"""LLM proposers for the probe agent loop.

``FakeLLMProposer`` round-robins registered mitigations (offline tests).
``LiteLLMProposer`` calls LiteLLM when ``amd-aorta[agent]`` is installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

log = logging.getLogger(__name__)

# Why the proposer set ``stop=True`` (drives CLI/report outcome labels).
StopReason = Literal[
    "baseline_pass",
    "exhausted_candidates",
    "agent_requested",
]

AUTOPSY_CATEGORIES: frozenset[str] = frozenset(
    {
        "rccl_hang",
        "thermal_throttle",
        "illegal_mem",
        "oom_fragment",
        "checkpoint_race",
        "launch_error",
        "perf_regression",
        "unknown",
    }
)

_BASELINE_CELL = "none-none"


@dataclass(frozen=True)
class AgentStep:
    """Structured output from one agent decision step.

    ``next_diagnostics`` is the second probe axis. A probe recipe has always
    had both (``mitigation_axis`` x ``diagnostic_axis``), but only the
    mitigation axis was ever an *action*: the diagnostic was fixed by whoever
    wrote the recipe, so the policy could buy a causal test and never buy
    information. It defaults to empty, and an empty list is exactly the old
    behaviour -- see ``loop._list_candidate_diagnostics`` for why nothing is
    offered on that axis unless the operator asks.
    """

    category: str
    hypothesis: str
    next_mitigations: list[str]
    confidence: float
    stop: bool
    stop_reason: StopReason | None = None
    next_diagnostics: list[str] = field(default_factory=list)
    #: Diagnostics the model proposed that the filter in
    #: :func:`_step_from_content` dropped, because they were not on the
    #: offered set: never registered, already tried, or on an axis the
    #: operator did not arm. Set by the proposer, never read from the model's
    #: reply. Distinct from ``AxisGrowth.rejected_diagnostics``, which is a
    #: *good* name the cell budget had no room for -- a rejected name still
    #: appears in ``next_diagnostics``, an unresolved one never does, so
    #: without this field it is retained nowhere and the trajectory records a
    #: proposal the model did not make.
    unresolved_diagnostics: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AgentStep:
        # Accept only a genuine JSON boolean: bool("false") is True, so a
        # malformed/untrusted "stop": "false" must not prematurely stop the
        # loop. Anything that isn't a real bool defaults to not-stopping.
        stop_raw = raw.get("stop", False)
        stop = stop_raw if isinstance(stop_raw, bool) else False
        reason_raw = raw.get("stop_reason")
        stop_reason: StopReason | None = None
        if stop and isinstance(reason_raw, str) and reason_raw in (
            "baseline_pass",
            "exhausted_candidates",
            "agent_requested",
        ):
            stop_reason = reason_raw  # type: ignore[assignment]
        # Defensive coercion: a real (or buggy) LLM can send a bare string,
        # null, or object for these fields. Only accept a genuine list for
        # next_mitigations -- never list("tf32_off"), which explodes into
        # single characters -- and fall back to a safe confidence instead of
        # raising on a non-numeric value. PolicyValidation re-checks names.
        raw_mitigations = raw.get("next_mitigations")
        next_mitigations = (
            [str(m) for m in raw_mitigations] if isinstance(raw_mitigations, list) else []
        )
        # Same rule, same reason, for the diagnostic axis: a bare string must
        # not become list("xnack"). Absent key -> [], which is what makes a
        # reply written against the old schema behave exactly as before.
        raw_diagnostics = raw.get("next_diagnostics")
        next_diagnostics = (
            [str(d) for d in raw_diagnostics] if isinstance(raw_diagnostics, list) else []
        )
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        # Type-aware, not str(): a null/non-string category or hypothesis from
        # the LLM must NOT become the literal "None"/"null" (which fails policy
        # validation for category and pollutes the report for hypothesis).
        category_raw = raw.get("category")
        category = (
            category_raw
            if isinstance(category_raw, str) and category_raw.strip()
            else "unknown"
        )
        hypothesis_raw = raw.get("hypothesis")
        hypothesis = hypothesis_raw if isinstance(hypothesis_raw, str) else ""
        return cls(
            category=category,
            hypothesis=hypothesis,
            next_mitigations=next_mitigations,
            confidence=confidence,
            stop=stop,
            stop_reason=stop_reason,
            next_diagnostics=next_diagnostics,
        )


class LLMProposer(Protocol):
    """Protocol for agent step proposers.

    ``diagnostic_candidates`` / ``tried_diagnostics`` default to None so a
    proposer written against the single-axis protocol keeps type-checking; the
    loop passes them on every call, and None reads as "no diagnostic action
    offered", which is the pre-joint-axis behaviour.
    """

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
        diagnostic_candidates: list[str] | None = None,
        tried_diagnostics: list[str] | None = None,
    ) -> AgentStep: ...


def _infer_category_from_detectors(detectors: list[str]) -> str:
    joined = " ".join(detectors).lower()
    if "tier2" in joined or "hang" in joined or "rccl" in joined:
        return "rccl_hang"
    if "oom" in joined or "137" in joined:
        return "oom_fragment"
    if "hip_error" in joined or "illegal" in joined or "memory" in joined:
        return "illegal_mem"
    if "checkpoint" in joined or "barrier" in joined:
        return "checkpoint_race"
    if "tier1:exit" in joined or "launch" in joined:
        return "launch_error"
    return "unknown"


class FakeLLMProposer:
    """Deterministic proposer: heuristic category + round-robin mitigations.

    It takes the diagnostic axis one name at a time, in registry order, and
    only when the caller offers one -- so the offline default stays exactly
    the single-axis walk it was whenever nothing is offered.
    """

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
        diagnostic_candidates: list[str] | None = None,
        tried_diagnostics: list[str] | None = None,
    ) -> AgentStep:
        last = cell_summaries[-1] if cell_summaries else {}
        detectors = list(last.get("failure_detectors_fired") or [])
        category = _infer_category_from_detectors(detectors)
        if symptom and category == "unknown":
            low = symptom.lower()
            if "hang" in low or "nccl" in low or "rccl" in low:
                category = "rccl_hang"
            elif "memory" in low or "illegal" in low:
                category = "illegal_mem"
            elif "oom" in low:
                category = "oom_fragment"

        # Baseline pass wins even when the allowlist has no further mitigations.
        for summary in cell_summaries:
            if summary.get("cell_name") == _BASELINE_CELL and summary.get("verdict") == "pass":
                return AgentStep(
                    category="unknown",
                    hypothesis="Baseline cell passed; no mitigation search needed.",
                    next_mitigations=[],
                    confidence=1.0,
                    stop=True,
                    stop_reason="baseline_pass",
                )

        remaining = [c for c in candidates if c not in tried and c != "none"]
        if not remaining:
            return AgentStep(
                category=category,
                hypothesis="No remaining registered mitigations to try.",
                next_mitigations=[],
                confidence=0.9,
                stop=True,
                stop_reason="exhausted_candidates",
            )

        next_m = remaining[0]
        remaining_d = _remaining_candidates(
            list(diagnostic_candidates or []), list(tried_diagnostics or [])
        )
        next_d = remaining_d[:1]
        return AgentStep(
            category=category,
            hypothesis=(
                f"Try mitigation {next_m!r} based on detectors {detectors!r}."
                + (f" Diagnostic {next_d[0]!r} for evidence." if next_d else "")
                + (f" Symptom: {symptom}" if symptom else "")
            ),
            next_mitigations=[next_m],
            confidence=0.5,
            stop=False,
            next_diagnostics=next_d,
        )


def _remaining_candidates(candidates: list[str], tried: list[str]) -> list[str]:
    return [c for c in candidates if c not in tried and c != "none"]


def _exhausted_step() -> AgentStep:
    """Stop without spending tokens, and without needing a backend installed."""
    return AgentStep(
        category="unknown",
        hypothesis="No remaining registered mitigations to try.",
        next_mitigations=[],
        confidence=0.9,
        stop=True,
        stop_reason="exhausted_candidates",
    )


def _safe_stop(hypothesis: str) -> AgentStep:
    """Turn an unusable model response into a stop the loop can still report on."""
    return AgentStep(
        category="unknown",
        hypothesis=hypothesis,
        next_mitigations=[],
        confidence=0.0,
        stop=True,
        stop_reason="agent_requested",
    )


def _build_prompt(
    symptom: str | None,
    cell_summaries: list[dict[str, Any]],
    remaining: list[str],
    tried: list[str],
    remaining_diagnostics: list[str] | None = None,
    tried_diagnostics: list[str] | None = None,
) -> tuple[str, str]:
    """The system and user messages, shared by every real proposer.

    One definition so the two backends cannot drift into asking for different
    JSON, which is the failure a shared provider layer is supposed to prevent.

    The diagnostic axis is described only when something is offered on it. A
    prompt that names a ``next_diagnostics`` key with an empty candidate list
    invites a reply the loop would then have to discard, and it would also
    change the bytes of every existing single-axis prompt.
    """
    offered_diagnostics = list(remaining_diagnostics or [])
    system = (
        "You are an AORTA probe agent. Propose ONLY registered mitigation "
        "names from the candidate list. Never propose shell commands or argv. "
        "Return strict JSON with keys: category, hypothesis, next_mitigations "
        "(list of strings), confidence (0-1), stop (bool). "
        f"category must be one of: {sorted(AUTOPSY_CATEGORIES)}."
    )
    payload: dict[str, Any] = {
        "symptom": symptom,
        "cell_summaries": cell_summaries,
        "candidates": remaining,
        "already_tried": tried,
    }
    if offered_diagnostics:
        system = (
            "You are an AORTA probe agent localising the root cause of a GPU "
            "failure at minimum cost. You have two actions, and you may use "
            "either or both in one step. next_mitigations tests a causal "
            "hypothesis: a mitigation that makes the repro pass names the "
            "cause. next_diagnostics buys evidence: it switches on an "
            "observability knob that does not change the outcome but makes "
            "the mechanism visible. Propose ONLY names from the matching "
            "candidate list. Never propose shell commands or argv.\n"
            "Cost matters and the two axes multiply, not add: the probe runs "
            "the cross product of the two axes, so adding one diagnostic "
            "re-runs every mitigation under it. Propose the smallest set that "
            "would discriminate between your hypotheses.\n"
            "Return strict JSON with keys: category, hypothesis, "
            "next_mitigations (list of strings), next_diagnostics (list of "
            "strings), confidence (0-1), stop (bool). "
            f"category must be one of: {sorted(AUTOPSY_CATEGORIES)}."
        )
        payload["diagnostic_candidates"] = offered_diagnostics
        payload["already_tried_diagnostics"] = list(tried_diagnostics or [])
    user = json.dumps(payload, indent=2)
    return system, user


def _strip_code_fence(content: str) -> str:
    """Unwrap a ```json fenced block.

    Needed on the chat-provider path, which has no ``response_format`` knob to
    ask for a bare object; a fenced reply is otherwise a parse failure and a
    wasted iteration.
    """
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()


def _step_from_content(
    content: str | None,
    remaining: list[str],
    remaining_diagnostics: list[str],
) -> AgentStep:
    """Parse a model reply into an :class:`AgentStep`, failing safe.

    Providers return malformed or partial JSON, a non-object, or nothing at all
    even when asked for strict JSON. Every one of those becomes a stop rather
    than an exception, so the loop still writes a report.

    An empty ``remaining_diagnostics`` filters every proposed diagnostic away,
    and that policy is unchanged: the filter below exists so the model cannot
    widen its own allowlist, and a caller that named no diagnostic axis has
    authorised nothing on it.

    What changed is that ``remaining_diagnostics`` is now **required**. It used
    to default to None, so "I am deliberately offering nothing" and "I forgot
    to pass this" produced the same silent 100% discard -- one recorded
    nowhere, on the argument that decides whether the model's entire
    diagnostic proposal survives. Requiring it turns the second case into a
    TypeError at the call site, and the drops the first case makes are now
    recorded on ``unresolved_diagnostics``.
    """
    if not content or not content.strip():
        return _safe_stop("Empty LLM response")
    try:
        raw = json.loads(_strip_code_fence(content))
        if not isinstance(raw, dict):
            raise TypeError(f"expected a JSON object, got {type(raw).__name__}")
        step = AgentStep.from_dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _safe_stop(f"LLM returned unparseable response: {exc}")

    # Never let the model widen its own allowlist: PolicyValidation re-checks,
    # but a name outside `remaining` is a mitigation already tried or never
    # registered, and running it is not the agent's call.
    filtered = [m for m in step.next_mitigations if m in remaining]
    filtered_diagnostics = [
        d for d in step.next_diagnostics if d in remaining_diagnostics
    ]
    # Keep what the diagnostic filter dropped. A dropped name is retained
    # nowhere else: it never reaches plan_axis_growth, so it cannot appear in
    # axis_growth.rejected_diagnostics, and the loop-level allowlist guard
    # that would have raised on it sits after this filter has already removed
    # it. So the trajectory records a proposal the model did not make, and
    # the discarded half cannot be reconstructed -- only discarded. Not
    # de-duplicated: this is the audit trail of what the model emitted.
    unresolved_diagnostics = [
        d for d in step.next_diagnostics if d not in remaining_diagnostics
    ]
    if unresolved_diagnostics:
        log.warning(
            "proposer named %d diagnostic(s) that are not on the offered set "
            "and were dropped: %s (offered: %s)",
            len(unresolved_diagnostics),
            sorted(set(unresolved_diagnostics)),
            sorted(remaining_diagnostics),
        )
    stop_reason = step.stop_reason
    if step.stop and stop_reason is None:
        stop_reason = "agent_requested"
    return AgentStep(
        category=step.category,
        hypothesis=step.hypothesis,
        next_mitigations=filtered,
        confidence=step.confidence,
        stop=step.stop,
        stop_reason=stop_reason,
        next_diagnostics=filtered_diagnostics,
        unresolved_diagnostics=unresolved_diagnostics,
    )


class LiteLLMProposer:
    """LiteLLM-backed proposer (requires ``pip install 'amd-aorta[agent]'``).

    Retained after Phase 5b as the path that works on an ``[agent]``-only
    install, where the chat provider layer is not present. ``--llm-backend
    litellm`` has shipped and must keep working without the chat extra, so
    :func:`make_proposer` prefers the shared layer and falls back to this.
    """

    def __init__(self, *, model: str = "gpt-4o-mini") -> None:
        self._model = model

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
        diagnostic_candidates: list[str] | None = None,
        tried_diagnostics: list[str] | None = None,
    ) -> AgentStep:
        remaining = _remaining_candidates(candidates, tried)
        remaining_d = _remaining_candidates(
            list(diagnostic_candidates or []), list(tried_diagnostics or [])
        )
        # Exhaustion is still judged on the mitigation axis alone: a run with
        # diagnostics left but no mitigations left can buy evidence it has no
        # remaining causal test to spend it on, which is not a search.
        if not remaining:
            return _exhausted_step()

        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "LiteLLM is required for --llm-backend=litellm. "
                "Install it with either:\n"
                "  pip install litellm\n"
                "  pip install -e '.[agent]'   # from the aorta repo root (editable + extra)\n"
                "If pip says the 'agent' extra does not exist, your installed amd-aorta "
                "distribution is stale — reinstall from this repo with -e '.[agent]'."
            ) from exc

        system, user = _build_prompt(
            symptom, cell_summaries, remaining, tried, remaining_d, tried_diagnostics
        )
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return _step_from_content(
            response.choices[0].message.content, remaining, remaining_d
        )


#: Backends resolved through the shared chat provider layer (Decision 7a). The
#: names are the chat factory's own, so ``--llm-backend`` and
#: ``--llm-provider`` mean the same thing on both front doors.
CHAT_PROVIDER_BACKENDS: frozenset[str] = frozenset({"litellm", "openai", "vllm"})

_CHAT_EXTRA_HINT = (
    "--llm-backend={backend} is configured through the shared chat provider "
    "layer, which needs the chat-cli extra.\n"
    "Install it with:  pip install 'amd-aorta[chat-cli]'\n"
    "The provider itself (endpoint, model, API key, gateway headers) is then "
    "configured once, in ~/.config/aorta/chat.toml or AORTA_CHAT_* -- the same "
    "settings 'aorta chat' uses.\n"
    "(missing: {missing})"
)


class ChatProviderProposer:
    """Proposer on the shared chat provider layer (locked Decision 7a).

    The point is single configuration: ``vllm`` / ``openai`` / ``litellm`` are
    set up once, in the chat profile, and both front doors read it. Before this,
    ``aorta agent`` reached litellm directly and knew nothing about the endpoint,
    gateway headers or auth scheme that ``aorta chat`` had already been told.

    Unlike :class:`LiteLLMProposer` there is no ``response_format`` to lean on
    -- the layer returns a LangChain chat model, not a raw completion call -- so
    the reply is fence-tolerant and every parse failure still fails safe.
    """

    def __init__(self, provider: str, *, model: str | None = None) -> None:
        self._provider = provider
        self._model = model

    def _chat_model(self) -> Any:
        """Resolve the configured chat model, or explain which extra is missing.

        Imported inside the call, not at module scope: this is the one place
        core reaches into ``aorta.chat``, and deferring it keeps ``import
        aorta.agent`` (and so ``aorta --help``) free of langchain.
        """
        try:
            from aorta.chat.config import apply_cli_overrides
            from aorta.chat.inference.providers.factory import get_backend
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            if missing.startswith("aorta.chat"):
                # A broken aorta.chat submodule is a real bug, not a missing
                # extra; advising an install would bury it. Same rule as
                # cli/chat.py's _load and cli/bench.py.
                raise
            raise ImportError(
                _CHAT_EXTRA_HINT.format(backend=self._provider, missing=missing)
            ) from exc

        # Reuse the chat layer's own precedence, which also knows whether a
        # model name belongs to the local or the remote field.
        apply_cli_overrides(provider=self._provider, model=self._model)
        return get_backend(self._provider).get_chat_model(temperature=0.0, streaming=False)

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
        diagnostic_candidates: list[str] | None = None,
        tried_diagnostics: list[str] | None = None,
    ) -> AgentStep:
        remaining = _remaining_candidates(candidates, tried)
        remaining_d = _remaining_candidates(
            list(diagnostic_candidates or []), list(tried_diagnostics or [])
        )
        # Checked before the import, so an exhausted loop neither spends tokens
        # nor requires the extra to be installed. Mirrors both siblings.
        if not remaining:
            return _exhausted_step()

        system, user = _build_prompt(
            symptom, cell_summaries, remaining, tried, remaining_d, tried_diagnostics
        )
        # Role tuples rather than langchain message classes: one fewer import on
        # a path that only needs to say who said what.
        response = self._chat_model().invoke([("system", system), ("human", user)])
        return _step_from_content(
            getattr(response, "content", None), remaining, remaining_d
        )


def _chat_layer_available() -> bool:
    """Whether the chat provider layer can be imported at all.

    ``find_spec`` rather than an import: this is asked on the ``litellm`` path
    to choose between two working implementations, and it must not pull in
    langchain for an install that is going to use the direct path anyway.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("langchain_core") is not None
    except (ImportError, ValueError):
        return False


def make_proposer(backend: str, *, model: str | None = None) -> LLMProposer:
    """Build the proposer for ``--llm-backend``.

    Phase 5b (locked Decision 7a) put ``vllm`` / ``openai`` / ``litellm`` onto
    the shared chat provider layer, so a provider is configured once and both
    front doors read that configuration.

    ``litellm`` is the one backend with a shipped contract to keep: it has
    worked on an ``[agent]``-only install since before ``aorta.chat`` existed.
    So it prefers the shared layer and falls back to the direct
    :class:`LiteLLMProposer` when the chat extra is absent, rather than
    breaking an install that used to work. ``vllm`` and ``openai`` are new, have
    no such history, and say plainly which extra they need.

    ``fake`` stays the default and stays fully offline -- it imports nothing and
    reaches nothing, which is what makes the test suite and ``--dry-run``
    hermetic.
    """
    if backend == "fake":
        return FakeLLMProposer()
    if backend == "litellm" and not _chat_layer_available():
        return LiteLLMProposer(model=model or "gpt-4o-mini")
    if backend in CHAT_PROVIDER_BACKENDS:
        return ChatProviderProposer(backend, model=model)
    raise ValueError(
        f"unknown agent LLM backend: {backend!r} "
        f"(expected one of {', '.join(sorted({'fake', *CHAT_PROVIDER_BACKENDS}))})"
    )


__all__ = [
    "AUTOPSY_CATEGORIES",
    "CHAT_PROVIDER_BACKENDS",
    "AgentStep",
    "ChatProviderProposer",
    "FakeLLMProposer",
    "LLMProposer",
    "LiteLLMProposer",
    "StopReason",
    "make_proposer",
]
