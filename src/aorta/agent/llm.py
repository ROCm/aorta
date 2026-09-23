"""LLM proposers for the probe agent loop.

``FakeLLMProposer`` round-robins registered mitigations (offline tests).
``LiteLLMProposer`` calls LiteLLM when ``amd-aorta[agent]`` is installed.
``LayaProposer`` answers the same decision from a local calibrated encoder when
``amd-aorta[laya]`` is.

**Every model import in this module is deferred, and the rule is tested.**
``tests/cli/test_chat_boundaries.py`` imports this module in a clean interpreter
and fails if any of ``_HEAVY_PREFIXES`` -- torch, onnxruntime, fastembed, the
langchain stack, pydantic, chainlit -- reaches ``sys.modules``. The reason is
``--llm-backend=fake``: it is the default, it is what the test suite and
``--dry-run`` depend on, and it has to keep working on a base install of
``pip install amd-aorta``, which is ``pyyaml`` plus ``click``. So the chat seam
lives inside ``ChatProviderProposer._chat_model`` and the Laya seam inside
``LayaProposer.propose``, and neither costs anything to anyone who does not
select it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # Annotations only: `from __future__ import annotations` means
    # this costs nothing at run time, which is what the import-boundary probe
    # above measures.
    from aorta.laya.predictor import LayaPredictor

# Why the proposer set ``stop=True`` (drives CLI/report outcome labels).
StopReason = Literal[
    "baseline_pass",
    "exhausted_candidates",
    "agent_requested",
]

#: Reachable only from instrument evidence: a sanitizer that watched two waves
#: collide, a debugger that read a stopped wave, or a tool that could not run at
#: all. Nothing the probe agent does can establish one of these -- it reaches a
#: category by trying mitigations and seeing what changes, which cannot observe
#: a race or distinguish "the sanitizer found nothing" from "the sanitizer never
#: ran".
EVIDENCE_ONLY_CATEGORIES: frozenset[str] = frozenset(
    {
        "gpu_race",
        "numeric_silent",
        "tooling_gap",
    }
)

#: Every category either front door may return. The probe agent reaches a
#: category by trying mitigations; :mod:`aorta.cia` reaches one by reading
#: instrument evidence. They answer different questions and share this
#: vocabulary, so a verdict means the same thing whichever produced it -- and
#: anything reading a report validates against this.
AUTOPSY_CATEGORIES: frozenset[str] = (
    frozenset(
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
    | EVIDENCE_ONLY_CATEGORIES
)

#: What the probe agent may propose: the shared vocabulary less what only an
#: instrument can establish.
#:
#: Derived rather than written out a second time. Listing it by hand is how the
#: two drift, and the drift is silent -- offering the probe model a category it
#: has no way to reach teaches it to guess one, and the guess validates.
PROBE_CATEGORIES: frozenset[str] = AUTOPSY_CATEGORIES - EVIDENCE_ONLY_CATEGORIES

_BASELINE_CELL = "none-none"


@dataclass(frozen=True)
class AgentStep:
    """Structured output from one agent decision step."""

    category: str
    hypothesis: str
    next_mitigations: list[str]
    confidence: float
    stop: bool
    stop_reason: StopReason | None = None

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
        )


class LLMProposer(Protocol):
    """Protocol for agent step proposers."""

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
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
    """Deterministic proposer: heuristic category + round-robin mitigations."""

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
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
        return AgentStep(
            category=category,
            hypothesis=(
                f"Try mitigation {next_m!r} based on detectors {detectors!r}."
                + (f" Symptom: {symptom}" if symptom else "")
            ),
            next_mitigations=[next_m],
            confidence=0.5,
            stop=False,
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
) -> tuple[str, str]:
    """The system and user messages, shared by every real proposer.

    One definition so the two backends cannot drift into asking for different
    JSON, which is the failure a shared provider layer is supposed to prevent.
    """
    system = (
        "You are an AORTA probe agent. Propose ONLY registered mitigation "
        "names from the candidate list. Never propose shell commands or argv. "
        "Return strict JSON with keys: category, hypothesis, next_mitigations "
        "(list of strings), confidence (0-1), stop (bool). "
        f"category must be one of: {sorted(PROBE_CATEGORIES)}."
    )
    user = json.dumps(
        {
            "symptom": symptom,
            "cell_summaries": cell_summaries,
            "candidates": remaining,
            "already_tried": tried,
        },
        indent=2,
    )
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


def _step_from_content(content: str | None, remaining: list[str]) -> AgentStep:
    """Parse a model reply into an :class:`AgentStep`, failing safe.

    Providers return malformed or partial JSON, a non-object, or nothing at all
    even when asked for strict JSON. Every one of those becomes a stop rather
    than an exception, so the loop still writes a report.
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
    ) -> AgentStep:
        remaining = _remaining_candidates(candidates, tried)
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

        system, user = _build_prompt(symptom, cell_summaries, remaining, tried)
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return _step_from_content(response.choices[0].message.content, remaining)


#: Backends resolved through the shared chat provider layer (Decision 7a). The
#: names are the chat factory's own, so ``--llm-backend`` and
#: ``--llm-provider`` mean the same thing on both front doors.
CHAT_PROVIDER_BACKENDS: frozenset[str] = frozenset({"litellm", "openai", "vllm"})

#: Every value ``--llm-backend`` accepts, which is what :func:`make_proposer`
#: resolves.
#:
#: Exported because the Click choice in ``aorta/cli/agent_mitigate.py`` is
#: hard-coded -- that decorator runs at import time and ``aorta --help`` must not
#: pay for this module -- so the two can drift silently. The hard-coded list is
#: checked against this one in ``tests/agent/test_llm_providers.py``, which is
#: the only thing standing between adding a backend here and it being
#: unreachable from the command line.
AGENT_LLM_BACKENDS: frozenset[str] = frozenset({"fake", "laya"}) | CHAT_PROVIDER_BACKENDS

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
    ) -> AgentStep:
        remaining = _remaining_candidates(candidates, tried)
        # Checked before the import, so an exhausted loop neither spends tokens
        # nor requires the extra to be installed. Mirrors both siblings.
        if not remaining:
            return _exhausted_step()

        system, user = _build_prompt(symptom, cell_summaries, remaining, tried)
        # Role tuples rather than langchain message classes: one fewer import on
        # a path that only needs to say who said what.
        response = self._chat_model().invoke([("system", system), ("human", user)])
        return _step_from_content(getattr(response, "content", None), remaining)


#: The three questions a Laya proposer asks, spelled as text rather than as
#: :class:`aorta.laya.predictor.Choice` / ``Noul`` instances so that naming them
#: costs no import.
#:
#: Public because ``aorta/laya/corpus/proposer.py`` labels a corpus against these
#: same three questions, and the two have to be the same strings. If they drift,
#: the encoder is fine-tuned on one question and asked another -- which does not
#: fail, it just answers slightly worse for a reason nobody would look for. That
#: module already imports :data:`PROBE_CATEGORIES` from here; the questions
#: belong in the same place, and ``tests/agent/test_laya_proposer.py`` fails if
#: the two copies stop matching.
LAYA_MITIGATION_QUESTION = (
    "Which of these candidate mitigations will make this failure stop reproducing?"
)
LAYA_CATEGORY_QUESTION = "Which category of failure is this?"
LAYA_STOP_QUESTION = (
    "Should the mitigation search stop here, rather than trying another "
    "registered mitigation?"
)
LAYA_STOP_WHEN_TRUE = "no untried mitigation can plausibly clear this failure"
LAYA_STOP_WHEN_FALSE = "at least one untried mitigation is still worth running"

#: What ``--llm-backend=laya`` loads when ``--llm-model`` names nothing.
#:
#: The fine-tuned checkpoint rather than the base one, because the model card's
#: own numbers put base ``laya`` below a majority-class baseline on its typed
#: decisions and call it "a fast base to specialise, not a zero-shot decision
#: engine". Defaulting to the weaker of the two would make the first thing anyone
#: tries the worst version of it.
DEFAULT_LAYA_CHECKPOINT = "laya-typed-decisions"

#: How much of the stop noul's probability mass it takes to end the search.
#:
#: Above the 0.5 midpoint because the two mistakes do not cost the same. A false
#: stop ends an investigation and reports a category nobody went on to test; a
#: false continue costs one more probe cell, and ``AgentPolicy`` already bounds
#: how many of those there can be. So the asymmetry is paid for in cells.
#:
#: **This number is not a calibration figure and must not be reported as one.**
#: Nothing here applies the per-(question type, option count) temperature fit
#: that Phase 1 exists to produce, so it is a policy choice about which error to
#: prefer, made in the absence of a fit rather than derived from one. Re-derive
#: it when there is one -- and read ``docs/laya-packaging.md`` on why a
#: probability is a function of the checkpoint *and* the fit applied to it.
#:
#: Whether the stop noul's own bucket was trustworthy is not a thing this
#: constant can know. An earlier revision of this comment asserted that a noul
#: is at least not in the bucket the library clamps, which is true of today's
#: published checkpoint and is not a property of nouls. The step asks
#: ``Calibration.caveat`` per question instead, so a checkpoint that clamped the
#: noul bucket says so on the very step that thresholded against it.
DEFAULT_LAYA_STOP_THRESHOLD = 0.75


class LayaProposer:
    """Proposer on a local calibrated encoder (requires ``pip install 'amd-aorta[laya]'``).

    The same decision as :class:`FakeLLMProposer` and :class:`ChatProviderProposer`,
    reached without generating a token. ``AgentStep`` happens to be almost exactly
    the shape Laya answers in: ``next_mitigations`` is a choice over the
    candidates that are actually left, ``category`` a choice over
    :data:`PROBE_CATEGORIES`, ``stop`` a noul, and ``confidence`` the probability
    the chosen answer carries rather than a float a prompt asked a model to
    invent. All three are one forward pass, because Laya batches M questions over
    one state -- see :class:`aorta.laya.predictor.LayaPredictor`.

    **``confidence`` is not yet a calibrated probability, and the step says so
    out loud.** Two separate things are wrong with it, and the disclosure is in
    two halves because the two things are known at different times.

    The first is the *question*: a probability read off an N-way choice is not
    comparable to a noul's or to an LLM's self-report without N. The argmax of a
    21-way answer is a strong preference at 0.24 and pure noise at 0.048, and
    both look like "low confidence" to a threshold picked for a producer with a
    different answer shape. N and the library's bucket for it
    (``aorta.laya.predictor.bucket_for``) are pure and offline, so they are on
    every step whether or not a checkpoint ever loaded.

    The second is the *checkpoint*: whether that bucket's shipped temperature was
    trustworthy. ``Calibration.caveat`` answers it from the checkpoint's own
    tables, and that is deliberately not re-derived here. An earlier revision
    hardcoded "eleven or more options is the broken bucket", which is true of the
    published checkpoint and false for exactly the artifact Phase 1 exists to
    produce -- a fine-tune with a sane ``choice:11+`` and a broken
    ``choice:3-5``. That version would have gone quiet on the bucket that was
    actually broken while still warning about one that was fine.

    The number itself is kept, because it is the best ranking signal available
    and a 0.0 would destroy information while colliding with ``_safe_stop``'s.
    Anything downstream that thresholds ``AgentStep.confidence`` must read the
    hypothesis before trusting it. A figure that looks calibrated and is not is
    precisely the defect this integration exists to remove, and shipping one here
    under a new name would be worse than the self-reported float it replaces.

    ``hypothesis`` stays templated, following :class:`FakeLLMProposer`'s. Laya
    never emits text, so there is nothing to ask it for, and a sentence built
    from the answers is more honest than one built from nothing. It is also the
    only field the caveat above can ride on without a call-site change.

    **The candidate list is the answer space, so this proposer cannot name a
    mitigation that was not offered.** ``_step_from_content`` filters the LLM
    backends' replies for exactly that reason; here the filter is structural.
    ``PolicyValidation`` still re-checks every name downstream, and should: this
    class is one of three proposers and the guard has to hold for all of them.
    """

    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        device: str | None = None,
        stop_threshold: float = DEFAULT_LAYA_STOP_THRESHOLD,
        predictor: LayaPredictor | None = None,
    ) -> None:
        self._checkpoint = checkpoint or DEFAULT_LAYA_CHECKPOINT
        self._device = device
        self._stop_threshold = stop_threshold
        # Injected by tests against ``FakeLayaPredictor``, which needs no weights
        # and no extra. Nothing on the CLI path reaches this argument, so a run
        # cannot end up reporting a hash function's output as a verdict.
        self._predictor = predictor

    def _laya_predictor(self) -> LayaPredictor:
        """Resolve the predictor once, importing the seam here and not above.

        Deferred for the reason this module's docstring gives, and cached for a
        second one: the loop calls :meth:`propose` once per iteration, up to
        ``--max-iterations``, and a checkpoint rebuilt per call would cost
        seconds per iteration -- the card measures a 7.4 s median reload on CPU.
        The weights load on the first question and stay loaded, which is
        :class:`aorta.laya.predictor.LayaAgentPredictor`'s own behaviour.
        """
        if self._predictor is None:
            from aorta.laya.predictor import make_predictor

            # ``checkpoint=`` rather than ``backend=`` so that ``--llm-model``
            # can name either a published checkpoint or a local fine-tune
            # directory. It also means ``fake`` is unreachable from here, which
            # is deliberate.
            self._predictor = make_predictor(
                checkpoint=self._checkpoint, device=self._device
            )
        return self._predictor

    def propose(
        self,
        *,
        symptom: str | None,
        cell_summaries: list[dict[str, Any]],
        candidates: list[str],
        tried: list[str],
    ) -> AgentStep:
        remaining = _remaining_candidates(candidates, tried)
        # Checked before the import, so an exhausted loop neither loads weights
        # nor requires the extra to be installed. Mirrors both siblings.
        if not remaining:
            return _exhausted_step()

        from aorta.laya.predictor import (
            Choice,
            ChoiceAnswer,
            Noul,
            NoulAnswer,
            ask_one,
            bucket_for,
        )

        mitigation_q = Choice(
            question=LAYA_MITIGATION_QUESTION, options=tuple(remaining)
        )
        # The derived probe set, not AUTOPSY_CATEGORIES. Offering a category the
        # agent has no way to reach teaches it to guess one, and the guess
        # validates -- see the comment above PROBE_CATEGORIES.
        category_q = Choice(
            question=LAYA_CATEGORY_QUESTION, options=tuple(sorted(PROBE_CATEGORIES))
        )
        stop_q = Noul(
            question=LAYA_STOP_QUESTION,
            when_true=LAYA_STOP_WHEN_TRUE,
            when_false=LAYA_STOP_WHEN_FALSE,
        )

        predictor = self._laya_predictor()
        # The user half of the prompt the LLM backends send, verbatim, because
        # that is what `aorta/laya/corpus/proposer.py` recorded as the state when
        # it built the training corpus. Serialising it a second way here would be
        # train/serve skew introduced by a helper nobody would suspect -- and it
        # would also stop a Phase 1 comparison between the two backends being a
        # comparison, since they would then differ in their input as well as in
        # their model.
        _system, state = _build_prompt(symptom, cell_summaries, remaining, tried)
        mitigation, category, stopping = ask_one(
            predictor, state, [mitigation_q, category_q, stop_q]
        )
        # The check ``ask_noul`` and ``ask_choice`` perform, carried along rather
        # than borrowed: asking through those helpers would cost three forward
        # passes where the whole point is one. A predictor answering the wrong
        # shape must not have its answer read as a probability.
        if not (
            isinstance(mitigation, ChoiceAnswer)
            and isinstance(category, ChoiceAnswer)
            and isinstance(stopping, NoulAnswer)
        ):
            raise TypeError(
                f"{type(predictor).__name__} answered the proposer's three questions with "
                f"{type(mitigation).__name__}, {type(category).__name__} and "
                f"{type(stopping).__name__}"
            )

        model_id = predictor.model_id()
        # Read *after* the answers, and the order is load-bearing. `calibration()`
        # deliberately does not load a checkpoint -- an accessor should not cost
        # seven seconds of model build -- so asked before the first forward pass
        # it reports "not loaded yet" and every step would disclose an unknown
        # calibration on a run whose calibration was perfectly knowable. By here
        # `ask_one` has touched the weights, so the tables it diffs are there.
        calibration = predictor.calibration()

        # `.at()` rather than a bare boolean, so the threshold is named at the
        # point it is applied; NoulAnswer deliberately has no default reading.
        if stopping.at(self._stop_threshold):
            return AgentStep(
                category=category.option,
                # The stop noul gets its own caveat rather than inheriting the
                # choice's. Which bucket clamped is a fact about the checkpoint,
                # so a fine-tune that clamped the noul bucket has to say so on
                # the step that thresholded against it.
                hypothesis=(
                    "Stopping: no untried mitigation looks likely to clear this failure. "
                    f"[{model_id}: p(stop)={stopping.probability:.2f} "
                    f"at threshold {self._stop_threshold:.2f} "
                    f"({bucket_for(stop_q)}){calibration.caveat(stop_q)}]"
                ),
                next_mitigations=[],
                # Named rather than left to the loop to infer. `_resolve_stop_outcome`
                # falls back to reading the hypothesis text when `stop_reason` is
                # None, and a proposer that knows why it stopped should not make it
                # guess from prose.
                stop=True,
                stop_reason="agent_requested",
                confidence=stopping.probability,
            )

        next_m = mitigation.option
        last = cell_summaries[-1] if cell_summaries else {}
        detectors = list(last.get("failure_detectors_fired") or [])
        return AgentStep(
            category=category.option,
            # `FakeLLMProposer`'s sentence, plus what answered it. Decision 22
            # asks that every report a Laya verdict reaches records which
            # checkpoint produced it *inline*, because a report gets copied,
            # archived and attached to a ticket away from anything beside it --
            # and `hypothesis` is the only field on an AgentStep that both
            # `agent_log.jsonl` and `agent_report.md` carry through unchanged.
            # The width and its bucket come from the question and are printed
            # whatever the checkpoint turns out to be; the caveat comes from the
            # checkpoint and is empty when there is nothing to say. Splitting
            # them is what lets a run whose weights failed to load still carry
            # the half of the disclosure that never needed them.
            hypothesis=(
                f"Try mitigation {next_m!r} based on detectors {detectors!r}."
                + (f" Symptom: {symptom}" if symptom else "")
                + f" [{model_id}: p({next_m})={mitigation.probability:.2f} "
                f"of {len(remaining)} candidates ({bucket_for(mitigation_q)}), "
                f"p(stop)={stopping.probability:.2f}"
                f"{calibration.caveat(mitigation_q)}]"
            ),
            next_mitigations=[next_m],
            # Off the head that answered the decision this step reports -- the
            # probability of the mitigation being proposed, not of the stop that
            # was declined. No temperature fit is applied on this path, and the
            # hypothesis says whether the checkpoint's own fit for this bucket
            # survived, so the number ranks candidates and does not measure
            # certainty. This is the field that would otherwise be read as
            # though it did.
            confidence=mitigation.probability,
            stop=False,
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

    ``laya`` is the odd one out: it is not a provider at all but a local
    encoder, so it reads no chat settings and ``model`` names a *checkpoint*
    rather than a model on someone's endpoint.

    ``fake`` stays the default and stays fully offline -- it imports nothing and
    reaches nothing, which is what makes the test suite and ``--dry-run``
    hermetic.
    """
    if backend == "fake":
        return FakeLLMProposer()
    if backend == "laya":
        return LayaProposer(checkpoint=model)
    if backend == "litellm" and not _chat_layer_available():
        return LiteLLMProposer(model=model or "gpt-4o-mini")
    if backend in CHAT_PROVIDER_BACKENDS:
        return ChatProviderProposer(backend, model=model)
    raise ValueError(
        f"unknown agent LLM backend: {backend!r} "
        f"(expected one of {', '.join(sorted(AGENT_LLM_BACKENDS))})"
    )


__all__ = [
    "AGENT_LLM_BACKENDS",
    "AUTOPSY_CATEGORIES",
    "DEFAULT_LAYA_CHECKPOINT",
    "DEFAULT_LAYA_STOP_THRESHOLD",
    "EVIDENCE_ONLY_CATEGORIES",
    "LAYA_CATEGORY_QUESTION",
    "LAYA_MITIGATION_QUESTION",
    "LAYA_STOP_QUESTION",
    "LAYA_STOP_WHEN_FALSE",
    "LAYA_STOP_WHEN_TRUE",
    "PROBE_CATEGORIES",
    "CHAT_PROVIDER_BACKENDS",
    "AgentStep",
    "ChatProviderProposer",
    "FakeLLMProposer",
    "LLMProposer",
    "LayaProposer",
    "LiteLLMProposer",
    "StopReason",
    "make_proposer",
]
