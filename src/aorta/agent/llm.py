"""LLM proposers for the probe agent loop.

``FakeLLMProposer`` round-robins registered mitigations (offline tests).
``LiteLLMProposer`` calls LiteLLM when ``amd-aorta[agent]`` is installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

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


class LiteLLMProposer:
    """LiteLLM-backed proposer (requires ``pip install 'amd-aorta[agent]'``)."""

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
        remaining = [c for c in candidates if c not in tried and c != "none"]
        # Nothing left to try: stop without spending tokens (and without
        # needing litellm installed at all). Mirrors FakeLLMProposer.
        if not remaining:
            return AgentStep(
                category="unknown",
                hypothesis="No remaining registered mitigations to try.",
                next_mitigations=[],
                confidence=0.9,
                stop=True,
                stop_reason="exhausted_candidates",
            )

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

        system = (
            "You are an AORTA probe agent. Propose ONLY registered mitigation "
            "names from the candidate list. Never propose shell commands or argv. "
            "Return strict JSON with keys: category, hypothesis, next_mitigations "
            "(list of strings), confidence (0-1), stop (bool). "
            f"category must be one of: {sorted(AUTOPSY_CATEGORIES)}."
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
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            return AgentStep(
                category="unknown",
                hypothesis="Empty LLM response",
                next_mitigations=[],
                confidence=0.0,
                stop=True,
                stop_reason="agent_requested",
            )
        # Even with response_format=json_object, providers can return
        # malformed/partial JSON or a non-object. Convert any parse/shape
        # failure into a safe stop so the loop still emits a report.
        try:
            raw = json.loads(content)
            if not isinstance(raw, dict):
                raise TypeError(f"expected a JSON object, got {type(raw).__name__}")
            step = AgentStep.from_dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return AgentStep(
                category="unknown",
                hypothesis=f"LLM returned unparseable response: {exc}",
                next_mitigations=[],
                confidence=0.0,
                stop=True,
                stop_reason="agent_requested",
            )
        # Filter to remaining candidates only
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


def make_proposer(backend: str, *, model: str = "gpt-4o-mini") -> LLMProposer:
    if backend == "fake":
        return FakeLLMProposer()
    if backend == "litellm":
        return LiteLLMProposer(model=model)
    raise ValueError(f"unknown agent LLM backend: {backend!r}")


__all__ = [
    "AUTOPSY_CATEGORIES",
    "AgentStep",
    "FakeLLMProposer",
    "LLMProposer",
    "LiteLLMProposer",
    "StopReason",
    "make_proposer",
]
