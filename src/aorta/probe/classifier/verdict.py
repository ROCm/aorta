"""Verdict precedence resolver for ``aorta probe`` Phase 2 (issue #188).

Combines the per-tier detector outputs into the final
``result.json`` shape. The rules (rubric §2.B FR 2.8) are:

(a) Any Tier 1–4 detector fires OR any ``custom_patterns[*]``
    with ``on_match: fail`` fires → ``verdict = "fail"``.

(b) ``result.json::failure_detectors_fired`` lists ALL fired in a
    fixed encounter order: Tier 1, Tier 2, Tier 3, Tier 4, then
    Tier 5 (``custom_patterns`` failures). The order is set by
    :func:`resolve` (and the :class:`VerdictInputs` field order it
    iterates), independent of when each tier physically fires
    relative to the workload exit. The in-flight Tier 2 hang monitor
    contributes its detector IDs to ``inputs.tier2`` before
    :func:`resolve` is called, so even though the predicate fires
    mid-run, the IDs always land between Tier 1 and Tier 3 in the
    serialised list.

(c) Any ``custom_patterns[*]`` with ``required_for_pass: true``
    AND none of them fired → add ``meta:missing_pass_signal`` to
    ``failure_detectors_fired`` AND ``verdict = "fail"``.

(d) Otherwise → ``verdict = "pass"``.

``on_match: warn`` populates ``warn_detectors_fired`` only.
``on_match: info`` populates ``capture`` only.

The resolver consumes :class:`VerdictInputs` (a plain bundle of
per-tier outputs) and returns :class:`Verdict`. Both are
dataclasses so the call site stays declarative and unit tests can
build inputs in one line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aorta.probe.classifier.tier5_custom import CompiledPattern, CustomScanResult

# Synthesised detector ID for "the recipe asked for one required
# pattern to fire and none did". The ``meta:`` prefix marks it as
# coming from the verdict resolver, not from any tier directly.
DETECTOR_MISSING_PASS_SIGNAL = "meta:missing_pass_signal"


@dataclass
class VerdictInputs:
    """Per-tier outputs the resolver merges.

    Order MATTERS for ``failure_detectors_fired``: the resolver
    walks the lists in the order they appear on this struct
    (tier1, tier2, tier3, tier4, custom-fail, meta) and appends in
    that order so the final list is reproducible across runs
    given the same inputs.

    ``custom_required_patterns`` is the list of ``CompiledPattern``
    objects with ``required_for_pass=True``, regardless of whether
    they fired. The resolver compares against
    ``custom_result.fired_required_ids`` to decide whether to
    inject :data:`DETECTOR_MISSING_PASS_SIGNAL`.
    """

    tier1: list[str] = field(default_factory=list)
    tier2: list[str] = field(default_factory=list)
    tier3: list[str] = field(default_factory=list)
    tier4: list[str] = field(default_factory=list)
    custom_result: CustomScanResult = field(default_factory=CustomScanResult)
    custom_required_patterns: tuple[CompiledPattern, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """Final resolution: ``verdict``, fired-detector lists, capture.

    Each list is freshly constructed (not a reference to a
    caller's mutable list) so a downstream consumer can mutate the
    result without affecting cached classifier state.
    """

    verdict: str  # "pass" | "fail"
    failure_detectors_fired: list[str]
    warn_detectors_fired: list[str]
    capture: dict[str, str | float | int]


def resolve(inputs: VerdictInputs) -> Verdict:
    """Apply the precedence rules and return the trial's verdict.

    Pure: no FS / subprocess. Suited to table-driven parametrised
    testing (rubric §2.B FR 2.8 — verdict precedence is a hard
    gate).
    """
    failures: list[str] = []
    failures.extend(inputs.tier1)
    failures.extend(inputs.tier2)
    failures.extend(inputs.tier3)
    failures.extend(inputs.tier4)
    failures.extend(inputs.custom_result.fail_detectors)

    # required_for_pass: every pattern in the recipe whose flag is
    # True must have fired. If at least one didn't, synthesise
    # ``meta:missing_pass_signal`` so the trial fails with a
    # clear, namespaced reason. The check considers ONLY patterns
    # whose .required_for_pass is True; non-required patterns
    # have no effect here.
    required_ids = {p.detector_id for p in inputs.custom_required_patterns if p.required_for_pass}
    if required_ids and not (required_ids <= inputs.custom_result.fired_required_ids):
        failures.append(DETECTOR_MISSING_PASS_SIGNAL)

    warns = list(inputs.custom_result.warn_detectors)
    capture = dict(inputs.custom_result.capture)

    verdict = "fail" if failures else "pass"
    return Verdict(
        verdict=verdict,
        failure_detectors_fired=failures,
        warn_detectors_fired=warns,
        capture=capture,
    )


__all__ = [
    "DETECTOR_MISSING_PASS_SIGNAL",
    "Verdict",
    "VerdictInputs",
    "resolve",
]
