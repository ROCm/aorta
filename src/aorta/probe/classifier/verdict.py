"""Verdict precedence resolver for ``aorta probe`` Phase 2 (issue #188).

Combines the per-tier detector outputs into the final
``result.json`` shape. The three-way verdict (issue #230) is one of
``pass`` / ``fail`` / ``error``:

(a) ``fail`` -- the event of interest manifested. Any Tier 1–4
    detector fires (other than the *error* detectors below) OR any
    ``custom_patterns[*]`` with ``on_match: fail`` fires.

(b) ``error`` -- the trial broke for an infrastructure reason and
    produced *no valid observation* of the thing under test: the
    command never launched (``tier1:exec_failed``) or it ran to the
    deadline without the hang monitor recognising a hang
    (``tier1:timeout`` with no co-firing ``tier2:hang``). The
    error detectors that :func:`resolve` recognises are exactly
    :data:`ERROR_DETECTOR_IDS` -- :func:`partition_detectors` treats no
    other ID as an error. The ``error_detectors_fired`` list in
    ``result.json`` can still carry additional ``meta:`` infra-error IDs,
    but those are written **directly** by a workload that bypasses
    :func:`resolve` entirely (e.g.
    :class:`aorta.workloads.SubprocessWorkload` emits a hand-built
    ``error`` ``result.json`` with ``meta:env_file_validation_failed``);
    they are not produced by, and do not flow through, this resolver.
    ``error`` is excluded from the matrix event-rate denominator so an
    infra flake doesn't inflate (or deflate) the reproduction rate.

(c) ``pass`` -- nothing fired.

Precedence is **fail > error > pass**: if any genuine failure
detector fired, the trial failed *even if* an error detector also
fired (a timeout that the monitor recognised as a hang fires both
``tier1:timeout`` and ``tier2:hang`` -> the hang wins -> ``fail``).
Only when error detectors fired and no failure detector did does
the verdict become ``error``.

``result.json`` carries two ordered lists: ``failure_detectors_fired``
(the fail signals) and ``error_detectors_fired`` (the error signals),
each in a fixed encounter order: Tier 1, Tier 2, Tier 3, Tier 4,
then Tier 5 (``custom_patterns`` failures). The order is set
explicitly by :func:`resolve` as a fixed tier sequence (not by the
:class:`VerdictInputs` field order), independent of when each tier
physically fires relative to the workload exit. The in-flight Tier 2
hang monitor contributes its detector IDs to ``inputs.tier2`` before
:func:`resolve` is called, so even though the predicate fires
mid-run, the IDs always land between Tier 1 and Tier 3 in the
serialised list.

``custom_patterns[*]`` with ``required_for_pass: true`` AND none of
them fired → add ``meta:missing_pass_signal`` to
``failure_detectors_fired`` AND ``verdict = "fail"``.

``on_match: warn`` populates ``warn_detectors_fired`` only.
``on_match: info`` populates ``capture`` only.

The resolver consumes :class:`VerdictInputs` (a plain bundle of
per-tier outputs) and returns :class:`Verdict`. Both are
dataclasses so the call site stays declarative and unit tests can
build inputs in one line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aorta.probe.classifier.tier1_process import (
    DETECTOR_EXEC_FAILED,
    DETECTOR_TIMEOUT,
)
from aorta.probe.classifier.tier5_custom import CompiledPattern, CustomScanResult

# Synthesised detector ID for "the recipe asked for one required
# pattern to fire and none did". The ``meta:`` prefix marks it as
# coming from the verdict resolver, not from any tier directly.
DETECTOR_MISSING_PASS_SIGNAL = "meta:missing_pass_signal"

# Detector IDs that mean "no valid observation" rather than "the event
# fired" (issue #230). A trial whose only fired detectors are in this
# set resolves to ``error``; one that also fired a genuine failure
# detector resolves to ``fail`` (fail > error). Keeping this set tiny
# and explicit is deliberate -- a plain non-zero exit
# (``tier1:exit_nonzero``), a signal death, a coredump, a kernel fault,
# or a log-pattern match are all *valid observations of a failure* and
# stay ``fail``. Only launch failures and unrecognised timeouts are
# infrastructure noise.
ERROR_DETECTOR_IDS = frozenset({DETECTOR_TIMEOUT, DETECTOR_EXEC_FAILED})

# Canonical three-way verdict vocabulary (issue #230). The single source
# of truth downstream code (matrix aggregation, the ``stop_after`` rule,
# the verify-run-output skill) validates against, so a fourth bucket can
# never be introduced by a typo in one layer. Ordered by precedence for
# readability; membership, not order, is the contract.
VALID_VERDICTS = frozenset({"fail", "error", "pass"})


def partition_detectors(fired: list[str]) -> tuple[list[str], list[str]]:
    """Split fired detector IDs into ``(failure, error)`` preserving order.

    A detector is an *error* signal iff it is in
    :data:`ERROR_DETECTOR_IDS`; everything else is a *failure* signal.
    Shared by :func:`resolve` and the Tier-1-only fallback in
    :mod:`aorta.workloads._subprocess` so both agree on the three-way
    split.
    """
    failures = [d for d in fired if d not in ERROR_DETECTOR_IDS]
    errors = [d for d in fired if d in ERROR_DETECTOR_IDS]
    return failures, errors


def verdict_from_detectors(failures: list[str], errors: list[str]) -> str:
    """Apply the **fail > error > pass** precedence to split detector lists."""
    if failures:
        return "fail"
    if errors:
        return "error"
    return "pass"


@dataclass
class VerdictInputs:
    """Per-tier outputs the resolver merges.

    Field order on this struct does NOT drive ``failure_detectors_fired``
    ordering: :func:`resolve` appends the failure lists in a fixed tier
    sequence (tier1, tier2, tier3, tier4, custom-fail, then the
    synthesised ``meta:`` signal), so the serialised list is reproducible
    across runs given the same inputs regardless of how the fields are
    laid out here.

    ``tier3_warn`` carries advisory Tier-3 detectors (e.g.
    ``tier3:vram_growth``) that contribute ONLY to
    ``warn_detectors_fired`` -- they never appear in
    ``failure_detectors_fired`` and never flip the verdict to fail.

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
    # Advisory Tier-3 detectors (e.g. ``tier3:vram_growth``) that surface as
    # warns rather than failures -- they never flip the verdict to fail.
    tier3_warn: list[str] = field(default_factory=list)
    custom_result: CustomScanResult = field(default_factory=CustomScanResult)
    custom_required_patterns: tuple[CompiledPattern, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """Final resolution: ``verdict``, fired-detector lists, capture.

    Each list is freshly constructed (not a reference to a
    caller's mutable list) so a downstream consumer can mutate the
    result without affecting cached classifier state.

    ``error_detectors_fired`` carries the infrastructure-error signals
    (see :data:`ERROR_DETECTOR_IDS`) separately from the genuine
    failure signals in ``failure_detectors_fired`` so downstream
    tooling can tell "the bug reproduced" from "the trial never
    validly ran".
    """

    verdict: str  # "pass" | "fail" | "error"
    failure_detectors_fired: list[str]
    warn_detectors_fired: list[str]
    capture: dict[str, str | float | int]
    error_detectors_fired: list[str] = field(default_factory=list)


def resolve(inputs: VerdictInputs) -> Verdict:
    """Apply the precedence rules and return the trial's verdict.

    Pure: no FS / subprocess. Suited to table-driven parametrised
    testing (rubric §2.B FR 2.8 — verdict precedence is a hard
    gate).
    """
    fired: list[str] = []
    fired.extend(inputs.tier1)
    fired.extend(inputs.tier2)
    fired.extend(inputs.tier3)
    fired.extend(inputs.tier4)
    fired.extend(inputs.custom_result.fail_detectors)

    # required_for_pass: every pattern in the recipe whose flag is
    # True must have fired. If at least one didn't, synthesise
    # ``meta:missing_pass_signal`` so the trial fails with a
    # clear, namespaced reason. The check considers ONLY patterns
    # whose .required_for_pass is True; non-required patterns
    # have no effect here. (A missing required signal is a genuine
    # failure, never an infra error.)
    required_ids = {p.detector_id for p in inputs.custom_required_patterns if p.required_for_pass}
    if required_ids and not (required_ids <= inputs.custom_result.fired_required_ids):
        fired.append(DETECTOR_MISSING_PASS_SIGNAL)

    # Built-in advisory (warn) detectors precede user custom warns, mirroring
    # the tier-before-custom ordering used for failures above.
    warns = list(inputs.tier3_warn)
    warns.extend(inputs.custom_result.warn_detectors)
    capture = dict(inputs.custom_result.capture)

    # Partition into genuine failures vs infrastructure errors, then
    # apply fail > error > pass precedence (issue #230).
    failures, errors = partition_detectors(fired)
    return Verdict(
        verdict=verdict_from_detectors(failures, errors),
        failure_detectors_fired=failures,
        warn_detectors_fired=warns,
        capture=capture,
        error_detectors_fired=errors,
    )


__all__ = [
    "DETECTOR_MISSING_PASS_SIGNAL",
    "ERROR_DETECTOR_IDS",
    "VALID_VERDICTS",
    "Verdict",
    "VerdictInputs",
    "partition_detectors",
    "resolve",
    "verdict_from_detectors",
]
