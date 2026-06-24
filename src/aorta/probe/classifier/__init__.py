"""Five-tier classifier for ``aorta probe`` (issue #188 Phase 2).

Single entry point :func:`classify_trial` consumed by
:class:`aorta.workloads._subprocess.SubprocessWorkload.run` post-exit.
The tiers are individually importable for unit tests and the
``aorta probe --list-patterns`` subcommand.

The classifier intentionally lives OUTSIDE the workload module so
the workload stays small and testable: the workload owns the
subprocess; the classifier owns the verdict. Cross-tier ordering
(``failure_detectors_fired`` reflects Tier 1 → Tier 2 → Tier 3 →
Tier 4 → Tier 5) lives in :mod:`aorta.probe.classifier.verdict`.

This package is the only module in :mod:`aorta.probe` whose name
contains "classifier"; per the rubric's engine-reuse gate, no
file under :mod:`aorta.probe` may include "runner" or "dispatcher"
in its name (the shared-engine test enforces this — the classifier
hangs off ``SubprocessWorkload.run()`` post-exit, never replaces
the dispatcher).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aorta.probe.classifier import (
    tier1_process,
    tier3_kernel,
    tier4_patterns,
    tier5_custom,
)
from aorta.probe.classifier.tier1_process import Tier1Context
from aorta.probe.classifier.tier2_hang import DETECTOR_HANG
from aorta.probe.classifier.tier3_kernel import (
    TIER3_WARN_DETECTOR_IDS,
    AmdSmiSnapshot,
    Tier3State,
)
from aorta.probe.classifier.tier5_custom import CompiledPattern
from aorta.probe.classifier.verdict import Verdict, VerdictInputs, resolve


@dataclass(frozen=True)
class TrialContext:
    """Inputs to :func:`classify_trial`.

    ``hang_detected`` is the post-monitor flag captured by
    :class:`aorta.probe.classifier.tier2_hang.HangMonitor`. Pure
    bool so the workload doesn't have to hold a live ``HangMonitor``
    reference after stopping it.

    ``tier3_state`` carries the "tier3 disabled" warning latch so
    the classifier can scan dmesg / amd-smi without double-logging
    across cells/trials (rubric §2.B FR 2.11). The runner owns
    one instance per ``aorta probe`` invocation.

    ``amd_smi_pre`` / ``amd_smi_post`` are the snapshots polled
    before and after the trial; either being None disables Tier 3
    GPU counters (fail-soft per FR 2.3).
    """

    exit_code: int
    timed_out: bool
    walltime_sec: float
    trial_dir: Path
    log_text: str
    custom_patterns: tuple[CompiledPattern, ...] = ()
    hang_detected: bool = False
    # Exec-time ``Popen`` failure -- the wrapped command never launched.
    # Routed to ``tier1:exec_failed`` (an error detector, issue #230) so
    # a command-not-found resolves to ``error``, not ``fail``.
    exec_failed: bool = False
    peak_vram_mib: int | None = None
    dmesg_text: str | None = None
    tier3_state: Tier3State | None = None
    amd_smi_pre: AmdSmiSnapshot | None = None
    amd_smi_post: AmdSmiSnapshot | None = None
    # Tier 3 detector IDs the caller has already collected (e.g. the
    # workload pre-invoked ``scan_dmesg`` with a known ``--since``
    # window and prefers not to round-trip through ``dmesg_text``).
    # Merged with the in-classifier scan results; empty tuple is the
    # legacy-equivalent no-op. Distinct from ``dmesg_text`` so callers
    # can supply IDs without the source text and the classifier can
    # supply the source text without the IDs -- the two paths union.
    tier3_extra: tuple[str, ...] = ()
    tier3_vram_growth: bool = True
    # Issue #229: operator-supplied detector-disable knobs. A disabled
    # whole tier (``disabled_tiers``, e.g. ``"tier3"``) is not evaluated
    # at all -- ``classify_trial`` skips its scan, and the workload
    # (:meth:`SubprocessWorkload.run`) skips the Tier-3 collection too, so
    # the side-effecting probes (dmesg / amd-smi) don't run. A disabled
    # detector id (``disabled_detectors``,
    # e.g. ``"tier2:hang"``) is filtered out of the tier's result after
    # the (cheap, side-effect-free) Tier 1-4 scan. Disabled Tier 5
    # ``custom:*`` detectors are instead filtered *before* the scan,
    # because a custom scan runs sandbox ``condition``s and populates
    # ``capture`` -- post-hoc filtering would let a silenced detector
    # still leave observable side effects. Either way a disabled detector
    # never reaches :func:`resolve`, so it cannot flip the verdict or
    # appear in ``failure_detectors_fired`` / ``warn_detectors_fired``.
    # Validated upstream by :mod:`aorta.probe.classifier.disables`.
    disabled_tiers: frozenset[str] = frozenset()
    disabled_detectors: frozenset[str] = frozenset()


def classify_trial(ctx: TrialContext) -> tuple[Verdict, dict[str, float]]:
    """Run all five tiers and resolve the trial's verdict.

    Returns ``(verdict, tier_durations_ms)`` where the second value
    is the wall-clock time spent in each tier (logged to
    ``result.json::tier_durations_ms`` for budget audits).

    Tier 3 is fail-soft on missing binaries; passing a None
    ``tier3_state`` skips Tier 3 entirely (rubric §2.B FR 2.3 +
    FR 2.11). Tier 5 is always invoked so its measured duration
    appears in ``tier_durations_ms``; with an empty
    ``custom_patterns`` tuple the call returns an empty
    :class:`~aorta.probe.classifier.tier5_custom.CustomScanResult`
    in microseconds.
    """
    import time

    tier_durations_ms: dict[str, float] = {}
    disabled_tiers = ctx.disabled_tiers
    disabled_detectors = ctx.disabled_detectors

    def _keep(ids: list[str]) -> list[str]:
        """Drop any operator-disabled detector ids, preserving order."""
        if not disabled_detectors:
            return ids
        return [d for d in ids if d not in disabled_detectors]

    t0 = time.perf_counter()
    if "tier1" in disabled_tiers:
        tier1: list[str] = []
    else:
        tier1 = _keep(
            tier1_process.detect(
                Tier1Context(
                    exit_code=ctx.exit_code,
                    timed_out=ctx.timed_out,
                    trial_dir=ctx.trial_dir,
                    exec_failed=ctx.exec_failed,
                )
            )
        )
    tier_durations_ms["tier1"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if "tier2" in disabled_tiers:
        tier2: list[str] = []
    else:
        tier2 = _keep([DETECTOR_HANG] if ctx.hang_detected else [])
    tier_durations_ms["tier2"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    tier3: list[str] = []
    if "tier3" not in disabled_tiers and ctx.tier3_state is not None:
        # Caller-pre-collected IDs come first so the call order in
        # ``failure_detectors_fired`` matches the chronological order
        # of detection (workload polls dmesg before classify_trial
        # runs; in-classifier scans run after).
        tier3.extend(ctx.tier3_extra)
        if ctx.dmesg_text:
            # Only call scan_dmesg_text when there's actual content --
            # an empty string is the workload's "no new dmesg lines"
            # signal and shouldn't redundantly re-run the regex set.
            tier3.extend(tier3_kernel.scan_dmesg_text(ctx.dmesg_text))
        if ctx.amd_smi_pre is not None and ctx.amd_smi_post is not None:
            # ``tier3:vram_growth`` is an operator-disable-able warn
            # detector. Honour the disable *before* the scan -- not only
            # via the post-hoc ``_keep`` filter below -- so a silenced
            # detector never runs its VRAM delta check, mirroring the
            # ``tier3_vram_growth`` recipe knob. (oyazdanb review)
            check_vram_growth = (
                ctx.tier3_vram_growth
                and tier3_kernel.DETECTOR_VRAM_GROWTH not in disabled_detectors
            )
            tier3.extend(
                tier3_kernel.scan_amd_smi(
                    ctx.tier3_state,
                    ctx.amd_smi_pre,
                    ctx.amd_smi_post,
                    check_vram_growth=check_vram_growth,
                )
            )
    tier3 = _keep(tier3)
    tier_durations_ms["tier3"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if "tier4" in disabled_tiers:
        tier4: list[str] = []
    else:
        tier4 = _keep(tier4_patterns.scan(ctx.log_text))
    tier_durations_ms["tier4"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    # ``tier5`` is the tier token for the custom-pattern scanner; when
    # disabled, skip the scan entirely (an empty result is the no-op the
    # resolver expects) so no ``custom:*`` id can fire.
    if "tier5" in disabled_tiers:
        custom_result = tier5_custom.CustomScanResult()
        active_custom_patterns: tuple[CompiledPattern, ...] = ()
    else:
        # Unlike Tiers 1-4 (cheap, side-effect-free scans we can filter
        # post-hoc), a custom detector's scan runs its sandbox
        # ``condition`` and populates ``capture``. Filtering disabled ids
        # out *before* the scan keeps "disabled == not evaluated" honest:
        # no condition runs and no capture surfaces for a silenced
        # detector. The same disable-filtered set also seeds the resolver's
        # required-pattern subset below, so a disabled ``required_for_pass``
        # pattern can't synthesise ``meta:missing_pass_signal``.
        if disabled_detectors:
            active_custom_patterns = tuple(
                p for p in ctx.custom_patterns if p.detector_id not in disabled_detectors
            )
        else:
            active_custom_patterns = ctx.custom_patterns
        custom_result = tier5_custom.scan(
            ctx.log_text,
            active_custom_patterns,
            exit_code=ctx.exit_code,
            walltime_sec=ctx.walltime_sec,
            peak_vram_mib=ctx.peak_vram_mib,
        )
    tier_durations_ms["tier5"] = (time.perf_counter() - t0) * 1000.0

    # Split Tier-3 IDs into hard failures vs. advisory warns. ``vram_growth``
    # is advisory (see TIER3_WARN_DETECTOR_IDS) so it never flips the verdict;
    # kernel-fault IDs stay failures. Preserve encounter order within each.
    tier3_fail = [d for d in tier3 if d not in TIER3_WARN_DETECTOR_IDS]
    tier3_warn = [d for d in tier3 if d in TIER3_WARN_DETECTOR_IDS]

    verdict = resolve(
        VerdictInputs(
            tier1=tier1,
            tier2=tier2,
            tier3=tier3_fail,
            tier4=tier4,
            tier3_warn=tier3_warn,
            custom_result=custom_result,
            # ``VerdictInputs.custom_required_patterns`` is contractually the
            # ``required_for_pass=True`` subset (the resolver only inspects
            # those). Narrow here rather than handing over the full active
            # set so the field matches its documented meaning. Derived from
            # the disable-filtered ``active_custom_patterns``, so a disabled
            # required pattern stays excluded.
            custom_required_patterns=tuple(
                p for p in active_custom_patterns if p.required_for_pass
            ),
        )
    )
    return verdict, tier_durations_ms


__all__ = [
    "TrialContext",
    "Verdict",
    "VerdictInputs",
    "classify_trial",
    "resolve",
    "tier1_process",
    "tier3_kernel",
    "tier4_patterns",
    "tier5_custom",
]
