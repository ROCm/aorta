"""Contingency-table data structure and per-cell aggregation for triage matrices.

Consumes B1's :class:`aorta.run.TrialResult` list for a single cell and emits
a :class:`CellStats` record with the fields the matrix.md table, confound
detection, and matrix.json all need.

Step-time source order (per cell, per trial):

1. ``trial.result["step_times_ms"]`` -- B1 surfaces the workload's
   ``WorkloadResult.step_times_ms`` list here. Preferred signal for
   confound detection since it comes from the workload's own clocks.
   Tagged ``"per_step"``.
2. ``trial.result["elapsed_sec"] / trial.result["total_iterations"] * 1000``
   -- fallback when the workload didn't provide per-iteration step times.
   Tagged ``"elapsed_per_iter"``.
3. ``trial.wall_clock_sec / <steps>`` when both of the above are absent
   (happens for workloads that fail in ``setup()`` before they compute any
   timing at all); attributed to the cell's resolved step count. Folds
   setup / teardown into the per-step number, so the figure is honest about
   total time but **not** a clean iteration-rate signal. Tagged
   ``"wall_clock_total"``.

The chosen branch is recorded on :class:`CellStats` as ``step_time_source``
and persisted in matrix.json so confound classification can refuse to
compare cells whose timing came from different fallback levels (mixing
"per_step" against "wall_clock_total" is comparing different kinds of
numbers; see :func:`aorta.triage.confound.classify`).

A trial is counted as a failure (``failed_count += 1``) if either its
``exit_status != "ok"`` OR the wrapped ``WorkloadResult.passed`` is False.
The aggregator does NOT inspect *why* a trial failed (NaN vs corruption vs
divergence vs infrastructure crash); that's :class:`WorkloadResult` /
``exit_status`` territory and is preserved separately via the
``exit_status_counts`` histogram so callers can distinguish e.g.
``workload_failed`` from ``infrastructure_failed`` when triaging.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

StepTimeSource = Literal["per_step", "elapsed_per_iter", "wall_clock_total", "missing"]

# Lower index == higher fidelity. The cell-level ``step_time_source`` is the
# WORST source any trial in the cell actually contributed samples from -- if
# even one trial fell back to ``wall_clock_total``, the cell's mean folds
# setup/teardown for at least one trial and we can't honestly claim per-step
# fidelity for the aggregate. ``missing`` is reported only when no trial
# produced any timing data at all.
_SOURCE_RANK: dict[str, int] = {
    "per_step": 0,
    "elapsed_per_iter": 1,
    "wall_clock_total": 2,
    "missing": 3,
}


@dataclass(frozen=True)
class CellStats:
    """Aggregated statistics for one matrix cell.

    ``error`` is non-None when the whole cell failed (docker pull failure,
    environment resolve error, etc.) -- the matrix row is preserved so the
    matrix is complete but all numeric fields are zero / NaN.

    ``step_times_ms`` is the concatenation of every trial's step-time
    samples. Kept on the dataclass so matrix.json can embed the raw series
    for downstream analysis; matrix.md shows only the mean. The min / max /
    p50 / p90 / p99 fields summarise this series for the matrix.json
    consumer without forcing them to recompute.

    ``exit_status_counts`` is a histogram keyed by ``TrialResult.exit_status``
    (``"ok"``, ``"workload_failed"``, ``"infrastructure_failed"``, ...). It
    lets matrix.json consumers distinguish failure modes that ``failure_rate``
    alone collapses into a single number.

    ``step_time_source`` records which branch of the fallback ladder
    populated ``step_times_ms`` -- ``"per_step"``, ``"elapsed_per_iter"``,
    ``"wall_clock_total"``, or ``"missing"``. When trials in the same cell
    used different branches the cell-level value is the worst (lowest
    fidelity) source any trial fell back to. Confound classification reads
    this field to refuse cell-vs-baseline ratios whose numerators and
    denominators were derived from incomparable signals (issue #160 /
    Sonbol's review on PR #160).
    """

    name: str
    mitigations: tuple[str, ...]
    environment: str
    extra_env: dict[str, str]
    resolved_env_vars: dict[str, str]
    trials: int
    passed_count: int
    failed_count: int
    mean_step_time_ms: float
    std_step_time_ms: float
    min_step_time_ms: float
    max_step_time_ms: float
    p50_step_time_ms: float
    p90_step_time_ms: float
    p99_step_time_ms: float
    mean_wall_clock_sec: float
    exit_status_counts: dict[str, int] = field(default_factory=dict)
    step_times_ms: list[float] = field(default_factory=list)
    trial_paths: list[str] = field(default_factory=list)
    error: str | None = None
    step_time_source: StepTimeSource = "missing"

    @property
    def failure_rate(self) -> float:
        """Fraction of trials that failed for ANY reason. 0.0 for an empty cell.

        Counts every trial whose ``exit_status != "ok"`` or whose wrapped
        ``WorkloadResult.passed`` is False. This is *not* a NaN-specific
        rate -- :class:`aorta.run.WorkloadResult.failure_count` is generic
        (NaN / corruption / divergence / infrastructure crash) and the
        aggregator does not try to disambiguate. Callers that need to
        distinguish failure modes should read ``exit_status_counts``.
        """
        if self.trials == 0:
            return 0.0
        return self.failed_count / self.trials


def _step_times_from_trial(trial: Any, effective_steps: int) -> tuple[list[float], StepTimeSource]:
    """Pull per-step times from a trial result, applying the fallback ladder.

    Returns ``(times, source)`` where ``source`` names which branch fired:
    ``"per_step"``, ``"elapsed_per_iter"``, ``"wall_clock_total"``, or
    ``"missing"`` (the trial produced no usable timing). The aggregator
    uses ``source`` to compute the cell-level ``step_time_source`` so
    downstream confound detection can reject incomparable comparisons.
    """
    result = getattr(trial, "result", None)
    if isinstance(result, dict):
        times = result.get("step_times_ms")
        if isinstance(times, list) and times:
            cleaned = [float(t) for t in times if isinstance(t, (int, float))]
            if cleaned:
                return cleaned, "per_step"
        iters = result.get("total_iterations")
        elapsed = result.get("elapsed_sec")
        if (
            isinstance(iters, int)
            and iters > 0
            and isinstance(elapsed, (int, float))
            and elapsed > 0
        ):
            return [float(elapsed) / iters * 1000.0], "elapsed_per_iter"
    wall = getattr(trial, "wall_clock_sec", 0.0) or 0.0
    if wall > 0 and effective_steps > 0:
        return [float(wall) / effective_steps * 1000.0], "wall_clock_total"
    return [], "missing"


def _reduce_step_time_sources(sources: list[StepTimeSource]) -> StepTimeSource:
    """Pick the worst (lowest-fidelity) source any contributing trial used.

    Trials that produced no samples (``"missing"``) are filtered out so a
    single setup-crashing trial in an otherwise-healthy cell does not poison
    the cell's source label. If every trial was ``"missing"`` (or ``sources``
    is empty), the cell is reported as ``"missing"``.
    """
    contributing = [s for s in sources if s != "missing"]
    if not contributing:
        return "missing"
    return max(contributing, key=lambda s: _SOURCE_RANK.get(s, 99))


def _trial_passed(trial: Any) -> bool:
    """A trial passed iff its exit_status is ok AND its WorkloadResult.passed is True."""
    status = getattr(trial, "exit_status", None)
    if status != "ok":
        return False
    result = getattr(trial, "result", None)
    if isinstance(result, dict):
        passed = result.get("passed")
        if passed is False:
            return False
    return True


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    data = sorted(samples)
    k = (len(data) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] + (data[c] - data[f]) * (k - f)


def aggregate_cell(
    name: str,
    mitigations: tuple[str, ...],
    environment: str,
    extra_env: dict[str, str],
    resolved_env_vars: dict[str, str],
    trials: list[Any],
    effective_steps: int,
    trial_paths: list[str] | None = None,
    error: str | None = None,
) -> CellStats:
    """Aggregate a list of TrialResult-shaped objects into a :class:`CellStats`.

    Any object with ``exit_status``, ``wall_clock_sec``, and ``result`` (dict)
    is accepted -- the aggregator does not import :class:`aorta.run.TrialResult`
    so tests can pass plain dataclasses or ``SimpleNamespace`` stand-ins.

    Args:
        name: Cell name from the recipe.
        mitigations: Tuple of mitigation names applied.
        environment: Resolved environment name (possibly ``_inline_<hash>``).
        extra_env: Ad-hoc overrides from the cell (for audit, recorded as-is).
        resolved_env_vars: Final env-var set applied to the trials (union of
            mitigation bundles + ``extra_env``).
        trials: List of trial results. Empty list is allowed for error cells.
        effective_steps: The per-trial step count the cell was configured
            with; used for the step-time fallback when the workload did not
            surface per-step times.
        trial_paths: Optional list of filesystem paths to per-trial JSON
            files; recorded in matrix.json.
        error: Cell-level error message; when set, the cell is marked as an
            error row and all numeric aggregates are forced to zero / 0.0.

    Returns:
        :class:`CellStats` populated from the trials.
    """
    if error is not None:
        return CellStats(
            name=name,
            mitigations=mitigations,
            environment=environment,
            extra_env=dict(extra_env),
            resolved_env_vars=dict(resolved_env_vars),
            trials=len(trials),
            passed_count=0,
            failed_count=len(trials),
            mean_step_time_ms=0.0,
            std_step_time_ms=0.0,
            min_step_time_ms=0.0,
            max_step_time_ms=0.0,
            p50_step_time_ms=0.0,
            p90_step_time_ms=0.0,
            p99_step_time_ms=0.0,
            mean_wall_clock_sec=0.0,
            exit_status_counts={},
            step_times_ms=[],
            trial_paths=list(trial_paths or []),
            error=error,
            step_time_source="missing",
        )

    trial_count = len(trials)
    passed = sum(1 for t in trials if _trial_passed(t))
    failed = trial_count - passed

    all_step_times: list[float] = []
    wall_clocks: list[float] = []
    status_counter: Counter[str] = Counter()
    trial_sources: list[StepTimeSource] = []
    for trial in trials:
        times, source = _step_times_from_trial(trial, effective_steps)
        all_step_times.extend(times)
        trial_sources.append(source)
        wall = getattr(trial, "wall_clock_sec", 0.0) or 0.0
        wall_clocks.append(float(wall))
        # Histogram by raw exit_status so callers can distinguish e.g.
        # "workload_failed" (the workload returned a failed WorkloadResult)
        # from "infrastructure_failed" (run_trials never got a result back).
        # Falling back to "unknown" keeps the histogram total == trial_count
        # even for stand-in trial objects that omit the attribute.
        status = getattr(trial, "exit_status", None) or "unknown"
        status_counter[str(status)] += 1

    cell_source = _reduce_step_time_sources(trial_sources)

    if all_step_times:
        mean_step = float(statistics.fmean(all_step_times))
        std_step = float(statistics.pstdev(all_step_times)) if len(all_step_times) > 1 else 0.0
        min_step = float(min(all_step_times))
        max_step = float(max(all_step_times))
        p50 = _percentile(all_step_times, 0.50)
        p90 = _percentile(all_step_times, 0.90)
        p99 = _percentile(all_step_times, 0.99)
    else:
        mean_step = std_step = min_step = max_step = p50 = p90 = p99 = 0.0

    mean_wall = float(statistics.fmean(wall_clocks)) if wall_clocks else 0.0

    return CellStats(
        name=name,
        mitigations=mitigations,
        environment=environment,
        extra_env=dict(extra_env),
        resolved_env_vars=dict(resolved_env_vars),
        trials=trial_count,
        passed_count=passed,
        failed_count=failed,
        mean_step_time_ms=mean_step,
        std_step_time_ms=std_step,
        min_step_time_ms=min_step,
        max_step_time_ms=max_step,
        p50_step_time_ms=p50,
        p90_step_time_ms=p90,
        p99_step_time_ms=p99,
        mean_wall_clock_sec=mean_wall,
        exit_status_counts=dict(status_counter),
        step_times_ms=all_step_times,
        trial_paths=list(trial_paths or []),
        error=None,
        step_time_source=cell_source,
    )


__all__ = ["CellStats", "StepTimeSource", "aggregate_cell"]
