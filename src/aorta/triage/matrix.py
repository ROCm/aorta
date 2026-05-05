"""Contingency-table data structure and per-cell aggregation for triage matrices.

Consumes B1's :class:`aorta.run.TrialResult` list for a single cell and emits
a :class:`CellStats` record with the fields the matrix.md table, confound
detection, and matrix.json all need.

Step-time source order (per cell, per trial):

1. ``trial.result["step_times_ms"]`` -- B1 surfaces the workload's
   ``WorkloadResult.step_times_ms`` list here. Preferred signal for
   confound detection since it comes from the workload's own clocks.
2. ``trial.result["elapsed_sec"] / trial.result["total_iterations"] * 1000``
   -- fallback when the workload didn't provide per-iteration step times.
3. ``trial.wall_clock_sec / <steps>`` when both of the above are absent
   (happens for workloads that fail in ``setup()`` before they compute any
   timing at all); attributed to the cell's resolved step count.

A trial is counted as a failure (``failed_count += 1``) if either its
``exit_status != "ok"`` OR the wrapped ``WorkloadResult.passed`` is False.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CellStats:
    """Aggregated statistics for one matrix cell.

    ``error`` is non-None when the whole cell failed (docker pull failure,
    environment resolve error, etc.) -- the matrix row is preserved so the
    matrix is complete but all numeric fields are zero/NaN.

    ``step_times_ms`` is the concatenation of every trial's step-time
    samples. Kept on the dataclass so matrix.json can embed the raw series
    for downstream analysis; matrix.md shows only the mean.
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
    p50_step_time_ms: float
    p99_step_time_ms: float
    mean_wall_clock_sec: float
    step_times_ms: list[float] = field(default_factory=list)
    trial_paths: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def nan_rate(self) -> float:
        """Fraction of trials that failed. 0.0 for an empty cell."""
        if self.trials == 0:
            return 0.0
        return self.failed_count / self.trials


def _step_times_from_trial(trial: Any, effective_steps: int) -> list[float]:
    """Pull per-step times from a trial result, applying the fallback ladder."""
    result = getattr(trial, "result", None)
    if isinstance(result, dict):
        times = result.get("step_times_ms")
        if isinstance(times, list) and times:
            return [float(t) for t in times if isinstance(t, (int, float))]
        iters = result.get("total_iterations")
        elapsed = result.get("elapsed_sec")
        if (
            isinstance(iters, int)
            and iters > 0
            and isinstance(elapsed, (int, float))
            and elapsed > 0
        ):
            return [float(elapsed) / iters * 1000.0]
    wall = getattr(trial, "wall_clock_sec", 0.0) or 0.0
    if wall > 0 and effective_steps > 0:
        return [float(wall) / effective_steps * 1000.0]
    return []


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
            p50_step_time_ms=0.0,
            p99_step_time_ms=0.0,
            mean_wall_clock_sec=0.0,
            step_times_ms=[],
            trial_paths=list(trial_paths or []),
            error=error,
        )

    trial_count = len(trials)
    passed = sum(1 for t in trials if _trial_passed(t))
    failed = trial_count - passed

    all_step_times: list[float] = []
    wall_clocks: list[float] = []
    for trial in trials:
        all_step_times.extend(_step_times_from_trial(trial, effective_steps))
        wall = getattr(trial, "wall_clock_sec", 0.0) or 0.0
        wall_clocks.append(float(wall))

    if all_step_times:
        mean_step = float(statistics.fmean(all_step_times))
        std_step = float(statistics.pstdev(all_step_times)) if len(all_step_times) > 1 else 0.0
        p50 = _percentile(all_step_times, 0.50)
        p99 = _percentile(all_step_times, 0.99)
    else:
        mean_step = std_step = p50 = p99 = 0.0

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
        p50_step_time_ms=p50,
        p99_step_time_ms=p99,
        mean_wall_clock_sec=mean_wall,
        step_times_ms=all_step_times,
        trial_paths=list(trial_paths or []),
        error=None,
    )


__all__ = ["CellStats", "aggregate_cell"]
