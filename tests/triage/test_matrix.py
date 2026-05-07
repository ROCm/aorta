"""Tests for src/aorta/triage/matrix.py: CellStats + aggregate_cell."""

from __future__ import annotations

from types import SimpleNamespace

from aorta.triage.matrix import CellStats, aggregate_cell


def _trial(
    passed: bool = True,
    exit_status: str = "ok",
    step_times_ms: list[float] | None = None,
    total_iterations: int | None = None,
    elapsed_sec: float | None = None,
    wall_clock_sec: float = 1.0,
):
    """Build a TrialResult-shaped stand-in. aggregate_cell uses duck typing."""
    result: dict = {"passed": passed}
    if step_times_ms is not None:
        result["step_times_ms"] = step_times_ms
    if total_iterations is not None:
        result["total_iterations"] = total_iterations
    if elapsed_sec is not None:
        result["elapsed_sec"] = elapsed_sec
    return SimpleNamespace(
        exit_status=exit_status,
        wall_clock_sec=wall_clock_sec,
        result=result,
    )


def _default_call(**overrides):
    kwargs = {
        "name": "cell",
        "mitigations": ("none",),
        "environment": "local",
        "extra_env": {},
        "resolved_env_vars": {},
        "trials": [],
        "effective_steps": 100,
    }
    kwargs.update(overrides)
    return aggregate_cell(**kwargs)


def test_all_pass_no_failures():
    trials = [_trial(passed=True) for _ in range(4)]
    stats = _default_call(trials=trials)
    assert stats.trials == 4
    assert stats.passed_count == 4
    assert stats.failed_count == 0
    assert stats.failure_rate == 0.0
    assert stats.error is None


def test_mixed_pass_fail_counts_correctly():
    trials = [_trial(passed=True), _trial(passed=False), _trial(passed=True)]
    stats = _default_call(trials=trials)
    assert stats.passed_count == 2
    assert stats.failed_count == 1
    assert abs(stats.failure_rate - (1 / 3)) < 1e-9


def test_infrastructure_failed_exit_status_counts_as_failure():
    trials = [
        _trial(passed=True, exit_status="ok"),
        _trial(passed=True, exit_status="infrastructure_failed"),
    ]
    stats = _default_call(trials=trials)
    assert stats.passed_count == 1
    assert stats.failed_count == 1


def test_step_times_preferred_over_fallback():
    trials = [_trial(step_times_ms=[100.0, 200.0, 300.0], wall_clock_sec=10.0)]
    stats = _default_call(trials=trials)
    assert stats.mean_step_time_ms == 200.0
    assert stats.p50_step_time_ms == 200.0
    # p99 at n=3 interpolates between idx 1 and idx 2 -> close to 300
    assert 290.0 < stats.p99_step_time_ms <= 300.0
    assert stats.std_step_time_ms > 0


def test_step_times_fallback_to_total_iterations_and_elapsed():
    trials = [_trial(total_iterations=10, elapsed_sec=1.0)]
    stats = _default_call(trials=trials)
    # 1.0s / 10 iters = 100ms/iter
    assert abs(stats.mean_step_time_ms - 100.0) < 1e-9


def test_step_times_fallback_to_wall_clock_over_steps():
    # No step_times_ms, no total_iterations/elapsed -> wall_clock / effective_steps
    trials = [_trial(wall_clock_sec=2.0)]
    stats = _default_call(trials=trials, effective_steps=100)
    # 2.0s / 100 steps = 20ms/step
    assert abs(stats.mean_step_time_ms - 20.0) < 1e-9


def test_no_timing_data_produces_zero():
    trials = [_trial(wall_clock_sec=0.0)]
    stats = _default_call(trials=trials)
    assert stats.mean_step_time_ms == 0.0
    assert stats.p50_step_time_ms == 0.0


def test_error_cell_preserves_row():
    stats = _default_call(
        trials=[],
        error="docker pull failed: image not found",
    )
    assert isinstance(stats, CellStats)
    assert stats.error is not None
    assert stats.passed_count == 0
    assert stats.failed_count == 0  # len(trials) == 0
    assert stats.trials == 0


def test_error_cell_zeroes_all_numeric_stats():
    stats = _default_call(
        trials=[_trial()],  # even with trials present
        error="whole cell failed to start",
    )
    assert stats.mean_step_time_ms == 0.0
    assert stats.std_step_time_ms == 0.0
    assert stats.min_step_time_ms == 0.0
    assert stats.max_step_time_ms == 0.0
    assert stats.p50_step_time_ms == 0.0
    assert stats.p90_step_time_ms == 0.0
    assert stats.p99_step_time_ms == 0.0
    assert stats.mean_wall_clock_sec == 0.0
    assert stats.exit_status_counts == {}


def test_resolved_env_vars_and_extra_env_recorded():
    stats = _default_call(
        extra_env={"X": "1"},
        resolved_env_vars={"HSA_XNACK": "1", "X": "1"},
        trials=[_trial()],
    )
    assert stats.extra_env == {"X": "1"}
    assert stats.resolved_env_vars == {"HSA_XNACK": "1", "X": "1"}


def test_trial_paths_recorded():
    paths = ["/tmp/cell/trial_0.json", "/tmp/cell/trial_1.json"]
    stats = _default_call(trials=[_trial(), _trial()], trial_paths=paths)
    assert stats.trial_paths == paths


def test_step_time_percentile_series():
    """With >1 step-time samples across trials we get a real p50/p99 computation."""
    trials = [
        _trial(step_times_ms=[100.0, 200.0]),
        _trial(step_times_ms=[300.0, 400.0]),
    ]
    stats = _default_call(trials=trials)
    assert stats.mean_step_time_ms == 250.0
    assert stats.step_times_ms == [100.0, 200.0, 300.0, 400.0]
    assert stats.std_step_time_ms > 0


def test_min_max_p90_step_times_populated():
    """Pinning the new min/max/p90 fields requested in PR review."""
    trials = [
        _trial(
            step_times_ms=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
        )
    ]
    stats = _default_call(trials=trials)
    assert stats.min_step_time_ms == 100.0
    assert stats.max_step_time_ms == 1000.0
    # p90 of 10 evenly-spaced samples interpolates between idx 8 (900) and 9 (1000)
    assert 900.0 <= stats.p90_step_time_ms <= 1000.0
    # p99 should be even closer to the max
    assert stats.p99_step_time_ms >= stats.p90_step_time_ms


def test_exit_status_histogram_distinguishes_failure_modes():
    """Pin the new exit_status_counts field. Failure rate alone is generic;
    callers triaging a cell need to tell workload_failed from infrastructure_failed.
    """
    trials = [
        _trial(passed=True, exit_status="ok"),
        _trial(passed=True, exit_status="ok"),
        _trial(passed=False, exit_status="workload_failed"),
        _trial(passed=False, exit_status="infrastructure_failed"),
        _trial(passed=False, exit_status="infrastructure_failed"),
    ]
    stats = _default_call(trials=trials)
    assert stats.exit_status_counts == {
        "ok": 2,
        "workload_failed": 1,
        "infrastructure_failed": 2,
    }
    # Histogram total must equal trial count, by construction.
    assert sum(stats.exit_status_counts.values()) == stats.trials
    # And the failure_rate is consistent with the histogram (3 of 5 not "ok").
    assert abs(stats.failure_rate - 0.6) < 1e-9


def test_failure_rate_docstring_is_general_not_nan_specific():
    """Regression: the property must NOT be called nan_rate.

    Prior naming claimed "NaN rate" but counted every non-ok exit. Pin the
    new name so future renames don't silently re-introduce the misnomer.
    """
    assert hasattr(CellStats, "failure_rate")
    assert not hasattr(CellStats, "nan_rate")


# ---- step_time_source lineage --------------------------------------------
#
# Sonbol's review on PR #160 flagged that ``_step_times_from_trial`` falls
# through three different timing signals -- per-step, elapsed/iter, and
# wall-clock-total -- and that comparing a cell that landed on branch 1
# against one that only reached branch 3 silently mixes iteration time with
# setup-and-teardown time. The cell-level ``step_time_source`` field labels
# which branch the cell actually used so confound detection can refuse the
# bad comparison instead of pretending the ratio is meaningful.


def test_step_time_source_per_step_when_step_times_ms_present():
    """Branch 1: workload reports per-step times -> source is 'per_step'."""
    trials = [_trial(step_times_ms=[100.0, 200.0])]
    stats = _default_call(trials=trials)
    assert stats.step_time_source == "per_step"


def test_step_time_source_elapsed_per_iter_fallback():
    """Branch 2: no step_times_ms but elapsed/iters available."""
    trials = [_trial(total_iterations=10, elapsed_sec=1.0)]
    stats = _default_call(trials=trials)
    assert stats.step_time_source == "elapsed_per_iter"


def test_step_time_source_wall_clock_total_fallback():
    """Branch 3: only wall_clock_sec available; folds setup + teardown in."""
    trials = [_trial(wall_clock_sec=2.0)]
    stats = _default_call(trials=trials, effective_steps=100)
    assert stats.step_time_source == "wall_clock_total"


def test_step_time_source_missing_when_no_timing_data():
    """No samples produced -> source must be 'missing', not a default lie."""
    trials = [_trial(wall_clock_sec=0.0)]
    stats = _default_call(trials=trials)
    assert stats.step_time_source == "missing"
    assert stats.mean_step_time_ms == 0.0


def test_step_time_source_is_worst_across_mixed_trials():
    """If any contributing trial fell back to a lower-fidelity branch the
    cell-level source must reflect that.

    The cell's mean folds together one trial of true per-step samples and
    one trial of wall-clock-divided-by-step-count -- the latter includes
    setup and teardown. We can't honestly claim per-step fidelity for the
    aggregate, so the cell labels itself by the worst source any contributing
    trial used.
    """
    trials = [
        _trial(step_times_ms=[100.0, 200.0]),
        _trial(wall_clock_sec=1.5),  # branch 3
    ]
    stats = _default_call(trials=trials, effective_steps=100)
    assert stats.step_time_source == "wall_clock_total"


def test_step_time_source_ignores_missing_trials_when_others_have_data():
    """A single setup-crashing trial (no timing at all) must not poison the
    cell's source label when the rest of the cell measured cleanly.

    'missing' is a per-trial honest report ("this trial produced nothing"),
    not a fidelity tier; a cell with three per-step trials and one missing
    trial is still a per-step cell.
    """
    trials = [
        _trial(step_times_ms=[100.0]),
        _trial(step_times_ms=[110.0]),
        _trial(wall_clock_sec=0.0),  # produced nothing
    ]
    stats = _default_call(trials=trials)
    assert stats.step_time_source == "per_step"


def test_step_time_source_for_error_cell_is_missing():
    """Error cells short-circuit aggregation; the source must report 'missing'
    rather than a stale default that implies real timing exists."""
    stats = _default_call(trials=[], error="docker pull failed")
    assert stats.step_time_source == "missing"
