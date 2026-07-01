"""Tests for src/aorta/triage/matrix.py: CellStats + aggregate_cell."""

from __future__ import annotations

from types import SimpleNamespace

from aorta.triage.matrix import (
    OUTCOME_COMPLETED,
    OUTCOME_CRASHED_AFTER_ITERATIONS,
    OUTCOME_DID_NOT_RUN,
    OUTCOME_UNKNOWN,
    CellStats,
    aggregate_cell,
)


def _trial(
    passed: bool = True,
    exit_status: str = "ok",
    step_times_ms: list[float] | None = None,
    total_iterations: int | None = None,
    elapsed_sec: float | None = None,
    wall_clock_sec: float = 1.0,
    failure_details: list[dict] | None = None,
    main_work_started: bool | None = None,
    executed_iterations: int | None = None,
    configured_iterations: int | None = None,
):
    """Build a TrialResult-shaped stand-in. aggregate_cell uses duck typing."""
    result: dict = {"passed": passed}
    if step_times_ms is not None:
        result["step_times_ms"] = step_times_ms
    if total_iterations is not None:
        result["total_iterations"] = total_iterations
    if elapsed_sec is not None:
        result["elapsed_sec"] = elapsed_sec
    if failure_details is not None:
        result["failure_details"] = failure_details
    if main_work_started is not None:
        result["main_work_started"] = main_work_started
    if executed_iterations is not None:
        result["executed_iterations"] = executed_iterations
    if configured_iterations is not None:
        result["configured_iterations"] = configured_iterations
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


def test_infrastructure_failed_exit_status_counts_as_error():
    """Issue #230: infra crashes are ``error`` (no valid observation), not
    ``fail`` -- they're excluded from the event-rate denominator."""
    trials = [
        _trial(passed=True, exit_status="ok"),
        _trial(passed=True, exit_status="infrastructure_failed"),
    ]
    stats = _default_call(trials=trials)
    assert stats.passed_count == 1
    assert stats.failed_count == 0
    assert stats.error_count == 1
    # Event rate excludes the error trial from the denominator: 0 fails / 1
    # valid trial == 0.0 (not 0/2 either -- the error trial isn't counted).
    assert stats.failure_rate == 0.0
    assert stats.error_rate == 0.5


def test_workload_setup_failed_counts_as_error():
    """``workload_setup_failed`` -> ``error`` (setup died before the
    measurement; issue #230)."""
    trials = [
        _trial(passed=False, exit_status="workload_failed"),
        _trial(passed=False, exit_status="workload_setup_failed"),
    ]
    stats = _default_call(trials=trials)
    assert stats.passed_count == 0
    assert stats.failed_count == 1
    assert stats.error_count == 1
    # 1 genuine fail / 1 valid trial == 1.0; the setup-failed error is excluded.
    assert stats.failure_rate == 1.0
    assert stats.error_rate == 0.5


def test_all_error_cell_has_zero_event_rate():
    """A cell where every trial errored has event_rate 0.0 (no valid trials),
    not a division-by-zero."""
    trials = [_trial(passed=False, exit_status="infrastructure_failed") for _ in range(3)]
    stats = _default_call(trials=trials)
    assert stats.passed_count == 0
    assert stats.failed_count == 0
    assert stats.error_count == 3
    assert stats.failure_rate == 0.0
    assert stats.error_rate == 1.0
    assert stats.is_unreliable is True


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
    callers triaging a cell need to tell workload_failed from
    workload_setup_failed from infrastructure_failed.
    """
    trials = [
        _trial(passed=True, exit_status="ok"),
        _trial(passed=True, exit_status="ok"),
        _trial(passed=False, exit_status="workload_failed"),
        _trial(passed=False, exit_status="workload_setup_failed"),
        _trial(passed=False, exit_status="infrastructure_failed"),
        _trial(passed=False, exit_status="infrastructure_failed"),
    ]
    stats = _default_call(trials=trials)
    assert stats.exit_status_counts == {
        "ok": 2,
        "workload_failed": 1,
        "workload_setup_failed": 1,
        "infrastructure_failed": 2,
    }
    # Histogram total must equal trial count, by construction.
    assert sum(stats.exit_status_counts.values()) == stats.trials
    # Issue #230 three-way split: ok=2 (pass), workload_failed=1 (fail),
    # workload_setup_failed + 2x infrastructure_failed = 3 (error).
    assert stats.passed_count == 2
    assert stats.failed_count == 1
    assert stats.error_count == 3
    # Event rate excludes the 3 error trials: 1 fail / (2 pass + 1 fail) = 1/3.
    assert abs(stats.failure_rate - (1.0 / 3.0)) < 1e-9
    assert abs(stats.error_rate - (3.0 / 6.0)) < 1e-9


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


# ---- failure_hints aggregation -------------------------------------------
#
# Workloads that can't run (image / config mismatch, missing dependency)
# emit an explanatory `hint` string in `failure_details[*].hint`. The
# aggregator surfaces those at the cell level so matrix.md can show them
# without a reader having to open per-trial JSON.


def test_failure_hints_default_empty_when_no_hints():
    trials = [_trial(passed=True), _trial(passed=False)]
    stats = _default_call(trials=trials)
    assert stats.failure_hints == []


def test_failure_hints_collects_and_counts_duplicates():
    hint = "shampoo import failed: try shampoo_api='old'"
    trials = [
        _trial(passed=False, failure_details=[{"status": "crash", "hint": hint}]),
        _trial(passed=False, failure_details=[{"status": "crash", "hint": hint}]),
    ]
    stats = _default_call(trials=trials)
    assert stats.failure_hints == [(hint, 2)]


def test_failure_hints_preserve_first_seen_order_with_distinct_hints():
    h1 = "first hint"
    h2 = "second hint"
    trials = [
        _trial(passed=False, failure_details=[{"hint": h1}]),
        _trial(passed=False, failure_details=[{"hint": h2}]),
        _trial(passed=False, failure_details=[{"hint": h1}]),
    ]
    stats = _default_call(trials=trials)
    assert stats.failure_hints == [(h1, 2), (h2, 1)]


def test_failure_hints_skips_entries_without_hint_key():
    """A trial that crashed but didn't emit a hint contributes nothing.

    Pin: the aggregator must NOT invent a hint from status / stderr.
    """
    trials = [
        _trial(passed=False, failure_details=[{"status": "crash", "returncode": 1}]),
        _trial(passed=False, failure_details=[{"hint": "real hint"}]),
    ]
    stats = _default_call(trials=trials)
    assert stats.failure_hints == [("real hint", 1)]


def test_failure_hints_skips_none_and_empty_hint_values():
    trials = [
        _trial(passed=False, failure_details=[{"hint": None}]),
        _trial(passed=False, failure_details=[{"hint": ""}]),
    ]
    stats = _default_call(trials=trials)
    assert stats.failure_hints == []


def test_failure_hints_handles_multiple_details_per_trial():
    trials = [
        _trial(
            passed=False,
            failure_details=[{"hint": "h1"}, {"hint": "h2"}],
        ),
    ]
    stats = _default_call(trials=trials)
    assert stats.failure_hints == [("h1", 1), ("h2", 1)]


def test_failure_hints_dedupes_within_trial_so_count_is_bounded_by_trials():
    """A single trial that emits the same hint in two failure_details entries
    must contribute 1 to the count, not 2.

    The rendered ``({count}/{cell.trials} trials)`` would otherwise be
    nonsense (e.g. ``2/1 trials``) once a workload emits multiple
    ``failure_details`` per trial. Pin the per-trial-presence semantic
    so the fraction stays interpretable.
    """
    trials = [
        _trial(
            passed=False,
            failure_details=[{"hint": "x"}, {"hint": "x"}],
        ),
        _trial(
            passed=False,
            failure_details=[{"hint": "x"}],
        ),
    ]
    stats = _default_call(trials=trials)
    # Two trials emitted the hint, even though it appears 3 times across details.
    assert stats.failure_hints == [("x", 2)]


def test_failure_hints_for_error_cell_reflects_supplied_trials():
    """Error cells short-circuit numeric aggregation but still expose hints
    from any trials that did execute before the cell-level error fired."""
    trials = [_trial(passed=False, failure_details=[{"hint": "boom"}])]
    stats = _default_call(trials=trials, error="cell crashed")
    assert stats.failure_hints == [("boom", 1)]


# ---- did_not_run outcome aggregation (issue #173) ------------------------
#
# When a workload populates the new `main_work_started` /
# `executed_iterations` / `configured_iterations` fields the aggregator
# must (a) classify each trial against the platform-level outcome enum,
# (b) suppress the wall_clock_total step-time fallback for did_not_run
# trials so matrix.json never carries a fake iteration-time number, and
# (c) pre-render the cell-level Iters column. Workloads that don't
# populate the new fields must continue to behave exactly as today.


def test_platform_inference_did_not_run_for_legacy_setup_crash():
    """Legacy workload that died in setup -- main_work_started absent,
    but the observable signature (exit_status="workload_failed",
    step_times_ms=[], elapsed_sec=0.0) is unmistakably "didn't run".

    The platform must classify the trial as did_not_run from these
    signals alone, without requiring the workload to be updated to the
    new contract. This is the recom_repro nan-repro demo case from the
    issue: trials died on ImportError at 3.5s, produced no per-step
    times, no elapsed time, but did report passed=False.
    """
    trials = [
        _trial(
            passed=False,
            exit_status="workload_failed",
            wall_clock_sec=3.5,
            # main_work_started intentionally absent -- legacy workload
        )
        for _ in range(2)
    ]
    stats = _default_call(trials=trials, effective_steps=50)
    assert stats.outcome_counts == {OUTCOME_DID_NOT_RUN: 2}
    # The wall_clock_total fallback is suppressed for inferred
    # did_not_run trials, same as for explicit main_work_started=False.
    assert stats.mean_step_time_ms == 0.0
    assert stats.step_time_source == "missing"


def test_platform_inference_does_not_fire_when_step_times_present():
    """Even with non-ok exit_status, a trial that produced ANY per-step
    timing data is NOT inferred as did_not_run. That signature means the
    workload reached its measurement loop and crashed mid-run -- the
    wall_clock fallback path / explicit crashed_after_iterations
    classification handles it correctly. Pin the predicate's strictness.
    """
    trials = [
        _trial(
            passed=False,
            exit_status="workload_failed",
            step_times_ms=[100.0, 110.0, 95.0],
            wall_clock_sec=3.5,
        )
    ]
    stats = _default_call(trials=trials, effective_steps=50)
    assert stats.outcome_counts == {OUTCOME_UNKNOWN: 1}
    assert stats.mean_step_time_ms > 0  # per-step timing preserved


def test_platform_inference_does_not_fire_on_passing_trials():
    """A trial with exit_status="ok" never triggers inference, even if
    it has no step_times / elapsed (e.g. a workload that doesn't track
    timing). Inference is strictly for failure paths."""
    trials = [_trial(passed=True, exit_status="ok", wall_clock_sec=1.0)]
    stats = _default_call(trials=trials)
    assert stats.outcome_counts == {OUTCOME_UNKNOWN: 1}


def test_platform_inference_does_not_fire_when_elapsed_sec_present():
    """A workload that measured *something* (elapsed_sec > 0) is not
    inferred as did_not_run, even if it failed and produced no per-step
    times. That signature is a workload that finished some measurable
    work then crashed -- classification falls to the existing wall-clock
    fallback path."""
    trials = [
        _trial(
            passed=False,
            exit_status="workload_failed",
            elapsed_sec=2.0,
            wall_clock_sec=2.5,
        )
    ]
    stats = _default_call(trials=trials, effective_steps=50)
    assert stats.outcome_counts == {OUTCOME_UNKNOWN: 1}


def test_outcome_counts_did_not_run_when_main_work_did_not_start():
    trials = [
        _trial(
            passed=False,
            main_work_started=False,
            executed_iterations=0,
            configured_iterations=50,
            wall_clock_sec=3.5,
        )
        for _ in range(2)
    ]
    stats = _default_call(trials=trials, effective_steps=50)
    assert stats.outcome_counts == {OUTCOME_DID_NOT_RUN: 2}


def test_outcome_counts_completed_when_executed_matches_configured():
    trials = [
        _trial(
            passed=True,
            main_work_started=True,
            executed_iterations=50,
            configured_iterations=50,
        ),
    ]
    stats = _default_call(trials=trials)
    assert stats.outcome_counts == {OUTCOME_COMPLETED: 1}


def test_outcome_counts_crashed_after_iterations():
    trials = [
        _trial(
            passed=False,
            main_work_started=True,
            executed_iterations=12,
            configured_iterations=50,
        ),
    ]
    stats = _default_call(trials=trials)
    assert stats.outcome_counts == {OUTCOME_CRASHED_AFTER_ITERATIONS: 1}


def test_outcome_counts_mixed_outcomes_in_one_cell():
    trials = [
        _trial(
            main_work_started=True,
            executed_iterations=50,
            configured_iterations=50,
            passed=True,
        ),
        _trial(
            main_work_started=False,
            executed_iterations=0,
            configured_iterations=50,
            passed=False,
        ),
    ]
    stats = _default_call(trials=trials)
    assert stats.outcome_counts == {OUTCOME_COMPLETED: 1, OUTCOME_DID_NOT_RUN: 1}


def test_outcome_counts_unknown_for_legacy_passing_trials():
    """Legacy passing trials -> ``{"unknown": N}`` (always populated for
    non-error cells now that platform inference is live).

    The matrix.md ``did_not_run`` legend gate moved to "actual rendered
    confound tag" rather than "outcome_counts presence", so an
    all-unknown legacy cell can no longer leak the legend even though
    its outcome_counts is non-empty. ``is_did_not_run_cell`` likewise
    requires ``set(counts) == {DID_NOT_RUN}``, so an all-unknown cell
    is never false-flagged.
    """
    trials = [_trial(passed=True), _trial(passed=True)]
    stats = _default_call(trials=trials)
    assert stats.outcome_counts == {OUTCOME_UNKNOWN: 2}
    assert sum(stats.outcome_counts.values()) == stats.trials


def test_outcome_counts_built_when_only_iter_fields_populated():
    """Workload populates ``configured_iterations`` / ``executed_iterations``
    but not ``main_work_started`` -> the new contract IS in use, even
    partially. ``outcome_counts`` must be built (silent ``main_work_started``
    classifies the trial as ``unknown``) so matrix.json doesn't contradict
    matrix.md / the terminal log, both of which already treat the cell
    as new-contract.

    Round-4 Copilot catch on PR #175: previously the predicate was
    narrow (``main_work_started is not None`` only) while runner.py and
    output.py used the wider predicate, producing a partial-contract
    cell that looked legacy in matrix.json but new-contract everywhere
    else.
    """
    trials = [
        _trial(executed_iterations=50, configured_iterations=50, passed=True),
        _trial(executed_iterations=50, configured_iterations=50, passed=True),
    ]
    stats = _default_call(trials=trials)
    # main_work_started absent -> classified as OUTCOME_UNKNOWN (not
    # OUTCOME_COMPLETED -- completion requires main_work_started=True).
    # The point is that the histogram is BUILT, not that the workload
    # gets credit for completing.
    assert stats.outcome_counts == {OUTCOME_UNKNOWN: 2}
    assert sum(stats.outcome_counts.values()) == stats.trials


def test_outcome_counts_includes_unknown_for_silent_trials_in_mixed_cell():
    """As soon as ONE trial speaks the new contract, all trials are
    counted -- silent ones legitimately classify as ``unknown`` so the
    histogram total matches ``trial_count``. This is the "mixed cell"
    case where some trials use the new contract and others don't."""
    trials = [
        _trial(
            main_work_started=True,
            executed_iterations=50,
            configured_iterations=50,
            passed=True,
        ),
        _trial(passed=False),  # silent -- no main_work_started
    ]
    stats = _default_call(trials=trials)
    assert stats.outcome_counts == {OUTCOME_COMPLETED: 1, OUTCOME_UNKNOWN: 1}
    assert sum(stats.outcome_counts.values()) == stats.trials


def test_did_not_run_trials_suppress_wall_clock_step_time_fallback():
    """Symmetric refusal: a trial that died in setup must not contribute
    a fake step-time number derived from setup-only wall clock.

    Demo case: the import-error trial in the issue's MI350 SHAMPOO run
    produced 3.5s wall clock and 50 configured steps; the old aggregator
    surfaced 70 ms/step. With the suppression the cell reports no
    timing -- matrix.md will render n/a, matrix.json carries no
    misleading number.
    """
    trials = [
        _trial(
            passed=False,
            main_work_started=False,
            executed_iterations=0,
            configured_iterations=50,
            wall_clock_sec=3.5,
        )
    ]
    stats = _default_call(trials=trials, effective_steps=50)
    assert stats.mean_step_time_ms == 0.0
    assert stats.step_time_source == "missing"


def test_iters_display_uniform_when_all_trials_executed_same_count():
    trials = [
        _trial(
            main_work_started=True, executed_iterations=50, configured_iterations=50, passed=True
        )
        for _ in range(3)
    ]
    stats = _default_call(trials=trials)
    assert stats.iters_display == "50/50"
    assert stats.executed_iter_min == 50
    assert stats.executed_iter_max == 50
    assert stats.configured_iters == 50


def test_iters_display_min_max_when_executed_counts_differ():
    trials = [
        _trial(
            main_work_started=True, executed_iterations=10, configured_iterations=50, passed=False
        ),
        _trial(
            main_work_started=True, executed_iterations=42, configured_iterations=50, passed=False
        ),
    ]
    stats = _default_call(trials=trials)
    assert stats.iters_display == "10..42/50"
    assert stats.executed_iter_min == 10
    assert stats.executed_iter_max == 42


def test_iters_display_question_marks_when_configured_disagrees():
    """Defensive: a recipe pins `steps` per cell, so trials in the same
    cell shouldn't disagree on configured_iterations. If they do, surface
    the contradiction instead of silently picking one value."""
    trials = [
        _trial(
            main_work_started=True, executed_iterations=10, configured_iterations=50, passed=True
        ),
        _trial(
            main_work_started=True, executed_iterations=10, configured_iterations=100, passed=True
        ),
    ]
    stats = _default_call(trials=trials)
    assert stats.iters_display == "?/?"
    assert stats.configured_iters is None


def test_iters_display_dash_when_no_trial_populates_iter_fields():
    """Legacy workload: configured_iters is None so the display is the
    em-dash placeholder. The output renderer hides the column entirely
    when every cell lands here."""
    trials = [_trial(passed=True), _trial(passed=False)]
    stats = _default_call(trials=trials)
    assert stats.iters_display == "—"
    assert stats.configured_iters is None
    assert stats.executed_iter_min is None
    assert stats.executed_iter_max is None


def test_iters_display_renders_configured_when_executed_partial():
    """Mixed populated/None ``executed_iterations`` across trials -> render
    ``"—/<configured>"`` so the operator sees the budget that was set up
    even when execution count is missing for some trials. Hiding the row
    entirely (the previous behaviour) swallowed useful partial data --
    Copilot's round-2 catch on PR #175.

    The configured number is the load-bearing piece: it tells the reader
    "the workload was asked to run N iterations". The em-dash on the
    numerator says honestly "we don't know how far they got".
    """
    trials = [
        _trial(
            main_work_started=True, executed_iterations=42, configured_iterations=50, passed=True
        ),
        _trial(main_work_started=True, configured_iterations=50, passed=False),
    ]
    stats = _default_call(trials=trials)
    assert stats.iters_display == "—/50"
    assert stats.configured_iters == 50


def test_error_cell_outcome_counts_empty_and_iters_display_dash():
    """Error cells short-circuit aggregation: no outcome histogram, no
    iters display."""
    stats = _default_call(trials=[_trial()], error="docker pull failed")
    assert stats.outcome_counts == {}
    assert stats.iters_display == "—"
    assert stats.configured_iters is None


def test_workload_config_defaults_to_empty_dict():
    assert _default_call(trials=[_trial()]).workload_config == {}


def test_workload_config_persisted_on_cellstats():
    stats = _default_call(trials=[_trial()], workload_config={"shampoo_api": "old"})
    assert stats.workload_config == {"shampoo_api": "old"}


def test_workload_config_persisted_on_error_cell():
    stats = _default_call(trials=[_trial()], error="docker pull failed",
                          workload_config={"shampoo_api": "old"})
    assert stats.workload_config == {"shampoo_api": "old"}
