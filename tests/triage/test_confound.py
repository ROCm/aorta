"""Tests for src/aorta/triage/confound.py: baseline resolution + classify()."""

from __future__ import annotations

import pytest

from aorta.triage.confound import (
    CONFOUND_BASELINE,
    CONFOUND_ERROR,
    CONFOUND_NEUTRAL,
    CONFOUND_NO_EFFECT,
    classify,
    classify_all,
    resolve_baseline,
)
from aorta.triage.matrix import CellStats
from aorta.triage.recipe import Cell, RecipeCellError


def _stats(
    name: str,
    mean_step_time_ms: float = 100.0,
    passed_count: int = 0,
    trials: int = 8,
    error: str | None = None,
) -> CellStats:
    return CellStats(
        name=name,
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        passed_count=passed_count,
        failed_count=trials - passed_count,
        mean_step_time_ms=mean_step_time_ms,
        std_step_time_ms=0.0,
        min_step_time_ms=mean_step_time_ms,
        max_step_time_ms=mean_step_time_ms,
        p50_step_time_ms=mean_step_time_ms,
        p90_step_time_ms=mean_step_time_ms,
        p99_step_time_ms=mean_step_time_ms,
        mean_wall_clock_sec=1.0,
        exit_status_counts={},
        step_times_ms=[mean_step_time_ms],
        trial_paths=[],
        error=error,
    )


# ---- resolve_baseline -----------------------------------------------------


def test_explicit_baseline_name_wins():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="baseline-local", mitigations=("none",), environment="local"),
    ]
    chosen = resolve_baseline(cells, explicit_name="tf32-local")
    assert chosen.name == "tf32-local"


def test_explicit_baseline_name_not_found_raises():
    cells = [Cell(name="a", mitigations=("none",), environment="local")]
    with pytest.raises(RecipeCellError, match="does not match any cell"):
        resolve_baseline(cells, explicit_name="nope")


def test_default_picks_first_baseline_dash_prefix():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="baseline-local", mitigations=("none",), environment="local"),
        Cell(name="baseline-docker", mitigations=("none",), environment="local"),
    ]
    assert resolve_baseline(cells, explicit_name=None).name == "baseline-local"


def test_default_falls_back_to_mitigations_none():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="vanilla", mitigations=("none",), environment="local"),
    ]
    assert resolve_baseline(cells, explicit_name=None).name == "vanilla"


def test_single_cell_is_its_own_baseline():
    cells = [Cell(name="only", mitigations=("tf32_off",), environment="local")]
    assert resolve_baseline(cells, explicit_name=None).name == "only"


def test_no_baseline_resolution_raises():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="xnack-local", mitigations=("xnack",), environment="local"),
    ]
    with pytest.raises(RecipeCellError, match="cannot resolve baseline cell"):
        resolve_baseline(cells, explicit_name=None)


# ---- classify -------------------------------------------------------------


def test_classify_baseline():
    base = _stats("b", mean_step_time_ms=100.0)
    tag, ratio = classify(base, base, threshold=1.15)
    assert tag == CONFOUND_BASELINE
    assert ratio is None


def test_classify_speed_confound_plus_25_percent():
    base = _stats("b", mean_step_time_ms=400.0, passed_count=4)  # failure_rate=0.5
    slow = _stats("tf32-local", mean_step_time_ms=500.0, passed_count=8)  # failure_rate=0
    tag, ratio = classify(slow, base, threshold=1.15)
    assert tag == "speed (+25%)"
    assert ratio is not None and abs(ratio - 1.25) < 1e-9


def test_classify_neutral_when_ratio_one_and_failure_rate_drops():
    base = _stats("b", mean_step_time_ms=100.0, passed_count=0)  # failure_rate 1.0
    cell = _stats("c", mean_step_time_ms=100.0, passed_count=8)  # failure_rate 0.0
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NEUTRAL
    assert ratio == 1.0


def test_classify_no_effect_when_failure_rate_unchanged_and_no_slowdown():
    base = _stats("b", mean_step_time_ms=100.0, passed_count=0)  # failure_rate 1.0
    cell = _stats("c", mean_step_time_ms=105.0, passed_count=0)  # ratio 1.05, failure_rate 1.0
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NO_EFFECT
    assert ratio == 1.05


def test_classify_error_cell_tag():
    base = _stats("b", mean_step_time_ms=100.0)
    err = _stats("c", error="docker pull failed")
    tag, ratio = classify(err, base, threshold=1.15)
    assert tag == CONFOUND_ERROR
    assert ratio is None


def test_classify_baseline_errored_forces_no_ratio():
    base = _stats("b", mean_step_time_ms=0.0, error="baseline crashed")
    cell = _stats("c", mean_step_time_ms=100.0, passed_count=8)
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NEUTRAL
    assert ratio is None


def test_classify_baseline_zero_step_time_forces_no_ratio():
    base = _stats("b", mean_step_time_ms=0.0)
    cell = _stats("c", mean_step_time_ms=100.0, passed_count=8)
    tag, ratio = classify(cell, base, threshold=1.15)
    assert ratio is None


# ---- classify_all ---------------------------------------------------------


def test_classify_all_returns_tag_per_cell():
    base = _stats("baseline-local", mean_step_time_ms=400.0, passed_count=4)
    slow = _stats("tf32-local", mean_step_time_ms=500.0, passed_count=8)
    neutral = _stats("xnack-local", mean_step_time_ms=400.0, passed_count=8)
    tags = classify_all([base, slow, neutral], baseline_name="baseline-local", threshold=1.15)
    assert tags["baseline-local"][0] == CONFOUND_BASELINE
    assert tags["tf32-local"][0] == "speed (+25%)"
    assert tags["xnack-local"][0] == CONFOUND_NEUTRAL


def test_classify_all_missing_baseline_raises():
    cell = _stats("c", mean_step_time_ms=100.0)
    with pytest.raises(RecipeCellError, match="baseline cell"):
        classify_all([cell], baseline_name="not_present", threshold=1.15)
