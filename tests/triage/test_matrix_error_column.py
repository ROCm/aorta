"""Tests for the issue #230 ``Errors`` column + unreliable flag in matrix.md.

The column appears ONLY when at least one cell has an ``error_count`` > 0,
so legacy / error-free runs render byte-equivalently. Mirrors the
visibility pattern of the Phase-2 ``Top failure`` columns.
"""

from __future__ import annotations

import json
from pathlib import Path

from aorta.triage.matrix import CellStats
from aorta.triage.output import write_matrix_json, write_matrix_md
from aorta.triage.recipe import Recipe, build_recipe_from_flags


def _minimal_recipe() -> Recipe:
    return build_recipe_from_flags(
        workload="echo",
        mitigation_axis="none",
        environment_axis="local",
        trials=1,
        steps=1,
    )


def _stats(
    name: str,
    *,
    trials: int,
    passed: int,
    failed: int,
    error: int,
) -> CellStats:
    return CellStats(
        name=name,
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        passed_count=passed,
        failed_count=failed,
        mean_step_time_ms=10.0,
        std_step_time_ms=0.0,
        min_step_time_ms=10.0,
        max_step_time_ms=10.0,
        p50_step_time_ms=10.0,
        p90_step_time_ms=10.0,
        p99_step_time_ms=10.0,
        mean_wall_clock_sec=1.0,
        step_time_source="per_step",
        error_count=error,
    )


def _write_md(tmp_path: Path, stats: list[CellStats]) -> str:
    out = tmp_path / "matrix.md"
    write_matrix_md(
        out,
        _minimal_recipe(),
        stats,
        baseline=stats[0],
        confound_tags={s.name: ("(baseline)", None) for s in stats},
        warnings=[],
        run_timestamp="2026-01-01T00:00:00Z",
    )
    return out.read_text(encoding="utf-8")


def test_errors_column_hidden_when_no_errors(tmp_path: Path):
    stats = [_stats("none-local", trials=4, passed=4, failed=0, error=0)]
    text = _write_md(tmp_path, stats)
    assert "Errors" not in text
    assert "unreliable" not in text


def test_errors_column_visible_when_a_cell_errors(tmp_path: Path):
    # 1 error out of 4 (25% >= 10% threshold) -> unreliable.
    stats = [_stats("c", trials=4, passed=2, failed=1, error=1)]
    text = _write_md(tmp_path, stats)
    assert "| Errors " in text
    assert "1 / 4 (unreliable)" in text
    # Failure rate is computed over valid trials (2 pass + 1 fail = 3): 1/3 ~ 33%.
    assert "33%" in text
    # Failures column matches the rate's denominator: 1 / 3.
    assert "1 / 3" in text


def test_errors_column_no_unreliable_below_threshold(tmp_path: Path):
    # 1 error out of 20 == 5% < 10% threshold: column shows, no flag.
    stats = [_stats("c", trials=20, passed=15, failed=4, error=1)]
    text = _write_md(tmp_path, stats)
    assert "| Errors " in text
    assert "1 / 20" in text
    # The cell value must not carry the marker (the legend prose still
    # explains what ``(unreliable)`` means -- that's expected).
    assert "1 / 20 (unreliable)" not in text


def test_failures_column_flags_no_valid_trials(tmp_path: Path):
    # Every trial errored (no whole-cell error, but zero valid trials):
    # the Failures column must read "0 / 0 (no valid trials)" rather than
    # a bare, degenerate "0 / 0" (issue #230 review).
    stats = [_stats("c", trials=3, passed=0, failed=0, error=3)]
    text = _write_md(tmp_path, stats)
    assert "0 / 0 (no valid trials)" in text


def test_matrix_json_carries_error_fields(tmp_path: Path):
    stats = [_stats("c", trials=4, passed=2, failed=1, error=1)]
    out = tmp_path / "matrix.json"
    write_matrix_json(
        out,
        _minimal_recipe(),
        stats,
        baseline_name="c",
        confound_tags={"c": ("(baseline)", None)},
        run_timestamp="2026-01-01T00:00:00Z",
        warnings=[],
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    cell = doc["cells"][0]
    assert cell["error_count"] == 1
    assert abs(cell["error_rate"] - 0.25) < 1e-9
    # failure_rate is the event rate over valid trials: 1 / (2 + 1).
    assert abs(cell["failure_rate"] - (1.0 / 3.0)) < 1e-9
    assert cell["unreliable"] is True
