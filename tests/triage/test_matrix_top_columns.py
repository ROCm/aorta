"""Tests for the Phase-2 ``Top failure`` / ``Top warn`` columns in matrix.md (FR 2.10).

These tests live under ``tests/triage/`` because the renderer is
:func:`aorta.triage.output.write_matrix_md` (mode-agnostic). The
columns appear ONLY when a cell populates the new
:attr:`aorta.triage.matrix.CellStats.top_failure_detector_id` /
``top_warn_detector_id`` fields — so triage-mode runs (which never set
them) render byte-equivalently to Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aorta.triage.matrix import CellStats, aggregate_cell
from aorta.triage.output import write_matrix_md
from aorta.triage.recipe import Recipe, build_recipe_from_flags


def _minimal_recipe() -> Recipe:
    """A no-op triage-mode recipe; the renderer only reads workload/ticket/etc."""
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
    top_failure: str | None = None,
    top_warn: str | None = None,
    trials: int = 1,
    failed: int = 0,
) -> CellStats:
    return CellStats(
        name=name,
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        passed_count=trials - failed,
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
        top_failure_detector_id=top_failure,
        top_warn_detector_id=top_warn,
    )


def test_columns_hidden_when_no_cell_populates_detectors(tmp_path: Path):
    """Triage-mode default: no cell sets the detector fields -> no columns."""
    recipe = _minimal_recipe()
    stats = [_stats("none-local")]
    out = tmp_path / "matrix.md"
    write_matrix_md(
        out,
        recipe,
        stats,
        baseline=stats[0],
        confound_tags={"none-local": ("(baseline)", None)},
        warnings=[],
        run_timestamp="2026-01-01T00:00:00Z",
    )
    text = out.read_text(encoding="utf-8")
    # When NO cell carries detector data, neither the header nor the
    # legend should mention Top failure / Top warn (rubric §2.B FR 2.10).
    assert "Top failure" not in text
    assert "Top warn" not in text


def test_top_failure_column_visible_when_a_cell_has_detector(tmp_path: Path):
    """A single cell with a top_failure_detector_id surfaces the column."""
    recipe = _minimal_recipe()
    stats = [_stats("c", top_failure="tier4:hip_error", failed=1)]
    out = tmp_path / "matrix.md"
    write_matrix_md(
        out,
        recipe,
        stats,
        baseline=stats[0],
        confound_tags={"c": ("(baseline)", None)},
        warnings=[],
        run_timestamp="2026-01-01T00:00:00Z",
    )
    text = out.read_text(encoding="utf-8")
    assert "Top failure" in text
    assert "tier4:hip_error" in text
    # The HEADER row should not contain a 'Top warn' column when no cell
    # populates a warn detector. The legend mentions both column names
    # together; we check the table separators row to disambiguate.
    header_line = next(line for line in text.splitlines() if line.startswith("| Cell"))
    assert "Top warn" not in header_line


def test_top_warn_column_visible_when_a_cell_has_warn(tmp_path: Path):
    recipe = _minimal_recipe()
    stats = [_stats("c", top_warn="custom:slow_iter")]
    out = tmp_path / "matrix.md"
    write_matrix_md(
        out,
        recipe,
        stats,
        baseline=stats[0],
        confound_tags={"c": ("(baseline)", None)},
        warnings=[],
        run_timestamp="2026-01-01T00:00:00Z",
    )
    text = out.read_text(encoding="utf-8")
    header_line = next(line for line in text.splitlines() if line.startswith("| Cell"))
    assert "Top warn" in header_line
    assert "custom:slow_iter" in text


def test_both_columns_present_when_both_kinds_fire(tmp_path: Path):
    recipe = _minimal_recipe()
    stats = [
        _stats(
            "c",
            top_failure="tier4:python_traceback",
            top_warn="custom:slow_iter",
            failed=1,
        )
    ]
    out = tmp_path / "matrix.md"
    write_matrix_md(
        out,
        recipe,
        stats,
        baseline=stats[0],
        confound_tags={"c": ("(baseline)", None)},
        warnings=[],
        run_timestamp="2026-01-01T00:00:00Z",
    )
    text = out.read_text(encoding="utf-8")
    header_line = next(line for line in text.splitlines() if line.startswith("| Cell"))
    assert "Top failure" in header_line
    assert "Top warn" in header_line
    assert "tier4:python_traceback" in text
    assert "custom:slow_iter" in text


def test_empty_cells_render_em_dash_in_visible_columns(tmp_path: Path):
    """Cells without a top_failure_detector_id render '—' when the column shows."""
    recipe = _minimal_recipe()
    stats = [
        _stats("a", top_failure="tier4:hip_error", failed=1),
        _stats("b"),
    ]
    out = tmp_path / "matrix.md"
    write_matrix_md(
        out,
        recipe,
        stats,
        baseline=stats[0],
        confound_tags={"a": ("(baseline)", None), "b": ("-", 1.0)},
        warnings=[],
        run_timestamp="2026-01-01T00:00:00Z",
    )
    text = out.read_text(encoding="utf-8")
    header_line = next(line for line in text.splitlines() if line.startswith("| Cell"))
    assert "Top failure" in header_line
    # ``b``'s row should contain an em-dash placeholder in the Top failure column.
    b_line = next(line for line in text.splitlines() if "| b " in line)
    assert "—" in b_line


# ---- aggregate_cell() picks the highest-frequency detector ID ----------


def _trial_with_detectors(
    *, failures: list[str] = None, warns: list[str] = None
) -> SimpleNamespace:
    """A minimal trial-shaped object matching what the aggregator reads."""
    return SimpleNamespace(
        exit_status="ok",
        wall_clock_sec=1.0,
        result={
            "passed": True,
            "step_times_ms": [100.0],
            "metrics": {
                "failure_detectors_fired": list(failures or []),
                "warn_detectors_fired": list(warns or []),
            },
        },
    )


def test_aggregator_picks_most_frequent_detector():
    """Across trials, the detector ID fired the most wins (FR 2.10)."""
    trials = [
        _trial_with_detectors(failures=["tier4:hip_error"]),
        _trial_with_detectors(failures=["tier4:hip_error"]),
        _trial_with_detectors(failures=["tier4:nan_signature"]),
    ]
    stats = aggregate_cell(
        name="c",
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        effective_steps=1,
    )
    assert stats.top_failure_detector_id == "tier4:hip_error"


def test_aggregator_ties_resolve_by_encounter_order():
    """A tie picks the first-encountered detector ID (rubric §2.10)."""
    trials = [
        _trial_with_detectors(failures=["tier4:nan_signature"]),
        _trial_with_detectors(failures=["tier4:hip_error"]),
    ]
    stats = aggregate_cell(
        name="c",
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        effective_steps=1,
    )
    assert stats.top_failure_detector_id == "tier4:nan_signature"


def test_aggregator_returns_none_when_no_detectors_fired():
    """No detectors -> ``None`` (matrix.md hides the column)."""
    trials = [_trial_with_detectors()]
    stats = aggregate_cell(
        name="c",
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        effective_steps=1,
    )
    assert stats.top_failure_detector_id is None
    assert stats.top_warn_detector_id is None
