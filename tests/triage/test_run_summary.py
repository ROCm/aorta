"""Tests for the concise end-of-run sweep summary (issue #280).

``aorta sweep run`` is otherwise silent during a run and prints only a single
``Wrote matrix to ...`` line at the end, so an operator has no way to see which
cells failed or where their captured output landed. These tests pin:

* the pure :func:`aorta.triage.output.format_run_summary` formatter -- totals
  line, per-cell fail/error lines, artifact-directory pointers (both layouts),
  failure hints, and the ``-v`` tip suppression; and
* that a real ``aorta sweep run`` (through the shared engine) actually emits the
  summary to stdout -- including on a degraded (``MatrixIncompleteError``) run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import aorta.triage.runner as runner
from aorta.cli import main
from aorta.triage.matrix import aggregate_cell
from aorta.triage.output import format_run_summary

# ---- trial stand-ins ------------------------------------------------------


def _pass_trial() -> SimpleNamespace:
    return SimpleNamespace(
        exit_status="ok",
        wall_clock_sec=1.0,
        result={"passed": True, "step_times_ms": [100.0]},
    )


def _fail_trial(hint: str | None = None) -> SimpleNamespace:
    result: dict = {"passed": False, "step_times_ms": [100.0]}
    if hint is not None:
        result["failure_details"] = [{"hint": hint}]
    return SimpleNamespace(exit_status="workload_failed", wall_clock_sec=1.0, result=result)


def _error_trial() -> SimpleNamespace:
    # setup() raised -> infra error verdict, no valid observation.
    return SimpleNamespace(
        exit_status="workload_setup_failed",
        wall_clock_sec=0.0,
        result={"passed": False, "step_times_ms": [], "elapsed_sec": 0.0},
    )


def _cell(name, mitigation, trials, error=None):
    return aggregate_cell(
        name=name,
        mitigations=(mitigation,),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        effective_steps=50,
        error=error,
    )


# ---- format_run_summary: unit --------------------------------------------


def test_all_clean_is_a_single_line():
    clean = _cell("baseline-local", "none", [_pass_trial(), _pass_trial()])
    lines = format_run_summary([clean], Path("/tmp/run"))
    assert lines == ["Sweep summary: all 1 cell(s) passed."]


def test_totals_line_counts_clean_failing_and_errored_cells():
    stats = [
        _cell("baseline-local", "none", [_pass_trial(), _pass_trial()]),
        _cell("tf32-local", "tf32_off", [_fail_trial(), _pass_trial()]),
        _cell("hsa-local", "hsa_xnack", [_error_trial(), _error_trial()]),
        _cell("bad-local", "bad", [], error="UnknownMitigationError: bad"),
    ]
    lines = format_run_summary(stats, Path("/tmp/run"))
    assert lines[0] == (
        "Sweep summary: 4 cell(s) -- 1 clean, 1 with failing trials, 2 with errors."
    )


def test_failing_cell_line_has_counts_hint_and_relative_dir():
    stats = [
        _cell("baseline-local", "none", [_pass_trial(), _pass_trial()]),
        _cell("tf32-local", "tf32_off", [_fail_trial("NaN at layer 3"), _pass_trial()]),
    ]
    lines = format_run_summary(stats, Path("/tmp/run"))
    # Clean cells never get their own line.
    assert not any("baseline-local" in ln for ln in lines[1:])
    fail_line = next(ln for ln in lines if "tf32-local" in ln)
    assert fail_line.startswith("  [fail] tf32-local:")
    assert "1 failed of 2 trial(s)" in fail_line
    assert "(NaN at layer 3)" in fail_line
    # Timestamped (triage) layout nests cells under cells/.
    assert fail_line.rstrip().endswith("-> cells/tf32-local/")


def test_error_only_cell_is_tagged_error_not_fail():
    stats = [
        _cell("baseline-local", "none", [_pass_trial()]),
        _cell("hsa-local", "hsa_xnack", [_error_trial(), _error_trial()]),
    ]
    lines = format_run_summary(stats, Path("/tmp/run"))
    err_line = next(ln for ln in lines if "hsa-local" in ln)
    assert err_line.startswith("  [error] hsa-local:")
    assert "2 errored of 2 trial(s)" in err_line


def test_whole_cell_error_line_and_message_is_single_line():
    stats = [
        _cell("baseline-local", "none", [_pass_trial()]),
        _cell("bad-local", "bad", [], error="UnknownMitigationError: no\nsuch\nname"),
    ]
    lines = format_run_summary(stats, Path("/tmp/run"))
    err_line = next(ln for ln in lines if "bad-local" in ln)
    assert err_line.startswith("  [error] bad-local: cell did not run (")
    # Newlines in the error message are folded so each cell stays one line.
    assert "\n" not in err_line
    assert "UnknownMitigationError: no such name" in err_line


def test_flat_resume_layout_points_at_sibling_dirs():
    stats = [
        _cell("none-smoke", "none", [_pass_trial()]),
        _cell("tf32-smoke", "tf32_off", [_fail_trial()]),
    ]
    lines = format_run_summary(stats, Path("/tmp/run"), layout="flat_resume")
    fail_line = next(ln for ln in lines if "tf32-smoke" in ln)
    # flat_resume puts cell artifacts directly under the run dir (no cells/).
    assert fail_line.rstrip().endswith("-> tf32-smoke/")


def test_footer_points_at_matrix_and_offers_verbose_tip():
    stats = [_cell("tf32-local", "tf32_off", [_fail_trial()])]
    lines = format_run_summary(stats, Path("/tmp/run"))
    assert f"Full matrix: {Path('/tmp/run') / 'matrix.md'}" in lines
    assert any(ln.startswith("Tip: re-run with -v") for ln in lines)


def test_verbose_run_suppresses_the_verbose_tip():
    stats = [_cell("tf32-local", "tf32_off", [_fail_trial()])]
    lines = format_run_summary(stats, Path("/tmp/run"), verbose_active=True)
    assert not any("re-run with -v" in ln for ln in lines)
    # The matrix pointer is still there -- it's useful regardless of verbosity.
    assert any(ln.startswith("Full matrix:") for ln in lines)


# ---- aorta sweep run: end-to-end -----------------------------------------

_TRIAGE_RECIPE = """\
schema_version: 1
ticket: SUM-1
workload: fsdp
trials: 2
steps: 5
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
  - name: tf32-local
    mitigations: [tf32_off]
    environment: local
"""


@pytest.fixture
def _hermetic_engine(monkeypatch):
    """Patch the runner's env probe so the summary tests don't touch host state."""
    monkeypatch.setattr(
        runner,
        "collect_env",
        lambda: SimpleNamespace(to_dict=lambda: {}, partial=False, partial_reasons=[]),
    )


def test_sweep_run_prints_summary_with_failing_cell(tmp_path, monkeypatch, _hermetic_engine):
    def fake_run_trials(request):
        if "tf32_off" in request.mitigations:
            return [_fail_trial("NaN at layer 3"), _pass_trial()]
        return [_pass_trial(), _pass_trial()]

    monkeypatch.setattr(runner, "run_trials", fake_run_trials)

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(_TRIAGE_RECIPE, encoding="utf-8")
    result = CliRunner().invoke(
        main,
        ["sweep", "run", "--recipe", str(recipe), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "Sweep summary: 2 cell(s) -- 1 clean, 1 with failing trials" in result.output
    assert "[fail] tf32-local:" in result.output
    assert "(NaN at layer 3)" in result.output
    assert "cells/tf32-local/" in result.output
    # The pre-existing final line is preserved.
    assert "Wrote matrix to" in result.output


def test_sweep_run_all_pass_is_one_summary_line(tmp_path, monkeypatch, _hermetic_engine):
    monkeypatch.setattr(runner, "run_trials", lambda request: [_pass_trial(), _pass_trial()])
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(_TRIAGE_RECIPE, encoding="utf-8")
    result = CliRunner().invoke(
        main,
        ["sweep", "run", "--recipe", str(recipe), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "Sweep summary: all 2 cell(s) passed." in result.output
    assert "[fail]" not in result.output


def test_summary_prints_even_on_degraded_run(tmp_path, monkeypatch, _hermetic_engine):
    """A run whose explicit baseline never ran still gets a summary + exit!=0."""
    monkeypatch.setattr(runner, "run_trials", lambda request: [_error_trial(), _error_trial()])
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        _TRIAGE_RECIPE + "confound:\n  baseline_cell: baseline-local\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        main,
        ["sweep", "run", "--recipe", str(recipe), "--output", str(tmp_path / "out")],
    )
    # Degraded: the baseline produced only errored trials.
    assert result.exit_code != 0, result.output
    assert "Sweep summary:" in result.output
    assert "[error] baseline-local:" in result.output
