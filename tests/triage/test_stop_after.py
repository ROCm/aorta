"""Tests for the ``stop_after`` collect-until-N stopping rule (issue #232).

Coverage:

* schema parse + validation (``_parse_stop_after`` via the loaders);
* the dispatcher's event predicate + early-stop loop;
* the matrix "stopped early" vs "cap reached" annotation;
* the ``--stop-after-events`` / ``--max-trials`` CLI overlay;
* end-to-end via ``aorta probe`` (real subprocess argv), including the
  resume short-circuit for a cell that already satisfied its rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aorta.cli.probe import probe
from aorta.probe.cli_helpers import ProbeUsageError, apply_recipe_overrides
from aorta.run.dispatcher import RunRequest, _trial_is_event, run_trials
from aorta.triage.recipe import RecipeSchemaError, StopAfter, load_recipe
from aorta.triage.runner import _stop_after_note
from aorta.workloads import Workload, WorkloadResult


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "r.yaml"
    p.write_text(text, encoding="utf-8")
    return p


_PROBE_HEAD = (
    "schema_version: 1\n"
    "mode: probe\n"
    "trials: 1\n"
    "mitigation_axis: [none]\n"
    "diagnostic_axis: [none]\n"
)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
def test_stop_after_parses(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  events: 3\n  max_trials: 160\n"
    r = load_recipe(_write(tmp_path, text))
    assert r.stop_after == StopAfter(events=3, max_trials=160, event_verdict="fail")


def test_stop_after_event_verdict_pass(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  events: 2\n  max_trials: 10\n  event_verdict: pass\n"
    r = load_recipe(_write(tmp_path, text))
    assert r.stop_after.event_verdict == "pass"


def test_stop_after_absent_is_none(tmp_path):
    r = load_recipe(_write(tmp_path, _PROBE_HEAD))
    assert r.stop_after is None


def test_stop_after_requires_max_trials(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  events: 3\n"
    with pytest.raises(RecipeSchemaError, match="max_trials' is required"):
        load_recipe(_write(tmp_path, text))


def test_stop_after_requires_events(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  max_trials: 10\n"
    with pytest.raises(RecipeSchemaError, match="missing required key 'events'"):
        load_recipe(_write(tmp_path, text))


def test_stop_after_cap_below_target_rejected(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  events: 9\n  max_trials: 3\n"
    with pytest.raises(RecipeSchemaError, match="must be >="):
        load_recipe(_write(tmp_path, text))


def test_stop_after_error_verdict_now_accepted(tmp_path):
    """Issue #230 landed the three-way verdict, so ``event_verdict: error``
    is now a valid stopping rule (e.g. bail out of an infra-flaky sweep)."""
    text = _PROBE_HEAD + "stop_after:\n  events: 1\n  max_trials: 5\n  event_verdict: error\n"
    recipe = load_recipe(_write(tmp_path, text))
    assert recipe.stop_after is not None
    assert recipe.stop_after.event_verdict == "error"


def test_stop_after_unknown_verdict_rejected(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  events: 1\n  max_trials: 5\n  event_verdict: bogus\n"
    with pytest.raises(RecipeSchemaError, match="event_verdict"):
        load_recipe(_write(tmp_path, text))


def test_stop_after_non_int_rejected(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  events: 1.5\n  max_trials: 5\n"
    with pytest.raises(RecipeSchemaError, match="must be an integer"):
        load_recipe(_write(tmp_path, text))


def test_stop_after_unknown_key_rejected(tmp_path):
    text = _PROBE_HEAD + "stop_after:\n  events: 1\n  max_trials: 5\n  bogus: 1\n"
    with pytest.raises(RecipeSchemaError, match="unknown keys"):
        load_recipe(_write(tmp_path, text))


def test_stop_after_valid_in_triage_mode(tmp_path):
    text = (
        "schema_version: 1\n"
        "workload: fsdp\n"
        "trials: 2\n"
        "steps: 10\n"
        "stop_after:\n  events: 1\n  max_trials: 8\n"
        "cells:\n"
        "  - name: baseline-local\n"
        "    mitigations: [none]\n"
        "    environment: local\n"
    )
    r = load_recipe(_write(tmp_path, text))
    assert r.stop_after == StopAfter(events=1, max_trials=8, event_verdict="fail")


def test_stop_after_rejects_per_cell_trials_override(tmp_path):
    # Issue #232: stop_after.max_trials is the per-cell cap, so a per-cell
    # ``trials:`` override would be silently ignored at run time. Reject the
    # combination at load time instead of surprising the operator.
    text = (
        "schema_version: 1\n"
        "workload: fsdp\n"
        "trials: 2\n"
        "steps: 10\n"
        "stop_after:\n  events: 1\n  max_trials: 8\n"
        "cells:\n"
        "  - name: baseline-local\n"
        "    mitigations: [none]\n"
        "    environment: local\n"
        "    trials: 3\n"
    )
    with pytest.raises(RecipeSchemaError, match="incompatible with per-cell 'trials'"):
        load_recipe(_write(tmp_path, text))


# --------------------------------------------------------------------------
# Event predicate
# --------------------------------------------------------------------------
def _tr(exit_status: str) -> SimpleNamespace:
    return SimpleNamespace(exit_status=exit_status)


@pytest.mark.parametrize(
    "exit_status,verdict,expected",
    [
        ("ok", "fail", False),
        ("workload_failed", "fail", True),
        # Issue #230: an infra crash is an ``error`` event, NOT a ``fail``
        # event -- it never validly reproduced the bug.
        ("infrastructure_failed", "fail", False),
        ("infrastructure_failed", "error", True),
        ("workload_setup_failed", "error", True),
        ("workload_failed", "error", False),
        ("ok", "pass", True),
        ("workload_failed", "pass", False),
        ("infrastructure_failed", "pass", False),
    ],
)
def test_trial_is_event(exit_status, verdict, expected):
    assert _trial_is_event(_tr(exit_status), verdict) is expected


@pytest.mark.parametrize("bad_verdict", ["FAIL", "fial", "", None, 1])
def test_trial_is_event_rejects_unknown_verdict(bad_verdict):
    # An unknown/typo'd verdict must raise rather than silently fall through
    # to the "fail" branch -- otherwise a programmatic caller that bypassed
    # the recipe loader would get misinterpreted semantics (#236 r3 pattern).
    # ``error`` is a valid three-way event verdict (#230), so it is no longer
    # in this rejected set.
    with pytest.raises(ValueError, match="event_verdict must be one of"):
        _trial_is_event(_tr("ok"), bad_verdict)


# --------------------------------------------------------------------------
# Dispatcher early-stop loop
# --------------------------------------------------------------------------
class _FailWL(Workload):
    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(passed=False, failure_count=1)

    def cleanup(self) -> None:
        pass


class _PassWL(Workload):
    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        pass

    def run(self) -> WorkloadResult:
        return WorkloadResult(passed=True, total_iterations=1, elapsed_sec=0.1)

    def cleanup(self) -> None:
        pass


class _ErrorWL(Workload):
    """Setup raises -> dispatcher records exit_status='workload_setup_failed'
    -> three-way verdict 'error' (issue #230)."""

    launch_mode = "single_process"
    min_world_size = 1

    def setup(self) -> None:
        raise RuntimeError("infra boom")

    def run(self) -> WorkloadResult:  # pragma: no cover - never reached
        return WorkloadResult(passed=True)

    def cleanup(self) -> None:
        pass


def _run_trials_with(workload_cls, stop_after, tmp_path, trials):
    mock_ep = MagicMock()
    mock_ep.name = "w"
    mock_ep.load.return_value = workload_cls
    mock_eps = MagicMock()
    mock_eps.select.return_value = [mock_ep]
    with patch("importlib.metadata.entry_points", return_value=mock_eps):
        req = RunRequest(
            workload="w", trials=trials, results_dir=tmp_path, stop_after=stop_after
        )
        return run_trials(req)


def test_dispatcher_stops_early_on_events(tmp_path):
    sa = StopAfter(events=2, max_trials=10)
    results = _run_trials_with(_FailWL, sa, tmp_path, trials=sa.max_trials)
    assert len(results) == 2  # stopped after the 2nd failing trial


def test_dispatcher_runs_to_cap_when_target_unmet(tmp_path):
    sa = StopAfter(events=3, max_trials=4)  # passing workload -> 0 fail events
    results = _run_trials_with(_PassWL, sa, tmp_path, trials=sa.max_trials)
    assert len(results) == 4  # cap reached


def test_dispatcher_event_verdict_pass(tmp_path):
    sa = StopAfter(events=2, max_trials=10, event_verdict="pass")
    results = _run_trials_with(_PassWL, sa, tmp_path, trials=sa.max_trials)
    assert len(results) == 2


def test_dispatcher_no_stop_after_runs_all(tmp_path):
    results = _run_trials_with(_FailWL, None, tmp_path, trials=3)
    assert len(results) == 3


def test_dispatcher_event_verdict_error_stops_early(tmp_path):
    """Issue #230: ``event_verdict: error`` stops a cell once enough trials
    have errored (e.g. bail out of an infra-flaky sweep)."""
    sa = StopAfter(events=2, max_trials=10, event_verdict="error")
    results = _run_trials_with(_ErrorWL, sa, tmp_path, trials=sa.max_trials)
    assert len(results) == 2


def test_dispatcher_errors_do_not_count_as_fail_events(tmp_path):
    """An all-error workload never satisfies an ``event_verdict: fail`` rule,
    so the cell runs to the cap (errors are not fails -- issue #230)."""
    sa = StopAfter(events=2, max_trials=3, event_verdict="fail")
    results = _run_trials_with(_ErrorWL, sa, tmp_path, trials=sa.max_trials)
    assert len(results) == 3  # cap reached, never stopped early on a "fail"


def test_dispatcher_log_says_early_when_budget_remains(tmp_path, caplog):
    sa = StopAfter(events=2, max_trials=10)
    with caplog.at_level("INFO", logger="aorta.run.dispatcher"):
        _run_trials_with(_FailWL, sa, tmp_path, trials=sa.max_trials)
    stop_logs = [r.getMessage() for r in caplog.records if "stop_after:" in r.getMessage()]
    assert stop_logs and "stopping cell early" in stop_logs[-1]


def test_dispatcher_log_says_cap_reached_on_final_trial(tmp_path, caplog):
    # Target met exactly on the last allowed trial -> not "early", it's a cap reach.
    sa = StopAfter(events=3, max_trials=3)
    with caplog.at_level("INFO", logger="aorta.run.dispatcher"):
        _run_trials_with(_FailWL, sa, tmp_path, trials=sa.max_trials)
    stop_logs = [r.getMessage() for r in caplog.records if "stop_after:" in r.getMessage()]
    assert stop_logs and "cap reached" in stop_logs[-1]
    assert "stopping cell early" not in stop_logs[-1]


# --------------------------------------------------------------------------
# Matrix annotation
# --------------------------------------------------------------------------
def test_stop_after_note_stopped_early():
    sa = StopAfter(events=2, max_trials=10)
    trials = [_tr("workload_failed"), _tr("ok"), _tr("workload_failed")]
    note = _stop_after_note(sa, trials)
    assert note.startswith("stopped early")
    assert "2 fail event(s) in 3 trial(s)" in note


def test_stop_after_note_cap_reached():
    sa = StopAfter(events=3, max_trials=3)
    trials = [_tr("ok"), _tr("ok"), _tr("ok")]
    note = _stop_after_note(sa, trials)
    assert note.startswith("cap reached")
    assert "0 fail event(s) in 3 trial(s)" in note


def test_stop_after_column_shown_when_all_cells_error(tmp_path):
    # The "Stop after" column must reflect the *configured* rule, not the
    # success of individual cells: an all-errored run leaves every
    # ``stop_after_note`` None, but the rule was still active and
    # matrix.json still carries ``stop_after``. Gating on any cell's note
    # would hide the column here. Errored cells render "—" in the column.
    from aorta.triage.matrix import aggregate_cell
    from aorta.triage.output import write_matrix_md

    recipe = _probe_recipe(tmp_path, "stop_after:\n  events: 2\n  max_trials: 6\n")
    errored = aggregate_cell(
        name="none-none",
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=[],
        effective_steps=10,
        error="infrastructure_failed: boom",
    )
    out = tmp_path / "matrix.md"
    write_matrix_md(out, recipe, [errored], errored, {}, [], "2026-06-17T00:00:00Z")
    md = out.read_text(encoding="utf-8")
    assert "Stop after" in md
    note_row = next(line for line in md.splitlines() if line.startswith("| none-none "))
    assert "| — " in note_row


def test_matrix_md_trials_per_cell_reflects_cap(tmp_path):
    # The matrix.md "Trials per cell" header must show the stop_after cap
    # ("up to N"), not ``recipe.trials`` (often 1 on probe recipes), so the
    # header matches the real per-cell budget. A rule-less recipe still
    # renders the bare ``recipe.trials``.
    from aorta.triage.matrix import aggregate_cell
    from aorta.triage.output import write_matrix_md

    def _render(recipe):
        cell = aggregate_cell(
            name="none-none",
            mitigations=("none",),
            environment="local",
            extra_env={},
            resolved_env_vars={},
            trials=[],
            effective_steps=10,
            error="infrastructure_failed: boom",
        )
        out = tmp_path / "matrix.md"
        write_matrix_md(out, recipe, [cell], cell, {}, [], "2026-06-17T00:00:00Z")
        return out.read_text(encoding="utf-8")

    with_rule = _render(_probe_recipe(tmp_path, "stop_after:\n  events: 2\n  max_trials: 6\n"))
    assert "**Trials per cell**: up to 6" in with_rule

    without_rule = _render(_probe_recipe(tmp_path))
    assert "**Trials per cell**: 1" in without_rule


# --------------------------------------------------------------------------
# CLI overlay (apply_recipe_overrides)
# --------------------------------------------------------------------------
def _probe_recipe(tmp_path, stop_after_block=""):
    return load_recipe(_write(tmp_path, _PROBE_HEAD + stop_after_block))


def test_cli_overlay_builds_stop_after(tmp_path):
    r = _probe_recipe(tmp_path)
    out = apply_recipe_overrides(
        r, ticket=None, cli_passthrough_mode=None,
        cli_stop_after_events=3, cli_max_trials=20,
    )
    assert out.stop_after == StopAfter(events=3, max_trials=20, event_verdict="fail")


def test_cli_overlay_half_falls_back_to_recipe(tmp_path):
    r = _probe_recipe(tmp_path, "stop_after:\n  events: 2\n  max_trials: 9\n  event_verdict: pass\n")
    # Only override the cap; events + verdict come from the recipe.
    out = apply_recipe_overrides(
        r, ticket=None, cli_passthrough_mode=None, cli_max_trials=50
    )
    assert out.stop_after == StopAfter(events=2, max_trials=50, event_verdict="pass")


def test_cli_overlay_events_without_cap_rejected(tmp_path):
    r = _probe_recipe(tmp_path)
    with pytest.raises(ProbeUsageError, match="requires --max-trials"):
        apply_recipe_overrides(
            r, ticket=None, cli_passthrough_mode=None, cli_stop_after_events=3
        )


def test_cli_overlay_cap_below_target_rejected(tmp_path):
    r = _probe_recipe(tmp_path)
    with pytest.raises(ProbeUsageError, match="must be >="):
        apply_recipe_overrides(
            r, ticket=None, cli_passthrough_mode=None,
            cli_stop_after_events=9, cli_max_trials=2,
        )


# --------------------------------------------------------------------------
# End-to-end via aorta probe
# --------------------------------------------------------------------------
def _probe_e2e(output: Path, recipe: Path, argv: list[str], extra: list[str] | None = None):
    args = [
        "--recipe", str(recipe), "--output", str(output), "--ticket", "SA-1",
        *(extra or []), "--", *argv,
    ]
    result = CliRunner().invoke(probe, args)
    if result.exit_code != 0 and result.exception:
        import traceback

        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )
    return result


def _matrix_cell(output: Path):
    doc = json.loads((output / "SA-1" / "matrix.json").read_text(encoding="utf-8"))
    return doc, doc["cells"][0]


def test_e2e_stops_early_on_failures(tmp_path):
    recipe = _write(tmp_path, _PROBE_HEAD + "stop_after:\n  events: 2\n  max_trials: 6\n")
    output = tmp_path / "out"
    res = _probe_e2e(output, recipe, ["sh", "-c", "exit 1"])
    assert res.exit_code == 0, res.output
    cell_dir = output / "SA-1" / "none-none"
    # Stopped at the 2nd failure: trial_0 + trial_1 exist, trial_2 does not.
    assert (cell_dir / "trial_0" / "result.json").is_file()
    assert (cell_dir / "trial_1" / "result.json").is_file()
    assert not (cell_dir / "trial_2").exists()
    doc, cell = _matrix_cell(output)
    assert cell["trials"] == 2
    assert cell["failed_count"] == 2
    assert "stopped early" in (cell["stop_after_note"] or "")
    assert doc["stop_after"] == {"events": 2, "max_trials": 6, "event_verdict": "fail"}
    # trials_per_cell reflects the cap (max_trials), not recipe.trials (1).
    assert doc["trials_per_cell"] == 6


def test_e2e_cap_reached_when_all_pass(tmp_path):
    recipe = _write(tmp_path, _PROBE_HEAD + "stop_after:\n  events: 2\n  max_trials: 3\n")
    output = tmp_path / "out"
    res = _probe_e2e(output, recipe, ["sh", "-c", "exit 0"])
    assert res.exit_code == 0, res.output
    _doc, cell = _matrix_cell(output)
    assert cell["trials"] == 3  # ran the full cap
    assert _doc["trials_per_cell"] == 3  # cap, not recipe.trials (1)
    assert cell["passed_count"] == 3
    assert "cap reached" in (cell["stop_after_note"] or "")
    # matrix.md surfaces the column when a stop_after rule is active.
    md = (output / "SA-1" / "matrix.md").read_text(encoding="utf-8")
    assert "Stop after" in md


def test_e2e_cli_flags_override_recipe(tmp_path):
    # Recipe has no stop_after; the CLI flags introduce one.
    recipe = _write(tmp_path, _PROBE_HEAD)
    output = tmp_path / "out"
    res = _probe_e2e(
        output, recipe, ["sh", "-c", "exit 1"],
        extra=["--stop-after-events", "1", "--max-trials", "5"],
    )
    assert res.exit_code == 0, res.output
    _doc, cell = _matrix_cell(output)
    assert cell["trials"] == 1  # stopped at the first failure


def test_e2e_resume_skips_satisfied_cell(tmp_path):
    recipe = _write(tmp_path, _PROBE_HEAD + "stop_after:\n  events: 2\n  max_trials: 6\n")
    output = tmp_path / "out"
    assert _probe_e2e(output, recipe, ["sh", "-c", "exit 1"]).exit_code == 0
    trial1 = output / "SA-1" / "none-none" / "trial_1" / "result.json"
    first_mtime = trial1.stat().st_mtime
    # Re-run: the on-disk prefix already has 2 failures == target, so the
    # cell must be skipped (no re-execution, mtime unchanged).
    assert _probe_e2e(output, recipe, ["sh", "-c", "exit 1"]).exit_code == 0
    assert trial1.stat().st_mtime == first_mtime
    _doc, cell = _matrix_cell(output)
    assert cell["trials"] == 2
