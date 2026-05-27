"""Resume semantics tests for ``aorta probe`` (issue #188 FR 1.4)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from aorta.cli.probe import probe
from aorta.probe.resume import is_trial_complete

FIXTURES = Path(__file__).parent / "fixtures"


# ---- Pure resume-helper unit tests ---------------------------------------


def test_is_trial_complete_missing(tmp_path):
    """No result.json -> incomplete."""
    assert is_trial_complete(tmp_path) is False


def test_is_trial_complete_empty_file(tmp_path):
    (tmp_path / "result.json").write_text("", encoding="utf-8")
    assert is_trial_complete(tmp_path) is False


def test_is_trial_complete_invalid_json(tmp_path):
    (tmp_path / "result.json").write_text("{not json", encoding="utf-8")
    assert is_trial_complete(tmp_path) is False


def test_is_trial_complete_missing_verdict(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    assert is_trial_complete(tmp_path) is False


def test_is_trial_complete_blank_verdict(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"verdict": ""}), encoding="utf-8")
    assert is_trial_complete(tmp_path) is False


def test_is_trial_complete_pass(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps({"verdict": "pass", "exit_code": 0}), encoding="utf-8"
    )
    assert is_trial_complete(tmp_path) is True


def test_is_trial_complete_fail_still_counts(tmp_path):
    """A failed trial is still 'complete' -- the operator can decide to re-run."""
    (tmp_path / "result.json").write_text(
        json.dumps({"verdict": "fail", "exit_code": 1}), encoding="utf-8"
    )
    assert is_trial_complete(tmp_path) is True


# ---- End-to-end resume via the CLI ---------------------------------------


def _invoke_probe(output: Path, recipe: Path) -> int:
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(recipe),
            "--output",
            str(output),
            "--ticket",
            "RESUME-1",
            "--",
            "sh",
            "-c",
            "echo run-marker; exit 0",
        ],
    )
    if result.exit_code != 0:
        print(result.output)
        if result.exception:
            import traceback

            traceback.print_exception(
                type(result.exception), result.exception, result.exception.__traceback__
            )
    return result.exit_code


def test_skips_completed_cell(tmp_path):
    """Second invocation does NOT re-run a completed cell."""
    output = tmp_path / "out"
    rc1 = _invoke_probe(output, FIXTURES / "probe_minimal.yaml")
    assert rc1 == 0
    cell_dir = output / "RESUME-1" / "none-none"
    trial0 = cell_dir / "trial_0"
    result_path = trial0 / "result.json"
    assert result_path.is_file()
    first_mtime = result_path.stat().st_mtime
    stdout_first = (trial0 / "stdout.log").read_text(encoding="utf-8")
    assert "run-marker" in stdout_first

    # Second invocation: same output dir / ticket -> cell is already done,
    # the runner must skip it. result.json must be byte-equivalent (same
    # mtime) because the trial was not re-executed.
    rc2 = _invoke_probe(output, FIXTURES / "probe_minimal.yaml")
    assert rc2 == 0
    second_mtime = result_path.stat().st_mtime
    assert second_mtime == first_mtime, (
        f"result.json mtime changed ({first_mtime} -> {second_mtime}); "
        "cell was re-executed when it should have been skipped"
    )


def test_skipped_cell_records_real_counts_in_matrix(tmp_path):
    """Resume short-circuit must surface real trial counts in matrix.json.

    Regression for PR #194 review: the short-circuit previously returned
    ``trials=[]`` which made ``aggregate_cell`` record
    ``trials=0/passed=0/failed=0`` for a skipped-but-complete cell -- a
    silently incorrect matrix on every resumed run. The runner must
    hydrate the dispatcher's per-trial JSONs into TrialResult objects so
    the matrix reflects what actually ran.
    """
    output = tmp_path / "out"
    rc1 = _invoke_probe(output, FIXTURES / "probe_minimal.yaml")
    assert rc1 == 0
    matrix_path = output / "RESUME-1" / "matrix.json"
    first_doc = json.loads(matrix_path.read_text(encoding="utf-8"))
    first_cells = {c["name"]: c for c in first_doc["cells"]}
    assert first_cells["none-none"]["trials"] == 1
    assert first_cells["none-none"]["passed_count"] == 1

    rc2 = _invoke_probe(output, FIXTURES / "probe_minimal.yaml")
    assert rc2 == 0
    second_doc = json.loads(matrix_path.read_text(encoding="utf-8"))
    second_cells = {c["name"]: c for c in second_doc["cells"]}
    assert second_cells["none-none"]["trials"] == 1, (
        "resumed cell reported trials=0; the short-circuit returned an empty "
        "trials list to aggregate_cell instead of hydrating dispatcher JSONs"
    )
    assert second_cells["none-none"]["passed_count"] == 1
    assert second_cells["none-none"]["failed_count"] == 0


def test_resume_with_reduced_trial_count_slices_to_current_recipe(tmp_path):
    """When the resumed directory has MORE completed trials than the
    current recipe asks for, the short-circuit must slice the
    hydrated list (and trial paths) down to ``effective_trials``.

    Regression for PR #194 review: a user who re-ran with ``trials: 1``
    after a previous ``trials: 3`` left a directory with three
    completed trial dirs on disk. Without the slice, ``aggregate_cell``
    saw all three and matrix.md reported the stale extra trials. We
    only assert the count is correct here; the stale dirs on disk are
    left in place so the operator can still inspect prior runs.
    """
    output = tmp_path / "out"

    # First run with trials: 3.
    recipe_3 = tmp_path / "trials_3.yaml"
    recipe_3.write_text(
        "schema_version: 1\n"
        "mode: probe\n"
        "ticket: RESUME-SLICE\n"
        "trials: 3\n"
        "mitigation_axis: [none]\n"
        "diagnostic_axis: [none]\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    rc1 = runner.invoke(
        probe,
        [
            "--recipe",
            str(recipe_3),
            "--output",
            str(output),
            "--ticket",
            "RESUME-SLICE",
            "--",
            "sh",
            "-c",
            "exit 0",
        ],
    ).exit_code
    assert rc1 == 0
    cell_dir = output / "RESUME-SLICE" / "none-none"
    # Three trial dirs landed.
    assert (cell_dir / "trial_0" / "result.json").is_file()
    assert (cell_dir / "trial_1" / "result.json").is_file()
    assert (cell_dir / "trial_2" / "result.json").is_file()

    # Re-run with the SAME output/ticket but trials: 1.
    recipe_1 = tmp_path / "trials_1.yaml"
    recipe_1.write_text(
        "schema_version: 1\n"
        "mode: probe\n"
        "ticket: RESUME-SLICE\n"
        "trials: 1\n"
        "mitigation_axis: [none]\n"
        "diagnostic_axis: [none]\n",
        encoding="utf-8",
    )
    rc2 = runner.invoke(
        probe,
        [
            "--recipe",
            str(recipe_1),
            "--output",
            str(output),
            "--ticket",
            "RESUME-SLICE",
            "--",
            "sh",
            "-c",
            "exit 0",
        ],
    ).exit_code
    assert rc2 == 0

    matrix_doc = json.loads((output / "RESUME-SLICE" / "matrix.json").read_text(encoding="utf-8"))
    cells = {c["name"]: c for c in matrix_doc["cells"]}
    assert cells["none-none"]["trials"] == 1, (
        f"matrix reported {cells['none-none']['trials']} trials; expected 1 "
        "(the resume short-circuit returned stale extra trials instead of "
        "slicing to the current recipe's effective_trials)"
    )


def test_reruns_truncated_result_json(tmp_path):
    """Half a `{` in result.json -> trial is re-executed."""
    output = tmp_path / "out"
    rc1 = _invoke_probe(output, FIXTURES / "probe_minimal.yaml")
    assert rc1 == 0
    trial0 = output / "RESUME-1" / "none-none" / "trial_0"
    result_path = trial0 / "result.json"

    # Truncate to half a JSON object.
    result_path.write_text("{", encoding="utf-8")
    first_mtime = result_path.stat().st_mtime

    rc2 = _invoke_probe(output, FIXTURES / "probe_minimal.yaml")
    assert rc2 == 0
    # The trial was re-run; result.json was overwritten with a valid doc.
    doc = json.loads(result_path.read_text(encoding="utf-8"))
    assert doc["verdict"] == "pass"
    second_mtime = result_path.stat().st_mtime
    # mtime should change since we overwrote the truncated file.
    assert second_mtime >= first_mtime
