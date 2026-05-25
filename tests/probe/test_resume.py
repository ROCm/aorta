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
