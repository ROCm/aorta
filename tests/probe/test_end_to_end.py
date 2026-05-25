"""End-to-end smoke for ``aorta probe`` (issue #188 FR 1.3, 1.14).

Invokes the real ``aorta probe`` Click command via :class:`CliRunner`,
runs a 1-cell recipe with ``bash -c 'echo hi'`` as the user command,
and asserts the artifact tree matches the rubric.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from aorta.cli.probe import probe

FIXTURES = Path(__file__).parent / "fixtures"


def test_smoke_passes(tmp_path):
    """FR 1.3 -- full smoke with the documented artifact tree."""
    output = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            str(output),
            "--ticket",
            "SMOKE-1",
            "--",
            "bash",
            "-c",
            "echo hi; exit 0",
        ],
    )
    assert result.exit_code == 0, result.output
    cell_dir = output / "SMOKE-1" / "none-none"
    trial = cell_dir / "trial_0"
    for artifact in ("stdout.log", "stderr.log", "result.json"):
        assert (trial / artifact).is_file(), f"missing {artifact} in {trial}"
    doc = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "pass"
    assert doc["exit_code"] == 0
    stdout = (trial / "stdout.log").read_text(encoding="utf-8")
    assert "hi" in stdout


def test_no_ticket_routed_to_no_ticket_slug(tmp_path):
    """FR 1.14 -- omitted --ticket routes to '_no_ticket_/' under output."""
    # The fixture has a ticket; override it to None by writing a fresh recipe.
    recipe = tmp_path / "no_ticket.yaml"
    recipe.write_text(
        "schema_version: 1\n"
        "mode: probe\n"
        "trials: 1\n"
        "mitigation_axis: [none]\n"
        "diagnostic_axis: [none]\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(recipe),
            "--output",
            str(output),
            "--",
            "bash",
            "-c",
            "echo hi",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output / "_no_ticket_" / "none-none" / "trial_0" / "result.json").is_file()
