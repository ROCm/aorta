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


def test_custom_patterns_recipe_does_not_break_dispatcher_json(tmp_path):
    """Regression for PR #197 round-7 review: a probe-mode recipe with
    ``custom_patterns`` used to crash the dispatcher's per-trial
    JSON write because ``probe_extras_payload['custom_patterns']``
    held :class:`CompiledPattern` objects (compiled ``re.Pattern`` +
    ``CodeType``), neither of which is JSON-serializable. The
    dispatcher now sanitizes the tuple down to a JSON-safe summary
    list before ``json.dump``; this test pins the contract by
    running a real probe through ``CliRunner`` with
    ``custom_patterns`` present and asserting the dispatcher's
    ``trial_d*_m*_t*.json`` is both written and round-trippable.
    """
    output = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_with_phase_2_keys.yaml"),
            "--output",
            str(output),
            "--ticket",
            "CUSTOM-PATTERNS-1",
            "--",
            "bash",
            "-c",
            "echo hi; exit 0",
        ],
    )
    assert result.exit_code == 0, result.output
    cell_dir = output / "CUSTOM-PATTERNS-1" / "none-none"
    # The dispatcher per-trial JSONs live under the workload
    # subdirectory (``_subprocess`` for ``aorta probe``); the
    # cell-root ``trial_<n>/`` holds the SubprocessWorkload's own
    # ``result.json``.
    dispatcher_jsons = list((cell_dir / "_subprocess").glob("trial_d*_m*_t*.json"))
    assert dispatcher_jsons, (
        "dispatcher per-trial JSON was not written -- probe-mode + "
        "custom_patterns regressed the dispatcher's JSON-serialization "
        "path"
    )
    # Round-trip the dispatcher JSON: if any field still carries a
    # CompiledPattern / re.Pattern, ``json.loads`` would never have
    # gotten here, but the explicit reload + sanity check makes the
    # regression obvious in failure output.
    doc = json.loads(dispatcher_jsons[0].read_text(encoding="utf-8"))
    summarized = doc["config"]["_aorta_probe_extras"]["custom_patterns"]
    assert isinstance(summarized, list) and summarized, (
        f"_aorta_probe_extras.custom_patterns should be a non-empty "
        f"list of summary dicts after sanitization; got {summarized!r}"
    )
    entry = summarized[0]
    assert entry["detector_id"] == "custom:hip_oom"
    assert entry["regex"] == "hipErrorOutOfMemory"
    assert entry["on_match"] == "fail"


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
