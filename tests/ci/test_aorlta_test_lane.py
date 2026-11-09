"""AORLTA's tests must run in CI and gate merges, without slowing other pull requests.

``aorlta/`` is a separate package with its own pins and Python, so the general
CPU job neither installs it nor collects its tests (``pytest.ini`` has
``testpaths = tests``). Its own lane in cpu-tests.yml runs them, decided by an
allow-list of paths, and feeds the existing required ``CPU tests`` aggregate.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "cpu-tests.yml"


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _script(job: dict) -> str:
    return "\n".join(str(step.get("run") or "") for step in job["steps"])


def _gate_step(step_id: str) -> str:
    return next(s for s in _jobs()["changes"]["steps"] if s.get("id") == step_id)["run"]


def test_the_lane_installs_the_package_with_its_pins_and_runs_its_tests():
    job = _jobs()["aorlta_tests"]
    script = _script(job)

    assert job["defaults"]["run"]["working-directory"] == "aorlta"
    assert "-r requirements-dev.txt" in script and "pip install -e . --no-deps" in script
    assert "download.pytorch.org/whl/cpu" in script
    assert "pytest -q -n auto" in script


def test_the_lane_runs_on_its_own_paths_and_fails_open():
    job = _jobs()["aorlta_tests"]
    gate = _gate_step("aorlta")

    assert job["if"] == "needs.changes.outputs.aorlta == 'true'"
    assert _jobs()["changes"]["outputs"]["aorlta"] == "${{ steps.aorlta.outputs.aorlta }}"
    assert "aorlta/*|.github/workflows/cpu-tests.yml)" in gate
    assert gate.count('echo "aorlta=true" >> "$GITHUB_OUTPUT"') >= 4  # not a PR, and each way the list can fail


def test_the_general_suite_skips_a_change_only_to_aorlta():
    assert "'aorlta/*'" in _gate_step("filter")


def test_the_required_aggregate_waits_for_it_and_rejects_a_failure():
    required = _jobs()["required"]
    script = _script(required)

    assert "aorlta_tests" in required["needs"]
    assert "AORLTA_TESTS_RESULT" in script and "AORLTA suite did not pass" in script
