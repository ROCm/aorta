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
    assert 'pytest -q -n "$PYTEST_WORKERS" -m "$PYTEST_MARKERS"' in script


def test_the_lane_runs_every_test_in_one_of_its_three_parts():
    job = _jobs()["aorlta_tests"]
    legs = job["strategy"]["matrix"]["include"]
    parts = {leg["part"]: (leg["workers"], leg["threads"], leg["markers"]) for leg in legs}

    # never more torch threads than the runner's four cores
    assert parts == {"light": ("4", "1", "not heavy and not cpu_run"), "heavy": ("2", "2", "heavy and not cpu_run"),
                     "cpu run": ("0", "4", "cpu_run")}
    for heavy in (False, True):  # every test, whatever its markers, runs in exactly one part
        for cpu_run in (False, True):
            marks = {"heavy": heavy, "cpu_run": cpu_run}
            assert sum(eval(m, {}, marks) for _, _, m in parts.values()) == 1, marks
    step = next(s for s in job["steps"] if "pytest" in str(s.get("run")))
    assert step["env"]["OMP_NUM_THREADS"] == "${{ matrix.threads }}"
    assert job["strategy"]["fail-fast"] is False
    assert 'if [ "$rc" -eq 5 ]' in _script(job)  # a part with no test on a branch (the markers come later) passes


def test_the_lane_gives_freed_memory_back_and_reports_its_peak():
    step = next(s for s in _jobs()["aorlta_tests"]["steps"] if "pytest" in str(s.get("run")))

    assert step["env"]["MALLOC_MMAP_THRESHOLD_"] == "65536" and step["env"]["MALLOC_ARENA_MAX"] == "2"
    assert "peak memory in use" in step["run"] and 'exit "$rc"' in step["run"]


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
