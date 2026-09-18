"""The CIA regressions must run in CI, not merely exist in the tree.

The general CPU job deliberately installs no ``[cia]`` extra, and
``tests/cia/conftest.py`` therefore collect-ignores the directory. The chat
workflow neither installs CIA nor names ``tests/cia``. Every required check was
green while all of the security, provider, and concurrency tests were running
nowhere.

This reads the workflow as a contract: the lane installs CIA and tests
together, proves DSPy is importable before pytest can skip anything, spans the
supported Python range, runs the directory explicitly, and feeds the existing
required CPU aggregate.
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


class TestTheCIALaneExists:
    def test_it_has_an_explicit_job(self):
        assert "cia_tests" in _jobs()

    def test_it_spans_every_supported_python(self):
        versions = _jobs()["cia_tests"]["strategy"]["matrix"]["python-version"]

        assert versions == ["3.10", "3.11", "3.12", "3.13", "3.14"]

    def test_it_installs_the_feature_and_tests_in_one_resolution(self):
        script = _script(_jobs()["cia_tests"])

        assert 'pip install -e ".[tests,cia]"' in script

    def test_it_verifies_dspy_before_pytest_can_skip_the_suite(self):
        steps = _jobs()["cia_tests"]["steps"]
        verify = next(i for i, step in enumerate(steps) if "Verify the CIA" in step["name"])
        run = next(i for i, step in enumerate(steps) if "Run the CIA" in step["name"])

        assert "import dspy" in steps[verify]["run"]
        assert verify < run

    def test_it_runs_the_cia_directory_explicitly(self):
        script = _script(_jobs()["cia_tests"])

        assert "pytest tests/cia" in script


class TestItIsPartOfTheRequiredGate:
    def test_the_cpu_aggregate_waits_for_it(self):
        needs = _jobs()["required"]["needs"]

        assert "cia_tests" in needs

    def test_the_aggregate_rejects_a_failed_cia_matrix(self):
        script = _script(_jobs()["required"])

        assert "CIA_TESTS_RESULT" in script
        assert "CIA matrix did not pass" in script
