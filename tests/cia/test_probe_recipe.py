"""Escalation re-runs the job's own recipe, or it does not run.

The probe used to ``find /root /mnt -name Residual-NaN-Repro.yaml`` on the node
and sweep whatever came back, whichever job was under autopsy. It escalates
below 0.85 confidence, which is exactly where a static wait-hazard finding
lands -- so on a machine with that demo installed, a WaitCheck autopsy swept the
NaN demo and folded the results into its verdict, with nothing in the verdict
saying so.

The plan rejects this in §1: a canned tool is worse than an absent one, because
it gets picked and silently substitutes the demo. Here it substituted the demo
inside a verdict.
"""

from __future__ import annotations

import inspect

import pytest

from aorta.cia.autopsy import probe
from aorta.cia.launch.job import JobRecord


def _job(**overrides) -> JobRecord:
    fields = {
        "job_id": "cia-test",
        "node": "node1",
        "recipe": "rms_norm NaN at step 5",  # a label, not a path
        "launched_at": "2026-01-01T00:00:00Z",
        "log_path": "/tmp/cia-test/watch.log",
        "aorta_output": "/tmp/cia-test/aorta",
    }
    return JobRecord(**{**fields, **overrides})


class _Launched(Exception):
    """Raised once the sweep has been issued, to stop before the four-hour wait."""

    def __init__(self, cmd: str):
        self.cmd = cmd


def _capture_sweep(monkeypatch) -> list[str]:
    """Record the command the probe issues, and go no further.

    ``run_aorta_probe`` polls for matrix.json for PROBE_TIMEOUT_SEC and then
    scps the result, neither of which belongs in a unit test. What is under
    test is the command it decided to run.
    """
    sent: list[str] = []

    def fake_ssh(node, cmd, **kwargs):
        sent.append(cmd)
        raise _Launched(cmd)

    monkeypatch.setattr(probe, "_ssh", fake_ssh)
    return sent


def _executable_source(module) -> str:
    """The module's code with its docstrings dropped.

    The docstrings name the demo recipe deliberately, to say what was removed
    and why. Asserting on raw source would make that explanation fail the test
    that the explanation is about.
    """
    import ast

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body.pop(0)
    return ast.unparse(tree)


class TestTheDemoIsGone:
    def test_no_recipe_name_is_hardcoded(self):
        assert "Residual-NaN-Repro" not in _executable_source(probe)

    def test_it_does_not_go_looking_for_one_on_the_node(self):
        code = _executable_source(probe)
        assert "find /root" not in code
        assert "/mnt" not in code


class TestResolvingTheRecipe:
    def test_a_job_that_recorded_one_supplies_it(self, tmp_path):
        job = _job(recipe_path="/jobs/mine.yaml", sidecar_path="/jobs/mine.json")
        assert probe.resolve_recipe(tmp_path, job) == ("/jobs/mine.yaml", "/jobs/mine.json")

    def test_a_job_that_recorded_none_resolves_to_nothing(self, tmp_path):
        """The label in `recipe` is prose, and prose is not a path."""
        assert probe.resolve_recipe(tmp_path, _job()) == ("", "")

    def test_the_manifest_is_consulted_second(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text(
            "paths:\n  recipe: from/manifest.yaml\n  mitigations: from/manifest.json\n"
        )
        assert probe.resolve_recipe(tmp_path, _job()) == (
            "from/manifest.yaml",
            "from/manifest.json",
        )

    def test_the_job_wins_over_the_manifest(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text("paths:\n  recipe: from/manifest.yaml\n")
        job = _job(recipe_path="from/the/job.yaml")
        assert probe.resolve_recipe(tmp_path, job)[0] == "from/the/job.yaml"

    def test_a_manifest_without_a_recipe_resolves_to_nothing(self, tmp_path):
        (tmp_path / "manifest.yaml").write_text("paths:\n  stderr: logs/watch.stderr.log\n")
        assert probe.resolve_recipe(tmp_path, _job()) == ("", "")


class TestRefusingToEscalate:
    @pytest.fixture(autouse=True)
    def _head_node(self, monkeypatch):
        monkeypatch.setenv("CIA_SSH_HOST", "head.example.invalid")

    def test_an_unresolvable_recipe_stops_the_sweep(self, tmp_path, monkeypatch, capsys):
        """Better no escalation than an escalation of somebody else's workload."""
        sent = _capture_sweep(monkeypatch)

        assert probe.run_aorta_probe(tmp_path, _job()) is None
        assert not sent, "nothing may be run when the recipe is unknown"
        assert "no recipe" in capsys.readouterr().out

    def test_a_resolvable_recipe_is_the_one_swept(self, tmp_path, monkeypatch):
        sent = _capture_sweep(monkeypatch)

        job = _job(recipe_path="/jobs/cia-test/recipe.yaml")
        with pytest.raises(_Launched):
            probe.run_aorta_probe(tmp_path, job)

        assert "/jobs/cia-test/recipe.yaml" in sent[0]


class TestTheSweepCommand:
    @pytest.fixture(autouse=True)
    def _head_node(self, monkeypatch):
        monkeypatch.setenv("CIA_SSH_HOST", "head.example.invalid")

    def test_a_sanitizer_recipe_is_not_given_a_sidecar_it_rejects(
        self, tmp_path, monkeypatch
    ):
        """check_recipe_mode says a sanitizer recipe refuses --mitigations-file."""
        recipe = tmp_path / "san.yaml"
        recipe.write_text("mode: sanitizer\n")
        sent = _capture_sweep(monkeypatch)

        job = _job(recipe_path=str(recipe), sidecar_path="/jobs/side.json")
        with pytest.raises(_Launched):
            probe.run_aorta_probe(tmp_path, job)

        assert "--mitigations-file" not in sent[0]

    def test_a_recipe_that_takes_one_gets_it(self, tmp_path, monkeypatch):
        recipe = tmp_path / "sweep.yaml"
        recipe.write_text("mode: matrix\n")
        sent = _capture_sweep(monkeypatch)

        job = _job(recipe_path=str(recipe), sidecar_path="/jobs/side.json")
        with pytest.raises(_Launched):
            probe.run_aorta_probe(tmp_path, job)

        assert "--mitigations-file" in sent[0]

    def test_a_path_with_a_space_does_not_split_the_command(self, tmp_path, monkeypatch):
        recipe = tmp_path / "my recipe.yaml"
        recipe.write_text("mode: matrix\n")
        sent = _capture_sweep(monkeypatch)

        with pytest.raises(_Launched):
            probe.run_aorta_probe(tmp_path, _job(recipe_path=str(recipe)))

        assert "my recipe.yaml" in sent[0]
        assert "'" in sent[0], "the path has to survive as one argument"
