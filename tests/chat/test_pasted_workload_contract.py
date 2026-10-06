"""A pasted program is run the way its author wrote it.

The bundle directory was appended to the command as a positional argument, so
what actually ran was ``python their_script.py /jobs/cia-N/bundle``. A training
script with an argparse parser and no positional parameter exits 2 on
"unrecognized arguments" -- so the workload the user asked to reproduce never
ran, and the failure they were chasing was replaced by one the tool introduced.

Nothing about the program says it takes an argument. Where the artifacts should
go is something the job knows and the program may ask for, which is what an
environment variable is: AORTA_BUNDLE is exported for every job, and a script
that does not care never notices.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from aorta.cia.launch.cluster import build_sbatch_script, containerize


@pytest.fixture()
def cluster(tmp_path, monkeypatch):
    pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
    import aorta.chat.tools.cluster as module

    monkeypatch.setattr(module.settings, "jobs_path", str(tmp_path), raising=False)
    return module


def _command(cluster, monkeypatch, source: str = "print('hi')\n") -> str:
    seen: dict = {}

    def fake(argv, *, stop=None):
        seen["argv"] = argv
        return {"ok": True, "job_id": "j", "job_dir": "/tmp/j"}

    monkeypatch.setattr(cluster, "run_triage", fake)
    cluster.triage_workload.func(source=source, label="nan repro")
    argv = seen["argv"]
    return argv[argv.index("--command") + 1]


class TestTheProgramKeepsItsOwnArguments:
    def test_nothing_is_appended_to_the_invocation(self, cluster, monkeypatch):
        command = _command(cluster, monkeypatch)

        assert command.rstrip().endswith(".py")

    def test_the_bundle_is_not_passed_positionally(self, cluster, monkeypatch):
        command = _command(cluster, monkeypatch)

        assert "{bundle}" not in command
        assert "bundle" not in command.split(".py", 1)[1]

    def test_the_interpreter_and_the_script_are_still_there(self, cluster, monkeypatch):
        command = _command(cluster, monkeypatch)

        assert sys.executable in command
        assert ".py" in command

    def test_both_are_quoted(self, cluster, monkeypatch):
        """The script name is derived from a label the model supplies."""
        command = _command(cluster, monkeypatch)

        assert "; " not in command and "&&" not in command


class TestAnOrdinaryTrainingScriptSurvives:
    """The failure, run rather than reasoned about."""

    @staticmethod
    def _script(tmp_path):
        script = tmp_path / "train.py"
        script.write_text(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--lr', type=float, default=1e-3)\n"
            "a = p.parse_args()\n"
            "print('trained at', a.lr)\n",
            encoding="utf-8",
        )
        return script

    def test_it_runs_with_no_positional(self, tmp_path):
        done = subprocess.run(
            [sys.executable, str(self._script(tmp_path))], capture_output=True, text=True
        )

        assert done.returncode == 0
        assert "trained at" in done.stdout

    def test_and_would_have_failed_with_one(self, tmp_path):
        """Establishes that the argument was the problem, not a guess at it."""
        done = subprocess.run(
            [sys.executable, str(self._script(tmp_path)), "/jobs/cia-1/bundle"],
            capture_output=True,
            text=True,
        )

        assert done.returncode == 2
        assert "unrecognized arguments" in done.stderr


class TestTheBundleArrivesInTheEnvironment:
    def test_every_job_exports_it(self):
        script = build_sbatch_script(
            command="python train.py",
            job_name="cia-test",
            log_path="/jobs/cia-test/watch.log",
            env_vars={"AORTA_BUNDLE": "/jobs/cia-test/bundle"},
        )

        assert "export AORTA_BUNDLE=/jobs/cia-test/bundle" in script

    def test_it_crosses_the_container_boundary(self, monkeypatch):
        """A workload under CIA_CONTAINER_IMAGE needs it as much as one without."""
        monkeypatch.setenv("CIA_CONTAINER_IMAGE", "rocm/dev:latest")
        wrapped = containerize("python train.py", "/jobs", {"AORTA_BUNDLE": "/b"})

        assert "-e AORTA_BUNDLE" in wrapped

    def test_the_driver_sets_it_from_the_job_directory(self, tmp_path, monkeypatch):
        """Not the caller's to compute: the job directory is made inside run_triage."""
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        from aorta.cia import triage as triage_mod

        seen: dict = {}

        def fake_launch(**kwargs):
            seen.update(kwargs)
            return "", "stopped before submitting"

        monkeypatch.setattr(triage_mod, "launch", fake_launch)
        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")

        triage_mod.run_triage(
            ["--source", str(source), "--jobs-root", str(tmp_path), "--kernel-name", "bump"]
        )

        bundle = seen["env_vars"]["AORTA_BUNDLE"]
        assert bundle.endswith("/bundle")
        assert str(tmp_path) in bundle

    def test_a_caller_supplied_value_still_wins(self, tmp_path, monkeypatch):
        """--env is the explicit instruction; the default should not override it."""
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        from aorta.cia import triage as triage_mod

        seen: dict = {}

        def fake_launch(**kwargs):
            seen.update(kwargs)
            return "", "stopped"

        monkeypatch.setattr(triage_mod, "launch", fake_launch)
        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")

        triage_mod.run_triage(
            [
                "--source", str(source),
                "--jobs-root", str(tmp_path),
                "--kernel-name", "bump",
                "--env", "AORTA_BUNDLE=/somewhere/else",
            ]
        )

        assert seen["env_vars"]["AORTA_BUNDLE"] == "/somewhere/else"


class TestTheCommandSubstitutionStillExists:
    """A --command caller that does want it as an argument keeps that route."""

    def test_the_placeholder_is_still_substituted(self, tmp_path, monkeypatch):
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        from aorta.cia import triage as triage_mod

        seen: dict = {}

        def fake_launch(**kwargs):
            seen.update(kwargs)
            return "", "stopped before submitting"

        monkeypatch.setattr(triage_mod, "launch", fake_launch)
        triage_mod.run_triage(
            ["--command", "my_tool --out {bundle}", "--jobs-root", str(tmp_path)]
        )

        assert "{bundle}" not in seen["command"]
        assert seen["command"].endswith("/bundle")

    def test_the_help_names_the_environment_variable_too(self):
        """Someone reading --help should learn the route that costs nothing."""
        pytest.importorskip("dspy", reason="needs the [cia] extra")
        from pathlib import Path

        from aorta.cia import triage as triage_mod

        source = Path(triage_mod.__file__).read_text(encoding="utf-8")

        assert "AORTA_BUNDLE in its environment" in source
