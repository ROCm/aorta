"""Launch is reached through one function, and forwards what it is given.

``launch`` exists so a scheduler-less backend is a new branch in one place
rather than an edit at every call site. It is Slurm today and the tests below
are about Slurm, but they go through the seam, which is what keeps the seam
honest -- an unused abstraction rots.

The env-var test is the interesting one. Those variables used to reach the job
by being set in a subprocess that submitted it; when the driver moved
in-process nothing set them, the sanitizer binary was simply absent, and the
sweep reported "not checked" while every layer above it carried on.
"""

from __future__ import annotations

import pytest

from aorta.cia.launch import launch
from aorta.cia.launch.cluster import build_sbatch_script


def test_the_seam_reports_a_missing_scheduler_rather_than_raising(tmp_path, monkeypatch):
    """A launch that did not happen must not read as one that did."""
    monkeypatch.setattr("aorta.cia.launch.cluster.sbatch_available", lambda: False)
    job_id, error = launch(
        command="true",
        job_name="j",
        log_path=str(tmp_path / "j.log"),
        script_path=tmp_path / "j.sbatch",
    )
    assert job_id == ""
    assert "sbatch" in error


def test_requested_variables_are_exported_in_the_job():
    """The regression this file exists for.

    The sanitizer needs its backend path and a preload; both must appear as
    exports in the script, because the submitting process's environment is no
    longer the job's environment.
    """
    script = build_sbatch_script(
        command="aorta sweep run --recipe r.yaml",
        job_name="sweep",
        log_path="/tmp/sweep.log",
        env_vars={"ROCJITSU_BUILD": "/opt/rj", "LD_PRELOAD": "/opt/libstdc++.so.6"},
    )
    assert "export ROCJITSU_BUILD=/opt/rj" in script
    assert "export LD_PRELOAD=/opt/libstdc++.so.6" in script


def test_a_value_needing_quoting_survives_it():
    script = build_sbatch_script(
        command="true",
        job_name="j",
        log_path="/tmp/j.log",
        env_vars={"CIA_SEARCH_ROOTS": "/a b:/c"},
    )
    assert "export CIA_SEARCH_ROOTS='/a b:/c'" in script


def test_the_command_and_log_reach_the_script():
    script = build_sbatch_script(
        command="python train.py", job_name="train", log_path="/tmp/train.log"
    )
    assert "python train.py" in script
    assert "/tmp/train.log" in script
    assert "--job-name=train" in script


def test_a_pinned_node_becomes_a_directive_only_when_it_exists(monkeypatch):
    """An unknown node must not become a directive that never schedules."""
    monkeypatch.setattr("aorta.cia.launch.cluster.node_exists", lambda n: False)
    script = build_sbatch_script(
        command="true", job_name="j", log_path="/tmp/j.log", node="ghost-node"
    )
    assert "--nodelist" not in script


@pytest.mark.parametrize("missing", ["command", "job_name", "log_path", "script_path"])
def test_the_seam_requires_what_any_backend_would_need(missing):
    """Keyword-only and required: the signature is the backend contract."""
    import inspect

    parameter = inspect.signature(launch).parameters[missing]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    if missing != "log_path":
        assert parameter.default is inspect.Parameter.empty
