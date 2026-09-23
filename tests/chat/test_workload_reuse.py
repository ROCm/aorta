"""A workload triage is reused, and runs under an interpreter that can.

Two things an end-to-end run turned up, both specific to the workload path --
the one that runs longest and had the least care taken over it.

The kernel and assembly paths have reused a finished run since the cache was
introduced. This one did not, so a critic that rejected an answer sent the
agent round again and the second pass submitted a second cluster job for a
question already answered: another GPU allocation, and another two minutes to
reach the verdict already in hand.

And it ran the pasted script under ``sys.executable`` -- the interpreter
serving the chat, which needs langchain and Chainlit and need not have a GPU
build of torch. On a server whose torch was CPU-only the job died with "Torch
not compiled with CUDA enabled" before reaching the bug it was submitted to
find, which reads downstream as a workload with no NaN in it rather than as a
workload that never ran.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")

from aorta.chat.config import reset_settings
from aorta.chat.tools import cluster
from aorta.chat.tools.cache import ToolCache, use_tool_cache

_VERDICT = "Autopsy verdict:\n  category: numeric_silent\n  confidence: 0.62"
_SOURCE = "import torch\nprint('train')\n"


@pytest.fixture()
def submissions(tmp_path, monkeypatch):
    """Every triage that reached the cluster, with the tools pointed at tmp."""
    monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path))
    monkeypatch.setenv("AORTA_CHAT_ALLOW_CLUSTER_JOBS", "true")
    reset_settings()
    seen: list[list[str]] = []

    def fake(extra_args, label):
        seen.append(extra_args)
        return _VERDICT

    monkeypatch.setattr(cluster, "_run_triage", fake)
    yield seen
    reset_settings()


class TestAFinishedRunIsReused:
    def test_the_same_source_does_not_submit_twice(self, submissions):
        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")
            cluster.triage_workload.func(source=_SOURCE, label="train")

        assert len(submissions) == 1, "a second cluster job was submitted"

    def test_the_reuse_is_disclosed(self, submissions):
        """The user is told why the answer came back instantly."""
        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")
            again = cluster.triage_workload.func(source=_SOURCE, label="train")

        assert "no second cluster job" in again
        assert "numeric_silent" in again, "the verdict itself must still be there"

    def test_a_different_workload_still_runs(self, submissions):
        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")
            cluster.triage_workload.func(source="import torch  # other\n", label="train")

        assert len(submissions) == 2

    def test_a_command_is_reused_too(self, submissions):
        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(command="python train.py", label="t")
            cluster.triage_workload.func(command="python train.py", label="t")

        assert len(submissions) == 1

    def test_one_conversation_does_not_answer_another(self, submissions):
        """The cache is per conversation, as it is for the other two tools."""
        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")
        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")

        assert len(submissions) == 2


class TestOnlyAFinishedRunIsRemembered:
    def test_a_launch_failure_is_retried(self, tmp_path, monkeypatch):
        """A transient failure cached would replay for the whole conversation."""
        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path))
        monkeypatch.setenv("AORTA_CHAT_ALLOW_CLUSTER_JOBS", "true")
        reset_settings()
        seen: list[list[str]] = []

        def failing(extra_args, label):
            seen.append(extra_args)
            return "The triage did not finish within 1800s."

        monkeypatch.setattr(cluster, "_run_triage", failing)
        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")
            cluster.triage_workload.func(source=_SOURCE, label="train")
        reset_settings()

        assert len(seen) == 2, "a failed run was remembered instead of retried"


class TestTheInterpreterIsTheOneThatCanRunIt:
    def test_it_defaults_to_the_server_interpreter(self, submissions):
        import sys

        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")

        assert sys.executable in submissions[0][1]

    def test_a_configured_interpreter_is_used_instead(self, submissions, monkeypatch):
        """The chat server's torch need not be the one with a GPU in it."""
        monkeypatch.setenv("AORTA_CHAT_WORKLOAD_PYTHON", "/opt/rocm-venv/bin/python")
        reset_settings()

        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")

        assert "/opt/rocm-venv/bin/python" in submissions[0][1]

    def test_the_script_still_runs_as_the_user_would_run_it(self, submissions, monkeypatch):
        """No arguments appended: that broke argparse workloads once already."""
        monkeypatch.setenv("AORTA_CHAT_WORKLOAD_PYTHON", "/opt/rocm-venv/bin/python")
        reset_settings()

        with use_tool_cache(ToolCache()):
            cluster.triage_workload.func(source=_SOURCE, label="train")

        command = submissions[0][1]
        assert command.count(" ") == 1, f"arguments were appended: {command}"
