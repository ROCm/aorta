"""Reaching the assembler, when the way there is not available.

The assemble went through ``srun`` with nothing catching what that call can
raise. Two ordinary situations came out as a traceback from inside a LangChain
tool: a machine with no scheduler installed raised ``FileNotFoundError``, and a
queue that never got round to the job raised ``TimeoutExpired``. Neither is a
programming error and neither is the user's fault, but both reached the chat as
one.

The scheduler is also not a requirement here, only a lift. Assembling a
fragment is about a second of CPU and needs no GPU; ``srun`` is how it reaches
a node with ROCm installed. Somewhere without a scheduler, the same assemble
runs perfectly well on the machine already running the chat.
"""

from __future__ import annotations

import subprocess

import pytest


@pytest.fixture()
def cluster(tmp_path, monkeypatch):
    pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
    import aorta.chat.tools.cluster as module

    monkeypatch.setattr(module.settings, "jobs_path", str(tmp_path), raising=False)
    return module


def _asm(cluster, source="s_endpgm\n"):
    return cluster.triage_assembly_source.func(source)


class TestAMachineWithNoScheduler:
    def test_the_assemble_runs_here_instead(self, cluster, monkeypatch):
        """The fallback: srun is a lift, not a requirement."""
        seen: list[list[str]] = []
        real = subprocess.run

        def no_srun(argv, **kwargs):
            seen.append(argv)
            if argv[0] == "srun":
                raise FileNotFoundError(2, "No such file or directory", "srun")
            return real(argv, **kwargs)

        monkeypatch.setattr(cluster.subprocess, "run", no_srun)
        cluster._assemble("true")

        assert [a[0] for a in seen] == ["srun", "bash"]

    def test_it_is_the_same_command_either_way(self, cluster, monkeypatch):
        """The fallback must not quietly assemble something different."""
        commands: list[str] = []

        def no_srun(argv, **kwargs):
            commands.append(argv[-1])
            if argv[0] == "srun":
                raise FileNotFoundError(2, "No such file or directory", "srun")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(cluster.subprocess, "run", no_srun)
        cluster._assemble("the-assemble-command")

        assert commands == ["the-assemble-command", "the-assemble-command"]

    def test_the_tool_answers_rather_than_raising(self, cluster, monkeypatch):
        """End to end: a missing scheduler used to be a traceback."""

        def no_srun(argv, **kwargs):
            if argv[0] == "srun":
                raise FileNotFoundError(2, "No such file or directory", "srun")
            return subprocess.CompletedProcess(argv, 127, "", cluster._NO_ASSEMBLER)

        monkeypatch.setattr(cluster.subprocess, "run", no_srun)
        answer = _asm(cluster)

        assert isinstance(answer, str)
        assert "No AMD assembler was found" in answer


class TestAQueueThatNeverGetsRoundToUs:
    def test_the_timeout_is_answered_not_raised(self, cluster, monkeypatch):
        def too_slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 300)

        monkeypatch.setattr(cluster.subprocess, "run", too_slow)
        answer = _asm(cluster)

        assert isinstance(answer, str)
        assert "did not finish" in answer

    def test_it_does_not_read_as_a_problem_with_the_paste(self, cluster, monkeypatch):
        """A busy cluster must not send someone off to rewrite good assembly."""

        def too_slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 300)

        monkeypatch.setattr(cluster.subprocess, "run", too_slow)
        answer = _asm(cluster)

        assert "did not assemble" not in answer
        assert "busy queue" in answer

    def test_the_wrapped_source_is_still_named(self, cluster, monkeypatch):
        """It is the only way back to what was actually submitted."""

        def too_slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 300)

        monkeypatch.setattr(cluster.subprocess, "run", too_slow)

        assert "Wrapped source kept at:" in _asm(cluster)


class TestOtherWaysTheCallCanFail:
    def test_a_broken_environment_is_reported_not_raised(self, cluster, monkeypatch):
        """OSError covers more than a missing file -- permissions, no fork."""

        def broken(argv, **kwargs):
            raise PermissionError(13, "Permission denied", "srun")

        monkeypatch.setattr(cluster.subprocess, "run", broken)
        answer = _asm(cluster)

        assert isinstance(answer, str)
        assert "could not be reached" in answer

    def test_a_timeout_is_not_swallowed_by_the_oserror_clause(self, cluster):
        """They need separate clauses; TimeoutExpired is not an OSError."""
        assert not issubclass(subprocess.TimeoutExpired, OSError)

    def test_a_missing_bash_in_the_fallback_still_answers(self, cluster, monkeypatch):
        """The fallback can fail too, and it is inside the same guard."""

        def nothing_works(argv, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", argv[0])

        monkeypatch.setattr(cluster.subprocess, "run", nothing_works)
        answer = _asm(cluster)

        assert isinstance(answer, str)
        assert "could not be reached" in answer


class TestTheOrdinaryPathIsUnchanged:
    def test_srun_is_used_when_it_is_there(self, cluster, monkeypatch):
        seen: list[str] = []

        def ok(argv, **kwargs):
            seen.append(argv[0])
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(cluster.subprocess, "run", ok)
        cluster._assemble("true")

        assert seen == ["srun"]

    def test_the_node_gets_a_single_node_and_a_time_limit(self, cluster, monkeypatch):
        """Unchanged from before, and worth keeping: this queues on a cluster."""
        seen: list[list[str]] = []

        def ok(argv, **kwargs):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(cluster.subprocess, "run", ok)
        cluster._assemble("true")

        assert "--nodes=1" in seen[0]
        assert seen[0][seen[0].index("-t") + 1] == "5"
