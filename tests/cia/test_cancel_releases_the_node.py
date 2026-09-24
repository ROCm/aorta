"""Giving up on the answer has to give back the node.

The waits became interruptible so an abandoned triage would stop occupying a
worker thread and stop calling the model. It stopped waiting, and that was all
it stopped: the job it had already submitted stayed in the scheduler, holding a
GPU node until its own time limit -- four hours by default.

That is the more expensive leak of the two. A chat turn that timed out left a
node running work whose result nobody would read, and the next person to ask for
one queued behind it.
"""

from __future__ import annotations

import subprocess

import pytest

from aorta.cia.launch import cancel
from aorta.cia.launch.cluster import cancel_sbatch


def _scancel(monkeypatch, *, returncode: int = 0, stderr: str = "", raises=None):
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, "", stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


class TestCancellingAnAllocation:
    def test_it_asks_the_scheduler_to_release_the_job(self, monkeypatch):
        seen = _scancel(monkeypatch)

        assert cancel_sbatch("12345") == (True, "")
        assert seen["argv"] == ["scancel", "12345"]

    def test_a_job_that_already_finished_is_not_an_error(self, monkeypatch):
        """The race between a job ending and this call is ordinary."""
        _scancel(monkeypatch, returncode=0)

        cancelled, why = cancel_sbatch("12345")

        assert cancelled and why == ""

    def test_a_refusal_is_reported_rather_than_raised(self, monkeypatch):
        _scancel(monkeypatch, returncode=1, stderr="Invalid job id specified\n")

        cancelled, why = cancel_sbatch("12345")

        assert not cancelled
        assert "Invalid job id" in why

    def test_no_scheduler_is_reported_rather_than_raised(self, monkeypatch):
        """The caller is already giving up; this must not raise on top of that."""
        _scancel(monkeypatch, raises=FileNotFoundError(2, "no such file", "scancel"))

        cancelled, why = cancel_sbatch("12345")

        assert not cancelled
        assert "scancel" in why

    def test_an_empty_job_id_asks_for_nothing(self, monkeypatch):
        seen = _scancel(monkeypatch)

        cancelled, why = cancel_sbatch("")

        assert not cancelled and "no scheduler job id" in why
        assert "argv" not in seen, "it should not have run scancel at all"


class TestItGoesThroughTheSeam:
    """The same reason launch() is a seam: one place a backend branches."""

    def test_cancel_reaches_the_slurm_implementation(self, monkeypatch):
        seen = _scancel(monkeypatch)

        assert cancel("777") == (True, "")
        assert seen["argv"] == ["scancel", "777"]

    def test_the_seam_exports_both_halves(self):
        import aorta.cia.launch as seam

        assert callable(seam.launch) and callable(seam.cancel)


def _triage_stopping_at(tmp_path, monkeypatch, state: str, *, cancels: bool = True):
    """Run a triage whose wait ends in *state*, with the scheduler stubbed."""
    from aorta.cia import triage as triage_mod

    cancelled: list = []

    monkeypatch.setattr(triage_mod, "launch", lambda **kw: ("99999", ""))
    monkeypatch.setattr(
        triage_mod,
        "cancel",
        lambda job_id: (cancelled.append(job_id) or (cancels, "" if cancels else "no scancel")),
    )
    monkeypatch.setattr(triage_mod, "wait_for_job", lambda *a, **k: state)

    source = tmp_path / "k.hip"
    source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")
    result = triage_mod.run_triage(
        ["--source", str(source), "--jobs-root", str(tmp_path), "--kernel-name", "bump"]
    )
    return result, cancelled


class TestAnAbandonedTriageReleasesItsNode:
    @staticmethod
    def _run(tmp_path, monkeypatch, *, cancels: bool = True) -> tuple[dict, list]:
        # The caller gave up while the job was still running.
        return _triage_stopping_at(
            tmp_path, monkeypatch, "ABANDONED(RUNNING)", cancels=cancels
        )

    def test_the_allocation_is_cancelled(self, tmp_path, monkeypatch):
        _result, cancelled = self._run(tmp_path, monkeypatch)

        assert cancelled == ["99999"]

    def test_the_result_says_what_happened(self, tmp_path, monkeypatch):
        result, _ = self._run(tmp_path, monkeypatch)

        assert result["ok"] is False
        assert result["error"] == "abandoned by caller"
        assert result["cancelled"] is True
        assert result["slurm_job_id"] == "99999"

    def test_a_failed_cancellation_is_reported_not_hidden(self, tmp_path, monkeypatch):
        """The node is still held; saying so is the only way anyone finds out."""
        result, _ = self._run(tmp_path, monkeypatch, cancels=False)

        assert result["cancelled"] is False

    def test_the_job_record_is_marked_cancelled(self, tmp_path, monkeypatch):
        import json

        result, _ = self._run(tmp_path, monkeypatch)
        record = json.loads(
            (tmp_path / result["job_id"] / "job.json").read_text(encoding="utf-8")
        )

        assert record["status"] == "cancelled"

    def test_a_completed_run_is_not_cancelled(self, tmp_path, monkeypatch):
        """Only an abandoned wait owes the cluster anything."""
        from aorta.cia import triage as triage_mod

        cancelled: list = []
        monkeypatch.setattr(triage_mod, "launch", lambda **kw: ("99999", ""))
        monkeypatch.setattr(triage_mod, "cancel", lambda job_id: cancelled.append(job_id))
        monkeypatch.setattr(triage_mod, "wait_for_job", lambda *a, **k: "COMPLETED")

        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")
        triage_mod.run_triage(
            ["--source", str(source), "--jobs-root", str(tmp_path), "--kernel-name", "bump"]
        )

        assert cancelled == []


class TestATimedOutTriageReleasesItsNodeToo:
    """The other way a wait ends, which used to keep the allocation.

    ``wait_for_job`` has two ways of stopping without an answer: the caller
    gives up, which returns ``ABANDONED(state)``, or ``job_timeout`` expires,
    which returns ``TIMEOUT_WAITING(state)``. Only the first reached scancel.
    The second fell through to the grace window and the bundle fallback with the
    job still queued or running, so it stayed in the scheduler until its own
    time limit -- four hours by default.

    That made the leak worst exactly where it costs most. A job hits the
    internal timeout because it is slow, so the runs that held a node for the
    full four hours were the ones already using it hardest.
    """

    @staticmethod
    def _run(tmp_path, monkeypatch, *, cancels: bool = True) -> tuple[dict, list]:
        return _triage_stopping_at(
            tmp_path, monkeypatch, "TIMEOUT_WAITING(RUNNING)", cancels=cancels
        )

    def test_the_allocation_is_cancelled(self, tmp_path, monkeypatch):
        _result, cancelled = self._run(tmp_path, monkeypatch)

        assert cancelled == ["99999"]

    def test_a_job_that_never_started_is_released_as_well(self, tmp_path, monkeypatch):
        """Timing out while still PENDING holds a queue slot, not a node."""
        _result, cancelled = _triage_stopping_at(
            tmp_path, monkeypatch, "TIMEOUT_WAITING(PENDING)"
        )

        assert cancelled == ["99999"]

    def test_the_result_says_it_timed_out_rather_than_that_it_was_abandoned(
        self, tmp_path, monkeypatch
    ):
        """Both end the job; which one happened is what the reader needs."""
        result, _ = self._run(tmp_path, monkeypatch)

        assert result["ok"] is False
        assert result["stage"] == "wait"
        assert "timed out" in result["error"]
        assert "abandoned" not in result["error"]

    def test_it_names_the_job_it_gave_up_on(self, tmp_path, monkeypatch):
        result, _ = self._run(tmp_path, monkeypatch)

        assert result["slurm_job_id"] == "99999"
        assert result["cancelled"] is True

    def test_a_failed_cancellation_is_reported_not_hidden(self, tmp_path, monkeypatch):
        """The node is still held; saying so is the only way anyone finds out."""
        result, _ = self._run(tmp_path, monkeypatch, cancels=False)

        assert result["cancelled"] is False

    def test_the_job_record_is_marked_cancelled(self, tmp_path, monkeypatch):
        import json

        result, _ = self._run(tmp_path, monkeypatch)
        record = json.loads(
            (tmp_path / result["job_id"] / "job.json").read_text(encoding="utf-8")
        )

        assert record["status"] == "cancelled"

    def test_watch_stops_being_handed_the_dead_job(self, tmp_path, monkeypatch):
        """A terminal status is what takes it out of the polling set.

        scan_active_jobs selects on status == 'running'. Left there, every later
        round would re-read a log that stopped growing when the job was killed.
        """
        from aorta.cia.launch.registry import scan_active_jobs

        self._run(tmp_path, monkeypatch)

        assert scan_active_jobs(tmp_path) == []
