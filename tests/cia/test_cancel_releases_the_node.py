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


class TestAnAbandonedTriageReleasesItsNode:
    @staticmethod
    def _run(tmp_path, monkeypatch, *, cancels: bool = True) -> tuple[dict, list]:
        from aorta.cia import triage as triage_mod

        cancelled: list = []

        monkeypatch.setattr(triage_mod, "launch", lambda **kw: ("99999", ""))
        monkeypatch.setattr(
            triage_mod,
            "cancel",
            lambda job_id: (cancelled.append(job_id) or (cancels, "" if cancels else "no scancel")),
        )
        # The caller gave up while the job was still running.
        monkeypatch.setattr(triage_mod, "wait_for_job", lambda *a, **k: "ABANDONED(RUNNING)")

        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")
        result = triage_mod.run_triage(
            ["--source", str(source), "--jobs-root", str(tmp_path), "--kernel-name", "bump"]
        )
        return result, cancelled

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
