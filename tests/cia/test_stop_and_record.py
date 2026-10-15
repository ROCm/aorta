"""Cancellation races have one state-persistence implementation."""

from __future__ import annotations

import logging
from pathlib import Path

from aorta.cia.launch.job import JobRecord, read_job_json, write_job_json
from aorta.cia.triage import _stop_and_record


class _Backend:
    name = "test"

    def __init__(
        self,
        *,
        states: list[str],
        cancel_result: tuple[bool, str],
        noop_reason: str = "",
    ):
        self.states = iter(states)
        self.cancel_result = cancel_result
        self.noop_reason = noop_reason
        self.recorded: list[tuple[str, Path, bool]] = []

    def state(self, job_id: str, job_dir: Path) -> str:
        return next(self.states)

    def cancel(self, job_id: str) -> tuple[bool, str]:
        return self.cancel_result

    def cancel_was_noop(self, message: str) -> bool:
        return bool(self.noop_reason) and message == self.noop_reason

    def record_cancellation(
        self,
        job_id: str,
        job_dir: Path,
        *,
        preserve_terminal: bool = True,
    ) -> None:
        self.recorded.append((job_id, job_dir, preserve_terminal))


def _job(jobs_root: Path) -> Path:
    record = JobRecord(
        job_id="cia-test",
        node="",
        recipe="test",
        launched_at="2026-01-01T00:00:00Z",
        log_path=str(jobs_root / "cia-test" / "watch.log"),
        aorta_output=str(jobs_root / "cia-test" / "bundle" / "aorta"),
        scheduler="test",
        scheduler_job_id="native-1",
        status="running",
    )
    write_job_json(record, jobs_root)
    return jobs_root / record.job_id


def _status(jobs_root: Path) -> str:
    return read_job_json(jobs_root / "cia-test" / "job.json").status


def test_a_real_cancellation_records_cancelled(tmp_path):
    job_dir = _job(tmp_path)
    backend = _Backend(states=["RUNNING"], cancel_result=(True, ""))

    result = _stop_and_record(
        tmp_path,
        "cia-test",
        job_dir,
        "native-1",
        backend,
    )

    assert result == (True, "")
    assert _status(tmp_path) == "cancelled"
    assert backend.recorded == [("native-1", job_dir, False)]


def test_a_terminal_race_preserves_completion(tmp_path):
    job_dir = _job(tmp_path)
    backend = _Backend(states=["COMPLETED"], cancel_result=(True, ""))

    _stop_and_record(tmp_path, "cia-test", job_dir, "native-1", backend)

    assert _status(tmp_path) == "completed"
    assert backend.recorded == []


def test_an_already_finished_result_rereads_the_terminal_state(tmp_path):
    job_dir = _job(tmp_path)
    backend = _Backend(
        states=["RUNNING", "FAILED"],
        cancel_result=(True, "already finished"),
        noop_reason="already finished",
    )

    _stop_and_record(tmp_path, "cia-test", job_dir, "native-1", backend)

    assert _status(tmp_path) == "failed"
    assert backend.recorded == []


def test_a_failed_cancellation_stays_running_and_warns(tmp_path, caplog):
    job_dir = _job(tmp_path)
    backend = _Backend(
        states=["RUNNING"],
        cancel_result=(False, "permission denied"),
    )

    with caplog.at_level(logging.WARNING, logger="aorta.cia.triage"):
        result = _stop_and_record(
            tmp_path,
            "cia-test",
            job_dir,
            "native-1",
            backend,
        )

    assert result == (False, "permission denied")
    assert _status(tmp_path) == "running"
    assert "could not be cancelled (permission denied)" in caplog.text
