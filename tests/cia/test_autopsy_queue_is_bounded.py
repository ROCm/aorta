"""Autopsy admission is bounded and excess work survives a restart.

ThreadPoolExecutor limits running workers, not submitted tasks. One alerting
round could therefore enqueue every active job behind two workers, each of
which may spend four hours in a production sweep. Every queued job was marked
``queued`` and given the same five-hour lease as running work, so killing Watch
lost the in-memory queue while the state file suppressed recovery for hours.

Admission now equals worker count. Excess jobs become durable ``deferred``
records without spending an attempt; a later round or a new Watch process
submits them from their persisted bundle. A task admitted as ``queued`` gets a
short lease because it has not started external work, while ``running`` keeps
the lease that outlasts a production sweep.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the watch loop needs the [cia] extra")

import aorta.cia.watch.poll as poll_mod
from aorta.cia.watch.poll import autopsy_state, poll_jobs, record_autopsy_state


def _write_job(root: Path, job_id: str) -> Path:
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    log = job_dir / "watch.log"
    log.write_text("step=1 loss=nan\n", encoding="utf-8")
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "node": "node1",
                "recipe": "a recipe",
                "launched_at": "2026-01-01T00:00:00Z",
                "log_path": str(log),
                "aorta_output": str(job_dir / "aorta"),
                "status": "running",
                "schema_version": "0.1",
                "watch_files": [str(log)],
            }
        ),
        encoding="utf-8",
    )
    return job_dir


@pytest.fixture()
def alerting(monkeypatch):
    class Pred:
        signal = "WATCH_NUMERIC_NAN"
        healthy = False
        confidence = 0.99
        evidence = "loss=nan"
        assessment = "diverged"

    class FakeWatcher:
        def forward(self, **_kwargs):
            return Pred()

    monkeypatch.setattr(poll_mod, "LogWatcher", lambda *a, **k: FakeWatcher())
    monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: object())
    # poll_jobs sleeps once after its last bounded round. Avoid a minute-long
    # test; Event.wait below remains a real clock.
    monkeypatch.setattr(poll_mod.time, "sleep", lambda _seconds: None)

    def write_bundle(job, job_dir, evidence, signal):
        bundle = job_dir / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        return bundle

    monkeypatch.setattr(
        "aorta.cia.watch.bundle_writer.write_bundle", write_bundle
    )


class TestAdmissionIsActuallyBounded:
    def test_capacity_is_the_worker_count(self):
        """No accepted task waits in ThreadPoolExecutor's hidden queue."""
        assert poll_mod.AUTOPSY_CAPACITY == poll_mod.AUTOPSY_WORKERS

    def test_one_alerting_round_defers_every_job_beyond_capacity(
        self, tmp_path, alerting, monkeypatch
    ):
        started: list[str] = []
        lock = threading.Lock()
        enough_started = threading.Event()
        release = threading.Event()

        def blocked(bundle, job, jobs_root, stop=None):
            with lock:
                started.append(job.job_id)
                if len(started) == poll_mod.AUTOPSY_CAPACITY:
                    enough_started.set()
            release.wait(timeout=20)

        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy", blocked
        )
        job_ids = [f"cia-{i:03d}" for i in range(7)]
        for job_id in job_ids:
            _write_job(tmp_path, job_id)

        runner = threading.Thread(
            target=poll_jobs,
            args=(tmp_path,),
            kwargs={"max_rounds": 1},
            daemon=True,
        )
        runner.start()
        assert enough_started.wait(timeout=10), "the workers never filled"

        # Wait until the Watch thread has considered every job and persisted
        # either admission or deferral. Event.wait is untouched by the fixture.
        deadline = datetime.now(timezone.utc) + timedelta(seconds=10)
        states: dict[str, dict] = {}
        while datetime.now(timezone.utc) < deadline:
            states = {job_id: autopsy_state(tmp_path / job_id) for job_id in job_ids}
            if all(state.get("state") for state in states.values()):
                break
            threading.Event().wait(0.02)

        try:
            active = [
                job_id
                for job_id, state in states.items()
                if state.get("state") in {"queued", "running"}
            ]
            deferred = [
                job_id
                for job_id, state in states.items()
                if state.get("state") == "deferred"
            ]

            assert len(active) == poll_mod.AUTOPSY_CAPACITY
            assert len(deferred) == len(job_ids) - poll_mod.AUTOPSY_CAPACITY
            assert len(started) == poll_mod.AUTOPSY_CAPACITY
        finally:
            release.set()
            runner.join(timeout=20)

        assert not runner.is_alive()

    def test_deferral_does_not_spend_an_attempt(
        self, tmp_path, alerting, monkeypatch
    ):
        release = threading.Event()
        started = threading.Event()

        def blocked(*_args, **_kwargs):
            started.set()
            release.wait(timeout=20)

        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy", blocked
        )
        for i in range(poll_mod.AUTOPSY_CAPACITY + 1):
            _write_job(tmp_path, f"cia-{i:03d}")

        runner = threading.Thread(
            target=poll_jobs,
            args=(tmp_path,),
            kwargs={"max_rounds": 1},
            daemon=True,
        )
        runner.start()
        assert started.wait(timeout=10)

        deferred_dir = tmp_path / f"cia-{poll_mod.AUTOPSY_CAPACITY:03d}"
        deadline = datetime.now(timezone.utc) + timedelta(seconds=10)
        state = {}
        while datetime.now(timezone.utc) < deadline:
            state = autopsy_state(deferred_dir)
            if state.get("state") == "deferred":
                break
            threading.Event().wait(0.02)

        try:
            assert state["state"] == "deferred"
            assert state["attempts"] == 0
            assert state["next_attempt"] == 1
            assert "capacity" in state["reason"]
        finally:
            release.set()
            runner.join(timeout=20)


class TestThePersistedQueueRecovers:
    def test_a_deferred_job_runs_without_new_log_bytes(
        self, tmp_path, alerting, monkeypatch
    ):
        calls: list[str] = []
        job_dir = _write_job(tmp_path, "cia-aaa")
        (job_dir / "bundle").mkdir()
        record_autopsy_state(
            job_dir,
            "deferred",
            job_id="cia-aaa",
            attempts=0,
            next_attempt=1,
            reason="watch capacity is full",
        )
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda bundle, job, jobs_root, stop=None: calls.append(job.job_id),
        )

        # A new poll_jobs invocation models a restarted Watch. The log cursor is
        # irrelevant: persisted deferred state takes the retry path first.
        poll_jobs(tmp_path, max_rounds=1)

        assert calls == ["cia-aaa"]
        assert autopsy_state(job_dir)["state"] == "done"

    def test_an_orphaned_queued_record_uses_the_short_lease(self, tmp_path):
        queued = {
            "state": "queued",
            "ts": (
                datetime.now(timezone.utc)
                - timedelta(seconds=poll_mod.AUTOPSY_QUEUED_STALE_AFTER_SEC + 1)
            ).isoformat().replace("+00:00", "Z"),
            "attempts": 1,
        }
        job_dir = tmp_path / "cia-aaa"
        job_dir.mkdir()
        (job_dir / "autopsy.state.json").write_text(
            json.dumps(queued), encoding="utf-8"
        )

        assert poll_mod.autopsy_is_settled(job_dir) is False

    def test_running_work_keeps_the_long_lease(self, tmp_path):
        running = {
            "state": "running",
            "ts": (
                datetime.now(timezone.utc)
                - timedelta(seconds=poll_mod.AUTOPSY_QUEUED_STALE_AFTER_SEC + 1)
            ).isoformat().replace("+00:00", "Z"),
            "attempts": 1,
        }
        job_dir = tmp_path / "cia-aaa"
        job_dir.mkdir()
        (job_dir / "autopsy.state.json").write_text(
            json.dumps(running), encoding="utf-8"
        )

        assert poll_mod.autopsy_is_settled(job_dir) is True
