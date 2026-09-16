"""One Autopsy must not stop every other job being watched.

``trigger_autopsy`` was called inline in the per-job loop. Autopsy is an LLM
ReAct loop that can escalate to a production sweep with a four-hour limit, so
one alert stopped every other active job being monitored for as long as it
took -- and the jobs that most need watching are the ones running beside a
failure.

The alert state was also a set in the function, so a watcher that restarted
re-diagnosed everything it had already alerted on and spent its rounds
re-running escalations it had already paid for.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the watch loop needs the [cia] extra")

import aorta.cia.watch.poll as poll_mod
from aorta.cia.watch.poll import autopsy_state, poll_jobs, record_autopsy_state


def _write_job(root: Path, job_id: str, *, text: str = "step=1 loss=nan\n") -> Path:
    """A job on disk, laid out the way scan_active_jobs expects."""
    job_dir = root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log = job_dir / "watch.log"
    log.write_text(text, encoding="utf-8")
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
    """Every job alerts, and Autopsy is a stand-in we control the timing of."""

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
    monkeypatch.setattr(poll_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "aorta.cia.watch.bundle_writer.write_bundle",
        lambda job, job_dir, evidence, signal: job_dir / "bundle",
    )


class TestTheLoopKeepsGoing:
    def test_a_slow_autopsy_does_not_hold_up_the_other_jobs(
        self, tmp_path, alerting, monkeypatch
    ):
        """The finding itself: three jobs, the first one slow to diagnose."""
        started = threading.Event()
        release = threading.Event()
        examined: list[str] = []

        def slow_autopsy(bundle, job, jobs_root, stop=None):
            started.set()
            release.wait(timeout=10)

        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy", slow_autopsy
        )

        for job_id in ("cia-aaa", "cia-bbb", "cia-ccc"):
            _write_job(tmp_path, job_id)

        def watch():
            poll_jobs(tmp_path, max_rounds=1)

        runner = threading.Thread(target=watch, daemon=True)
        runner.start()
        # The first Autopsy is now blocked. If the loop were serial, the other
        # two jobs would never be reached while it is.
        assert started.wait(timeout=10), "no autopsy started"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            examined = [
                d.name for d in tmp_path.iterdir() if autopsy_state(d).get("state")
            ]
            if len(examined) == 3:
                break
            time.sleep(0.05)
        release.set()
        runner.join(timeout=15)

        assert sorted(examined) == ["cia-aaa", "cia-bbb", "cia-ccc"], (
            "a blocked autopsy stopped the other jobs being examined"
        )

    def test_every_alert_is_queued_not_run_inline(self, tmp_path, alerting, monkeypatch):
        """Which thread it ran on is the whole point."""
        threads: list[str] = []
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda *a, **k: threads.append(threading.current_thread().name),
        )
        _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)

        assert threads, "autopsy never ran"
        assert all(name.startswith("cia-autopsy") for name in threads), threads


class TestTheStateSurvivesTheProcess:
    def test_an_alert_is_written_beside_the_job(self, tmp_path, alerting, monkeypatch):
        monkeypatch.setattr("aorta.cia.watch.trigger.trigger_autopsy", lambda *a, **k: None)
        job_dir = _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)

        assert autopsy_state(job_dir).get("state") in {"queued", "running", "done"}

    def test_a_finished_autopsy_says_done(self, tmp_path, alerting, monkeypatch):
        monkeypatch.setattr("aorta.cia.watch.trigger.trigger_autopsy", lambda *a, **k: None)
        job_dir = _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)

        assert autopsy_state(job_dir)["state"] == "done"

    def test_a_failing_autopsy_says_failed_rather_than_vanishing(
        self, tmp_path, alerting, monkeypatch
    ):
        """A worker that raises is otherwise silent: the pool swallows it."""
        def boom(*_a, **_k):
            raise RuntimeError("autopsy exploded")

        monkeypatch.setattr("aorta.cia.watch.trigger.trigger_autopsy", boom)
        job_dir = _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)

        assert autopsy_state(job_dir)["state"] == "failed"

    def test_a_restart_does_not_diagnose_it_again(self, tmp_path, alerting, monkeypatch):
        """The set was in the function, so a restart paid for it twice."""
        calls: list[str] = []
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda bundle, job, jobs_root, stop=None: calls.append(job.job_id),
        )
        _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)
        poll_jobs(tmp_path, max_rounds=1)  # a fresh process would start here

        assert calls == ["cia-aaa"], f"diagnosed {len(calls)} times"

    def test_an_abandoned_one_is_eligible_again(self, tmp_path, alerting, monkeypatch):
        """Cancelled before it started is not the same as finished."""
        calls: list[str] = []
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda bundle, job, jobs_root, stop=None: calls.append(job.job_id),
        )
        job_dir = _write_job(tmp_path, "cia-aaa")
        record_autopsy_state(job_dir, "abandoned", job_id="cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)

        assert calls == ["cia-aaa"], "an abandoned autopsy was never retried"


class TestThePoolIsBounded:
    def test_it_has_a_ceiling(self):
        assert 1 <= poll_mod.AUTOPSY_WORKERS <= 4

    def test_the_workers_are_named_for_it(self, tmp_path, alerting, monkeypatch):
        """So a stuck autopsy is identifiable in a thread dump."""
        seen: list[str] = []
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda *a, **k: seen.append(threading.current_thread().name),
        )
        _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)

        assert seen and "cia-autopsy" in seen[0]
