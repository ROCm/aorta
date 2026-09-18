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
    def fake_write_bundle(job, job_dir, evidence, signal):
        # The real one creates the directory, and the retry path checks for it
        # before re-queueing -- a bundle that was never written is nothing to
        # re-run.
        bundle = job_dir / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        return bundle

    monkeypatch.setattr(
        "aorta.cia.watch.bundle_writer.write_bundle", fake_write_bundle
    )


class TestTheLoopKeepsGoing:
    def test_a_slow_autopsy_does_not_hold_up_the_other_jobs(
        self, tmp_path, alerting, monkeypatch
    ):
        """The finding itself: three jobs, the first one slow to diagnose."""
        started = threading.Event()
        release = threading.Event()
        examined: list[str] = []

        def slow_autopsy(bundle, job, jobs_root):
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
            lambda bundle, job, jobs_root: calls.append(job.job_id),
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
            lambda bundle, job, jobs_root: calls.append(job.job_id),
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


class TestAWatcherThatDiedHoldingOne:
    """Persistence must not become a way to suppress work that never happened.

    The first version of this recorded "queued" before enqueueing and treated
    anything but "abandoned" as settled. That stops duplicate work, which was
    the point -- and it also meant a watcher killed mid-Autopsy left the record
    saying "running" for ever, so no later watcher would ever pick the job up.
    The state that existed to prevent waste became a way to lose the diagnosis
    entirely.
    """

    @staticmethod
    def _write_state(job_dir: Path, state: str, age_hours: float, attempts: int = 1):
        from datetime import datetime, timedelta, timezone

        stamped = (
            datetime.now(timezone.utc) - timedelta(hours=age_hours)
        ).isoformat().replace("+00:00", "Z")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "autopsy.state.json").write_text(
            json.dumps({"state": state, "ts": stamped, "attempts": attempts}),
            encoding="utf-8",
        )

    @pytest.mark.parametrize(
        ("state", "age_hours"),
        [
            ("queued", 0.01),
            ("running", 0.5),
        ],
    )
    def test_a_fresh_one_is_left_alone(self, tmp_path, state, age_hours):
        """Finishing slowly is not dying; re-queueing a live one wastes a node."""
        self._write_state(tmp_path, state, age_hours=age_hours)

        assert poll_mod.autopsy_is_settled(tmp_path) is True

    @pytest.mark.parametrize(
        ("state", "age_hours"),
        [
            ("queued", 0.5),
            ("running", 9),
        ],
    )
    def test_a_stale_one_goes_back_in_the_queue(self, tmp_path, state, age_hours):
        self._write_state(tmp_path, state, age_hours=age_hours)

        assert poll_mod.autopsy_is_settled(tmp_path) is False

    def test_the_window_outlasts_the_production_sweep(self):
        """Four hours is the sweep's own limit; reclaiming sooner kills live work."""
        assert poll_mod.AUTOPSY_STALE_AFTER_SEC > 4 * 60 * 60

    def test_a_queued_task_has_a_shorter_lease_than_running_work(self):
        """Queued means no external work started, so five hours was indefensible."""
        assert (
            poll_mod.AUTOPSY_QUEUED_STALE_AFTER_SEC
            < poll_mod.AUTOPSY_STALE_AFTER_SEC
        )

    def test_an_unparseable_timestamp_is_treated_as_lost(self, tmp_path):
        """A record we cannot date is not evidence that something is running."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "autopsy.state.json").write_text(
            json.dumps({"state": "running", "ts": "not a date"}), encoding="utf-8"
        )

        assert poll_mod.autopsy_is_settled(tmp_path) is False


class TestFailureIsRetriedAndThenGivenUpOn:
    def test_a_failed_autopsy_is_not_settled(self, tmp_path):
        """The case the attempt counter exists for."""
        poll_mod.record_autopsy_state(tmp_path, "failed", attempts=1)

        assert poll_mod.autopsy_is_settled(tmp_path) is False

    def test_a_failing_autopsy_is_tried_again(self, tmp_path, alerting, monkeypatch):
        calls: list[str] = []

        def boom(bundle, job, jobs_root, stop=None):
            calls.append(job.job_id)
            raise RuntimeError("autopsy exploded")

        monkeypatch.setattr("aorta.cia.watch.trigger.trigger_autopsy", boom)
        _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)
        poll_jobs(tmp_path, max_rounds=1)

        assert len(calls) == 2, f"a failed autopsy was retried {len(calls) - 1} times"

    def test_it_retries_while_the_same_watch_process_stays_alive(
        self, tmp_path, alerting, monkeypatch
    ):
        """The old test restarted poll_jobs, which reset the suppressing set."""
        calls: list[str] = []
        failed_persisted = threading.Event()
        real_record = poll_mod.record_autopsy_state

        def recording(job_dir, state, **fields):
            persisted = real_record(job_dir, state, **fields)
            if state == "failed":
                # Set after the state is on disk, so the next round is testing
                # the exact ordering that failed in production.
                failed_persisted.set()
            return persisted

        def between_rounds(_seconds):
            assert failed_persisted.wait(timeout=10), "the first Autopsy never failed"

        def boom(bundle, job, jobs_root, stop=None):
            calls.append(job.job_id)
            raise RuntimeError("autopsy exploded")

        monkeypatch.setattr(poll_mod, "record_autopsy_state", recording)
        monkeypatch.setattr(poll_mod.time, "sleep", between_rounds)
        monkeypatch.setattr("aorta.cia.watch.trigger.trigger_autopsy", boom)
        _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=3)

        assert len(calls) == 2, (
            "the persisted failed state never overrode the in-memory alert claim"
        )

    def test_the_same_process_still_stops_at_the_attempt_limit(
        self, tmp_path, alerting, monkeypatch
    ):
        """Recoverable must remain bounded when every attempt fails."""
        calls: list[str] = []
        condition = threading.Condition()
        failure_count = 0
        sleeps = 0
        real_record = poll_mod.record_autopsy_state

        def recording(job_dir, state, **fields):
            nonlocal failure_count
            persisted = real_record(job_dir, state, **fields)
            if state == "failed":
                with condition:
                    failure_count += 1
                    condition.notify_all()
            return persisted

        def between_rounds(_seconds):
            nonlocal sleeps
            sleeps += 1
            target = min(sleeps, poll_mod.AUTOPSY_MAX_ATTEMPTS)
            with condition:
                assert condition.wait_for(
                    lambda: failure_count >= target, timeout=10
                ), f"Autopsy failure {target} was never persisted"

        def boom(bundle, job, jobs_root, stop=None):
            calls.append(job.job_id)
            raise RuntimeError("autopsy exploded")

        monkeypatch.setattr(poll_mod, "record_autopsy_state", recording)
        monkeypatch.setattr(poll_mod.time, "sleep", between_rounds)
        monkeypatch.setattr("aorta.cia.watch.trigger.trigger_autopsy", boom)
        job_dir = _write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=4)

        assert len(calls) == poll_mod.AUTOPSY_MAX_ATTEMPTS
        assert autopsy_state(job_dir)["state"] == "gave_up"

    def test_it_stops_after_the_attempt_limit(self, tmp_path, alerting, monkeypatch):
        """A crash loop must not spend every round on one job."""
        calls: list[str] = []

        def boom(bundle, job, jobs_root, stop=None):
            calls.append(job.job_id)
            raise RuntimeError("autopsy exploded")

        monkeypatch.setattr("aorta.cia.watch.trigger.trigger_autopsy", boom)
        job_dir = _write_job(tmp_path, "cia-aaa")

        for _ in range(5):
            poll_jobs(tmp_path, max_rounds=1)

        assert len(calls) == poll_mod.AUTOPSY_MAX_ATTEMPTS, calls
        assert autopsy_state(job_dir)["state"] == "gave_up"

    def test_giving_up_is_terminal(self, tmp_path):
        poll_mod.record_autopsy_state(tmp_path, "gave_up", attempts=2)

        assert poll_mod.autopsy_is_settled(tmp_path) is True

    def test_a_successful_one_is_not_repeated(self, tmp_path):
        poll_mod.record_autopsy_state(tmp_path, "done", attempts=1)

        assert poll_mod.autopsy_is_settled(tmp_path) is True
