"""Cancellation closes Watch admission instead of only waking its sleeps."""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the watch and Autopsy paths need [cia]")

from aorta.cia.autopsy.orchestrator import run_autopsy
from aorta.cia.launch.job import JobRecord
from aorta.cia.watch import poll as poll_mod


def _job(root: Path, job_id: str = "cia-cancel") -> tuple[JobRecord, Path]:
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    log = job_dir / "watch.log"
    log.write_text("step=1 loss=nan\n", encoding="utf-8")
    job = JobRecord(
        job_id=job_id,
        node="node1",
        recipe="a recipe",
        recipe_path="/tmp/recipe.yaml",
        launched_at="2026-01-01T00:00:00Z",
        log_path=str(log),
        aorta_output=str(job_dir / "aorta"),
        status="running",
        watch_files=[str(log)],
    )
    (job_dir / "job.json").write_text(
        json.dumps(job.to_dict()),
        encoding="utf-8",
    )
    return job, job_dir


def _alerting_watch(monkeypatch, *, on_assessment=None):
    class Prediction:
        signal = "WATCH_NUMERIC_NAN"
        healthy = False
        confidence = 0.99
        evidence = "loss=nan"
        assessment = "diverged"

    class Watcher:
        def forward(self, **_kwargs):
            if on_assessment:
                on_assessment()
            return Prediction()

    monkeypatch.setattr(poll_mod, "LogWatcher", lambda: Watcher())
    monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: object())

    def write_bundle(job, job_dir, evidence, signal):
        bundle = job_dir / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        return bundle

    monkeypatch.setattr(
        "aorta.cia.watch.bundle_writer.write_bundle",
        write_bundle,
    )


class TestAdmissionClosesPromptly:
    def test_a_stop_during_assessment_starts_no_autopsy(self, tmp_path, monkeypatch):
        stop = threading.Event()
        _job(tmp_path)
        _alerting_watch(monkeypatch, on_assessment=stop.set)
        calls = []
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda *args, **kwargs: calls.append(args),
        )

        poll_mod.poll_jobs(tmp_path, max_rounds=1, stop=stop)

        assert calls == []
        assert poll_mod.autopsy_state(tmp_path / "cia-cancel") == {}

    def test_shutdown_is_bounded_and_unfinished_work_is_recoverable(self, tmp_path, monkeypatch):
        stop = threading.Event()
        release = threading.Event()
        started = threading.Event()
        _job(tmp_path)
        _alerting_watch(monkeypatch)
        monkeypatch.setattr(poll_mod, "AUTOPSY_STOP_GRACE_SEC", 0.05)

        def blocked(*_args, **_kwargs):
            started.set()
            stop.set()
            release.wait(timeout=5)

        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            blocked,
        )

        before = time.monotonic()
        poll_mod.poll_jobs(tmp_path, max_rounds=1, stop=stop)
        elapsed = time.monotonic() - before
        state = poll_mod.autopsy_state(tmp_path / "cia-cancel")
        release.set()

        assert started.is_set()
        assert elapsed < 1
        assert state["state"] == "deferred"
        assert state["attempts"] == 1
        assert state["retry_after"]
        assert poll_mod.autopsy_is_settled(tmp_path / "cia-cancel")

    def test_shutdown_workers_do_not_hold_interpreter_exit(self):
        release = threading.Event()
        started = threading.Event()
        pool = poll_mod._DaemonExecutor(1, "cia-test")

        def blocked():
            started.set()
            release.wait(timeout=5)

        future = pool.submit(blocked)
        assert started.wait(timeout=1)
        before = time.monotonic()
        pool.shutdown(wait=False)
        elapsed = time.monotonic() - before
        try:
            assert elapsed < 1
            assert all(thread.daemon for thread in pool._threads)
            assert not future.done()
        finally:
            release.set()


class TestQueuedWorkKeepsItsRecoveryIdentity:
    def test_abandoning_a_claimed_attempt_keeps_its_number(self, tmp_path):
        _job(tmp_path)
        job_dir = tmp_path / "cia-cancel"
        poll_mod.record_autopsy_state(
            job_dir,
            "queued",
            job_id="cia-cancel",
            attempts=1,
            signal="WATCH_NUMERIC_NAN",
        )

        poll_mod.abandon_autopsy(job_dir, reason="cancelled")

        state = poll_mod.autopsy_state(job_dir)
        assert state["state"] == "abandoned"
        assert state["attempts"] == 1
        assert state["signal"] == "WATCH_NUMERIC_NAN"

    def test_shutdown_deferral_never_overwrites_done(self, tmp_path):
        _job(tmp_path)
        job_dir = tmp_path / "cia-cancel"
        poll_mod.record_autopsy_state(
            job_dir,
            "done",
            job_id="cia-cancel",
            attempts=1,
        )

        poll_mod.defer_stopping_autopsy(
            job_dir,
            reason="shutdown race",
            attempt=1,
        )

        assert poll_mod.autopsy_state(job_dir)["state"] == "done"

    def test_a_worker_stopped_before_start_runs_nothing(self, tmp_path, monkeypatch):
        stop = threading.Event()
        stop.set()
        job, job_dir = _job(tmp_path)
        bundle = job_dir / "bundle"
        bundle.mkdir()
        poll_mod.record_autopsy_state(
            job_dir,
            "queued",
            job_id=job.job_id,
            attempts=1,
        )
        calls = []
        monkeypatch.setattr(
            "aorta.cia.watch.trigger.trigger_autopsy",
            lambda *args, **kwargs: calls.append(args),
        )

        poll_mod._run_autopsy_off_the_loop(bundle, job, tmp_path, job_dir, stop, 1)

        assert calls == []
        assert poll_mod.autopsy_state(job_dir)["state"] == "abandoned"
        assert poll_mod.autopsy_state(job_dir)["attempts"] == 1


class TestNoSweepBeginsAfterCancellation:
    def test_a_stop_during_prelaunch_work_submits_nothing(self, tmp_path, monkeypatch):
        from aorta.cia import triage as triage_mod

        stop = threading.Event()
        launches = []

        def reconcile_then_stop(_jobs_root):
            stop.set()
            return 0

        monkeypatch.setattr(triage_mod, "reconcile_stale_jobs", reconcile_then_stop)
        monkeypatch.setattr(
            triage_mod,
            "launch",
            lambda **kwargs: launches.append(kwargs),
        )

        result = triage_mod.run_triage(
            ["--command", "true", "--jobs-root", str(tmp_path)],
            stop=stop,
        )

        assert launches == []
        assert result["stage"] == "launch"
        assert result["error"] == "abandoned by caller before launch"

    def test_a_stop_racing_with_submission_cancels_before_watch(self, tmp_path, monkeypatch):
        from aorta.cia import triage as triage_mod

        stop = threading.Event()
        cancelled = []

        def launch_then_stop(**_kwargs):
            stop.set()
            return "12345", ""

        def watch_must_not_start(**_kwargs):
            raise AssertionError("Watch started after launch was cancelled")

        monkeypatch.setattr(triage_mod, "launch", launch_then_stop)
        monkeypatch.setattr(
            triage_mod,
            "cancel",
            lambda job_id: (cancelled.append(job_id) or (True, "")),
        )
        monkeypatch.setattr(triage_mod.threading, "Thread", watch_must_not_start)

        result = triage_mod.run_triage(
            ["--command", "true", "--jobs-root", str(tmp_path)],
            stop=stop,
        )

        record = json.loads((tmp_path / result["job_id"] / "job.json").read_text(encoding="utf-8"))
        assert cancelled == ["12345"]
        assert result["stage"] == "launch"
        assert result["slurm_job_id"] == "12345"
        assert result["cancelled"] is True
        assert record["status"] == "cancelled"

    def test_an_already_stopped_probe_never_opens_ssh(self, tmp_path, monkeypatch):
        from aorta.cia.autopsy import probe as probe_mod

        stop = threading.Event()
        stop.set()
        job, _job_dir = _job(tmp_path)
        calls = []
        monkeypatch.setattr(
            probe_mod,
            "_ssh",
            lambda *args, **kwargs: calls.append(args),
        )

        assert probe_mod.run_aorta_probe(tmp_path, job, head_node="head", stop=stop) is None
        assert calls == []

    def test_a_stop_during_router_review_blocks_escalation(self, tmp_path, monkeypatch):
        from aorta.cia.autopsy import probe as probe_mod
        from aorta.cia.autopsy import router as router_mod

        bundle = tmp_path / "bundle"
        (bundle / "logs").mkdir(parents=True)
        (bundle / "logs" / "watch.stderr.log").write_text(
            "step=1 loss=nan\n",
            encoding="utf-8",
        )
        (bundle / "manifest.yaml").write_text(
            "job_id: cia-cancel\npaths:\n" "  stderr: logs/watch.stderr.log\n",
            encoding="utf-8",
        )
        stop = threading.Event()
        job, _job_dir = _job(tmp_path / "jobs")
        probes = []

        class Prediction:
            category = "numeric_silent"
            confidence = 0.5
            rationale = "needs a sweep"
            next_probe = "aorta sweep run"
            next_probe_reason = "low confidence"

        class Router:
            def __init__(self, _root):
                pass

            def __call__(self, **_kwargs):
                stop.set()
                return Prediction()

        monkeypatch.setattr(router_mod, "TriageRouter", Router)
        monkeypatch.setattr(
            probe_mod,
            "run_aorta_probe",
            lambda *args, **kwargs: probes.append(args),
        )

        result = run_autopsy(bundle, job=job, stop=stop)

        assert result["category"] == "numeric_silent"
        assert probes == []

    def test_post_probe_autopsy_keeps_the_stop_token(self, tmp_path, monkeypatch):
        from aorta.cia.autopsy import probe as probe_mod
        from aorta.cia.autopsy import router as router_mod

        bundle = tmp_path / "bundle"
        (bundle / "logs").mkdir(parents=True)
        (bundle / "logs" / "watch.stderr.log").write_text(
            "step=1 loss=nan\n",
            encoding="utf-8",
        )
        (bundle / "manifest.yaml").write_text(
            "job_id: cia-cancel\npaths:\n" "  stderr: logs/watch.stderr.log\n",
            encoding="utf-8",
        )
        stop = threading.Event()
        job, _job_dir = _job(tmp_path / "jobs")
        router_calls = []

        class Prediction:
            category = "numeric_silent"
            confidence = 0.5
            rationale = "needs a sweep"
            next_probe = "aorta sweep run"
            next_probe_reason = "low confidence"

        class Router:
            def __init__(self, _root):
                pass

            def __call__(self, **_kwargs):
                router_calls.append(True)
                return Prediction()

        def successful_probe(*_args, **_kwargs):
            stop.set()
            return bundle / "aorta" / "matrix.json"

        monkeypatch.setattr(router_mod, "TriageRouter", Router)
        monkeypatch.setattr(
            probe_mod,
            "run_aorta_probe",
            successful_probe,
        )

        run_autopsy(bundle, job=job, stop=stop)

        assert len(router_calls) == 1


class TestCliInterruptsUseTheSameStopSignal:
    def test_first_sigint_requests_cooperative_shutdown(self, monkeypatch, capsys):
        from aorta.cia import triage as triage_mod

        installed = {}
        restored = object()
        monkeypatch.setattr(
            triage_mod.signal,
            "getsignal",
            lambda _sig: restored,
        )
        monkeypatch.setattr(
            triage_mod.signal,
            "signal",
            lambda sig, handler: installed.__setitem__(sig, handler),
        )

        def run(*, stop):
            installed[signal.SIGINT](signal.SIGINT, None)
            assert stop.is_set()
            return {"ok": False, "error": "abandoned by caller"}

        monkeypatch.setattr(triage_mod, "run_triage", run)

        assert triage_mod.main() == 1
        assert installed[signal.SIGINT] is restored
        assert "abandoned by caller" in capsys.readouterr().out
