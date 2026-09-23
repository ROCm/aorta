"""A triage the chat gave up on has to stop, not just stop being listened to.

``_run_triage`` runs the pipeline on a worker thread and stops waiting after
``triage_timeout``. Stopping waiting was all that happened: the thread stayed in
its sleep loops, kept polling sacct and kept calling the model, and a fresh
one-worker pool was built for every call.

Two consequences, both about the server rather than the answer. Pool threads are
not daemons and ``concurrent.futures`` joins them at interpreter exit, so a chat
server that had timed out a triage could not shut down until that triage ran
itself out -- roughly eighteen minutes of bounded sleeps, and every timed-out
call added another thread to wait for. Nothing capped how many there could be.

The waits now take a stop flag and check it, and there is one pool with a small
fixed number of workers.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

from aorta.cia.cancellation import pause, stopped


@pytest.fixture()
def cluster(tmp_path, monkeypatch):
    pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
    import aorta.chat.tools.cluster as module

    monkeypatch.setattr(module.settings, "jobs_path", str(tmp_path), raising=False)
    monkeypatch.setattr(module.settings, "cia_demo_node", "", raising=False)
    return module


class TestTheAbandonedWorkIsAskedToStop:
    def test_the_flag_is_set_when_the_tool_times_out(self, cluster, monkeypatch):
        """The bug: nothing told the run that nobody was waiting any more."""
        monkeypatch.setattr(cluster.settings, "triage_timeout", 0.2, raising=False)
        seen: dict[str, threading.Event] = {}
        released = threading.Event()

        def slow_triage(argv, *, stop=None):
            seen["stop"] = stop
            # Stands in for the sleep loops: wakes as soon as the flag is set.
            pause(stop, 30)
            released.set()
            return {"ok": True}

        monkeypatch.setattr(cluster, "run_triage", slow_triage)
        answer = cluster._run_triage(["--source", "k.hip"], "label")

        assert "exceeded" in answer
        assert released.wait(timeout=10), "the run never noticed it had been abandoned"
        assert stopped(seen["stop"])

    def test_and_it_stops_promptly_rather_than_running_its_course(self, cluster, monkeypatch):
        """The whole point: the thread is free long before its own deadline."""
        monkeypatch.setattr(cluster.settings, "triage_timeout", 0.2, raising=False)
        stopped_at: list[float] = []

        def slow_triage(argv, *, stop=None):
            started = time.monotonic()
            pause(stop, 30)
            stopped_at.append(time.monotonic() - started)
            return {"ok": True}

        monkeypatch.setattr(cluster, "run_triage", slow_triage)
        cluster._run_triage(["--source", "k.hip"], "label")

        deadline = time.monotonic() + 10
        while not stopped_at and time.monotonic() < deadline:
            time.sleep(0.05)

        assert stopped_at, "the run never returned"
        assert stopped_at[0] < 5, f"took {stopped_at[0]:.1f}s to notice"


class TestThreadsAreBounded:
    def test_one_pool_is_reused_across_calls(self, cluster, monkeypatch):
        """A pool per call is a thread per call, for the life of the server."""
        monkeypatch.setattr(cluster.settings, "triage_timeout", 5, raising=False)
        monkeypatch.setattr(
            cluster,
            "run_triage",
            lambda argv, *, stop=None: {"ok": True, "job_id": "j1", "job_dir": "/tmp/j1"},
        )

        pools = set()
        for _ in range(4):
            cluster._run_triage(["--source", "k.hip"], "label")
            pools.add(id(cluster._triage_pool()))

        assert len(pools) == 1

    def test_timed_out_runs_do_not_accumulate_threads(self, cluster, monkeypatch):
        """The leak, counted: six abandoned triages used to mean six threads."""
        monkeypatch.setattr(cluster.settings, "triage_timeout", 0.2, raising=False)

        def slow_triage(argv, *, stop=None):
            pause(stop, 30)
            return {"ok": True}

        monkeypatch.setattr(cluster, "run_triage", slow_triage)
        before = threading.active_count()
        for _ in range(6):
            cluster._run_triage(["--source", "k.hip"], "label")

        grown = threading.active_count() - before
        assert grown <= cluster._TRIAGE_WORKERS, f"{grown} threads for 6 timed-out calls"

    def test_a_saturated_pool_still_answers(self, cluster, monkeypatch):
        """A bounded pool must not turn a leak into a hang.

        With every worker busy a submission waits in the queue, so the timeout
        has to cover the wait as well as the run.
        """
        monkeypatch.setattr(cluster.settings, "triage_timeout", 0.3, raising=False)

        def slow_triage(argv, *, stop=None):
            pause(stop, 30)
            return {"ok": True}

        monkeypatch.setattr(cluster, "run_triage", slow_triage)
        for _ in range(cluster._TRIAGE_WORKERS):
            cluster._run_triage(["--source", "k.hip"], "label")

        started = time.monotonic()
        answer = cluster._run_triage(["--source", "k.hip"], "label")

        assert "exceeded" in answer
        assert time.monotonic() - started < 5


class TestTheProcessCanExit:
    """The reported symptom, measured on a real interpreter.

    Nothing in-process can assert this: the join happens during interpreter
    shutdown, after the last test would have finished. So a child process runs
    the same shape and is timed from outside.
    """

    SCRIPT = textwrap.dedent(
        """
        import sys, time, threading
        sys.path.insert(0, {src!r})
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FT
        from aorta.cia.cancellation import pause

        def wedged(stop):
            pause(stop, 12)          # a wait loop with nobody left to answer

        pool = ThreadPoolExecutor(max_workers=2)
        stop = threading.Event()
        f = pool.submit(wedged, stop)
        try:
            f.result(timeout=0.5)
        except FT:
            {cancel}
        print("returned", flush=True)
        """
    )

    def _exit_seconds(self, tmp_path, cancel: str) -> float:
        script = tmp_path / "exit_shape.py"
        script.write_text(
            self.SCRIPT.format(src="/apps/avsharma/aorta/src", cancel=cancel),
            encoding="utf-8",
        )
        started = time.monotonic()
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=60
        )
        assert "returned" in done.stdout
        return time.monotonic() - started

    def test_without_the_flag_exit_waits_for_the_abandoned_run(self, tmp_path):
        """Establishes that the mechanism is real before claiming it is fixed."""
        assert self._exit_seconds(tmp_path, cancel="pass") > 8

    def test_setting_the_flag_lets_the_interpreter_go(self, tmp_path):
        assert self._exit_seconds(tmp_path, cancel="stop.set()") < 5


class TestThePauseHelper:
    def test_it_reports_that_it_was_stopped(self):
        event = threading.Event()
        event.set()
        assert pause(event, 30) is True

    def test_it_reports_a_plain_elapsed_sleep(self):
        assert pause(threading.Event(), 0.01) is False

    def test_a_caller_that_never_cancels_still_sleeps(self):
        started = time.monotonic()
        assert pause(None, 0.05) is False
        assert time.monotonic() - started >= 0.04

    def test_it_wakes_as_soon_as_the_flag_is_set(self):
        """A sleep that only checks on expiry would wait out the interval."""
        event = threading.Event()
        threading.Timer(0.1, event.set).start()

        started = time.monotonic()
        assert pause(event, 30) is True
        assert time.monotonic() - started < 5

    def test_stopped_is_false_for_a_caller_with_no_flag(self):
        assert stopped(None) is False
