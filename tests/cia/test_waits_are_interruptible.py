"""The waits themselves, checked where they actually happen.

A triage is mostly sleeping: fifteen minutes of five-second polls waiting for
Slurm, three more for Watch to assemble a bundle, and on the escalation path
four hours of thirty-second polls waiting for a sweep to drop its matrix on a
compute node. Whoever asked for the triage has usually stopped waiting long
before any of that runs out.

Each of those loops now takes a stop flag. What matters is not that the flag
exists but that a loop wakes for it rather than at the end of the interval it
happened to be in -- a check placed only at the top of the loop body still
leaves thirty seconds of dead sleep on the probe, and four hours of them adds
up to the process hanging around.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from aorta.cia import triage as triage_mod
from aorta.cia.cancellation import pause
from aorta.cia.launch.job import JobRecord


def _backend(state, *, poll_interval: float = 5.0, subject: str = "slurm"):
    state_fn = state if callable(state) else lambda *_args: state
    return SimpleNamespace(
        name="test",
        wait_subject=subject,
        poll_interval=poll_interval,
        state=state_fn,
    )


@pytest.fixture()
def already_stopped() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


class TestWaitingForSlurm:
    """``wait_for_job``: up to fifteen minutes of five-second polls."""

    def test_it_gives_up_when_the_caller_has(self, already_stopped):
        started = time.monotonic()
        state = triage_mod.wait_for_job(
            "123",
            timeout=900,
            stop=already_stopped,
            backend=_backend("RUNNING"),
        )

        assert state == "ABANDONED(RUNNING)"
        assert time.monotonic() - started < 5, "it waited out an interval first"

    def test_it_says_which_state_it_walked_away_from(self, already_stopped):
        """The job is still out there; the record should say what it was doing."""
        state = triage_mod.wait_for_job(
            "1",
            timeout=900,
            stop=already_stopped,
            backend=_backend("PENDING"),
        )

        assert state == "ABANDONED(PENDING)"

    def test_a_finished_job_is_still_reported_normally(self, already_stopped):
        """Stopping must not mask an answer that had already arrived."""
        state = triage_mod.wait_for_job(
            "1",
            timeout=900,
            stop=already_stopped,
            backend=_backend("COMPLETED"),
        )

        assert state == "COMPLETED"

    def test_without_a_flag_it_times_out_as_before(self):
        state = triage_mod.wait_for_job(
            "1",
            timeout=0.05,
            interval=0.01,
            backend=_backend("RUNNING"),
        )

        assert state == "TIMEOUT_WAITING(RUNNING)"

    def test_it_wakes_mid_sleep_rather_than_at_the_interval(self):
        """The five-second poll is where the time goes."""
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()

        started = time.monotonic()
        triage_mod.wait_for_job(
            "1",
            timeout=900,
            interval=30,
            stop=stop,
            backend=_backend("RUNNING"),
        )

        assert time.monotonic() - started < 5


class TestWaitProgressLogging:
    def test_local_poll_stays_fast_without_repeating_the_same_state(self, monkeypatch, caplog):
        states = iter(("RUNNING", "RUNNING", "RUNNING", "COMPLETED"))
        intervals: list[float] = []
        monkeypatch.setattr(
            triage_mod,
            "pause",
            lambda _stop, interval: intervals.append(interval) or False,
        )

        with caplog.at_level(logging.INFO, logger=triage_mod.log.name):
            state = triage_mod.wait_for_job(
                "local:123:456",
                timeout=900,
                backend=_backend(
                    lambda *_args: next(states),
                    poll_interval=0.2,
                    subject="local process",
                ),
            )

        messages = [record.getMessage() for record in caplog.records]
        assert state == "COMPLETED"
        assert intervals == [0.2, 0.2, 0.2]
        assert messages.count("local process local:123:456 state=RUNNING ...") == 1
        assert messages[-1] == "local process local:123:456 reached COMPLETED"


class TestTheWatchLoop:
    """``poll_jobs`` keeps calling the model; that costs money after the ask."""

    def test_it_ends_when_the_caller_gives_up(self, tmp_path, already_stopped):
        from aorta.cia.watch import poll as poll_mod

        started = time.monotonic()
        poll_mod.poll_jobs(tmp_path, max_rounds=100, stop=already_stopped)

        assert time.monotonic() - started < 5

    def test_it_does_not_scan_at_all_once_stopped(self, tmp_path, monkeypatch, already_stopped):
        from aorta.cia.watch import poll as poll_mod

        scans = []
        monkeypatch.setattr(poll_mod, "scan_active_jobs", lambda root: scans.append(root) or [])
        poll_mod.poll_jobs(tmp_path, max_rounds=100, stop=already_stopped)

        assert scans == []


class TestTheFourHourProbe:
    """``run_aorta_probe`` waits longest and is the most likely to be orphaned."""

    def test_it_stops_waiting_for_the_matrix(self, tmp_path, monkeypatch):
        from aorta.cia.autopsy import probe as probe_mod

        stop = threading.Event()
        checks = []

        class _Result:
            stdout = "WAITING"
            stderr = ""

        def fake_ssh(host, cmd, background=False):
            checks.append(cmd)
            stop.set()  # the caller gives up while the sweep is still running
            return _Result()

        monkeypatch.setattr(probe_mod, "_ssh", fake_ssh)
        monkeypatch.setattr(
            probe_mod, "resolve_recipe", lambda root, job: ("recipe.yaml", "sidecar")
        )
        job = JobRecord(
            job_id="cia-test",
            node="node1",
            recipe="a recipe",
            launched_at="2026-01-01T00:00:00Z",
            log_path="/tmp/cia-test/watch.log",
            aorta_output="/tmp/cia-test/aorta",
        )

        started = time.monotonic()
        result = probe_mod.run_aorta_probe(tmp_path, job, head_node="head", stop=stop)

        assert result is None
        assert time.monotonic() - started < 5, "it slept out a thirty-second poll"


class TestThePauseContract:
    """Everything above depends on this one helper behaving."""

    def test_a_set_flag_returns_immediately_and_says_so(self):
        event = threading.Event()
        event.set()

        started = time.monotonic()
        assert pause(event, 3600) is True
        assert time.monotonic() - started < 1

    def test_an_unset_flag_sleeps_the_interval(self):
        started = time.monotonic()
        assert pause(threading.Event(), 0.05) is False
        assert time.monotonic() - started >= 0.04
