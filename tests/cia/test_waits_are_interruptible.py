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

import threading
import time
from pathlib import Path

import pytest

from aorta.cia import triage as triage_mod
from aorta.cia.cancellation import pause
from aorta.cia.launch.job import JobRecord


@pytest.fixture()
def already_stopped() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


class TestWaitingForSlurm:
    """``wait_for_job``: up to fifteen minutes of five-second polls."""

    def test_it_gives_up_when_the_caller_has(self, monkeypatch, already_stopped):
        monkeypatch.setattr(triage_mod, "sacct_state", lambda _id: "RUNNING")

        started = time.monotonic()
        state = triage_mod.wait_for_job("123", timeout=900, stop=already_stopped)

        assert state == "ABANDONED(RUNNING)"
        assert time.monotonic() - started < 5, "it waited out an interval first"

    def test_it_says_which_state_it_walked_away_from(self, monkeypatch, already_stopped):
        """The job is still out there; the record should say what it was doing."""
        monkeypatch.setattr(triage_mod, "sacct_state", lambda _id: "PENDING")

        state = triage_mod.wait_for_job("1", timeout=900, stop=already_stopped)

        assert state == "ABANDONED(PENDING)"

    def test_a_finished_job_is_still_reported_normally(self, monkeypatch, already_stopped):
        """Stopping must not mask an answer that had already arrived."""
        monkeypatch.setattr(triage_mod, "sacct_state", lambda _id: "COMPLETED")

        state = triage_mod.wait_for_job("1", timeout=900, stop=already_stopped)

        assert state == "COMPLETED"

    def test_without_a_flag_it_times_out_as_before(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "sacct_state", lambda _id: "RUNNING")

        state = triage_mod.wait_for_job("1", timeout=0.05, interval=0.01)

        assert state == "TIMEOUT_WAITING(RUNNING)"

    def test_it_wakes_mid_sleep_rather_than_at_the_interval(self, monkeypatch):
        """The five-second poll is where the time goes."""
        monkeypatch.setattr(triage_mod, "sacct_state", lambda _id: "RUNNING")
        stop = threading.Event()
        threading.Timer(0.1, stop.set).start()

        started = time.monotonic()
        triage_mod.wait_for_job("1", timeout=900, interval=30, stop=stop)

        assert time.monotonic() - started < 5


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
