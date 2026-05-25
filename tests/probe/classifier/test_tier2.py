"""Tests for Tier 2 hang detector (FR 2.2).

The predicate is pure (operates on a :class:`HangSignals` snapshot)
so it's tested directly. ``HangMonitor`` is exercised through its
public ``hang_detected`` flag with synthetic stdout / IO probes —
no real subprocess hang required.
"""

from __future__ import annotations

import time
from pathlib import Path

from aorta.probe.classifier import tier2_hang
from aorta.probe.classifier.tier2_hang import (
    DEFAULT_HANG_GRACE_SEC,
    DEFAULT_HANG_WINDOW_SEC,
    HangMonitor,
    HangSignals,
    evaluate_predicate,
)


def _sig(stdout: bool, gpu: bool, io: bool, *, elapsed: float = 120.0) -> HangSignals:
    return HangSignals(
        stdout_silent=stdout,
        gpu_idle=gpu,
        io_idle=io,
        elapsed_sec=elapsed,
    )


def test_two_of_three_required():
    """Single-signal never fires; two-of-three trips it."""
    assert not evaluate_predicate(_sig(True, False, False))
    assert not evaluate_predicate(_sig(False, True, False))
    assert not evaluate_predicate(_sig(False, False, True))
    assert evaluate_predicate(_sig(True, True, False))
    assert evaluate_predicate(_sig(True, False, True))
    assert evaluate_predicate(_sig(False, True, True))
    assert evaluate_predicate(_sig(True, True, True))


def test_grace_window_suppresses_early_detection():
    """During ``hang_grace_period_at_start``, all-three signals don't fire."""
    early = HangSignals(stdout_silent=True, gpu_idle=True, io_idle=True, elapsed_sec=10.0)
    assert not evaluate_predicate(early, grace_period_sec=60.0)


def test_grace_window_default_is_60s():
    """Default grace period matches the rubric's stated default."""
    assert DEFAULT_HANG_GRACE_SEC == 60.0
    assert DEFAULT_HANG_WINDOW_SEC == 30.0


def test_grace_boundary_fires_at_or_above():
    """At exactly the grace boundary, two-of-three trips immediately."""
    boundary = HangSignals(
        stdout_silent=True,
        gpu_idle=True,
        io_idle=False,
        elapsed_sec=DEFAULT_HANG_GRACE_SEC,
    )
    assert evaluate_predicate(boundary)


# ---- HangMonitor integration (no real hang, just signal plumbing) -------


def test_monitor_starts_and_stops_cleanly(tmp_path: Path):
    """``HangMonitor`` start/stop round-trips even without a subprocess."""
    stdout = tmp_path / "stdout.log"
    stdout.write_text("hello\n", encoding="utf-8")
    # PID 1 always exists; the monitor will read /proc/1/io and either
    # get a value or fail-soft to 0 on hosts that don't expose it.
    mon = HangMonitor(
        pid=1,
        stdout_path=stdout,
        hang_window_sec=0.05,
        hang_grace_period_at_start=0.0,
        poll_interval_sec=0.01,
    )
    mon.start()
    time.sleep(0.05)
    mon.stop()
    # No assertion on hang_detected -- depends on whether /proc/1/io
    # was readable and how the test scheduler interleaved; this test
    # is purely about clean start/stop.


def test_monitor_no_hang_during_grace(tmp_path: Path):
    """The monitor's ``hang_detected`` stays False while inside grace."""
    stdout = tmp_path / "stdout.log"
    stdout.write_text("hello\n", encoding="utf-8")
    mon = HangMonitor(
        pid=1,
        stdout_path=stdout,
        hang_window_sec=0.05,
        hang_grace_period_at_start=10.0,  # well above the test window
        poll_interval_sec=0.01,
    )
    mon.start()
    time.sleep(0.05)
    mon.stop()
    assert mon.hang_detected is False


def test_module_detector_id():
    """Detector ID is the rubric-mandated ``tier2:hang``."""
    assert tier2_hang.DETECTOR_HANG == "tier2:hang"
