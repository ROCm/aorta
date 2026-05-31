"""Tier 2 hang detector for ``aorta probe`` (issue #188).

Fires :data:`DETECTOR_HANG` (``tier2:hang``) when **at least two of
three** in-flight signals agree, AND only after the trial's grace
period has elapsed:

* ``stdout_silent`` -- no stdout writes for ``hang_window_sec``.
* ``gpu_idle`` -- ``amd-smi`` reports zero activity for the window.
* ``io_idle`` -- ``/proc/<pid>/io``'s ``rchar+wchar`` is unchanged
  for the window.

Both knobs come from the recipe (``hang_window_sec`` defaults to
30, ``hang_grace_period_at_start`` to 60 — chosen to be longer than
typical PyTorch import + dataloader warm-up). Recipes pin them via
the new ``hang_window_sec`` and ``hang_grace_period_at_start``
top-level keys (accepted only in ``mode: probe`` recipes).

The detector is split into two surfaces:

* :func:`evaluate_predicate` -- pure, takes a :class:`HangSignals`
  snapshot and returns ``True`` iff two-of-three agree AND the
  grace window has elapsed. Unit-testable without a process.
* :class:`HangMonitor` -- a polling loop runnable in a background
  thread; consumes the subprocess's PID, the trial's stdout path,
  and an amd-smi shim, and calls ``evaluate_predicate`` once per
  poll. The first ``True`` evaluation flags the trial as hung
  (the workload itself decides whether to kill the process — the
  workload owns process lifecycle).

The ``aorta probe`` SubprocessWorkload runs the monitor in a
background thread alongside the synchronous ``proc.wait(...)``
call. The monitor's sole side-effect is to flip the
:attr:`HangMonitor.hang_detected` flag once the predicate trips;
the workload reads it post-exit to decide whether to add
``tier2:hang`` to ``failure_detectors_fired``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DETECTOR_HANG = "tier2:hang"

DEFAULT_HANG_WINDOW_SEC = 30.0
DEFAULT_HANG_GRACE_SEC = 60.0

# Upper-bound on how long :attr:`HangMonitor.gpu_idle_probe` can
# block. Mirrors the live ``amd-smi monitor`` subprocess timeout in
# :mod:`aorta.probe.classifier.tier3_kernel` (10s) plus a 1s
# buffer. Used by :meth:`HangMonitor.stop` to size the thread-join
# budget so a poll currently blocked inside ``amd-smi`` is given
# time to return before teardown -- if ``stop()`` returned early,
# the poll thread would race the *next* trial's monitor for
# ``Tier3State`` reads (cross-trial state corruption). Per Sonbol's
# PR #197 review.
#
# Hard-coded rather than imported from tier3 so the tier2 module
# keeps no compile-time dependency on tier3 (the monitor takes
# ``gpu_idle_probe`` as a callable specifically so the two tiers
# stay decoupled). If the amd-smi timeout grows materially in
# tier3, raise this constant in lockstep.
GPU_IDLE_PROBE_MAX_BLOCK_SEC = 11.0


@dataclass(frozen=True)
class HangSignals:
    """One sample of the three two-of-three signals.

    ``elapsed_sec`` is the time since trial start; the predicate
    short-circuits to ``False`` when ``elapsed_sec < grace_period``
    so the early phase of the trial (PyTorch import, dataloader
    setup) can be silent without flagging.
    """

    stdout_silent: bool
    gpu_idle: bool
    io_idle: bool
    elapsed_sec: float


def evaluate_predicate(
    signals: HangSignals,
    *,
    grace_period_sec: float = DEFAULT_HANG_GRACE_SEC,
) -> bool:
    """Return True iff the trial looks hung per the two-of-three rule.

    Returns ``False`` while ``elapsed_sec < grace_period_sec`` no
    matter how many signals agree — the grace window deliberately
    suppresses every false-positive during workload startup.

    The "two of three" choice (rather than "all three") is a
    deliberate trade-off: a workload that is hung but periodically
    flushes a heartbeat to stderr (stdout silent + GPU idle + IO
    non-idle because of the heartbeat) would not trip a strict
    "all three" rule, and the AORTA team has seen that pattern in
    NCCL collective hangs. Two-of-three trips it; if false
    positives become a problem in practice the recipe can raise
    ``hang_window_sec``.
    """
    if signals.elapsed_sec < grace_period_sec:
        return False
    agreeing = sum((signals.stdout_silent, signals.gpu_idle, signals.io_idle))
    return agreeing >= 2


@dataclass
class HangMonitor:
    """Polls the three signals in a background thread.

    Attributes:
        pid: PID of the user subprocess (read /proc/<pid>/io).
        stdout_path: Trial stdout log; mtime advances when the
            child writes.
        hang_window_sec: How long each signal must hold to count.
        hang_grace_period_at_start: How long to wait before
            firing at all (rubric default 60s).
        poll_interval_sec: How often the monitor wakes. Small
            enough to catch a hang within a window; large enough
            to keep monitoring overhead negligible.
        gpu_idle_probe: A callable returning the current GPU
            "idle" boolean — typically a closure over
            :func:`aorta.probe.classifier.tier3_kernel.poll_amd_smi`
            results. ``None`` (default) is treated as "GPU idle
            unknown" → contributes False to the two-of-three.
        hang_detected: Flips to True the first time
            :func:`evaluate_predicate` returns True. The workload
            reads this after ``proc.wait()`` to decide whether
            ``tier2:hang`` belongs in ``failure_detectors_fired``.
    """

    pid: int
    stdout_path: Path
    hang_window_sec: float = DEFAULT_HANG_WINDOW_SEC
    hang_grace_period_at_start: float = DEFAULT_HANG_GRACE_SEC
    poll_interval_sec: float = 5.0
    gpu_idle_probe: Callable[[], bool] | None = None

    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    hang_detected: bool = False
    started_at: float = 0.0

    def start(self) -> None:
        """Spawn the polling thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.started_at = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"aorta-probe-hang-monitor-{self.pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to exit and join, with a budget that covers the slowest probe.

        The join budget must outlast the longest in-flight call the
        monitor thread can be inside when ``stop()`` runs. The
        ``gpu_idle_probe`` closure typically wraps
        ``tier3_kernel.poll_amd_smi``, whose subprocess timeout is
        ~10s; a naive ``join(timeout=poll_interval + 1)`` (=6s)
        returns *before* a stuck amd-smi call gives up, leaving an
        orphan thread that lives into the next trial and races for
        ``Tier3State`` reads. Take the max of the two budgets; log
        a warning if the thread is still alive after that (it
        means a probe wedged past its own timeout — rare, but the
        caller deserves a breadcrumb in the run log).
        Per Sonbol's PR #197 review.
        """
        self._stop.set()
        if self._thread is None:
            return
        join_budget = max(self.poll_interval_sec + 1.0, GPU_IDLE_PROBE_MAX_BLOCK_SEC)
        self._thread.join(timeout=join_budget)
        if self._thread.is_alive():
            log.warning(
                "HangMonitor.stop(): poll thread for pid=%d still alive after "
                "%.1fs join budget; continuing teardown (next trial may see "
                "stale tier-3 state).",
                self.pid,
                join_budget,
            )
        self._thread = None

    def _run(self) -> None:
        """Polling loop. Exits when ``_stop`` is set or hang is detected.

        The loop computes the three signals each iteration and
        feeds them to :func:`evaluate_predicate`. The first True
        flips ``hang_detected`` and exits; the workload reads the
        flag once ``proc.wait()`` returns.
        """
        last_stdout_mtime = self._stdout_mtime()
        last_io_total = self._io_total()
        last_stdout_seen_at = time.monotonic()
        last_io_seen_at = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - self.started_at

            current_mtime = self._stdout_mtime()
            if current_mtime != last_stdout_mtime:
                last_stdout_mtime = current_mtime
                last_stdout_seen_at = now

            # ``_io_total`` returns None when /proc/<pid>/io can't be
            # read. "Unknown" must NOT vote io_idle=True -- if we
            # treated None as a value, two consecutive Nones would
            # look like a stable I/O counter and the io_idle window
            # would tick over on its own. Skip the staleness check on
            # missing-source iterations and bind io_idle=False so the
            # detector can never fire from the I/O leg alone.
            current_io = self._io_total()
            if current_io is None:
                io_idle = False
            else:
                if current_io != last_io_total:
                    last_io_total = current_io
                    last_io_seen_at = now
                io_idle = (now - last_io_seen_at) >= self.hang_window_sec

            stdout_silent = (now - last_stdout_seen_at) >= self.hang_window_sec
            gpu_idle = bool(self.gpu_idle_probe()) if self.gpu_idle_probe else False

            signals = HangSignals(
                stdout_silent=stdout_silent,
                gpu_idle=gpu_idle,
                io_idle=io_idle,
                elapsed_sec=elapsed,
            )
            if evaluate_predicate(
                signals,
                grace_period_sec=self.hang_grace_period_at_start,
            ):
                self.hang_detected = True
                return

            # Sleep with the stop event so a quick stop() during a
            # poll wakes the thread immediately rather than waiting
            # for the full poll_interval_sec.
            if self._stop.wait(self.poll_interval_sec):
                return

    def _stdout_mtime(self) -> float:
        """Last-modified timestamp of the trial's stdout log.

        Returns ``0.0`` when the file does not yet exist — the
        workload may not have opened it before the monitor first
        polled. ``0.0`` is a safe initial value because a real
        mtime will always differ from it on the next poll.
        """
        try:
            return self.stdout_path.stat().st_mtime
        except FileNotFoundError:
            return 0.0
        except OSError:
            return 0.0

    def _io_total(self) -> int | None:
        """``rchar + wchar`` from ``/proc/<pid>/io``.

        Returns ``None`` when the file is unreadable (process already
        exited, permission denied, ``ptrace_scope`` restricted, NFS
        with no proc visibility). The monitor's ``_run`` loop treats
        ``None`` as "I/O availability unknown" and contributes
        ``io_idle=False`` to the predicate -- without this distinction,
        an unreadable ``/proc/<pid>/io`` would look like a permanently
        stable I/O counter (``current == last`` because both are
        ``0``) and could flip ``io_idle=True`` after
        ``hang_window_sec``, false-firing ``tier2:hang`` on hosts where
        I/O telemetry simply isn't readable.
        """
        try:
            with open(f"/proc/{self.pid}/io", encoding="utf-8") as fh:
                rchar = wchar = 0
                for line in fh:
                    if line.startswith("rchar:"):
                        rchar = int(line.split(":", 1)[1].strip())
                    elif line.startswith("wchar:"):
                        wchar = int(line.split(":", 1)[1].strip())
                return rchar + wchar
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            return None


def read_proc_io_total(pid: int) -> int | None:
    """Convenience for tests: read ``rchar+wchar`` for ``pid``.

    Returns ``None`` when ``/proc/<pid>/io`` is unreadable so the
    caller can distinguish "0 bytes done" from "no information".
    """
    try:
        with open(f"/proc/{pid}/io", encoding="utf-8") as fh:
            rchar = wchar = 0
            for line in fh:
                if line.startswith("rchar:"):
                    rchar = int(line.split(":", 1)[1].strip())
                elif line.startswith("wchar:"):
                    wchar = int(line.split(":", 1)[1].strip())
            return rchar + wchar
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


# Re-export so tests can ``from aorta.probe.classifier.tier2_hang
# import _ALL_DETECTOR_IDS`` (parity with other tiers).
ALL_DETECTOR_IDS = (DETECTOR_HANG,)


__all__ = [
    "ALL_DETECTOR_IDS",
    "DEFAULT_HANG_GRACE_SEC",
    "DEFAULT_HANG_WINDOW_SEC",
    "DETECTOR_HANG",
    "HangMonitor",
    "HangSignals",
    "evaluate_predicate",
    "read_proc_io_total",
]
# Keep ``os`` imported -- read_proc_io_total opens via path and the
# linter would otherwise flag the unused stdlib import while we keep
# the option open for switching to ``os.read``-based I/O.
_ = os.path
