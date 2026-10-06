"""Asking long-running cluster work to stop.

A triage is a chain of waits -- for Slurm to finish, for Watch to notice, for a
sweep to drop its matrix on a compute node -- and the caller that started it has
its own patience, which is usually shorter. The chat tool gives up after
``triage_timeout`` and tells the user so.

Giving up on the *answer* used to be all that happened. The work carried on:
the thread stayed in its sleep loop, kept polling ``sacct``, kept calling the
model, and because the pool's threads are not daemons and ``concurrent.futures``
joins them at exit, the process could not shut down until every abandoned
triage had run itself out. Each timed-out call added another.

So the waits take a stop flag and check it. Nothing here can interrupt a call
already inside the kernel -- an ``ssh`` mid-handshake still runs to its own
timeout -- but the sleeps between attempts are where these loops spend nearly
all of their time, and a loop that wakes to find the flag set returns instead
of starting another attempt.

The shape is the one :class:`aorta.probe.classifier.tier2_hang.HangMonitor`
already uses: an :class:`~threading.Event` for the flag, and ``wait`` on it as
the sleep, so a stop lands as soon as it is signalled rather than at the end of
whatever interval was in progress.
"""

from __future__ import annotations

import threading
import time

#: A stop flag, or ``None`` for callers that never cancel -- the CLI, the
#: tests, anything whose triage owns the process it runs in.
Stop = threading.Event | None


def stopped(stop: Stop) -> bool:
    """Whether the caller has asked this work to stop."""
    return stop is not None and stop.is_set()


def pause(stop: Stop, seconds: float) -> bool:
    """Sleep for *seconds*, returning early and ``True`` if asked to stop.

    The return value is what the loop branches on, so a caller reads as
    ``if pause(stop, interval): return`` and cannot mistake "time passed" for
    "carry on" the way a bare :func:`time.sleep` invites.

    A caller with no flag sleeps through :func:`time.sleep` rather than through
    an event nobody can set. It is the same wait either way, but the poll loops
    are tested by patching ``time.sleep`` -- to skip the wait, and in places to
    use it as the hook that writes the next chunk of log -- and waiting on a
    private event instead would quietly step around the seam those tests hold.
    """
    if stop is None:
        time.sleep(seconds)
        return False
    return stop.wait(seconds)


__all__ = ["Stop", "pause", "stopped"]
