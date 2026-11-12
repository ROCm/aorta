"""The watchdog (design Section 6.2, Watchdog).

- Every leg has a timeout (`reward.timeouts`: gate 60 s, compile 120 s, simulator run 120 s, ASan
  request 900 s). A job past it is `unsettled(timeout)` and its worker is restarted; the result is
  never cached and not retried (it says nothing about the candidate, and nothing new would be learnt
  by waiting again; the orchestrator's retry policy is what re-submits it).
- A worker that is silent for `stall_s` (300 s) while it holds work is killed and restarted; its work
  becomes `unsettled(worker_stalled)` and is retried once elsewhere. Silence is the absence of a
  heartbeat: each worker's heartbeat thread asks its adapter for `health()` every `heartbeat_s`
  (10 s) and beats only on a prompt, responsive answer.
- The two meet when a job passes its deadline: if its worker has also missed three heartbeats, the
  worker is what hung, so it is a stall (retried once elsewhere); if the worker is still beating,
  the job itself was slow, so it is a timeout (not retried). Without this, every simulator leg
  (deadline 120 s) would be called a timeout long before `stall_s` (300 s) could see the stall.
- If no leg completes anywhere for `no_progress_s` (600 s) while work is pending, the service raises
  an alarm and dumps its state.

These are the only places the service creates a timeout, and it is always `unsettled`. A timeout the
checker attributes to the candidate (`fail`, reason `timeout`) can only come from an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerView:
    worker_id: str
    busy: bool  # holds a job, a control run or a request
    t_item: float | None  # when the current item started (clock seconds)
    limit_s: float | None  # the current item's timeout
    last_beat: float


@dataclass(frozen=True)
class Verdict:
    worker_id: str
    reason: str  # "timeout" | "worker_stalled"
    silent_s: float
    running_s: float


def check_workers(views: list[WorkerView], *, now: float, stall_s: float, silent_after_s: float) -> list[Verdict]:
    """The workers to restart now, and why: past the deadline on a worker silent for `silent_after_s`
    (three heartbeats) or silent for `stall_s` while busy is a stall; past the deadline on a live
    worker is a timeout."""
    out = []
    for v in views:
        if not v.busy or v.t_item is None:
            continue
        running = now - v.t_item
        silent = now - v.last_beat
        if v.limit_s is not None and running > v.limit_s:
            out.append(Verdict(v.worker_id, "worker_stalled" if silent > silent_after_s else "timeout", silent, running))
        elif silent > stall_s:
            out.append(Verdict(v.worker_id, "worker_stalled", silent, running))
    return out


@dataclass
class ProgressAlarm:
    """Raised once when no leg has completed for `no_progress_s` while work is pending; re-armed by
    the next completion."""

    no_progress_s: float = 600.0
    t_last_progress: float | None = None
    raised: bool = False

    def progress(self, now: float) -> None:
        self.t_last_progress = now
        self.raised = False

    def check(self, now: float, pending: bool) -> bool:
        if not pending:
            self.t_last_progress = now
            return False
        if self.t_last_progress is None:
            self.t_last_progress = now
            return False
        if not self.raised and now - self.t_last_progress > self.no_progress_s:
            self.raised = True
            return True
        return False
