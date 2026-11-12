"""Warm workers (design Section 6.2, Warm workers).

A worker is a long-lived thread with its own adapter instance, so whatever the adapter keeps loaded
(a simulator, a container, a compiled harness) is paid for once, not per job. The coordinator hands
a worker one item at a time: a job, a control run, or an ASan request. The worker runs it with the
adapter and posts the result back; it never decides anything itself.

Each worker also has a heartbeat thread. Every `heartbeat_s` it asks the adapter for `health()`
and records a beat only if the answer is prompt and responsive (`base.responsive`). A hung adapter,
or one that reports a stall in progress, therefore goes silent, which is what the watchdog's
`stall_s` rule measures. A long job on a responsive adapter keeps beating and is caught by its leg
timeout instead.

When the watchdog gives up on a worker, the coordinator abandons it: its generation is retired (any
late result it posts is ignored), its adapter is closed (which interrupts a fake stall or kills a
subprocess), and a fresh worker with a new id takes its place.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .adapters.base import responsive, safe_run, safe_run_batch
from .clock import Clock

log = logging.getLogger(__name__)

STOP = object()
_generation = itertools.count(1)


@dataclass
class Item:
    kind: str  # "job" | "control" | "batch"
    payload: Any
    jobs: list  # the LegJobs to run (one for a job)
    limit_s: float  # the watchdog's timeout for this item
    t_start: float = 0.0  # clock seconds, set at dispatch


@dataclass
class Worker:
    worker_id: str
    pool: str
    adapter: Any
    clock: Clock
    post: Callable[[tuple], None]
    heartbeat_s: float = 10.0
    batched: bool = False
    generation: int = field(default_factory=lambda: next(_generation))
    inbox: queue.Queue = field(default_factory=queue.Queue)
    stop: threading.Event = field(default_factory=threading.Event)
    # coordinator-owned state
    state: str = "fresh"  # fresh | idle | busy | dead
    item: Item | None = None
    last_beat: float = 0.0
    jobs_since_control: int = 0
    t_last_control: float | None = None
    last_clean_control: str | None = None
    provisional: list = field(default_factory=list)
    quarantined: bool = False
    needs_control: bool = True
    dirty_streak: int = 0
    control_seq: int = 0
    jobs_run: int = 0

    def start(self) -> None:
        self.last_beat = self.clock.now()
        self._thread = threading.Thread(target=self._main, name=f"reward-worker-{self.worker_id}", daemon=True)
        self._beat = threading.Thread(target=self._beat_main, name=f"reward-beat-{self.worker_id}", daemon=True)
        self._thread.start()
        self._beat.start()

    def give(self, item: Item) -> None:
        item.t_start = self.clock.now()
        self.item = item
        self.state = "busy"
        self.inbox.put(item)

    def _main(self) -> None:
        while True:
            item = self.inbox.get()
            if item is STOP:
                return
            self.last_beat = self.clock.now()
            try:
                if item.kind == "batch" and isinstance(item.payload, dict) and item.payload.get("separate"):
                    nb, na = item.payload["n_before"], item.payload["n_after"]
                    parts = (item.jobs[:nb], item.jobs[nb:len(item.jobs) - na], item.jobs[len(item.jobs) - na:])
                    results = [r for part in parts if part for r in safe_run_batch(self.adapter, part, self.clock.wall)]
                elif item.kind == "batch":
                    results = safe_run_batch(self.adapter, item.jobs, self.clock.wall)
                elif item.kind == "control" and self.batched:
                    results = safe_run_batch(self.adapter, item.jobs, self.clock.wall)
                else:
                    results = [safe_run(self.adapter, job, self.clock.wall) for job in item.jobs]
            except BaseException as exc:  # never let a worker thread die silently
                log.exception("worker %s: %s", self.worker_id, type(exc).__name__)
                results = None
            self.last_beat = self.clock.now()
            self.post(("item_done", self.worker_id, self.generation, item, results))

    def _beat_main(self) -> None:
        while not self.stop.is_set():
            try:
                ok = responsive(self.adapter.health())
            except Exception:
                ok = False
            if ok and not self.stop.is_set():
                self.last_beat = self.clock.now()
            # never faster than every 20 ms of real time, however fast a test's clock runs
            self.stop.wait(max(self.clock.real(self.heartbeat_s), 0.02))

    def abandon(self) -> None:
        """Retire this worker: stop its threads and close its adapter (interrupting any run)."""
        self.state = "dead"
        self.stop.set()
        self.inbox.put(STOP)
        close = getattr(self.adapter, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                log.warning("worker %s: adapter close failed", self.worker_id)

    def view(self):
        from .watchdog import WorkerView
        busy = self.state == "busy" and self.item is not None
        return WorkerView(self.worker_id, busy, self.item.t_start if busy else None,
                          self.item.limit_s if busy else None, self.last_beat)
