"""Trusted controls (design Section 6.2; decision D15).

Simulator workers have no natural batch once checks stream, so the bracket is per worker: each worker
runs a trusted control before its first job, and then after every `controls.every_jobs` jobs or
`controls.every_s` seconds, whichever comes first. A result on a bracketed worker is provisional
until the next control on that worker:

- after-control clean: every provisional result gets `bracket = {before, after, clean: True}`;
- after-control dirty (or lost with its worker): every result since the last clean control becomes
  `unsettled(control_dirty)` (`control_lost`) with `clean: False`, and is retried once on another
  worker; the worker is quarantined until a control comes back clean.

A worker that is idle while holding provisional results also runs its control at once, so the last
results of an iteration do not wait `every_s` for their bracket (only ever more controls, never
fewer). ASan requests are bracketed per request instead: controls travel as the first and last
packages of the request, or as separate requests before and after it.

A control run is clean only if every one of its control jobs comes back `clean`: an unsettled
control cannot vouch for anything.

`ControlLedger` records every control run: worker, time, verdict.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..schemas import LegResult


def control_clean(results: list[LegResult]) -> bool:
    return bool(results) and all(r.state == "clean" for r in results)


def bracket(before: str | None, after: str, clean: bool) -> dict:
    return {"before": before, "after": after, "clean": bool(clean and before is not None)}


@dataclass
class ControlRecord:
    control_run_id: str
    worker_id: str
    pool: str
    t_start: float  # UTC
    t_end: float
    clean: bool
    states: dict[str, str]  # leg -> state of its control job
    brackets: int  # provisional results it closed


@dataclass
class ControlLedger:
    records: list[ControlRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, record: ControlRecord) -> None:
        with self._lock:
            self.records.append(record)

    def snapshot(self) -> list[ControlRecord]:
        with self._lock:
            return list(self.records)

    def summary(self) -> dict:
        recs = self.snapshot()
        return {"runs": len(recs), "dirty": sum(not r.clean for r in recs),
                "by_pool": {p: sum(r.pool == p for r in recs) for p in sorted({r.pool for r in recs})}}


def due(*, jobs_since: int, t_last: float | None, now: float, every_jobs: int, every_s: float) -> bool:
    """Whether a worker's next item must be a control: before its first job, then every
    `every_jobs` jobs or `every_s` seconds, whichever comes first."""
    if t_last is None:
        return True
    return jobs_since >= every_jobs or (now - t_last) >= every_s
