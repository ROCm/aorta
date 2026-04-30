"""
NaN-correlated eBPF event timeline for distributed training debugging.

Joins user-space NaN detection timestamps with kernel-level eBPF events
(queue dispatch, memory migration, race detection, DMA overlap) to produce
a unified timeline that answers *why* NaN appeared, not just *where*.

Typical flow:
1. User-space sanitizer detects NaN at step N and records a timestamp.
2. eBPF tracers (queue, memory, race, DMA, RCCL) run concurrently.
3. ``NaNCorrelator`` merges all event streams and extracts the kernel
   events within a configurable window around each NaN detection.
4. Output: per-NaN JSON report with root-cause indicators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class NaNDetection:
    """A single NaN detection event from user-space instrumentation."""

    timestamp_ns: int
    step: int
    rank: int = 0
    source: str = ""
    details: str = ""

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000


@dataclass
class CorrelatedNaNReport:
    """Kernel-level context around a single NaN detection event."""

    nan_event: NaNDetection
    window_ms: float

    queue_events_in_window: int = 0
    memory_events_in_window: int = 0
    evictions_in_window: int = 0
    bo_moves_in_window: int = 0
    race_events_in_window: int = 0
    dma_overlaps_in_window: int = 0
    collective_races_in_window: int = 0

    dispatch_gap_spike: bool = False
    avg_dispatch_gap_us: float = 0.0
    max_dispatch_gap_us: float = 0.0
    concurrent_rings: List[int] = field(default_factory=list)
    migration_bytes_in_window: int = 0

    diagnosis: str = ""
    confidence: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nan_detected_at_ns": self.nan_event.timestamp_ns,
            "nan_step": self.nan_event.step,
            "nan_rank": self.nan_event.rank,
            "nan_source": self.nan_event.source,
            "window_ms": self.window_ms,
            "kernel_events": {
                "queue_events": self.queue_events_in_window,
                "memory_events": self.memory_events_in_window,
                "evictions": self.evictions_in_window,
                "bo_moves": self.bo_moves_in_window,
                "race_events": self.race_events_in_window,
                "dma_overlaps": self.dma_overlaps_in_window,
                "collective_races": self.collective_races_in_window,
                "dispatch_gap_spike": self.dispatch_gap_spike,
                "avg_dispatch_gap_us": self.avg_dispatch_gap_us,
                "max_dispatch_gap_us": self.max_dispatch_gap_us,
                "concurrent_rings": self.concurrent_rings,
                "migration_bytes": self.migration_bytes_in_window,
            },
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
        }


class NaNCorrelator:
    """
    Correlate NaN detections with kernel-level eBPF events.

    Takes timestamped NaN detection events and the raw event lists from
    all eBPF tracers, then produces a ``CorrelatedNaNReport`` for each
    NaN showing what was happening at the kernel level in the surrounding
    time window.

    Usage::

        correlator = NaNCorrelator(window_ms=100.0)
        correlator.add_nan_event(NaNDetection(
            timestamp_ns=..., step=42, rank=0, source="post-all_reduce"
        ))
        correlator.set_queue_events(queue_tracer_events)
        correlator.set_memory_events(memory_tracer_events)
        correlator.set_race_events(race_detector_metrics.race_events)
        correlator.set_dma_overlaps(dma_tracer_metrics.overlap_events)
        correlator.set_collective_races(rccl_tracer_metrics.race_events)

        reports = correlator.correlate()
        for r in reports:
            print(json.dumps(r.to_dict(), indent=2))
    """

    def __init__(self, window_ms: float = 100.0):
        self._window_ms = window_ms
        self._nan_events: List[NaNDetection] = []
        self._queue_events: List[Any] = []
        self._memory_events: List[Any] = []
        self._race_events: List[Dict[str, Any]] = []
        self._dma_overlaps: List[Dict[str, Any]] = []
        self._collective_races: List[Dict[str, Any]] = []

    def add_nan_event(self, event: NaNDetection) -> None:
        self._nan_events.append(event)

    def add_nan_events_from_log(
        self,
        log_path: str | Path,
        include_pre: bool = False,
    ) -> int:
        """Parse NaN detection events from a sanitizer log file.

        Looks for lines matching the stream sanitizer format::

            [NaN POST-<op>] rank=<r> step=<s>: nan=<count>
            [NaN PRE-<op>]  rank=<r> step=<s>: nan=<count>
            [NaN DATADIST-<op>] rank=<r> step=<s>: nan=<count>

        By default only ``POST-*`` and ``DATADIST-*`` detections are
        recorded.  ``PRE-*`` markers indicate NaN values that already
        existed before the operation ran (i.e. they originated upstream)
        and are typically noise when correlating against kernel events
        for the *current* op, so they are skipped.

        Pass ``include_pre=True`` to also record ``PRE-*`` events when
        you want to correlate against the upstream operation that
        produced them.

        Returns the number of events parsed.
        """
        import re

        pattern = re.compile(
            r"\[NaN\s+(POST|PRE|DATADIST)-([^\]]+)\]\s*rank=(\d+)\s+step=(\d+)"
        )
        path = Path(log_path)
        if not path.exists():
            return 0

        accepted_kinds = {"POST", "DATADIST"}
        if include_pre:
            accepted_kinds.add("PRE")

        count = 0
        with open(path) as f:
            for line in f:
                m = pattern.search(line)
                if m and m.group(1) in accepted_kinds:
                    rank = int(m.group(3))
                    step = int(m.group(4))
                    source = f"{m.group(1)}-{m.group(2)}"
                    self._nan_events.append(
                        NaNDetection(
                            timestamp_ns=0,
                            step=step,
                            rank=rank,
                            source=source,
                            details=line.strip(),
                        )
                    )
                    count += 1
        return count

    def set_queue_events(self, events: List[Any]) -> None:
        """Accept DriverQueueEvent objects (or any with timestamp_ns, ring)."""
        self._queue_events = list(events)

    def set_memory_events(self, events: List[Any]) -> None:
        """Accept MemoryTraceEvent objects (or any with timestamp_ns, event_type, size_bytes)."""
        self._memory_events = list(events)

    def set_race_events(self, events: List[Any]) -> None:
        """Accept RaceEvent objects or dicts with ``timestamp_ns``."""
        self._race_events = [
            e.to_dict() if hasattr(e, "to_dict") else e for e in events
        ]

    def set_dma_overlaps(self, events: List[Any]) -> None:
        self._dma_overlaps = [
            e.to_dict() if hasattr(e, "to_dict") else e for e in events
        ]

    def set_collective_races(self, events: List[Any]) -> None:
        self._collective_races = [
            e.to_dict() if hasattr(e, "to_dict") else e for e in events
        ]

    def correlate(self) -> List[CorrelatedNaNReport]:
        """Produce a correlated report for each NaN detection event."""
        reports: List[CorrelatedNaNReport] = []

        for nan_ev in self._nan_events:
            report = self._build_report(nan_ev)
            reports.append(report)

        return reports

    def _build_report(self, nan_ev: NaNDetection) -> CorrelatedNaNReport:
        window_ns = int(self._window_ms * 1_000_000)
        ts = nan_ev.timestamp_ns
        lo = ts - window_ns
        hi = ts + window_ns

        report = CorrelatedNaNReport(
            nan_event=nan_ev,
            window_ms=self._window_ms,
        )

        use_timestamp_filter = ts > 0

        # --- Queue events ---
        # ``dispatch_gap_*`` is meant to capture how often dispatches fire,
        # so only consider dispatch events when computing gaps -- mixing in
        # submits would let submit cadence skew or trigger the spike
        # heuristic.  ``queue_events_in_window`` and ``concurrent_rings``
        # still reflect every queue event in the window.
        dispatch_gaps: List[float] = []
        ring_set: set[int] = set()
        q_in_window = []
        dispatches_in_window: List[Any] = []
        for ev in self._queue_events:
            if use_timestamp_filter and not (lo <= ev.timestamp_ns <= hi):
                continue
            q_in_window.append(ev)
            ring_set.add(ev.ring)
            if getattr(ev, "event_type", None) == "dispatch":
                dispatches_in_window.append(ev)

        report.queue_events_in_window = len(q_in_window)
        report.concurrent_rings = sorted(ring_set)

        dispatches_in_window.sort(key=lambda e: e.timestamp_ns)
        for i in range(1, len(dispatches_in_window)):
            gap_us = (
                dispatches_in_window[i].timestamp_ns
                - dispatches_in_window[i - 1].timestamp_ns
            ) / 1_000.0
            dispatch_gaps.append(gap_us)

        if dispatch_gaps:
            report.avg_dispatch_gap_us = sum(dispatch_gaps) / len(dispatch_gaps)
            report.max_dispatch_gap_us = max(dispatch_gaps)
            baseline = report.avg_dispatch_gap_us
            if baseline > 0 and report.max_dispatch_gap_us > baseline * 5:
                report.dispatch_gap_spike = True

        # --- Memory events ---
        for ev in self._memory_events:
            if use_timestamp_filter and not (lo <= ev.timestamp_ns <= hi):
                continue
            report.memory_events_in_window += 1
            if ev.event_type == "evict":
                report.evictions_in_window += 1
            elif ev.event_type == "bo_move":
                report.bo_moves_in_window += 1
                report.migration_bytes_in_window += ev.size_bytes

        # --- Race events ---
        for rev in self._race_events:
            rev_ts = rev.get("timestamp_ns", 0)
            if use_timestamp_filter and not (lo <= rev_ts <= hi):
                continue
            report.race_events_in_window += 1

        # --- DMA overlaps ---
        for dev in self._dma_overlaps:
            dev_ts = dev.get("timestamp_ns", 0)
            if use_timestamp_filter and not (lo <= dev_ts <= hi):
                continue
            report.dma_overlaps_in_window += 1

        # --- Collective races ---
        for cev in self._collective_races:
            cev_ts = cev.get("timestamp_ns", 0)
            if use_timestamp_filter and not (lo <= cev_ts <= hi):
                continue
            report.collective_races_in_window += 1

        report.diagnosis = self._diagnose(report)
        report.confidence = self._assess_confidence(report)

        return report

    @staticmethod
    def _diagnose(report: CorrelatedNaNReport) -> str:
        """Produce a human-readable root-cause hypothesis."""
        causes: List[str] = []

        if report.race_events_in_window > 0:
            causes.append("stream_race_detected")
        if report.dma_overlaps_in_window > 0:
            causes.append("h2d_dma_overlap")
        if report.collective_races_in_window > 0:
            causes.append("collective_compute_race")
        if report.evictions_in_window > 0:
            causes.append("memory_eviction")
        if report.dispatch_gap_spike:
            causes.append("dispatch_stall")
        if report.bo_moves_in_window > 10:
            causes.append("memory_thrashing")

        if not causes:
            if report.queue_events_in_window == 0 and report.memory_events_in_window == 0:
                return "no_kernel_events_in_window"
            return "no_clear_kernel_cause"

        return "+".join(causes)

    @staticmethod
    def _assess_confidence(report: CorrelatedNaNReport) -> str:
        """Rate confidence in the diagnosis."""
        score = 0
        if report.race_events_in_window > 0:
            score += 3
        if report.dma_overlaps_in_window > 0:
            score += 2
        if report.collective_races_in_window > 0:
            score += 2
        if report.evictions_in_window > 0:
            score += 1
        if report.dispatch_gap_spike:
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    def export_reports(self, reports: List[CorrelatedNaNReport], filepath: str | Path) -> None:
        """Write all correlated reports to a JSON file."""
        data = {
            "nan_count": len(reports),
            "window_ms": self._window_ms,
            "reports": [r.to_dict() for r in reports],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
