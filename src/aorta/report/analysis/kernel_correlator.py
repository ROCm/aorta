"""Correlate kernel-trace events with user-space training metrics.

Reads ``rank_XX_metrics.jsonl`` produced by the FSDP trainer (and any
other workload that emits the ``profile`` + ``kernel_trace`` schema) and
produces:

  - A per-iteration aggregate with kernel-event counts joined to the
    user-space loss / overlap data.
  - A simple "events preceding failure" view: for any iteration with
    ``loss == NaN`` or ``failure``, list kernel events from a configurable
    look-back window of preceding iterations.
  - A summary table suitable for direct printing or downstream rendering
    by the report generators.

The module avoids a hard dependency on pandas/numpy so it is usable from
the small base install (``pip install amd-aorta``); richer rendering lives
in ``aorta.report.generators.kernel_report``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Flattened view of a single ``rank_XX_metrics.jsonl`` line."""

    rank: int
    global_step: int
    epoch: int
    step: int
    loss: float
    overlap_ms: dict[str, float] = field(default_factory=dict)
    per_stream_ms: dict[str, float] = field(default_factory=dict)
    kernel_summary: dict[str, int] = field(default_factory=dict)
    kernel_event_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_nan(self) -> bool:
        return isinstance(self.loss, float) and math.isnan(self.loss)


@dataclass
class CorrelationFinding:
    """One iteration of interest plus the kernel events preceding it."""

    target: IterationRecord
    preceding_window: list[IterationRecord]
    kernel_event_total: dict[str, int]


class KernelEventCorrelator:
    """Join StreamProfiler overlap data with KernelTraceProfiler events."""

    def __init__(self, lookback_iterations: int = 5) -> None:
        self.lookback_iterations = lookback_iterations

    def load_metrics(self, path: Path) -> list[IterationRecord]:
        """Parse a single ``rank_XX_metrics.jsonl`` file."""
        records: list[IterationRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    log.warning("Skipping malformed line %d in %s: %s", line_no, path, exc)
                    continue
                records.append(_payload_to_record(payload))
        return records

    def load_metrics_glob(
        self, directory: Path, pattern: str = "rank_*_metrics.jsonl"
    ) -> list[IterationRecord]:
        """Load and concatenate every metrics file matching ``pattern``."""
        all_records: list[IterationRecord] = []
        for path in sorted(directory.glob(pattern)):
            all_records.extend(self.load_metrics(path))
        all_records.sort(key=lambda r: (r.global_step, r.rank))
        return all_records

    def find_failures(self, records: Iterable[IterationRecord]) -> list[CorrelationFinding]:
        """Return findings for each NaN iteration with preceding context."""
        sorted_records = sorted(records, key=lambda r: (r.rank, r.global_step))
        per_rank: dict[int, list[IterationRecord]] = {}
        for record in sorted_records:
            per_rank.setdefault(record.rank, []).append(record)

        findings: list[CorrelationFinding] = []
        for rank, rank_records in per_rank.items():
            for idx, record in enumerate(rank_records):
                if not record.is_nan:
                    continue
                start = max(0, idx - self.lookback_iterations)
                window = rank_records[start:idx]
                totals = _sum_kernel_summaries(window + [record])
                findings.append(
                    CorrelationFinding(
                        target=record,
                        preceding_window=window,
                        kernel_event_total=totals,
                    )
                )
            log.debug("Rank %d: %d findings", rank, sum(1 for r in rank_records if r.is_nan))
        return findings

    def summarise(self, records: Iterable[IterationRecord]) -> dict[str, Any]:
        """Aggregate kernel-event totals across all iterations."""
        records = list(records)
        totals = _sum_kernel_summaries(records)
        nan_iterations = [r for r in records if r.is_nan]
        return {
            "total_iterations": len(records),
            "nan_iterations": len(nan_iterations),
            "kernel_event_totals": totals,
            "ranks": sorted({r.rank for r in records}),
        }


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort ``int()`` that survives explicit JSON ``null`` values.

    ``dict.get(key, default)`` only returns ``default`` when the key is
    absent; if the key is present with a JSON ``null``, ``.get()``
    returns ``None`` and ``int(None)`` raises ``TypeError``. Older or
    partial trainer schemas can emit explicit nulls, so coerce
    defensively rather than dropping the whole row.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    """``_coerce_int`` counterpart for floats; same null-safety rationale."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int_dict(raw: Any) -> dict[str, int]:
    """Coerce a ``{name: count}`` dict, dropping any non-coercible value.

    A JSON ``"foo": null`` for a kernel-event count is treated as zero
    so the rest of the row is still usable.
    """
    if not isinstance(raw, dict):
        return {}
    return {str(k): _coerce_int(v) for k, v in raw.items()}


def _coerce_float_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): _coerce_float(v, 0.0) for k, v in raw.items()}


def _payload_to_record(payload: dict[str, Any]) -> IterationRecord:
    profile = payload.get("profile") or {}
    overlap = profile.get("overlap") or {}
    kernel_trace = payload.get("kernel_trace") or {}
    summary = kernel_trace.get("summary") or {}

    # ``kernel_trace.get("event_count")`` may be an explicit ``null``
    # (older/partial schema), in which case ``dict.get`` returns
    # ``None`` rather than the fallback. Compute the fallback length
    # eagerly so it is also used on the null branch.
    raw_event_count = kernel_trace.get("event_count")
    events_field = kernel_trace.get("events")
    fallback_event_count = len(events_field) if isinstance(events_field, list) else 0

    return IterationRecord(
        rank=_coerce_int(payload.get("rank")),
        global_step=_coerce_int(payload.get("global_step")),
        epoch=_coerce_int(payload.get("epoch")),
        step=_coerce_int(payload.get("step")),
        loss=_coerce_float(payload.get("loss"), float("nan")),
        overlap_ms=_coerce_float_dict(overlap.get("overlap_ms")),
        per_stream_ms=_coerce_float_dict(overlap.get("per_stream_ms")),
        kernel_summary=_coerce_int_dict(summary),
        kernel_event_count=_coerce_int(raw_event_count, fallback_event_count),
        raw=payload,
    )


def _sum_kernel_summaries(records: Iterable[IterationRecord]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        for key, value in record.kernel_summary.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def iter_findings_table(findings: Iterable[CorrelationFinding]) -> Iterator[dict[str, Any]]:
    """Flatten findings into rows suitable for tabular rendering."""
    for finding in findings:
        target = finding.target
        yield {
            "rank": target.rank,
            "global_step": target.global_step,
            "loss": target.loss,
            "lookback_iterations": len(finding.preceding_window),
            **{f"kernel_{k}": v for k, v in finding.kernel_event_total.items()},
        }


__all__ = [
    "CorrelationFinding",
    "IterationRecord",
    "KernelEventCorrelator",
    "iter_findings_table",
]
