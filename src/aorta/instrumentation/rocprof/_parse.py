"""Fail-soft summary parsing of a ``rocprofv3`` output directory.

Two verified behaviours drive the shape of this module:

* Where ``rocprofv3`` puts its CSVs under the ``-d`` directory depends on
  whether ``-o`` was passed: with it they are flat
  (``<out_dir>/<basename>_kernel_stats.csv``), without it they nest under a
  hostname directory (``<out_dir>/<hostname>/<pid>_kernel_stats.csv``). Every
  lookup is therefore a recursive glob rather than a fixed path, so an
  operator's own ``rocprofv3 -d`` tree parses as well as aorta's.
* When the profiled command performs no GPU work, ``rocprofv3`` writes **no
  files at all**. That is a legitimate outcome (e.g. a probe of ``/bin/echo``),
  not a failure, so a missing/empty/malformed artifact tree yields fewer
  metrics rather than an exception.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: How many kernel names to carry in the non-numeric ``rocprof_top_kernels``
#: channel. Kept small: it is a triage breadcrumb, not a report.
TOP_N = 5

_NS_PER_MS = 1_000_000.0

# rocprofv3 ``--stats`` column names (rocprofiler-sdk 1.x).
_STATS_NAME = "Name"
_STATS_CALLS = "Calls"
_STATS_TOTAL_NS = "TotalDurationNs"

# rocprofv3 ``--kernel-trace`` column names.
_TRACE_KIND = "Kind"
_TRACE_NAME = "Kernel_Name"
_TRACE_START = "Start_Timestamp"
_TRACE_END = "End_Timestamp"
_KERNEL_DISPATCH = "KERNEL_DISPATCH"


def _iter_rows(path: Path) -> Iterator[dict[str, str]]:
    """Stream a CSV as dict rows, yielding nothing more on a read/parse error.

    A generator rather than a list: with ``stats: false`` a
    ``*_kernel_trace.csv`` carries one row per dispatch and can reach hundreds
    of MB, and materialising that before aggregating would add the whole file
    to the trial process's peak RSS. Both aggregators make a single forward
    pass, so no caller needs the rows twice.

    Fail-soft like the rest of the module: a truncated or undecodable file
    contributes the rows read so far and then stops, rather than raising.
    """
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                yield dict(row)
    except (OSError, csv.Error, UnicodeDecodeError):
        return


def _to_float(value: Any) -> float | None:
    """Parse a CSV cell into a *finite* float, or ``None`` if it is not one.

    ``float()`` accepts ``"NaN"`` / ``"Infinity"``, and a metric built from
    those is both meaningless and unserialisable as strict JSON --
    :func:`json.dump` writes the non-standard ``NaN`` / ``Infinity`` tokens,
    which a downstream reader is entitled to reject. A row carrying one is
    dropped like any other malformed row.
    """
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _totals_from_stats(paths: list[Path]) -> tuple[dict[str, float], dict[str, int]]:
    """Aggregate ``*_kernel_stats.csv`` into per-kernel ns totals + call counts."""
    ns_by_kernel: dict[str, float] = {}
    calls_by_kernel: dict[str, int] = {}
    for path in paths:
        for row in _iter_rows(path):
            name = (row.get(_STATS_NAME) or "").strip()
            total_ns = _to_float(row.get(_STATS_TOTAL_NS))
            calls = _to_float(row.get(_STATS_CALLS))
            # A negative duration is as malformed as an unparseable one: no
            # dispatch takes less than no time, and letting it through would
            # silently subtract from the run's total.
            if not name or total_ns is None or total_ns < 0:
                continue
            ns_by_kernel[name] = ns_by_kernel.get(name, 0.0) + total_ns
            # A row with no readable ``Calls`` column still evidences one
            # dispatch, so it counts as one. A readable zero is taken at its
            # word: ``rocprof_kernel_count`` claims dispatches, so inventing
            # one would over-count.
            calls_by_kernel[name] = calls_by_kernel.get(name, 0) + (
                1 if calls is None or calls < 0 else int(calls)
            )
    return ns_by_kernel, calls_by_kernel


def _totals_from_trace(paths: list[Path]) -> tuple[dict[str, float], dict[str, int]]:
    """Aggregate ``*_kernel_trace.csv`` dispatch rows into the same shape."""
    ns_by_kernel: dict[str, float] = {}
    calls_by_kernel: dict[str, int] = {}
    for path in paths:
        for row in _iter_rows(path):
            kind = (row.get(_TRACE_KIND) or "").strip()
            if kind and kind != _KERNEL_DISPATCH:
                continue
            name = (row.get(_TRACE_NAME) or "").strip()
            start = _to_float(row.get(_TRACE_START))
            end = _to_float(row.get(_TRACE_END))
            if not name or start is None or end is None or end < start:
                continue
            ns_by_kernel[name] = ns_by_kernel.get(name, 0.0) + (end - start)
            calls_by_kernel[name] = calls_by_kernel.get(name, 0) + 1
    return ns_by_kernel, calls_by_kernel


def parse_summary(out_dir: Path | str) -> dict[str, Any]:
    """Summarise a ``rocprofv3`` output directory into trial metrics.

    Prefers the ``--stats`` CSVs (rocprofv3 already did the aggregation) and
    falls back to summing dispatch spans from the kernel trace whenever the
    stats CSVs are absent *or* yield no usable rows -- a truncated or
    column-renamed stats file must not mask a readable trace.

    Args:
        out_dir: The directory passed to ``rocprofv3 -d``.

    Returns:
        ``{}`` when the directory does not exist. Otherwise
        ``{"rocprof_artifact_dir": str}`` plus, when kernel data was found,
        the flat numeric ``rocprof_kernel_count`` / ``rocprof_gpu_time_ms`` /
        ``rocprof_top_kernel_ms`` and the non-numeric ``rocprof_top_kernels``
        name list. Never raises: an unreadable or malformed artifact tree
        degrades to the artifact-dir-only result.
    """
    root = Path(out_dir)
    try:
        if not root.is_dir():
            return {}
        stats_paths = sorted(root.rglob("*_kernel_stats.csv"))
        trace_paths = sorted(root.rglob("*_kernel_trace.csv"))
    except OSError:
        return {}

    metrics: dict[str, Any] = {"rocprof_artifact_dir": str(root)}

    # The fallback keys off absence of *data*, not absence of files. A stats
    # CSV that exists but yields nothing -- rocprofv3 killed mid-write by a
    # trial timeout, or a future release renaming the columns pinned above --
    # would otherwise suppress a complete kernel trace sitting beside it, and
    # report no kernels on a host where profiling works.
    ns_by_kernel: dict[str, float] = {}
    calls_by_kernel: dict[str, int] = {}
    if stats_paths:
        ns_by_kernel, calls_by_kernel = _totals_from_stats(stats_paths)
    if not ns_by_kernel and trace_paths:
        ns_by_kernel, calls_by_kernel = _totals_from_trace(trace_paths)
    if not ns_by_kernel:
        return metrics

    ranked = sorted(ns_by_kernel.items(), key=lambda item: item[1], reverse=True)
    total_ms = sum(ns_by_kernel.values()) / _NS_PER_MS
    top_ms = ranked[0][1] / _NS_PER_MS
    # Per-row finiteness is not enough: finite rows still sum to infinity
    # (1e308 + 1e308), in the aggregate or within one kernel's accumulator.
    # The metrics channel promises strictly serialisable JSON, so a total that
    # overflowed degrades to the artifact directory like any other unusable
    # capture rather than publishing an ``Infinity`` token.
    if not (math.isfinite(total_ms) and math.isfinite(top_ms)):
        return metrics
    metrics["rocprof_kernel_count"] = sum(calls_by_kernel.values())
    metrics["rocprof_gpu_time_ms"] = total_ms
    metrics["rocprof_top_kernel_ms"] = top_ms
    metrics["rocprof_top_kernels"] = [name for name, _ in ranked[:TOP_N]]
    return metrics


__all__ = ["TOP_N", "parse_summary"]
