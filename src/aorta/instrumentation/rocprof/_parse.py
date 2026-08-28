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
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

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


def _iter_rows(stream: TextIO) -> Iterator[dict[str, str]]:
    """Stream an open CSV handle as dict rows, stopping on a read/parse error.

    A generator rather than a list: with ``stats: false`` a
    ``*_kernel_trace.csv`` carries one row per dispatch and can reach hundreds
    of MB, and materialising that before aggregating would add the whole file
    to the trial process's peak RSS. Both aggregators make a single forward
    pass, so no caller needs the rows twice.

    Fail-soft like the rest of the module: a truncated or undecodable file
    contributes the rows read so far and then stops, rather than raising. The
    caller owns the handle -- taking an already-open stream (rather than a path)
    lets the collector supply one opened ``O_NOFOLLOW`` under a directory fd, so
    a payload symlink swapped in after the guard cannot redirect the read.
    """
    try:
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


def _totals_from_stats(
    streams: Iterable[TextIO],
) -> tuple[dict[str, float], dict[str, int] | None]:
    """Aggregate ``*_kernel_stats.csv`` into per-kernel ns totals + call counts.

    Returns ``None`` for the counts when any otherwise-usable row had an
    unreadable ``Calls`` column. A stats row is aggregated **per kernel, not
    per dispatch**, so an unreadable count could stand for one dispatch or ten
    thousand; substituting 1 would publish a confident-looking
    ``rocprof_kernel_count`` that is simply wrong. Omitting the metric while
    keeping the timings is the fail-soft behaviour the module promises -- fewer
    metrics, never invented ones.
    """
    ns_by_kernel: dict[str, float] = {}
    calls_by_kernel: dict[str, int] = {}
    counts_trustworthy = True
    for stream in streams:
        for row in _iter_rows(stream):
            name = (row.get(_STATS_NAME) or "").strip()
            total_ns = _to_float(row.get(_STATS_TOTAL_NS))
            calls = _to_float(row.get(_STATS_CALLS))
            # A negative duration is as malformed as an unparseable one: no
            # dispatch takes less than no time, and letting it through would
            # silently subtract from the run's total.
            if not name or total_ns is None or total_ns < 0:
                continue
            ns_by_kernel[name] = ns_by_kernel.get(name, 0.0) + total_ns
            # A fractional count is as unreadable as a missing one: dispatches
            # are whole, and ``int()`` would silently truncate 1.9 to 1 and
            # publish it as measured.
            if calls is None or calls < 0 or not calls.is_integer():
                counts_trustworthy = False
                continue
            # A readable zero is taken at its word: ``rocprof_kernel_count``
            # claims dispatches, so rounding it up would over-count.
            calls_by_kernel[name] = calls_by_kernel.get(name, 0) + int(calls)
    return ns_by_kernel, (calls_by_kernel if counts_trustworthy else None)


def _totals_from_trace(
    streams: Iterable[TextIO],
) -> tuple[dict[str, float], dict[str, int] | None]:
    """Aggregate ``*_kernel_trace.csv`` dispatch rows into the same shape.

    Counts are always trustworthy here: a trace row *is* one dispatch, so
    there is no per-kernel aggregate to misread.
    """
    ns_by_kernel: dict[str, float] = {}
    calls_by_kernel: dict[str, int] = {}
    for stream in streams:
        for row in _iter_rows(stream):
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


def parse_summary_from_streams(
    artifact_dir: str,
    stats_streams: Iterable[TextIO],
    trace_streams: Iterable[TextIO],
) -> dict[str, Any]:
    """Summarise pre-opened ``rocprofv3`` CSV handles into trial metrics.

    The stream-taking core of :func:`parse_summary`. The caller supplies the
    ``*_kernel_stats.csv`` and ``*_kernel_trace.csv`` handles -- the collector
    path opens them ``O_NOFOLLOW`` under a directory fd so a payload symlink
    swapped in after the guard cannot redirect the read -- and the display
    string for ``rocprof_artifact_dir``. Same metrics contract as
    :func:`parse_summary`; never raises.

    Each group may be a **lazy** iterator that opens one file at a time (both
    callers pass one), so a multi-rank capture never needs a descriptor per
    artifact. Each is consumed at most once, and the trace group is not
    consumed at all when the stats group yielded data.
    """
    metrics: dict[str, Any] = {"rocprof_artifact_dir": artifact_dir}

    # The fallback keys off absence of *data*, not absence of files. A stats
    # CSV that exists but yields nothing -- rocprofv3 killed mid-write by a
    # trial timeout, or a future release renaming the columns pinned above --
    # would otherwise suppress a complete kernel trace sitting beside it, and
    # report no kernels on a host where profiling works. An empty stats group
    # aggregates to nothing and falls through the same way.
    ns_by_kernel, calls_by_kernel = _totals_from_stats(stats_streams)
    if not ns_by_kernel:
        ns_by_kernel, calls_by_kernel = _totals_from_trace(trace_streams)
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
    # Omitted rather than fabricated when a stats row's ``Calls`` was
    # unreadable; the timings above are still sound.
    if calls_by_kernel is not None:
        metrics["rocprof_kernel_count"] = sum(calls_by_kernel.values())
    metrics["rocprof_gpu_time_ms"] = total_ms
    metrics["rocprof_top_kernel_ms"] = top_ms
    metrics["rocprof_top_kernels"] = [name for name, _ in ranked[:TOP_N]]
    return metrics


def parse_summary(out_dir: Path | str) -> dict[str, Any]:
    """Summarise a ``rocprofv3`` output directory into trial metrics.

    Prefers the ``--stats`` CSVs (rocprofv3 already did the aggregation) and
    falls back to summing dispatch spans from the kernel trace whenever the
    stats CSVs are absent *or* yield no usable rows -- a truncated or
    column-renamed stats file must not mask a readable trace.

    Path-based convenience wrapper over :func:`parse_summary_from_streams`: it
    globs the two CSV families and opens them itself, one at a time -- a
    distributed run emits a CSV pair per rank and holding them all open at once
    could exceed ``RLIMIT_NOFILE``, which would have dropped the later ranks
    from the totals without saying so. The collector post-run path uses the
    stream entrypoint instead, opening each file ``O_NOFOLLOW`` under a
    directory fd so a payload symlink cannot redirect the read; this wrapper
    follows symlinks like any ``open`` and is for callers that already hold a
    trusted directory (tests, an operator's own ``rocprofv3 -d`` tree).

    Args:
        out_dir: The directory passed to ``rocprofv3 -d``.

    Returns:
        ``{}`` when the directory does not exist. Otherwise
        ``{"rocprof_artifact_dir": str}`` plus, when kernel data was found,
        the flat numeric ``rocprof_kernel_count`` / ``rocprof_gpu_time_ms`` /
        ``rocprof_top_kernel_ms`` and the non-numeric ``rocprof_top_kernels``
        name list. ``rocprof_kernel_count`` is omitted -- while the timings
        are still reported -- when a stats row's ``Calls`` column was
        unreadable, because a per-kernel aggregate row gives no basis for
        guessing how many dispatches it stood for. Never raises: an unreadable
        or malformed artifact tree degrades to the artifact-dir-only result.
    """
    root = Path(out_dir)
    try:
        if not root.is_dir():
            return {}
        stats_paths = sorted(root.rglob("*_kernel_stats.csv"))
        trace_paths = sorted(root.rglob("*_kernel_trace.csv"))
    except OSError:
        return {}
    return parse_summary_from_streams(
        str(root), _open_each(stats_paths), _open_each(trace_paths)
    )


def _open_each(paths: Sequence[Path]) -> Iterator[TextIO]:
    """Yield each CSV as an open handle, closing it before opening the next.

    One descriptor at a time: ``rocprofv3`` writes a stats/trace pair per
    process, so a large distributed capture holds more artifacts than the soft
    fd limit allows, and an ``EMFILE`` partway through a batch open would have
    left the remaining ranks out of the totals with nothing in the output to
    say so. A file that cannot be opened is skipped -- this wrapper promises
    never to raise.
    """
    for path in paths:
        try:
            handle = path.open(newline="", encoding="utf-8")
        except OSError:
            continue
        with handle:
            yield handle


__all__ = ["TOP_N", "parse_summary", "parse_summary_from_streams"]
