"""Fail-soft summary parsing of a Proton ``.hatchet`` profile.

A ``.hatchet`` file is JSON: a list whose first element is a nested tree of
frames (``{"frame": {"name": ...}, "metrics": {...}, "children": [...]}``) and
whose remaining elements are device metadata. Leaf frames carry the exclusive
per-kernel measurements, so the summary walks to the leaves and aggregates
those.

Like the rocprof parser this never raises: ``data: trace``, a crashed workload,
or a Proton build that emitted nothing all degrade to fewer metrics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

#: How many kernel names to carry in ``proton_top_kernels``.
TOP_N = 5

#: Multiplier from each supported ``time (<unit>)`` metric key to milliseconds.
_TIME_UNIT_TO_MS: dict[str, float] = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1e3,
}


def _time_ms(metrics: dict[str, Any]) -> float | None:
    """Return a node's exclusive GPU time in ms, or ``None`` if it has none.

    Proton names the metric ``time (<unit>)``. ``cpu_time (...)`` and the
    derived ``(inc)`` columns are deliberately not matched: the former is host
    time and the latter would double-count parents.

    A value that is unreadable, non-finite, or negative is skipped and the
    remaining metric keys are still inspected: ``NaN`` / infinity would
    propagate into ``proton_gpu_time_ms`` and be written to the trial JSON as
    a non-standard token, and no kernel runs for a negative time.
    """
    for key, value in metrics.items():
        head, _, tail = key.partition("(")
        if head.strip().lower() != "time":
            continue
        unit = tail.rstrip(")").strip().lower()
        factor = _TIME_UNIT_TO_MS.get(unit)
        if factor is None:
            continue
        try:
            elapsed_ms = float(value) * factor
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            continue
        return elapsed_ms
    return None


def _count(metrics: dict[str, Any]) -> int | None:
    """Return a leaf's dispatch count, or ``None`` when it is not readable.

    A Proton leaf aggregates every launch of that kernel -- the checked-in real
    fixture carries ``count: 6`` -- so a missing, non-finite, negative or
    otherwise malformed value gives no basis for assuming one dispatch.
    Returning ``None`` lets :func:`parse_summary` drop
    ``proton_kernel_count`` while still publishing the timings, rather than
    fabricating a count that reads as measured.

    ``OverflowError`` is caught alongside the parse errors because it is what
    a huge integer count raises on the conversion to float, and ``int()`` of
    an infinity raises it too -- both would otherwise escape
    :func:`parse_summary` and break its never-raises contract.
    """
    if "count" not in metrics:
        return None
    try:
        parsed = float(metrics["count"])
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return int(parsed)


def _walk(node: Any, by_name: dict[str, float], counts: dict[str, int | None]) -> None:
    """Accumulate leaf-frame time/count into ``by_name`` / ``counts``.

    A ``None`` in ``counts`` is sticky: once one leaf of a kernel had an
    unreadable count, the total for that name can no longer be trusted.
    """
    if not isinstance(node, dict):
        return
    children = node.get("children")
    children = children if isinstance(children, list) else []
    for child in children:
        _walk(child, by_name, counts)
    if children:
        return
    frame = node.get("frame")
    metrics = node.get("metrics")
    if not isinstance(frame, dict) or not isinstance(metrics, dict):
        return
    name = str(frame.get("name") or "").strip()
    elapsed_ms = _time_ms(metrics)
    if not name or elapsed_ms is None:
        return
    by_name[name] = by_name.get(name, 0.0) + elapsed_ms
    leaf_count = _count(metrics)
    previous = counts.get(name, 0)
    counts[name] = None if leaf_count is None or previous is None else previous + leaf_count


def parse_profile_stream(stream: TextIO) -> tuple[dict[str, float], dict[str, int | None]]:
    """Aggregate one open ``.hatchet`` handle into per-kernel ms totals + counts.

    The stream-taking core of :func:`parse_profile`. The caller owns the handle
    -- the collector path opens it ``O_NOFOLLOW`` under a directory fd so a
    payload symlink swapped in after the guard cannot redirect the read.
    Returns ``({}, {})`` when the handle is unreadable or not a Proton tree
    profile. A ``None`` count means that kernel's launch total was not readable,
    so it must not be summed into a published metric.
    """
    by_name: dict[str, float] = {}
    counts: dict[str, int | None] = {}
    try:
        database = json.load(stream)
    except (OSError, ValueError, UnicodeDecodeError):
        return by_name, counts
    roots = database if isinstance(database, list) else [database]
    for root in roots:
        _walk(root, by_name, counts)
    return by_name, counts


def parse_profile(path: Path | str) -> tuple[dict[str, float], dict[str, int | None]]:
    """Aggregate one ``.hatchet`` file into per-kernel ms totals + call counts.

    Path-based convenience wrapper over :func:`parse_profile_stream`. Returns
    ``({}, {})`` when the file is missing, unreadable, or not a Proton tree
    profile.
    """
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return parse_profile_stream(stream)
    except OSError:
        return {}, {}


def parse_summary_from_streams(
    artifact_dir: str, profile_streams: Iterable[TextIO]
) -> dict[str, Any]:
    """Summarise pre-opened ``.hatchet`` handles into trial metrics.

    The stream-taking core of :func:`parse_summary`. The caller supplies the
    profile handles -- the collector path opens them ``O_NOFOLLOW`` under a
    directory fd so a payload symlink swapped in after the guard cannot
    redirect the read -- and the display string for ``proton_artifact_dir``.
    Same metrics contract as :func:`parse_summary`; never raises.

    ``profile_streams`` may be a **lazy** iterator that opens one profile at a
    time (both callers pass one): each handle is aggregated and closed before
    the next opens, so a run with a profile per rank never needs a descriptor
    per rank.
    """
    metrics: dict[str, Any] = {"proton_artifact_dir": artifact_dir}
    by_name: dict[str, float] = {}
    counts: dict[str, int | None] = {}
    for stream in profile_streams:
        file_by_name, file_counts = parse_profile_stream(stream)
        for name, elapsed_ms in file_by_name.items():
            by_name[name] = by_name.get(name, 0.0) + elapsed_ms
        for name, calls in file_counts.items():
            previous = counts.get(name, 0)
            counts[name] = None if calls is None or previous is None else previous + calls
    if not by_name:
        return metrics

    ranked = sorted(by_name.items(), key=lambda item: item[1], reverse=True)
    total_ms = sum(by_name.values())
    top_ms = ranked[0][1]
    # The per-leaf finiteness check in ``_time_ms`` does not survive addition:
    # finite leaves still sum to infinity, across the tree or within one
    # kernel name. Degrade to the artifact directory rather than writing an
    # ``Infinity`` token into the trial JSON.
    if not (math.isfinite(total_ms) and math.isfinite(top_ms)):
        return metrics
    # Omitted rather than fabricated when any leaf's launch count was
    # unreadable; a Proton leaf aggregates launches, so there is no safe
    # substitute. The timings above stand on their own.
    if all(value is not None for value in counts.values()):
        metrics["proton_kernel_count"] = sum(counts.values())  # type: ignore[arg-type]
    metrics["proton_gpu_time_ms"] = total_ms
    metrics["proton_top_kernel_ms"] = top_ms
    metrics["proton_top_kernels"] = [name for name, _ in ranked[:TOP_N]]
    return metrics


def parse_summary(out_dir: Path | str) -> dict[str, Any]:
    """Summarise a Proton output directory into trial metrics.

    Path-based convenience wrapper over :func:`parse_summary_from_streams`: it
    globs the ``.hatchet`` profiles and opens them itself, one at a time -- a
    run can emit a profile per rank and holding them all open at once could
    exceed ``RLIMIT_NOFILE``, which would have dropped the later ranks from the
    totals without saying so. The collector post-run path uses the stream
    entrypoint instead, opening each file ``O_NOFOLLOW`` under a directory fd so
    a payload symlink cannot redirect the read; this wrapper follows symlinks
    like any ``open`` and is for callers that already hold a trusted directory
    (tests, an operator's own tree).

    Args:
        out_dir: The directory Proton's ``-n <out_dir>/<name>`` wrote into.

    Returns:
        ``{}`` when the directory does not exist. Otherwise
        ``{"proton_artifact_dir": str}`` plus, when a tree profile with timing
        was found, the flat numeric ``proton_kernel_count`` /
        ``proton_gpu_time_ms`` / ``proton_top_kernel_ms`` and the non-numeric
        ``proton_top_kernels`` name list. ``proton_kernel_count`` is omitted --
        while the timings are still reported -- when any leaf's launch count
        was unreadable, since a leaf aggregates launches and there is no safe
        substitute for the real number.
    """
    root = Path(out_dir)
    try:
        if not root.is_dir():
            return {}
        profiles = sorted(root.rglob("*.hatchet"))
    except OSError:
        return {}
    return parse_summary_from_streams(str(root), _open_each(profiles))


def _open_each(paths: Sequence[Path]) -> Iterator[TextIO]:
    """Yield each profile as an open handle, closing it before opening the next.

    One descriptor at a time: Proton writes a profile per rank, so a large
    distributed capture holds more artifacts than the soft fd limit allows, and
    an ``EMFILE`` partway through a batch open would have left the remaining
    ranks out of the totals with nothing in the output to say so. A file that
    cannot be opened is skipped -- this wrapper promises never to raise.
    """
    for path in paths:
        try:
            handle = path.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            yield handle


__all__ = ["TOP_N", "parse_profile", "parse_profile_stream", "parse_summary", "parse_summary_from_streams"]
