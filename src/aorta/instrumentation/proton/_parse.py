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
from pathlib import Path
from typing import Any

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


def _count(metrics: dict[str, Any]) -> int:
    """Return a leaf's dispatch count, defaulting to 1 when unreadable.

    ``OverflowError`` is caught alongside the parse errors because it is what
    a huge integer count raises on the conversion to float, and ``int()`` of
    an infinity raises it too -- both would otherwise escape
    :func:`parse_summary` and break its never-raises contract.
    """
    try:
        parsed = float(metrics.get("count", 1))
    except (TypeError, ValueError, OverflowError):
        return 1
    if not math.isfinite(parsed) or parsed < 0:
        return 1
    return int(parsed)


def _walk(node: Any, by_name: dict[str, float], counts: dict[str, int]) -> None:
    """Accumulate leaf-frame time/count into ``by_name`` / ``counts``."""
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
    counts[name] = counts.get(name, 0) + _count(metrics)


def parse_profile(path: Path | str) -> tuple[dict[str, float], dict[str, int]]:
    """Aggregate one ``.hatchet`` file into per-kernel ms totals + call counts.

    Returns ``({}, {})`` when the file is missing, unreadable, or not a Proton
    tree profile.
    """
    by_name: dict[str, float] = {}
    counts: dict[str, int] = {}
    try:
        with Path(path).open(encoding="utf-8") as stream:
            database = json.load(stream)
    except (OSError, ValueError, UnicodeDecodeError):
        return by_name, counts
    roots = database if isinstance(database, list) else [database]
    for root in roots:
        _walk(root, by_name, counts)
    return by_name, counts


def parse_summary(out_dir: Path | str) -> dict[str, Any]:
    """Summarise a Proton output directory into trial metrics.

    Args:
        out_dir: The directory Proton's ``-n <out_dir>/<name>`` wrote into.

    Returns:
        ``{}`` when the directory does not exist. Otherwise
        ``{"proton_artifact_dir": str}`` plus, when a tree profile with timing
        was found, the flat numeric ``proton_kernel_count`` /
        ``proton_gpu_time_ms`` / ``proton_top_kernel_ms`` and the non-numeric
        ``proton_top_kernels`` name list.
    """
    root = Path(out_dir)
    try:
        if not root.is_dir():
            return {}
        profiles = sorted(root.rglob("*.hatchet"))
    except OSError:
        return {}

    metrics: dict[str, Any] = {"proton_artifact_dir": str(root)}
    by_name: dict[str, float] = {}
    counts: dict[str, int] = {}
    for profile in profiles:
        file_by_name, file_counts = parse_profile(profile)
        for name, elapsed_ms in file_by_name.items():
            by_name[name] = by_name.get(name, 0.0) + elapsed_ms
        for name, calls in file_counts.items():
            counts[name] = counts.get(name, 0) + calls
    if not by_name:
        return metrics

    ranked = sorted(by_name.items(), key=lambda item: item[1], reverse=True)
    metrics["proton_kernel_count"] = sum(counts.values())
    metrics["proton_gpu_time_ms"] = sum(by_name.values())
    metrics["proton_top_kernel_ms"] = ranked[0][1]
    metrics["proton_top_kernels"] = [name for name, _ in ranked[:TOP_N]]
    return metrics


__all__ = ["TOP_N", "parse_profile", "parse_summary"]
