"""Pure-logic helpers for the nightly evaluation harness.

Kept dependency-light (stdlib only) and side-effect free so it can be unit
tested on the CPU gate without torch / aorta / a GPU. The runner
(``nightly_eval.py``) and the baseline refresher (``refresh_baselines.py``)
import these functions; only those modules shell out to ``aorta`` / read the GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cell_key(entry_name: str, cell_name: str) -> str:
    """Baseline / results key for one (matrix entry, recipe cell)."""
    return f"{entry_name}::{cell_name}"


def cell_passed(cell: dict[str, Any]) -> bool:
    """A cell 'passed' iff it ran cleanly: no whole-cell error, no failing or
    erroring trials, and at least one trial actually passed.

    Mirrors aorta's own "clean cell" definition (``error is None and
    failed_count == 0 and error_count == 0``) with an explicit ``passed_count > 0``
    so a cell that never produced a valid observation is not counted as a pass.
    """
    return (
        cell.get("error") is None
        and int(cell.get("failed_count", 0) or 0) == 0
        and int(cell.get("error_count", 0) or 0) == 0
        and int(cell.get("passed_count", 0) or 0) > 0
    )


def extract_metrics(cell: dict[str, Any]) -> dict[str, Any]:
    """Pull the trend-worthy metrics out of a matrix.json cell entry.

    ``metrics_summary`` is ``{metric_name: {mean, ...}}`` (throughput-style
    metrics a workload reported, e.g. ``gflops`` / ``gbps``); we keep the mean.
    """
    metrics: dict[str, Any] = {
        "mean_step_time_ms": cell.get("mean_step_time_ms"),
        "mean_wall_clock_sec": cell.get("mean_wall_clock_sec"),
        "step_time_source": cell.get("step_time_source"),
    }
    throughput: dict[str, float] = {}
    for name, stats in (cell.get("metrics_summary") or {}).items():
        if isinstance(stats, dict) and stats.get("mean") is not None:
            throughput[name] = stats["mean"]
    metrics["throughput"] = throughput
    return metrics


def harvest_matrix_json(matrix_path: Path) -> list[dict[str, Any]]:
    """Parse a matrix.json into a list of per-cell harvest dicts."""
    doc = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    harvested: list[dict[str, Any]] = []
    for cell in doc.get("cells", []) or []:
        harvested.append(
            {
                "cell": cell.get("name"),
                "passed": cell_passed(cell),
                "error": cell.get("error"),
                "trials": cell.get("trials"),
                "passed_count": cell.get("passed_count"),
                "failed_count": cell.get("failed_count"),
                "error_count": cell.get("error_count", 0),
                "metrics": extract_metrics(cell),
            }
        )
    return harvested


def compare_to_baseline(
    harvested: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare one harvested cell against its blessed baseline (or record-only).

    Returns ``{verdict, reasons, deltas}`` where verdict is one of:
      * ``record`` -- no baseline yet; observed metrics recorded, treated as pass.
      * ``pass``   -- baseline present and all checks satisfied.
      * ``fail``   -- baseline present and at least one check failed.
    """
    if baseline is None:
        return {"verdict": "record", "reasons": ["no baseline (record-only)"], "deltas": {}}

    reasons: list[str] = []
    deltas: dict[str, Any] = {}
    metrics = harvested.get("metrics", {})

    # Correctness: require the cell to have passed if the baseline expects it.
    if baseline.get("passed", True) and not harvested.get("passed", False):
        err = harvested.get("error")
        reasons.append(
            f"expected passing cell but it did not pass"
            + (f" (error: {err})" if err else "")
        )

    # Step-time ceiling.
    st = baseline.get("step_time_ms") or {}
    st_max = st.get("max")
    observed_st = metrics.get("mean_step_time_ms")
    if st_max is not None and observed_st is not None:
        deltas["mean_step_time_ms"] = {"observed": observed_st, "max": st_max}
        if observed_st > st_max:
            reasons.append(f"mean_step_time_ms {observed_st:.3f} > max {st_max:.3f}")

    # Throughput floors.
    tp_baseline = baseline.get("throughput") or {}
    observed_tp = metrics.get("throughput") or {}
    for name, bounds in tp_baseline.items():
        floor = (bounds or {}).get("min")
        observed = observed_tp.get(name)
        if floor is None:
            continue
        deltas.setdefault("throughput", {})[name] = {"observed": observed, "min": floor}
        if observed is None:
            reasons.append(f"throughput '{name}' missing (expected >= {floor})")
        elif observed < floor:
            reasons.append(f"throughput '{name}' {observed:.3f} < min {floor:.3f}")

    return {
        "verdict": "fail" if reasons else "pass",
        "reasons": reasons,
        "deltas": deltas,
    }


def summarize(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Tally verdicts across all result entries."""
    summary = {"total": len(entries), "pass": 0, "fail": 0, "record": 0, "skip": 0}
    for e in entries:
        summary[e.get("verdict", "skip")] = summary.get(e.get("verdict", "skip"), 0) + 1
    return summary
