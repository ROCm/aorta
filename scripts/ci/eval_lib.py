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


# Explicit comparison policy per metric name. Not every metric is a
# higher-is-better throughput number: latencies are upper-bounded, a checksum
# must match exactly. Unknown metrics default to "min" (treated as throughput)
# but that default is only used for trend capture, never to silently gate.
_METRIC_POLICIES: dict[str, str] = {
    "gflops": "min",
    "gbps": "min",
    "tokens_per_sec": "min",
    "samples_per_sec": "min",
    "throughput": "min",
    "prefill_latency_ms": "max",
    "decode_latency_ms": "max",
    "latency_ms": "max",
    "mean_step_time_ms": "max",
    "logits_checksum": "equal",
}


def metric_policy(name: str) -> str | None:
    """Return the comparison policy (min/max/equal) for a metric, or None."""
    return _METRIC_POLICIES.get(name)


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

    ``metrics_summary`` is ``{metric_name: {mean, ...}}`` for whatever the
    workload reported -- throughput, latency, checksums, counts, etc. We keep the
    mean under ``summary`` (a generic name->value map); the comparison policy for
    each is decided by ``_METRIC_POLICIES``, not by assuming everything is
    throughput.
    """
    metrics: dict[str, Any] = {
        "mean_step_time_ms": cell.get("mean_step_time_ms"),
        "mean_wall_clock_sec": cell.get("mean_wall_clock_sec"),
        "step_time_source": cell.get("step_time_source"),
    }
    summary: dict[str, float] = {}
    for name, stats in (cell.get("metrics_summary") or {}).items():
        if isinstance(stats, dict) and stats.get("mean") is not None:
            summary[name] = stats["mean"]
    metrics["summary"] = summary
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
    # BLOCKER fix: a cell that did not pass is a FAIL regardless of whether a
    # baseline exists -- never record-only. This must be checked before the
    # baseline-None short-circuit so a failed unbaselined workload can't slip
    # through as a benign "record".
    if not harvested.get("passed", False):
        detail = harvested.get("error") or "cell failed or did not run"
        return {"verdict": "fail", "reasons": [str(detail)], "deltas": {}}

    if baseline is None:
        return {"verdict": "record", "reasons": ["no baseline (record-only)"], "deltas": {}}

    reasons: list[str] = []
    deltas: dict[str, Any] = {}
    metrics = harvested.get("metrics", {})
    observed_summary = metrics.get("summary") or {}

    # Step-time ceiling. A configured max with a missing observation is a FAIL
    # (we cannot confirm the bound held), not a silent pass.
    st = baseline.get("step_time_ms") or {}
    st_max = st.get("max")
    observed_st = metrics.get("mean_step_time_ms")
    if st_max is not None:
        deltas["mean_step_time_ms"] = {"observed": observed_st, "max": st_max}
        if observed_st is None:
            reasons.append(f"mean_step_time_ms missing (expected <= {st_max:.3f})")
        elif observed_st > st_max:
            reasons.append(f"mean_step_time_ms {observed_st:.3f} > max {st_max:.3f}")

    # Generic metric bounds with explicit policy. Each baseline metric entry is
    # ``{policy: min|max|equal, value: X, required: bool}``; policy falls back to
    # _METRIC_POLICIES when omitted. A required (default True) metric that is
    # absent from the observation is a FAIL.
    for name, spec in (baseline.get("metrics") or {}).items():
        spec = spec or {}
        policy = spec.get("policy") or metric_policy(name) or "min"
        threshold = spec.get("value")
        required = spec.get("required", True)
        observed = observed_summary.get(name)
        deltas.setdefault("metrics", {})[name] = {
            "observed": observed, "policy": policy, "value": threshold,
        }
        if observed is None:
            if required:
                reasons.append(f"metric '{name}' missing (policy {policy} {threshold})")
            continue
        if threshold is None:
            continue
        if policy == "min" and observed < threshold:
            reasons.append(f"metric '{name}' {observed:.4g} < min {threshold:.4g}")
        elif policy == "max" and observed > threshold:
            reasons.append(f"metric '{name}' {observed:.4g} > max {threshold:.4g}")
        elif policy == "equal" and observed != threshold:
            reasons.append(f"metric '{name}' {observed} != expected {threshold}")

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
