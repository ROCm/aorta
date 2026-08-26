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


VALID_POLICIES = ("min", "max", "equal")

# Explicit comparison policy per metric name (allowlist). Unknown metrics are
# NEVER gated automatically -- they're captured for trends only. `equal` metrics
# are correctness checks (e.g. a logits checksum must match); `min`/`max` metrics
# are performance thresholds (throughput floor / latency ceiling).
_METRIC_POLICIES: dict[str, str] = {
    "gflops": "min",
    "gbps": "min",
    "tokens_per_sec": "min",
    "samples_per_sec": "min",
    "throughput": "min",
    "prefill_latency_ms": "max",
    "decode_latency_ms": "max",
    "latency_ms": "max",
    # Online-serving metrics, as `tokenspeed bench serve` spells them (the
    # `tokenspeed_serve` workload passes its export through verbatim).
    # Deliberately NOT aliased onto prefill/decode_latency_ms above: TTFT
    # includes queueing delay and TPOT is an inter-token average, so gating them
    # under those names would gate a differently-defined quantity. Medians and
    # p99s only -- means are the noisiest summary of a latency distribution and
    # the least useful thing to gate on.
    "median_ttft_ms": "max",
    "p99_ttft_ms": "max",
    "median_tpot_ms": "max",
    "p99_tpot_ms": "max",
    # ITL is gateable but read the value before blessing a bound on it: the
    # gateway delivers several tokens per SSE chunk, so most recorded gaps are
    # ~0 and `median_itl_ms` sits near zero while the real gaps land in the
    # tail. A relative margin around ~0 is noise, so a median-ITL gate will
    # flap. Prefer `median_tpot_ms` for per-token latency and treat
    # `p99_itl_ms` as the useful half of this pair.
    "median_itl_ms": "max",
    "p99_itl_ms": "max",
    "median_e2el_ms": "max",
    "p99_e2el_ms": "max",
    "output_throughput": "min",
    "total_token_throughput": "min",
    "request_throughput": "min",
    "logits_checksum": "equal",
    "output_checksum": "equal",
    "checksum": "equal",
}


def metric_policy(name: str) -> str | None:
    """Return the comparison policy (min/max/equal) for a metric, or None if the
    metric is not in the gating allowlist."""
    return _METRIC_POLICIES.get(name)


def is_correctness_metric(name: str) -> bool:
    """True for allowlisted metrics whose policy is exact-equality (checksums).

    These are correctness gates (a wrong-but-finite output must be caught), so
    they are blessed in the default baseline mode, not only under --perf-gate.
    """
    return _METRIC_POLICIES.get(name) == "equal"


def is_performance_metric(name: str) -> bool:
    """True for allowlisted min/max metrics (throughput floors, latency ceilings)."""
    return _METRIC_POLICIES.get(name) in ("min", "max")


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
    passed = harvested.get("passed", False)

    if baseline is None:
        # A cell that did not pass is a FAIL even without a baseline -- never
        # record-only. A passing unbaselined cell is recorded.
        if not passed:
            detail = harvested.get("error") or "cell failed or did not run"
            return {"verdict": "fail", "reasons": [str(detail)], "deltas": {}}
        return {"verdict": "record", "reasons": ["no baseline (record-only)"], "deltas": {}}

    reasons: list[str] = []
    deltas: dict[str, Any] = {}
    metrics = harvested.get("metrics", {})
    observed_summary = metrics.get("summary") or {}

    # Honor the baseline's expected outcome. Default is passed=True; an
    # expected-failure baseline (passed=False) inverts it.
    expected_pass = baseline.get("passed", True)
    if expected_pass and not passed:
        detail = harvested.get("error") or "cell did not pass"
        return {"verdict": "fail", "reasons": [f"expected pass but {detail}"], "deltas": {}}
    if not expected_pass and passed:
        return {"verdict": "fail",
                "reasons": ["expected failure but cell passed (stale expected-failure baseline?)"],
                "deltas": {}}
    if not passed:
        # Expected failure that did fail: as expected. Metrics from a failed cell
        # aren't meaningful, so don't check them.
        return {"verdict": "pass", "reasons": [], "deltas": {}}

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

    # Generic metric bounds with an EXPLICIT policy. Each baseline metric entry is
    # ``{policy: min|max|equal, value: X, required: bool}``; policy falls back to
    # the allowlist only. An invalid/unknown policy is a hard error, and a
    # required (default True) metric absent from the observation is a FAIL.
    for name, spec in (baseline.get("metrics") or {}).items():
        spec = spec or {}
        policy = spec.get("policy") or metric_policy(name)
        threshold = spec.get("value")
        required = spec.get("required", True)
        observed = observed_summary.get(name)
        deltas.setdefault("metrics", {})[name] = {
            "observed": observed, "policy": policy, "value": threshold,
        }
        if policy not in VALID_POLICIES:
            reasons.append(f"metric '{name}' has invalid/unknown policy {policy!r}")
            continue
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
