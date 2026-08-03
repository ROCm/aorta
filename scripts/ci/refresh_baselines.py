#!/usr/bin/env python3
"""Regenerate candidate baselines from a live run on the runner.

Runs the nightly eval matrix, captures each passing cell's observed metrics, and
writes config/ci/regression_baselines.yaml with tolerance margins applied. The
refresh-baselines workflow runs this on the GPU runner and opens a PR so a human
reviews/blesses the numbers -- this script never commits.

Usage:
    python scripts/ci/refresh_baselines.py --step-time-margin 0.25 --throughput-margin 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_lib  # noqa: E402
import nightly_eval  # noqa: E402


def build_baselines(
    matrix_doc: dict[str, Any],
    out_dir: Path,
    step_time_margin: float,
    throughput_margin: float,
    perf_gate: bool,
) -> dict[str, Any]:
    ngpu = nightly_eval.gpu_count()
    baselines: dict[str, Any] = {}
    incomplete: list[str] = []

    for entry in matrix_doc.get("entries") or []:
        name = entry["name"]
        min_gpus = int(entry.get("min_gpus", entry.get("nproc", 1)))
        if ngpu < min_gpus:
            incomplete.append(f"{name} (skipped: needs {min_gpus} GPU(s), have {ngpu})")
            continue

        rc, matrix_path, timed_out = nightly_eval.run_entry(entry, out_dir)
        if timed_out:
            incomplete.append(f"{name} (timed out)")
            continue
        if matrix_path is None:
            incomplete.append(f"{name} (no matrix.json; exit {rc})")
            continue

        harvested_cells = eval_lib.harvest_matrix_json(matrix_path)
        if not harvested_cells:
            incomplete.append(f"{name} (matrix.json had no cells)")
            continue

        for harvested in harvested_cells:
            key = eval_lib.cell_key(name, harvested["cell"])
            if not harvested["passed"]:
                incomplete.append(f"{key} (did not pass)")
                continue

            spec: dict[str, Any] = {"passed": True}
            metrics = harvested["metrics"]
            summary = metrics.get("summary") or {}
            metric_specs: dict[str, Any] = {}

            # Correctness metrics (equal-policy checksums) are blessed in the
            # DEFAULT mode too -- a wrong-but-finite output must be caught even
            # without perf gating.
            for mname, value in summary.items():
                if value is not None and eval_lib.is_correctness_metric(mname):
                    metric_specs[mname] = {"policy": "equal", "value": value}

            # Performance thresholds (min/max) are opt-in via --perf-gate. Only
            # allowlisted metrics are gated; unknown metrics (step_time_p99,
            # final_loss, ...) are NEVER auto-gated as min.
            if perf_gate:
                st = metrics.get("mean_step_time_ms")
                if st is not None:
                    spec["step_time_ms"] = {"max": round(st * (1.0 + step_time_margin), 4)}
                for mname, value in summary.items():
                    if value is None or not eval_lib.is_performance_metric(mname):
                        continue
                    policy = eval_lib.metric_policy(mname)  # "min" or "max"
                    if policy == "min":
                        bound = round(value * (1.0 - throughput_margin), 4)
                    else:  # "max"
                        bound = round(value * (1.0 + step_time_margin), 4)
                    metric_specs[mname] = {"policy": policy, "value": bound}

            if metric_specs:
                spec["metrics"] = metric_specs
            baselines[key] = spec

    # MAJOR fix: refuse a partial refresh. Regenerating from a run where some
    # eligible entries/cells were skipped/failed/missing would drop their gates
    # from the (fully-replaced) baseline file; merging that PR silently reverts
    # those workloads to record-only. Fail atomically before writing anything.
    if incomplete:
        raise SystemExit(
            "refusing partial baseline refresh -- these entries/cells did not "
            "produce a blessable pass:\n  - " + "\n  - ".join(incomplete)
        )

    return {"version": 1, "baselines": baselines}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-time-margin", type=float, default=0.25,
                    help="fractional headroom above observed mean step time")
    ap.add_argument("--throughput-margin", type=float, default=0.15,
                    help="fractional floor below observed throughput")
    ap.add_argument("--work-dir", type=Path, default=nightly_eval.REPO_ROOT / ".refresh-baselines")
    ap.add_argument("--out", type=Path, default=nightly_eval.BASELINES)
    ap.add_argument("--perf-gate", action="store_true",
                    help="also emit step-time/throughput bounds (Phase 5 perf gating); "
                         "default is correctness-only baselines")
    args = ap.parse_args()

    matrix_doc = nightly_eval._load_yaml(nightly_eval.MATRIX)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    doc = build_baselines(
        matrix_doc, args.work_dir, args.step_time_margin, args.throughput_margin, args.perf_gate
    )

    header = (
        "# Expected-outcome baselines for the nightly evaluation.\n"
        "# Regenerated by scripts/ci/refresh_baselines.py; review the diff before merging.\n"
        "# Empty baselines => record-only (see scripts/ci/nightly_eval.py).\n"
    )
    args.out.write_text(header + yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(doc['baselines'])} baseline entries to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
