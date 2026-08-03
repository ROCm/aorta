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
    existing_baselines: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_baselines = existing_baselines or {}
    ngpu = nightly_eval.gpu_count()
    baselines: dict[str, Any] = {}
    incomplete: list[str] = []
    preserved: list[str] = []
    dropped: list[str] = []

    for entry in matrix_doc.get("entries") or []:
        name = entry["name"]
        min_gpus = int(entry.get("min_gpus", entry.get("nproc", 1)))
        if ngpu < min_gpus:
            # This runner physically can't exercise the entry (e.g. an 8-GPU
            # distributed variant on a 1-GPU box). Refusing the whole refresh
            # here would make baselines un-blessable on smaller runners, so we
            # instead CARRY OVER any baselines this entry already has (never
            # silently reverting a live gate to record-only). If it has none
            # yet, note it as still-unblessed rather than failing.
            carried = {k: v for k, v in existing_baselines.items()
                       if k.split("::", 1)[0] == name}
            if carried:
                baselines.update(carried)
                preserved.append(
                    f"{name} (needs {min_gpus} GPU(s), have {ngpu}; "
                    f"kept {len(carried)} existing baseline(s))"
                )
            else:
                dropped.append(f"{name} (needs {min_gpus} GPU(s), have {ngpu}; no existing baseline)")
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

    # Refuse a partial refresh caused by entries that RAN but didn't produce a
    # blessable pass (timeout / no matrix.json / empty cells / did-not-pass).
    # Regenerating from such a run would drop those gates from the (fully
    # replaced) baseline file, silently reverting those workloads to
    # record-only. Fail atomically before writing anything. Note: entries the
    # runner can't physically exercise (insufficient GPUs) are NOT fatal --
    # their existing baselines are carried over above.
    if incomplete:
        raise SystemExit(
            "refusing partial baseline refresh -- these entries/cells ran but did "
            "not produce a blessable pass:\n  - " + "\n  - ".join(incomplete)
        )

    for note in preserved:
        print(f"[refresh] carried over baseline(s): {note}", flush=True)
    for note in dropped:
        print(f"[refresh] WARNING still unblessed (insufficient GPUs, no prior baseline): {note}",
              flush=True)

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

    existing_baselines: dict[str, Any] = {}
    if args.out.exists():
        existing_doc = nightly_eval._load_yaml(args.out) or {}
        existing_baselines = existing_doc.get("baselines") or {}

    doc = build_baselines(
        matrix_doc, args.work_dir, args.step_time_margin, args.throughput_margin,
        args.perf_gate, existing_baselines,
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
