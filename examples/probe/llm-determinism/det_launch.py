"""Standalone launcher for aorta's llm_determinism workload.

Shape follows docs/llm-determinism.md "Quick Start": construct the workload,
setup / run / cleanup, exit 0 on pass and 1 on divergence. The configuration
arrives as JSON in DET_CFG so one script covers every cell without editing
code between runs, and DET_CAPTURE_DIR (when set) overrides capture_dir so
each probe trial captures into its own directory.

Nothing here perturbs the workload. The same doc also describes a
*fault-injection* variant that monkeypatches _run_once to force divergence;
that lives in det_launch_injected.py and is used ONLY to validate that the
detector fires on this node. Its output is never a training scenario.
"""

from __future__ import annotations

import json
import os
import sys

from aorta.workloads.llm_determinism import LlmDeterminismWorkload


def build_config() -> dict:
    cfg = json.loads(os.environ["DET_CFG"])
    capture_dir = os.environ.get("DET_CAPTURE_DIR", "").strip()
    if capture_dir:
        cfg["capture_dir"] = capture_dir
    return cfg


def report(result, rank: str, extra: dict | None = None) -> None:
    metrics = result.metrics or {}
    print(
        f"[rank {rank}] passed={result.passed} "
        f"failures={result.failure_count} "
        f"elapsed={result.elapsed_sec:.2f}s "
        f"ranks_with_divergence={metrics.get('ranks_with_divergence')}",
        flush=True,
    )
    for reason in (metrics.get("divergence_reasons") or [])[:8]:
        print(f"[rank {rank}] reason: {reason}", flush=True)

    if rank in ("0", "?"):
        doc = {
            "passed": bool(result.passed),
            "failure_count": result.failure_count,
            "elapsed_sec": round(result.elapsed_sec, 3),
            "ranks_with_divergence": metrics.get("ranks_with_divergence"),
            "dtype": metrics.get("dtype"),
            "num_layers": metrics.get("num_layers"),
            "num_experts": metrics.get("num_experts"),
            "checksum_mode": metrics.get("checksum_mode"),
            "steps": len(metrics.get("steps") or []),
            "reasons": (metrics.get("divergence_reasons") or [])[:8],
            "disable_tf32": os.environ.get("DISABLE_TF32"),
            "hip_launch_blocking": os.environ.get("HIP_LAUNCH_BLOCKING"),
        }
        doc.update(extra or {})
        print("DET_JSON " + json.dumps(doc), flush=True)


def main() -> int:
    rank = os.environ.get("RANK", "?")
    workload = LlmDeterminismWorkload(build_config())
    workload.setup()
    try:
        result = workload.run()
    finally:
        workload.cleanup()
    report(result, rank)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
