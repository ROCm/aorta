"""DETECTOR VALIDATION ONLY -- this deliberately injects a divergence.

docs/llm-determinism.md, "Validate the detector actually flags real
divergence": a green smoke alone does not prove the detector works in your
environment, so the doc ships this perturbation -- bump one input token
between replay 1 and replay 2, which the snapshot/restore does not undo.

The verdict this produces is a property of the injection, NOT of the hardware
or the software stack. It exists to answer one question: on this node, does a
real divergence actually reach the probe's Tier-1 exit-code detector and the
agent loop? Anything it writes is harness evidence and must never be ingested
as a training scenario.
"""

from __future__ import annotations

import os
import sys

from aorta.workloads.llm_determinism import LlmDeterminismWorkload

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_launch import build_config, report  # noqa: E402


def main() -> int:
    rank = os.environ.get("RANK", "?")
    workload = LlmDeterminismWorkload(build_config())
    workload.setup()

    original = workload._run_once
    calls = [0]

    def patched():
        calls[0] += 1
        if calls[0] == 2:  # perturb after r1, before r2
            workload._input_ids[0, 0] = (
                int(workload._input_ids[0, 0]) + 1
            ) % workload._cfg.vocab_size
        return original()

    workload._run_once = patched

    try:
        result = workload.run()
    finally:
        workload.cleanup()

    report(result, rank, extra={"INJECTED_FAULT": True})
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
