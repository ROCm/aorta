"""``gpu_smoke`` workload: minimal single-process GPU sanity check.

Runs a trivial CUDA/HIP kernel (`x.add_(1.0)` over a small tensor) and verifies
the result, then reports a one-line pass/fail. It is **single-process**
(``launch_mode = "single_process"``, ``min_world_size = 1``) so it needs no
``torchrun`` — which makes it the smallest end-to-end workload that exercises a
real GPU through the triage path.

Primary purpose: a hardware-free **emulator / CI smoke test**. Run the whole
``aorta triage run`` under the mirage GPU emulator (rocjitsu) and this workload's
GPU kernel executes on the simulated device:

    mirage run --profile mi350x -- \
        aorta triage run --recipe recipes/gpu-smoke-emulated.yaml

It is also a useful "is the GPU usable at all?" probe on real hardware.

Config keys (all optional; ``steps`` is dispatcher-supplied):
* ``n``      -- element count (default 8).
* ``steps``  -- number of add iterations (default 1).
* ``dtype``  -- ``"float32"`` (default) / ``"float16"`` / ``"bfloat16"``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from aorta.workloads._base import Workload, WorkloadResult

log = logging.getLogger(__name__)


class GpuSmokeWorkload(Workload):
    """Single-process GPU smoke workload (trivial kernel + verification)."""

    launch_mode = "single_process"
    min_world_size = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._device = None
        self._dtype = None

    def setup(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("gpu_smoke: torch.cuda.is_available() is False")
        torch.cuda.set_device(0)
        self._device = torch.device("cuda:0")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        # Validate rather than silently defaulting an unknown/typo'd dtype to
        # float32 -- a silent fallback can mask a misconfigured run as green.
        dtype_name = str(self.config.get("dtype", "float32"))
        if dtype_name not in dtype_map:
            raise ValueError(
                f"gpu_smoke: unknown dtype {dtype_name!r}; "
                f"allowed: {sorted(dtype_map)}"
            )
        self._dtype = dtype_map[dtype_name]
        log.info(
            "gpu_smoke setup: device=%s name=%s",
            self._device,
            torch.cuda.get_device_name(0),
        )

    def run(self) -> WorkloadResult:
        import torch

        n = int(self.config.get("n", 8))
        # Explicit ``is None`` defaulting so an intentional ``steps: 0`` (a
        # degenerate but valid "allocate + verify, no adds" config) is honored
        # rather than silently treated as missing by a falsy-``0`` check.
        steps_cfg = self.config.get("steps")
        steps = 1 if steps_cfg is None else int(steps_cfg)
        if n <= 0:
            raise ValueError(f"gpu_smoke: n must be >= 1, got {n}")
        if steps < 0:
            raise ValueError(f"gpu_smoke: steps must be >= 0, got {steps}")
        t0 = time.perf_counter()

        x = torch.zeros(n, device=self._device, dtype=self._dtype)
        for _ in range(steps):
            x.add_(1.0)
        torch.cuda.synchronize()

        total = float(x.sum().item())
        expected = float(n * steps)
        # Tolerance-based compare: reduced-precision dtypes (float16/bfloat16)
        # and larger n*steps accumulate rounding error, so an exact ``==`` would
        # produce false FAILs. A small relative tolerance is sufficient for a
        # smoke test (we only care that the kernel ran and produced ~the right
        # value, not bit-exactness).
        passed = math.isclose(total, expected, rel_tol=1e-2, abs_tol=1e-3)
        elapsed = time.perf_counter() - t0

        if passed:
            log.info("gpu_smoke PASS: sum=%s expected=%s", total, expected)
        else:
            log.error("gpu_smoke FAIL: sum=%s expected=%s", total, expected)

        return WorkloadResult(
            passed=passed,
            failure_count=0 if passed else 1,
            first_failure_iteration=None if passed else 0,
            failure_details=[] if passed else [{"sum": total, "expected": expected}],
            total_iterations=steps,
            elapsed_sec=elapsed,
            main_work_started=True,
            executed_iterations=steps,
            configured_iterations=steps,
            metrics={"sum": total, "expected": expected, "n": n},
        )


__all__ = ["GpuSmokeWorkload"]
