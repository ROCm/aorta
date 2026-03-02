"""GPU hardware control integration via Magpie.

Provides GPU power/frequency management for controlled benchmarking by
wrapping Magpie's GPUController / MultiGPUController.

Falls back gracefully when Magpie is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    from Magpie.utils.gpu import (
        GPUConfig,
        GPUController,
        GPUHardwareInfo,
        GPUVendor,
        MultiGPUConfig,
        MultiGPUController,
        detect_gpu,
    )

    HAS_MAGPIE = True
except ImportError:
    HAS_MAGPIE = False


@dataclass
class GPUControlConfig:
    """Configuration for GPU hardware control in aorta benchmarks.

    Attributes:
        enabled: Whether GPU control is active. Requires Magpie to be installed.
        power_limit_watts: GPU power cap in watts (None = unchanged).
        gpu_clock_level: AMD clock level 0-7 (None = unchanged).
        mem_clock_level: AMD memory clock level (None = unchanged).
        gpu_clock_mhz: GPU clock range (min, max) in MHz for NVIDIA (None = unchanged).
        mem_clock_mhz: Memory clock range (min, max) in MHz for NVIDIA (None = unchanged).
        reset_on_exit: Reset GPU settings after benchmark completes.
        device_ids: Specific GPU IDs to manage (None = all available).
    """

    enabled: bool = False
    power_limit_watts: Optional[int] = None
    gpu_clock_level: Optional[int] = None
    mem_clock_level: Optional[int] = None
    gpu_clock_mhz: Optional[Tuple[int, int]] = None
    mem_clock_mhz: Optional[Tuple[int, int]] = None
    reset_on_exit: bool = True
    device_ids: Optional[List[int]] = None


class GPUControlManager:
    """Manages GPU power/frequency state for deterministic benchmarking.

    Uses Magpie's GPUController under the hood. Designed as a context manager
    so GPU settings are automatically restored after the benchmark.

    Usage::

        mgr = GPUControlManager(config)
        with mgr:
            # GPU clocks are now locked
            run_benchmark()
        # GPU clocks restored to defaults
    """

    def __init__(self, config: GPUControlConfig) -> None:
        self.config = config
        self._controller: Any = None  # MultiGPUController when active
        self._applied = False

    @property
    def available(self) -> bool:
        """True if Magpie GPU control is available."""
        return HAS_MAGPIE and self.config.enabled

    def apply(self) -> Dict[str, Any]:
        """Apply GPU configuration and return hardware snapshot.

        Returns:
            Dictionary with pre-benchmark GPU hardware state for metadata.
        """
        if not self.available:
            if self.config.enabled and not HAS_MAGPIE:
                log.warning(
                    "GPU control requested but Magpie is not installed. "
                    "Install with: pip install magpie-eval"
                )
            return {}

        self._controller = MultiGPUController(device_ids=self.config.device_ids)

        gpu_cfg = GPUConfig(
            power_limit_watts=self.config.power_limit_watts,
            gpu_clock_level=self.config.gpu_clock_level,
            mem_clock_level=self.config.mem_clock_level,
            gpu_clock_mhz=self.config.gpu_clock_mhz,
            mem_clock_mhz=self.config.mem_clock_mhz,
        )

        multi_cfg = MultiGPUConfig(
            default_config=gpu_cfg,
            device_ids=self.config.device_ids,
            parallel=True,
        )

        results = self._controller.apply_config(multi_cfg)
        self._applied = True

        success_count = sum(1 for v in results.values() if v)
        total = len(results)
        log.info(
            "GPU control applied to %d/%d GPUs (power=%s W, gpu_clk_level=%s, mem_clk_level=%s)",
            success_count,
            total,
            self.config.power_limit_watts,
            self.config.gpu_clock_level,
            self.config.mem_clock_level,
        )

        return self._snapshot()

    def reset(self) -> None:
        """Reset GPUs to default settings."""
        if not self._applied or self._controller is None:
            return

        results = self._controller.reset_all()
        success_count = sum(1 for v in results.values() if v)
        log.info("GPU control reset on %d/%d GPUs", success_count, len(results))
        self._applied = False

    def _snapshot(self) -> Dict[str, Any]:
        """Capture current GPU hardware state for result metadata."""
        if self._controller is None:
            return {}

        try:
            infos = self._controller.get_all_hardware_info()
            snapshot = {}
            for dev_id, info in infos.items():
                snapshot[f"gpu_{dev_id}"] = info.to_dict()
            return {"gpu_hardware_state": snapshot}
        except Exception as e:
            log.debug("Failed to capture GPU hardware snapshot: %s", e)
            return {}

    def __enter__(self) -> "GPUControlManager":
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.config.reset_on_exit:
            self.reset()
        return None
