"""
Device-side eBPF integration stub for future bpftime SPIR-V support.

This module is a placeholder for device-side eBPF profiling that will enable
per-warp, per-CU (Compute Unit) profiling inside GPU kernels.  Current
host-side profiling (rocprof, PyTorch profiler) only captures kernel-launch
granularity.  Device eBPF can reveal CU load imbalance, warp scheduling
patterns, and memory access locality.

Current status:
- bpftime (https://github.com/eunomia-bpf/bpftime) compiles eBPF to PTX
  for NVIDIA GPUs.  An AMD SPIR-V / GCN / CDNA backend is not yet available.
- The gpu_ext paper (arXiv:2512.12615) proposes SPIR-V as a portability
  path (Section 7).
- In the interim, ROCm's ``rocprof --att`` (Assembly Tracing Tool) provides
  per-instruction visibility on AMD GPUs.

Once the bpftime AMD backend is available, this module will implement:
- DeviceEBPFProfiler: attach points at kernel entry/exit and memory ops
- Per-CU utilization metrics
- Warp occupancy tracking
- Memory access heatmaps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DeviceEBPFConfig:
    """Configuration for device-side eBPF profiling (future)."""

    enabled: bool = False
    attach_points: List[str] = field(default_factory=lambda: ["kernel_entry", "kernel_exit"])
    sampling_rate: int = 1  # sample every Nth kernel launch
    collect_cu_utilization: bool = True
    collect_warp_occupancy: bool = True
    collect_memory_heatmap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "attach_points": self.attach_points,
            "sampling_rate": self.sampling_rate,
            "collect_cu_utilization": self.collect_cu_utilization,
            "collect_warp_occupancy": self.collect_warp_occupancy,
            "collect_memory_heatmap": self.collect_memory_heatmap,
        }


@dataclass
class DeviceEBPFMetrics:
    """Metrics from device-side eBPF profiling (future)."""

    per_cu_utilization: Dict[int, float] = field(default_factory=dict)
    warp_occupancy: float = 0.0
    avg_active_warps_per_cu: float = 0.0
    cu_imbalance_ratio: float = 0.0  # max/min utilization
    memory_access_pattern: str = ""  # "sequential", "strided", "random"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_cu_utilization": self.per_cu_utilization,
            "warp_occupancy": self.warp_occupancy,
            "avg_active_warps_per_cu": self.avg_active_warps_per_cu,
            "cu_imbalance_ratio": self.cu_imbalance_ratio,
            "memory_access_pattern": self.memory_access_pattern,
        }


class DeviceEBPFProfiler:
    """
    Device-side eBPF profiler stub.

    This class will wrap bpftime (or a future AMD equivalent) to run eBPF
    programs directly on the GPU device for per-CU, per-warp profiling.

    Currently raises NotImplementedError -- this is intentional as the
    underlying runtime support is not yet available for AMD GPUs.

    Interim alternative: use ``rocprof --att`` for per-instruction tracing::

        rocprof --att <kernel_name> -- python my_workload.py
    """

    def __init__(self, config: Optional[DeviceEBPFConfig] = None):
        self._config = config or DeviceEBPFConfig()

    def start(self) -> None:
        """Start device-side profiling."""
        raise NotImplementedError(
            "Device-side eBPF is not yet available for AMD GPUs. "
            "bpftime currently supports NVIDIA PTX only. "
            "Use 'rocprof --att' for per-instruction profiling on AMD."
        )

    def stop(self) -> DeviceEBPFMetrics:
        """Stop profiling and return metrics."""
        raise NotImplementedError(
            "Device-side eBPF is not yet available for AMD GPUs."
        )

    @staticmethod
    def is_available() -> bool:
        """Check if device-side eBPF is available on this system."""
        return False

    @staticmethod
    def rocprof_att_available() -> bool:
        """Check if rocprof --att (Assembly Tracing Tool) is available."""
        import shutil
        return shutil.which("rocprof") is not None
