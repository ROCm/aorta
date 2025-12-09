"""Device discovery and GPU property utilities.

This module combines device detection for distributed training (original aorta)
with comprehensive GPU property queries (from hw_queue_eval).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch


Accelerator = Literal["nvidia", "amd", "cpu"]

log = logging.getLogger(__name__)

# Detect if running on ROCm or CUDA
IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
BACKEND_NAME = "ROCm/HIP" if IS_ROCM else "CUDA"


# =============================================================================
# Original aorta device utilities (for distributed training)
# =============================================================================


def detect_accelerator() -> Accelerator:
    """Detect the active accelerator type."""
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "amd"
    return "nvidia"


def _visible_device_indices() -> List[int]:
    """Return list of visible device indices from environment variables."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    if env:
        indices = [token.strip() for token in env.split(",") if token.strip()]
        try:
            return [int(token) for token in indices]
        except ValueError:
            log.warning("Non-integer device identifiers in *_VISIBLE_DEVICES=%s", env)
            return list(range(len(indices)))

    try:
        count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unable to query device count via torch: %s", exc)
        count = 0
    return list(range(count))


def get_device(local_rank: int) -> torch.device:
    """Return the torch.device for the current process.

    Args:
        local_rank: Local rank of the current process

    Returns:
        torch.device for the current process
    """
    accelerator = detect_accelerator()
    if accelerator == "cpu":
        return torch.device("cpu")

    visible = _visible_device_indices()
    if not visible:
        raise RuntimeError("CUDA/HIP backend reports zero visible devices")

    mapped_index = visible[local_rank % len(visible)]
    if local_rank >= len(visible):
        log.warning(
            "local_rank %s exceeds visible device count %s; remapping to device %s",
            local_rank,
            len(visible),
            mapped_index,
        )

    torch.cuda.set_device(mapped_index)
    return torch.device("cuda", mapped_index)


def get_distributed_backend() -> str:
    """Select the distributed backend based on accelerator."""
    accelerator = detect_accelerator()
    if accelerator == "cpu":
        return "gloo"
    return "nccl"


# =============================================================================
# Extended device utilities (from hw_queue_eval)
# =============================================================================


@dataclass
class DeviceProperties:
    """GPU device properties relevant to queue evaluation and benchmarking."""

    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    multi_processor_count: int
    max_threads_per_mp: int
    warp_size: int
    is_rocm: bool
    device_index: int

    # ROCm-specific properties (may be None on CUDA)
    gcn_arch: Optional[str] = None
    compute_units: Optional[int] = None

    def __str__(self) -> str:
        backend = "ROCm" if self.is_rocm else "CUDA"
        return (
            f"{self.name} ({backend})\n"
            f"  Memory: {self.total_memory_gb:.1f} GB\n"
            f"  Compute Units/SMs: {self.multi_processor_count}\n"
            f"  Warp/Wavefront Size: {self.warp_size}"
        )


def get_device_properties(device: str = "cuda:0") -> DeviceProperties:
    """Get properties of the specified GPU device.

    Args:
        device: Device string (e.g., "cuda:0", "cuda:1")

    Returns:
        DeviceProperties object with device information
    """
    device_obj = torch.device(device)
    device_idx = device_obj.index if device_obj.index is not None else 0

    props = torch.cuda.get_device_properties(device_idx)

    # Get compute capability
    if IS_ROCM:
        # ROCm reports capability differently
        compute_cap = (props.major, props.minor)
        gcn_arch = getattr(props, "gcnArchName", None)
    else:
        compute_cap = (props.major, props.minor)
        gcn_arch = None

    return DeviceProperties(
        name=props.name,
        compute_capability=compute_cap,
        total_memory_gb=props.total_memory / (1024**3),
        multi_processor_count=props.multi_processor_count,
        max_threads_per_mp=props.max_threads_per_multi_processor,
        warp_size=props.warp_size if hasattr(props, "warp_size") else 64 if IS_ROCM else 32,
        is_rocm=IS_ROCM,
        device_index=device_idx,
        gcn_arch=gcn_arch,
        compute_units=props.multi_processor_count if IS_ROCM else None,
    )


def ensure_gpu_available(device: str = "cuda:0") -> bool:
    """Check if GPU is available and accessible.

    Args:
        device: Device string to check

    Returns:
        True if GPU is available and accessible
    """
    if not torch.cuda.is_available():
        return False

    try:
        device_obj = torch.device(device)
        device_idx = device_obj.index if device_obj.index is not None else 0
        if device_idx >= torch.cuda.device_count():
            return False
        # Try to allocate a small tensor
        with torch.cuda.device(device_idx):
            _ = torch.empty(1, device=device)
        return True
    except Exception:
        return False


def get_memory_stats(device: str = "cuda:0") -> dict:
    """Get current GPU memory statistics.

    Args:
        device: Device to query

    Returns:
        Dictionary with memory statistics
    """
    device_idx = torch.device(device).index or 0

    return {
        "allocated_gb": torch.cuda.memory_allocated(device_idx) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device_idx) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device_idx) / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved(device_idx) / (1024**3),
    }


def reset_memory_stats(device: str = "cuda:0") -> None:
    """Reset memory statistics for device."""
    device_idx = torch.device(device).index or 0
    torch.cuda.reset_peak_memory_stats(device_idx)


def get_rocm_env_info() -> dict:
    """Get ROCm-specific environment information.

    Returns:
        Dictionary with ROCm environment details
    """
    info = {
        "is_rocm": IS_ROCM,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }

    if IS_ROCM:
        # ROCm-specific environment variables
        rocm_env_vars = [
            "ROCM_PATH",
            "HIP_VISIBLE_DEVICES",
            "AMD_LOG_LEVEL",
            "HSA_TOOLS_LIB",
            "GPU_MAX_HW_QUEUES",
            "HSA_ENABLE_SDMA",
        ]
        info["env_vars"] = {var: os.environ.get(var) for var in rocm_env_vars}

    return info


__all__ = [
    # Type aliases
    "Accelerator",
    # Constants
    "IS_ROCM",
    "BACKEND_NAME",
    # Original aorta functions
    "detect_accelerator",
    "get_device",
    "get_distributed_backend",
    # Extended device utilities
    "DeviceProperties",
    "get_device_properties",
    "ensure_gpu_available",
    "get_memory_stats",
    "reset_memory_stats",
    "get_rocm_env_info",
]
