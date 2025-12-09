"""Utility helpers for the AORTA toolkit.

This module provides shared utilities for:
- Device discovery and GPU properties
- Stream creation and management
- GPU timing utilities
- Configuration loading
- Logging setup
"""

from .config import load_config, merge_cli_overrides
from .device import (
    # Constants
    BACKEND_NAME,
    IS_ROCM,
    # Type aliases
    Accelerator,
    # Classes
    DeviceProperties,
    # Original aorta functions
    detect_accelerator,
    ensure_gpu_available,
    get_device,
    # Extended device utilities
    get_device_properties,
    get_distributed_backend,
    get_memory_stats,
    get_rocm_env_info,
    reset_memory_stats,
)
from .logging import setup_logging
from .streams import (
    create_streams,
    cuda_stream_context,
    get_stream_id,
    sync_all_streams,
    sync_stream,
    warmup_gpu,
)
from .timing import (
    CPUTimer,
    EventTiming,
    StreamTimer,
    TimingContext,
)

__all__ = [
    # Config
    "load_config",
    "merge_cli_overrides",
    # Logging
    "setup_logging",
    # Device - constants
    "IS_ROCM",
    "BACKEND_NAME",
    "Accelerator",
    # Device - classes
    "DeviceProperties",
    # Device - functions (original aorta)
    "detect_accelerator",
    "get_device",
    "get_distributed_backend",
    # Device - functions (extended)
    "get_device_properties",
    "ensure_gpu_available",
    "get_memory_stats",
    "reset_memory_stats",
    "get_rocm_env_info",
    # Streams
    "create_streams",
    "sync_all_streams",
    "sync_stream",
    "cuda_stream_context",
    "get_stream_id",
    "warmup_gpu",
    # Timing
    "EventTiming",
    "TimingContext",
    "StreamTimer",
    "CPUTimer",
]
