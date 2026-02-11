"""Utility helpers for the AORTA toolkit.

This module provides shared utilities for:
- Device discovery and GPU properties
- Stream creation and management
- GPU timing utilities
- Configuration loading
- Logging setup
- Distributed training warmup
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
    get_driver_info,
    get_memory_stats,
    get_rocm_env_info,
    get_system_info,
    log_environment_info,
    reset_memory_stats,
)
from .logging import setup_logging
from .streams import (
    create_multi_gpu_streams,
    create_streams,
    create_streams_per_device,
    cuda_stream_context,
    get_available_devices,
    get_stream_id,
    sync_all_streams,
    sync_stream,
    warmup_all_gpus,
    warmup_gpu,
)
from .timing import (
    CPUTimer,
    EventTiming,
    StreamTimer,
    TimingContext,
)
from .warmup import (
    manual_sync_params,
    warmup_rccl_communicators,
    warmup_training_collectives,
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
    "get_driver_info",
    "get_system_info",
    "log_environment_info",
    # Streams
    "create_streams",
    "create_multi_gpu_streams",
    "create_streams_per_device",
    "get_available_devices",
    "sync_all_streams",
    "sync_stream",
    "cuda_stream_context",
    "get_stream_id",
    "warmup_gpu",
    "warmup_all_gpus",
    # Timing
    "EventTiming",
    "TimingContext",
    "StreamTimer",
    "CPUTimer",
    # Warmup
    "warmup_rccl_communicators",
    "manual_sync_params",
    "warmup_training_collectives",
]
