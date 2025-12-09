"""
Core components for the hardware queue evaluation framework.

This module contains:
- StreamHarness: Parameterized test harness for running workloads
- MetricsCollector: Performance metrics collection and aggregation
- ROCmProfiler: Integration with ROCm profiling tools
- TorchProfilerWrapper: PyTorch profiler for Chrome/TensorBoard traces
- Utility functions for stream management and timing
"""

from aorta.hw_queue_eval.core.harness import HarnessConfig, HarnessResult, StreamHarness
from aorta.hw_queue_eval.core.metrics import LatencyMetrics, MetricsCollector, SwitchLatencyMetrics
from aorta.hw_queue_eval.core.profiler import ROCmProfiler
from aorta.hw_queue_eval.core.torch_profiler import (
    ProfilerConfig,
    ProfilerResult,
    TorchProfilerWrapper,
    generate_profile_summary,
    profile_workload_run,
)
from aorta.utils import (
    create_streams,
    get_device_properties,
    sync_all_streams,
    TimingContext,
)

__all__ = [
    "HarnessConfig",
    "HarnessResult",
    "StreamHarness",
    "LatencyMetrics",
    "MetricsCollector",
    "SwitchLatencyMetrics",
    "ROCmProfiler",
    "ProfilerConfig",
    "ProfilerResult",
    "TorchProfilerWrapper",
    "generate_profile_summary",
    "profile_workload_run",
    "create_streams",
    "get_device_properties",
    "sync_all_streams",
    "TimingContext",
]
