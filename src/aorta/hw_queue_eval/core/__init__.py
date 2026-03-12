"""
Core components for the hardware queue evaluation framework.

This module contains:
- StreamHarness: Parameterized test harness for running workloads
- MetricsCollector: Performance metrics collection and aggregation
- ROCmProfiler: Integration with ROCm profiling tools
- TorchProfilerWrapper: PyTorch profiler for Chrome/TensorBoard traces
- eBPF tracers: Kernel-level queue and memory tracing
- PolicyEvaluator: Scheduling/memory policy comparison framework
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
from aorta.hw_queue_eval.core.ebpf_tracer import (
    BPFQueueTracer,
    DriverQueueMetrics,
    EBPFCapabilities,
    check_ebpf_capabilities,
)
from aorta.hw_queue_eval.core.ebpf_memory_tracer import BPFMemoryTracer, MemoryTraceMetrics
from aorta.hw_queue_eval.core.policy_evaluator import (
    BUILTIN_POLICIES,
    PolicyComparison,
    PolicyConfig,
    PolicyEvaluator,
)
from aorta.hw_queue_eval.core.device_ebpf import DeviceEBPFConfig, DeviceEBPFProfiler
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
    # eBPF
    "BPFQueueTracer",
    "DriverQueueMetrics",
    "EBPFCapabilities",
    "check_ebpf_capabilities",
    "BPFMemoryTracer",
    "MemoryTraceMetrics",
    # Policy evaluation
    "BUILTIN_POLICIES",
    "PolicyComparison",
    "PolicyConfig",
    "PolicyEvaluator",
    # Device eBPF (stub)
    "DeviceEBPFConfig",
    "DeviceEBPFProfiler",
    # Utilities
    "create_streams",
    "get_device_properties",
    "sync_all_streams",
    "TimingContext",
]
