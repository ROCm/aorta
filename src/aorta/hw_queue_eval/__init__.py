"""
GPU Hardware Queue Concurrency Evaluation Framework

A comprehensive framework for stress-testing AMD GPU hardware queue mapping
with workloads requiring >4 concurrent hardware queues.
"""

__version__ = "0.1.0"
__author__ = "AMD ROCm Team"

from aorta.hw_queue_eval.core.harness import HarnessConfig, HarnessResult, StreamHarness
from aorta.hw_queue_eval.core.metrics import LatencyMetrics, MetricsCollector, SwitchLatencyMetrics
from aorta.hw_queue_eval.workloads.base import BaseWorkload

__all__ = [
    "HarnessConfig",
    "HarnessResult",
    "StreamHarness",
    "LatencyMetrics",
    "MetricsCollector",
    "SwitchLatencyMetrics",
    "BaseWorkload",
    "__version__",
]
