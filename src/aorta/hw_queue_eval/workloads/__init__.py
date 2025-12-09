"""
Workload implementations for hardware queue evaluation.

Workload categories:
- distributed: FSDP, TP, MoE, gradient accumulation patterns
- inference: Speculative decoding, continuous batching, RAG pipelines
- pipeline: Async data loading, ZeRO offload, torch.compile patterns
- latency_sensitive: Heterogeneous kernels, graph subgraph execution
"""

from aorta.hw_queue_eval.workloads.base import BaseWorkload
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry, get_workload, list_workloads

__all__ = [
    "BaseWorkload",
    "WorkloadRegistry",
    "get_workload",
    "list_workloads",
]
