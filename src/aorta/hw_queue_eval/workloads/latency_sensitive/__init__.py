"""
Latency-sensitive workloads for hardware queue evaluation.

These workloads are specifically designed to stress queue switch latency:
- Heterogeneous kernel mixing (tiny + large GEMMs)
- Independent subgraph execution patterns
"""

from aorta.hw_queue_eval.workloads.latency_sensitive.graph_subgraphs import GraphSubgraphsWorkload
from aorta.hw_queue_eval.workloads.latency_sensitive.hetero_kernels import HeterogeneousKernelWorkload

__all__ = [
    "HeterogeneousKernelWorkload",
    "GraphSubgraphsWorkload",
]
