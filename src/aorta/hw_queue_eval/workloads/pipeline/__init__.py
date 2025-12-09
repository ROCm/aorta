"""
Pipeline workloads for hardware queue evaluation.

These workloads simulate data pipeline and memory offload patterns:
- Async data loading with GPU preprocessing
- ZeRO-offload memory management patterns
- torch.compile multi-region execution
"""

from aorta.hw_queue_eval.workloads.pipeline.async_dataload import AsyncDataLoadWorkload
from aorta.hw_queue_eval.workloads.pipeline.torch_compile import TorchCompileWorkload
from aorta.hw_queue_eval.workloads.pipeline.zero_offload import ZeROOffloadWorkload

__all__ = [
    "AsyncDataLoadWorkload",
    "ZeROOffloadWorkload",
    "TorchCompileWorkload",
]
