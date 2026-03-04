"""
Distributed training workloads for hardware queue evaluation.

These workloads simulate patterns from distributed training:
- Comm-compute overlap with configurable GEMM and collectives
- FSDP + Tensor Parallelism (3D parallelism)
- Mixture of Experts (MoE) with parallel expert execution
- Activation checkpointing with recomputation streams
- Gradient accumulation with early reduction
"""

from aorta.hw_queue_eval.workloads.distributed.activation_ckpt import ActivationCheckpointWorkload
from aorta.hw_queue_eval.workloads.distributed.comms_compute_overlap import CommsComputeOverlapWorkload
from aorta.hw_queue_eval.workloads.distributed.fsdp_tp import FSDPTPWorkload
from aorta.hw_queue_eval.workloads.distributed.grad_accum import GradientAccumulationWorkload
from aorta.hw_queue_eval.workloads.distributed.moe import MoEWorkload

__all__ = [
    "CommsComputeOverlapWorkload",
    "FSDPTPWorkload",
    "MoEWorkload",
    "ActivationCheckpointWorkload",
    "GradientAccumulationWorkload",
]
