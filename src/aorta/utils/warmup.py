"""Distributed training warmup utilities.

This module provides functions to warm up RCCL communicators and training
collectives before the main training loop starts. These help avoid race
conditions in RCCL/RDMA during FSDP initialization.

Design Notes:
    - Training warmup is reduced to 1 step (from 3) to preserve timing
      variability while still exercising the collectives.
    - RCCL warmup in build_fsdp_model handles communicator initialization
      separately from training collectives warmup.
    - Set skip_training_warmup=True to maximize timing variability for
      race condition testing.
"""

import logging
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

log = logging.getLogger(__name__)


def warmup_rccl_communicators(
    shard_group: Optional[dist.ProcessGroup],
    replicate_group: Optional[dist.ProcessGroup],
    device: torch.device,
    num_warmup_ops: int = 5,
) -> None:
    """
    Warm up RCCL communicators with small operations before heavy FSDP usage.

    This ensures inter-node communicators are fully established before the
    _sync_params_and_buffers broadcasts. The race condition in RCCL/RoCE RDMA
    setup can cause hangs during FSDP initialization if broadcasts are issued
    before the communicators are ready.

    Args:
        shard_group: Intra-node shard process group (may be None)
        replicate_group: Inter-node replicate process group (may be None)
        device: GPU device to use for warmup tensors
        num_warmup_ops: Number of warmup operations to perform (default: 5)
    """
    rank = dist.get_rank()
    # Use a larger tensor for more thorough warmup
    warmup_tensor = torch.ones(8192, device=device, dtype=torch.float32)

    log.info("Starting RCCL communicator warmup with %d iterations (rank=%d)...", num_warmup_ops, rank)
    log.info("Warmup will test: all_reduce, broadcast, reduce_scatter, all_gather")

    # First, warmup the global world group
    log.info("Warming up global world group...")
    for i in range(num_warmup_ops):
        dist.all_reduce(warmup_tensor)
        dist.broadcast(warmup_tensor, src=0)
        torch.cuda.synchronize()

    dist.barrier()
    log.info("Global world group warmup complete")

    # Then warmup the shard and replicate groups
    # IMPORTANT: Must test ALL collective types that FSDP uses:
    # - reduce_scatter (FSDP backward gradient reduction)
    # - all_gather (FSDP forward parameter gathering)
    # - all_reduce (HYBRID_SHARD inter-node gradient sync)
    # - broadcast (parameter sync)
    if shard_group is None and replicate_group is None:
        log.warning(
            "No shard_group or replicate_group provided; only the global world "
            "process group will be warmed up. Subsequent iterations will run "
            "only barriers and CUDA synchronizations without group-specific "
            "collective operations (rank=%d).",
            rank,
        )
    for i in range(num_warmup_ops):
        # Warmup intra-node shard group
        if shard_group is not None:
            shard_size = dist.get_world_size(shard_group)

            # all_reduce (used in some FSDP operations)
            dist.all_reduce(warmup_tensor, group=shard_group)

            # reduce_scatter - CRITICAL: this is what FSDP backward uses!
            # Create input tensor that's shard_size times larger
            rs_input = torch.ones(8192 * shard_size, device=device, dtype=torch.float32)
            rs_output = torch.empty(8192, device=device, dtype=torch.float32)
            dist.reduce_scatter_tensor(rs_output, rs_input, group=shard_group)

            # all_gather - used in FSDP forward
            ag_input = torch.ones(8192, device=device, dtype=torch.float32)
            ag_output = torch.empty(8192 * shard_size, device=device, dtype=torch.float32)
            dist.all_gather_into_tensor(ag_output, ag_input, group=shard_group)

            # broadcast from first rank in shard group
            shard_ranks = dist.get_process_group_ranks(shard_group)
            dist.broadcast(warmup_tensor, src=shard_ranks[0], group=shard_group)

        # Warmup inter-node replicate group (this is where the race condition occurs)
        if replicate_group is not None:
            # Get the ranks in this replicate group and use the first one as source
            # Note: dist.get_process_group_ranks returns global ranks in the group
            group_ranks = dist.get_process_group_ranks(replicate_group)
            src_global_rank = group_ranks[0]  # First rank in the group
            dist.broadcast(warmup_tensor, src=src_global_rank, group=replicate_group)
            dist.all_reduce(warmup_tensor, group=replicate_group)

        # Synchronize GPU and global barrier between iterations
        torch.cuda.synchronize()
        dist.barrier()

    # Final synchronization with extra delay
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()

    log.info("RCCL communicator warmup complete (rank=%d)", rank)


def manual_sync_params(
    model: FSDP,
    replicate_group: Optional[dist.ProcessGroup],
) -> None:
    """
    Manually synchronize FSDP parameters from the first rank in each replicate group.

    This replaces the automatic sync_module_states with controlled synchronization
    to avoid race conditions in RCCL/RDMA during FSDP initialization. Parameters
    are broadcast from the first rank in each replicate group to ensure consistency.

    Args:
        model: The FSDP-wrapped model
        replicate_group: Inter-node replicate process group for broadcasting
    """
    rank = dist.get_rank()

    log.info("Starting manual parameter synchronization (rank=%d)...", rank)

    # Synchronize before param sync
    torch.cuda.synchronize()
    dist.barrier()

    # Determine the source rank for this replicate group
    # Each replicate group contains ranks with the same local_rank across nodes
    # e.g., group for local_rank 2: [2, 10, 18] - we broadcast from rank 2 (first in group)
    src_global_rank = None
    if replicate_group is not None:
        group_ranks = dist.get_process_group_ranks(replicate_group)
        src_global_rank = group_ranks[0]  # First rank in the group
        log.info("Manual sync: replicate group ranks=%s, src_rank=%d", group_ranks, src_global_rank)

    param_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_meta:
                log.debug("Skipping meta parameter: %s", name)
                continue

            # Broadcast from the first rank within this replicate group
            if replicate_group is not None and src_global_rank is not None:
                dist.broadcast(param.data, src=src_global_rank, group=replicate_group)

            param_count += 1

            # Periodic sync to prevent overwhelming the network
            if param_count % 10 == 0:
                torch.cuda.synchronize()

    # Final barrier to ensure all ranks complete
    torch.cuda.synchronize()
    dist.barrier()

    log.info("Manual parameter synchronization complete (rank=%d, params=%d)", rank, param_count)


def warmup_training_collectives(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    scaler: Optional[torch.cuda.amp.GradScaler],
    loss_fn: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    num_warmup_steps: int = 3,
) -> None:
    """
    Warm up training collectives by running dummy forward/backward/optimizer steps.

    This exercises all the collective operations used during training (all-gather,
    reduce-scatter, all-reduce) to ensure RCCL communicators are fully established
    before the main training loop starts.

    Args:
        model: The model (FSDP-wrapped)
        optimizer: The optimizer
        dataloader: Training dataloader
        device: GPU device
        autocast_dtype: Mixed precision dtype (or None)
        scaler: Gradient scaler for fp16 (or None)
        loss_fn: Loss function that takes (scores, batch) and returns loss tensor
        num_warmup_steps: Number of warmup steps to run
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Get an iterator from the dataloader
    data_iter = iter(dataloader)

    for warmup_step in range(num_warmup_steps):
        try:
            cpu_batch = next(data_iter)
        except StopIteration:
            # Restart iterator if dataloader is exhausted
            data_iter = iter(dataloader)
            cpu_batch = next(data_iter)

        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v
                 for k, v in cpu_batch.items()}
        torch.cuda.synchronize()

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                scores = model(batch)
                loss = loss_fn(scores, batch)
        else:
            scores = model(batch)
            loss = loss_fn(scores, batch)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Synchronize all ranks after each warmup step
        torch.cuda.synchronize()
        dist.barrier()

        log.debug("Warmup step %d complete (rank=%d, loss=%.4f)", warmup_step, rank, loss.item())

    # Reset optimizer state after warmup to not affect actual training
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()


__all__ = [
    "warmup_rccl_communicators",
    "manual_sync_params",
    "warmup_training_collectives",
]
