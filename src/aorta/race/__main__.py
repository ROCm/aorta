"""
CLI entry point for the RCCL race condition reproducer.

Supports multiple workload modes via --mode:
  - default: TorchRec-like pattern (H2D + all_to_all + all_reduce)
  - ddp:     DDP pattern (H2D prefetch + gradient all_reduce)
  - minimal: Legacy monolithic reproducer (backward compat)

Usage:
    # Default mode (TorchRec-like)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 -m aorta.race \
        --warmup 100 --verify 10000

    # DDP mode (gradient all_reduce + H2D prefetch)
    torchrun --nproc_per_node=8 -m aorta.race --mode ddp \
        --warmup 100 --verify 10000

    # Same-stream mode (definitive runtime bug test)
    torchrun --nproc_per_node=8 -m aorta.race --same-stream

    # Reduced parallelism (comparison baseline)
    GPU_MAX_HW_QUEUES=2 torchrun --nproc_per_node=8 -m aorta.race

Environment variables to test:
    GPU_MAX_HW_QUEUES=4          # Full parallelism (use 2 to reduce)
    ROC_SIGNAL_POOL_SIZE=16384   # HSA signal pool size
    HSA_ENABLE_SDMA=0            # Disable SDMA engine
    GPU_FORCE_BLIT_COPY_SIZE=128 # Force blit copy threshold
    NCCL_LAUNCH_ORDER_IMPLICIT=1 # Serialize NCCL operations
    RCCL_GFX9_CHEAP_FENCE_OFF=1  # Disable fence optimization
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

from aorta.race.config import ReproducerConfig, ReproducerResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RCCL race condition reproducer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="default",
        choices=["default", "ddp", "fsdp"],
        help=(
            "Workload mode. "
            "default: TorchRec-like (H2D + all_to_all + all_reduce). "
            "ddp: DDP (H2D prefetch + gradient all_reduce). "
            "fsdp: FSDP (per-layer all_gather + reduce_scatter). "
            "Default: default"
        ),
    )

    # Iteration settings
    parser.add_argument(
        "--warmup", type=int, default=100,
        help="Number of warmup iterations (no verification). Default: 100"
    )
    parser.add_argument(
        "--verify", type=int, default=10000,
        help="Number of verification iterations. Default: 10000"
    )
    parser.add_argument(
        "--no-stop-on-first", action="store_true",
        help="Don't stop on first corruption (continue counting)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="Log progress every N iterations. Default: 100"
    )

    # Tensor sizes
    parser.add_argument(
        "--h2d-size", type=int, default=1_000_000,
        help="H2D tensor size (elements). Default: 1M"
    )
    parser.add_argument(
        "--a2a-size", type=int, default=100_000,
        help="all_to_all tensor size per rank (elements). Default: 100K"
    )
    parser.add_argument(
        "--ar-size", type=int, default=100_000,
        help="all_reduce tensor size (elements). Default: 100K"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type. Default: bfloat16"
    )

    # Compute simulation (enabled by default, use --no-compute to disable)
    parser.add_argument(
        "--no-compute", action="store_true",
        help="Disable compute simulation (fast but may not trigger bug)"
    )
    parser.add_argument(
        "--gemm-size", type=int, default=5120,
        help="GEMM matrix size. Default: 5120 (~500ms/step on MI300X)"
    )
    parser.add_argument(
        "--gemm-layers", type=int, default=26,
        help="Number of GEMM layers. Default: 26 (~500ms/step on MI300X)"
    )
    parser.add_argument(
        "--no-backward", action="store_true",
        help="Skip backward pass simulation"
    )
    parser.add_argument(
        "--compute-type", type=str, default="gemm",
        choices=["gemm", "transformer"],
        help="Compute pattern type. gemm: stacked matrix multiplications. "
             "transformer: self-attention + FFN layers. Default: gemm"
    )

    # Transformer compute settings
    parser.add_argument(
        "--model-dim", type=int, default=2048,
        help="Transformer hidden dim (d_model). Default: 2048"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4,
        help="Number of transformer layers. Default: 4"
    )

    # FSDP-specific
    parser.add_argument(
        "--fsdp-shard-size", type=int, default=100_000,
        help="FSDP shard size per rank (elements). Default: 100K"
    )

    # H2D buffering strategy
    parser.add_argument(
        "--prefetch", action="store_true",
        help="Use double-buffered H2D prefetch (overlap next batch with backward)"
    )

    # Stream configuration
    parser.add_argument(
        "--same-stream", action="store_true",
        help="Put H2D and datadist on same stream (definitive runtime bug test)"
    )

    # Optimizer (for modes that support it)
    parser.add_argument(
        "--optimizer", type=str, default="none",
        choices=["none", "adamw", "sgd", "shampoo"],
        help="Optimizer for weight updates (used by DDP mode). Default: none"
    )

    # DDP-specific
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Enable deterministic mode (required for DDP gradient verification)"
    )
    parser.add_argument(
        "--bucketed", action="store_true",
        help="Use bucketed per-layer gradient all_reduce overlapping with backward (DDP mode)"
    )

    # Hardware settings (GPU_MAX_HW_QUEUES)
    parser.add_argument(
        "--hw-queues", type=int, default=None,
        help="Set GPU_MAX_HW_QUEUES (4 exposes bug, 2 masks it)"
    )

    # ==========================================================================
    # Environment variable flags
    # ==========================================================================

    parser.add_argument(
        "--signal-pool-size", type=int, default=None,
        help="Set ROC_SIGNAL_POOL_SIZE (default 64)"
    )
    parser.add_argument(
        "--disable-sdma", action="store_true",
        help="Set HSA_ENABLE_SDMA=0 (disable SDMA engine)"
    )
    parser.add_argument(
        "--blit-copy-size", type=int, default=None,
        help="Set GPU_FORCE_BLIT_COPY_SIZE threshold"
    )
    parser.add_argument(
        "--nccl-implicit-order", action="store_true",
        help="Set NCCL_LAUNCH_ORDER_IMPLICIT=1 (serializes NCCL ops)"
    )
    parser.add_argument(
        "--disable-cheap-fence", action="store_true",
        help="Set RCCL_GFX9_CHEAP_FENCE_OFF=1 and RCCL_GFX942_CHEAP_FENCE_OFF=1"
    )
    parser.add_argument(
        "--disable-clr-batch", action="store_true",
        help="Set DEBUG_CLR_BATCH_CPU_SYNC_SIZE=0 (disable CLR batching)"
    )

    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    """Initialize distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return rank, world_size


def apply_env_vars(args: argparse.Namespace) -> dict[str, str]:
    """Apply environment variables from CLI flags.

    Returns dict of variables that were set for logging.
    """
    applied = {}

    if args.hw_queues is not None:
        os.environ["GPU_MAX_HW_QUEUES"] = str(args.hw_queues)
        applied["GPU_MAX_HW_QUEUES"] = str(args.hw_queues)

    if args.signal_pool_size is not None:
        os.environ["ROC_SIGNAL_POOL_SIZE"] = str(args.signal_pool_size)
        applied["ROC_SIGNAL_POOL_SIZE"] = str(args.signal_pool_size)

    if args.disable_sdma:
        os.environ["HSA_ENABLE_SDMA"] = "0"
        applied["HSA_ENABLE_SDMA"] = "0"

    if args.blit_copy_size is not None:
        os.environ["GPU_FORCE_BLIT_COPY_SIZE"] = str(args.blit_copy_size)
        applied["GPU_FORCE_BLIT_COPY_SIZE"] = str(args.blit_copy_size)

    if args.nccl_implicit_order:
        os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
        applied["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"

    if args.disable_cheap_fence:
        os.environ["RCCL_GFX9_CHEAP_FENCE_OFF"] = "1"
        os.environ["RCCL_GFX942_CHEAP_FENCE_OFF"] = "1"
        applied["RCCL_GFX9_CHEAP_FENCE_OFF"] = "1"
        applied["RCCL_GFX942_CHEAP_FENCE_OFF"] = "1"

    if args.disable_clr_batch:
        os.environ["DEBUG_CLR_BATCH_CPU_SYNC_SIZE"] = "0"
        applied["DEBUG_CLR_BATCH_CPU_SYNC_SIZE"] = "0"

    return applied


def run_with_mode(config: ReproducerConfig, rank: int, world_size: int) -> ReproducerResult:
    """
    Dispatch to the appropriate reproducer based on config.mode.

    Uses the modular system (base.py + modes/) via create_reproducer().
    """
    from aorta.race.modes import create_reproducer

    reproducer = create_reproducer(config, rank, world_size)
    return reproducer.run()


def main():
    """Main entry point."""
    args = parse_args()

    # Apply environment variables from CLI flags (before any CUDA init)
    applied_env = apply_env_vars(args)

    # Initialize distributed
    rank, world_size = init_distributed()

    if rank == 0:
        log.info("=" * 70)
        log.info("RCCL RACE CONDITION REPRODUCER")
        log.info("=" * 70)
        log.info("")
        log.info("This reproducer uses PROPER SYNCHRONIZATION everywhere.")
        log.info("If corruption occurs, it indicates a RUNTIME BUG in RCCL/HIP,")
        log.info("not an application-level issue.")
        log.info("")
        log.info(f"Mode: {args.mode}")
        log.info(f"World size: {world_size}")
        log.info(f"Warmup iterations: {args.warmup}")
        log.info(f"Verify iterations: {args.verify}")
        log.info(f"Simulate compute: {not args.no_compute}")
        log.info(f"Compute type: {args.compute_type}")
        if args.compute_type == "transformer":
            log.info(f"  model_dim: {args.model_dim}, num_layers: {args.num_layers}")
        else:
            log.info(f"  gemm_size: {args.gemm_size}, gemm_layers: {args.gemm_layers}")
        log.info(f"H2D prefetch: {args.prefetch}")
        log.info(f"Same stream mode: {args.same_stream}")
        log.info(f"Optimizer: {args.optimizer}")
        log.info(f"Deterministic: {args.deterministic}")
        log.info("")

        # Log relevant env vars
        env_vars = [
            "GPU_MAX_HW_QUEUES",
            "ROC_SIGNAL_POOL_SIZE",
            "HSA_ENABLE_SDMA",
            "GPU_FORCE_BLIT_COPY_SIZE",
            "NCCL_LAUNCH_ORDER_IMPLICIT",
            "RCCL_GFX9_CHEAP_FENCE_OFF",
            "RCCL_GFX942_CHEAP_FENCE_OFF",
            "DEBUG_CLR_BATCH_CPU_SYNC_SIZE",
        ]
        log.info("Environment variables:")
        for var in env_vars:
            value = os.environ.get(var, "(not set)")
            source = " (via CLI)" if var in applied_env else ""
            log.info(f"  {var}={value}{source}")
        log.info("")
        log.info("=" * 70)

    # Build config
    config = ReproducerConfig(
        mode=args.mode,
        warmup_iterations=args.warmup,
        verify_iterations=args.verify,
        stop_on_first_corruption=not args.no_stop_on_first,
        log_interval=args.log_interval,
        h2d_tensor_size=args.h2d_size,
        alltoall_tensor_size=args.a2a_size,
        allreduce_tensor_size=args.ar_size,
        fsdp_shard_size=args.fsdp_shard_size,
        dtype=args.dtype,
        simulate_compute=not args.no_compute,
        compute_type=args.compute_type,
        h2d_prefetch=args.prefetch,
        gemm_size=args.gemm_size,
        gemm_layers=args.gemm_layers,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        include_backward_compute=not args.no_backward,
        same_stream_mode=args.same_stream,
        gpu_max_hw_queues=args.hw_queues,
        optimizer=args.optimizer,
        deterministic=args.deterministic,
        ddp_bucketed=args.bucketed,
    )

    # Run reproducer via mode dispatch
    result = run_with_mode(config, rank, world_size)

    # Sync all ranks
    dist.barrier()

    # Report results (rank 0 only)
    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"Mode: {args.mode}")
        log.info(f"Passed: {result.passed}")
        log.info(f"Total iterations: {result.total_iterations}")
        log.info(f"Corruption count: {result.corruption_count}")
        log.info(f"Elapsed time: {result.elapsed_time_sec:.2f}s")
        log.info(f"Avg step time: {result.avg_step_time_ms:.2f}ms")

        if result.first_corruption_iter is not None:
            log.info(f"First corruption at iteration: {result.first_corruption_iter}")

        if result.passed:
            log.info("")
            log.info("VERDICT: No runtime bug detected with current settings.")
            log.info("If corruption still occurs in real workloads, check for")
            log.info("application-level synchronization issues.")
        else:
            log.info("")
            log.info("VERDICT: RUNTIME BUG DETECTED!")
            log.info("Corruption occurred DESPITE proper synchronization.")
            log.info("This is a bug in RCCL/HIP runtime - report to AMD.")

        log.info("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
