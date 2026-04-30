"""
Command-line interface for the hardware queue evaluation framework.

Example usage:
    # Run single workload with specific stream count
    python -m aorta.hw_queue_eval run hetero_kernels --streams 8

    # Run with PyTorch profiler (generates Chrome trace and TensorBoard logs)
    python -m aorta.hw_queue_eval run hetero_kernels --streams 8 --profile
    python -m aorta.hw_queue_eval run moe --streams 16 --profile --profile-dir traces/

    # Run with real distributed collectives (requires torchrun)
    torchrun --nproc_per_node=4 -m aorta.hw_queue_eval run comms_compute_overlap \\
        --streams 4 --real-collectives --async-op --backend nccl \\
        --process-groups "[0,1,2,3]" --profile --profile-dir traces/

    # Run stream count sweep
    python -m aorta.hw_queue_eval sweep hetero_kernels --streams 1,2,4,8,16,32

    # Run all P0 workloads
    python -m aorta.hw_queue_eval run-priority P0

    # Compare baseline vs modified runtime
    python -m aorta.hw_queue_eval compare --baseline results_baseline.json --test results_modified.json

    # Profile with rocprof (ROCm-specific)
    python -m aorta.hw_queue_eval profile hetero_kernels --streams 8 --output traces/

    # List available workloads
    python -m aorta.hw_queue_eval list
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List, Optional

import click

from aorta.hw_queue_eval import __version__


# Priority classifications for workloads
PRIORITY_WORKLOADS = {
    "P0": [  # Most critical - implement and test first
        "hetero_kernels",
        "tiny_kernel_stress",
        "large_gemm_only",
    ],
    "P1": [  # High priority - important real-world patterns
        "fsdp_tp",
        "moe",
        "speculative_decode",
        "continuous_batch",
    ],
    "P2": [  # Medium priority - additional coverage
        "activation_ckpt",
        "grad_accum",
        "rag_pipeline",
        "graph_subgraphs",
    ],
    "P3": [  # Lower priority - nice to have
        "async_dataload",
        "zero_offload",
        "torch_compile",
    ],
}


def _parse_size(value: str) -> int:
    """Parse a size string with optional K/M/G suffix into bytes."""
    original = value
    value = value.strip().upper()
    multipliers = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
    try:
        result: int
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                result = int(float(value[:-1]) * mult)
                break
        else:
            result = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid size value '{original}'. "
            "Expected an integer or a number with K/M/G suffix, e.g. '128M' or '1024'."
        ) from None
    if result < 0:
        raise ValueError(
            f"Invalid size value '{original}'. Size must be non-negative."
        )
    return result


def get_workload_instance(name: str, **kwargs):
    """Get a workload instance by name."""
    from aorta.hw_queue_eval.workloads.registry import get_workload
    return get_workload(name, **kwargs)


def list_available_workloads():
    """List all available workloads."""
    from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry

    try:
        # Import for side effect (registry self-registration); not all
        # workload modules are guaranteed to be importable on every host.
        from aorta.hw_queue_eval.workloads import (  # noqa: F401
            distributed,
            inference,
            latency_sensitive,
            pipeline,
        )
    except ImportError:
        pass

    return WorkloadRegistry.list_all()


@click.group()
@click.version_option(version=__version__)
def cli():
    """Hardware Queue Evaluation Framework for AMD ROCm.

    Stress-test GPU hardware queue mapping with workloads
    requiring high concurrent stream counts.
    """
    pass


@cli.command()
@click.argument("workload")
@click.option("--streams", "-s", default=4, help="Number of streams")
@click.option("--iterations", "-i", default=100, help="Measurement iterations")
@click.option("--warmup", "-w", default=10, help="Warmup iterations")
@click.option("--output", "-o", default=None, help="Output JSON file")
@click.option("--device", "-d", default="cuda:0", help="Target device")
@click.option("--sync-mode", type=click.Choice(["per_iteration", "end_only", "none"]),
              default="per_iteration", help="Synchronization mode")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.option("--profile", "-p", is_flag=True, help="Enable PyTorch profiler (generates Chrome trace and TensorBoard logs)")
@click.option("--profile-dir", default="profiles", help="Output directory for profiler traces")
@click.option("--real-collectives", is_flag=True,
              help="Use real torch.distributed collectives (requires torchrun)")
@click.option("--async-op", is_flag=True,
              help="Issue non-blocking collectives (only with --real-collectives)")
@click.option("--backend", type=click.Choice(["nccl", "gloo"]), default="nccl",
              help="Distributed backend (only with --real-collectives)")
@click.option("--process-groups", default=None,
              help='Process group spec, e.g. "[0,1,2,3],[4,5,6,7]"')
@click.option("--mode", "wl_mode", default=None,
              type=click.Choice(["compute_only", "comms_only", "comms_compute"]),
              help="Workload mode (comms_compute_overlap only)")
@click.option("--mm-dim", default=None,
              help="GEMM dimensions M,N,K e.g. 2048,2048,2048")
@click.option("--num-compute", default=None, type=int,
              help="Number of GEMM ops per iteration per compute stream")
@click.option("--num-coll", default=None, type=int,
              help="Number of collective ops per iteration")
@click.option("--comm-size", default=None, type=str,
              help="Communication tensor size in bytes (supports M/G suffix, e.g. 128M)")
@click.option("--compute-streams", default=None, type=int,
              help="Number of compute streams (independent of --streams)")
@click.option("--comp-dtype", default=None,
              type=click.Choice(["float32", "float16", "bfloat16"]),
              help="Data type for compute (GEMM) tensors")
@click.option("--comm-dtype", default=None,
              type=click.Choice(["float32", "float16", "bfloat16"]),
              help="Data type for communication tensors")
@click.option("--lock-clocks", type=int, default=None,
              help="Lock GPU clock level (AMD: 0-7) for deterministic results")
@click.option("--power-limit", type=int, default=None,
              help="Set GPU power limit in watts")
@click.option("--ebpf-trace", is_flag=True, default=False,
              help="Enable eBPF queue tracing (requires bpftrace + root)")
@click.option("--ebpf-memory-trace", is_flag=True, default=False,
              help="Enable eBPF memory tracing for BO migrations and process "
                   "eviction/restore cycles (not literal GPU page faults)")
def run(workload: str, streams: int, iterations: int, warmup: int,
        output: Optional[str], device: str, sync_mode: str, quiet: bool,
        profile: bool, profile_dir: str,
        real_collectives: bool, async_op: bool, backend: str,
        process_groups: Optional[str],
        wl_mode: Optional[str], mm_dim: Optional[str],
        num_compute: Optional[int], num_coll: Optional[int],
        comm_size: Optional[str], compute_streams: Optional[int],
        comp_dtype: Optional[str], comm_dtype: Optional[str],
        lock_clocks: Optional[int], power_limit: Optional[int],
        ebpf_trace: bool, ebpf_memory_trace: bool):
    """Run a single workload evaluation.

    WORKLOAD: Name of the workload to run (e.g., hetero_kernels, fsdp_tp)

    For distributed mode (real NCCL/RCCL collectives), launch via torchrun:

        torchrun --nproc_per_node=4 -m aorta.hw_queue_eval run comms_compute_overlap
            --streams 4 --real-collectives --backend nccl
    """
    import os

    from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness
    from aorta.hw_queue_eval.core.torch_profiler import TorchProfilerWrapper, generate_profile_summary
    from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry
    from aorta.utils.gpu_control import GPUControlConfig

    # Build GPU control config from CLI flags
    gpu_ctl_enabled = lock_clocks is not None or power_limit is not None
    gpu_control = GPUControlConfig(
        enabled=gpu_ctl_enabled,
        gpu_clock_level=lock_clocks,
        power_limit_watts=power_limit,
    ) if gpu_ctl_enabled else None

    # --- Distributed device auto-detection ---
    # When --real-collectives is set, override device to cuda:{LOCAL_RANK}
    # so that each torchrun process uses its own GPU.
    dist_rank = 0
    if real_collectives:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist_rank = int(os.environ.get("RANK", "0"))
        device = f"cuda:{local_rank}"

    # Only rank 0 prints verbose output in distributed mode
    is_main = dist_rank == 0

    try:
        # Get workload -- pass applicable kwargs
        kwargs = {}
        if real_collectives:
            kwargs.update(
                simulate_collectives=False,
                async_op=async_op,
                backend=backend,
                process_groups=process_groups,
            )

        # Workload-specific parameters (comms_compute_overlap only)
        _overlap_opts = (
            wl_mode, mm_dim, num_compute, num_coll,
            comm_size, compute_streams, comp_dtype, comm_dtype,
        )
        if any(v is not None for v in _overlap_opts):
            if workload != "comms_compute_overlap":
                click.echo(
                    f"Error: Options --mode, --mm-dim, --num-compute, --num-coll, "
                    f"--comm-size, --compute-streams, --comp-dtype, --comm-dtype "
                    f"are only valid for the comms_compute_overlap workload, "
                    f"not '{workload}'.",
                    err=True,
                )
                sys.exit(1)

            if wl_mode is not None:
                kwargs["mode"] = wl_mode
            if mm_dim is not None:
                try:
                    parts = [int(x.strip()) for x in mm_dim.split(",")]
                except ValueError:
                    click.echo(
                        "Error: --mm-dim must contain integers, e.g. '2048,2048,2048'",
                        err=True,
                    )
                    sys.exit(1)
                if len(parts) == 1:
                    kwargs["mm_dim"] = (parts[0], parts[0], parts[0])
                elif len(parts) == 3:
                    kwargs["mm_dim"] = tuple(parts)
                else:
                    click.echo("Error: --mm-dim must be M,N,K or a single value", err=True)
                    sys.exit(1)
                if any(d <= 0 for d in kwargs["mm_dim"]):
                    click.echo("Error: --mm-dim dimensions must be positive integers", err=True)
                    sys.exit(1)
            if num_compute is not None:
                if num_compute <= 0:
                    click.echo("Error: --num-compute must be a positive integer", err=True)
                    sys.exit(1)
                kwargs["num_compute_per_iter"] = num_compute
            if num_coll is not None:
                if num_coll <= 0:
                    click.echo("Error: --num-coll must be a positive integer", err=True)
                    sys.exit(1)
                kwargs["num_coll_per_iter"] = num_coll
            if comm_size is not None:
                kwargs["comm_size_bytes"] = _parse_size(comm_size)
            if compute_streams is not None:
                if compute_streams <= 0:
                    click.echo("Error: --compute-streams must be a positive integer", err=True)
                    sys.exit(1)
                kwargs["compute_streams"] = compute_streams
            if comp_dtype is not None:
                kwargs["comp_data_type"] = comp_dtype
            if comm_dtype is not None:
                kwargs["comm_data_type"] = comm_dtype

        wl = get_workload_instance(workload, **kwargs)
        info = WorkloadRegistry.get_info(workload)

        # Check stream count compatibility
        if not wl.supports_stream_count(streams):
            click.echo(
                f"Error: Workload {workload} supports {wl.min_streams}-{wl.max_streams} streams",
                err=True
            )
            sys.exit(1)

        # Print header with workload info (rank 0 only in distributed mode)
        if is_main:
            click.echo("=" * 70)
            click.echo("GPU HARDWARE QUEUE EVALUATION")
            click.echo("=" * 70)
            click.echo()
            click.echo(f"WORKLOAD: {workload}")
            click.echo(f"  {info.description}")
            click.echo()
            click.echo("PURPOSE:")
            _print_workload_purpose(workload, info)
            click.echo()
            click.echo("TEST CONFIGURATION:")
            click.echo(f"  Concurrent streams:     {streams} (recommended: {info.recommended_streams})")
            click.echo(f"  Measurement iterations: {iterations}")
            click.echo(f"  Warmup iterations:      {warmup}")
            click.echo(f"  Device:                 {device}")
            click.echo(f"  Switch sensitivity:     {info.switch_latency_sensitivity}")
            if real_collectives:
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                click.echo(f"  Distributed:            yes (backend={backend}, world_size={world_size})")
                click.echo(f"  Async collectives:      {async_op}")
                if process_groups:
                    click.echo(f"  Process groups:         {process_groups}")
            click.echo()
            click.echo("-" * 70)
            click.echo("Running workload...")
            if profile:
                click.echo("  (Profiling enabled - will generate Chrome trace and TensorBoard logs)")
            click.echo()

        # Create harness
        config = HarnessConfig(
            stream_count=streams,
            warmup_iterations=warmup,
            measurement_iterations=iterations,
            sync_mode=sync_mode,
            device=device,
            gpu_control=gpu_control,
            ebpf_tracing=ebpf_trace,
            ebpf_memory_tracing=ebpf_memory_trace,
        )
        harness = StreamHarness(config)

        if gpu_ctl_enabled:
            click.echo("GPU CONTROL:")
            if lock_clocks is not None:
                click.echo(f"  Clock level locked to: {lock_clocks}")
            if power_limit is not None:
                click.echo(f"  Power limit set to:    {power_limit} W")
            click.echo()

        if ebpf_trace or ebpf_memory_trace:
            click.echo("eBPF TRACING:")
            if ebpf_trace:
                click.echo("  Queue tracing: enabled")
            if ebpf_memory_trace:
                click.echo("  Memory tracing: enabled")
            click.echo()

        # Run workload (with optional profiling)
        profile_result = None
        if profile:
            import torch
            from aorta.utils import create_streams

            # Setup profiler
            profiler_wrapper = TorchProfilerWrapper(output_dir=profile_dir)

            # Setup workload and streams
            wl.setup(streams, device)
            cuda_streams = create_streams(streams, device)

            # Profile the workload
            def run_iteration():
                wl.run_iteration(cuda_streams)
                torch.cuda.synchronize()

            profile_result = profiler_wrapper.profile_workload(
                run_iteration,
                name=f"{workload}_{streams}s",
                iterations=iterations,
                warmup=warmup,
            )

        # Run regular benchmark (always, for metrics)
        result = harness.run_workload(wl)

        # Print results with interpretation (rank 0 only)
        if is_main:
            click.echo("-" * 70)
            click.echo("RESULTS")
            click.echo("-" * 70)
            click.echo()

            click.echo("THROUGHPUT:")
            click.echo(f"  {result.throughput:,.2f} {result.throughput_unit}")
            click.echo("  (Higher is better - measures how much work completed per second)")
            click.echo()

            click.echo("LATENCY (per iteration):")
            click.echo(f"  Mean:  {result.latency_ms['mean']:.3f} ms")
            click.echo(f"  P50:   {result.latency_ms['p50']:.3f} ms  (median - 50% of iterations faster than this)")
            click.echo(f"  P95:   {result.latency_ms['p95']:.3f} ms  (95% of iterations faster than this)")
            click.echo(f"  P99:   {result.latency_ms['p99']:.3f} ms  (99% of iterations faster than this)")
            click.echo(f"  (Lower is better - time to complete one iteration across all {streams} streams)")
            click.echo()

            # Latency variance analysis
            latency_ratio = result.latency_ms['p99'] / result.latency_ms['p50'] if result.latency_ms['p50'] > 0 else 1
            if latency_ratio > 2.0:
                click.echo(f"  WARNING: High latency variance (P99/P50 = {latency_ratio:.1f}x)")
                click.echo("    This may indicate queue contention or scheduling issues")
            elif latency_ratio > 1.5:
                click.echo(f"  Note: Moderate latency variance (P99/P50 = {latency_ratio:.1f}x)")
            click.echo()

            click.echo("QUEUE SWITCH ANALYSIS:")
            if result.switch_latency:
                switch_overhead = result.switch_latency['estimated_switch_overhead_ms']
                inter_gap = result.switch_latency['inter_stream_gap_ms']
                intra_gap = result.switch_latency['intra_stream_gap_ms']

                click.echo(f"  Inter-stream gap: {inter_gap:.3f} ms (avg gap between kernels on DIFFERENT streams)")
                click.echo(f"  Intra-stream gap: {intra_gap:.3f} ms (avg gap between kernels on SAME stream)")
                click.echo(f"  Est. switch overhead: {switch_overhead:.3f} ms")

                if switch_overhead > 0.1:
                    click.echo("  Significant queue switch overhead detected")
                    click.echo(f"    This suggests hardware queue contention at {streams} streams")
                elif switch_overhead > 0.01:
                    click.echo("  Moderate queue switch overhead")
                else:
                    click.echo("  Minimal queue switch overhead")
            else:
                click.echo("  (Not enough data to estimate switch overhead)")
            click.echo()

            click.echo("TIMING:")
            click.echo(f"  Total measurement time: {result.total_time_ms:.2f} ms ({result.total_time_ms/1000:.2f} sec)")
            click.echo()

            # eBPF driver-level metrics
            if result.ebpf_queue_metrics:
                click.echo("eBPF DRIVER-LEVEL QUEUE METRICS:")
                eqm = result.ebpf_queue_metrics
                total_sub = eqm.get('total_submissions', 0)
                total_disp = eqm.get('total_dispatches', 0)
                click.echo(f"  Total dispatches:   {total_disp}")
                click.echo(f"  HW rings used:      {eqm.get('rings_used', [])}")
                click.echo(f"  Dispatch rate:      {eqm.get('dispatch_rate_per_sec', 0):.0f} /sec")

                if total_sub > 0:
                    click.echo(f"  Total submissions:  {total_sub}")
                    avg_us = eqm.get("avg_submit_to_dispatch_us", 0.0)
                    p99_us = eqm.get("p99_submit_to_dispatch_us", 0.0)
                    click.echo(f"  Submit→dispatch avg:  {avg_us:.1f} us")
                    click.echo(f"  Submit→dispatch P99:  {p99_us:.1f} us")
                else:
                    avg_gap = eqm.get("avg_inter_dispatch_gap_us", 0.0)
                    p99_gap = eqm.get("p99_inter_dispatch_gap_us", 0.0)
                    click.echo(f"  Inter-dispatch gap avg: {avg_gap:.1f} us")
                    click.echo(f"  Inter-dispatch gap P99: {p99_gap:.1f} us")
                    click.echo("  (ROCm/KFD path -- submit events not visible via amdgpu_cs_ioctl)")
                click.echo()

            if result.ebpf_vs_cuda:
                click.echo("eBPF vs CUDA COMPARISON:")
                cmp = result.ebpf_vs_cuda
                click.echo(f"  eBPF dispatch gap avg: {cmp.get('ebpf_avg_dispatch_gap_ms', cmp.get('ebpf_avg_submit_to_dispatch_ms', 0)):.3f} ms")
                click.echo(f"  CUDA switch overhead:  {cmp.get('cuda_estimated_switch_overhead_ms', 0):.3f} ms")
                click.echo(f"  Measurement accuracy:  {cmp.get('accuracy_pct', 0):.1f}%")
                click.echo()

            if result.ebpf_memory_metrics:
                click.echo("eBPF MEMORY METRICS:")
                emm = result.ebpf_memory_metrics
                bo_moves = emm.get('total_bo_moves', 0)
                bo_maps = emm.get('total_bo_maps', 0)
                bo_unmaps = emm.get('total_bo_unmaps', 0)
                evictions = emm.get('total_evictions', 0)
                migration = emm.get('migration_bytes', 0)

                click.echo(f"  BO moves (migrations): {bo_moves}  ({emm.get('bo_move_rate_per_sec', 0):.0f} /sec)")
                if migration > 0:
                    if migration >= 1024 * 1024:
                        click.echo(f"  Migration volume:      {migration / (1024*1024):.1f} MB")
                    else:
                        click.echo(f"  Migration volume:      {migration} bytes")
                click.echo(f"  BO maps / unmaps:      {bo_maps} / {bo_unmaps}")
                click.echo(f"  Evictions / restores:  {evictions} / {emm.get('total_restores', 0)}")
                if evictions > 0:
                    click.echo(f"  Eviction rate:         {emm.get('fault_rate_per_sec', 0):.1f} /sec")
                    click.echo(f"  Avg evict latency:     {emm.get('avg_fault_latency_us', 0):.1f} us")
                if bo_moves == 0 and bo_maps == 0 and evictions == 0:
                    click.echo("  (No memory events captured -- workload may not trigger")
                    click.echo("   migrations. Try a memory-intensive workload or check")
                    click.echo("   tracepoint availability with: aorta ebpf-info)")
                click.echo()

            # Summary
            click.echo("-" * 70)
            click.echo("INTERPRETATION")
            click.echo("-" * 70)
            _print_interpretation(workload, info, result, streams)

            # Print profile results if profiling was enabled
            if profile and profile_result:
                click.echo()
                click.echo("-" * 70)
                click.echo("PROFILER OUTPUT")
                click.echo("-" * 70)
                click.echo()
                click.echo(generate_profile_summary(profile_result))

            # Save to file if requested
            if output:
                result.to_json(output)
                click.echo()
                click.echo(f"Results saved to: {output}")

            click.echo()
            click.echo("=" * 70)
            click.echo("To compare with different stream counts, run:")
            click.echo(f"  python -m aorta.hw_queue_eval sweep {workload} --streams 1,2,4,8,16,32")
            if profile:
                click.echo()
                click.echo("Profile traces saved to:")
                if profile_result and profile_result.chrome_trace_path:
                    click.echo(f"  Chrome trace: {profile_result.chrome_trace_path}")
                    click.echo("    View: Open chrome://tracing and load the JSON file")
                if profile_result and profile_result.tensorboard_dir:
                    click.echo(f"  TensorBoard: {profile_result.tensorboard_dir}")
                    click.echo(f"    View: tensorboard --logdir={profile_result.tensorboard_dir}")
            click.echo("=" * 70)

    except KeyError as e:
        click.echo(f"Error: Workload not found: {e}", err=True)
        click.echo(f"Available workloads: {', '.join(list_available_workloads())}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error running workload: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _print_workload_purpose(workload: str, info) -> None:
    """Print workload-specific purpose description."""
    purposes = {
        "hetero_kernels": (
            "  Tests queue switch latency by interleaving tiny kernels (~10us) with\n"
            "  large GEMMs (~10ms). Poor queue mapping causes 'convoy effect' where\n"
            "  tiny kernels wait behind large ones on shared queues."
        ),
        "tiny_kernel_stress": (
            "  Stress test with ONLY tiny kernels. Makes queue switch overhead the\n"
            "  dominant factor. If throughput doesn't scale with streams, it indicates\n"
            "  queue scheduling bottlenecks."
        ),
        "large_gemm_only": (
            "  Compute-bound baseline with large GEMMs only. Should scale well with\n"
            "  streams. Use this to establish baseline GPU compute capability."
        ),
        "fsdp_tp": (
            "  Simulates FSDP + Tensor Parallelism (3D parallelism) with overlapped\n"
            "  communication and compute. Tests stream coordination for distributed\n"
            "  training patterns."
        ),
        "moe": (
            "  Mixture of Experts with parallel expert execution. Each expert runs\n"
            "  on its own stream. Tests high stream count scalability (8-16+ streams)."
        ),
        "speculative_decode": (
            "  Draft + verify speculative decoding pattern. Has tight latency\n"
            "  requirements - switch overhead directly impacts tokens/sec."
        ),
        "continuous_batch": (
            "  Overlapped prefill (compute-heavy) and decode (memory-bound) phases.\n"
            "  Tests ability to run different workload types concurrently."
        ),
        "activation_ckpt": (
            "  Activation checkpointing with recomputation streams. Tests overlap\n"
            "  of forward recomputation with backward gradient computation."
        ),
        "grad_accum": (
            "  Gradient accumulation with early reduction. Tests overlapping\n"
            "  gradient computation with all-reduce communication."
        ),
        "rag_pipeline": (
            "  Multi-model RAG pipeline: embedding -> retrieval -> reranking -> generation.\n"
            "  Tests concurrent execution of different model types."
        ),
        "graph_subgraphs": (
            "  Independent computation subgraphs with no dependencies. Tests maximum\n"
            "  parallel execution capability when work is fully parallelizable."
        ),
        "async_dataload": (
            "  Async data loading with GPU preprocessing overlap. Tests ability to\n"
            "  hide H2D transfer latency with concurrent compute."
        ),
        "zero_offload": (
            "  ZeRO-offload memory management. Tests overlap of CPU<->GPU transfers\n"
            "  with computation for memory-constrained training."
        ),
        "torch_compile": (
            "  Multi-region torch.compile execution. Tests interaction between\n"
            "  compiled code and manual stream management."
        ),
        "comms_compute_overlap": (
            "  Synthetic comm-compute overlap benchmark. Runs GEMM compute on\n"
            "  dedicated streams while collective communication runs on a separate\n"
            "  stream. Supports simulated or real NCCL/RCCL collectives, async ops,\n"
            "  and configurable process groups."
        ),
    }
    click.echo(purposes.get(workload, f"  Tests {info.category} patterns with {info.switch_latency_sensitivity} switch sensitivity."))


def _print_interpretation(workload: str, info, result, streams: int) -> None:
    """Print interpretation guidance based on results."""
    click.echo()

    # General guidance
    click.echo(f"This workload has '{info.switch_latency_sensitivity}' sensitivity to queue switch latency.")
    click.echo()

    if info.switch_latency_sensitivity == "critical":
        click.echo("For CRITICAL sensitivity workloads like this:")
        click.echo("  - Throughput should ideally scale linearly with stream count")
        click.echo("  - Latency variance (P99/P50) should stay below 1.5x")
        click.echo("  - Switch overhead > 0.1ms indicates queue contention")
        click.echo()
        click.echo("If throughput plateaus or latency spikes at higher stream counts,")
        click.echo("it suggests the hardware queue limit has been reached.")

    elif info.switch_latency_sensitivity == "high":
        click.echo("For HIGH sensitivity workloads:")
        click.echo("  - Some throughput degradation at high stream counts is expected")
        click.echo("  - Watch for latency variance > 2x (P99/P50)")
        click.echo("  - Communication/compute overlap effectiveness is key")

    else:
        click.echo("For MEDIUM/LOW sensitivity workloads:")
        click.echo("  - Focus on overall throughput rather than latency variance")
        click.echo("  - These workloads are less affected by queue switch overhead")

    click.echo()
    click.echo("NEXT STEPS:")
    click.echo("  1. Run a sweep to find the optimal stream count:")
    click.echo(f"     python -m aorta.hw_queue_eval sweep {workload} -s 1,2,4,8,16,32")
    click.echo("  2. Compare results before/after runtime changes:")
    click.echo("     python -m aorta.hw_queue_eval compare -b baseline.json -t test.json")


@cli.command()
@click.argument("workload")
@click.option("--streams", "-s", default="1,2,4,8,16,32",
              help="Comma-separated stream counts to test")
@click.option("--iterations", "-i", default=100, help="Measurement iterations per config")
@click.option("--warmup", "-w", default=10, help="Warmup iterations")
@click.option("--output", "-o", default=None, help="Output JSON file")
@click.option("--device", "-d", default="cuda:0", help="Target device")
@click.option("--lock-clocks", type=int, default=None,
              help="Lock GPU clock level (AMD: 0-7) for deterministic results")
@click.option("--power-limit", type=int, default=None,
              help="Set GPU power limit in watts")
@click.option("--ebpf-trace", is_flag=True, default=False,
              help="Enable eBPF queue tracing (requires bpftrace + root)")
@click.option("--ebpf-memory-trace", is_flag=True, default=False,
              help="Enable eBPF memory tracing for BO migrations and process "
                   "eviction/restore cycles (not literal GPU page faults)")
def sweep(workload: str, streams: str, iterations: int, warmup: int,
          output: Optional[str], device: str, lock_clocks: Optional[int],
          power_limit: Optional[int], ebpf_trace: bool, ebpf_memory_trace: bool):
    """Run workload across multiple stream counts.

    WORKLOAD: Name of the workload to sweep
    """
    from aorta.hw_queue_eval.core.harness import (
        HarnessConfig, StreamHarness, analyze_sweep_results,
        format_results_table, save_sweep_results
    )
    from aorta.utils.gpu_control import GPUControlConfig

    # Build GPU control config from CLI flags
    gpu_ctl_enabled = lock_clocks is not None or power_limit is not None
    gpu_control = GPUControlConfig(
        enabled=gpu_ctl_enabled,
        gpu_clock_level=lock_clocks,
        power_limit_watts=power_limit,
    ) if gpu_ctl_enabled else None

    # Parse stream counts
    stream_counts = [int(s.strip()) for s in streams.split(",")]

    click.echo(f"Sweeping workload: {workload}")
    click.echo(f"  Stream counts: {stream_counts}")
    if gpu_ctl_enabled:
        click.echo(f"  GPU control: clock_level={lock_clocks}, power_limit={power_limit}W")
    click.echo()

    try:
        wl = get_workload_instance(workload)

        # Filter stream counts by workload limits
        valid_counts = [c for c in stream_counts if wl.supports_stream_count(c)]
        if not valid_counts:
            click.echo(f"Error: No valid stream counts for {workload}", err=True)
            click.echo(f"Workload supports {wl.min_streams}-{wl.max_streams} streams", err=True)
            sys.exit(1)

        if len(valid_counts) < len(stream_counts):
            click.echo(f"Note: Filtered to valid stream counts: {valid_counts}")

        # Run sweep
        results = []
        for count in valid_counts:
            click.echo(f"Running with {count} streams...", nl=False)

            config = HarnessConfig(
                stream_count=count,
                warmup_iterations=warmup,
                measurement_iterations=iterations,
                device=device,
                gpu_control=gpu_control,
                ebpf_tracing=ebpf_trace,
                ebpf_memory_tracing=ebpf_memory_trace,
            )
            harness = StreamHarness(config)
            result = harness.run_workload(wl)
            results.append(result)

            click.echo(f" {result.throughput:.2f} {result.throughput_unit}")

        # Print summary table
        click.echo()
        click.echo("Summary:")
        click.echo(format_results_table(results))

        # Analyze scaling
        analysis = analyze_sweep_results(results)
        click.echo()
        click.echo("Scaling Analysis:")
        click.echo(f"  Peak throughput at: {analysis.peak_stream_count} streams")
        if analysis.inflection_point:
            click.echo(f"  Inflection point: {analysis.inflection_point} streams")

        # Save results
        if output:
            save_sweep_results(results, output)
            click.echo(f"\nResults saved to: {output}")

    except KeyError as e:
        click.echo(f"Error: Workload not found: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("run-priority")
@click.argument("priority", type=click.Choice(["P0", "P1", "P2", "P3", "all"]))
@click.option("--streams", "-s", default="1,2,4,8,16",
              help="Comma-separated stream counts to test")
@click.option("--iterations", "-i", default=50, help="Measurement iterations")
@click.option("--output-dir", "-o", default="results", help="Output directory")
@click.option("--device", "-d", default="cuda:0", help="Target device")
@click.option("--profile", "-p", is_flag=True, help="Enable PyTorch profiler for each workload")
@click.option("--profile-dir", default="profiles", help="Output directory for profiler traces")
def run_priority(priority: str, streams: str, iterations: int,
                 output_dir: str, device: str, profile: bool, profile_dir: str):
    """Run all workloads of a given priority level.

    PRIORITY: Priority level (P0, P1, P2, P3, or 'all')
    """
    import torch
    from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness, save_sweep_results
    from aorta.utils.device import log_environment_info

    # Get workloads for priority
    if priority == "all":
        workloads = []
        for p in ["P0", "P1", "P2", "P3"]:
            workloads.extend(PRIORITY_WORKLOADS.get(p, []))
    else:
        workloads = PRIORITY_WORKLOADS.get(priority, [])

    if not workloads:
        click.echo(f"No workloads found for priority {priority}", err=True)
        sys.exit(1)

    # Parse stream counts
    stream_counts = [int(s.strip()) for s in streams.split(",")]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Log comprehensive environment info (also saves to output_dir/environment_info.json).
    # Return value is intentionally unused; the side effect (disk write + log)
    # is the goal.
    log_environment_info(
        stream_counts=stream_counts,
        iterations=iterations,
        output_dir=output_dir,
    )
    click.echo()

    click.echo(f"Running {len(workloads)} workloads at priority {priority}")
    click.echo(f"Stream counts: {stream_counts}")
    click.echo(f"Output directory: {output_path}")
    if profile:
        click.echo(f"Profiling enabled: {profile_dir}")
    click.echo()

    # Setup profiler if enabled
    profile_path = None
    if profile:
        from aorta.hw_queue_eval.core.torch_profiler import TorchProfilerWrapper
        profile_path = Path(profile_dir)
        profile_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    failed = []

    for workload_name in workloads:
        click.echo(f"[{workload_name}]")

        try:
            wl = get_workload_instance(workload_name)
            valid_counts = [c for c in stream_counts if wl.supports_stream_count(c)]

            if not valid_counts:
                click.echo("  Skipped (no valid stream counts)")
                continue

            results = []
            for count in valid_counts:
                click.echo(f"  {count} streams...", nl=False)

                config = HarnessConfig(
                    stream_count=count,
                    warmup_iterations=10,
                    measurement_iterations=iterations,
                    device=device,
                )
                harness = StreamHarness(config)

                # Run with profiling if enabled
                if profile and profile_path:
                    from aorta.utils import create_streams
                    profiler_wrapper = TorchProfilerWrapper(output_dir=str(profile_path))

                    # Setup workload and streams
                    wl.setup(count, device)
                    cuda_streams = create_streams(count, device)

                    def run_iteration():
                        wl.run_iteration(cuda_streams)
                        torch.cuda.synchronize()

                    profiler_wrapper.profile_workload(
                        run_iteration,
                        name=f"{workload_name}_{count}s",
                        iterations=min(iterations, 20),  # Limit profiled iterations
                        warmup=5,
                    )

                result = harness.run_workload(wl)
                results.append(result)
                click.echo(f" {result.throughput:.2f} {result.throughput_unit}")

            # Save results
            output_file = output_path / f"{workload_name}_results.json"
            save_sweep_results(results, output_file)
            all_results[workload_name] = results

        except Exception as e:
            click.echo(f"  Error: {e}")
            failed.append((workload_name, str(e)))

    # Summary
    click.echo()
    click.echo("=" * 50)
    click.echo(f"Completed: {len(all_results)}/{len(workloads)} workloads")
    if failed:
        click.echo(f"Failed: {len(failed)}")
        for name, error in failed:
            click.echo(f"  - {name}: {error}")


@cli.command()
@click.option("--baseline", "-b", required=True, help="Baseline results JSON")
@click.option("--test", "-t", required=True, help="Test results JSON")
@click.option("--threshold", default=0.05, help="Regression threshold (fraction)")
def compare(baseline: str, test: str, threshold: float):
    """Compare baseline and test results for regressions."""
    from aorta.hw_queue_eval.core.metrics import compare_results

    # Load results
    with open(baseline) as f:
        baseline_data = json.load(f)
    with open(test) as f:
        test_data = json.load(f)

    # Handle both single result and sweep result formats
    if "results" in baseline_data:
        baseline_results = baseline_data["results"]
        test_results = test_data.get("results", [])
    else:
        baseline_results = [baseline_data]
        test_results = [test_data]

    click.echo(f"Comparing results with {threshold*100:.1f}% threshold")
    click.echo()

    has_regressions = False

    for i, (b, t) in enumerate(zip(baseline_results, test_results)):
        stream_count = b.get("stream_count", i + 1)
        comparison = compare_results(b, t, threshold)

        click.echo(f"Stream count: {stream_count}")

        if comparison["regressions"]:
            has_regressions = True
            click.echo("  REGRESSIONS:")
            for reg in comparison["regressions"]:
                click.echo(f"    - {reg['metric']}: {reg['baseline']:.3f} -> {reg['test']:.3f} "
                          f"({reg['change_pct']:+.1f}%)")

        if comparison["improvements"]:
            click.echo("  Improvements:")
            for imp in comparison["improvements"]:
                click.echo(f"    + {imp['metric']}: {imp['baseline']:.3f} -> {imp['test']:.3f} "
                          f"({imp['change_pct']:+.1f}%)")

        if not comparison["regressions"] and not comparison["improvements"]:
            click.echo("  No significant changes")

        click.echo()

    if has_regressions:
        click.echo("RESULT: REGRESSIONS DETECTED", err=True)
        sys.exit(1)
    else:
        click.echo("RESULT: No regressions detected")


@cli.command("list")
@click.option("--category", "-c", default=None,
              help="Filter by category (distributed, inference, pipeline, latency_sensitive)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def list_workloads(category: Optional[str], verbose: bool):
    """List available workloads."""
    from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry

    # Import workload modules to register them (side-effect imports).
    try:
        from aorta.hw_queue_eval.workloads import (  # noqa: F401
            distributed,
            inference,
            latency_sensitive,
            pipeline,
        )
    except ImportError as e:
        click.echo(f"Warning: Could not import all workloads: {e}", err=True)

    workloads = WorkloadRegistry.list_all()

    if category:
        workloads = WorkloadRegistry.list_by_category(category)

    if not workloads:
        click.echo("No workloads found")
        return

    click.echo(f"Available workloads ({len(workloads)}):")
    click.echo()

    for name in sorted(workloads):
        try:
            info = WorkloadRegistry.get_info(name)

            if verbose:
                click.echo(f"  {name}")
                click.echo(f"    Description: {info.description}")
                click.echo(f"    Category: {info.category}")
                click.echo(f"    Streams: {info.min_streams}-{info.max_streams} (recommended: {info.recommended_streams})")
                click.echo(f"    Switch sensitivity: {info.switch_latency_sensitivity}")
                click.echo()
            else:
                click.echo(f"  {name:<25} [{info.category}] {info.description}")

        except Exception:
            click.echo(f"  {name:<25} (error loading info)")


@cli.command()
@click.argument("workload")
@click.option("--streams", "-s", default=8, help="Number of streams")
@click.option("--output", "-o", default="profiles", help="Output directory")
@click.option("--metrics", "-m", default=None,
              help="Comma-separated hardware metrics to collect")
def profile(workload: str, streams: int, output: str, metrics: Optional[str]):
    """Profile a workload with ROCm tools.

    WORKLOAD: Name of the workload to profile
    """
    from aorta.hw_queue_eval.core.profiler import ROCmProfiler, create_profiling_script

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Profiling workload: {workload}")
    click.echo(f"  Streams: {streams}")
    click.echo(f"  Output: {output_path}")

    profiler = ROCmProfiler(output_path)

    if not profiler.rocprof_available:
        click.echo("Warning: rocprof not available. Creating profiling script instead.")

        # Create a standalone script
        script_path = create_profiling_script(workload, streams, output_path)
        click.echo(f"\nCreated profiling script: {script_path}")
        click.echo(f"Run with: rocprof --hip-trace -o {output_path}/{workload}.csv python {script_path}")
        return

    # Create and run profiling
    script_path = create_profiling_script(workload, streams, output_path)

    metrics_list = None
    if metrics:
        metrics_list = [m.strip() for m in metrics.split(",")]

    click.echo("Running profiler...")

    try:
        trace_file = profiler.profile_with_rocprof(
            ["python", str(script_path)],
            metrics=metrics_list,
            output_name=f"{workload}_{streams}s",
        )

        # Parse results
        queue_info = profiler.parse_queue_info(trace_file)

        click.echo()
        click.echo("Queue Information:")
        click.echo(f"  Number of queues used: {queue_info.num_queues}")
        click.echo(f"  Kernels per queue: {queue_info.kernels_per_queue}")

        # Generate timeline
        timeline_file = profiler.generate_timeline(trace_file)
        click.echo(f"\nTimeline generated: {timeline_file}")
        click.echo(f"Trace file: {trace_file}")

    except Exception as e:
        click.echo(f"Error during profiling: {e}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show environment and configuration information."""
    import torch
    from aorta.utils import get_device_properties, get_rocm_env_info, ensure_gpu_available

    click.echo("Hardware Queue Evaluation Framework")
    click.echo(f"Version: {__version__}")
    click.echo()

    # PyTorch info
    click.echo("PyTorch:")
    click.echo(f"  Version: {torch.__version__}")
    click.echo(f"  CUDA available: {torch.cuda.is_available()}")
    click.echo(f"  CUDA version: {torch.version.cuda}")

    # ROCm info
    rocm_info = get_rocm_env_info()
    click.echo()
    click.echo("ROCm:")
    click.echo(f"  Is ROCm: {rocm_info['is_rocm']}")
    if rocm_info['is_rocm']:
        click.echo(f"  HIP version: {rocm_info['hip_version']}")
        if 'env_vars' in rocm_info:
            click.echo("  Environment variables:")
            for var, val in rocm_info['env_vars'].items():
                if val:
                    click.echo(f"    {var}={val}")

    # GPU info
    if torch.cuda.is_available():
        click.echo()
        click.echo("GPU:")
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            if ensure_gpu_available(device):
                props = get_device_properties(device)
                click.echo(f"  [{i}] {props.name}")
                click.echo(f"      Memory: {props.total_memory_gb:.1f} GB")
                click.echo(f"      Compute Units: {props.multi_processor_count}")

    # eBPF info
    click.echo()
    click.echo("eBPF:")
    try:
        from aorta.hw_queue_eval.core.ebpf_tracer import check_ebpf_capabilities

        caps = check_ebpf_capabilities()
        click.echo(f"  bpftrace: {caps.bpftrace_version or 'not installed'}")
        click.echo(f"  amdgpu tracepoints: {'yes' if caps.has_amdgpu_tracepoints else 'no'}")
        click.echo(f"  amdkfd tracepoints: {'yes' if caps.has_amdkfd_tracepoints else 'no'}")
        click.echo(f"  Available: {'yes' if caps.available else 'no'}")
    except Exception:
        click.echo("  (detection failed)")


@cli.command("policy-sweep")
@click.argument("workload")
@click.option("--streams", "-s", default=4, help="Number of streams")
@click.option("--iterations", "-i", default=100, help="Measurement iterations per policy")
@click.option("--warmup", "-w", default=10, help="Warmup iterations")
@click.option("--policies", "-p", default=None,
              help="Comma-separated policy names (default: baseline,priority_lc,priority_be)")
@click.option("--output", "-o", default=None, help="Output JSON file")
@click.option("--device", "-d", default="cuda:0", help="Target device")
def policy_sweep(workload: str, streams: int, iterations: int, warmup: int,
                 policies: Optional[str], output: Optional[str], device: str):
    """Evaluate a workload across different scheduling/memory policies.

    WORKLOAD: Name of the workload to evaluate

    Available built-in policies:
      baseline, priority_lc, priority_be, multi_tenant_fair,
      high_queue, default_uvm, xnack_off
    """
    from aorta.hw_queue_eval.core.harness import HarnessConfig
    from aorta.hw_queue_eval.core.policy_evaluator import PolicyEvaluator

    policy_names = (
        [p.strip() for p in policies.split(",")]
        if policies
        else ["baseline", "priority_lc", "priority_be"]
    )

    click.echo(f"Policy sweep: {workload}")
    click.echo(f"  Streams:  {streams}")
    click.echo(f"  Policies: {policy_names}")
    click.echo()

    try:
        wl = get_workload_instance(workload)

        base_config = HarnessConfig(
            stream_count=streams,
            warmup_iterations=warmup,
            measurement_iterations=iterations,
            device=device,
        )

        evaluator = PolicyEvaluator(base_config, wl)
        comparison = evaluator.evaluate(policy_names=policy_names)

        click.echo(comparison.summary_table())

        if output:
            comparison.save(output)
            click.echo(f"\nResults saved to: {output}")

    except (KeyError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command("ebpf-info")
def ebpf_info():
    """Show eBPF capabilities and available tracepoints."""
    from aorta.hw_queue_eval.core.ebpf_tracer import check_ebpf_capabilities

    caps = check_ebpf_capabilities()

    click.echo("eBPF Capabilities")
    click.echo("=" * 50)
    click.echo()
    click.echo(f"Kernel version:    {caps.kernel_version}")
    click.echo(f"bpftrace:          {caps.bpftrace_version or 'not installed'}")
    # The check accepts uid 0 *or* CAP_SYS_ADMIN / CAP_BPF / CAP_PERFMON in
    # the effective capability set (parsed from /proc/self/status CapEff).
    click.echo(
        f"eBPF privilege:    {'yes' if caps.has_root_or_cap else 'no'} "
        "(root or CAP_BPF / CAP_PERFMON / CAP_SYS_ADMIN)"
    )
    click.echo(f"Overall available: {'yes' if caps.available else 'no'}")
    click.echo()

    click.echo("amdgpu tracepoints:")
    if caps.has_amdgpu_tracepoints:
        for tp in caps.amdgpu_tracepoints:
            click.echo(f"  - {tp}")
    else:
        click.echo("  (not accessible -- mount debugfs or run as root)")

    click.echo()
    click.echo("amdkfd tracepoints:")
    if caps.has_amdkfd_tracepoints:
        for tp in caps.amdkfd_tracepoints:
            click.echo(f"  - {tp}")
    else:
        click.echo("  (not accessible -- mount debugfs or run as root)")


@cli.command("ebpf-attach")
@click.option("--pid", "-p", type=int, default=None,
              help="PID of the running training process to attach to")
@click.option("--duration", "-d", default="60s",
              help="Tracing duration (e.g. 30s, 2m, 120s)")
@click.option("--output", "-o", default=None,
              help="Output JSON file for the diagnostic report")
@click.option("--tracers", "-t", default="queue,memory,race,dma,rccl",
              help="Comma-separated tracers to run (queue,memory,race,dma,rccl)")
@click.option("--nan-log", default=None,
              help="Path to sanitizer log file for NaN correlation")
@click.option("--race-window", default=100.0, type=float,
              help="Race detection window in microseconds")
@click.option("--dma-window", default=500.0, type=float,
              help="DMA overlap detection window in microseconds")
@click.option("--rccl-window", default=500.0, type=float,
              help="RCCL collective race detection window in microseconds")
def ebpf_attach(pid: Optional[int], duration: str, output: Optional[str],
                tracers: str, nan_log: Optional[str],
                race_window: float, dma_window: float, rccl_window: float):
    """Attach eBPF tracers to a running training process for NaN debugging.

    This command attaches kernel-level tracers to a live process without
    requiring a restart.  It collects queue dispatch, memory migration,
    stream race, DMA overlap, and RCCL collective data, then produces a
    diagnostic report.

    Requires bpftrace and root/CAP_BPF privileges.

    Examples:

        \b
        # Attach to PID 12345 for 60 seconds with all tracers
        python -m aorta.hw_queue_eval ebpf-attach --pid 12345

        \b
        # Quick 30-second capture with queue and race tracers only
        python -m aorta.hw_queue_eval ebpf-attach --pid 12345 -d 30s -t queue,race

        \b
        # Full capture with NaN log correlation
        python -m aorta.hw_queue_eval ebpf-attach --pid 12345 -d 120s \\
            --nan-log /path/to/training.log -o nan_diagnosis.json
    """
    import time as _time

    from aorta.hw_queue_eval.core.ebpf_tracer import check_ebpf_capabilities

    # Parse duration
    duration_sec = _parse_duration(duration)

    # Check prerequisites.  We do not require the current CLI process to
    # already have root/CAP_BPF here because the tracers default to
    # ``sudo=True`` and may elevate themselves.  Hard-fail only on the
    # things that no amount of sudo can fix.
    #
    # Missing tracepoints can be a *false negative* for an unprivileged
    # caller because debugfs is often unreadable without elevated
    # privileges, so the signal is only authoritative when we already
    # have root/CAP_BPF.  Otherwise downgrade to a warning and let the
    # tracers' own sudo-elevation try.
    caps = check_ebpf_capabilities()
    if caps.bpftrace_path is None:
        click.echo("Error: bpftrace is not installed.", err=True)
        sys.exit(1)
    if not (caps.has_amdgpu_tracepoints or caps.has_amdkfd_tracepoints):
        if caps.has_root_or_cap:
            click.echo(
                "Error: no amdgpu/amdkfd tracepoints found "
                "(is debugfs mounted and the amdgpu driver loaded?).",
                err=True,
            )
            sys.exit(1)
        click.echo(
            "Warning: no amdgpu/amdkfd tracepoints were visible to the "
            "current process. This may be a false negative if debugfs "
            "is unreadable without elevated privileges; tracer startup "
            "will still attempt sudo-based elevation.",
            err=True,
        )
    if not caps.has_root_or_cap:
        click.echo(
            "Note: current process lacks root/CAP_BPF; "
            "tracer startup will use sudo to elevate.",
            err=True,
        )

    tracer_names = [t.strip() for t in tracers.split(",")]
    valid_tracers = {"queue", "memory", "race", "dma", "rccl"}
    invalid = set(tracer_names) - valid_tracers
    if invalid:
        click.echo(f"Error: Unknown tracers: {invalid}", err=True)
        click.echo(f"Valid tracers: {', '.join(sorted(valid_tracers))}", err=True)
        sys.exit(1)

    click.echo("=" * 70)
    click.echo("eBPF LIVE ATTACH - NaN Debugging")
    click.echo("=" * 70)
    click.echo()
    click.echo(f"  Target PID:   {pid or 'all processes'}")
    click.echo(f"  Duration:     {duration_sec}s")
    click.echo(f"  Tracers:      {', '.join(tracer_names)}")
    if nan_log:
        click.echo(f"  NaN log:      {nan_log}")
    click.echo()

    # Instantiate tracers
    active_tracers: dict[str, Any] = {}

    try:
        if "queue" in tracer_names:
            from aorta.hw_queue_eval.core.ebpf_tracer import BPFQueueTracer
            active_tracers["queue"] = BPFQueueTracer(target_pid=pid)

        if "memory" in tracer_names:
            from aorta.hw_queue_eval.core.ebpf_memory_tracer import BPFMemoryTracer
            active_tracers["memory"] = BPFMemoryTracer(target_pid=pid)

        if "race" in tracer_names:
            from aorta.hw_queue_eval.core.ebpf_race_detector import BPFRaceDetector
            active_tracers["race"] = BPFRaceDetector(
                target_pid=pid, race_window_us=race_window,
            )

        if "dma" in tracer_names:
            from aorta.hw_queue_eval.core.ebpf_dma_tracer import BPFDMATracer
            active_tracers["dma"] = BPFDMATracer(
                target_pid=pid, overlap_window_us=dma_window,
            )

        if "rccl" in tracer_names:
            from aorta.hw_queue_eval.core.ebpf_rccl_tracer import BPFRCCLTracer
            active_tracers["rccl"] = BPFRCCLTracer(
                target_pid=pid, race_window_us=rccl_window,
            )

        # Start all tracers
        click.echo("Starting tracers...")
        started: List[str] = []
        for name, tracer in active_tracers.items():
            try:
                tracer.start()
                started.append(name)
                click.echo(f"  [{name}] started")
            except RuntimeError as e:
                click.echo(f"  [{name}] failed: {e}", err=True)

        if not started:
            click.echo("Error: No tracers started successfully.", err=True)
            sys.exit(1)

        click.echo()
        click.echo(f"Collecting data for {duration_sec} seconds...")
        click.echo("  (Press Ctrl+C to stop early)")
        click.echo()

        try:
            _time.sleep(duration_sec)
        except KeyboardInterrupt:
            click.echo()
            click.echo("Interrupted -- stopping tracers...")

        # Stop all tracers and collect results
        click.echo()
        click.echo("Stopping tracers and parsing results...")
        results: dict[str, Any] = {}
        for name in started:
            tracer = active_tracers[name]
            try:
                metric = tracer.stop()
                results[name] = metric
                click.echo(f"  [{name}] stopped")
            except Exception as e:
                # If ``stop()`` raises, the tracer's bpftrace subprocess
                # may still be attached; without an explicit cleanup it
                # would keep running and writing to the temp log after
                # the CLI exits.  Best-effort release.
                click.echo(f"  [{name}] stop failed: {e}", err=True)
                cleanup = getattr(tracer, "cleanup", None)
                if callable(cleanup):
                    try:
                        cleanup()
                        click.echo(f"  [{name}] cleanup completed")
                    except Exception as cleanup_error:
                        click.echo(
                            f"  [{name}] cleanup failed: {cleanup_error}",
                            err=True,
                        )

        # Print results
        click.echo()
        click.echo("-" * 70)
        click.echo("DIAGNOSTIC RESULTS")
        click.echo("-" * 70)
        click.echo()

        _print_attach_results(results)

        # NaN correlation -- only run when the user supplied a log;
        # ``_run_nan_correlation`` short-circuits on a falsy ``nan_log``
        # anyway, so the previous ``any(_has_issues(...))`` half of this
        # condition was dead code.
        nan_reports = None
        if nan_log:
            nan_reports = _run_nan_correlation(results, nan_log)

        # Export to JSON
        if output:
            _export_attach_results(results, nan_reports, output)
            click.echo(f"\nDiagnostic report saved to: {output}")

        click.echo()
        click.echo("=" * 70)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        for tracer in active_tracers.values():
            try:
                tracer.cleanup()
            except Exception:
                pass
        sys.exit(1)


def _parse_duration(duration: str) -> float:
    """Parse a duration string like '30s', '2m', '120s' into seconds."""
    duration = duration.strip().lower()
    if duration.endswith("m"):
        return float(duration[:-1]) * 60
    if duration.endswith("s"):
        return float(duration[:-1])
    return float(duration)


def _has_issues(name: str, metric: Any) -> bool:
    """Check if a tracer's metrics indicate potential issues."""
    if name == "race":
        return getattr(metric, "races_detected", 0) > 0
    if name == "dma":
        return getattr(metric, "overlaps_detected", 0) > 0
    if name == "rccl":
        return getattr(metric, "races_detected", 0) > 0
    if name == "memory":
        return getattr(metric, "total_evictions", 0) > 0
    return False


def _print_attach_results(results: dict) -> None:
    """Print diagnostic results from all tracers."""
    if "queue" in results:
        m = results["queue"]
        click.echo("QUEUE TRACER:")
        d = m.to_dict()
        click.echo(f"  Dispatches:    {d.get('total_dispatches', 0)}")
        click.echo(f"  Rings used:    {d.get('rings_used', [])}")
        click.echo(f"  Dispatch rate: {d.get('dispatch_rate_per_sec', 0):.0f} /sec")
        avg_gap = d.get("avg_inter_dispatch_gap_us", 0.0)
        p99_gap = d.get("p99_inter_dispatch_gap_us", 0.0)
        if avg_gap > 0:
            click.echo(f"  Avg dispatch gap: {avg_gap:.1f} us")
            click.echo(f"  P99 dispatch gap: {p99_gap:.1f} us")
        click.echo()

    if "memory" in results:
        m = results["memory"]
        d = m.to_dict()
        click.echo("MEMORY TRACER:")
        click.echo(f"  BO moves:      {d.get('total_bo_moves', 0)}")
        click.echo(f"  Evictions:     {d.get('total_evictions', 0)}")
        click.echo(f"  Restores:      {d.get('total_restores', 0)}")
        migration = d.get("migration_bytes", 0)
        if migration > 0:
            click.echo(f"  Migration:     {migration / (1024*1024):.1f} MB")
        if d.get("total_evictions", 0) > 0:
            click.echo("  WARNING: Evictions detected -- memory pressure may cause NaN")
        click.echo()

    if "race" in results:
        m = results["race"]
        d = m.to_dict()
        click.echo("RACE DETECTOR:")
        click.echo(f"  Submissions:   {d.get('total_submissions', 0)}")
        click.echo(f"  Races found:   {d.get('races_detected', 0)}")
        if d.get("races_detected", 0) > 0:
            click.echo(f"  Affected rings: {d.get('rings_with_races', [])}")
            click.echo("  ALERT: Stream races detected -- likely cause of NaN!")
            for i, rev in enumerate(d.get("race_events", [])[:5]):
                click.echo(f"    Race #{i+1}: ring={rev.get('ring')} "
                           f"gap={rev.get('gap_us', 0):.1f}us "
                           f"fence_gap={rev.get('fence_gap', 0)}")
        else:
            click.echo("  No races detected")
        click.echo()

    if "dma" in results:
        m = results["dma"]
        d = m.to_dict()
        click.echo("DMA/H2D TRACER:")
        click.echo(f"  BO moves:      {d.get('total_bo_moves', 0)}")
        click.echo(f"  Compute subs:  {d.get('total_compute_submits', 0)}")
        click.echo(f"  Overlaps:      {d.get('overlaps_detected', 0)}")
        if d.get("overlaps_detected", 0) > 0:
            click.echo(f"  Max overlap:   {d.get('max_overlap_us', 0):.1f} us")
            click.echo("  ALERT: H2D DMA-compute overlaps detected!")
        click.echo()

    if "rccl" in results:
        m = results["rccl"]
        d = m.to_dict()
        click.echo("RCCL TRACER:")
        click.echo(f"  Collective submissions: {d.get('collective_submissions', 0)}")
        click.echo(f"  Compute submissions:    {d.get('compute_submissions', 0)}")
        click.echo(f"  Collective rings:       {d.get('collective_rings', [])}")
        click.echo(f"  Compute rings:          {d.get('compute_rings', [])}")
        click.echo(f"  Races detected:         {d.get('races_detected', 0)}")
        cross = d.get("cross_ring_observations", 0)
        if cross:
            click.echo(
                f"  Cross-ring observations: {cross} "
                "(within window but on different rings; not confirmed races)"
            )
        if d.get("races_detected", 0) > 0:
            click.echo("  ALERT: Collective-compute races detected!")
        click.echo()


def _run_nan_correlation(results: dict, nan_log: Optional[str]) -> Optional[list]:
    """Run NaN correlation against the supplied sanitizer log.

    Requires ``nan_log`` to be set; if it is falsy this function returns
    ``None`` without doing any correlation.  An earlier "issues were
    found" auto-trigger was removed because the ``NaNCorrelator`` API
    needs explicit NaN events (timestamp + rank + step) to anchor the
    correlation window, and tracer-level "issues" alone don't provide
    that anchor.

    Returns a list of per-NaN correlation reports, or ``None`` when no
    NaN events were available to correlate.
    """
    from aorta.hw_queue_eval.core.ebpf_nan_correlator import NaNCorrelator

    if not nan_log:
        return None

    correlator = NaNCorrelator(window_ms=100.0)
    count = correlator.add_nan_events_from_log(nan_log)
    if count > 0:
        click.echo(f"NaN CORRELATION: Parsed {count} NaN events from log")
    else:
        click.echo(f"NaN CORRELATION: No NaN events found in {nan_log}")
        return None

    if "queue" in results:
        correlator.set_queue_events(results["queue"].events)
    if "memory" in results:
        correlator.set_memory_events(results["memory"].events)
    if "race" in results:
        correlator.set_race_events(results["race"].race_events)
    if "dma" in results:
        correlator.set_dma_overlaps(results["dma"].overlap_events)
    if "rccl" in results:
        correlator.set_collective_races(results["rccl"].race_events)

    reports = correlator.correlate()

    if reports:
        click.echo()
        click.echo("NaN CORRELATION RESULTS:")
        for i, r in enumerate(reports[:10]):
            d = r.to_dict()
            click.echo(f"  NaN #{i+1} (step={d['nan_step']}, rank={d['nan_rank']}):")
            click.echo(f"    Diagnosis:  {d['diagnosis']}")
            click.echo(f"    Confidence: {d['confidence']}")
            ke = d["kernel_events"]
            if ke["race_events"] > 0:
                click.echo(f"    Stream races in window: {ke['race_events']}")
            if ke["dma_overlaps"] > 0:
                click.echo(f"    DMA overlaps in window: {ke['dma_overlaps']}")
            if ke["collective_races"] > 0:
                click.echo(f"    Collective races:       {ke['collective_races']}")
            if ke["evictions"] > 0:
                click.echo(f"    Evictions in window:    {ke['evictions']}")

    return [r.to_dict() for r in reports]


def _export_attach_results(results: dict, nan_reports: Optional[list],
                           filepath: str) -> None:
    """Export all diagnostic results to a JSON file."""
    data: dict[str, Any] = {}

    for name, metric in results.items():
        data[f"{name}_metrics"] = metric.to_dict()

    if nan_reports:
        data["nan_correlation"] = nan_reports

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
