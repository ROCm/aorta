"""
Command-line interface for the hardware queue evaluation framework.

Example usage:
    # Run single workload with specific stream count
    python -m aorta.hw_queue_eval run hetero_kernels --streams 8

    # Run with PyTorch profiler (generates Chrome trace and TensorBoard logs)
    python -m aorta.hw_queue_eval run hetero_kernels --streams 8 --profile
    python -m aorta.hw_queue_eval run moe --streams 16 --profile --profile-dir traces/

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
from typing import List, Optional

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


def get_workload_instance(name: str, **kwargs):
    """Get a workload instance by name."""
    from aorta.hw_queue_eval.workloads.registry import get_workload
    return get_workload(name, **kwargs)


def list_available_workloads():
    """List all available workloads."""
    from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry

    try:
        from aorta.hw_queue_eval.workloads import distributed, inference, latency_sensitive, pipeline
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
def run(workload: str, streams: int, iterations: int, warmup: int,
        output: Optional[str], device: str, sync_mode: str, quiet: bool,
        profile: bool, profile_dir: str):
    """Run a single workload evaluation.

    WORKLOAD: Name of the workload to run (e.g., hetero_kernels, fsdp_tp)
    """
    from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness
    from aorta.hw_queue_eval.core.torch_profiler import TorchProfilerWrapper, generate_profile_summary
    from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry

    try:
        # Get workload
        wl = get_workload_instance(workload)
        info = WorkloadRegistry.get_info(workload)

        # Check stream count compatibility
        if not wl.supports_stream_count(streams):
            click.echo(
                f"Error: Workload {workload} supports {wl.min_streams}-{wl.max_streams} streams",
                err=True
            )
            sys.exit(1)

        # Print header with workload info
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
        )
        harness = StreamHarness(config)

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

        # Print results with interpretation
        click.echo("-" * 70)
        click.echo("RESULTS")
        click.echo("-" * 70)
        click.echo()

        click.echo("THROUGHPUT:")
        click.echo(f"  {result.throughput:,.2f} {result.throughput_unit}")
        click.echo(f"  (Higher is better - measures how much work completed per second)")
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
            click.echo(f"    This may indicate queue contention or scheduling issues")
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
                click.echo(f"  Significant queue switch overhead detected")
                click.echo(f"    This suggests hardware queue contention at {streams} streams")
            elif switch_overhead > 0.01:
                click.echo(f"  Moderate queue switch overhead")
            else:
                click.echo(f"  Minimal queue switch overhead")
        else:
            click.echo(f"  (Not enough data to estimate switch overhead)")
        click.echo()

        click.echo("TIMING:")
        click.echo(f"  Total measurement time: {result.total_time_ms:.2f} ms ({result.total_time_ms/1000:.2f} sec)")
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
                click.echo(f"    View: Open chrome://tracing and load the JSON file")
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
    click.echo(f"  1. Run a sweep to find the optimal stream count:")
    click.echo(f"     python -m aorta.hw_queue_eval sweep {workload} -s 1,2,4,8,16,32")
    click.echo(f"  2. Compare results before/after runtime changes:")
    click.echo(f"     python -m aorta.hw_queue_eval compare -b baseline.json -t test.json")


@cli.command()
@click.argument("workload")
@click.option("--streams", "-s", default="1,2,4,8,16,32",
              help="Comma-separated stream counts to test")
@click.option("--iterations", "-i", default=100, help="Measurement iterations per config")
@click.option("--warmup", "-w", default=10, help="Warmup iterations")
@click.option("--output", "-o", default=None, help="Output JSON file")
@click.option("--device", "-d", default="cuda:0", help="Target device")
def sweep(workload: str, streams: str, iterations: int, warmup: int,
          output: Optional[str], device: str):
    """Run workload across multiple stream counts.

    WORKLOAD: Name of the workload to sweep
    """
    from aorta.hw_queue_eval.core.harness import (
        HarnessConfig, StreamHarness, analyze_sweep_results,
        format_results_table, save_sweep_results
    )

    # Parse stream counts
    stream_counts = [int(s.strip()) for s in streams.split(",")]

    click.echo(f"Sweeping workload: {workload}")
    click.echo(f"  Stream counts: {stream_counts}")
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
def run_priority(priority: str, streams: str, iterations: int,
                 output_dir: str, device: str):
    """Run all workloads of a given priority level.

    PRIORITY: Priority level (P0, P1, P2, P3, or 'all')
    """
    from aorta.hw_queue_eval.core.harness import HarnessConfig, StreamHarness, save_sweep_results

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

    click.echo(f"Running {len(workloads)} workloads at priority {priority}")
    click.echo(f"Stream counts: {stream_counts}")
    click.echo(f"Output directory: {output_path}")
    click.echo()

    all_results = {}
    failed = []

    for workload_name in workloads:
        click.echo(f"[{workload_name}]")

        try:
            wl = get_workload_instance(workload_name)
            valid_counts = [c for c in stream_counts if wl.supports_stream_count(c)]

            if not valid_counts:
                click.echo(f"  Skipped (no valid stream counts)")
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

    # Import workload modules to register them
    try:
        from aorta.hw_queue_eval.workloads import distributed, inference, latency_sensitive, pipeline
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


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
