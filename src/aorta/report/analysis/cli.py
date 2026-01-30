"""CLI commands for TraceLens analysis.

This module provides the 'analyze' command group with subcommands:
  - single: Analyze a single configuration trace directory
  - sweep: Analyze a sweep directory with multiple configurations
  - gemm: Analyze GEMM kernels from TraceLens reports
"""

import click
from pathlib import Path


@click.group()
@click.pass_context
def analyze(ctx):
    """Run TraceLens analysis on traces.

    \b
    Commands:
      single  - Analyze a single configuration trace directory
      sweep   - Analyze a sweep directory with multiple configurations
      gemm    - Analyze GEMM kernels from TraceLens reports
    """
    pass


@analyze.command("single")
@click.argument("trace_dir", type=click.Path(exists=True))
@click.option("--individual-only", is_flag=True, help="Generate only individual reports")
@click.option("--collective-only", is_flag=True, help="Generate only collective report")
@click.option("--geo-mean", is_flag=True, help="Use geometric mean for timeline aggregation")
@click.option("--short-kernel-threshold", default=50, type=int,
              help="Threshold for short kernel study (microseconds)")
@click.option("--topk-ops", default=100, type=int,
              help="Number of top operations to include")
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.pass_context
def analyze_single(ctx, trace_dir, individual_only, collective_only, geo_mean,
                   short_kernel_threshold, topk_ops, output):
    """Analyze a single configuration trace directory.

    TRACE_DIR: Path to the trace directory containing rank subdirectories.

    \b
    Examples:
      aorta-report analyze single /path/to/traces
      aorta-report analyze single /path/to/traces --individual-only
      aorta-report analyze single /path/to/traces -o ./results
    """
    from . import analyze_single_config

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    run_individual = not collective_only
    run_collective = not individual_only

    try:
        results = analyze_single_config(
            input_dir=Path(trace_dir),
            output_dir=Path(output) if output else None,
            run_individual=run_individual,
            run_collective=run_collective,
            aggregate_timeline=run_individual,
            use_geo_mean=geo_mean,
            short_kernel_threshold_us=short_kernel_threshold,
            topk_ops=topk_ops,
            verbose=verbose,
        )
        if not quiet:
            click.echo(f"\nAnalysis complete: {results['output_dir']}")
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


@analyze.command("sweep")
@click.argument("sweep_dir", type=click.Path(exists=True))
@click.option("--skip-tracelens", is_flag=True,
              help="Skip TraceLens analysis, only aggregate existing reports")
@click.option("--geo-mean", is_flag=True, help="Use geometric mean instead of arithmetic mean")
@click.option("--short-kernel-threshold", default=50, type=int,
              help="Threshold for short kernel study (microseconds)")
@click.option("--topk-ops", default=100, type=int,
              help="Number of top operations to include")
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.pass_context
def analyze_sweep(ctx, sweep_dir, skip_tracelens, geo_mean, short_kernel_threshold,
                  topk_ops, output):
    """Analyze a sweep directory with multiple configurations.

    SWEEP_DIR: Path to the sweep directory containing thread/channel subdirectories.

    By default, runs TraceLens analysis on all configurations first, then
    aggregates the results. Use --skip-tracelens to only aggregate existing reports.

    \b
    Expected directory structure:
      sweep_dir/
      ├── 256thread/
      │   ├── nccl_28channels/
      │   │   └── torch_profiler/rank*/
      │   └── nccl_42channels/
      └── 512thread/
          └── ...

    \b
    Examples:
      # Run TraceLens + aggregate (default)
      aorta-report analyze sweep /path/to/sweep_20251124

      # Only aggregate existing reports
      aorta-report analyze sweep /path/to/sweep --skip-tracelens

      # With geometric mean aggregation
      aorta-report analyze sweep /path/to/sweep --geo-mean
    """
    from . import analyze_sweep_config

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        output_path = analyze_sweep_config(
            sweep_dir=Path(sweep_dir),
            output_dir=Path(output) if output else None,
            use_geo_mean=geo_mean,
            skip_tracelens=skip_tracelens,
            short_kernel_threshold_us=short_kernel_threshold,
            topk_ops=topk_ops,
            verbose=verbose,
        )
        if not quiet and output_path:
            click.echo(f"\nAnalysis complete: {output_path}")
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


@analyze.command("gemm")
@click.argument("reports_dir", type=click.Path(exists=True))
@click.option("--threads", "-t", multiple=True, type=int, default=(256, 512),
              help="Thread configurations to analyze (can be specified multiple times)")
@click.option("--channels", "-c", multiple=True, type=int, default=(28, 42, 56, 70),
              help="Channel configurations to analyze (can be specified multiple times)")
@click.option("--ranks", "-r", multiple=True, type=int,
              help="Ranks to analyze (default: 0-7)")
@click.option("--top-k", default=5, type=int, help="Number of top kernels to extract per file")
@click.option("-o", "--output", type=click.Path(),
              default="top5_gemm_kernels_time_variance.csv", help="Output CSV file")
@click.pass_context
def analyze_gemm(ctx, reports_dir, threads, channels, ranks, top_k, output):
    """Analyze GEMM kernels from TraceLens reports.

    REPORTS_DIR: Path to tracelens_analysis directory containing
    {threads}thread/individual_reports/ subdirectories.

    \b
    Examples:
      aorta-report analyze gemm /path/to/tracelens_analysis
      aorta-report analyze gemm /path/to/reports --top-k 10 -o gemm_analysis.csv
      aorta-report analyze gemm /path/to/reports -t 256 -t 512 -c 28 -c 42
    """
    from . import analyze_gemm_reports

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    # Convert tuples to lists, use defaults if not specified
    threads_list = list(threads) if threads else [256, 512]
    channels_list = list(channels) if channels else [28, 42, 56, 70]
    ranks_list = list(ranks) if ranks else list(range(8))

    try:
        output_path = analyze_gemm_reports(
            base_path=Path(reports_dir),
            threads=threads_list,
            channels=channels_list,
            ranks=ranks_list,
            top_k=top_k,
            output_file=output,
            verbose=verbose,
        )
        if not quiet and output_path:
            click.echo(f"\nAnalysis complete: {output_path}")
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))

