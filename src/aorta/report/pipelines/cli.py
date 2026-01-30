"""CLI commands for complete analysis pipelines.

This module provides the 'pipeline' command group with subcommands:
  - summary: Run complete summary analysis pipeline (GPU + NCCL)
  - gemm: Run GEMM variance analysis pipeline
"""

import click
from pathlib import Path


@click.group()
@click.pass_context
def pipeline(ctx):
    """Run complete analysis pipelines.

    \b
    Commands:
      summary  - Run complete summary analysis pipeline (GPU + NCCL)
      gemm     - Run GEMM variance analysis pipeline
    """
    pass


@pipeline.command("summary")
@click.option(
    "-b", "--baseline", required=True, type=click.Path(exists=True), help="Baseline trace directory"
)
@click.option(
    "-t", "--test", required=True, type=click.Path(exists=True), help="Test trace directory"
)
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output directory for results"
)
@click.option("--baseline-label", default=None, help="Label for baseline (default: directory name)")
@click.option("--test-label", default=None, help="Label for test (default: directory name)")
@click.option("--skip-tracelens", is_flag=True, help="Skip TraceLens analysis (if already done)")
@click.option(
    "--gpu-timeline/--no-gpu-timeline", default=True, help="Enable/disable GPU timeline comparison"
)
@click.option(
    "--collective/--no-collective", default=True, help="Enable/disable collective comparison"
)
@click.option(
    "--final-report/--no-final-report", default=True, help="Enable/disable final Excel report"
)
@click.option("--plots/--no-plots", default=True, help="Enable/disable plot generation")
@click.option("--html/--no-html", default=True, help="Enable/disable HTML report generation")
@click.pass_context
def pipeline_summary(
    ctx,
    baseline,
    test,
    output,
    baseline_label,
    test_label,
    skip_tracelens,
    gpu_timeline,
    collective,
    final_report,
    plots,
    html,
):
    """Run complete summary analysis pipeline.

    Orchestrates the full TraceLens analysis workflow:

    \b
    1. TraceLens Analysis (optional, skip with --skip-tracelens)
    2. Process GPU timelines
    3. Compare GPU timelines (baseline vs test)
    4. Compare collective/NCCL metrics
    5. Generate final Excel report
    6. Generate visualization plots
    7. Generate HTML report

    \b
    Examples:
      # Full pipeline
      aorta-report pipeline summary \\
          -b /path/to/baseline -t /path/to/test -o /path/to/output

      # Skip TraceLens (already done)
      aorta-report pipeline summary \\
          -b /path/to/baseline -t /path/to/test -o /path/to/output \\
          --skip-tracelens

      # Only GPU timeline comparison
      aorta-report pipeline summary \\
          -b /path/to/baseline -t /path/to/test -o /path/to/output \\
          --no-collective --no-final-report --no-plots --no-html
    """
    from . import run_summary_pipeline, SummaryPipelineConfig

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    # Resolve paths to absolute to ensure all outputs go to correct location
    baseline_path = Path(baseline).resolve()
    test_path = Path(test).resolve()
    output_path = Path(output).resolve()

    config = SummaryPipelineConfig(
        baseline_path=baseline_path,
        test_path=test_path,
        output_dir=output_path,
        baseline_label=baseline_label,
        test_label=test_label,
        skip_tracelens=skip_tracelens,
        gpu_timeline=gpu_timeline,
        collective=collective,
        final_report=final_report,
        plots=plots,
        html=html,
        verbose=verbose,
    )

    if not quiet:
        click.echo("=" * 60)
        click.echo("SUMMARY ANALYSIS PIPELINE")
        click.echo("=" * 60)
        click.echo(f"Baseline: {baseline}")
        click.echo(f"Test: {test}")
        click.echo(f"Output: {output}")
        click.echo(f"Labels: {baseline_label or '(auto)'} vs {test_label or '(auto)'}")
        click.echo(f"Options: skip_tracelens={skip_tracelens}, gpu_timeline={gpu_timeline}")
        click.echo(f"         collective={collective}, final_report={final_report}")
        click.echo(f"         plots={plots}, html={html}")

    result = run_summary_pipeline(config)

    if not quiet:
        click.echo("\n" + "=" * 60)
        click.echo("PIPELINE COMPLETE!" if result.success else "PIPELINE FAILED!")
        click.echo("=" * 60)

        if result.steps_completed:
            click.echo("\nSteps completed:")
            for step in result.steps_completed:
                click.echo(f"  ✓ {step}")

        if result.steps_skipped:
            click.echo("\nSteps skipped:")
            for step in result.steps_skipped:
                click.echo(f"  - {step}")

        if result.errors:
            click.echo("\nErrors:")
            for err in result.errors:
                click.echo(f"  ✗ {err}")

        if result.files_generated:
            click.echo(f"\nOutput directory: {result.output_dir}")
            click.echo("\nGenerated files:")
            for name, path in result.files_generated.items():
                if isinstance(path, Path):
                    click.echo(f"  - {path}")

    if not result.success:
        raise click.ClickException("Pipeline failed")


@pipeline.command("gemm")
@click.option(
    "--sweep-dir",
    required=True,
    type=click.Path(exists=True),
    help="Sweep directory with thread/channel subdirectories",
)
@click.option(
    "-o", "--output", required=True, type=click.Path(), help="Output directory for results"
)
@click.option(
    "--skip-tracelens",
    is_flag=True,
    help="Skip TraceLens analysis, use existing reports",
)
@click.option(
    "--short-kernel-threshold",
    default=50,
    type=int,
    help="Threshold for short kernel study in microseconds (default: 50)",
)
@click.option(
    "--topk-ops",
    default=100,
    type=int,
    help="Number of top operations to include in TraceLens analysis (default: 100)",
)
@click.option("--top-k", default=5, type=int, help="Number of top GEMM kernels to extract (default: 5)")
@click.option(
    "--threads",
    "-t",
    multiple=True,
    type=int,
    default=(256, 512),
    help="Thread configurations (can specify multiple)",
)
@click.option(
    "--channels",
    "-c",
    multiple=True,
    type=int,
    default=(28, 42, 56, 70),
    help="Channel configurations (can specify multiple)",
)
@click.option(
    "--timestamps/--no-timestamps", default=True, help="Enhance with timestamps (default: True)"
)
@click.option("--plots/--no-plots", default=True, help="Generate plots (default: True)")
@click.option("--html/--no-html", default=True, help="Generate HTML report (default: True)")
@click.pass_context
def pipeline_gemm(ctx, sweep_dir, output, skip_tracelens, short_kernel_threshold, topk_ops,
                  top_k, threads, channels, timestamps, plots, html):
    """Run GEMM variance analysis pipeline.

    By default, runs TraceLens on all configurations first, then analyzes GEMM kernels.
    Use --skip-tracelens to use existing TraceLens reports.

    \b
    Steps:
      0. Run TraceLens on sweep (default, skip with --skip-tracelens)
      1. Analyze GEMM reports to extract top-K kernels with highest variance
      2. Enhance with timestamps (optional)
      3. Generate variance plots (optional)
      4. Generate HTML report (optional)

    \b
    Examples:
      # Full pipeline (runs TraceLens + GEMM analysis)
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o /path/to/output

      # Skip TraceLens, use existing reports
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o ./output --skip-tracelens

      # Custom top-k
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o ./output --top-k 10

      # Skip plots and HTML
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o ./output --no-plots --no-html

      # Custom thread/channel configurations
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o ./output \\
          -t 256 -t 512 -c 28 -c 42 -c 56 -c 70
    """
    from . import run_gemm_pipeline, GemmPipelineConfig

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    # Resolve paths to absolute to ensure all outputs go to correct location
    sweep_path = Path(sweep_dir).resolve()
    output_path = Path(output).resolve()

    config = GemmPipelineConfig(
        sweep_dir=sweep_path,
        output_dir=output_path,
        top_k=top_k,
        threads=list(threads),
        channels=list(channels),
        skip_tracelens=skip_tracelens,
        short_kernel_threshold_us=short_kernel_threshold,
        topk_ops=topk_ops,
        timestamps=timestamps,
        plots=plots,
        html=html,
        verbose=verbose,
    )

    if not quiet:
        click.echo("=" * 60)
        click.echo("GEMM VARIANCE ANALYSIS PIPELINE")
        click.echo("=" * 60)
        click.echo(f"Sweep dir: {sweep_dir}")
        click.echo(f"Output: {output}")
        click.echo(f"TraceLens: {'skip' if skip_tracelens else 'run'}")
        click.echo(f"Top-K GEMM kernels: {top_k}")
        click.echo(f"Threads: {list(threads)}")
        click.echo(f"Channels: {list(channels)}")
        click.echo(f"Options: timestamps={timestamps}, plots={plots}, html={html}")

    result = run_gemm_pipeline(config)

    if not quiet:
        click.echo("\n" + "=" * 60)
        click.echo("PIPELINE COMPLETE!" if result.success else "PIPELINE FAILED!")
        click.echo("=" * 60)

        if result.steps_completed:
            click.echo("\nSteps completed:")
            for step in result.steps_completed:
                click.echo(f"  ✓ {step}")

        if result.steps_skipped:
            click.echo("\nSteps skipped:")
            for step in result.steps_skipped:
                click.echo(f"  - {step}")

        if result.errors:
            click.echo("\nErrors:")
            for err in result.errors:
                click.echo(f"  ✗ {err}")

        click.echo(f"\nOutput directory: {result.output_dir}")
        click.echo("\nGenerated files:")
        if result.csv_path:
            click.echo(f"  - {result.csv_path}")
        if result.csv_with_timestamps_path:
            click.echo(f"  - {result.csv_with_timestamps_path}")
        if result.plots_dir:
            click.echo(f"  - {result.plots_dir}/ (5 plots)")
        if result.html_path:
            click.echo(f"  - {result.html_path}")

    if not result.success:
        raise click.ClickException("Pipeline failed")
