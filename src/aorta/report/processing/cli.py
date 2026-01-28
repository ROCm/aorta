"""CLI commands for data processing utilities.

This module provides the 'process' command group with subcommands:
  - gpu-timeline: Process GPU timeline data from TraceLens reports
  - comms: Process NCCL communication data
  - gemm-variance: Enhance GEMM variance with timestamps
"""

import click
from pathlib import Path


@click.group()
@click.pass_context
def process(ctx):
    """Data processing utilities.

    \b
    Commands:
      gpu-timeline   - Process GPU timeline data from TraceLens reports
      comms          - Process communication data
      gemm-variance  - Enhance GEMM variance with timestamps
    """
    pass


@process.command("gpu-timeline")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--mode", type=click.Choice(["auto", "single", "sweep"]), default="auto",
              help="Processing mode: auto-detect, single config, or sweep")
@click.option("--geo-mean", is_flag=True, help="Use geometric mean instead of arithmetic mean")
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.pass_context
def process_gpu_timeline(ctx, input_dir, mode, geo_mean, output):
    """Process GPU timeline data from TraceLens reports.

    INPUT_DIR: Path to reports directory or sweep directory.

    Supports both single-config and sweep directory structures.
    Auto-detects the structure by default.

    \b
    Single mode: Processes perf_rank*.xlsx files from individual_reports/
    Sweep mode: Processes perf_*ch_rank*.xlsx files from tracelens_analysis/

    \b
    Examples:
      aorta-report process gpu-timeline /path/to/reports
      aorta-report process gpu-timeline /path/to/individual_reports --mode single
      aorta-report process gpu-timeline /path/to/sweep --mode sweep --geo-mean
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    input_path = Path(input_dir)

    # Auto-detect mode
    if mode == "auto":
        # Check for sweep structure (tracelens_analysis with thread directories)
        tracelens_dir = input_path / "tracelens_analysis"
        if tracelens_dir.exists():
            thread_dirs = [d for d in tracelens_dir.iterdir() if d.is_dir() and "thread" in d.name]
            if thread_dirs:
                mode = "sweep"
            else:
                mode = "single"
        elif input_path.name == "individual_reports" or list(input_path.glob("perf_rank*.xlsx")):
            mode = "single"
        elif list(input_path.glob("perf_*ch_rank*.xlsx")):
            mode = "sweep"
        else:
            raise click.ClickException(
                "Could not auto-detect mode. Please specify --mode single or --mode sweep"
            )

        if verbose:
            click.echo(f"Auto-detected mode: {mode}")

    try:
        if mode == "single":
            from . import process_single_config
            output_path = process_single_config(
                reports_dir=input_path,
                use_geo_mean=geo_mean,
                output_path=Path(output) if output else None,
                verbose=verbose,
            )
        else:  # sweep
            from . import process_sweep_config
            output_path = process_sweep_config(
                sweep_dir=input_path,
                use_geo_mean=geo_mean,
                output_path=Path(output) if output else None,
                verbose=verbose,
            )

        if not quiet and output_path:
            click.echo(f"\nProcessing complete: {output_path}")
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


@process.command("comms")
@click.argument("sweep_dir", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.pass_context
def process_comms(ctx, sweep_dir, output):
    """Process NCCL communication data from collective reports.

    SWEEP_DIR: Path to sweep directory containing tracelens_analysis/

    Reads nccl_summary_implicit_sync sheet from collective_*.xlsx files,
    combines data across all configurations, and generates master files.

    \b
    Output files:
      - nccl_master_all_configs.xlsx (for pivot tables)
      - nccl_master_all_configs.csv (for pandas/scripts)

    \b
    Examples:
      aorta-report process comms /path/to/sweep
      aorta-report process comms /path/to/sweep -o ./nccl_analysis/
    """
    from . import process_nccl_data

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        excel_path, csv_path = process_nccl_data(
            sweep_dir=Path(sweep_dir),
            output_dir=Path(output) if output else None,
            verbose=verbose,
        )
        if not quiet and excel_path:
            click.echo(f"\nProcessing complete:")
            click.echo(f"  Excel: {excel_path}")
            click.echo(f"  CSV: {csv_path}")
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


@process.command("gemm-variance")
@click.argument("input_csv", type=click.Path(exists=True))
@click.option("--base-path", required=True, type=click.Path(exists=True),
              help="Base path to sweep directory containing trace files")
@click.option("--tolerance", default=0.01, type=float,
              help="Duration matching tolerance as fraction (default: 0.01 = 1%)")
@click.option("-o", "--output", type=click.Path(), help="Output CSV file")
@click.pass_context
def process_gemm_variance(ctx, input_csv, base_path, tolerance, output):
    """Enhance GEMM variance CSV with kernel timestamps.

    INPUT_CSV: CSV file with GEMM variance data (from 'analyze gemm' command).

    For each row, finds the corresponding trace file and extracts timestamps
    for the kernel instances with minimum and maximum durations.

    \b
    Added columns:
      - min_duration_timestamp_ms: When shortest instance occurred
      - max_duration_timestamp_ms: When longest instance occurred
      - time_between_min_max_ms: Time difference between occurrences

    \b
    Examples:
      aorta-report process gemm-variance ./gemm_variance.csv --base-path /path/to/sweep
      aorta-report process gemm-variance ./variance.csv --base-path /path/to/sweep \\
          --tolerance 0.02 -o ./enhanced.csv
    """
    from . import enhance_gemm_variance

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        output_path = enhance_gemm_variance(
            input_csv=Path(input_csv),
            base_path=Path(base_path),
            output_csv=Path(output) if output else None,
            tolerance=tolerance,
            verbose=verbose,
        )
        if not quiet and output_path:
            click.echo(f"\nProcessing complete: {output_path}")
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))

