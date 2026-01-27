"""
aorta-report CLI - Unified interface for TraceLens analysis and report generation.

Usage:
    aorta-report --help
    aorta-report analyze --help
    aorta-report compare --help
    aorta-report generate --help
    aorta-report process --help
    aorta-report pipeline --help
"""

import click

from . import __version__


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.version_option(version=__version__, prog_name="aorta-report")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--quiet", is_flag=True, help="Suppress non-error output")
@click.pass_context
def cli(ctx, verbose, quiet):
    """aorta-report: Unified CLI for TraceLens analysis and report generation.

    Analyze PyTorch profiler traces, process GPU timeline data,
    and generate comprehensive comparison reports.

    \b
    Command Groups:
      analyze   - Run TraceLens analysis on traces
      compare   - Compare traces and reports
      generate  - Generate reports (HTML, Excel, plots)
      process   - Data processing utilities
      pipeline  - Run complete analysis pipelines
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


# =============================================================================
# ANALYZE Group
# =============================================================================


@cli.group()
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
    from pathlib import Path
    from .analysis import analyze_single_config

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
@click.option("--geo-mean", is_flag=True, help="Use geometric mean instead of arithmetic mean")
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.pass_context
def analyze_sweep(ctx, sweep_dir, geo_mean, output):
    """Analyze a sweep directory with multiple configurations.

    SWEEP_DIR: Path to the sweep directory containing tracelens_analysis/
    with multiple thread/channel configs.

    \b
    Examples:
      aorta-report analyze sweep /path/to/sweep_20251124
      aorta-report analyze sweep /path/to/sweep --geo-mean
    """
    from pathlib import Path
    from .analysis import analyze_sweep_config

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        output_path = analyze_sweep_config(
            sweep_dir=Path(sweep_dir),
            output_dir=Path(output) if output else None,
            use_geo_mean=geo_mean,
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
    from pathlib import Path
    from .analysis import analyze_gemm_reports

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


# =============================================================================
# COMPARE Group
# =============================================================================


@cli.group()
@click.pass_context
def compare(ctx):
    """Compare baseline and test TraceLens reports.

    \b
    Supported comparison types:
      gpu_timeline  - Compare GPU timeline reports
      collective    - Compare collective/NCCL reports
    """
    pass


@compare.command("gpu_timeline")
@click.option("-b", "--baseline", required=True, type=click.Path(exists=True),
              help="Path to baseline gpu_timeline_summary_mean.xlsx")
@click.option("-t", "--test", required=True, type=click.Path(exists=True),
              help="Path to test gpu_timeline_summary_mean.xlsx")
@click.option("--baseline-label", default=None,
              help="Label for baseline (default: extracted from path)")
@click.option("--test-label", default=None,
              help="Label for test (default: extracted from path)")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output Excel file path")
@click.pass_context
def compare_gpu_timeline(ctx, baseline, test, baseline_label, test_label, output):
    """Compare two GPU timeline reports.

    Combines baseline and test files, then adds comparison sheets
    with diff, percent_change, and status columns.

    \b
    Output sheets:
      - Summary, All_Ranks_Combined, Per_Rank_* (combined data)
      - Comparison_By_Rank (per-rank comparison)
      - Summary_Comparison (overall comparison)

    \b
    Examples:
      aorta-report compare gpu_timeline \\
          -b baseline/gpu_timeline_summary_mean.xlsx \\
          -t test/gpu_timeline_summary_mean.xlsx \\
          -o comparison.xlsx

      aorta-report compare gpu_timeline \\
          -b baseline/gpu.xlsx -t test/gpu.xlsx \\
          --baseline-label "ROCm 6.0" --test-label "ROCm 7.0" \\
          -o comparison.xlsx
    """
    from pathlib import Path
    from .comparison import (
        combine_excel_files,
        add_gpu_timeline_comparison,
        save_with_formatting,
    )
    from .comparison.combine import extract_label_from_path

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    baseline_path = Path(baseline)
    test_path = Path(test)
    output_path = Path(output)

    # Extract labels from paths if not provided
    if baseline_label is None:
        baseline_label = extract_label_from_path(baseline_path, "baseline")
    if test_label is None:
        test_label = extract_label_from_path(test_path, "test")

    if not quiet:
        click.echo("=" * 60)
        click.echo("GPU Timeline Comparison")
        click.echo("=" * 60)
        click.echo(f"Baseline: {baseline_path}")
        click.echo(f"Test: {test_path}")
        click.echo(f"Baseline label: {baseline_label}")
        click.echo(f"Test label: {test_label}")

    try:
        # Step 1: Combine Excel files
        if not quiet:
            click.echo("\nStep 1: Combining Excel files")
        combined = combine_excel_files(
            baseline_path,
            test_path,
            baseline_label,
            test_label,
            verbose=verbose,
        )

        # Step 2: Add comparison sheets
        if not quiet:
            click.echo("\nStep 2: Adding comparison sheets")
        result = add_gpu_timeline_comparison(
            combined,
            baseline_label,
            test_label,
            verbose=verbose,
        )

        # Step 3: Save with formatting
        if not quiet:
            click.echo("\nStep 3: Saving with formatting")
        format_columns = {
            "Comparison_By_Rank": ["percent_change"],
            "Summary_Comparison": ["percent_change"],
        }
        save_with_formatting(result, output_path, format_columns, verbose=verbose)

        if not quiet:
            click.echo("\n" + "=" * 60)
            click.echo("Comparison Complete!")
            click.echo("=" * 60)
            click.echo(f"\nOutput: {output_path}")
            click.echo("\nSheets:")
            for sheet_name in result.keys():
                click.echo(f"  - {sheet_name}")
            click.echo("\npercent_change interpretation:")
            click.echo("  Positive = test is faster/better")
            click.echo("  Negative = test is slower/worse")

    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


@compare.command("collective")
@click.option("-b", "--baseline", required=True, type=click.Path(exists=True),
              help="Path to baseline collective_all_ranks.xlsx")
@click.option("-t", "--test", required=True, type=click.Path(exists=True),
              help="Path to test collective_all_ranks.xlsx")
@click.option("--baseline-label", default=None,
              help="Label for baseline (default: extracted from path)")
@click.option("--test-label", default=None,
              help="Label for test (default: extracted from path)")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output Excel file path")
@click.pass_context
def compare_collective(ctx, baseline, test, baseline_label, test_label, output):
    """Compare two collective/NCCL reports.

    Combines baseline and test files, then adds comparison sheets
    for NCCL summary data with latency and bandwidth metrics.

    \b
    Output sheets:
      - nccl_summary_* (combined summary sheets)
      - nccl_implicit_sync_cmp (comparison)
      - nccl_long_cmp (comparison)

    \b
    Examples:
      aorta-report compare collective \\
          -b baseline/collective_all_ranks.xlsx \\
          -t test/collective_all_ranks.xlsx \\
          -o collective_comparison.xlsx

      aorta-report compare collective \\
          -b baseline/coll.xlsx -t test/coll.xlsx \\
          --baseline-label "ROCm 6.0" --test-label "ROCm 7.0" \\
          -o comparison.xlsx
    """
    from pathlib import Path
    from .comparison import (
        combine_excel_files,
        add_collective_comparison,
        save_with_formatting,
    )
    from .comparison.combine import extract_label_from_path
    from .comparison.collective_comparison import get_percent_change_columns

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    baseline_path = Path(baseline)
    test_path = Path(test)
    output_path = Path(output)

    # Extract labels from paths if not provided
    if baseline_label is None:
        baseline_label = extract_label_from_path(baseline_path, "baseline")
    if test_label is None:
        test_label = extract_label_from_path(test_path, "test")

    if not quiet:
        click.echo("=" * 60)
        click.echo("Collective/NCCL Comparison")
        click.echo("=" * 60)
        click.echo(f"Baseline: {baseline_path}")
        click.echo(f"Test: {test_path}")
        click.echo(f"Baseline label: {baseline_label}")
        click.echo(f"Test label: {test_label}")

    try:
        # Step 1: Combine Excel files (filter to summary sheets only)
        if not quiet:
            click.echo("\nStep 1: Combining Excel files")
        combined = combine_excel_files(
            baseline_path,
            test_path,
            baseline_label,
            test_label,
            filter_summary_only=True,
            verbose=verbose,
        )

        # Step 2: Add comparison sheets
        if not quiet:
            click.echo("\nStep 2: Adding comparison sheets")
        result = add_collective_comparison(
            combined,
            baseline_label,
            test_label,
            verbose=verbose,
        )

        # Step 3: Save with formatting
        if not quiet:
            click.echo("\nStep 3: Saving with formatting")

        # Build format_columns for all comparison sheets
        format_columns = {}
        for sheet_name, df in result.items():
            if sheet_name.endswith("_cmp"):
                pct_cols = get_percent_change_columns(df)
                if pct_cols:
                    format_columns[sheet_name] = pct_cols

        save_with_formatting(result, output_path, format_columns, verbose=verbose)

        if not quiet:
            click.echo("\n" + "=" * 60)
            click.echo("Comparison Complete!")
            click.echo("=" * 60)
            click.echo(f"\nOutput: {output_path}")
            click.echo("\nSheets:")
            for sheet_name in result.keys():
                click.echo(f"  - {sheet_name}")
            click.echo("\npercent_change interpretation:")
            click.echo("  For latency/time: Positive = faster (better)")
            click.echo("  For bandwidth: Positive = higher bandwidth (better)")

    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))


# =============================================================================
# GENERATE Group
# =============================================================================


@cli.group()
@click.pass_context
def generate(ctx):
    """Generate reports and visualizations.

    \b
    Commands:
      html   - Generate HTML report with embedded images
      excel  - Generate comprehensive Excel report
      plots  - Generate visualization plots
    """
    pass


@generate.command("html")
@click.option("--mode", type=click.Choice(["sweep", "performance"]), required=True,
              help="Report mode: 'sweep' for GEMM variance comparison, 'performance' for GPU/NCCL analysis")
# Sweep mode options
@click.option("--sweep1", type=click.Path(exists=True),
              help="[sweep mode] First sweep directory")
@click.option("--sweep2", type=click.Path(exists=True),
              help="[sweep mode] Second sweep directory")
@click.option("--label1", help="[sweep mode] Label for first sweep")
@click.option("--label2", help="[sweep mode] Label for second sweep")
# Performance mode options
@click.option("--plots-dir", type=click.Path(exists=True),
              help="[performance mode] Directory containing pre-generated plots")
# Common options
@click.option("-o", "--output", required=True, type=click.Path(), help="Output HTML file")
@click.pass_context
def generate_html(ctx, mode, sweep1, sweep2, label1, label2, plots_dir, output):
    """Generate HTML report with embedded images.

    Two modes available:

    \b
    SWEEP MODE (--mode sweep):
      Compare GEMM kernel variance between two experiment sweeps.
      Requires: --sweep1, --sweep2
      Optional: --label1, --label2

    \b
    PERFORMANCE MODE (--mode performance):
      Generate GPU/NCCL performance analysis report.
      Requires: --plots-dir (directory with pre-generated plots)

    \b
    Examples:
      # Sweep comparison (GEMM variance)
      aorta-report generate html --mode sweep \\
          --sweep1 ./exp1 --sweep2 ./exp2 \\
          --label1 "Baseline" --label2 "Optimized" \\
          -o comparison.html

      # Performance report (GPU/NCCL analysis)
      aorta-report generate html --mode performance \\
          --plots-dir ./output/plots \\
          -o performance_report.html
    """
    from pathlib import Path
    from .generators import generate_html as do_generate_html

    verbose = ctx.obj.get("verbose", False)

    try:
        output_path = do_generate_html(
            mode=mode,
            output=Path(output),
            sweep1=Path(sweep1) if sweep1 else None,
            sweep2=Path(sweep2) if sweep2 else None,
            label1=label1,
            label2=label2,
            plots_dir=Path(plots_dir) if plots_dir else None,
            verbose=verbose,
        )
        if not ctx.obj.get("quiet", False):
            click.echo(f"\nReport generated successfully: {output_path}")
    except ValueError as e:
        raise click.UsageError(str(e))
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


@generate.command("excel")
@click.option("--gpu-combined", required=True, type=click.Path(exists=True),
              help="GPU combined report file")
@click.option("--gpu-comparison", required=True, type=click.Path(exists=True),
              help="GPU comparison report file")
@click.option("--coll-combined", required=True, type=click.Path(exists=True),
              help="Collective combined report file")
@click.option("--coll-comparison", required=True, type=click.Path(exists=True),
              help="Collective comparison report file")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output Excel file")
@click.option("--baseline-label", help="Label for baseline")
@click.option("--test-label", help="Label for test")
@click.pass_context
def generate_excel(ctx, gpu_combined, gpu_comparison, coll_combined, coll_comparison,
                   output, baseline_label, test_label):
    """Generate comprehensive Excel report.

    Combines GPU timeline and collective comparison data into a single report.

    \b
    Examples:
      aorta-report generate excel \\
          --gpu-combined gpu_combined.xlsx \\
          --gpu-comparison gpu_comparison.xlsx \\
          --coll-combined coll_combined.xlsx \\
          --coll-comparison coll_comparison.xlsx \\
          -o final_report.xlsx
    """
    click.echo(f"[generate excel] gpu_combined={gpu_combined}")
    click.echo(f"  gpu_comparison={gpu_comparison}")
    click.echo(f"  coll_combined={coll_combined}")
    click.echo(f"  coll_comparison={coll_comparison}")
    click.echo(f"  output={output}")
    click.echo("  [NOT IMPLEMENTED]")


@generate.command("plots")
@click.option("-i", "--input", "input_file", required=True, type=click.Path(exists=True),
              help="Input Excel report")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--type", "plot_type", type=click.Choice(["all", "gpu-timeline", "gemm-variance"]),
              default="all", help="Type of plots to generate")
@click.pass_context
def generate_plots(ctx, input_file, output, plot_type):
    """Generate visualization plots.

    \b
    Examples:
      aorta-report generate plots -i final_report.xlsx -o ./plots/
      aorta-report generate plots -i report.xlsx -o ./plots/ --type gemm-variance
    """
    click.echo(f"[generate plots] input={input_file}")
    click.echo(f"  output={output}, type={plot_type}")
    click.echo("  [NOT IMPLEMENTED]")


# =============================================================================
# PROCESS Group
# =============================================================================


@cli.group()
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
    from pathlib import Path

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
            from .processing import process_single_config
            output_path = process_single_config(
                reports_dir=input_path,
                use_geo_mean=geo_mean,
                output_path=Path(output) if output else None,
                verbose=verbose,
            )
        else:  # sweep
            from .processing import process_sweep_config
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
    from pathlib import Path
    from .processing import process_nccl_data

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
    from pathlib import Path
    from .processing import enhance_gemm_variance

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


# =============================================================================
# PIPELINE Group
# =============================================================================


@cli.group()
@click.pass_context
def pipeline(ctx):
    """Run complete analysis pipelines.

    \b
    Commands:
      full  - Run complete analysis pipeline with comparisons
      gemm  - Run GEMM-focused analysis pipeline
    """
    pass


@pipeline.command("full")
@click.option("-b", "--baseline", required=True, type=click.Path(exists=True),
              help="Baseline trace directory")
@click.option("-t", "--test", multiple=True, required=True, type=click.Path(exists=True),
              help="Test trace directory (can be specified multiple times)")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--skip-tracelens", is_flag=True, help="Skip TraceLens report generation")
@click.option("--gpu-timeline/--no-gpu-timeline", default=True,
              help="Perform GPU timeline comparison")
@click.option("--collective/--no-collective", default=True,
              help="Perform collective/NCCL comparison")
@click.option("--final-report/--no-final-report", default=True,
              help="Create comprehensive final report")
@click.option("--plots/--no-plots", default=True,
              help="Generate visualization plots")
@click.pass_context
def pipeline_full(ctx, baseline, test, output, skip_tracelens, gpu_timeline,
                  collective, final_report, plots):
    """Run complete analysis pipeline with comparisons.

    \b
    Examples:
      aorta-report pipeline full \\
          --baseline /path/to/baseline \\
          --test /path/to/test \\
          --output /path/to/output

      aorta-report pipeline full \\
          -b /path/to/baseline \\
          -t /path/to/test1 -t /path/to/test2 \\
          -o /path/to/output \\
          --skip-tracelens --plots
    """
    click.echo(f"[pipeline full] baseline={baseline}")
    click.echo(f"  test={test}")
    click.echo(f"  output={output}")
    click.echo(f"  skip_tracelens={skip_tracelens}")
    click.echo(f"  gpu_timeline={gpu_timeline}, collective={collective}")
    click.echo(f"  final_report={final_report}, plots={plots}")
    click.echo("  [NOT IMPLEMENTED]")


@pipeline.command("gemm")
@click.option("--sweep-dir", required=True, type=click.Path(exists=True),
              help="Sweep directory to analyze")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.option("--top-k", default=5, type=int, help="Number of top kernels to extract")
@click.pass_context
def pipeline_gemm(ctx, sweep_dir, output, top_k):
    """Run GEMM-focused analysis pipeline.

    \b
    Examples:
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o /path/to/output
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o ./results --top-k 10
    """
    click.echo(f"[pipeline gemm] sweep_dir={sweep_dir}")
    click.echo(f"  output={output}, top_k={top_k}")
    click.echo("  [NOT IMPLEMENTED]")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

