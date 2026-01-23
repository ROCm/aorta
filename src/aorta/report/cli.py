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
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.pass_context
def analyze_single(ctx, trace_dir, individual_only, collective_only, output):
    """Analyze a single configuration trace directory.

    TRACE_DIR: Path to the trace directory containing rank subdirectories.

    \b
    Examples:
      aorta-report analyze single /path/to/traces
      aorta-report analyze single /path/to/traces --individual-only
      aorta-report analyze single /path/to/traces -o ./results
    """
    click.echo(f"[analyze single] trace_dir={trace_dir}")
    click.echo(f"  individual_only={individual_only}, collective_only={collective_only}")
    click.echo(f"  output={output}")
    click.echo("  [NOT IMPLEMENTED]")


@analyze.command("sweep")
@click.argument("sweep_dir", type=click.Path(exists=True))
@click.option("--rocprof", is_flag=True, help="Use rocprof traces instead of PyTorch profiler")
@click.option("-o", "--output", type=click.Path(), help="Output directory")
@click.pass_context
def analyze_sweep(ctx, sweep_dir, rocprof, output):
    """Analyze a sweep directory with multiple configurations.

    SWEEP_DIR: Path to the sweep directory containing multiple thread/channel configs.

    \b
    Examples:
      aorta-report analyze sweep /path/to/sweep_20251124
      aorta-report analyze sweep /path/to/sweep --rocprof
    """
    click.echo(f"[analyze sweep] sweep_dir={sweep_dir}")
    click.echo(f"  rocprof={rocprof}, output={output}")
    click.echo("  [NOT IMPLEMENTED]")


@analyze.command("gemm")
@click.argument("reports_dir", type=click.Path(exists=True))
@click.option("--top-k", default=5, type=int, help="Number of top kernels to extract")
@click.option("-o", "--output", type=click.Path(), help="Output CSV file")
@click.pass_context
def analyze_gemm(ctx, reports_dir, top_k, output):
    """Analyze GEMM kernels from TraceLens reports.

    REPORTS_DIR: Path to directory containing TraceLens Excel reports.

    \b
    Examples:
      aorta-report analyze gemm /path/to/reports
      aorta-report analyze gemm /path/to/reports --top-k 10 -o gemm_analysis.csv
    """
    click.echo(f"[analyze gemm] reports_dir={reports_dir}")
    click.echo(f"  top_k={top_k}, output={output}")
    click.echo("  [NOT IMPLEMENTED]")


# =============================================================================
# COMPARE Group
# =============================================================================


@cli.group()
@click.pass_context
def compare(ctx):
    """Compare traces and reports.

    \b
    Commands:
      runs       - Compare multiple TraceLens analysis runs
      reports    - Combine and compare two reports
      collective - Add collective operation comparison sheets
    """
    pass


@compare.command("runs")
@click.option("-i", "--inputs", multiple=True, required=True, type=click.Path(exists=True),
              help="Input directories (can be specified multiple times)")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory")
@click.pass_context
def compare_runs(ctx, inputs, output):
    """Compare multiple TraceLens analysis runs.

    \b
    Examples:
      aorta-report compare runs -i /path/to/run1 -i /path/to/run2 -o /path/to/output
    """
    click.echo(f"[compare runs] inputs={inputs}")
    click.echo(f"  output={output}")
    click.echo("  [NOT IMPLEMENTED]")


@compare.command("reports")
@click.option("-b", "--baseline", required=True, type=click.Path(exists=True),
              help="Baseline report file")
@click.option("-t", "--test", required=True, type=click.Path(exists=True),
              help="Test report file")
@click.option("--baseline-label", help="Label for baseline")
@click.option("--test-label", help="Label for test")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output file")
@click.pass_context
def compare_reports(ctx, baseline, test, baseline_label, test_label, output):
    """Combine and compare two reports.

    \b
    Examples:
      aorta-report compare reports -b baseline.xlsx -t test.xlsx -o comparison.xlsx
      aorta-report compare reports -b baseline.xlsx -t test.xlsx \\
          --baseline-label "ROCm 6.0" --test-label "ROCm 7.0" -o comparison.xlsx
    """
    click.echo(f"[compare reports] baseline={baseline}, test={test}")
    click.echo(f"  baseline_label={baseline_label}, test_label={test_label}")
    click.echo(f"  output={output}")
    click.echo("  [NOT IMPLEMENTED]")


@compare.command("collective")
@click.option("-i", "--input", "input_file", required=True, type=click.Path(exists=True),
              help="Input combined report file")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output file")
@click.option("--baseline-label", help="Label for baseline")
@click.option("--test-label", help="Label for test")
@click.pass_context
def compare_collective(ctx, input_file, output, baseline_label, test_label):
    """Add collective operation comparison sheets.

    \b
    Examples:
      aorta-report compare collective -i combined.xlsx -o collective_comparison.xlsx
    """
    click.echo(f"[compare collective] input={input_file}")
    click.echo(f"  baseline_label={baseline_label}, test_label={test_label}")
    click.echo(f"  output={output}")
    click.echo("  [NOT IMPLEMENTED]")


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
@click.option("--sweep1", required=True, type=click.Path(exists=True),
              help="First sweep directory")
@click.option("--sweep2", type=click.Path(exists=True),
              help="Second sweep directory (for comparison)")
@click.option("--label1", help="Label for first sweep")
@click.option("--label2", help="Label for second sweep")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output HTML file")
@click.pass_context
def generate_html(ctx, sweep1, sweep2, label1, label2, output):
    """Generate HTML report with embedded images.

    \b
    Examples:
      aorta-report generate html --sweep1 ./exp1 -o report.html
      aorta-report generate html --sweep1 ./exp1 --sweep2 ./exp2 \\
          --label1 "Baseline" --label2 "Optimized" -o comparison.html
    """
    click.echo(f"[generate html] sweep1={sweep1}, sweep2={sweep2}")
    click.echo(f"  label1={label1}, label2={label2}")
    click.echo(f"  output={output}")
    click.echo("  [NOT IMPLEMENTED]")


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
    Examples:
      aorta-report process gpu-timeline /path/to/reports
      aorta-report process gpu-timeline /path/to/individual_reports --mode single
      aorta-report process gpu-timeline /path/to/sweep --mode sweep --geo-mean
    """
    click.echo(f"[process gpu-timeline] input_dir={input_dir}")
    click.echo(f"  mode={mode}, geo_mean={geo_mean}, output={output}")
    click.echo("  [NOT IMPLEMENTED]")


@process.command("comms")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.pass_context
def process_comms(ctx, input_dir, output):
    """Process communication data.

    INPUT_DIR: Path to directory containing trace data.

    \b
    Examples:
      aorta-report process comms /path/to/traces
      aorta-report process comms /path/to/traces -o comms_processed.xlsx
    """
    click.echo(f"[process comms] input_dir={input_dir}")
    click.echo(f"  output={output}")
    click.echo("  [NOT IMPLEMENTED]")


@process.command("gemm-variance")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--timestamps", is_flag=True, help="Include timestamp data")
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.pass_context
def process_gemm_variance(ctx, input_file, timestamps, output):
    """Enhance GEMM variance with timestamps.

    INPUT_FILE: Path to GEMM report file.

    \b
    Examples:
      aorta-report process gemm-variance report.xlsx
      aorta-report process gemm-variance report.xlsx --timestamps -o enhanced.xlsx
    """
    click.echo(f"[process gemm-variance] input_file={input_file}")
    click.echo(f"  timestamps={timestamps}, output={output}")
    click.echo("  [NOT IMPLEMENTED]")


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

