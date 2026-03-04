"""
aorta-report CLI - Unified interface for TraceLens analysis and report generation.

This is the main entry point that orchestrates all command groups.
Each command group is defined in its respective package's cli.py module:

  - analysis/cli.py    → analyze commands
  - comparison/cli.py  → compare commands
  - generators/cli.py  → generate commands
  - processing/cli.py  → process commands
  - pipelines/cli.py   → pipeline commands

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
# Register Command Groups from Subpackages
# =============================================================================

# Import command groups from their respective packages
from .analysis.cli import analyze
from .comparison.cli import compare
from .generators.cli import generate
from .processing.cli import process
from .pipelines.cli import pipeline

# Register all command groups with the main CLI
cli.add_command(analyze)
cli.add_command(compare)
cli.add_command(generate)
cli.add_command(process)
cli.add_command(pipeline)


# =============================================================================
# Magpie Integration Commands
# =============================================================================

@cli.group()
@click.pass_context
def magpie(ctx):
    """Magpie benchmark integration commands.

    \b
    Import and analyze Magpie benchmark workspaces:
      list      - List Magpie benchmark workspaces
      show      - Show details of a Magpie benchmark run
      import    - Import a Magpie workspace for aorta analysis
      compare   - Quick comparison of two Magpie benchmark runs
    """
    pass


@magpie.command("list")
@click.argument("results_dir", required=False, default="./results")
@click.pass_context
def magpie_list(ctx, results_dir):
    """List Magpie benchmark workspaces in RESULTS_DIR."""
    from .magpie_adapter import locate_magpie_workspaces, read_magpie_report

    workspaces = locate_magpie_workspaces(results_dir)
    if not workspaces:
        click.echo(f"No Magpie workspaces found in {results_dir}")
        return

    click.echo(f"Found {len(workspaces)} Magpie benchmark workspace(s):\n")
    for ws in workspaces:
        report = read_magpie_report(ws)
        status = "OK" if report.get("success") else "FAIL"
        fw = report.get("framework", "?")
        model = report.get("model", "?")
        tp = report.get("throughput") or {}
        req_tp = tp.get("request_throughput", 0)
        tl = "Y" if report.get("has_tracelens") else "N"

        click.echo(
            f"  {ws.name}  [{status}] {fw} {model}  "
            f"{req_tp:.1f} req/s  TraceLens={tl}"
        )


@magpie.command("show")
@click.argument("workspace")
@click.pass_context
def magpie_show(ctx, workspace):
    """Show details of a Magpie benchmark workspace."""
    import json as _json
    from .magpie_adapter import read_magpie_report

    report = read_magpie_report(workspace)
    click.echo(_json.dumps(report, indent=2))


@magpie.command("import")
@click.argument("workspace")
@click.option("-o", "--output", required=True, help="Output directory for aorta-format data")
@click.option("--run-tracelens", is_flag=True,
              help="Run TraceLens analysis on torch traces if not already present")
@click.option("--num-ranks", default=8, help="Number of ranks for multi-rank analysis")
@click.pass_context
def magpie_import(ctx, workspace, output, run_tracelens, num_ranks):
    """Import a Magpie workspace into aorta-report format.

    Copies TraceLens output and torch traces so aorta-report commands
    (analyze, compare, generate) can operate on them.
    """
    from .magpie_adapter import import_magpie_workspace

    result = import_magpie_workspace(
        workspace=workspace,
        output_dir=output,
        run_tracelens=run_tracelens,
        num_ranks=num_ranks,
    )

    click.echo(f"Imported {len(result.get('imported_files', []))} items to {output}")
    for f in result.get("imported_files", []):
        click.echo(f"  {f}")

    if result.get("tracelens_ran"):
        click.echo("TraceLens analysis completed.")
    elif result.get("tracelens_error"):
        click.echo(f"TraceLens error: {result['tracelens_error']}", err=True)


@magpie.command("compare")
@click.option("-b", "--baseline", required=True, help="Baseline Magpie workspace directory")
@click.option("-t", "--test", required=True, help="Test Magpie workspace directory")
@click.option("-o", "--output", default=None, help="Output JSON file for comparison")
@click.pass_context
def magpie_compare(ctx, baseline, test, output):
    """Quick comparison of two Magpie benchmark runs.

    Computes throughput and latency deltas.

    \b
    Example:
      aorta-report magpie compare \\
          -b results/benchmark_vllm_20260301_120000 \\
          -t results/benchmark_vllm_20260301_140000
    """
    import json as _json
    from .magpie_adapter import compare_magpie_reports

    comparison = compare_magpie_reports(baseline, test)

    if "error" in comparison:
        click.echo(f"Error: {comparison['error']}", err=True)
        return

    # Print summary
    summary = comparison.get("summary", {})
    overall = summary.get("overall", "unknown")
    click.echo(f"Overall: {overall.upper()}\n")

    click.echo("Throughput:")
    for key, val in comparison.get("throughput", {}).items():
        click.echo(
            f"  {key}: {val['baseline']:.2f} -> {val['test']:.2f} "
            f"({val['percent_change']:+.1f}%) [{val['status']}]"
        )

    click.echo("\nLatency:")
    for key, val in comparison.get("latency", {}).items():
        click.echo(
            f"  {key}: {val['baseline']:.2f} -> {val['test']:.2f} ms "
            f"({val['percent_change']:+.1f}%) [{val['status']}]"
        )

    if output:
        with open(output, "w") as f:
            _json.dump(comparison, f, indent=2)
        click.echo(f"\nComparison saved to: {output}")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
