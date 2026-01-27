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
# Entry Point
# =============================================================================


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
