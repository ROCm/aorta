"""CLI commands for TraceLens report comparison.

This module provides the 'compare' command group with subcommands:
  - gpu_timeline: Compare two GPU timeline reports
  - collective: Compare two collective/NCCL reports
"""

import click
from pathlib import Path


@click.group()
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
    from . import (
        combine_excel_files,
        add_gpu_timeline_comparison,
        save_with_formatting,
    )
    from .combine import extract_label_from_path

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
    from . import (
        combine_excel_files,
        add_collective_comparison,
        save_with_formatting,
    )
    from .combine import extract_label_from_path
    from .collective_comparison import get_percent_change_columns

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

