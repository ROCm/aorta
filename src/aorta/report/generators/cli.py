"""CLI commands for report generation.

This module provides the 'generate' command group with subcommands:
  - html: Generate HTML report with embedded images
  - excel: Generate comprehensive Excel report
  - plots: Generate visualization plots
"""

import click
from pathlib import Path


@click.group()
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
    from . import generate_html as do_generate_html

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
@click.option("--baseline-label", default="Baseline", help="Label for baseline (default: Baseline)")
@click.option("--test-label", default="Test", help="Label for test (default: Test)")
@click.pass_context
def generate_excel(ctx, gpu_combined, gpu_comparison, coll_combined, coll_comparison,
                   output, baseline_label, test_label):
    """Generate comprehensive Excel report.

    Combines GPU timeline and collective comparison data into a single
    well-organized Excel report with:

    \b
    - Summary Dashboard (first sheet, key metrics at a glance)
    - Comparison sheets (visible, with color-coded changes)
    - Raw data sheets (hidden, accessible via Unhide)
    - Excel table formatting with filters

    \b
    Examples:
      aorta-report generate excel \\
          --gpu-combined gpu_combined.xlsx \\
          --gpu-comparison gpu_comparison.xlsx \\
          --coll-combined coll_combined.xlsx \\
          --coll-comparison coll_comparison.xlsx \\
          -o final_report.xlsx

      aorta-report generate excel \\
          --gpu-combined gpu_combined.xlsx \\
          --gpu-comparison gpu_comparison.xlsx \\
          --coll-combined coll_combined.xlsx \\
          --coll-comparison coll_comparison.xlsx \\
          --baseline-label "ROCm 6.0" --test-label "ROCm 7.0" \\
          -o final_report.xlsx
    """
    from . import create_final_excel_report

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    if not quiet:
        click.echo("=" * 60)
        click.echo("Creating Final Excel Report")
        click.echo("=" * 60)
        click.echo(f"GPU Combined:     {gpu_combined}")
        click.echo(f"GPU Comparison:   {gpu_comparison}")
        click.echo(f"Coll Combined:    {coll_combined}")
        click.echo(f"Coll Comparison:  {coll_comparison}")
        click.echo(f"Output:           {output}")
        click.echo(f"Baseline label:   {baseline_label}")
        click.echo(f"Test label:       {test_label}")

    try:
        result = create_final_excel_report(
            gpu_combined_path=Path(gpu_combined),
            gpu_comparison_path=Path(gpu_comparison),
            coll_combined_path=Path(coll_combined),
            coll_comparison_path=Path(coll_comparison),
            output_path=Path(output),
            baseline_label=baseline_label,
            test_label=test_label,
            verbose=verbose,
        )

        if not quiet:
            click.echo("\n" + "=" * 60)
            click.echo("Report Complete!")
            click.echo("=" * 60)
            click.echo(f"\nOutput: {result['output_path']}")
            click.echo("\nReport Structure:")
            click.echo("  Visible Sheets (Analysis):")
            for sheet in result["visible_sheets"]:
                click.echo(f"    - {sheet}")
            click.echo("\n  Hidden Sheets (Raw Data):")
            for sheet in result["hidden_sheets"]:
                click.echo(f"    - {sheet}")
            click.echo("\nFeatures:")
            click.echo("  - All data formatted as Excel tables with filters")
            click.echo("  - Percent change columns are color-coded (green=better, red=worse)")
            click.echo("  - Unhide raw data: Right-click sheet tab → Unhide")

    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Error creating report: {e}")


@generate.command("plots")
@click.option("-i", "--input", "input_file", type=click.Path(exists=True),
              help="Input file (Excel for summary, CSV for gemm)")
@click.option("--excel-input", type=click.Path(exists=True),
              help="Excel report file (for --type all)")
@click.option("--gemm-csv", type=click.Path(exists=True),
              help="GEMM variance CSV (for --type all)")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory for PNG files")
@click.option("--type", "plot_type",
              type=click.Choice(["all", "summary", "gemm"]),
              default="all", help="Type of plots to generate")
@click.option("--dpi", default=150, type=int,
              help="DPI for output images (default: 150)")
@click.pass_context
def generate_plots_cmd(ctx, input_file, excel_input, gemm_csv, output, plot_type, dpi):
    """Generate visualization plots.

    \b
    Plot Types:
      summary  - GPU timeline & NCCL charts from Excel report
      gemm     - GEMM variance distribution from CSV
      all      - Both summary and gemm plots

    \b
    Examples:
      # Summary plots from Excel report
      aorta-report generate plots -i final_report.xlsx -o ./plots/ --type summary

      # GEMM plots from CSV
      aorta-report generate plots -i gemm_variance.csv -o ./plots/ --type gemm

      # All plots (both inputs required)
      aorta-report generate plots \\
          --excel-input final_report.xlsx \\
          --gemm-csv gemm_variance.csv \\
          -o ./plots/ --type all
    """
    from . import generate_plots

    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    # Resolve inputs based on plot_type
    excel_path = None
    csv_path = None

    if plot_type == "summary":
        if input_file is None and excel_input is None:
            raise click.UsageError("--input or --excel-input required for summary plots")
        excel_path = Path(input_file or excel_input)
    elif plot_type == "gemm":
        if input_file is None and gemm_csv is None:
            raise click.UsageError("--input or --gemm-csv required for gemm plots")
        csv_path = Path(input_file or gemm_csv)
    else:  # all
        if excel_input is None:
            raise click.UsageError("--excel-input required for --type all")
        if gemm_csv is None:
            raise click.UsageError("--gemm-csv required for --type all")
        excel_path = Path(excel_input)
        csv_path = Path(gemm_csv)

    if not quiet:
        click.echo("=" * 60)
        click.echo("Generating Plots")
        click.echo("=" * 60)
        click.echo(f"Plot type: {plot_type}")
        if excel_path:
            click.echo(f"Excel input: {excel_path}")
        if csv_path:
            click.echo(f"GEMM CSV: {csv_path}")
        click.echo(f"Output: {output}")
        click.echo(f"DPI: {dpi}")

    try:
        results = generate_plots(
            plot_type=plot_type,
            output_dir=Path(output),
            excel_input=excel_path,
            gemm_csv=csv_path,
            dpi=dpi,
            verbose=verbose,
        )

        if not quiet:
            click.echo("\n" + "=" * 60)
            click.echo("Plots Generated!")
            click.echo("=" * 60)
            total = 0
            for category, files in results.items():
                click.echo(f"\n{category.upper()} plots:")
                for f in files:
                    click.echo(f"  - {f.name}")
                total += len(files)
            click.echo(f"\nTotal: {total} files generated in {output}")

    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Error generating plots: {e}")

