"""Plot generation orchestrator.

Provides unified interface for generating summary and GEMM plots.
"""

from pathlib import Path
from typing import Dict, List, Optional

from .plot_helper import (
    configure_style,
    # Summary
    get_labels_from_excel,
    plot_improvement_chart,
    plot_abs_time_comparison,
    plot_gpu_metrics_by_rank,
    plot_gpu_percent_change_grid,
    plot_gpu_heatmap,
    plot_nccl_comparison,
    plot_nccl_percent_change,
    # GEMM
    read_gemm_csv_data,
    print_gemm_statistics,
    plot_variance_by_threads,
    plot_variance_by_channels,
    plot_variance_by_ranks,
    plot_variance_violin_combined,
    plot_thread_channel_interaction,
)


def generate_summary_plots(
    excel_path: Path,
    output_dir: Path,
    labels: Optional[List[str]] = None,
    dpi: int = 150,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate all summary plots from Excel report (comparison mode).

    Args:
        excel_path: Path to final Excel report
        output_dir: Output directory for PNG files
        labels: Optional list of labels [baseline, test]. If None, extracted from Excel.
        dpi: DPI for output images
        verbose: Print progress

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: List[Path] = []

    if verbose:
        print(f"\nGenerating summary plots from: {excel_path}")

    # Use provided labels or extract from Excel
    if labels is None:
        labels = get_labels_from_excel(excel_path)
    if verbose:
        print(f"  Labels: {labels}")

    configure_style()

    # Dashboard plots
    if verbose:
        print("  Creating dashboard plots...")
    output_files.append(plot_improvement_chart(excel_path, output_dir, dpi))
    output_files.append(plot_abs_time_comparison(excel_path, output_dir, labels, dpi))

    # GPU plots
    if verbose:
        print("  Creating GPU plots...")
    output_files.extend(
        plot_gpu_metrics_by_rank(excel_path, output_dir, labels, dpi=dpi, single_config_mode=False)
    )
    output_files.append(plot_gpu_percent_change_grid(excel_path, output_dir, dpi))
    output_files.append(plot_gpu_heatmap(excel_path, output_dir, dpi))

    # NCCL plots
    if verbose:
        print("  Creating NCCL plots...")
    nccl_files = plot_nccl_comparison(excel_path, output_dir, labels, dpi, single_config_mode=False)
    output_files.extend(nccl_files)

    nccl_pct_file = plot_nccl_percent_change(excel_path, output_dir, dpi)
    if nccl_pct_file:
        output_files.append(nccl_pct_file)

    if verbose:
        print(f"  Generated {len(output_files)} summary plots")

    return output_files


def generate_single_config_plots(
    gpu_excel_path: Path,
    output_dir: Path,
    label: str,
    coll_excel_path: Optional[Path] = None,
    dpi: int = 150,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate plots for single configuration (no comparison).

    Reuses existing plot functions with single-element labels list.
    Skips percentage change plots (require comparison).
    Skips GPU heatmap.

    Args:
        gpu_excel_path: Path to GPU timeline summary Excel
        output_dir: Output directory for PNG files
        label: Configuration label
        coll_excel_path: Optional path to collective Excel (for NCCL plots)
        dpi: DPI for output images
        verbose: Print progress

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: List[Path] = []
    labels = [label]  # Single-element list

    if verbose:
        print(f"\nGenerating single-config plots")
        print(f"  Config: {label}")
        print(f"  GPU data: {gpu_excel_path}")

    configure_style()

    # GPU Summary Bar Chart (single bar per metric)
    if verbose:
        print("  Creating GPU summary bar chart...")
    try:
        output_files.append(
            plot_abs_time_comparison(
                gpu_excel_path, output_dir, labels, dpi, single_config_mode=True
            )
        )
    except Exception as e:
        if verbose:
            print(f"    Skipped (error: {e})")

    # GPU By Rank Line Plots (single line per metric)
    if verbose:
        print("  Creating GPU by rank plots...")
    try:
        output_files.extend(
            plot_gpu_metrics_by_rank(
                gpu_excel_path, output_dir, labels, dpi=dpi, single_config_mode=True
            )
        )
    except Exception as e:
        if verbose:
            print(f"    Skipped (error: {e})")

    # SKIP: GPU Heatmap (requires percent_change)
    # SKIP: GPU Percent Change Grid (requires comparison)
    # SKIP: Improvement Chart (requires comparison)

    # NCCL Charts (single bars)
    if coll_excel_path and coll_excel_path.exists():
        if verbose:
            print("  Creating NCCL charts...")
        try:
            nccl_files = plot_nccl_comparison(
                coll_excel_path, output_dir, labels, dpi, single_config_mode=True
            )
            output_files.extend(nccl_files)
        except Exception as e:
            if verbose:
                print(f"    Skipped (error: {e})")

    # SKIP: NCCL Percent Change (requires comparison)

    if verbose:
        print(f"  Generated {len(output_files)} plots")

    return output_files


def generate_gemm_plots(
    csv_path: Path,
    output_dir: Path,
    dpi: int = 150,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate all GEMM variance plots from CSV.

    Args:
        csv_path: Path to GEMM variance CSV
        output_dir: Output directory for PNG files
        dpi: DPI for output images
        verbose: Print progress

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: List[Path] = []

    if verbose:
        print(f"\nGenerating GEMM plots from: {csv_path}")

    data = read_gemm_csv_data(csv_path)

    if verbose:
        print(f"  Total data points: {len(data['all'])}")
        print_gemm_statistics(data)

    # Boxplots
    if verbose:
        print("  Creating boxplots...")
    output_files.append(plot_variance_by_threads(data, output_dir, dpi))
    output_files.append(plot_variance_by_channels(data, output_dir, dpi))
    output_files.append(plot_variance_by_ranks(data, output_dir, dpi))

    # Violin and interaction
    if verbose:
        print("  Creating violin and interaction plots...")
    output_files.append(plot_variance_violin_combined(data, output_dir, dpi))
    output_files.append(plot_thread_channel_interaction(data, output_dir, dpi))

    if verbose:
        print(f"  Generated {len(output_files)} GEMM plots")

    return output_files


def generate_plots(
    plot_type: str,
    output_dir: Path,
    excel_input: Optional[Path] = None,
    gemm_csv: Optional[Path] = None,
    dpi: int = 150,
    verbose: bool = False,
) -> Dict[str, List[Path]]:
    """
    Generate plots based on type.

    Args:
        plot_type: "summary", "gemm", or "all"
        output_dir: Output directory for PNG files
        excel_input: Path to Excel report (for summary/all)
        gemm_csv: Path to GEMM CSV (for gemm/all)
        dpi: DPI for output images
        verbose: Print progress

    Returns:
        Dict mapping category to list of generated file paths

    Raises:
        ValueError: If required inputs not provided for plot_type
        FileNotFoundError: If input files don't exist
    """
    configure_style()
    results: Dict[str, List[Path]] = {}

    if plot_type in ("summary", "all"):
        if excel_input is None:
            raise ValueError("Excel input required for summary plots")
        if not excel_input.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_input}")
        results["summary"] = generate_summary_plots(
            excel_input, output_dir, dpi, verbose
        )

    if plot_type in ("gemm", "all"):
        if gemm_csv is None:
            raise ValueError("GEMM CSV required for gemm plots")
        if not gemm_csv.exists():
            raise FileNotFoundError(f"CSV file not found: {gemm_csv}")
        results["gemm"] = generate_gemm_plots(gemm_csv, output_dir, dpi, verbose)

    return results
