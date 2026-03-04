"""Summary analysis pipeline.

Orchestrates complete TraceLens analysis workflow:
1. TraceLens Analysis (optional)
2. Process GPU Timelines
3. Compare GPU Timelines
4. Compare Collective
5. Generate Final Excel Report
6. Generate Plots
7. Generate HTML Report
"""

from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class SummaryPipelineConfig:
    """Configuration for summary pipeline."""

    test_path: Path
    output_dir: Path
    baseline_path: Optional[Path] = None  # Optional for single-config mode
    baseline_label: Optional[str] = None
    test_label: Optional[str] = None
    skip_tracelens: bool = False
    skip_baseline_gpu_processing: bool = False  # Skip baseline GPU processing (use existing file)
    gpu_timeline: bool = True
    collective: bool = True
    final_report: bool = True
    plots: bool = True
    html: bool = True
    verbose: bool = False

    @property
    def is_comparison_mode(self) -> bool:
        """True if baseline provided (comparison mode), False for single-config."""
        return self.baseline_path is not None


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    success: bool
    output_dir: Path
    files_generated: Dict[str, Path] = field(default_factory=dict)
    steps_completed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def run_summary_pipeline(config: SummaryPipelineConfig) -> PipelineResult:
    """
    Run the complete summary pipeline.

    Supports two modes:
    - Comparison mode: baseline and test provided, generates comparison reports
    - Single-config mode: only test provided, generates single-config plots

    Returns PipelineResult with success status and generated files.
    """
    result = PipelineResult(
        success=True,
        output_dir=config.output_dir,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract labels from directory names if not provided
    test_label = config.test_label or config.test_path.name
    baseline_label = config.baseline_label or (
        config.baseline_path.name if config.baseline_path else None
    )

    try:
        # Step 1: TraceLens Analysis
        if not config.skip_tracelens:
            _step_tracelens_analysis(config, result)
        else:
            result.steps_skipped.append("tracelens_analysis")

        # Validate analysis directories exist
        test_analysis = config.test_path / "tracelens_analysis"
        if not test_analysis.exists():
            raise FileNotFoundError(
                f"Test analysis not found: {test_analysis}. "
                "Run without --skip-tracelens first."
            )

        if config.is_comparison_mode:
            baseline_analysis = config.baseline_path / "tracelens_analysis"
            if not baseline_analysis.exists():
                raise FileNotFoundError(
                    f"Baseline analysis not found: {baseline_analysis}. "
                    "Run without --skip-tracelens first."
                )

        # Step 2: Process GPU Timelines
        if config.gpu_timeline:
            _step_process_gpu_timelines(config, result)

        # BRANCHING: Comparison vs Single-Config mode
        if config.is_comparison_mode:
            # === COMPARISON MODE ===
            # Step 3: Compare GPU Timelines
            if config.gpu_timeline:
                _step_compare_gpu_timeline(config, result, baseline_label, test_label)
            else:
                result.steps_skipped.append("compare_gpu_timeline")

            # Step 4: Compare Collective
            if config.collective:
                _step_compare_collective(config, result, baseline_label, test_label)
            else:
                result.steps_skipped.append("compare_collective")

            # Step 5: Generate Final Report
            if (
                config.final_report
                and config.gpu_timeline
                and config.collective
                and "gpu_combined" in result.files_generated
                and "coll_combined" in result.files_generated
            ):
                _step_generate_final_report(config, result, baseline_label, test_label)
            elif config.final_report:
                result.steps_skipped.append("final_report (requires both gpu_timeline and collective)")

            # Step 6: Generate Plots (comparison mode)
            if config.plots and "final_report" in result.files_generated:
                _step_generate_plots(config, result, [baseline_label, test_label])
            elif config.plots:
                result.steps_skipped.append("plots (requires final_report)")

        else:
            # === SINGLE-CONFIG MODE ===
            result.steps_skipped.append("compare_gpu_timeline (single-config mode)")
            result.steps_skipped.append("compare_collective (single-config mode)")
            result.steps_skipped.append("final_report (single-config mode)")

            # Step 6: Generate Single-Config Plots
            if config.plots:
                _step_generate_single_config_plots(config, result, test_label)

        # Step 7: Generate HTML
        if config.html and "plots_dir" in result.files_generated:
            _step_generate_html(config, result)
        elif config.html:
            result.steps_skipped.append("html (requires plots)")

    except Exception as e:
        result.success = False
        result.errors.append(str(e))

    return result


def _step_tracelens_analysis(config: SummaryPipelineConfig, result: PipelineResult) -> None:
    """Step 1: Run TraceLens analysis on baseline (if provided) and test."""
    from ..analysis import analyze_single_config

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 1: TraceLens Analysis")
        print("=" * 60)

    # Analyze baseline (if provided)
    if config.is_comparison_mode:
        if config.verbose:
            print(f"\nAnalyzing baseline: {config.baseline_path}")
        analyze_single_config(config.baseline_path, verbose=config.verbose)

    # Analyze test/single config
    if config.verbose:
        print(f"\nAnalyzing test: {config.test_path}")
    analyze_single_config(config.test_path, verbose=config.verbose)

    result.steps_completed.append("tracelens_analysis")


def _step_process_gpu_timelines(config: SummaryPipelineConfig, result: PipelineResult) -> None:
    """Step 2: Process GPU timelines for baseline (if provided) and test."""
    from ..processing import process_single_config

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Process GPU Timelines")
        print("=" * 60)

    # Process baseline (if provided, and not skipped)
    if config.is_comparison_mode and not config.skip_baseline_gpu_processing:
        baseline_reports = config.baseline_path / "tracelens_analysis" / "individual_reports"
        if config.verbose:
            print(f"\nProcessing baseline: {baseline_reports}")
        process_single_config(baseline_reports, verbose=config.verbose)
    elif config.is_comparison_mode and config.skip_baseline_gpu_processing:
        if config.verbose:
            print("\nSkipping baseline GPU processing (--skip-baseline-gpu-processing)")
        result.steps_skipped.append("process_baseline_gpu_timelines")

        baseline_gpu_timeline = config.baseline_path / "gpu_timeline_summary_mean.xlsx"
        if not baseline_gpu_timeline.is_file():
            msg = (
                "Baseline GPU timeline file not found. Cannot use "
                "--skip-baseline-gpu-processing without existing "
                "gpu_timeline_summary_mean.xlsx."
            )
            if config.verbose:
                print(msg)
            raise FileNotFoundError(msg)
    # Process test/single config
    test_reports = config.test_path / "tracelens_analysis" / "individual_reports"
    if config.verbose:
        print(f"\nProcessing test: {test_reports}")
    process_single_config(test_reports, verbose=config.verbose)

    result.steps_completed.append("process_gpu_timelines")


def _step_compare_gpu_timeline(
    config: SummaryPipelineConfig,
    result: PipelineResult,
    baseline_label: str,
    test_label: str,
) -> None:
    """Step 3: Compare GPU timelines."""
    from ..comparison import (
        combine_excel_files,
        add_gpu_timeline_comparison,
        save_with_formatting,
    )

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Compare GPU Timelines")
        print("=" * 60)

    baseline_gpu = config.baseline_path / "tracelens_analysis" / "gpu_timeline_summary_mean.xlsx"
    test_gpu = config.test_path / "tracelens_analysis" / "gpu_timeline_summary_mean.xlsx"

    if not baseline_gpu.exists():
        raise FileNotFoundError(f"Baseline GPU timeline not found: {baseline_gpu}")
    if not test_gpu.exists():
        raise FileNotFoundError(f"Test GPU timeline not found: {test_gpu}")

    # Combine
    combined = combine_excel_files(
        baseline_gpu, test_gpu, baseline_label, test_label, verbose=config.verbose
    )

    # Save combined
    combined_path = config.output_dir / "gpu_timeline_combined.xlsx"
    save_with_formatting(combined, combined_path, {})
    result.files_generated["gpu_combined"] = combined_path

    # Add comparison
    comparison = add_gpu_timeline_comparison(
        combined, baseline_label, test_label, verbose=config.verbose
    )

    # Save comparison
    comparison_path = config.output_dir / "gpu_timeline_comparison.xlsx"
    format_columns = {
        "Comparison_By_Rank": ["percent_change"],
        "Summary_Comparison": ["percent_change"],
    }
    save_with_formatting(comparison, comparison_path, format_columns)
    result.files_generated["gpu_comparison"] = comparison_path

    if config.verbose:
        print(f"  GPU timeline combined: {combined_path}")
        print(f"  GPU timeline comparison: {comparison_path}")

    result.steps_completed.append("compare_gpu_timeline")


def _step_compare_collective(
    config: SummaryPipelineConfig,
    result: PipelineResult,
    baseline_label: str,
    test_label: str,
) -> None:
    """Step 4: Compare collective/NCCL."""
    from ..comparison import (
        combine_excel_files,
        add_collective_comparison,
        save_with_formatting,
    )
    from ..comparison.collective_comparison import get_percent_change_columns

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 4: Compare Collective/NCCL")
        print("=" * 60)

    baseline_coll = (
        config.baseline_path
        / "tracelens_analysis"
        / "collective_reports"
        / "collective_all_ranks.xlsx"
    )
    test_coll = (
        config.test_path / "tracelens_analysis" / "collective_reports" / "collective_all_ranks.xlsx"
    )

    if not baseline_coll.exists():
        raise FileNotFoundError(f"Baseline collective not found: {baseline_coll}")
    if not test_coll.exists():
        raise FileNotFoundError(f"Test collective not found: {test_coll}")

    # Combine (filter summary sheets only)
    combined = combine_excel_files(
        baseline_coll,
        test_coll,
        baseline_label,
        test_label,
        filter_summary_only=True,
        verbose=config.verbose,
    )

    # Save combined
    combined_path = config.output_dir / "collective_combined.xlsx"
    save_with_formatting(combined, combined_path, {})
    result.files_generated["coll_combined"] = combined_path

    # Add comparison
    comparison = add_collective_comparison(
        combined, baseline_label, test_label, verbose=config.verbose
    )

    # Save comparison
    comparison_path = config.output_dir / "collective_comparison.xlsx"
    format_columns: Dict[str, List[str]] = {}
    for sheet_name, df in comparison.items():
        if sheet_name.endswith("_cmp"):
            pct_cols = get_percent_change_columns(df)
            if pct_cols:
                format_columns[sheet_name] = pct_cols
    save_with_formatting(comparison, comparison_path, format_columns)
    result.files_generated["coll_comparison"] = comparison_path

    if config.verbose:
        print(f"  Collective combined: {combined_path}")
        print(f"  Collective comparison: {comparison_path}")

    result.steps_completed.append("compare_collective")


def _step_generate_final_report(
    config: SummaryPipelineConfig,
    result: PipelineResult,
    baseline_label: str,
    test_label: str,
) -> None:
    """Step 5: Generate final Excel report."""
    from ..generators import create_final_excel_report

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 5: Generate Final Excel Report")
        print("=" * 60)

    final_report_path = config.output_dir / "final_analysis_report.xlsx"

    create_final_excel_report(
        gpu_combined_path=result.files_generated["gpu_combined"],
        gpu_comparison_path=result.files_generated["gpu_comparison"],
        coll_combined_path=result.files_generated["coll_combined"],
        coll_comparison_path=result.files_generated["coll_comparison"],
        output_path=final_report_path,
        baseline_label=baseline_label,
        test_label=test_label,
        verbose=config.verbose,
    )

    result.files_generated["final_report"] = final_report_path

    if config.verbose:
        print(f"  Final report: {final_report_path}")

    result.steps_completed.append("final_report")


def _step_generate_plots(
    config: SummaryPipelineConfig,
    result: PipelineResult,
    labels: List[str],
) -> None:
    """Step 6: Generate plots (comparison mode)."""
    from ..generators import generate_summary_plots

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 6: Generate Plots (Comparison)")
        print("=" * 60)

    plots_dir = config.output_dir / "plots"

    plot_files = generate_summary_plots(
        excel_path=result.files_generated["final_report"],
        output_dir=plots_dir,
        labels=labels,
        verbose=config.verbose,
    )

    result.files_generated["plots_dir"] = plots_dir

    if config.verbose:
        print(f"  Plots directory: {plots_dir}")
        print(f"  Generated {len(plot_files)} plots")

    result.steps_completed.append("plots")


def _step_generate_single_config_plots(
    config: SummaryPipelineConfig,
    result: PipelineResult,
    label: str,
) -> None:
    """Step 6 (single-config): Generate single-config plots."""
    from ..generators import generate_single_config_plots

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 6: Generate Plots (Single-Config)")
        print("=" * 60)

    plots_dir = config.output_dir / "plots"

    # Use the processed GPU timeline summary directly
    gpu_summary_path = config.test_path / "tracelens_analysis" / "gpu_timeline_summary_mean.xlsx"
    coll_path = (
        config.test_path / "tracelens_analysis" / "collective_reports" / "collective_all_ranks.xlsx"
    )

    plot_files = generate_single_config_plots(
        gpu_excel_path=gpu_summary_path,
        output_dir=plots_dir,
        label=label,
        coll_excel_path=coll_path if coll_path.exists() else None,
        verbose=config.verbose,
    )

    result.files_generated["plots_dir"] = plots_dir

    if config.verbose:
        print(f"  Plots directory: {plots_dir}")
        print(f"  Generated {len(plot_files)} plots")

    result.steps_completed.append("plots (single-config)")


def _step_generate_html(config: SummaryPipelineConfig, result: PipelineResult) -> None:
    """Step 7: Generate HTML report."""
    from ..generators import generate_html

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 7: Generate HTML Report")
        print("=" * 60)

    html_path = config.output_dir / "performance_analysis_report.html"

    generate_html(
        mode="performance",
        output=html_path,
        plots_dir=result.files_generated["plots_dir"],
        verbose=config.verbose,
    )

    result.files_generated["html_report"] = html_path

    if config.verbose:
        print(f"  HTML report: {html_path}")

    result.steps_completed.append("html")
