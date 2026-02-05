"""GEMM variance analysis pipeline.

Orchestrates GEMM kernel variance analysis:
1. Analyze GEMM Reports
2. Enhance with Timestamps (optional)
3. Generate GEMM Plots (optional)
"""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class GemmPipelineConfig:
    """Configuration for GEMM pipeline."""

    sweep_dir: Path
    output_dir: Path
    top_k: int = 5
    threads: List[int] = field(default_factory=lambda: [256, 512])
    channels: List[int] = field(default_factory=lambda: [28, 42, 56, 70])
    ranks: List[int] = field(default_factory=lambda: list(range(8)))
    timestamps: bool = True
    plots: bool = True
    html: bool = True
    verbose: bool = False


@dataclass
class GemmPipelineResult:
    """Result from GEMM pipeline execution."""

    success: bool
    output_dir: Path
    csv_path: Optional[Path] = None
    csv_with_timestamps_path: Optional[Path] = None
    plots_dir: Optional[Path] = None
    html_path: Optional[Path] = None
    steps_completed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def run_gemm_pipeline(config: GemmPipelineConfig) -> GemmPipelineResult:
    """
    Run the complete GEMM analysis pipeline.

    Returns GemmPipelineResult with success status and generated files.
    """
    result = GemmPipelineResult(
        success=True,
        output_dir=config.output_dir,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Analyze GEMM Reports
        _step_analyze_gemm(config, result)

        # Step 2: Enhance with Timestamps
        if config.timestamps and result.csv_path:
            _step_enhance_timestamps(config, result)
        elif config.timestamps:
            result.steps_skipped.append("timestamps (analyze_gemm failed)")
        else:
            result.steps_skipped.append("timestamps")

        # Step 3: Generate GEMM Plots
        if config.plots and result.csv_path:
            _step_generate_plots(config, result)
        elif config.plots:
            result.steps_skipped.append("plots (analyze_gemm failed)")
        else:
            result.steps_skipped.append("plots")

        # Step 4: Generate HTML Report
        if config.html and result.plots_dir:
            _step_generate_html(config, result)
        elif config.html:
            result.steps_skipped.append("html (plots not generated)")
        else:
            result.steps_skipped.append("html")

    except Exception as e:
        result.success = False
        result.errors.append(str(e))

    return result


def _step_analyze_gemm(config: GemmPipelineConfig, result: GemmPipelineResult) -> None:
    """Step 1: Analyze GEMM reports."""
    from ..analysis import analyze_gemm_reports

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 1: Analyze GEMM Reports")
        print("=" * 60)

    reports_dir = config.sweep_dir / "tracelens_analysis"

    if not reports_dir.exists():
        raise FileNotFoundError(f"TraceLens analysis directory not found: {reports_dir}")

    output_file = f"top{config.top_k}_gemm_kernels_time_variance.csv"
    output_path = config.output_dir / output_file

    if config.verbose:
        print(f"  Reports dir: {reports_dir}")
        print(f"  Top-K: {config.top_k}")
        print(f"  Threads: {config.threads}")
        print(f"  Channels: {config.channels}")
        print(f"  Ranks: {config.ranks}")

    csv_path = analyze_gemm_reports(
        base_path=reports_dir,
        threads=config.threads,
        channels=config.channels,
        ranks=config.ranks,
        top_k=config.top_k,
        output_file=str(output_path),
        verbose=config.verbose,
    )

    result.csv_path = csv_path

    if config.verbose:
        print(f"  Output: {csv_path}")

    result.steps_completed.append("analyze_gemm")


def _step_enhance_timestamps(config: GemmPipelineConfig, result: GemmPipelineResult) -> None:
    """Step 2: Enhance with timestamps."""
    from ..processing import enhance_gemm_variance

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Enhance with Timestamps")
        print("=" * 60)

    if result.csv_path is None:
        result.steps_skipped.append("timestamps (no CSV path)")
        return

    output_csv = result.csv_path.with_name(result.csv_path.stem + "_with_timestamps.csv")

    try:
        enhanced_path = enhance_gemm_variance(
            input_csv=result.csv_path,
            base_path=config.sweep_dir,
            output_csv=output_csv,
            verbose=config.verbose,
        )
        result.csv_with_timestamps_path = enhanced_path

        if config.verbose:
            print(f"  Output: {enhanced_path}")

        result.steps_completed.append("timestamps")
    except Exception as e:
        result.errors.append(f"Timestamp enhancement failed: {e}")
        result.steps_skipped.append("timestamps (failed)")


def _step_generate_plots(config: GemmPipelineConfig, result: GemmPipelineResult) -> None:
    """Step 3: Generate GEMM plots."""
    from ..generators import generate_gemm_plots

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Generate GEMM Plots")
        print("=" * 60)

    if result.csv_path is None:
        result.steps_skipped.append("plots (no CSV path)")
        return

    plots_dir = config.output_dir / "plots"

    plot_files = generate_gemm_plots(
        csv_path=result.csv_path,
        output_dir=plots_dir,
        verbose=config.verbose,
    )

    result.plots_dir = plots_dir

    if config.verbose:
        print(f"  Plots directory: {plots_dir}")
        print(f"  Generated {len(plot_files)} plots")

    result.steps_completed.append("plots")


def _step_generate_html(config: GemmPipelineConfig, result: GemmPipelineResult) -> None:
    """Step 4: Generate HTML report."""
    from ..generators import generate_html

    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 4: Generate HTML Report")
        print("=" * 60)

    if result.plots_dir is None:
        result.steps_skipped.append("html (no plots directory)")
        return

    output_html = config.output_dir / "gemm_variance_report.html"

    try:
        html_path = generate_html(
            mode="gemm",
            output=output_html,
            plots_dir=result.plots_dir,
            sweep_dir=config.sweep_dir,
            label=config.sweep_dir.name,
            csv_path=result.csv_path,
            verbose=config.verbose,
        )
        result.html_path = html_path

        if config.verbose:
            print(f"  Output: {html_path}")

        result.steps_completed.append("html")
    except Exception as e:
        result.errors.append(f"HTML generation failed: {e}")
        result.steps_skipped.append("html (failed)")
