"""HW Queue Eval analysis pipeline.

Orchestrates hw_queue_eval JSON analysis:
- Mode A: Single workload, single run
- Mode B: Single workload, sweep (multiple stream counts)
- Mode C: Multi-workload comparison (baseline vs test directories)

Steps:
1. Load and validate JSON data
2. Generate Excel report(s)
3. Generate plots (optional)
4. Generate HTML report (optional)
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class HWQueuePipelineConfig:
    """Configuration for HW Queue pipeline."""

    # Output directory (required)
    output_dir: Path

    # Mode A/B: Single input file
    input_path: Optional[Path] = None

    # Mode C: Comparison directories
    baseline_dir: Optional[Path] = None
    test_dir: Optional[Path] = None
    baseline_label: Optional[str] = None
    test_label: Optional[str] = None

    # Comparison options
    threshold: float = 0.05  # Regression threshold (5%)

    # Output options
    excel: bool = True
    plots: bool = True
    html: bool = True
    verbose: bool = False

    def get_mode(self) -> str:
        """Determine pipeline mode from config."""
        if self.baseline_dir and self.test_dir:
            return "comparison"
        elif self.input_path:
            return "single_input"
        else:
            raise ValueError(
                "Invalid config: must provide either --input or both --baseline-dir and --test-dir"
            )

    def validate(self) -> None:
        """Validate configuration."""
        if self.input_path and (self.baseline_dir or self.test_dir):
            raise ValueError(
                "Cannot specify both --input and --baseline-dir/--test-dir"
            )

        if self.baseline_dir and not self.test_dir:
            raise ValueError("--baseline-dir requires --test-dir")

        if self.test_dir and not self.baseline_dir:
            raise ValueError("--test-dir requires --baseline-dir")

        if not self.input_path and not (self.baseline_dir and self.test_dir):
            raise ValueError(
                "Must provide either --input or both --baseline-dir and --test-dir"
            )


@dataclass
class HWQueuePipelineResult:
    """Result from HW Queue pipeline execution."""

    success: bool
    mode: str  # "single_run", "sweep", "comparison"
    output_dir: Path

    # Generated files
    files_generated: Dict[str, Path] = field(default_factory=dict)

    # Comparison mode info
    common_workloads: List[str] = field(default_factory=list)
    missing_from_baseline: List[str] = field(default_factory=list)
    missing_from_test: List[str] = field(default_factory=list)

    # Comparison results
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)

    # Pipeline status
    steps_completed: List[str] = field(default_factory=list)
    steps_skipped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def run_hwqueue_pipeline(config: HWQueuePipelineConfig) -> HWQueuePipelineResult:
    """
    Run the HW Queue Eval analysis pipeline.

    Supports three modes:
    - Mode A (single_run): Single workload, single stream count
    - Mode B (sweep): Single workload, multiple stream counts
    - Mode C (comparison): Multiple workloads, baseline vs test

    Returns HWQueuePipelineResult with success status and generated files.
    """
    # Validate config
    try:
        config.validate()
        mode = config.get_mode()
    except ValueError as e:
        return HWQueuePipelineResult(
            success=False,
            mode="unknown",
            output_dir=config.output_dir,
            errors=[str(e)],
        )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "comparison":
        return _run_comparison_pipeline(config)
    else:
        return _run_single_input_pipeline(config)


def _run_single_input_pipeline(config: HWQueuePipelineConfig) -> HWQueuePipelineResult:
    """Run pipeline for single input file (Mode A or Mode B)."""
    from ..processing.hwqueue_loader import HWQueueLoader, HWQueueLoaderError
    from ..generators.hwqueue_excel import generate_hwqueue_excel

    result = HWQueuePipelineResult(
        success=True,
        mode="unknown",  # Will be updated after loading
        output_dir=config.output_dir,
    )

    try:
        # Step 1: Load and detect format
        if config.verbose:
            print("\n" + "=" * 60)
            print("STEP 1: Load JSON Data")
            print("=" * 60)

        format_type, data = HWQueueLoader.load_auto(config.input_path)
        result.mode = format_type

        if config.verbose:
            print(f"  Input: {config.input_path}")
            print(f"  Format: {format_type}")
            if format_type == "single_run":
                print(f"  Workload: {data.workload_name}")
                print(f"  Streams: {data.stream_count}")
                print(f"  Throughput: {data.throughput:.2f} {data.throughput_unit}")
            else:
                print(f"  Workload: {data.workload_name}")
                print(f"  Results: {len(data.results)} stream counts")
                best_s, best_t = data.get_best_throughput()
                print(f"  Best: {best_t:.2f} at {best_s} streams")

        result.steps_completed.append("load_data")

        # Step 2: Generate Excel
        if config.excel:
            if config.verbose:
                print("\n" + "=" * 60)
                print("STEP 2: Generate Excel Report")
                print("=" * 60)

            excel_filename = f"hwqueue_{data.workload_name}_analysis.xlsx"
            excel_path = config.output_dir / excel_filename

            try:
                output_file = generate_hwqueue_excel(data, excel_path, verbose=config.verbose)
                result.files_generated["excel"] = output_file
                result.steps_completed.append("excel")
                if config.verbose:
                    print(f"  ✓ Generated: {output_file.name}")
            except Exception as e:
                result.warnings.append(f"Failed to generate Excel: {e}")
                result.steps_skipped.append("excel (failed)")
        else:
            result.steps_skipped.append("excel (disabled)")

        # Step 3: Generate Plots
        if config.plots:
            if config.verbose:
                print("\n" + "=" * 60)
                print("STEP 3: Generate Plots")
                print("=" * 60)

            try:
                from ..generators.hwqueue_plots import generate_hwqueue_plots

                plots_dir = config.output_dir / "plots"
                plot_files = generate_hwqueue_plots(data, plots_dir, verbose=config.verbose)
                result.files_generated["plots"] = plot_files
                result.steps_completed.append("plots")

                if config.verbose:
                    print(f"  ✓ Generated {len(plot_files)} plot(s)")
            except Exception as e:
                result.warnings.append(f"Failed to generate plots: {e}")
                result.steps_skipped.append("plots (failed)")
        else:
            result.steps_skipped.append("plots (disabled)")

        # Step 4: Generate HTML Report
        if config.html:
            if config.verbose:
                print("\n" + "=" * 60)
                print("STEP 4: Generate HTML Report")
                print("=" * 60)

            try:
                from ..generators.hwqueue_html import generate_hwqueue_html

                plots_dir = config.output_dir / "plots"
                html_filename = f"hwqueue_{data.workload_name}_report.html"
                html_path = config.output_dir / html_filename

                output_file = generate_hwqueue_html(data, plots_dir, html_path, verbose=config.verbose)
                result.files_generated["html"] = output_file
                result.steps_completed.append("html")

                if config.verbose:
                    print(f"  ✓ Generated: {output_file.name}")
            except Exception as e:
                result.warnings.append(f"Failed to generate HTML: {e}")
                result.steps_skipped.append("html (failed)")
        else:
            result.steps_skipped.append("html (disabled)")

    except HWQueueLoaderError as e:
        result.success = False
        result.errors.append(f"Failed to load data: {e}")
    except Exception as e:
        result.success = False
        result.errors.append(f"Unexpected error: {e}")

    return result


def _run_comparison_pipeline(config: HWQueuePipelineConfig) -> HWQueuePipelineResult:
    """Run pipeline for comparison mode (Mode C)."""
    from ..processing.hwqueue_loader import HWQueueLoader, HWQueueLoaderError
    from ..generators.hwqueue_excel import generate_comparison_excel

    result = HWQueuePipelineResult(
        success=True,
        mode="comparison",
        output_dir=config.output_dir,
    )

    try:
        # Step 1: Load and validate directories
        if config.verbose:
            print("\n" + "=" * 60)
            print("STEP 1: Load Comparison Data")
            print("=" * 60)
            print(f"  Baseline: {config.baseline_dir}")
            print(f"  Test: {config.test_dir}")

        # Find common workloads
        common, baseline_only, test_only = HWQueueLoader.find_common_workloads(
            config.baseline_dir, config.test_dir
        )

        result.common_workloads = common
        result.missing_from_test = baseline_only
        result.missing_from_baseline = test_only

        if config.verbose:
            print(f"\n  Common workloads ({len(common)}): {common}")

        # Report missing workloads
        if baseline_only:
            msg = f"Workloads in baseline but not test: {baseline_only}"
            result.warnings.append(msg)
            if config.verbose:
                print(f"  ⚠ {msg}")

        if test_only:
            msg = f"Workloads in test but not baseline: {test_only}"
            result.warnings.append(msg)
            if config.verbose:
                print(f"  ⚠ {msg}")

        if not common:
            raise HWQueueLoaderError("No common workloads found between baseline and test")

        # Load data for common workloads
        baseline_data, test_data, _, _, _ = HWQueueLoader.load_comparison_data(
            config.baseline_dir, config.test_dir
        )

        if config.verbose:
            print(f"\n  Loaded {len(baseline_data)} baseline workloads")
            print(f"  Loaded {len(test_data)} test workloads")
            for wl_name in common:
                b = baseline_data[wl_name]
                t = test_data[wl_name]
                b_best_s, b_best_t = b.get_best_throughput()
                t_best_s, t_best_t = t.get_best_throughput()
                change = ((t_best_t - b_best_t) / b_best_t * 100) if b_best_t > 0 else 0
                print(f"    {wl_name}: {b_best_t:.0f} -> {t_best_t:.0f} ({change:+.1f}%)")

        result.steps_completed.append("load_data")

        # Store labels for later use
        baseline_label = config.baseline_label or config.baseline_dir.name
        test_label = config.test_label or config.test_dir.name

        if config.verbose:
            print(f"\n  Labels: {baseline_label} vs {test_label}")
            print(f"  Threshold: {config.threshold * 100:.1f}%")

        # Step 2: Generate Comparison Excel
        if config.excel:
            if config.verbose:
                print("\n" + "=" * 60)
                print("STEP 2: Generate Comparison Excel")
                print("=" * 60)

            excel_path = config.output_dir / "all_workloads_comparison.xlsx"

            try:
                output_file, regressions, improvements = generate_comparison_excel(
                    baseline_data=baseline_data,
                    test_data=test_data,
                    common_workloads=common,
                    baseline_only=baseline_only,
                    test_only=test_only,
                    output_path=excel_path,
                    baseline_label=baseline_label,
                    test_label=test_label,
                    threshold=config.threshold,
                    verbose=config.verbose,
                )
                result.files_generated["excel"] = output_file
                result.regressions = regressions
                result.improvements = improvements
                result.steps_completed.append("excel")

                if config.verbose:
                    print(f"  ✓ Generated: {output_file.name}")
                    if regressions:
                        print(f"  ⚠ Regressions: {len(regressions)}")
                    if improvements:
                        print(f"  ✓ Improvements: {len(improvements)}")

            except Exception as e:
                result.warnings.append(f"Failed to generate Excel: {e}")
                result.steps_skipped.append("excel (failed)")
        else:
            result.steps_skipped.append("excel (disabled)")

        # Step 3: Generate Comparison Plots
        if config.plots:
            if config.verbose:
                print("\n" + "=" * 60)
                print("STEP 3: Generate Comparison Plots")
                print("=" * 60)

            try:
                from ..generators.hwqueue_plots import generate_comparison_plots

                plots_dir = config.output_dir / "plots"
                plot_files = generate_comparison_plots(
                    baseline_data=baseline_data,
                    test_data=test_data,
                    common_workloads=common,
                    output_dir=plots_dir,
                    baseline_label=baseline_label,
                    test_label=test_label,
                    threshold=config.threshold,
                    verbose=config.verbose,
                )
                result.files_generated["plots"] = plot_files
                result.steps_completed.append("plots")

                if config.verbose:
                    print(f"  ✓ Generated {len(plot_files)} comparison plot(s)")
            except Exception as e:
                result.warnings.append(f"Failed to generate plots: {e}")
                result.steps_skipped.append("plots (failed)")
        else:
            result.steps_skipped.append("plots (disabled)")

        # Step 4: Generate HTML Report
        if config.html:
            if config.verbose:
                print("\n" + "=" * 60)
                print("STEP 4: Generate HTML Report")
                print("=" * 60)

            try:
                from ..generators.hwqueue_html import generate_comparison_html

                plots_dir = config.output_dir / "plots"
                html_path = config.output_dir / "hwqueue_comparison_report.html"

                output_file = generate_comparison_html(
                    baseline_data=baseline_data,
                    test_data=test_data,
                    common_workloads=common,
                    baseline_only=baseline_only,
                    test_only=test_only,
                    regressions=result.regressions,
                    improvements=result.improvements,
                    plots_dir=plots_dir,
                    output_path=html_path,
                    baseline_label=baseline_label,
                    test_label=test_label,
                    threshold=config.threshold,
                    verbose=config.verbose,
                )
                result.files_generated["html"] = output_file
                result.steps_completed.append("html")

                if config.verbose:
                    print(f"  ✓ Generated: {output_file.name}")
            except Exception as e:
                result.warnings.append(f"Failed to generate HTML: {e}")
                result.steps_skipped.append("html (failed)")
        else:
            result.steps_skipped.append("html (disabled)")

    except HWQueueLoaderError as e:
        result.success = False
        result.errors.append(f"Failed to load data: {e}")
    except Exception as e:
        result.success = False
        result.errors.append(f"Unexpected error: {e}")

    return result
