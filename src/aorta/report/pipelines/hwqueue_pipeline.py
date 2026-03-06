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
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Summary Data Classes
# =============================================================================


@dataclass
class SweepSummary:
    """Quick verdict for sweep analysis."""

    # Peak performance
    peak_throughput: float
    peak_streams: int
    peak_efficiency: float
    throughput_unit: str

    # Optimal config
    optimal_streams: int

    # Verdicts (status, explanation)
    scaling_verdict: Tuple[str, str]
    latency_verdict: Tuple[str, str]

    # Raw values for detailed display
    latency_ratio: float  # P99 at peak / P99 at 1 stream

    # Normalized status fields used for logic/CSS: "good", "warning", "poor"
    scaling_status: str = "good"
    latency_status: str = "good"

    def __post_init__(self) -> None:
        """Normalize status fields based on verdicts if not explicitly set."""
        valid_statuses = {"good", "warning", "poor"}

        if self.scaling_status not in valid_statuses:
            self.scaling_status = self._infer_status_from_verdict_text(
                self.scaling_verdict[0]
            )

        if self.latency_status not in valid_statuses:
            self.latency_status = self._infer_status_from_verdict_text(
                self.latency_verdict[0]
            )

    @staticmethod
    def _infer_status_from_verdict_text(verdict_text: str) -> str:
        """Best-effort mapping from legacy verdict text to normalized status."""
        # Preserve existing behavior: treat "✗" as poor, "⚠" as warning, else good.
        if "✗" in verdict_text:
            return "poor"
        if "⚠" in verdict_text:
            return "warning"
        return "good"

    def get_overall_status(self) -> str:
        """Get CSS class for overall status based on normalized statuses."""
        if self.scaling_status == "poor" or self.latency_status == "poor":
            return "poor"
        if self.scaling_status == "warning" or self.latency_status == "warning":
            return "warning"
        return "good"


@dataclass
class ComparisonSummary:
    """Quick verdict for comparison analysis."""

    # Overall verdict
    verdict: str
    verdict_class: str  # CSS class: "improved", "degraded", "mixed", "unchanged"

    # Counts
    total_workloads: int
    num_improved: int
    num_degraded: int
    num_unchanged: int

    # Average change
    avg_change: float

    # Top performers
    top_improvement: Optional[Tuple[str, float]] = None  # (workload, change%)
    top_regression: Optional[Tuple[str, float]] = None  # (workload, change%)

    # Average changes by category
    avg_improved_change: float = 0.0
    avg_degraded_change: float = 0.0


# =============================================================================
# Summary Generation Functions
# =============================================================================


def generate_sweep_summary(data: "SweepData") -> SweepSummary:
    """
    Generate quick verdict for sweep analysis.

    Args:
        data: SweepData object with results

    Returns:
        SweepSummary with verdict information
    """
    # Get throughputs and efficiencies from analysis or compute
    if hasattr(data, "analysis") and data.analysis and data.analysis.throughputs:
        # Access dataclass attributes directly (not a dict)
        throughputs = data.analysis.throughputs
        efficiencies = data.analysis.efficiencies
        stream_counts = data.analysis.stream_counts
    else:
        # Compute from results
        throughputs = [r.throughput for r in data.results]
        stream_counts = [r.stream_count for r in data.results]
        # Compute efficiency (actual / ideal where ideal uses normalized single-stream throughput)
        base_throughput = throughputs[0] if throughputs else 1
        baseline_streams = stream_counts[0] if stream_counts else 1
        if base_throughput > 0 and baseline_streams > 0:
            single_stream_throughput = base_throughput / baseline_streams
        else:
            single_stream_throughput = 0
        efficiencies = [
            (t / (single_stream_throughput * s)) if single_stream_throughput > 0 and s > 0 else 0
            for t, s in zip(throughputs, stream_counts)
        ]

    # Find peak
    if not throughputs:
        # Return empty summary
        return SweepSummary(
            peak_throughput=0,
            peak_streams=0,
            peak_efficiency=0,
            throughput_unit="ops/sec",
            optimal_streams=0,
            scaling_verdict=("⚠ Unknown", "No data available"),
            latency_verdict=("⚠ Unknown", "No data available"),
            latency_ratio=1.0,
        )

    peak_idx = throughputs.index(max(throughputs))
    peak_throughput = throughputs[peak_idx]
    peak_streams = stream_counts[peak_idx]
    peak_efficiency = efficiencies[peak_idx] if peak_idx < len(efficiencies) else 0

    # Get throughput unit
    throughput_unit = data.results[0].throughput_unit if data.results else "ops/sec"

    # Find optimal (highest stream count with efficiency >= 70% AND throughput >= 80% of peak)
    optimal_idx = 0
    for i, (eff, tput, sc) in enumerate(zip(efficiencies, throughputs, stream_counts)):
        if eff >= 0.70 and tput >= 0.80 * peak_throughput:
            optimal_idx = i
    optimal_streams = stream_counts[optimal_idx] if stream_counts else peak_streams

    # Check latency trend (P99 at peak vs P99 at lowest stream count)
    sorted_results = sorted(data.results, key=lambda r: r.stream_count)
    if sorted_results:
        p99_at_1 = sorted_results[0].latency.p99
        # Find the result corresponding to the peak stream count to get correct peak latency
        peak_result = next(
            (r for r in sorted_results if r.stream_count == peak_streams),
            sorted_results[-1],
        )
        p99_at_peak = peak_result.latency.p99
        latency_ratio = p99_at_peak / p99_at_1 if p99_at_1 > 0 else 1.0
    else:
        latency_ratio = 1.0

    # Determine scaling verdict
    if peak_efficiency >= 0.80:
        scaling_verdict = ("✓ Excellent", "Near-linear scaling")
    elif peak_efficiency >= 0.60:
        scaling_verdict = ("✓ Good", "Good scaling with some overhead")
    elif peak_efficiency >= 0.40:
        scaling_verdict = ("⚠ Fair", "Significant overhead at high stream counts")
    else:
        scaling_verdict = ("✗ Poor", "Severe diminishing returns")

    # Determine latency verdict
    if latency_ratio <= 1.5:
        latency_verdict = ("✓ Stable", "Latency remains controlled")
    elif latency_ratio <= 2.5:
        latency_verdict = ("⚠ Increasing", f"P99 latency {latency_ratio:.1f}x higher at peak")
    else:
        latency_verdict = ("✗ Degraded", f"P99 latency {latency_ratio:.1f}x higher - investigate")

    return SweepSummary(
        peak_throughput=peak_throughput,
        peak_streams=peak_streams,
        peak_efficiency=peak_efficiency,
        throughput_unit=throughput_unit,
        optimal_streams=optimal_streams,
        scaling_verdict=scaling_verdict,
        latency_verdict=latency_verdict,
        latency_ratio=latency_ratio,
    )


def generate_comparison_summary(
    baseline_data: Dict[str, "SweepData"],
    test_data: Dict[str, "SweepData"],
    common_workloads: List[str],
    regressions: List[Dict],
    improvements: List[Dict],
    threshold: float = 0.05,
) -> ComparisonSummary:
    """
    Generate quick verdict for comparison analysis.

    Args:
        baseline_data: Dict of workload_name -> SweepData for baseline
        test_data: Dict of workload_name -> SweepData for test
        common_workloads: List of workloads in both baseline and test
        regressions: List of regression dicts
        improvements: List of improvement dicts
        threshold: Regression threshold (fraction)

    Returns:
        ComparisonSummary with verdict information
    """
    total = len(common_workloads)

    # Count unique workloads with regressions/improvements
    workloads_with_regression = set(r["Workload"] for r in regressions)
    workloads_with_improvement = set(i["Workload"] for i in improvements)

    num_improved = len(workloads_with_improvement - workloads_with_regression)
    num_degraded = len(workloads_with_regression - workloads_with_improvement)
    # Workloads with both are counted as mixed - we'll count them as degraded for simplicity
    num_mixed = len(workloads_with_regression & workloads_with_improvement)
    num_degraded += num_mixed
    num_unchanged = total - num_improved - num_degraded

    # Calculate throughput changes for each workload
    changes = []
    improved_changes = []
    degraded_changes = []

    threshold_pct = threshold * 100

    for workload in common_workloads:
        b_best_s, b_best_t = baseline_data[workload].get_best_throughput()
        t_best_s, t_best_t = test_data[workload].get_best_throughput()
        change_pct = ((t_best_t - b_best_t) / b_best_t * 100) if b_best_t > 0 else 0
        changes.append((workload, change_pct))

        if change_pct > threshold_pct:
            improved_changes.append(change_pct)
        elif change_pct < -threshold_pct:
            degraded_changes.append(change_pct)

    avg_change = sum(c[1] for c in changes) / len(changes) if changes else 0
    avg_improved_change = sum(improved_changes) / len(improved_changes) if improved_changes else 0
    avg_degraded_change = sum(degraded_changes) / len(degraded_changes) if degraded_changes else 0

    # Find top improvement and regression
    improvements_sorted = sorted(changes, key=lambda x: x[1], reverse=True)
    regressions_sorted = sorted(changes, key=lambda x: x[1])

    top_improvement = None
    if improvements_sorted and improvements_sorted[0][1] > threshold_pct:
        top_improvement = improvements_sorted[0]

    top_regression = None
    if regressions_sorted and regressions_sorted[0][1] < -threshold_pct:
        top_regression = regressions_sorted[0]

    # Determine overall verdict
    if num_degraded == 0 and num_improved > 0:
        verdict = "✓ ALL IMPROVED"
        verdict_class = "improved"
    elif num_improved == 0 and num_degraded > 0:
        verdict = "✗ ALL DEGRADED"
        verdict_class = "degraded"
    elif num_improved > num_degraded * 2:
        verdict = "✓ MOSTLY IMPROVED"
        verdict_class = "improved"
    elif num_degraded > num_improved * 2:
        verdict = "⚠ MOSTLY DEGRADED"
        verdict_class = "degraded"
    elif num_improved > 0 or num_degraded > 0:
        verdict = "⚠ MIXED RESULTS"
        verdict_class = "mixed"
    else:
        verdict = "─ NO SIGNIFICANT CHANGE"
        verdict_class = "unchanged"

    return ComparisonSummary(
        verdict=verdict,
        verdict_class=verdict_class,
        total_workloads=total,
        num_improved=num_improved,
        num_degraded=num_degraded,
        num_unchanged=num_unchanged,
        avg_change=avg_change,
        top_improvement=top_improvement,
        top_regression=top_regression,
        avg_improved_change=avg_improved_change,
        avg_degraded_change=avg_degraded_change,
    )


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

                # Generate summary for sweep data
                summary = None
                if format_type == "sweep":
                    summary = generate_sweep_summary(data)
                    if config.verbose:
                        print(f"  Summary: {summary.scaling_verdict[0]}, {summary.latency_verdict[0]}")

                output_file = generate_hwqueue_html(
                    data, plots_dir, html_path, summary=summary, verbose=config.verbose
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

                # Generate comparison summary
                summary = generate_comparison_summary(
                    baseline_data=baseline_data,
                    test_data=test_data,
                    common_workloads=common,
                    regressions=result.regressions,
                    improvements=result.improvements,
                    threshold=config.threshold,
                )

                if config.verbose:
                    print(f"  Summary: {summary.verdict}")
                    if summary.top_improvement:
                        print(f"    Top improvement: {summary.top_improvement[0]} (+{summary.top_improvement[1]:.1f}%)")
                    if summary.top_regression:
                        print(f"    Top regression: {summary.top_regression[0]} ({summary.top_regression[1]:.1f}%)")

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
                    summary=summary,
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
