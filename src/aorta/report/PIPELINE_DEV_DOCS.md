# Pipeline Commands - Developer Documentation

**Version:** 1.0  
**Date:** January 2026  
**Status:** ✅ Implemented

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Summary](#2-pipeline-summary)
3. [Pipeline GEMM](#3-pipeline-gemm)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Module Details](#5-module-details)
6. [Implementation Plan](#6-implementation-plan)

---

## 1. Overview

The pipeline commands orchestrate multi-step analysis workflows, combining existing commands into end-to-end automation.

### Pipeline Commands

| Command | Description | Steps |
|---------|-------------|-------|
| `pipeline summary` | Complete TraceLens analysis (GPU + NCCL) | 7 steps |
| `pipeline gemm` | GEMM kernel variance analysis | 3 steps |

### Design Principles

1. **Reuse Existing Functions**: Call existing module functions directly (no subprocess)
2. **Configurable Steps**: Enable/disable individual steps via flags
3. **Progress Reporting**: Clear step-by-step progress output
4. **Error Handling**: Continue on non-critical errors, fail fast on critical ones
5. **Dataclass Config**: Clean configuration management

---

## 2. Pipeline Summary

### 2.1 Source Script

**Location:** `scripts/tracelens_single_config/run_full_analysis.py` (529 lines)

### 2.2 Pipeline Steps

| Step | Description | Existing Function | Skippable |
|------|-------------|-------------------|-----------|
| 1 | TraceLens Analysis | `analyze_single_config()` | Yes (`--skip-tracelens`) |
| 2 | Process GPU Timelines | `process_single_config()` | No |
| 3 | Compare GPU Timelines | `compare gpu_timeline` logic | Yes (`--no-gpu-timeline`) |
| 4 | Compare Collective | `compare collective` logic | Yes (`--no-collective`) |
| 5 | Generate Final Excel | `create_final_excel_report()` | Yes (`--no-final-report`) |
| 6 | Generate Plots | `generate_summary_plots()` | Yes (`--no-plots`) |
| 7 | Generate HTML | `generate_html(mode="performance")` | Yes (`--no-html`) |

### 2.3 CLI Specification

```bash
aorta-report pipeline summary \
    -b/--baseline <path>           # Required: Baseline trace directory
    -t/--test <path>               # Required: Test trace directory
    -o/--output <path>             # Required: Output directory
    [--baseline-label <label>]     # Optional: Label for baseline (default: dir name)
    [--test-label <label>]         # Optional: Label for test (default: dir name)
    [--skip-tracelens]             # Skip TraceLens analysis
    [--gpu-timeline/--no-gpu-timeline]    # Default: True
    [--collective/--no-collective]        # Default: True
    [--final-report/--no-final-report]    # Default: True
    [--plots/--no-plots]                  # Default: True
    [--html/--no-html]                    # Default: True
```

### 2.4 Examples

```bash
# Full pipeline
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output

# Skip TraceLens (analysis already done)
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output \
    --skip-tracelens

# Only GPU timeline comparison
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output \
    --no-collective --no-final-report --no-plots --no-html

# Custom labels
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 7.0"
```

### 2.5 Data Flow

```
INPUTS:
├── baseline/                     (trace directory)
│   └── rank_*/pytorch_trace.json.gz
└── test/                         (trace directory)
    └── rank_*/pytorch_trace.json.gz

STEP 1: TraceLens Analysis (--skip-tracelens to skip)
────────────────────────────────────────────────────
baseline/ ──► baseline/tracelens_analysis/
              ├── individual_reports/perf_rank*.xlsx
              └── collective_reports/collective_all_ranks.xlsx
test/     ──► test/tracelens_analysis/
              └── (same structure)

STEP 2: Process GPU Timelines
─────────────────────────────
individual_reports/ ──► gpu_timeline_summary_mean.xlsx

STEP 3: Compare GPU Timeline
────────────────────────────
baseline/gpu_timeline_summary_mean.xlsx ─┬──► gpu_timeline_combined.xlsx
test/gpu_timeline_summary_mean.xlsx ─────┘──► gpu_timeline_comparison.xlsx

STEP 4: Compare Collective
──────────────────────────
baseline/collective_all_ranks.xlsx ──┬──► collective_combined.xlsx
test/collective_all_ranks.xlsx ──────┘──► collective_comparison.xlsx

STEP 5: Final Excel Report
──────────────────────────
gpu_combined + gpu_comparison ──┬──► final_analysis_report.xlsx
coll_combined + coll_comparison ┘

STEP 6: Generate Plots
──────────────────────
final_analysis_report.xlsx ──► plots/*.png (13 files)

STEP 7: Generate HTML
─────────────────────
plots/ ──► performance_analysis_report.html

FINAL OUTPUT:
└── output/
    ├── gpu_timeline_combined.xlsx
    ├── gpu_timeline_comparison.xlsx
    ├── collective_combined.xlsx
    ├── collective_comparison.xlsx
    ├── final_analysis_report.xlsx
    ├── plots/
    │   ├── improvement_chart.png
    │   ├── gpu_time_heatmap.png
    │   └── ... (13 files)
    └── performance_analysis_report.html
```

---

## 3. Pipeline GEMM

### 3.1 Purpose

Automates GEMM kernel variance analysis for sweep experiments, extracting top-K kernels with highest time variance and generating visualization plots.

### 3.2 Pipeline Steps

| Step | Description | Existing Function | Skippable |
|------|-------------|-------------------|-----------|
| 1 | Analyze GEMM Reports | `analyze_gemm_reports()` | No |
| 2 | Enhance with Timestamps | `enhance_gemm_variance()` | Yes (`--no-timestamps`) |
| 3 | Generate GEMM Plots | `generate_gemm_plots()` | Yes (`--no-plots`) |

### 3.3 CLI Specification

```bash
aorta-report pipeline gemm \
    --sweep-dir <path>             # Required: Sweep directory with tracelens_analysis/
    -o/--output <path>             # Required: Output directory
    [--top-k <int>]                # Top K kernels to extract (default: 5)
    [--threads <int>...]           # Thread configs (default: 256, 512)
    [--channels <int>...]          # Channel configs (default: 28, 42, 56, 70)
    [--timestamps/--no-timestamps] # Enhance with timestamps (default: True)
    [--plots/--no-plots]           # Generate plots (default: True)
```

### 3.4 Examples

```bash
# Full GEMM pipeline
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o /path/to/output

# Custom top-k and configs
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o /path/to/output \
    --top-k 10 \
    --threads 256 512 \
    --channels 28 42 56 70

# Skip plots
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o /path/to/output \
    --no-plots

# Skip timestamp enhancement
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o /path/to/output \
    --no-timestamps
```

### 3.5 Data Flow

```
INPUT:
└── sweep_dir/
    └── tracelens_analysis/
        ├── 256thread/
        │   └── individual_reports/
        │       └── perf_*ch_rank*.xlsx
        └── 512thread/
            └── individual_reports/
                └── perf_*ch_rank*.xlsx

STEP 1: Analyze GEMM Reports
────────────────────────────
tracelens_analysis/ ──► top{k}_gemm_kernels_time_variance.csv

STEP 2: Enhance with Timestamps (--no-timestamps to skip)
─────────────────────────────────────────────────────────
variance.csv + trace files ──► variance_with_timestamps.csv

STEP 3: Generate GEMM Plots (--no-plots to skip)
────────────────────────────────────────────────
variance.csv ──► plots/
                 ├── variance_by_threads_boxplot.png
                 ├── variance_by_channels_boxplot.png
                 ├── variance_by_ranks_boxplot.png
                 ├── variance_violin_combined.png
                 └── variance_thread_channel_interaction.png

FINAL OUTPUT:
└── output/
    ├── top5_gemm_kernels_time_variance.csv
    ├── top5_gemm_kernels_time_variance_with_timestamps.csv (if --timestamps)
    └── plots/
        └── *.png (5 files)
```

---

## 4. Implementation Architecture

### 4.1 File Structure

```
src/aorta/report/
├── pipelines/                       # NEW: Pipeline orchestrators
│   ├── __init__.py                  # Package exports
│   ├── summary_pipeline.py          # Summary pipeline (~250 lines)
│   └── gemm_pipeline.py             # GEMM pipeline (~150 lines)
└── cli.py                           # Update pipeline commands
```

### 4.2 Shared Components

```python
# pipelines/__init__.py

from .summary_pipeline import run_summary_pipeline, SummaryPipelineConfig
from .gemm_pipeline import run_gemm_pipeline, GemmPipelineConfig

__all__ = [
    "run_summary_pipeline",
    "SummaryPipelineConfig",
    "run_gemm_pipeline",
    "GemmPipelineConfig",
]
```

---

## 5. Module Details

### 5.1 `pipelines/summary_pipeline.py`

```python
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
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class SummaryPipelineConfig:
    """Configuration for summary pipeline."""
    baseline_path: Path
    test_path: Path
    output_dir: Path
    baseline_label: Optional[str] = None
    test_label: Optional[str] = None
    skip_tracelens: bool = False
    gpu_timeline: bool = True
    collective: bool = True
    final_report: bool = True
    plots: bool = True
    html: bool = True
    verbose: bool = False


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
    
    Returns PipelineResult with success status and generated files.
    """
    result = PipelineResult(
        success=True,
        output_dir=config.output_dir,
    )
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract labels from directory names if not provided
    baseline_label = config.baseline_label or config.baseline_path.name
    test_label = config.test_label or config.test_path.name
    
    try:
        # Step 1: TraceLens Analysis
        if not config.skip_tracelens:
            _step_tracelens_analysis(config, result)
        else:
            result.steps_skipped.append("tracelens_analysis")
        
        # Step 2: Process GPU Timelines
        if config.gpu_timeline:
            _step_process_gpu_timelines(config, result)
        
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
        if config.final_report and config.gpu_timeline and config.collective:
            _step_generate_final_report(config, result, baseline_label, test_label)
        elif config.final_report:
            result.steps_skipped.append("final_report (requires both gpu_timeline and collective)")
        
        # Step 6: Generate Plots
        if config.plots and "final_analysis_report.xlsx" in str(result.files_generated):
            _step_generate_plots(config, result)
        elif config.plots:
            result.steps_skipped.append("plots (requires final_report)")
        
        # Step 7: Generate HTML
        if config.html and result.files_generated.get("plots_dir"):
            _step_generate_html(config, result)
        elif config.html:
            result.steps_skipped.append("html (requires plots)")
            
    except Exception as e:
        result.success = False
        result.errors.append(str(e))
    
    return result


def _step_tracelens_analysis(config: SummaryPipelineConfig, result: PipelineResult) -> None:
    """Step 1: Run TraceLens analysis on baseline and test."""
    from ..analysis import analyze_single_config
    
    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 1: TraceLens Analysis")
        print("=" * 60)
    
    # Analyze baseline
    if config.verbose:
        print(f"\nAnalyzing baseline: {config.baseline_path}")
    analyze_single_config(config.baseline_path, verbose=config.verbose)
    
    # Analyze test
    if config.verbose:
        print(f"\nAnalyzing test: {config.test_path}")
    analyze_single_config(config.test_path, verbose=config.verbose)
    
    result.steps_completed.append("tracelens_analysis")


def _step_process_gpu_timelines(config: SummaryPipelineConfig, result: PipelineResult) -> None:
    """Step 2: Process GPU timelines for both baseline and test."""
    from ..processing import process_single_config
    
    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Process GPU Timelines")
        print("=" * 60)
    
    baseline_reports = config.baseline_path / "tracelens_analysis" / "individual_reports"
    test_reports = config.test_path / "tracelens_analysis" / "individual_reports"
    
    if config.verbose:
        print(f"\nProcessing baseline: {baseline_reports}")
    process_single_config(baseline_reports, verbose=config.verbose)
    
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
    from ..comparison import combine_excel_files, add_gpu_timeline_comparison, save_with_formatting
    
    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Compare GPU Timelines")
        print("=" * 60)
    
    baseline_gpu = config.baseline_path / "tracelens_analysis" / "gpu_timeline_summary_mean.xlsx"
    test_gpu = config.test_path / "tracelens_analysis" / "gpu_timeline_summary_mean.xlsx"
    
    # Combine
    combined = combine_excel_files(baseline_gpu, test_gpu, baseline_label, test_label, verbose=config.verbose)
    
    # Save combined
    combined_path = config.output_dir / "gpu_timeline_combined.xlsx"
    save_with_formatting(combined, combined_path, {})
    result.files_generated["gpu_combined"] = combined_path
    
    # Add comparison
    comparison = add_gpu_timeline_comparison(combined, baseline_label, test_label, verbose=config.verbose)
    
    # Save comparison
    comparison_path = config.output_dir / "gpu_timeline_comparison.xlsx"
    format_columns = {
        "Comparison_By_Rank": ["percent_change"],
        "Summary_Comparison": ["percent_change"],
    }
    save_with_formatting(comparison, comparison_path, format_columns)
    result.files_generated["gpu_comparison"] = comparison_path
    
    result.steps_completed.append("compare_gpu_timeline")


def _step_compare_collective(
    config: SummaryPipelineConfig,
    result: PipelineResult,
    baseline_label: str,
    test_label: str,
) -> None:
    """Step 4: Compare collective/NCCL."""
    from ..comparison import combine_excel_files, add_collective_comparison, save_with_formatting
    from ..comparison.collective_comparison import get_percent_change_columns
    
    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 4: Compare Collective/NCCL")
        print("=" * 60)
    
    baseline_coll = config.baseline_path / "tracelens_analysis" / "collective_reports" / "collective_all_ranks.xlsx"
    test_coll = config.test_path / "tracelens_analysis" / "collective_reports" / "collective_all_ranks.xlsx"
    
    # Combine (filter summary sheets only)
    combined = combine_excel_files(
        baseline_coll, test_coll, baseline_label, test_label,
        filter_summary_only=True, verbose=config.verbose
    )
    
    # Save combined
    combined_path = config.output_dir / "collective_combined.xlsx"
    save_with_formatting(combined, combined_path, {})
    result.files_generated["coll_combined"] = combined_path
    
    # Add comparison
    comparison = add_collective_comparison(combined, baseline_label, test_label, verbose=config.verbose)
    
    # Save comparison
    comparison_path = config.output_dir / "collective_comparison.xlsx"
    format_columns = {}
    for sheet_name, df in comparison.items():
        if sheet_name.endswith("_cmp"):
            pct_cols = get_percent_change_columns(df)
            if pct_cols:
                format_columns[sheet_name] = pct_cols
    save_with_formatting(comparison, comparison_path, format_columns)
    result.files_generated["coll_comparison"] = comparison_path
    
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
    result.steps_completed.append("final_report")


def _step_generate_plots(config: SummaryPipelineConfig, result: PipelineResult) -> None:
    """Step 6: Generate plots."""
    from ..generators import generate_summary_plots
    
    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 6: Generate Plots")
        print("=" * 60)
    
    plots_dir = config.output_dir / "plots"
    
    generate_summary_plots(
        excel_path=result.files_generated["final_report"],
        output_dir=plots_dir,
        verbose=config.verbose,
    )
    
    result.files_generated["plots_dir"] = plots_dir
    result.steps_completed.append("plots")


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
    result.steps_completed.append("html")
```

---

### 5.2 `pipelines/gemm_pipeline.py`

```python
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
    verbose: bool = False


@dataclass
class GemmPipelineResult:
    """Result from GEMM pipeline execution."""
    success: bool
    output_dir: Path
    csv_path: Optional[Path] = None
    csv_with_timestamps_path: Optional[Path] = None
    plots_dir: Optional[Path] = None
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
    output_file = f"top{config.top_k}_gemm_kernels_time_variance.csv"
    
    csv_path = analyze_gemm_reports(
        base_path=reports_dir,
        threads=config.threads,
        channels=config.channels,
        ranks=config.ranks,
        top_k=config.top_k,
        output_file=str(config.output_dir / output_file),
        verbose=config.verbose,
    )
    
    result.csv_path = csv_path
    result.steps_completed.append("analyze_gemm")


def _step_enhance_timestamps(config: GemmPipelineConfig, result: GemmPipelineResult) -> None:
    """Step 2: Enhance with timestamps."""
    from ..processing import enhance_gemm_variance
    
    if config.verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Enhance with Timestamps")
        print("=" * 60)
    
    output_csv = result.csv_path.with_name(
        result.csv_path.stem + "_with_timestamps.csv"
    )
    
    try:
        enhanced_path = enhance_gemm_variance(
            input_csv=result.csv_path,
            base_path=config.sweep_dir,
            output_csv=output_csv,
            verbose=config.verbose,
        )
        result.csv_with_timestamps_path = enhanced_path
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
    
    plots_dir = config.output_dir / "plots"
    
    generate_gemm_plots(
        csv_path=result.csv_path,
        output_dir=plots_dir,
        verbose=config.verbose,
    )
    
    result.plots_dir = plots_dir
    result.steps_completed.append("plots")
```

---

### 5.3 `pipelines/__init__.py`

```python
"""Pipeline orchestrators for multi-step analysis workflows."""

from .summary_pipeline import run_summary_pipeline, SummaryPipelineConfig, PipelineResult
from .gemm_pipeline import run_gemm_pipeline, GemmPipelineConfig, GemmPipelineResult

__all__ = [
    "run_summary_pipeline",
    "SummaryPipelineConfig",
    "PipelineResult",
    "run_gemm_pipeline",
    "GemmPipelineConfig",
    "GemmPipelineResult",
]
```

---

### 5.4 CLI Updates

#### `pipeline summary` Command

```python
@pipeline.command("summary")
@click.option("-b", "--baseline", required=True, type=click.Path(exists=True),
              help="Baseline trace directory")
@click.option("-t", "--test", required=True, type=click.Path(exists=True),
              help="Test trace directory")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory for results")
@click.option("--baseline-label", default=None,
              help="Label for baseline (default: directory name)")
@click.option("--test-label", default=None,
              help="Label for test (default: directory name)")
@click.option("--skip-tracelens", is_flag=True,
              help="Skip TraceLens analysis (if already done)")
@click.option("--gpu-timeline/--no-gpu-timeline", default=True,
              help="Enable/disable GPU timeline comparison")
@click.option("--collective/--no-collective", default=True,
              help="Enable/disable collective comparison")
@click.option("--final-report/--no-final-report", default=True,
              help="Enable/disable final Excel report")
@click.option("--plots/--no-plots", default=True,
              help="Enable/disable plot generation")
@click.option("--html/--no-html", default=True,
              help="Enable/disable HTML report generation")
@click.pass_context
def pipeline_summary(ctx, baseline, test, output, baseline_label, test_label,
                     skip_tracelens, gpu_timeline, collective, final_report, plots, html):
    """Run complete summary analysis pipeline.
    
    Orchestrates the full TraceLens analysis workflow:
    
    \b
    1. TraceLens Analysis (optional, skip with --skip-tracelens)
    2. Process GPU timelines
    3. Compare GPU timelines (baseline vs test)
    4. Compare collective/NCCL metrics
    5. Generate final Excel report
    6. Generate visualization plots
    7. Generate HTML report
    
    \b
    Examples:
      # Full pipeline
      aorta-report pipeline summary \\
          -b /path/to/baseline -t /path/to/test -o /path/to/output
      
      # Skip TraceLens (already done)
      aorta-report pipeline summary \\
          -b /path/to/baseline -t /path/to/test -o /path/to/output \\
          --skip-tracelens
    """
    from pathlib import Path
    from .pipelines import run_summary_pipeline, SummaryPipelineConfig
    
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    
    config = SummaryPipelineConfig(
        baseline_path=Path(baseline),
        test_path=Path(test),
        output_dir=Path(output),
        baseline_label=baseline_label,
        test_label=test_label,
        skip_tracelens=skip_tracelens,
        gpu_timeline=gpu_timeline,
        collective=collective,
        final_report=final_report,
        plots=plots,
        html=html,
        verbose=verbose,
    )
    
    if not quiet:
        click.echo("=" * 60)
        click.echo("SUMMARY ANALYSIS PIPELINE")
        click.echo("=" * 60)
        click.echo(f"Baseline: {baseline}")
        click.echo(f"Test: {test}")
        click.echo(f"Output: {output}")
        click.echo(f"Options: skip_tracelens={skip_tracelens}, gpu_timeline={gpu_timeline}")
        click.echo(f"         collective={collective}, final_report={final_report}")
        click.echo(f"         plots={plots}, html={html}")
    
    result = run_summary_pipeline(config)
    
    if not quiet:
        click.echo("\n" + "=" * 60)
        click.echo("PIPELINE COMPLETE!" if result.success else "PIPELINE FAILED!")
        click.echo("=" * 60)
        
        if result.steps_completed:
            click.echo("\nSteps completed:")
            for step in result.steps_completed:
                click.echo(f"  ✓ {step}")
        
        if result.steps_skipped:
            click.echo("\nSteps skipped:")
            for step in result.steps_skipped:
                click.echo(f"  - {step}")
        
        if result.errors:
            click.echo("\nErrors:")
            for err in result.errors:
                click.echo(f"  ✗ {err}")
        
        if result.files_generated:
            click.echo(f"\nOutput directory: {result.output_dir}")
            click.echo("Generated files:")
            for name, path in result.files_generated.items():
                click.echo(f"  - {path.name}")
    
    if not result.success:
        raise click.ClickException("Pipeline failed")
```

#### `pipeline gemm` Command

```python
@pipeline.command("gemm")
@click.option("--sweep-dir", required=True, type=click.Path(exists=True),
              help="Sweep directory containing tracelens_analysis/")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory for results")
@click.option("--top-k", default=5, type=int,
              help="Number of top kernels to extract (default: 5)")
@click.option("--threads", "-t", multiple=True, type=int, default=(256, 512),
              help="Thread configurations (can specify multiple)")
@click.option("--channels", "-c", multiple=True, type=int, default=(28, 42, 56, 70),
              help="Channel configurations (can specify multiple)")
@click.option("--timestamps/--no-timestamps", default=True,
              help="Enhance with timestamps (default: True)")
@click.option("--plots/--no-plots", default=True,
              help="Generate plots (default: True)")
@click.pass_context
def pipeline_gemm(ctx, sweep_dir, output, top_k, threads, channels, timestamps, plots):
    """Run GEMM variance analysis pipeline.
    
    Analyzes GEMM kernel time variance across configurations:
    
    \b
    1. Analyze GEMM reports to extract top-K kernels
    2. Enhance with timestamps (optional)
    3. Generate variance plots (optional)
    
    \b
    Examples:
      # Full pipeline
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o /path/to/output
      
      # Custom top-k
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o ./output --top-k 10
      
      # Skip plots
      aorta-report pipeline gemm --sweep-dir /path/to/sweep -o ./output --no-plots
    """
    from pathlib import Path
    from .pipelines import run_gemm_pipeline, GemmPipelineConfig
    
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    
    config = GemmPipelineConfig(
        sweep_dir=Path(sweep_dir),
        output_dir=Path(output),
        top_k=top_k,
        threads=list(threads),
        channels=list(channels),
        timestamps=timestamps,
        plots=plots,
        verbose=verbose,
    )
    
    if not quiet:
        click.echo("=" * 60)
        click.echo("GEMM VARIANCE ANALYSIS PIPELINE")
        click.echo("=" * 60)
        click.echo(f"Sweep dir: {sweep_dir}")
        click.echo(f"Output: {output}")
        click.echo(f"Top-K: {top_k}")
        click.echo(f"Threads: {list(threads)}")
        click.echo(f"Channels: {list(channels)}")
        click.echo(f"Options: timestamps={timestamps}, plots={plots}")
    
    result = run_gemm_pipeline(config)
    
    if not quiet:
        click.echo("\n" + "=" * 60)
        click.echo("PIPELINE COMPLETE!" if result.success else "PIPELINE FAILED!")
        click.echo("=" * 60)
        
        if result.steps_completed:
            click.echo("\nSteps completed:")
            for step in result.steps_completed:
                click.echo(f"  ✓ {step}")
        
        if result.steps_skipped:
            click.echo("\nSteps skipped:")
            for step in result.steps_skipped:
                click.echo(f"  - {step}")
        
        if result.errors:
            click.echo("\nErrors:")
            for err in result.errors:
                click.echo(f"  ✗ {err}")
        
        click.echo(f"\nOutput directory: {result.output_dir}")
        if result.csv_path:
            click.echo(f"  - {result.csv_path.name}")
        if result.csv_with_timestamps_path:
            click.echo(f"  - {result.csv_with_timestamps_path.name}")
        if result.plots_dir:
            click.echo(f"  - plots/ (5 files)")
    
    if not result.success:
        raise click.ClickException("Pipeline failed")
```

---

## 6. Implementation Plan

### Phase 1: Create Pipeline Module (~15 min)

| Task | Est. Time |
|------|-----------|
| Create `pipelines/` directory | 2 min |
| Create `pipelines/__init__.py` | 3 min |
| Create dataclasses for configs and results | 10 min |

### Phase 2: Implement Summary Pipeline (~40 min)

| Task | Est. Time |
|------|-----------|
| `run_summary_pipeline()` orchestrator | 10 min |
| `_step_tracelens_analysis()` | 5 min |
| `_step_process_gpu_timelines()` | 5 min |
| `_step_compare_gpu_timeline()` | 5 min |
| `_step_compare_collective()` | 5 min |
| `_step_generate_final_report()` | 3 min |
| `_step_generate_plots()` | 3 min |
| `_step_generate_html()` | 4 min |

### Phase 3: Implement GEMM Pipeline (~20 min)

| Task | Est. Time |
|------|-----------|
| `run_gemm_pipeline()` orchestrator | 5 min |
| `_step_analyze_gemm()` | 5 min |
| `_step_enhance_timestamps()` | 5 min |
| `_step_generate_plots()` | 5 min |

### Phase 4: Update CLI (~15 min)

| Task | Est. Time |
|------|-----------|
| Rename/update `pipeline full` → `pipeline summary` | 5 min |
| Update `pipeline gemm` command | 5 min |
| Update help text | 5 min |

### Phase 5: Testing (~20 min)

| Task | Est. Time |
|------|-----------|
| Test summary pipeline | 10 min |
| Test gemm pipeline | 10 min |

**Total Estimated Time: ~2 hours**

---

## Appendix A: Design Decisions

1. **Rename:** `pipeline full` → `pipeline summary` (as requested)
2. **Dataclass Config:** Clean configuration management with defaults
3. **Result Objects:** Track steps completed, skipped, errors, and files
4. **Direct Function Calls:** Use existing module functions (no subprocess)
5. **Graceful Degradation:** Continue on non-critical errors
6. **Progress Reporting:** Clear step-by-step output with `verbose` flag
7. **Label Auto-Extract:** Use directory names as default labels

