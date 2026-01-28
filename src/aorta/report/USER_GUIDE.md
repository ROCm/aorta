# aorta-report User Guide

**Version:** 1.0  
**Date:** January 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Command Reference](#4-command-reference)
   - [analyze](#41-analyze-commands)
   - [compare](#42-compare-commands)
   - [generate](#43-generate-commands)
   - [process](#44-process-commands)
   - [pipeline](#45-pipeline-commands)
5. [Common Workflows](#5-common-workflows)
6. [Output Files](#6-output-files)
7. [Implementation Status](#7-implementation-status)

---

## 1. Overview

`aorta-report` is a unified CLI tool for TraceLens analysis and report generation. It provides commands for:

- **Analyzing** PyTorch profiler traces
- **Comparing** baseline and test configurations
- **Generating** Excel reports, plots, and HTML dashboards
- **Processing** GPU timeline and NCCL communication data
- **Running pipelines** that orchestrate multiple steps

### Global Options

```bash
aorta-report [OPTIONS] COMMAND [ARGS]...

Options:
  --version        Show version and exit
  -v, --verbose    Enable verbose output
  --quiet          Suppress non-error output
  --help           Show help message
```

---

## 2. Installation

```bash
# From the aorta directory
cd aorta
pip install -e .

# Verify installation
aorta-report --version
aorta-report --help
```

### Dependencies

- Python 3.8+
- pandas
- openpyxl
- matplotlib
- seaborn
- click

---

## 3. Quick Start

### Full Analysis Pipeline (Recommended)

The easiest way to run a complete analysis:

```bash
# Compare baseline vs test with full analysis
aorta-report pipeline summary \
    -b /path/to/baseline/traces \
    -t /path/to/test/traces \
    -o /path/to/output

# If TraceLens analysis is already done
aorta-report pipeline summary \
    -b /path/to/baseline/traces \
    -t /path/to/test/traces \
    -o /path/to/output \
    --skip-tracelens
```

This generates:
- GPU timeline comparison Excel
- Collective/NCCL comparison Excel
- Final comprehensive report
- Visualization plots
- HTML report

### GEMM Variance Analysis

```bash
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o /path/to/output
```

---

## 4. Command Reference

### 4.1 Analyze Commands

Commands for running TraceLens analysis on trace data.

#### `analyze single`

Analyze a single configuration trace directory.

```bash
aorta-report analyze single <TRACE_DIR> [OPTIONS]

Arguments:
  TRACE_DIR              Path to trace directory containing rank subdirectories

Options:
  --individual-only      Generate only individual reports
  --collective-only      Generate only collective report
  --geo-mean             Use geometric mean for timeline aggregation
  --short-kernel-threshold INT  Threshold for short kernel study (default: 50 µs)
  --topk-ops INT         Number of top operations to include (default: 100)
  -o, --output PATH      Output directory
```

**Examples:**

```bash
# Basic analysis
aorta-report analyze single /path/to/traces

# Individual reports only
aorta-report analyze single /path/to/traces --individual-only

# Custom output directory
aorta-report analyze single /path/to/traces -o ./results
```

---

#### `analyze sweep`

Analyze a sweep directory with multiple configurations.

```bash
aorta-report analyze sweep <SWEEP_DIR> [OPTIONS]

Arguments:
  SWEEP_DIR              Path to sweep directory containing tracelens_analysis/

Options:
  --geo-mean             Use geometric mean instead of arithmetic mean
  -o, --output PATH      Output directory
```

**Examples:**

```bash
aorta-report analyze sweep /path/to/sweep_20251124
aorta-report analyze sweep /path/to/sweep --geo-mean
```

---

#### `analyze gemm`

Analyze GEMM kernels from TraceLens reports.

```bash
aorta-report analyze gemm <REPORTS_DIR> [OPTIONS]

Arguments:
  REPORTS_DIR            Path to tracelens_analysis directory

Options:
  -t, --threads INT      Thread configurations (can specify multiple, default: 256, 512)
  -c, --channels INT     Channel configurations (can specify multiple, default: 28, 42, 56, 70)
  -r, --ranks INT        Ranks to analyze (default: 0-7)
  --top-k INT            Number of top kernels to extract (default: 5)
  -o, --output PATH      Output CSV file
```

**Examples:**

```bash
# Default analysis
aorta-report analyze gemm /path/to/tracelens_analysis

# Custom top-k
aorta-report analyze gemm /path/to/reports --top-k 10 -o gemm_analysis.csv

# Custom configurations
aorta-report analyze gemm /path/to/reports -t 256 -t 512 -c 28 -c 42
```

---

### 4.2 Compare Commands

Commands for comparing baseline and test TraceLens reports.

#### `compare gpu_timeline`

Compare two GPU timeline reports.

```bash
aorta-report compare gpu_timeline [OPTIONS]

Options:
  -b, --baseline PATH    Path to baseline gpu_timeline_summary_mean.xlsx (required)
  -t, --test PATH        Path to test gpu_timeline_summary_mean.xlsx (required)
  --baseline-label TEXT  Label for baseline (default: extracted from path)
  --test-label TEXT      Label for test (default: extracted from path)
  -o, --output PATH      Output Excel file path (required)
```

**Examples:**

```bash
# Basic comparison
aorta-report compare gpu_timeline \
    -b baseline/gpu_timeline_summary_mean.xlsx \
    -t test/gpu_timeline_summary_mean.xlsx \
    -o comparison.xlsx

# With custom labels
aorta-report compare gpu_timeline \
    -b baseline/gpu.xlsx -t test/gpu.xlsx \
    --baseline-label "ROCm 6.0" --test-label "ROCm 7.0" \
    -o comparison.xlsx
```

**Output Sheets:**
- `Summary` - Combined summary data
- `All_Ranks_Combined` - Combined per-rank data
- `Comparison_By_Rank` - Per-rank comparison with percent_change
- `Summary_Comparison` - Overall comparison

---

#### `compare collective`

Compare two collective/NCCL reports.

```bash
aorta-report compare collective [OPTIONS]

Options:
  -b, --baseline PATH    Path to baseline collective_all_ranks.xlsx (required)
  -t, --test PATH        Path to test collective_all_ranks.xlsx (required)
  --baseline-label TEXT  Label for baseline (default: extracted from path)
  --test-label TEXT      Label for test (default: extracted from path)
  -o, --output PATH      Output Excel file path (required)
```

**Examples:**

```bash
aorta-report compare collective \
    -b baseline/collective_all_ranks.xlsx \
    -t test/collective_all_ranks.xlsx \
    -o collective_comparison.xlsx
```

**Output Sheets:**
- `nccl_summary_implicit_sync` - Combined summary data
- `nccl_summary_long` - Combined long operation data
- `nccl_implicit_sync_cmp` - Comparison with latency/bandwidth metrics
- `nccl_long_cmp` - Long operation comparison

---

### 4.3 Generate Commands

Commands for generating reports and visualizations.

#### `generate html`

Generate HTML report with embedded images.

```bash
aorta-report generate html [OPTIONS]

Options:
  --mode [sweep|performance]  Report mode (required)
  --sweep1 PATH          [sweep mode] First sweep directory
  --sweep2 PATH          [sweep mode] Second sweep directory
  --label1 TEXT          [sweep mode] Label for first sweep
  --label2 TEXT          [sweep mode] Label for second sweep
  --plots-dir PATH       [performance mode] Directory with pre-generated plots
  -o, --output PATH      Output HTML file (required)
```

**Examples:**

```bash
# Sweep comparison (GEMM variance)
aorta-report generate html --mode sweep \
    --sweep1 ./exp1 --sweep2 ./exp2 \
    --label1 "Baseline" --label2 "Optimized" \
    -o comparison.html

# Performance report
aorta-report generate html --mode performance \
    --plots-dir ./output/plots \
    -o performance_report.html
```

---

#### `generate excel`

Generate comprehensive final Excel report.

```bash
aorta-report generate excel [OPTIONS]

Options:
  --gpu-combined PATH     GPU combined report file (required)
  --gpu-comparison PATH   GPU comparison report file (required)
  --coll-combined PATH    Collective combined report file (required)
  --coll-comparison PATH  Collective comparison report file (required)
  --baseline-label TEXT   Label for baseline (default: "Baseline")
  --test-label TEXT       Label for test (default: "Test")
  -o, --output PATH       Output Excel file (required)
```

**Examples:**

```bash
aorta-report generate excel \
    --gpu-combined gpu_combined.xlsx \
    --gpu-comparison gpu_comparison.xlsx \
    --coll-combined coll_combined.xlsx \
    --coll-comparison coll_comparison.xlsx \
    --baseline-label "ROCm 6.0" --test-label "ROCm 7.0" \
    -o final_report.xlsx
```

**Output Structure:**
- `Summary_Dashboard` - Key metrics at a glance (visible, first sheet)
- `GPU_Summary_Cmp`, `GPU_ByRank_Cmp` - GPU comparisons (visible)
- `NCCL_*_Cmp` - NCCL comparisons (visible)
- `*_Raw` sheets - Raw data (hidden, accessible via Unhide)

---

#### `generate plots`

Generate visualization plots.

```bash
aorta-report generate plots [OPTIONS]

Options:
  -i, --input PATH       Input file (Excel for summary, CSV for gemm)
  --excel-input PATH     Excel report file (for --type all)
  --gemm-csv PATH        GEMM variance CSV (for --type all)
  -o, --output PATH      Output directory for PNG files (required)
  --type [all|summary|gemm]  Type of plots (default: all)
  --dpi INT              DPI for output images (default: 150)
```

**Examples:**

```bash
# Summary plots from Excel report
aorta-report generate plots \
    -i final_report.xlsx \
    -o ./plots/ \
    --type summary

# GEMM plots from CSV
aorta-report generate plots \
    -i gemm_variance.csv \
    -o ./plots/ \
    --type gemm

# All plots
aorta-report generate plots \
    --excel-input final_report.xlsx \
    --gemm-csv gemm_variance.csv \
    -o ./plots/ \
    --type all
```

**Summary Plots (13 files):**
- `improvement_chart.png` - Percent improvement bar chart
- `abs_time_comparison.png` - Absolute time comparison
- `*_by_rank.png` - Metrics by rank (4 files)
- `gpu_time_heatmap.png` - Heatmap of changes
- `gpu_time_change_percentage_summary_by_rank.png` - 2×4 grid
- `NCCL_*.png` - NCCL comparison charts (5 files)

**GEMM Plots (5 files):**
- `variance_by_threads_boxplot.png`
- `variance_by_channels_boxplot.png`
- `variance_by_ranks_boxplot.png`
- `variance_violin_combined.png`
- `variance_thread_channel_interaction.png`

---

### 4.4 Process Commands

Data processing utilities.

#### `process gpu-timeline`

Process GPU timeline data from TraceLens reports.

```bash
aorta-report process gpu-timeline <INPUT_DIR> [OPTIONS]

Arguments:
  INPUT_DIR              Path to reports directory or sweep directory

Options:
  --mode [auto|single|sweep]  Processing mode (default: auto)
  --geo-mean             Use geometric mean instead of arithmetic mean
  -o, --output PATH      Output file path
```

**Examples:**

```bash
# Auto-detect mode
aorta-report process gpu-timeline /path/to/reports

# Single config mode
aorta-report process gpu-timeline /path/to/individual_reports --mode single

# Sweep mode with geometric mean
aorta-report process gpu-timeline /path/to/sweep --mode sweep --geo-mean
```

---

#### `process comms`

Process NCCL communication data from collective reports.

```bash
aorta-report process comms <SWEEP_DIR> [OPTIONS]

Arguments:
  SWEEP_DIR              Path to sweep directory containing tracelens_analysis/

Options:
  -o, --output PATH      Output directory
```

**Examples:**

```bash
aorta-report process comms /path/to/sweep
aorta-report process comms /path/to/sweep -o ./nccl_analysis/
```

**Output Files:**
- `nccl_master_all_configs.xlsx` - For pivot tables
- `nccl_master_all_configs.csv` - For pandas/scripts

---

#### `process gemm-variance`

Enhance GEMM variance CSV with kernel timestamps.

```bash
aorta-report process gemm-variance <INPUT_CSV> [OPTIONS]

Arguments:
  INPUT_CSV              CSV file with GEMM variance data

Options:
  --base-path PATH       Base path to sweep directory (required)
  --tolerance FLOAT      Duration matching tolerance (default: 0.01 = 1%)
  -o, --output PATH      Output CSV file
```

**Examples:**

```bash
aorta-report process gemm-variance ./gemm_variance.csv \
    --base-path /path/to/sweep

aorta-report process gemm-variance ./variance.csv \
    --base-path /path/to/sweep \
    --tolerance 0.02 \
    -o ./enhanced.csv
```

**Added Columns:**
- `min_duration_timestamp_ms` - When shortest instance occurred
- `max_duration_timestamp_ms` - When longest instance occurred
- `time_between_min_max_ms` - Time difference between occurrences

---

### 4.5 Pipeline Commands

End-to-end analysis pipelines that orchestrate multiple steps.

#### `pipeline summary`

Run complete summary analysis pipeline (GPU + NCCL comparison).

```bash
aorta-report pipeline summary [OPTIONS]

Options:
  -b, --baseline PATH    Baseline trace directory (required)
  -t, --test PATH        Test trace directory (required)
  -o, --output PATH      Output directory (required)
  --baseline-label TEXT  Label for baseline (default: directory name)
  --test-label TEXT      Label for test (default: directory name)
  --skip-tracelens       Skip TraceLens analysis (if already done)
  --gpu-timeline/--no-gpu-timeline    Enable/disable GPU timeline (default: True)
  --collective/--no-collective        Enable/disable collective (default: True)
  --final-report/--no-final-report    Enable/disable final report (default: True)
  --plots/--no-plots                  Enable/disable plots (default: True)
  --html/--no-html                    Enable/disable HTML report (default: True)
```

**Pipeline Steps:**
1. TraceLens Analysis (skippable)
2. Process GPU Timelines
3. Compare GPU Timelines
4. Compare Collective/NCCL
5. Generate Final Excel Report
6. Generate Visualization Plots
7. Generate HTML Report

**Examples:**

```bash
# Full pipeline
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output

# Skip TraceLens (already done)
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output \
    --skip-tracelens

# Custom labels
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output \
    --baseline-label "ROCm 6.0" \
    --test-label "ROCm 7.0"

# Only GPU timeline comparison
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output \
    --no-collective --no-final-report --no-plots --no-html
```

---

#### `pipeline gemm`

Run GEMM variance analysis pipeline.

```bash
aorta-report pipeline gemm [OPTIONS]

Options:
  --sweep-dir PATH       Sweep directory containing tracelens_analysis/ (required)
  -o, --output PATH      Output directory (required)
  --top-k INT            Number of top kernels to extract (default: 5)
  -t, --threads INT      Thread configurations (can specify multiple)
  -c, --channels INT     Channel configurations (can specify multiple)
  --timestamps/--no-timestamps  Enhance with timestamps (default: True)
  --plots/--no-plots            Generate plots (default: True)
```

**Pipeline Steps:**
1. Analyze GEMM Reports
2. Enhance with Timestamps (optional)
3. Generate GEMM Plots (optional)

**Examples:**

```bash
# Full pipeline
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o /path/to/output

# Custom top-k
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o ./output \
    --top-k 10

# Skip plots
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o ./output \
    --no-plots

# Custom configurations
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o ./output \
    -t 256 -t 512 -c 28 -c 42 -c 56 -c 70
```

---

## 5. Common Workflows

### Workflow 1: Compare Two Configurations

```bash
# Step 1: Run TraceLens analysis on both (or use existing)
aorta-report analyze single /path/to/baseline/traces
aorta-report analyze single /path/to/test/traces

# Step 2: Process GPU timelines
aorta-report process gpu-timeline /path/to/baseline/tracelens_analysis/individual_reports
aorta-report process gpu-timeline /path/to/test/tracelens_analysis/individual_reports

# Step 3: Compare GPU timelines
aorta-report compare gpu_timeline \
    -b baseline/tracelens_analysis/gpu_timeline_summary_mean.xlsx \
    -t test/tracelens_analysis/gpu_timeline_summary_mean.xlsx \
    -o output/gpu_comparison.xlsx

# Step 4: Compare collective/NCCL
aorta-report compare collective \
    -b baseline/tracelens_analysis/collective_reports/collective_all_ranks.xlsx \
    -t test/tracelens_analysis/collective_reports/collective_all_ranks.xlsx \
    -o output/collective_comparison.xlsx

# Step 5: Generate plots
aorta-report generate plots \
    -i output/gpu_comparison.xlsx \
    -o output/plots/ \
    --type summary
```

**OR use the pipeline (recommended):**

```bash
aorta-report pipeline summary \
    -b /path/to/baseline \
    -t /path/to/test \
    -o /path/to/output
```

---

### Workflow 2: GEMM Kernel Variance Analysis

```bash
# Full pipeline
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep \
    -o /path/to/output \
    --top-k 10

# OR step by step:
aorta-report analyze gemm /path/to/sweep/tracelens_analysis --top-k 10 -o variance.csv
aorta-report process gemm-variance variance.csv --base-path /path/to/sweep -o enhanced.csv
aorta-report generate plots -i variance.csv -o ./plots/ --type gemm
```

---

### Workflow 3: Generate Reports from Existing Comparisons

```bash
# If you already have comparison files:
aorta-report generate excel \
    --gpu-combined gpu_combined.xlsx \
    --gpu-comparison gpu_comparison.xlsx \
    --coll-combined coll_combined.xlsx \
    --coll-comparison coll_comparison.xlsx \
    -o final_report.xlsx

aorta-report generate plots \
    -i final_report.xlsx \
    -o ./plots/ \
    --type summary

aorta-report generate html \
    --mode performance \
    --plots-dir ./plots/ \
    -o report.html
```

---

## 6. Output Files

### Summary Pipeline Output

```
output/
├── gpu_timeline_combined.xlsx       # Combined baseline + test GPU data
├── gpu_timeline_comparison.xlsx     # GPU comparison with percent_change
├── collective_combined.xlsx         # Combined NCCL data
├── collective_comparison.xlsx       # NCCL comparison
├── final_analysis_report.xlsx       # Comprehensive report
│   ├── Summary_Dashboard (visible)
│   ├── GPU_Summary_Cmp (visible)
│   ├── GPU_ByRank_Cmp (visible)
│   ├── NCCL_*_Cmp (visible)
│   └── *_Raw (hidden)
├── plots/
│   ├── improvement_chart.png
│   ├── abs_time_comparison.png
│   ├── gpu_time_heatmap.png
│   ├── total_time_by_rank.png
│   ├── computation_time_by_rank.png
│   ├── total_comm_time_by_rank.png
│   ├── idle_time_by_rank.png
│   ├── gpu_time_change_percentage_summary_by_rank.png
│   ├── NCCL_Communication_Latency_comparison.png
│   ├── NCCL_Algorithm_Bandwidth_comparison.png
│   ├── NCCL_Bus_Bandwidth_comparison.png
│   ├── NCCL_Total_Communication_Latency_comparison.png
│   └── NCCL_Performance_Percentage_Change_comparison.png
└── performance_analysis_report.html  # Self-contained HTML report
```

### GEMM Pipeline Output

```
output/
├── top5_gemm_kernels_time_variance.csv
├── top5_gemm_kernels_time_variance_with_timestamps.csv
└── plots/
    ├── variance_by_threads_boxplot.png
    ├── variance_by_channels_boxplot.png
    ├── variance_by_ranks_boxplot.png
    ├── variance_violin_combined.png
    └── variance_thread_channel_interaction.png
```

---

## 7. Implementation Status

### ✅ Fully Implemented

| Command | Description |
|---------|-------------|
| `analyze single` | Analyze single configuration traces |
| `analyze sweep` | Analyze sweep with multiple configs |
| `analyze gemm` | Analyze GEMM kernels |
| `compare gpu_timeline` | Compare two GPU timeline reports |
| `compare collective` | Compare two collective/NCCL reports |
| `generate html` | Generate HTML report |
| `generate excel` | Generate comprehensive Excel report |
| `generate plots` | Generate visualization plots |
| `process gpu-timeline` | Process GPU timeline data |
| `process comms` | Process NCCL communication data |
| `process gemm-variance` | Enhance GEMM variance with timestamps |
| `pipeline summary` | Complete summary analysis pipeline |
| `pipeline gemm` | GEMM variance analysis pipeline |

### ⏸️ Planned (Not Yet Implemented)

| Command | Description |
|---------|-------------|
| `compare runs` | Compare N TraceLens runs (N-way comparison) |

---

## Appendix: Troubleshooting

### Common Issues

1. **"Baseline/Test analysis not found"**
   - Run without `--skip-tracelens` first, or ensure `tracelens_analysis/` exists

2. **"Individual reports not found"**
   - Ensure TraceLens analysis completed successfully
   - Check for `individual_reports/` directory with `perf_rank*.xlsx` files

3. **"Collective reports not found"**
   - Ensure collective analysis was run (not `--individual-only`)
   - Check for `collective_reports/collective_all_ranks.xlsx`

4. **Excel formatting issues**
   - Ensure `openpyxl` is installed
   - Hidden sheets can be revealed: Right-click sheet tab → Unhide

### Getting Help

```bash
# General help
aorta-report --help

# Command group help
aorta-report analyze --help
aorta-report compare --help
aorta-report generate --help
aorta-report process --help
aorta-report pipeline --help

# Specific command help
aorta-report pipeline summary --help
aorta-report compare gpu_timeline --help
```

