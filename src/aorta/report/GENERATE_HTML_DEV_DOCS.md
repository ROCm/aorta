# `generate html` Command - Developer Documentation

**Version:** 1.0  
**Date:** January 2026  
**Status:** Implemented

---

## Table of Contents

1. [Overview](#1-overview)
2. [Command Modes](#2-command-modes)
3. [Pipeline Flows](#3-pipeline-flows)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Expected Plot Files](#5-expected-plot-files)
6. [Usage Examples](#6-usage-examples)
7. [Output Format](#7-output-format)
8. [Source Script Mapping](#8-source-script-mapping)

---

## 1. Overview

The `generate html` command creates self-contained HTML reports with embedded base64 images. It consolidates two previously separate scripts into a unified interface with two modes:

| Mode | Purpose | Source Script |
|------|---------|---------------|
| `sweep` | GEMM kernel variance comparison | `create_embeded_html_report.py` |
| `performance` | GPU/NCCL performance analysis | `create_final_html.py` |

### Key Features

- **Self-contained HTML**: All images embedded as base64 for easy sharing
- **Clear plot status**: Reports expected/found/missing plots before generation
- **Graceful degradation**: Missing plots show placeholder messages instead of breaking
- **Consistent interface**: Both modes use the same CLI structure

---

## 2. Command Modes

### 2.1 Sweep Mode (`--mode sweep`)

**Purpose:** Compare GEMM kernel variance between two experiment sweeps (A/B testing).

**Use Case:** Comparing different RCCL configurations (thread counts, channel counts) to analyze kernel timing variance.

```bash
aorta-report generate html --mode sweep \
    --sweep1 /path/to/sweep1 \
    --sweep2 /path/to/sweep2 \
    --label1 "Baseline" \
    --label2 "Optimized" \
    -o comparison.html
```

**Required Options:**
- `--sweep1`: First sweep directory
- `--sweep2`: Second sweep directory

**Optional Options:**
- `--label1`: Label for first sweep (default: directory name)
- `--label2`: Label for second sweep (default: directory name)

**Input Structure:**
```
sweep_dir/
└── tracelens_analysis/
    └── plots/
        ├── variance_by_threads_boxplot.png
        ├── variance_by_channels_boxplot.png
        ├── variance_by_ranks_boxplot.png
        ├── variance_violin_combined.png
        └── variance_thread_channel_interaction.png
```

### 2.2 Performance Mode (`--mode performance`)

**Purpose:** Generate GPU/NCCL performance analysis report comparing baseline vs test.

**Use Case:** Visualizing results from the full analysis pipeline showing GPU metrics, cross-rank comparisons, and NCCL operations.

```bash
aorta-report generate html --mode performance \
    --plots-dir /path/to/output/plots \
    -o performance_report.html
```

**Required Options:**
- `--plots-dir`: Directory containing pre-generated plots

**Input Structure:**
```
plots_dir/
├── improvement_chart.png
├── abs_time_comparison.png
├── gpu_time_heatmap.png
├── total_time_by_rank.png
├── computation_time_by_rank.png
├── total_comm_time_by_rank.png
├── idle_time_by_rank.png
├── gpu_time_change_percentage_summary_by_rank.png
├── NCCL_Communication_Latency_comparison.png
├── NCCL_Algorithm_Bandwidth_comparison.png
├── NCCL_Bus_Bandwidth_comparison.png
├── NCCL_Performance_Percentage_Change_comparison.png
└── NCCL_Total_Communication_Latency_comparison.png
```

---

## 3. Pipeline Flows

### 3.1 Sweep Mode Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GEMM SWEEP ANALYSIS PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: TraceLens Analysis (per sweep)                                     │
│  ───────────────────────────────────────                                    │
│  run_tracelens_analysis.sh <sweep_dir>                                      │
│      ↓                                                                      │
│  sweep/tracelens_analysis/                                                  │
│      ├── 256thread/individual_reports/perf_*ch_rank*.xlsx                   │
│      └── 512thread/individual_reports/perf_*ch_rank*.xlsx                   │
│                                                                             │
│  STEP 2: Analyze GEMM Reports                                               │
│  ────────────────────────────                                               │
│  analyze_gemm_reports.py --base-path sweep/tracelens_analysis               │
│      ↓                                                                      │
│  sweep/tracelens_analysis/top5_gemm_kernels_time_variance.csv               │
│                                                                             │
│  STEP 3: Plot GEMM Variance                                                 │
│  ──────────────────────────                                                 │
│  plot_gemm_variance.py --csv-path top5_gemm_kernels_time_variance.csv       │
│      ↓                                                                      │
│  sweep/tracelens_analysis/plots/variance_*.png                              │
│                                                                             │
│  STEP 4: Generate HTML (THIS COMMAND)                                       │
│  ────────────────────────────────────                                       │
│  aorta-report generate html --mode sweep --sweep1 ... --sweep2 ...          │
│      ↓                                                                      │
│  comparison.html (side-by-side GEMM variance comparison)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Performance Mode Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FULL ANALYSIS PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: TraceLens Analysis                                                 │
│  ─────────────────────────────                                              │
│  run_tracelens_single_config.sh                                             │
│      ↓                                                                      │
│  baseline/tracelens_analysis/individual_reports/                            │
│  test/tracelens_analysis/individual_reports/                                │
│                                                                             │
│  STEP 2: GPU Timeline Processing                                            │
│  ────────────────────────────────                                           │
│  process_gpu_timeline.py                                                    │
│      ↓                                                                      │
│  gpu_timeline_summary_mean.xlsx                                             │
│                                                                             │
│  STEP 3: Combine & Compare                                                  │
│  ─────────────────────────────                                              │
│  combine_reports.py → add_comparison_sheets.py                              │
│  combine_reports.py → add_collective_comparison.py                          │
│      ↓                                                                      │
│  output/gpu_timeline_comparison.xlsx                                        │
│  output/collective_comparison.xlsx                                          │
│                                                                             │
│  STEP 4: Create Final Report                                                │
│  ───────────────────────────────                                            │
│  create_final_report.py                                                     │
│      ↓                                                                      │
│  output/final_analysis_report.xlsx                                          │
│                                                                             │
│  STEP 5: Generate Plots                                                     │
│  ────────────────────────                                                   │
│  create_final_plots.py --input final_analysis_report.xlsx                   │
│      ↓                                                                      │
│  output/plots/*.png                                                         │
│                                                                             │
│  STEP 6: Generate HTML (THIS COMMAND)                                       │
│  ─────────────────────────                                                  │
│  aorta-report generate html --mode performance --plots-dir output/plots     │
│      ↓                                                                      │
│  performance_analysis_report.html                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Architecture

### 4.1 File Structure

```
src/aorta/report/
├── cli.py                               # CLI definition with generate_html command
├── generators/
│   ├── __init__.py                      # Exports generate_html, image_to_base64
│   ├── html_generator.py                # Unified entry point + shared utilities
│   ├── sweep_comparison.py              # Sweep mode generator
│   └── performance_report.py            # Performance mode generator
└── templates/
    ├── __init__.py                      # Exports all templates
    ├── sweep_comparison_template.py     # HTML template for sweep comparison
    └── performance_report_template.py   # HTML template + chart configs
```

### 4.2 Module Responsibilities

#### `html_generator.py` (Entry Point)

```python
# Shared utilities
def image_to_base64(image_path: Path) -> Optional[str]
def validate_directory(path: Path, name: str) -> None
def find_plots_directory(base_path: Path) -> Path
def check_plots_status(plots_dir, expected_files) -> Tuple[found, missing]
def print_plot_status(expected, found, missing, plots_dir, label)

# Main entry point
def generate_html(mode, output, sweep1, sweep2, label1, label2, plots_dir, verbose)

# Mode handlers (private)
def _generate_sweep_mode(...)
def _generate_performance_mode(...)
```

#### `sweep_comparison.py`

```python
SWEEP_PLOT_FILES = {
    "threads": "variance_by_threads_boxplot.png",
    "channels": "variance_by_channels_boxplot.png",
    "ranks": "variance_by_ranks_boxplot.png",
    "violin": "variance_violin_combined.png",
    "interaction": "variance_thread_channel_interaction.png",
}

def generate_sweep_comparison(plots_dir1, plots_dir2, label1, label2, 
                              sweep1_path, sweep2_path, output, found1, found2, verbose)
```

#### `performance_report.py`

```python
PERFORMANCE_PLOT_FILES = {
    # Built dynamically from OVERALL_GPU_CHARTS, CROSS_RANK_CHARTS, NCCL_CHARTS
}

def create_chart_html(chart_config, found) -> str
def create_section_html(title, charts, found) -> str
def generate_performance_report(plots_dir, output, found, verbose)
```

### 4.3 Data Flow

```
CLI (cli.py)
    │
    ▼
html_generator.generate_html(mode, ...)
    │
    ├── validate inputs
    ├── find plots directory
    ├── check_plots_status() → print status
    │
    ├── [sweep mode] ──────────────────────────────►  sweep_comparison.generate_sweep_comparison()
    │                                                         │
    │                                                         ├── load images as base64
    │                                                         ├── call template
    │                                                         └── write HTML
    │
    └── [performance mode] ────────────────────────►  performance_report.generate_performance_report()
                                                              │
                                                              ├── create section HTML
                                                              ├── embed images
                                                              └── write HTML
```

---

## 5. Expected Plot Files

### 5.1 Sweep Mode Plots

| Key | Filename | Description |
|-----|----------|-------------|
| `threads` | `variance_by_threads_boxplot.png` | Box plot of variance by thread count |
| `channels` | `variance_by_channels_boxplot.png` | Box plot of variance by channel count |
| `ranks` | `variance_by_ranks_boxplot.png` | Box plot of variance by rank |
| `violin` | `variance_violin_combined.png` | Combined violin plots |
| `interaction` | `variance_thread_channel_interaction.png` | Thread-channel interaction plot |

### 5.2 Performance Mode Plots

#### Overall GPU Charts
| Filename | Description |
|----------|-------------|
| `improvement_chart.png` | Percentage change overview |
| `abs_time_comparison.png` | Absolute time comparison |

#### Cross-Rank Charts
| Filename | Description |
|----------|-------------|
| `gpu_time_heatmap.png` | Performance heatmap by rank |
| `total_time_by_rank.png` | Total execution time by rank |
| `computation_time_by_rank.png` | Computation time by rank |
| `total_comm_time_by_rank.png` | Communication time by rank |
| `idle_time_by_rank.png` | Idle time by rank |
| `gpu_time_change_percentage_summary_by_rank.png` | Detailed % change by metric |

#### NCCL Charts
| Filename | Description |
|----------|-------------|
| `NCCL_Communication_Latency_comparison.png` | Communication latency |
| `NCCL_Algorithm_Bandwidth_comparison.png` | Algorithm bandwidth |
| `NCCL_Bus_Bandwidth_comparison.png` | Bus bandwidth |
| `NCCL_Performance_Percentage_Change_comparison.png` | Performance % change |
| `NCCL_Total_Communication_Latency_comparison.png` | Total latency |

---

## 6. Usage Examples

### 6.1 Sweep Comparison

```bash
# Basic usage
aorta-report generate html --mode sweep \
    --sweep1 experiments/sweep_baseline \
    --sweep2 experiments/sweep_optimized \
    -o comparison.html

# With custom labels
aorta-report generate html --mode sweep \
    --sweep1 experiments/sweep_20251121 \
    --sweep2 experiments/sweep_20251124 \
    --label1 "ROCm 6.0" \
    --label2 "ROCm 7.0" \
    -o rocm_comparison.html
```

### 6.2 Performance Report

```bash
# From pipeline output
aorta-report generate html --mode performance \
    --plots-dir ./output/plots \
    -o performance_report.html

# With verbose output
aorta-report -v generate html --mode performance \
    --plots-dir /path/to/analysis/plots \
    -o detailed_report.html
```

### 6.3 As Part of Full Pipeline

```bash
# Run full analysis pipeline (generates plots)
aorta-report pipeline full \
    --baseline /path/to/baseline \
    --test /path/to/test \
    --output /path/to/output \
    --plots

# Then generate HTML from the plots
aorta-report generate html --mode performance \
    --plots-dir /path/to/output/plots \
    -o final_report.html
```

---

## 7. Output Format

### 7.1 Console Output (Plot Status)

```
============================================================
Plot Status (Sweep 1: baseline_sweep)
============================================================
Directory: /path/to/baseline_sweep/tracelens_analysis/plots

Expected plots (5):
  [✓ FOUND]   variance_by_threads_boxplot.png
  [✓ FOUND]   variance_by_channels_boxplot.png
  [✓ FOUND]   variance_by_ranks_boxplot.png
  [✓ FOUND]   variance_violin_combined.png
  [✗ MISSING] variance_thread_channel_interaction.png

Summary: 4 found, 1 missing
============================================================

Encoding images for baseline_sweep...
Encoding images for optimized_sweep...

✓ HTML report created: comparison.html
  File size: 2.45 MB
```

### 7.2 HTML Output Structure

**Sweep Mode:**
- Side-by-side comparison tables
- Each section shows both sweeps
- Missing images show placeholder message

**Performance Mode:**
- Sequential sections (Overall GPU, Cross-Rank, NCCL)
- Charts with descriptions
- Missing charts show placeholder message

---

## 8. Source Script Mapping

| Original Script | New Command | Notes |
|-----------------|-------------|-------|
| `gemm_analysis/create_embeded_html_report.py` | `aorta-report generate html --mode sweep` | Side-by-side comparison |
| `gemm_analysis/html_template.py` | `templates/sweep_comparison_template.py` | Migrated template |
| `tracelens_single_config/create_final_html.py` | `aorta-report generate html --mode performance` | Performance report |
| `tracelens_single_config/html_report_config.py` | `templates/performance_report_template.py` | Migrated template |

---

## Appendix A: Adding New Plot Types

To add a new plot to an existing mode:

### For Sweep Mode

1. Add entry to `SWEEP_PLOT_FILES` in `sweep_comparison.py`:
   ```python
   SWEEP_PLOT_FILES = {
       ...
       "new_plot": "new_plot_filename.png",
   }
   ```

2. Update `sweep_comparison_template.py` to include the new plot in the HTML.

### For Performance Mode

1. Add chart config to appropriate list in `performance_report_template.py`:
   ```python
   OVERALL_GPU_CHARTS = [
       ...
       {
           "name": "New Chart Name",
           "file": "new_chart.png",
           "alt": "Alt text",
           "description": "Chart description.",
       },
   ]
   ```

The `PERFORMANCE_PLOT_FILES` dict is built dynamically from these lists.

---

## Appendix B: Troubleshooting

### "Sweep mode requires both --sweep1 and --sweep2"
Ensure both sweep directories are provided when using `--mode sweep`.

### "Performance mode requires --plots-dir"
Provide the path to the plots directory. This is typically `output/plots/` after running the full pipeline.

### Many plots showing as MISSING
The plots need to be pre-generated by earlier pipeline steps:
- **Sweep mode:** Run `plot_gemm_variance.py` first
- **Performance mode:** Run `create_final_plots.py` first, or use `aorta-report pipeline full --plots`

### HTML file is very large
This is expected - images are embedded as base64. A typical report with 10+ plots can be 5-10 MB.

