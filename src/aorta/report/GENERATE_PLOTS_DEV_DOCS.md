# `generate plots` Command - Developer Documentation

**Version:** 1.1  
**Date:** January 2026  
**Status:** ✅ Implemented

---

## Table of Contents

1. [Overview](#1-overview)
2. [Source Scripts Analysis](#2-source-scripts-analysis)
3. [Command Specification](#3-command-specification)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Module Details](#5-module-details)
6. [Data Flow](#6-data-flow)
7. [Implementation Order](#7-implementation-order)
8. [Output Files](#8-output-files)

---

## 1. Overview

The `generate plots` command creates visualization plots from analysis reports. It merges functionality from two existing scripts into a unified interface with two plot types.

### Plot Types

| Type | Source Script | Input | Description |
|------|---------------|-------|-------------|
| `summary` | `create_final_plots.py` | Excel report | GPU timeline & NCCL comparison charts |
| `gemm` | `plot_gemm_variance.py` | CSV file | GEMM kernel variance distribution plots |

### Scripts Being Merged

| Script | Lines | Location |
|--------|-------|----------|
| `create_final_plots.py` | 333 | `scripts/tracelens_single_config/` |
| `plot_gemm_variance.py` | 423 | `scripts/gemm_analysis/` |
| **Total** | **756** | - |

---

## 2. Source Scripts Analysis

### 2.1 `create_final_plots.py` → Plot Type: `summary`

**Input:** Final Excel report (output of `generate excel`)  
**Required Sheets:** `Summary_Dashboard`, `GPU_ByRank_Cmp`, `NCCL_ImplicitSyncCmp`

#### Functions & Output Files

| Function | Output File | Description |
|----------|-------------|-------------|
| `plot_improvement_chart()` | `improvement_chart.png` | Horizontal bar chart showing % improvement per metric |
| `plot_abs_time_comparison()` | `abs_time_comparison.png` | Grouped bar chart: baseline vs test absolute times |
| `create_gpu_time_accross_all_ranks()` | `{metric}_by_rank.png` | Line plots showing metric values across ranks (4 files) |
| `create_gpu_time_change_percentage_summaryby_rank()` | `gpu_time_change_percentage_summary_by_rank.png` | 2×4 grid of bar charts per metric type |
| `create_gpu_time_heatmap()` | `gpu_time_heatmap.png` | Seaborn heatmap: percent_change by (metric × rank) |
| `create_nccl_charts()` | `NCCL_*.png` | 5 NCCL comparison charts |

**Total Output Files:** ~13 PNG files

---

### 2.2 `plot_gemm_variance.py` → Plot Type: `gemm`

**Input:** GEMM variance CSV (output of `analyze gemm` + optional `process gemm-variance`)  
**Required Columns:** `threads`, `channel`, `rank`, `time_diff_us`, `kernel_name`

#### Functions & Output Files

| Function | Output File | Description |
|----------|-------------|-------------|
| `create_boxplot_by_threads()` | `variance_by_threads_boxplot.png` | Box plot: variance distribution by thread count |
| `create_boxplot_by_channels()` | `variance_by_channels_boxplot.png` | Box plot: variance distribution by channel count |
| `create_boxplot_by_ranks()` | `variance_by_ranks_boxplot.png` | Box plot: variance distribution by rank |
| `create_violin_plot_combined()` | `variance_violin_combined.png` | 1×3 grid: violin plots for all dimensions |
| `create_interaction_plot()` | `variance_thread_channel_interaction.png` | Line plot: thread-channel interaction |

**Total Output Files:** 5 PNG files

---

## 3. Command Specification

### CLI Interface

```bash
# Summary plots (GPU timeline + NCCL from Excel)
aorta-report generate plots \
    -i final_report.xlsx \
    -o ./plots/ \
    --type summary

# GEMM variance plots (from CSV)
aorta-report generate plots \
    -i gemm_variance.csv \
    -o ./plots/ \
    --type gemm

# All plots (requires both inputs)
aorta-report generate plots \
    --excel-input final_report.xlsx \
    --gemm-csv gemm_variance.csv \
    -o ./plots/ \
    --type all
```

### Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `-i, --input` | Conditional | - | Input file (Excel for summary, CSV for gemm) |
| `--excel-input` | For `all` | - | Excel report file (for `--type all`) |
| `--gemm-csv` | For `all` | - | GEMM variance CSV (for `--type all`) |
| `-o, --output` | Yes | - | Output directory for PNG files |
| `--type` | No | `all` | Plot type: `summary`, `gemm`, or `all` |
| `--dpi` | No | `150` | DPI for output images |

### Validation Rules

1. If `--type summary`: `-i` must be an Excel file with required sheets
2. If `--type gemm`: `-i` must be a CSV file with required columns
3. If `--type all`: Both `--excel-input` and `--gemm-csv` must be provided

---

## 4. Implementation Architecture

### 4.1 File Structure

```
src/aorta/report/
└── generators/
    ├── __init__.py                    # Update exports
    ├── html_generator.py              # Existing
    ├── excel_report.py                # Existing
    ├── plot_generator.py              # NEW: Thin orchestrator (~100 lines)
    └── plot_helper/                   # NEW: Internal package
        ├── __init__.py                # Exports all plot functions
        ├── common.py                  # Shared utilities, colors, styles
        │
        │ # Summary plots (from create_final_plots.py)
        ├── summary_dashboard.py       # improvement_chart, abs_time_comparison
        ├── gpu_by_rank.py             # GPU metrics by rank line plots
        ├── gpu_percent_change.py      # 2x4 grid of percent change bars
        ├── gpu_heatmap.py             # Seaborn heatmap
        ├── nccl_charts.py             # NCCL comparison charts
        │
        │ # GEMM plots (from plot_gemm_variance.py)
        ├── gemm_data.py               # CSV reader, statistics
        ├── gemm_boxplots.py           # Boxplots by threads/channels/ranks
        ├── gemm_violin.py             # Combined violin plot
        └── gemm_interaction.py        # Thread-channel interaction plot
```

### 4.2 File Size Estimates

| File | Functions | Lines (est.) |
|------|-----------|--------------|
| **Common** | | |
| `common.py` | `configure_style()`, `COLORS`, `save_figure()` | ~50 |
| **Summary Plots** | | |
| `summary_dashboard.py` | `plot_improvement_chart()`, `plot_abs_time_comparison()`, `get_labels()` | ~80 |
| `gpu_by_rank.py` | `plot_gpu_metrics_by_rank()` | ~70 |
| `gpu_percent_change.py` | `plot_gpu_percent_change_grid()` | ~60 |
| `gpu_heatmap.py` | `plot_gpu_heatmap()` | ~50 |
| `nccl_charts.py` | `plot_nccl_comparison()`, `plot_nccl_percent_change()` | ~120 |
| **GEMM Plots** | | |
| `gemm_data.py` | `read_gemm_csv_data()`, `print_statistics()` | ~60 |
| `gemm_boxplots.py` | `create_boxplot()`, `plot_by_threads()`, `plot_by_channels()`, `plot_by_ranks()` | ~100 |
| `gemm_violin.py` | `plot_variance_violin_combined()` | ~80 |
| `gemm_interaction.py` | `plot_thread_channel_interaction()` | ~60 |
| **Orchestrator** | | |
| `plot_generator.py` | `generate_summary_plots()`, `generate_gemm_plots()`, `generate_plots()` | ~100 |
| **Total** | | **~830** |

---

## 5. Module Details

### 5.1 `plot_helper/common.py`

Shared utilities, colors, and styling for all plots.

```python
"""Common utilities for plot generation."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Color Palette
# =============================================================================

COLORS = {
    "positive": "#2ecc71",    # Green - improvements
    "negative": "#e74c3c",    # Red - regressions
    "baseline": "#3498db",    # Blue - baseline data
    "test": "#e67e22",        # Orange - test data
    "neutral": "#95a5a6",     # Gray - neutral
}

# Extended palette for multi-series
PALETTE_MULTI = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c"]


# =============================================================================
# Plot Configuration
# =============================================================================

DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (10, 6)


def configure_style() -> None:
    """Configure matplotlib/seaborn style for consistent plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def remove_spines(ax) -> None:
    """Remove all spines from an axis."""
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)


def save_figure(
    fig,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> Path:
    """Save figure and optionally close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return output_path


def get_improvement_colors(values) -> list:
    """Return green/red colors based on positive/negative values."""
    return [COLORS["positive"] if v > 0 else COLORS["negative"] for v in values]
```

---

### 5.2 `plot_helper/summary_dashboard.py`

Dashboard-level plots from Summary_Dashboard sheet.

```python
"""Summary dashboard plots: improvement chart and absolute time comparison."""

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .common import (
    COLORS, DEFAULT_DPI, DEFAULT_FIGSIZE,
    remove_spines, save_figure, get_improvement_colors,
)


def get_labels_from_excel(excel_path: Path) -> List[str]:
    """Extract baseline/test labels from Summary_Dashboard sheet."""
    df = pd.read_excel(excel_path, sheet_name="Summary_Dashboard")
    cols = df.columns.tolist()
    return [cols[1], cols[2]]  # Baseline and Test column names


def plot_improvement_chart(
    excel_path: Path,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Create horizontal bar chart of percent improvement.
    
    Reads Summary_Dashboard sheet, plots Metric vs Improvement (%).
    Green bars for positive (better), red for negative (worse).
    """
    df = pd.read_excel(excel_path, sheet_name="Summary_Dashboard")
    
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    
    colors = get_improvement_colors(df["Improvement (%)"])
    ax.barh(df["Metric"], df["Improvement (%)"], color=colors)
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)
    remove_spines(ax)
    
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_xlabel("Change (%)", fontsize=12)
    ax.set_title(
        "GPU Metrics Percentage Change (Test vs Baseline)\n(Positive = Test is better)",
        fontsize=14, fontweight="bold",
    )
    
    plt.tight_layout()
    return save_figure(fig, output_dir / "improvement_chart.png", dpi)


def plot_abs_time_comparison(
    excel_path: Path,
    output_dir: Path,
    labels: List[str],
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Create grouped bar chart of baseline vs test absolute times.
    
    Reads Summary_Dashboard sheet, plots side-by-side bars for each metric.
    """
    df = pd.read_excel(excel_path, sheet_name="Summary_Dashboard")
    
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    
    x = np.arange(len(df))
    width = 0.35
    colors = [COLORS["baseline"], COLORS["test"]]
    
    for i, label in enumerate(labels):
        offset = (i - len(labels) / 2 + 0.5) * width
        ax.bar(x + offset, df[label], width, label=label, color=colors[i])
    
    ax.xaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
    ax.set_axisbelow(True)
    remove_spines(ax)
    
    ax.set_xlabel("Metric Type", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("GPU Metrics Absolute Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"], rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    return save_figure(fig, output_dir / "abs_time_comparison.png", dpi)
```

---

### 5.3 `plot_helper/gpu_by_rank.py`

Line plots showing GPU metrics across ranks.

```python
"""GPU metrics by rank line plots."""

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from .common import COLORS, DEFAULT_DPI, save_figure


METRICS_TO_PLOT = ["total_time", "computation_time", "total_comm_time", "idle_time"]


def plot_gpu_metrics_by_rank(
    excel_path: Path,
    output_dir: Path,
    labels: List[str],
    metrics: List[str] = None,
    dpi: int = DEFAULT_DPI,
) -> List[Path]:
    """
    Create line plots for GPU metrics across ranks.
    
    Reads GPU_ByRank_Cmp sheet, creates one plot per metric type.
    Each plot shows baseline vs test values across all ranks.
    
    Returns list of generated file paths.
    """
    df = pd.read_excel(excel_path, sheet_name="GPU_ByRank_Cmp")
    metrics = metrics or METRICS_TO_PLOT
    
    output_files = []
    colors = [COLORS["baseline"], COLORS["test"]]
    markers = ["o", "s"]
    
    for metric in metrics:
        metric_df = df[df["type"] == metric]
        if metric_df.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, label in enumerate(labels):
            col_name = f"{label}_time_ms"
            if col_name in metric_df.columns:
                ax.plot(
                    metric_df["rank"],
                    metric_df[col_name],
                    marker=markers[i],
                    linewidth=2,
                    markersize=8,
                    color=colors[i],
                    label=label,
                )
        
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
        ax.set_axisbelow(True)
        
        ax.set_xlabel("Rank", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title(f"{metric} Comparison across all ranks", fontsize=14, fontweight="bold")
        ax.legend()
        
        plt.tight_layout()
        output_path = save_figure(fig, output_dir / f"{metric}_by_rank.png", dpi)
        output_files.append(output_path)
    
    return output_files
```

---

### 5.4 `plot_helper/gpu_percent_change.py`

2×4 grid of percent change bar charts.

```python
"""GPU percent change grid plot."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .common import DEFAULT_DPI, save_figure, get_improvement_colors


METRIC_TYPES = [
    "busy_time", "computation_time", "exposed_comm_time", "exposed_memcpy_time",
    "idle_time", "total_comm_time", "total_memcpy_time", "total_time",
]


def plot_gpu_percent_change_grid(
    excel_path: Path,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Create 2x4 grid of percent change bar charts by rank.
    
    Reads GPU_ByRank_Cmp sheet, creates one subplot per metric type.
    Each subplot shows percent_change for all ranks as bar chart.
    """
    df = pd.read_excel(excel_path, sheet_name="GPU_ByRank_Cmp")
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 8))
    
    for i, metric_type in enumerate(METRIC_TYPES):
        ax = axes[i // 4, i % 4]
        type_df = df[df["type"] == metric_type]
        
        if type_df.empty:
            ax.set_visible(False)
            continue
        
        colors = get_improvement_colors(type_df["percent_change"])
        ax.bar(type_df["rank"].astype(str), type_df["percent_change"], color=colors)
        
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
        ax.set_axisbelow(True)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Percent Change (%)")
        ax.set_title(metric_type, fontsize=10)
    
    fig.suptitle(
        "GPU Metrics Percent Change by Rank\n(Positive = Better)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    return save_figure(fig, output_dir / "gpu_time_change_percentage_summary_by_rank.png", dpi)
```

---

### 5.5 `plot_helper/gpu_heatmap.py`

Seaborn heatmap of percent change.

```python
"""GPU percent change heatmap."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .common import DEFAULT_DPI, save_figure


def plot_gpu_heatmap(
    excel_path: Path,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Create heatmap of percent_change by metric type and rank.
    
    Reads GPU_ByRank_Cmp sheet, pivots to (metric × rank) matrix,
    and creates color-coded heatmap (green=better, red=worse).
    """
    df = pd.read_excel(excel_path, sheet_name="GPU_ByRank_Cmp")
    pivot_df = df.pivot(index="type", columns="rank", values="percent_change")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Percent Change (%)"},
        ax=ax,
    )
    
    ax.set_title(
        "GPU Metric Percentage Change by Rank (HeatMap)\n(Positive = Better Test)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Metric Type", fontsize=12)
    
    plt.tight_layout()
    return save_figure(fig, output_dir / "gpu_time_heatmap.png", dpi)
```

---

### 5.6 `plot_helper/nccl_charts.py`

NCCL comparison charts.

```python
"""NCCL comparison charts."""

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .common import COLORS, DEFAULT_DPI, save_figure, get_improvement_colors


NCCL_METRICS = {
    "NCCL Communication Latency": {
        "y_col": "comm_latency_mean",
        "y_label": "Communication Latency (ms)",
    },
    "NCCL Algorithm Bandwidth": {
        "y_col": "algo bw (GB/s)_mean",
        "y_label": "Algorithm Bandwidth (GB/s)",
    },
    "NCCL Bus Bandwidth": {
        "y_col": "bus bw (GB/s)_mean",
        "y_label": "Bus Bandwidth (GB/s)",
    },
    "NCCL Total Communication Latency": {
        "y_col": "Total comm latency (ms)",
        "y_label": "Total Communication Latency (ms)",
    },
}

NCCL_PERCENT_METRICS = {
    "Comm Latency": "percent_change_comm_latency_mean",
    "Algo BW": "percent_change_algo bw (GB/s)_mean",
    "Bus BW": "percent_change_bus bw (GB/s)_mean",
}


def plot_nccl_comparison(
    excel_path: Path,
    output_dir: Path,
    labels: List[str],
    dpi: int = DEFAULT_DPI,
) -> List[Path]:
    """
    Create NCCL metric comparison bar charts.
    
    Reads NCCL_ImplicitSyncCmp sheet, creates grouped bar charts
    for each metric (latency, bandwidth).
    """
    df = pd.read_excel(excel_path, sheet_name="NCCL_ImplicitSyncCmp")
    df["label"] = df["Collective name"] + "\n" + df["In msg nelems"].astype(str)
    
    x = np.arange(len(df))
    width = 0.35
    colors = [COLORS["baseline"], COLORS["test"]]
    output_files = []
    
    for title, config in NCCL_METRICS.items():
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, label in enumerate(labels):
            col_name = f"{label}_{config['y_col']}"
            if col_name in df.columns:
                offset = (i - len(labels) / 2 + 0.5) * width
                ax.bar(x + offset, df[col_name], width, label=label, color=colors[i])
        
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Collective Operation (Message Size)", fontsize=12)
        ax.set_ylabel(config["y_label"], fontsize=12)
        ax.set_title(f"{title} Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        
        plt.tight_layout()
        filename = f'{title.replace(" ", "_")}_comparison.png'
        output_files.append(save_figure(fig, output_dir / filename, dpi))
    
    return output_files


def plot_nccl_percent_change(
    excel_path: Path,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Create 1x3 grid of NCCL percent change horizontal bar charts.
    """
    df = pd.read_excel(excel_path, sheet_name="NCCL_ImplicitSyncCmp")
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
    
    for i, (title, col_name) in enumerate(NCCL_PERCENT_METRICS.items()):
        ax = axes[i]
        if col_name not in df.columns:
            ax.set_visible(False)
            continue
        
        colors = get_improvement_colors(df[col_name])
        ax.barh(df["In msg nelems"].astype(str), df[col_name], color=colors)
        
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
        ax.set_axisbelow(True)
        ax.set_xlabel("Percent Change (%)")
        ax.set_title(f"{title}\nPercent Change (Positive = better)")
    
    fig.suptitle(
        "NCCL Performance Percentage Change By Message Size",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    return save_figure(fig, output_dir / "NCCL_Performance_Percentage_Change_comparison.png", dpi)
```

---

### 5.7 `plot_helper/gemm_data.py`

GEMM CSV reader and statistics.

```python
"""GEMM variance data loading and statistics."""

import csv
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def read_gemm_csv_data(csv_path: Path) -> Dict[str, Any]:
    """
    Read GEMM variance CSV and organize by dimensions.
    
    Returns:
        {
            "threads": {256: [values], 512: [values]},
            "channels": {28: [values], 42: [values], ...},
            "ranks": {0: [values], 1: [values], ...},
            "all": [list of row dicts],
        }
    """
    data = {
        "threads": defaultdict(list),
        "channels": defaultdict(list),
        "ranks": defaultdict(list),
        "all": [],
    }
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                threads = int(row["threads"])
                channel = int(row["channel"])
                rank = int(row["rank"])
                time_diff = float(row["time_diff_us"])
                
                data["threads"][threads].append(time_diff)
                data["channels"][channel].append(time_diff)
                data["ranks"][rank].append(time_diff)
                data["all"].append({
                    "threads": threads,
                    "channel": channel,
                    "rank": rank,
                    "time_diff": time_diff,
                    "kernel_name": row["kernel_name"],
                })
            except (ValueError, KeyError) as e:
                continue
    
    return data


def _calculate_median(values: List[float]) -> float:
    """Calculate median of a list of values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def print_gemm_statistics(data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Print and return summary statistics."""
    stats = {}
    
    if verbose:
        print("\n" + "=" * 70)
        print("VARIANCE DISTRIBUTION STATISTICS")
        print("=" * 70)
    
    for dimension, label_fmt in [
        ("threads", "{} threads"),
        ("channels", "{}ch"),
        ("ranks", "Rank {}"),
    ]:
        stats[dimension] = {}
        if verbose:
            print(f"\nBy {dimension.title()}:")
        
        for key in sorted(data[dimension].keys()):
            values = data[dimension][key]
            mean_val = sum(values) / len(values)
            median_val = _calculate_median(values)
            
            stats[dimension][key] = {
                "mean": mean_val,
                "median": median_val,
                "max": max(values),
                "count": len(values),
            }
            
            if verbose:
                label = label_fmt.format(key)
                print(f"  {label}: mean={mean_val:.2f}us, median={median_val:.2f}us, "
                      f"max={max(values):.2f}us, n={len(values)}")
    
    if verbose:
        print("=" * 70 + "\n")
    
    return stats
```

---

### 5.8 `plot_helper/gemm_boxplots.py`

GEMM variance boxplots.

```python
"""GEMM variance boxplot generators."""

from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt

from .common import DEFAULT_DPI, save_figure


def _create_boxplot(
    data_dict: Dict[int, List[float]],
    output_path: Path,
    label_fmt: str,
    xlabel: str,
    title: str,
    colors: List[str],
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generic boxplot creation helper."""
    fig, ax = plt.subplots(figsize=figsize)
    
    keys_list = sorted(data_dict.keys())
    plot_data = [data_dict[k] for k in keys_list]
    labels = [label_fmt.format(k) for k in keys_list]
    
    bp = ax.boxplot(
        plot_data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )
    
    # Handle color assignment
    if colors == "viridis":
        colors = plt.cm.viridis([i / len(keys_list) for i in range(len(keys_list))])
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel("Time Difference (us)", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_figure(fig, output_path, dpi)


def plot_variance_by_threads(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Create boxplot of variance by thread count."""
    return _create_boxplot(
        data_dict=data["threads"],
        output_path=output_dir / "variance_by_threads_boxplot.png",
        label_fmt="{} threads",
        xlabel="Thread Configuration",
        title="GEMM Kernel Time Variance by Thread Count",
        colors=["lightblue", "lightcoral"],
        figsize=(10, 6),
        dpi=dpi,
    )


def plot_variance_by_channels(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Create boxplot of variance by channel count."""
    return _create_boxplot(
        data_dict=data["channels"],
        output_path=output_dir / "variance_by_channels_boxplot.png",
        label_fmt="{}ch",
        xlabel="Channel Configuration",
        title="GEMM Kernel Time Variance by Channel Count",
        colors=["#e6f2ff", "#99ccff", "#4da6ff", "#0073e6"],
        figsize=(12, 6),
        dpi=dpi,
    )


def plot_variance_by_ranks(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Create boxplot of variance by rank."""
    return _create_boxplot(
        data_dict=data["ranks"],
        output_path=output_dir / "variance_by_ranks_boxplot.png",
        label_fmt="Rank {}",
        xlabel="Rank",
        title="GEMM Kernel Time Variance by Rank",
        colors="viridis",
        figsize=(14, 6),
        dpi=dpi,
    )
```

---

### 5.9 `plot_helper/gemm_violin.py`

Combined violin plot.

```python
"""GEMM variance violin plot."""

from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt

from .common import DEFAULT_DPI, save_figure


def _prepare_violin_data(data_dict: Dict[int, List[float]], label_fmt: str) -> List[Dict]:
    """Prepare data for violin plot from a dictionary."""
    result = []
    for key, values in sorted(data_dict.items()):
        for val in values:
            result.append({"config": label_fmt.format(key), "time_diff": val})
    return result


def plot_variance_violin_combined(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Create combined violin plot (1x3 grid) for all dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    configs = [
        {
            "data": _prepare_violin_data(data["threads"], "{}t"),
            "sort_key": lambda x: int(x[:-1]),
            "color": "lightblue",
            "xlabel": "Threads",
            "title": "By Thread Count",
        },
        {
            "data": _prepare_violin_data(data["channels"], "{}ch"),
            "sort_key": lambda x: int(x[:-2]),
            "color": "lightcoral",
            "xlabel": "Channels",
            "title": "By Channel Count",
        },
        {
            "data": _prepare_violin_data(data["ranks"], "R{}"),
            "sort_key": lambda x: int(x[1:]),
            "color": "lightgreen",
            "xlabel": "Ranks",
            "title": "By Rank",
        },
    ]
    
    for ax, cfg in zip(axes, configs):
        violin_data = cfg["data"]
        configs_list = sorted(set(d["config"] for d in violin_data), key=cfg["sort_key"])
        values = [[d["time_diff"] for d in violin_data if d["config"] == c] for c in configs_list]
        
        parts = ax.violinplot(
            values,
            positions=range(len(configs_list)),
            showmeans=True,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(cfg["color"])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(configs_list)))
        ax.set_xticklabels(configs_list)
        ax.set_ylabel("Time Difference (us)", fontsize=12, fontweight="bold")
        ax.set_xlabel(cfg["xlabel"], fontsize=12, fontweight="bold")
        ax.set_title(cfg["title"], fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    
    fig.suptitle(
        "GEMM Kernel Time Variance Distribution",
        fontsize=18, fontweight="bold", y=1.02,
    )
    
    plt.tight_layout()
    return save_figure(fig, output_dir / "variance_violin_combined.png", dpi)
```

---

### 5.10 `plot_helper/gemm_interaction.py`

Thread-channel interaction plot.

```python
"""GEMM thread-channel interaction plot."""

from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import matplotlib.pyplot as plt

from .common import DEFAULT_DPI, save_figure


def plot_thread_channel_interaction(
    data: Dict[str, Any],
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Create thread-channel interaction line plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Organize data by threads and channels
    thread_channel_data = defaultdict(lambda: defaultdict(list))
    for row in data["all"]:
        thread_channel_data[row["threads"]][row["channel"]].append(row["time_diff"])
    
    threads = sorted(thread_channel_data.keys())
    channels = sorted(set(
        ch for t_data in thread_channel_data.values() for ch in t_data.keys()
    ))
    
    markers = ["o", "s", "^", "D"]
    
    for i, thread in enumerate(threads):
        means = []
        for channel in channels:
            if channel in thread_channel_data[thread]:
                values = thread_channel_data[thread][channel]
                means.append(sum(values) / len(values))
            else:
                means.append(0)
        
        ax.plot(
            channels, means,
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=10,
            label=f"{thread} threads",
        )
    
    ax.set_xlabel("Channel Count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Time Difference (us)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Thread-Channel Interaction: Mean Variance",
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.set_xticks(channels)
    ax.set_xticklabels([f"{c}ch" for c in channels])
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_figure(fig, output_dir / "variance_thread_channel_interaction.png", dpi)
```

---

### 5.11 `plot_helper/__init__.py`

Package exports.

```python
"""Plot helper functions for summary and GEMM visualizations."""

from .common import configure_style, COLORS, save_figure, get_improvement_colors

# Summary plots
from .summary_dashboard import (
    get_labels_from_excel,
    plot_improvement_chart,
    plot_abs_time_comparison,
)
from .gpu_by_rank import plot_gpu_metrics_by_rank
from .gpu_percent_change import plot_gpu_percent_change_grid
from .gpu_heatmap import plot_gpu_heatmap
from .nccl_charts import plot_nccl_comparison, plot_nccl_percent_change

# GEMM plots
from .gemm_data import read_gemm_csv_data, print_gemm_statistics
from .gemm_boxplots import (
    plot_variance_by_threads,
    plot_variance_by_channels,
    plot_variance_by_ranks,
)
from .gemm_violin import plot_variance_violin_combined
from .gemm_interaction import plot_thread_channel_interaction

__all__ = [
    # Common
    "configure_style",
    "COLORS",
    "save_figure",
    "get_improvement_colors",
    # Summary
    "get_labels_from_excel",
    "plot_improvement_chart",
    "plot_abs_time_comparison",
    "plot_gpu_metrics_by_rank",
    "plot_gpu_percent_change_grid",
    "plot_gpu_heatmap",
    "plot_nccl_comparison",
    "plot_nccl_percent_change",
    # GEMM
    "read_gemm_csv_data",
    "print_gemm_statistics",
    "plot_variance_by_threads",
    "plot_variance_by_channels",
    "plot_variance_by_ranks",
    "plot_variance_violin_combined",
    "plot_thread_channel_interaction",
]
```

---

### 5.12 `generators/plot_generator.py`

Main orchestrator (thin wrapper).

```python
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
    dpi: int = 150,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate all summary plots from Excel report.
    
    Returns list of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    
    if verbose:
        print(f"\nGenerating summary plots from: {excel_path}")
    
    labels = get_labels_from_excel(excel_path)
    if verbose:
        print(f"  Labels: {labels}")
    
    # Dashboard plots
    output_files.append(plot_improvement_chart(excel_path, output_dir, dpi))
    output_files.append(plot_abs_time_comparison(excel_path, output_dir, labels, dpi))
    
    # GPU plots
    output_files.extend(plot_gpu_metrics_by_rank(excel_path, output_dir, labels, dpi=dpi))
    output_files.append(plot_gpu_percent_change_grid(excel_path, output_dir, dpi))
    output_files.append(plot_gpu_heatmap(excel_path, output_dir, dpi))
    
    # NCCL plots
    output_files.extend(plot_nccl_comparison(excel_path, output_dir, labels, dpi))
    output_files.append(plot_nccl_percent_change(excel_path, output_dir, dpi))
    
    if verbose:
        print(f"  Generated {len(output_files)} summary plots")
    
    return output_files


def generate_gemm_plots(
    csv_path: Path,
    output_dir: Path,
    dpi: int = 150,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate all GEMM variance plots from CSV.
    
    Returns list of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    
    if verbose:
        print(f"\nGenerating GEMM plots from: {csv_path}")
    
    data = read_gemm_csv_data(csv_path)
    
    if verbose:
        print(f"  Total data points: {len(data['all'])}")
        print_gemm_statistics(data)
    
    # Boxplots
    output_files.append(plot_variance_by_threads(data, output_dir, dpi))
    output_files.append(plot_variance_by_channels(data, output_dir, dpi))
    output_files.append(plot_variance_by_ranks(data, output_dir, dpi))
    
    # Violin and interaction
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
    results = {}
    
    if plot_type in ("summary", "all"):
        if excel_input is None:
            raise ValueError("Excel input required for summary plots")
        if not excel_input.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_input}")
        results["summary"] = generate_summary_plots(excel_input, output_dir, dpi, verbose)
    
    if plot_type in ("gemm", "all"):
        if gemm_csv is None:
            raise ValueError("GEMM CSV required for gemm plots")
        if not gemm_csv.exists():
            raise FileNotFoundError(f"CSV file not found: {gemm_csv}")
        results["gemm"] = generate_gemm_plots(gemm_csv, output_dir, dpi, verbose)
    
    return results
```

---

## 6. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          generate plots                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  --type summary                                                             │
│  ────────────────                                                           │
│  INPUT: final_report.xlsx                                                   │
│    ├── Summary_Dashboard → summary_dashboard.py                             │
│    ├── GPU_ByRank_Cmp → gpu_by_rank.py, gpu_percent_change.py, gpu_heatmap.py│
│    └── NCCL_ImplicitSyncCmp → nccl_charts.py                                │
│                                                                             │
│  OUTPUT: ./plots/ (13 files)                                                │
│    ├── improvement_chart.png                                                │
│    ├── abs_time_comparison.png                                              │
│    ├── {metric}_by_rank.png (4 files)                                       │
│    ├── gpu_time_change_percentage_summary_by_rank.png                       │
│    ├── gpu_time_heatmap.png                                                 │
│    └── NCCL_*.png (5 files)                                                 │
│                                                                             │
│  --type gemm                                                                │
│  ──────────────                                                             │
│  INPUT: gemm_variance.csv                                                   │
│    └── gemm_data.py → gemm_boxplots.py, gemm_violin.py, gemm_interaction.py │
│                                                                             │
│  OUTPUT: ./plots/ (5 files)                                                 │
│    ├── variance_by_threads_boxplot.png                                      │
│    ├── variance_by_channels_boxplot.png                                     │
│    ├── variance_by_ranks_boxplot.png                                        │
│    ├── variance_violin_combined.png                                         │
│    └── variance_thread_channel_interaction.png                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Order

| Phase | Task | Est. Time |
|-------|------|-----------|
| **1** | Create `plot_helper/` package structure | 5 min |
| **2** | Implement `common.py` | 10 min |
| **3** | Implement `summary_dashboard.py` | 15 min |
| **4** | Implement `gpu_by_rank.py` | 10 min |
| **5** | Implement `gpu_percent_change.py` | 10 min |
| **6** | Implement `gpu_heatmap.py` | 10 min |
| **7** | Implement `nccl_charts.py` | 20 min |
| **8** | Implement `gemm_data.py` | 10 min |
| **9** | Implement `gemm_boxplots.py` | 15 min |
| **10** | Implement `gemm_violin.py` | 15 min |
| **11** | Implement `gemm_interaction.py` | 10 min |
| **12** | Implement `plot_helper/__init__.py` | 5 min |
| **13** | Implement `plot_generator.py` orchestrator | 15 min |
| **14** | Update `generators/__init__.py` | 5 min |
| **15** | Update CLI in `cli.py` | 15 min |
| **16** | Testing | 20 min |

**Total estimated time: ~3 hours**

---

## 8. Output Files

### Summary Plots (13 files)

| File | Source Module | Description |
|------|---------------|-------------|
| `improvement_chart.png` | `summary_dashboard.py` | Horizontal bar chart |
| `abs_time_comparison.png` | `summary_dashboard.py` | Grouped bar chart |
| `total_time_by_rank.png` | `gpu_by_rank.py` | Line plot |
| `computation_time_by_rank.png` | `gpu_by_rank.py` | Line plot |
| `total_comm_time_by_rank.png` | `gpu_by_rank.py` | Line plot |
| `idle_time_by_rank.png` | `gpu_by_rank.py` | Line plot |
| `gpu_time_change_percentage_summary_by_rank.png` | `gpu_percent_change.py` | 2×4 grid |
| `gpu_time_heatmap.png` | `gpu_heatmap.py` | Seaborn heatmap |
| `NCCL_Communication_Latency_comparison.png` | `nccl_charts.py` | Grouped bars |
| `NCCL_Algorithm_Bandwidth_comparison.png` | `nccl_charts.py` | Grouped bars |
| `NCCL_Bus_Bandwidth_comparison.png` | `nccl_charts.py` | Grouped bars |
| `NCCL_Total_Communication_Latency_comparison.png` | `nccl_charts.py` | Grouped bars |
| `NCCL_Performance_Percentage_Change_comparison.png` | `nccl_charts.py` | 1×3 grid |

### GEMM Plots (5 files)

| File | Source Module | Description |
|------|---------------|-------------|
| `variance_by_threads_boxplot.png` | `gemm_boxplots.py` | Boxplot |
| `variance_by_channels_boxplot.png` | `gemm_boxplots.py` | Boxplot |
| `variance_by_ranks_boxplot.png` | `gemm_boxplots.py` | Boxplot |
| `variance_violin_combined.png` | `gemm_violin.py` | 1×3 violin |
| `variance_thread_channel_interaction.png` | `gemm_interaction.py` | Line plot |

---

## Appendix A: Design Decisions

1. **Modular Structure:** One file per logical group of plots (~50-120 lines each)
2. **Plot Types:** `summary` and `gemm` as requested
3. **Internal Package:** `plot_helper/` keeps implementation details separate from public API
4. **Thin Orchestrator:** `plot_generator.py` imports from `plot_helper/` and provides CLI-facing API
5. **Consistent Style:** All plots use shared `common.py` utilities
6. **Easy Extension:** Adding new plot types = new file in `plot_helper/`
