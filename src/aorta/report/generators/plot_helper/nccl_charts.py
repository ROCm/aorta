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
    try:
        df = pd.read_excel(excel_path, sheet_name="NCCL_ImplicitSyncCmp")
    except ValueError:
        # Sheet might not exist
        return []
    
    df["label"] = df["Collective name"] + "\n" + df["In msg nelems"].astype(str)
    
    x = np.arange(len(df))
    width = 0.35
    colors = [COLORS["baseline"], COLORS["test"]]
    output_files = []
    
    for title, config in NCCL_METRICS.items():
        fig, ax = plt.subplots(figsize=(14, 6))
        
        has_data = False
        for i, label in enumerate(labels):
            col_name = f"{label}_{config['y_col']}"
            if col_name in df.columns:
                offset = (i - len(labels) / 2 + 0.5) * width
                ax.bar(x + offset, df[col_name], width, label=label, color=colors[i])
                has_data = True
        
        if not has_data:
            plt.close(fig)
            continue
        
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
    try:
        df = pd.read_excel(excel_path, sheet_name="NCCL_ImplicitSyncCmp")
    except ValueError:
        # Sheet might not exist
        return None
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
    
    has_any_data = False
    for i, (title, col_name) in enumerate(NCCL_PERCENT_METRICS.items()):
        ax = axes[i]
        if col_name not in df.columns:
            ax.set_visible(False)
            continue
        
        has_any_data = True
        colors = get_improvement_colors(df[col_name])
        ax.barh(df["In msg nelems"].astype(str), df[col_name], color=colors)
        
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
        ax.set_axisbelow(True)
        ax.set_xlabel("Percent Change (%)")
        ax.set_title(f"{title}\nPercent Change (Positive = better)")
    
    if not has_any_data:
        plt.close(fig)
        return None
    
    fig.suptitle(
        "NCCL Performance Percentage Change By Message Size",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    return save_figure(
        fig, output_dir / "NCCL_Performance_Percentage_Change_comparison.png", dpi
    )

