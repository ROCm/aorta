"""GPU percent change grid plot."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .common import DEFAULT_DPI, save_figure, get_improvement_colors


METRIC_TYPES = [
    "busy_time",
    "computation_time",
    "exposed_comm_time",
    "exposed_memcpy_time",
    "idle_time",
    "total_comm_time",
    "total_memcpy_time",
    "total_time",
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
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return save_figure(
        fig, output_dir / "gpu_time_change_percentage_summary_by_rank.png", dpi
    )

