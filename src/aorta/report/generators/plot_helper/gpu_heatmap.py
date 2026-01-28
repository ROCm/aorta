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
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Metric Type", fontsize=12)
    
    plt.tight_layout()
    return save_figure(fig, output_dir / "gpu_time_heatmap.png", dpi)

