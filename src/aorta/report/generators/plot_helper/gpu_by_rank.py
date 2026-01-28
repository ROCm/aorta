"""GPU metrics by rank line plots."""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

from .common import COLORS, DEFAULT_DPI, save_figure


METRICS_TO_PLOT = ["total_time", "computation_time", "total_comm_time", "idle_time"]


def plot_gpu_metrics_by_rank(
    excel_path: Path,
    output_dir: Path,
    labels: List[str],
    metrics: Optional[List[str]] = None,
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
        ax.set_title(
            f"{metric} Comparison across all ranks",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        
        plt.tight_layout()
        output_path = save_figure(fig, output_dir / f"{metric}_by_rank.png", dpi)
        output_files.append(output_path)
    
    return output_files

