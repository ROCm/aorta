"""Summary dashboard plots: improvement chart and absolute time comparison."""

from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .common import (
    COLORS,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    remove_spines,
    save_figure,
    get_improvement_colors,
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
        fontsize=14,
        fontweight="bold",
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
        if label in df.columns:
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
