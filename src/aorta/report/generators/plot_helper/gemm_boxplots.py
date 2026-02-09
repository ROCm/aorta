"""GEMM variance boxplot generators."""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Union

import matplotlib.pyplot as plt

from .common import DEFAULT_DPI, save_figure


def _create_boxplot(
    data_dict: Dict[int, List[float]],
    output_path: Path,
    label_fmt: str,
    xlabel: str,
    title: str,
    colors: Union[List[str], str],
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
        color_list = plt.cm.viridis(
            [i / len(keys_list) for i in range(len(keys_list))]
        )
    else:
        color_list = colors

    for patch, color in zip(bp["boxes"], color_list):
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
