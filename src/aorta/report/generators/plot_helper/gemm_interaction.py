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
    thread_channel_data: Dict[int, Dict[int, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in data["all"]:
        thread_channel_data[row["threads"]][row["channel"]].append(row["time_diff"])

    threads = sorted(thread_channel_data.keys())
    # Use ALL channels from data["channels"] to ensure all channels appear on X-axis
    # even if some thread/channel combinations don't have data
    channels = sorted(data["channels"].keys())

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
            channels,
            means,
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=10,
            label=f"{thread} threads",
        )

    ax.set_xlabel("Channel Count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Time Difference (us)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Thread-Channel Interaction: Mean Variance",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(channels)
    ax.set_xticklabels([f"{c}ch" for c in channels])
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_figure(
        fig, output_dir / "variance_thread_channel_interaction.png", dpi
    )
