#!/usr/bin/env python3
"""
Create variance distribution plots from GEMM analysis results.
Shows time_diff distribution across different configurations.
"""

import csv
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_context("talk")  # Larger fonts for presentations


def read_csv_data(csv_path):
    """Read the CSV file and return data organized by different dimensions."""
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
                data["all"].append(
                    {
                        "threads": threads,
                        "channel": channel,
                        "rank": rank,
                        "time_diff": time_diff,
                        "kernel_name": row["kernel_name"],
                    }
                )
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}")
                continue

    return data


def create_boxplot(
    data_dict, output_dir, dimension, figsize, label_fmt, xlabel, title, colors
):
    """
    Generic box plot function for time_diff distribution.

    Args:
        data_dict: Dictionary mapping keys to lists of time_diff values
        output_dir: Path to save the plot
        dimension: Name of dimension for filename (e.g., 'threads', 'channels', 'ranks')
        figsize: Tuple of (width, height) for the figure
        label_fmt: Format string for labels (e.g., '{} threads', '{}ch', 'Rank {}')
        xlabel: Label for x-axis
        title: Plot title
        colors: List of colors or 'viridis' for colormap
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for boxplot
    keys_list = sorted(data_dict.keys())
    plot_data = [data_dict[k] for k in keys_list]
    labels = [label_fmt.format(k) for k in keys_list]

    bp = ax.boxplot(
        plot_data, tick_labels=labels, patch_artist=True, showmeans=True, meanline=True
    )

    # Color the boxes
    if colors == "viridis":
        colors = plt.cm.viridis([i / len(keys_list) for i in range(len(keys_list))])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Time Difference (µs)", fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"variance_by_{dimension}_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def create_boxplot_by_threads(data, output_dir):
    """Create box plot showing time_diff distribution by thread count."""
    create_boxplot(
        data_dict=data["threads"],
        output_dir=output_dir,
        dimension="threads",
        figsize=(10, 6),
        label_fmt="{} threads",
        xlabel="Thread Configuration",
        title="GEMM Kernel Time Variance by Thread Count",
        colors=["lightblue", "lightcoral"],
    )


def create_boxplot_by_channels(data, output_dir):
    """Create box plot showing time_diff distribution by channel count."""
    create_boxplot(
        data_dict=data["channels"],
        output_dir=output_dir,
        dimension="channels",
        figsize=(12, 6),
        label_fmt="{}ch",
        xlabel="Channel Configuration",
        title="GEMM Kernel Time Variance by Channel Count",
        colors=["#e6f2ff", "#99ccff", "#4da6ff", "#0073e6"],
    )


def create_boxplot_by_ranks(data, output_dir):
    """Create box plot showing time_diff distribution by rank."""
    create_boxplot(
        data_dict=data["ranks"],
        output_dir=output_dir,
        dimension="ranks",
        figsize=(14, 6),
        label_fmt="Rank {}",
        xlabel="Rank",
        title="GEMM Kernel Time Variance by Rank",
        colors="viridis",
    )


def _prepare_violin_data(data_dict, label_fmt):
    """
    Prepare data for violin plot from a dictionary.

    Args:
        data_dict: Dictionary mapping keys to lists of values
        label_fmt: Format string for config labels (e.g., '{}t', '{}ch', 'R{}')

    Returns:
        List of dicts with 'config' and 'time_diff' keys
    """
    result = []
    for key, values in sorted(data_dict.items()):
        for val in values:
            result.append({"config": label_fmt.format(key), "time_diff": val})
    return result


def _create_violin_subplot(ax, data, sort_key_fn, color, xlabel, title):
    """
    Create a single violin subplot.

    Args:
        ax: Matplotlib axis
        data: List of dicts with 'config' and 'time_diff' keys
        sort_key_fn: Function to extract sort key from config string
        color: Face color for violin bodies
        xlabel: Label for x-axis
        title: Subplot title
    """
    configs = sorted(set(d["config"] for d in data), key=sort_key_fn)
    values = [[d["time_diff"] for d in data if d["config"] == c] for c in configs]

    parts = ax.violinplot(
        values, positions=range(len(configs)), showmeans=True, showmedians=True
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_ylabel("Time Difference (µs)", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")


def create_violin_plot_combined(data, output_dir):
    """Create a combined violin plot showing all three dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Define subplot configurations
    subplot_configs = [
        {
            "data": _prepare_violin_data(data["threads"], "{}t"),
            "sort_key_fn": lambda x: int(x[:-1]),
            "color": "lightblue",
            "xlabel": "Threads",
            "title": "By Thread Count",
        },
        {
            "data": _prepare_violin_data(data["channels"], "{}ch"),
            "sort_key_fn": lambda x: int(x[:-2]),
            "color": "lightcoral",
            "xlabel": "Channels",
            "title": "By Channel Count",
        },
        {
            "data": _prepare_violin_data(data["ranks"], "R{}"),
            "sort_key_fn": lambda x: int(x[1:]),
            "color": "lightgreen",
            "xlabel": "Ranks",
            "title": "By Rank",
        },
    ]

    # Create each subplot
    for ax, config in zip(axes, subplot_configs):
        _create_violin_subplot(ax, **config)

    fig.suptitle(
        "GEMM Kernel Time Variance Distribution", fontsize=18, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "variance_violin_combined.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'variance_violin_combined.png'}")
    plt.close()


def create_interaction_plot(data, output_dir):
    """Create a plot showing interaction between threads and channels."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Organize data by threads and channels
    thread_channel_data = defaultdict(lambda: defaultdict(list))
    for row in data["all"]:
        thread_channel_data[row["threads"]][row["channel"]].append(row["time_diff"])

    # Calculate means
    threads = sorted(thread_channel_data.keys())
    channels = [28, 42, 56, 70]

    for thread in threads:
        means = []
        for channel in channels:
            if channel in thread_channel_data[thread]:
                mean_val = sum(thread_channel_data[thread][channel]) / len(
                    thread_channel_data[thread][channel]
                )
                means.append(mean_val)
            else:
                means.append(0)

        marker = "o" if thread == 256 else "s"
        label = f"{thread} threads"
        ax.plot(channels, means, marker=marker, linewidth=2, markersize=10, label=label)

    ax.set_xlabel("Channel Count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Time Difference (µs)", fontsize=14, fontweight="bold")
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
    plt.savefig(
        output_dir / "variance_thread_channel_interaction.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {output_dir / 'variance_thread_channel_interaction.png'}")
    plt.close()


def _calculate_median(values):
    """Calculate median of a list of values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def _print_dimension_stats(data_dict, section_title, label_fmt):
    """
    Print statistics for a single dimension.

    Args:
        data_dict: Dictionary mapping keys to lists of values
        section_title: Title for this section (e.g., "By Thread Count:")
        label_fmt: Format string for labels (e.g., '{} threads', '{}ch', 'Rank {}')
    """
    print(f"\n{section_title}")
    for key in sorted(data_dict.keys()):
        values = data_dict[key]
        mean_val = sum(values) / len(values)
        median_val = _calculate_median(values)
        label = label_fmt.format(key)
        print(
            f"  {label}: mean={mean_val:.2f}µs, "
            f"median={median_val:.2f}µs, "
            f"max={max(values):.2f}µs, n={len(values)}"
        )


def print_statistics(data):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("VARIANCE DISTRIBUTION STATISTICS")
    print("=" * 70)

    _print_dimension_stats(data["threads"], "By Thread Count:", "{} threads")
    _print_dimension_stats(data["channels"], "By Channel Count:", "{}ch")
    _print_dimension_stats(data["ranks"], "By Rank:", "Rank {}")

    print("=" * 70 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create variance distribution plots from GEMM analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python plot_gemm_variance.py

  # Specify custom CSV file and output directory
  python plot_gemm_variance.py \\
    --csv-path experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv \\
    --output-dir experiments/sweep_20251124_222204/tracelens_analysis/plots

  # Using absolute paths
  python plot_gemm_variance.py \\
    --csv-path /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis/top5_gemm_kernels_time_variance.csv \\
    --output-dir /home/oyazdanb/aorta/experiments/sweep_20251124_222204/tracelens_analysis/plots
        """,
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(
            "~/aorta/experiments/sweep_20251121_155219/tracelens_analysis/top5_gemm_kernels_time_variance.csv"
        ),
        help="Path to the GEMM variance CSV file (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same directory as CSV with /plots suffix)",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    csv_path = args.csv_path

    # Set default output directory if not specified
    if args.output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = args.output_dir

    # Validate CSV file exists
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    print("GEMM Variance Plotting")
    print("=" * 70)
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Reading data from CSV...")
    data = read_csv_data(csv_path)

    print(f"Total data points: {len(data['all'])}")

    # Print statistics
    print_statistics(data)

    # Create plots
    print("\nGenerating plots...")
    create_boxplot_by_threads(data, output_dir)
    create_boxplot_by_channels(data, output_dir)
    create_boxplot_by_ranks(data, output_dir)
    create_violin_plot_combined(data, output_dir)
    create_interaction_plot(data, output_dir)

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. variance_by_threads_boxplot.png - Box plot by thread count")
    print("  2. variance_by_channels_boxplot.png - Box plot by channel count")
    print("  3. variance_by_ranks_boxplot.png - Box plot by rank")
    print("  4. variance_violin_combined.png - Combined violin plots")
    print("  5. variance_thread_channel_interaction.png - Interaction plot")


if __name__ == "__main__":
    main()
