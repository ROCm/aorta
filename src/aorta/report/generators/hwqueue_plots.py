"""
HW Queue Eval plot generator.

Generates plots for:
- Single run analysis (Mode A)
- Sweep analysis (Mode B)

Phase 6 will add comparison mode plots (Mode C).
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from .plot_helper.common import (
    COLORS,
    PALETTE_MULTI,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    configure_style,
    save_figure,
)

from ..processing.hwqueue_loader import SingleRunData, SweepData


# =============================================================================
# Single Run Plots (Mode A)
# =============================================================================


def plot_latency_histogram(
    data: SingleRunData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Plot latency distribution histogram from iteration times.

    Args:
        data: SingleRunData object
        output_dir: Output directory for PNG
        dpi: DPI for output image
        verbose: Print progress

    Returns:
        Path to generated file, or None if no data
    """
    if not data.iteration_times_ms:
        if verbose:
            print("    Skipping latency histogram (no iteration_times_ms)")
        return None

    configure_style()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    times = np.array(data.iteration_times_ms)

    # Create histogram
    ax.hist(times, bins=30, color=COLORS["baseline"], edgecolor="white", alpha=0.8)

    # Add vertical lines for percentiles
    ax.axvline(data.latency.p50, color=COLORS["positive"], linestyle="--",
               linewidth=2, label=f"P50: {data.latency.p50:.2f} ms")
    ax.axvline(data.latency.p95, color=COLORS["test"], linestyle="--",
               linewidth=2, label=f"P95: {data.latency.p95:.2f} ms")
    ax.axvline(data.latency.p99, color=COLORS["negative"], linestyle="--",
               linewidth=2, label=f"P99: {data.latency.p99:.2f} ms")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{data.workload_name} - Latency Distribution ({data.stream_count} streams)")
    ax.legend(loc="upper right")

    output_path = output_dir / f"latency_histogram_{data.workload_name}.png"
    save_figure(fig, output_path, dpi)

    if verbose:
        print(f"    Generated: {output_path.name}")

    return output_path


def plot_latency_percentiles(
    data: SingleRunData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> Path:
    """
    Plot latency percentiles as bar chart.

    Args:
        data: SingleRunData object
        output_dir: Output directory for PNG
        dpi: DPI for output image
        verbose: Print progress

    Returns:
        Path to generated file
    """
    configure_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ["Mean", "P50", "P95", "P99", "Min", "Max"]
    values = [
        data.latency.mean,
        data.latency.p50,
        data.latency.p95,
        data.latency.p99,
        data.latency.min,
        data.latency.max,
    ]

    colors = [COLORS["baseline"], COLORS["positive"], COLORS["test"],
              COLORS["negative"], COLORS["neutral"], COLORS["neutral"]]

    bars = ax.bar(metrics, values, color=colors, edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{data.workload_name} - Latency Percentiles ({data.stream_count} streams)")

    output_path = output_dir / f"latency_percentiles_{data.workload_name}.png"
    save_figure(fig, output_path, dpi)

    if verbose:
        print(f"    Generated: {output_path.name}")

    return output_path


def plot_per_stream_times(
    data: SingleRunData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Plot per-stream execution times as bar chart.

    Args:
        data: SingleRunData object
        output_dir: Output directory for PNG
        dpi: DPI for output image
        verbose: Print progress

    Returns:
        Path to generated file, or None if no data
    """
    if not data.per_stream_times_ms:
        if verbose:
            print("    Skipping per-stream times (no data)")
        return None

    configure_style()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    stream_indices = list(range(len(data.per_stream_times_ms)))
    times = data.per_stream_times_ms

    # Color bars by relative time
    mean_time = np.mean(times)
    colors = [COLORS["positive"] if t <= mean_time else COLORS["negative"] for t in times]

    bars = ax.bar(stream_indices, times, color=colors, edgecolor="white")

    # Add mean line
    ax.axhline(mean_time, color=COLORS["baseline"], linestyle="--",
               linewidth=2, label=f"Mean: {mean_time:.2f} ms")

    ax.set_xlabel("Stream Index")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{data.workload_name} - Per-Stream Execution Times")
    ax.legend(loc="upper right")

    output_path = output_dir / f"per_stream_times_{data.workload_name}.png"
    save_figure(fig, output_path, dpi)

    if verbose:
        print(f"    Generated: {output_path.name}")

    return output_path


def generate_single_run_plots(
    data: SingleRunData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate all plots for single run data (Mode A).

    Args:
        data: SingleRunData object
        output_dir: Output directory for PNG files
        dpi: DPI for output images
        verbose: Print progress

    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    if verbose:
        print(f"  Generating single run plots for: {data.workload_name}")

    # Latency histogram
    p = plot_latency_histogram(data, output_dir, dpi, verbose)
    if p:
        plots.append(p)

    # Latency percentiles
    plots.append(plot_latency_percentiles(data, output_dir, dpi, verbose))

    # Per-stream times
    p = plot_per_stream_times(data, output_dir, dpi, verbose)
    if p:
        plots.append(p)

    return plots


# =============================================================================
# Sweep Plots (Mode B)
# =============================================================================


def plot_throughput_scaling(
    data: SweepData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> Path:
    """
    Plot throughput scaling curve with ideal line.

    Args:
        data: SweepData object
        output_dir: Output directory for PNG
        dpi: DPI for output image
        verbose: Print progress

    Returns:
        Path to generated file
    """
    configure_style()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Sort by stream count
    sorted_results = sorted(data.results, key=lambda r: r.stream_count)
    stream_counts = [r.stream_count for r in sorted_results]
    throughputs = [r.throughput for r in sorted_results]

    # Actual throughput
    ax.plot(stream_counts, throughputs, "o-", color=COLORS["baseline"],
            linewidth=2, markersize=8, label="Actual Throughput")

    # Ideal linear scaling (based on first point)
    if throughputs[0] > 0:
        ideal = [throughputs[0] * (sc / stream_counts[0]) for sc in stream_counts]
        ax.plot(stream_counts, ideal, "--", color=COLORS["neutral"],
                linewidth=1.5, alpha=0.7, label="Ideal Linear Scaling")

    # Mark best throughput
    best_streams, best_throughput = data.get_best_throughput()
    ax.scatter([best_streams], [best_throughput], color=COLORS["positive"],
               s=150, zorder=5, marker="*", label=f"Best: {best_throughput:.0f} @ {best_streams} streams")

    # Mark inflection point if available
    if data.analysis.inflection_point:
        inflection = data.analysis.inflection_point
        # Find throughput at inflection
        for r in sorted_results:
            if r.stream_count == inflection:
                ax.axvline(inflection, color=COLORS["test"], linestyle=":",
                           linewidth=2, alpha=0.7, label=f"Inflection: {inflection} streams")
                break

    ax.set_xlabel("Stream Count")
    ax.set_ylabel(f"Throughput ({data.results[0].throughput_unit})")
    ax.set_title(f"{data.workload_name} - Throughput Scaling")
    ax.legend(loc="best")
    ax.set_xticks(stream_counts)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / f"throughput_scaling_{data.workload_name}.png"
    save_figure(fig, output_path, dpi)

    if verbose:
        print(f"    Generated: {output_path.name}")

    return output_path


def plot_scaling_efficiency(
    data: SweepData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> Path:
    """
    Plot scaling efficiency curve (actual/ideal).

    Args:
        data: SweepData object
        output_dir: Output directory for PNG
        dpi: DPI for output image
        verbose: Print progress

    Returns:
        Path to generated file
    """
    configure_style()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Sort by stream count
    sorted_results = sorted(data.results, key=lambda r: r.stream_count)
    stream_counts = [r.stream_count for r in sorted_results]
    throughputs = [r.throughput for r in sorted_results]

    # Calculate efficiency (actual / ideal)
    base_throughput = throughputs[0]
    base_streams = stream_counts[0]

    efficiencies = []
    for sc, tp in zip(stream_counts, throughputs):
        if base_throughput > 0:
            ideal = base_throughput * (sc / base_streams)
            efficiency = (tp / ideal) * 100 if ideal > 0 else 0
        else:
            efficiency = 0
        efficiencies.append(efficiency)

    # Create color gradient based on efficiency
    colors = [COLORS["positive"] if e >= 80 else
              (COLORS["test"] if e >= 60 else COLORS["negative"]) for e in efficiencies]

    ax.bar(range(len(stream_counts)), efficiencies, color=colors, edgecolor="white")
    ax.set_xticks(range(len(stream_counts)))
    ax.set_xticklabels(stream_counts)

    # Add 100% reference line
    ax.axhline(100, color=COLORS["neutral"], linestyle="--", linewidth=1.5, alpha=0.7)

    # Add threshold lines
    ax.axhline(80, color=COLORS["positive"], linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(60, color=COLORS["test"], linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Stream Count")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title(f"{data.workload_name} - Scaling Efficiency")
    ax.set_ylim(0, max(110, max(efficiencies) + 10))

    output_path = output_dir / f"scaling_efficiency_{data.workload_name}.png"
    save_figure(fig, output_path, dpi)

    if verbose:
        print(f"    Generated: {output_path.name}")

    return output_path


def plot_latency_vs_streams(
    data: SweepData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> Path:
    """
    Plot latency percentiles vs stream count (multi-line).

    Args:
        data: SweepData object
        output_dir: Output directory for PNG
        dpi: DPI for output image
        verbose: Print progress

    Returns:
        Path to generated file
    """
    configure_style()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Sort by stream count
    sorted_results = sorted(data.results, key=lambda r: r.stream_count)
    stream_counts = [r.stream_count for r in sorted_results]

    # Extract latency percentiles
    p50_values = [r.latency.p50 for r in sorted_results]
    p95_values = [r.latency.p95 for r in sorted_results]
    p99_values = [r.latency.p99 for r in sorted_results]

    ax.plot(stream_counts, p50_values, "o-", color=COLORS["positive"],
            linewidth=2, markersize=6, label="P50")
    ax.plot(stream_counts, p95_values, "s-", color=COLORS["test"],
            linewidth=2, markersize=6, label="P95")
    ax.plot(stream_counts, p99_values, "^-", color=COLORS["negative"],
            linewidth=2, markersize=6, label="P99")

    ax.set_xlabel("Stream Count")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{data.workload_name} - Latency vs Stream Count")
    ax.legend(loc="best")
    ax.set_xticks(stream_counts)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / f"latency_vs_streams_{data.workload_name}.png"
    save_figure(fig, output_path, dpi)

    if verbose:
        print(f"    Generated: {output_path.name}")

    return output_path


def plot_latency_heatmap(
    data: SweepData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> Path:
    """
    Plot latency variance heatmap (P99/P50 ratio across streams).

    Args:
        data: SweepData object
        output_dir: Output directory for PNG
        dpi: DPI for output image
        verbose: Print progress

    Returns:
        Path to generated file
    """
    configure_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    # Sort by stream count
    sorted_results = sorted(data.results, key=lambda r: r.stream_count)
    stream_counts = [str(r.stream_count) for r in sorted_results]

    # Build heatmap data: rows = metrics, cols = stream counts
    metrics = ["Mean", "P50", "P95", "P99"]
    heatmap_data = []
    for metric in metrics:
        row = []
        for r in sorted_results:
            if metric == "Mean":
                row.append(r.latency.mean)
            elif metric == "P50":
                row.append(r.latency.p50)
            elif metric == "P95":
                row.append(r.latency.p95)
            elif metric == "P99":
                row.append(r.latency.p99)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)

    # Create heatmap
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Latency (ms)")

    # Set ticks
    ax.set_xticks(range(len(stream_counts)))
    ax.set_xticklabels(stream_counts)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)

    ax.set_xlabel("Stream Count")
    ax.set_ylabel("Metric")
    ax.set_title(f"{data.workload_name} - Latency Heatmap")

    # Add value annotations
    for i in range(len(metrics)):
        for j in range(len(stream_counts)):
            val = heatmap_data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > heatmap_data.mean() else "black", fontsize=9)

    output_path = output_dir / f"latency_heatmap_{data.workload_name}.png"
    save_figure(fig, output_path, dpi)

    if verbose:
        print(f"    Generated: {output_path.name}")

    return output_path


def generate_sweep_plots(
    data: SweepData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate all plots for sweep data (Mode B).

    Args:
        data: SweepData object
        output_dir: Output directory for PNG files
        dpi: DPI for output images
        verbose: Print progress

    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    if verbose:
        print(f"  Generating sweep plots for: {data.workload_name}")

    # Throughput scaling
    plots.append(plot_throughput_scaling(data, output_dir, dpi, verbose))

    # Scaling efficiency
    plots.append(plot_scaling_efficiency(data, output_dir, dpi, verbose))

    # Latency vs streams
    plots.append(plot_latency_vs_streams(data, output_dir, dpi, verbose))

    # Latency heatmap
    plots.append(plot_latency_heatmap(data, output_dir, dpi, verbose))

    return plots


# =============================================================================
# Auto-dispatch Function
# =============================================================================


def generate_hwqueue_plots(
    data: SingleRunData | SweepData,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
) -> List[Path]:
    """
    Generate plots for single run or sweep data.

    Automatically detects data type and calls the appropriate generator.

    Args:
        data: SingleRunData or SweepData object
        output_dir: Output directory for PNG files
        dpi: DPI for output images
        verbose: Print progress

    Returns:
        List of generated file paths
    """
    if isinstance(data, SweepData):
        return generate_sweep_plots(data, output_dir, dpi, verbose)
    elif isinstance(data, SingleRunData):
        return generate_single_run_plots(data, output_dir, dpi, verbose)
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

