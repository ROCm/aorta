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
        'threads': defaultdict(list),
        'channels': defaultdict(list),
        'ranks': defaultdict(list),
        'all': []
    }

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                threads = int(row['threads'])
                channel = int(row['channel'])
                rank = int(row['rank'])
                time_diff = float(row['time_diff_us'])

                data['threads'][threads].append(time_diff)
                data['channels'][channel].append(time_diff)
                data['ranks'][rank].append(time_diff)
                data['all'].append({
                    'threads': threads,
                    'channel': channel,
                    'rank': rank,
                    'time_diff': time_diff,
                    'kernel_name': row['kernel_name']
                })
            except (ValueError, KeyError) as e:
                print(f"Skipping row due to error: {e}")
                continue

    return data

def create_boxplot_by_threads(data, output_dir):
    """Create box plot showing time_diff distribution by thread count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for boxplot
    threads_list = sorted(data['threads'].keys())
    plot_data = [data['threads'][t] for t in threads_list]
    labels = [f'{t} threads' for t in threads_list]

    bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Time Difference (µs)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Thread Configuration', fontsize=14, fontweight='bold')
    ax.set_title('GEMM Kernel Time Variance by Thread Count',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_by_threads_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'variance_by_threads_boxplot.png'}")
    plt.close()

def create_boxplot_by_channels(data, output_dir):
    """Create box plot showing time_diff distribution by channel count."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for boxplot
    channels_list = sorted(data['channels'].keys())
    plot_data = [data['channels'][c] for c in channels_list]
    labels = [f'{c}ch' for c in channels_list]

    bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color the boxes with gradient
    colors = ['#e6f2ff', '#99ccff', '#4da6ff', '#0073e6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Time Difference (µs)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Channel Configuration', fontsize=14, fontweight='bold')
    ax.set_title('GEMM Kernel Time Variance by Channel Count',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_by_channels_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'variance_by_channels_boxplot.png'}")
    plt.close()

def create_boxplot_by_ranks(data, output_dir):
    """Create box plot showing time_diff distribution by rank."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data for boxplot
    ranks_list = sorted(data['ranks'].keys())
    plot_data = [data['ranks'][r] for r in ranks_list]
    labels = [f'Rank {r}' for r in ranks_list]

    bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color the boxes with gradient
    colors = plt.cm.viridis([i/len(ranks_list) for i in range(len(ranks_list))])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Time Difference (µs)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rank', fontsize=14, fontweight='bold')
    ax.set_title('GEMM Kernel Time Variance by Rank',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_by_ranks_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'variance_by_ranks_boxplot.png'}")
    plt.close()

def create_violin_plot_combined(data, output_dir):
    """Create a combined violin plot showing all three dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Prepare data for violin plots
    threads_data = []
    for threads, values in sorted(data['threads'].items()):
        for val in values:
            threads_data.append({'config': f'{threads}t', 'time_diff': val})

    channels_data = []
    for channel, values in sorted(data['channels'].items()):
        for val in values:
            channels_data.append({'config': f'{channel}ch', 'time_diff': val})

    ranks_data = []
    for rank, values in sorted(data['ranks'].items()):
        for val in values:
            ranks_data.append({'config': f'R{rank}', 'time_diff': val})

    # Plot 1: Threads
    ax = axes[0]
    threads_configs = sorted(set(d['config'] for d in threads_data))
    threads_values = [[d['time_diff'] for d in threads_data if d['config'] == c]
                      for c in threads_configs]

    parts = ax.violinplot(threads_values, positions=range(len(threads_configs)),
                          showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(threads_configs)))
    ax.set_xticklabels(threads_configs)
    ax.set_ylabel('Time Difference (µs)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Threads', fontsize=12, fontweight='bold')
    ax.set_title('By Thread Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Channels
    ax = axes[1]
    channels_configs = sorted(set(d['config'] for d in channels_data),
                             key=lambda x: int(x[:-2]))
    channels_values = [[d['time_diff'] for d in channels_data if d['config'] == c]
                       for c in channels_configs]

    parts = ax.violinplot(channels_values, positions=range(len(channels_configs)),
                          showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(channels_configs)))
    ax.set_xticklabels(channels_configs)
    ax.set_ylabel('Time Difference (µs)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Channels', fontsize=12, fontweight='bold')
    ax.set_title('By Channel Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Ranks
    ax = axes[2]
    ranks_configs = sorted(set(d['config'] for d in ranks_data),
                          key=lambda x: int(x[1:]))
    ranks_values = [[d['time_diff'] for d in ranks_data if d['config'] == c]
                    for c in ranks_configs]

    parts = ax.violinplot(ranks_values, positions=range(len(ranks_configs)),
                          showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(ranks_configs)))
    ax.set_xticklabels(ranks_configs)
    ax.set_ylabel('Time Difference (µs)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Ranks', fontsize=12, fontweight='bold')
    ax.set_title('By Rank', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('GEMM Kernel Time Variance Distribution',
                 fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_violin_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'variance_violin_combined.png'}")
    plt.close()

def create_interaction_plot(data, output_dir):
    """Create a plot showing interaction between threads and channels."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Organize data by threads and channels
    thread_channel_data = defaultdict(lambda: defaultdict(list))
    for row in data['all']:
        thread_channel_data[row['threads']][row['channel']].append(row['time_diff'])

    # Calculate means
    threads = sorted(thread_channel_data.keys())
    channels = [28, 42, 56, 70]

    for thread in threads:
        means = []
        for channel in channels:
            if channel in thread_channel_data[thread]:
                mean_val = sum(thread_channel_data[thread][channel]) / len(thread_channel_data[thread][channel])
                means.append(mean_val)
            else:
                means.append(0)

        marker = 'o' if thread == 256 else 's'
        label = f'{thread} threads'
        ax.plot(channels, means, marker=marker, linewidth=2, markersize=10, label=label)

    ax.set_xlabel('Channel Count', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Time Difference (µs)', fontsize=14, fontweight='bold')
    ax.set_title('Thread-Channel Interaction: Mean Variance',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(channels)
    ax.set_xticklabels([f'{c}ch' for c in channels])
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_thread_channel_interaction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'variance_thread_channel_interaction.png'}")
    plt.close()

def print_statistics(data):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("VARIANCE DISTRIBUTION STATISTICS")
    print("="*70)

    print("\nBy Thread Count:")
    for threads in sorted(data['threads'].keys()):
        values = data['threads'][threads]
        sorted_vals = sorted(values)
        n = len(values)
        median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
        print(f"  {threads} threads: mean={sum(values)/len(values):.2f}µs, "
              f"median={median:.2f}µs, "
              f"max={max(values):.2f}µs, n={len(values)}")

    print("\nBy Channel Count:")
    for channel in sorted(data['channels'].keys()):
        values = data['channels'][channel]
        sorted_vals = sorted(values)
        n = len(values)
        median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
        print(f"  {channel}ch: mean={sum(values)/len(values):.2f}µs, "
              f"median={median:.2f}µs, "
              f"max={max(values):.2f}µs, n={len(values)}")

    print("\nBy Rank:")
    for rank in sorted(data['ranks'].keys()):
        values = data['ranks'][rank]
        sorted_vals = sorted(values)
        n = len(values)
        median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
        print(f"  Rank {rank}: mean={sum(values)/len(values):.2f}µs, "
              f"median={median:.2f}µs, "
              f"max={max(values):.2f}µs, n={len(values)}")
    print("="*70 + "\n")

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
        """
    )

    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path("/home/oyazdanb/aorta/experiments/sweep_20251121_155219/tracelens_analysis/top5_gemm_kernels_time_variance.csv"),
        help='Path to the GEMM variance CSV file (default: %(default)s)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots (default: same directory as CSV with /plots suffix)'
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
