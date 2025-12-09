#!/usr/bin/env python3
"""
GPU Timeline Aggregation Script

Combines per-rank individual reports and aggregates gpu_timeline data
across all ranks using mean or geometric mean.

Usage:
    python process_gpu_timeline.py --sweep-dir /path/to/sweep_directory [--geo-mean]

Example:
    python process_gpu_timeline.py --sweep-dir experiments/sweep_20251124_222204
"""

import pandas as pd
import os
import glob
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gpu_timeline_utils import (
    read_gpu_timeline_from_excel,
    aggregate_gpu_timeline,
    get_method_suffix,
    get_aggregation_description,
    print_section,
)


# =============================================================================
# Utility Functions
# =============================================================================


def parse_perf_filename(filename):
    """
    Parse performance filename to extract channel config and rank.

    Args:
        filename: e.g., 'perf_28ch_rank0.xlsx'

    Returns:
        tuple: (channel_config, rank) e.g., ('28ch', 0)
    """
    parts = filename.replace("perf_", "").replace(".xlsx", "").split("_")
    channel_config = parts[0]  # e.g., "28ch"
    rank = int(parts[1].replace("rank", ""))
    return channel_config, rank


def group_files_by_channel(perf_files):
    """
    Group performance files by channel configuration.

    Args:
        perf_files: List of file paths

    Returns:
        dict: {channel_config: [(rank, file_path), ...]}
    """
    channel_groups = {}
    for file_path in perf_files:
        filename = os.path.basename(file_path)
        channel_config, rank = parse_perf_filename(filename)

        if channel_config not in channel_groups:
            channel_groups[channel_config] = []
        channel_groups[channel_config].append((rank, file_path))

    return channel_groups


# =============================================================================
# Data Processing Functions
# =============================================================================


def read_rank_data(rank_files):
    """
    Read gpu_timeline data from all rank files.

    Args:
        rank_files: List of (rank, file_path) tuples

    Returns:
        list: List of DataFrames with rank column added
    """
    rank_data = []
    for rank, file_path in rank_files:
        df, success = read_gpu_timeline_from_excel(Path(file_path), rank=rank)
        if success:
            rank_data.append(df)
    return rank_data


def add_config_metadata(df, thread_config, channel_config, num_ranks):
    """
    Add configuration metadata columns to the dataframe.

    Args:
        df: Source DataFrame
        thread_config: Thread configuration string (e.g., '256thread')
        channel_config: Channel configuration string (e.g., '28ch')
        num_ranks: Number of ranks

    Returns:
        DataFrame: DataFrame with added metadata columns
    """
    df["thread_config"] = thread_config
    df["threads_num"] = int(thread_config.replace("thread", ""))
    df["channel_config"] = channel_config
    df["channels_num"] = int(channel_config.replace("ch", ""))
    df["full_config"] = f"{thread_config}_{channel_config}"
    df["num_ranks"] = num_ranks
    return df


def process_channel_config(channel_config, channel_groups, use_geo_mean, thread_config):
    """
    Process a single channel configuration.

    Args:
        channel_config: Channel configuration string
        channel_groups: Dict of channel groups
        use_geo_mean: Whether to use geometric mean
        thread_config: Thread configuration string

    Returns:
        DataFrame or None: Aggregated data, or None if no valid data
    """
    rank_files = sorted(channel_groups[channel_config], key=lambda x: x[0])
    num_ranks = len(rank_files)

    print(f"  {channel_config}: Processing {num_ranks} ranks...")

    rank_data = read_rank_data(rank_files)

    if not rank_data:
        print(f"    No valid data for {channel_config}")
        return None

    aggregated = aggregate_gpu_timeline(rank_data, use_geo_mean)
    aggregated = add_config_metadata(aggregated, thread_config, channel_config, num_ranks)
    print(f"    [OK] Aggregated across {num_ranks} ranks")

    return aggregated


def process_thread_config(thread_config, tracelens_dir, use_geo_mean):
    """
    Process a single thread configuration.

    Args:
        thread_config: Thread configuration string
        tracelens_dir: Path to tracelens_analysis directory
        use_geo_mean: Whether to use geometric mean

    Returns:
        list: List of aggregated DataFrames
    """
    individual_reports_dir = tracelens_dir / thread_config / "individual_reports"

    if not individual_reports_dir.exists():
        print(f"  Warning: {individual_reports_dir} not found, skipping...")
        return []

    print(f"\nProcessing: {thread_config}")
    print("-" * 60)

    perf_files = sorted(glob.glob(str(individual_reports_dir / "perf_*ch_rank*.xlsx")))

    if not perf_files:
        print(f"  Warning: No performance files found in {individual_reports_dir}")
        return []

    channel_groups = group_files_by_channel(perf_files)
    results = []

    # Process each channel configuration (sorted by channel number)
    sorted_channels = sorted(channel_groups.keys(), key=lambda x: int(x.replace("ch", "")))
    for channel_config in sorted_channels:
        aggregated = process_channel_config(
            channel_config, channel_groups, use_geo_mean, thread_config
        )
        if aggregated is not None:
            results.append(aggregated)

    return results


# =============================================================================
# Excel Output Functions
# =============================================================================


def create_pivot_sheet(df, value_col):
    """
    Create a pivot table from the dataframe.

    Args:
        df: Source DataFrame
        value_col: Column to use for values

    Returns:
        DataFrame: Pivot table
    """
    return df.pivot_table(values=value_col, index="type", columns="full_config", aggfunc="first")


def create_summary_sheet(df):
    """
    Create a summary sheet with key metrics per configuration.

    Args:
        df: Source DataFrame

    Returns:
        DataFrame: Summary table
    """
    summary = (
        df.groupby("full_config")
        .agg({"threads_num": "first", "channels_num": "first", "num_ranks": "first"})
        .reset_index()
    )

    # Add key metrics for each config
    key_metrics = [
        "computation_time",
        "exposed_comm_time",
        "busy_time",
        "idle_time",
        "total_time",
    ]
    for metric_type in key_metrics:
        metric_data = df[df["type"] == metric_type].set_index("full_config")["time ms"]
        summary[f"{metric_type}_ms"] = summary["full_config"].map(metric_data)

    return summary


def save_excel_output(final_df, output_path):
    """
    Save results to Excel with multiple sheets.

    Args:
        final_df: Final DataFrame to save
        output_path: Path to output file
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="All_Data", index=False)
        create_pivot_sheet(final_df, "time ms").to_excel(writer, sheet_name="Pivot_Time_ms")
        create_pivot_sheet(final_df, "percent").to_excel(writer, sheet_name="Pivot_Percent")
        create_summary_sheet(final_df).to_excel(writer, sheet_name="Summary_By_Config", index=False)

    print(f"[SAVED] {output_path}")
    print("  Sheets created:")
    print("    1. All_Data - Complete dataset")
    print("    2. Pivot_Time_ms - Matrix view of time (ms)")
    print("    3. Pivot_Percent - Matrix view of percentages")
    print("    4. Summary_By_Config - Key metrics per configuration")


# =============================================================================
# Reporting Functions
# =============================================================================


def print_metric_comparison(df, metric_type, description):
    """
    Print a metric comparison table.

    Args:
        df: Source DataFrame
        metric_type: Type of metric to filter
        description: Description to print
    """
    metric_data = df[df["type"] == metric_type][["full_config", "time ms", "percent"]].sort_values(
        "time ms"
    )
    print(f"\n{description}:")
    print(metric_data.to_string(index=False))


def print_summary_report(final_df):
    """Print summary statistics and comparisons."""
    print_section("SUMMARY")

    print("\nMetric Types Found:")
    for metric_type in sorted(final_df["type"].unique()):
        count = len(final_df[final_df["type"] == metric_type])
        print(f"  {metric_type:<25} ({count} configurations)")

    print("\nConfigurations Processed:")
    configs = final_df.groupby("full_config")["num_ranks"].first().sort_index()
    for config, num_ranks in configs.items():
        print(f"  {config:<25} ({num_ranks} ranks)")

    print_section("KEY METRICS COMPARISON (Sorted by Busy Time)")
    print_metric_comparison(final_df, "busy_time", "Busy Time (lower is better)")
    print_metric_comparison(final_df, "idle_time", "Idle Time (lower is better)")


# =============================================================================
# Main Processing Function
# =============================================================================


def process_gpu_timeline_data(sweep_dir, use_geo_mean=False):
    """
    Process GPU timeline data from all individual reports.

    Args:
        sweep_dir: Path to sweep directory
        use_geo_mean: If True, use geometric mean; otherwise use arithmetic mean
    """
    sweep_path = Path(sweep_dir)
    tracelens_dir = sweep_path / "tracelens_analysis"

    if not tracelens_dir.exists():
        print(f"Error: tracelens_analysis directory not found in {sweep_dir}")
        return

    agg_method = get_aggregation_description(use_geo_mean)
    print("=" * 80)
    print(f"Processing GPU Timeline data from: {sweep_dir}")
    print(f"Aggregation method: {agg_method}")
    print("=" * 80)

    # Find all thread configurations
    thread_configs = [d.name for d in tracelens_dir.iterdir() if d.is_dir() and "thread" in d.name]

    if not thread_configs:
        print("Error: No thread configuration directories found")
        return

    print(f"\nFound thread configurations: {sorted(thread_configs)}")

    # Process all thread configurations
    all_results = []
    for thread_config in sorted(thread_configs):
        results = process_thread_config(thread_config, tracelens_dir, use_geo_mean)
        all_results.extend(results)

    if not all_results:
        print("\nError: No data was processed")
        return

    # Combine and format results
    print_section("CREATING OUTPUT FILE")

    final_df = pd.concat(all_results, ignore_index=True)

    # Reorder and sort
    column_order = [
        "full_config",
        "threads_num",
        "thread_config",
        "channels_num",
        "channel_config",
        "num_ranks",
        "type",
        "time ms",
        "percent",
    ]
    final_df = final_df[column_order]
    final_df = final_df.sort_values(["threads_num", "channels_num", "type"])

    # Save to Excel
    method_suffix = get_method_suffix(use_geo_mean)
    output_path = tracelens_dir / f"gpu_timeline_all_configs_{method_suffix}.xlsx"
    save_excel_output(final_df, output_path)

    # Print summary
    print_summary_report(final_df)

    print_section("COMPLETE!")
    print(f"\nOutput file: {output_path}")
    print("Open in Excel to create custom pivots and charts!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Process GPU timeline data from individual reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with arithmetic mean (default)
  python process_gpu_timeline.py --sweep-dir /path/to/sweep_20251124_222204

  # Process with geometric mean
  python process_gpu_timeline.py --sweep-dir /path/to/sweep_20251124_222204 --geo-mean
        """,
    )

    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Path to sweep directory (e.g., sweep_20251124_222204)",
    )

    parser.add_argument(
        "--geo-mean",
        action="store_true",
        help="Use geometric mean instead of arithmetic mean for aggregation",
    )

    args = parser.parse_args()

    # Validate sweep directory
    sweep_path = Path(args.sweep_dir)
    if not sweep_path.exists():
        print(f"Error: Sweep directory does not exist: {args.sweep_dir}")
        return 1

    # Process the sweep
    try:
        process_gpu_timeline_data(args.sweep_dir, args.geo_mean)
        return 0
    except Exception as e:
        print(f"\nError processing sweep: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
