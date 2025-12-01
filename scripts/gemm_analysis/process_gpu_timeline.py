#!/usr/bin/env python3
"""
GPU Timeline Aggregation Script

Combines per-rank individual reports and aggregates gpu_timeline data
across all ranks using mean or geometric mean.

Usage:
    python process_gpu_timeline.py --sweep-dir /path/to/sweep_directory [--geo-mean]

Example:
    python process_gpu_timeline.py --sweep-dir /home/oyazdanb/aorta/experiments/sweep_20251124_222204
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path


def geometric_mean(values):
    """Calculate geometric mean, handling zeros."""
    values = np.array(values)
    # Replace zeros with small value to avoid log(0)
    values = np.where(values == 0, 1e-10, values)
    return np.exp(np.mean(np.log(values)))


def process_gpu_timeline_data(sweep_dir, use_geo_mean=False):
    """
    Process GPU timeline data from all individual reports.

    Args:
        sweep_dir: Path to sweep directory
        use_geo_mean: If True, use geometric mean; otherwise use arithmetic mean
    """
    sweep_path = Path(sweep_dir)
    tracelens_dir = sweep_path / 'tracelens_analysis'

    if not tracelens_dir.exists():
        print(f"Error: tracelens_analysis directory not found in {sweep_dir}")
        return

    print("="*80)
    print(f"Processing GPU Timeline data from: {sweep_dir}")
    print(f"Aggregation method: {'Geometric Mean' if use_geo_mean else 'Arithmetic Mean'}")
    print("="*80)

    # Find all thread configurations
    thread_configs = [d.name for d in tracelens_dir.iterdir()
                     if d.is_dir() and 'thread' in d.name]

    if not thread_configs:
        print("Error: No thread configuration directories found")
        return

    print(f"\nFound thread configurations: {sorted(thread_configs)}")

    all_results = []

    # Process each thread configuration
    for thread_config in sorted(thread_configs):
        individual_reports_dir = tracelens_dir / thread_config / 'individual_reports'

        if not individual_reports_dir.exists():
            print(f"  Warning: {individual_reports_dir} not found, skipping...")
            continue

        print(f"\nProcessing: {thread_config}")
        print("-" * 60)

        # Find all perf_*ch_rank*.xlsx files
        perf_files = sorted(glob.glob(str(individual_reports_dir / 'perf_*ch_rank*.xlsx')))

        if not perf_files:
            print(f"  Warning: No performance files found in {individual_reports_dir}")
            continue

        # Group files by channel configuration
        channel_groups = {}
        for file_path in perf_files:
            filename = os.path.basename(file_path)
            # Extract channel config: perf_28ch_rank0.xlsx -> 28ch
            parts = filename.replace('perf_', '').replace('.xlsx', '').split('_')
            channel_config = parts[0]  # e.g., "28ch"
            rank = int(parts[1].replace('rank', ''))

            if channel_config not in channel_groups:
                channel_groups[channel_config] = []
            channel_groups[channel_config].append((rank, file_path))

        # Process each channel configuration
        for channel_config in sorted(channel_groups.keys(), key=lambda x: int(x.replace('ch', ''))):
            rank_files = sorted(channel_groups[channel_config], key=lambda x: x[0])
            num_ranks = len(rank_files)

            print(f"  {channel_config}: Processing {num_ranks} ranks...")

            # Read gpu_timeline data from all ranks
            rank_data = []
            for rank, file_path in rank_files:
                try:
                    df = pd.read_excel(file_path, sheet_name='gpu_timeline')
                    df['rank'] = rank
                    rank_data.append(df)
                except Exception as e:
                    print(f"    Warning: Could not read {os.path.basename(file_path)}: {e}")

            if not rank_data:
                print(f"    No valid data for {channel_config}")
                continue

            # Combine all ranks
            combined = pd.concat(rank_data, ignore_index=True)

            # Aggregate by type across ranks
            agg_func = geometric_mean if use_geo_mean else 'mean'
            aggregated = combined.groupby('type').agg({
                'time ms': agg_func,
                'percent': agg_func
            }).reset_index()

            # Add metadata
            threads_num = int(thread_config.replace('thread', ''))
            channels_num = int(channel_config.replace('ch', ''))

            aggregated['thread_config'] = thread_config
            aggregated['threads_num'] = threads_num
            aggregated['channel_config'] = channel_config
            aggregated['channels_num'] = channels_num
            aggregated['full_config'] = f"{thread_config}_{channel_config}"
            aggregated['num_ranks'] = num_ranks

            all_results.append(aggregated)
            print(f"    ✓ Aggregated across {num_ranks} ranks")

    if not all_results:
        print("\nError: No data was processed")
        return

    # Combine all results
    print("\n" + "="*80)
    print("CREATING OUTPUT FILE")
    print("="*80)

    final_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns
    column_order = [
        'full_config',
        'threads_num',
        'thread_config',
        'channels_num',
        'channel_config',
        'num_ranks',
        'type',
        'time ms',
        'percent'
    ]
    final_df = final_df[column_order]

    # Sort by configuration
    final_df = final_df.sort_values(['threads_num', 'channels_num', 'type'])

    # Save to Excel
    method_suffix = 'geomean' if use_geo_mean else 'mean'
    output_path = tracelens_dir / f'gpu_timeline_all_configs_{method_suffix}.xlsx'

    # Create multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: All data in long format
        final_df.to_excel(writer, sheet_name='All_Data', index=False)

        # Sheet 2: Pivot - Time (ms) by type and config
        pivot_time = final_df.pivot_table(
            values='time ms',
            index='type',
            columns='full_config',
            aggfunc='first'
        )
        pivot_time.to_excel(writer, sheet_name='Pivot_Time_ms')

        # Sheet 3: Pivot - Percent by type and config
        pivot_percent = final_df.pivot_table(
            values='percent',
            index='type',
            columns='full_config',
            aggfunc='first'
        )
        pivot_percent.to_excel(writer, sheet_name='Pivot_Percent')

        # Sheet 4: Summary by configuration
        summary = final_df.groupby('full_config').agg({
            'threads_num': 'first',
            'channels_num': 'first',
            'num_ranks': 'first'
        }).reset_index()

        # Add key metrics for each config
        for metric_type in ['computation_time', 'exposed_comm_time', 'busy_time', 'idle_time', 'total_time']:
            metric_data = final_df[final_df['type'] == metric_type].set_index('full_config')['time ms']
            summary[f'{metric_type}_ms'] = summary['full_config'].map(metric_data)

        summary.to_excel(writer, sheet_name='Summary_By_Config', index=False)

    print(f"✓ Saved: {output_path}")
    print(f"  Sheets created:")
    print(f"    1. All_Data - Complete dataset")
    print(f"    2. Pivot_Time_ms - Matrix view of time (ms)")
    print(f"    3. Pivot_Percent - Matrix view of percentages")
    print(f"    4. Summary_By_Config - Key metrics per configuration")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nMetric Types Found:")
    for metric_type in sorted(final_df['type'].unique()):
        count = len(final_df[final_df['type'] == metric_type])
        print(f"  {metric_type:<25} ({count} configurations)")

    print("\nConfigurations Processed:")
    configs = final_df.groupby('full_config')['num_ranks'].first().sort_index()
    for config, num_ranks in configs.items():
        print(f"  {config:<25} ({num_ranks} ranks)")

    # Show key metrics comparison
    print("\n" + "="*80)
    print("KEY METRICS COMPARISON (Sorted by Busy Time)")
    print("="*80)

    busy_time_data = final_df[final_df['type'] == 'busy_time'][['full_config', 'time ms', 'percent']].sort_values('time ms')
    print("\nBusy Time (lower is better):")
    print(busy_time_data.to_string(index=False))

    idle_time_data = final_df[final_df['type'] == 'idle_time'][['full_config', 'time ms', 'percent']].sort_values('time ms')
    print("\nIdle Time (lower is better):")
    print(idle_time_data.to_string(index=False))

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput file: {output_path}")
    print("Open in Excel to create custom pivots and charts!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Process GPU timeline data from individual reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with arithmetic mean (default)
  python process_gpu_timeline.py --sweep-dir /path/to/sweep_20251124_222204

  # Process with geometric mean
  python process_gpu_timeline.py --sweep-dir /path/to/sweep_20251124_222204 --geo-mean
        """
    )

    parser.add_argument(
        '--sweep-dir',
        required=True,
        help='Path to sweep directory (e.g., sweep_20251124_222204)'
    )

    parser.add_argument(
        '--geo-mean',
        action='store_true',
        help='Use geometric mean instead of arithmetic mean for aggregation'
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


if __name__ == '__main__':
    exit(main())
