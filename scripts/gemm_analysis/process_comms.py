#!/usr/bin/env python3
"""
NCCL Data Processing to CSV

Simple script to process NCCL collective reports and generate CSV files.
No plotting or analysis - just data extraction and aggregation.

Usage:
    python process_nccl_to_csv.py --sweep-dir /path/to/sweep_directory

Example:
    python process_nccl_to_csv.py --sweep-dir /home/oyazdanb/aorta/experiments/sweep_20251124_222204
"""

import pandas as pd
import os
import glob
import argparse
from pathlib import Path


def process_nccl_sweep(sweep_dir):
    """
    Process NCCL data from a sweep directory and generate CSV files.

    Args:
        sweep_dir: Path to sweep directory (e.g., sweep_20251124_222204)
    """
    sweep_path = Path(sweep_dir)
    tracelens_dir = sweep_path / 'tracelens_analysis'

    if not tracelens_dir.exists():
        print(f"Error: tracelens_analysis directory not found in {sweep_dir}")
        return

    print("="*80)
    print(f"Processing NCCL data from: {sweep_dir}")
    print("="*80)

    # Find all thread configurations
    thread_configs = [d.name for d in tracelens_dir.iterdir()
                     if d.is_dir() and 'thread' in d.name]

    if not thread_configs:
        print("Error: No thread configuration directories found")
        return

    print(f"\nFound thread configurations: {sorted(thread_configs)}")

    all_data = []

    # Process each thread configuration
    for thread_config in sorted(thread_configs):
        collective_dir = tracelens_dir / thread_config / 'collective_reports'

        if not collective_dir.exists():
            print(f"  Warning: {collective_dir} not found, skipping...")
            continue

        print(f"\nProcessing: {thread_config}")
        print("-" * 60)

        # Find all Excel files
        excel_files = sorted(glob.glob(str(collective_dir / 'collective_*.xlsx')))

        if not excel_files:
            print(f"  Warning: No collective_*.xlsx files found in {collective_dir}")
            continue

        for file_path in excel_files:
            filename = os.path.basename(file_path)
            channel_config = filename.replace('collective_', '').replace('.xlsx', '')
            channels_num = int(channel_config.replace('ch', ''))
            threads_num = int(thread_config.replace('thread', ''))

            print(f"  Reading: {filename}")

            try:
                # Read the nccl_summary_implicit_sync sheet
                df = pd.read_excel(file_path, sheet_name='nccl_summary_implicit_sync')

                # Add metadata columns
                df['thread_config'] = thread_config
                df['threads_num'] = threads_num
                df['channel_config'] = channel_config
                df['channels_num'] = channels_num
                df['source_file'] = filename
                df['full_config'] = f"{thread_config}_{channel_config}"

                all_data.append(df)
                print(f"    ✓ Loaded {len(df)} rows")

            except Exception as e:
                print(f"    ✗ Error reading {filename}: {e}")

    if not all_data:
        print("\nError: No data was loaded")
        return

    # Combine all data
    print("\n" + "="*80)
    print("COMBINING AND PROCESSING DATA")
    print("="*80)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")

    # Create unique operation IDs based on message size
    print("\nCreating unique operation IDs...")
    unique_sizes = sorted(combined_df['Full msg size (MB)'].unique())
    size_to_id = {size: f"OP_{i+1:02d}" for i, size in enumerate(unique_sizes)}
    combined_df['operation_id'] = combined_df['Full msg size (MB)'].map(size_to_id)

    # Create operation name
    def create_op_name(row):
        size_mb = row['Full msg size (MB)']
        if size_mb < 0.01:
            return f"tiny_{size_mb*1000:.3f}KB"
        elif size_mb < 100:
            return f"medium_{size_mb:.2f}MB"
        else:
            return f"large_{size_mb:.2f}MB"

    combined_df['operation_name'] = combined_df.apply(create_op_name, axis=1)

    # Reorder columns for better readability
    column_order = [
        # Unique identifiers
        'operation_id',
        'operation_name',
        'Full msg size (MB)',
        'In msg nelems',

        # Configuration
        'threads_num',
        'thread_config',
        'channels_num',
        'channel_config',
        'full_config',

        # Operation info
        'Collective name',
        'dtype',
        'Group size',
        'count',

        # Communication Latency
        'comm_latency_mean',
        'comm_latency_median',
        'comm_latency_min',
        'comm_latency_max',
        'Total comm latency (ms)',

        # Algorithm Bandwidth
        'algo bw (GB/s)_mean',
        'algo bw (GB/s)_median',
        'algo bw (GB/s)_min',
        'algo bw (GB/s)_max',

        # Bus Bandwidth
        'bus bw (GB/s)_mean',
        'bus bw (GB/s)_median',
        'bus bw (GB/s)_min',
        'bus bw (GB/s)_max',

        # Start Time Skew
        'skew in start time_mean',
        'skew in start time_median',
        'skew in start time_min',
        'skew in start time_max',

        # End Time Skew
        'skew in end time_mean',
        'skew in end time_median',
        'skew in end time_min',
        'skew in end time_max',

        # Process Group Info
        'Process Group Name',
        'source_file'
    ]

    combined_df = combined_df[column_order]
    combined_df = combined_df.sort_values(['operation_id', 'threads_num', 'channels_num'])

    # Save as Excel file (better for pivot tables)
    print("\n" + "="*80)
    print("SAVING DATA FILE")
    print("="*80)

    output_dir = tracelens_dir

    # Save as Excel (primary format - supports pivot tables)
    excel_path = output_dir / 'nccl_master_all_configs.xlsx'
    combined_df.to_excel(excel_path, index=False, sheet_name='NCCL_Data')
    print(f"✓ Saved Excel: {excel_path}")
    print(f"  Rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")

    # Also save as CSV (for pandas/command-line analysis)
    csv_path = output_dir / 'nccl_master_all_configs.csv'
    combined_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    print(f"  (Use Excel file for pivot tables, CSV for pandas/scripts)")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nOperation ID Mapping:")
    print("-" * 60)
    for op_id in sorted(combined_df['operation_id'].unique()):
        row = combined_df[combined_df['operation_id'] == op_id].iloc[0]
        print(f"  {op_id}: {row['Full msg size (MB)']:>12.6f} MB  ({int(row['In msg nelems']):>10} elements)  {row['operation_name']}")

    print("\nConfigurations:")
    print("-" * 60)
    configs = combined_df.groupby(['thread_config', 'channel_config']).size().reset_index(name='operations')
    for _, row in configs.iterrows():
        print(f"  {row['thread_config']:<12} {row['channel_config']:<8} -> {row['operations']} operations")

    print("\nTotal Communication Time by Configuration:")
    print("-" * 60)
    total_by_config = combined_df.groupby('full_config')['Total comm latency (ms)'].sum().sort_values()
    for config, total in total_by_config.items():
        print(f"  {config:<25}: {total:>10.2f} ms")

    print("\nBest Configuration by Operation:")
    print("-" * 60)
    for op_id in sorted(combined_df['operation_id'].unique()):
        op_data = combined_df[combined_df['operation_id'] == op_id]
        best = op_data.loc[op_data['comm_latency_mean'].idxmin()]
        print(f"  {op_id} ({best['Full msg size (MB)']:>8.2f} MB): {best['full_config']:<20} ({best['comm_latency_mean']:>8.2f} ms)")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {excel_path} (Excel - use for pivot tables)")
    print(f"  2. {csv_path} (CSV - use for pandas/scripts)")
    print("\nRecommended workflow:")
    print("  1. Open Excel file: libreoffice nccl_master_all_configs.xlsx")
    print("  2. Create pivot table: Select all → Insert → Pivot Table")
    print("  3. Setup: Rows=operation_id, Columns=full_config, Values=comm_latency_mean")
    print("\nSee README for more examples!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Process NCCL collective reports and generate CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process sweep directory
  python process_nccl_to_csv.py --sweep-dir /path/to/sweep_20251124_222204

  # Process and save to different location
  python process_nccl_to_csv.py --sweep-dir /path/to/sweep --output-dir /path/to/output
        """
    )

    parser.add_argument(
        '--sweep-dir',
        required=True,
        help='Path to sweep directory (e.g., sweep_20251124_222204)'
    )

    args = parser.parse_args()

    # Validate sweep directory
    sweep_path = Path(args.sweep_dir)
    if not sweep_path.exists():
        print(f"Error: Sweep directory does not exist: {args.sweep_dir}")
        return 1

    # Process the sweep
    try:
        process_nccl_sweep(args.sweep_dir)
        return 0
    except Exception as e:
        print(f"\nError processing sweep: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
