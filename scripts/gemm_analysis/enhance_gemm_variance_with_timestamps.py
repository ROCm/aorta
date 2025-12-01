#!/usr/bin/env python3
"""
Enhance the top5_gemm_kernels_time_variance.csv with timestamp information.
For each row, this script will find the specific GEMM kernel instances with
min and max durations and add their timestamps.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

def find_min_max_kernel_timestamps(trace_file: Path, kernel_name: str,
                                 min_duration_us: float, max_duration_us: float) -> Dict[str, Optional[float]]:
    """
    Find timestamps for kernel instances with min and max durations.
    Durations are in microseconds to match trace file format.
    Returns dict with 'min_timestamp_ms', 'max_timestamp_ms', and found durations for verification.
    """

    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {trace_file}: {e}")
        return {'min_timestamp_ms': None, 'max_timestamp_ms': None}

    if 'traceEvents' not in data:
        return {'min_timestamp_ms': None, 'max_timestamp_ms': None}

    events = data['traceEvents']

    # Find all instances of this kernel
    kernel_instances = []
    for event in events:
        if (event.get('cat') == 'kernel' and
            event.get('name', '').startswith(kernel_name)):

            # Duration and timestamp are in microseconds in PyTorch trace file
            duration_us = event.get('dur', 0)
            timestamp_us = event.get('ts', 0)
            timestamp_ms = timestamp_us / 1000.0

            kernel_instances.append({
                'duration_us': duration_us,
                'timestamp_ms': timestamp_ms,
                'timestamp_us': timestamp_us
            })

    if not kernel_instances:
        print(f"  Warning: No instances of kernel {kernel_name[:50]}... found")
        return {
            'min_timestamp_ms': None,
            'max_timestamp_ms': None,
            'min_duration_found_us': None,
            'max_duration_found_us': None
        }

    # Sort by duration
    kernel_instances.sort(key=lambda x: x['duration_us'])

    # Option 1: Use all instances (current behavior - RECOMMENDED)
    # Get the actual minimum and maximum instances
    min_instance = kernel_instances[0]  # Shortest duration
    max_instance = kernel_instances[-1]  # Longest duration

    # Option 2: Match TraceLens top 100 (uncomment to use)
    # if len(kernel_instances) > 100:
    #     # Take only the top 100 longest-running instances
    #     kernel_instances_top100 = kernel_instances[-100:]
    #     min_instance = kernel_instances_top100[0]  # Min of top 100
    #     max_instance = kernel_instances_top100[-1]  # Max of top 100
    #     print(f"  Note: Using top 100 of {len(kernel_instances)} instances")
    # else:
    #     min_instance = kernel_instances[0]
    #     max_instance = kernel_instances[-1]

    # Verify the matches are reasonably close (within 1% tolerance)
    min_tolerance = min_duration_us * 0.01
    max_tolerance = max_duration_us * 0.01

    result = {
        'min_timestamp_ms': min_instance['timestamp_ms'],
        'max_timestamp_ms': max_instance['timestamp_ms'],
        'min_duration_found_us': min_instance['duration_us'],
        'max_duration_found_us': max_instance['duration_us']
    }

    # Print warnings if mismatch
    if abs(min_instance['duration_us'] - min_duration_us) > min_tolerance:
        print(f"  Warning: Min duration mismatch: found {min_instance['duration_us']:.3f}µs vs expected {min_duration_us:.3f}µs")

    if abs(max_instance['duration_us'] - max_duration_us) > max_tolerance:
        print(f"  Warning: Max duration mismatch: found {max_instance['duration_us']:.3f}µs vs expected {max_duration_us:.3f}µs")

    return result

def get_trace_file_path(base_path: Path, threads: int, channel: int, rank: int) -> Optional[Path]:
    """Find the trace file for a given configuration."""

    trace_dir = base_path / f"{threads}thread" / f"nccl_{channel}channels" / "torch_profiler" / f"rank{rank}"

    if not trace_dir.exists():
        return None

    # Look for JSON trace files
    trace_files = list(trace_dir.glob("*.json"))

    if not trace_files:
        return None

    # Prefer customer_trace files, but use any available
    for pattern in ["customer_trace*.json", "*.json"]:
        matches = list(trace_dir.glob(pattern))
        if matches:
            return matches[0]

    return trace_files[0] if trace_files else None

def enhance_csv_with_timestamps(input_csv: Path, output_csv: Path, base_path: Path):
    """Add timestamp columns to the variance CSV file."""

    # Read the existing CSV
    df = pd.read_csv(input_csv)

    # Add new columns
    df['min_duration_timestamp_ms'] = pd.NA
    df['max_duration_timestamp_ms'] = pd.NA
    df['time_between_min_max_ms'] = pd.NA
    df['min_duration_found_us'] = pd.NA  # For verification
    df['max_duration_found_us'] = pd.NA  # For verification

    total_rows = len(df)
    print(f"Processing {total_rows} rows...")

    for idx, row in df.iterrows():
        print(f"\nProcessing row {idx + 1}/{total_rows}")

        # Extract configuration
        threads = int(row['threads'])
        channel = int(row['channel'])
        rank = int(row['rank'])
        kernel_name = row['kernel_name']

        # Get durations in microseconds
        min_duration_us = float(row['kernel_time_min_us'])
        max_duration_us = float(row['kernel_time_max_us'])

        print(f"  Config: {threads}thread/{channel}ch/rank{rank}")
        print(f"  Kernel: {kernel_name[:60]}...")
        print(f"  Duration range: [{min_duration_us:.3f}, {max_duration_us:.3f}] µs")

        # Find trace file
        trace_file = get_trace_file_path(base_path, threads, channel, rank)

        if trace_file is None:
            print(f"  Warning: No trace file found")
            continue

        print(f"  Using trace: {trace_file.name}")

        # Find timestamps
        timestamps = find_min_max_kernel_timestamps(
            trace_file, kernel_name, min_duration_us, max_duration_us
        )

        if timestamps['min_timestamp_ms'] is not None:
            df.at[idx, 'min_duration_timestamp_ms'] = timestamps['min_timestamp_ms']

        if timestamps['max_timestamp_ms'] is not None:
            df.at[idx, 'max_duration_timestamp_ms'] = timestamps['max_timestamp_ms']

        # Store found durations for verification
        if timestamps['min_duration_found_us'] is not None:
            df.at[idx, 'min_duration_found_us'] = timestamps['min_duration_found_us']

        if timestamps['max_duration_found_us'] is not None:
            df.at[idx, 'max_duration_found_us'] = timestamps['max_duration_found_us']

        # Calculate time between min and max occurrences
        if timestamps['min_timestamp_ms'] is not None and timestamps['max_timestamp_ms'] is not None:
            time_diff = abs(timestamps['max_timestamp_ms'] - timestamps['min_timestamp_ms'])
            df.at[idx, 'time_between_min_max_ms'] = time_diff
            print(f"  Found timestamps: min at {timestamps['min_timestamp_ms']:.3f}ms, "
                  f"max at {timestamps['max_timestamp_ms']:.3f}ms (diff: {time_diff:.3f}ms)")
            print(f"  Verification: found min={timestamps['min_duration_found_us']:.3f}µs (expected {min_duration_us:.3f}µs), "
                  f"found max={timestamps['max_duration_found_us']:.3f}µs (expected {max_duration_us:.3f}µs)")

    # Save enhanced CSV
    df.to_csv(output_csv, index=False)
    print(f"\nEnhanced CSV saved to: {output_csv}")

    # Print summary statistics
    valid_timestamps = df['min_duration_timestamp_ms'].notna().sum()
    print(f"\nSummary:")
    print(f"  Total rows: {total_rows}")
    print(f"  Rows with timestamps: {valid_timestamps}")
    print(f"  Success rate: {valid_timestamps/total_rows*100:.1f}%")

    if valid_timestamps > 0:
        time_diffs = df['time_between_min_max_ms'].dropna()
        if len(time_diffs) > 0:
            print(f"\nTime between min/max occurrences:")
            print(f"  Mean: {time_diffs.mean():.3f} ms")
            print(f"  Median: {time_diffs.median():.3f} ms")
            print(f"  Max: {time_diffs.max():.3f} ms")
            print(f"  Min: {time_diffs.min():.3f} ms")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhance GEMM variance CSV with timestamp information",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input-csv',
        type=Path,
        default=Path("/home/oyazdanb/aorta/experiments/sweep_20251121_155219/tracelens_analysis/top5_gemm_kernels_time_variance.csv"),
        help='Input CSV file with GEMM variance data'
    )

    parser.add_argument(
        '--output-csv',
        type=Path,
        default=None,
        help='Output CSV file (default: input_file_with_timestamps.csv)'
    )

    parser.add_argument(
        '--base-path',
        type=Path,
        default=Path("/home/oyazdanb/aorta/experiments/sweep_20251121_155219"),
        help='Base path to sweep directory containing trace files'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Set default output file if not specified
    if args.output_csv is None:
        args.output_csv = args.input_csv.parent / f"{args.input_csv.stem}_with_timestamps.csv"

    print("GEMM Variance Timestamp Enhancement")
    print("=" * 60)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Base path: {args.base_path}")
    print()

    # Verify input file exists
    if not args.input_csv.exists():
        print(f"Error: Input CSV not found: {args.input_csv}")
        return

    # Enhance the CSV with timestamps
    enhance_csv_with_timestamps(args.input_csv, args.output_csv, args.base_path)

    print("\n✅ Enhancement complete!")

if __name__ == "__main__":
    main()
