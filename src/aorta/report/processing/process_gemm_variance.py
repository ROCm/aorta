"""
GEMM Variance Timestamp Enhancement.

Enhances GEMM variance CSV (from analyze gemm) with actual kernel timestamps
by finding the specific kernel instances with min and max durations in the
original trace files.

Source: scripts/gemm_analysis/enhance_gemm_variance_with_timestamps.py
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def get_trace_file_path(
    base_path: Path, threads: int, channel: int, rank: int
) -> Optional[Path]:
    """
    Find the trace file for a given configuration.

    Args:
        base_path: Base path to sweep directory
        threads: Thread count
        channel: Channel count
        rank: Rank number

    Returns:
        Path to trace file or None if not found
    """
    trace_dir = (
        base_path
        / f"{threads}thread"
        / f"nccl_{channel}channels"
        / "torch_profiler"
        / f"rank{rank}"
    )

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


def find_min_max_kernel_timestamps(
    trace_file: Path,
    kernel_name: str,
    min_duration_us: float,
    max_duration_us: float,
    tolerance: float = 0.01,
) -> Dict[str, Optional[float]]:
    """
    Find timestamps for kernel instances with min and max durations.

    Args:
        trace_file: Path to trace JSON file
        kernel_name: Kernel name to search for
        min_duration_us: Expected minimum duration in microseconds
        max_duration_us: Expected maximum duration in microseconds
        tolerance: Tolerance for duration matching (fraction)

    Returns:
        Dict with min_timestamp_ms, max_timestamp_ms, and found durations
    """
    try:
        with open(trace_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {trace_file}: {e}")
        return {
            "min_timestamp_ms": None,
            "max_timestamp_ms": None,
            "min_duration_found_us": None,
            "max_duration_found_us": None,
        }

    if "traceEvents" not in data:
        return {
            "min_timestamp_ms": None,
            "max_timestamp_ms": None,
            "min_duration_found_us": None,
            "max_duration_found_us": None,
        }

    events = data["traceEvents"]

    # Find all instances of this kernel
    kernel_instances = []
    for event in events:
        if event.get("cat") == "kernel" and event.get("name", "").startswith(
            kernel_name
        ):
            # Duration and timestamp are in microseconds in PyTorch trace file
            duration_us = event.get("dur")
            timestamp_us = event.get("ts")

            # Skip events without proper duration or timestamp
            if duration_us is None or timestamp_us is None:
                continue

            timestamp_ms = timestamp_us / 1000.0

            kernel_instances.append(
                {
                    "duration_us": duration_us,
                    "timestamp_ms": timestamp_ms,
                    "timestamp_us": timestamp_us,
                }
            )

    if not kernel_instances:
        print(f"  Warning: No valid instances of kernel {kernel_name[:50]}... found")
        return {
            "min_timestamp_ms": None,
            "max_timestamp_ms": None,
            "min_duration_found_us": None,
            "max_duration_found_us": None,
        }

    # Sort by duration (guaranteed to exist and be non-None)
    kernel_instances.sort(key=lambda x: x["duration_us"])

    # Get the actual minimum and maximum instances
    min_instance = kernel_instances[0]  # Shortest duration
    max_instance = kernel_instances[-1]  # Longest duration

    # Verify the matches are reasonably close
    min_tolerance = min_duration_us * tolerance
    max_tolerance = max_duration_us * tolerance

    result = {
        "min_timestamp_ms": min_instance["timestamp_ms"],
        "max_timestamp_ms": max_instance["timestamp_ms"],
        "min_duration_found_us": min_instance["duration_us"],
        "max_duration_found_us": max_instance["duration_us"],
    }

    # Print warnings if mismatch
    if abs(min_instance["duration_us"] - min_duration_us) > min_tolerance:
        print(
            f"  Warning: Min duration mismatch: found {min_instance['duration_us']:.3f}us "
            f"vs expected {min_duration_us:.3f}us"
        )

    if abs(max_instance["duration_us"] - max_duration_us) > max_tolerance:
        print(
            f"  Warning: Max duration mismatch: found {max_instance['duration_us']:.3f}us "
            f"vs expected {max_duration_us:.3f}us"
        )

    return result


def enhance_gemm_variance(
    input_csv: Path,
    base_path: Path,
    output_csv: Optional[Path] = None,
    tolerance: float = 0.01,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Enhance GEMM variance CSV with timestamp information.

    For each row in the variance CSV, finds the corresponding trace file,
    searches for kernel instances with min/max durations, and adds their
    timestamps to the output.

    Args:
        input_csv: Input CSV file with GEMM variance data
        base_path: Base path to sweep directory containing trace files
        output_csv: Output CSV path (default: input_with_timestamps.csv)
        tolerance: Duration matching tolerance as fraction (default: 0.01 = 1%)
        verbose: Print verbose output

    Returns:
        Path to output CSV file or None if processing failed
    """
    input_path = Path(input_csv)
    base_path = Path(base_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    # Set default output file if not specified
    if output_csv is None:
        output_csv = input_path.parent / f"{input_path.stem}_with_timestamps.csv"
    else:
        output_csv = Path(output_csv)

    print("GEMM Variance Timestamp Enhancement")
    print("=" * 60)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Base path: {base_path}")
    print(f"Tolerance: {tolerance * 100:.1f}%")
    print()

    # Read the existing CSV
    df = pd.read_csv(input_path)

    # Add new columns
    df["min_duration_timestamp_ms"] = pd.NA
    df["max_duration_timestamp_ms"] = pd.NA
    df["time_between_min_max_ms"] = pd.NA
    df["min_duration_found_us"] = pd.NA
    df["max_duration_found_us"] = pd.NA

    total_rows = len(df)
    print(f"Processing {total_rows} rows...")

    successful_rows = 0

    for idx, row in df.iterrows():
        if verbose or (idx + 1) % 10 == 0 or idx == 0:
            print(f"\nProcessing row {idx + 1}/{total_rows}")

        # Extract configuration
        threads = int(row["threads"])
        channel = int(row["channel"])
        rank = int(row["rank"])
        kernel_name = row["kernel_name"]

        # Get durations in microseconds
        min_duration_us = float(row["kernel_time_min_us"])
        max_duration_us = float(row["kernel_time_max_us"])

        if verbose:
            print(f"  Config: {threads}thread/{channel}ch/rank{rank}")
            print(f"  Kernel: {kernel_name[:60]}...")
            print(f"  Duration range: [{min_duration_us:.3f}, {max_duration_us:.3f}] us")

        # Find trace file
        trace_file = get_trace_file_path(base_path, threads, channel, rank)

        if trace_file is None:
            if verbose:
                print("  Warning: No trace file found")
            continue

        if verbose:
            print(f"  Using trace: {trace_file.name}")

        # Find timestamps
        timestamps = find_min_max_kernel_timestamps(
            trace_file, kernel_name, min_duration_us, max_duration_us, tolerance
        )

        if timestamps["min_timestamp_ms"] is not None:
            df.at[idx, "min_duration_timestamp_ms"] = timestamps["min_timestamp_ms"]

        if timestamps["max_timestamp_ms"] is not None:
            df.at[idx, "max_duration_timestamp_ms"] = timestamps["max_timestamp_ms"]

        # Store found durations for verification
        if timestamps["min_duration_found_us"] is not None:
            df.at[idx, "min_duration_found_us"] = timestamps["min_duration_found_us"]

        if timestamps["max_duration_found_us"] is not None:
            df.at[idx, "max_duration_found_us"] = timestamps["max_duration_found_us"]

        # Calculate time between min and max occurrences
        if (
            timestamps["min_timestamp_ms"] is not None
            and timestamps["max_timestamp_ms"] is not None
        ):
            time_diff = abs(
                timestamps["max_timestamp_ms"] - timestamps["min_timestamp_ms"]
            )
            df.at[idx, "time_between_min_max_ms"] = time_diff
            successful_rows += 1

            if verbose:
                print(
                    f"  Found timestamps: min at {timestamps['min_timestamp_ms']:.3f}ms, "
                    f"max at {timestamps['max_timestamp_ms']:.3f}ms (diff: {time_diff:.3f}ms)"
                )

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save enhanced CSV
    df.to_csv(output_csv, index=False)
    print(f"\nEnhanced CSV saved to: {output_csv}")

    # Print summary statistics
    valid_timestamps = df["min_duration_timestamp_ms"].notna().sum()
    print(f"\nSummary:")
    print(f"  Total rows: {total_rows}")
    print(f"  Rows with timestamps: {valid_timestamps}")
    print(f"  Success rate: {valid_timestamps/total_rows*100:.1f}%")

    if valid_timestamps > 0:
        time_diffs = df["time_between_min_max_ms"].dropna()
        if len(time_diffs) > 0:
            print(f"\nTime between min/max occurrences:")
            print(f"  Mean: {time_diffs.mean():.3f} ms")
            print(f"  Median: {time_diffs.median():.3f} ms")
            print(f"  Max: {time_diffs.max():.3f} ms")
            print(f"  Min: {time_diffs.min():.3f} ms")

    print("\n[OK] Enhancement complete!")

    return output_csv

