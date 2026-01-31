"""
Sweep configuration analysis - analyze traces from parameter sweep experiments.

Processes GPU timeline data from TraceLens individual reports across multiple
thread and channel configurations, aggregating across ranks.

Supports two modes:
1. Run TraceLens on all configurations then aggregate (default)
2. Aggregate existing TraceLens reports only (--skip-tracelens)
"""

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .analyze_single import analyze_single_config


def discover_and_run_tracelens(
    sweep_dir: Path,
    short_kernel_threshold_us: int = 50,
    topk_ops: int = 100,
    verbose: bool = False,
) -> Path:
    """
    Discover thread/channel configs and run TraceLens on each.

    Expected input structure:
        sweep_dir/
        ├── 256thread/
        │   ├── nccl_28channels/
        │   │   └── torch_profiler/rank*/
        │   └── nccl_42channels/
        └── 512thread/
            └── ...

    Output structure:
        sweep_dir/
        └── tracelens_analysis/
            ├── 256thread/
            │   └── individual_reports/
            │       ├── perf_28ch_rank0.xlsx
            │       └── ...
            └── 512thread/
                └── ...

    Args:
        sweep_dir: Path to sweep directory with thread/channel subdirectories
        short_kernel_threshold_us: Threshold for short kernel study
        topk_ops: Number of top operations to include
        verbose: Whether to print verbose output

    Returns:
        Path to tracelens_analysis output directory
    """
    sweep_path = Path(sweep_dir)
    output_base = sweep_path / "tracelens_analysis"

    # Discover thread configurations (e.g., "256thread", "512thread")
    thread_dirs = sorted([
        d for d in sweep_path.iterdir()
        if d.is_dir() and "thread" in d.name
    ])

    if not thread_dirs:
        raise ValueError(f"No thread configurations found in {sweep_dir}")

    print("=" * 80)
    print("Step 0: Running TraceLens on All Configurations")
    print("=" * 80)
    print(f"\nDiscovered thread configs: {[d.name for d in thread_dirs]}")

    for thread_dir in thread_dirs:
        thread_name = thread_dir.name  # e.g., "256thread"

        # Find channel configs (e.g., "nccl_28channels")
        channel_dirs = sorted([
            d for d in thread_dir.iterdir()
            if d.is_dir() and "channel" in d.name
        ])

        if not channel_dirs:
            print(f"  [WARN] No channel configs in {thread_name}")
            continue

        print(f"\n{thread_name}: {[d.name for d in channel_dirs]}")

        for channel_dir in channel_dirs:
            # Extract channel number (e.g., "nccl_28channels" -> "28")
            channel_name = channel_dir.name
            channel_match = re.search(r"(\d+)", channel_name)
            channel_num = channel_match.group(1) if channel_match else "0"

            # Look for torch_profiler directory
            trace_dir = channel_dir / "torch_profiler"
            if not trace_dir.exists():
                print(f"    [SKIP] {channel_name} - no torch_profiler/")
                continue

            # Output to: tracelens_analysis/{thread}/individual_reports/
            output_dir = output_base / thread_name

            print(f"  Processing {channel_name}...")

            try:
                analyze_single_config(
                    input_dir=trace_dir,
                    output_dir=output_dir,
                    run_individual=True,
                    run_collective=False,  # Skip collective for sweep
                    aggregate_timeline=False,  # Will aggregate at sweep level
                    short_kernel_threshold_us=short_kernel_threshold_us,
                    topk_ops=topk_ops,
                    verbose=verbose,
                    output_prefix=f"{channel_num}ch",  # e.g., "28ch"
                )
                print(f"    [OK] {channel_name}")
            except Exception as e:
                print(f"    [ERROR] {channel_name}: {e}")

    print("\n" + "=" * 80)
    print("TraceLens Analysis Complete")
    print("=" * 80)

    return output_base


def geometric_mean(values: np.ndarray) -> float:
    """Calculate geometric mean, handling zeros."""
    values = np.array(values)
    # Replace zeros with small value to avoid log(0)
    values = np.where(values == 0, 1e-10, values)
    return float(np.exp(np.mean(np.log(values))))


def parse_perf_filename(filename: str) -> tuple:
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


def group_files_by_channel(perf_files: List[str]) -> Dict[str, List[tuple]]:
    """
    Group performance files by channel configuration.

    Args:
        perf_files: List of file paths

    Returns:
        dict: {channel_config: [(rank, file_path), ...]}
    """
    channel_groups = {}
    for file_path in perf_files:
        filename = Path(file_path).name
        channel_config, rank = parse_perf_filename(filename)

        if channel_config not in channel_groups:
            channel_groups[channel_config] = []
        channel_groups[channel_config].append((rank, file_path))

    return channel_groups


def read_rank_data(rank_files: List[tuple], verbose: bool = False) -> List[pd.DataFrame]:
    """
    Read gpu_timeline data from all rank files.

    Args:
        rank_files: List of (rank, file_path) tuples
        verbose: Whether to print verbose output

    Returns:
        list: List of DataFrames with rank column added
    """
    rank_data = []
    for rank, file_path in rank_files:
        try:
            df = pd.read_excel(file_path, sheet_name="gpu_timeline")
            df["rank"] = rank
            rank_data.append(df)
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not read {Path(file_path).name}: {e}")
    return rank_data


def aggregate_rank_data(
    rank_data: List[pd.DataFrame],
    thread_config: str,
    channel_config: str,
    num_ranks: int,
    use_geo_mean: bool,
) -> pd.DataFrame:
    """
    Aggregate data across ranks and add metadata.

    Args:
        rank_data: List of DataFrames
        thread_config: Thread configuration string (e.g., '256thread')
        channel_config: Channel configuration string (e.g., '28ch')
        num_ranks: Number of ranks
        use_geo_mean: Whether to use geometric mean

    Returns:
        DataFrame: Aggregated data with metadata
    """
    combined = pd.concat(rank_data, ignore_index=True)

    agg_func = geometric_mean if use_geo_mean else "mean"
    aggregated = (
        combined.groupby("type")
        .agg({"time ms": agg_func, "percent": agg_func})
        .reset_index()
    )

    # Add metadata
    aggregated["thread_config"] = thread_config
    aggregated["threads_num"] = int(thread_config.replace("thread", ""))
    aggregated["channel_config"] = channel_config
    aggregated["channels_num"] = int(channel_config.replace("ch", ""))
    aggregated["full_config"] = f"{thread_config}_{channel_config}"
    aggregated["num_ranks"] = num_ranks

    return aggregated


def process_channel_config(
    channel_config: str,
    channel_groups: Dict[str, List[tuple]],
    use_geo_mean: bool,
    thread_config: str,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Process a single channel configuration.

    Args:
        channel_config: Channel configuration string
        channel_groups: Dict of channel groups
        use_geo_mean: Whether to use geometric mean
        thread_config: Thread configuration string
        verbose: Whether to print verbose output

    Returns:
        DataFrame or None: Aggregated data, or None if no valid data
    """
    rank_files = sorted(channel_groups[channel_config], key=lambda x: x[0])
    num_ranks = len(rank_files)

    if verbose:
        print(f"  {channel_config}: Processing {num_ranks} ranks...")

    rank_data = read_rank_data(rank_files, verbose)

    if not rank_data:
        if verbose:
            print(f"    No valid data for {channel_config}")
        return None

    aggregated = aggregate_rank_data(
        rank_data, thread_config, channel_config, num_ranks, use_geo_mean
    )
    if verbose:
        print(f"    [OK] Aggregated across {num_ranks} ranks")

    return aggregated


def process_thread_config(
    thread_config: str,
    tracelens_dir: Path,
    use_geo_mean: bool,
    verbose: bool = False,
) -> List[pd.DataFrame]:
    """
    Process a single thread configuration.

    Args:
        thread_config: Thread configuration string
        tracelens_dir: Path to tracelens_analysis directory
        use_geo_mean: Whether to use geometric mean
        verbose: Whether to print verbose output

    Returns:
        list: List of aggregated DataFrames
    """
    individual_reports_dir = tracelens_dir / thread_config / "individual_reports"

    if not individual_reports_dir.exists():
        if verbose:
            print(f"  Warning: {individual_reports_dir} not found, skipping...")
        return []

    if verbose:
        print(f"\nProcessing: {thread_config}")
        print("-" * 60)

    perf_files = sorted(glob.glob(str(individual_reports_dir / "perf_*ch_rank*.xlsx")))

    if not perf_files:
        if verbose:
            print(f"  Warning: No performance files found in {individual_reports_dir}")
        return []

    channel_groups = group_files_by_channel(perf_files)
    results = []

    # Process each channel configuration (sorted by channel number)
    sorted_channels = sorted(
        channel_groups.keys(), key=lambda x: int(x.replace("ch", ""))
    )
    for channel_config in sorted_channels:
        aggregated = process_channel_config(
            channel_config, channel_groups, use_geo_mean, thread_config, verbose
        )
        if aggregated is not None:
            results.append(aggregated)

    return results


def create_pivot_sheet(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Create a pivot table from the dataframe.

    Args:
        df: Source DataFrame
        value_col: Column to use for values

    Returns:
        DataFrame: Pivot table
    """
    return df.pivot_table(
        values=value_col, index="type", columns="full_config", aggfunc="first"
    )


def create_summary_sheet(df: pd.DataFrame) -> pd.DataFrame:
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


def print_summary_report(final_df: pd.DataFrame, verbose: bool = False) -> None:
    """Print summary statistics and comparisons."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nMetric Types Found:")
    for metric_type in sorted(final_df["type"].unique()):
        count = len(final_df[final_df["type"] == metric_type])
        print(f"  {metric_type:<25} ({count} configurations)")

    print("\nConfigurations Processed:")
    configs = final_df.groupby("full_config")["num_ranks"].first().sort_index()
    for config, num_ranks in configs.items():
        print(f"  {config:<25} ({num_ranks} ranks)")

    if verbose:
        print("\n" + "=" * 80)
        print("KEY METRICS COMPARISON (Sorted by Busy Time)")
        print("=" * 80)

        for metric, desc in [
            ("busy_time", "Busy Time (lower is better)"),
            ("idle_time", "Idle Time (lower is better)"),
        ]:
            metric_data = final_df[final_df["type"] == metric][
                ["full_config", "time ms", "percent"]
            ].sort_values("time ms")
            print(f"\n{desc}:")
            print(metric_data.to_string(index=False))


def analyze_sweep_config(
    sweep_dir: Path,
    output_dir: Optional[Path] = None,
    use_geo_mean: bool = False,
    skip_tracelens: bool = False,
    short_kernel_threshold_us: int = 50,
    topk_ops: int = 100,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Analyze a sweep directory: run TraceLens on all configs and aggregate results.

    By default, runs TraceLens analysis on all thread/channel configurations first,
    then aggregates GPU timeline data. Use skip_tracelens=True to only aggregate
    existing reports.

    Args:
        sweep_dir: Path to sweep directory with thread/channel subdirectories
        output_dir: Output directory (default: sweep_dir/tracelens_analysis/)
        use_geo_mean: If True, use geometric mean; otherwise use arithmetic mean
        skip_tracelens: If True, skip TraceLens analysis (only aggregate existing)
        short_kernel_threshold_us: Threshold for short kernel study
        topk_ops: Number of top operations to include
        verbose: Whether to print verbose output

    Returns:
        Path to output Excel file or None if no data processed
    """
    sweep_path = Path(sweep_dir)
    tracelens_dir = sweep_path / "tracelens_analysis"

    # Step 1: Run TraceLens on all configurations (unless skipped)
    if not skip_tracelens:
        discover_and_run_tracelens(
            sweep_dir=sweep_path,
            short_kernel_threshold_us=short_kernel_threshold_us,
            topk_ops=topk_ops,
            verbose=verbose,
        )

    # Step 2: Aggregate results
    if not tracelens_dir.exists():
        raise FileNotFoundError(
            f"tracelens_analysis directory not found in {sweep_dir}"
        )

    agg_method = "Geometric Mean" if use_geo_mean else "Arithmetic Mean"
    print("=" * 80)
    print(f"Processing GPU Timeline data from: {sweep_dir}")
    print(f"Aggregation method: {agg_method}")
    print("=" * 80)

    # Find all thread configurations
    thread_configs = [
        d.name for d in tracelens_dir.iterdir() if d.is_dir() and "thread" in d.name
    ]

    if not thread_configs:
        raise ValueError("No thread configuration directories found")

    print(f"\nFound thread configurations: {sorted(thread_configs)}")

    # Process all thread configurations
    all_results = []
    for thread_config in sorted(thread_configs):
        results = process_thread_config(thread_config, tracelens_dir, use_geo_mean, verbose)
        all_results.extend(results)

    if not all_results:
        print("\nError: No data was processed")
        return None

    # Combine and format results
    print("\n" + "=" * 80)
    print("CREATING OUTPUT FILE")
    print("=" * 80)

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

    # Determine output path
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = tracelens_dir

    method_suffix = "geomean" if use_geo_mean else "mean"
    output_file = output_path / f"gpu_timeline_all_configs_{method_suffix}.xlsx"

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="All_Data", index=False)
        create_pivot_sheet(final_df, "time ms").to_excel(
            writer, sheet_name="Pivot_Time_ms"
        )
        create_pivot_sheet(final_df, "percent").to_excel(
            writer, sheet_name="Pivot_Percent"
        )
        create_summary_sheet(final_df).to_excel(
            writer, sheet_name="Summary_By_Config", index=False
        )

    print(f"[SAVED] {output_file}")
    print("  Sheets created:")
    print("    1. All_Data - Complete dataset")
    print("    2. Pivot_Time_ms - Matrix view of time (ms)")
    print("    3. Pivot_Percent - Matrix view of percentages")
    print("    4. Summary_By_Config - Key metrics per configuration")

    # Print summary
    print_summary_report(final_df, verbose)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nOutput file: {output_file}")
    print("Open in Excel to create custom pivots and charts!")

    return output_file

