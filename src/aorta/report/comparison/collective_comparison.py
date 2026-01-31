"""Collective/NCCL comparison logic.

Creates comparison sheets for NCCL collective data, matching the behavior of
scripts/tracelens_single_config/add_collective_comparison.py
"""

from typing import Dict, List, Optional

import pandas as pd


# Metrics to compare
NCCL_NUMERIC_COLS = [
    "comm_latency_mean",
    "algo bw (GB/s)_mean",
    "bus bw (GB/s)_mean",
    "Total comm latency (ms)",
    "count",
]

# Grouping columns for NCCL data
NCCL_GROUP_COLS = ["Collective name", "dtype", "In msg nelems"]

# Summary sheets to process
NCCL_SUMMARY_SHEETS = ["nccl_summary_implicit_sync", "nccl_summary_long"]


def add_collective_comparison(
    combined_data: Dict[str, pd.DataFrame],
    baseline_label: str,
    test_label: str,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Add comparison sheets for collective/NCCL data.

    Args:
        combined_data: Dict from combine_excel_files()
        baseline_label: Label for baseline
        test_label: Label for test
        verbose: Print progress messages

    Returns:
        Dict with summary sheets + new comparison sheets:
        - 'nccl_implicit_sync_cmp': Comparison for nccl_summary_implicit_sync
        - 'nccl_long_cmp': Comparison for nccl_summary_long

    Processes sheets:
        - 'nccl_summary_implicit_sync' → 'nccl_implicit_sync_cmp'
        - 'nccl_summary_long' → 'nccl_long_cmp'

    Groups by: ['Collective name', 'dtype', 'In msg nelems']

    For each metric, creates columns:
        - {baseline}_{metric}, {test}_{metric}
        - diff_{metric}, percent_change_{metric}, ratio_{metric}

    percent_change semantics (positive = better):
        - Latency/time: (baseline - test) / baseline × 100
        - Bandwidth: (test - baseline) / baseline × 100
    """
    result = combined_data.copy()

    if verbose:
        print(f"\nCreating comparison sheets...")

    for sheet_name in NCCL_SUMMARY_SHEETS:
        if sheet_name not in combined_data:
            if verbose:
                print(f"  Skipping {sheet_name} (not found)")
            continue

        df = combined_data[sheet_name]

        # Get actual source values
        sources = df["source"].unique()
        if len(sources) < 2:
            if verbose:
                print(f"  Skipping {sheet_name} (only {len(sources)} source(s))")
            continue

        actual_baseline = sources[0]
        actual_test = sources[1]

        # Create comparison
        comparison = _create_collective_comparison(
            df,
            actual_baseline,
            actual_test,
            baseline_label,
            test_label,
            verbose,
        )

        # Sheet name: nccl_summary_implicit_sync → nccl_implicit_sync_cmp
        comparison_sheet_name = sheet_name.replace("nccl_summary_", "nccl_") + "_cmp"
        result[comparison_sheet_name] = comparison

        if verbose:
            print(f"  Created {comparison_sheet_name} ({len(comparison)} rows)")

    return result


def _create_collective_comparison(
    df: pd.DataFrame,
    actual_baseline: str,
    actual_test: str,
    baseline_label: str,
    test_label: str,
    verbose: bool,
) -> pd.DataFrame:
    """Create comparison DataFrame for a single NCCL summary sheet."""
    baseline_df = df[df["source"] == actual_baseline].copy()
    test_df = df[df["source"] == actual_test].copy()

    if len(baseline_df) == 0 or len(test_df) == 0:
        return pd.DataFrame()

    # Determine grouping columns (use fallback if some columns are missing)
    group_cols = _get_available_group_cols(baseline_df)

    if verbose:
        print(f"    Grouping by: {group_cols}")

    rows = []

    # Group baseline data
    for name, base_group in baseline_df.groupby(group_cols, as_index=False):
        # Find matching test group
        test_group = _find_matching_group(test_df, group_cols, name)

        if test_group is None or len(test_group) == 0:
            continue

        # Build comparison row
        comp_row = {}

        # Copy grouping columns
        if isinstance(name, tuple):
            for col, val in zip(group_cols, name):
                comp_row[col] = val
        else:
            comp_row[group_cols[0]] = name

        # Compare each numeric metric
        for col in NCCL_NUMERIC_COLS:
            if col not in base_group.columns or col not in test_group.columns:
                continue

            base_val = base_group[col].values[0]
            test_val = test_group[col].values[0]

            # Store values
            comp_row[f"{actual_baseline}_{col}"] = base_val
            comp_row[f"{actual_test}_{col}"] = test_val
            comp_row[f"diff_{col}"] = test_val - base_val

            # Calculate percent_change with correct semantics
            pct_change = _calculate_percent_change(col, base_val, test_val)
            if pct_change is not None:
                comp_row[f"percent_change_{col}"] = pct_change

            # Ratio
            comp_row[f"ratio_{col}"] = test_val / base_val if base_val != 0 else 0

        rows.append(comp_row)

    return pd.DataFrame(rows)


def _get_available_group_cols(df: pd.DataFrame) -> List[str]:
    """Get available grouping columns from DataFrame."""
    available = [col for col in NCCL_GROUP_COLS if col in df.columns]
    if not available:
        # Fallback to just Collective name if nothing else available
        if "Collective name" in df.columns:
            return ["Collective name"]
        raise ValueError("No grouping columns found in DataFrame")
    return available


def _find_matching_group(
    test_df: pd.DataFrame,
    group_cols: List[str],
    name,
) -> Optional[pd.DataFrame]:
    """Find matching group in test DataFrame."""
    if isinstance(name, tuple):
        mask = pd.Series([True] * len(test_df), index=test_df.index)
        for col, val in zip(group_cols, name):
            mask = mask & (test_df[col] == val)
    else:
        mask = test_df[group_cols[0]] == name

    result = test_df.loc[mask]
    return result if len(result) > 0 else None


def _calculate_percent_change(
    col_name: str,
    base_val: float,
    test_val: float,
) -> Optional[float]:
    """
    Calculate percent_change with correct semantics for the metric type.

    For latency/time: Lower is better → positive when test is faster
        Formula: (baseline - test) / baseline × 100

    For bandwidth: Higher is better → positive when test has more bandwidth
        Formula: (test - baseline) / baseline × 100

    Returns:
        Percent change value, or None for metrics that shouldn't have percent_change
    """
    if base_val == 0:
        return 0.0

    col_lower = col_name.lower()

    if "latency" in col_lower or "time" in col_lower:
        # Lower is better - positive when test is faster
        return (base_val - test_val) / base_val * 100
    elif "bw" in col_lower or "bandwidth" in col_lower:
        # Higher is better - positive when test is better
        return (test_val - base_val) / base_val * 100
    elif "count" in col_lower:
        # Count doesn't need percent_change
        return None

    return None


def get_percent_change_columns(comparison_df: pd.DataFrame) -> List[str]:
    """Get list of percent_change columns in a comparison DataFrame."""
    return [col for col in comparison_df.columns if col.startswith("percent_change_")]
