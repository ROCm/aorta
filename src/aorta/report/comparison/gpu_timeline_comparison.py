"""GPU timeline comparison logic.

Creates comparison sheets for GPU timeline data, matching the behavior of
scripts/tracelens_single_config/add_comparison_sheets.py
"""

from typing import Dict

import pandas as pd


def add_gpu_timeline_comparison(
    combined_data: Dict[str, pd.DataFrame],
    baseline_label: str,
    test_label: str,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Add comparison sheets for GPU timeline data.

    Args:
        combined_data: Dict from combine_excel_files()
        baseline_label: Label for baseline (for column naming)
        test_label: Label for test (for column naming)
        verbose: Print progress messages

    Returns:
        Dict with original sheets + new comparison sheets:
        - 'Comparison_By_Rank': Per-rank comparison
        - 'Summary_Comparison': Overall comparison

    Expects combined_data to have:
        - 'All_Ranks_Combined' sheet with: source, rank, type, time ms, percent
        - 'Summary' sheet with: source, type, time ms, percent

    Comparison columns created:
        - {baseline_label}_time_ms, {test_label}_time_ms
        - diff_time_ms, percent_change, status, ratio
        - {baseline_label}_percent, {test_label}_percent, diff_percent

    percent_change formula: (baseline - test) / baseline × 100
        - Positive = test is faster (better)
        - Negative = test is slower (worse)

    status thresholds:
        - "Better" if percent_change > 1
        - "Worse" if percent_change < -1
        - "Similar" otherwise
    """
    result = combined_data.copy()

    # Get actual source values from the dataframe
    all_combined = combined_data.get("All_Ranks_Combined")
    if all_combined is None:
        raise ValueError("'All_Ranks_Combined' sheet not found in combined data")

    sources = all_combined["source"].unique()
    if len(sources) < 2:
        raise ValueError(f"Expected 2 sources, found {len(sources)}: {sources}")

    # First source is baseline, second is test
    actual_baseline = sources[0]
    actual_test = sources[1]

    if verbose:
        print(f"\nCreating comparison sheets...")
        print(f"  Baseline source: {actual_baseline}")
        print(f"  Test source: {actual_test}")

    # Create Comparison_By_Rank
    comparison_by_rank = _create_comparison_by_rank(
        all_combined,
        actual_baseline,
        actual_test,
        baseline_label,
        test_label,
        verbose,
    )
    result["Comparison_By_Rank"] = comparison_by_rank

    # Create Summary_Comparison
    summary = combined_data.get("Summary")
    if summary is not None:
        summary_comparison = _create_summary_comparison(
            summary,
            actual_baseline,
            actual_test,
            baseline_label,
            test_label,
            verbose,
        )
        result["Summary_Comparison"] = summary_comparison

    return result


def _create_comparison_by_rank(
    all_combined: pd.DataFrame,
    actual_baseline: str,
    actual_test: str,
    baseline_label: str,
    test_label: str,
    verbose: bool,
) -> pd.DataFrame:
    """Create per-rank comparison DataFrame."""
    baseline_data = all_combined[all_combined["source"] == actual_baseline]
    test_data = all_combined[all_combined["source"] == actual_test]

    rows = []

    for rank in sorted(baseline_data["rank"].unique()):
        base_rank = baseline_data[baseline_data["rank"] == rank].set_index("type")
        test_rank = test_data[test_data["rank"] == rank].set_index("type")

        for metric_type in base_rank.index:
            if metric_type not in test_rank.index:
                continue

            base_time = base_rank.loc[metric_type, "time ms"]
            test_time = test_rank.loc[metric_type, "time ms"]

            # Calculate metrics
            ratio_val = test_time / base_time if base_time != 0 else 0

            # percent_change: positive when test is faster (takes less time)
            pct_change = (
                (base_time - test_time) / base_time * 100
                if base_time != 0
                else 0
            )

            # Determine status
            if pct_change > 1:
                status = "Better"
            elif pct_change < -1:
                status = "Worse"
            else:
                status = "Similar"

            # Build row
            row = {
                "rank": rank,
                "type": metric_type,
                f"{baseline_label}_time_ms": base_time,
                f"{test_label}_time_ms": test_time,
                "diff_time_ms": test_time - base_time,
                "percent_change": pct_change,
                "status": status,
                "ratio": ratio_val,
                f"{baseline_label}_percent": base_rank.loc[metric_type, "percent"],
                f"{test_label}_percent": test_rank.loc[metric_type, "percent"],
                "diff_percent": (
                    test_rank.loc[metric_type, "percent"]
                    - base_rank.loc[metric_type, "percent"]
                ),
            }
            rows.append(row)

    comparison_by_rank = pd.DataFrame(rows)

    if verbose:
        num_ranks = baseline_data["rank"].nunique()
        num_types = baseline_data["type"].nunique()
        print(f"  Created Comparison_By_Rank ({len(comparison_by_rank)} rows)")
        print(f"    {num_ranks} ranks × {num_types} types")

    return comparison_by_rank


def _create_summary_comparison(
    summary: pd.DataFrame,
    actual_baseline: str,
    actual_test: str,
    baseline_label: str,
    test_label: str,
    verbose: bool,
) -> pd.DataFrame:
    """Create overall summary comparison DataFrame."""
    baseline_summary = summary[summary["source"] == actual_baseline].set_index("type")
    test_summary = summary[summary["source"] == actual_test].set_index("type")

    rows = []

    for metric_type in baseline_summary.index:
        if metric_type not in test_summary.index:
            continue

        base_time = baseline_summary.loc[metric_type, "time ms"]
        test_time = test_summary.loc[metric_type, "time ms"]

        # Calculate metrics
        ratio_val = test_time / base_time if base_time != 0 else 0

        # percent_change: positive when test is faster (takes less time)
        pct_change = (
            (base_time - test_time) / base_time * 100 if base_time != 0 else 0
        )

        # Build row
        row = {
            "type": metric_type,
            f"{baseline_label}_time_ms": base_time,
            f"{test_label}_time_ms": test_time,
            "diff_time_ms": test_time - base_time,
            "percent_change": pct_change,
            "ratio": ratio_val,
            f"{baseline_label}_percent": baseline_summary.loc[metric_type, "percent"],
            f"{test_label}_percent": test_summary.loc[metric_type, "percent"],
            "diff_percent": (
                test_summary.loc[metric_type, "percent"]
                - baseline_summary.loc[metric_type, "percent"]
            ),
        }
        rows.append(row)

    summary_comparison = pd.DataFrame(rows)

    if verbose:
        print(f"  Created Summary_Comparison ({len(summary_comparison)} rows)")

    return summary_comparison

