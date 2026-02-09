"""GEMM variance data loading and statistics."""

import csv
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def read_gemm_csv_data(csv_path: Path) -> Dict[str, Any]:
    """
    Read GEMM variance CSV and organize by dimensions.

    Returns:
        {
            "threads": {256: [values], 512: [values]},
            "channels": {28: [values], 42: [values], ...},
            "ranks": {0: [values], 1: [values], ...},
            "all": [list of row dicts],
        }
    """
    data: Dict[str, Any] = {
        "threads": defaultdict(list),
        "channels": defaultdict(list),
        "ranks": defaultdict(list),
        "all": [],
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                threads = int(row["threads"])
                channel = int(row["channel"])
                rank = int(row["rank"])
                time_diff = float(row["time_diff_us"])

                data["threads"][threads].append(time_diff)
                data["channels"][channel].append(time_diff)
                data["ranks"][rank].append(time_diff)
                data["all"].append({
                    "threads": threads,
                    "channel": channel,
                    "rank": rank,
                    "time_diff": time_diff,
                    "kernel_name": row.get("kernel_name", ""),
                })
            except (ValueError, KeyError):
                continue

    return data


def _calculate_median(values: List[float]) -> float:
    """Calculate median of a list of values."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def print_gemm_statistics(
    data: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Print and return summary statistics."""
    stats: Dict[str, Any] = {}

    if verbose:
        print("\n" + "=" * 70)
        print("VARIANCE DISTRIBUTION STATISTICS")
        print("=" * 70)

    for dimension, label_fmt in [
        ("threads", "{} threads"),
        ("channels", "{}ch"),
        ("ranks", "Rank {}"),
    ]:
        stats[dimension] = {}
        if verbose:
            print(f"\nBy {dimension.title()}:")

        for key in sorted(data[dimension].keys()):
            values = data[dimension][key]
            if not values:
                continue

            mean_val = sum(values) / len(values)
            median_val = _calculate_median(values)

            stats[dimension][key] = {
                "mean": mean_val,
                "median": median_val,
                "max": max(values),
                "count": len(values),
            }

            if verbose:
                label = label_fmt.format(key)
                print(
                    f"  {label}: mean={mean_val:.2f}us, median={median_val:.2f}us, "
                    f"max={max(values):.2f}us, n={len(values)}"
                )

    if verbose:
        print("=" * 70 + "\n")

    return stats
