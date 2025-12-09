#!/usr/bin/env python3
"""
GPU Timeline Processing for Single Configuration.

Aggregates gpu_timeline data across all ranks in a tracelens analysis directory.

Usage:
    python process_gpu_timeline.py --reports-dir /path/to/individual_reports [--geo-mean]
"""

import pandas as pd
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
)


def process_gpu_timeline(reports_dir: str, use_geo_mean: bool = False) -> int:
    """
    Create mean/geometric mean aggregated GPU timeline across all ranks.

    Args:
        reports_dir: Path to directory containing perf_rank*.xlsx files
        use_geo_mean: If True, use geometric mean; otherwise arithmetic mean

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    reports_path = Path(reports_dir)

    if not reports_path.exists():
        print(f"Error: Directory not found: {reports_dir}")
        return 1

    print(f"Processing GPU timeline from: {reports_dir}")
    print(f"Aggregation: {get_aggregation_description(use_geo_mean)}")

    perf_files = sorted(reports_path.glob("perf_rank*.xlsx"))

    if not perf_files:
        print("Error: No perf_rank*.xlsx files found")
        return 1

    print(f"Found {len(perf_files)} rank files")

    rank_data = []
    for file_path in perf_files:
        rank_num = int(file_path.stem.replace("perf_rank", ""))
        df, success = read_gpu_timeline_from_excel(file_path, rank=rank_num)
        if success:
            rank_data.append(df)
            print(f"  Rank {rank_num}: OK")
        else:
            print(f"  Rank {rank_num}: Error")

    if not rank_data:
        print("Error: No valid data loaded")
        return 1

    combined = pd.concat(rank_data, ignore_index=True)
    aggregated = aggregate_gpu_timeline(rank_data, use_geo_mean)
    aggregated["num_ranks"] = len(perf_files)

    method_suffix = get_method_suffix(use_geo_mean)
    output_path = reports_path.parent / f"gpu_timeline_summary_{method_suffix}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        aggregated.to_excel(writer, sheet_name="Summary", index=False)

        combined_sorted = combined.sort_values(["rank", "type"])
        combined_sorted.to_excel(writer, sheet_name="All_Ranks_Combined", index=False)

        per_rank = combined.pivot_table(
            values="time ms", index="type", columns="rank", aggfunc="first"
        )
        per_rank.to_excel(writer, sheet_name="Per_Rank_Time_ms")

        per_rank_pct = combined.pivot_table(
            values="percent", index="type", columns="rank", aggfunc="first"
        )
        per_rank_pct.to_excel(writer, sheet_name="Per_Rank_Percent")

    print(f"\nSaved: {output_path}")
    print("\nSummary:")
    print(aggregated.to_string(index=False))

    return 0


def main():
    parser = argparse.ArgumentParser(description="Aggregate GPU timeline across ranks")
    parser.add_argument("--reports-dir", required=True, help="Path to individual_reports directory")
    parser.add_argument("--geo-mean", action="store_true", help="Use geometric mean")

    args = parser.parse_args()

    return process_gpu_timeline(args.reports_dir, args.geo_mean)


if __name__ == "__main__":
    exit(main())
