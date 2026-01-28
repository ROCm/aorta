"""
GPU Timeline Processing - Single Configuration Mode.

Processes GPU timeline data from TraceLens individual reports for a single
configuration (no thread/channel variations). Aggregates across ranks.

Source: scripts/tracelens_single_config/process_gpu_timeline.py
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def geometric_mean(values: np.ndarray) -> float:
    """
    Calculate geometric mean, handling zeros.

    Args:
        values: Array of values

    Returns:
        Geometric mean value
    """
    values = np.array(values)
    values = np.where(values == 0, 1e-10, values)
    return float(np.exp(np.mean(np.log(values))))


def process_single_config(
    reports_dir: Path,
    use_geo_mean: bool = False,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Process GPU timeline from single config individual reports.

    Reads gpu_timeline sheet from each perf_rank*.xlsx file and aggregates
    across all ranks using mean or geometric mean.

    Args:
        reports_dir: Path to individual_reports directory containing perf_rank*.xlsx
        use_geo_mean: Use geometric mean instead of arithmetic mean
        output_path: Custom output path (default: parent/gpu_timeline_summary_{method}.xlsx)
        verbose: Print verbose output

    Returns:
        Path to output Excel file or None if processing failed
    """
    reports_path = Path(reports_dir)

    if not reports_path.exists():
        raise FileNotFoundError(f"Directory not found: {reports_dir}")

    agg_method = "Geometric Mean" if use_geo_mean else "Arithmetic Mean"
    print(f"Processing GPU timeline from: {reports_dir}")
    print(f"Aggregation: {agg_method}")

    # Find performance files
    perf_files = sorted(reports_path.glob("perf_rank*.xlsx"))

    if not perf_files:
        print("Error: No perf_rank*.xlsx files found")
        return None

    print(f"Found {len(perf_files)} rank files")

    # Read data from each rank
    rank_data = []
    for file_path in perf_files:
        rank_num = int(file_path.stem.replace("perf_rank", ""))
        try:
            df = pd.read_excel(file_path, sheet_name="gpu_timeline")
            df["rank"] = rank_num
            rank_data.append(df)
            if verbose:
                print(f"  Rank {rank_num}: OK")
        except Exception as e:
            print(f"  Rank {rank_num}: Error - {e}")

    if not rank_data:
        print("Error: No valid data loaded")
        return None

    # Combine all rank data
    combined = pd.concat(rank_data, ignore_index=True)

    # Aggregate across ranks
    agg_func = geometric_mean if use_geo_mean else "mean"
    aggregated = (
        combined.groupby("type")
        .agg({"time ms": agg_func, "percent": agg_func})
        .reset_index()
    )

    aggregated["num_ranks"] = len(perf_files)

    # Determine output path
    method_suffix = "geomean" if use_geo_mean else "mean"
    if output_path is None:
        output_path = reports_path.parent / f"gpu_timeline_summary_{method_suffix}.xlsx"
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Summary sheet - aggregated metrics
        aggregated.to_excel(writer, sheet_name="Summary", index=False)

        # All ranks combined - raw data with rank column
        combined_sorted = combined.sort_values(["rank", "type"])
        combined_sorted.to_excel(writer, sheet_name="All_Ranks_Combined", index=False)

        # Per-rank pivot - time values
        per_rank_time = combined.pivot_table(
            values="time ms", index="type", columns="rank", aggfunc="first"
        )
        per_rank_time.to_excel(writer, sheet_name="Per_Rank_Time_ms")

        # Per-rank pivot - percentages
        per_rank_pct = combined.pivot_table(
            values="percent", index="type", columns="rank", aggfunc="first"
        )
        per_rank_pct.to_excel(writer, sheet_name="Per_Rank_Percent")

    print(f"\nSaved: {output_path}")
    print("\nSheets created:")
    print("  1. Summary - Aggregated metrics across ranks")
    print("  2. All_Ranks_Combined - Raw data from all ranks")
    print("  3. Per_Rank_Time_ms - Pivot: type × rank (time)")
    print("  4. Per_Rank_Percent - Pivot: type × rank (percent)")

    # Print summary
    print("\nSummary:")
    print(aggregated.to_string(index=False))

    return output_path
