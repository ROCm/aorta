#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def geometric_mean(values):
    values = np.array(values)
    values = np.where(values == 0, 1e-10, values)
    return np.exp(np.mean(np.log(values)))


def process_gpu_timeline(reports_dir, use_geo_mean=False):
    reports_path = Path(reports_dir)

    if not reports_path.exists():
        print(f"Error: Directory not found: {reports_dir}")
        return 1

    print(f"Processing GPU timeline from: {reports_dir}")
    print(f"Aggregation: {'Geometric Mean' if use_geo_mean else 'Arithmetic Mean'}")

    perf_files = sorted(reports_path.glob('perf_rank*.xlsx'))

    if not perf_files:
        print("Error: No perf_rank*.xlsx files found")
        return 1

    print(f"Found {len(perf_files)} rank files")

    rank_data = []
    for file_path in perf_files:
        rank_num = int(file_path.stem.replace('perf_rank', ''))
        try:
            df = pd.read_excel(file_path, sheet_name='gpu_timeline')
            df['rank'] = rank_num
            rank_data.append(df)
            print(f"  Rank {rank_num}: OK")
        except Exception as e:
            print(f"  Rank {rank_num}: Error - {e}")

    if not rank_data:
        print("Error: No valid data loaded")
        return 1

    combined = pd.concat(rank_data, ignore_index=True)

    agg_func = geometric_mean if use_geo_mean else 'mean'
    aggregated = combined.groupby('type').agg({
        'time ms': agg_func,
        'percent': agg_func
    }).reset_index()

    aggregated['num_ranks'] = len(perf_files)

    method_suffix = 'geomean' if use_geo_mean else 'mean'
    output_path = reports_path.parent / f'gpu_timeline_summary_{method_suffix}.xlsx'

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        aggregated.to_excel(writer, sheet_name='Summary', index=False)

        combined_sorted = combined.sort_values(['rank', 'type'])
        combined_sorted.to_excel(writer, sheet_name='All_Ranks_Combined', index=False)

        per_rank = combined.pivot_table(
            values='time ms',
            index='type',
            columns='rank',
            aggfunc='first'
        )
        per_rank.to_excel(writer, sheet_name='Per_Rank_Time_ms')

        per_rank_pct = combined.pivot_table(
            values='percent',
            index='type',
            columns='rank',
            aggfunc='first'
        )
        per_rank_pct.to_excel(writer, sheet_name='Per_Rank_Percent')

    print(f"\nSaved: {output_path}")
    print("\nSummary:")
    print(aggregated.to_string(index=False))

    return 0


def main():
    parser = argparse.ArgumentParser(description='Aggregate GPU timeline across ranks')
    parser.add_argument('--reports-dir', required=True, help='Path to individual_reports directory')
    parser.add_argument('--geo-mean', action='store_true', help='Use geometric mean')

    args = parser.parse_args()

    return process_gpu_timeline(args.reports_dir, args.geo_mean)


if __name__ == '__main__':
    exit(main())
