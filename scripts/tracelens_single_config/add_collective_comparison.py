#!/usr/bin/env python3
import pandas as pd
import argparse
from openpyxl.styles import Color
from openpyxl.formatting.rule import ColorScaleRule


def add_collective_comparison_sheets(input_path, output_path):
    print(f"Loading: {input_path}")

    xl = pd.ExcelFile(input_path)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Copy only summary sheets
        for sheet_name in xl.sheet_names:
            # Only keep sheets with 'summary' in the name
            if 'summary' not in sheet_name.lower():
                print(f"  Skip {sheet_name} (keeping only summary sheets)")
                continue
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Copied {sheet_name}")

        # Process summary sheets for comparison
        for sheet_name in ['nccl_summary_implicit_sync', 'nccl_summary_long']:
            if sheet_name not in xl.sheet_names:
                continue

            df = pd.read_excel(input_path, sheet_name=sheet_name)

            # Separate baseline and saleelk
            baseline_df = df[df['source'] == 'baseline'].copy()
            saleelk_df = df[df['source'] == 'saleelk'].copy()

            if len(baseline_df) == 0 or len(saleelk_df) == 0:
                print(f"  Skip {sheet_name} - missing data")
                continue

            # Create comparison dataframe
            comparison = pd.DataFrame()

            # Identify key columns for grouping
            group_cols = ['Collective name', 'dtype', 'In msg nelems']
            if not all(col in baseline_df.columns for col in group_cols):
                group_cols = ['Collective name']

            # Group and compare
            baseline_grouped = baseline_df.groupby(group_cols, as_index=False)
            saleelk_grouped = saleelk_df.groupby(group_cols, as_index=False)

            for name, base_group in baseline_grouped:
                # Find matching saleelk group
                if isinstance(name, tuple):
                    mask = pd.Series([True] * len(saleelk_df), index=saleelk_df.index)
                    for col, val in zip(group_cols, name):
                        mask = mask & (saleelk_df[col] == val)
                else:
                    mask = (saleelk_df[group_cols[0]] == name)

                sale_group = saleelk_df.loc[mask]

                if len(sale_group) == 0:
                    continue

                # Create comparison row
                comp_row = {}

                # Copy grouping columns
                if isinstance(name, tuple):
                    for col, val in zip(group_cols, name):
                        comp_row[col] = val
                else:
                    comp_row[group_cols[0]] = name

                # Compare numeric columns
                numeric_cols = ['comm_latency_mean', 'algo bw (GB/s)_mean', 'bus bw (GB/s)_mean',
                               'Total comm latency (ms)', 'count']

                for col in numeric_cols:
                    if col not in base_group.columns or col not in sale_group.columns:
                        continue

                    base_val = base_group[col].values[0]
                    sale_val = sale_group[col].values[0]

                    comp_row[f'baseline_{col}'] = base_val
                    comp_row[f'saleelk_{col}'] = sale_val
                    comp_row[f'diff_{col}'] = sale_val - base_val

                    # For latency/time: positive percent_change means faster (less time)
                    # For bandwidth: positive percent_change means better (more bandwidth)
                    if 'latency' in col.lower() or 'time' in col.lower():
                        # Lower is better - positive when saleelk is faster
                        pct_change = (base_val - sale_val) / base_val * 100 if base_val != 0 else 0
                        comp_row[f'percent_change_{col}'] = pct_change
                    elif 'bw' in col.lower() or 'bandwidth' in col.lower():
                        # Higher is better - positive when saleelk is better
                        pct_change = (sale_val - base_val) / base_val * 100 if base_val != 0 else 0
                        comp_row[f'percent_change_{col}'] = pct_change

                    comp_row[f'ratio_{col}'] = sale_val / base_val if base_val != 0 else 0

                comparison = pd.concat([comparison, pd.DataFrame([comp_row])], ignore_index=True)

            # Write comparison sheet (shorten name to fit Excel's 31 char limit)
            # Replace 'nccl_summary_' with 'nccl_' and '_comparison' with '_cmp'
            comparison_sheet_name = sheet_name.replace('nccl_summary_', 'nccl_') + '_cmp'
            comparison.to_excel(writer, sheet_name=comparison_sheet_name, index=False)
            print(f"  Added {comparison_sheet_name}")

            # Add conditional formatting to percent_change columns
            print(f"    Applying conditional formatting to {comparison_sheet_name}...")

            ws = writer.sheets[comparison_sheet_name]

            # Format all percent_change columns with color scale
            for col_idx, col in enumerate(comparison.columns, start=1):
                if 'percent_change' in col:
                    # Convert column index to Excel letter (A, B, C, ...)
                    if col_idx <= 26:
                        col_letter = chr(64 + col_idx)
                    else:
                        col_letter = chr(64 + (col_idx // 26)) + chr(64 + (col_idx % 26))

                    data_range = f'{col_letter}2:{col_letter}{len(comparison)+1}'

                    # Color scale: red (min/negative) -> white (0) -> green (max/positive)
                    ws.conditional_formatting.add(data_range,
                        ColorScaleRule(
                            start_type='min', start_color='F8696B',  # Red
                            mid_type='num', mid_value=0, mid_color='FFFFFF',  # White
                            end_type='max', end_color='63BE7B'  # Green
                        ))

                    print(f"      Formatted {col}")

    print(f"\nSaved: {output_path}")
    print("\nNew comparison sheets added")
    print("percent_change interpretation:")
    print("  For latency/time: Positive = faster (less time)")
    print("  For bandwidth: Positive = better (more bandwidth)")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Add comparison sheets to combined collective reports')
    parser.add_argument('--input', required=True, help='Input combined collective Excel file')
    parser.add_argument('--output', required=True, help='Output Excel file with comparison sheets')

    args = parser.parse_args()

    return add_collective_comparison_sheets(args.input, args.output)


if __name__ == '__main__':
    exit(main())
