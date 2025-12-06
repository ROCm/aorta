#!/usr/bin/env python3
import pandas as pd
import argparse
from openpyxl.styles import Color
from openpyxl.formatting.rule import ColorScaleRule


def add_comparison_sheets(input_path, output_path):
    print(f"Loading: {input_path}")

    xl = pd.ExcelFile(input_path)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Copy all original sheets
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(input_path, sheet_name=sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Copied {sheet_name}")

        # Add comparison sheets
        all_combined = pd.read_excel(input_path, sheet_name='All_Ranks_Combined')

        # Comparison 1: Side-by-side by rank
        baseline_data = all_combined[all_combined['source'] == 'baseline']
        saleelk_data = all_combined[all_combined['source'] == 'saleelk']

        comparison_by_rank = pd.DataFrame()
        for rank in sorted(baseline_data['rank'].unique()):
            base_rank = baseline_data[baseline_data['rank'] == rank].set_index('type')
            sale_rank = saleelk_data[saleelk_data['rank'] == rank].set_index('type')

            for metric_type in base_rank.index:
                if metric_type in sale_rank.index:
                    base_time = base_rank.loc[metric_type, 'time ms']
                    sale_time = sale_rank.loc[metric_type, 'time ms']
                    ratio_val = sale_time / base_time if base_time != 0 else 0
                    # Percentage change: positive when saleelk is faster (takes less time)
                    pct_change = (base_time - sale_time) / base_time * 100 if base_time != 0 else 0

                    # Determine if better or worse
                    if pct_change > 1:
                        status = 'Better'
                    elif pct_change < -1:
                        status = 'Worse'
                    else:
                        status = 'Similar'

                    comparison_by_rank = pd.concat([comparison_by_rank, pd.DataFrame({
                        'rank': [rank],
                        'type': [metric_type],
                        'baseline_time_ms': [base_time],
                        'saleelk_time_ms': [sale_time],
                        'diff_time_ms': [sale_time - base_time],
                        'percent_change': [pct_change],
                        'status': [status],
                        'ratio': [ratio_val],
                        'baseline_percent': [base_rank.loc[metric_type, 'percent']],
                        'saleelk_percent': [sale_rank.loc[metric_type, 'percent']],
                        'diff_percent': [sale_rank.loc[metric_type, 'percent'] - base_rank.loc[metric_type, 'percent']]
                    })], ignore_index=True)

        comparison_by_rank.to_excel(writer, sheet_name='Comparison_By_Rank', index=False)
        print(f"  Added Comparison_By_Rank")

        # Comparison 2: Summary comparison
        summary = pd.read_excel(input_path, sheet_name='Summary')
        baseline_summary = summary[summary['source'] == 'baseline'].set_index('type')
        saleelk_summary = summary[summary['source'] == 'saleelk'].set_index('type')

        summary_comparison = pd.DataFrame()
        for metric_type in baseline_summary.index:
            if metric_type in saleelk_summary.index:
                base_time = baseline_summary.loc[metric_type, 'time ms']
                sale_time = saleelk_summary.loc[metric_type, 'time ms']
                ratio_val = sale_time / base_time if base_time != 0 else 0
                # Percentage change: positive when saleelk is faster (takes less time)
                pct_change = (base_time - sale_time) / base_time * 100 if base_time != 0 else 0

                summary_comparison = pd.concat([summary_comparison, pd.DataFrame({
                    'type': [metric_type],
                    'baseline_time_ms': [base_time],
                    'saleelk_time_ms': [sale_time],
                    'diff_time_ms': [sale_time - base_time],
                    'percent_change': [pct_change],
                    'ratio': [ratio_val],
                    'baseline_percent': [baseline_summary.loc[metric_type, 'percent']],
                    'saleelk_percent': [saleelk_summary.loc[metric_type, 'percent']],
                    'diff_percent': [saleelk_summary.loc[metric_type, 'percent'] - baseline_summary.loc[metric_type, 'percent']]
                })], ignore_index=True)

        summary_comparison.to_excel(writer, sheet_name='Summary_Comparison', index=False)
        print(f"  Added Summary_Comparison")

        # Add conditional formatting to percent_change columns
        print("\n  Applying conditional formatting...")

        # Create color scale: Red (negative) -> White (0) -> Green (positive)

        # Format Comparison_By_Rank
        ws_rank = writer.sheets['Comparison_By_Rank']
        # Find percent_change column
        for col_idx, col in enumerate(comparison_by_rank.columns, start=1):
            if col == 'percent_change':
                col_letter = chr(64 + col_idx)  # Convert to Excel column letter
                data_range = f'{col_letter}2:{col_letter}{len(comparison_by_rank)+1}'
                # Color scale: red (min) -> white (0) -> green (max)
                ws_rank.conditional_formatting.add(data_range,
                    ColorScaleRule(
                        start_type='min', start_color='F8696B',  # Red
                        mid_type='num', mid_value=0, mid_color='FFFFFF',  # White
                        end_type='max', end_color='63BE7B'  # Green
                    ))
                print(f"    Formatted Comparison_By_Rank column {col}")
                break

        # Format Summary_Comparison
        ws_summary = writer.sheets['Summary_Comparison']
        for col_idx, col in enumerate(summary_comparison.columns, start=1):
            if col == 'percent_change':
                col_letter = chr(64 + col_idx)
                data_range = f'{col_letter}2:{col_letter}{len(summary_comparison)+1}'
                # Color scale: red (min) -> white (0) -> green (max)
                ws_summary.conditional_formatting.add(data_range,
                    ColorScaleRule(
                        start_type='min', start_color='F8696B',  # Red
                        mid_type='num', mid_value=0, mid_color='FFFFFF',  # White
                        end_type='max', end_color='63BE7B'  # Green
                    ))
                print(f"    Formatted Summary_Comparison column {col}")
                break

    print(f"\nSaved: {output_path}")
    print("\nNew sheets:")
    print("  Comparison_By_Rank - Side-by-side comparison for each rank")
    print("  Summary_Comparison - Overall comparison")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Add comparison sheets to combined GPU timeline')
    parser.add_argument('--input', required=True, help='Input combined Excel file')
    parser.add_argument('--output', required=True, help='Output Excel file with comparison sheets')

    args = parser.parse_args()

    return add_comparison_sheets(args.input, args.output)


if __name__ == '__main__':
    exit(main())
