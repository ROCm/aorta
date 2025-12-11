#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path


def combine_collective_reports(baseline_path, test_path, output_path, baseline_label = "baseline", test_label="test"):
    """
    Combine two collective reports into a single Excel file by adding a source column to the data.
    """
    # Extract folder names from paths for labels
    #baseline_label = Path(baseline_path).parent.parent.name  # Get the config folder name
    #test_label = Path(test_path).parent.parent.name  # Get the config folder name

    print(f"Loading baseline ({baseline_label}): {baseline_path}")
    baseline_xl = pd.ExcelFile(baseline_path)

    print(f"Loading test ({test_label}): {test_path}")
    test_xl = pd.ExcelFile(test_path)

    print(f"\nBaseline sheets: {baseline_xl.sheet_names}")
    print(f"Test sheets: {test_xl.sheet_names}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in baseline_xl.sheet_names:
            if sheet_name not in test_xl.sheet_names:
                print(f"  Skip {sheet_name} - not in test file")
                continue

            baseline_df = pd.read_excel(baseline_path, sheet_name=sheet_name)
            test_df = pd.read_excel(test_path, sheet_name=sheet_name)

            baseline_df["source"] = baseline_label
            test_df["source"] = test_label

            combined = pd.concat([baseline_df, test_df], ignore_index=True)

            combined.to_excel(writer, sheet_name=sheet_name, index=False)
            print(
                f"  Combined {sheet_name}: {len(baseline_df)} + {len(test_df)} = {len(combined)} rows"
            )

    print(f"\nSaved: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Combine two collective reports")
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline collective_all_ranks.xlsx"
    )
    parser.add_argument(
        "--test", required=True, help="Path to test collective_all_ranks.xlsx"
    )
    parser.add_argument(
        "--baseline-label", default="baseline", help="Label for baseline data"
    )
    parser.add_argument(
        "--test-label", default="test", help="Label for test data"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for combined Excel file"
    )

    args = parser.parse_args()

    return combine_collective_reports(args.baseline, args.test, args.output, args.baseline_label, args.test_label)


if __name__ == "__main__":
    exit(main())
