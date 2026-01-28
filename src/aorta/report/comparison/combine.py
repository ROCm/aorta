"""Shared functionality to combine two Excel files."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def combine_excel_files(
    baseline_path: Path,
    test_path: Path,
    baseline_label: str,
    test_label: str,
    sheets_to_combine: Optional[List[str]] = None,
    filter_summary_only: bool = False,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Combine two Excel files by adding a 'source' column.

    Args:
        baseline_path: Path to baseline Excel file
        test_path: Path to test Excel file
        baseline_label: Label for baseline rows in 'source' column
        test_label: Label for test rows in 'source' column
        sheets_to_combine: Specific sheets to combine (None = all common sheets)
        filter_summary_only: If True, only keep sheets with 'summary' in name
        verbose: Print progress messages

    Returns:
        Dict mapping sheet_name to combined DataFrame

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no common sheets found
    """
    baseline_path = Path(baseline_path)
    test_path = Path(test_path)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    if verbose:
        print(f"Loading baseline ({baseline_label}): {baseline_path}")
        print(f"Loading test ({test_label}): {test_path}")

    # Load Excel files
    baseline_xl = pd.ExcelFile(baseline_path)
    test_xl = pd.ExcelFile(test_path)

    if verbose:
        print(f"\nBaseline sheets: {baseline_xl.sheet_names}")
        print(f"Test sheets: {test_xl.sheet_names}")

    # Determine sheets to combine
    if sheets_to_combine is not None:
        # Use specified sheets (must exist in both files)
        common_sheets = [
            s for s in sheets_to_combine
            if s in baseline_xl.sheet_names and s in test_xl.sheet_names
        ]
    else:
        # Find common sheets
        common_sheets = [
            s for s in baseline_xl.sheet_names
            if s in test_xl.sheet_names
        ]

    # Apply summary filter if requested
    if filter_summary_only:
        filtered_sheets = [s for s in common_sheets if "summary" in s.lower()]
        skipped_sheets = [s for s in common_sheets if "summary" not in s.lower()]

        if verbose and skipped_sheets:
            print(f"\nFiltering to summary sheets only...")
            print(f"  Skipped sheets (non-summary): {skipped_sheets}")

        common_sheets = filtered_sheets

    if not common_sheets:
        raise ValueError("No common sheets found between baseline and test files")

    if verbose:
        print(f"\nCombining sheets:")

    # Combine each sheet
    combined_data: Dict[str, pd.DataFrame] = {}

    for sheet_name in common_sheets:
        baseline_df = pd.read_excel(baseline_path, sheet_name=sheet_name)
        test_df = pd.read_excel(test_path, sheet_name=sheet_name)

        # Add source column
        baseline_df["source"] = baseline_label
        test_df["source"] = test_label

        # Concatenate
        combined = pd.concat([baseline_df, test_df], ignore_index=True)
        combined_data[sheet_name] = combined

        if verbose:
            print(f"  {sheet_name}: {len(baseline_df)} + {len(test_df)} = {len(combined)} rows")

    return combined_data


def extract_label_from_path(file_path: Path, default: str = "unknown") -> str:
    """
    Extract label from file path using grandparent directory name.

    Args:
        file_path: Path to the Excel file
        default: Default label if extraction fails

    Returns:
        Extracted label or default

    Examples:
        /path/to/56cu_256threads/tracelens_analysis/gpu_timeline.xlsx
        → "56cu_256threads"

        /path/to/run1/tracelens_analysis/collective_reports/collective.xlsx
        → "tracelens_analysis" (grandparent of file)
    """
    try:
        file_path = Path(file_path)
        # Go up to grandparent (skip filename and parent directory)
        grandparent = file_path.parent.parent.name
        if grandparent and grandparent not in [".", "..", ""]:
            return grandparent
    except Exception:
        pass
    return default
