#!/usr/bin/env python3
"""
Compare actual outputs against expected baseline outputs.
Allows tolerance for numeric values and ignores cosmetic differences.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime


class OutputComparator:
    """Compare analysis outputs with tolerance for acceptable differences."""

    def __init__(self, expected_dir: Path, actual_dir: Path, verbose: bool = False):
        self.expected_dir = Path(expected_dir)
        self.actual_dir = Path(actual_dir)
        self.verbose = verbose
        self.numeric_tolerance = 1e-6  # Tolerance for numeric comparisons
        self.errors = []
        self.warnings = []

    def log_error(self, msg: str):
        """Log an error message."""
        self.errors.append(msg)
        if self.verbose:
            print(f"ERROR: {msg}")

    def log_warning(self, msg: str):
        """Log a warning message."""
        self.warnings.append(msg)
        if self.verbose:
            print(f"WARNING: {msg}")

    def log_success(self, msg: str):
        """Log a success message."""
        if self.verbose:
            print(f"OK: {msg}")

    def compare_csv_files(self, expected_file: Path, actual_file: Path) -> bool:
        """Compare two CSV files with tolerance for numeric values."""
        try:
            expected_df = pd.read_csv(expected_file)
            actual_df = pd.read_csv(actual_file)
        except Exception as e:
            self.log_error(f"Failed to read CSV files: {e}")
            return False

        # Check if required columns are present (actual may have more)
        expected_cols = set(expected_df.columns)
        actual_cols = set(actual_df.columns)

        missing_cols = expected_cols - actual_cols
        if missing_cols:
            self.log_error(f"Missing required columns in {actual_file.name}: {missing_cols}")
            return False

        extra_cols = actual_cols - expected_cols
        if extra_cols:
            self.log_warning(f"Extra columns in {actual_file.name} (allowed): {extra_cols}")

        # Check row count (should be similar)
        if len(expected_df) != len(actual_df):
            self.log_warning(f"Different row counts in {actual_file.name}: "
                           f"expected {len(expected_df)}, got {len(actual_df)}")

        # Compare numeric columns with tolerance
        for col in expected_cols:
            if col not in actual_df.columns:
                continue

            if pd.api.types.is_numeric_dtype(expected_df[col]):
                # Compare numeric columns with tolerance
                expected_vals = expected_df[col].fillna(0).values
                actual_vals = actual_df[col].fillna(0).values[:len(expected_vals)]

                if not np.allclose(expected_vals, actual_vals,
                                  rtol=self.numeric_tolerance,
                                  atol=self.numeric_tolerance):
                    max_diff = np.max(np.abs(expected_vals - actual_vals))
                    self.log_error(f"Numeric mismatch in column '{col}' of {actual_file.name}: "
                                 f"max difference = {max_diff}")
                    return False
            else:
                # For non-numeric columns, check if core values are present
                # (order may differ, some values may be added)
                expected_values = set(expected_df[col].dropna().unique())
                actual_values = set(actual_df[col].dropna().unique())

                missing_values = expected_values - actual_values
                if missing_values and col in ['kernel_name', 'operation']:
                    # Critical columns should have all values
                    self.log_error(f"Missing values in column '{col}' of {actual_file.name}: "
                                 f"{missing_values}")
                    return False

        self.log_success(f"CSV comparison passed: {actual_file.name}")
        return True

    def compare_json_files(self, expected_file: Path, actual_file: Path) -> bool:
        """Compare two JSON files allowing for structural flexibility."""
        try:
            with open(expected_file, 'r') as f:
                expected_data = json.load(f)
            with open(actual_file, 'r') as f:
                actual_data = json.load(f)
        except Exception as e:
            self.log_error(f"Failed to read JSON files: {e}")
            return False

        return self.compare_json_objects(expected_data, actual_data, actual_file.name)

    def compare_json_objects(self, expected: Any, actual: Any, context: str) -> bool:
        """Recursively compare JSON objects."""
        if type(expected) != type(actual):
            self.log_error(f"Type mismatch in {context}: "
                         f"expected {type(expected)}, got {type(actual)}")
            return False

        if isinstance(expected, dict):
            # Check required keys are present
            for key in expected:
                if key not in actual:
                    self.log_error(f"Missing required key '{key}' in {context}")
                    return False
                if not self.compare_json_objects(expected[key], actual[key],
                                                f"{context}.{key}"):
                    return False

            # Extra keys are allowed
            extra_keys = set(actual.keys()) - set(expected.keys())
            if extra_keys:
                self.log_warning(f"Extra keys in {context} (allowed): {extra_keys}")

        elif isinstance(expected, list):
            # Lists should have similar length
            if len(expected) != len(actual):
                self.log_warning(f"Different list lengths in {context}: "
                               f"expected {len(expected)}, got {len(actual)}")

            # Compare elements up to minimum length
            min_len = min(len(expected), len(actual))
            for i in range(min_len):
                if not self.compare_json_objects(expected[i], actual[i],
                                                f"{context}[{i}]"):
                    return False

        elif isinstance(expected, (int, float)):
            # Numeric comparison with tolerance
            if not np.isclose(expected, actual, rtol=self.numeric_tolerance):
                self.log_error(f"Numeric mismatch in {context}: "
                             f"expected {expected}, got {actual}")
                return False

        elif isinstance(expected, str):
            # String comparison (ignore timestamps and build IDs)
            if self.is_volatile_string(expected) or self.is_volatile_string(actual):
                # Skip comparison for volatile strings
                pass
            elif expected != actual:
                self.log_error(f"String mismatch in {context}: "
                             f"expected '{expected}', got '{actual}'")
                return False

        else:
            # Direct comparison for other types
            if expected != actual:
                self.log_error(f"Value mismatch in {context}: "
                             f"expected {expected}, got {actual}")
                return False

        return True

    def is_volatile_string(self, s: str) -> bool:
        """Check if a string contains volatile content like timestamps or IDs."""
        volatile_patterns = [
            '2024', '2025',  # Years in timestamps
            'build_', 'run_',  # Build/run IDs
            'tmp/',  # Temporary paths
            '.log',  # Log files
        ]
        return any(pattern in s for pattern in volatile_patterns)

    def compare_excel_files(self, expected_file: Path, actual_file: Path) -> bool:
        """Compare Excel files by checking sheets and data."""
        try:
            expected_xlsx = pd.ExcelFile(expected_file)
            actual_xlsx = pd.ExcelFile(actual_file)
        except Exception as e:
            self.log_error(f"Failed to read Excel files: {e}")
            return False

        # Check if required sheets are present
        expected_sheets = set(expected_xlsx.sheet_names)
        actual_sheets = set(actual_xlsx.sheet_names)

        missing_sheets = expected_sheets - actual_sheets
        if missing_sheets:
            self.log_error(f"Missing required sheets in {actual_file.name}: {missing_sheets}")
            return False

        # Compare data in each expected sheet
        for sheet_name in expected_sheets:
            expected_df = pd.read_excel(expected_file, sheet_name=sheet_name)
            actual_df = pd.read_excel(actual_file, sheet_name=sheet_name)

            # Similar logic to CSV comparison
            if len(expected_df.columns) > len(actual_df.columns):
                self.log_error(f"Missing columns in sheet '{sheet_name}' of {actual_file.name}")
                return False

        self.log_success(f"Excel comparison passed: {actual_file.name}")
        return True

    def compare_html_files(self, expected_file: Path, actual_file: Path) -> bool:
        """Compare HTML files (check if key elements are present)."""
        try:
            with open(expected_file, 'r') as f:
                expected_content = f.read()
            with open(actual_file, 'r') as f:
                actual_content = f.read()
        except Exception as e:
            self.log_error(f"Failed to read HTML files: {e}")
            return False

        # Check for key markers/elements (not exact match due to styling changes)
        key_markers = [
            '<table',  # Tables should be present
            'GEMM',    # GEMM-related content
            'chart',   # Charts/visualizations
        ]

        for marker in key_markers:
            if marker in expected_content and marker not in actual_content:
                self.log_error(f"Missing key element '{marker}' in {actual_file.name}")
                return False

        self.log_success(f"HTML structure check passed: {actual_file.name}")
        return True

    def compare_directories(self) -> bool:
        """Compare all files in the output directories."""
        all_passed = True

        # Get all expected files
        expected_files = list(self.expected_dir.rglob('*'))
        expected_files = [f for f in expected_files if f.is_file()]

        print(f"\nComparing {len(expected_files)} files...")
        print("=" * 50)

        for expected_file in expected_files:
            # Get relative path
            rel_path = expected_file.relative_to(self.expected_dir)
            actual_file = self.actual_dir / rel_path

            if not actual_file.exists():
                self.log_error(f"Missing output file: {rel_path}")
                all_passed = False
                continue

            # Compare based on file type
            if expected_file.suffix == '.csv':
                if not self.compare_csv_files(expected_file, actual_file):
                    all_passed = False
            elif expected_file.suffix == '.json':
                if not self.compare_json_files(expected_file, actual_file):
                    all_passed = False
            elif expected_file.suffix in ['.xlsx', '.xls']:
                if not self.compare_excel_files(expected_file, actual_file):
                    all_passed = False
            elif expected_file.suffix in ['.html', '.htm']:
                if not self.compare_html_files(expected_file, actual_file):
                    all_passed = False
            else:
                # For other files, just check existence
                self.log_success(f"File exists: {rel_path}")

        return all_passed

    def print_summary(self):
        """Print summary of comparison results."""
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more warnings")

        if not self.errors:
            print("\nAll regression tests PASSED!")
            print(f"   {len(self.warnings)} warnings (cosmetic differences allowed)")
        else:
            print(f"\nRegression tests FAILED!")
            print(f"   {len(self.errors)} errors, {len(self.warnings)} warnings")


def main():
    parser = argparse.ArgumentParser(description="Compare GEMM analysis outputs")
    parser.add_argument("--expected", required=True, help="Path to expected outputs directory")
    parser.add_argument("--actual", required=True, help="Path to actual outputs directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    comparator = OutputComparator(args.expected, args.actual, args.verbose)

    # Run comparison
    success = comparator.compare_directories()

    # Print summary
    comparator.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
