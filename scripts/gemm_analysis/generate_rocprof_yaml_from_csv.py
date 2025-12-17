#!/usr/bin/env python3
"""
Generate a rocprof yaml configuration file from top5_gemm_kernels_time_variance.csv
This will create a yaml file that profiles only the kernels mentioned in the CSV.
"""

import argparse
import csv
import re
from pathlib import Path


def truncate_kernel_name(kernel_name):
    """
    Extract the shortest meaningful kernel name using delimiters.

    Logic:
    1. Remove 'void ' prefix if present
    2. Remove template parameters first (split by '<' and take first part)
    3. Remove function parameters (split by '(' and take first part)
    4. Split by '::' and take the last part (actual kernel/function name)

    Examples:
    - 'void at::native::elementwise_kernel_manual_unroll<...>' -> 'elementwise_kernel_manual_unroll'
    - 'Cijk_Ailk_Bjlk_BBS_BH_...' -> 'Cijk_Ailk_Bjlk_BBS_BH_...' (unchanged)
    """
    name = kernel_name.strip()

    # Remove 'void ' prefix if present (common for function kernels)
    if name.startswith('void '):
        name = name[5:]  # len('void ') = 5

    # Remove template parameters first (everything after '<')
    # This must be done before splitting by :: to avoid issues with nested templates
    if '<' in name:
        name = name.split('<')[0]

    # Remove function parameters (everything after '(')
    if '(' in name:
        name = name.split('(')[0]

    # Now split by '::' and take the last part (the actual function/kernel name)
    if '::' in name:
        name = name.split('::')[-1]

    return name.strip()


def escape_regex_special_chars(kernel_name):
    """Escape special regex characters in kernel name."""
    # Characters that have special meaning in regex
    special_chars = r'\.[]{}()*+?^$|'
    escaped = kernel_name
    for char in special_chars:
        escaped = escaped.replace(char, '\\' + char)
    return escaped


def generate_rocprof_yaml(csv_path, output_yaml_path):
    """Generate rocprof yaml from CSV file."""

    # Read CSV and extract unique kernel names
    kernel_names = set()
    kernel_mapping = {}  # Track original -> processed names

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_name_orig = row['kernel_name'].strip()
            if kernel_name_orig:
                # Apply generic truncation logic using delimiters
                kernel_name = truncate_kernel_name(kernel_name_orig)

                # Track if truncation occurred
                if kernel_name != kernel_name_orig:
                    kernel_mapping[kernel_name] = kernel_name_orig

                kernel_names.add(kernel_name)

    print(f"Found {len(kernel_names)} unique kernel patterns in CSV")
    if kernel_mapping:
        print(f"Truncated {len(kernel_mapping)} kernel name(s) using delimiter-based extraction")

    # Escape special regex characters and create regex pattern
    # No anchors - allow substring matching
    escaped_kernels = [escape_regex_special_chars(name) for name in sorted(kernel_names)]

    # Create regex pattern - join with OR operator
    # For rocprof, we want to match any of these kernel names
    kernel_regex = '(' + '|'.join(escaped_kernels) + ')'

    # Generate YAML content
    # Use single quotes for the regex to avoid YAML escape sequence issues
    yaml_content = f"""# Auto-generated rocprof configuration for top GEMM kernels
# Generated from: {csv_path}
# Number of unique kernel patterns: {len(kernel_names)}
#
# This configuration profiles only the specific kernels identified in the variance analysis.
# Kernel names are matched as substrings (no anchors) to allow flexible matching.
# Note: Kernel names are extracted using delimiters (::, <, space) for brevity.
# Use this with rocprofv3 for targeted profiling of high-variance GEMM kernels.

jobs:
  - kernel_include_regex: '{kernel_regex}'
    kernel_trace: true
    output_format: [json, csv]
    pmc:
      - SQ_BUSY_CU_CYCLES     # CU utilization (most important)
      - SQ_WAVES              # Active waves (occupancy indicator)
      - SQ_WAVE_CYCLES        # Total wave cycles
      - SQ_INSTS_MFMA         # Matrix ops (GEMM-specific)
"""

    # Write YAML file
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Generated rocprof yaml: {output_yaml_path}")
    print(f"Kernel regex length: {len(kernel_regex)} characters")

    # Also save a text file with the list of kernels for reference
    kernel_list_path = str(output_yaml_path).replace('.yaml', '_kernel_list.txt')
    with open(kernel_list_path, 'w') as f:
        f.write("List of kernel patterns included in rocprof configuration:\n")
        f.write("=" * 80 + "\n")
        f.write("Note: Kernel names are matched as substrings (no regex anchors).\n")
        f.write("Note: Names extracted using delimiters (::, <, (), 'void' prefix).\n")
        if kernel_mapping:
            f.write(f"Note: {len(kernel_mapping)} kernel name(s) were truncated for brevity.\n")
        f.write("\n")
        for i, kernel in enumerate(sorted(kernel_names), 1):
            f.write(f"{i}. {kernel}\n")
            if kernel in kernel_mapping:
                # Show first 100 chars of original, indicate if longer
                orig = kernel_mapping[kernel]
                if len(orig) > 100:
                    f.write(f"   (Original: {orig[:100]}...)\n")
                else:
                    f.write(f"   (Original: {orig})\n")

    print(f"Saved kernel list to: {kernel_list_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate rocprof yaml configuration from top5_gemm_kernels_time_variance.csv"
    )
    parser.add_argument(
        '--input-csv',
        type=str,
        required=True,
        help='Path to top5_gemm_kernels_time_variance.csv file'
    )
    parser.add_argument(
        '--output-yaml',
        type=str,
        default=None,
        help='Output yaml file path (default: rocprof_top5_kernels.yaml in same directory as CSV)'
    )

    args = parser.parse_args()

    csv_path = Path(args.input_csv)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Determine output path
    if args.output_yaml:
        output_yaml_path = Path(args.output_yaml)
    else:
        output_yaml_path = csv_path.parent / 'rocprof_top5_kernels.yaml'

    # Generate the yaml
    generate_rocprof_yaml(csv_path, output_yaml_path)

    return 0


if __name__ == '__main__':
    exit(main())
