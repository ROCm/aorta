#!/usr/bin/env python3
"""
AUTHOR: oyazdanb
TraceLens with GEMM Recognition Patches

This script applies GEMM recognition patches and runs TraceLens commands.

Usage:
    python tracelens_with_gemm_patch.py generate_perf_report [args...]
    python tracelens_with_gemm_patch.py generate_multi_rank_collective [args...]
    python tracelens_with_gemm_patch.py compare_perf_reports [args...]
"""

import argparse
import re
import sys


def apply_gemm_patches():
    """Apply all GEMM recognition patches to TraceLens."""

    print("Applying TraceLens GEMM recognition patches...")

    # Patch kernel_name_parser for enhanced ROCm GEMM recognition
    try:
        from TraceLens.PerfModel import kernel_name_parser

        def patched_is_rocm_gemm(kernel_name):
            """
            Enhanced ROCm GEMM pattern matching for Tensile kernels.
            Recognizes: Cijk_Alik_Bljk_... and variants with arbitrary prefixes.
            """
            pattern = r"^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$"
            return bool(re.match(pattern, kernel_name))

        def patched_parse_rocm_gemm(kernel_name):
            """Parse ROCm GEMM kernel details."""
            # Parse transpose flags
            trans_a, trans_b = None, None
            if "_Ailk_" in kernel_name:
                trans_a = False
            elif "_Alik_" in kernel_name:
                trans_a = True
            if "_Bljk_" in kernel_name:
                trans_b = False
            elif "_Bjlk_" in kernel_name:
                trans_b = True

            # Parse macro tile size (MT64x16x64)
            macro_tile_match = re.search(r"MT(\d+)x(\d+)x(\d+)", kernel_name)
            if macro_tile_match:
                mt_m = int(macro_tile_match.group(1))
                mt_n = int(macro_tile_match.group(2))
                depth_u = int(macro_tile_match.group(3))
            else:
                mt_m, mt_n, depth_u = None, None, None

            return {
                "transpose": (trans_a, trans_b),
                "mt_m": mt_m,
                "mt_n": mt_n,
                "depth_u": depth_u,
            }

        def patched_gemm_name_parser(kernel_name):
            """Enhanced GEMM name parser with better ROCm support."""
            if patched_is_rocm_gemm(kernel_name):
                return patched_parse_rocm_gemm(kernel_name)
            elif kernel_name_parser.is_cuda_gemm(kernel_name):
                return kernel_name_parser.parse_cuda_gemm(kernel_name)
            return None

        kernel_name_parser.is_rocm_gemm = patched_is_rocm_gemm
        kernel_name_parser.parse_rocm_gemm = patched_parse_rocm_gemm
        kernel_name_parser.gemm_name_parser = patched_gemm_name_parser

        print("  [OK] Patched kernel_name_parser (ROCm GEMM recognition)")
    except ImportError as e:
        print(f"  [WARN] Could not patch kernel_name_parser: {e}")

    # Patch Trace2Tree util for is_gemm_kernel function
    try:
        from TraceLens.Trace2Tree import util as trace_util

        def patched_is_gemm_kernel(kernel_event: dict) -> bool:
            """Enhanced GEMM kernel detection."""
            assert kernel_event["cat"] == "kernel"
            kernel_name = kernel_event["name"]

            # ROCm Tensile GEMM pattern: C[xyz]_A[xyz]_B[xyz]
            pattern = r"^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$"
            is_rocm_gemm = bool(re.match(pattern, kernel_name))

            # CUDA GEMM pattern
            is_cuda_gemm = kernel_name.startswith("nvjet") or "cublasLt" in kernel_name

            return is_rocm_gemm or is_cuda_gemm

        trace_util.is_gemm_kernel = patched_is_gemm_kernel
        print("  [OK] Patched Trace2Tree.util (is_gemm_kernel)")
    except ImportError as e:
        print(f"  [WARN] Could not patch Trace2Tree.util: {e}")

    # Patch TraceEventUtils to enhance GEMM keys
    try:
        from TraceLens import util as tracelens_util

        if hasattr(tracelens_util, "TraceEventUtils"):
            if hasattr(tracelens_util.TraceEventUtils, "JaxOpKeys"):
                original_gemm_keys = tracelens_util.TraceEventUtils.JaxOpKeys.GemmKeys
                enhanced_gemm_keys = [
                    "Cijk",
                    "gemm",
                    "nvjet",
                    "cublasLt",
                    "C[a-z]{3}_A[a-z]{3}_B[a-z]{3}",
                ]

                all_keys = list(set(original_gemm_keys + enhanced_gemm_keys))
                tracelens_util.TraceEventUtils.JaxOpKeys.GemmKeys = all_keys

                print("  [OK] Patched TraceEventUtils.JaxOpKeys (GEMM keys enhanced)")
    except (ImportError, AttributeError) as e:
        print(f"  [WARN] Could not patch TraceEventUtils: {e}")

    # Patch torch_op_mapping for better categorization
    try:
        from TraceLens.PerfModel import torch_op_mapping

        original_categorize = torch_op_mapping.categorize_torch_op

        def patched_categorize_torch_op(row):
            """Enhanced categorization with better GEMM detection."""
            result = original_categorize(row)

            # If result is 'other', check for GEMM patterns in kernel names
            if result == "other" and "kernel_details" in row and len(row["kernel_details"]) > 0:
                kernel_name = row["kernel_details"][0]["name"]
                pattern = r"^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$"
                if re.match(pattern, kernel_name):
                    return "GEMM"

            return result

        torch_op_mapping.categorize_torch_op = patched_categorize_torch_op
        print("  [OK] Patched torch_op_mapping (categorize_torch_op)")
    except ImportError as e:
        print(f"  [WARN] Could not patch torch_op_mapping: {e}")

    print("[OK] All GEMM patches applied successfully!\n")


def create_parser():
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="tracelens_with_gemm_patch.py",
        description="TraceLens with GEMM Recognition Patches - Apply GEMM recognition patches and run TraceLens commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tracelens_with_gemm_patch.py generate_perf_report --trace-dir /path/to/traces --output report.xlsx
    python tracelens_with_gemm_patch.py generate_multi_rank_collective --trace-dir /path/to/traces
    python tracelens_with_gemm_patch.py compare_perf_reports --baseline base.xlsx --test test.xlsx
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available TraceLens commands",
        help="Command to run (use '<command> --help' for command-specific help)",
    )

    # Subparser for generate_perf_report
    subparsers.add_parser(
        "generate_perf_report",
        help="Generate individual performance report from trace data",
        add_help=False,  # Let TraceLens handle its own --help
    )

    # Subparser for generate_multi_rank_collective
    subparsers.add_parser(
        "generate_multi_rank_collective",
        help="Generate multi-rank collective report from trace data",
        add_help=False,  # Let TraceLens handle its own --help
    )

    # Subparser for compare_perf_reports
    subparsers.add_parser(
        "compare_perf_reports",
        help="Compare two performance reports",
        add_help=False,  # Let TraceLens handle its own --help
    )

    return parser


def main():
    parser = create_parser()

    # Parse only the command, let TraceLens handle the rest
    args, remaining_args = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Apply patches before importing TraceLens reporting modules
    apply_gemm_patches()

    # Import TraceLens after patches are applied
    from TraceLens.Reporting.generate_perf_report_pytorch import main as generate_perf_report_main
    from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
        main as generate_multi_rank_collective_report_main,
    )
    from TraceLens.Reporting.compare_perf_reports_pytorch import main as compare_perf_reports_main

    # Update sys.argv so TraceLens sees only its arguments
    sys.argv = [sys.argv[0]] + remaining_args

    if args.command == "generate_perf_report":
        generate_perf_report_main()
    elif args.command == "generate_multi_rank_collective":
        generate_multi_rank_collective_report_main()
    elif args.command == "compare_perf_reports":
        compare_perf_reports_main()


if __name__ == "__main__":
    main()
