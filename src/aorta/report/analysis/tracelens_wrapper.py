"""
TraceLens wrapper with GEMM recognition patches.

Applies patches to TraceLens for better ROCm Tensile kernel recognition
and provides a clean Python API for TraceLens commands.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any


class TraceLensWrapper:
    """GEMM-patched TraceLens wrapper."""

    _patches_applied = False

    def __init__(self, verbose: bool = False):
        """Initialize wrapper and apply GEMM patches."""
        self.verbose = verbose
        if not TraceLensWrapper._patches_applied:
            self._apply_gemm_patches()
            TraceLensWrapper._patches_applied = True

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _apply_gemm_patches(self) -> None:
        """Apply all GEMM recognition patches to TraceLens."""
        self._log("Applying TraceLens GEMM recognition patches...")

        # Patch kernel_name_parser for enhanced ROCm GEMM recognition
        try:
            from TraceLens.PerfModel import kernel_name_parser

            def patched_is_rocm_gemm(kernel_name):
                """Enhanced ROCm GEMM pattern matching for Tensile kernels."""
                pattern = r"^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$"
                return bool(re.match(pattern, kernel_name))

            def patched_parse_rocm_gemm(kernel_name):
                """Parse ROCm GEMM kernel details."""
                trans_a, trans_b = None, None
                if "_Ailk_" in kernel_name:
                    trans_a = False
                elif "_Alik_" in kernel_name:
                    trans_a = True
                if "_Bljk_" in kernel_name:
                    trans_b = False
                elif "_Bjlk_" in kernel_name:
                    trans_b = True

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

            self._log("  [OK] Patched kernel_name_parser (ROCm GEMM recognition)")
        except ImportError as e:
            self._log(f"  [WARN] Could not patch kernel_name_parser: {e}")

        # Patch Trace2Tree util for is_gemm_kernel function
        try:
            from TraceLens.Trace2Tree import util as trace_util

            def patched_is_gemm_kernel(kernel_event: dict) -> bool:
                """Enhanced GEMM kernel detection."""
                assert kernel_event["cat"] == "kernel"
                kernel_name = kernel_event["name"]

                pattern = r"^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$"
                is_rocm_gemm = bool(re.match(pattern, kernel_name))
                is_cuda_gemm = kernel_name.startswith("nvjet") or "cublasLt" in kernel_name

                return is_rocm_gemm or is_cuda_gemm

            trace_util.is_gemm_kernel = patched_is_gemm_kernel
            self._log("  [OK] Patched Trace2Tree.util (is_gemm_kernel)")
        except ImportError as e:
            self._log(f"  [WARN] Could not patch Trace2Tree.util: {e}")

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
                    self._log("  [OK] Patched TraceEventUtils.JaxOpKeys (GEMM keys enhanced)")
        except (ImportError, AttributeError) as e:
            self._log(f"  [WARN] Could not patch TraceEventUtils: {e}")

        # Patch torch_op_mapping for better categorization
        try:
            from TraceLens.PerfModel import torch_op_mapping

            original_categorize = torch_op_mapping.categorize_torch_op

            def patched_categorize_torch_op(row):
                """Enhanced categorization with better GEMM detection."""
                result = original_categorize(row)

                if result == "other" and "kernel_details" in row and len(row["kernel_details"]) > 0:
                    kernel_name = row["kernel_details"][0]["name"]
                    pattern = r"^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$"
                    if re.match(pattern, kernel_name):
                        return "GEMM"

                return result

            torch_op_mapping.categorize_torch_op = patched_categorize_torch_op
            self._log("  [OK] Patched torch_op_mapping (categorize_torch_op)")
        except ImportError as e:
            self._log(f"  [WARN] Could not patch torch_op_mapping: {e}")

        self._log("[OK] GEMM patches applied\n")

    def generate_perf_report(
        self,
        trace_path: Path,
        output_path: Path,
        include_unlinked_kernels: bool = True,
        short_kernel_study: bool = True,
        short_kernel_threshold_us: int = 50,
        topk_ops: int = 100,
        topk_roofline_ops: int = 100,
        enable_kernel_summary: bool = False,
    ) -> Path:
        """
        Generate individual performance report from trace data.

        Args:
            trace_path: Path to the trace JSON file
            output_path: Path for output Excel file
            include_unlinked_kernels: Include unlinked kernels in report
            short_kernel_study: Enable short kernel study
            short_kernel_threshold_us: Threshold for short kernels (microseconds)
            topk_ops: Number of top operations to include
            topk_roofline_ops: Number of top roofline operations
            enable_kernel_summary: Enable kernel summary sheet

        Returns:
            Path to generated report
        """
        from TraceLens.Reporting.generate_perf_report_pytorch import main as generate_main

        # Build argument list
        args = [
            "--profile_json_path", str(trace_path),
            "--output_xlsx_path", str(output_path),
        ]

        if include_unlinked_kernels:
            args.append("--include_unlinked_kernels")
        if short_kernel_study:
            args.append("--short_kernel_study")
            args.extend(["--short_kernel_threshold_us", str(short_kernel_threshold_us)])
        if topk_ops:
            args.extend(["--topk_ops", str(topk_ops)])
        if topk_roofline_ops:
            args.extend(["--topk_roofline_ops", str(topk_roofline_ops)])
        if enable_kernel_summary:
            args.append("--enable_kernel_summary")

        # Save original argv and replace
        original_argv = sys.argv
        sys.argv = ["generate_perf_report_pytorch"] + args

        try:
            generate_main()
        finally:
            sys.argv = original_argv

        return output_path

    def generate_perf_report_rocprof(
        self,
        trace_path: Path,
        output_path: Path,
        kernel_details: bool = True,
        short_kernel_study: bool = True,
        short_kernel_threshold_us: int = 50,
        topk_kernels: int = 100,
    ) -> Path:
        """
        Generate performance report from rocprof trace data.

        Args:
            trace_path: Path to the rocprof results JSON file
            output_path: Path for output Excel file
            kernel_details: Include kernel details
            short_kernel_study: Enable short kernel study
            short_kernel_threshold_us: Threshold for short kernels
            topk_kernels: Number of top kernels to include

        Returns:
            Path to generated report
        """
        from TraceLens.Reporting.generate_perf_report_rocprof import main as generate_main

        args = [
            "--profile_json_path", str(trace_path),
            "--output_xlsx_path", str(output_path),
        ]

        if kernel_details:
            args.append("--kernel_details")
        if short_kernel_study:
            args.append("--short_kernel_study")
            args.extend(["--short_kernel_threshold_us", str(short_kernel_threshold_us)])
        if topk_kernels:
            args.extend(["--topk_kernels", str(topk_kernels)])

        original_argv = sys.argv
        sys.argv = ["generate_perf_report_rocprof"] + args

        try:
            generate_main()
        finally:
            sys.argv = original_argv

        return output_path

    def generate_collective_report(
        self,
        trace_pattern: str,
        world_size: int,
        output_path: Path,
        detailed_analysis: bool = True,
        use_multiprocessing: bool = True,
    ) -> Path:
        """
        Generate multi-rank collective report.

        Args:
            trace_pattern: Glob pattern for trace files (e.g., "rank*/trace.json")
            world_size: Number of ranks
            output_path: Path for output Excel file
            detailed_analysis: Enable detailed analysis
            use_multiprocessing: Use multiprocessing for parallel analysis

        Returns:
            Path to generated report
        """
        from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
            main as generate_main,
        )

        args = [
            "--trace_pattern", str(trace_pattern),
            "--world_size", str(world_size),
            "--output_xlsx_path", str(output_path),
        ]

        if detailed_analysis:
            args.append("--detailed_analysis")
        if use_multiprocessing:
            args.append("--use_multiprocessing")

        original_argv = sys.argv
        sys.argv = ["generate_multi_rank_collective_report_pytorch"] + args

        try:
            generate_main()
        finally:
            sys.argv = original_argv

        return output_path

    def compare_reports(
        self,
        report_paths: List[Path],
        names: List[str],
        output_path: Path,
        sheets: Optional[List[str]] = None,
    ) -> Path:
        """
        Compare multiple performance reports.

        Args:
            report_paths: List of paths to Excel reports
            names: Names for each report
            output_path: Path for output comparison file
            sheets: Sheets to compare (default: gpu_timeline, ops_summary)

        Returns:
            Path to generated comparison report
        """
        from TraceLens.Reporting.compare_perf_reports_pytorch import main as compare_main

        if sheets is None:
            sheets = ["gpu_timeline", "ops_summary"]

        args = [str(p) for p in report_paths]
        args.extend(["--names"] + names)
        args.extend(["--sheets"] + sheets)
        args.extend(["-o", str(output_path)])

        original_argv = sys.argv
        sys.argv = ["compare_perf_reports_pytorch"] + args

        try:
            compare_main()
        finally:
            sys.argv = original_argv

        return output_path

