"""Analysis modules for TraceLens trace processing."""

from .tracelens_wrapper import TraceLensWrapper
from .analyze_gemm import analyze_gemm_reports
from .analyze_single import analyze_single_config
from .analyze_sweep import analyze_sweep_config, discover_and_run_tracelens
from .kernel_correlator import (
    CorrelationFinding,
    IterationRecord,
    KernelEventCorrelator,
    iter_findings_table,
)

__all__ = [
    "TraceLensWrapper",
    "analyze_gemm_reports",
    "analyze_single_config",
    "analyze_sweep_config",
    "discover_and_run_tracelens",
    "CorrelationFinding",
    "IterationRecord",
    "KernelEventCorrelator",
    "iter_findings_table",
]
