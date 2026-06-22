"""Analysis modules for TraceLens trace processing.

The GEMM / single-config / sweep analysers each pull pandas / numpy /
openpyxl, which live behind ``amd-aorta[report]``. ``kernel_correlator`` is
deliberately stdlib-only so the NaN-correlation pipeline runs on a base
install. To keep that promise, only the lightweight modules
(``tracelens_wrapper``, ``kernel_correlator``) are imported eagerly here;
the heavy ones resolve via PEP 562 ``__getattr__`` on first access.
Issue surfaced by Copilot review on PR #162: importing any symbol from
this package used to drag in pandas/openpyxl regardless of which
analyser the caller actually wanted.
"""

from __future__ import annotations

from typing import Any

from .kernel_correlator import (
    CorrelationFinding,
    IterationRecord,
    KernelEventCorrelator,
    iter_findings_table,
)
from .tracelens_wrapper import TraceLensWrapper

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


_LAZY_SOURCES = {
    "analyze_gemm_reports": ("analyze_gemm", "analyze_gemm_reports"),
    "analyze_single_config": ("analyze_single", "analyze_single_config"),
    "analyze_sweep_config": ("analyze_sweep", "analyze_sweep_config"),
    "discover_and_run_tracelens": ("analyze_sweep", "discover_and_run_tracelens"),
}


def __getattr__(name: str) -> Any:
    """Resolve the heavy analysers on first access.

    Same rationale as ``aorta.report.generators.__getattr__``: keeps the
    kernel-trace path runnable on a base install while preserving the
    package-level public surface for callers that have the extras.
    """
    try:
        module_name, attr = _LAZY_SOURCES[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'aorta.report.analysis' has no attribute {name!r}"
        ) from exc
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
