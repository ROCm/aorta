"""aorta-report: Unified CLI for TraceLens analysis and report generation.

This package provides a command-line interface for:
- Analyzing PyTorch profiler traces with TraceLens
- Processing GPU timeline data
- Generating comparison reports (HTML, Excel, plots)
- Running full analysis pipelines

Usage:
    aorta-report --help
    python -m aorta.report --help

The ``cli`` / ``main`` re-exports below are loaded via PEP 562
``__getattr__`` so that lightweight subpackages (notably the
stdlib-only ``aorta.report.generators.kernel_report`` and
``aorta.report.analysis.kernel_correlator``) can be imported on a base
install without dragging in the heavy CLI chain (pandas / openpyxl /
matplotlib via ``aorta.report.cli`` -> ``comparison.cli``). Issue
surfaced by Copilot review on PR #162.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["cli", "main", "__version__"]


def __getattr__(name: str) -> Any:
    if name in {"cli", "main"}:
        from .cli import cli, main  # type: ignore[import-untyped]

        globals()["cli"] = cli
        globals()["main"] = main
        return {"cli": cli, "main": main}[name]
    raise AttributeError(f"module 'aorta.report' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
