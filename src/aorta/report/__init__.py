"""
aorta-report: Unified CLI for TraceLens analysis and report generation.

This package provides a command-line interface for:
- Analyzing PyTorch profiler traces with TraceLens
- Processing GPU timeline data
- Generating comparison reports (HTML, Excel, plots)
- Running full analysis pipelines

Usage:
    aorta-report --help
    python -m aorta.report --help
"""

__version__ = "0.1.0"

from .cli import cli, main

__all__ = ["cli", "main", "__version__"]

