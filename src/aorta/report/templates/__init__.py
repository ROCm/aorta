"""HTML templates for report generation."""

from .sweep_comparison_template import get_comparison_template
from .performance_report_template import (
    HTML_HEADER,
    HTML_FOOTER,
    OVERALL_GPU_CHARTS,
    CROSS_RANK_CHARTS,
    NCCL_CHARTS,
)

__all__ = [
    "get_comparison_template",
    "HTML_HEADER",
    "HTML_FOOTER",
    "OVERALL_GPU_CHARTS",
    "CROSS_RANK_CHARTS",
    "NCCL_CHARTS",
]
