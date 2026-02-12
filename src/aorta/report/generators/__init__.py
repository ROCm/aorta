"""Report generators for HTML, Excel, and plots."""

from .html_generator import generate_html, image_to_base64
from .excel_report import create_final_excel_report
from .plot_generator import (
    generate_plots,
    generate_summary_plots,
    generate_gemm_plots,
    generate_single_config_plots,
)
from .hwqueue_excel import (
    generate_hwqueue_excel,
    generate_single_run_excel,
    generate_sweep_excel,
    generate_comparison_excel,
)
from .hwqueue_plots import (
    generate_hwqueue_plots,
    generate_single_run_plots,
    generate_sweep_plots,
    generate_comparison_plots,
)
from .hwqueue_html import (
    generate_hwqueue_html,
    generate_single_run_html,
    generate_sweep_html,
    generate_comparison_html,
)

__all__ = [
    "generate_html",
    "image_to_base64",
    "create_final_excel_report",
    "generate_plots",
    "generate_summary_plots",
    "generate_gemm_plots",
    "generate_single_config_plots",
    "generate_hwqueue_excel",
    "generate_single_run_excel",
    "generate_sweep_excel",
    "generate_comparison_excel",
    "generate_hwqueue_plots",
    "generate_single_run_plots",
    "generate_sweep_plots",
    "generate_comparison_plots",
    "generate_hwqueue_html",
    "generate_single_run_html",
    "generate_sweep_html",
    "generate_comparison_html",
]
