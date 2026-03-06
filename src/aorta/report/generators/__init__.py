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
]
