"""Report generators for HTML and Excel."""

from .html_generator import generate_html, image_to_base64
from .excel_report import create_final_excel_report

__all__ = [
    "generate_html",
    "image_to_base64",
    "create_final_excel_report",
]

