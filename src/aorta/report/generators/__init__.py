"""Report generators for HTML, Excel, plots, and kernel-trace.

The HTML / Excel / plot generators each pull a heavy optional dep
(matplotlib, openpyxl, pandas) that is *not* part of the base ``aorta``
install -- they live behind the ``aorta[report]`` extra. The kernel-
trace generator, by contrast, deliberately depends only on the standard
library so a NaN-correlation report can be produced on a stock host.

To keep that promise, this module loads only the lightweight
``generate_kernel_report`` symbol eagerly and uses a PEP 562
``__getattr__`` shim to defer the heavy generators until they are
actually requested. ``from aorta.report.generators import generate_html``
still works for callers that have the extras installed; importing
anything from a kernel-trace-only path no longer triggers the heavy
chain (issue raised by Copilot review on PR #162).
"""

from __future__ import annotations

from typing import Any

from .kernel_report import generate_kernel_report

__all__ = [
    "generate_html",
    "image_to_base64",
    "create_final_excel_report",
    "generate_kernel_report",
    "generate_plots",
    "generate_summary_plots",
    "generate_gemm_plots",
    "generate_single_config_plots",
]


_LAZY_SOURCES = {
    "generate_html": ("html_generator", "generate_html"),
    "image_to_base64": ("html_generator", "image_to_base64"),
    "create_final_excel_report": ("excel_report", "create_final_excel_report"),
    "generate_plots": ("plot_generator", "generate_plots"),
    "generate_summary_plots": ("plot_generator", "generate_summary_plots"),
    "generate_gemm_plots": ("plot_generator", "generate_gemm_plots"),
    "generate_single_config_plots": ("plot_generator", "generate_single_config_plots"),
}


def __getattr__(name: str) -> Any:
    """Resolve the heavy generators on first access.

    Eagerly importing them at package init would force every consumer
    of the kernel-trace path -- including the dependency-free CLI
    subcommand -- to also have ``matplotlib`` / ``openpyxl`` / ``pandas``
    installed. Deferring lets a base install get the kernel-trace
    artefact bundle without those extras.
    """
    try:
        module_name, attr = _LAZY_SOURCES[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'aorta.report.generators' has no attribute {name!r}"
        ) from exc
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
