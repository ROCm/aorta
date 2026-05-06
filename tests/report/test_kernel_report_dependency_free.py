"""Regression test for the kernel-trace report's dependency-free promise.

Pre-fix (Copilot review on PR #162) every entry into the kernel-trace
path went through ``aorta.report.__init__`` -> ``aorta.report.cli``,
which in turn imports the comparison subpackage and ultimately
``pandas`` / ``openpyxl`` / ``matplotlib``. That defeated the whole
selling point of ``aorta-report generate kernel-trace`` running on the
base install.

This test pins the contract that importing
``aorta.report.generators.kernel_report.generate_kernel_report`` and
``aorta.report.analysis.kernel_correlator.KernelEventCorrelator``
(plus the package-level re-exports) does *not* drag in any of those
heavy modules. Any new eager import in the chain will fail this test
loudly.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_HEAVY_MODULES = ("pandas", "openpyxl", "matplotlib", "seaborn", "numpy")


def _run_in_clean_subprocess(import_block: str) -> set[str]:
    """Execute ``import_block`` in a fresh interpreter and return any
    heavy modules that ended up loaded."""
    script = textwrap.dedent(
        f"""
        import sys
        {import_block}
        loaded = sorted(
            m for m in sys.modules
            if any(m == h or m.startswith(h + ".") for h in {_HEAVY_MODULES!r})
        )
        for name in loaded:
            print(name)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


class TestKernelReportImportIsDependencyFree:
    def test_leaf_module_import_does_not_load_pandas_openpyxl_matplotlib(self):
        loaded = _run_in_clean_subprocess(
            "from aorta.report.generators.kernel_report import generate_kernel_report"
        )
        assert loaded == set(), (
            f"importing the leaf module pulled in heavy deps: {sorted(loaded)}; "
            "the dependency-free promise of the kernel-trace path is broken"
        )

    def test_generators_package_level_import_stays_lazy(self):
        loaded = _run_in_clean_subprocess(
            "from aorta.report.generators import generate_kernel_report"
        )
        assert loaded == set(), (
            f"package-level kernel-trace import dragged in {sorted(loaded)}; "
            "aorta.report.generators.__init__ must not eagerly load the heavy "
            "html/excel/plot generators just to expose generate_kernel_report"
        )

    def test_analysis_package_level_correlator_import_stays_lazy(self):
        loaded = _run_in_clean_subprocess(
            "from aorta.report.analysis import KernelEventCorrelator"
        )
        assert loaded == set(), (
            f"package-level KernelEventCorrelator import dragged in {sorted(loaded)}; "
            "aorta.report.analysis.__init__ must not eagerly load the analyse_* "
            "modules just to expose KernelEventCorrelator"
        )

    def test_aorta_report_top_level_does_not_eagerly_load_cli_chain(self):
        # Just importing the top-level package used to load the CLI
        # dispatch chain (comparison.combine -> pandas). Verify the
        # __getattr__ shim defers that until ``cli`` / ``main`` is
        # actually requested.
        loaded = _run_in_clean_subprocess("import aorta.report")
        assert loaded == set(), (
            f"`import aorta.report` loaded {sorted(loaded)} eagerly; the "
            "package __init__ must keep the CLI imports lazy via __getattr__"
        )
