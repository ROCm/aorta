"""NumPy must be the real module, not a lazy proxy waiting to re-execute it.

DSPy proxies its optional heavy dependencies: ``require`` puts a ``_LazyModule``
in ``sys.modules`` and runs the real module on first attribute access. It hands
back an already-imported module untouched, so the proxy only appears if NumPy is
missing when DSPy first asks -- and then the deferred execution is a second
initialization of a C extension that initializes once. NumPy warns about the
reload; what follows is a module whose dtypes stop resolving, ``mu.dtype("bool")``
raising ``TypeError: data type 'bool' not understood`` inside its own bootstrap.

The damage lands on whoever touches NumPy next on that xdist worker, which is
how one cause produced three unrelated-looking failures: an index build failing
on a dtype, a fastembed cache probe answering False because its import raised,
and a doctor check reporting "warn" for a cache that was on disk. All three
passed when run on their own.

tests/conftest.py imports NumPy up front so DSPy always takes the branch that
returns the real module. These tests hold that from the outside.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("numpy", reason="numpy arrives with the cia and chat extras")


class TestNumpyIsRealHere:
    def test_it_is_imported_before_any_test_runs(self):
        """The conftest pre-import, observed from a test."""
        assert "numpy" in sys.modules

    def test_it_is_a_module_and_not_a_proxy(self):
        import numpy

        assert type(sys.modules["numpy"]).__name__ == "module"
        assert sys.modules["numpy"] is numpy

    def test_its_dtypes_resolve(self):
        """The symptom a reload produces, asserted directly."""
        import numpy

        assert numpy.dtype("bool").name == "bool"
        assert numpy.array([True, False]).dtype == numpy.bool_


class TestDspyLeavesItAlone:
    def test_require_hands_back_the_real_module(self):
        """DSPy's own early-return, which the pre-import is there to reach."""
        pytest.importorskip("dspy", reason="dspy arrives with the cia extra")
        import numpy
        from dspy.utils.lazy_import import require

        assert require("numpy") is numpy

    def test_the_c_backed_modules_are_not_proxied(self):
        """A proxy is only dangerous over a module that initializes once.

        DSPy proxying ``transformers`` is fine and expected -- nothing has
        imported it, so its deferred execution is an ordinary first import. The
        damage needs a proxy standing in front of an extension module already
        loaded in this process, which for this suite means NumPy and SciPy.
        """
        pytest.importorskip("dspy", reason="dspy arrives with the cia extra")

        proxied = [
            name
            for name in ("numpy", "scipy")
            if type(sys.modules.get(name)).__name__ == "_LazyModule"
        ]

        assert proxied == []
