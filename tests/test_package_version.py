"""Contract tests for the lazily resolved ``aorta.__version__`` (issue #417).

``aorta/__init__.py`` resolves ``__version__`` from the installed
distribution's metadata behind a PEP 562 module ``__getattr__``, because
importing ``importlib.metadata`` and walking ``sys.path`` for dist-info
dominated the cost of a bare ``import aorta`` for an attribute almost nothing
reads.

Deferring an export has an introspection cost that is easy to ship by
accident: the name vanishes from ``dir()`` until something has already read
it, so tab-completion and ``help()`` stop advertising it. ``__dir__`` is what
buys it back, matching ``aorta.report``, ``aorta.report.analysis`` and
``aorta.report.generators``. Raised by Copilot review on PR #428.

The laziness assertions run in a subprocess: by the time this module executes
the pytest session has already imported ``aorta``, so the in-process module
state says nothing about what a fresh ``import aorta`` does.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

import aorta


def _probe(*statements: str) -> str:
    """Run statements in a fresh interpreter and return the last line printed."""
    completed = subprocess.run(
        [sys.executable, "-c", "\n".join(statements)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.splitlines()[-1]


def test_version_resolves_from_the_distribution_metadata() -> None:
    assert aorta.__version__ == version("amd-aorta")


def test_dir_advertises_version_before_it_has_been_read() -> None:
    """``dir(aorta)`` must list ``__version__`` on a fresh import.

    Without ``__dir__`` the name is missing until first access, which
    contradicts ``__all__`` and regresses introspection against the eager
    module this replaced.
    """
    assert (
        _probe(
            "import aorta",
            "print('__version__' in dir(aorta))",
        )
        == "True"
    )


def test_dir_matches_all_before_the_lazy_export_is_read() -> None:
    """Nothing ``__all__`` promises may be missing from a fresh ``dir()``."""
    assert (
        _probe(
            "import aorta",
            "print(set(aorta.__all__) <= set(dir(aorta)))",
        )
        == "True"
    )


def test_first_access_caches_the_resolved_version() -> None:
    """The resolved value lands in the module globals, so metadata is read once."""
    assert (
        _probe(
            "import aorta",
            "assert '__version__' not in vars(aorta)",
            "aorta.__version__",
            "print('__version__' in vars(aorta))",
        )
        == "True"
    )


def test_importing_aorta_does_not_read_distribution_metadata() -> None:
    """The point of the deferral: ``importlib.metadata`` stays unimported."""
    assert (
        _probe(
            "import sys, aorta",
            "print('importlib.metadata' in sys.modules)",
        )
        == "False"
    )


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """The ``__getattr__`` shim must not turn typos into something else."""
    assert (
        _probe(
            "import aorta",
            "try:",
            "    aorta.no_such_attribute",
            "except AttributeError as exc:",
            "    print('no_such_attribute' in str(exc))",
        )
        == "True"
    )
