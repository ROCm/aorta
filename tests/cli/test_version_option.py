"""Regression tests for ``aorta --version`` (issue #429).

``click.version_option`` resolves its ``package_name`` through
``importlib.metadata.version`` and, when that name is not an installed
distribution, falls back to ``packages_distributions()`` to map an import name
onto the distribution providing it. This project's distribution is
``amd-aorta`` and only its import package is ``aorta``, so passing the import
name always took that fallback -- and the fallback raises ``RuntimeError`` when
the mapping is ambiguous.

Two ordinary conditions make it ambiguous, neither needing a stale artefact:

* Any editable install of this src layout. The ``.pth`` puts ``src/`` on the
  path and the editable build itself leaves ``src/amd_aorta.egg-info`` there,
  so it is discovered as a *second* distribution also named ``amd-aorta``. A
  fresh ``pip install -e .`` in a clean checkout is enough.
* Any install, wheel included, on a distro where ``sys.platlibdir`` is
  ``lib64`` (the RHEL and Fedora family). ``python -m venv`` makes ``lib64`` a
  symlink to ``lib`` and ``site`` puts both on ``sys.path``, so *every*
  ``.dist-info`` is enumerated twice.

Naming the distribution skips the fallback.

These run in a subprocess because ``version_option`` caches the resolved
version in its closure: once anything in the process has asked for
``--version``, the resolution under test is skipped and the assertions would
pass vacuously.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

# What a doubly-enumerated ``amd-aorta`` looks like to Click, from either
# cause: the import package maps to the same distribution twice.
_AMBIGUOUS_MAPPING = """
import importlib.metadata

importlib.metadata.packages_distributions = lambda: {"aorta": ["amd-aorta", "amd-aorta"]}
"""

_INVOKE_VERSION = """
from click.testing import CliRunner

from aorta.cli import main

result = CliRunner().invoke(main, ["--version"], prog_name="aorta")
assert result.exit_code == 0, (
    f"exit={result.exit_code} exception={result.exception!r} output={result.output!r}"
)
print(result.output.strip())
"""


def _version_output(*prelude: str) -> str:
    """Run ``--version`` in a fresh interpreter and return the line it printed."""
    completed = subprocess.run(
        [sys.executable, "-c", "\n".join([*prelude, _INVOKE_VERSION])],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.splitlines()[-1]


def test_version_reports_the_installed_distribution() -> None:
    """``--version`` prints the version of the ``amd-aorta`` distribution."""
    assert _version_output() == f"aorta, version {version('amd-aorta')}"


def test_version_does_not_depend_on_the_import_name_mapping() -> None:
    """An ambiguous import-name mapping must not reach ``--version``.

    Fails with Click's ``'aorta' maps to multiple installed distributions``
    ``RuntimeError`` if the option goes back to naming the import package.
    """
    assert _version_output(_AMBIGUOUS_MAPPING) == f"aorta, version {version('amd-aorta')}"
