"""Dependency / discipline tests: keep ``aorta bundle`` stdlib-only.

Issue #196 acceptance criterion 7: no new top-level dependencies
(uses stdlib ``tarfile`` + ``json``). The rubric for #188 has the
matching "no new third-party dependencies" guardrail. We pin that
here at the import level so a future drive-by ``import requests``
in the bundle module surfaces as a unit-test failure rather than
landing silently.

We also pin two design invariants that the rubric forbids:

* No ``runner`` / ``dispatcher`` filename under ``src/aorta/bundle/``
  (the engine-reuse gate -- bundle is a writer, not a runner).
* No ``subprocess`` import inside ``src/aorta/bundle/`` (the
  command operates purely on filesystem artifacts; spawning
  children is out of scope and would dodge the redactor).
"""

from __future__ import annotations

import ast
import pkgutil
from pathlib import Path

import aorta.bundle as _bundle_pkg

# Anything outside this set on a `from ... import ...` line in any
# bundle module is a new third-party dependency by definition.
# stdlib packages don't need to be enumerated -- isort's `known_first_party`
# already separates them, but we want a positive allowlist so an
# accidental top-level vendor package gets flagged here too.
_ALLOWED_THIRD_PARTY = frozenset({"click"})
_AORTA_INTERNAL_PREFIX = "aorta."


def _bundle_source_files() -> list[Path]:
    pkg_dir = Path(_bundle_pkg.__file__).parent
    return [p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_no_third_party_imports_beyond_click():
    """Acceptance criterion 7: stdlib + click only.

    Walks every ``.py`` under ``src/aorta/bundle/`` and asserts the
    top-level module of every import is either stdlib or
    ``aorta.*`` or in the small allowlist.
    """
    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", set()))
    for path in _bundle_source_files():
        for name in _imports(path):
            top = name.split(".", 1)[0]
            if top in stdlib:
                continue
            if name.startswith(_AORTA_INTERNAL_PREFIX) or name == "aorta":
                continue
            assert top in _ALLOWED_THIRD_PARTY, (
                f"{path}: new third-party import {name!r}; the bundle "
                "module is stdlib + click only by issue #196 acceptance."
            )


def test_no_subprocess_import_in_bundle():
    """Bundle is a filesystem-only writer; no children should be spawned."""
    for path in _bundle_source_files():
        for name in _imports(path):
            assert name != "subprocess", (
                f"{path}: 'subprocess' import not allowed in aorta.bundle "
                "(filesystem-only writer; spawning children would dodge "
                "the redactor)."
            )


def test_no_runner_or_dispatcher_filename_under_bundle():
    """Engine-reuse gate from the #188 rubric §X.4 applied to bundle.

    Bundle is not a runner; never let a future refactor smuggle a
    runner under its package name.
    """
    pkg_dir = Path(_bundle_pkg.__file__).parent
    for path in pkg_dir.rglob("*.py"):
        stem = path.stem.lower()
        assert "runner" not in stem, f"{path}: 'runner' is reserved for engine code"
        assert "dispatcher" not in stem, f"{path}: 'dispatcher' is reserved for engine code"


def test_bundle_submodules_collected():
    """Sanity: the package actually exposes the four submodules."""
    submodules = {m.name for m in pkgutil.iter_modules(_bundle_pkg.__path__)}
    assert {"errors", "manifest", "redactor", "writer"} <= submodules
