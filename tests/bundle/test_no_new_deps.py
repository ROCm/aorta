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
            # Relative imports (``from .errors import ...``, level > 0)
            # are internal by definition -- ``node.module`` is a bare
            # sibling name with no package prefix, so recording it would
            # mis-flag it as a third-party top-level import. Skip them.
            if node.level > 0:
                continue
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


def test_bundle_only_imports_its_own_aorta_subpackage():
    """The bundle package must not import sibling ``aorta`` packages.

    ``aorta.triage.output`` (the old home of ``safe_slug``) imports
    PyYAML and the registry stack, so a ``from aorta.triage... import``
    silently busts the stdlib+click import budget (PR #199 review).
    Only ``aorta.bundle.*`` internal imports are allowed.
    """
    for path in _bundle_source_files():
        for name in _imports(path):
            if name == "aorta" or name.startswith(_AORTA_INTERNAL_PREFIX):
                assert name.startswith("aorta.bundle"), (
                    f"{path}: cross-package import {name!r}; aorta.bundle must "
                    "depend only on its own subpackage to stay stdlib+click "
                    "(no transitive PyYAML via aorta.triage)."
                )


def test_bundle_safe_slug_matches_triage_canonical():
    """The inlined ``aorta.bundle._slug`` copy must stay byte-for-byte
    equivalent to the canonical ``aorta.triage.output.safe_slug`` so the
    decoupling (PR #199) cannot silently drift -- e.g. if triage tightens
    the slug regex for a path-traversal fix, this test fails until the
    bundle copy is updated too.
    """
    from aorta.bundle._slug import NO_TICKET_SLUG as BUNDLE_NO_TICKET
    from aorta.bundle._slug import safe_slug as bundle_slug
    from aorta.triage.output import NO_TICKET_SLUG as TRIAGE_NO_TICKET
    from aorta.triage.output import safe_slug as triage_slug

    assert BUNDLE_NO_TICKET == TRIAGE_NO_TICKET
    samples = [
        "PROJ-123",
        "TKT 1",
        "a/b/c",
        "..",
        ".",
        "",
        "weird:name*x",
        "a.b-c_d",
        "../../etc/passwd",
        "with space and / slash",
    ]
    for s in samples:
        assert bundle_slug(s) == triage_slug(s), f"slug drift on {s!r}"


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
