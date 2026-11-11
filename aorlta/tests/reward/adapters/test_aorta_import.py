"""Importing aorta's backends: only from the snapshot whose commit the checkers' versions name."""

import importlib
import sys

import pytest

from aorlta.reward.adapters import aorta_import


def _package(root, where):
    d = root / "aorta" / "instrumentation" / "rocjitsu_sanitizers"
    d.mkdir(parents=True)
    for p in (root / "aorta", root / "aorta" / "instrumentation", d):
        (p / "__init__.py").write_text("")
    (d / "probe.py").write_text(f"WHERE = {where!r}\n")


@pytest.fixture
def no_aorta(monkeypatch):
    def aorta_modules():
        return [n for n in sys.modules if n == "aorta" or n.startswith("aorta.")]

    saved = {n: sys.modules.pop(n) for n in aorta_modules()}
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for n in aorta_modules():
        del sys.modules[n]
    sys.modules.update(saved)


def test_a_backend_imported_from_outside_the_snapshot_is_refused(tmp_path, no_aorta):
    _package(tmp_path / "installed", "installed")
    _package(tmp_path / "snapshot" / "src", "snapshot")
    sys.path.insert(0, str(tmp_path / "installed"))
    importlib.invalidate_caches()
    mod, why = aorta_import.load("probe", str(tmp_path / "snapshot" / "src"))
    assert mod is None and "not from the snapshot" in why


def test_the_snapshot_is_imported_when_nothing_else_provides_aorta(tmp_path, no_aorta):
    _package(tmp_path / "snapshot" / "src", "snapshot")
    importlib.invalidate_caches()
    mod, why = aorta_import.load("probe", str(tmp_path / "snapshot" / "src"))
    assert why is None and mod.WHERE == "snapshot"
