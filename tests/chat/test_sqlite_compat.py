"""The sqlite3 swap that lets Chroma run on RHEL 9 / CentOS Stream 9.

The interesting cases are counterfactual -- this box's sqlite is whatever it is
-- so the version is patched rather than assumed, and ``sys.modules`` is
restored afterwards so a swap never leaks into another test.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from aorta.chat.rag import sqlite_compat
from aorta.chat.rag.sqlite_compat import (
    MIN_SQLITE_VERSION,
    _version_tuple,
    ensure_modern_sqlite,
)


@pytest.fixture()
def restore_sqlite_modules():
    """Undo any ``sys.modules`` swap the code under test performs."""
    saved = {
        name: sys.modules.get(name) for name in ("sqlite3", "sqlite3.dbapi2")
    }
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _fake_pysqlite3(version: str = "3.51.1"):
    return SimpleNamespace(
        sqlite_version=version,
        dbapi2=SimpleNamespace(sqlite_version=version),
    )


class TestVersionTuple:
    def test_parses_a_normal_version(self):
        assert _version_tuple("3.34.1") == (3, 34, 1)

    def test_ignores_non_numeric_parts(self):
        """Some builds append a suffix; it must not raise."""
        assert _version_tuple("3.45.0beta") == (3, 45)

    def test_the_floor_matches_chroma_s_documented_requirement(self):
        assert MIN_SQLITE_VERSION == (3, 35, 0)
        assert _version_tuple("3.34.1") < MIN_SQLITE_VERSION
        assert _version_tuple("3.35.0") >= MIN_SQLITE_VERSION


class TestEnsureModernSqlite:
    def test_a_new_enough_stdlib_is_left_completely_alone(
        self, monkeypatch, restore_sqlite_modules
    ):
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.45.0")
        before = sys.modules["sqlite3"]
        ensure_modern_sqlite()
        assert sys.modules["sqlite3"] is before

    def test_an_old_stdlib_is_replaced_by_pysqlite3(
        self, monkeypatch, restore_sqlite_modules
    ):
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.34.1")
        fake = _fake_pysqlite3()
        monkeypatch.setitem(sys.modules, "pysqlite3", fake)
        ensure_modern_sqlite()
        assert sys.modules["sqlite3"] is fake
        assert sys.modules["sqlite3.dbapi2"] is fake.dbapi2

    def test_it_is_idempotent(self, monkeypatch, restore_sqlite_modules):
        """Both indexer.py and retriever.py call it; the second must be safe."""
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.34.1")
        fake = _fake_pysqlite3()
        monkeypatch.setitem(sys.modules, "pysqlite3", fake)
        ensure_modern_sqlite()
        ensure_modern_sqlite()
        assert sys.modules["sqlite3"] is fake

    def test_an_old_stdlib_with_no_fallback_names_the_package_to_install(
        self, monkeypatch, restore_sqlite_modules
    ):
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.34.1")
        monkeypatch.setitem(sys.modules, "pysqlite3", None)
        with pytest.raises(RuntimeError) as excinfo:
            ensure_modern_sqlite()
        message = str(excinfo.value)
        assert "pip install pysqlite3-binary" in message
        # The version actually found, so the reader can tell how far off it is.
        assert "3.34.1" in message
        # Chroma's own error links to docs instead of naming the fix.
        assert "3.35.0" in message
