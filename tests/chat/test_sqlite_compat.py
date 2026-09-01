"""The sqlite3 guards that let sqlite-vec run on RHEL 9 / CentOS Stream 9.

The interesting cases are counterfactual -- this box's sqlite is whatever it is,
and its Python either loads extensions or does not -- so both the version and the
connection are patched rather than assumed, and ``sys.modules`` is restored
afterwards so a swap never leaks into another test.
"""

from __future__ import annotations

import sqlite3 as stdlib_sqlite3
import sys
from types import SimpleNamespace

import pytest

from aorta.chat.rag import sqlite_compat
from aorta.chat.rag.sqlite_compat import (
    MIN_SQLITE_VERSION,
    _version_tuple,
    ensure_loadable_extensions,
    ensure_modern_sqlite,
)


@pytest.fixture()
def restore_sqlite_modules():
    """Undo every swap the code under test performs.

    ``ensure_modern_sqlite`` rebinds the module global as well as
    ``sys.modules``, and ``ensure_loadable_extensions`` memoises its verdict, so
    all three have to come back or the next test inherits a fake sqlite.
    """
    saved = {name: sys.modules.get(name) for name in ("sqlite3", "sqlite3.dbapi2")}
    saved_attr = sqlite_compat.sqlite3
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    sqlite_compat.sqlite3 = saved_attr
    sqlite_compat._extensions_checked = False


def _fake_pysqlite3(version: str = "3.51.1"):
    return SimpleNamespace(
        sqlite_version=version,
        dbapi2=SimpleNamespace(sqlite_version=version),
    )


def _sqlite_without_extensions():
    """A sqlite3 stand-in whose connections have no enable_load_extension."""
    return SimpleNamespace(
        NotSupportedError=stdlib_sqlite3.NotSupportedError,
        connect=lambda _target: SimpleNamespace(close=lambda: None),
    )


def _sqlite_refusing_extensions():
    """A build that keeps the method but refuses the call."""

    def _refuse(_enabled):
        raise stdlib_sqlite3.NotSupportedError("extension loading is disabled")

    return SimpleNamespace(
        NotSupportedError=stdlib_sqlite3.NotSupportedError,
        connect=lambda _target: SimpleNamespace(enable_load_extension=_refuse, close=lambda: None),
    )


class TestVersionTuple:
    def test_parses_a_normal_version(self):
        assert _version_tuple("3.34.1") == (3, 34, 1)

    def test_ignores_non_numeric_parts(self):
        """Some builds append a suffix; it must not raise."""
        assert _version_tuple("3.45.0beta") == (3, 45)

    def test_the_floor_matches_the_knn_query_shape_the_retriever_uses(self):
        """3.41 is where LIMIT reaches a virtual table, which vec0 KNN needs."""
        assert MIN_SQLITE_VERSION == (3, 41, 0)
        assert _version_tuple("3.40.1") < MIN_SQLITE_VERSION
        assert _version_tuple("3.41.0") >= MIN_SQLITE_VERSION

    def test_the_floor_is_above_what_rhel_9_ships(self):
        """The whole shim exists for 3.34.1; a floor below it would be dead code."""
        assert _version_tuple("3.34.1") < MIN_SQLITE_VERSION


class TestEnsureModernSqlite:
    def test_a_new_enough_stdlib_is_left_completely_alone(
        self, monkeypatch, restore_sqlite_modules
    ):
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.45.0")
        before = sys.modules["sqlite3"]
        ensure_modern_sqlite()
        assert sys.modules["sqlite3"] is before

    def test_an_old_stdlib_is_replaced_by_pysqlite3(self, monkeypatch, restore_sqlite_modules):
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.34.1")
        fake = _fake_pysqlite3()
        monkeypatch.setitem(sys.modules, "pysqlite3", fake)
        ensure_modern_sqlite()
        assert sys.modules["sqlite3"] is fake
        assert sys.modules["sqlite3.dbapi2"] is fake.dbapi2

    def test_the_swap_rebinds_the_module_global_too(self, monkeypatch, restore_sqlite_modules):
        """The extension check must interrogate pysqlite3, not the stdlib build."""
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.34.1")
        fake = _fake_pysqlite3()
        monkeypatch.setitem(sys.modules, "pysqlite3", fake)
        ensure_modern_sqlite()
        assert sqlite_compat.sqlite3 is fake

    def test_it_is_idempotent(self, monkeypatch, restore_sqlite_modules):
        """Both indexer.py and retriever.py reach it; the second must be safe."""
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
        assert "pip install 'amd-aorta[chat-sqlite]'" in message
        assert "pip install pysqlite3-binary" in message
        # The version actually found, so the reader can tell how far off it is.
        assert "3.34.1" in message
        # And the version wanted, which sqlite-vec's own OperationalError omits.
        assert "3.41.0" in message


class TestEnsureLoadableExtensions:
    def test_a_build_that_loads_extensions_passes_silently(self, restore_sqlite_modules):
        sqlite_compat._extensions_checked = False
        ensure_loadable_extensions()

    def test_a_missing_method_names_the_extra_to_install(self, monkeypatch, restore_sqlite_modules):
        """CPython omits enable_load_extension entirely when the flag is off."""
        sqlite_compat._extensions_checked = False
        monkeypatch.setattr(sqlite_compat, "sqlite3", _sqlite_without_extensions())
        with pytest.raises(RuntimeError) as excinfo:
            ensure_loadable_extensions()
        message = str(excinfo.value)
        assert "pip install 'amd-aorta[chat-sqlite]'" in message
        assert "enable_load_extension" in message
        # The version guard is a separate failure and must not be blamed here.
        assert "3.41.0" not in message

    def test_a_build_that_refuses_the_call_is_treated_the_same(
        self, monkeypatch, restore_sqlite_modules
    ):
        sqlite_compat._extensions_checked = False
        monkeypatch.setattr(sqlite_compat, "sqlite3", _sqlite_refusing_extensions())
        with pytest.raises(RuntimeError, match="chat-sqlite"):
            ensure_loadable_extensions()

    def test_a_pass_is_memoised(self, monkeypatch, restore_sqlite_modules):
        """It runs per connection, so the second call must not reopen anything."""
        sqlite_compat._extensions_checked = False
        ensure_loadable_extensions()

        def _explode(_target):
            raise AssertionError("ensure_loadable_extensions reconnected")

        monkeypatch.setattr(
            sqlite_compat,
            "sqlite3",
            SimpleNamespace(NotSupportedError=stdlib_sqlite3.NotSupportedError, connect=_explode),
        )
        ensure_loadable_extensions()

    def test_a_swap_reopens_the_question(self, monkeypatch, restore_sqlite_modules):
        """pysqlite3 is a different build, so an earlier pass says nothing."""
        sqlite_compat._extensions_checked = True
        monkeypatch.setattr(sqlite_compat.sqlite3, "sqlite_version", "3.34.1")
        monkeypatch.setitem(sys.modules, "pysqlite3", _fake_pysqlite3())
        ensure_modern_sqlite()
        assert sqlite_compat._extensions_checked is False
