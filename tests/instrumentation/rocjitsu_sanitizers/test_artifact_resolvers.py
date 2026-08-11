"""Resolver tests for the prebuilt bundle vs. CMake build-tree layouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.consan import resolve_consan_hook
from aorta.instrumentation.rocjitsu_sanitizers.recipe import _resolve_waitcheck_binary

_BUILD_HOOK = Path("lib") / "rocjitsu" / "src" / "rocjitsu" / "hooks" / "librocjitsu_dbi_hooks.so"
_PREBUILT_HOOK = Path("lib") / "librocjitsu_dbi_hooks.so"


def _make_prebuilt_bundle(root: Path) -> Path:
    """Materialize a flattened prebuilt bundle (bin/ + lib/) under ``root``."""

    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir(parents=True)
    (root / "bin" / "rj_waitcheck").write_bytes(b"waitcheck")
    (root / _PREBUILT_HOOK).write_bytes(b"hook")
    return root


def _make_build_tree(root: Path) -> Path:
    """Materialize a raw CMake build tree (tools/ + nested lib/) under ``root``."""

    (root / "tools").mkdir(parents=True)
    (root / "tools" / "rj_waitcheck").write_bytes(b"waitcheck")
    (root / _BUILD_HOOK).parent.mkdir(parents=True)
    (root / _BUILD_HOOK).write_bytes(b"hook")
    return root


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROCJITSU_PREBUILT", raising=False)
    monkeypatch.delenv("ROCJITSU_BUILD", raising=False)


def test_waitcheck_resolver_prefers_prebuilt_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _make_prebuilt_bundle(tmp_path / "bundle")
    monkeypatch.setenv("ROCJITSU_PREBUILT", str(bundle))

    assert _resolve_waitcheck_binary() == bundle / "bin" / "rj_waitcheck"


def test_waitcheck_resolver_accepts_build_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _make_build_tree(tmp_path / "build")
    monkeypatch.setenv("ROCJITSU_BUILD", str(build))

    assert _resolve_waitcheck_binary() == build / "tools" / "rj_waitcheck"


def test_waitcheck_resolver_prebuilt_wins_over_build_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _make_prebuilt_bundle(tmp_path / "bundle")
    build = _make_build_tree(tmp_path / "build")
    monkeypatch.setenv("ROCJITSU_PREBUILT", str(bundle))
    monkeypatch.setenv("ROCJITSU_BUILD", str(build))

    assert _resolve_waitcheck_binary() == bundle / "bin" / "rj_waitcheck"


def test_waitcheck_resolver_none_when_unset() -> None:
    assert _resolve_waitcheck_binary() is None


def test_waitcheck_resolver_ignores_missing_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A bundle root that exists but lacks bin/rj_waitcheck must not resolve.
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("ROCJITSU_PREBUILT", str(tmp_path / "empty"))

    assert _resolve_waitcheck_binary() is None


def test_consan_hook_resolver_prefers_prebuilt_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _make_prebuilt_bundle(tmp_path / "bundle")
    monkeypatch.setenv("ROCJITSU_PREBUILT", str(bundle))

    assert resolve_consan_hook() == bundle / _PREBUILT_HOOK


def test_consan_hook_resolver_accepts_build_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = _make_build_tree(tmp_path / "build")
    monkeypatch.setenv("ROCJITSU_BUILD", str(build))

    assert resolve_consan_hook() == build / _BUILD_HOOK


def test_consan_hook_resolver_prebuilt_wins_over_build_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _make_prebuilt_bundle(tmp_path / "bundle")
    build = _make_build_tree(tmp_path / "build")
    monkeypatch.setenv("ROCJITSU_PREBUILT", str(bundle))
    monkeypatch.setenv("ROCJITSU_BUILD", str(build))

    assert resolve_consan_hook() == bundle / _PREBUILT_HOOK


def test_consan_hook_resolver_explicit_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _make_prebuilt_bundle(tmp_path / "bundle")
    monkeypatch.setenv("ROCJITSU_PREBUILT", str(bundle))
    explicit = tmp_path / "explicit.so"

    assert resolve_consan_hook(explicit) == explicit


def test_consan_hook_resolver_none_when_unset() -> None:
    assert resolve_consan_hook() is None
