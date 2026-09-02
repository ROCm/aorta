"""CLI tests for the ``aorta bench`` shim group.

Tests the shim contract only — not hw_queue_eval internals:
- ``aorta bench --help`` lists hw_queue_eval as a subcommand.
- ``aorta bench hw_queue_eval --help`` exposes the same commands that
  hw_queue_eval's own CLI group registers (derived at runtime, not hardcoded).
- When hw_queue_eval is unavailable, bare invoke exits non-zero with a clear
  install hint; ``--help`` exits 0 but shows the hint instead of empty help.
- Only a missing *external* dependency is read as "extra not installed": a
  missing aorta.hw_queue_eval sub-module, or any other ImportError, propagates.
- ``bench`` is correctly wired under the top-level ``aorta`` CLI.
"""

from __future__ import annotations

import importlib
import types

import click
import pytest
from click.testing import CliRunner

from aorta.cli import main
from aorta.cli.bench import bench


def _hw_queue_available() -> bool:
    """Report whether the [hw-queue] extra is installed.

    Availability is determined by actually attempting the import, not
    find_spec: aorta.hw_queue_eval files are always present in the source tree,
    but the import fails on a base install because hw_queue_eval.__init__
    pulls torch.
    """
    try:
        importlib.import_module("aorta.hw_queue_eval.cli")
    except ModuleNotFoundError:
        return False
    return True


_HW_QUEUE_AVAILABLE = _hw_queue_available()


def test_bench_help_lists_hw_queue_eval() -> None:
    """``aorta bench --help`` exits 0 and lists hw_queue_eval."""
    result = CliRunner().invoke(bench, ["--help"])
    assert result.exit_code == 0, result.output
    assert "hw_queue_eval" in result.output


def test_bench_wired_under_main() -> None:
    """``aorta bench --help`` via the top-level CLI exits 0 (validates __init__.py wiring)."""
    result = CliRunner().invoke(main, ["bench", "--help"])
    assert result.exit_code == 0, result.output
    assert "hw_queue_eval" in result.output


@pytest.mark.skipif(not _HW_QUEUE_AVAILABLE, reason="amd-aorta[hw-queue] not installed")
def test_hw_queue_eval_help_lists_subcommands() -> None:
    """``aorta bench hw_queue_eval --help`` lists every registered subcommand."""
    from aorta.hw_queue_eval.cli import cli as hw_queue_eval_cli

    expected = set(hw_queue_eval_cli.commands)
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    for sub in expected:
        assert sub in result.output, f"--help missing {sub!r}: {result.output!r}"


@pytest.mark.skipif(_HW_QUEUE_AVAILABLE, reason="hw_queue_eval is installed — error path not active")
def test_hw_queue_unavailable_invoke_shows_install_hint() -> None:
    """Bare invoke when hw_queue_eval is missing exits non-zero with install hint."""
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert result.exit_code != 0
    assert "amd-aorta[hw-queue]" in result.output


@pytest.mark.skipif(_HW_QUEUE_AVAILABLE, reason="hw_queue_eval is installed — error path not active")
def test_hw_queue_unavailable_help_shows_install_hint() -> None:
    """``--help`` when hw_queue_eval is missing shows install hint.

    Click always exits 0 for --help; the hint text is the signal, not the
    exit code.
    """
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    assert "amd-aorta[hw-queue]" in result.output


def _fail_import(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Make resolving a lazy command raise ``exc``, whatever is installed locally."""

    def raise_it(_name: str) -> None:
        raise exc

    monkeypatch.setattr("aorta.cli._lazy_group.import_module", raise_it)


def test_missing_external_dependency_becomes_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent third-party dep means the extra is not installed: show the hint."""
    _fail_import(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert result.exit_code != 0
    assert "amd-aorta[hw-queue]" in result.output


def test_missing_internal_module_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing aorta.hw_queue_eval sub-module is our bug, not a missing extra.

    Reporting it as "install the extra" would send users down a dead end, so it
    has to surface as the ModuleNotFoundError it is.
    """
    broken = ModuleNotFoundError(
        "No module named 'aorta.hw_queue_eval.sweep'",
        name="aorta.hw_queue_eval.sweep",
    )
    _fail_import(monkeypatch, broken)
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert isinstance(result.exception, ModuleNotFoundError)
    assert result.exception.name == "aorta.hw_queue_eval.sweep"


def test_other_import_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything that is not a ModuleNotFoundError is a real bug and must surface."""
    _fail_import(monkeypatch, ImportError("cannot import name 'cli'"))
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert isinstance(result.exception, ImportError)


def test_hw_queue_group_is_resolved_when_the_extra_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the import succeeding, ``bench`` hands Click the real inner group.

    Stubbed rather than skipped so the happy path stays covered on the base
    install, where torch (and so hw_queue_eval) is absent.
    """

    @click.group(name="hw_queue_eval")
    def stub() -> None:
        """Stand-in hw_queue_eval group."""

    @stub.command()
    def sweep() -> None:
        """Stand-in subcommand."""

    monkeypatch.setattr(
        "aorta.cli._lazy_group.import_module",
        lambda _name: types.SimpleNamespace(cli=stub),
    )
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    assert "sweep" in result.output
