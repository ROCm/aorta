"""CLI tests for the ``aorta bench`` shim group.

Tests the shim contract only — not hw_queue_eval internals:
- ``aorta bench --help`` lists hw_queue_eval as a subcommand.
- ``aorta bench hw_queue_eval --help`` exposes the same commands that
  hw_queue_eval's own CLI group registers (derived at runtime, not hardcoded).
- When hw_queue_eval is unavailable, every entry point (bare invoke, --help)
  exits non-zero with a clear install hint.
- ``bench`` is correctly wired under the top-level ``aorta`` CLI.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from aorta.cli.bench import _load_hw_queue_cli, bench
from aorta.cli import main

# Availability is determined by actually attempting the import, not find_spec:
# aorta.hw_queue_eval files are always present in the source tree, but the
# import fails on a base install because hw_queue_eval.__init__ pulls torch.
_HW_QUEUE_AVAILABLE = _load_hw_queue_cli() is not None


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
    """``--help`` when hw_queue_eval is missing shows install hint instead of empty help."""
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert "amd-aorta[hw-queue]" in result.output
