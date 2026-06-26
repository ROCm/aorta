"""CLI tests for the ``aorta bench`` shim group.

Tests the shim contract only — not hw_queue_eval internals:
- ``aorta bench --help`` lists hw_queue_eval as a subcommand.
- ``aorta bench hw_queue_eval --help`` exposes the same commands that
  hw_queue_eval's own CLI group registers (derived at runtime, not hardcoded).
- When hw_queue_eval is unavailable, invoking the subcommand exits non-zero
  with a clear install hint.
"""

from __future__ import annotations

import importlib

import pytest
from click.testing import CliRunner

from aorta.cli.bench import bench

_HW_QUEUE_AVAILABLE = importlib.util.find_spec("aorta.hw_queue_eval") is not None


def test_bench_help_lists_hw_queue_eval() -> None:
    """``aorta bench --help`` exits 0 and lists hw_queue_eval."""
    result = CliRunner().invoke(bench, ["--help"])
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
def test_hw_queue_unavailable_exits_with_install_hint() -> None:
    """When hw_queue_eval is missing, invoking the subcommand exits non-zero with a hint."""
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert result.exit_code != 0
    assert "amd-aorta[hw-queue]" in result.output
