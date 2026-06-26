"""CLI tests for the ``aorta bench`` shim group.

Tests the shim contract only — not hw_queue_eval internals:
- ``aorta bench --help`` lists hw_queue_eval as a subcommand.
- ``aorta bench hw_queue_eval --help`` exposes the same commands that
  hw_queue_eval's own CLI group registers (derived at runtime, not hardcoded).
- When hw_queue_eval is unavailable, the stub exits with a clear install message.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from aorta.cli.bench import _hw_queue_available, bench


def test_bench_help_lists_hw_queue_eval():
    """``aorta bench --help`` exits 0 and lists hw_queue_eval."""
    result = CliRunner().invoke(bench, ["--help"])
    assert result.exit_code == 0, result.output
    assert "hw_queue_eval" in result.output


@pytest.mark.skipif(not _hw_queue_available, reason="amd-aorta[hw-queue] not installed")
def test_hw_queue_eval_help_lists_subcommands():
    """``aorta bench hw_queue_eval --help`` lists every registered subcommand."""
    from aorta.hw_queue_eval.cli import cli as hw_queue_eval_cli

    expected = set(hw_queue_eval_cli.commands)
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    for sub in expected:
        assert sub in result.output, f"--help missing {sub!r}: {result.output!r}"


@pytest.mark.skipif(_hw_queue_available, reason="hw_queue_eval is installed — stub not active")
def test_hw_queue_unavailable_stub_exits_with_install_hint():
    """When hw_queue_eval is missing, the stub exits non-zero with an install hint."""
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert result.exit_code != 0
    assert "amd-aorta[hw-queue]" in result.output
