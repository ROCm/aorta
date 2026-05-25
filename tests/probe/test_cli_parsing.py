"""CLI parsing tests for ``aorta probe`` (issue #188 Phase 1).

Covers FR 1.1 (documented flags appear in --help), FR 1.15 (handler is a
thin shim), and FR 1.18 (invalid recipe / empty argv exit non-zero).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from click.testing import CliRunner

from aorta.cli.probe import probe

FIXTURES = Path(__file__).parent / "fixtures"


# ---- FR 1.1 (documented flags) -------------------------------------------


def test_help_lists_documented_flags():
    """`aorta probe --help` shows the rubric-documented flag set."""
    runner = CliRunner()
    result = runner.invoke(probe, ["--help"])
    assert result.exit_code == 0
    out = result.output
    for flag in ("--recipe", "--output", "--ticket", "--dry-run", "--env-passthrough-mode"):
        assert flag in out, f"missing flag {flag} in --help output"
    # Trailing-argv usage line:
    assert "ARGV" in out or "argv" in out


# ---- FR 1.15 (thin-shim handler) -----------------------------------------


def test_handler_is_thin_shim():
    """The handler body is bounded so orchestration can't drift in.

    Mirrors ``tests/run/test_cli_parsing.py::TestCliHandlerIsThinShell``.
    Rubric pins the cap at <= 60 lines.
    """
    fn = probe.callback
    source = inspect.getsource(fn)
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    body_start = func_def.body[0].lineno
    body_end = func_def.end_lineno
    assert body_end is not None
    body_lines = body_end - body_start + 1
    assert body_lines <= 60, (
        f"Click handler body has grown to {body_lines} lines -- "
        "move logic into aorta.probe.cli_helpers / aorta.probe.recipe_builder."
    )


def test_no_per_trial_loop_in_handler():
    """The handler must not contain a ``for ... in range(...)`` loop."""
    fn = probe.callback
    source = inspect.getsource(fn)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                raise AssertionError("Click handler contains a `for ... in range(...)` loop")


# ---- FR 1.18 (invalid inputs exit non-zero) ------------------------------


def test_empty_argv_nonzero_exit(tmp_path):
    """`aorta probe --recipe X --` with nothing after `--` exits non-zero."""
    runner = CliRunner()
    result = runner.invoke(
        probe,
        ["--recipe", str(FIXTURES / "probe_minimal.yaml"), "--output", str(tmp_path), "--"],
    )
    assert result.exit_code != 0
    assert "no trailing argv" in result.output.lower() or "no trailing argv" in str(
        result.exception
    )


def test_invalid_recipe_nonzero_exit(tmp_path):
    """`aorta probe --recipe <bogus_path>` exits non-zero with a ClickException."""
    runner = CliRunner()
    result = runner.invoke(
        probe,
        ["--recipe", str(tmp_path / "nonexistent.yaml"), "--", "echo", "hi"],
    )
    assert result.exit_code != 0


def test_triage_mode_recipe_rejected(tmp_path):
    """A non-probe-mode recipe surfaces a ClickException."""
    recipe_path = tmp_path / "triage.yaml"
    recipe_path.write_text(
        "schema_version: 1\n"
        "workload: fsdp\n"
        "trials: 1\n"
        "steps: 1\n"
        "cells:\n"
        "  - name: c\n"
        "    mitigations: [none]\n"
        "    environment: local\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(probe, ["--recipe", str(recipe_path), "--", "echo", "hi"])
    assert result.exit_code != 0
    assert "probe-mode" in result.output.lower()


def test_invalid_env_passthrough_mode():
    """Click's Choice validator rejects bogus modes pre-handler."""
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--env-passthrough-mode",
            "bogus",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output or "Invalid value" in result.output


# ---- FR 1.18 defensive parsing (post-bug-report from real-world misuse) --


def test_flag_shaped_value_for_output_rejected(tmp_path):
    """``--output --ticket X`` (i.e. $TMPDIR unset) is refused, not silently accepted.

    Real-world bug: shell expands ``--output $TMPDIR --ticket SMOKE-1``
    to ``--output --ticket SMOKE-1`` when TMPDIR is unset. Click would
    otherwise pass ``--ticket`` as the value of ``--output``, silently
    creating a directory literally named ``--ticket`` and dropping the
    real ``--ticket`` flag entirely.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            "--ticket",
            "SMOKE-1",
            "--",
            "bash",
            "-c",
            "echo hi",
        ],
    )
    assert result.exit_code != 0
    assert "looks like another flag" in result.output


def test_flag_shaped_value_for_recipe_rejected():
    """``--recipe --ticket X`` is refused (either by exists= or our callback).

    For ``--recipe``, Click's ``Path(exists=True)`` validator fires before
    the post-conversion callback, so the user sees "File '--ticket' does
    not exist" -- which is also a clear, non-zero-exit error pointing
    them at the misparse. Either message is acceptable; the contract is
    that the bug isn't silently accepted.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        ["--recipe", "--ticket", "X", "--", "echo", "hi"],
    )
    assert result.exit_code != 0
    assert "looks like another flag" in result.output or "does not exist" in result.output


def test_flag_shaped_value_for_ticket_rejected(tmp_path):
    """``--ticket --output X`` (transposed flags) is refused."""
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--ticket",
            "--output",
            str(tmp_path),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "looks like another flag" in result.output


def test_missing_double_dash_separator_rejected(tmp_path):
    """Without ``--`` in raw argv, the CLI refuses to silently sweep positionals.

    Real-world bug: a stray positional (e.g. ``SMOKE-1``) before any
    ``--`` got swept into the user command argv as the executable name,
    producing an exit-127 "fail" trial that was actually a CLI misparse.
    The double-dash separator is mandatory.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            str(tmp_path),
            "SMOKE-1",
            "bash",
            "-c",
            "echo hi",
        ],
    )
    assert result.exit_code != 0
    assert "missing '--' separator" in result.output


def test_user_command_starting_with_dash_rejected(tmp_path):
    """A user command whose first token starts with ``-`` is refused.

    Catches the residual case where ``--`` was present but a leftover
    aorta-shaped flag (e.g. ``-v``) still leaked into ``argv[0]``.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            str(tmp_path),
            "--",
            "-v",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "looks like a flag" in result.output
