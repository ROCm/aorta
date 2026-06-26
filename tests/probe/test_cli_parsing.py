"""CLI parsing tests for ``aorta probe`` (issue #188 Phase 1).

Covers FR 1.1 (documented flags appear in --help), FR 1.15 (handler is a
thin shim), and FR 1.18 (invalid recipe / empty argv exit non-zero).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

import aorta.cli.probe as probe_cli
from aorta.cli.probe import probe

FIXTURES = Path(__file__).parent / "fixtures"


# ---- FR 1.1 (documented flags) -------------------------------------------


def test_help_lists_documented_flags():
    """`aorta probe --help` shows the rubric-documented flag set."""
    runner = CliRunner()
    result = runner.invoke(probe, ["--help"])
    assert result.exit_code == 0
    out = result.output
    for flag in (
        "--recipe",
        "--output",
        "--ticket",
        "--dry-run",
        "--env-passthrough-mode",
        "--mitigations-file",
    ):
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


# ---- FR 1.10 (CLI flag precedence over recipe env_passthrough_mode) ------


def _invoke_probe_capturing_recipe(monkeypatch, tmp_path, *, recipe_text, cli_extra):
    """Run ``aorta probe`` against an in-memory recipe and return the
    :class:`Recipe` the CLI handed to ``run_recipe``.

    Both modules bind ``run_recipe`` at import time so patching only the
    runner module would miss the CLI binding -- patch ``aorta.cli.probe``
    directly. Mirrors the pattern in ``tests/probe/test_shared_engine.py``.
    """
    mock = MagicMock(return_value=tmp_path / "run-dir")
    monkeypatch.setattr(probe_cli, "run_recipe", mock)
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(recipe_text, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(recipe_path),
            "--output",
            str(tmp_path / "out"),
            *cli_extra,
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    mock.assert_called_once()
    args, kwargs = mock.call_args
    recipe_arg = args[0] if args else kwargs["recipe"]
    return recipe_arg


_RECIPE_WITH_FILE_MODE = (
    "schema_version: 1\n"
    "mode: probe\n"
    "ticket: PROBE-188-PRECEDENCE\n"
    "trials: 1\n"
    "mitigation_axis: [none]\n"
    "diagnostic_axis: [none]\n"
    "env_passthrough_mode: file\n"
)

_RECIPE_NO_MODE = (
    "schema_version: 1\n"
    "mode: probe\n"
    "ticket: PROBE-188-PRECEDENCE\n"
    "trials: 1\n"
    "mitigation_axis: [none]\n"
    "diagnostic_axis: [none]\n"
)


def test_recipe_env_passthrough_mode_honored_when_cli_omits_flag(monkeypatch, tmp_path):
    """Recipe says ``env_passthrough_mode: file`` and CLI omits the flag -> ``file`` wins.

    Regression for PR #194 review: the Click option used to default to
    ``"inherit"`` and unconditionally overwrite ``recipe.probe_extras
    .env_passthrough_mode``, making the recipe key impossible to honour
    when the user didn't pass the flag. Default is now ``None`` and the
    handler only overrides when the user actually supplied the flag.
    """
    recipe_arg = _invoke_probe_capturing_recipe(
        monkeypatch,
        tmp_path,
        recipe_text=_RECIPE_WITH_FILE_MODE,
        cli_extra=[],
    )
    assert recipe_arg.probe_extras is not None
    assert recipe_arg.probe_extras.env_passthrough_mode == "file"


def test_cli_env_passthrough_mode_overrides_recipe(monkeypatch, tmp_path):
    """Recipe says ``file`` and CLI says ``inherit`` -> CLI wins (FR 1.10)."""
    recipe_arg = _invoke_probe_capturing_recipe(
        monkeypatch,
        tmp_path,
        recipe_text=_RECIPE_WITH_FILE_MODE,
        cli_extra=["--env-passthrough-mode", "inherit"],
    )
    assert recipe_arg.probe_extras is not None
    assert recipe_arg.probe_extras.env_passthrough_mode == "inherit"


def test_default_passthrough_mode_when_neither_set(monkeypatch, tmp_path):
    """Neither CLI nor recipe sets the mode -> recipe-builder default ``"inherit"``."""
    recipe_arg = _invoke_probe_capturing_recipe(
        monkeypatch,
        tmp_path,
        recipe_text=_RECIPE_NO_MODE,
        cli_extra=[],
    )
    assert recipe_arg.probe_extras is not None
    assert recipe_arg.probe_extras.env_passthrough_mode == "inherit"


# ---- issue #229: --disable-detector overlay ------------------------------


def test_cli_disable_detector_unions_onto_recipe(monkeypatch, tmp_path):
    """``--disable-detector`` adds to (does not replace) the recipe's set."""
    recipe_text = _RECIPE_NO_MODE + "disable_detectors: [custom:from_recipe]\n"
    recipe_arg = _invoke_probe_capturing_recipe(
        monkeypatch,
        tmp_path,
        recipe_text=recipe_text,
        cli_extra=["--disable-detector", "tier2:hang", "--disable-detector", "tier3"],
    )
    assert recipe_arg.probe_extras is not None
    assert recipe_arg.probe_extras.disable_detectors == ("custom:from_recipe", "tier2:hang")
    assert recipe_arg.probe_extras.disable_detector_tiers == ("tier3",)


def test_cli_disable_detector_invalid_token_errors(monkeypatch, tmp_path):
    """A malformed token surfaces a friendly CLI error, not a traceback."""
    mock = MagicMock(return_value=tmp_path / "run-dir")
    monkeypatch.setattr(probe_cli, "run_recipe", mock)
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(_RECIPE_NO_MODE, encoding="utf-8")
    result = CliRunner().invoke(
        probe,
        [
            "--recipe",
            str(recipe_path),
            "--output",
            str(tmp_path / "out"),
            "--disable-detector",
            "tier9:nope",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "--disable-detector" in result.output
    mock.assert_not_called()


def test_cli_env_passthrough_mode_file_overrides_no_recipe_key(monkeypatch, tmp_path):
    """Recipe omits the key, CLI says ``file`` -> ``file`` wins."""
    recipe_arg = _invoke_probe_capturing_recipe(
        monkeypatch,
        tmp_path,
        recipe_text=_RECIPE_NO_MODE,
        cli_extra=["--env-passthrough-mode", "file"],
    )
    assert recipe_arg.probe_extras is not None
    assert recipe_arg.probe_extras.env_passthrough_mode == "file"


# ---- PR #194 round-3 review: --help bypass must be option-zone-scoped ------


def test_user_command_help_does_not_bypass_separator(tmp_path):
    """A ``--help`` token AFTER the user command must NOT bypass the
    mandatory ``--`` separator.

    Regression for PR #194 review: ``aorta probe --recipe r --output o
    echo --help`` previously short-circuited the separator check
    because ``"--help" in args`` is True -- but here ``--help`` is the
    user command's flag, not aorta's. The fix scopes the bypass to
    help tokens that sit in the aorta-option zone (before the
    user-command boundary). See
    :func:`aorta.probe.cli_helpers.help_token_in_option_zone`.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            str(tmp_path),
            "echo",
            "--help",
        ],
    )
    assert result.exit_code != 0
    assert "missing '--' separator" in result.output


def test_aorta_help_still_works(tmp_path):
    """``aorta probe --help`` (no user command, --help before any
    positional) still renders help. Pins the upper bound of the
    scoped bypass: real help invocations are not regressed.
    """
    runner = CliRunner()
    result = runner.invoke(probe, ["--help"])
    assert result.exit_code == 0
    assert "ARGV" in result.output or "argv" in result.output


def test_aorta_help_after_value_taking_option_still_works(tmp_path):
    """``aorta probe --recipe r --help`` -- the help token follows
    ``--recipe`` (a value-taking option that consumes ``r``) but stays
    in the option zone because no user-command positional has been
    seen yet. Must still render help, not raise the separator error.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--help",
        ],
    )
    assert result.exit_code == 0
    assert "ARGV" in result.output or "argv" in result.output


# ---- PR #198 round-2 review: --mitigations-file plumbed through to load_recipe ----


def test_mitigations_file_resolves_sidecar_only_name(monkeypatch, tmp_path):
    """``aorta probe --mitigations-file <sidecar>`` makes sidecar-only
    names resolve at recipe load time.

    Regression for PR #198 review (issue #195): the new flag is forwarded
    to ``load_recipe(..., sidecar_files=...)``. Without coverage, a typo
    in the kwarg name or a missing ``or None`` normalisation would slip
    through. Patches ``run_recipe`` so this is a CLI-plumbing test, not
    an end-to-end one.
    """
    mock = MagicMock(return_value=tmp_path / "run-dir")
    monkeypatch.setattr(probe_cli, "run_recipe", mock)
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_needs_sidecar.yaml"),
            "--mitigations-file",
            str(FIXTURES / "probe_sidecar.json"),
            "--output",
            str(tmp_path / "out"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    mock.assert_called_once()


def test_missing_mitigations_file_for_sidecar_recipe_fails(tmp_path):
    """Omitting ``--mitigations-file`` for a recipe that references a
    sidecar-only mitigation surfaces ``UnknownMitigationError`` as a
    ``ClickException`` (exit non-zero, no traceback).

    The error message must name the missing mitigation so operators
    can map back to the sidecar JSON they forgot to pass.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_needs_sidecar.yaml"),
            "--output",
            str(tmp_path / "out"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "sidecar_only_mitigation" in result.output


def test_mitigations_file_is_repeatable(monkeypatch, tmp_path):
    """Two ``--mitigations-file`` flags both contribute names to the
    resolver. Pins the ``multiple=True`` contract.
    """
    mock = MagicMock(return_value=tmp_path / "run-dir")
    monkeypatch.setattr(probe_cli, "run_recipe", mock)
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_needs_two_sidecars.yaml"),
            "--mitigations-file",
            str(FIXTURES / "probe_sidecar.json"),
            "--mitigations-file",
            str(FIXTURES / "probe_sidecar_two.json"),
            "--output",
            str(tmp_path / "out"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    mock.assert_called_once()


def test_mitigations_file_nonexistent_path_rejected(tmp_path):
    """Click's ``Path(exists=True)`` fires for ``--mitigations-file`` so a
    typo'd path exits non-zero with a clear message instead of being
    silently dropped on the way into ``load_recipe``.
    """
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--mitigations-file",
            str(tmp_path / "does-not-exist.json"),
            "--output",
            str(tmp_path / "out"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output


# ---- PR #194 round-3 review: exec-time errors land as Tier-1 fails ----
