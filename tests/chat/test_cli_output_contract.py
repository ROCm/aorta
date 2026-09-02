"""What ``aorta chat`` puts on stdout, and what it exits with.

Both are contracts the piping modes rest on, and both were broken in ways that
only show up downstream:

* ``input(prompt)`` writes the prompt to **stdout**, so a ``--json`` REPL
  interleaved ``aorta> `` with the JSON objects and produced something that is
  not JSONL -- the very thing the banner goes to stderr to avoid.
* A graph failure was rendered as an ordinary reply and swallowed, so a one-shot
  ``ask`` exited 0 having answered nothing, reporting success to automation.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from aorta.cli import chat as cli


class TestReplPromptStaysOffStdout:
    @pytest.mark.asyncio
    async def test_a_piped_json_repl_emits_only_json_on_stdout(self, capsys, monkeypatch):
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)
        monkeypatch.setattr(cli, "_enable_line_editing", lambda _path: False)
        monkeypatch.setattr(cli, "_history_path", lambda: __import__("pathlib").Path("/dev/null"))
        monkeypatch.setattr("builtins.input", _inputs(["what is aorta?", "exit"]))

        invoke = AsyncMock(return_value=("an answer", [], {}))
        await cli._interactive_loop(invoke, "json", quiet=False)

        captured = capsys.readouterr()
        for line in filter(None, captured.out.splitlines()):
            json.loads(line)  # raises if the prompt leaked onto stdout
        assert cli.PROMPT not in captured.out
        assert cli.PROMPT in captured.err

    @pytest.mark.asyncio
    async def test_a_tty_still_hands_the_prompt_to_input(self, capsys, monkeypatch):
        """readline measures the prompt from input(); losing that erases it."""
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli, "_enable_line_editing", lambda _path: True)
        monkeypatch.setattr(cli, "_history_path", lambda: __import__("pathlib").Path("/dev/null"))
        seen: list[str] = []

        def _input(prompt=""):
            seen.append(prompt)
            return "exit"

        monkeypatch.setattr("builtins.input", _input)
        await cli._interactive_loop(AsyncMock(), "rich", quiet=False)

        assert seen and seen[0] != ""


class TestAskReportsFailure:
    @pytest.mark.asyncio
    async def test_a_graph_failure_is_reported_as_a_failure(self, capsys):
        invoke = AsyncMock(side_effect=RuntimeError("provider exploded"))
        _, ok = await cli._ask_once(invoke, "q", [], "json", quiet=False)
        assert ok is False

    @pytest.mark.asyncio
    async def test_a_successful_answer_reports_success(self, capsys):
        invoke = AsyncMock(return_value=("an answer", [], {}))
        _, ok = await cli._ask_once(invoke, "q", [], "json", quiet=False)
        assert ok is True

    def test_dispatch_turns_a_failed_answer_into_a_nonzero_exit(self, monkeypatch):
        """The status is what a pipeline reads; the message is already printed."""
        monkeypatch.setattr(cli, "_load", lambda _name: _NullConfig())
        monkeypatch.setattr(cli, "_setup_logging", lambda _verbose: False)
        monkeypatch.setattr(cli, "_run", AsyncMock(return_value=False))

        with pytest.raises(SystemExit) as exc:
            cli._dispatch("q", False, False, None, None, False, False, False)
        assert exc.value.code == 1

    def test_dispatch_exits_zero_when_the_answer_succeeded(self, monkeypatch):
        monkeypatch.setattr(cli, "_load", lambda _name: _NullConfig())
        monkeypatch.setattr(cli, "_setup_logging", lambda _verbose: False)
        monkeypatch.setattr(cli, "_run", AsyncMock(return_value=True))

        cli._dispatch("q", False, False, None, None, False, False, False)


class _NullConfig:
    """Stands in for the lazily imported ``aorta.chat.config`` module."""

    @staticmethod
    def apply_cli_overrides(**_kwargs) -> None:
        return None


class TestStderrIsRestored:
    def test_suppression_is_scoped_rather_than_permanent(self, monkeypatch):
        """A REPL's first query must not silence the rest of the session."""
        monkeypatch.setattr(cli, "_stderr_suppressed", False)
        real = cli.sys.stderr
        with cli._suppress_stderr_noise():
            assert cli.sys.stderr is not real
        assert cli.sys.stderr is real


def _inputs(answers: list[str]):
    """An ``input`` stand-in returning each answer in turn."""
    remaining = list(answers)

    def _input(prompt=""):
        if prompt:
            raise AssertionError("the prompt must not reach stdout via input()")
        return remaining.pop(0)

    return _input
