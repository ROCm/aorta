"""REPL line editing and command history.

The interactive loop read input with bare ``input()`` and never imported
``readline``, so arrow keys arrived as raw escape sequences (``^[[A``) instead
of recalling the previous query. Importing readline is the whole mechanism;
these tests pin that it happens, that history survives a session, and that no
history problem can take the REPL down with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aorta.cli import chat as cli


@pytest.fixture()
def history_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "history"
    monkeypatch.setattr(cli, "HISTORY_FILE", path)
    return path


class TestReadlineIsEnabled:
    def test_readline_gets_imported(self, history_file):
        """Without the import, input() has no editing keymap at all."""
        cli._enable_line_editing()
        assert "readline" in sys.modules

    def test_the_interactive_loop_turns_it_on(self):
        """Guards against the call being dropped in a future refactor."""
        import inspect

        assert "_enable_line_editing()" in inspect.getsource(cli._interactive_loop)

    def test_a_history_length_is_set(self, history_file):
        import readline

        cli._enable_line_editing()
        assert readline.get_history_length() == cli.HISTORY_LENGTH


class TestHistoryPersistence:
    def test_previous_queries_are_read_back(self, history_file):
        history_file.write_text(
            "what does the probe engine do\nfind all mitigations\n", encoding="utf-8"
        )
        import readline

        readline.clear_history()
        cli._enable_line_editing()
        items = [
            readline.get_history_item(i)
            for i in range(1, readline.get_current_history_length() + 1)
        ]
        assert "find all mitigations" in items

    def test_saving_writes_the_file(self, history_file):
        import readline

        readline.clear_history()
        readline.add_history("a remembered query")
        cli._save_history(readline, history_file)
        assert "a remembered query" in history_file.read_text(encoding="utf-8")


class TestFailuresAreNotFatal:
    def test_a_missing_history_file_is_fine(self, tmp_path: Path, monkeypatch):
        """The first ever run has no file, which is not an error."""
        monkeypatch.setattr(cli, "HISTORY_FILE", tmp_path / "nope" / "history")
        cli._enable_line_editing()

    def test_an_unwritable_history_path_does_not_raise(self, tmp_path: Path):
        """A read-only or full home must not end the session."""
        import readline

        unwritable = tmp_path / "no-such-dir" / "history"
        cli._save_history(readline, unwritable)

    def test_a_missing_readline_module_is_tolerated(self, history_file, monkeypatch):
        """Windows has no stdlib readline; the REPL should still run."""
        monkeypatch.setitem(sys.modules, "readline", None)
        cli._enable_line_editing()
