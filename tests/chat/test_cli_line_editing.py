"""REPL line editing and command history.

The interactive loop read input with bare ``input()`` and never imported
``readline``, so arrow keys arrived as raw escape sequences (``^[[A``) instead
of recalling the previous query. Importing readline is the whole mechanism;
these tests pin that it happens, that history survives a session, and that no
history problem can take the REPL down with it.

The REPL moved to ``aorta/cli/chat.py`` in the merge. The history path is now
passed in rather than read from a module global, so these tests hand it a
``tmp_path`` instead of monkeypatching a constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aorta.cli import chat as cli


@pytest.fixture()
def history_file(tmp_path: Path) -> Path:
    return tmp_path / "history"


class TestReadlineIsEnabled:
    def test_readline_gets_imported(self, history_file):
        """Without the import, input() has no editing keymap at all."""
        cli._enable_line_editing(history_file)
        assert "readline" in sys.modules

    def test_the_interactive_loop_turns_it_on(self):
        """Guards against the call being dropped in a future refactor."""
        import inspect

        assert "_enable_line_editing(" in inspect.getsource(cli._interactive_loop)

    def test_a_history_length_is_set(self, history_file):
        import readline

        cli._enable_line_editing(history_file)
        assert readline.get_history_length() == cli.HISTORY_LENGTH


class TestHistoryPersistence:
    """A session's queries survive into the next one.

    Asserted through the readline API rather than against the file's bytes: the
    two implementations Python may be linked against write different formats
    (GNU readline writes plain lines, libedit writes a ``_HiStOrY_V2_`` header
    and escapes spaces as ``\\040``), and neither is the behaviour under test.
    """

    def test_previous_queries_are_read_back(self, history_file):
        import readline

        readline.clear_history()
        readline.add_history("what does the probe engine do")
        readline.add_history("find all mitigations")
        cli._save_history(readline, history_file)

        readline.clear_history()
        cli._enable_line_editing(history_file)
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
        assert history_file.exists()
        assert history_file.stat().st_size > 0

    def test_saving_creates_the_cache_directory(self, tmp_path: Path):
        """The XDG cache dir does not exist on a first-ever run."""
        import readline

        target = tmp_path / "never" / "existed" / "repl_history"
        readline.clear_history()
        readline.add_history("first ever query")
        cli._save_history(readline, target)
        assert target.exists()


class TestFailuresAreNotFatal:
    def test_a_missing_history_file_is_fine(self, tmp_path: Path):
        """The first ever run has no file, which is not an error."""
        cli._enable_line_editing(tmp_path / "nope" / "history")

    def test_an_unwritable_history_path_does_not_raise(self, tmp_path: Path):
        """A read-only or full home must not end the session."""
        import readline

        readonly = tmp_path / "readonly"
        readonly.mkdir(mode=0o500)
        cli._save_history(readline, readonly / "history")

    def test_a_missing_readline_module_is_tolerated(self, history_file, monkeypatch):
        """Windows has no stdlib readline; the REPL should still run."""
        monkeypatch.setitem(sys.modules, "readline", None)
        cli._enable_line_editing(history_file)


class TestHistoryLocation:
    def test_history_lives_under_the_xdg_cache(self, monkeypatch, tmp_path: Path):
        """Not a dotfile in ``$HOME``: chat's state is XDG-anchored now."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        path = cli._history_path()
        assert path.parent == tmp_path / "aorta" / "chat"
