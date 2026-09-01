"""Tests for run_terminal_command tool with mocked subprocess."""

from __future__ import annotations

from unittest.mock import patch

from aorta.chat.tools.run import run_terminal_command


class TestRunTerminalCommand:
    @patch("aorta.chat.tools.run.settings")
    def test_denied_command(self, mock_settings, tmp_path):
        mock_settings.allowed_commands = ["ls"]
        mock_settings.aorta_root = tmp_path
        result = run_terminal_command.invoke({"command": "curl http://evil.com"})
        assert "DENIED" in result
        assert "Blocked" in result

    @patch("aorta.chat.tools.run.settings")
    def test_disallowed_executable(self, mock_settings, tmp_path):
        mock_settings.allowed_commands = ["ls", "grep"]
        mock_settings.aorta_root = tmp_path
        result = run_terminal_command.invoke({"command": "docker ps"})
        assert "DENIED" in result
        assert "not in the allowlist" in result

    @patch("aorta.chat.tools.run.settings")
    def test_nonexistent_aorta_path(self, mock_settings, tmp_path):
        mock_settings.allowed_commands = ["ls"]
        mock_settings.aorta_root = tmp_path / "nonexistent"
        mock_settings.command_timeout = 10
        result = run_terminal_command.invoke({"command": "ls"})
        assert "does not exist" in result

    @patch("aorta.chat.tools.run.settings")
    def test_successful_command(self, mock_settings, tmp_path):
        mock_settings.allowed_commands = ["ls"]
        mock_settings.aorta_root = tmp_path
        mock_settings.command_timeout = 10
        (tmp_path / "file.txt").write_text("hello", encoding="utf-8")
        result = run_terminal_command.invoke({"command": "ls"})
        assert "Exit code: 0" in result
        assert "file.txt" in result

    @patch("aorta.chat.tools.run.settings")
    def test_grep_command(self, mock_settings, tmp_path):
        mock_settings.allowed_commands = ["grep"]
        mock_settings.aorta_root = tmp_path
        mock_settings.command_timeout = 10
        (tmp_path / "data.txt").write_text("foo\nbar\nbaz\n", encoding="utf-8")
        result = run_terminal_command.invoke({"command": "grep bar data.txt"})
        assert "Exit code: 0" in result
        assert "bar" in result
