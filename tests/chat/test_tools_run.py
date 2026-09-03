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
    def test_a_path_qualified_lookalike_does_not_inherit_the_allowlist(
        self, mock_settings, tmp_path
    ):
        """The allowlist names commands, not basenames.

        Comparing only ``parts[0].split("/")[-1]`` meant an operator who
        narrowed the list to one trusted binary got every binary sharing its
        name -- including one the model had just written to /tmp. Relative
        forms resolve against the command's working directory, which is the
        AORTA tree, so they are never the interpreter on PATH.
        """
        mock_settings.allowed_commands = ["ls"]
        mock_settings.aorta_root = tmp_path
        for command in ("/tmp/ls", "./ls", "../ls", "bin/ls"):
            result = run_terminal_command.invoke({"command": command})
            assert "DENIED" in result, command
            assert "not in the allowlist" in result

    @patch("aorta.chat.tools.run.settings")
    def test_the_refusal_explains_why_a_matching_name_was_not_enough(
        self, mock_settings, tmp_path
    ):
        """'ls is allowed but /tmp/ls is not' is confusing without a reason."""
        mock_settings.allowed_commands = ["ls"]
        mock_settings.aorta_root = tmp_path
        result = run_terminal_command.invoke({"command": "/tmp/ls"})
        assert "not the 'ls' on PATH" in result
        assert "allowed_commands" in result

    @patch("aorta.chat.tools.run.settings")
    def test_the_real_binary_behind_an_allowlisted_name_is_still_accepted(
        self, mock_settings, tmp_path
    ):
        """``/usr/bin/ls`` is the ``ls`` a bare ``ls`` would have run.

        Refusing every path would take away a form that was previously
        accepted, so the two are told apart by resolving them rather than by
        comparing basenames -- which is the thing that could not tell them
        apart in the first place.
        """
        import shutil

        resolved = shutil.which("ls")
        assert resolved, "no ls on PATH to test against"
        mock_settings.allowed_commands = ["ls"]
        mock_settings.aorta_root = tmp_path
        mock_settings.command_timeout = 10
        assert "DENIED" not in run_terminal_command.invoke({"command": resolved})

    @patch("aorta.chat.tools.run.settings")
    def test_an_impostor_beside_the_real_binary_is_refused(self, mock_settings, tmp_path):
        """The case the whole change is about, with a real file on disk."""
        impostor = tmp_path / "ls"
        impostor.write_text("#!/bin/sh\necho pwned\n", encoding="utf-8")
        impostor.chmod(0o755)
        mock_settings.allowed_commands = ["ls"]
        mock_settings.aorta_root = tmp_path
        mock_settings.command_timeout = 10
        result = run_terminal_command.invoke({"command": str(impostor)})
        assert "DENIED" in result
        assert "pwned" not in result

    @patch("aorta.chat.tools.run.settings")
    def test_an_explicitly_allowlisted_path_is_still_permitted(self, mock_settings, tmp_path):
        """Refusing paths outright would take away a capability an operator has.

        The rule is that the allowlist entry has to match the command, not that
        a path can never appear in one.
        """
        script = tmp_path / "hello.sh"
        script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        script.chmod(0o755)
        mock_settings.allowed_commands = [str(script)]
        mock_settings.aorta_root = tmp_path
        mock_settings.command_timeout = 10
        result = run_terminal_command.invoke({"command": str(script)})
        assert "DENIED" not in result
        assert "hi" in result

    @patch("aorta.chat.tools.run.settings")
    def test_every_pipeline_stage_gets_the_same_check(self, mock_settings, tmp_path):
        """A later stage must not be the way round the whole-token rule."""
        mock_settings.allowed_commands = ["ls", "grep"]
        mock_settings.aorta_root = tmp_path
        result = run_terminal_command.invoke({"command": "ls | /tmp/grep x"})
        assert "DENIED" in result

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
