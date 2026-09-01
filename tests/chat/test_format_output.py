"""Tests for _format_output() from aorta/chat/tools/run.py."""

from __future__ import annotations

from aorta.chat.tools.run import _format_output


class TestFormatOutput:
    def test_short_output_preserved(self):
        result = _format_output("hello world", 0)
        assert result == "Exit code: 0\nhello world"

    def test_exit_code_in_first_line(self):
        result = _format_output("some output", 1)
        first_line = result.splitlines()[0]
        assert first_line == "Exit code: 1"

    def test_nonzero_exit_code(self):
        result = _format_output("error occurred", 127)
        assert "Exit code: 127" in result

    def test_long_output_truncated(self):
        long_text = "x" * 5000
        result = _format_output(long_text, 0)
        assert "... (truncated)" in result
        assert len(result) < len(long_text) + 100

    def test_output_at_limit_not_truncated(self):
        text = "x" * 4000
        result = _format_output(text, 0)
        assert "truncated" not in result

    def test_output_just_over_limit_truncated(self):
        text = "x" * 4001
        result = _format_output(text, 0)
        assert "... (truncated)" in result

    def test_empty_output(self):
        result = _format_output("", 0)
        assert result == "Exit code: 0\n"
