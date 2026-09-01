"""Tests for _parse_action() regex parser from aorta/chat/graph/nodes.py."""

from __future__ import annotations

from aorta.chat.graph.nodes import _parse_action


class TestParseActionValid:
    def test_single_string_arg(self):
        result = _parse_action('ACTION: search_code(query="test")')
        assert result is not None
        name, kwargs = result
        assert name == "search_code"
        assert kwargs == {"query": "test"}

    def test_path_arg(self):
        result = _parse_action('ACTION: list_files(path="src/")')
        assert result is not None
        name, kwargs = result
        assert name == "list_files"
        assert kwargs == {"path": "src/"}

    def test_file_path_arg(self):
        result = _parse_action('ACTION: read_file(file_path="src/main.py")')
        assert result is not None
        name, kwargs = result
        assert name == "read_file"
        assert kwargs == {"file_path": "src/main.py"}

    def test_integer_arg(self):
        result = _parse_action('ACTION: search_code(query="test", k=3)')
        assert result is not None
        name, kwargs = result
        assert name == "search_code"
        assert kwargs == {"query": "test", "k": 3}

    def test_single_quoted_arg(self):
        result = _parse_action("ACTION: search_code(query='hello world')")
        assert result is not None
        name, kwargs = result
        assert name == "search_code"
        assert kwargs == {"query": "hello world"}

    def test_command_arg(self):
        result = _parse_action('ACTION: run_terminal_command(command="ls -la")')
        assert result is not None
        name, kwargs = result
        assert name == "run_terminal_command"
        assert kwargs == {"command": "ls -la"}

    def test_action_embedded_in_text(self):
        text = (
            "Let me search for that.\n"
            'ACTION: search_code(query="main entry")\n'
            "I will check the results."
        )
        result = _parse_action(text)
        assert result is not None
        name, kwargs = result
        assert name == "search_code"
        assert kwargs == {"query": "main entry"}

    def test_no_args(self):
        result = _parse_action("ACTION: list_files()")
        assert result is not None
        name, kwargs = result
        assert name == "list_files"
        assert kwargs == {}


class TestParseActionInvalid:
    def test_unknown_tool_returns_none(self):
        result = _parse_action('ACTION: unknown_tool(arg="val")')
        assert result is None

    def test_no_action_line(self):
        result = _parse_action("This is just a normal response with no tool call.")
        assert result is None

    def test_empty_string(self):
        result = _parse_action("")
        assert result is None

    def test_malformed_action(self):
        result = _parse_action("ACTION: ")
        assert result is None

    def test_partial_action_no_parens(self):
        result = _parse_action("ACTION: search_code")
        assert result is None


class TestParseActionEdgeCases:
    def test_command_with_parentheses_in_value(self):
        text = 'ACTION: run_terminal_command(command="ls -td experiments/multinode_* 2>/dev/null | head -1")'
        result = _parse_action(text)
        assert result is not None
        name, kwargs = result
        assert name == "run_terminal_command"
        assert "ls -td" in kwargs["command"]

    def test_command_with_subshell(self):
        text = 'ACTION: run_terminal_command(command="find . -name \'*.py\' -exec grep -l main {} +")'
        result = _parse_action(text)
        assert result is not None
        name, kwargs = result
        assert name == "run_terminal_command"

    def test_action_in_markdown_code_block(self):
        text = (
            "Let me check.\n"
            "```\n"
            'ACTION: list_files(path="src")\n'
            "```\n"
        )
        result = _parse_action(text)
        assert result is not None
        name, kwargs = result
        assert name == "list_files"
        assert kwargs == {"path": "src"}

    def test_action_in_labeled_code_block(self):
        text = (
            "```python\n"
            'ACTION: search_code(query="config")\n'
            "```\n"
        )
        result = _parse_action(text)
        assert result is not None
        name, kwargs = result
        assert name == "search_code"

    def test_action_with_preceding_text_on_same_line(self):
        text = 'I will now call ACTION: list_files(path=".")'
        result = _parse_action(text)
        assert result is not None
        name, kwargs = result
        assert name == "list_files"
