"""Tests for _parse_action() regex parser from aorta/chat/graph/nodes.py."""

from __future__ import annotations

import pytest

from aorta.chat.graph.nodes import _parse_action
from aorta.chat.plugins import OPTIONAL_CHAT_TOOLS


@pytest.fixture()
def shell_tool_registered(monkeypatch):
    """Put ``run_terminal_command`` in the registry ``_parse_action`` consults.

    The shell tool is opt-in, and the registry is resolved once at import, so a
    setting flipped now would not reach it. These cases are about parsing a
    command string -- the argument values that made the old ``[^)]*`` capture
    stop early -- not about whether the tool is enabled.
    """
    from aorta.chat.graph import nodes

    monkeypatch.setitem(
        nodes.TOOL_REGISTRY,
        "run_terminal_command",
        OPTIONAL_CHAT_TOOLS["run_terminal_command"],
    )


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

    def test_command_arg(self, shell_tool_registered):
        result = _parse_action('ACTION: run_terminal_command(command="ls -la")')
        assert result is not None
        name, kwargs = result
        assert name == "run_terminal_command"
        assert kwargs == {"command": "ls -la"}


class TestEscapedQuotes:
    r"""An escaped quote is part of the value, not the end of it.

    Matching with ``[^"]*`` ended the value at the first ``"`` whatever
    preceded it, so ``query="say \"hi\""`` parsed as ``say \`` -- the tool ran
    and the answer was assembled from a query the model never asked for, with
    nothing anywhere to say so.
    """

    def test_an_escaped_quote_stays_inside_the_value(self):
        result = _parse_action(r'ACTION: search_code(query="say \"hi\"")')
        assert result == ("search_code", {"query": 'say "hi"'})

    def test_an_escaped_quote_does_not_truncate_a_later_argument(self):
        result = _parse_action(r'ACTION: search_code(query="a \"b\" c", k=2)')
        assert result == ("search_code", {"query": 'a "b" c', "k": 2})

    def test_an_escaped_single_quote_in_a_single_quoted_value(self):
        result = _parse_action(r"ACTION: search_code(query='it\'s here')")
        assert result == ("search_code", {"query": "it's here"})

    def test_a_backslash_at_the_end_of_a_value_survives(self):
        result = _parse_action(r'ACTION: search_code(query="path\\")')
        assert result == ("search_code", {"query": "path\\"})


class TestUnparseableArgumentsAreRefused:
    """Half-reading a call is worse than not reading it.

    A refusal reaches the critic and earns another attempt. A call that parsed
    with some arguments dropped cannot be noticed by anything downstream.
    """

    def test_an_unterminated_value_is_refused(self):
        assert _parse_action('ACTION: search_code(query="unclosed') is None

    def test_a_positional_argument_is_refused(self):
        """There is no name to map it onto the tool's schema with."""
        assert _parse_action('ACTION: search_code("test")') is None

    def test_a_non_literal_value_is_refused(self):
        assert _parse_action("ACTION: search_code(query=some_variable)") is None

    def test_trailing_junk_is_not_passed_over_in_silence(self):
        assert _parse_action('ACTION: search_code(query="a" and then some)') is None

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
    def test_command_with_parentheses_in_value(self, shell_tool_registered):
        text = 'ACTION: run_terminal_command(command="ls -td experiments/multinode_* 2>/dev/null | head -1")'
        result = _parse_action(text)
        assert result is not None
        name, kwargs = result
        assert name == "run_terminal_command"
        assert "ls -td" in kwargs["command"]

    def test_command_with_subshell(self, shell_tool_registered):
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
