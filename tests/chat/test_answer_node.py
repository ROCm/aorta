"""The tool-free Q&A path.

Regression cover for a live failure against gpt-oss-20b: `answer_node` has no
tool-execution loop, but it was handed `SYSTEM_PROMPT`, which advertises six
tools and instructs the model to call them before answering. A reasoning model
obeys -- it emitted the tool call into its reasoning channel and returned empty
content -- so every question dead-ended on "I couldn't generate a response".
DeepSeek had masked the bug by ignoring the instruction and answering anyway.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aorta.chat.graph.nodes import (
    ANSWER_PROMPT,
    SYSTEM_PROMPT,
    _build_answer_message,
    answer_node,
)
from tests.chat.conftest import make_fake_llm

#: ``run_terminal_command`` is absent on purpose: it is opt-in, so the prompts
#: describe it only when ``enable_shell_tool`` registered it.
TOOL_NAMES = [
    "list_files",
    "read_file",
    "search_code",
    "grep_code",
    "search_repo_map",
]


def _state(context: str = "### src/probe.py\n```\ndef run(): ...\n```"):
    return {
        "messages": [HumanMessage(content="What does run() do?")],
        "retrieved_context": context,
        "route": "question",
        "plan": None,
        "command_output": None,
        "critic_feedback": None,
        "iteration": 0,
    }


class TestPromptOffersNoTools:
    @pytest.mark.parametrize("tool", TOOL_NAMES)
    def test_no_tool_is_advertised(self, tool: str):
        assert tool not in ANSWER_PROMPT

    def test_the_absence_of_tools_is_stated_explicitly(self):
        assert "NO tools" in ANSWER_PROMPT

    def test_the_model_is_told_to_admit_a_gap_instead_of_reaching_for_tools(self):
        assert "does not contain the answer" in ANSWER_PROMPT

    def test_the_act_prompt_still_offers_tools(self):
        """The tool-using path must be untouched by this split."""
        for tool in TOOL_NAMES:
            assert tool in SYSTEM_PROMPT

    def test_retrieved_context_is_interpolated(self):
        message = _build_answer_message("### a.py\n```\nx = 1\n```")
        assert "x = 1" in message.content
        assert "{context}" not in message.content

    def test_answer_node_uses_the_tool_free_prompt(self):
        """Guards against a future edit pointing this node back at SYSTEM_PROMPT."""
        import inspect

        source = inspect.getsource(answer_node)
        assert "_build_answer_message" in source
        assert "_build_system_message" not in source


class TestEmptyContentIsNotADeadEnd:
    @pytest.mark.asyncio
    async def test_a_usable_answer_is_passed_through_untouched(self):
        fake = make_fake_llm(["run() executes one trial. See src/probe.py:1."])
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await answer_node(_state())
        assert result["messages"][0].content.startswith("run() executes")

    @pytest.mark.asyncio
    async def test_empty_content_becomes_actionable_guidance(self):
        """What the user saw was the generic extract_reply fallback."""
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await answer_node(_state())
        reply = result["messages"][0].content
        assert reply
        assert "more specific" in reply

    @pytest.mark.asyncio
    async def test_whitespace_only_content_counts_as_empty(self):
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="   \n  "))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await answer_node(_state())
        assert "more specific" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_the_token_count_is_logged_so_the_cause_is_diagnosable(
        self, caplog
    ):
        """Output tokens with no content is the signature of a reasoning model."""
        response = AIMessage(content="")
        response.usage_metadata = {
            "input_tokens": 1627,
            "output_tokens": 75,
            "total_tokens": 1702,
        }
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=response)
        with caplog.at_level("WARNING"), patch(
            "aorta.chat.graph.nodes._get_llm", return_value=fake
        ):
            await answer_node(_state())
        assert "75 output tokens" in caplog.text

    @pytest.mark.asyncio
    async def test_a_reasoning_channel_is_logged_at_debug_when_present(
        self, caplog
    ):
        """langchain-openai drops gpt-oss's `reasoning` today; read it anyway."""
        response = AIMessage(
            content="",
            additional_kwargs={"reasoning": "I should call search_code here."},
        )
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=response)
        with caplog.at_level("DEBUG"), patch(
            "aorta.chat.graph.nodes._get_llm", return_value=fake
        ):
            await answer_node(_state())
        assert "I should call search_code here." in caplog.text
