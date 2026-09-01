"""The two tool-calling protocols, and the guards around both.

Findings from running gpt-oss-20b against a real gateway, each of which cost a
query and some tokens to discover:

* It never emits a parseable ``ACTION:`` line -- 0 of 8 rounds -- so the text
  protocol burned 11 billed calls and answered nothing. Hence LLM_TOOL_MODE.
* Offered tools on the final synthesis call, it keeps calling them and returns
  no prose, so a loop that gathered plenty still answered nothing.
* It repeats an identical tool call when a result disappoints, spending rounds
  that cannot teach it anything.
* The gateway leaked harmony markers into a tool name
  (``search_code<|channel|>commentary``), and the unguarded registry lookup
  turned that into a KeyError that aborted the whole graph run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aorta.chat.graph.nodes import (
    _MAX_UNPRODUCTIVE_ROUNDS,
    _NO_ANSWER_MSG,
    TOOL_REGISTRY,
    _execute_tool,
    _normalise_tool_name,
    act_node,
)


def _state(query: str = "find all mitigations"):
    """Default query is search-shaped, which is what forces the tool path."""
    return {
        "messages": [HumanMessage(content=query)],
        "retrieved_context": "### src/x.py\n```\nx = 1\n```",
        "route": "action",
        "plan": None,
        "command_output": None,
        "critic_feedback": None,
        "iteration": 0,
    }


def _tool_call(name: str, args: dict, call_id: str = "c1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _fake_llm(round_responses: list, final_text: str = "Final answer."):
    """A model whose bound form yields *round_responses*, unbound the final text.

    Mirrors the real split: the loop runs on ``bind_tools(...)``, and the
    synthesis call runs on the plain model so no tools are on offer.
    """
    plain = MagicMock()
    plain.ainvoke = AsyncMock(return_value=AIMessage(content=final_text))
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=round_responses)
    plain.bind_tools = MagicMock(return_value=bound)
    return plain, bound


@pytest.fixture()
def native_mode(monkeypatch):
    from aorta.chat.graph import nodes

    monkeypatch.setattr(nodes.settings, "llm_tool_mode", "native")
    monkeypatch.setattr(nodes.settings, "max_act_rounds", 5)
    monkeypatch.setattr(nodes.settings, "max_act_rounds_search", 8)


@pytest.fixture()
def text_mode(monkeypatch):
    from aorta.chat.graph import nodes

    monkeypatch.setattr(nodes.settings, "llm_tool_mode", "text")
    monkeypatch.setattr(nodes.settings, "max_act_rounds", 5)
    monkeypatch.setattr(nodes.settings, "max_act_rounds_search", 8)


class TestSuiteIsIndependentOfLocalConfig:
    def test_the_default_mode_under_test_is_text(self):
        """Pins the conftest guard.

        Without it, a developer `.env` containing LLM_TOOL_MODE=native -- the
        value a reasoning model requires -- silently sent four TestActNode
        tests down the native path and broke them. Deleting the guard should
        fail here rather than somewhere unrelated.
        """
        from aorta.chat.graph import nodes

        assert nodes.settings.llm_tool_mode == "text"


class TestModeDispatch:
    @pytest.mark.asyncio
    async def test_native_binds_tools(self, native_mode):
        plain, _bound = _fake_llm([AIMessage(content="Answered.")])
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        plain.bind_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_does_not_bind_tools(self, text_mode):
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="Answered."))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            await act_node(_state())
        fake.bind_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unknown_mode_is_rejected_loudly(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "llm_tool_mode", "function_calling")
        with pytest.raises(ValueError, match="unknown LLM_TOOL_MODE"):
            await act_node(_state())

    @pytest.mark.asyncio
    async def test_the_mode_name_is_case_and_space_tolerant(self, monkeypatch):
        from aorta.chat.graph import nodes

        monkeypatch.setattr(nodes.settings, "llm_tool_mode", "  NATIVE ")
        monkeypatch.setattr(nodes.settings, "max_act_rounds", 5)
        monkeypatch.setattr(nodes.settings, "max_act_rounds_search", 8)
        plain, _bound = _fake_llm([AIMessage(content="Answered.")])
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        assert result["messages"][0].content == "Answered."


class TestToolNameNormalisation:
    def test_harmony_markers_are_stripped(self):
        assert _normalise_tool_name("search_code<|channel|>commentary") == "search_code"

    def test_a_plain_name_is_untouched(self):
        assert _normalise_tool_name("search_code") == "search_code"

    def test_surrounding_whitespace_goes(self):
        assert _normalise_tool_name("  read_file  ") == "read_file"

    def test_a_mangled_name_still_reaches_its_tool(self):
        """The KeyError crash: the name was real, the marker made it unknown."""
        assert _normalise_tool_name("list_files<|channel|>x") in TOOL_REGISTRY


class TestExecuteToolNeverRaises:
    def test_an_unknown_tool_returns_a_readable_error(self):
        result = _execute_tool("no_such_tool", {})
        assert "no tool named" in result

    def test_the_error_lists_what_is_available(self):
        """So the model can correct itself on the next round."""
        result = _execute_tool("no_such_tool", {})
        for name in TOOL_REGISTRY:
            assert name in result

    def test_a_mangled_name_is_executed_rather_than_rejected(self):
        with patch.dict(
            TOOL_REGISTRY, {"list_files": MagicMock(invoke=lambda _: "ok")}
        ):
            assert _execute_tool("list_files<|channel|>commentary", {}) == "ok"

    def test_a_tool_that_raises_is_reported_not_propagated(self):
        boom = MagicMock()
        boom.invoke.side_effect = RuntimeError("disk on fire")
        with patch.dict(TOOL_REGISTRY, {"list_files": boom}):
            assert "disk on fire" in _execute_tool("list_files", {})


class TestNativeLoop:
    @pytest.mark.asyncio
    async def test_a_tool_call_is_executed_and_fed_back(self, native_mode):
        plain, bound = _fake_llm(
            [
                AIMessage(
                    content="", tool_calls=[_tool_call("list_files", {"path": "."})]
                ),
                AIMessage(content="There are three files."),
            ]
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py\nb.py") as ex,
        ):
            result = await act_node(_state())
        ex.assert_called_once_with("list_files", {"path": "."})
        assert result["messages"][0].content == "There are three files."
        sent = bound.ainvoke.call_args_list[-1][0][0]
        assert any(isinstance(m, ToolMessage) for m in sent)

    @pytest.mark.asyncio
    async def test_an_identical_repeated_call_is_not_run_twice(self, native_mode):
        call = _tool_call("list_files", {"path": "."})
        plain, _bound = _fake_llm(
            [
                AIMessage(content="", tool_calls=[call]),
                AIMessage(content="", tool_calls=[dict(call, id="c2")]),
                AIMessage(content="Done."),
            ]
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py") as ex,
        ):
            await act_node(_state())
        assert ex.call_count == 1

    @pytest.mark.asyncio
    async def test_differing_arguments_are_not_treated_as_a_repeat(self, native_mode):
        plain, _bound = _fake_llm(
            [
                AIMessage(
                    content="", tool_calls=[_tool_call("search_code", {"k": 10})]
                ),
                AIMessage(
                    content="", tool_calls=[_tool_call("search_code", {"k": 20}, "c2")]
                ),
                AIMessage(content="Done."),
            ]
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="hits") as ex,
        ):
            await act_node(_state())
        assert ex.call_count == 2

    @pytest.mark.asyncio
    async def test_the_final_call_offers_no_tools(self, native_mode):
        """The bug: bound to tools, the model kept calling them and said nothing."""
        plain, _bound = _fake_llm(
            [AIMessage(content="", tool_calls=[_tool_call("list_files", {})])] * 8,
            final_text="Synthesised answer.",
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="x"),
        ):
            result = await act_node(_state())
        plain.ainvoke.assert_awaited_once()
        assert result["messages"][0].content == "Synthesised answer."

    @pytest.mark.asyncio
    async def test_the_final_instruction_is_a_user_turn(self, native_mode):
        """As a SystemMessage it had no effect on Anthropic.

        LiteLLM hoists system messages into Anthropic's `system` parameter, so
        appending one never made it the last thing the model saw: the
        conversation still ended on tool results and the model kept working. One
        run's entire answer was "Installed. Now let me build the HIP binary."
        """
        from aorta.chat.graph.nodes import _FINAL_ANSWER_MSG

        plain, _bound = _fake_llm(
            [AIMessage(content="", tool_calls=[_tool_call("list_files", {})])] * 8,
            final_text="Complete answer.",
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="x"),
        ):
            await act_node(_state())

        sent = plain.ainvoke.call_args[0][0]
        assert sent[-1].content == _FINAL_ANSWER_MSG
        assert isinstance(sent[-1], HumanMessage)

    @pytest.mark.asyncio
    async def test_budget_exhaustion_is_logged(self, native_mode, caplog):
        """A truncated-looking answer should be explainable from the logs."""
        plain, _bound = _fake_llm(
            [AIMessage(content="", tool_calls=[_tool_call("list_files", {})])] * 8,
            final_text="Complete answer.",
        )
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="x"),
        ):
            await act_node(_state())
        assert "budget" in caplog.text
        assert "MAX_ACT_ROUNDS" in caplog.text

    @pytest.mark.asyncio
    async def test_an_empty_final_answer_becomes_guidance(self, native_mode):
        plain, _bound = _fake_llm(
            [AIMessage(content="", tool_calls=[_tool_call("list_files", {})])] * 8,
            final_text="",
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="x"),
        ):
            result = await act_node(_state())
        assert result["messages"][0].content == _NO_ANSWER_MSG


class TestWastedCallGuards:
    @pytest.mark.asyncio
    async def test_the_text_loop_gives_up_instead_of_spending_the_budget(
        self, text_mode
    ):
        """gpt-oss burned 11 calls here; the cap is 2 unproductive rounds."""
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        assert fake.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS
        assert result["messages"][0].content == _NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_empty_content_is_never_returned_as_the_answer(self, text_mode):
        """It was, and extract_reply then showed the generic failure message."""
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="   "))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        assert result["messages"][0].content.strip()

    @pytest.mark.asyncio
    async def test_a_real_text_answer_returns_immediately(self, text_mode):
        """A non-search query needs no tools, so one round is enough."""
        fake = MagicMock()
        fake.ainvoke = AsyncMock(
            return_value=AIMessage(content="AORTA uses Python 3.10+.")
        )
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state("what python version does AORTA need"))
        assert result["messages"][0].content == "AORTA uses Python 3.10+."
        assert fake.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_a_search_query_is_re_prompted_once_then_accepted(self, text_mode):
        """Pre-existing behaviour, now bounded: it used to re-prompt eight times."""
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="Answer from context."))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state("find all config files"))
        assert fake.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS
        assert result["messages"][0].content == "Answer from context."

    @pytest.mark.asyncio
    async def test_the_native_loop_also_stops_when_nothing_comes_back(
        self, native_mode
    ):
        plain, bound = _fake_llm([AIMessage(content="")] * 8, final_text="")
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        assert bound.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS
        assert result["messages"][0].content == _NO_ANSWER_MSG
