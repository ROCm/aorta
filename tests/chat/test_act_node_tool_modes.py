"""The two tool-calling protocols, and the guards around both.

Findings from running gpt-oss-20b against a real gateway, each of which cost a
query and some tokens to discover:

* It never emits a parseable ``ACTION:`` line -- 0 of 8 rounds -- so the text
  protocol burned 11 billed calls and answered nothing. Hence ``llm_tool_mode``.
* Offered tools on the final synthesis call, it keeps calling them and returns
  no prose, so a loop that gathered plenty still answered nothing.
* It repeats an identical tool call when a result disappoints, spending rounds
  that cannot teach it anything.
* The gateway leaked harmony markers into a tool name
  (``search_code<|channel|>commentary``), and the unguarded registry lookup
  turned that into a KeyError that aborted the whole graph run.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aorta.chat.graph import nodes
from aorta.chat.graph.nodes import (
    _MAX_ESCALATED_ROUNDS,
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

        Without it, a developer profile or exported
        ``AORTA_CHAT_LLM_TOOL_MODE=native`` -- the value a reasoning model
        requires -- silently sent four TestActNode tests down the native path
        and broke them. Deleting the guard should fail here rather than
        somewhere unrelated.
        """
        from aorta.chat.graph import nodes

        assert nodes.settings.llm_tool_mode == "text"


class TestTheGiveUpMessageIsForTheUser:
    """It used to be a developer diagnostic printed in the answer slot.

    Four sentences naming an environment variable, an internal text protocol, a
    provider API and a class of model the user never chose -- phrased as a
    condition they cannot evaluate. The reporter's note was "the chatbot should
    not talk about model". The specifics are still logged, and the tool
    protocol is named on the startup ``LLM backend`` line; the answer slot gets
    one plain sentence and a command to run.
    """

    @pytest.mark.parametrize(
        "leak",
        [
            "AORTA_CHAT",
            "ACTION:",
            "reasoning model",
            "function-calling",
            "native",
            "protocol",
            "tool",
        ],
    )
    def test_it_names_no_internals(self, leak):
        assert leak.lower() not in _NO_ANSWER_MSG.lower()

    def test_it_leaves_the_user_able_to_reach_the_fix(self):
        """Shorter is not the goal -- reachable is. Hence the command."""
        assert "aorta chat doctor" in _NO_ANSWER_MSG

    def test_it_stays_short_enough_to_read_as_an_answer(self):
        assert len(_NO_ANSWER_MSG) < 200

    @pytest.mark.asyncio
    async def test_the_diagnostic_moved_to_the_log_rather_than_vanishing(
        self, text_mode, caplog
    ):
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
        ):
            await act_node(_state())
        assert "AORTA_CHAT_LLM_TOOL_MODE" in caplog.text


class TestTheRepromptMatchesTheProtocol:
    """The nudge asked for an ``ACTION:`` line in both loops.

    Harmless while only a deliberate setting reached the native path; wrong once
    auto-escalation puts users there, because the function-calling path does not
    parse that syntax and asking for it teaches the model to emit it.
    """

    @pytest.mark.asyncio
    async def test_the_native_loop_never_asks_for_an_action_line(self, native_mode):
        plain, bound = _fake_llm([AIMessage(content="")] * 8, final_text="")
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        sent = [
            message.content
            for call in bound.ainvoke.call_args_list
            for message in call[0][0]
        ]
        assert not any("ACTION:" in text for text in sent if isinstance(text, str))

    @pytest.mark.asyncio
    async def test_the_text_loop_still_does(self, text_mode):
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            await act_node(_state())
        sent = [
            message.content
            for call in fake.ainvoke.call_args_list
            for message in call[0][0]
        ]
        assert any("ACTION:" in text for text in sent if isinstance(text, str))


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
        with pytest.raises(ValueError, match="unknown llm_tool_mode"):
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
    """Awaited rather than called, so a tool cannot stall the event loop.

    ``grep_code`` walks the tree, ``search_code`` embeds a query and
    ``run_terminal_command`` runs a subprocess up to its timeout, all from a
    coroutine a Chainlit request handler is awaiting. ``BaseTool.ainvoke`` hands
    a tool with no coroutine of its own to ``run_in_executor``, so a synchronous
    tool still runs off the loop and needs no wrapper here (issue #444).
    """

    async def test_an_unknown_tool_returns_a_readable_error(self):
        result = await _execute_tool("no_such_tool", {})
        assert "no tool named" in result

    async def test_the_error_lists_what_is_available(self):
        """So the model can correct itself on the next round."""
        result = await _execute_tool("no_such_tool", {})
        for name in TOOL_REGISTRY:
            assert name in result

    async def test_a_mangled_name_is_executed_rather_than_rejected(self):
        stub = MagicMock(ainvoke=AsyncMock(return_value="ok"))
        with patch.dict(TOOL_REGISTRY, {"list_files": stub}):
            assert await _execute_tool("list_files<|channel|>commentary", {}) == "ok"

    async def test_a_tool_that_raises_is_reported_not_propagated(self):
        boom = MagicMock()
        boom.ainvoke = AsyncMock(side_effect=RuntimeError("disk on fire"))
        with patch.dict(TOOL_REGISTRY, {"list_files": boom}):
            assert "disk on fire" in await _execute_tool("list_files", {})

    async def test_a_real_registry_tool_still_runs_through_ainvoke(self):
        """Against the real ``BaseTool``, not a stub, so the shim is exercised."""
        result = await _execute_tool("list_files", {"path": "."})
        assert result and "no tool named" not in result

    async def test_a_synchronous_tool_runs_off_the_event_loop(self):
        """The claim that awaiting is enough for a tool with no coroutine.

        A stub ``AsyncMock`` would prove nothing -- it runs on the loop. This
        uses a real ``@tool``, so ``BaseTool``'s ``run_in_executor`` fallback is
        what is being tested, and thread identity is what shows it worked.
        """
        from langchain_core.tools import tool

        ran_on: list[int] = []

        @tool
        def record_thread(path: str = ".") -> str:
            """Record which thread this ran on."""
            ran_on.append(threading.get_ident())
            return "ok"

        with patch.dict(TOOL_REGISTRY, {"list_files": record_thread}):
            assert await _execute_tool("list_files", {"path": "."}) == "ok"

        assert ran_on, "the tool never ran"
        assert threading.get_ident() not in ran_on


class TestNativeLoop:
    @pytest.mark.asyncio
    async def test_a_tool_call_is_executed_and_fed_back(self, native_mode):
        plain, bound = _fake_llm(
            [
                AIMessage(content="", tool_calls=[_tool_call("list_files", {"path": "."})]),
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
                AIMessage(content="", tool_calls=[_tool_call("search_code", {"k": 10})]),
                AIMessage(content="", tool_calls=[_tool_call("search_code", {"k": 20}, "c2")]),
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


def _dead_end_reply(
    output_tokens: int = 105,
    reasoning: str | None = None,
    finish_reason: str = "stop",
):
    """The signature docs/chat/providers.md calls distinctive.

    Empty ``content``, a non-zero output-token count and ``finish_reason:
    stop``: the model spent tokens, finished normally and returned no text, so
    the text went somewhere this protocol cannot read. Taken from the
    reporter's transcript -- "round 1 produced no text despite 105 output
    tokens".
    """
    return AIMessage(
        content="",
        additional_kwargs={"reasoning": reasoning} if reasoning else {},
        response_metadata={"finish_reason": finish_reason} if finish_reason else {},
        usage_metadata={
            "input_tokens": 900,
            "output_tokens": output_tokens,
            "total_tokens": 900 + output_tokens,
        },
    )


def _escalating_llm(native_reply: str = "Answered natively."):
    """A model that dead-ends on the text protocol and answers on the native one.

    Which is the whole claim under test, and the one the docs measured: the same
    query drove 8 real tool calls and a complete answer in ``native`` after 0
    parseable actions in 8 rounds of ``text``.
    """
    plain = MagicMock()
    plain.ainvoke = AsyncMock(return_value=_dead_end_reply())
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=AIMessage(content=native_reply))
    plain.bind_tools = MagicMock(return_value=bound)
    return plain, bound


@pytest.fixture()
def tool_mode_not_chosen(monkeypatch):
    """Nobody set ``llm_tool_mode``, so the default is in force.

    The suite exports ``AORTA_CHAT_LLM_TOOL_MODE=text`` in ``conftest`` to keep
    itself independent of the developer's profile, which makes the setting
    explicit for every test -- and an explicit setting is authoritative. Tests
    about the escalation therefore have to say which of the two they mean.
    """
    monkeypatch.setattr(nodes, "_tool_mode_is_explicit", lambda: False)


class TestTheDeadEndSignature:
    """The trigger, and why it has two halves rather than one.

    **The live check behind this was not run.** Review asked for the trigger to
    be sharpened with "and the reasoning channel is populated", which
    needs one fact from outside this repository: whether the reporter's AMD APIM
    gateway populates ``additional_kwargs["reasoning"]``. Establishing that
    needs a query against their endpoint, and there are no credentials for it
    here.

    So the trigger fires on the documented token signature, tightens itself when
    the channel happens to be present, and ``_REQUIRE_REASONING_CHANNEL`` turns
    the channel into a requirement in one line if it is ever shown to be
    dependable. Both paths are tested because only one of them is verified.
    """

    def test_tokens_spent_on_no_text_is_the_signature(self):
        assert nodes._is_reasoning_dead_end(_dead_end_reply())

    def test_a_populated_reasoning_channel_is_too(self):
        """Even with no usage metadata: a gateway may report one and not both."""
        assert nodes._is_reasoning_dead_end(
            AIMessage(content="", additional_kwargs={"reasoning": "let me think"})
        )

    def test_vllms_field_name_is_read_as_well(self):
        """``langchain-openai`` says ``reasoning``; vLLM says ``reasoning_content``."""
        assert nodes._is_reasoning_dead_end(
            AIMessage(content="", additional_kwargs={"reasoning_content": "hmm"})
        )

    def test_an_empty_reply_that_spent_nothing_is_not_the_signature(self):
        """A truncation or a dropped request must not change the protocol."""
        assert not nodes._is_reasoning_dead_end(AIMessage(content=""))
        assert not nodes._is_reasoning_dead_end(_dead_end_reply(output_tokens=0))

    def test_a_malformed_token_count_is_read_as_none(self):
        reply = AIMessage(content="")
        reply.usage_metadata = {"output_tokens": "many"}  # type: ignore[assignment]
        assert not nodes._is_reasoning_dead_end(reply)

    @pytest.mark.parametrize("reason", ["length", "content_filter", "max_tokens"])
    def test_a_stated_cutoff_disqualifies_the_token_signal(self, reason):
        """Tokens spent with no text also describes a truncation or a filter.

        Neither is a protocol problem -- native would run out of tokens too --
        and this is the half of the trigger that would otherwise switch the
        whole process on one of them.
        """
        assert not nodes._is_reasoning_dead_end(
            _dead_end_reply(finish_reason=reason)
        )

    def test_anthropics_field_name_is_read_as_well(self):
        """LiteLLM normalises to ``finish_reason``; langchain-anthropic does not."""
        reply = _dead_end_reply(finish_reason="")
        reply.response_metadata = {"stop_reason": "max_tokens"}
        assert not nodes._is_reasoning_dead_end(reply)

    def test_an_unstated_finish_reason_does_not_disqualify_it(self):
        """The field is optional, so requiring it would silence the escalation.

        Same asymmetry as the reasoning channel: a *stated* cutoff is evidence,
        an absent one is not, and a gateway that omits it is the one this was
        written for.
        """
        assert nodes._is_reasoning_dead_end(_dead_end_reply(finish_reason=""))

    def test_the_reasoning_channel_survives_a_cutoff(self):
        """It is the stronger signal: the answer demonstrably went elsewhere."""
        assert nodes._is_reasoning_dead_end(
            _dead_end_reply(reasoning="analysis...", finish_reason="length")
        )

    def test_requiring_the_channel_is_a_one_line_change(self, monkeypatch):
        """What flipping the flag buys, so the option stays real rather than aspirational."""
        monkeypatch.setattr(nodes, "_REQUIRE_REASONING_CHANNEL", True)
        assert not nodes._is_reasoning_dead_end(_dead_end_reply())
        assert nodes._is_reasoning_dead_end(_dead_end_reply(reasoning="analysis..."))


class TestAnExplicitProtocolIsAuthoritative:
    """``AORTA_CHAT_LLM_TOOL_MODE`` is documented, so a choice must stick.

    A stock local vLLM started without ``--enable-auto-tool-choice`` cannot
    serve the native protocol at all, so escalating there would trade a bad
    answer for a failed request. The escalation exists for the user who never
    made a choice.
    """

    def test_the_built_in_default_is_not_a_choice(self, monkeypatch):
        from aorta.chat import config

        monkeypatch.delenv("AORTA_CHAT_LLM_TOOL_MODE", raising=False)
        config.reset_settings()
        assert nodes._tool_mode_is_explicit() is False

    def test_the_environment_variable_is_a_choice(self, monkeypatch):
        from aorta.chat import config

        monkeypatch.setenv("AORTA_CHAT_LLM_TOOL_MODE", "text")
        config.reset_settings()
        assert nodes._tool_mode_is_explicit() is True

    def test_the_profile_file_is_a_choice(self, chat_profile, monkeypatch):
        from aorta.chat import config

        monkeypatch.delenv("AORTA_CHAT_LLM_TOOL_MODE", raising=False)
        chat_profile.write_text('llm_tool_mode = "text"\n', encoding="utf-8")
        config.reset_settings()
        assert nodes._tool_mode_is_explicit() is True

    def test_an_unreadable_fields_set_is_treated_as_a_choice(self, monkeypatch):
        """The gate has to fail towards *not* overriding.

        ``model_fields_set`` is how a stated preference is told apart from the
        built-in default, so a settings object that cannot answer leaves the
        question open -- and of the two answers, ``False`` is the one that lets
        the escalation move a mode the user may well have chosen. Guessing in
        that direction would break the only promise this gate makes.
        """
        class _StandInSettings:
            """What a test double looks like: no real ``model_fields_set``."""

            llm_tool_mode = "text"
            model_fields_set = MagicMock()

        monkeypatch.setattr(nodes, "settings", _StandInSettings())
        assert nodes._tool_mode_is_explicit() is True

    @pytest.mark.asyncio
    async def test_a_chosen_text_mode_survives_the_dead_end(self, text_mode):
        """The suite's own conftest export is the explicit setting here."""
        plain, _bound = _escalating_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        plain.bind_tools.assert_not_called()
        assert result["messages"][0].content == _NO_ANSWER_MSG


class TestAutoEscalationToNative:
    """Between "we recognised this exact failure" and "we stopped", try the fix.

    The reporter's transcript: 4 billed calls, `act_node` logging the signature
    on both rounds, and no answer. The docs already recorded that the same query
    in ``native`` drove 8 real tool calls and answered completely, so the one
    thing missing was the retry.
    """

    @pytest.mark.asyncio
    async def test_the_query_is_retried_on_the_native_protocol(
        self, text_mode, tool_mode_not_chosen
    ):
        plain, bound = _escalating_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        plain.bind_tools.assert_called_once()
        assert result["messages"][0].content == "Answered natively."

    @pytest.mark.asyncio
    async def test_it_costs_one_extra_call_not_a_second_loop(
        self, text_mode, tool_mode_not_chosen
    ):
        """The bounded budget: 4 wasted calls in the act node, not more.

        Two text rounds are what the query already paid for; the retry adds one.
        """
        plain, bound = _escalating_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        assert plain.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS
        assert bound.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_a_retry_that_also_dead_ends_stops_after_one_round(
        self, text_mode, tool_mode_not_chosen
    ):
        """A model that says nothing on both protocols has answered the question.

        A second native round could not add information, and the final synthesis
        call is skipped too: asking a model that has said nothing three times to
        summarise is one more billed call for the same result.
        """
        plain, bound = _escalating_llm()
        bound.ainvoke = AsyncMock(return_value=_dead_end_reply())
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        assert bound.ainvoke.await_count == _MAX_ESCALATED_ROUNDS
        # Two text rounds, then one retrieval fallback. No final synthesis.
        assert plain.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS + 1
        assert result["messages"][0].content == _NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_the_worst_case_budget_is_named(self, text_mode, tool_mode_not_chosen):
        """Everything above and below, added up, for the case where nothing works.

        The reporter's failure billed 4 calls -- router, plan, two act rounds --
        and answered nothing. Total act-node spend when every layer fails is two
        text rounds, one native round and one retrieval fallback, so the query
        costs 6 calls rather than 4. Both extra calls buy an attempt at an
        answer; if either succeeds the user gets one where they previously got
        none. Pinned here so a later change to any one layer has to face the sum.
        """
        plain, bound = _escalating_llm()
        bound.ainvoke = AsyncMock(return_value=_dead_end_reply())
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        assert plain.ainvoke.await_count + bound.ainvoke.await_count == 4

    @pytest.mark.asyncio
    async def test_a_plain_empty_reply_does_not_change_the_protocol(
        self, text_mode, tool_mode_not_chosen
    ):
        """Only the documented signature escalates, not any empty answer."""
        plain, _bound = _escalating_llm()
        plain.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        plain.bind_tools.assert_not_called()
        assert result["messages"][0].content == _NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_it_is_kept_for_the_process(self, text_mode, tool_mode_not_chosen):
        """Otherwise every query pays the two wasted rounds again to rediscover it."""
        plain, bound = _escalating_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
            plain.ainvoke.reset_mock()
            await act_node(_state())
        # Second query went straight to native: no text rounds at all.
        plain.ainvoke.assert_not_awaited()
        assert bound.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_it_is_announced_once_and_says_who_decides(
        self, text_mode, tool_mode_not_chosen, caplog
    ):
        plain, _bound = _escalating_llm()
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
        ):
            await act_node(_state())
            first = caplog.text.count("this process will use native from here")
            await act_node(_state())
        assert first == 1
        assert caplog.text.count("this process will use native from here") == 1
        assert "AORTA_CHAT_LLM_TOOL_MODE" in caplog.text
        # Names the protocol itself rather than sending the reader to the
        # startup banner, which `aorta chat ui` never prints (#468). Not
        # `aorta chat doctor` either: it reports extras, the backend, the index
        # and the model cache, but nothing about the tool protocol.
        assert "'text' protocol" in caplog.text
        assert "aorta chat doctor" not in caplog.text

    @pytest.mark.asyncio
    async def test_the_announcement_reports_only_what_was_observed(
        self, text_mode, tool_mode_not_chosen, caplog
    ):
        """It used to assert reasoning came back even when nothing said so.

        ``_REQUIRE_REASONING_CHANNEL`` is False, so the trigger fires on tokens
        alone far more often than on a populated channel -- and a log line that
        claims an unobserved fact is the kind that sends someone hunting for a
        reasoning field their gateway never sent.
        """
        plain, _bound = _escalating_llm()
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
        ):
            await act_node(_state())
        assert "105 output tokens spent on empty content" in caplog.text
        assert "side channel" not in caplog.text

    @pytest.mark.asyncio
    async def test_an_observed_reasoning_channel_is_named(
        self, text_mode, tool_mode_not_chosen, caplog
    ):
        plain, _bound = _escalating_llm()
        plain.ainvoke = AsyncMock(return_value=_dead_end_reply(reasoning="hmm"))
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
        ):
            await act_node(_state())
        assert "reasoning returned on a side channel" in caplog.text


class TestConcurrentQueriesShareTheSwitch:
    """Two requests already in flight, and who is owed the retry.

    ``aorta chat ui`` serves many browser sessions from one process, so two
    queries can be inside the text loop at once. Making the decision and the
    state transition the same act meant the second one read its own dead end as
    somebody else's business: it saw the switch already thrown, returned False,
    and took the degraded fallback without trying the protocol just chosen for
    it. The decision records nothing now, so the ordering cannot arise.
    """

    def test_a_second_dead_end_still_gets_the_retry(
        self, text_mode, tool_mode_not_chosen
    ):
        """Called directly, because an event loop will not schedule this on demand.

        The interleaving under test is both requests past the gates and then one
        of them committing -- which is not something two concurrent ``act_node``
        runs can be made to reproduce reliably.
        """
        assert nodes._escalate_to_native(_dead_end_reply()) is True
        nodes._commit_escalation("105 output tokens spent on empty content")
        assert nodes._escalate_to_native(_dead_end_reply()) is True

    def test_the_decision_records_nothing_by_itself(
        self, text_mode, tool_mode_not_chosen, caplog
    ):
        """Deciding to try native is not the same as having moved to it."""
        with caplog.at_level("WARNING"):
            assert nodes._escalate_to_native(_dead_end_reply()) is True
        assert nodes._resolved_tool_mode() == "text"
        assert caplog.text == ""

    def test_an_explicit_protocol_still_comes_first(self, text_mode):
        """Restructuring the gates must not have dropped the one that matters."""
        assert nodes._escalate_to_native(_dead_end_reply()) is False

    def test_a_reply_without_the_signature_still_comes_first(
        self, text_mode, tool_mode_not_chosen
    ):
        assert nodes._escalate_to_native(AIMessage(content="")) is False


class TestAnEndpointThatRefusesNative:
    """The escalation is speculative, so it must not make the query worse.

    A stock local vLLM started without ``--enable-auto-tool-choice`` rejects any
    request carrying ``tools`` -- and its user is exactly who the escalation
    targets, because never setting ``llm_tool_mode`` is what makes them eligible.
    """

    @staticmethod
    def _refusing_llm():
        plain = MagicMock()
        plain.ainvoke = AsyncMock(return_value=_dead_end_reply())
        bound = MagicMock()
        bound.ainvoke = AsyncMock(
            side_effect=RuntimeError(
                "400: 'auto' tool choice requires --enable-auto-tool-choice"
            )
        )
        plain.bind_tools = MagicMock(return_value=bound)
        return plain, bound

    @pytest.mark.asyncio
    async def test_the_refusal_does_not_escape_the_graph(
        self, text_mode, tool_mode_not_chosen
    ):
        """Unhandled it replaced a poor answer with a traceback out of the run."""
        plain, _bound = self._refusing_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        assert result["messages"][0].content == _NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_the_switch_is_never_committed(self, text_mode, tool_mode_not_chosen):
        """Thrown up front, it sent every later query into the same refusal."""
        plain, _bound = self._refusing_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        assert nodes._resolved_tool_mode() == "text"

    @pytest.mark.asyncio
    async def test_one_failure_does_not_strand_the_process(
        self, text_mode, tool_mode_not_chosen
    ):
        """A timeout and a permanent refusal raise the same way here.

        The broad catch cannot tell them apart, so writing native off on the
        first failure would let one bad minute keep a long-lived UI server on
        the protocol its model cannot drive.
        """
        plain, bound = self._refusing_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        assert nodes._escalate_to_native(_dead_end_reply()) is True
        assert bound.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_a_repeating_failure_writes_native_off(
        self, text_mode, tool_mode_not_chosen
    ):
        """What tells a refusal from a blip is that it repeats."""
        plain, bound = self._refusing_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            for _ in range(nodes._MAX_NATIVE_FAILURES + 2):
                await act_node(_state())
        assert bound.ainvoke.await_count == nodes._MAX_NATIVE_FAILURES

    @pytest.mark.asyncio
    async def test_the_failure_is_counted_out_loud_not_diagnosed(
        self, text_mode, tool_mode_not_chosen, caplog
    ):
        """The log says which attempt this was rather than naming a cause.

        Calling it a refusal on the strength of one broad ``except`` would be
        the same overstatement the escalation warning was corrected for.
        """
        plain, _bound = self._refusing_llm()
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
        ):
            await act_node(_state())
        assert f"attempt 1 of {nodes._MAX_NATIVE_FAILURES}" in caplog.text
        assert "a later query may try again" in caplog.text
        assert "--enable-auto-tool-choice" in caplog.text

    @pytest.mark.asyncio
    async def test_a_loop_that_ran_a_tool_keeps_its_trace(
        self, text_mode, tool_mode_not_chosen
    ):
        """The label depends on it, and the sentinel used to drop it.

        One tool call, then two silent rounds: tools *did* run, so the fallback
        must not tell the user it could not use any.
        """
        from aorta.chat.graph.nodes import _DEGRADED_ANSWER_PREFIX

        plain, _bound = self._refusing_llm()
        plain.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content='ACTION: list_files(path=".")'),
                _dead_end_reply(),
                _dead_end_reply(),
            ]
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
        ):
            result = await act_node(_state())
        assert _DEGRADED_ANSWER_PREFIX not in result["messages"][0].content
        assert result["tool_trace"]


class TestAnEndpointThatAnswersNothingOnNativeEither:
    """The other way the retry fails: it returns, and it says nothing.

    ``_act_native`` hands back a state update on every path out, including the
    two that gave up, so "the call returned" was never evidence the protocol
    works -- and committing on it moved the whole process onto a protocol that
    had answered nothing. Reading the answer back off the result does not work
    either: ``_abandoned_result`` blanks ``command_output`` for the critic's
    sake and the synthesis path substitutes the give-up notice for empty text,
    so the shapes collide. Hence the explicit signal.
    """

    @staticmethod
    def _silent_llm():
        """Dead-ends on text, and is just as silent on the native retry."""
        plain, bound = _escalating_llm()
        bound.ainvoke = AsyncMock(return_value=_dead_end_reply())
        return plain, bound

    @pytest.mark.asyncio
    async def test_the_switch_is_never_committed(self, text_mode, tool_mode_not_chosen):
        plain, _bound = self._silent_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        assert nodes._resolved_tool_mode() == "text"

    @pytest.mark.asyncio
    async def test_the_next_query_still_starts_on_text(
        self, text_mode, tool_mode_not_chosen
    ):
        """The behavioural half: committing here skipped text for every query."""
        plain, _bound = self._silent_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
            plain.ainvoke.reset_mock()
            await act_node(_state())
        # Text rounds ran again, so the protocol did not move under the user.
        assert plain.ainvoke.await_count >= _MAX_UNPRODUCTIVE_ROUNDS

    @pytest.mark.asyncio
    async def test_a_silent_retry_spends_the_same_budget_as_a_failed_one(
        self, text_mode, tool_mode_not_chosen
    ):
        """Otherwise the retry is billed on every query, forever.

        Nothing was learned about the endpoint, so there is no reason to keep
        paying a native round to rediscover it -- which is the same argument
        ``_MAX_UNPRODUCTIVE_ROUNDS`` exists for one level down.
        """
        plain, bound = self._silent_llm()
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            for _ in range(nodes._MAX_NATIVE_FAILURES + 2):
                await act_node(_state())
        assert bound.ainvoke.await_count == nodes._MAX_NATIVE_FAILURES

    @pytest.mark.asyncio
    async def test_it_is_reported_as_an_attempt_and_not_as_a_config_fault(
        self, text_mode, tool_mode_not_chosen, caplog
    ):
        """The endpoint served the request, so naming a server flag would mislead."""
        plain, _bound = self._silent_llm()
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
        ):
            await act_node(_state())
        assert "no answer and no tool call either" in caplog.text
        assert f"attempt 1 of {nodes._MAX_NATIVE_FAILURES}" in caplog.text
        assert "this process will use native from here" not in caplog.text
        assert "--enable-auto-tool-choice" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_text_loop_that_ran_a_tool_keeps_its_trace(
        self, text_mode, tool_mode_not_chosen
    ):
        """The label is about the query, not about the protocol that gave up last.

        ``TestAnEndpointThatRefusesNative`` pins this for the raising path,
        where the caller passes the text trace to ``_abandoned_result`` itself.
        The silent path builds its result *inside* the native loop, whose own
        trace is empty by construction -- so without the trace being threaded
        down, a query that had already run a tool under ``text`` came back
        labelled "I could not use my tools for this question".
        """
        from aorta.chat.graph.nodes import _DEGRADED_ANSWER_PREFIX

        plain, _bound = self._silent_llm()
        plain.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content='ACTION: list_files(path=".")'),
                _dead_end_reply(),
                _dead_end_reply(),
            ]
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
        ):
            result = await act_node(_state())
        assert result["tool_trace"]
        assert _DEGRADED_ANSWER_PREFIX not in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_the_give_up_answer_costs_one_fallback_call_not_two(
        self, text_mode, tool_mode_not_chosen
    ):
        """Threading the trace must not become a second `_abandoned_result`.

        The native loop has already built one by the time the caller sees the
        outcome, so rebuilding it with the right trace would bill the retrieval
        fallback twice on the query that has already paid the most.
        """
        plain, _bound = self._silent_llm()
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch(
                "aorta.chat.graph.nodes._fallback_retrieval_answer",
                new=AsyncMock(return_value=""),
            ) as fallback,
        ):
            await act_node(_state())
        assert fallback.await_count == 1

    @pytest.mark.asyncio
    async def test_a_tool_call_commits_even_when_the_synthesis_is_empty(
        self, text_mode, tool_mode_not_chosen
    ):
        """A tool call *is* the protocol working, whatever happened after it.

        The model drove ``tools`` successfully; that the synthesis which
        followed came back empty is a different failure, and refusing to commit
        on it would make every later query pay the text rounds again to
        rediscover a protocol already shown to work.
        """
        plain, bound = self._silent_llm()
        bound.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("list_files", {"path": "."})],
                ),
                _dead_end_reply(),
            ]
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
        ):
            await act_node(_state())
        assert nodes._resolved_tool_mode() == "native"


class TestTheDegradedRetrievalFallback:
    """The act loop abandoning is survivable, and the reporter proved it.

    The same information need, asked twice a minute apart against one model and
    one index: 4 billed calls and nothing through the ``action`` route, 2 calls
    and a correct answer naming both files through ``question``. The index,
    retrieval and model were all fine -- the routing decision was the sole
    determinant of success -- so the fallback's output is demonstrated rather
    than hoped for.
    """

    @staticmethod
    def _llm(fallback_text: str):
        """Dead-ends every act round, answers the tool-free fallback call."""
        replies = [AIMessage(content=""), AIMessage(content="")]
        replies.append(AIMessage(content=fallback_text))
        fake = MagicMock()
        fake.ainvoke = AsyncMock(side_effect=replies)
        return fake

    @pytest.mark.asyncio
    async def test_it_answers_instead_of_dead_ending(self, text_mode):
        fake = self._llm("The TokenSpeed docs are docs/tokenspeed.md.")
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        assert "docs/tokenspeed.md" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_the_answer_is_labelled_as_degraded(self, text_mode):
        """Unlabelled, this would be #433's defect in mirror image."""
        from aorta.chat.graph.nodes import _DEGRADED_ANSWER_PREFIX

        fake = self._llm("The TokenSpeed docs are docs/tokenspeed.md.")
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        reply = result["messages"][0].content
        assert reply.startswith(_DEGRADED_ANSWER_PREFIX)
        assert "could not use my tools" in reply

    @pytest.mark.asyncio
    async def test_the_label_names_no_internals_either(self):
        """Same rule as the give-up message: it lands in the answer slot."""
        from aorta.chat.graph.nodes import _DEGRADED_ANSWER_PREFIX

        for leak in ("AORTA_CHAT", "ACTION:", "act loop", "retrieval", "native"):
            assert leak.lower() not in _DEGRADED_ANSWER_PREFIX.lower()

    @pytest.mark.asyncio
    async def test_it_is_capped_at_one_attempt(self, text_mode):
        fake = self._llm("An answer from context.")
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            await act_node(_state())
        assert fake.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS + 1

    @pytest.mark.asyncio
    async def test_it_cannot_re_enter_the_act_loop(self, text_mode):
        """Which is what returns the call-count problem the cap exists for.

        ``command_output`` is what the critic judges, and it stays empty -- so
        the critic returns no feedback and the graph ends. An answer with no tool
        output behind it would otherwise be rejected as ungrounded and sent
        straight back into the loop that just gave up.
        """
        from aorta.chat.graph.nodes import critic_node

        fake = self._llm("An answer from context.")
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        assert result.get("command_output") == ""

        # Subscripted, not ``.get``: ``route_after_critic`` reads this with
        # ``state.get(...)``, so a missing key is indistinguishable from ``None``
        # and ``.get(...) is None`` would pass vacuously if the critic stopped
        # returning the field at all.
        verdict = await critic_node({**_state(), **result, "iteration": 0})
        assert verdict["critic_feedback"] is None

    @pytest.mark.asyncio
    async def test_a_fallback_that_also_says_nothing_reports_the_dead_end(
        self, text_mode
    ):
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        assert result["messages"][0].content == _NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_a_fallback_that_raises_still_reports_the_dead_end(self, text_mode):
        """This call is an addition to a path that previously made none.

        The likeliest reason the act loop dead-ended is an unwell backend, which
        is exactly when this extra call raises -- and letting it through would
        turn the give-up notice into a traceback on the query that needed the
        notice most.
        """
        fake = MagicMock()
        fake.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content=""),
                AIMessage(content=""),
                RuntimeError("connection reset by peer"),
            ]
        )
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        assert result["messages"][0].content == _NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_a_failed_fallback_is_diagnosable(self, text_mode, caplog):
        fake = MagicMock()
        fake.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content=""),
                AIMessage(content=""),
                RuntimeError("connection reset by peer"),
            ]
        )
        with (
            caplog.at_level("WARNING"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
        ):
            await act_node(_state())
        assert "connection reset by peer" in caplog.text

    @pytest.mark.asyncio
    async def test_a_loop_that_did_run_a_tool_is_not_labelled_as_toolless(
        self, native_mode
    ):
        """The label has to stay true, so this case gets the plain notice.

        One tool call, then silence: tools *did* run, so "I could not use my
        tools" would be false -- and an inaccurate label is the exact thing this
        fallback is careful about.
        """
        from aorta.chat.graph.nodes import _DEGRADED_ANSWER_PREFIX

        plain, _bound = _fake_llm(
            [
                AIMessage(content="", tool_calls=[_tool_call("list_files", {})]),
                AIMessage(content=""),
                AIMessage(content=""),
            ],
            final_text="",
        )
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
        ):
            result = await act_node(_state())
        assert _DEGRADED_ANSWER_PREFIX not in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_the_fallback_is_reported_in_the_log(self, text_mode, caplog):
        fake = self._llm("An answer from context.")
        with (
            caplog.at_level("INFO"),
            patch("aorta.chat.graph.nodes._get_llm", return_value=fake),
        ):
            await act_node(_state())
        assert "Answered from retrieved context" in caplog.text


class TestWastedCallGuards:
    @pytest.mark.asyncio
    async def test_the_text_loop_gives_up_instead_of_spending_the_budget(self, text_mode):
        """gpt-oss burned 11 calls here; the cap is 2 unproductive rounds.

        Plus the single retrieval fallback that now follows an abandoned loop --
        which is where the answer comes from when it works, and is capped at one
        attempt. Three calls in total, not a budget's worth.
        """
        fake = MagicMock()
        fake.ainvoke = AsyncMock(return_value=AIMessage(content=""))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=fake):
            result = await act_node(_state())
        assert fake.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS + 1
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
        fake.ainvoke = AsyncMock(return_value=AIMessage(content="AORTA uses Python 3.10+."))
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
    async def test_the_native_loop_also_stops_when_nothing_comes_back(self, native_mode):
        plain, bound = _fake_llm([AIMessage(content="")] * 8, final_text="")
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        assert bound.ainvoke.await_count == _MAX_UNPRODUCTIVE_ROUNDS
        assert result["messages"][0].content == _NO_ANSWER_MSG


class TestABackendThatFallsOverPartWayThroughNative:
    """A failure *after* native worked is not a failure of native.

    The escalated retry's progress lives in the loop's locals, so an exception
    used to discard it: the tool results went missing, and the endpoint was
    charged a native failure on the strength of a round that had demonstrably
    driven ``tools``. Two of those wrote native off for the process.
    """

    @staticmethod
    def _llm_that_breaks_after_one_tool_call():
        """Text dead-ends; native calls a tool, then the backend goes away."""
        plain = MagicMock()
        plain.ainvoke = AsyncMock(return_value=_dead_end_reply())
        bound = MagicMock()
        bound.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="", tool_calls=[_tool_call("list_files", {"path": "."})]
                ),
                RuntimeError("503 Service Unavailable"),
            ]
        )
        plain.bind_tools = MagicMock(return_value=bound)
        return plain, bound

    @pytest.mark.asyncio
    async def test_the_native_tool_result_survives_the_failure(
        self, text_mode, tool_mode_not_chosen
    ):
        """The results were gathered; losing them loses the query's only material."""
        plain, _bound = self._llm_that_breaks_after_one_tool_call()
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py\nb.py"),
        ):
            result = await act_node(_state())
        assert any("a.py" in entry for entry in result["tool_trace"])

    @pytest.mark.asyncio
    async def test_the_answer_is_not_labelled_as_toolless(
        self, text_mode, tool_mode_not_chosen
    ):
        """A tool ran, so the degraded label would be a false statement."""
        from aorta.chat.graph.nodes import _DEGRADED_ANSWER_PREFIX

        plain, _bound = self._llm_that_breaks_after_one_tool_call()
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
        ):
            result = await act_node(_state())
        assert _DEGRADED_ANSWER_PREFIX not in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_it_does_not_count_against_the_native_failure_budget(
        self, text_mode, tool_mode_not_chosen
    ):
        """Two transient 503s must not strand the process on ``text`` for good."""
        plain, _bound = self._llm_that_breaks_after_one_tool_call()
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
        ):
            await act_node(_state())
        assert nodes._escalation.native_failures == 0

    @pytest.mark.asyncio
    async def test_the_protocol_moves_because_tool_calling_demonstrably_worked(
        self, text_mode, tool_mode_not_chosen
    ):
        """The escalation asks one question, and this run answered it yes.

        Deliberately different from ``TestAnEndpointThatRefusesNative``, where
        the refusal arrives *before* any tool call and so proves the opposite.
        The two cases reach the same ``except`` and must not reach the same
        conclusion.
        """
        plain, _bound = self._llm_that_breaks_after_one_tool_call()
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
        ):
            await act_node(_state())
        assert nodes._escalation.escalated is True

    @pytest.mark.asyncio
    async def test_a_refusal_before_any_tool_call_still_counts_as_a_failure(
        self, text_mode, tool_mode_not_chosen
    ):
        """The other half of the split, so the new branch cannot swallow both."""
        plain = MagicMock()
        plain.ainvoke = AsyncMock(return_value=_dead_end_reply())
        bound = MagicMock()
        bound.ainvoke = AsyncMock(side_effect=RuntimeError("400: tools not supported"))
        plain.bind_tools = MagicMock(return_value=bound)
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            await act_node(_state())
        assert nodes._escalation.native_failures == 1
        assert nodes._escalation.escalated is False

    @pytest.mark.asyncio
    async def test_a_backend_that_cannot_even_be_built_is_still_caught(
        self, text_mode, tool_mode_not_chosen
    ):
        """Only the calls *inside* the loop carry progress.

        The loop resolves the backend and binds the tool schemas before it makes
        any request, and those raise plainly rather than as a
        ``_NativeLoopError``. Narrowing the caller's ``except`` to the new
        exception alone would have put the traceback back on the query this
        whole path exists to keep an answer on.
        """
        plain = MagicMock()
        plain.ainvoke = AsyncMock(return_value=_dead_end_reply())
        plain.bind_tools = MagicMock(side_effect=RuntimeError("no tool schema"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=plain):
            result = await act_node(_state())
        assert result["messages"][0].content
        assert nodes._escalation.native_failures == 1


class TestTheFallbackDoesNotAskForToolOutputItCannotHave:
    """The retrieval fallback runs without tools, so its nudge must not want any.

    ``_ensure_ends_with_user`` appends a user turn whenever the conversation
    ends on an assistant one, which is exactly the shape a critic-triggered
    retry produces. The default nudge tells the model to ground every claim in
    tool output "obtained in this turn" -- an instruction this request makes
    impossible, on the one path guaranteed to receive it.
    """

    @staticmethod
    def _state_after_a_rejected_answer():
        state = _state()
        state["messages"] = [
            HumanMessage(content="find all mitigations"),
            AIMessage(content="A rejected answer."),
        ]
        state["critic_feedback"] = "Not grounded in any tool output."
        return state

    @pytest.mark.asyncio
    async def test_the_fallback_nudge_replaces_the_tool_grounding_one(self, text_mode):
        from aorta.chat.graph.nodes import (
            _FALLBACK_RETRY_NUDGE,
            _RETRY_NUDGE,
            _fallback_retrieval_answer,
        )

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="From the docs: ..."))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            answer = await _fallback_retrieval_answer(
                self._state_after_a_rejected_answer()
            )
        assert answer == "From the docs: ..."
        sent = [str(m.content) for m in llm.ainvoke.await_args.args[0]]
        assert _FALLBACK_RETRY_NUDGE in sent
        assert _RETRY_NUDGE not in sent

    @pytest.mark.asyncio
    async def test_the_trailing_turn_is_still_a_user_turn(self, text_mode):
        """The nudge's other job: models that disallow prefill reject the request."""
        from aorta.chat.graph.nodes import _fallback_retrieval_answer

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
            await _fallback_retrieval_answer(self._state_after_a_rejected_answer())
        assert isinstance(llm.ainvoke.await_args.args[0][-1], HumanMessage)


class TestTheExtraSendsDoNotRepeatTheRedactionNotice:
    """Two new outbound paths in one turn, and still one notice.

    The notice is owed once per session, not once per redacting turn, and this
    PR is the first thing to put three separate ``_send`` call sites inside a
    single act turn: the text rounds, the escalated native round, and the
    retrieval fallback. Each redacts, so each *could* have announced itself.
    Pinned here rather than in ``test_redaction.py`` because what is new is
    these paths, not the notice.
    """

    @pytest.mark.asyncio
    async def test_one_notice_covers_every_send_in_an_escalated_turn(
        self, text_mode, tool_mode_not_chosen
    ):
        import io

        from aorta.chat import redaction

        redaction.reset_session_notice()
        stderr = io.StringIO()

        async def plain_invoke(messages, **_kwargs):
            joined = " ".join(str(getattr(m, "content", "")) for m in messages)
            # Empty on the text protocol, fine on the tool-free fallback.
            return _dead_end_reply() if "ACTION:" in joined else AIMessage(content="ok")

        plain = MagicMock()
        plain.ainvoke = AsyncMock(side_effect=plain_invoke)
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=_dead_end_reply())
        plain.bind_tools = MagicMock(return_value=bound)
        state = _state("find mitigations in /home/cust7/secret/run.log")
        state["retrieved_context"] = "### /home/cust7/secret/run.log\n```\nx\n```"

        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch.object(redaction, "_notice_stream", lambda: stderr),
        ):
            await act_node(state)

        # The turn really did make the extra calls this test is about.
        assert plain.ainvoke.await_count + bound.ainvoke.await_count >= 3
        assert stderr.getvalue().count("aorta chat: redacted") == 1


class TestTheGiveUpLogDoesNotBlameTheRoundBudget:
    """The line an operator reads has to name the limit that was actually hit.

    A native loop that gathered results and *then* went quiet fell through to
    the budget message -- "hit its N-round budget with the model still calling
    tools" -- when neither half was true. It sent the reader to MAX_ACT_ROUNDS,
    the one knob that would not have helped.
    """

    @pytest.mark.asyncio
    async def test_a_loop_that_went_quiet_after_a_tool_call_says_so(
        self, native_mode, caplog
    ):
        plain = MagicMock()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="", tool_calls=[_tool_call("list_files", {"path": "."})]
                ),
                AIMessage(content=""),
                AIMessage(content=""),
            ]
        )
        plain.ainvoke = AsyncMock(return_value=AIMessage(content="Final answer."))
        plain.bind_tools = MagicMock(return_value=bound)
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
            caplog.at_level(logging.WARNING),
        ):
            result = await act_node(_state())
        assert "gave up after" in caplog.text
        assert "tool result(s) already gathered" in caplog.text
        assert "round budget was not the limit" in caplog.text
        assert "still calling tools" not in caplog.text
        # The synthesis still runs -- only the log line was wrong.
        assert result["messages"][0].content == "Final answer."

    @pytest.mark.asyncio
    async def test_a_loop_that_really_ran_out_of_rounds_still_says_that(
        self, native_mode, caplog
    ):
        """The message this fix must not take away from the case it belongs to."""
        calls = [
            AIMessage(content="", tool_calls=[_tool_call("list_files", {"path": f"{i}"})])
            for i in range(12)
        ]
        plain, _bound = _fake_llm(calls, final_text="Final answer.")
        with (
            patch("aorta.chat.graph.nodes._get_llm", return_value=plain),
            patch("aorta.chat.graph.nodes._execute_tool", return_value="a.py"),
            caplog.at_level(logging.WARNING),
        ):
            await act_node(_state())
        assert "still calling tools" in caplog.text
        assert "Raise MAX_ACT_ROUNDS" in caplog.text
        assert "gave up after" not in caplog.text
