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


@pytest.fixture(autouse=True)
def _no_sticky_escalation():
    """Undo any auto-escalation, which is process-wide by design.

    ``_escalated_to_native`` is deliberately a module global: "keep it for the
    session" is the point, and a test process is one session for every test in
    it. Without this an escalation in one test silently puts the next one on the
    native protocol.
    """
    nodes.reset_tool_mode_escalation()
    yield
    nodes.reset_tool_mode_escalation()


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
    not talk about model". The specifics are still logged, and ``aorta chat
    doctor`` reports the resolved protocol; the answer slot gets one plain
    sentence and a command to run.
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


def _dead_end_reply(output_tokens: int = 105, reasoning: str | None = None):
    """The signature docs/chat/providers.md calls distinctive.

    Empty ``content`` with a non-zero output-token count: the model spent tokens
    and returned no text, so the text went somewhere this protocol cannot read.
    Taken from the reporter's transcript -- "round 1 produced no text despite
    105 output tokens".
    """
    return AIMessage(
        content="",
        additional_kwargs={"reasoning": reasoning} if reasoning else {},
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

    **The live check behind this was not run.** The register's decision was to
    sharpen the trigger with "and the reasoning channel is populated", which
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
        """The budget the register was explicit about: 4 wasted calls, not more.

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
    async def test_it_is_kept_for_the_session(self, text_mode, tool_mode_not_chosen):
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
            first = caplog.text.count("Retrying this query on native")
            await act_node(_state())
        assert first == 1
        assert caplog.text.count("Retrying this query on native") == 1
        assert "AORTA_CHAT_LLM_TOOL_MODE" in caplog.text
        assert "aorta chat doctor" in caplog.text


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
        assert result["command_output"] == ""

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
