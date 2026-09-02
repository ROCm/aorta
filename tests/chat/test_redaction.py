"""The egress redaction gate (Decision 16).

Three properties matter here and none of them is about the scrubbers, which
belong to ``aorta.probe.redaction`` and are tested with the bundle:

1. It is **on by default**, and off only when asked.
2. It rewrites the copy that leaves the machine and *not* the conversation
   state, so the user still sees their own paths echoed back.
3. The notice fires **once**, on stderr, and names both what went and how to
   stop it. A gate nobody is told about is a gate that gets blamed for a wrong
   answer.
"""

from __future__ import annotations

import io

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aorta.chat import redaction
from aorta.chat.config import configure, reset_settings

CUSTOMER_TEXT = (
    "the run under /home/cust7/models/llama-70b failed on host 10.42.7.9 "
    "and the replica at [2001:db8::1] hung"
)


@pytest.fixture(autouse=True)
def _clean_session():
    """Every test starts with the notice unshown and the default settings."""
    redaction.reset_session_notice()
    reset_settings()
    yield
    redaction.reset_session_notice()
    reset_settings()


class TestDefaults:
    def test_redaction_is_on_without_configuration(self):
        """The default has to be the safe one; nobody opts in to not leaking."""
        text, summary = redaction.redact_text(CUSTOMER_TEXT)
        assert "/home/cust7" not in text
        assert "10.42.7.9" not in text
        assert summary.total == 3

    def test_no_redact_passes_text_through_byte_for_byte(self):
        configure(redact=False)
        text, summary = redaction.redact_text(CUSTOMER_TEXT)
        assert text == CUSTOMER_TEXT
        assert not summary

    def test_placeholders_replace_each_kind(self):
        text, _ = redaction.redact_text(CUSTOMER_TEXT)
        assert "<PATH:0>" in text
        assert "<IPV4:0>" in text
        assert "<IPV6:0>" in text


class TestMessageRewriting:
    def test_conversation_state_is_not_mutated(self):
        """The user's own history must keep their paths; only the wire copy loses them."""
        original = HumanMessage(content=CUSTOMER_TEXT)
        messages = [original]
        out, _ = redaction.redact_messages(messages)
        assert original.content == CUSTOMER_TEXT
        assert out[0] is not original
        assert "/home/cust7" not in out[0].content

    def test_unchanged_messages_keep_their_identity(self):
        """A no-op scrub must not clone.

        LangChain pairs a ToolMessage with the AIMessage that requested it, so
        needlessly rebuilding untouched messages risks that bookkeeping for no
        gain.
        """
        clean = SystemMessage(content="You are an assistant.")
        out, summary = redaction.redact_messages([clean])
        assert out[0] is clean
        assert not summary

    def test_non_string_content_is_passed_through_untouched(self):
        """Multimodal / tool-call blocks are not modelled, so they are not rewritten.

        Coercing one to text would corrupt the request rather than protect
        anything.
        """
        structured = AIMessage(content=[{"type": "text", "text": "/home/cust7/x"}])
        out, summary = redaction.redact_messages([structured])
        assert out[0] is structured
        assert not summary

    def test_counts_accumulate_across_the_whole_message_list(self):
        messages = [
            HumanMessage(content="/home/a/b failed"),
            ToolMessage(content="also /home/c/d and 10.0.0.1", tool_call_id="1"),
        ]
        _, summary = redaction.redact_messages(messages)
        assert summary.paths == 2
        assert summary.ipv4 == 1

    def test_no_redact_returns_the_same_list_object(self):
        configure(redact=False)
        messages = [HumanMessage(content=CUSTOMER_TEXT)]
        out, _ = redaction.redact_messages(messages)
        assert out is messages


class TestNoticeIsSessionLocal:
    """The notice state must not be shared by every session in the process.

    Under ``aorta chat ui`` one process serves many browser sessions. With a
    module-level flag, the first session to trigger a redaction consumed the
    disclosure for everybody, and every later user was silently redacted -- the
    one thing Decision 16 says must not happen. The CLI is one session per
    process, so it keeps using the process-wide state and binds nothing.
    """

    def test_two_bound_sessions_are_each_told(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        first, second = redaction.NoticeState(), redaction.NoticeState()
        one, two = io.StringIO(), io.StringIO()
        with redaction.use_notice_state(first):
            assert redaction.emit_notice_once(summary, stream=one) is True
        with redaction.use_notice_state(second):
            assert redaction.emit_notice_once(summary, stream=two) is True
        assert one.getvalue().count("aorta chat: redacted") == 1
        assert two.getvalue().count("aorta chat: redacted") == 1

    def test_one_session_is_still_told_only_once(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        stream = io.StringIO()
        for _ in range(3):
            with redaction.use_notice_state(state):
                redaction.emit_notice_once(summary, stream=stream)
        assert stream.getvalue().count("aorta chat: redacted") == 1

    def test_a_bound_session_does_not_consume_the_process_notice(self):
        """A UI session must leave the CLI fallback state untouched."""
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        with redaction.use_notice_state(redaction.NoticeState()):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert redaction.current_notice_state().emitted is False

    def test_the_binding_is_undone_on_the_way_out(self):
        state = redaction.NoticeState()
        with redaction.use_notice_state(state):
            assert redaction.current_notice_state() is state
        assert redaction.current_notice_state() is not state

    def test_state_survives_the_tasks_one_turn_spans(self):
        """The reason this is a mutable object and not a ``ContextVar[bool]``.

        A task copies the context at creation and its writes do not propagate
        back, so a bare flag flipped inside a graph node would be forgotten by
        the next turn and the notice would fire every time. Shared state does
        not have that problem.
        """
        import asyncio

        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        stream = io.StringIO()

        async def turn() -> None:
            # A nested task is what LangGraph runs nodes in.
            await asyncio.create_task(
                asyncio.to_thread(redaction.emit_notice_once, summary, stream)
            )

        async def session() -> None:
            with redaction.use_notice_state(state):
                await turn()
                await turn()

        asyncio.run(session())
        assert stream.getvalue().count("aorta chat: redacted") == 1


class TestNoticeDelivery:
    """A front door that is not a terminal has to be able to render it itself."""

    def test_the_line_is_parked_for_the_caller_to_show(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        with redaction.use_notice_state(state):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert state.pending is not None
        assert "aorta chat: redacted" in state.pending

    def test_draining_it_yields_it_exactly_once(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        with redaction.use_notice_state(state):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert redaction.take_pending_notice(state) is not None
        assert redaction.take_pending_notice(state) is None

    def test_nothing_is_parked_when_nothing_was_redacted(self):
        state = redaction.NoticeState()
        _, summary = redaction.redact_text("what does the router node do?")
        with redaction.use_notice_state(state):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert redaction.take_pending_notice(state) is None


class TestNotice:
    def test_notice_names_what_went_and_how_to_disable_it(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        line = redaction.notice_line(summary)
        assert "filesystem path" in line
        assert "IPv4" in line
        assert "--no-redact" in line
        assert "redact = false" in line

    def test_notice_fires_once_per_session(self):
        stream = io.StringIO()
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        assert redaction.emit_notice_once(summary, stream=stream) is True
        assert redaction.emit_notice_once(summary, stream=stream) is False
        assert stream.getvalue().count("aorta chat: redacted") == 1

    def test_nothing_is_announced_when_nothing_was_redacted(self):
        """A session of path-free prompts must not be told about a redaction."""
        stream = io.StringIO()
        _, summary = redaction.redact_text("what does the router node do?")
        assert redaction.emit_notice_once(summary, stream=stream) is False
        assert stream.getvalue() == ""

    def test_redact_for_send_writes_the_notice_to_stderr(self, capfd):
        """stderr, so --json and --plain stay machine-parseable on stdout.

        ``capfd`` rather than ``capsys``: the notice deliberately targets
        ``sys.__stderr__``, because quiet mode -- the default -- repoints
        ``sys.stderr`` at ``os.devnull``. Only fd-level capture sees it, which
        is also the proof that it survives that repointing.
        """
        redaction.redact_for_send([HumanMessage(content=CUSTOMER_TEXT)])
        captured = capfd.readouterr()
        assert captured.out == ""
        assert "aorta chat: redacted" in captured.err

    def test_the_notice_survives_quiet_mode_repointing_stderr(self, capfd, monkeypatch):
        """The regression this guards is Decision 16's whole point.

        ``aorta chat`` without ``-v`` sends stderr to /dev/null before the first
        query. A notice written to ``sys.stderr`` would be silently discarded
        exactly when the user most needs it.
        """
        import os

        with open(os.devnull, "w") as devnull:
            monkeypatch.setattr(sys_module(), "stderr", devnull)
            redaction.redact_for_send([HumanMessage(content=CUSTOMER_TEXT)])
        assert "aorta chat: redacted" in capfd.readouterr().err


class TestSummaryWording:
    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            (redaction.RedactionSummary(), "nothing"),
            (redaction.RedactionSummary(paths=1), "1 filesystem path"),
            (redaction.RedactionSummary(paths=2), "2 filesystem paths"),
            (redaction.RedactionSummary(ipv4=1), "1 IPv4 address"),
            (redaction.RedactionSummary(ipv4=3), "3 IPv4 addresses"),
            (
                redaction.RedactionSummary(paths=1, ipv4=1),
                "1 filesystem path and 1 IPv4 address",
            ),
            (
                redaction.RedactionSummary(paths=1, ipv4=1, ipv6=2),
                "1 filesystem path, 1 IPv4 address and 2 IPv6 addresses",
            ),
        ],
    )
    def test_describe_reads_as_prose(self, summary, expected):
        assert summary.describe() == expected


class TestGraphChokepoint:
    async def test_every_node_send_goes_through_the_gate(self):
        """``_send`` is the single seam, so a node added later cannot bypass it."""
        from unittest.mock import AsyncMock, MagicMock

        from aorta.chat.graph import nodes

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        await nodes._send(llm, [HumanMessage(content=CUSTOMER_TEXT)])

        sent = llm.ainvoke.await_args.args[0]
        assert "/home/cust7" not in sent[0].content

    def test_no_node_calls_ainvoke_directly(self):
        """The gate is only worth anything if it is the only way out.

        Asserted against the source rather than by mocking, because the failure
        being guarded against is a *new* call site, which no existing test would
        exercise.
        """
        import ast
        from pathlib import Path

        source = Path(nodes_path()).read_text(encoding="utf-8")
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Attribute) or node.attr != "ainvoke":
                continue
            # `_send` itself is the sanctioned caller.
            offenders.append(node.lineno)
        # One permitted occurrence: the call inside `_send`.
        assert len(offenders) == 1, (
            f"graph/nodes.py calls .ainvoke at lines {offenders}; every outbound "
            "call must go through _send so the redaction gate applies."
        )


def nodes_path() -> str:
    from aorta.chat.graph import nodes

    return nodes.__file__


def sys_module():
    import sys

    return sys
