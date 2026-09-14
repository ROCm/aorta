"""Progress is not the answer, and must not be able to cost it.

Streaming replaced ``ainvoke`` with ``astream`` so the browser could be told
what was happening during a tool call that runs for minutes. The two awaits
that told it were unguarded, so anything the callback raised came out of
``astream`` and ended the query.

The callback the UI passes talks to Chainlit, which fails once the session is
gone. So a user who closed the tab during a five-minute cluster job took the
run down with it -- the job kept burning a GPU node with nobody left to receive
the verdict, and the history recorded a question with no answer. The ``ainvoke``
path this replaced had no such coupling, which made it a regression for anyone
passing a callback.

Cancellation is deliberately not swallowed. It is how a caller stops this on
purpose, and catching it here would make the run unstoppable.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

from aorta.chat import session


def _graph(chunks):
    """A graph that streams *chunks* and nothing else."""

    async def astream(initial, stream_mode=None):
        for chunk in chunks:
            yield chunk

    return type("FakeGraph", (), {"astream": staticmethod(astream)})()


_ANSWER = ("values", {"messages": [AIMessage(content="the kernel races on line 4")]})
_CHUNKS = [
    ("updates", {"router": {"route": "action"}}),
    ("custom", {"tool": "triage_kernel_source"}),
    ("updates", {"act": {}}),
    _ANSWER,
]


@pytest.fixture()
def streaming(monkeypatch):
    monkeypatch.setattr(session, "agent_graph", _graph(_CHUNKS))


class TestACallbackThatFailsDoesNotCostTheAnswer:
    async def test_the_reply_still_arrives(self, streaming):
        async def dead(name, payload):
            raise RuntimeError("Session not found")

        reply, _history, _result = await session.invoke_agent("why?", [], on_step=dead)

        assert reply == "the kernel races on line 4"

    async def test_the_history_records_the_exchange(self, streaming):
        async def dead(name, payload):
            raise RuntimeError("Session not found")

        _reply, history, _result = await session.invoke_agent("why?", [], on_step=dead)

        assert len(history) == 2, "the question should not be left unanswered"

    async def test_the_result_is_still_returned(self, streaming):
        async def dead(name, payload):
            raise RuntimeError("Session not found")

        _reply, _history, result = await session.invoke_agent("why?", [], on_step=dead)

        assert result.get("messages"), "the caller still needs the state"

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("session gone"), ValueError("bad payload"), KeyError("sid")],
        ids=["runtime", "value", "key"],
    )
    async def test_whatever_the_callback_raises(self, streaming, error):
        async def dead(name, payload):
            raise error

        reply, _history, _result = await session.invoke_agent("q", [], on_step=dead)

        assert reply == "the kernel races on line 4"

    async def test_a_callback_that_fails_once_does_not_stop_the_rest(self, streaming):
        seen: list[str] = []

        async def flaky(name, payload):
            seen.append(name)
            if len(seen) == 1:
                raise RuntimeError("the first one failed")

        await session.invoke_agent("q", [], on_step=flaky)

        assert len(seen) == 3, "every step should still be offered"


class TestItIsSaidOnceRatherThanPerChunk:
    async def test_a_dead_session_logs_a_single_warning(self, streaming, caplog):
        async def dead(name, payload):
            raise RuntimeError("Session not found")

        with caplog.at_level(logging.WARNING, logger="aorta.chat.session"):
            await session.invoke_agent("q", [], on_step=dead)

        warnings = [r for r in caplog.records if "Progress reporting failed" in r.message]
        assert len(warnings) == 1, "a dead session fails every chunk; say so once"

    async def test_the_warning_says_the_answer_is_unaffected(self, streaming, caplog):
        async def dead(name, payload):
            raise RuntimeError("Session not found")

        with caplog.at_level(logging.WARNING, logger="aorta.chat.session"):
            await session.invoke_agent("q", [], on_step=dead)

        assert "answer is unaffected" in caplog.text


class TestCancellationStillWorks:
    """A blanket except would make the run impossible to stop."""

    async def test_it_propagates(self, streaming):
        async def cancelling(name, payload):
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await session.invoke_agent("q", [], on_step=cancelling)


class TestTheOrdinaryPathIsUnchanged:
    async def test_a_working_callback_sees_every_step(self, streaming):
        seen: list[str] = []

        async def ok(name, payload):
            seen.append(name)

        reply, _history, _result = await session.invoke_agent("q", [], on_step=ok)

        assert seen == ["router", "tool", "act"]
        assert reply == "the kernel races on line 4"

    async def test_no_callback_uses_the_awaited_path(self, monkeypatch):
        """The CLI passes none, and must not be routed through streaming."""
        called: list[str] = []

        async def ainvoke(initial):
            called.append("ainvoke")
            return {"messages": [AIMessage(content="answered")]}

        monkeypatch.setattr(
            session, "agent_graph", type("G", (), {"ainvoke": staticmethod(ainvoke)})()
        )
        reply, _history, _result = await session.invoke_agent("q", [])

        assert called == ["ainvoke"]
        assert reply == "answered"
