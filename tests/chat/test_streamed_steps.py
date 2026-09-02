"""Showing the work, and not changing the answer by showing it.

A diagnostic tool holds the act node for minutes. Steps that appear only when a
node finishes therefore say nothing for the entire part of the run a user is
actually waiting through, which is indistinguishable from a hang -- and was
reported as one.

So there are two mechanisms, and both are tested here: a step per node as it
completes, and a tool announcing itself *before* it blocks. The second is the
one that matters, and the ordering assertion is the whole point of it.

The other half is that observation must not change the observed. Passing no
callback has to leave the existing path exactly as it was, because the CLI uses
it and a streamed graph and an awaited one are not obviously the same thing.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aorta.chat.graph.nodes import _execute_tool_async
from aorta.chat.session import invoke_agent


def _fake_llm(*replies: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[AIMessage(content=r) for r in replies])
    return llm


# ── a tool announces itself before it blocks ──────────────────────────────


async def test_a_tool_is_announced_before_it_runs():
    """The ordering is the feature.

    An event after the call would be a log line; an event before it is what
    lets a UI name what it is waiting on while it waits.
    """
    order: list[str] = []

    def _writer(payload: dict) -> None:
        order.append(f"announced:{payload['tool']}")

    def _slow(name: str, kwargs: dict) -> str:
        order.append(f"ran:{name}")
        return "done"

    with (
        patch("aorta.chat.graph.nodes.get_stream_writer", return_value=_writer),
        patch("aorta.chat.graph.nodes._execute_tool", _slow),
    ):
        await _execute_tool_async("triage_workload", {"source": "x"})

    assert order == ["announced:triage_workload", "ran:triage_workload"]


async def test_a_tool_runs_off_the_event_loop():
    """A blocking tool must not stall the loop.

    With the diagnostic tools this is not a nicety: they block for minutes, and
    inline that stops Chainlit answering at all, so the browser reports the
    backend as unreachable rather than busy.
    """
    ran_on = {}

    def _record(name: str, kwargs: dict) -> str:
        ran_on["thread"] = __import__("threading").current_thread().name
        return "done"

    with (
        patch("aorta.chat.graph.nodes.get_stream_writer", side_effect=RuntimeError),
        patch("aorta.chat.graph.nodes._execute_tool", _record),
    ):
        await _execute_tool_async("list_files", {})

    assert ran_on["thread"] != __import__("threading").current_thread().name


async def test_no_stream_to_announce_to_is_not_an_error():
    """The CLI runs the same node without streaming."""
    with (
        patch("aorta.chat.graph.nodes.get_stream_writer", side_effect=RuntimeError),
        patch("aorta.chat.graph.nodes._execute_tool", lambda n, k: "fine"),
    ):
        assert await _execute_tool_async("list_files", {}) == "fine"


# ── the callback, and the answer it must not change ───────────────────────


@pytest.fixture
def scripted_graph():
    """A graph run that reaches an answer without a model or a network."""

    async def _run(on_step=None):
        # Awaited inside the patch: returning the coroutine would leave the
        # context before the graph runs, and the nodes would reach a real model.
        with patch(
            "aorta.chat.graph.nodes._get_llm",
            side_effect=lambda **kw: _fake_llm("question", "the answer"),
        ):
            return await invoke_agent("what does the router do?", [], on_step=on_step)

    return _run


async def test_the_callback_is_told_which_node_produced_what(scripted_graph):
    seen: list[tuple[str, set]] = []

    async def record(node: str, delta: dict) -> None:
        seen.append((node, set(delta)))

    await scripted_graph(on_step=record)
    nodes = [node for node, _ in seen]
    assert "router" in nodes
    assert all(isinstance(keys, set) for _, keys in seen)


async def test_streaming_returns_what_awaiting_returns(scripted_graph):
    """Observation must not change the observed.

    ``astream`` accumulates state differently from ``ainvoke``; taking the last
    "values" chunk is what makes them agree, and this is the assertion that
    keeps that true.
    """
    awaited_reply, _, awaited_state = await scripted_graph()
    streamed_reply, _, streamed_state = await scripted_graph(on_step=AsyncMock())

    assert streamed_reply == awaited_reply
    assert set(streamed_state) == set(awaited_state)
    assert "messages" in streamed_state


async def test_the_history_is_the_same_either_way(scripted_graph):
    _, awaited_history, _ = await scripted_graph()
    _, streamed_history, _ = await scripted_graph(on_step=AsyncMock())
    assert [type(m) for m in streamed_history] == [type(m) for m in awaited_history]


async def test_a_slow_callback_does_not_lose_events(scripted_graph):
    """Rendering a step is I/O; the run must wait for it rather than race it."""
    seen = []

    async def slow(node: str, delta: dict) -> None:
        await asyncio.sleep(0)
        seen.append(node)

    await scripted_graph(on_step=slow)
    assert seen, "no steps reached a callback that yielded"
