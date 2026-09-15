"""A tool step must stay open for as long as the tool runs.

Chainlit renders a step as completed once it carries an end timestamp, and
``async with cl.Step`` stamps one on the way out of the block. Wrapping only
the announcement in that block meant the step was closed before the tool had
started -- so the five-minute cluster job this feature exists to make visible
rendered as already finished, for the whole of the wait.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langgraph", reason="requires the chat extra")

from aorta.chat.graph import nodes


class Recorder:
    """Collects what the graph puts on the stream, in order."""

    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, payload):
        self.events.append(payload)

    def starts(self):
        return [e for e in self.events if not e.get("done")]

    def ends(self):
        return [e for e in self.events if e.get("done")]


@pytest.fixture()
def stream(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(nodes, "get_stream_writer", lambda: recorder)
    return recorder


async def run_tool(monkeypatch, body):
    """Drive _execute_tool_async with *body* standing in for the tool."""
    monkeypatch.setattr(nodes, "_execute_tool", lambda name, kwargs: body())
    return await nodes._execute_tool_async("triage_kernel_source", {})


async def test_a_completion_is_emitted(stream, monkeypatch):
    """The event the UI needs in order to close the step at the right time."""
    await run_tool(monkeypatch, lambda: "ok")
    assert len(stream.ends()) == 1, "nothing tells the UI the tool finished"


async def test_the_completion_arrives_after_the_tool_returns(stream, monkeypatch):
    """Not merely present -- ordered, or it closes the step just as early."""
    order: list[str] = []

    def body():
        order.append("tool ran")
        return "ok"

    await run_tool(monkeypatch, body)
    # The announcement precedes the work, the completion follows it.
    assert stream.events[0].get("done") is None
    assert order == ["tool ran"]
    assert stream.events[-1].get("done") is True


async def test_start_and_end_carry_the_same_call_id(stream, monkeypatch):
    """The UI closes a specific step, so the pair has to be identifiable."""
    await run_tool(monkeypatch, lambda: "ok")
    assert stream.starts()[0]["id"] == stream.ends()[0]["id"]


async def test_repeat_calls_of_one_tool_are_distinguishable(stream, monkeypatch):
    """A turn calls the same tool twice with different arguments."""
    await run_tool(monkeypatch, lambda: "ok")
    await run_tool(monkeypatch, lambda: "ok")
    ids = [e["id"] for e in stream.starts()]
    assert len(set(ids)) == 2, f"calls collide under one id: {ids}"


async def test_a_failing_tool_still_completes(stream, monkeypatch):
    """Otherwise a raise leaves 'Running ...' on screen with no end."""

    def body():
        raise RuntimeError("cluster unreachable")

    with pytest.raises(RuntimeError):
        await run_tool(monkeypatch, body)
    assert len(stream.ends()) == 1, "a failed tool left its step open"


async def test_a_cancelled_turn_still_completes(stream, monkeypatch):
    """A user navigating away must not leave the step running forever."""

    def body():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_tool(monkeypatch, body)
    assert len(stream.ends()) == 1, "a cancelled tool left its step open"


async def test_the_duration_is_reported(stream, monkeypatch):
    """What the completion says the wait cost."""
    await run_tool(monkeypatch, lambda: "ok")
    assert isinstance(stream.ends()[0].get("seconds"), (int, float))


async def test_announcing_survives_no_stream(monkeypatch):
    """Outside a graph run there is no writer, and that is not an error."""

    def no_stream():
        raise RuntimeError("not streaming")

    monkeypatch.setattr(nodes, "get_stream_writer", no_stream)
    monkeypatch.setattr(nodes, "_execute_tool", lambda name, kwargs: "ok")
    assert await nodes._execute_tool_async("read_file", {}) == "ok"
