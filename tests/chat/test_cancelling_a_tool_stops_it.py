"""Cancelling the turn has to reach the work, not just the screen.

Every tool here is synchronous and runs on a worker thread. Cancelling the
coroutine that awaited it cannot touch that thread -- Python has no way to
interrupt one -- so the tool ran on. For a triage that meant the full timeout
with a Slurm allocation held behind it, for a turn nobody was waiting for.

The completion event made it worse rather than visible. It was emitted from a
``finally``, so cancelling produced "done" immediately while the job was still
running: the one signal a caller reads as "this is over" said so first.

Two things now. A cancellation token rides the context into the worker, which
is the same route the per-conversation cache already takes, and the triage
watches it -- so giving up reaches `scancel`. And completion waits for the
thread to actually end, bounded, saying which of the two happened rather than
claiming the good one.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("langchain_core", reason="the graph needs the chat-cli extra")

import aorta.chat.graph.nodes as nodes
from aorta.chat.cancellation import cancelled, current_cancel_token


@pytest.fixture()
def announced(monkeypatch):
    """Collect the progress events instead of streaming them."""
    events: list[dict] = []
    monkeypatch.setattr(nodes, "_announce_tool", events.append)
    return events


@pytest.fixture()
def quick_grace(monkeypatch):
    """A grace short enough to assert against."""
    monkeypatch.setattr(nodes, "_CANCEL_GRACE_SEC", 1.0)


def _run(coro_factory):
    async def main():
        task = asyncio.create_task(coro_factory())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return "finished"

    return asyncio.run(main())


class TestTheTokenReachesTheTool:
    def test_a_tool_can_see_its_token(self, monkeypatch, announced):
        seen: dict = {}

        def tool(name, kwargs):
            seen["token"] = current_cancel_token()
            return "ok"

        monkeypatch.setattr(nodes, "_execute_tool", tool)
        asyncio.run(nodes._execute_tool_async("triage_workload", {}))

        assert isinstance(seen["token"], threading.Event)

    def test_it_is_not_set_while_the_caller_is_still_waiting(
        self, monkeypatch, announced
    ):
        seen: dict = {}

        def tool(name, kwargs):
            seen["cancelled"] = cancelled()
            return "ok"

        monkeypatch.setattr(nodes, "_execute_tool", tool)
        asyncio.run(nodes._execute_tool_async("triage_workload", {}))

        assert seen["cancelled"] is False

    def test_two_calls_at_once_do_not_share_one(self, monkeypatch, announced):
        """Cancelling one turn must not stop another's tool."""
        tokens: list = []
        gate = threading.Barrier(2, timeout=10)

        def tool(name, kwargs):
            tokens.append(current_cancel_token())
            gate.wait()  # both inside the tool at the same moment
            return "ok"

        monkeypatch.setattr(nodes, "_execute_tool", tool)

        async def both():
            await asyncio.gather(
                nodes._execute_tool_async("triage_workload", {}),
                nodes._execute_tool_async("triage_workload", {}),
            )

        asyncio.run(both())

        assert len(tokens) == 2
        assert tokens[0] is not tokens[1]


class TestCancellingSetsIt:
    def test_a_watching_tool_is_told(self, monkeypatch, announced):
        noticed = threading.Event()

        def tool(name, kwargs):
            token = current_cancel_token()
            for _ in range(200):
                if token is not None and token.is_set():
                    noticed.set()
                    return "stopped early"
                time.sleep(0.05)
            return "ran to completion"

        monkeypatch.setattr(nodes, "_execute_tool", tool)
        outcome = _run(lambda: nodes._execute_tool_async("triage_kernel_source", {}))

        assert outcome == "cancelled"
        assert noticed.is_set()

    def test_and_it_stops_promptly_rather_than_running_its_course(
        self, monkeypatch, announced
    ):
        def tool(name, kwargs):
            token = current_cancel_token()
            for _ in range(200):  # 10s if never cancelled
                if token is not None and token.is_set():
                    return "stopped early"
                time.sleep(0.05)
            return "ran to completion"

        monkeypatch.setattr(nodes, "_execute_tool", tool)
        started = time.monotonic()
        _run(lambda: nodes._execute_tool_async("triage_kernel_source", {}))

        assert time.monotonic() - started < 3.0


class TestCompletionWaitsForTheWorker:
    def test_a_tool_that_stops_is_reported_as_stopped(self, monkeypatch, announced):
        def tool(name, kwargs):
            token = current_cancel_token()
            while not (token is not None and token.is_set()):
                time.sleep(0.02)
            return "stopped"

        monkeypatch.setattr(nodes, "_execute_tool", tool)
        _run(lambda: nodes._execute_tool_async("triage_kernel_source", {}))

        done = [e for e in announced if e.get("done")]
        assert done and done[0]["cancelled"] == "stopped"

    def test_a_tool_that_ignores_the_token_is_not_called_stopped(
        self, monkeypatch, announced, quick_grace
    ):
        """The honest answer, since the cluster job may still be running."""

        def tool(name, kwargs):
            time.sleep(5.0)
            return "ignored it"

        monkeypatch.setattr(nodes, "_execute_tool", tool)
        _run(lambda: nodes._execute_tool_async("triage_workload", {}))

        done = [e for e in announced if e.get("done")]
        assert done and done[0]["cancelled"] == "still running"

    def test_waiting_is_bounded_by_the_grace(self, monkeypatch, announced, quick_grace):
        """A tool that never looks must not hold the turn open for ever."""

        def tool(name, kwargs):
            time.sleep(5.0)
            return "ignored it"

        monkeypatch.setattr(nodes, "_execute_tool", tool)
        started = time.monotonic()
        _run(lambda: nodes._execute_tool_async("triage_workload", {}))

        assert time.monotonic() - started < 3.0

    def test_the_grace_is_read_at_the_time_it_is_used(self):
        """A default argument would fix it at import and ignore the constant."""
        import inspect

        signature = inspect.signature(nodes._wait_for_worker)

        assert signature.parameters["grace"].default is None


class TestAnUncancelledCallIsUnchanged:
    def test_it_returns_the_tool_result(self, monkeypatch, announced):
        monkeypatch.setattr(nodes, "_execute_tool", lambda name, kwargs: "the answer")

        assert asyncio.run(nodes._execute_tool_async("triage_workload", {})) == "the answer"

    def test_it_announces_a_start_and_a_finish(self, monkeypatch, announced):
        monkeypatch.setattr(nodes, "_execute_tool", lambda name, kwargs: "ok")
        asyncio.run(nodes._execute_tool_async("triage_workload", {}))

        assert len(announced) == 2
        assert not announced[0].get("done")
        assert announced[1]["done"] is True

    def test_nothing_claims_it_was_cancelled(self, monkeypatch, announced):
        monkeypatch.setattr(nodes, "_execute_tool", lambda name, kwargs: "ok")
        asyncio.run(nodes._execute_tool_async("triage_workload", {}))

        assert "cancelled" not in announced[1]

    def test_the_two_events_share_an_id(self, monkeypatch, announced):
        """The consumer closes the step it opened, not whichever is newest."""
        monkeypatch.setattr(nodes, "_execute_tool", lambda name, kwargs: "ok")
        asyncio.run(nodes._execute_tool_async("triage_workload", {}))

        assert announced[0]["id"] == announced[1]["id"]


class TestTheTriageWatchesIt:
    def test_run_triage_adopts_the_ambient_token(self):
        """Without this the token stops at the tool and never reaches scancel."""
        pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")
        import inspect

        from aorta.chat.tools import cluster

        source = inspect.getsource(cluster._run_triage)

        assert "current_cancel_token()" in source

    def test_it_still_works_outside_a_tool_call(self):
        """Called directly -- from a test or the CLI -- there is no token."""
        assert current_cancel_token() is None
        assert cancelled() is False
