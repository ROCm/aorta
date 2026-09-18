"""Cancelling a chat task must reach the tool and its scheduler job.

``BaseTool.ainvoke`` keeps synchronous tools off the event loop, but cancelling
the asyncio waiter cannot interrupt the callable already running in that
executor. The old ``finally`` immediately announced ``done`` while a triage
continued until its internal timeout, holding a Slurm allocation after the
browser disconnected.

Each call now carries a context-local event through LangChain's executor into
``_run_triage``. Cancellation sets it, the triage waits for ``run_triage`` to
observe it and cancel Slurm, and the completion event belongs to the worker
Task rather than to the cancelled waiter.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from langchain_core.tools import tool

import aorta.chat.graph.nodes as nodes
from aorta.chat.cancellation import current_cancel_token


async def _started(event: threading.Event) -> None:
    deadline = asyncio.get_running_loop().time() + 10
    while not event.is_set():
        assert asyncio.get_running_loop().time() < deadline, "tool never started"
        await asyncio.sleep(0.01)


@pytest.fixture()
def announcements(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(nodes, "_announce_tool", events.append)
    return events


class TestCancellationCrossesTheLangChainExecutor:
    async def test_the_sync_tool_receives_a_per_call_handle(
        self, monkeypatch, announcements
    ):
        started = threading.Event()
        stopped = threading.Event()
        seen: dict = {}

        @tool
        def waits_for_cancel() -> str:
            """Wait until the caller gives up."""
            token = current_cancel_token()
            seen["token"] = token
            started.set()
            assert token is not None
            token.wait(timeout=10)
            stopped.set()
            return "stopped"

        monkeypatch.setitem(nodes.TOOL_REGISTRY, waits_for_cancel.name, waits_for_cancel)
        task = asyncio.create_task(
            nodes._execute_tool_async(waits_for_cancel.name, {})
        )
        await _started(started)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert isinstance(seen["token"], threading.Event)
        assert seen["token"].is_set()
        assert stopped.is_set()

    async def test_two_calls_do_not_share_a_handle(self, monkeypatch, announcements):
        started = [threading.Event(), threading.Event()]
        release = threading.Event()
        tokens: list[threading.Event] = []
        lock = threading.Lock()

        @tool
        def concurrent_call(slot: int) -> str:
            """Record this call's cancellation token."""
            token = current_cancel_token()
            assert token is not None
            with lock:
                tokens.append(token)
            started[slot].set()
            release.wait(timeout=10)
            return "done"

        monkeypatch.setitem(nodes.TOOL_REGISTRY, concurrent_call.name, concurrent_call)
        tasks = [
            asyncio.create_task(
                nodes._execute_tool_async(concurrent_call.name, {"slot": slot})
            )
            for slot in range(2)
        ]
        await asyncio.gather(*(_started(event) for event in started))
        release.set()
        await asyncio.gather(*tasks)

        assert len(tokens) == 2
        assert tokens[0] is not tokens[1]


class TestCompletionBelongsToTheWorker:
    async def test_done_is_not_emitted_while_the_callable_still_runs(
        self, monkeypatch, announcements
    ):
        started = threading.Event()
        release = threading.Event()
        exited = threading.Event()

        @tool
        def ignores_cancel_until_released() -> str:
            """Model a synchronous tool winding down after cancellation."""
            started.set()
            release.wait(timeout=10)
            exited.set()
            return "done"

        monkeypatch.setitem(
            nodes.TOOL_REGISTRY,
            ignores_cancel_until_released.name,
            ignores_cancel_until_released,
        )
        task = asyncio.create_task(
            nodes._execute_tool_async(ignores_cancel_until_released.name, {})
        )
        await _started(started)
        task.cancel()
        await asyncio.sleep(0.1)

        assert not exited.is_set()
        assert not any(event.get("done") for event in announcements)
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        done = [event for event in announcements if event.get("done")]
        assert exited.is_set()
        assert len(done) == 1
        assert done[0]["cancelled"] is True

    async def test_normal_completion_still_announces_once(
        self, monkeypatch, announcements
    ):
        @tool
        def quick_tool() -> str:
            """Return immediately."""
            return "answer"

        monkeypatch.setitem(nodes.TOOL_REGISTRY, quick_tool.name, quick_tool)

        assert await nodes._execute_tool_async(quick_tool.name, {}) == "answer"
        assert len(announcements) == 2
        assert announcements[0].get("done") is None
        assert announcements[1]["done"] is True
        assert announcements[1]["cancelled"] is False
        assert announcements[0]["id"] == announcements[1]["id"]


class TestCancellationReachesRunTriage:
    async def test_the_inner_worker_stops_before_the_chat_task_finishes(
        self, tmp_path, monkeypatch, announcements
    ):
        import aorta.chat.tools.cluster as cluster

        entered = threading.Event()
        exited = threading.Event()
        seen: dict = {}

        def fake_run_triage(argv, *, stop=None):
            seen["stop"] = stop
            entered.set()
            assert stop is not None
            stop.wait(timeout=10)
            exited.set()
            return {
                "ok": False,
                "stage": "wait",
                "error": "abandoned by caller",
            }

        monkeypatch.setattr(cluster, "run_triage", fake_run_triage)
        monkeypatch.setattr(cluster.settings, "jobs_path", str(tmp_path), raising=False)

        @tool
        def diagnostic() -> str:
            """Run the synchronous triage seam."""
            return cluster._run_triage(["--source", "kernel.hip"], "test")

        monkeypatch.setitem(nodes.TOOL_REGISTRY, diagnostic.name, diagnostic)
        task = asyncio.create_task(nodes._execute_tool_async(diagnostic.name, {}))
        await _started(entered)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert seen["stop"] is not None and seen["stop"].is_set()
        assert exited.is_set()
        done = [event for event in announcements if event.get("done")]
        assert len(done) == 1
        assert done[0]["cancelled"] is True
