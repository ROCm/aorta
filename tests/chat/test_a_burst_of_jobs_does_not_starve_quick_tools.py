"""Cluster jobs must not occupy a small outer pool needed by quick tools.

#425 originally fixed a four-worker outer executor by splitting it into job and
quick pools. Updated ``main`` made ``_execute_tool`` asynchronous through
``BaseTool.ainvoke`` itself, and #424 now wraps that coroutine only for progress
and cancellation. Keeping the old pools would reintroduce both nesting and a
worse bug: submitting an async function to a thread returns a coroutine object
instead of running the tool.

The stronger merged invariant is therefore that no outer tool executor exists.
LangChain owns off-loop execution for synchronous tools, while the bounded
triage pool owns cluster admission.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from langchain_core.tools import tool

import aorta.chat.graph.nodes as nodes


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(nodes, "_announce_tool", lambda payload: None)


class TestThereIsNoNestedToolPool:
    def test_the_obsolete_outer_pool_is_gone(self):
        assert not hasattr(nodes, "_tool_pool")
        assert not hasattr(nodes, "_TOOL_WORKERS")
        assert not hasattr(nodes, "_JOB_WORKERS")
        assert not hasattr(nodes, "_QUICK_WORKERS")

    def test_the_progress_wrapper_awaits_the_async_tool_seam(self):
        import inspect

        source = inspect.getsource(nodes._execute_tool_async)

        assert "await _execute_tool(" in source
        assert "run_in_executor" not in source
        assert ".submit(" not in source


class TestAQuickToolAnswersDuringTheReportedBurst:
    async def test_four_long_sync_tools_do_not_hold_it(
        self, monkeypatch
    ):
        """Four was the complete outer pool before; a fifth call could not run."""
        release = threading.Event()
        lock = threading.Lock()
        started = 0

        @tool
        def long_tool(slot: int) -> str:
            """Wait until the test releases this synchronous tool."""
            nonlocal started
            with lock:
                started += 1
            release.wait(timeout=10)
            return f"long-{slot}"

        @tool
        def quick_tool() -> str:
            """Return without waiting for cluster work."""
            return "quick"

        monkeypatch.setitem(nodes.TOOL_REGISTRY, long_tool.name, long_tool)
        monkeypatch.setitem(nodes.TOOL_REGISTRY, quick_tool.name, quick_tool)
        held = [
            asyncio.create_task(
                nodes._execute_tool_async(long_tool.name, {"slot": slot})
            )
            for slot in range(4)
        ]

        deadline = asyncio.get_running_loop().time() + 10
        while True:
            with lock:
                all_started = started == 4
            if all_started:
                break
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

        began = time.monotonic()
        answer = await nodes._execute_tool_async(quick_tool.name, {})
        latency = time.monotonic() - began
        release.set()
        await asyncio.gather(*held)

        assert answer == "quick"
        assert latency < 1.0
