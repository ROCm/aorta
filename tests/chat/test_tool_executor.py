"""Tool execution uses LangChain's async seam without losing context or progress.

Updated ``main`` made ``_execute_tool`` await ``BaseTool.ainvoke``. LangChain
already sends synchronous tools to an executor and copies context there, so the
extra outer pool #425 originally added became both redundant and wrong: calling
an async function in that pool returned its coroutine object as the tool result.

These tests assert the merged contract at its boundaries: synchronous tools are
off-loop, context variables survive, and the progress event remains small and
ordered around the actual work.
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import AsyncMock, patch

from langchain_core.tools import tool

from aorta.chat.graph import nodes
from aorta.chat.tools.cache import ToolCache, current_tool_cache, use_tool_cache


async def _run_sync_tool(monkeypatch, tool_fn, kwargs: dict | None = None) -> str:
    monkeypatch.setitem(nodes.TOOL_REGISTRY, tool_fn.name, tool_fn)
    return await nodes._execute_tool_async(tool_fn.name, kwargs or {})


class TestThereIsNoNestedOuterExecutor:
    def test_the_progress_wrapper_awaits_the_async_tool_seam(self):
        import inspect

        source = inspect.getsource(nodes._execute_tool_async)

        assert "await _execute_tool(" in source
        assert "run_in_executor" not in source
        assert ".submit(" not in source

    def test_the_obsolete_pool_is_gone(self):
        assert not hasattr(nodes, "_tool_pool")
        assert not hasattr(nodes, "_TOOL_WORKERS")


class TestContextReachesTheSynchronousWorker:
    async def test_the_conversation_cache_is_still_bound(self, monkeypatch):
        seen: dict = {}

        @tool
        def capture_cache() -> str:
            """Capture the cache bound to this conversation."""
            seen["cache"] = current_tool_cache()
            return "ok"

        with use_tool_cache(ToolCache()) as bound:
            await _run_sync_tool(monkeypatch, capture_cache)

        assert seen["cache"] is bound

    async def test_two_turns_do_not_share_a_cache(self, monkeypatch):
        seen: list[ToolCache] = []

        @tool
        def capture_cache() -> str:
            """Capture each conversation's cache."""
            seen.append(current_tool_cache())
            return "ok"

        monkeypatch.setitem(nodes.TOOL_REGISTRY, capture_cache.name, capture_cache)
        for _ in range(2):
            with use_tool_cache(ToolCache()):
                await nodes._execute_tool_async(capture_cache.name, {})

        assert seen[0] is not seen[1]

    async def test_the_redaction_notice_is_still_bound(self, monkeypatch):
        from aorta.chat import redaction

        seen: dict = {}

        @tool
        def capture_notice() -> str:
            """Capture the redaction disclosure state."""
            seen["state"] = redaction.current_notice_state()
            return "ok"

        state = redaction.NoticeState()
        with redaction.use_notice_state(state):
            await _run_sync_tool(monkeypatch, capture_notice)

        assert seen["state"] is state


class TestSynchronousToolsStillRunOffTheLoop:
    async def test_the_call_happens_on_another_thread(self, monkeypatch):
        seen: dict = {}

        @tool
        def capture_thread() -> str:
            """Capture the thread that executes this tool."""
            seen["thread"] = threading.current_thread().name
            return "ok"

        await _run_sync_tool(monkeypatch, capture_thread)

        assert seen["thread"] != threading.main_thread().name

    async def test_the_loop_keeps_answering_while_a_tool_blocks(self, monkeypatch):
        ticks = 0

        @tool
        def slow_tool() -> str:
            """Block briefly on the executor."""
            threading.Event().wait(0.3)
            return "done"

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        result = await _run_sync_tool(monkeypatch, slow_tool)
        beat.cancel()

        assert result == "done"
        assert ticks > 3

    async def test_the_result_is_returned_unchanged(self, monkeypatch):
        @tool
        def answer_tool() -> str:
            """Return the tool's answer."""
            return "the tool said this"

        assert await _run_sync_tool(monkeypatch, answer_tool) == "the tool said this"


class TestTheAnnouncementIsSmallAndOrdered:
    @staticmethod
    async def _announced(tool_name: str, kwargs: dict) -> list[dict]:
        sent: list[dict] = []
        execute = AsyncMock(return_value="ok")
        with patch.object(nodes, "_execute_tool", execute), patch.object(
            nodes, "get_stream_writer", lambda: sent.append
        ):
            await nodes._execute_tool_async(tool_name, kwargs)
        return sent

    async def test_the_name_is_present_but_arguments_are_not(self):
        kernel = "__global__ void k(float* o) { o[0] = 1; }" * 40
        events = await self._announced(
            "triage_kernel_source", {"source": kernel}
        )

        assert events[0]["tool"] == "triage_kernel_source"
        assert "args" not in events[0]
        assert kernel not in str(events[0])

    async def test_the_payload_size_does_not_follow_the_paste(self):
        events = await self._announced(
            "triage_kernel_source", {"source": "x" * 20_000}
        )

        assert len(json.dumps(events[0])) < 100

    async def test_start_and_completion_share_an_id(self):
        events = await self._announced("read_file", {"file_path": "README.md"})

        assert len(events) == 2
        assert events[0].get("done") is None
        assert events[1]["done"] is True
        assert events[0]["id"] == events[1]["id"]

    async def test_not_being_streamed_is_not_an_error(self):
        execute = AsyncMock(return_value="ok")

        def no_writer():
            raise RuntimeError("not in a stream")

        with patch.object(nodes, "_execute_tool", execute), patch.object(
            nodes, "get_stream_writer", no_writer
        ):
            assert await nodes._execute_tool_async("read_file", {}) == "ok"
