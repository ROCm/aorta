"""Tool calls run on an executor somebody chose the size of.

``asyncio.to_thread`` uses the loop's default executor, which is
``min(32, cpu_count + 4)`` -- thirty-two on the machine this was written on,
and not a bound anyone picked for work that blocks for minutes while a cluster
job runs. It is also shared with every other ``to_thread`` in the process, so a
burst of tool calls and unrelated work starve each other.

The thing that makes this more than a tidy-up is the contextvar. ``to_thread``
copies the calling context into the worker; ``run_in_executor`` does not. The
per-conversation ``ToolCache`` and the redaction notice are both bound that
way, so moving to an executor without carrying the context by hand would leave
the worker reading the process-wide fallback -- silently reintroducing the
cross-session cache bug the ToolCache exists to fix, in a place no test was
looking.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import pytest

from aorta.chat.graph import nodes
from aorta.chat.tools.cache import ToolCache, current_tool_cache, use_tool_cache


class TestTheExecutorIsChosenRatherThanInherited:
    def test_it_has_an_explicit_size(self):
        assert nodes._TOOL_WORKERS == 4

    def test_which_is_smaller_than_the_default_would_be(self):
        """min(32, cpu+4) is whatever the machine happens to have."""
        import os

        assert nodes._TOOL_WORKERS < min(32, (os.cpu_count() or 1) + 4)

    def test_it_leaves_room_beside_the_triage_pool(self):
        """A quick tool should still answer while a triage is out."""
        pytest.importorskip("dspy", reason="the triage pool needs the [cia] extra")
        from aorta.chat.tools import cluster

        assert nodes._TOOL_WORKERS > cluster._TRIAGE_WORKERS

    def test_one_pool_is_reused(self):
        assert nodes._tool_pool() is nodes._tool_pool()

    def test_its_threads_are_named_for_it(self):
        """So a stack dump says which pool a stuck thread belongs to."""
        nodes._tool_pool().submit(lambda: None).result(timeout=10)

        assert any("aorta-tool" in t.name for t in threading.enumerate())


class TestTheContextReachesTheWorker:
    """run_in_executor drops it; to_thread did not."""

    async def test_the_conversation_cache_is_still_bound(self):
        seen: dict = {}

        def spy(name, kwargs):
            seen["cache"] = current_tool_cache()
            return "ok"

        with patch.object(nodes, "_execute_tool", spy):
            with use_tool_cache(ToolCache()) as bound:
                await nodes._execute_tool_async("some_tool", {})

        assert seen["cache"] is bound, "the worker fell back to the process cache"

    async def test_two_turns_do_not_share_a_cache(self):
        """The property the binding exists for, measured through the executor."""
        seen: list = []

        def spy(name, kwargs):
            seen.append(current_tool_cache())
            return "ok"

        with patch.object(nodes, "_execute_tool", spy):
            for _ in range(2):
                with use_tool_cache(ToolCache()):
                    await nodes._execute_tool_async("some_tool", {})

        assert seen[0] is not seen[1]

    async def test_the_redaction_notice_is_bound_too(self):
        """Same mechanism, and a session-scoped disclosure rides on it."""
        from aorta.chat import redaction

        seen: dict = {}

        def spy(name, kwargs):
            seen["state"] = redaction.current_notice_state()
            return "ok"

        state = redaction.NoticeState()
        with patch.object(nodes, "_execute_tool", spy):
            with redaction.use_notice_state(state):
                await nodes._execute_tool_async("some_tool", {})

        assert seen["state"] is state


class TestItStillRunsOffTheLoop:
    """The reason this is not simply called inline."""

    async def test_the_call_happens_on_another_thread(self):
        seen: dict = {}

        def spy(name, kwargs):
            seen["thread"] = threading.current_thread().name
            return "ok"

        with patch.object(nodes, "_execute_tool", spy):
            await nodes._execute_tool_async("some_tool", {})

        assert seen["thread"] != threading.main_thread().name

    async def test_on_our_executor_and_not_the_default_one(self):
        """The size above is only a bound if the work actually runs there.

        Named threads are how that is visible: the default executor calls its
        own "asyncio_%d", so the prefix says which pool answered.
        """
        seen: dict = {}

        def spy(name, kwargs):
            seen["thread"] = threading.current_thread().name
            return "ok"

        with patch.object(nodes, "_execute_tool", spy):
            await nodes._execute_tool_async("some_tool", {})

        assert seen["thread"].startswith("aorta-tool"), seen["thread"]

    async def test_concurrent_calls_stay_within_the_bound(self):
        """Four workers means at most four tools in flight, not thirty-two."""
        live = 0
        peak = 0
        guard = threading.Lock()

        def slow(name, kwargs):
            nonlocal live, peak
            with guard:
                live += 1
                peak = max(peak, live)
            threading.Event().wait(0.15)
            with guard:
                live -= 1
            return "ok"

        with patch.object(nodes, "_execute_tool", slow):
            await asyncio.gather(
                *(nodes._execute_tool_async("t", {}) for _ in range(12))
            )

        assert peak <= nodes._TOOL_WORKERS, f"{peak} tools ran at once"

    async def test_the_loop_keeps_answering_while_a_tool_blocks(self):
        """A tool that blocks must not stall the server."""
        ticks = 0

        def slow(name, kwargs):
            threading.Event().wait(0.3)
            return "done"

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        with patch.object(nodes, "_execute_tool", slow):
            result = await nodes._execute_tool_async("some_tool", {})
        beat.cancel()

        assert result == "done"
        assert ticks > 3, f"the loop only ticked {ticks} times while a tool ran"

    async def test_the_result_is_returned_unchanged(self):
        with patch.object(nodes, "_execute_tool", lambda n, k: "the tool said this"):
            assert await nodes._execute_tool_async("t", {}) == "the tool said this"


# ── what the announcement carries ─────────────────────────────────────────


class TestTheAnnouncementIsJustTheName:
    """The arguments rode along and nothing read them.

    ``show_step`` renders ``delta.get("tool")`` and drops the rest, while
    ``kwargs`` for ``triage_kernel_source`` is the user's entire pasted kernel.
    So every tool call pushed the whole paste through the custom stream for a
    consumer that wanted eighteen characters of it.
    """

    @staticmethod
    async def _announced(tool_name: str, kwargs: dict) -> dict:
        sent: list[dict] = []

        with patch.object(nodes, "_execute_tool", lambda n, k: "ok"), patch.object(
            nodes, "get_stream_writer", lambda: sent.append
        ):
            await nodes._execute_tool_async(tool_name, kwargs)
        return sent[0] if sent else {}

    async def test_the_name_is_there(self):
        payload = await self._announced("triage_kernel_source", {"source": "__global__"})

        assert payload["tool"] == "triage_kernel_source"

    async def test_the_arguments_are_not(self):
        kernel = "__global__ void k(float* o) { o[0] = 1; }" * 40
        payload = await self._announced("triage_kernel_source", {"source": kernel})

        assert "args" not in payload
        assert kernel not in str(payload)

    async def test_the_payload_stays_small(self):
        """Independent of how much the user pasted."""
        import json

        big = await self._announced("triage_kernel_source", {"source": "x" * 20_000})

        assert len(json.dumps(big)) < 100

    async def test_it_is_still_announced_before_the_tool_blocks(self):
        """Dropping the payload must not drop the announcement."""
        order: list[str] = []

        def slow(name, kwargs):
            order.append("tool ran")
            return "ok"

        with patch.object(nodes, "_execute_tool", slow), patch.object(
            nodes, "get_stream_writer", lambda: lambda p: order.append("announced")
        ):
            await nodes._execute_tool_async("triage_kernel_source", {})

        assert order == ["announced", "tool ran"]

    async def test_not_being_streamed_is_not_an_error(self):
        """The CLI passes no callback, so there is no writer to announce to."""

        def no_writer():
            raise RuntimeError("not in a stream")

        with patch.object(nodes, "_execute_tool", lambda n, k: "ok"), patch.object(
            nodes, "get_stream_writer", no_writer
        ):
            assert await nodes._execute_tool_async("t", {}) == "ok"
