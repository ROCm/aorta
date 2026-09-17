"""A burst of cluster jobs must not take the threads a file read needs.

Tool calls ran on one pool of four, against a cluster pool of two, on the
reasoning that four is more than two and the difference is room for the quick
tools. It is not. A triage occupies a tool worker and *then* waits for a slot
in the cluster pool, so the third and fourth concurrent triages sat in the tool
pool holding the very capacity they were meant to be leaving free. Four at once
took every worker, and reading a file queued behind a GPU job for minutes.

Two pools now, and the job half is sized to the cluster pool so a worker here
maps to a worker there and never waits for admission. A third triage queues as
a job rather than as a tool, and the quick tools have workers a burst cannot
reach.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("langchain_core", reason="the graph needs the chat-cli extra")

import aorta.chat.graph.nodes as nodes


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(nodes, "_announce_tool", lambda payload: None)


class TestTheTwoKindsAreToldApart:
    def test_a_triage_is_a_job(self):
        assert nodes._is_job_tool("triage_kernel_source") is True

    def test_so_are_the_other_two(self):
        assert nodes._is_job_tool("triage_assembly_source") is True
        assert nodes._is_job_tool("triage_workload") is True

    def test_reading_a_report_is_not(self):
        """It reads a file the job already wrote; nothing is submitted."""
        assert nodes._is_job_tool("read_autopsy_report") is False

    def test_nor_is_listing_jobs(self):
        assert nodes._is_job_tool("list_cluster_jobs") is False

    def test_nor_is_an_ordinary_file_tool(self):
        assert nodes._is_job_tool("read_file") is False

    def test_an_unknown_tool_is_treated_as_quick(self):
        """A plugin tool is not assumed to hold a GPU node."""
        assert nodes._is_job_tool("something_from_a_plugin") is False


class TestTheyGetDifferentPools:
    def test_a_job_and_a_quick_tool_do_not_share(self):
        assert nodes._tool_pool("triage_kernel_source") is not nodes._tool_pool("read_file")

    def test_the_job_pool_is_reused(self):
        assert nodes._tool_pool("triage_kernel_source") is nodes._tool_pool("triage_workload")

    def test_the_quick_pool_is_reused(self):
        assert nodes._tool_pool("read_file") is nodes._tool_pool("list_cluster_jobs")

    def test_the_job_pool_matches_the_cluster_pool(self):
        """The sizing is the fix: one worker here per worker there.

        Larger and a job worker waits for an inner slot while holding a tool
        thread, which is the bug. Smaller and the cluster pool is never full.
        """
        pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster

        assert nodes._JOB_WORKERS == cluster._TRIAGE_WORKERS

    def test_the_quick_pool_has_room_of_its_own(self):
        assert nodes._QUICK_WORKERS >= 2


class TestAQuickToolAnswersDuringABurst:
    """The behaviour the sizing exists for, measured rather than reasoned."""

    @staticmethod
    def _burst(monkeypatch, jobs: int) -> float:
        """Latency of a quick tool while *jobs* triages are held open."""
        holding = threading.Event()

        def slow(name, kwargs):
            holding.wait(10)
            return "slow"

        def quick(name, kwargs):
            return "quick"

        async def main() -> float:
            monkeypatch.setattr(nodes, "_execute_tool", slow)
            held = [
                asyncio.create_task(
                    nodes._execute_tool_async("triage_kernel_source", {})
                )
                for _ in range(jobs)
            ]
            await asyncio.sleep(0.3)  # let them occupy what they are going to
            monkeypatch.setattr(nodes, "_execute_tool", quick)
            started = time.monotonic()
            answer = await nodes._execute_tool_async("list_cluster_jobs", {})
            latency = time.monotonic() - started
            assert answer == "quick"
            holding.set()
            await asyncio.gather(*held)
            return latency

        return asyncio.run(main())

    def test_more_triages_than_workers_do_not_block_it(self, monkeypatch):
        """Six at once, against two job workers and four quick ones."""
        assert self._burst(monkeypatch, jobs=6) < 1.0

    def test_nor_does_exactly_filling_the_job_pool(self, monkeypatch):
        assert self._burst(monkeypatch, jobs=nodes._JOB_WORKERS) < 1.0

    def test_nor_does_one_more_than_the_old_single_pool_held(self, monkeypatch):
        """Four was the whole pool before; a file read waited behind them."""
        assert self._burst(monkeypatch, jobs=4) < 1.0
