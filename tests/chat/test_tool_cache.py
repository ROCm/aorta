"""Reuse of a cluster result has to stop at the edge of the conversation.

Triaging a kernel costs minutes on a GPU node, so an identical paste is
answered from the previous run rather than submitted again. The dict holding
those answers was a module global, and a module global in the Chainlit server
is one dict shared by every browser session it serves. Two things followed.

The first is a correctness and privacy problem: user B pasting a kernel user A
had already submitted got user A's verdict, over user A's code, under a line
saying the run had happened "in this conversation". It had not.

The second is a leak. Keyed on the full pasted source with no cap, the dicts
held every kernel anyone had ever submitted for as long as the process ran.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from aorta.chat.tools.cache import (
    MAX_ENTRIES,
    BoundedCache,
    ToolCache,
    current_tool_cache,
    use_tool_cache,
)

_KERNEL = """\
__global__ void bump(float* out) {
    out[threadIdx.x] += 1.0f;
}
"""

_OTHER_KERNEL = """\
__global__ void scale(float* out) {
    out[threadIdx.x] *= 2.0f;
}
"""

_ASM = "s_load_dword s4, s[0:1], 0x10\nv_mov_b32 v0, s4\ns_endpgm\n"

_VERDICT = "Autopsy verdict: gpu_race (0.92)"


# ── the tool, driven with the cluster stubbed out ──────────────────────────


@pytest.fixture()
def cluster(tmp_path, monkeypatch):
    """The cluster tools with the parts that need a GPU replaced.

    Everything up to and including the cache decision is the real code; only
    the job itself is stubbed, and it counts its calls, which is how these
    tests tell reuse from a fresh submission.
    """
    pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
    import aorta.chat.tools.cluster as module

    # jobs_root is derived; jobs_path is the field behind it.
    monkeypatch.setattr(module.settings, "jobs_path", str(tmp_path), raising=False)
    monkeypatch.setattr(module.settings, "cia_demo_node", "", raising=False)

    calls: list[str] = []

    def fake_triage(extra_args, label):
        calls.append(label)
        return _VERDICT

    monkeypatch.setattr(module, "_run_triage", fake_triage)
    module.triage_calls = calls
    return module


@pytest.fixture(autouse=True)
def _fresh_process_cache():
    """No test inherits what an earlier one left in the fallback cache."""
    current_tool_cache().clear()
    yield
    current_tool_cache().clear()


def _triage(cluster, source=_KERNEL, **kwargs) -> str:
    return cluster.triage_kernel_source.func(source, **kwargs)


class TestOneConversationDoesNotAnswerAnother:
    """The finding itself."""

    def test_a_second_session_runs_its_own_job(self, cluster):
        """The bug: B was handed A's verdict without a job being submitted."""
        with use_tool_cache(ToolCache()):
            first = _triage(cluster)

        with use_tool_cache(ToolCache()):
            second = _triage(cluster)

        assert len(cluster.triage_calls) == 2, "the second session reused the first's run"
        assert "Reusing" not in second
        assert _VERDICT in first and _VERDICT in second

    def test_and_is_not_told_a_run_it_never_made_was_reused(self, cluster):
        """The wording is the part a user would act on."""
        with use_tool_cache(ToolCache()):
            _triage(cluster)

        with use_tool_cache(ToolCache()):
            second = _triage(cluster)

        assert "in this conversation" not in second

    def test_assembly_is_scoped_the_same_way(self, cluster, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            cluster,
            "prepare_asm",
            lambda *a, **k: (_ for _ in ()).throw(cluster.AsmHarnessError("stop")),
        )

        def record(source):
            seen.append(source)
            return cluster.triage_assembly_source.func(source)

        with use_tool_cache(ToolCache()):
            record(_ASM)
        with use_tool_cache(ToolCache()):
            second = record(_ASM)

        assert "Reusing" not in second


class TestReuseStillWorksWhereItShould:
    """The caching exists for a reason; scoping it must not switch it off."""

    def test_the_same_kernel_twice_in_one_conversation_runs_one_job(self, cluster):
        with use_tool_cache(ToolCache()):
            _triage(cluster)
            second = _triage(cluster)

        assert len(cluster.triage_calls) == 1
        assert "Reusing" in second
        assert _VERDICT in second

    def test_the_single_session_front_doors_need_no_binding(self, cluster):
        """``aorta chat`` is one conversation per process, so the default holds.

        Nothing in the CLI binds a cache, and reuse still has to work there.
        """
        _triage(cluster)
        second = _triage(cluster)

        assert len(cluster.triage_calls) == 1
        assert "Reusing" in second

    def test_a_different_geometry_is_a_different_question(self, cluster):
        with use_tool_cache(ToolCache()):
            _triage(cluster, block_size=64)
            _triage(cluster, block_size=128)

        assert len(cluster.triage_calls) == 2

    def test_force_resubmits_and_refreshes_the_entry(self, cluster):
        with use_tool_cache(ToolCache()):
            _triage(cluster)
            forced = _triage(cluster, force=True)

        assert len(cluster.triage_calls) == 2
        assert "Reusing" not in forced

    def test_a_failed_run_is_not_remembered(self, cluster, monkeypatch):
        """A transient launch failure should be retried, not cached."""
        monkeypatch.setattr(
            cluster, "_run_triage", lambda extra_args, label: "Error: triage failed: boom"
        )

        with use_tool_cache(ToolCache()):
            _triage(cluster)
            cache = current_tool_cache()

        assert len(cache.triage) == 0


class TestItStopsGrowing:
    """Keyed on the paste with no cap, these held every kernel ever submitted."""

    def test_entries_are_capped(self):
        cache = BoundedCache(max_entries=3)
        for i in range(10):
            cache.put(f"kernel-{i}", "verdict")

        assert len(cache) == 3

    def test_the_oldest_is_what_goes(self):
        cache = BoundedCache(max_entries=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")

        assert "a" not in cache
        assert cache.get("b") == "2" and cache.get("c") == "3"

    def test_reuse_keeps_an_entry_alive(self):
        """Least *recently used*: the kernel a conversation keeps returning to
        should not be evicted by one it touched once."""
        cache = BoundedCache(max_entries=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.get("a")
        cache.put("c", "3")

        assert cache.get("a") == "1"
        assert "b" not in cache

    def test_a_conversation_that_keeps_pasting_stays_bounded(self, cluster):
        with use_tool_cache(ToolCache(max_entries=4)) as cache:
            for i in range(12):
                _triage(cluster, source=_KERNEL.replace("bump", f"bump{i}"))

            assert len(cache.triage) == 4
        assert len(cluster.triage_calls) == 12

    def test_cheap_results_cannot_evict_expensive_ones(self):
        """A triage verdict cost minutes; an assembly analysis cost a second.

        One shared bound would let a run of pasted fragments push out the
        answers that were expensive to get.
        """
        cache = ToolCache(max_entries=2)
        cache.triage.put("kernel", "verdict")
        for i in range(10):
            cache.asm.put(f"frag-{i}", "hazards")

        assert cache.triage.get("kernel") == "verdict"


class TestTheChainlitServerBindsOnePerSession:
    """Where the scoping is actually established."""

    @pytest.fixture()
    def app(self, monkeypatch):
        sent: list[str] = []

        class _Message:
            def __init__(self, content="", **kwargs):
                self.content = content

            async def send(self):
                sent.append(self.content)

            async def remove(self):
                return None

        class _UserSession:
            def __init__(self):
                self._values = {}

            def get(self, key, default=None):
                return self._values.get(key, default)

            def set(self, key, value):
                self._values[key] = value

        chainlit = ModuleType("chainlit")
        chainlit.Message = _Message
        chainlit.user_session = _UserSession()
        chainlit.on_chat_start = lambda fn: fn
        chainlit.on_message = lambda fn: fn

        monkeypatch.setitem(sys.modules, "chainlit", chainlit)
        monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)
        import aorta.chat.ui.app as app_module

        app_module.cl.user_session = _UserSession()
        app_module.cl.user_session.set("history", [])
        app_module.cl.user_session.set("backend_error", None)
        app_module.sent = sent
        yield app_module
        monkeypatch.delitem(sys.modules, "aorta.chat.ui.app", raising=False)

    async def test_a_turn_runs_against_this_session_s_cache(self, app, monkeypatch):
        session_cache = ToolCache()
        app.cl.user_session.set(app._TOOL_CACHE_KEY, session_cache)
        seen = []

        async def _answer(question, history, on_step=None):  # noqa: ARG001 - signature match
            seen.append(current_tool_cache())
            return "answer", [], {}

        monkeypatch.setattr(app, "invoke_agent", _answer)
        await app.on_message(SimpleNamespace(content="triage this"))

        assert seen == [session_cache]

    async def test_a_session_that_predates_the_key_gets_its_own(self, app, monkeypatch):
        """A reconnect must not fall back to the process-wide cache, which is
        what would let one browser session read another's run."""
        seen = []

        async def _answer(question, history, on_step=None):  # noqa: ARG001 - signature match
            seen.append(current_tool_cache())
            return "answer", [], {}

        monkeypatch.setattr(app, "invoke_agent", _answer)
        await app.on_message(SimpleNamespace(content="triage this"))

        assert seen and seen[0] is not None
        assert seen[0] is app.cl.user_session.get(app._TOOL_CACHE_KEY)

    async def test_starting_a_chat_gives_the_session_a_cache(self, app, monkeypatch):
        monkeypatch.setattr(
            app, "get_backend", lambda: SimpleNamespace(describe=lambda: "x", preflight=None)
        )
        monkeypatch.setattr(app, "_SKIP_PREFLIGHT", True)
        await app.on_start()

        assert isinstance(app.cl.user_session.get(app._TOOL_CACHE_KEY), ToolCache)


class TestItSurvivesTheWayTheGraphRuns:
    """A turn is not one flat call, and the var has to hold across the shape.

    A task copies the context at creation, so writes to a ``ContextVar`` inside
    one do not propagate back out. That is why the var carries a handle to a
    mutable cache rather than being the cache: entries stored while a node was
    answering would otherwise be forgotten by the time the next message
    arrived, and every turn would resubmit.
    """

    async def test_an_entry_stored_inside_a_task_is_there_afterwards(self):
        async def deeper():
            current_tool_cache().triage.put("kernel", "verdict")

        with use_tool_cache(ToolCache()) as cache:
            await asyncio.create_task(deeper())
            assert cache.triage.get("kernel") == "verdict"

    async def test_and_one_stored_on_a_worker_thread_too(self):
        """``_run_triage`` hands the job to a thread; the cache write happens
        back in the caller, but a node moving off-thread must not break it."""
        with use_tool_cache(ToolCache()) as cache:
            await asyncio.to_thread(lambda: current_tool_cache().asm.put("frag", "hazards"))

        assert cache.asm.get("frag") == "hazards"

    async def test_two_concurrent_sessions_stay_separate(self):
        """What the Chainlit server actually does: many turns at once."""
        async def turn(cache, key):
            with use_tool_cache(cache):
                await asyncio.sleep(0)
                current_tool_cache().triage.put(key, f"verdict for {key}")
                await asyncio.sleep(0)
                return current_tool_cache()

        a, b = ToolCache(), ToolCache()
        seen = await asyncio.gather(turn(a, "alice"), turn(b, "bob"))

        assert seen == [a, b]
        assert "bob" not in a.triage
        assert "alice" not in b.triage


class TestTheBindingUnwinds:
    def test_the_cache_does_not_outlive_its_block(self):
        outer = current_tool_cache()
        with use_tool_cache(ToolCache()) as inner:
            assert current_tool_cache() is inner
        assert current_tool_cache() is outer

    def test_the_default_is_shared_only_by_callers_that_bind_nothing(self):
        assert current_tool_cache() is current_tool_cache()

    def test_the_cap_is_a_real_number(self):
        assert MAX_ENTRIES > 0
