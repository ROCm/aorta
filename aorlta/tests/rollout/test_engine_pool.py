"""The engine pool (design Section 6.1: Engine start, Group formation, Failure modes)."""

from __future__ import annotations

import math
import random
import time

import pytest

from aorlta.rollout.engine_pool import (
    EnginePool, PrefixIndex, RolloutStop, engine_kwargs, engine_port, expected_server_args, logprobs_aligned,
    lpt_assign, max_model_len, run_self_test, server_arg_problems,
)
from aorlta.rollout.engine_proc import InProcessEngine
from aorlta.rollout.fake_engine import FakeEngine, FakeEngineConfig

CFG = FakeEngineConfig(seed=1, min_tokens=60, max_tokens=150)


def test_engine_kwargs_are_the_section_6_1_table():
    kw = engine_kwargs(model_dir="/m", dtype="bfloat16", P=32, K=8, n_engines=3, max_len=28118,
                       gpu_memory_utilization=0.45, run_seed=5)
    assert kw == {"model": "/m", "tokenizer": "/m", "dtype": "bfloat16", "sampling_backend": "triton",
                  "enable_output_logprobs": True, "enable_prefix_caching": True, "disable_kvstore": True,
                  "disable_autotune": True, "max_num_seqs": math.ceil(32 * 8 / 3), "max_model_len": 28118,
                  "gpu_memory_utilization": 0.45, "seed": 5, "log_level": "warning"}
    assert [engine_port(i) for i in range(4)] == [20000, 22000, 24000, 26000]
    assert max_model_len(11696, 4096, 2067) == 11696 + 3 * 4096 + 2 * 2067


@pytest.mark.parametrize("bad, field", [({"sampling_backend": None}, "sampling_backend"),
                                        ({"enable_output_logprobs": False}, "enable_output_logprobs"),
                                        ({"enable_prefix_caching": False}, "enable_prefix_caching"),
                                        ({"dtype": "auto"}, "dtype")])
def test_start_refuses_an_engine_not_set_as_asked(rt, bad, field):
    # sampling_backend None resolves to "greedy" on non-NVIDIA hardware (A2), as the fake does.
    fake = FakeEngine(CFG, **{**rt.ENGINE_KW, **bad})
    with pytest.raises(RolloutStop, match=field):
        rt.make_pool([fake])
    assert server_arg_problems(dict(fake.tokenizer_manager.server_args.__dict__), expected_server_args())


class SeedBlind(InProcessEngine):
    def generate(self, ids, sp, *, timeout_s, extra=None):
        return super().generate(ids, dict(sp, seed=1), timeout_s=timeout_s, extra=extra)


class NoLogprobs(InProcessEngine):
    def generate(self, ids, sp, *, timeout_s, extra=None):
        res = super().generate(ids, sp, timeout_s=timeout_s, extra=extra)
        return dict(res, lp_vals=None, lp_ids=None)


def test_the_self_test_passes_a_healthy_engine_one_request_at_a_time(rt):
    fake = FakeEngine(CFG, **rt.ENGINE_KW)
    h = InProcessEngine(0, fake)
    h.start()
    tok = rt.SpecialByteTokenizer()
    test = run_self_test(h, [tok.encode(f"p{i}") for i in range(4)], 7, 30)
    assert test.passed and test.differ == 4 and test.aligned == 8 and test.requests == 8
    assert [r.sampling_params["max_new_tokens"] for r in fake.requests] == [128] * 8
    assert {r.sampling_params["temperature"] for r in fake.requests} == {0.7}
    assert len({r.sampling_params["seed"] for r in fake.requests}) == 8


@pytest.mark.parametrize("handle", [SeedBlind, NoLogprobs])
def test_the_self_test_refuses_an_engine_that_ignores_seeds_or_loses_logprobs(rt, handle):
    with pytest.raises(RolloutStop, match="self-test"):
        rt.make_pool([handle(0, FakeEngine(CFG, **rt.ENGINE_KW))])


def test_the_self_test_threshold_is_three_of_four(rt):
    class OneBlind(InProcessEngine):  # the seed is ignored for the first n prompts
        n = 0

        def generate(self, ids, sp, *, timeout_s, extra=None):
            if ids[-1] - 10000 - ord("0") < self.n:
                sp = dict(sp, seed=1)
            return super().generate(ids, sp, timeout_s=timeout_s, extra=extra)

    tok = rt.SpecialByteTokenizer()
    prompts = [tok.encode(f"p{i}") for i in range(4)]
    for n, ok in ((0, True), (1, True), (2, False)):
        h = OneBlind(0, FakeEngine(CFG, **rt.ENGINE_KW))
        h.n = n
        h.start()
        test = run_self_test(h, prompts, 7, 30)
        assert (test.differ, test.passed) == (4 - n, ok)


def test_logprobs_aligned():
    good = {"output_ids": [1, 2], "lp_vals": [-0.1, -2.0], "lp_ids": [1, 2]}
    assert logprobs_aligned(good)
    assert not logprobs_aligned(dict(good, lp_ids=[1, 3]))
    assert not logprobs_aligned(dict(good, lp_vals=[-0.1]))
    assert not logprobs_aligned(dict(good, lp_vals=[-0.1, float("nan")]))
    assert not logprobs_aligned(dict(good, lp_vals=[-0.1, 0.5]))
    assert not logprobs_aligned(dict(good, output_ids=[]))


def test_lpt_balances_estimates_and_caps_groups_per_engine():
    est = [100, 90, 80, 70, 60, 50, 40, 30]
    assert lpt_assign(est, 1) == [0] * 8
    a = lpt_assign(est, 2)
    loads = [sum(e for e, x in zip(est, a) if x == k) for k in range(2)]
    assert sorted(loads) == [260, 260] and [a.count(k) for k in range(2)] == [4, 4]
    skewed = [1000, 1, 1, 1, 1, 1, 1, 1]
    s = lpt_assign(skewed, 4)
    assert [s.count(k) for k in range(4)] == [2, 2, 2, 2]  # never more than ceil(8/4): max_num_seqs holds
    assert lpt_assign(skewed, 4) == s  # deterministic
    assert lpt_assign([5, 5, 5], 2) == [0, 1, 0]


def test_the_prefix_index_matches_brute_force():
    rng = random.Random(3)
    index, seen = PrefixIndex(), []

    def brute(q):
        best = 0
        for s in seen:
            n = 0
            while n < min(len(s), len(q)) and s[n] == q[n]:
                n += 1
            best = max(best, n)
        return best

    base = [rng.randrange(5) for _ in range(40)]
    for _ in range(400):
        cut = rng.randrange(len(base))
        q = base[:cut] + [rng.randrange(5) for _ in range(rng.randrange(30))]
        assert index.longest_prefix(q) == brute(q)
        if rng.random() < 0.6:
            index.insert(q)
            seen.append(q)
    index.clear()
    assert index.longest_prefix(base) == 0


def test_ideal_cached_tokens_counts_inputs_from_submit_and_outputs_from_reply(rt):
    pool = rt.make_pool([FakeEngine(CFG, **rt.ENGINE_KW)])
    sp = {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 50, "seed": 1, "stop_token_ids": [248046]}
    a = pool.generate(0, [1, 2, 3, 4], sp)
    assert a.ideal_cached_tokens == 0  # the self-test is not counted
    b = pool.generate(0, [1, 2, 3, 4, 5], sp)
    assert b.ideal_cached_tokens == 4
    follow = [1, 2, 3, 4] + a.result["output_ids"] + [9]
    assert pool.generate(0, follow, sp).ideal_cached_tokens == 4 + len(a.result["output_ids"])
    pool.note_flush(0)
    assert pool.generate(0, follow, sp).ideal_cached_tokens == 0


class Mortal(InProcessEngine):
    """Dies on the first request after `armed`, then restarts healthy."""

    armed = False

    def generate(self, ids, sp, *, timeout_s, extra=None):
        if self.armed:
            self.armed = False
            return {"error": "connection lost", "kind": "dead"}
        return super().generate(ids, sp, timeout_s=timeout_s, extra=extra)


def _sp(seed=1):
    return {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 20, "seed": seed,
            "stop_token_ids": [248046]}


@pytest.mark.parametrize("landing_ok", [True, False])
def test_a_dead_engine_restarts_and_serves_only_after_the_landing_check(rt, landing_ok):
    h = Mortal(0, FakeEngine(CFG, **rt.ENGINE_KW))
    calls = []
    pool = rt.make_pool([h], on_restart=lambda i: calls.append(i) or landing_ok)
    pool.begin_iteration(1)
    h.armed = True
    out = pool.generate(0, [1, 2], _sp())
    assert out.result["kind"] == "dead"
    pool.wait_restarts(30)
    assert calls == [0] and pool.available == [landing_ok]
    assert pool.deaths[0]["iteration"] == 1 and pool.stop_reason is None
    assert len(pool.self_tests) == 2  # the restarted engine passed its self-test again
    after = pool.generate(0, [1, 2], _sp())
    assert ("error" not in after.result) == landing_ok


def test_a_second_death_in_one_iteration_stops_the_run(rt):
    h = Mortal(0, FakeEngine(CFG, **rt.ENGINE_KW))
    pool = rt.make_pool([h], on_restart=lambda i: True)
    pool.begin_iteration(2)
    h.armed = True
    pool.generate(0, [1], _sp())
    pool.wait_restarts(30)
    assert pool.available == [True] and pool.stop_reason is None
    h.armed = True
    pool.generate(0, [1], _sp())
    assert pool.stop_reason and "second engine death in iteration 2" in pool.stop_reason
    assert pool.available == [False]
    pool.begin_iteration(3)  # a new iteration does not clear the stop
    assert pool.stop_reason


def test_a_timeout_kills_and_restarts_the_engine(rt):
    fake = FakeEngine(FakeEngineConfig(seed=1, min_tokens=60, max_tokens=150), **rt.ENGINE_KW)

    class SlowOnce:  # an engine whose next request does not return in time
        slow = False
        tokenizer_manager = fake.tokenizer_manager

        def generate(self, **kw):
            if self.slow:
                self.slow = False
                time.sleep(1.0)
            return fake.generate(**kw)

        def restart(self):
            fake.restart()

    engine = SlowOnce()
    pool = rt.make_pool([InProcessEngine(0, engine)], on_restart=lambda i: True, request_timeout_s=0.3)
    pool.begin_iteration(1)
    engine.slow = True
    t0 = time.monotonic()
    out = pool.generate(0, [1], _sp())
    assert out.result["kind"] == "timeout" and time.monotonic() - t0 < 2.0
    pool.wait_restarts(30)
    assert pool.deaths[0]["kind"] == "timeout" and pool.available == [True]


def test_a_run_needs_every_engine_in_service(rt):
    h = Mortal(0, FakeEngine(CFG, **rt.ENGINE_KW))
    pool = rt.make_pool([h])  # no landing-check hook: a restarted engine stays out of service
    pool.begin_iteration(1)
    h.armed = True
    pool.generate(0, [1], _sp())
    pool.wait_restarts(30)
    assert pool.available == [False] and pool.deaths[0]["restart"] == "awaiting landing check"
    mgr = rt.make_manager(pool, K=2)
    with pytest.raises(RolloutStop, match="not in service"):
        mgr.run(rt.make_prompts(rt.SpecialByteTokenizer(), 1), 2, "v")
