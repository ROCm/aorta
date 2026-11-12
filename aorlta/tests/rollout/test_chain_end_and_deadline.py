"""The chain-end callback and the generation deadline of the rollout manager (Section 6.6, steps 3-5; the RolloutPort's
contract), on the fake engine: each chain is handed over whole the moment it ends, so the orchestrator can release
groups during generation; and past the deadline no chain starts another turn: it ends `engine_error`, unscored, and
its id is reported as outstanding."""

from __future__ import annotations

import threading
import time

from aorlta.rollout.engine_proc import InProcessEngine
from aorlta.rollout.fake_engine import FakeEngineConfig

SHORT = dict(min_tokens=120, max_tokens=300)


class Delayed(InProcessEngine):
    """Every request takes `delay` seconds."""

    def __init__(self, index, engine, delay):
        super().__init__(index, engine)
        self.delay = delay

    def generate(self, ids, sp, *, timeout_s, extra=None):
        time.sleep(self.delay)
        return super().generate(ids, sp, timeout_s=timeout_s, extra=extra)


def test_every_chain_is_handed_over_whole_as_it_ends_and_before_the_rollout_returns(rt):
    tok = rt.SpecialByteTokenizer()
    pool = rt.make_pool(rt.fake_engines(1, FakeEngineConfig(seed=1, compile_error_rate=0.5, **SHORT)))
    mgr = rt.make_manager(pool, tok=tok, K=4)
    prompts = rt.make_prompts(tok, 3)
    ended, cands, lock = [], [], threading.Lock()

    def on_end(o):
        with lock:
            ended.append((time.monotonic(), o))

    result = mgr.run(prompts, 1, "v", on_candidate=cands.append, on_chain_end=on_end)
    t_return = time.monotonic()
    assert sorted(o.record.chain_id for _, o in ended) == sorted(c.chain_id for c in result.chains)
    assert all(t <= t_return for t, _ in ended)
    by_id = {o.record.chain_id: o for _, o in ended}
    for c in result.chains:
        o = by_id[c.chain_id]
        assert o.record is c or o.record == c
        assert (o.candidate is None) == (c.end_reason in ("engine_error", "quarantined"))
    assert sorted(x.candidate_id for x in cands) == sorted(o.candidate.candidate_id for _, o in ended if o.candidate)
    assert result.deadline_hit is False and result.outstanding == [] and result.metrics["cut_by_deadline"] == 0


def test_a_failing_chain_end_callback_is_a_stop_reason_never_dropped(rt):
    tok = rt.SpecialByteTokenizer()
    pool = rt.make_pool(rt.fake_engines(1, FakeEngineConfig(seed=1, **SHORT)))
    mgr = rt.make_manager(pool, tok=tok, K=2)

    def boom(o):
        raise RuntimeError("the ledger refused it")

    result = mgr.run(rt.make_prompts(tok, 2), 1, "v", on_chain_end=boom)
    assert len(result.stop_reasons) == len(result.chains) and "the ledger refused it" in result.stop_reasons[0]


def test_past_the_deadline_no_chain_starts_another_turn_and_each_is_outstanding(rt):
    tok = rt.SpecialByteTokenizer()
    fake = rt.fake_engines(1, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT))[0]  # every turn repairs
    pool = rt.make_pool([Delayed(0, fake, 0.4)])
    mgr = rt.make_manager(pool, tok=tok, K=2)
    prompts = rt.make_prompts(tok, 2)
    deadline = mgr.spans.now() + 0.6  # turn 0 (0.4 s) runs; the deadline passes during turn 1
    result = mgr.run(prompts, 1, "v", deadline=deadline)
    assert result.deadline_hit is True and sorted(result.outstanding) == sorted(c.chain_id for c in result.chains)
    for c in result.chains:
        assert c.end_reason == "engine_error"
        assert 1 <= len(c.turns) <= 2  # what was running at the deadline finished; nothing new started
        assert all(t.t_submit < deadline for t in c.turns)
    assert result.candidates == [] and result.samples == []  # cut chains are unscored
    assert result.metrics["cut_by_deadline"] == len(result.chains)


def test_a_deadline_already_passed_starts_nothing(rt):
    tok = rt.SpecialByteTokenizer()
    fakes = rt.fake_engines(1, FakeEngineConfig(seed=1, **SHORT))
    pool = rt.make_pool(fakes)
    before = len(fakes[0].requests)
    mgr = rt.make_manager(pool, tok=tok, K=2)
    result = mgr.run(rt.make_prompts(tok, 2), 1, "v", deadline=mgr.spans.now() - 1.0)
    assert len(fakes[0].requests) == before
    assert {c.end_reason for c in result.chains} == {"engine_error"} and all(not c.turns for c in result.chains)
    assert len(result.outstanding) == 4
