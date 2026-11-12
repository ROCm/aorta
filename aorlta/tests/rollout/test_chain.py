"""C1.T5 and C1.T6 (design Section 6.1), and the per-turn seeds of C1.T3, on the fake engine.

C1.T6 covers every edge of the chain state machine:
  SUBMIT -> VALIDATE -> END(logprob_error) | END(engine_error)
  VALIDATE -> PARSE -> END(no_code_block)
  PARSE -> COMPILE -> END(compiled)
  COMPILE fail -> END(turns_exhausted) at t == max_repair_turns
  COMPILE fail -> FEEDBACK -> LEAK CHECK -> END(quarantined)
  LEAK CHECK pass -> END(context_limit)
  LEAK CHECK pass -> SUBMIT(t+1)
plus asynchronous turns (no barrier) and one request per chain per turn.
"""

from __future__ import annotations

import threading
import time

import pytest

from aorlta.ids import IterationSeeds, sha256_ids, sha256_text
from aorlta.rollout.chain import SCORED_ENDS, TRAINED_ENDS, chain_requests_expected
from aorlta.rollout.engine_pool import RolloutStop
from aorlta.rollout.engine_proc import InProcessEngine
from aorlta.rollout.fake_engine import COMPILE_ERROR_MARKER, FakeEngineConfig
from aorlta.rollout.feedback import HEADER

SHORT = dict(min_tokens=120, max_tokens=300)


class SpyLeak:
    name = "fake.leak_check"

    def __init__(self, log: list, verdict=True):
        self.log, self.verdict = log, verdict

    def check(self, feedback, prompt):
        self.log.append(("leak", sha256_text(feedback)))
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict(feedback) if callable(self.verdict) else self.verdict


class Armed(InProcessEngine):
    """A healthy engine through start-up and self-test; once armed, `fault(res)` rewrites each reply
    (or replaces it with an error), as an engine that degrades mid-run would."""

    def __init__(self, index, engine, fault):
        super().__init__(index, engine)
        self.fault, self.armed = fault, False

    def generate(self, ids, sp, *, timeout_s, extra=None):
        res = super().generate(ids, sp, timeout_s=timeout_s, extra=extra)
        return self.fault(dict(res)) if self.armed and "error" not in res else res


def _bad_logprobs(res):
    res["lp_ids"] = res["lp_ids"][:-1] + [res["lp_ids"][-1] + 1]
    return res


def _engine_error(res):
    return {"error": "injected engine error", "kind": "engine"}


def _run(rt, cfg: FakeEngineConfig, *, P=3, K=4, engines=1, leak=None, tok=None, prompts=None, fault=None, **kw):
    fakes = rt.fake_engines(engines, cfg)
    handles = [Armed(i, f, fault) for i, f in enumerate(fakes)] if fault else fakes
    pool = rt.make_pool(handles)
    for h in pool.handles:
        h.armed = True
    before = [len(f.requests) for f in fakes]
    tok = tok or rt.SpecialByteTokenizer()
    mgr = rt.make_manager(pool, tok=tok, leak_check=leak, K=K, **kw)
    prompts = prompts or rt.make_prompts(tok, P)
    got = []
    result = mgr.run(prompts, 1, "wp2-test:v0", on_candidate=got.append)
    reqs = [r for f, b in zip(fakes, before) for r in f.requests[b:]]
    return result, reqs, got, prompts, mgr


def _ends(result) -> set:
    return {c.end_reason for c in result.chains}


def test_c1_t6_compiled_at_turn0(rt):
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, **SHORT))
    assert _ends(result) == {"compiled"} and all(len(c.turns) == 1 for c in result.chains)
    assert all(c.turns[0].compile.ok for c in result.chains)
    assert len(got) == len(result.samples) == len(result.chains)


def test_c1_t6_no_code_block_ends_without_a_repair_turn(rt):
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, no_code_block_rate=1.0, **SHORT))
    assert _ends(result) == {"no_code_block"}
    assert all(len(c.turns) == 1 and c.turns[0].compile is None for c in result.chains)
    assert len(reqs) == len(result.chains)  # nothing was resubmitted
    assert len(got) == len(result.samples) == len(result.chains)  # scored (parse gate) and trained


def test_c1_t6_turns_exhausted_after_two_repairs(rt):
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT))
    assert _ends(result) == {"turns_exhausted"}
    for c in result.chains:
        assert [t.turn for t in c.turns] == [0, 1, 2] and c.final_turn == 2
        assert all(not t.compile.ok for t in c.turns)
        assert [t.feedback_sha256 is not None for t in c.turns] == [True, True, False]
        assert [t.leak_check for t in c.turns] == ["not_run"] * 3  # stand-in run, no leak check configured
    assert len(got) == len(result.chains)


def test_c1_t6_repair_then_compiled_and_the_append_rule_on_the_wire(rt):
    result, reqs, got, prompts, mgr = _run(rt, FakeEngineConfig(seed=3, compile_error_rate=0.5, **SHORT), P=4, K=8)
    rescued = [c for c in result.chains if c.end_reason == "compiled" and len(c.turns) >= 2]
    assert rescued, "no chain failed to compile and was then repaired"
    assert "turns_exhausted" in _ends(result) and any(len(c.turns) == 1 for c in result.chains)
    # Each repair turn's input is the previous input + reply + the feedback suffix; check the hashes.
    tok = rt.SpecialByteTokenizer()
    from aorlta.rollout.feedback import build_feedback
    from aorlta.rollout.templates import next_turn_ids, turn0_ids
    by_id = {p.prompt_id: p for p in prompts}
    for c in result.chains:
        ids = turn0_ids(by_id[c.prompt_id], tok, "off")
        for t, turn in enumerate(c.turns):
            assert turn.input_ids_sha256 == sha256_ids(ids) and turn.input_len == len(ids)
            if turn.compile is not None and not turn.compile.ok and t < len(c.turns) - 1:
                fb = build_feedback(turn.compile.errors).text
                assert turn.feedback_sha256 == sha256_text(fb)
                assert f"`{COMPILE_ERROR_MARKER}`" in fb
                ids = next_turn_ids(ids, turn.response_ids, fb, tok, "off")
    # The training sample of a chain is its final turn's exact input and response.
    samples = {s.chain_id: s for s in result.samples}
    for c in rescued:
        s = samples[c.chain_id]
        assert sha256_ids(s.prompt_ids) == c.turns[-1].input_ids_sha256
        assert s.response_ids == c.turns[-1].response_ids and s.response_logprobs == c.turns[-1].response_logprobs
        assert s.group_key == (1, c.prompt_id)


def test_c1_t6_logprob_error_is_scored_but_never_trained(rt):
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT), fault=_bad_logprobs)
    assert _ends(result) == {"logprob_error"}
    assert all(len(c.turns) == 1 and c.turns[0].response_logprobs == [] for c in result.chains)
    assert len(got) == len(result.chains) and result.samples == []
    assert result.metrics["logprobs_aligned_turns"] == 0


def test_c1_t6_the_fakes_own_bad_logprobs_are_caught_by_the_self_test(rt):
    with pytest.raises(RolloutStop, match="self-test"):
        rt.make_pool(rt.fake_engines(1, FakeEngineConfig(seed=1, bad_logprobs_rate=1.0, **SHORT)))


def test_c1_t6_engine_error_is_unscored(rt):
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, **SHORT), fault=_engine_error)
    assert _ends(result) == {"engine_error"}
    assert got == [] and result.samples == [] and all(c.turns == [] and c.final_turn == -1 for c in result.chains)


def test_c1_t6_a_failed_repair_request_is_an_engine_error_too(rt):
    calls = {"n": 0}

    def second_turn_fails(res):
        return res if res["prompt_tokens"] < 400 else _engine_error(res)

    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT),
                                   fault=second_turn_fails, P=1, K=2)
    assert _ends(result) == {"engine_error"} and got == []
    assert all(len(c.turns) >= 1 and c.final_turn == len(c.turns) - 1 for c in result.chains)


class FinishAbort(InProcessEngine):
    def generate(self, ids, sp, *, timeout_s, extra=None):
        res = super().generate(ids, sp, timeout_s=timeout_s, extra=extra)
        if "error" not in res and sp["max_new_tokens"] != 128:  # leave the self-test alone
            res["finish_reason"] = {"type": "abort", "message": "aborted"}
        return res


def test_c1_t6_an_unknown_finish_reason_is_an_engine_error(rt):
    fake = rt.fake_engines(1, FakeEngineConfig(seed=1, **SHORT))[0]
    pool = rt.make_pool([FinishAbort(0, fake)])
    result = rt.make_manager(pool, K=2).run(rt.make_prompts(rt.SpecialByteTokenizer(), 2), 1, "v")
    assert _ends(result) == {"engine_error"} and result.candidates == []


def test_c1_t6_context_limit_makes_the_last_answer_final(rt):
    tok = rt.SpecialByteTokenizer()
    prompts = rt.make_prompts(tok, 2)
    from aorlta.rollout.templates import turn0_ids
    longest = max(len(turn0_ids(p, tok, "off")) for p in prompts)
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT), P=2,
                                   prompts=prompts, max_new_tokens=600, max_model_len=longest + 600 + 50)
    assert _ends(result) == {"context_limit"}
    assert all(len(c.turns) == 1 for c in result.chains) and len(reqs) == len(result.chains)
    assert len(got) == len(result.samples) == len(result.chains)
    assert all(g.response_tokens == len(c.turns[0].response_ids) for g, c in
               zip(sorted(got, key=lambda g: g.candidate_id), sorted(result.chains, key=lambda c: c.chain_id)))


def test_c1_t5_the_leak_check_runs_before_the_append(rt):
    log: list = []
    tok = rt.SpecialByteTokenizer(log=log)
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT), tok=tok,
                                   leak=SpyLeak(log), P=2, K=2)
    feedback_encodes = [i for i, (kind, text) in enumerate(log) if kind == "encode" and HEADER in text]
    leaks = [i for i, (kind, _) in enumerate(log) if kind == "leak"]
    assert len(feedback_encodes) == len(leaks) == 2 * 2 * 2  # two repair turns per chain
    pending: dict[str, int] = {}  # checks not yet followed by their append, by feedback hash
    for kind, payload in log:  # in the order things happened, across all chain threads
        if kind == "leak":
            pending[payload] = pending.get(payload, 0) + 1
        elif HEADER in payload:
            h = sha256_text(payload.split("\n<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
            assert pending.get(h, 0) > 0, "feedback entered the token sequence before its leak check"
            pending[h] -= 1
    assert all(t.leak_check == "pass" for c in result.chains for t in c.turns[:2])


@pytest.mark.parametrize("verdict", [False, RuntimeError("leak service down"), "yes"])
def test_c1_t5_a_failing_leak_check_quarantines_and_nothing_is_submitted(rt, verdict):
    log: list = []
    tok = rt.SpecialByteTokenizer(log=log)
    result, reqs, got, _, _ = _run(rt, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT), tok=tok,
                                   leak=SpyLeak(log, verdict), P=2, K=2)
    assert _ends(result) == {"quarantined"}
    assert len(reqs) == len(result.chains)  # turn 0 only: the failed feedback was never submitted
    assert not any(kind == "encode" and HEADER in text for kind, text in log)  # and never appended
    assert got == [] and result.samples == []  # never scored or trained
    for c in result.chains:
        assert len(c.turns) == 1 and c.turns[0].leak_check == "fail" and c.turns[0].response_ids == []
        assert c.turns[0].feedback_sha256 is not None  # logged by hash only


def test_c1_t5_a_prompt_of_another_source_is_refused_and_only_the_none_leak_check_is_built(rt):
    tok = rt.SpecialByteTokenizer()
    pool = rt.make_pool(rt.fake_engines(1))
    other = rt.make_prompts(tok, 1, source="other")
    with pytest.raises(RolloutStop, match="source"):
        rt.make_manager(pool, K=2).run(other, 1, "v")  # another source's prompt in a stand-in run
    from aorlta.rollout.chain import make_leak_check
    assert make_leak_check("none") is None
    with pytest.raises(RolloutStop, match="no leak check named"):
        make_leak_check("standin.leak_check")


def test_c1_t6_one_request_per_chain_per_turn_with_the_turns_own_seed(rt):
    result, reqs, got, prompts, mgr = _run(rt, FakeEngineConfig(seed=5, compile_error_rate=0.6, **SHORT), P=3, K=4)
    assert len(reqs) == chain_requests_expected(result.chains) == result.metrics["requests"]
    assert all("n" not in r.sampling_params for r in reqs)
    seeds = IterationSeeds(7, 1, [p.prompt_id for p in prompts], 4, 3)
    sent = sorted((r.input_ids_sha256, r.sampling_params["seed"]) for r in reqs)
    expected = sorted((t.input_ids_sha256, seeds.seed_for(c.prompt_id, c.k, t.turn)) for c in result.chains
                      for t in c.turns)
    assert sent == expected
    for c in result.chains:
        assert [t.seed for t in c.turns] == [seeds.seed_for(c.prompt_id, c.k, t.turn) for t in c.turns]
        assert len({t.seed for t in c.turns}) == len(c.turns)  # each turn has its own seed
    assert max(len(c.turns) for c in result.chains) == 3


def test_c1_t3_rollout_seeds_differ_by_turn_and_collisions_are_recorded(rt):
    result, reqs, got, prompts, mgr = _run(rt, FakeEngineConfig(seed=2, compile_error_rate=1.0, **SHORT), P=2, K=2)
    for c in result.chains:
        turn_seeds = [t.seed for t in c.turns]
        assert len(set(turn_seeds)) == 3 and all(0 <= s < 2**31 for s in turn_seeds)
    assert result.metrics["salted_seeds"] == []


class Delayed(InProcessEngine):
    """Turn-0 requests of one prompt (marked by its text) are slow; everything else is immediate."""

    def __init__(self, index, engine, slow_ids, delay):
        super().__init__(index, engine)
        self.slow_ids, self.delay = slow_ids, delay

    def generate(self, ids, sp, *, timeout_s, extra=None):
        if list(ids) == self.slow_ids:
            time.sleep(self.delay)
        return super().generate(ids, sp, timeout_s=timeout_s, extra=extra)


def test_c1_t6_turns_are_asynchronous_and_candidates_stream(rt):
    tok = rt.SpecialByteTokenizer()
    prompts = rt.make_prompts(tok, 2)
    from aorlta.rollout.templates import turn0_ids
    slow = turn0_ids(prompts[0], tok, "off")
    fake = rt.fake_engines(1, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT))[0]
    pool = rt.make_pool([Delayed(0, fake, slow, 1.5)])
    mgr = rt.make_manager(pool, tok=tok, K=2)
    stamps = []
    result = mgr.run(prompts, 1, "v", on_candidate=lambda c: stamps.append((time.time(), c.candidate_id)))
    a = [c for c in result.chains if c.prompt_id == prompts[0].prompt_id]
    b = [c for c in result.chains if c.prompt_id == prompts[1].prompt_id]
    # Prompt B's chains finished all three turns while prompt A's turn 0 was still running.
    assert all(cb.turns[2].t_submit < ca.turns[0].t_done for cb in b for ca in a)
    first_b = min(t for t, cid in stamps if prompts[1].prompt_id in cid)
    last_a_end = max(c.turns[-1].t_done for c in a)
    assert first_b < last_a_end  # handed over while other chains were still generating


class SeedBlind(InProcessEngine):
    """Passes the read-back (triton) but ignores every request seed except during the self-test."""

    def generate(self, ids, sp, *, timeout_s, extra=None):
        if sp["max_new_tokens"] != 128:
            sp = dict(sp, seed=1)
        return super().generate(ids, sp, timeout_s=timeout_s, extra=extra)


def test_identical_groups_above_5_percent_stop_the_run(rt):
    fake = rt.fake_engines(1, FakeEngineConfig(seed=1, **SHORT))[0]
    pool = rt.make_pool([SeedBlind(0, fake)])
    result = rt.make_manager(pool, K=4).run(rt.make_prompts(rt.SpecialByteTokenizer(), 3), 1, "v")
    m = result.metrics
    assert m["distinct_turn0_per_group"] == [1, 1, 1] and m["groups_all_identical"] == 3
    assert any("sampler suspected" in r for r in result.stop_reasons)
    healthy, *_ = _run(rt, FakeEngineConfig(seed=1, **SHORT), P=3, K=4)
    assert healthy.metrics["distinct_turn0_mean"] == 4 and healthy.stop_reasons == []


def test_prompts_go_to_engines_by_lpt_and_groups_stay_on_one_engine(rt):
    result, reqs, got, prompts, mgr = _run(rt, FakeEngineConfig(seed=1, compile_error_rate=0.5, **SHORT), P=6, K=3,
                                           engines=3)
    by_prompt = {}
    for c in result.chains:
        by_prompt.setdefault(c.prompt_id, set()).add(c.engine_id)
    assert all(len(v) == 1 for v in by_prompt.values())  # all K of a group on one engine
    assert sorted(m["groups"] for m in result.metrics["engines"].values()) == [2, 2, 2]
    assert result.metrics["engine_finish_spread_s"] is not None


def test_repair_turns_reuse_the_prefix_and_ideal_is_the_previous_sequence(rt):
    cfg = FakeEngineConfig(seed=1, compile_error_rate=1.0, prefix_cache=True, **SHORT)
    result, *_ = _run(rt, cfg, P=2, K=2)
    for c in result.chains:
        for prev, turn in zip(c.turns, c.turns[1:]):
            assert turn.ideal_cached_tokens == prev.input_len + len(prev.response_ids)
            assert turn.cached_tokens == turn.ideal_cached_tokens - 1  # the last sampled token was never fed
    m = result.metrics
    assert m["repair_turns"] == 8 and m["repair_cache_ok"] == 8 and m["repair_cache_miss_rate"] == 0


def test_a_cache_miss_is_counted(rt):
    class NoCache(InProcessEngine):
        def generate(self, ids, sp, *, timeout_s, extra=None):
            return dict(super().generate(ids, sp, timeout_s=timeout_s, extra=extra), cached_tokens=0)

    fake = rt.fake_engines(1, FakeEngineConfig(seed=1, compile_error_rate=1.0, **SHORT))[0]
    pool = rt.make_pool([NoCache(0, fake)])
    result = rt.make_manager(pool, K=2).run(rt.make_prompts(rt.SpecialByteTokenizer(), 1), 1, "v")
    assert result.metrics["repair_cache_miss_rate"] == 1.0


def test_candidates_carry_the_final_answer(rt):
    result, reqs, got, prompts, mgr = _run(rt, FakeEngineConfig(seed=4, compile_error_rate=0.5, **SHORT), P=2, K=4)
    tok = rt.SpecialByteTokenizer()
    chains = {c.chain_id: c for c in result.chains}
    assert {g.candidate_id for g in got} == {c.chain_id for c in result.chains if c.end_reason in SCORED_ENDS}
    for g in got:
        final = chains[g.candidate_id].turns[-1]
        assert g.response_text == tok.decode(final.response_ids[:-1] if final.response_ids[-1] == 248046 else final.response_ids)
        assert g.response_tokens == len(final.response_ids) and g.max_new_tokens == 600
        assert g.check_profile == "standin-static" and g.policy_version == "wp2-test:v0"
        assert g.rollout_compile == final.compile
    assert {s.chain_id for s in result.samples} == {c.chain_id for c in result.chains if c.end_reason in TRAINED_ENDS}


def test_a_failed_candidate_hand_over_is_a_stop_reason(rt):
    def boom(c):
        raise RuntimeError("reward service refused")

    fakes = rt.fake_engines(1)
    result = rt.make_manager(rt.make_pool(fakes), K=2).run(rt.make_prompts(rt.SpecialByteTokenizer(), 1), 1, "v",
                                                           on_candidate=boom)
    assert any("hand-over failed" in r for r in result.stop_reasons)


def test_the_iteration_checks_its_prompts(rt):
    tok = rt.SpecialByteTokenizer()
    pool = rt.make_pool(rt.fake_engines(1))
    prompts = rt.make_prompts(tok, 2)
    with pytest.raises(RolloutStop, match="repeats"):
        rt.make_manager(pool, K=2).run([prompts[0], prompts[0]], 1, "v")
    with pytest.raises(RolloutStop, match="rollout.P"):
        rt.make_manager(pool, K=2, P=3).run(prompts, 1, "v")
    with pytest.raises(RolloutStop, match="no room"):
        rt.make_manager(pool, K=2, max_model_len=10).run(prompts, 1, "v")


def test_the_dev_config_gives_the_section_6_1_settings():
    import dataclasses
    from pathlib import Path

    from aorlta.config import load_config
    from aorlta.rollout.chain import RolloutSettings

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "dev-standin.yaml")
    s = RolloutSettings.from_config(cfg, 28118)
    assert (s.K, s.P, s.max_repair_turns, s.thinking, s.temperature, s.top_p, s.top_k, s.max_new_tokens,
            s.max_feedback_chars, s.source, s.check_profile) == (8, 32, 2, "off", 0.7, 0.95, -1, 4096, 2000,
                                                               "standin", "standin-static")
    assert s.sampling_params(5) == {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 4096,
                                    "seed": 5, "stop_token_ids": [248046]}
    for field, value in (("parse_adapter", "standin.other"), ("feedback_template", "standin.other")):
        bad = dataclasses.replace(cfg, rollout=dataclasses.replace(cfg.rollout, **{field: value}))
        with pytest.raises(RolloutStop, match="not available"):
            RolloutSettings.from_config(bad, 28118)


def test_only_built_compile_adapters_can_be_named(tmp_path):
    from aorlta.rollout.compile_check import CompileUnavailable, FakeCompiler, make_compiler

    assert isinstance(make_compiler("fake.compile", tmp_path), FakeCompiler)
    with pytest.raises(CompileUnavailable, match="not available"):
        make_compiler("standin.other", tmp_path)
