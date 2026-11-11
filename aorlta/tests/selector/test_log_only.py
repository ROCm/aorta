"""The selector in the reward service (Section 6.3): C3.T1, C3.T3, C3.T4 and C3.T5 through the real
service with fake adapters, and the failure modes of the planner.

Every test compares a run with a selector against a run of the same candidates with none, or against
an independent restatement (of the audit coin, of the safety switch).
"""

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from aorlta.config import ConfigError, RewardAsan, RewardSection, SelectorSection
from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeLegProfile, FakeOutcome
from aorlta.reward.clock import Clock
from aorlta.reward.profiles import load_profile
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.selector.base import CheckPlan, PlanRequest, SkipAtRandom, SkipEverything, SkipNothing, plan_json, wilson_upper
from aorlta.selector.log_only import SelectorHook, selector_hook
from selector_helpers import (
    PLAN_TIMEOUT_S,
    PLANNED,
    SCALE,
    SEED,
    Answers,
    Gated,
    Hanging,
    Raising,
    Recording,
    Sleeper,
    cand,
    hook,
    leg_view,
    outcomes,
    plan_with,
    restated_coin,
    reward_view,
    rig,
)

PLANNERS = {
    "skip_nothing": lambda: SkipNothing(),
    "skip_everything": lambda: SkipEverything(),
    "skip_at_random": lambda: SkipAtRandom(seed=3, p_skip=0.5),
    "no_planner": lambda: None,
}


def off_run(tmp_path, adapter, cands, **kw):
    r = rig(tmp_path / "off", adapter=adapter, **kw)
    try:
        return r.run(cands), r.service.stats().leg_runs
    finally:
        r.close()


def selector_run(tmp_path, selector, adapter, cands, **kw):
    r = rig(tmp_path / "on", selector=selector, adapter=adapter, **kw)
    try:
        return r.run(cands), r.service.stats().leg_runs, r
    finally:
        r.close()


def spans_of(r, cid, names=PLANNED):
    return [s for s in r.spans.spans() if s.attrs.get("candidate_id") == cid and s.name in names]


# --- C3.T1: in log_only every leg runs whatever the plan says ----------------------------------------


@pytest.mark.parametrize("name", sorted(PLANNERS))
def test_c3_t1_in_log_only_every_leg_runs_whatever_the_plan_says(tmp_path, name):
    """Same candidates, with no selector and with this planner: the same rewards, the same result on
    every leg, the same leg runs, and every plan logged as it was answered."""
    cands = [cand(i) for i in range(48)]
    base, base_runs = off_run(tmp_path, outcomes(cands), cands, scale=SCALE)
    planner = PLANNERS[name]()
    h = hook(planner=planner, plan_timeout_s=PLAN_TIMEOUT_S)
    results, runs, _ = selector_run(tmp_path, h, outcomes(cands), cands, scale=SCALE)
    assert runs == base_runs
    planned_skips = 0
    for a, b, c in zip(base, results, cands):
        assert reward_view(b) == reward_view(a), c.candidate_id
        assert leg_view(b) == leg_view(a), c.candidate_id
        assert b.full_check and not any(x.state == "skipped_by_plan" for x in b.legs.values())
        assert b.audit == restated_coin(SEED, c.candidate_id, 0.2)
        plan = b.plan
        if a.legs["gate.compile"].state != "clean":
            assert (plan["status"], plan["reason"]) == ("not_requested", "decided_before_plan")
            continue
        if planner is None:
            assert (plan["status"], plan["reason"]) == ("plan_unavailable", "no_planner")
            continue
        assert plan["status"] == "ok" and plan["applied"] is False and plan["skipped"] == []
        expected = plan_json(planner.plan(PlanRequest(c.candidate_id, "x", {}, list(PLANNED), {}, None)))["skip"]
        assert plan["skip"] == expected
        planned_skips += len(expected)
        for leg in plan["skip"]:  # a leg the plan skipped still ran
            assert b.legs[leg].state in ("clean", "finding", "fail", "unsettled") and b.legs[leg].attempts >= 1
    states = {x.state for r in base for x in r.legs.values()}
    assert {"finding", "unsettled", "fail", "not_run"} <= states  # the run had something to lose
    if name in ("skip_everything", "skip_at_random"):
        assert planned_skips > 0


def test_c3_t1_with_early_exits_a_planner_that_skips_everything_changes_no_reward(tmp_path):
    cands = [cand(i) for i in range(64)]
    base, _ = off_run(tmp_path, outcomes(cands, early_exit=True), cands, scale=SCALE)
    results, _, _ = selector_run(tmp_path, hook(planner=SkipEverything()), outcomes(cands, early_exit=True), cands,
                                 scale=SCALE)
    assert [reward_view(r) for r in results] == [reward_view(r) for r in base]
    assert all(r.full_check and not any(x.state == "skipped_by_plan" for x in r.legs.values()) for r in results)
    assert {r.reason for r in base} >= {None, "sanitizer_not_settled"}
    assert any(r.parts and r.parts["base"] in (-0.7, -0.8) for r in base)  # decided early by a reference or a fault


# --- log_only: never wait; log every plan that answers in time ---------------------------------------


def test_log_only_does_not_wait_for_the_plan_but_logs_every_plan_that_answers_in_time(tmp_path):
    """The planner answers 3 s into a 30 s deadline. The legs start at once; a candidate decided before
    its plan arrived is held until it does, so every plan is logged as ok. The 3 s are 0.3 s of real time: the legs
    must start within that, also on a busy host."""
    clock = Clock(0.1)
    cands = [cand(i) for i in range(24)]
    h = hook(planner=Sleeper(SkipEverything(), clock, 3.0), plan_timeout_s=30.0, threads=len(cands))
    results, _, run = selector_run(tmp_path, h, FakeAdapterConfig(), cands, clock=clock)
    held = 0
    for res, c in zip(results, cands):
        plan = res.plan
        assert plan["status"] == "ok", plan
        assert 3.0 - 1e-6 <= plan["answered_after_s"] < 30.0
        answer_at = plan["requested_at"] + plan["answered_after_s"]
        starts = [s.t_start for s in spans_of(run, c.candidate_id, ("ref.native", "ref.edge", "san.consan",
                                                                     "san.waitcheck"))]
        assert starts and max(starts) < answer_at  # every S2 leg started before the plan answered
        assert res.reward == 1.0 + -0.05 * (c.response_tokens / 4096)
        # decided before its plan answered, and published with it
        held += res.timings["total_s"] < res.timings["stages"]["S2"]["start_s"] + plan["answered_after_s"]
    assert held >= len(cands) // 2


def test_a_late_plan_is_plan_unavailable_timeout_and_the_result_does_not_wait_for_it(tmp_path):
    """The planner answers only after every result is out: the results were released at the plan
    deadline, and the late answers are counted and change nothing. The 100 s deadline is 0.2 s of real time; a
    request still queued at its deadline is dropped by the hook's threads, so only those the planner took answer."""
    gated = Gated(SkipEverything())
    h = hook(planner=gated, plan_timeout_s=100.0)
    cands = [cand(i) for i in range(8)]
    r = rig(tmp_path, selector=h)
    try:
        results = r.run(cands)
        assert not gated.release.is_set()
        for res in results:
            assert (res.plan["status"], res.plan["reason"]) == ("plan_unavailable", "timeout")
            assert res.plan["legs"] is None and res.plan["skip"] == [] and res.status == "scored"
            assert res.plan["answered_after_s"] is None
        gated.release.set()
        deadline = time.monotonic() + 30
        while h.stats().get("late_answers", 0) < gated.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert 1 <= gated.calls <= len(cands) and h.stats()["late_answers"] == gated.calls
        assert len(h.log_lines()) == len(cands)  # nothing logged twice, nothing rewritten
        assert all(line["result"]["plan"]["status"] == "plan_unavailable" for line in h.log_lines())
    finally:
        gated.release.set()
        r.close()


class ManualClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def wall(self):
        return 1.8e9 + self.t


def test_a_plan_answered_after_its_deadline_is_plan_unavailable_timeout():
    """The hook alone, on a clock the test moves: an answer timed past the deadline is a timeout even
    if it reaches the coordinator before the deadline's tick; a request still pending at its deadline
    expires; an answer after that is counted and dropped."""
    gated = Gated(SkipNothing())  # the hook's threads never answer; the test answers for them
    clock, posted = ManualClock(), []
    h = hook(planner=gated, plan_timeout_s=5.0)
    h.attach(profile=load_profile("fake"), clock=clock, post=posted.append)
    try:
        cs = [cand(i) for i in range(3)]
        cps = [h.on_submit(c, f"{i:064x}") for i, c in enumerate(cs)]
        for cp, c in zip(cps, cs):
            h.request(cp, c, {})
        plans = [SkipNothing().plan(PlanRequest(c.candidate_id, "x", {}, list(PLANNED), {}, None)) for c in cs]
        assert h.next_deadline() == clock.t + 5.0
        on_time = h.on_answer(cps[0].token, ("ok", plans[0], clock.t + 4.999))
        assert (on_time.status, on_time.answered_after_s) == ("ok", pytest.approx(4.999))
        late = h.on_answer(cps[1].token, ("ok", plans[1], clock.t + 5.001))
        assert (late.status, late.reason, late.plan) == ("plan_unavailable", "timeout", None)
        assert h.expire(clock.t + 4.9) == []
        assert h.expire(clock.t + 5.0) == [cps[2]] and (cps[2].status, cps[2].reason) == ("plan_unavailable", "timeout")
        assert h.on_answer(cps[2].token, ("ok", plans[2], clock.t + 5.1)) is None
        assert h.stats()["late_answers"] == 2 and h.next_deadline() is None
        assert h.record(cps[1])["status"] == "plan_unavailable" and h.record(cps[0])["skip"] == []
    finally:
        gated.release.set()
        h.close()


@pytest.mark.parametrize("planner, reason, rejected", [
    (Raising(), "planner_error:RuntimeError", []),
    (Answers(lambda req: CheckPlan("v/1", "p", {**plan_with(req, {}).legs, "san.mystery": {
        "decision": "skip", "p_finding": None, "score": None}}, None)), "rejected:unknown_leg", ["san.mystery"]),
    (Answers(lambda req: CheckPlan("v/1", "p", {k: v for k, v in plan_with(req, {}).legs.items() if k != "san.asan"},
                                   None)), "rejected:missing_leg", ["san.asan"]),
    (Answers(lambda req: CheckPlan("v/1", "p", {k: dict(v, decision="perhaps") for k, v in
                                                plan_with(req, {}).legs.items()}, None)), "rejected:bad_decision", []),
    (Answers(lambda req: CheckPlan("v/1", "p", {k: dict(v, score=10**400) for k, v in
                                                plan_with(req, {}).legs.items()}, None)), "rejected:bad_score", []),
    (Answers(lambda req: {"skip": ["san.asan"]}), "rejected:not_a_check_plan", []),
])
def test_a_planner_that_fails_or_breaks_the_interface_is_plan_unavailable_and_every_leg_runs(
        tmp_path, planner, reason, rejected):
    cands = [cand(i) for i in range(16)]
    base, base_runs = off_run(tmp_path, outcomes(cands), cands, scale=SCALE)
    h = hook(planner=planner, plan_timeout_s=PLAN_TIMEOUT_S)
    results, runs, _ = selector_run(tmp_path, h, outcomes(cands), cands, scale=SCALE)
    assert runs == base_runs
    for a, b in zip(base, results):
        assert reward_view(b) == reward_view(a) and leg_view(b) == leg_view(a)
        if b.plan["status"] != "not_requested":
            assert (b.plan["status"], b.plan["reason"]) == ("plan_unavailable", reason)
            assert b.plan["rejected_legs"] == rejected and b.plan["legs"] is None
    assert Raising.CANARY not in json.dumps(h.log_lines())


def test_the_plan_is_requested_after_s1_and_not_for_a_candidate_decided_before(tmp_path):
    cands = [cand(i) for i in range(12)]
    overrides = {(cands[0].candidate_id, "gate.static"): FakeOutcome("fail", "static"),
                 (cands[1].candidate_id, "gate.compile"): FakeOutcome("fail", "compile_error")}
    results, _, _ = selector_run(tmp_path, hook(planner=SkipNothing()), FakeAdapterConfig(overrides=overrides), cands)
    assert results[0].reward == -1.0 + -0.05 * (cands[0].response_tokens / 4096)
    for res in results[:2]:
        assert (res.plan["status"], res.plan["reason"], res.plan["requested_at"]) == ("not_requested",
                                                                                      "decided_before_plan", None)
    for res in results[2:]:
        assert res.plan["status"] == "ok"
        assert res.plan["requested_at"] >= res.legs["gate.compile"].t_end
        assert res.plan["requested_at"] >= max(x.t_end for n, x in res.legs.items() if n.startswith("gate."))
        assert res.plan["available_legs"] == list(PLANNED)


def test_a_reward_cache_hit_carries_its_own_coin_and_no_plan(tmp_path):
    text = "```cpp\n// the same kernel\n```"
    cands = [cand(i, text=text, tokens=500) for i in range(40)]
    r = rig(tmp_path, selector=hook(planner=SkipEverything(), audit_fraction=0.5))
    try:
        first = r.service.submit(cands[0]).result(timeout=60)  # fills the reward cache
        rest = r.run(cands[1:])
    finally:
        r.close()
    assert first.plan["status"] == "ok"
    assert all(x.timings.get("reward_cache_hit") for x in rest)
    for x, c in zip(rest, cands[1:]):
        assert (x.plan["status"], x.plan["reason"]) == ("not_requested", "reward_cache_hit")
        assert x.audit == restated_coin(SEED, c.candidate_id, 0.5) and x.reward == first.reward
    assert len({x.audit for x in rest}) == 2  # the coins differ between candidates of the same content


def test_the_plan_log_carries_ids_and_numbers_never_text(tmp_path):
    canary_src, canary_ctx = "SOURCE-CANARY-7f3a", "CONTEXT-CANARY-91b2"
    seen = {}

    def make(req):
        seen[req.candidate_id] = req.kernel_features["response_text"]
        return CheckPlan("v/1", f"p:{req.candidate_id}", plan_with(req, {"san.asan": "skip"}).legs,
                         f"{canary_ctx} {req.kernel_features['response_text']}")

    cands = [cand(i, text=f"```cpp\n// {canary_src} {i}\n```") for i in range(8)]
    h = hook(planner=Answers(make))
    results, _, _ = selector_run(tmp_path, h, FakeAdapterConfig(), cands)
    assert all(canary_src in seen[c.candidate_id] for c in cands)  # the planner reads the source
    path = h.write_plan_log(tmp_path / "plan-log.jsonl")
    text = path.read_text()
    assert canary_src not in text and canary_ctx not in text
    lines = [json.loads(t) for t in text.splitlines()]
    assert len(lines) == len(cands)
    assert all(line["result"]["plan"]["context_sha256"] and line["result"]["plan"]["context_chars"] > 20 for line in lines)
    assert {line["candidate"]["candidate_id"] for line in lines} == {c.candidate_id for c in cands}


def test_the_drain_deadline_and_close_never_leave_a_plan_or_a_result_waiting(tmp_path):
    hanging = Hanging()
    try:
        h = hook(planner=hanging, plan_timeout_s=1e5)
        r = rig(tmp_path / "drain", selector=h)
        try:
            futures = [r.service.submit(cand(i)) for i in range(8)]
            report = r.service.drain(r.clock.now() + 200.0)
            results = [f.result(timeout=30) for f in futures]
            assert report.resolved == 8
            assert all((x.plan["status"], x.plan["reason"]) == ("plan_unavailable", "drain_deadline") for x in results)
            assert all(x.status == "scored" for x in results)  # the legs had finished; only the plan was missing
        finally:
            r.close()
        h2 = hook(planner=hanging, plan_timeout_s=1e5)
        r2 = rig(tmp_path / "close", selector=h2)
        futures = [r2.service.submit(cand(i, iteration=2)) for i in range(4)]
        deadline = time.monotonic() + 30
        while not all(c.held is not None for c in r2.service._cands.values()) and time.monotonic() < deadline:
            time.sleep(0.01)
        r2.close()
        results = [f.result(timeout=10) for f in futures]
        assert all((x.plan["status"], x.plan["reason"]) == ("plan_unavailable", "service_closed") for x in results)
        assert all(x.status == "scored" for x in results)
    finally:
        hanging.release.set()


def test_the_leg_costs_the_planner_sees_are_running_medians_of_settled_runs(tmp_path):
    rec = Recording(SkipNothing())
    lat = FakeAdapterConfig(per_leg={"ref.native": FakeLegProfile(latency_s=4.0), "san.consan": FakeLegProfile(latency_s=9.0)})
    h = hook(planner=rec)
    rec.hook = h
    r = rig(tmp_path, selector=h, adapter=lat, scale=SCALE)
    try:
        for i in range(6):  # one at a time, so each request sees the runs before it
            r.service.submit(cand(i)).result(timeout=60)
    finally:
        r.close()
    reqs = [rec.requests[cand(i).candidate_id] for i in range(6)]
    assert reqs[0].leg_cost_s == {}  # nothing has run yet: absent, not free
    # 0.25 s of clock is 12.5 ms of real time at this scale: room for the fake's sleep to wake late on a busy host
    assert reqs[-1].leg_cost_s["ref.native"] == pytest.approx(4.0, abs=0.25)
    assert reqs[-1].leg_cost_s["san.consan"] == pytest.approx(9.0, abs=0.25)
    assert all(req.budget_s is None and req.available_legs == list(PLANNED) for req in reqs)
    assert all("response_text" in req.kernel_features and "legs" in req.kernel_features for req in reqs)


# --- configuration -----------------------------------------------------------------------------------


def test_a_run_config_gives_off_or_log_only_and_never_plan_mode():
    assert selector_hook(SelectorSection(mode="off"), None, audit_seed=1) is None
    h = selector_hook(SelectorSection(), None, audit_seed=1)
    assert isinstance(h, SelectorHook) and h.mode == "log_only" and h.audit_fraction == 0.2 and h.plan_timeout_s == 5.0
    with pytest.raises(ConfigError, match="plan"):
        SelectorSection(mode="plan")
    with pytest.raises(ValueError, match="endpoint"):
        selector_hook(SelectorSection(endpoint="http://selector:8000"), None, audit_seed=1)
    with pytest.raises(ValueError, match="D6"):
        SelectorHook(mode="plan", planner=SkipNothing(), audit_seed=1)
    with pytest.raises(ValueError, match="cap"):
        SelectorHook(mode="plan", planner=SkipNothing(), audit_seed=1, allow_plan_mode=True, scoring="capped")


# --- C3.T3: plan mode (built, off): a skipped leg never yields clean ---------------------------------


@pytest.mark.parametrize("name, f", [("skip_everything", 0.0), ("skip_at_random", 0.0), ("skip_at_random", 0.3),
                                     ("skip_everything", 0.5)])
def test_c3_t3_in_plan_mode_a_skipped_leg_never_yields_clean_and_full_check_is_false(tmp_path, name, f):
    cands = [cand(i) for i in range(64)]
    base, _ = off_run(tmp_path, outcomes(cands, early_exit=True), cands, scale=SCALE)
    h = hook("plan", PLANNERS[name](), audit_fraction=f, max_miss_rate=1.0, plan_timeout_s=PLAN_TIMEOUT_S)
    results, _, run = selector_run(tmp_path, h, outcomes(cands, early_exit=True), cands, scale=SCALE)
    skipped_samples = 0
    for a, b, c in zip(base, results, cands):
        skipped = sorted(n for n, x in b.legs.items() if x.state == "skipped_by_plan")
        assert skipped == sorted(b.plan["skipped"])  # what the hook skipped is what the record shows
        if b.plan["skipped"]:
            skipped_samples += 1
            assert b.full_check is False
            assert not StandinV11.claims_clean(b.reward, b.parts)
            assert b.reward is None or b.parts["base"] < 0
            assert b.plan["applied"] is True and not b.audit
            for leg in skipped:  # a skipped leg never ran
                assert b.legs[leg].attempts == 0 and b.legs[leg].worker_id is None
                assert not spans_of(run, c.candidate_id, (leg,))
        else:
            assert b.full_check is True
        if b.audit:
            assert not skipped and reward_view(b) == reward_view(a)
            assert b.plan["withheld"] == b.plan["skip"]
    assert skipped_samples > 0
    if f > 0:
        assert any(r.audit for r in results)
    assert any(StandinV11.claims_clean(a.reward, a.parts) for a, b in zip(base, results)
               if any(x.state == "skipped_by_plan" for x in b.legs.values()))  # clean under full checks


# --- C3.T4: the coin is fixed before the plan and no planner moves it --------------------------------


def test_c3_t4_the_audit_coin_is_fixed_before_the_plan_is_requested_and_no_planner_or_mode_moves_it(tmp_path):
    cands = [cand(i) for i in range(48)]
    expected = {c.candidate_id: restated_coin(SEED, c.candidate_id, 0.3) for c in cands}
    assert 0 < sum(expected.values()) < len(cands)
    for n, (name, mode) in enumerate([("skip_nothing", "log_only"), ("skip_everything", "log_only"),
                                      ("skip_at_random", "log_only"), ("no_planner", "log_only"),
                                      ("skip_everything", "plan"), ("skip_at_random", "plan")]):
        inner = PLANNERS[name]()
        rec = None if inner is None else Recording(inner)
        h = hook(mode, rec, audit_fraction=0.3, max_miss_rate=1.0)
        if rec is not None:
            rec.hook = h
        results, _, _ = selector_run(tmp_path / str(n), h, outcomes(cands), cands)
        lines = {line["candidate"]["candidate_id"]: line for line in h.log_lines()}
        for res, c in zip(results, cands):
            cid = c.candidate_id
            assert res.audit == expected[cid] and lines[cid]["audit"]["coin"] == expected[cid]
            if rec is not None and cid in rec.seen:
                assert rec.seen[cid] == expected[cid]  # already drawn when the planner was asked
            if res.plan["requested_at"] is not None:
                assert lines[cid]["audit"]["drawn_at"] <= res.plan["requested_at"]


# --- C3.T5: in plan mode the switch turns skipping off for a leg that misses -------------------------


def switch_state_after(observations, m):
    """The switch state after these observations (the doc's rule with the one-miss arming margin)."""
    state = "armed" if wilson_upper(1, 1) <= m else "warming"
    k = 0
    for t, miss in enumerate(observations, 1):
        k += miss
        if state == "warming" and wilson_upper(k + 1, t + 1) <= m:
            state = "armed"
        elif state == "armed" and wilson_upper(k, t) > m:
            state = "tripped"
    return state


def test_c3_t5_in_plan_mode_the_switch_stops_skipping_a_leg_whose_audited_miss_rate_exceeds_the_limit(tmp_path):
    """Every leg clean, except ASan findings from candidate 40 on. Audited samples (f = 0.5) feed the
    switch; an unaudited one skips exactly the legs the switch has armed at that moment."""
    m, f, n = 0.3, 0.5, 100
    cands = [cand(i) for i in range(n)]
    overrides = {(c.candidate_id, "san.asan"): FakeOutcome("finding") for c in cands[40:]}
    h = hook("plan", SkipEverything(), audit_fraction=f, max_miss_rate=m, plan_timeout_s=PLAN_TIMEOUT_S)
    section = RewardSection(profile="fake", reward_fn="standin.v11", sim_workers=8, asan=RewardAsan(max_wait_s=0.5))
    r = rig(tmp_path, selector=h, adapter=FakeAdapterConfig(overrides=overrides), reward=section, scale=SCALE)
    try:
        results = [r.service.submit(c).result(timeout=60) for c in cands]  # one at a time: a known order
    finally:
        r.close()
    coins = [restated_coin(SEED, c.candidate_id, f) for c in cands]
    asan_obs, other_obs = [], []
    states_before = []
    for i in range(n):
        states_before.append((switch_state_after(asan_obs, m), switch_state_after(other_obs, m)))
        if coins[i]:
            asan_obs.append(i >= 40)  # the plan skips ASan and ASan has a finding: a miss
            other_obs.append(False)
    seen = [s for s, _ in states_before]
    assert "armed" in seen and seen[-1] == "tripped"  # the test exercises both transitions
    for i, (res, (asan_state, other_state)) in enumerate(zip(results, states_before)):
        assert res.audit == coins[i]
        if res.audit:
            assert res.plan["skipped"] == [] and res.full_check
            continue
        expect = [leg for leg in PLANNED if (asan_state if leg == "san.asan" else other_state) == "armed"]
        assert sorted(res.plan["skipped"]) == sorted(expect), i
        if asan_state == "tripped":
            assert "san.asan" in res.plan["withheld"] and res.legs["san.asan"].state in ("clean", "finding")
    assert h.switch.state("san.asan") == "tripped" and h.switch.state("san.consan") == "armed"
    hidden = [i for i, res in enumerate(results) if i >= 40 and "san.asan" in res.plan["skipped"]]
    assert hidden  # defects plan mode hid before the switch fired; the audits found their kind
