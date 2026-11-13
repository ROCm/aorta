"""The offline evaluator (Section 6.3): C3.T2 on a fake log with a known miss pattern, the option
rewards and advantages on a log small enough to score by hand, and two end-to-end checks against the
real service: the log it writes reproduces independent counts, and what the evaluator predicts for
option A + C from a log-only run is what plan mode then really scores.
"""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction

import pytest

from aorlta.ids import candidate_id
from aorlta.reward.profiles import load_profile
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.schemas import Candidate, LegResult, RewardResult, TaskRef
from aorlta.selector.base import SkipAtRandom, wilson_upper
from aorlta.selector.log_only import CandidatePlan, plan_log_line, plan_record
from aorlta.selector.plan_eval import (
    UNDETERMINED,
    Evaluator,
    PlanEvalError,
    evaluate,
    fake_log,
    main,
    parse_plan_log,
    read_plan_log,
    replay_switch,
    write_jsonl,
)
from selector_helpers import PLAN_TIMEOUT_S, PLANNED, SCALE, SEED, cand, hook, outcomes, restated_coin, rig

GATES = ("gate.parse", "gate.contract", "gate.static", "gate.reference", "gate.compile", "gate.no_artifact",
         "gate.spec_invalid", "gate.static_reject", "gate.no_change_on_failing_parent")
COST = {"ref.native": 11.0, "ref.edge": 11.0, "san.consan": 11.0, "san.waitcheck": 11.0, "san.asan": 8.5}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    path = write_jsonl(tmp_path_factory.mktemp("fake") / "fake-log.jsonl", fake_log())
    samples = read_plan_log(path)
    return samples, evaluate(samples, audit_fractions=[0.1, 0.5])


# --- C3.T2: the fake log's known answers -------------------------------------------------------------


def truth(i: int) -> dict:
    """The fake log's pattern, written out again from its docstring."""
    a, b, c, d = i % 8, i % 3, i % 5, i % 7
    t = {"compile_error": d == 6, "plan_ok": d <= 4, "native_fail": c == 4 and b == 2}
    t["legs"] = {
        "ref.native": "fail" if t["native_fail"] else "clean",
        "ref.edge": "clean",
        "san.consan": "finding" if c == 0 else "clean",
        "san.waitcheck": "unsettled" if (a == 7 and c == 1) else ("finding" if a == 1 else "clean"),
        "san.asan": "not_run" if t["native_fail"] else ("finding" if a == 3 else "clean"),
    }
    t["skip"] = set()
    if b == 0:
        t["skip"].add("san.consan")
    if a in (0, 2, 4, 6):
        t["skip"].add("san.waitcheck")
    if (a != 3 and b != 0) or (a == 3 and d == 0):
        t["skip"].add("san.asan")
    return t


def brute_force(indices) -> dict:
    out = {}
    for leg in PLANNED:
        rows = [truth(i) for i in indices]
        plans = [t for t in rows if not t["compile_error"] and t["plan_ok"]]
        skipped = [t for t in plans if leg in t["skip"]]
        settled = [t for t in plans if t["legs"][leg] in ("clean", "finding", "fail")]
        defects = [t for t in settled if t["legs"][leg] in ("finding", "fail")]
        missed = [t for t in defects if leg in t["skip"]]
        unknown = [t for t in plans if t["legs"][leg] not in ("clean", "finding", "fail")]
        ran = [t for t in rows if not t["compile_error"] and t["legs"][leg] != "not_run"]
        cost = lambda t: 120.0 if t["legs"][leg] == "unsettled" else COST[leg]  # noqa: E731
        out[leg] = {"plans": len(plans), "skipped": len(skipped), "settled": len(settled), "defects": len(defects),
                    "missed": len(missed), "unknown": len(unknown), "unknown_skipped": sum(leg in t["skip"] for t in unknown),
                    "run_s": sum(cost(t) for t in ran),
                    "saved_s": sum(cost(t) for t in skipped if t["legs"][leg] != "not_run")}
    return out


# Derived by hand from the pattern: over 2520 samples every (i mod 8, 3, 5, 7) appears exactly 3 times.
HAND = {
    "ref.native": {"plans": 1800, "skipped": 0, "settled": 1800, "defects": 120, "missed": 0, "unknown": 0},
    "ref.edge": {"plans": 1800, "skipped": 0, "settled": 1800, "defects": 0, "missed": 0, "unknown": 0},
    "san.consan": {"plans": 1800, "skipped": 600, "settled": 1800, "defects": 360, "missed": 120, "unknown": 0},
    "san.waitcheck": {"plans": 1800, "skipped": 900, "settled": 1755, "defects": 225, "missed": 0, "unknown": 45},
    "san.asan": {"plans": 1800, "skipped": 1095, "settled": 1680, "defects": 210, "missed": 42, "unknown": 120,
                 "unknown_skipped": 108},
}
HAND_RATES = {  # skip rate, miss rate per sample, share of defects missed
    "san.consan": (Fraction(1, 3), Fraction(1, 15), Fraction(1, 3)),
    "san.waitcheck": (Fraction(1, 2), Fraction(0), Fraction(0)),
    "san.asan": (Fraction(73, 120), Fraction(1, 40), Fraction(1, 5)),
}


def test_c3_t2_the_evaluator_reproduces_the_fake_logs_skip_and_miss_rates_exactly(fake):
    samples, report = fake
    legs = report["legs"]
    assert len(samples) == 2520 and report["log"]["reproduced"] == 2520
    for leg, expected in HAND.items():
        assert {k: legs[leg][k] for k in expected} == expected, leg
    for leg, (skip, miss, share) in HAND_RATES.items():
        t = legs[leg]
        assert t["skip_rate"] == float(skip)
        assert t["miss_rate"]["rate"] == float(miss) and t["defects_missed"]["rate"] == float(share)
        assert t["miss_rate"]["ci95"][1] == wilson_upper(t["missed"], t["settled"])
    assert legs == {leg: dict(legs[leg], **brute_force(range(2520))[leg]) for leg in PLANNED}
    assert {leg: legs[leg]["saved_s"] for leg in PLANNED} == {"ref.native": 0.0, "ref.edge": 0.0, "san.consan": 6600.0,
                                                             "san.waitcheck": 9900.0, "san.asan": 8389.5}
    b = report["options"]["B"]
    assert b["saved_s"] == 24889.5 and sum(b["run_s_by_resource"].values()) == 118062.0
    assert b["saved_s_by_resource"] == {"external_gpu": 8389.5, "sim_cpu": 16500.0}


def test_c3_t2_the_audited_rates_are_the_same_counts_over_the_audited_samples(fake):
    samples, report = fake
    audited = [i for i, s in enumerate(samples) if restated_coin(SEED, s.cand.candidate_id, 0.2)]
    assert [s.audit for s in samples] == [i in set(audited) for i in range(len(samples))]
    expected = brute_force(audited)
    for leg in PLANNED:
        assert {k: report["audited_legs"][leg][k] for k in expected[leg]} == expected[leg]
    unaudited_saved = sum(row["saved_s"] for row in brute_force(sorted(set(range(2520)) - set(audited))).values())
    assert report["options"]["B+C"]["saved_s"] == report["options"]["A+C"]["saved_s"] == unaudited_saved


def test_the_fake_logs_options_keep_their_rules(fake):
    samples, report = fake
    ev = Evaluator(load_profile("fake"), StandinV11())
    for s in samples:
        a = ev.option_reward(s, "A", s.audit)
        b = ev.option_reward(s, "B", s.audit)
        bc = ev.option_reward(s, "B+C", s.audit)
        assert UNDETERMINED not in (a, b, bc)  # the fake plan never skips a reference leg
        if s.skip:
            assert a is None or a < 0  # A never scores a sample with a skip as clean
            assert b is None or b <= 0.5  # B caps a clean result
        else:
            assert a == b == s.result.reward
        if s.audit:
            assert bc == s.result.reward
    full, opt_b = report["options"]["full"], report["options"]["B"]
    assert full["defects_in_skipped_legs"] == opt_b["defects_in_skipped_legs"] == 159 == opt_b["hidden_defects"]
    assert full["hidden_defects"] == 0 and report["options"]["A"]["hidden_defects_scored_as_clean"] == 0
    assert opt_b["hidden_defects_scored_as_clean"] > 0  # B alone is exploitable
    assert full["clean_minus_defect_gap"] > report["options"]["B+C"]["clean_minus_defect_gap"] > opt_b["clean_minus_defect_gap"]


def test_with_the_switch_only_the_legs_it_has_armed_are_skipped(fake):
    """B + C + S: walking the log in order, an unaudited sample skips the planned legs the switch had
    armed from the audited samples before it, and nothing else."""
    samples, report = fake
    coins = {s.cand.candidate_id: s.audit for s in samples}
    _, switched = replay_switch(samples, PLANNED, 0.02, coins)
    k = {leg: 0 for leg in PLANNED}
    n = {leg: 0 for leg in PLANNED}
    state = {leg: "warming" for leg in PLANNED}
    saved = 0.0
    for s in samples:
        want = {leg for leg in s.skip if not s.audit and state[leg] == "armed"}
        assert switched[s.cand.candidate_id] == want
        saved += sum(COST[leg] for leg in want if s.result.legs[leg].state != "not_run")
        if s.audit and s.plan_ok:
            for leg in PLANNED:
                r = s.result.legs[leg]
                if r.state in ("clean", "finding", "fail"):
                    n[leg] += 1
                    k[leg] += leg in s.skip and r.state in ("finding", "fail")
                    if state[leg] == "warming" and wilson_upper(k[leg] + 1, n[leg] + 1) <= 0.02:
                        state[leg] = "armed"
                    elif state[leg] == "armed" and wilson_upper(k[leg], n[leg]) > 0.02:
                        state[leg] = "tripped"
    assert {leg for v in switched.values() for leg in v} == {"san.waitcheck"}  # the one leg that never misses
    bcs = report["options"]["B+C+S"]
    assert bcs["saved_s"] == saved > 0 and bcs["hidden_defects"] == 0
    assert bcs["saved_s"] < report["options"]["B+C"]["saved_s"] / 5


def test_the_switch_replay_arms_a_leg_that_never_misses_after_279_audited_observations(fake):
    samples, report = fake
    observed = [s for s in samples if s.audit and s.plan_ok and s.result.legs["san.waitcheck"].state in
                ("clean", "finding", "fail")]
    wc = report["switch"]["legs"]["san.waitcheck"]
    assert wc["state"] == "armed" and wc["misses"] == 0 and wc["observations"] == len(observed)
    assert wc["armed_with"] == observed[278].cand.candidate_id
    for leg in ("san.consan", "san.asan"):  # missing at 6.7% and 2.5%: never trusted at a 2% limit
        assert report["switch"]["legs"][leg]["state"] == "warming"


# --- a log small enough to score by hand -------------------------------------------------------------

BR = {"before": "c1", "after": "c2", "clean": True}


def legs_of(**spec) -> dict[str, LegResult]:
    out = {g: LegResult(g, "clean", None, [], 1, 1, "g", None, False, 0.0, 0.0, 0.5) for g in GATES}
    for leg in PLANNED:
        state, reason = (spec.get(leg, "clean"), None) if isinstance(spec.get(leg, "clean"), str) else spec[leg]
        if state in ("not_run",):
            out[leg] = LegResult(leg, "not_run", reason, [], 0, 0, None, None, False, 0.0, None, None)
        elif state == "unsettled":
            out[leg] = LegResult(leg, "unsettled", reason, [], 0, 1, "w", None, False, 0.0, 1.0, 1.0 + COST[leg])
        else:
            out[leg] = LegResult(leg, state, reason, [{"kind": "f"}] if state == "finding" else [], 1, 1, "w", BR, False,
                                 0.0, 1.0, 1.0 + COST[leg])
    return out


def tiny_line(i: int, legs: dict, *, skip=None, u: float = 0.9, f: float = 0.2, prompt: str = "standin:t0",
              tokens: int = 0) -> dict:
    cid = candidate_id("tiny", 1, prompt, i)
    c = Candidate(cid, "tiny", 1, prompt, i, "tiny:v0", "", tokens, 4096,
                  TaskRef("c", None, None, "frozen", "0" * 64), "fake", None)
    rf = StandinV11()
    reward, raw = rf.compute(legs, c)
    cp = CandidatePlan(cid, "e" * 64, u, u < f, 0.0)
    cp.available_legs, cp.requested_at, cp.t_requested, cp.deadline = PLANNED, 1.0, 1.0, 6.0
    if skip is None:
        cp.status, cp.reason = "plan_unavailable", "timeout"
    else:
        cp.status, cp.answered_after_s = "ok", 0.1
        cp.plan = {"selector_version": "hand/1", "plan_id": f"hand:{i}",
                   "legs": {leg: {"decision": "skip" if leg in skip else "run", "p_finding": None, "score": None}
                            for leg in PLANNED},
                   "skip": sorted(skip), "context_sha256": None, "context_chars": None}
    result = RewardResult(cid, reward, "scored" if reward is not None else "unscored",
                          None if reward is not None else raw["reason"], raw if reward is not None else None,
                          {"name": rf.name, "version": rf.version, "code_sha256": rf.code_sha256}, legs, True,
                          plan_record(cp, "log_only"), cp.audit, "k", {"total_s": 0.0, "stages": {}})
    return plan_log_line(cp, c, result, f)


def tiny_group():
    """One group of four, with no length term (0 tokens), scored by hand from Appendix B:

    s0  clean; the plan skips ASan                full +1   A None (ASan unsettled)   B min(1, cap) = 0.5
    s1  only ASan finds; the plan skips ASan,     full -1/6 A None                    B 0.5: the hidden
        and s1 is audited                                                               defect scores as clean
    s2  ConSan and ASan find; the plan skips ASan full -1/3 A -1/6 (ASan in round)    B -1/6
    s3  waitcheck finds; no plan (timeout)        full -1/6 = A = B
    """
    return [tiny_line(0, legs_of(), skip={"san.asan"}),
            tiny_line(1, legs_of(**{"san.asan": "finding"}), skip={"san.asan"}, u=0.05),
            tiny_line(2, legs_of(**{"san.consan": "finding", "san.asan": "finding"}), skip={"san.asan"}),
            tiny_line(3, legs_of(**{"san.waitcheck": "finding"}))]


def test_the_option_rewards_and_advantages_are_the_hand_computed_ones():
    samples = parse_plan_log(tiny_group())
    ev = Evaluator(load_profile("fake"), StandinV11(), cap=0.5)
    sixth, third = -0.5 / 3, -0.5 * 2 / 3
    expected = {
        "full": [1.0, sixth, third, sixth],
        "A": [None, None, sixth, sixth],
        "A+C": [None, sixth, sixth, sixth],
        "B": [0.5, 0.5, sixth, sixth],
        "B+C": [0.5, sixth, sixth, sixth],
    }
    for option, rewards in expected.items():
        assert [ev.option_reward(s, option, s.audit) for s in samples] == rewards, option
    report = evaluate(samples, cap=0.5)
    opt = report["options"]

    def adv(rs):
        scored = [r for r in rs if r is not None]
        mean = sum(scored) / len(scored)
        return [None if r is None else r - mean for r in rs]

    full_adv = adv(expected["full"])
    for option, rewards in expected.items():
        a = adv(rewards)
        deltas = [abs(x - y) for x, y in zip(a, full_adv) if x is not None]
        assert opt[option]["mean_abs_delta_advantage"] == pytest.approx(sum(deltas) / len(deltas), abs=1e-12)
        assert opt[option]["scored"] == sum(r is not None for r in rewards)
        assert opt[option]["changed"] == sum(r != f for r, f in zip(rewards, expected["full"]))
    assert opt["B"]["hidden_defects"] == 2 and opt["B"]["hidden_defects_scored_as_clean"] == 1  # s1
    assert opt["B+C"]["hidden_defects"] == 1 and opt["B+C"]["hidden_defects_scored_as_clean"] == 0  # s1 is audited
    assert opt["B"]["clean_minus_defect_gap"] == pytest.approx(0.5 - (0.5 + sixth) / 2)
    assert opt["full"]["clean_minus_defect_gap"] == pytest.approx(1.0 - (sixth + third) / 2)
    assert opt["B"]["hidden_defects_advantage_up"] == 2 and opt["A"]["advantage_sign_flips"] == 0
    assert opt["A"]["saved_s"] == opt["B"]["saved_s"] == 3 * 8.5 and opt["B+C"]["saved_s"] == 2 * 8.5
    for option in ("A+C+S", "B+C+S"):  # four samples are far too few to trust any leg: nothing is skipped
        assert (opt[option]["changed"], opt[option]["saved_s"], opt[option]["hidden_defects"]) == (0, 0.0, 0)


def test_a_counterfactual_that_needs_a_leg_early_exit_kept_from_running_is_undetermined():
    """ref.native failed, so v1 never ran ASan. A plan that skips ref.native leaves the reward to
    ASan, which the log does not have: undetermined, even with a ConSan finding (ASan could add a
    second). A plan that skips only ASan changes nothing: ref.native still decides."""
    native_fail = {"ref.native": ("fail", "wrong_output"), "san.asan": ("not_run", "decided_before_run")}
    lines = [tiny_line(0, legs_of(**native_fail), skip={"ref.native"}),
             tiny_line(1, legs_of(**native_fail, **{"san.consan": "finding"}), skip={"ref.native"}),
             tiny_line(2, legs_of(**native_fail), skip={"san.asan"}),
             tiny_line(3, legs_of(), skip=set())]
    samples = parse_plan_log(lines)
    ev = Evaluator(load_profile("fake"), StandinV11())
    for option in ("A", "B"):
        assert [ev.option_reward(s, option, False) for s in samples] == [UNDETERMINED, UNDETERMINED, -0.7, 1.0]
        assert ev.decide(samples[0], samples[0].skip, "clean" if option == "B" else "skipped")[2] == ("san.asan",)
    report = evaluate(samples)
    assert report["options"]["A"]["undetermined"] == 2 and report["options"]["A"]["groups"] == {"undetermined": 1}
    assert report["options"]["A+C"]["undetermined"] == 2


def test_a_leg_that_requires_a_skipped_leg_does_not_run_in_the_counterfactual():
    profile = load_profile("fake")
    requires = dict(profile.requires, **{"san.asan": ("gate.compile", "san.consan")})
    dep = dataclasses.replace(profile, requires=requires)
    [s] = parse_plan_log([tiny_line(0, legs_of(), skip={"san.consan"})])
    assert Evaluator(profile, StandinV11()).option_reward(s, "B", False) == 0.5  # ASan ran clean: capped
    ev = Evaluator(dep, StandinV11())
    assert ev.option_reward(s, "B", False) is None  # ASan cannot run without ConSan: not settled
    assert ev._record(s, s.skip, "clean")["san.asan"].reason == "requirement_skipped"


# --- refusals ----------------------------------------------------------------------------------------


def test_the_evaluator_refuses_a_log_it_cannot_trust():
    good = tiny_group()
    assert evaluate(parse_plan_log(good))["log"]["reproduced"] == 4
    tampered = json.loads(json.dumps(good))
    tampered[0]["result"]["reward"] = 0.75
    tampered[0]["result"]["parts"]["base"] = 0.75
    with pytest.raises(PlanEvalError, match="is not what"):
        evaluate(parse_plan_log(tampered))
    other_fn = json.loads(json.dumps(good))
    for line in other_fn:
        line["result"]["reward_fn"]["code_sha256"] = "0" * 64
    with pytest.raises(PlanEvalError, match="scored by"):
        evaluate(parse_plan_log(other_fn))
    coin = json.loads(json.dumps(good))
    coin[1]["audit"]["u"] = 0.95
    with pytest.raises(PlanEvalError, match="coin"):
        parse_plan_log(coin)
    with pytest.raises(PlanEvalError, match="twice"):
        parse_plan_log(good + good[:1])
    with pytest.raises(PlanEvalError, match="empty"):
        parse_plan_log([])
    bad = json.loads(json.dumps(good))
    bad[2]["schema"] = "aorlta.selector.plan_log/0"
    with pytest.raises(PlanEvalError, match="line 3"):
        parse_plan_log(bad)
    with pytest.raises(PlanEvalError, match="cap"):
        Evaluator(load_profile("fake"), StandinV11(), cap=2.0)


# --- end to end against the service ------------------------------------------------------------------


def test_the_log_the_service_writes_gives_the_counts_an_independent_tally_gives(tmp_path):
    cands = [cand(i) for i in range(96)]
    planner = SkipAtRandom(seed=9, p_skip=0.4)
    h = hook(planner=planner)
    r = rig(tmp_path, selector=h, adapter=outcomes(cands, early_exit=True))
    try:
        results = r.run(cands)
    finally:
        r.close()
    samples = read_plan_log(h.write_plan_log(tmp_path / "plan-log.jsonl"))
    assert {s.cand.candidate_id for s in samples} == {c.candidate_id for c in cands}
    report = evaluate(samples)
    assert report["log"]["reproduced"] == len(cands)
    for leg in PLANNED:
        with_plan = [x for x in results if x.plan["status"] == "ok"]
        skips = [x for x in with_plan if planner.draw(x.candidate_id, leg) < 0.4]
        settled = [x for x in with_plan if x.legs[leg].state in ("clean", "finding", "fail")]
        missed = [x for x in skips if x.legs[leg].state in ("finding", "fail")]
        t = report["legs"][leg]
        assert (t["plans"], t["skipped"], t["settled"], t["missed"]) == (len(with_plan), len(skips), len(settled),
                                                                         len(missed)), leg
    assert report["legs"]["san.consan"]["missed"] > 0


def test_c3_t3_plan_mode_scores_what_the_evaluator_predicted_from_the_log_only_run(tmp_path):
    """The same candidates and planner, first in log_only (every leg runs, the plan is logged), then in
    plan mode with the same audit coins and a switch that always allows skipping. For every candidate
    the evaluator could determine from the first log, its option A + C reward is the reward plan mode
    really gives."""
    cands = [cand(i) for i in range(96)]
    planner = SkipAtRandom(seed=4, p_skip={"ref.native": 0.15, "san.consan": 0.5, "san.waitcheck": 0.5,
                                           "san.asan": 0.6})
    adapter = outcomes(cands, early_exit=True)
    h1 = hook(planner=planner, audit_fraction=0.3, plan_timeout_s=PLAN_TIMEOUT_S)
    r1 = rig(tmp_path / "log", selector=h1, adapter=adapter, scale=SCALE)
    try:
        r1.run(cands)
    finally:
        r1.close()
    samples = read_plan_log(h1.write_plan_log(tmp_path / "plan-log.jsonl"))
    h2 = hook("plan", planner, audit_fraction=0.3, max_miss_rate=1.0, plan_timeout_s=PLAN_TIMEOUT_S)
    r2 = rig(tmp_path / "plan", selector=h2, adapter=adapter, scale=SCALE)
    try:
        actual = {x.candidate_id: x for x in r2.run(cands)}
    finally:
        r2.close()
    ev = Evaluator(load_profile("fake"), StandinV11())
    determined = undetermined = changed = 0
    for s in samples:
        predicted = ev.option_reward(s, "A+C", s.audit)
        got = actual[s.cand.candidate_id]
        assert got.audit == s.audit
        if predicted == UNDETERMINED:
            undetermined += 1
            continue
        determined += 1
        assert got.reward == predicted, (s.cand.candidate_id, got.reward, predicted, got.reason)
        changed += predicted != s.result.reward
    assert determined >= 0.8 * len(cands) and changed > 0 and undetermined > 0


def test_the_cli_writes_the_report_and_refuses_a_bad_log(tmp_path, capsys):
    assert main(["--fake", str(tmp_path / "fake.jsonl"), "--out", str(tmp_path / "out")]) == 0
    md = (tmp_path / "out" / "plan-eval.md").read_text()
    assert "### Per leg" in md and "### Options (Section 6.3)" in md and "| B+C |" in md
    report = json.loads((tmp_path / "out" / "plan-eval.json").read_text())
    assert report["schema"] == "aorlta.selector.plan_eval/1" and report["legs"]["san.asan"]["missed"] == 42
    (tmp_path / "bad.jsonl").write_text('{"schema": "nope"}\n')
    assert main([str(tmp_path / "bad.jsonl")]) == 2
    assert "plan_eval:" in capsys.readouterr().err


TOY_V11_PLUGIN = '''
from pathlib import Path

from aorlta import plugins
from aorlta.reward.profiles import PROFILE_DIR
from aorlta.reward.rewardfn.standin_v11 import StandinV11


class ToyV11(StandinV11):
    name = "toy.v11"


here = Path(__file__).parent
(here / "toy-fake.yaml").write_text((PROFILE_DIR / "fake.yaml").read_text().replace("name: fake", "name: toy-fake")
                                    .replace("reward_fn: standin.v11", "reward_fn: toy.v11"))
plugins.register_profile("toy-fake", here / "toy-fake.yaml")
plugins.register_reward_fn("toy.v11", lambda profile: ToyV11(clean_repeats=profile.clean_repeats))
'''


def test_a_log_whose_profile_and_reward_a_plugin_registered_is_evaluated_once_the_plugin_is_loaded(tmp_path, capsys):
    """The evaluator refused any reward function but standin.v11 before asking the registry; now `--plugins` loads
    the plugin and the log is re-scored by the reward it registered, with the same counts as the built-in one."""
    from aorlta import plugins

    saved = {id(t): dict(t) for t in plugins._TABLES}
    loaded = dict(plugins.LOADED)
    try:
        (tmp_path / "toy_v11_plugin.py").write_text(TOY_V11_PLUGIN)
        lines = fake_log(iterations=1, prompts=4, K=4)
        ident = None
        for line in lines:
            line["candidate"]["check_profile"] = "toy-fake"
            if ident is None:
                from aorlta.reward.rewardfn.standin_v11 import StandinV11

                ident = dict(line["result"]["reward_fn"], name="toy.v11")
                assert line["result"]["reward_fn"]["name"] == StandinV11.name
            line["result"]["reward_fn"] = ident
        log = write_jsonl(tmp_path / "toy.jsonl", lines)
        assert main([str(log), "--audit-fractions", ""]) == 2
        err = capsys.readouterr().err
        assert "check profile 'toy-fake'" in err and "--plugins" in err
        out = tmp_path / "out"
        assert main([str(log), "--audit-fractions", "", "--plugins", str(tmp_path / "toy_v11_plugin.py"),
                     "--out", str(out)]) == 0
        report = json.loads((out / "plan-eval.json").read_text())
        assert report["log"]["reward_fn"]["name"] == "toy.v11" and report["log"]["profile"] == "toy-fake"
        assert report["log"]["reproduced"] == len(lines)
    finally:
        for t in plugins._TABLES:
            t.clear()
            t.update(saved[id(t)])
        plugins.LOADED.clear()
        plugins.LOADED.update(loaded)
