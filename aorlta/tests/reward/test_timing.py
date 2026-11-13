"""The timing measurement (reward/timing.py) and the reward that grades by it (reward/rewardfn/speedup.py), on CPU.

The statistics are checked on hand-made series and, with Hypothesis, against an independent restatement of the rule;
the reward is checked, as standin.v11 is in test_standin_v11.py, for the service's invariants and for `decide` never
giving an answer `compute` would contradict. Nothing a timing check could not decide may be clean.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aorlta.reward import timing
from aorlta.reward.adapters.base import checked_result, make_result
from aorlta.reward.rewardfn.base import Decided, NeedMore, Unscored
from aorlta.reward.rewardfn.speedup import WithSpeedup
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import REWARD_MAX, REWARD_MIN
from aorlta.schemas import LegJob
from reward_helpers import cand, leg

POLICY = timing.TimingPolicy(ratio=1.1, warmup=5, repeats=30, max_rel_mad=0.05, min_noise_floor=0.02)


def series(values, warmup=5, warm=1000.0):
    """`warmup` slow warm-up samples, then `values`."""
    return [warm] * warmup + list(values)


def flat(x, n=30):
    return [x] * n


# --- the statistics ------------------------------------------------------------------------------


def test_median_and_relative_mad():
    assert timing.median([3.0, 1.0, 2.0]) == 2.0 and timing.median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert timing.rel_mad([10.0, 10.0, 10.0]) == 0.0
    assert timing.rel_mad([9.0, 10.0, 11.0, 10.0, 10.0]) == pytest.approx(0.0)  # half the deviations are 0
    assert timing.rel_mad([8.0, 9.0, 10.0, 11.0, 12.0]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        timing.median([])


def test_the_warmups_are_dropped_and_only_they():
    # 5 warm-up samples a hundred times slower: dropped, they move nothing
    sp = timing.judge(series(flat(8.0)), series(flat(10.0)), POLICY)
    assert sp.verdict == "faster" and sp.speedup == 1.25 and sp.samples == 30
    assert (sp.candidate_median, sp.reference_median, sp.candidate_rel_mad) == (8.0, 10.0, 0.0)
    # kept, the same warm-up samples are 5 of 35: still the median's minority, but now the candidate's series spreads
    kept = timing.judge(series(flat(8.0)), series(flat(10.0)), timing.TimingPolicy(warmup=0, repeats=30))
    assert kept.samples == 35 and kept.candidate_rel_mad == 0.0
    # a sixth slow sample is past the warm-up and counts
    sp = timing.judge([1000.0] * 6 + flat(8.0, 29), series(flat(10.0)), POLICY)
    assert sp.samples == 30 and sp.candidate_median == 8.0
    sp = timing.judge([1000.0] * 21 + flat(8.0, 14), series(flat(10.0)), POLICY)
    assert sp.candidate_median == 1000.0 and sp.verdict == "not_faster"


def test_the_ratio_and_the_noise_floor_decide():
    ref = series(flat(10.0))
    floor = POLICY.min_noise_floor  # both series are flat: the floor is its minimum
    at = 10.0 / (1.1 * (1 + floor))  # the candidate time whose speedup just clears the ratio past the floor
    assert timing.judge(series(flat(at * 0.9999)), ref, POLICY).verdict == "faster"
    assert timing.judge(series(flat(at * 1.0001)), ref, POLICY).verdict == "within_noise"
    assert timing.judge(series(flat(10.0 / 1.1)), ref, POLICY).verdict == "within_noise"  # exactly the ratio
    below = 10.0 / (1.1 * (1 - floor))
    assert timing.judge(series(flat(below * 1.0001)), ref, POLICY).verdict == "not_faster"
    assert timing.judge(series(flat(below * 0.9999)), ref, POLICY).verdict == "within_noise"
    assert timing.judge(series(flat(10.0)), ref, POLICY).verdict == "not_faster"  # itself: speedup 1, ratio 1.1
    assert timing.judge(series(flat(10.0)), ref, timing.TimingPolicy(ratio=1.0)).verdict == "within_noise"


def test_the_spread_widens_the_noise_floor():
    jitter = [10.0 * (1 + d) for d in (-0.03, 0.03) * 15]  # relative MAD 0.03 on each side
    ref = series(jitter)
    sp = timing.judge(series([x / 1.16 for x in jitter]), ref, POLICY)
    assert sp.noise_floor == pytest.approx(0.06) and sp.speedup == pytest.approx(1.16)
    assert sp.verdict == "within_noise"  # 1.16 < 1.1 * 1.06: it would clear the bare 2% floor
    assert timing.judge(series([x / 1.17 for x in jitter]), ref, POLICY).verdict == "faster"


def test_a_noisy_series_is_not_decided():
    noisy = [10.0 * (1 + d) for d in (-0.1, 0.1) * 15]
    assert timing.judge(series(flat(5.0)), series(noisy), POLICY).verdict == "noisy"  # the reference
    assert timing.judge(series(noisy), series(flat(50.0)), POLICY).verdict == "noisy"  # the candidate
    edge = [10.0 * (1 + d) for d in (-0.05, 0.05) * 15]
    assert timing.judge(series(flat(5.0)), series(edge), POLICY).verdict == "faster"  # 0.05 is allowed


@pytest.mark.parametrize("cand_, ref_, verdict", [
    (series(flat(8.0, 29)), series(flat(10.0)), "too_few"),
    (series(flat(8.0)), series(flat(10.0, 29)), "too_few"),
    ([], [], "too_few"),
    (series(flat(8.0)[:-1] + [float("nan")]), series(flat(10.0)), "invalid"),
    (series(flat(8.0)), series(flat(10.0)[:-1] + [float("inf")]), "invalid"),
    (series(flat(8.0)[:-1] + [0.0]), series(flat(10.0)), "invalid"),
    (series(flat(8.0)[:-1] + [-1.0]), series(flat(10.0)), "invalid"),
    (series(flat(8.0)[:-1] + [True]), series(flat(10.0)), "invalid"),
    ([1000.0, float("nan")] + flat(8.0, 33), series(flat(10.0)), "invalid"),  # a warm-up sample is checked too
])
def test_too_few_or_impossible_samples_are_not_decided(cand_, ref_, verdict):
    sp = timing.judge(cand_, ref_, POLICY)
    assert sp.verdict == verdict and sp.speedup is None
    assert timing.leg_outcome(sp) == ("unsettled", f"timing_{verdict}", [])


@pytest.mark.parametrize("kw", [dict(ratio=0), dict(ratio=-1.0), dict(ratio=float("nan")), dict(ratio=True),
                                dict(warmup=-1), dict(warmup=1.0), dict(repeats=4), dict(max_rel_mad=0),
                                dict(max_rel_mad=1.0), dict(min_noise_floor=-0.1), dict(min_noise_floor=1.0)])
def test_a_policy_refuses_settings_that_cannot_decide(kw):
    with pytest.raises(ValueError):
        timing.TimingPolicy(**kw)


def test_a_policy_reads_its_own_leg_settings_and_ignores_the_checks():
    p = timing.TimingPolicy.from_settings({"ratio": 1.5, "warmup": 2, "repeats": 9, "target": "gfx950",
                                           "launches": 20, "max_rel_mad": 0.1, "min_noise_floor": 0.0})
    assert p == timing.TimingPolicy(ratio=1.5, warmup=2, repeats=9, max_rel_mad=0.1, min_noise_floor=0.0)
    assert timing.TimingPolicy.from_settings({}) == timing.TimingPolicy()


# --- the leg and its record ----------------------------------------------------------------------


def test_only_faster_is_clean_and_only_not_faster_is_settled():
    for verdict in timing.VERDICTS:
        sp = timing.Speedup(verdict, 1.1, speedup=1.0, noise_floor=0.02, samples=30)
        state, reason, findings = timing.leg_outcome(sp)
        assert (state == "clean") == (verdict == "faster")
        assert (state == "fail") == (verdict == "not_faster") and (reason == "not_faster") == (verdict == "not_faster")
        assert state in ("clean", "fail", "unsettled") and bool(findings) == (state != "unsettled")


def job(name="ref.speedup"):
    return LegJob(job_id="j", leg=name, candidate_id="c", control_id=None, source_sha256="s" * 64,
                  task_sha256="t" * 64)


def test_a_clean_leg_may_carry_its_measurement_and_nothing_else():
    sp = timing.judge(series(flat(8.0)), series(flat(10.0)), POLICY)
    state, reason, findings = timing.leg_outcome(sp)
    ok = checked_result(job(), make_result(job(), state, reason, findings=findings, t_start=0, t_end=1), 0, 1)
    assert ok.state == "clean" and timing.measurement(ok)["speedup"] == 1.25
    other = make_result(job(), "clean", findings=[*findings, {"kind": "other"}], t_start=0, t_end=1)
    assert checked_result(job(), other, 0, 1).reason == "clean_with_findings"


@pytest.mark.parametrize("change", [
    {"kind": "other"}, {"metric": "throughput"}, {"speedup": 0.0}, {"speedup": float("nan")}, {"speedup": None},
    {"speedup": True}, {"ratio": -1.0}, {"noise_floor": 1.0}, {"noise_floor": -0.1}, {"noise_floor": "0.02"},
])
def test_a_record_that_does_not_read_back_is_none(change):
    rec = {**timing.record(timing.judge(series(flat(8.0)), series(flat(10.0)), POLICY)), **change}
    assert timing.measurement(leg("ref.speedup", findings=[rec])) is None


def test_two_records_or_none_read_back_as_none():
    rec = timing.record(timing.judge(series(flat(8.0)), series(flat(10.0)), POLICY))
    assert timing.measurement(leg("ref.speedup", findings=[rec, rec])) is None
    assert timing.measurement(leg("ref.speedup")) is None


def test_the_grades():
    def rec(s, ratio=1.1, floor=0.02):
        return {"speedup": s, "ratio": ratio, "noise_floor": floor}

    assert timing.speedup_base("clean", rec(1.1 * 1.02)) == pytest.approx(0.75 + 0.25 * math.log2(1.02))
    assert timing.speedup_base("clean", rec(2.2)) == 1.0 and timing.speedup_base("clean", rec(9.0)) == 1.0
    assert timing.speedup_base("fail", rec(1.0)) == pytest.approx(0.5 / 1.1)
    assert timing.speedup_base("fail", rec(0.1)) == pytest.approx(0.05 / 1.1)
    # a state its record contradicts grades to nothing
    assert timing.speedup_base("clean", rec(1.1)) is None and timing.speedup_base("fail", rec(1.1)) is None
    assert timing.speedup_base("unsettled", rec(2.0)) is None


SAMPLES = st.lists(st.floats(0.5, 2.0, allow_nan=False), min_size=35, max_size=40)
PROPERTY = settings(max_examples=1500, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def restated(cand_, ref_, p):
    """The rule, restated from the module docstring without its helpers."""
    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2

    c, r = cand_[p.warmup:], ref_[p.warmup:]
    mc, mr = med(c), med(r)
    dc, dr = med([abs(x - mc) for x in c]) / mc, med([abs(x - mr) for x in r]) / mr
    if max(dc, dr) > p.max_rel_mad:
        return "noisy"
    floor, s = max(p.min_noise_floor, dc + dr), mr / mc
    return "faster" if s >= p.ratio * (1 + floor) else "not_faster" if s <= p.ratio * (1 - floor) else "within_noise"


@PROPERTY
@given(SAMPLES, SAMPLES, st.floats(0.5, 2.0), st.floats(0.01, 0.3), st.floats(0.0, 0.1))
def test_the_verdict_matches_the_rule_and_only_a_cleared_ratio_is_clean(c, r, ratio, max_mad, min_floor):
    p = timing.TimingPolicy(ratio=ratio, warmup=5, repeats=30, max_rel_mad=max_mad, min_noise_floor=min_floor)
    sp = timing.judge(c, r, p)
    assert sp.verdict == restated(c, r, p)
    state, _, findings = timing.leg_outcome(sp)
    if state == "clean":
        assert sp.speedup >= ratio * (1 + sp.noise_floor) and sp.noise_floor >= min_floor
        assert max(sp.candidate_rel_mad, sp.reference_rel_mad) <= max_mad
    if state != "unsettled":
        base = timing.speedup_base(state, timing.measurement(leg("ref.speedup", state, findings=findings)))
        assert base is not None and ((base >= timing.CLEAN_BASE) == (state == "clean")) and 0 < base <= 1.0


# --- the reward ----------------------------------------------------------------------------------

INNER = StandinV11()
FN = WithSpeedup(INNER, name="test.speed")
GOOD = timing.record(timing.judge(series(flat(8.0)), series(flat(10.0)), POLICY))  # 1.25: clean
SLOW = timing.record(timing.judge(series(flat(20.0)), series(flat(10.0)), POLICY))  # 0.5: not faster


@st.composite
def speedup_leg(draw):
    state = draw(st.sampled_from(["clean", "clean", "fail", "fail", "unsettled", "not_run", "skipped_by_plan",
                                  "finding"]))
    reason = {"fail": draw(st.sampled_from(["not_faster", "not_faster", "timeout"])),
              "unsettled": draw(st.sampled_from(["timing_noisy", "timing_within_noise", "timing_timeout"]))
              }.get(state)
    findings = []
    if state in ("clean", "fail", "finding"):
        findings = draw(st.sampled_from([[GOOD], [SLOW], [], [GOOD, SLOW], [{"kind": "fake_finding"}]]))
    return leg("ref.speedup", state, reason, findings=findings or None if state == "finding" else findings)


@st.composite
def record(draw):
    """A standin.v11 record clean but for up to two legs, and a timing leg."""
    rec = {name: leg(name) for name in INNER.legs}
    for name in draw(st.sets(st.sampled_from(INNER.legs), max_size=2)):
        rec[name] = leg(name, *draw(st.sampled_from([("fail", "wrong_output"), ("fail", "compile_error"),
                                                     ("unsettled", "timeout"), ("finding", None), ("not_run", None)])))
    rec["ref.speedup"] = draw(speedup_leg())
    return rec


@st.composite
def candidate(draw):
    max_new = draw(st.integers(1, 8192))
    return cand(0, tokens=draw(st.integers(0, max_new)), max_new=max_new,
                parent_healthy=draw(st.sampled_from([None, True, False])),
                reference_kind=draw(st.sampled_from(["frozen", "self_written"])))


@PROPERTY
@given(record(), candidate())
def test_the_reward_keeps_the_services_invariants(rec, c):
    reward, parts = FN.compute(rec, c)
    if reward is None:
        assert isinstance(parts.get("reason"), str) and parts["reason"] and not FN.claims_clean(reward, parts)
        return
    assert type(reward) is float and REWARD_MIN <= reward <= REWARD_MAX
    assert set(parts) == {"base", "broke_healthy", "length"} and all(type(v) is float for v in parts.values())
    assert parts["base"] + parts["broke_healthy"] + parts["length"] == reward
    inner_reward, inner_parts = INNER.compute(rec, c)
    if FN.claims_clean(reward, parts):  # clean: the inner reward was clean and the timing leg cleared the ratio
        assert INNER.claims_clean(inner_reward, inner_parts) and rec["ref.speedup"].state == "clean"
    elif INNER.claims_clean(inner_reward, inner_parts):  # measured, correct and not faster
        assert rec["ref.speedup"].state == "fail" and 0 < parts["base"] < 0.5
    else:  # whatever the inner function scored stands
        assert (reward, parts) == (inner_reward, inner_parts)


@PROPERTY
@given(record(), candidate())
def test_nothing_the_timing_did_not_settle_is_scored_on_a_clean_inner_reward(rec, c):
    inner_reward, inner_parts = INNER.compute(rec, c)
    t = rec["ref.speedup"]
    settled = (t.state == "clean" and t.findings == [GOOD]) or (t.state == "fail" and t.reason == "not_faster"
                                                                and t.findings == [SLOW])
    if INNER.claims_clean(inner_reward, inner_parts):
        assert (FN.compute(rec, c)[0] is not None) == settled


@PROPERTY
@given(record(), candidate(), st.permutations(list(INNER.legs) + ["ref.speedup"]))
def test_decide_never_contradicts_compute(rec, c, order):
    reward, parts = FN.compute(rec, c)
    for k in range(len(order) + 1):
        partial = {n: rec[n] for n in order[:k]}
        d = FN.decide(partial, c)
        if isinstance(d, Decided):
            assert (d.reward, d.parts) == (reward, parts)
        elif isinstance(d, Unscored):
            assert reward is None and d.reason == parts["reason"]
        else:
            assert isinstance(d, NeedMore) and d.legs and not set(d.legs) & set(partial)


def test_the_value_table():
    c = cand(0, tokens=0, reference_kind="frozen")
    clean = {name: leg(name) for name in INNER.legs}

    def score(t, **over):
        return FN.compute({**clean, **over, "ref.speedup": t}, c)

    assert score(leg("ref.speedup", findings=[GOOD]))[1]["base"] == pytest.approx(0.75 + 0.25 * math.log2(1.25 / 1.1))
    assert score(leg("ref.speedup", "fail", "not_faster", findings=[SLOW]))[1]["base"] == pytest.approx(0.5 * 0.5 / 1.1)
    assert score(leg("ref.speedup", "unsettled", "timing_noisy")) == (None, {"reason": "ref.speedup:timing_noisy"})
    assert score(leg("ref.speedup", "not_run")) == (None, {"reason": "ref.speedup:not_run"})
    assert score(leg("ref.speedup", "fail", "timeout", findings=[SLOW])) == (None, {"reason": "ref.speedup:timeout"})
    assert score(leg("ref.speedup", findings=[SLOW])) == (None, {"reason": "ref.speedup:no_measurement"})
    assert score(leg("ref.speedup", findings=[])) == (None, {"reason": "ref.speedup:no_measurement"})
    # a candidate that is not correct is never timed: its reward is the inner function's, timing leg or none
    wrong = score(leg("ref.speedup", findings=[GOOD]), **{"ref.native": leg("ref.native", "fail", "wrong_output")})
    assert wrong == INNER.compute({**clean, "ref.native": leg("ref.native", "fail", "wrong_output"),
                                   "ref.speedup": leg("ref.speedup")}, c)
    assert wrong[1]["base"] == -0.7


class _Overlapping:
    """An inner function whose not-clean reward has a base in the clean range of the wrapper."""

    name, version, code_sha256, legs = "overlap", "1", "0" * 64, ("ref.native",)

    def compute(self, legs, cand_):
        return 0.8, {"base": 0.8, "broke_healthy": 0.0, "length": 0.0}

    def decide(self, partial, cand_):
        return Decided(*self.compute(partial, cand_))

    @staticmethod
    def claims_clean(reward, parts):
        return False


def test_an_inner_reward_that_would_read_as_clean_is_unscored():
    fn = WithSpeedup(_Overlapping(), name="x")
    legs = {"ref.native": leg("ref.native"), "ref.speedup": leg("ref.speedup", findings=[GOOD])}
    assert fn.compute(legs, cand(0)) == (None, {"reason": "inner_base_overlaps_clean"})
    assert fn.decide(legs, cand(0)) == Unscored("inner_base_overlaps_clean")


def test_the_wrapper_names_its_parts():
    assert FN.legs == INNER.legs + ("ref.speedup",) and FN.version == "11+speedup-1"
    assert FN.settings == {"inner": "standin.v11", "inner_settings": {"clean_repeats": 1}, "speedup_leg": "ref.speedup"}
    assert FN.code_sha256 != INNER.code_sha256 and len(FN.code_sha256) == 64
    with pytest.raises(ValueError):
        WithSpeedup(INNER, name="x", leg="ref.native")
