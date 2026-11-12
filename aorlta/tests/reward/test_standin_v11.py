"""C2.T10 and C2.T3 on `standin.v11` (design Appendix B), with Hypothesis.

The properties are checked against independent restatements of the rules, not against the
implementation's own helpers, so a mutant in the ladder cannot also bend the test.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aorlta.reward.rewardfn.base import Decided, NeedMore, Unscored
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from reward_helpers import cand, leg

GATES = ["parse", "contract", "static", "reference", "compile", "no_artifact", "spec_invalid", "static_reject",
         "no_change_on_failing_parent"]
GATE_LEGS = [f"gate.{g}" for g in GATES]
LADDER = ["ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan"]
ORDER = GATE_LEGS + LADDER
SAN = ["san.consan", "san.waitcheck", "san.asan"]
KNOWN_GATE = {g for g in GATES if g != "compile"} | {"compile_error"}
# Appendix B's own list of why a sample can be unscored.
APPENDIX_B_REASONS = ({f"gate.{g}" for g in GATES} | {"ref.native", "ref.edge"} | {f"{n}.control" for n in LADDER}
                      | {"unknown_gate_reason", "sanitizer_not_settled", "clean_not_proven"})
PENALTY_BASES = {-1.0, -0.9, -0.8, -0.7}
FN = StandinV11()

BRACKETS = st.sampled_from([None, {"before": "c1", "after": "c2", "clean": True},
                            {"before": "c1", "after": "c2", "clean": False}])


@st.composite
def leg_result(draw, name: str):
    if name.startswith("gate."):
        state = draw(st.sampled_from(["clean", "clean", "clean", "fail", "unsettled", "not_run", "skipped_by_plan"]))
        reason = draw(st.sampled_from(sorted(KNOWN_GATE) + ["no_such_reason"])) if state == "fail" else None
        return leg(name, state, reason, repeats=draw(st.integers(0, 2)))
    states = ["clean", "clean", "clean", "fail", "unsettled", "not_run", "skipped_by_plan"]
    if name.startswith("san."):
        states += ["finding", "finding"]
    state = draw(st.sampled_from(states))
    reason = None
    if state == "fail":
        reason = draw(st.sampled_from(["wrong_output", "edge_mismatch", "crash", "fault", "timeout", "segv_unknown"]))
    if state == "unsettled":
        reason = draw(st.sampled_from(["timeout", "worker_stalled", "control_dirty", "adapter_error"]))
    return leg(name, state, reason, repeats=draw(st.integers(0, 2)), bracket=draw(BRACKETS))


@st.composite
def random_record(draw):
    return {name: draw(leg_result(name)) for name in ORDER}


@st.composite
def near_clean_record(draw):
    """Clean except for up to three legs, so the clean and late-ladder paths are reached often."""
    rec = {name: leg(name, repeats=draw(st.integers(1, 2))) for name in ORDER}
    for name in draw(st.sets(st.sampled_from(ORDER), max_size=3)):
        rec[name] = draw(leg_result(name))
    return rec


def record():
    return st.one_of(random_record(), near_clean_record())


@st.composite
def record_with_finding(draw):
    """Any record, with at least one sanitizer finding."""
    rec = draw(record())
    for name in draw(st.sets(st.sampled_from(SAN), min_size=1)):
        rec[name] = leg(name, "finding")
    return rec


@st.composite
def record_with_unsettled(draw):
    """A mostly clean record with one leg unsettled (or skipped) and no sanitizer finding."""
    rec = draw(near_clean_record())
    for name in SAN:
        if rec[name].state == "finding":
            rec[name] = leg(name)
    name = draw(st.sampled_from(ORDER))
    rec[name] = leg(name, draw(st.sampled_from(["unsettled", "skipped_by_plan"])), "timeout")
    return rec


@st.composite
def penalty_record(draw):
    """A record whose ladder stops at a penalty rung: every leg before it clean, later legs anything."""
    rec = {name: leg(name) for name in ORDER}
    rung = draw(st.sampled_from(ORDER))
    if rung.startswith("gate."):
        reason = "compile_error" if rung == "gate.compile" else draw(st.sampled_from(sorted(KNOWN_GATE - {"compile_error"})))
        rec[rung] = leg(rung, "fail", reason)
    elif draw(st.booleans()):
        rec[rung] = leg(rung, "fail", draw(st.sampled_from(["wrong_output", "edge_mismatch"])))
    else:
        reasons = ["crash", "fault"] if rung == "san.consan" else ["crash", "fault", "timeout"]
        rec[rung] = leg(rung, "fail", draw(st.sampled_from(reasons)), bracket={"before": "a", "after": "b", "clean": True})
    for name in ORDER[ORDER.index(rung) + 1:]:
        rec[name] = draw(leg_result(name))
    return rec


@st.composite
def defect_record(draw):
    """Every gate and reference clean; at least one sanitizer finding; the others anything that is not a
    decisive failure."""
    rec = {name: leg(name) for name in ORDER}
    found = draw(st.sets(st.sampled_from(SAN), min_size=1))
    for name in SAN:
        rec[name] = leg(name, "finding") if name in found else leg(
            name, draw(st.sampled_from(["clean", "unsettled", "not_run", "skipped_by_plan"])))
    return rec


ANY = object()


@st.composite
def candidate(draw, parent_healthy=ANY):
    max_new = draw(st.integers(1, 8192))
    tokens = draw(st.integers(0, max_new))
    ph = draw(st.sampled_from([None, True, False])) if parent_healthy is ANY else parent_healthy
    return cand(0, tokens=tokens, max_new=max_new, parent_healthy=ph,
                reference_kind=draw(st.sampled_from(["frozen", "self_written", None])))


def clean_record(**over):
    rec = {name: leg(name) for name in ORDER}
    rec.update(over)
    return rec


# --- an independent restatement of when the ladder decides ---------------------------------------


def norm_state(name: str, r) -> str:
    """What Appendix B can make of a leg: skipped, a finding off a sanitizer, or a fail with a reason the
    ladder does not know is unsettled (the stand-in's three fail-closed normalisations)."""
    if r.state == "skipped_by_plan":
        return "unsettled"
    if r.state == "finding" and not name.startswith("san."):
        return "unsettled"
    if r.state == "fail" and not name.startswith("gate.") and r.reason not in (
            "wrong_output", "edge_mismatch", "crash", "fault", "timeout"):
        return "unsettled"
    return r.state


def decisive(name: str, r) -> bool:
    """A rung that ends the ladder on its own: a gate failure, a mismatch, or an attributed fault (a
    ConSan-only timeout does not end it)."""
    s = norm_state(name, r)
    if s != "fail":
        return False
    if name.startswith("gate."):
        return True
    return not (name == "san.consan" and r.reason == "timeout")


def confirmed(rec) -> bool:
    return any(norm_state(n, rec[n]) == "finding" for n in SAN)


def reached_unsettled(rec) -> bool:
    """Some unsettled leg the ladder reaches: a gate or reference leg with no decisive rung before it,
    or a sanitizer leg with no decisive rung anywhere."""
    for i, name in enumerate(ORDER):
        s = norm_state(name, rec[name])
        if s == "unsettled":
            before = ORDER[:i] if not name.startswith("san.") else ORDER
            if not any(decisive(n, rec[n]) for n in before if n != name):
                return True
        if decisive(name, rec[name]) and not name.startswith("san."):
            return False
    return False


# --- C2.T10 -----------------------------------------------------------------------------------------

PROPERTY = settings(max_examples=3000, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])


@PROPERTY
@given(record(), candidate())
def test_reward_is_none_or_in_range(rec, c):
    reward, parts = FN.compute(rec, c)
    assert reward is None or (-1.55 <= reward <= 1.0)


@PROPERTY
@given(record(), candidate())
def test_length_term_is_in_range_and_is_the_token_share(rec, c):
    reward, parts = FN.compute(rec, c)
    assume(reward is not None)
    assert -0.05 <= parts["length"] <= 0.0
    assert parts["length"] == -0.05 * (c.response_tokens / c.max_new_tokens)
    assert parts["base"] + parts["broke_healthy"] + parts["length"] == reward


@pytest.mark.parametrize("tokens", [3, 4096, 6007])
def test_a_full_length_answer_costs_exactly_the_cap(tokens):
    """At tokens == max_new_tokens the term is -0.05, not -0.05000000000000001, so the worst reward is
    -1.55 and not one ulp below the range (found by the property above)."""
    rec = clean_record(**{"gate.static": leg("gate.static", "fail", "static")})
    reward, parts = FN.compute(rec, cand(0, tokens=tokens, max_new=tokens, parent_healthy=True))
    assert parts["length"] == -0.05
    assert reward == -1.55


@PROPERTY
@given(penalty_record(), defect_record(), st.sampled_from([None, True, False]), st.data())
def test_every_penalty_is_below_the_lowest_defect_reward_for_the_same_parent_health(ra, rb, ph, data):
    ca, cb = data.draw(candidate(parent_healthy=ph)), data.draw(candidate(parent_healthy=ph))
    a, pa = FN.compute(ra, ca)
    b, pb = FN.compute(rb, cb)
    assert a is not None and pa["base"] in PENALTY_BASES
    assert b is not None and -0.5 <= pb["base"] <= -0.5 / 3
    assert a < b


def test_the_penalty_defect_order_holds_at_its_extremes():
    """The closest pair: the mildest penalty with no length against the worst defect at full length."""
    worst_defect = clean_record(**{n: leg(n, "finding") for n in SAN})
    mild_penalty = clean_record(**{"ref.native": leg("ref.native", "fail", "wrong_output")})
    for ph in (None, True, False):
        a, _ = FN.compute(mild_penalty, cand(0, tokens=0, parent_healthy=ph))
        b, _ = FN.compute(worst_defect, cand(0, tokens=4096, max_new=4096, parent_healthy=ph))
        assert a < b


@PROPERTY
@given(st.one_of(record(), record_with_unsettled()), candidate())
def test_an_unsettled_leg_the_ladder_reaches_without_a_confirmed_finding_gives_none(rec, c):
    assume(not confirmed(rec) and reached_unsettled(rec))
    reward, parts = FN.compute(rec, c)
    assert reward is None


@PROPERTY
@given(record(), candidate())
def test_none_arises_only_for_the_reasons_appendix_b_lists(rec, c):
    reward, parts = FN.compute(rec, c)
    if reward is None:
        assert parts["reason"] in APPENDIX_B_REASONS


@PROPERTY
@given(record(), candidate())
def test_a_clean_reward_needs_every_sanitizer_clean_and_nothing_unsettled(rec, c):
    reward, parts = FN.compute(rec, c)
    assume(FN.claims_clean(reward, parts))
    assert all(rec[n].state == "clean" and rec[n].repeats >= 1 for n in SAN)
    assert rec["gate.compile"].state == "clean"
    assert not any(norm_state(n, rec[n]) in ("unsettled", "fail", "finding") for n in ORDER if not n.startswith("san."))
    assert parts["base"] == (1.0 if c.task.reference_kind == "frozen" else 0.5)


@PROPERTY
@given(record_with_finding(), candidate())
def test_a_confirmed_finding_is_always_penalised(rec, c):
    """Rule 5, except where Appendix B's ladder decides first: a gate failing for an unknown reason."""
    assume(confirmed(rec))
    unknown_gate_first = False
    for name in GATE_LEGS:
        s = norm_state(name, rec[name])
        if s == "fail":
            unknown_gate_first = rec[name].reason not in KNOWN_GATE
            break
    assume(not unknown_gate_first)
    reward, parts = FN.compute(rec, c)
    assert reward is not None and parts["base"] < 0


@PROPERTY
@given(record(), candidate())
def test_a_skipped_leg_never_counts_as_clean(rec, c):
    """Plan mode's skipped legs reach the reward as unsettled (Section 6.3 option A), never as clean."""
    names = [n for n in ORDER if rec[n].state == "skipped_by_plan"]
    assume(names)
    reward, parts = FN.compute(rec, c)
    assert not FN.claims_clean(reward, parts)


@pytest.mark.parametrize("name", ORDER)
def test_one_skipped_leg_in_an_otherwise_clean_record_is_unscored(name):
    rec = clean_record(**{name: leg(name, "skipped_by_plan")})
    reward, parts = FN.compute(rec, cand(0))
    assert reward is None


# --- early exit: decide never disagrees with compute ------------------------------------------------


@settings(max_examples=4000, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(record(), record(), st.lists(st.booleans(), min_size=len(ORDER), max_size=len(ORDER)), candidate())
def test_decide_answers_only_what_every_completion_agrees_on(known, completion, mask, c):
    partial = {n: known[n] for n, keep in zip(ORDER, mask) if keep}
    d = FN.decide(partial, c)
    full = {n: partial.get(n, completion[n]) for n in ORDER}
    reward, parts = FN.compute(full, c)
    if isinstance(d, Decided):
        assert (d.reward, d.parts) == (reward, parts)
    elif isinstance(d, Unscored):
        assert reward is None and parts["reason"] == d.reason
    else:
        assert isinstance(d, NeedMore) and d.legs and all(n not in partial for n in d.legs)


@PROPERTY
@given(record(), candidate())
def test_decide_on_a_complete_record_is_compute(rec, c):
    d = FN.decide(rec, c)
    reward, parts = FN.compute(rec, c)
    assert not isinstance(d, NeedMore)
    assert (d.reward, d.parts) == (reward, parts) if isinstance(d, Decided) else (reward is None and parts["reason"] == d.reason)


def test_early_exit_cases():
    c = cand(0)
    gates_clean = {n: leg(n) for n in GATE_LEGS}
    # a compile error decides at S1, whatever S2/S3 would say
    p = dict(gates_clean, **{"gate.compile": leg("gate.compile", "fail", "compile_error")})
    assert isinstance(FN.decide(p, c), Decided) and FN.decide(p, c).reward == -0.9 + -0.05 * (1024 / 4096)
    # a native mismatch decides at S2 without the sanitizers
    p = dict(gates_clean, **{"ref.native": leg("ref.native", "fail", "wrong_output")})
    assert isinstance(FN.decide(p, c), Decided)
    # a ConSan finding does not: ASan could add a second one
    p = dict(gates_clean, **{n: leg(n) for n in ("ref.native", "ref.edge", "san.waitcheck")},
             **{"san.consan": leg("san.consan", "finding")})
    assert isinstance(FN.decide(p, c), NeedMore) and FN.decide(p, c).legs == ("san.asan",)
    # an unsettled gate waits for the sanitizers (a confirmed finding is always penalised) ...
    p = dict(gates_clean, **{"gate.static": leg("gate.static", "unsettled", "timeout")})
    assert isinstance(FN.decide(p, c), NeedMore)
    # ... unless none can run any more
    p.update({n: leg(n, "not_run") for n in SAN})
    assert FN.decide(p, c) == Unscored("gate.static")


# --- C2.T3 on the reward function -------------------------------------------------------------------


@pytest.mark.parametrize("ph", [None, True, False])
@pytest.mark.parametrize("asan_state", ["unsettled", "not_run"])
def test_a_confirmed_finding_with_an_unsettled_asan_leg_is_penalised(ph, asan_state):
    """Appendix B rule 5: in-round counts every sanitizer that ran, unsettled or not."""
    rec = clean_record(**{"san.consan": leg("san.consan", "finding"),
                          "san.asan": leg("san.asan", asan_state, "timeout" if asan_state == "unsettled" else None)})
    c = cand(0, tokens=1000, max_new=4000, parent_healthy=ph)
    reward, parts = FN.compute(rec, c)
    in_round = 3 if asan_state == "unsettled" else 2
    base = -0.5 * 1 / in_round
    broke = -0.5 if ph else 0.0
    length = -0.05 * (1000 / 4000)
    assert parts == {"base": base, "broke_healthy": broke, "length": length}
    assert reward == base + broke + length


def test_appendix_b_values():
    c = cand(0, tokens=0)
    assert FN.compute(clean_record(), c)[0] == 1.0
    assert FN.compute(clean_record(), cand(0, tokens=0, reference_kind="self_written"))[0] == 0.5
    assert FN.compute(clean_record(**{"gate.static": leg("gate.static", "fail", "static")}), c)[0] == -1.0
    assert FN.compute(clean_record(**{"gate.compile": leg("gate.compile", "fail", "compile_error")}), c)[0] == -0.9
    assert FN.compute(clean_record(**{"ref.edge": leg("ref.edge", "fail", "edge_mismatch")}), c)[0] == -0.7
    br = {"before": "a", "after": "b", "clean": True}
    assert FN.compute(clean_record(**{"ref.native": leg("ref.native", "fail", "crash", bracket=br)}), c)[0] == -0.8
    dirty = dict(br, clean=False)
    assert FN.compute(clean_record(**{"ref.native": leg("ref.native", "fail", "crash", bracket=dirty)}), c) == \
        (None, {"reason": "ref.native.control"})
    assert FN.compute(clean_record(**{"san.consan": leg("san.consan", "fail", "timeout", bracket=br)}), c) == \
        (None, {"reason": "sanitizer_not_settled"})
    assert FN.compute(clean_record(**{"gate.static": leg("gate.static", "fail", "no_such")}), c) == \
        (None, {"reason": "unknown_gate_reason"})
    two = clean_record(**{"san.consan": leg("san.consan", "finding"), "san.asan": leg("san.asan", "finding")})
    assert FN.compute(two, c)[0] == -0.5 * 2 / 3
    assert FN.compute(clean_record(), cand(0, tokens=0, parent_healthy=True))[0] == 1.0  # clean: no broke_healthy
    assert FN.compute(two, cand(0, tokens=0, parent_healthy=True))[0] == -0.5 * 2 / 3 - 0.5
    short = clean_record(**{"san.asan": leg("san.asan", repeats=1)})
    assert StandinV11(clean_repeats=2).compute(short, c) == (None, {"reason": "clean_not_proven"})


def test_an_unknown_reference_fail_reason_is_not_ignored():
    """Appendix B's code passes over a `fail` it has no rung for; the stand-in treats it as unsettled."""
    rec = clean_record(**{"ref.native": leg("ref.native", "fail", "segfault_but_not_said_so")})
    assert FN.compute(rec, cand(0)) == (None, {"reason": "ref.native"})


def test_the_code_hash_covers_the_module():
    assert len(FN.code_sha256) == 64
    assert dataclasses.is_dataclass(leg("san.asan"))
