"""The selector's records, plan validation, audit coin, Wilson bound and safety switch (Section 6.3).

C3.T4 (the coin is unbiased; it depends on the run seed and the candidate id alone) and C3.T5 (the
safety switch fires at the configured miss rate) are checked here against independent restatements:
the hash and the Wilson bound are written out again in this file, in a different form.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import pytest

from aorlta.ids import candidate_id
from aorlta.selector.base import (
    AUDIT_SCHEMA,
    CheckPlan,
    CheckPlanner,
    PlanRequest,
    SafetySwitch,
    SkipAtRandom,
    SkipEverything,
    SkipNothing,
    audit_coin,
    audit_draw,
    plan_json,
    plan_problem,
    safe_leg_names,
    wilson_interval,
)
from selector_helpers import restated_draw

PLANNED = ["ref.native", "ref.edge", "san.consan", "san.waitcheck", "san.asan"]
Z = 1.959963984540054
CHI2_1DF_P001 = 10.828  # the chi-squared critical value for 1 degree of freedom at p = 0.001


def req(cid: str = "run/1/standin:p0/0", legs=PLANNED) -> PlanRequest:
    return PlanRequest(cid, "a" * 64, {"response_text": "x"}, list(legs), {}, None)


def plan(legs: dict, **kw) -> CheckPlan:
    fields = dict(selector_version="sel/1", plan_id="p-1", legs=legs, context=None)
    fields.update(kw)
    return CheckPlan(**fields)


def entry(decision="run", p=None, score=None):
    return {"decision": decision, "p_finding": p, "score": score}


def full(**over):
    legs = {leg: entry() for leg in PLANNED}
    legs.update(over)
    return legs


# --- plan validation -------------------------------------------------------------------------------


def test_a_complete_plan_is_usable():
    assert plan_problem(plan(full()), PLANNED) is None
    assert plan_problem(plan(full(**{"san.asan": entry("skip", 0.1, -3)})), PLANNED) is None
    assert plan_problem(plan(full(**{"san.asan": entry("skip", 1, 2.5)})), PLANNED) is None  # ints are numbers


@pytest.mark.parametrize("bad, reason", [
    (lambda: "not a plan", "not_a_check_plan"),
    (lambda: plan(full(), selector_version=""), "bad_selector_version"),
    (lambda: plan(full(), selector_version="has a space"), "bad_selector_version"),
    (lambda: plan(full(), plan_id="x" * 129), "bad_plan_id"),
    (lambda: plan(full(), context=7), "bad_context"),
    (lambda: plan([("san.asan", entry())]), "bad_legs"),
    (lambda: plan(full(**{"san.mystery": entry()})), "unknown_leg"),
    (lambda: plan({k: v for k, v in full().items() if k != "san.asan"}), "missing_leg"),
    (lambda: plan(full(**{"san.asan": {"decision": "run"}})), "bad_leg_entry"),
    (lambda: plan(full(**{"san.asan": dict(entry(), extra=1)})), "bad_leg_entry"),
    (lambda: plan(full(**{"san.asan": entry("maybe")})), "bad_decision"),
    (lambda: plan(full(**{"san.asan": entry("skip", 1.5)})), "bad_p_finding"),
    (lambda: plan(full(**{"san.asan": entry("skip", True)})), "bad_p_finding"),
    (lambda: plan(full(**{"san.asan": entry("skip", float("nan"))})), "bad_p_finding"),
    (lambda: plan(full(**{"san.asan": entry("skip", 10**400)})), "bad_p_finding"),  # no float holds it
    (lambda: plan(full(**{"san.asan": entry("skip", None, float("inf"))})), "bad_score"),
    (lambda: plan(full(**{"san.asan": entry("skip", None, "high")})), "bad_score"),
    (lambda: plan(full(**{"san.asan": entry("skip", None, 10**400)})), "bad_score"),
])
def test_a_plan_that_breaks_the_interface_is_rejected(bad, reason):
    assert plan_problem(bad(), PLANNED) == reason


def test_rejected_leg_names_are_logged_only_if_they_are_leg_names():
    names = safe_leg_names(["san.mystery", "int main() { return 0; }", 7])
    assert len(names) == 3 and "san.mystery" in names
    assert all(n == "san.mystery" or (n.startswith("sha256:") and len(n) == 19) for n in names)
    assert not any("main" in n for n in names)


def test_a_logged_plan_carries_the_contexts_hash_never_its_text():
    text = "__global__ void k(float* x) { x[threadIdx.x] = 0; }"
    logged = plan_json(plan(full(**{"san.consan": entry("skip", 0.25, 1)}), context=text))
    assert text not in json.dumps(logged)
    assert logged["context_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert logged["context_chars"] == len(text)
    assert logged["skip"] == ["san.consan"]
    assert logged["legs"]["san.consan"] == {"decision": "skip", "p_finding": 0.25, "score": 1.0}


# --- the test planners -----------------------------------------------------------------------------


@pytest.mark.parametrize("planner", [SkipNothing(), SkipEverything(), SkipAtRandom(seed=3, p_skip=0.5)])
def test_the_test_planners_answer_valid_plans(planner):
    assert isinstance(planner, CheckPlanner)
    p = planner.plan(req())
    assert plan_problem(p, PLANNED) is None
    assert planner.score({"bug": 1}, [{}, {}]) == [{"score": None, "context": None}] * 2


def test_skip_nothing_and_skip_everything_mean_what_they_say():
    assert set(plan_json(SkipNothing().plan(req()))["skip"]) == set()
    assert set(plan_json(SkipEverything().plan(req()))["skip"]) == set(PLANNED)


def test_skip_at_random_is_a_pure_function_of_its_seed_the_candidate_and_the_leg():
    a, b = SkipAtRandom(seed=1, p_skip=0.3), SkipAtRandom(seed=1, p_skip=0.3)
    reqs = [req(candidate_id("r", 1, f"standin:p{i}", 0)) for i in range(4000)]
    assert [a.plan(r) for r in reqs] == [b.plan(r) for r in reqs]
    skips = sum(len(plan_json(a.plan(r))["skip"]) for r in reqs)
    assert abs(skips / (len(reqs) * len(PLANNED)) - 0.3) < 0.02
    other = SkipAtRandom(seed=2, p_skip=0.3)
    assert [a.plan(r).legs for r in reqs[:50]] != [other.plan(r).legs for r in reqs[:50]]
    per_leg = SkipAtRandom(seed=1, p_skip={"san.asan": 1.0})
    assert plan_json(per_leg.plan(req()))["skip"] == ["san.asan"]


# --- C3.T4: the audit coin --------------------------------------------------------------------------


def ids(n: int) -> list[str]:
    return [candidate_id("wp6-chi2", 1 + i // 4096, f"standin:p{(i // 8) % 512}", i % 8) for i in range(n)]


def test_c3_t4_the_coin_is_a_hash_of_the_run_seed_and_the_candidate_id_alone():
    assert AUDIT_SCHEMA == "aorlta.selector.audit/1"
    for cid in ids(200):
        assert audit_draw(20261102, cid) == restated_draw(20261102, cid)
    assert audit_draw(1, "a/1/p/0") != audit_draw(2, "a/1/p/0")
    assert list(PlanRequest.__dataclass_fields__) == ["candidate_id", "source_sha256", "kernel_features",
                                                      "available_legs", "leg_cost_s", "budget_s"]


@pytest.mark.parametrize("f", [0.05, 0.2, 0.5])
def test_c3_t4_the_coin_is_unbiased_chi_squared_over_1e5_draws(f):
    n = 100_000
    audited = sum(audit_coin(audit_draw(7, cid), f) for cid in ids(n))
    chi2 = (audited - n * f) ** 2 / (n * f) + ((n - audited) - n * (1 - f)) ** 2 / (n * (1 - f))
    assert chi2 < CHI2_1DF_P001, (audited, chi2)


def test_c3_t4_the_coin_is_independent_of_a_plan_chi_squared_over_1e5_draws():
    """A 2 x 2 table of (audited, the plan skips ASan) over 10^5 candidates."""
    n, f = 100_000, 0.2
    planner = SkipAtRandom(seed=11, p_skip=0.6)
    table = [[0, 0], [0, 0]]
    for cid in ids(n):
        table[audit_coin(audit_draw(7, cid), f)][planner.draw(cid, "san.asan") < 0.6] += 1
    rows, cols = [sum(r) for r in table], [table[0][j] + table[1][j] for j in (0, 1)]
    chi2 = sum((table[i][j] - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n) for i in (0, 1) for j in (0, 1))
    assert chi2 < CHI2_1DF_P001, (table, chi2)


def test_the_coin_at_two_fractions_nests_and_the_edges_hold():
    for cid in ids(2000):
        u = audit_draw(5, cid)
        assert 0.0 <= u < 1.0
        assert not audit_coin(u, 0.0) and audit_coin(u, 1.0)
        assert (not audit_coin(u, 0.2)) or audit_coin(u, 0.3)


# --- the Wilson interval -----------------------------------------------------------------------------


def restated_upper(k: int, n: int, z: float = Z) -> float:
    """(2np + z^2 + z sqrt(z^2 + 4np(1 - p))) / (2(n + z^2)), the closed form of the upper bound."""
    if n == 0:
        return 1.0
    p = k / n
    return min(1.0, (2 * n * p + z * z + z * math.sqrt(z * z + 4 * n * p * (1 - p))) / (2 * (n + z * z)))


@pytest.mark.parametrize("k, n", [(0, 1), (0, 189), (1, 10), (3, 120), (28, 1120), (80, 1200), (10, 10), (0, 0)])
def test_wilson_interval_matches_the_closed_form(k, n):
    lo, hi = wilson_interval(k, n)
    assert hi == pytest.approx(restated_upper(k, n), abs=1e-12)
    assert 0.0 <= lo <= hi <= 1.0
    if n:
        assert lo <= k / n <= hi
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_reference_values():
    """Published values: 0 of 10 gives [0, 0.2775]; 5 of 10 gives [0.2366, 0.7634]."""
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))


# --- C3.T5: the safety switch -----------------------------------------------------------------------


def n_to_arm(m: float, z: float = Z) -> int:
    """Observations with no miss before the bound would stay at most m even after one more, a miss."""
    n = 0
    while restated_upper(1, n + 1, z) > m:
        n += 1
    return n


def misses_at_rate(p: float, n: int) -> list[bool]:
    """Observation t is a miss iff floor((t + 1) p) > floor(t p): exactly floor(n p) misses in n."""
    return [math.floor((t + 1) * p) > math.floor(t * p) for t in range(n)]


def expected_trip(pattern: list[bool], m: float) -> tuple[int | None, int | None]:
    """The 1-based observation at which the restated switch arms (the bound with one more miss is at
    most m), and the one at which it fires (the bound exceeds m)."""
    state, k, armed, tripped = ("armed" if restated_upper(1, 1) <= m else "warming"), 0, None, None
    for t, miss in enumerate(pattern, 1):
        k += miss
        if state == "warming" and restated_upper(k + 1, t + 1) <= m:
            state, armed = "armed", t
        elif state == "armed" and restated_upper(k, t) > m:
            state, tripped = "tripped", t
            break
    return armed, tripped


@pytest.mark.parametrize("m", [0.01, 0.02, 0.05])
def test_c3_t5_a_selector_that_never_misses_arms_with_a_one_miss_margin_and_never_fires(m):
    sw = SafetySwitch(["san.asan"], m)
    n = n_to_arm(m)
    assert m != 0.02 or n == 279
    for t in range(1, n + 5000):
        state = sw.observe("san.asan", would_skip=True, defect=False)
        assert state == ("warming" if t < n else "armed"), t
    assert sw.allows_skip("san.asan")


@pytest.mark.parametrize("m", [0.01, 0.02, 0.05])
@pytest.mark.parametrize("factor", [1.5, 3.0])
def test_c3_t5_the_switch_fires_at_the_configured_miss_rate(m, factor):
    """Armed by a clean record, then missing at factor x m: it fires at the first observation whose
    Wilson upper bound exceeds m, and not one observation earlier or later."""
    n0 = n_to_arm(m) + 200
    pattern = [False] * n0 + misses_at_rate(factor * m, 40_000)
    armed, tripped = expected_trip(pattern, m)
    assert armed is not None and tripped is not None
    sw = SafetySwitch(["san.consan"], m)
    for t, miss in enumerate(pattern, 1):
        state = sw.observe("san.consan", would_skip=True, defect=miss)
        if t < tripped:
            assert state == ("armed" if t >= armed else "warming"), t
        else:
            assert state == "tripped" and not sw.allows_skip("san.consan")
            break
    assert sw.snapshot()["san.consan"]["tripped_at"] == tripped
    assert sw.upper("san.consan") > m


@pytest.mark.parametrize("m", [0.01, 0.02, 0.05])
def test_c3_t5_a_miss_rate_well_under_the_limit_never_fires(m):
    pattern = misses_at_rate(m / 3, 60_000)
    armed, tripped = expected_trip(pattern, m)
    assert armed is not None and tripped is None
    sw = SafetySwitch(["san.asan"], m)
    for miss in pattern:
        sw.observe("san.asan", would_skip=True, defect=miss)
    assert sw.state("san.asan") == "armed" and sw.snapshot()["san.asan"]["armed_at"] == armed


def test_c3_t5_the_switch_stays_off_until_reset_by_hand():
    sw = SafetySwitch(["san.asan", "san.consan"], 0.02)
    for _ in range(279):
        sw.observe("san.asan", would_skip=True, defect=False)
    assert sw.allows_skip("san.asan")
    while sw.state("san.asan") != "tripped":
        sw.observe("san.asan", would_skip=True, defect=True)
    for _ in range(20_000):  # however good the evidence becomes, it stays off
        sw.observe("san.asan", would_skip=True, defect=False)
    assert sw.state("san.asan") == "tripped" and sw.upper("san.asan") < 0.02
    assert sw.state("san.consan") == "warming"  # per leg
    sw.reset("san.asan")
    assert sw.snapshot()["san.asan"] == {"state": "warming", "observations": 0, "misses": 0, "upper": 1.0,
                                         "armed_at": None, "tripped_at": None}


def test_only_a_defect_in_a_leg_the_plan_would_skip_is_a_miss():
    sw = SafetySwitch(["san.asan"], 0.02)
    for _ in range(400):
        sw.observe("san.asan", would_skip=False, defect=True)  # found, and the plan would have run it
        sw.observe("san.asan", would_skip=True, defect=False)  # skipped, and nothing to find
    assert sw.snapshot()["san.asan"]["misses"] == 0 and sw.state("san.asan") == "armed"


def test_the_switch_edges():
    assert SafetySwitch(["san.asan"], 1.0).allows_skip("san.asan")  # a limit of 1 is no limit: armed at once
    never = SafetySwitch(["san.asan"], 0.0)
    for _ in range(100_000 // 10):
        never.observe("san.asan", would_skip=True, defect=False)
    assert never.state("san.asan") == "warming"  # a limit of 0 never trusts a skip
    with pytest.raises(ValueError):
        SafetySwitch(["san.asan"], 1.5)
    with pytest.raises(KeyError):
        SafetySwitch(["san.asan"], 0.02).observe("san.unknown", would_skip=True, defect=True)


def test_records_are_the_docs():
    assert [f.name for f in dataclasses.fields(CheckPlan)] == ["selector_version", "plan_id", "legs", "context"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        req().budget_s = 1.0
