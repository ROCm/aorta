# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the RL reward seams under ``examples/rl``.

These are examples rather than package modules, so they are loaded by path.
They are still worth testing: both are scorers, and a scorer that silently stops
detecting what it claims to detect is the failure mode the whole design is
guarding against. The novelty gate and the degenerate-policy floor are the two
things most likely to rot unnoticed, so they are what these pin down.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rl"


def _load(name: str):
    """Import an example module by path, registered so dataclasses resolve.

    ``@dataclass`` looks the owning module up in ``sys.modules`` to resolve
    string annotations, so a module loaded without registering it there raises
    on the first dataclass. Hence the assignment before ``exec_module``.
    """
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipe_reward():
    return _load("recipe_reward")


@pytest.fixture(scope="module")
def triage_reward():
    return _load("triage_reward")


@pytest.fixture(scope="module")
def proposal_reward():
    return _load("proposal_reward")


def _real_detector_ids() -> set[str]:
    """Every detector ID the classifier tiers can actually emit.

    Read from the tier modules' own ``DETECTOR_*`` constants and ID frozensets
    so a rename shows up as a test failure rather than as a fixture quietly
    citing a detector that no longer exists.
    """
    ids: set[str] = set()
    for name in (
        "tier1_process",
        "tier2_hang",
        "tier3_kernel",
        "tier4_patterns",
        "tier5_custom",
        "verdict",
    ):
        module = importlib.import_module(f"aorta.probe.classifier.{name}")
        for key, value in vars(module).items():
            if key.startswith("DETECTOR") and isinstance(value, str) and ":" in value:
                ids.add(value)
            elif key.endswith("DETECTOR_IDS") and isinstance(value, frozenset):
                ids |= {v for v in value if isinstance(v, str) and ":" in v}
    return ids


# --------------------------------------------------------------------------- #
# recipe_reward: the novelty gate
# --------------------------------------------------------------------------- #


def test_the_taper_runs_from_full_reward_to_none(recipe_reward):
    m = recipe_reward
    assert m.novelty_multiplier(0.0) == 1.0
    assert m.novelty_multiplier(m.MEMORISATION_SOFT - 0.01) == 1.0
    assert m.novelty_multiplier(m.MEMORISATION_HARD) == 0.0
    assert m.novelty_multiplier(1.0) == 0.0
    midpoint = (m.MEMORISATION_SOFT + m.MEMORISATION_HARD) / 2
    assert 0.0 < m.novelty_multiplier(midpoint) < 1.0


def test_the_taper_is_monotonic_so_copying_more_never_pays_more(recipe_reward):
    m = recipe_reward
    steps = [i / 100 for i in range(0, 101)]
    values = [m.novelty_multiplier(s) for s in steps]
    assert values == sorted(values, reverse=True)


def test_canonicalising_ignores_edits_that_change_nothing(recipe_reward):
    """The evasions the gate has to survive: rename, reorder, reindent, uncomment."""
    original = (
        "# a leading comment\n"
        "schema_version: 1\n"
        "ticket: TICKET-ONE\n"
        "workload: tokenspeed_serve\n"
        "trials: 1\n"
        "steps: 1\n"
        "cells:\n"
        "  - name: only\n"
        "    mitigations: [none]\n"
        "    environment: local\n"
    )
    evaded = (
        "steps: 1\n"
        "trials: 1\n"
        "workload: tokenspeed_serve\n"
        "ticket: TICKET-RENAMED-TO-EVADE\n"
        "schema_version: 1\n"
        "cells:\n"
        "    -   name: only\n"
        "        environment: local\n"
        "        mitigations:\n"
        "            - none\n"
    )
    assert recipe_reward.canonicalise(original) == recipe_reward.canonicalise(evaded)


def test_canonicalising_still_separates_recipes_that_differ(recipe_reward):
    base = "schema_version: 1\nticket: T\nworkload: tokenspeed_serve\ntrials: 1\n"
    changed = base.replace("trials: 1", "trials: 8")
    assert recipe_reward.canonicalise(base) != recipe_reward.canonicalise(changed)


def test_unparseable_text_canonicalises_without_raising(recipe_reward):
    broken = "ticket: [unclosed\n"
    assert recipe_reward.canonicalise(broken) == broken


def test_a_verbatim_corpus_copy_earns_nothing(recipe_reward):
    """The whole point: full tier marks, zero reward."""
    corpus = {"recipes/committed.yaml": recipe_reward._GOOD}
    grade = recipe_reward.grade_recipe_text(recipe_reward._GOOD, corpus=corpus)

    assert grade.tier == recipe_reward.MAX_TIER
    assert grade.tier_reward == 1.0
    assert grade.memorised is True
    assert grade.novelty_multiplier == 0.0
    assert grade.reward == 0.0
    assert grade.nearest_committed is not None
    assert grade.nearest_committed[1] == pytest.approx(1.0)


def test_a_cosmetically_edited_copy_earns_nothing_either(recipe_reward):
    corpus = {"recipes/committed.yaml": recipe_reward._GOOD}
    evaded = recipe_reward._cosmetic_mutation(recipe_reward._GOOD)

    # Not the same bytes -- so a raw-text comparison would have paid out.
    assert evaded != recipe_reward._GOOD
    grade = recipe_reward.grade_recipe_text(evaded, corpus=corpus)
    assert grade.tier == recipe_reward.MAX_TIER
    assert grade.memorised is True
    assert grade.reward == 0.0


def test_a_genuinely_novel_valid_recipe_keeps_its_full_reward(recipe_reward):
    """Otherwise the gate is a difficulty penalty, not a novelty term."""
    novel = recipe_reward._GOOD
    # A committed corpus of something structurally unrelated.
    corpus = {
        "recipes/other.yaml": yaml.safe_dump(
            {
                "schema_version": 1,
                "ticket": "OTHER-1",
                "workload": "some_other_workload",
                "trials": 9,
                "steps": 4,
                "cells": [{"name": f"c{i}", "mitigations": ["none"]} for i in range(6)],
            }
        )
    }
    grade = recipe_reward.grade_recipe_text(novel, corpus=corpus)

    assert grade.tier == recipe_reward.MAX_TIER
    assert grade.memorised is False
    assert grade.novelty_multiplier == 1.0
    assert grade.reward == 1.0


def test_without_a_corpus_the_gate_cannot_fire(recipe_reward):
    """No corpus means no novelty claim, so the tier reward stands unmodified."""
    grade = recipe_reward.grade_recipe_text(recipe_reward._GOOD, corpus=None)
    assert grade.reward == grade.tier_reward
    assert grade.memorised is False
    assert grade.nearest_committed is None


def test_a_malformed_copy_is_not_rescued_by_being_a_copy(recipe_reward):
    """The gate scales the tier reward; it never invents one."""
    corpus = {"recipes/committed.yaml": recipe_reward._BAD_YAML}
    grade = recipe_reward.grade_recipe_text(recipe_reward._BAD_YAML, corpus=corpus)
    assert grade.tier == 0
    assert grade.reward == 0.0


# --------------------------------------------------------------------------- #
# triage_reward: labelling and the degenerate floor
# --------------------------------------------------------------------------- #


def test_labels_come_from_the_resolver_not_the_stored_field(triage_reward):
    """A stored verdict that disagrees with the rules is corrected and flagged."""
    doc = triage_reward._run("mislabelled", "pass", ["tier1:exit_nonzero"], [])
    label = triage_reward.label_run(doc)

    assert label.verdict == "fail"
    assert label.stored_verdict == "pass"
    assert label.stale is True


def test_an_agreeing_run_is_not_flagged_stale(triage_reward):
    doc = triage_reward._run("clean", "pass", [], [])
    label = triage_reward.label_run(doc)
    assert label.verdict == "pass"
    assert label.stale is False


def test_an_infra_only_run_is_an_error_not_a_failure(triage_reward):
    doc = triage_reward._run("launch", "error", [], ["tier1:exec_failed"])
    assert triage_reward.label_run(doc).verdict == "error"


def test_a_failure_alongside_infra_noise_still_reproduced(triage_reward):
    """fail > error: the bug reproducing outranks the trial also being flaky."""
    doc = triage_reward._run("both", "fail", ["tier1:sigabrt"], ["tier1:timeout"])
    label = triage_reward.label_run(doc)
    assert label.verdict == "fail"
    assert "tier1:sigabrt" in label.failure_detectors
    assert "tier1:timeout" in label.error_detectors


def test_a_detector_recorded_on_the_wrong_side_is_re_partitioned(triage_reward):
    """The recorded split is recombined and re-split through aorta's own rule."""
    doc = triage_reward._run("misfiled", "error", ["tier1:timeout"], [])
    label = triage_reward.label_run(doc)
    assert label.error_detectors == ["tier1:timeout"]
    assert label.failure_detectors == []
    assert label.verdict == "error"


def test_a_correct_answer_earns_the_full_reward(triage_reward):
    doc = triage_reward._run("x", "fail", ["tier1:sigsegv", "tier3:amdgpu_reset"], [])
    label = triage_reward.label_run(doc)
    answer = triage_reward.Answer("fail", ["tier3:amdgpu_reset", "tier1:sigsegv"])
    score = triage_reward.score_answer(answer, label)

    assert score.verdict_correct is True
    assert score.attribution_f1 == pytest.approx(1.0)
    assert score.reward == pytest.approx(1.0)


def test_a_clean_run_needs_no_citation_to_score_perfectly(triage_reward):
    label = triage_reward.label_run(triage_reward._run("clean", "pass", [], []))
    score = triage_reward.score_answer(triage_reward.Answer("pass", []), label)
    assert score.reward == pytest.approx(1.0)


def test_the_right_verdict_with_an_invented_reason_is_docked(triage_reward):
    """The 'right answer, wrong reason' case the attribution term exists for."""
    label = triage_reward.label_run(
        triage_reward._run("x", "fail", ["tier1:exit_nonzero"], [])
    )
    score = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier9:invented"]), label
    )

    assert score.verdict_correct is True
    assert score.attribution_f1 == 0.0
    assert score.reward == pytest.approx(triage_reward.VERDICT_WEIGHT)
    assert score.reward < 1.0


def test_a_partly_right_citation_earns_partial_credit(triage_reward):
    label = triage_reward.label_run(
        triage_reward._run("x", "fail", ["tier1:sigsegv", "tier3:amdgpu_reset"], [])
    )
    half = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier1:sigsegv"]), label
    )
    none = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier9:invented"]), label
    )
    full = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier1:sigsegv", "tier3:amdgpu_reset"]), label
    )
    assert none.reward < half.reward < full.reward


def test_confusing_a_failure_for_an_infra_error_loses_the_verdict_term(triage_reward):
    label = triage_reward.label_run(
        triage_reward._run("x", "fail", ["tier1:exit_nonzero"], [])
    )
    score = triage_reward.score_answer(
        triage_reward.Answer("error", ["tier1:exit_nonzero"]), label
    )
    assert score.verdict_correct is False
    assert score.reward == pytest.approx(triage_reward.ATTRIBUTION_WEIGHT)


def test_the_always_pass_policy_scores_well_enough_to_need_reporting(triage_reward):
    """The floor a real policy has to clear, on a corpus skewed towards pass."""
    labelled = [
        (f"synthetic:{d['cell_name']}", triage_reward.label_run(d))
        for d in triage_reward.FIXTURES
    ]
    always_pass = triage_reward.score_policy(
        "always pass", lambda _: triage_reward.Answer("pass", []), labelled
    )
    oracle = triage_reward.score_policy(
        "oracle",
        lambda lb: triage_reward.Answer(lb.verdict, sorted(lb.cited_detectors)),
        labelled,
    )

    assert oracle["mean_reward"] == pytest.approx(1.0)
    # Reading nothing is worth real reward, which is exactly why the demo prints
    # this number next to the policy's.
    assert always_pass["mean_reward"] > 0.2
    assert always_pass["mean_reward"] < oracle["mean_reward"]


def test_the_fixtures_cover_all_three_verdicts(triage_reward):
    """A fixture set missing a class would hide the term that detects it."""
    verdicts = {triage_reward.label_run(d).verdict for d in triage_reward.FIXTURES}
    assert verdicts == {"pass", "fail", "error"}


def test_no_fixture_is_stale_against_the_current_rules(triage_reward):
    """The fixtures encode today's precedence; if this fails, the rules moved."""
    stale = [d["cell_name"] for d in triage_reward.FIXTURES
             if triage_reward.label_run(d).stale]
    assert stale == []


def test_every_fixture_cites_only_detectors_a_tier_can_emit(triage_reward):
    """A fixture citing an invented ID trains attribution on a fake vocabulary.

    The attribution term is scored against exactly these IDs, so a fixture that
    names something no classifier tier produces rewards the model for citing a
    detector it will never see in a real run. Collected from the tier modules'
    own constants rather than hard-coded, so a renamed detector fails here
    instead of rotting silently.
    """
    real = _real_detector_ids()
    assert real, "no detector constants found; the classifier layout moved"
    for doc in triage_reward.FIXTURES:
        label = triage_reward.label_run(doc)
        invented = sorted(label.cited_detectors - real)
        assert not invented, f"{doc['cell_name']} cites unknown {invented}"


def test_the_debugging_fixtures_cover_the_failure_shapes_the_agent_sees(
    triage_reward,
):
    """The vertical is debugging, so the corpus has to contain its shapes.

    One representative per autopsy category the proposal contract enumerates
    and the classifier can actually evidence. Without these the fixture set
    only exercises signal/exit-code failures, which is not what `aorta agent`
    is pointed at.
    """
    cited = set()
    for doc in triage_reward.FIXTURES:
        cited |= triage_reward.label_run(doc).cited_detectors
    for required in (
        "tier4:collective_timeout",  # rccl_hang
        "tier4:hip_error",  # illegal_mem
        "tier3:vm_l2_fault",  # illegal_mem, the underlying fault
        "tier3:thermal_throttle",  # thermal_throttle
        "tier3:xgmi_link_error",  # fabric
        "tier4:nan_signature",  # numerics, with a zero exit code
    ):
        assert required in cited, f"no fixture evidences {required}"


def test_an_advisory_warn_is_never_part_of_the_justification(triage_reward):
    """`tier3:vram_growth` is a warn, so citing it must not earn attribution.

    The reset-with-warn fixture carries it precisely as a red herring: a policy
    that lists every detector it can see, warns included, should be docked.
    """
    doc = next(
        d for d in triage_reward.FIXTURES if d["cell_name"] == "reset-with-warn"
    )
    label = triage_reward.label_run(doc)
    assert "tier3:vram_growth" not in label.cited_detectors
    assert label.verdict == "fail"

    everything = triage_reward.Answer(
        verdict="fail",
        detectors=sorted(label.cited_detectors | {"tier3:vram_growth"}),
    )
    exact = triage_reward.Answer(
        verdict="fail", detectors=sorted(label.cited_detectors)
    )
    assert (
        triage_reward.score_answer(everything, label).reward
        < triage_reward.score_answer(exact, label).reward
    )


# --------------------------------------------------------------------------- #
# proposal_reward: the contract `aorta agent` actually enforces
# --------------------------------------------------------------------------- #


def test_an_on_contract_proposal_reaches_the_top_tier(proposal_reward):
    proposal = proposal_reward.Proposal(
        "valid",
        json.dumps(
            {
                "category": "rccl_hang",
                "hypothesis": "collective timed out on every rank",
                "next_mitigations": ["nccl_launch_order_implicit"],
                "confidence": 0.6,
                "stop": False,
            }
        ),
        ["nccl_launch_order_implicit", "tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.tier == proposal_reward.MAX_TIER
    assert score.reward == 1.0
    assert score.consumer_outcome == "accepted"


def test_a_hallucinated_mitigation_is_docked_and_would_stop_the_search(
    proposal_reward,
):
    """The defect this reward exists for.

    `LiteLLMProposer` filters unrecognised names out before the policy sees
    them, so a proposal naming only invented mitigations reaches the loop as a
    well-formed step with nothing to try, and the search ends. Nothing raises.
    The reward has to notice what the consumer does not.
    """
    proposal = proposal_reward.Proposal(
        "hallucinated",
        json.dumps(
            {
                "category": "rccl_hang",
                "hypothesis": "disable peer-to-peer",
                "next_mitigations": ["rccl_p2p_disable"],
                "confidence": 0.9,
                "stop": False,
            }
        ),
        ["nccl_launch_order_implicit", "tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.stopped_at == "tier4_registry"
    assert score.reward < 1.0
    assert score.consumer_outcome == "silent_stop"


def test_an_invented_category_is_rejected_loudly_by_the_consumer(proposal_reward):
    """A bad category is the one thing `AgentPolicy` raises on."""
    proposal = proposal_reward.Proposal(
        "bad category",
        json.dumps(
            {
                "category": "rccl_timeout",
                "hypothesis": "h",
                "next_mitigations": ["tf32_off"],
                "confidence": 0.5,
                "stop": False,
            }
        ),
        ["tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.stopped_at == "tier3_category"
    assert score.consumer_outcome == "policy_stop"


def test_coerced_fields_still_lose_reward_even_though_the_consumer_accepts(
    proposal_reward,
):
    """`AgentStep.from_dict` repairs a bad type; the contract still says no.

    This is why the consumer cannot be used as the oracle: it accepts a
    proposal whose confidence it silently zeroed.
    """
    proposal = proposal_reward.Proposal(
        "string confidence",
        json.dumps(
            {
                "category": "rccl_hang",
                "hypothesis": "h",
                "next_mitigations": ["tf32_off"],
                "confidence": "high",
                "stop": False,
            }
        ),
        ["tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.stopped_at == "tier2_schema"
    assert score.consumer_outcome == "accepted"


def test_prose_around_the_object_earns_nothing(proposal_reward):
    proposal = proposal_reward.Proposal(
        "prose", 'Here you go: {"category": "rccl_hang"}', ["tf32_off"]
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.tier == 0
    assert score.reward == 0.0


def test_a_registered_but_unavailable_mitigation_is_docked_one_tier(
    proposal_reward,
):
    """Registered is not the same as offered.

    Re-proposing something already tried is silently filtered too, so it costs
    an iteration rather than raising.
    """
    body = {
        "category": "rccl_hang",
        "hypothesis": "h",
        "next_mitigations": ["hsa_no_sdma"],
        "confidence": 0.5,
        "stop": False,
    }
    proposal = proposal_reward.Proposal(
        "already tried",
        json.dumps(body),
        ["hsa_no_sdma", "tf32_off"],
        tried=["hsa_no_sdma"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.tier == 4
    assert score.stopped_at == "tier5_available"
    assert score.consumer_outcome == "silent_stop"


def test_the_ladder_is_monotonic_so_being_more_wrong_never_pays_more(
    proposal_reward,
):
    """The tier ladder still spans 0 to MAX_TIER, and the ungraded part is flat.

    ``reward == tier / MAX_TIER`` used to hold for every fixture. It no longer
    does, deliberately: that identity is exactly what made the reward saturate,
    since it left the reward blind to *how* a tier was reached. It still holds
    wherever neither graded term applies -- a committed category and a
    mitigation list inside the free budget -- which covers every fixture that
    predates fixes 1 and 2, so a regression in the ungraded path is still
    caught here.
    """
    m = proposal_reward
    tiers = [m.score_proposal(f).tier for f in m.FIXTURES]
    assert min(tiers) == 0
    assert max(tiers) == m.MAX_TIER
    for f in m.FIXTURES:
        score = m.score_proposal(f)
        ungraded = (
            score.category_credit == 1.0
            and score.n_mitigations <= m.FREE_MITIGATIONS
        )
        if ungraded:
            assert score.reward == pytest.approx(score.tier / m.MAX_TIER)
        else:
            assert score.reward < score.tier / m.MAX_TIER


def test_the_reward_is_the_documented_sum_of_graded_tier_steps(proposal_reward):
    """Pins the formula, so the two graded terms cannot drift from the docstring."""
    m = proposal_reward
    for f in m.FIXTURES:
        score = m.score_proposal(f)
        expected = m.TIER_STEP * min(score.tier, 2)
        if score.tier >= 3:
            expected += m.TIER_STEP * score.category_credit
        if score.tier >= 4:
            expected += m.TIER_STEP * (score.tier - 3) * score.precision
        assert score.reward == pytest.approx(expected)


def test_every_fixture_earns_the_reward_it_is_meant_to(proposal_reward):
    """The companion to the tier map: four fixtures now share tier 5 and differ."""
    expected = proposal_reward._reward_expectations()
    actual = {
        f.name: proposal_reward.score_proposal(f).reward
        for f in proposal_reward.FIXTURES
    }
    assert set(actual) == set(expected)
    for name, want in expected.items():
        assert actual[name] == pytest.approx(want), name
    # The point of the whole exercise: tier 5 is no longer a single value.
    top = {
        f.name: proposal_reward.score_proposal(f).reward
        for f in proposal_reward.FIXTURES
        if proposal_reward.score_proposal(f).tier == proposal_reward.MAX_TIER
    }
    assert len(set(top.values())) > 1, top


# --------------------------------------------------------------------------- #
# Fix 1: declining to classify must not earn full marks
#
# `unknown` is a member of the closed autopsy set, so a proposal that refuses
# the classification task used to clear the tier that exists to test it and
# reach 1.0. The trap in fixing this is that on the committed corpus `unknown`
# is frequently the *honest* answer -- 8 of 9 scenarios have no correct category
# available -- so a penalty that pushes the policy towards a confident wrong
# label is worse than the saturation it removes. These tests pin the ordering
# that keeps that from happening.
# --------------------------------------------------------------------------- #


def _proposal(proposal_reward, category="rccl_hang", mitigations=None, **over):
    body = {
        "category": category,
        "hypothesis": "h",
        "next_mitigations": (
            ["nccl_launch_order_implicit"] if mitigations is None else mitigations
        ),
        "confidence": 0.5,
        "stop": False,
    }
    body.update(over)
    return proposal_reward.Proposal(
        "under test",
        json.dumps(body),
        ["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off", "xnack"],
    )


def test_declining_to_classify_no_longer_earns_full_marks(proposal_reward):
    """The headline of fix 1."""
    m = proposal_reward
    score = m.score_proposal(_proposal(m, category="unknown"))
    assert score.tier == m.MAX_TIER
    assert score.reward < 1.0
    assert score.category_credit == m.ABSTENTION_CREDIT
    assert score.abstained is True


def test_an_abstention_still_clears_the_tier_the_consumer_accepts(proposal_reward):
    """Docked, not rejected -- the reward must not disagree with `AgentPolicy`.

    `unknown` is in `AUTOPSY_CATEGORIES` and `validate_step` accepts it, so a
    grader that failed the tier would be scoring a contract the consumer does
    not enforce. That drift is the failure mode this module is built to avoid,
    which is why fix 1 grades the step instead of closing the set.
    """
    m = proposal_reward
    score = m.score_proposal(_proposal(m, category="unknown"))
    assert score.tier >= 3
    assert score.stopped_at == ""
    assert score.consumer_outcome == "accepted"


def test_declining_beats_leaving_the_closed_set(proposal_reward):
    """The ordering that stops fix 1 rewarding dishonesty.

    committed > declined > outside the set. The middle term is the load-bearing
    one: an honest `unknown` has to stay worth more than an invented category,
    or the reward pays a policy to guess its way out of the abstention penalty.
    """
    m = proposal_reward
    committed = m.score_proposal(_proposal(m, category="rccl_hang")).reward
    declined = m.score_proposal(_proposal(m, category="unknown")).reward
    outside = m.score_proposal(_proposal(m, category="rccl_timeout")).reward
    assert committed > declined > outside


def test_the_abstention_dock_is_one_partial_step_not_a_whole_tier(proposal_reward):
    """Pins how *small* the dock is, which is the actual design decision.

    Excluding `unknown` from tier 3's accepted set -- the first option the
    report offered -- would have dropped an abstention to 0.4 while any in-set
    category, right or wrong, still earned 1.0: a 0.6 gradient pointing at
    "invent a confident label". Grading the step instead costs one partial
    step. If this test starts failing upwards, that gradient is being rebuilt.
    """
    m = proposal_reward
    committed = m.score_proposal(_proposal(m, category="rccl_hang")).reward
    declined = m.score_proposal(_proposal(m, category="unknown")).reward
    dock = committed - declined
    assert dock == pytest.approx(m.TIER_STEP * (1.0 - m.ABSTENTION_CREDIT))
    # Strictly smaller than dropping the abstention a whole tier would be.
    assert dock < m.TIER_STEP
    # And far smaller than the exclusion alternative it was chosen over.
    assert dock < (committed - 2 * m.TIER_STEP)


# --------------------------------------------------------------------------- #
# Fix 2: hedging across the candidate set has to cost something
#
# `run_agent_loop` appends every proposed name to the mitigation axis and runs a
# probe cell for each, while charging the whole proposal one unit of the
# iteration budget -- so shotgunning is free against the limit the policy
# enforces and expensive in the resource the operator pays for.
# --------------------------------------------------------------------------- #


def test_a_shotgun_proposal_no_longer_ties_a_targeted_one(proposal_reward):
    """The headline of fix 2, at the size the real run produced."""
    m = proposal_reward
    offered = [f"m{i}" for i in range(17)]
    wide = m.precision_credit(len(offered))
    narrow = m.precision_credit(1)
    assert narrow == 1.0
    assert wide < narrow
    # Same tier, same category, different reward.
    one = m.score_proposal(_proposal(m, mitigations=["nccl_launch_order_implicit"]))
    many = m.score_proposal(
        _proposal(
            m,
            mitigations=["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off"],
        )
    )
    assert one.tier == many.tier == m.MAX_TIER
    assert many.reward < one.reward


def test_naming_a_primary_and_one_fallback_costs_nothing(proposal_reward):
    """The perversity a pure brevity term creates, neutralised at its own margin.

    With no correctness signal the reward cannot tell a right name from a wrong
    one, so any brevity term makes a one-name proposal beat a two-name proposal
    that *contains* the right name. `1/len` puts the largest step the term can
    produce exactly there. A free pair makes the two tie instead, so the reward
    never pays a policy to drop a correct name in order to look decisive.

    It only relocates the problem: at three names and up a confident wrong
    single name still wins. That needs a correctness signal, not a better shape.
    """
    m = proposal_reward
    assert m.FREE_MITIGATIONS >= 2
    one = m.score_proposal(_proposal(m, mitigations=["nccl_launch_order_implicit"]))
    pair = m.score_proposal(
        _proposal(m, mitigations=["nccl_launch_order_implicit", "hsa_no_sdma"])
    )
    assert pair.reward == pytest.approx(one.reward)
    # And the relocation is real, so it is pinned rather than left implicit.
    three = m.score_proposal(
        _proposal(
            m,
            mitigations=["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off"],
        )
    )
    assert three.reward < one.reward


def test_hedging_never_pays_more_than_being_precise(proposal_reward):
    m = proposal_reward
    credits = [m.precision_credit(n) for n in range(1, 41)]
    assert credits == sorted(credits, reverse=True)
    assert credits[0] == 1.0
    assert credits[-1] < 0.1


def test_precision_is_the_reciprocal_of_the_cell_count_past_the_free_pair(
    proposal_reward,
):
    """The cost model stated as arithmetic: k names is k probe cells."""
    m = proposal_reward
    for n in range(1, m.FREE_MITIGATIONS + 1):
        assert m.precision_credit(n) == 1.0
    for n in range(m.FREE_MITIGATIONS + 1, 25):
        assert m.precision_credit(n) == pytest.approx(m.FREE_MITIGATIONS / n)
    # An empty list never reaches the block, but the function is still total.
    assert m.precision_credit(0) == 0.0


def test_the_two_saturation_routes_are_now_separately_visible(proposal_reward):
    """Abstaining and shotgunning were both routes to 1.0; now they compound.

    The recorded model did both at once on most scenarios, so the reward has to
    dock both independently rather than collapsing them into one penalty.
    """
    m = proposal_reward
    wide = ["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off", "xnack"]
    clean = m.score_proposal(_proposal(m)).reward
    abstains = m.score_proposal(_proposal(m, category="unknown")).reward
    shotguns = m.score_proposal(_proposal(m, mitigations=wide)).reward
    both = m.score_proposal(
        _proposal(m, category="unknown", mitigations=wide)
    ).reward
    assert both < abstains < clean
    assert both < shotguns < clean


def test_a_constant_that_reads_nothing_is_no_longer_worth_a_diagnosis(
    proposal_reward,
):
    """The saturation, restated as the comparison that failed before.

    Both abstaining constants scored 1.0 on the shipped fixtures, tying the
    on-contract baseline. They no longer do.
    """
    rows = {row["policy"]: row for row in proposal_reward.baselines()}
    diagnosis = rows["always the same valid proposal"]["mean_reward"]
    assert diagnosis == 1.0
    for constant in ("always abstain, one mitigation", "always abstain, shotgun everything"):
        assert rows[constant]["mean_reward"] < diagnosis
    # The one that stays uncomfortably high, and is worth keeping in view: a
    # single-name abstention reads no input and still clears 0.85, because a
    # cheap answer is most of what a form reward can see.
    assert rows["always abstain, one mitigation"]["mean_reward"] > 0.85


def test_every_fixture_stops_where_it_is_meant_to(proposal_reward):
    """Pins each failure mode to its tier, so a loosened check is visible."""
    expected = proposal_reward._fixture_expectations()
    actual = {
        f.name: proposal_reward.score_proposal(f).tier
        for f in proposal_reward.FIXTURES
    }
    assert actual == expected


def test_the_categories_come_from_the_agent_not_a_copy(proposal_reward):
    """The closed set is imported, so adding a category updates the reward."""
    from aorta.agent.llm import AUTOPSY_CATEGORIES

    assert proposal_reward.AUTOPSY_CATEGORIES is AUTOPSY_CATEGORIES


def test_a_fixed_valid_proposal_scores_full_marks_without_diagnosing_anything(
    proposal_reward,
):
    """The ceiling of a format reward, stated as a test.

    If this ever fails, the reward has started measuring substance and the
    docstring's claim that it is only a gate is wrong.
    """
    rows = {row["policy"]: row for row in proposal_reward.baselines()}
    fixed = rows["always the same valid proposal"]
    assert fixed["mean_reward"] == 1.0
    assert fixed["accepted_rate"] == 1.0


# --------------------------------------------------------------------------- #
# triage_reward: the sanitizer label source
#
# Waitcheck and ConSan are the two tools the CIA architecture highlights, so
# their reports are on-domain evidence for the root-cause half rather than a
# separate concern. These run against the reports committed under
# recipes/sanitizers/survey, which are the only real labelled failure evidence
# in the tree.
# --------------------------------------------------------------------------- #

_SURVEY = Path(__file__).resolve().parents[2] / "recipes" / "sanitizers" / "survey"


def test_the_committed_sanitizer_reports_all_label(triage_reward):
    """Every committed report loads and yields a verdict from the tools' own set."""
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    assert len(labelled) == 6, "the committed survey reports moved or changed"
    verdicts = {label.verdict for _, label in labelled}
    assert verdicts <= {"pass", "warn", "fail", "not_checked", "error"}
    assert verdicts == {"pass", "warn", "error"}


def test_a_wait_hazard_is_cited_by_its_namespaced_code(triage_reward):
    """Attribution reads like a detector ID and cannot be confused with one."""
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    hazard = next(
        label for src, label in labelled if "gemm_f32_waitcheck" in src
    )
    assert hazard.verdict == "warn"
    assert hazard.cited_detectors == {"waitcheck:wait_hazard"}


def test_a_sanitizer_that_did_not_run_attributes_nothing(triage_reward):
    """An `error` verdict means no observation, so there is nothing to cite.

    Mirrors the probe side, where a trial that never validly ran is `error` and
    carries no failure detectors.
    """
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    errored = [label for src, label in labelled if "consan" in src]
    assert errored, "expected the ConSan reports to be present"
    for label in errored:
        assert label.verdict == "error"
        assert label.failure_detectors == []


def test_a_rotted_report_is_rejected_rather_than_relabelled(triage_reward, tmp_path):
    """The corpus-rot signal for this source is aorta's own consistency check.

    `SanitizerReport.from_dict` recomputes the overall verdict from the checks
    and raises when the stored value contradicts it, so a report whose verdict
    has been tampered with cannot be silently trained on.
    """
    import copy

    source = next(_SURVEY.rglob("sanitizer_report.json"))
    doc = json.loads(source.read_text(encoding="utf-8"))

    tampered = copy.deepcopy(doc)
    tampered["overall_verdict"] = "pass" if doc["overall_verdict"] != "pass" else "fail"
    with pytest.raises((ValueError, KeyError, TypeError)):
        triage_reward.label_sanitizer_report(tampered)

    # And the loader skips it instead of aborting the whole corpus.
    (tmp_path / "sanitizer_report.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    assert triage_reward.load_sanitizer_reports(tmp_path) == []


def test_the_untampered_report_still_labels(triage_reward):
    """Guards the test above: rejection must be caused by the tampering."""
    source = next(_SURVEY.rglob("sanitizer_report.json"))
    doc = json.loads(source.read_text(encoding="utf-8"))
    label = triage_reward.label_sanitizer_report(doc, source=str(source))
    assert label.verdict == doc["overall_verdict"]
    assert label.stale is False


def test_scoring_works_unchanged_across_both_label_sources(triage_reward):
    """One scorer, two label spaces.

    The sanitizer vocabulary includes `warn`, which the probe split has no
    equivalent for, so the two spaces are deliberately not mapped onto each
    other. `score_answer` is string equality plus set F1, so it spans both
    without needing to know which it is looking at.
    """
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    for _, label in labelled:
        oracle = triage_reward.Answer(label.verdict, sorted(label.cited_detectors))
        assert triage_reward.score_answer(oracle, label).reward == pytest.approx(1.0)

        wrong = triage_reward.Answer("pass" if label.verdict != "pass" else "fail", [])
        assert triage_reward.score_answer(wrong, label).reward < 1.0


# --------------------------------------------------------------------------- #
# build_corpus.py -- turning real sanitizer runs into scorable examples.
#
# The two properties worth pinning are the ones that would quietly ruin a
# corpus: counting lanes of one race as many examples, and losing the workload
# family that makes the corpus splittable.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def build_corpus():
    return _load("build_corpus")


def _build(build_corpus, tmp_path, results):
    out = tmp_path / "corpus"
    build_corpus.main([
        "--results", str(results),
        "--baselines", str(
            Path(__file__).resolve().parents[2]
            / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
        ),
        "--out", str(out),
    ])
    return out, json.loads((out / "manifest.json").read_text())


def test_the_committed_reports_build_into_a_corpus(build_corpus, tmp_path):
    _, manifest = _build(build_corpus, tmp_path, _SURVEY)
    assert manifest["scenarios"] == 6
    assert manifest["examples"]["triage"] == 6
    assert manifest["rejected_reports"] == 0


def test_every_example_carries_a_workload_family(build_corpus, tmp_path):
    """Stratification is free now and expensive to retrofit, so it is enforced."""
    out, manifest = _build(build_corpus, tmp_path, _SURVEY)
    assert "unknown" not in manifest["workload_families"]
    for name in ("triage.jsonl", "proposal.jsonl"):
        rows = [json.loads(x) for x in (out / name).read_text().splitlines() if x]
        assert rows
        for row in rows:
            assert row["workload_family"] != "unknown"


def test_lanes_of_one_race_collapse_to_one_site(build_corpus, tmp_path):
    """64 findings from one race are one piece of evidence, not 64.

    Built by replaying a real finding across lane masks, which is exactly the
    shape ConSan emits for a two-wave LDS race: same instruction pair, one
    record per lane.
    """
    source = json.loads(
        (_SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json")
        .read_text()
    )
    # Reduced to the one check carrying the lanes, so the counts below are the
    # dedup rule and nothing else.
    check = source["checks"][0]
    source["checks"] = [check]
    template = (check.get("findings") or [])[0]
    check["kernel_results"] = []
    lanes = []
    for index in range(64):
        lane = json.loads(json.dumps(template))
        lane["metadata"] = dict(lane.get("metadata") or {})
        lane["metadata"].update({
            "first_inst": "0x8", "second_inst": "0x28", "kind": "1",
            "first_lane_mask": hex(1 << (index % 32)),
            "first_lds": f"[{index * 4},{index * 4 + 4})",
        })
        lanes.append(lane)
    check["findings"] = lanes

    results = tmp_path / "results" / "lds_reduce_consan"
    results.mkdir(parents=True)
    (results / "sanitizer_report.json").write_text(json.dumps(source))

    _, manifest = _build(build_corpus, tmp_path, tmp_path / "results")
    assert manifest["findings"]["raw"] == 64
    assert manifest["findings"]["distinct_sites"] == 1


def test_the_corpus_scores_through_the_triage_scorer(
    build_corpus, triage_reward, tmp_path
):
    """The corpus is consumable with no conversion pass, and the oracle is perfect."""
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    rows = triage_reward.load_corpus(out / "triage.jsonl")
    assert len(rows) == 6
    for _, label, family in rows:
        assert family != "unknown"
        oracle = triage_reward.Answer(label.verdict, sorted(label.cited_detectors))
        assert triage_reward.score_answer(oracle, label).reward == pytest.approx(1.0)


def test_the_corpus_scores_through_the_proposal_scorer(
    build_corpus, proposal_reward, tmp_path
):
    """Every proposal variant lands on the tier it was synthesised to land on."""
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    rows = proposal_reward.load_corpus(out / "proposal.jsonl")
    assert rows
    by_variant = {}
    for proposal, _ in rows:
        by_variant.setdefault(proposal.name.rsplit(":", 1)[1], set()).add(
            proposal_reward.score_proposal(proposal).tier
        )
    assert by_variant["valid"] == {5}
    assert by_variant["hallucinated_name"] == {3}
    assert by_variant["invalid_category"] == {2}
    assert by_variant["already_tried"] == {4}


# --------------------------------------------------------------------------- #
# rescore_e2e: re-reading a recorded run under a changed grader
#
# The offline half of `run_e2e.py`. The property worth pinning hardest is the
# one that decides whether the acceptance test can be satisfied at all: a group
# whose completions are byte-identical cannot have within-group spread under
# *any* reward, so a zero there is a statement about the rollout and not about
# the grader. Conflating the two would have the reward blamed for a sampling
# defect, or a sampling fix credited to the reward.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rescore_e2e():
    return _load("rescore_e2e")


def _recorded(completions, candidates=None, tried=None):
    """A minimal `run_e2e.py` results document, one scenario per completion list."""
    candidates = candidates or ["hip_launch_blocking", "amd_log_level_4", "none"]
    tried = tried or []
    proposals = []
    for scenario, raws in completions.items():
        for index, raw in enumerate(raws):
            proposals.append({
                "scenario_id": scenario,
                "sample": index,
                "raw": raw,
                # The "before" reward, as the original run recorded it.
                "reward": 1.0,
                "tier": 5,
            })
    return {
        "meta": {
            "condition": "test",
            "model": "test-model",
            "candidates": candidates,
            "tried": tried,
        },
        "proposals": proposals,
    }


def _completion(category="unknown", mitigations=("amd_log_level_4",), hypothesis="h"):
    return json.dumps({
        "category": category,
        "hypothesis": hypothesis,
        "next_mitigations": list(mitigations),
        "confidence": 0.7,
        "stop": False,
    })


def test_rescoring_reads_the_raw_completion_not_the_stored_reward(rescore_e2e):
    """Otherwise the re-score would just echo the number it was meant to replace."""
    doc = _recorded({"s": [_completion()] * 3})
    rows = rescore_e2e.rescore_recorded(doc)
    assert len(rows) == 3
    for row in rows:
        assert row["reward_before"] == 1.0
        assert row["reward_after"] < 1.0


def test_identical_completions_cannot_produce_within_group_spread(rescore_e2e):
    """The reason acceptance criterion 3 cannot be met on the recorded rollouts.

    A reward is a function of the completion and the loop state, and the loop
    state is constant inside a group. So five copies of one completion earn five
    copies of one reward, and the within-group spread GRPO needs is exactly zero
    no matter what the grader does. This is the rollout's defect, not the
    reward's: `LiteLLMProposer.propose` sends no temperature.
    """
    doc = _recorded({"s": [_completion()] * 5})
    result = rescore_e2e.analyse(doc)
    group = result["per_scenario"]["s"]

    assert group["distinct_completions"] == 1
    assert group["spread_within_group"] == 0.0
    assert result["criteria"]["3_within_group_spread_nonzero"]["holds"] is False


def test_differing_completions_do_produce_within_group_spread(rescore_e2e):
    """The guard on the test above: the zero must come from the data, not the grader.

    Same grader, same scenario, same loop state -- only the completions differ,
    and the spread appears. So the reward is not what blocks criterion 3, and a
    rollout sampled at a non-zero temperature would produce a usable advantage.
    """
    doc = _recorded({
        "s": [
            _completion(category="rccl_hang", mitigations=("amd_log_level_4",)),
            _completion(category="unknown", mitigations=("amd_log_level_4",)),
            _completion(
                category="unknown",
                mitigations=("amd_log_level_4", "hip_launch_blocking"),
            ),
            "not JSON at all",
        ]
    })
    result = rescore_e2e.analyse(doc)
    group = result["per_scenario"]["s"]

    assert group["distinct_completions"] == 4
    assert group["spread_within_group"] > 0.0
    assert result["criteria"]["3_within_group_spread_nonzero"]["holds"] is True


def test_the_constant_templates_are_scored_on_the_models_own_loop_state(
    rescore_e2e, proposal_reward
):
    """A constant measured against a different candidate set proves nothing."""
    doc = _recorded({"s": [_completion()]})
    result = rescore_e2e.analyse(doc)
    references = result["references"]

    # Two offered names, so the shotgun template names both.
    assert references["abstain_and_shotgun"]["n_mitigations"] == 2
    assert references["abstain_and_pick_first"]["n_mitigations"] == 1
    # Every constant is on contract except the prose one, which is the point:
    # they fail on substance, not on format.
    assert references["always_prose"]["tier"] == 0
    for name in ("oracle_contract_perfect", "abstain_and_shotgun", "honest_abstainer"):
        assert references[name]["tier"] == proposal_reward.MAX_TIER


def test_the_contract_perfect_reference_still_tops_the_ladder(rescore_e2e):
    """Criterion 2. It commits to a category and spends one cell, so it is 1.0.

    Worth reading with `REFERENCE_CATEGORY`'s comment: this reference is perfect
    against *this reward*, and on the committed corpus its category is knowingly
    wrong, because the closed set has no name for a kernel-level data race.
    """
    doc = _recorded({"s": [_completion()]})
    result = rescore_e2e.analyse(doc)
    assert result["references"]["oracle_contract_perfect"]["reward"] == 1.0
    assert result["criteria"]["2_reference_at_top"]["holds"] is True


def test_an_abstaining_one_name_constant_is_the_hard_case_for_criterion_one(
    rescore_e2e,
):
    """The honest failure the re-score found, pinned so it cannot be lost.

    `abstain_and_pick_first` and `honest_abstainer` are the same policy up to
    the hypothesis text, and this reward cannot tell them apart -- it does not
    read the hypothesis. So the cheapest legal abstention scores what the best
    honest answer scores, and on a wide candidate set it beats a model that
    hedges. Fixing that needs a term the constant cannot satisfy.
    """
    doc = _recorded({"s": [_completion()]})
    result = rescore_e2e.analyse(doc)
    references = result["references"]
    assert (
        references["abstain_and_pick_first"]["reward"]
        == references["honest_abstainer"]["reward"]
    )


def test_a_verdict_outside_the_vocabulary_is_rejected(triage_reward, tmp_path):
    """A corpus written by a newer builder fails loudly, not as a silent mismatch."""
    corpus = tmp_path / "triage.jsonl"
    corpus.write_text(json.dumps({
        "kind": "triage", "example_id": "triage:x", "workload_family": "f",
        "label": {"verdict": "catastrophe", "failure_detectors": [],
                  "error_detectors": []},
    }) + "\n")
    with pytest.raises(ValueError, match="outside this scorer's vocabulary"):
        triage_reward.load_corpus(corpus)
