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
import shutil
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


# --------------------------------------------------------------------------- #
# run_e2e: the rollout's sampling parameters
#
# The temperature and the seed live in the rollout driver rather than in
# `LiteLLMProposer`, and that placement is the point: the proposer is the
# production path, and a diagnostic tool that returns a different answer each
# time it is asked is worse, not better. These pin the placement and the
# per-sample derivation, plus the two aggregate fields that tell a sampling
# defect apart from a saturated reward.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def run_e2e():
    return _load("run_e2e")


def test_the_proposer_itself_still_sends_no_temperature(run_e2e):
    """The load-bearing constraint: `aorta agent` must stay reproducible.

    Sampling diversity is a property of how training data is collected. If it
    leaks into the shipped proposer, every real triage becomes stochastic --
    the same evidence stops yielding the same recommendation, and the nightly
    `llm_determinism` entry starts measuring noise. Asserted against the
    proposer's source rather than its behaviour, so it holds without a server.
    """
    import inspect

    from aorta.agent.llm import LiteLLMProposer

    source = inspect.getsource(LiteLLMProposer)
    assert "temperature" not in source
    assert "seed" not in source


def test_each_sample_in_a_group_gets_its_own_seed(run_e2e):
    """A single seed per run would leave the group identical, which is the bug."""
    seeds = [run_e2e.sample_seed(7, "consan-racy", i) for i in range(5)]
    assert len(set(seeds)) == 5


def test_the_seed_is_reproducible_from_one_integer(run_e2e):
    """Diversity that nobody can re-derive is not evidence."""
    first = [run_e2e.sample_seed(7, "consan-racy", i) for i in range(5)]
    again = [run_e2e.sample_seed(7, "consan-racy", i) for i in range(5)]
    assert first == again
    # And a different base gives a different rollout.
    assert first != [run_e2e.sample_seed(8, "consan-racy", i) for i in range(5)]


def test_seeds_do_not_collide_across_scenarios(run_e2e):
    """Otherwise two groups would share draws and the corpus would be smaller."""
    seeds = [
        run_e2e.sample_seed(7, scenario, i)
        for scenario in ("consan-racy", "consan-clean", "waitcheck", "waitcheck-tiny")
        for i in range(5)
    ]
    assert len(set(seeds)) == len(seeds)
    # Engines reject out-of-range seeds, so stay inside int32.
    assert all(0 <= s < 2**31 for s in seeds)


def test_the_seed_is_not_derived_from_the_salted_builtin_hash(run_e2e):
    """`hash()` is salted per process, so a run would not replay tomorrow."""
    import subprocess
    import sys

    code = (
        f"import sys; sys.path.insert(0, {str(_EXAMPLES)!r});"
        "from run_e2e import sample_seed;"
        "print(sample_seed(7, 'consan-racy', 0))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, "seed changed across processes"


def _proposal_rows(raws):
    return [
        {
            "scenario_id": scenario,
            "sample": index,
            "raw": raw,
            "reward": 1.0,
            "tier": 5,
            "failure_kind": "on_contract",
            "consumer_outcome": "accepted",
            "category_claimed": "unknown",
            "mitigations_claimed": ["amd_log_level_4"],
            "offered": ["hip_launch_blocking", "amd_log_level_4"],
            "transport_error": "",
        }
        for scenario, group in raws.items()
        for index, raw in enumerate(group)
    ]


def test_a_collapsed_group_is_reported_apart_from_a_degenerate_one(run_e2e):
    """The distinction that decides whether the reward or the rollout is at fault.

    Identical completions give zero spread under any reward, so a group that
    collapsed to one completion is a sampling defect. Zero spread across
    several distinct completions is the reward saturating. The first end-to-end
    run could not tell these apart and attributed all nine groups to the
    reward; both were true, and only one was fixable by grading.
    """
    collapsed = run_e2e.aggregate(_proposal_rows({"s": ["same"] * 5}), [])["proposal"]
    assert collapsed["per_scenario"]["s"]["distinct_completions"] == 1
    assert collapsed["collapsed_groups"] == 1
    assert collapsed["degenerate_groups"] == 1

    # Distinct completions that happen to score the same: degenerate, but the
    # rollout did its job, so the reward is what needs work.
    rows = _proposal_rows({"s": [f"different-{i}" for i in range(5)]})
    saturated = run_e2e.aggregate(rows, [])["proposal"]
    assert saturated["per_scenario"]["s"]["distinct_completions"] == 5
    assert saturated["collapsed_groups"] == 0
    assert saturated["degenerate_groups"] == 1


def test_the_effective_n_does_not_count_a_completion_twice(run_e2e):
    """45 draws that collapse to 9 completions are 9 observations, not 45."""
    collapsed = run_e2e.aggregate(
        _proposal_rows({f"s{g}": ["same"] * 5 for g in range(9)}), []
    )["proposal"]
    assert collapsed["n"] == 45
    assert collapsed["effective_n"] == 9

    sampled = run_e2e.aggregate(
        _proposal_rows({f"s{g}": [f"r{g}-{i}" for i in range(5)] for g in range(9)}), []
    )["proposal"]
    assert sampled["n"] == 45
    assert sampled["effective_n"] == 45


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


# --------------------------------------------------------------------------- #
# The second artifact shape: probe `trial_*/result.json` cells.
#
# `build_corpus.py` used to find one filename, which meant only sanitizer runs
# could become scenarios at all. These pin the union: that both shapes are
# found, that a tree of either or both works, that the labels stay *derived*
# rather than read off the artifact, and -- hardest to notice if it broke --
# that the sanitizer path is untouched down to the prompt bytes.
# --------------------------------------------------------------------------- #


def _trial(
    cell: str,
    verdict: str,
    failures: list[str],
    errors: list[str],
    index: int = 0,
    exit_code: int = 0,
    warns: list[str] | None = None,
) -> dict:
    """One `result.json`, shaped like what `SubprocessWorkload` writes."""
    return {
        "verdict": verdict,
        "exit_code": exit_code,
        "walltime_sec": 12.5,
        "peak_vram_mib": 4096,
        "argv": ["bash", "run_repro.sh"],
        "cell_name": cell,
        "trial_index": index,
        "failure_detectors_fired": failures,
        "error_detectors_fired": errors,
        "warn_detectors_fired": warns or [],
        "capture": {},
        "tier_durations_ms": {"tier1": 3.0},
    }


def _write_cell(root: Path, cell: str, trials: list[dict]) -> Path:
    """Lay a probe cell out on disk the way the probe runner does."""
    cell_dir = root / cell
    for trial in trials:
        trial_dir = cell_dir / f"trial_{trial['trial_index']}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(json.dumps(trial), encoding="utf-8")
    return cell_dir


# A flaky reproducer plus its control: the shape the corpus has no scenario of
# today, and the reason this pipeline change exists.
_FLAKY = [
    _trial("none-none", "pass", [], [], 0),
    _trial("none-none", "fail", ["tier4:nan_signature"], [], 1, exit_code=1),
    _trial("none-none", "pass", [], [], 2),
    _trial("none-none", "error", [], ["tier1:timeout"], 3, exit_code=-9),
]
_CONTROL = [_trial("tf32_off-none", "pass", [], [], i) for i in range(4)]

_PROBE_FAMILIES = {
    "none-none": "synthetic_probe_nan",
    "tf32_off-none": "synthetic_probe_nan",
}


def _build_with_families(build_corpus, tmp_path, results, families=None):
    """`_build`, with the workload-family override a probe tree needs."""
    out = tmp_path / "corpus"
    argv = [
        "--results", str(results),
        "--baselines", str(
            Path(__file__).resolve().parents[2]
            / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
        ),
        "--out", str(out),
    ]
    if families is not None:
        path = tmp_path / "families.json"
        path.write_text(json.dumps(families), encoding="utf-8")
        argv += ["--families", str(path)]
    assert build_corpus.main(argv) == 0
    return out, json.loads((out / "manifest.json").read_text())


def _rows(out: Path, name: str = "triage.jsonl") -> list[dict]:
    return [json.loads(x) for x in (out / name).read_text().splitlines() if x]


# --- the cell is the unit, and its label is derived ------------------------- #


def test_one_cell_is_one_scenario_not_one_trial_each(build_corpus, tmp_path):
    """Four trials of one configuration are one scenario, as 64 lanes are one site."""
    results = tmp_path / "results"
    _write_cell(results, "none-none", _FLAKY)

    out, manifest = _build_with_families(
        build_corpus, tmp_path, results, _PROBE_FAMILIES
    )
    assert manifest["scenarios"] == 1
    assert manifest["probe_trials"] == 4
    assert manifest["artifacts"] == {"probe_result": 1}

    (row,) = _rows(out)
    assert row["trial_counts"] == {
        "total": 4, "pass": 2, "fail": 1, "error": 1, "distinct_detectors": 2,
    }


def test_one_reproducing_trial_makes_the_cell_a_reproduction(build_corpus, tmp_path):
    """fail > error > pass, applied across trials rather than within one.

    Two of four trials were clean and one was infra noise. A cell that answered
    `pass` here would train the model to call a flaky reproducer clean, which is
    the single most expensive wrong answer in this domain.
    """
    results = tmp_path / "results"
    _write_cell(results, "none-none", _FLAKY)
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    (row,) = _rows(out)
    assert row["label"]["verdict"] == "fail"
    assert row["label"]["failure_detectors"] == ["tier4:nan_signature"]
    assert row["label"]["error_detectors"] == ["tier1:timeout"]


def test_an_all_clean_cell_is_a_pass_with_nothing_cited(build_corpus, tmp_path):
    results = tmp_path / "results"
    _write_cell(results, "tf32_off-none", _CONTROL)
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    (row,) = _rows(out)
    assert row["label"]["verdict"] == "pass"
    assert row["distinct_evidence"] == []


def test_an_infra_only_cell_is_an_error_not_a_failure(build_corpus, tmp_path):
    """No valid observation was made, so there is nothing to attribute a failure to."""
    results = tmp_path / "results"
    _write_cell(results, "none-none", [
        _trial("none-none", "error", [], ["tier1:timeout"], 0, exit_code=-9),
        _trial("none-none", "error", [], ["tier1:exec_failed"], 1, exit_code=127),
    ])
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    (row,) = _rows(out)
    assert row["label"]["verdict"] == "error"
    assert row["label"]["failure_detectors"] == []
    assert {e["kind"] for e in row["distinct_evidence"]} == {"error"}


def test_the_cell_label_is_recomputed_not_read_off_the_artifact(
    build_corpus, tmp_path
):
    """The corpus-rot signal for this shape.

    Every trial here *stores* `pass` while recording a detector that means
    otherwise. A pipeline that trusted the stored field would emit a clean
    scenario carrying failure evidence -- a label that is wrong in the one
    direction that trains the model to ignore its own evidence.
    """
    results = tmp_path / "results"
    _write_cell(results, "none-none", [
        _trial("none-none", "pass", ["tier4:nan_signature"], [], 0),
        _trial("none-none", "pass", ["tier4:nan_signature"], [], 1),
    ])
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    (row,) = _rows(out)
    assert row["label"]["verdict"] == "fail", "the stored 'pass' was believed"
    assert row["label"]["stored_verdict"] == "pass"
    assert row["label"]["stale"] is True
    assert row["ground_truth"]["observed_verdict"] == "fail"


def test_a_cell_detector_on_the_wrong_side_is_re_partitioned(
    build_corpus, tmp_path
):
    """`partition_detectors` owns the fail/error line, not the artifact.

    The per-trial version of this is asserted further up against `label_run`;
    this is the cell-level one, through the whole builder.
    """
    results = tmp_path / "results"
    _write_cell(results, "none-none", [
        # `tier1:timeout` filed as a failure and `tier4:nan_signature` as an
        # error -- both on the wrong side of the line.
        _trial("none-none", "fail", ["tier1:timeout"], ["tier4:nan_signature"], 0),
    ])
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    (row,) = _rows(out)
    assert row["label"]["failure_detectors"] == ["tier4:nan_signature"]
    assert row["label"]["error_detectors"] == ["tier1:timeout"]


def test_one_detector_firing_in_many_trials_is_one_piece_of_evidence(
    build_corpus, tmp_path
):
    """The probe analogue of collapsing 64 lanes of one race to one site."""
    results = tmp_path / "results"
    _write_cell(results, "none-none", [
        _trial("none-none", "fail", ["tier4:nan_signature"], [], i, exit_code=1)
        for i in range(8)
    ])
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    (row,) = _rows(out)
    assert row["trial_counts"]["total"] == 8
    assert len(row["distinct_evidence"]) == 1
    assert row["distinct_evidence"][0]["trials"] == list(range(8))


def test_the_per_trial_split_survives_into_the_row(build_corpus, tmp_path):
    """A 1-in-4 reproducer and a 4-in-4 one are not the same scenario.

    Both carry verdict `fail`, so the distinction exists only in the per-trial
    record. Reducing it away is what would make a flaky scenario indistinguish-
    able from a deterministic one.
    """
    results = tmp_path / "results"
    _write_cell(results / "flaky", "none-none", _FLAKY)
    _write_cell(results / "solid", "none-none", [
        _trial("none-none", "fail", ["tier4:nan_signature"], [], i, exit_code=1)
        for i in range(4)
    ])
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    rows = _rows(out)
    assert len(rows) == 2
    verdicts = {r["label"]["verdict"] for r in rows}
    assert verdicts == {"fail"}, "both are reproductions; only the rate differs"
    assert sorted(r["trial_counts"]["fail"] for r in rows) == [1, 4]


# --- trees: either shape, both, neither ------------------------------------ #


def test_a_tree_of_both_shapes_yields_both(build_corpus, tmp_path):
    """The union case, and the one that would regress silently.

    The sanitizer reports are the committed ones, so this also checks the two
    collectors do not interfere: six reports plus two cells, each labelled by
    its own resolver.
    """
    results = tmp_path / "results"
    shutil.copytree(_SURVEY / "reports", results / "reports")
    _write_cell(results / "cells", "none-none", _FLAKY)
    _write_cell(results / "cells", "tf32_off-none", _CONTROL)

    out, manifest = _build_with_families(
        build_corpus, tmp_path, results, _PROBE_FAMILIES
    )
    assert manifest["scenarios"] == 8
    assert manifest["artifacts"] == {"sanitizer_report": 6, "probe_result": 2}
    assert manifest["probe_trials"] == 8
    # The sanitizer evidence total is unaffected by the probe rows sharing the
    # tree, and vice versa.
    assert manifest["findings"]["raw"] == 32
    assert "unknown" not in manifest["workload_families"]

    rows = _rows(out)
    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind.setdefault(row.get("artifact", "sanitizer_report"), 0)
        by_kind[row.get("artifact", "sanitizer_report")] += 1
    assert by_kind == {"sanitizer_report": 6, "probe_result": 2}


def test_sanitizer_rows_in_a_mixed_tree_carry_no_artifact_key(
    build_corpus, tmp_path
):
    """Absence is the discriminator, and it is load-bearing.

    Stamping `artifact` onto sanitizer rows would be tidier and would change
    their bytes, which is what makes every reward number recorded against the
    committed corpus incomparable.
    """
    results = tmp_path / "results"
    shutil.copytree(_SURVEY / "reports", results / "reports")
    _write_cell(results / "cells", "none-none", _FLAKY)
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    for row in _rows(out):
        if row["scenario_id"] == "none-none":
            assert row["artifact"] == "probe_result"
        else:
            assert "artifact" not in row


def test_a_tree_with_neither_shape_fails_loudly(build_corpus, tmp_path):
    """An empty corpus is a build failure, not a zero-row success."""
    empty = tmp_path / "results"
    (empty / "nothing" / "here").mkdir(parents=True)
    (empty / "nothing" / "here" / "notes.txt").write_text("no artifacts", "utf-8")
    assert build_corpus.main([
        "--results", str(empty),
        "--baselines", str(
            Path(__file__).resolve().parents[2]
            / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
        ),
        "--out", str(tmp_path / "corpus"),
    ]) == 1


def test_a_result_json_outside_a_trial_dir_is_not_mistaken_for_a_cell(
    build_corpus, triage_reward, tmp_path
):
    """Only `trial_<N>/result.json` is the probe artifact.

    A stray `result.json` is some other tool's output. Inventing a cell around
    it would put a run into the corpus that no resolver ever labelled.
    """
    results = tmp_path / "results"
    results.mkdir()
    (results / "result.json").write_text(json.dumps(_trial("x", "fail", [], [])))
    assert triage_reward.find_probe_cells(results) == []


# --- malformed artifacts, one of each shape -------------------------------- #


def test_a_malformed_probe_trial_is_rejected_rather_than_guessed(
    build_corpus, triage_reward, tmp_path
):
    """Valid JSON of the wrong type survives the loader, so the seam must refuse it."""
    results = tmp_path / "results"
    cell = _write_cell(results, "none-none", _FLAKY)
    (cell / "trial_1" / "result.json").write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        triage_reward.label_trials([{"verdict": "pass"}, [1, 2, 3]])

    # And the build drops that scenario, naming it, rather than aborting.
    out, manifest = _build_with_families(
        build_corpus, tmp_path, results, _PROBE_FAMILIES
    )
    assert manifest["scenarios"] == 0
    assert manifest["rejected_reports"] == 1
    assert _rows(out) == []


def test_an_unreadable_probe_trial_is_skipped_and_the_cell_still_labels(
    build_corpus, tmp_path
):
    """Truncated JSON is the interrupted-write case, not a corpus-rot case.

    `read_trial_results` already skips it, so the cell is labelled from the
    trials that do parse and the trial count says how many that was.
    """
    results = tmp_path / "results"
    cell = _write_cell(results, "none-none", _FLAKY)
    (cell / "trial_2" / "result.json").write_text('{"verdict": ', encoding="utf-8")

    out, manifest = _build_with_families(
        build_corpus, tmp_path, results, _PROBE_FAMILIES
    )
    assert manifest["scenarios"] == 1
    (row,) = _rows(out)
    assert row["trial_counts"]["total"] == 3
    assert row["label"]["verdict"] == "fail"


def test_a_malformed_sanitizer_report_is_still_rejected(build_corpus, tmp_path):
    """The sanitizer shape's own rot check, unchanged by the union.

    A report whose stored verdict contradicts its checks fails
    `SanitizerReport.from_dict`, and a probe cell sharing the tree neither
    rescues it nor is dragged down with it.
    """
    import copy

    results = tmp_path / "results"
    case = results / "reports" / "gemm_f32_waitcheck"
    case.mkdir(parents=True)
    doc = json.loads(
        (_SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json")
        .read_text()
    )
    tampered = copy.deepcopy(doc)
    tampered["overall_verdict"] = "pass"
    (case / "sanitizer_report.json").write_text(json.dumps(tampered), encoding="utf-8")
    _write_cell(results / "cells", "none-none", _FLAKY)

    out, manifest = _build_with_families(
        build_corpus, tmp_path, results, _PROBE_FAMILIES
    )
    assert manifest["rejected_reports"] == 1
    assert manifest["artifacts"] == {"probe_result": 1}
    assert [r["scenario_id"] for r in _rows(out)] == ["none-none"]


def test_an_unreadable_sanitizer_report_is_skipped(build_corpus, tmp_path):
    results = tmp_path / "results"
    case = results / "reports" / "gemm_f32_waitcheck"
    case.mkdir(parents=True)
    (case / "sanitizer_report.json").write_text("{not json", encoding="utf-8")
    _write_cell(results / "cells", "tf32_off-none", _CONTROL)

    _, manifest = _build_with_families(
        build_corpus, tmp_path, results, _PROBE_FAMILIES
    )
    assert manifest["artifacts"] == {"probe_result": 1}
    assert manifest["rejected_reports"] == 0, "unreadable is skipped, not rejected"


# --- the corpus stays consumable by both scorers --------------------------- #


def test_probe_rows_score_through_the_triage_scorer_unchanged(
    build_corpus, triage_reward, tmp_path
):
    """No conversion pass, and the oracle is perfect on both shapes."""
    results = tmp_path / "results"
    shutil.copytree(_SURVEY / "reports", results / "reports")
    _write_cell(results / "cells", "none-none", _FLAKY)
    _write_cell(results / "cells", "tf32_off-none", _CONTROL)
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    rows = triage_reward.load_corpus(out / "triage.jsonl")
    assert len(rows) == 8
    for _, label, family in rows:
        assert family != "unknown"
        oracle = triage_reward.Answer(label.verdict, sorted(label.cited_detectors))
        assert triage_reward.score_answer(oracle, label).reward == pytest.approx(1.0)


def test_probe_rows_get_the_same_proposal_ladder(
    build_corpus, proposal_reward, tmp_path
):
    """The proposal contract is a property of the model's output, not the run.

    So the ladder must land identically for a probe scenario -- if it did not,
    the tier would be reporting the artifact shape rather than the proposal.
    """
    results = tmp_path / "results"
    _write_cell(results, "none-none", _FLAKY)
    out, _ = _build_with_families(build_corpus, tmp_path, results, _PROBE_FAMILIES)

    rows = proposal_reward.load_corpus(out / "proposal.jsonl")
    assert len(rows) == len(build_corpus.PROPOSAL_VARIANTS)
    tiers = {
        proposal.name.rsplit(":", 1)[1]: proposal_reward.score_proposal(proposal).tier
        for proposal, _ in rows
    }
    assert tiers["valid"] == 5
    assert tiers["hallucinated_name"] == 3
    assert tiers["invalid_category"] == 2
    assert tiers["already_tried"] == 4


def test_a_probe_cell_without_a_family_override_is_honestly_unknown(
    build_corpus, tmp_path
):
    """A cell name encodes the mitigation axis, not the workload.

    So there is nothing to infer a family from, and guessing one would make the
    corpus look stratifiable when it is not.
    """
    results = tmp_path / "results"
    _write_cell(results, "none-none", _FLAKY)
    _, manifest = _build_with_families(build_corpus, tmp_path, results, None)
    assert manifest["workload_families"] == {"unknown": 1}


# --- the frozen prompt ----------------------------------------------------- #


def test_the_sanitizer_symptom_line_is_frozen(run_e2e):
    """The regression that would invalidate every reward number we hold.

    Nothing would fail if this text were reworded -- no scorer reads it -- so
    the only protection is an exact-match assertion. Pinned against a real
    committed corpus row rather than a synthetic one, so a change to the row
    schema that altered the rendering also trips this.
    """
    row = json.loads(
        json.dumps({
            "scenario_id": "waitcheck",
            "workload_family": "tensile_gemm_object",
            "label": {
                "verdict": "warn",
                "failure_detectors": ["waitcheck:wait_hazard"],
                "error_detectors": [],
            },
            "checks": [{
                "sanitizer": "waitcheck", "verdict": "warn",
                "findings": 64, "kernel_names": ["Cijk_Ailk_Bljk"],
            }],
            "distinct_evidence": [],
            "finding_counts": {"raw": 64, "distinct_sites": 2},
        })
    )
    symptom, summaries = run_e2e.loop_state(row)
    assert symptom == (
        "Sanitizer run 'waitcheck' on gfx950 returned overall verdict 'warn'. "
        "Per-sanitizer: waitcheck=warn (64 findings). "
        "Workload family: tensile_gemm_object."
    )
    assert [s["cell_name"] for s in summaries] == ["none-none"]
    assert summaries[0]["kernel_names"] == ["Cijk_Ailk_Bljk"]


def test_real_built_rows_still_render_the_frozen_line(
    build_corpus, run_e2e, tmp_path
):
    """Belt and braces: rows from the builder, not hand-built ones.

    `examples/rl/corpus/*.jsonl` is generated and gitignored, so it cannot be
    the fixture here -- it is absent on a clean checkout. The committed survey
    reports are the durable input, and building from them exercises the whole
    path, so a change to the row schema that altered the rendering trips this
    even if the synthetic row above still matches.
    """
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    rows = _rows(out)
    assert len(rows) == 6
    for row in rows:
        symptom, _ = run_e2e.loop_state(row)
        assert symptom.startswith("Sanitizer run '")
        assert " on gfx950 returned overall verdict " in symptom
        assert symptom.endswith(f"Workload family: {row['workload_family']}.")


def test_a_probe_row_gets_its_own_symptom_line(run_e2e):
    """Additive, and it names no hardware the row cannot vouch for."""
    row = {
        "artifact": "probe_result",
        "scenario_id": "none-none",
        "workload_family": "synthetic_probe_nan",
        "label": {
            "verdict": "fail",
            "failure_detectors": ["tier4:nan_signature"],
            "error_detectors": ["tier1:timeout"],
        },
        "trials": _FLAKY,
        "trial_counts": {
            "total": 4, "pass": 2, "fail": 1, "error": 1, "distinct_detectors": 2,
        },
        "distinct_evidence": [
            {"detector": "tier4:nan_signature", "kind": "failure", "trials": [1]},
            {"detector": "tier1:timeout", "kind": "error", "trials": [3]},
        ],
    }
    symptom, summaries = run_e2e.loop_state(row)
    assert symptom == (
        "Probe cell 'none-none' returned overall verdict 'fail' over 4 trial(s). "
        "Per-trial: pass=2, fail=1, error=1. "
        "Detectors fired: tier1:timeout, tier4:nan_signature. "
        "Workload family: synthetic_probe_nan."
    )
    assert "gfx950" not in symptom
    assert "Sanitizer run" not in symptom
    (summary,) = summaries
    assert summary["failure_detectors_fired"] == ["tier4:nan_signature"]
    assert summary["error_detectors_fired"] == ["tier1:timeout"]
    assert summary["trial_counts"]["fail"] == 1


def test_the_artifact_discriminator_defaults_to_the_sanitizer_shape(triage_reward):
    """An older row, with no `artifact` key, must keep reading as a sanitizer row."""
    assert triage_reward.artifact_kind({}) == "sanitizer_report"
    assert triage_reward.artifact_kind({"artifact": None}) == "sanitizer_report"
    assert triage_reward.artifact_kind({"artifact": ""}) == "sanitizer_report"
    assert triage_reward.artifact_kind({"artifact": "probe_result"}) == "probe_result"


# --------------------------------------------------------------------------- #
# The fix half: did the proposed mitigation actually resolve the reproducer?
#
# The first reward term here that is checked by *execution* rather than by
# inspection, so what these pin down is different in kind from the tiers above.
# Two things matter and neither is a score. First, that the ground truth is
# *recovered* from the archived matrix rather than asserted -- through the same
# verdict resolver the triage label uses and the same `winning_mitigation`
# attribution rule the agent loop uses -- because a fabricated ground truth
# still trains. Second, that the term refuses to score a scenario where it
# cannot discriminate, rather than paying everyone the same constant, which is
# the degeneracy this whole term exists to escape.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fix_reward():
    return _load("fix_reward")


def _matrix(tmp_path: Path, cells: dict[str, list[dict]]) -> Path:
    """An archived probe run: one directory per `{mitigation}-{diagnostic}` cell."""
    root = tmp_path / "run"
    for cell, trials in cells.items():
        _write_cell(root, cell, trials)
    return root


def _passing(cell: str, n: int = 2) -> list[dict]:
    return [_trial(cell, "pass", [], [], i) for i in range(n)]


def _failing(cell: str, n: int = 2) -> list[dict]:
    return [
        _trial(cell, "fail", ["tier1:exit_nonzero"], [], i, exit_code=1)
        for i in range(n)
    ]


def test_the_resolver_is_recovered_from_the_archived_matrix(fix_reward, tmp_path):
    """The whole cost argument: ground truth is a directory listing, not a GPU run."""
    root = _matrix(tmp_path, {
        "none-none": _failing("none-none"),
        "tf32_off-none": _passing("tf32_off-none"),
    })
    resolution = fix_reward.resolution_from_matrix(root)

    assert resolution.resolvers == frozenset({"tf32_off"})
    assert resolution.baseline_failed is True
    assert resolution.scoreable is True
    assert resolution.cells == 2


def test_every_mitigation_that_resolved_it_counts_not_just_the_first(
    fix_reward, tmp_path
):
    """`docs/tokenspeed-rl-post-training.md` 4.3: a matrix can have several right
    answers, and a policy naming the second one is not wrong."""
    root = _matrix(tmp_path, {
        "none-none": _failing("none-none"),
        "tf32_off-none": _passing("tf32_off-none"),
        "xnack-none": _passing("xnack-none"),
        "hsa_no_sdma-none": _failing("hsa_no_sdma-none"),
    })
    resolution = fix_reward.resolution_from_matrix(root)

    assert resolution.resolvers == frozenset({"tf32_off", "xnack"})
    assert fix_reward.fix_credit(["xnack"], resolution) == 1.0
    assert fix_reward.fix_credit(["hsa_no_sdma"], resolution) == 0.0


def test_a_pass_that_is_not_attributable_to_the_mitigation_is_not_a_resolver(
    fix_reward, tmp_path
):
    """A diagnostic-only cell and a mitigation+diagnostic cell both passed.

    Neither pass is attributable to the mitigation alone, which is exactly what
    `winning_mitigation` refuses, and the reason this reads through it rather
    than scanning for any passing cell.
    """
    root = _matrix(tmp_path, {
        "none-none": _failing("none-none"),
        "none-xnack": _passing("none-xnack"),
        "tf32_off-xnack": _passing("tf32_off-xnack"),
    })
    resolution = fix_reward.resolution_from_matrix(root)

    assert resolution.resolvers == frozenset()
    assert resolution.scoreable is False


def test_the_baseline_passing_means_there_was_nothing_to_resolve(fix_reward, tmp_path):
    root = _matrix(tmp_path, {
        "none-none": _passing("none-none"),
        "tf32_off-none": _passing("tf32_off-none"),
    })
    resolution = fix_reward.resolution_from_matrix(root)

    assert resolution.baseline_failed is False
    assert resolution.scoreable is False
    assert "nothing to resolve" in resolution.withheld_because


def test_a_matrix_nothing_resolved_is_withheld_rather_than_scored_zero(
    fix_reward, tmp_path
):
    """The trap this term exists to avoid, reappearing one level up.

    If no mitigation resolved the failure, every policy earns 0.0 and the term
    is a constant added to every reward -- no gradient, and a constant that
    looks like a measurement. `scoreable` is how a caller is told to withhold it.
    """
    root = _matrix(tmp_path, {
        "none-none": _failing("none-none"),
        "tf32_off-none": _failing("tf32_off-none"),
    })
    resolution = fix_reward.resolution_from_matrix(root)

    assert resolution.resolvers == frozenset()
    assert resolution.scoreable is False
    assert "no mitigation" in resolution.withheld_because


def test_the_verdict_is_recomputed_not_read_off_the_cell(fix_reward, tmp_path):
    """An artifact claiming `pass` while its detectors say otherwise is not a win.

    Same seam as the triage label: the stored verdict is ignored and the
    detector IDs are re-resolved, so corpus rot cannot promote a failing cell
    into a resolving one.
    """
    lying = [
        _trial("tf32_off-none", "pass", ["tier1:exit_nonzero"], [], 0, exit_code=1),
    ]
    root = _matrix(tmp_path, {
        "none-none": _failing("none-none"),
        "tf32_off-none": lying,
    })

    assert fix_reward.resolution_from_matrix(root).resolvers == frozenset()


def test_none_is_never_a_resolving_mitigation(fix_reward, tmp_path):
    """`none-none` passing is a baseline that did not reproduce, not a fix."""
    root = _matrix(tmp_path, {"none-none": _passing("none-none")})

    assert fix_reward.resolution_from_matrix(root).resolvers == frozenset()


def test_fix_credit_is_membership_and_an_empty_proposal_earns_nothing(
    fix_reward, tmp_path
):
    root = _matrix(tmp_path, {
        "none-none": _failing("none-none"),
        "tf32_off-none": _passing("tf32_off-none"),
    })
    resolution = fix_reward.resolution_from_matrix(root)

    assert fix_reward.fix_credit([], resolution) == 0.0
    assert fix_reward.fix_credit(["xnack"], resolution) == 0.0
    assert fix_reward.fix_credit(["xnack", "tf32_off"], resolution) == 1.0


def test_the_composite_has_exactly_one_weight_and_it_is_bounded(fix_reward):
    assert fix_reward.composite_reward(1.0, 0.0, 0.5) == 0.5
    assert fix_reward.composite_reward(0.0, 1.0, 0.5) == 0.5
    assert fix_reward.composite_reward(0.8, 1.0, 0.0) == pytest.approx(0.8)
    assert fix_reward.composite_reward(0.8, 1.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fix_reward.composite_reward(1.0, 1.0, 1.5)


# --- what the fix half scores is what the loop would actually run ----------- #


def test_only_names_the_loop_would_run_can_earn_fix_credit(rescore_e2e):
    """A hallucinated or unoffered name never becomes a probe cell.

    `LiteLLMProposer.propose` filters it out before the loop sees it, so it
    cannot resolve anything. Replaying that filter here is what makes the fix
    half score the search rather than the sentence.
    """
    offered = ["tf32_off", "xnack"]
    raw = json.dumps({
        "category": "unknown",
        "hypothesis": "",
        "next_mitigations": ["tf32_off", "rccl_p2p_disable", "hsa_no_sdma"],
        "confidence": 0.5,
        "stop": False,
    })
    assert rescore_e2e.runnable_names(raw, offered) == ["tf32_off"]


def test_an_unparseable_completion_runs_nothing(rescore_e2e):
    assert rescore_e2e.runnable_names("not JSON at all", ["tf32_off"]) == []
    assert rescore_e2e.runnable_names('["tf32_off"]', ["tf32_off"]) == []
    assert rescore_e2e.runnable_names('{"next_mitigations": "tf32_off"}', ["tf32_off"]) == []


# --- the fix half must not disturb the form-only numbers -------------------- #


def _one_scenario_each(raws: list[str], offered: list[str]) -> dict:
    """`_recorded`, with one scenario per completion and an explicit offered set."""
    return _recorded(
        {f"s{i}": [raw] for i, raw in enumerate(raws)},
        candidates=[*offered, "none"],
    )


def test_without_a_resolution_the_reward_is_exactly_the_form_reward(rescore_e2e):
    """The regression guard on every recorded number: the fix half is opt-in.

    Every reward measured before this term existed was a form score, and they
    stay comparable only if omitting the resolution changes nothing.
    """
    doc = _one_scenario_each([_completion(mitigations=["tf32_off"])], ["tf32_off", "xnack"])
    result = rescore_e2e.analyse(doc)

    assert result["fix_half"]["active"] is False
    assert result["model_mean_after"] == result["fix_half"]["model_form_mean"]
    assert result["references"]["abstain_and_pick_first"]["reward"] == 0.9


def test_the_composite_moves_the_model_by_the_weighted_fix_credit(rescore_e2e):
    doc = _one_scenario_each([_completion(mitigations=["tf32_off"])], ["tf32_off", "xnack"])
    resolution = rescore_e2e.hypothetical_resolution(["tf32_off"])
    result = rescore_e2e.analyse(doc, resolution, fix_weight=0.5)

    assert result["fix_half"]["active"] is True
    assert result["fix_half"]["model_fix_rate"] == 1.0
    # form 0.9 (abstains, one name), fix 1.0, equal weights.
    assert result["model_mean_after"] == pytest.approx(0.95)


def test_a_withheld_resolution_leaves_the_form_reward_alone(rescore_e2e):
    """`scoreable` is honoured by the caller, not just reported by the callee."""
    doc = _one_scenario_each([_completion(mitigations=["tf32_off"])], ["tf32_off", "xnack"])
    empty = rescore_e2e.hypothetical_resolution([])
    result = rescore_e2e.analyse(doc, empty)

    assert empty.scoreable is False
    assert result["fix_half"]["active"] is False
    assert result["model_mean_after"] == pytest.approx(0.9)


def test_the_oracle_names_a_resolver_once_correctness_is_checked(rescore_e2e):
    """Under a form-only reward the oracle could name anything available.

    Once the reward checks whether the mitigation worked, an arbitrary name is
    a coin flip, and criterion 2 would be measuring whether `offered[0]`
    happened to be right rather than whether the ceiling is reachable.
    """
    offered = ["hip_launch_blocking", "tf32_off"]
    form_only = rescore_e2e.reference_policies(offered)
    assert json.loads(form_only["oracle_contract_perfect"])["next_mitigations"] == [
        "hip_launch_blocking"
    ]

    aware = rescore_e2e.reference_policies(
        offered, rescore_e2e.hypothetical_resolution(["tf32_off"])
    )
    assert json.loads(aware["oracle_contract_perfect"])["next_mitigations"] == ["tf32_off"]
    # The constants are constants: they must not learn the answer too.
    for name in ("abstain_and_pick_first", "abstain_and_shotgun", "honest_abstainer"):
        assert aware[name] == form_only[name]


# --- the sweep, which is how a term with no ground truth is reported -------- #


def test_the_sweep_scores_every_hypothesis_and_selects_none(rescore_e2e):
    offered = ["tf32_off", "xnack"]
    doc = _one_scenario_each(
        [_completion(mitigations=["tf32_off"]), _completion(mitigations=["xnack"])],
        offered,
    )
    sweep = rescore_e2e.resolver_sweep(doc)

    labels = [row["hypothesis"] for row in sweep["hypotheses"]]
    assert labels == [
        "(none: form half only)",
        "nothing resolves it",
        "tf32_off resolves it",
        "xnack resolves it",
    ]
    # Half the samples name each, so each hypothesis pays half of them.
    by_label = {row["hypothesis"]: row for row in sweep["hypotheses"]}
    assert by_label["tf32_off resolves it"]["model_fix_rate"] == 0.5
    assert by_label["xnack resolves it"]["model_fix_rate"] == 0.5


def test_the_null_hypothesis_is_reported_rather_than_dropped(rescore_e2e):
    """"Nothing resolves it" is live on this corpus and has to be visible.

    The offered names there are diagnostic and serialisation toggles that
    cannot repair a source-level race, so the honest sweep has to include the
    case where the term simply has no right answer to pay for.
    """
    doc = _one_scenario_each([_completion(mitigations=["tf32_off"])], ["tf32_off"])
    sweep = rescore_e2e.resolver_sweep(doc)

    null = next(r for r in sweep["hypotheses"] if r["hypothesis"] == "nothing resolves it")
    assert null["criterion_1"] is False
    assert "withheld" in null["note"]


def test_a_shotgun_earns_full_fix_credit_under_every_hypothesis(rescore_e2e):
    """The structural reason the fix half did not break the degeneracy.

    Fix credit is containment, and containment is monotone in list length: a
    policy that names the whole offered set names every possible resolver by
    construction. Pricing that is the form half's job, via `precision_credit`,
    and on the recorded rollouts it did not price it enough. Pinned here so the
    limitation is a property of the design someone can find, rather than a
    surprise in a later measurement.
    """
    offered = ["tf32_off", "xnack", "hsa_no_sdma"]
    doc = _one_scenario_each([_completion(mitigations=offered)], offered)
    sweep = rescore_e2e.resolver_sweep(doc)

    for row in sweep["hypotheses"]:
        if row.get("model_fix_rate") is not None:
            assert row["model_fix_rate"] == 1.0
