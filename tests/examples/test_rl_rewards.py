# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the RL reward seams under ``examples/rl``.

These are examples rather than package modules, so they are loaded by path.
They are still worth testing: both are scorers, and a scorer that silently stops
detecting what it claims to detect is the failure mode the whole design is
guarding against. The novelty gate and the degenerate-policy floor are the two
things most likely to rot unnoticed, so they are what these pin down.
"""

from __future__ import annotations

import importlib.util
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
