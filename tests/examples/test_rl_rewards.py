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
    tiers = [
        proposal_reward.score_proposal(f).tier for f in proposal_reward.FIXTURES
    ]
    assert min(tiers) == 0
    assert max(tiers) == proposal_reward.MAX_TIER
    for f in proposal_reward.FIXTURES:
        score = proposal_reward.score_proposal(f)
        assert score.reward == pytest.approx(score.tier / proposal_reward.MAX_TIER)


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
