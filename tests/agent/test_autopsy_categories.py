"""The closed autopsy label set, and everything that has to agree with it.

A closed set is referenced in more places than it looks: the validator that
rejects anything outside it, the prompt that tells the model what exists, the
offline heuristic that labels without a model, the operator docs, and the
corpus ground truth. A member added in one place and missing in another is
worse than not adding it at all -- either the validator accepts a label the
model was never told about, or the model emits one the validator kills the
search over.

So these pin the *agreement between* those places rather than a hardcoded list
of names, and keep working as the set grows. The exceptions are the three
labels added because the measured end-to-end run had nowhere to put its
evidence; those are named explicitly, because losing one silently is the
regression this file exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.agent.llm import (
    AUTOPSY_CATEGORIES,
    AUTOPSY_CATEGORY_GUIDANCE,
    AgentStep,
    FakeLLMProposer,
    _build_prompt,
    format_category_guidance,
)
from aorta.agent.policy import AgentPolicy, PolicyViolation

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DOC = REPO_ROOT / "docs" / "agent" / "aorta-probe-agent.md"
SCENARIO_LABELS = REPO_ROOT / "examples" / "rl" / "corpus" / "scenario_labels.json"

#: Added because an end-to-end run against nine real sanitizer scenarios
#: answered ``unknown`` on eight of them and mislabelled the ninth: no correct
#: label existed. Named here rather than derived so that dropping one fails.
ADDED_CATEGORIES = ("kernel_race", "nondeterminism", "numeric_instability")


class TestTheSetItself:
    def test_the_set_is_derived_from_the_guidance_so_the_two_cannot_drift(self):
        assert AUTOPSY_CATEGORIES == frozenset(AUTOPSY_CATEGORY_GUIDANCE)

    def test_every_label_carries_a_gloss(self):
        """A label the model is shown without a gloss is a label it cannot route on."""
        assert [n for n, gloss in AUTOPSY_CATEGORY_GUIDANCE.items() if not gloss.strip()] == []

    @pytest.mark.parametrize("category", ADDED_CATEGORIES)
    def test_the_labels_the_evidence_demanded_are_present(self, category):
        assert category in AUTOPSY_CATEGORIES

    def test_kernel_race_did_not_replace_checkpoint_race(self):
        """They are different failures with different diagnostics.

        A checkpoint race is a concurrency problem around checkpoint I/O; an
        intra-wave LDS conflict is not. Collapsing them back together would
        re-create the mislabel that motivated the split.
        """
        assert {"kernel_race", "checkpoint_race"} <= AUTOPSY_CATEGORIES

    def test_the_guidance_is_read_only(self):
        with pytest.raises(TypeError):
            AUTOPSY_CATEGORY_GUIDANCE["invented"] = "nope"  # type: ignore[index]


class TestTheValidator:
    @pytest.mark.parametrize("category", sorted(AUTOPSY_CATEGORIES))
    def test_every_member_survives_validate_step(self, category):
        step = AgentStep(
            category=category,
            hypothesis="",
            next_mitigations=["tf32_off"],
            confidence=0.5,
            stop=False,
        )
        assert AgentPolicy().validate_step(step).category == category

    def test_unknown_stays_valid(self):
        """The honest answer when the evidence supports no label.

        Widening the set does not make declining wrong, and `validate_step`
        accepting it is load-bearing: six of the nine corpus scenarios are
        controls or harness errors whose correct label is `unknown`.
        """
        step = AgentStep(
            category="unknown",
            hypothesis="",
            next_mitigations=[],
            confidence=0.0,
            stop=True,
        )
        assert AgentPolicy().validate_step(step).category == "unknown"

    @pytest.mark.parametrize("category", ["lds_race", "nan", "KERNEL_RACE", "", "kernel-race"])
    def test_a_label_outside_the_set_is_still_rejected(self, category):
        """Widening the set must not soften it -- it is still closed.

        `lds_race` in particular is the name `examples/rl/build_corpus.py` uses
        to synthesise its invalid-category negative example, so it has to stay
        outside.
        """
        step = AgentStep(
            category=category,
            hypothesis="",
            next_mitigations=[],
            confidence=0.5,
            stop=False,
        )
        with pytest.raises(PolicyViolation, match="invalid category"):
            AgentPolicy().validate_step(step)


class TestTheProposerIsTold:
    def test_the_prompt_carries_every_label_and_its_gloss(self):
        """The model cannot use a label nobody mentioned to it."""
        system, _ = _build_prompt(None, [], ["tf32_off"], [])
        for name, gloss in AUTOPSY_CATEGORY_GUIDANCE.items():
            assert f"- {name}: " in system, f"{name} missing from the prompt"
            assert gloss in system, f"{name}'s gloss missing from the prompt"

    def test_the_rendering_is_one_line_per_label(self):
        lines = format_category_guidance().splitlines()
        assert len(lines) == len(AUTOPSY_CATEGORIES)
        assert lines == sorted(lines), "sorted so the prompt is stable between runs"


def _fake_step(*, detectors: list[str] | None = None, symptom: str | None = None) -> AgentStep:
    summaries = [{"cell_name": "none-none", "verdict": "fail",
                  "failure_detectors_fired": detectors or []}]
    return FakeLLMProposer().propose(
        symptom=symptom, cell_summaries=summaries, candidates=["tf32_off"], tried=[]
    )


class TestTheOfflineHeuristic:
    """`FakeLLMProposer` labels with no model at all, so it needs the new names too."""

    @pytest.mark.parametrize(
        ("detector", "expected"),
        [
            # Pre-existing mappings, pinned so the new branches cannot shadow them.
            ("tier2:hang", "rccl_hang"),
            ("tier4:hip_error", "illegal_mem"),
            ("tier1:exit_nonzero", "launch_error"),
            # `tier4:nan_signature` has shipped since before the set had a home
            # for it, and fell through to `unknown` every time it fired.
            ("tier4:nan_signature", "numeric_instability"),
            ("consan:data_race", "kernel_race"),
            ("waitcheck:missing_waitcnt", "kernel_race"),
        ],
    )
    def test_detectors_route_to_the_right_label(self, detector, expected):
        assert _fake_step(detectors=[detector]).category == expected

    def test_barrier_evidence_is_a_kernel_race_not_a_checkpoint_race(self):
        """A barrier here is a GPU barrier -- ConSan sites, barrier patching.

        Routing it to `checkpoint_race` is how intra-kernel evidence ended up
        under a checkpoint-I/O label in the first place.
        """
        assert _fake_step(detectors=["consan:barrier_unpatched"]).category == "kernel_race"

    def test_checkpoint_still_wins_for_checkpoint_evidence(self):
        assert _fake_step(detectors=["checkpoint:race"]).category == "checkpoint_race"

    @pytest.mark.parametrize(
        ("symptom", "expected"),
        [
            ("loss is NaN after 200 steps", "numeric_instability"),
            ("nondeterministic eval scores between runs", "nondeterminism"),
            ("data race reported in the reduction kernel", "kernel_race"),
            ("race condition on the LDS tile", "kernel_race"),
        ],
    )
    def test_symptom_text_routes_to_the_right_label(self, symptom, expected):
        assert _fake_step(symptom=symptom).category == expected

    def test_a_traceback_is_not_a_race(self):
        """"race" is a substring of "traceback", which is a plausible symptom word."""
        assert _fake_step(symptom="python traceback at startup").category != "kernel_race"


class TestTheDocsAgree:
    @pytest.mark.parametrize("category", sorted(AUTOPSY_CATEGORIES))
    def test_the_operator_doc_lists_every_label(self, category):
        """The taxonomy table is what a human reads; it drifts silently otherwise."""
        assert f"`{category}`" in AGENT_DOC.read_text(encoding="utf-8")


class TestTheCorpusGroundTruth:
    """`scenario_labels.json` is the training signal the widened set exists for."""

    @staticmethod
    def _labels() -> dict:
        return json.loads(SCENARIO_LABELS.read_text(encoding="utf-8"))["scenarios"]

    def test_every_scenario_is_labelled_from_the_closed_set(self):
        offenders = {
            name: entry["category"]
            for name, entry in self._labels().items()
            if entry["category"] not in AUTOPSY_CATEGORIES
        }
        assert offenders == {}

    def test_every_scenario_names_a_recipe_that_exists(self):
        missing = [
            entry["recipe"]
            for entry in self._labels().values()
            if not (REPO_ROOT / entry["recipe"]).is_file()
        ]
        assert missing == []

    def test_every_scenario_says_why(self):
        assert [n for n, e in self._labels().items() if not e.get("why", "").strip()] == []

    def test_the_lds_race_is_labelled_a_kernel_race(self):
        """The one scenario the model labelled, and labelled wrong.

        It drew `checkpoint_race` because that was the nearest available name.
        """
        assert self._labels()["consan-racy"]["category"] == "kernel_race"
