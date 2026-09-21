"""The probe agent proposes only categories it can actually reach.

Unifying the taxonomy put ``gpu_race``, ``numeric_silent`` and ``tooling_gap``
into ``AUTOPSY_CATEGORIES`` so a verdict means the same thing whichever front
door produced it. But that set is also what ``_build_prompt`` offers the probe
model and what ``AgentPolicy.validate_step`` accepts — so the probe agent could
propose all three, and the policy would take them, despite the comment beside
them saying it never would.

Nothing a mitigation sweep does can establish those. It tries a change and sees
what moves; it cannot watch two waves collide, and it cannot tell "the sanitizer
found nothing" from "the sanitizer never ran". A probe verdict of ``gpu_race``
would be a guess wearing the same name an instrument uses — and downstream, one
is indistinguishable from the other.

The subset is derived, not written out again. A second hand-kept list is how the
two drift, and the drift is silent.
"""

from __future__ import annotations

import pytest

from aorta.agent.llm import (
    AUTOPSY_CATEGORIES,
    EVIDENCE_ONLY_CATEGORIES,
    PROBE_CATEGORIES,
    AgentStep,
)
from aorta.agent.policy import AgentPolicy, PolicyViolation


class TestTheVocabularies:
    def test_the_probe_subset_is_the_shared_set_less_the_evidence_only_ones(self):
        assert PROBE_CATEGORIES == AUTOPSY_CATEGORIES - EVIDENCE_ONLY_CATEGORIES

    def test_they_do_not_overlap(self):
        assert not (PROBE_CATEGORIES & EVIDENCE_ONLY_CATEGORIES)

    def test_the_shared_set_still_holds_everything(self):
        assert EVIDENCE_ONLY_CATEGORIES < AUTOPSY_CATEGORIES
        assert PROBE_CATEGORIES < AUTOPSY_CATEGORIES

    @pytest.mark.parametrize("category", ["gpu_race", "numeric_silent", "tooling_gap"])
    def test_the_evidence_only_ones_are_named(self, category):
        assert category in EVIDENCE_ONLY_CATEGORIES

    def test_the_probe_keeps_the_categories_it_could_always_reach(self):
        assert {
            "rccl_hang",
            "thermal_throttle",
            "illegal_mem",
            "oom_fragment",
            "checkpoint_race",
            "launch_error",
            "perf_regression",
            "unknown",
        } == PROBE_CATEGORIES


class TestThePolicyRefusesThem:
    def _step(self, category: str) -> AgentStep:
        return AgentStep(
            category=category,
            hypothesis="h",
            next_mitigations=[],
            confidence=0.5,
            stop=True,
        )

    @pytest.mark.parametrize("category", sorted(EVIDENCE_ONLY_CATEGORIES))
    def test_an_evidence_only_category_is_a_violation(self, category):
        with pytest.raises(PolicyViolation, match="invalid category"):
            AgentPolicy().validate_step(self._step(category))

    @pytest.mark.parametrize("category", sorted(PROBE_CATEGORIES))
    def test_a_probe_category_is_accepted(self, category):
        assert AgentPolicy().validate_step(self._step(category)).category == category

    def test_the_message_lists_only_what_is_allowed(self):
        with pytest.raises(PolicyViolation) as excinfo:
            AgentPolicy().validate_step(self._step("gpu_race"))
        message = str(excinfo.value)
        assert "rccl_hang" in message
        assert "numeric_silent" not in message, "the message must not offer it either"


class TestThePromptOffersThem:
    @staticmethod
    def _system_prompt() -> str:
        import inspect

        from aorta.agent import llm

        return inspect.getsource(llm._build_prompt)

    def test_it_is_built_from_the_probe_subset(self):
        assert "PROBE_CATEGORIES" in self._system_prompt()
        assert "AUTOPSY_CATEGORIES" not in self._system_prompt()


class TestTheOtherFrontDoorIsUnaffected:
    """CIA reads instruments, so it keeps the whole vocabulary."""

    @pytest.mark.parametrize("category", sorted(AUTOPSY_CATEGORIES))
    def test_the_router_still_accepts_every_category(self, category):
        from aorta.cia.autopsy.router import coerce_category

        assert coerce_category(category) == category

    def test_the_router_describes_the_whole_vocabulary(self):
        from aorta.cia.autopsy import router

        for category in EVIDENCE_ONLY_CATEGORIES:
            assert category in router._CATEGORY_DESC
