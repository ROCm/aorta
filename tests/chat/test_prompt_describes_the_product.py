"""The prompt should describe the assistant that is actually running.

Rule 1 said to answer questions about the AORTA codebase or the AORTA runs on
this machine and politely refuse anything else. A user pasting their own kernel
and asking why it races is asking about neither -- it is their code, not
AORTA's -- so rule 1 refuses it. Rules 12 and 13 then say to run a diagnostic
tool on exactly that and answer in three labelled parts.

Two products in one prompt, with nothing saying which wins. A model following
rule 1 refuses the question the selector was taught to route; a model following
rule 12 diagnoses it.

Rule 3 completed the gap: it named the sandboxed file, search and run-artifact
tools, so the three tools that do the work this product exists for appeared
nowhere in the prompt that asks for them.

Both clauses are conditional, because the tools are. They register only when
``allow_cluster_jobs`` is set, and a prompt naming tools the model cannot call
is what the removed run_nan_demo redirect did.
"""

from __future__ import annotations

import pytest

from aorta.chat.config import reset_settings
from aorta.chat.graph.nodes import ANSWER_PROMPT, _build_system_message

_TRIAGE = ("triage_kernel_source", "triage_assembly_source", "triage_workload")


@pytest.fixture()
def prompt(monkeypatch):
    def build(*, cluster_jobs: bool) -> str:
        monkeypatch.setenv("AORTA_CHAT_ALLOW_CLUSTER_JOBS", str(cluster_jobs).lower())
        reset_settings()
        return _build_system_message("SOME CONTEXT").content

    yield build
    reset_settings()


def _rule(text: str, number: int) -> str:
    start = text.index(f"\n{number}. ")
    end = text.index(f"\n{number + 1}. ", start)
    return " ".join(text[start:end].split())


class TestWithTheDiagnosticToolsOn:
    def test_rule_one_admits_pasted_code(self, prompt):
        """Otherwise it refuses the question rules 12 and 13 are written for."""
        pytest.importorskip("dspy", reason="the tools need the [cia] extra")

        assert "pastes for diagnosis" in _rule(prompt(cluster_jobs=True), 1)

    def test_rule_one_still_refuses_unrelated_work(self, prompt):
        """Widening the scope is not removing it."""
        assert "refuse" in _rule(prompt(cluster_jobs=True), 1)

    @pytest.mark.parametrize("tool", _TRIAGE)
    def test_rule_three_names_the_tool(self, prompt, tool):
        pytest.importorskip("dspy", reason="the tools need the [cia] extra")

        assert tool in _rule(prompt(cluster_jobs=True), 3)

    def test_rule_three_says_they_cost_real_work(self, prompt):
        """A minutes-long cluster job should not be spent on curiosity."""
        pytest.importorskip("dspy", reason="the tools need the [cia] extra")

        assert "minutes" in _rule(prompt(cluster_jobs=True), 3)

    def test_the_rules_that_use_them_are_still_there(self, prompt):
        text = prompt(cluster_jobs=True)

        assert "run the matching diagnostic tool" in text
        assert "three labelled parts" in text


class TestWithThemOff:
    """The default. A prompt must not offer what is not registered."""

    @pytest.mark.parametrize("tool", _TRIAGE)
    def test_no_tool_is_named(self, prompt, tool):
        assert tool not in prompt(cluster_jobs=False)

    def test_rule_one_does_not_offer_diagnosis(self, prompt):
        assert "pastes for diagnosis" not in _rule(prompt(cluster_jobs=False), 1)

    def test_rule_one_is_still_a_sentence(self, prompt):
        """The clause is interpolated, so its absence must not leave a seam."""
        rule = _rule(prompt(cluster_jobs=False), 1)

        assert "machine. Politely refuse" in rule
        assert "{" not in rule and "}" not in rule


class TestTheContextStillInterpolates:
    def test_the_retrieved_context_arrives(self, prompt):
        assert "SOME CONTEXT" in prompt(cluster_jobs=False)

    def test_no_placeholder_survives_either_way(self, prompt):
        for flag in (True, False):
            text = prompt(cluster_jobs=flag)
            assert "{diagnosis_scope}" not in text
            assert "{diagnostic_tools}" not in text
            assert "{context}" not in text


class TestTheToolFreePathSaysWhyItCannot:
    """ANSWER_PROMPT has no tools, so it cannot diagnose -- but refusing is wrong.

    A pasted-kernel question routed down the question path would have been
    refused as off-topic. It is on-topic and unanswerable here, which is a
    different thing and worth saying.
    """

    def test_it_does_not_refuse_pasted_code_outright(self):
        assert "pasted code of their own" in " ".join(ANSWER_PROMPT.split())

    def test_it_says_a_diagnostic_run_is_what_is_missing(self):
        assert "needs a diagnostic run" in " ".join(ANSWER_PROMPT.split())

    def test_it_promises_no_tools(self):
        """It has none; naming any would be the run_nan_demo failure again."""
        for tool in _TRIAGE:
            assert tool not in ANSWER_PROMPT


def _identity(text: str) -> str:
    """The sentence before the rules: what the assistant says it is."""
    return " ".join(text.split("RULES:")[0].split())


class TestTheIdentityAgreesWithTheRules:
    """The prompt has to introduce itself as the product it then describes.

    Rule 1 was widened to take pasted GPU code while the sentence above it
    still said "the AORTA codebase" and nothing else. A model reading the two
    in order is introduced to one product and then given the rules of another,
    and the identity sentence is the one it leans on when the rules are
    ambiguous -- which is how a diagnostic request gets politely refused by
    the assistant built to diagnose it.
    """

    def test_it_says_it_diagnoses_when_it_can(self, prompt):
        assert "diagnoses" in _identity(prompt(cluster_jobs=True))

    def test_it_names_what_it_diagnoses(self, prompt):
        identity = _identity(prompt(cluster_jobs=True))
        for subject in ("kernels", "assembly", "workloads"):
            assert subject in identity, f"{subject} missing from {identity!r}"

    def test_it_claims_nothing_extra_when_it_cannot(self, prompt):
        """Without the extra there is no GPU node to run anything on."""
        assert "diagnoses" not in _identity(prompt(cluster_jobs=False))

    def test_the_codebase_half_survives_both_ways(self, prompt):
        for enabled in (True, False):
            assert "AORTA codebase" in _identity(prompt(cluster_jobs=enabled))

    def test_the_identity_and_rule_one_agree(self, prompt):
        """Neither may offer what the other refuses."""
        for enabled in (True, False):
            text = prompt(cluster_jobs=enabled)
            offered = "diagnoses" in _identity(text)
            admitted = "pastes for diagnosis" in _rule(text, 1)
            assert offered == admitted, (
                f"identity says {offered}, rule 1 says {admitted}, with "
                f"cluster_jobs={enabled}"
            )
