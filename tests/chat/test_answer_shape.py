"""The answer a diagnostic run has to produce.

An engineer reading a verdict needs three things, and the middle one is the one
that gets dropped: what is wrong, *how we know*, and what to change. Without the
middle part a verdict is an assertion — the reader cannot tell a collision that
was observed from a hazard inferred off a path that may never execute, and the
two deserve different responses.

Which is why the confidence is reported rather than rounded up. 0.55 and 0.95
are both real answers about a real bug; presenting them identically throws away
the only signal that says how hard to trust it.

These assert the instruction, not a model's compliance with it — an offline
suite cannot test the latter. What it can do is fail when the instruction is
dropped, which is how it would actually be lost.
"""

from __future__ import annotations

import pytest

from aorta.chat.graph.nodes import SYSTEM_PROMPT


@pytest.mark.parametrize(
    "requirement",
    [
        "three labelled parts",
        "how we found it",
        "the fix",
    ],
)
def test_the_answer_shape_is_specified(requirement: str):
    assert requirement in SYSTEM_PROMPT


def test_the_evidence_part_is_told_what_to_cite():
    """"How we found it" is worthless if it only names the tool."""
    lowered = SYSTEM_PROMPT.lower()
    for citation in ("signal", "file", "line", "confidence"):
        assert citation in lowered, f"the prompt never asks for a {citation}"


def test_the_fix_is_asked_for_verbatim():
    """A paraphrased diff is one the reader has to reconstruct."""
    assert "verbatim" in SYSTEM_PROMPT


def test_confidence_is_reported_rather_than_rounded():
    """The instruction that keeps 0.55 from being presented as certainty."""
    assert "rounding it up" in SYSTEM_PROMPT or "rather than rounding" in SYSTEM_PROMPT


def test_the_prompt_still_names_no_tool_for_a_symptom():
    """The three-part rule must not smuggle routing back in.

    It is guidance about the shape of an answer. The moment it says which tool
    to use for which symptom, tool choice is a routing table again and the
    selector is decoration.
    """
    lowered = SYSTEM_PROMPT.lower()
    for coupling in ("if the user mentions nan", "for nan", "for a race", "if it is a race"):
        assert coupling not in lowered



# ── where the instruction sits, not just that it exists ───────────────────


class TestItSitsWhereTheModelWillRead:
    """Presence is not placement, and this file only checked presence.

    The instruction was appended after ``RETRIEVED CONTEXT:`` and the
    ``{context}`` interpolation, so it arrived below however many kilobytes of
    retrieved code chunks the turn had gathered -- numbered into a rules list it
    was nowhere near. Every assertion above is a substring check, which passes
    wherever the text sits, so the suite reported the instruction as present the
    whole time it was in the worst place in the prompt for one.
    """

    def test_it_comes_before_the_retrieved_context(self):
        assert SYSTEM_PROMPT.index("three labelled parts") < SYSTEM_PROMPT.index(
            "RETRIEVED CONTEXT"
        )

    def test_and_before_the_context_is_interpolated(self):
        """The marker and the interpolation are separate places to land after."""
        assert SYSTEM_PROMPT.index("three labelled parts") < SYSTEM_PROMPT.index(
            "{context}"
        )

    def test_it_sits_inside_the_numbered_list(self):
        """A rule numbered into a list the reader has already left is not a rule."""
        rules = SYSTEM_PROMPT[
            SYSTEM_PROMPT.index("RULES:") : SYSTEM_PROMPT.index("RETRIEVED CONTEXT")
        ]

        assert "three labelled parts" in rules

    def test_it_carries_a_number_of_its_own(self):
        """So a reader can see it belongs to the list rather than trailing it."""
        import re

        rules = SYSTEM_PROMPT[
            SYSTEM_PROMPT.index("RULES:") : SYSTEM_PROMPT.index("RETRIEVED CONTEXT")
        ]
        line_start = rules.rindex("\n", 0, rules.index("three labelled parts")) + 1

        assert re.match(r"\s*\d+\.", rules[line_start:]), rules[line_start:line_start + 40]

    def test_the_rules_run_without_a_gap_in_the_numbering(self):
        """A rule appended elsewhere shows up as a number missing from here."""
        import re

        rules = SYSTEM_PROMPT[
            SYSTEM_PROMPT.index("RULES:") : SYSTEM_PROMPT.index("RETRIEVED CONTEXT")
        ]
        numbers = [int(n) for n in re.findall(r"^\s*(\d+)\.", rules, re.MULTILINE)]

        assert numbers == list(range(1, len(numbers) + 1)), numbers
