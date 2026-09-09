"""The rule that shapes the answer arrives before the context, not after it.

The three-part rule was written below the ``RETRIEVED CONTEXT:`` block, so it
reached the model after however many thousand tokens the retriever had just
pasted in -- the last place a model looks, for the rule that governs the whole
reply. Nothing failed loudly; the answers just came back in whatever shape the
model preferred.
"""

from __future__ import annotations

import re

from aorta.chat.graph.nodes import SYSTEM_PROMPT


def test_the_three_part_rule_is_stated_before_the_retrieved_context():
    tail = SYSTEM_PROMPT[SYSTEM_PROMPT.index("RETRIEVED CONTEXT:") :]
    assert "three labelled parts" not in tail


def test_the_context_placeholder_is_the_last_thing_in_the_prompt():
    """Anything after it is a rule behind the retrieval dump."""
    assert SYSTEM_PROMPT.rstrip().endswith("{context}")


def test_the_rules_are_numbered_without_a_gap_or_a_repeat():
    """Two rules both numbered 12 is the shape this arrived in."""
    numbers = [int(n) for n in re.findall(r"^(\d+)\.", SYSTEM_PROMPT, re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1))


def test_the_answer_still_has_to_carry_all_three_parts():
    for part in ("the bug", "how we found it", "the fix"):
        assert part in SYSTEM_PROMPT.lower()


def test_confidence_is_reported_rather_than_rounded_up():
    assert "rather than rounding it up" in SYSTEM_PROMPT


def test_running_the_tool_and_reporting_it_are_separate_rules():
    """One says to run a tool, the other how to report what it found."""
    assert "run the matching diagnostic tool" in SYSTEM_PROMPT
    assert "three labelled parts" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.index("run the matching diagnostic tool") < SYSTEM_PROMPT.index(
        "three labelled parts"
    )
