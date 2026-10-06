"""The agent is told to run a diagnostic tool, not to reason about the code.

A pasted workload can be readable enough to solve by inspection, and a model
that can will: given a sixteen-line training script that goes NaN at step 5, it
answered correctly and wrote "traced the data flow through rms_norm" as its
evidence. Nothing had run, on a GPU or anywhere else. The answer was right that
time, and a reader has no way to tell that from a fluent guess.

Nothing in the prompt had asked for a tool to be run -- the rules only covered
searching the codebase.
"""

from __future__ import annotations

import re

from aorta.chat.graph.nodes import SYSTEM_PROMPT

_PROMPT = SYSTEM_PROMPT.lower()


def test_the_agent_is_told_to_run_a_tool_on_a_pasted_artefact():
    assert "run the matching diagnostic tool" in _PROMPT


def test_the_rule_names_what_users_actually_paste():
    for artefact in ("kernel", "assembly listing", "workload"):
        assert artefact in _PROMPT


def test_reading_the_code_is_not_an_accepted_substitute_for_running_it():
    assert "however clearly" in _PROMPT
    assert "only say a thing was observed if a tool observed it" in _PROMPT


def test_the_rule_is_stated_before_the_retrieved_context():
    """A rule below {context} arrives after the whole retrieval dump."""
    tail = SYSTEM_PROMPT[SYSTEM_PROMPT.index("RETRIEVED CONTEXT:") :]
    assert "run the matching diagnostic tool" not in tail


def test_the_context_placeholder_is_the_last_thing_in_the_prompt():
    assert SYSTEM_PROMPT.rstrip().endswith("{context}")


def test_the_rules_stay_numbered_without_a_gap_or_a_repeat():
    numbers = [int(n) for n in re.findall(r"^(\d+)\.", SYSTEM_PROMPT, re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1))
