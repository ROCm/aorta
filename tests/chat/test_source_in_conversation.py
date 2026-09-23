"""Whether the user gave us code is a question about the conversation.

A tool that analyses pasted source cannot run when nothing was pasted, and that
is the one thing the selector decides in code rather than leaving to the model.
It decided it by looking at the newest message only.

Pasting and asking are usually different turns. A kernel arrives, the agent
asks what block size to launch it with, the user says "yes, run the sanitizer
on it" -- and that reply contains no code, so the source tools were withdrawn
from exactly the turn that asked for them. The rationale then said the source
was needed "in the message", which was true and beside the point: it was in the
conversation.

The markers had a second problem. "def ", "class " and "import " are ordinary
English as well as keywords, so "I import the model and define a class for it"
read as a paste. That errs the harmless way -- it offers a tool rather than
withholding one -- but a signal that fires on a sentence about code is not
measuring what it claims to.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aorta.chat.graph.nodes import (
    _SOURCE_LOOKBACK,
    _conversation_has_source,
    _looks_like_pasted_source,
)
from aorta.chat.tools.capabilities import enforce_requirements

_KERNEL = "```\n__global__ void bump(float* o) { __shared__ float s[64]; }\n```"
_ASM = "```asm\ns_load_dword s4, s[0:1], 0x10\nv_mov_b32 v0, s4\n```"
_SOURCE_TOOLS = ["triage_kernel_source", "triage_assembly_source"]


def _kept(messages: list) -> list[str]:
    kept, _ = enforce_requirements(
        list(_SOURCE_TOOLS), has_pasted_source=_conversation_has_source(messages)
    )
    return kept


class TestAPasteFromAnEarlierTurn:
    """The finding: the turn that asks is rarely the turn that pasted."""

    def test_the_source_tools_survive_the_follow_up(self):
        convo = [
            HumanMessage(content=_KERNEL),
            AIMessage(content="What block size should I launch it with?"),
            HumanMessage(content="yes, run the sanitizer on it"),
        ]

        assert _kept(convo) == _SOURCE_TOOLS

    def test_a_longer_clarifying_exchange_still_counts(self):
        convo = [
            HumanMessage(content=_KERNEL),
            AIMessage(content="Is this the whole kernel?"),
            HumanMessage(content="that is all of it"),
            AIMessage(content="What block size?"),
            HumanMessage(content="256, go ahead"),
        ]

        assert _kept(convo) == _SOURCE_TOOLS

    def test_assembly_pasted_earlier_counts_too(self):
        convo = [
            HumanMessage(content=_ASM),
            AIMessage(content="Which architecture is this for?"),
            HumanMessage(content="gfx950, check the waits"),
        ]

        assert _kept(convo) == _SOURCE_TOOLS

    def test_a_conversation_with_no_paste_still_drops_them(self):
        """The guard has to keep guarding; this is what it is for."""
        convo = [
            HumanMessage(content="my training loss goes to NaN a few steps in"),
            AIMessage(content="Does it crash?"),
            HumanMessage(content="no, it just stops being a number"),
        ]

        assert _kept(convo) == []

    def test_and_says_it_was_the_conversation_that_had_none(self):
        """The old wording sent the reader to re-read their last message."""
        _, dropped = enforce_requirements(list(_SOURCE_TOOLS), has_pasted_source=False)

        assert dropped == _SOURCE_TOOLS


class TestOnlyWhatTheUserPasted:
    def test_the_agent_quoting_code_back_is_not_a_paste(self):
        """Otherwise the agent can talk itself into having source."""
        convo = [
            HumanMessage(content="what does a reduction kernel look like?"),
            AIMessage(content=_KERNEL),
            HumanMessage(content="run the sanitizer on it"),
        ]

        assert _kept(convo) == []

    def test_nor_is_a_tool_result_that_contains_code(self):
        convo = [
            HumanMessage(content="read reduce.hip for me"),
            ToolMessage(content=_KERNEL, tool_call_id="1"),
            HumanMessage(content="now sanitize it"),
        ]

        assert _kept(convo) == []

    def test_a_system_prompt_full_of_examples_is_not_a_paste(self):
        convo = [
            SystemMessage(content=_KERNEL),
            HumanMessage(content="run the sanitizer"),
        ]

        assert _kept(convo) == []


class TestTheLookbackIsBounded:
    def test_a_paste_within_the_window_counts(self):
        convo = [HumanMessage(content=_KERNEL)]
        convo += [HumanMessage(content="and?") for _ in range(_SOURCE_LOOKBACK - 1)]

        assert _kept(convo) == _SOURCE_TOOLS

    def test_a_paste_long_since_left_behind_does_not(self):
        """A kernel from far earlier should not decide an unrelated question."""
        convo = [HumanMessage(content=_KERNEL)]
        convo += [HumanMessage(content="and?") for _ in range(_SOURCE_LOOKBACK + 2)]

        assert _kept(convo) == []

    def test_an_empty_conversation_is_not_a_paste(self):
        assert _kept([]) == []


class TestStructureBeatsVocabulary:
    """Fences and __global__ do not appear in a sentence; "class" does."""

    @pytest.mark.parametrize(
        "prose",
        [
            "I import the model and define a class for it",
            "we class this as a hang, not a race",
            "the import failed halfway through",
            "can you def the forward pass for me",
        ],
    )
    def test_a_sentence_about_code_is_not_code(self, prose):
        assert not _looks_like_pasted_source(prose)

    @pytest.mark.parametrize(
        "pasted",
        [
            "import torch\nimport torch.nn as nn",
            "class RMSNorm(nn.Module):\n    pass",
            "def forward(self, x):\n    return x",
            "    def forward(self, x):\n        return x",
        ],
        ids=["import", "class", "def", "indented"],
    )
    def test_a_keyword_at_the_start_of_a_line_is_code(self, pasted):
        assert _looks_like_pasted_source(pasted)

    @pytest.mark.parametrize(
        "pasted", [_KERNEL, _ASM, "```python\nx = 1\n```"], ids=["hip", "asm", "fenced"]
    )
    def test_structural_markers_count_anywhere(self, pasted):
        assert _looks_like_pasted_source(pasted)

    def test_a_marker_mid_sentence_still_counts_when_it_is_structural(self):
        """__global__ in a sentence means they are quoting the code at us."""
        assert _looks_like_pasted_source("the __global__ void bump one, that one")
