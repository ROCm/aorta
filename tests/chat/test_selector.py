"""Choosing a tool, and what happens when the choice cannot be made.

The selector is advisory by design. It runs before the agent and narrows what
the agent is pointed at, so every failure has to widen rather than narrow: a
model that returns nonsense, or no model at all, must leave the agent seeing
every tool. Narrowing on a bad reading is how a chatbot confidently runs the
wrong thing.

The one exception is structural. A tool that reads pasted source cannot run
when nothing was pasted, so that filter is applied in code and not left to the
model's judgement.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aorta.chat.graph.nodes import _looks_like_pasted_source, selector_node

_HIP = "```\n__global__ void k(float* o) { __shared__ float s[64]; }\n```"
_ASM = "```asm\ns_load_dword s4, s[0:1], 0x10\nv_mov_b32 v0, s4\n```"
_PY = "```python\nclass RMSNorm(nn.Module):\n    def forward(self, x): return x\n```"
_PROSE = "My training loss goes to NaN a few steps in. No crash, it just stops being a number."


def _model(reply: str) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    return llm


async def _select(message: str, reply: str) -> dict:
    with patch("aorta.chat.graph.nodes._get_llm", return_value=_model(reply)):
        return await selector_node({"messages": [HumanMessage(content=message)]})


# ── detecting source ──────────────────────────────────────────────────────


@pytest.mark.parametrize("pasted", [_HIP, _ASM, _PY], ids=["hip", "assembly", "python"])
def test_pasted_code_is_recognised(pasted: str):
    assert _looks_like_pasted_source(pasted)


def test_a_description_of_code_is_not_code():
    """The distinction the structural filter rests on."""
    assert not _looks_like_pasted_source(_PROSE)
    assert not _looks_like_pasted_source("the kernel writes to shared memory then reads it back")


# ── reading the model's answer ────────────────────────────────────────────


async def test_a_ranked_shortlist_is_kept_in_order():
    reply = json.dumps({"tools": ["triage_kernel_source", "search_code"], "why": "shared memory"})
    out = await _select(_HIP, reply)
    assert out["candidate_tools"] == ["triage_kernel_source", "search_code"]
    assert "shared memory" in out["selection_rationale"]


async def test_prose_around_the_json_is_tolerated():
    """Models wrap JSON in fences and commentary; that is not a failure."""
    reply = 'Here is my answer:\n```json\n{"tools": ["search_code"], "why": "why"}\n```\nHope that helps.'
    out = await _select(_PROSE, reply)
    assert out["candidate_tools"] == ["search_code"]


async def test_a_tool_that_does_not_exist_is_discarded():
    """A hallucinated name must not reach the agent as a candidate."""
    reply = json.dumps({"tools": ["run_the_debugger", "search_code"], "why": ""})
    out = await _select(_PROSE, reply)
    assert out["candidate_tools"] == ["search_code"]


async def test_the_shortlist_is_capped():
    from aorta.chat.tools.capabilities import MAX_CANDIDATES

    reply = json.dumps({"tools": sorted(__import__("aorta.chat.graph.nodes", fromlist=["x"]).TOOL_REGISTRY), "why": ""})
    out = await _select(_PROSE, reply)
    assert len(out["candidate_tools"]) <= MAX_CANDIDATES


# ── failing wide ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply", ["", "I could not decide.", "{not json at all", '{"tools": "not-a-list"}'],
    ids=["empty", "prose", "malformed", "wrong-type"],
)
async def test_an_unreadable_answer_leaves_the_agent_every_tool(reply: str):
    out = await _select(_PROSE, reply)
    assert out["candidate_tools"] == [], (
        "an unreadable ranking must widen to every tool, not narrow to a guess"
    )


async def test_an_unreachable_model_does_not_raise():
    """The selector is an optimisation. Losing it degrades quality, not service."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=ConnectionError("connection refused"))
    with patch("aorta.chat.graph.nodes._get_llm", return_value=llm):
        out = await selector_node({"messages": [HumanMessage(content=_PROSE)]})
    assert out["candidate_tools"] == []


# ── the one rule enforced in code ─────────────────────────────────────────


async def test_a_source_reading_tool_is_dropped_when_nothing_was_pasted():
    """Not a judgement call: there is nothing for it to read."""
    reply = json.dumps({"tools": ["triage_kernel_source", "search_code"], "why": ""})
    out = await _select(_PROSE, reply)
    assert out["candidate_tools"] == ["search_code"]
    assert "triage_kernel_source" in out["selection_rationale"]


async def test_the_same_tool_survives_when_source_is_present():
    reply = json.dumps({"tools": ["triage_kernel_source", "search_code"], "why": ""})
    out = await _select(_HIP, reply)
    assert out["candidate_tools"] == ["triage_kernel_source", "search_code"]


async def test_dropping_is_explained_rather_than_silent():
    """A shortlist that quietly shortened is one nobody can debug."""
    reply = json.dumps({"tools": ["triage_assembly_source"], "why": "wait hazard"})
    out = await _select(_PROSE, reply)
    assert out["candidate_tools"] == []
    assert "needs source" in out["selection_rationale"]
