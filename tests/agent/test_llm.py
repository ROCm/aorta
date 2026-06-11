"""AgentStep.from_dict defensive coercion of untrusted LLM/JSON output."""

from __future__ import annotations

from aorta.agent.llm import AgentStep


def test_from_dict_null_category_and_hypothesis():
    """A null category/hypothesis must not become the literal "None"/"null"."""
    step = AgentStep.from_dict({"category": None, "hypothesis": None, "stop": True})
    assert step.category == "unknown"
    assert step.hypothesis == ""


def test_from_dict_non_string_category_and_hypothesis_fall_back():
    step = AgentStep.from_dict({"category": 123, "hypothesis": ["x"]})
    assert step.category == "unknown"
    assert step.hypothesis == ""


def test_from_dict_blank_category_falls_back():
    step = AgentStep.from_dict({"category": "   "})
    assert step.category == "unknown"


def test_from_dict_string_fields_preserved():
    step = AgentStep.from_dict({"category": "rccl_hang", "hypothesis": "ring hang"})
    assert step.category == "rccl_hang"
    assert step.hypothesis == "ring hang"


def test_from_dict_bare_string_mitigations_not_exploded():
    """A bare string must be ignored, never list("tf32_off") -> ['t','f',...]."""
    step = AgentStep.from_dict({"next_mitigations": "tf32_off"})
    assert step.next_mitigations == []


def test_from_dict_non_numeric_confidence_is_safe():
    step = AgentStep.from_dict({"confidence": "high"})
    assert step.confidence == 0.0
