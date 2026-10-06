"""Progress steps explain activity without exposing internal graph state."""

from __future__ import annotations

import pytest

pytest.importorskip("chainlit", reason="requires the chat-ui extra")

from aorta.chat.ui.app import _node_reasoning


def test_router_describes_the_decision_without_its_internal_route_name():
    rendered = _node_reasoning("router", {"route": "action"})

    assert "diagnostic evidence" in rendered
    assert "Route:" not in rendered
    assert "action" not in rendered


def test_selector_translates_tool_names_to_user_facing_activities():
    rendered = _node_reasoning(
        "select",
        {
            "candidate_tools": ["triage_assembly_source", "read_autopsy_report"],
            "selection_rationale": (
                "triage_assembly_source can identify the missing wait"
            ),
        },
    )

    assert "Analyzing the GPU assembly" in rendered
    assert "Reviewing the diagnostic report" in rendered
    assert "triage_assembly_source" not in rendered
    assert "read_autopsy_report" not in rendered


def test_unknown_selector_tool_uses_generic_copy():
    rendered = _node_reasoning(
        "select",
        {"candidate_tools": ["private_plugin_name"]},
    )

    assert "Gathering additional evidence" in rendered
    assert "private_plugin_name" not in rendered


def test_plan_does_not_render_model_authored_internal_names():
    rendered = _node_reasoning(
        "plan",
        {"plan": "Call triage_assembly_source and then read_autopsy_report."},
    )

    assert rendered == "AORTA has planned which evidence to gather and in what order."
    assert "triage_assembly_source" not in rendered


def test_tool_trace_is_summarized_instead_of_rendered():
    rendered = _node_reasoning(
        "act",
        {
            "tool_trace": [
                "[triage_assembly_source({'source': 'secret'})] -> finding",
                "[read_autopsy_report({'job_id': 'private'})] -> report",
            ]
        },
    )

    assert rendered == "AORTA gathered 2 diagnostic results for the answer."
    assert "triage_assembly_source" not in rendered
    assert "secret" not in rendered


def test_empty_tool_trace_renders_nothing():
    assert _node_reasoning("act", {"tool_trace": []}) == ""


def test_critic_does_not_render_raw_feedback():
    rendered = _node_reasoning(
        "critic",
        {
            "critic_feedback": (
                "The answer named triage_assembly_source without evidence."
            )
        },
    )

    assert "revising" in rendered
    assert "triage_assembly_source" not in rendered


def test_accepted_answer_is_explained_in_plain_language():
    rendered = _node_reasoning("critic", {"critic_feedback": None})

    assert "supported by the evidence" in rendered
