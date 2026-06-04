"""Agent report writer."""

from __future__ import annotations

from aorta.agent.report import write_agent_report
from aorta.agent.state import AgentState


def test_write_agent_report(tmp_path):
    state = AgentState(
        ticket="T-1",
        last_category="illegal_mem",
        last_hypothesis="GPU fault",
        winning_mitigation="tf32_off",
    )
    path = write_agent_report(
        tmp_path,
        state=state,
        cell_summaries=[
            {
                "cell_name": "none-none",
                "verdict": "fail",
                "failure_detectors_fired": ["tier4:hip_error"],
                "capture": {"loss": "nan"},
            }
        ],
        outcome="converged",
        recommended_action="Apply tf32_off.",
    )
    text = path.read_text(encoding="utf-8")
    assert "illegal_mem" in text
    assert "tf32_off" in text
    assert "tier4:hip_error" in text
