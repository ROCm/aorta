"""Stop-outcome resolution for the agent loop."""

from __future__ import annotations

from aorta.agent.llm import AgentStep
from aorta.agent.loop import _resolve_stop_outcome


def test_resolve_baseline_pass():
    step = AgentStep(
        category="unknown",
        hypothesis="Baseline cell passed; no mitigation search needed.",
        next_mitigations=[],
        confidence=1.0,
        stop=True,
        stop_reason="baseline_pass",
    )
    summaries = [{"cell_name": "none-none", "verdict": "pass"}]
    outcome, msg, reason = _resolve_stop_outcome(step, summaries)
    assert outcome == "baseline_pass"
    assert reason == "baseline_pass"
    assert "without mitigations" in msg


def test_resolve_exhausted_candidates():
    step = AgentStep(
        category="rccl_hang",
        hypothesis="No remaining registered mitigations to try.",
        next_mitigations=[],
        confidence=0.9,
        stop=True,
        stop_reason="exhausted_candidates",
    )
    outcome, msg, reason = _resolve_stop_outcome(step, [])
    assert outcome == "exhausted_candidates"
    assert reason == "exhausted_candidates"
    assert "No further registered" in msg


def test_resolve_infers_reason_when_proposer_omits_it():
    """stop=True with stop_reason=None must yield a non-None inferred reason."""
    step = AgentStep(
        category="rccl_hang",
        hypothesis="No remaining registered mitigations to try.",
        next_mitigations=[],
        confidence=0.5,
        stop=True,
        stop_reason=None,
    )
    outcome, _msg, reason = _resolve_stop_outcome(step, [])
    assert outcome == "exhausted_candidates"
    assert reason == "exhausted_candidates"


def test_resolve_downgrades_unconfirmed_baseline_pass():
    """A proposer-claimed baseline_pass must not be honored when the probe
    verdicts don't show none-none passing -- otherwise a misbehaving proposer
    could falsely report success and stop the search."""
    step = AgentStep(
        category="unknown",
        hypothesis="Claiming baseline passed without evidence.",
        next_mitigations=[],
        confidence=1.0,
        stop=True,
        stop_reason="baseline_pass",
    )
    summaries = [{"cell_name": "none-none", "verdict": "fail"}]
    outcome, _msg, reason = _resolve_stop_outcome(step, summaries)
    assert outcome == "agent_stop"
    assert reason == "agent_requested"
