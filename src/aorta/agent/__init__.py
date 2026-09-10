"""Agentic closed-loop mitigation search on top of ``aorta probe``.

The probe agent grows a mitigation axis iteratively, reusing
``run_recipe`` with ``flat_resume`` so completed cells are skipped.
Verdicts always come from the deterministic probe classifier; the LLM
only labels failures and proposes registered mitigation names.
"""

from aorta.agent.llm import (
    AUTOPSY_CATEGORIES,
    AUTOPSY_CATEGORY_GUIDANCE,
    AgentStep,
    FakeLLMProposer,
    LiteLLMProposer,
    LLMProposer,
    format_category_guidance,
)
from aorta.agent.loop import AgentConfig, AgentLoopResult, run_agent_loop
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.agent.report import write_agent_report
from aorta.agent.state import AgentState, append_log_event, wake

__all__ = [
    "AUTOPSY_CATEGORIES",
    "AUTOPSY_CATEGORY_GUIDANCE",
    "AgentConfig",
    "AgentLoopResult",
    "AgentPolicy",
    "AgentState",
    "AgentStep",
    "FakeLLMProposer",
    "LLMProposer",
    "LiteLLMProposer",
    "PolicyViolation",
    "append_log_event",
    "format_category_guidance",
    "run_agent_loop",
    "wake",
    "write_agent_report",
]
