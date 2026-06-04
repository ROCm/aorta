"""Policy guardrails for the probe agent."""

from __future__ import annotations

import pytest

from aorta.agent.llm import AgentStep
from aorta.agent.policy import AgentPolicy, PolicyViolation


def test_rejects_shell_like_mitigation_name():
    policy = AgentPolicy()
    step = AgentStep(
        category="unknown",
        hypothesis="bad",
        next_mitigations=["python3 -c evil"],
        confidence=0.5,
        stop=False,
    )
    with pytest.raises(PolicyViolation, match="shell"):
        policy.validate_step(step)


def test_rejects_unregistered_mitigation():
    policy = AgentPolicy()
    step = AgentStep(
        category="unknown",
        hypothesis="bad",
        next_mitigations=["not_a_real_mitigation_xyz"],
        confidence=0.5,
        stop=False,
    )
    with pytest.raises(PolicyViolation):
        policy.validate_step(step)


def test_accepts_registered_mitigation():
    policy = AgentPolicy()
    step = AgentStep(
        category="rccl_hang",
        hypothesis="try tf32",
        next_mitigations=["tf32_off"],
        confidence=0.5,
        stop=False,
    )
    validated = policy.validate_step(step)
    assert validated.next_mitigations == ["tf32_off"]


def test_iteration_budget():
    policy = AgentPolicy(max_iterations=2)
    policy.check_iteration_budget(0)
    policy.check_iteration_budget(1)
    with pytest.raises(PolicyViolation, match="budget"):
        policy.check_iteration_budget(2)
