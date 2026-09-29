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


@pytest.mark.parametrize("bad_name", ["tf32/off", "tf32:off", "foo@bar", ".hidden"])
def test_rejects_cell_unsafe_mitigation_name(bad_name):
    """A name that wouldn't round-trip through the probe cell-name builder is
    rejected, so the agent never records a scrubbed name as the winner."""
    policy = AgentPolicy()
    step = AgentStep(
        category="unknown",
        hypothesis="off-charset",
        next_mitigations=[bad_name],
        confidence=0.5,
        stop=False,
    )
    with pytest.raises(PolicyViolation, match="cell"):
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


def test_rejects_non_positive_max_iterations():
    with pytest.raises(PolicyViolation, match="max_iterations"):
        AgentPolicy(max_iterations=0)


@pytest.mark.parametrize("bad", [0, 0.0, -1, -30.5])
def test_rejects_non_positive_max_walltime(bad):
    with pytest.raises(PolicyViolation, match="max_walltime_sec"):
        AgentPolicy(max_walltime_sec=bad)


def test_accepts_none_and_positive_max_walltime():
    assert AgentPolicy(max_walltime_sec=None).max_walltime_sec is None
    assert AgentPolicy(max_walltime_sec=30.0).max_walltime_sec == 30.0


def test_nccl_launch_order_implicit_needs_approval():
    """It crashes every rank instead of resolving the hang (aorta#512), so an
    operator who asked to approve risky mitigations must be asked about it."""
    assert AgentPolicy(require_approval=True).needs_approval("nccl_launch_order_implicit")


def test_approval_gate_holds_only_gated_names_and_only_when_asked():
    proposed = ["xnack", "nccl_launch_order_implicit", "tf32_off", "hip_launch_blocking"]
    assert AgentPolicy(require_approval=True).pending_approvals(proposed) == [
        "nccl_launch_order_implicit",
        "hip_launch_blocking",
    ]
    assert AgentPolicy().pending_approvals(proposed) == []
