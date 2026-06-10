"""Closed-loop agent tests with mocked run_recipe."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aorta.agent.loop import AgentConfig, run_agent_loop
from aorta.agent.policy import AgentPolicy


@pytest.fixture
def mock_run_recipe(monkeypatch):
    mock = MagicMock(return_value=Path("/tmp/agent-run"))
    import aorta.agent.loop as loop_mod

    monkeypatch.setattr(loop_mod, "run_recipe", mock)
    return mock


def _summaries_sequence():
    """Baseline fail, then mitigation pass."""
    return [
        [
            {
                "cell_name": "none-none",
                "verdict": "fail",
                "failure_detectors_fired": ["tier1:exit_nonzero"],
                "capture": {},
            }
        ],
        [
            {
                "cell_name": "none-none",
                "verdict": "fail",
                "failure_detectors_fired": ["tier1:exit_nonzero"],
                "capture": {},
            },
            {
                "cell_name": "tf32_off-none",
                "verdict": "pass",
                "failure_detectors_fired": [],
                "capture": {},
            },
        ],
    ]


def test_loop_converges_with_fake_llm(tmp_path, monkeypatch, mock_run_recipe):
    import aorta.agent.loop as loop_mod

    seq = _summaries_sequence()
    calls = {"n": 0}

    def fake_summaries(run_dir: Path):
        idx = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[idx]

    monkeypatch.setattr(loop_mod, "_read_cell_summaries", fake_summaries)

    def fake_run_dir(config):
        return config.output_dir / "ROCM-AGENT-TEST"

    monkeypatch.setattr(loop_mod, "_run_dir", fake_run_dir)

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="ROCM-AGENT-TEST",
        subprocess_argv=("echo", "hi"),
        policy=AgentPolicy(max_iterations=5),
        mitigations_allowlist=("none", "tf32_off"),
        llm_backend="fake",
    )
    result = run_agent_loop(config)
    assert result.outcome == "converged"
    assert result.state.winning_mitigation == "tf32_off"
    assert mock_run_recipe.call_count >= 1
    last_recipe = mock_run_recipe.call_args_list[-1][0][0]
    assert "tf32_off" in last_recipe.probe_extras.mitigation_axis
    kwargs = mock_run_recipe.call_args_list[-1][1]
    assert kwargs.get("layout") == "flat_resume"
    assert kwargs.get("resume_existing") is True
    assert kwargs.get("subprocess_argv") == ("echo", "hi")


def test_loop_uses_flat_resume_engine(mock_run_recipe, tmp_path, monkeypatch):
    import aorta.agent.loop as loop_mod

    monkeypatch.setattr(
        loop_mod,
        "_read_cell_summaries",
        lambda _d: [
            {
                "cell_name": "none-none",
                "verdict": "pass",
                "failure_detectors_fired": [],
                "capture": {},
            }
        ],
    )
    monkeypatch.setattr(
        loop_mod,
        "_run_dir",
        lambda c: c.output_dir / "BASELINE-PASS",
    )

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="BASELINE-PASS",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=3),
        mitigations_allowlist=("none",),
    )
    result = run_agent_loop(config)
    assert result.outcome == "baseline_pass"
    assert "Baseline cell" in result.recommended_action
    mock_run_recipe.assert_called()
