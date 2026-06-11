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


def test_dry_run_writes_no_artifacts(mock_run_recipe, tmp_path, monkeypatch):
    """--dry-run must not write agent_log.jsonl / report, nor scan the cwd.

    run_recipe(dry_run=True) returns a sentinel Path("."); the loop must
    discard it instead of writing logs into the caller's working directory.
    """
    mock_run_recipe.return_value = Path(".")
    # _read_cell_summaries must never run in dry-run (no real run_dir to scan).
    import aorta.agent.loop as loop_mod

    def _boom(_run_dir):
        raise AssertionError("_read_cell_summaries called during dry-run")

    monkeypatch.setattr(loop_mod, "_read_cell_summaries", _boom)
    monkeypatch.chdir(tmp_path)

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="DRY-1",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=3),
        mitigations_allowlist=("none", "tf32_off"),
        dry_run=True,
    )
    result = run_agent_loop(config)

    assert result.outcome == "dry_run"
    assert result.report_path is None
    mock_run_recipe.assert_called_once()
    assert mock_run_recipe.call_args_list[-1][1].get("dry_run") is True
    # No log in the cwd (the discarded Path(".") sentinel) or the planned dir.
    assert not (tmp_path / "agent_log.jsonl").exists()
    assert not (tmp_path / "out" / "DRY-1" / "agent_log.jsonl").exists()
    assert not (tmp_path / "out" / "DRY-1" / "agent_report.md").exists()
