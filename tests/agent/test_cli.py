"""CLI wiring for ``aorta agent``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

import aorta.cli.agent as agent_cli
from aorta.agent.loop import AgentLoopResult
from aorta.agent.state import AgentState
from aorta.cli.agent import agent


def test_agent_cli_invokes_loop(monkeypatch, tmp_path):
    mock_result = AgentLoopResult(
        run_dir=tmp_path / "r",
        state=AgentState(ticket="T1"),
        report_path=tmp_path / "r" / "agent_report.md",
        outcome="converged",
        recommended_action="done",
    )
    mock_loop = MagicMock(return_value=mock_result)
    monkeypatch.setattr(agent_cli, "run_agent_loop", mock_loop)

    runner = CliRunner()
    result = runner.invoke(
        agent,
        [
            "--output",
            str(tmp_path / "out"),
            "--ticket",
            "T1",
            "--",
            "echo",
            "ok",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_loop.assert_called_once()
    config = mock_loop.call_args[0][0]
    assert config.subprocess_argv == ("echo", "ok")
    assert config.ticket == "T1"


def test_agent_requires_double_dash_separator():
    runner = CliRunner()
    result = runner.invoke(agent, ["--output", "/tmp/o", "echo", "hi"])
    assert result.exit_code != 0
    assert "separator" in result.output.lower() or "Usage" in result.output
