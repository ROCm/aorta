"""CLI wiring for ``aorta agent``."""

from __future__ import annotations

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


def test_error_outcome_has_dedicated_headline(monkeypatch, tmp_path):
    # run_agent_loop can return outcome="error" from its generic exception
    # handler; the CLI must show a specific headline, not the generic fallback.
    mock_result = AgentLoopResult(
        run_dir=tmp_path / "r",
        state=AgentState(ticket="T1"),
        report_path=tmp_path / "r" / "agent_report.md",
        outcome="error",
        recommended_action="inspect logs",
    )
    monkeypatch.setattr(agent_cli, "run_agent_loop", MagicMock(return_value=mock_result))

    runner = CliRunner()
    result = runner.invoke(agent, ["--output", str(tmp_path / "out"), "--", "echo", "ok"])
    assert result.exit_code == 0, result.output
    assert agent_cli._OUTCOME_HEADLINES["error"] in result.output
    assert "Finished with outcome: error" not in result.output


def test_bundle_error_is_echoed_to_operator(monkeypatch, tmp_path):
    # A --bundle failure is captured on the result; the CLI must tell the
    # operator the bundle does not exist instead of exiting silently.
    mock_result = AgentLoopResult(
        run_dir=tmp_path / "r",
        state=AgentState(ticket="T1"),
        report_path=tmp_path / "r" / "agent_report.md",
        outcome="converged",
        recommended_action="done",
        bundle_error="redaction config missing",
    )
    monkeypatch.setattr(agent_cli, "run_agent_loop", MagicMock(return_value=mock_result))

    runner = CliRunner()
    result = runner.invoke(agent, ["--output", str(tmp_path / "out"), "--", "echo", "ok"])
    assert result.exit_code == 0, result.output
    assert "bundling failed" in result.output
    assert "redaction config missing" in result.output
