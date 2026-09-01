"""CLI wiring for the ``aorta agent`` namespace and ``aorta agent mitigate``."""

from __future__ import annotations

from unittest.mock import MagicMock

import click
from click.testing import CliRunner

import aorta.cli.agent_mitigate as mitigate_cli
from aorta.agent.loop import AgentLoopResult
from aorta.agent.state import AgentState
from aorta.cli.agent import agent
from aorta.cli.agent_mitigate import mitigate

# The pre-namespace form; every token before the user command is dash-prefixed,
# which is what the group's legacy-invocation check keys off.
LEGACY_ARGV = ["--output", "/tmp/o", "--", "echo", "ok"]


def _mock_result(tmp_path, **overrides) -> AgentLoopResult:
    defaults = {
        "run_dir": tmp_path / "r",
        "state": AgentState(ticket="T1"),
        "report_path": tmp_path / "r" / "agent_report.md",
        "outcome": "converged",
        "recommended_action": "done",
    }
    return AgentLoopResult(**{**defaults, **overrides})


def _patch_loop(monkeypatch, tmp_path, **overrides) -> MagicMock:
    """Stub out the agent loop and hand back the mock for call assertions."""
    mock_loop = MagicMock(return_value=_mock_result(tmp_path, **overrides))
    monkeypatch.setattr(mitigate_cli, "run_agent_loop", mock_loop)
    return mock_loop


def test_agent_cli_invokes_loop(monkeypatch, tmp_path):
    mock_loop = _patch_loop(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        agent,
        [
            "mitigate",
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
    result = runner.invoke(mitigate, ["--output", "/tmp/o", "echo", "hi"])
    assert result.exit_code != 0
    assert "separator" in result.output.lower() or "Usage" in result.output


def test_agent_leaked_flag_usage_error_names_agent_not_probe():
    # A dash-prefixed trailing token (a leaked flag) trips
    # validate_trailing_argv; the usage hint must name the invoked command
    # (`aorta agent mitigate`), never the shared-helper default `aorta probe`.
    runner = CliRunner()
    result = runner.invoke(mitigate, ["--output", "/tmp/o", "--", "-c"])
    assert result.exit_code != 0
    assert "aorta agent mitigate" in result.output
    assert "aorta probe" not in result.output


def test_error_outcome_has_dedicated_headline(monkeypatch, tmp_path):
    # run_agent_loop can return outcome="error" from its generic exception
    # handler; the CLI must show a specific headline, not the generic fallback.
    _patch_loop(monkeypatch, tmp_path, outcome="error", recommended_action="inspect logs")

    runner = CliRunner()
    result = runner.invoke(mitigate, ["--output", str(tmp_path / "out"), "--", "echo", "ok"])
    assert result.exit_code == 0, result.output
    assert mitigate_cli._OUTCOME_HEADLINES["error"] in result.output
    assert "Finished with outcome: error" not in result.output


def test_bundle_error_is_echoed_to_operator(monkeypatch, tmp_path):
    # A --bundle failure is captured on the result; the CLI must tell the
    # operator the bundle does not exist instead of exiting silently.
    _patch_loop(monkeypatch, tmp_path, bundle_error="redaction config missing")

    runner = CliRunner()
    result = runner.invoke(mitigate, ["--output", str(tmp_path / "out"), "--", "echo", "ok"])
    assert result.exit_code == 0, result.output
    assert "bundling failed" in result.output
    assert "redaction config missing" in result.output


# --- namespace dispatch ------------------------------------------------------


def test_agent_group_lists_registered_agents():
    runner = CliRunner()
    result = runner.invoke(agent, ["--help"])
    assert result.exit_code == 0, result.output
    assert "mitigate" in result.output
    # The group help must not be the old flat command's help.
    assert "--llm-backend" not in result.output


def test_agent_group_dispatches_a_plugin_agent(monkeypatch):
    # `aorta agent <name>` resolves through the registry rather than a
    # hard-coded add_command list, so a third-party entry-point agent is
    # dispatchable without touching core.
    @click.command(name="autopsy")
    def _autopsy() -> None:
        click.echo("autopsy ran")

    class _Dist:
        name = "fake_plugin"

    class _EntryPoint:
        name = "autopsy"
        dist = _Dist()

        def load(self):
            return _autopsy

    monkeypatch.setattr("aorta.registry.agents.entry_points", lambda group: [_EntryPoint()])

    runner = CliRunner()
    result = runner.invoke(agent, ["autopsy"])
    assert result.exit_code == 0, result.output
    assert "autopsy ran" in result.output


def test_unknown_agent_names_the_available_ones():
    runner = CliRunner()
    result = runner.invoke(agent, ["nope"])
    assert result.exit_code != 0
    assert "unknown agent 'nope'" in result.output
    assert "mitigate" in result.output


# --- deprecation shim on the pre-namespace form ------------------------------


def test_legacy_bare_form_still_runs_the_loop(monkeypatch, tmp_path):
    mock_loop = _patch_loop(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        agent, ["--output", str(tmp_path / "out"), "--ticket", "T1", "--", "echo", "ok"]
    )
    assert result.exit_code == 0, result.output
    config = mock_loop.call_args[0][0]
    assert config.subprocess_argv == ("echo", "ok")
    assert config.ticket == "T1"


def test_legacy_bare_form_warns_on_stderr_only(monkeypatch, tmp_path):
    # Scripted callers parse stdout; the notice must not contaminate it.
    _patch_loop(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(agent, LEGACY_ARGV)
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.stderr
    assert "aorta agent mitigate" in result.stderr
    assert "deprecated" not in result.stdout


def test_namespaced_form_does_not_warn(monkeypatch, tmp_path):
    _patch_loop(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(agent, ["mitigate", *LEGACY_ARGV])
    assert result.exit_code == 0, result.output
    assert "deprecated" not in result.stderr


def test_group_help_is_not_treated_as_a_legacy_invocation():
    runner = CliRunner()
    for token in ("--help", "-h"):
        result = runner.invoke(agent, [token])
        assert "deprecated" not in result.stderr, token


def test_legacy_usage_errors_point_at_the_replacement():
    # The old form still fails the same way (exit 2, same guardrail), but the
    # usage hint names the command the user should migrate to.
    runner = CliRunner()
    result = runner.invoke(agent, ["--output", "/tmp/o", "echo", "hi"])
    assert result.exit_code == 2
    assert "deprecated" in result.stderr
    assert "aorta agent mitigate" in result.output
