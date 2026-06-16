"""Closed-loop agent tests with mocked run_recipe."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aorta.agent.loop import AgentConfig, run_agent_loop
from aorta.agent.policy import AgentPolicy


@pytest.fixture
def mock_run_recipe(monkeypatch, tmp_path):
    # Default to a per-test path under tmp_path so the loop's agent_log.jsonl /
    # report writes stay hermetic; individual tests still override return_value.
    mock = MagicMock(return_value=tmp_path / "agent-run")
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


def _baseline_pass_summaries(_run_dir):
    return [
        {
            "cell_name": "none-none",
            "verdict": "pass",
            "failure_detectors_fired": [],
            "capture": {},
        }
    ]


def test_baseline_pass_short_circuits_before_proposer(
    mock_run_recipe, tmp_path, monkeypatch
):
    """A passing baseline must stop the loop before the proposer/budget run.

    Baseline-pass is deterministic from probe results, so with an LLM/custom
    backend it must short-circuit -- never spending a propose() call (tokens)
    nor mis-reporting policy_stop when the iteration budget is tight.
    """
    import aorta.agent.loop as loop_mod

    mock_run_recipe.return_value = tmp_path / "run"
    monkeypatch.setattr(loop_mod, "_read_cell_summaries", _baseline_pass_summaries)

    class _ExplodingProposer:
        called = False

        def propose(self, **kwargs):
            type(self).called = True
            raise AssertionError("proposer must not be called when baseline passes")

    proposer = _ExplodingProposer()
    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="BASELINE-LLM",
        subprocess_argv=("true",),
        # Tight budget: a non-short-circuited loop would hit the budget check
        # and resolve to policy_stop instead of baseline_pass.
        policy=AgentPolicy(max_iterations=1),
        mitigations_allowlist=("none", "tf32_off"),
        llm_backend="litellm",
    )
    result = run_agent_loop(config, proposer=proposer)

    assert result.outcome == "baseline_pass"
    assert _ExplodingProposer.called is False


def test_whitespace_ticket_aligns_slug_and_recipe(mock_run_recipe, tmp_path, monkeypatch):
    """A whitespace-only ticket normalises to None so the agent slug and the
    probe recipe ticket agree, and the audit log keeps raw + slug distinct."""
    import json

    import aorta.agent.loop as loop_mod
    from aorta.triage.output import NO_TICKET_SLUG

    mock_run_recipe.return_value = tmp_path / "run"
    monkeypatch.setattr(loop_mod, "_read_cell_summaries", _baseline_pass_summaries)

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="   ",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=2),
        mitigations_allowlist=("none",),
    )
    run_agent_loop(config)

    # Probe recipe ticket normalised to None -> resolve_run_dir uses the slug.
    recipe = mock_run_recipe.call_args_list[-1][0][0]
    assert not recipe.ticket
    # session_start log records the raw ticket (None) and the slug separately,
    # under the same NO_TICKET_SLUG dir the probe would resolve to.
    log_path = tmp_path / "out" / NO_TICKET_SLUG / "agent_log.jsonl"
    assert log_path.is_file()
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    start = next(e for e in events if e["type"] == "session_start")
    assert start["ticket"] is None
    assert start["ticket_slug"] == NO_TICKET_SLUG


def test_bundle_receives_raw_ticket(mock_run_recipe, tmp_path, monkeypatch):
    """--bundle must pass the raw operator ID so the manifest keeps it (not the slug)."""
    import aorta.agent.loop as loop_mod
    import aorta.bundle as bundle_mod
    import aorta.probe.bundle_hook as hook_mod

    mock_run_recipe.return_value = tmp_path / "run"
    monkeypatch.setattr(loop_mod, "_read_cell_summaries", _baseline_pass_summaries)

    captured = {}

    def fake_bundle(run_dir, *, ticket=None, redactor=None, **kw):
        captured["ticket"] = ticket
        return tmp_path / "bundle.tar.gz"

    monkeypatch.setattr(bundle_mod, "bundle_run_dir", fake_bundle)
    monkeypatch.setattr(hook_mod, "build_redactor_from_recipe", lambda *a, **k: None)

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="TKT/with space",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=2),
        mitigations_allowlist=("none",),
        run_bundle=True,
    )
    run_agent_loop(config)
    assert captured["ticket"] == "TKT/with space"


def test_proposer_exception_still_writes_report(mock_run_recipe, tmp_path, monkeypatch):
    """A backend error from the proposer must not skip the report/audit trail."""
    import json

    import aorta.agent.loop as loop_mod

    mock_run_recipe.return_value = tmp_path / "run"
    monkeypatch.setattr(
        loop_mod,
        "_read_cell_summaries",
        lambda _d: [
            {
                "cell_name": "none-none",
                "verdict": "fail",
                "failure_detectors_fired": ["tier1:exit_nonzero"],
                "capture": {},
            }
        ],
    )

    class BoomProposer:
        def propose(self, **kwargs):
            raise RuntimeError("litellm provider exploded")

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="BOOM-1",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=3),
        mitigations_allowlist=("none", "tf32_off"),
    )
    result = run_agent_loop(config, proposer=BoomProposer())

    assert result.outcome == "error"
    assert result.report_path is not None and result.report_path.is_file()
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "agent_log.jsonl").read_text().splitlines()
    ]
    assert any(e["type"] == "error" and "litellm" in e["reason"] for e in events)


def test_registry_error_logs_matching_event_type(mock_run_recipe, tmp_path, monkeypatch):
    """An UnknownMitigationError surfaces outcome=registry_error and the audit
    log event type must match it (not the generic "error"), so agent_log.jsonl
    can be filtered by the same key the operator sees."""
    import json

    import aorta.agent.loop as loop_mod
    from aorta.registry.errors import UnknownMitigationError

    mock_run_recipe.return_value = tmp_path / "run"
    monkeypatch.setattr(
        loop_mod,
        "_read_cell_summaries",
        lambda _d: [
            {
                "cell_name": "none-none",
                "verdict": "fail",
                "failure_detectors_fired": ["tier1:exit_nonzero"],
                "capture": {},
            }
        ],
    )

    class UnknownMitigationProposer:
        def propose(self, **kwargs):
            raise UnknownMitigationError("totally_unknown")

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="REG-1",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=3),
        mitigations_allowlist=("none", "tf32_off"),
    )
    result = run_agent_loop(config, proposer=UnknownMitigationProposer())

    assert result.outcome == "registry_error"
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "agent_log.jsonl").read_text().splitlines()
    ]
    assert any(e["type"] == "registry_error" for e in events)
    assert not any(e["type"] == "error" for e in events)


def test_proposer_outside_allowlist_fails_fast(mock_run_recipe, tmp_path, monkeypatch):
    """A registered-but-not-allowlisted mitigation must not be executed.

    validate_step() only checks registry membership; the operator's
    --mitigation allowlist is a loop-level guardrail. A custom proposer
    returning a registered mitigation outside the allowlist must trip
    policy_stop, not silently widen the search.
    """
    import aorta.agent.loop as loop_mod
    from aorta.agent.llm import AgentStep

    mock_run_recipe.return_value = tmp_path / "run"
    monkeypatch.setattr(
        loop_mod,
        "_read_cell_summaries",
        lambda _d: [
            {
                "cell_name": "none-none",
                "verdict": "fail",
                "failure_detectors_fired": ["tier1:exit_nonzero"],
                "capture": {},
            }
        ],
    )

    class OutOfAllowlistProposer:
        def propose(self, **kwargs):
            # hsa_no_sdma is registered (passes validate_step) but is NOT in the
            # --mitigation allowlist below.
            return AgentStep(
                category="unknown",
                hypothesis="try something off-allowlist",
                next_mitigations=["hsa_no_sdma"],
                confidence=0.9,
                stop=False,
            )

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="ALLOW-1",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=3),
        mitigations_allowlist=("none", "tf32_off"),
    )
    result = run_agent_loop(config, proposer=OutOfAllowlistProposer())

    assert result.outcome == "policy_stop"
    assert "candidate set" in result.recommended_action
    # The off-allowlist mitigation must never reach a probe run: only the
    # initial baseline matrix runs.
    assert mock_run_recipe.call_count == 1


def test_bundle_failure_is_surfaced(mock_run_recipe, tmp_path, monkeypatch):
    """A --bundle failure must be reported on the result, not swallowed by a
    default-muted logger (the operator would otherwise assume a bundle exists)."""
    import aorta.agent.loop as loop_mod
    import aorta.bundle as bundle_mod
    import aorta.probe.bundle_hook as hook_mod

    mock_run_recipe.return_value = tmp_path / "run"
    monkeypatch.setattr(loop_mod, "_read_cell_summaries", _baseline_pass_summaries)

    def boom_bundle(*a, **k):
        raise RuntimeError("redaction config missing")

    monkeypatch.setattr(bundle_mod, "bundle_run_dir", boom_bundle)
    monkeypatch.setattr(hook_mod, "build_redactor_from_recipe", lambda *a, **k: None)

    config = AgentConfig(
        output_dir=tmp_path / "out",
        ticket="BND-FAIL",
        subprocess_argv=("true",),
        policy=AgentPolicy(max_iterations=2),
        mitigations_allowlist=("none",),
        run_bundle=True,
    )
    result = run_agent_loop(config)

    assert result.bundle_error is not None
    assert "redaction config missing" in result.bundle_error


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
