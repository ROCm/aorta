"""Closed-loop orchestration: grow probe matrix, call ``run_recipe``, repeat."""

from __future__ import annotations

import json
import logging
import time

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.agent.llm import AgentStep, LLMProposer, StopReason, _BASELINE_CELL, make_proposer
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.agent.report import write_agent_report
from aorta.agent.state import AgentState, append_log_event, wake
from aorta.probe.recipe_builder import build_probe_recipe_from_dict
from aorta.registry import load_mitigations
from aorta.registry.errors import UnknownMitigationError
from aorta.triage.output import NO_TICKET_SLUG, safe_slug
from aorta.triage.recipe import load_recipe
from aorta.triage.runner import run_recipe

log = logging.getLogger(__name__)

_BASELINE_MITIGATION = "none"
_BASELINE_DIAGNOSTIC = "none"


@dataclass
class AgentConfig:
    """Inputs for :func:`run_agent_loop`."""

    output_dir: Path
    ticket: str | None
    subprocess_argv: tuple[str, ...]
    symptom: str | None = None
    policy: AgentPolicy = field(default_factory=AgentPolicy)
    llm_backend: str = "fake"
    llm_model: str = "gpt-4o-mini"
    mitigations_allowlist: tuple[str, ...] | None = None
    recipe_path: Path | None = None
    dry_run: bool = False
    run_bundle: bool = False


@dataclass
class AgentLoopResult:
    """Outcome of a completed (or budget-stopped) agent loop."""

    run_dir: Path
    state: AgentState
    report_path: Path | None
    outcome: str
    recommended_action: str


def _ticket_slug(ticket: str | None) -> str:
    if ticket is None or not str(ticket).strip():
        return NO_TICKET_SLUG
    return safe_slug(ticket)


def _run_dir(config: AgentConfig) -> Path:
    return config.output_dir / _ticket_slug(config.ticket)


def _recipe_template_dict(config: AgentConfig) -> dict[str, Any]:
    """Load optional probe recipe YAML; return extra keys to merge into each run."""
    if config.recipe_path is None:
        return {}
    # Validate via the normal loader (sidecar merge, schema checks).
    recipe = load_recipe(
        config.recipe_path,
        sidecar_files=list(config.policy.sidecar_files) or None,
    )
    if recipe.probe_extras is None:
        raise ValueError(f"{config.recipe_path} is not a probe-mode recipe")
    raw = yaml.safe_load(config.recipe_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config.recipe_path}: expected a YAML mapping")
    template: dict[str, Any] = {}
    for key in (
        "trials",
        "diagnostic_axis",
        "timeout_per_trial",
        "env_passthrough_mode",
        "step_time_regex",
        "collect_paths",
        "custom_patterns",
        "hang_window_sec",
        "hang_grace_period_at_start",
    ):
        if key in raw:
            template[key] = raw[key]
    template["_mitigation_axis_order"] = list(recipe.probe_extras.mitigation_axis)
    if recipe.ticket:
        template["ticket"] = recipe.ticket
    return template


def _list_candidate_mitigations(
    config: AgentConfig,
    recipe_template: dict[str, Any],
) -> list[str]:
    if config.mitigations_allowlist:
        return list(config.mitigations_allowlist)
    axis = recipe_template.get("_mitigation_axis_order")
    if isinstance(axis, list) and axis:
        return [str(m) for m in axis if m != _BASELINE_MITIGATION]
    extra = list(config.policy.sidecar_files) if config.policy.sidecar_files else None
    names = sorted(load_mitigations(extra_files=extra).keys())
    return [n for n in names if n != _BASELINE_MITIGATION]


def _build_probe_recipe_dict(
    config: AgentConfig,
    mitigation_axis: list[str],
    recipe_template: dict[str, Any],
) -> dict[str, Any]:
    ticket = config.ticket or recipe_template.get("ticket")
    data: dict[str, Any] = {
        "schema_version": 1,
        "mode": "probe",
        "ticket": ticket,
        "trials": recipe_template.get("trials", 1),
        "mitigation_axis": mitigation_axis,
        "diagnostic_axis": recipe_template.get("diagnostic_axis", [_BASELINE_DIAGNOSTIC]),
    }
    for key in (
        "timeout_per_trial",
        "env_passthrough_mode",
        "step_time_regex",
        "collect_paths",
        "custom_patterns",
        "hang_window_sec",
        "hang_grace_period_at_start",
    ):
        if key in recipe_template:
            data[key] = recipe_template[key]
    return data


def _read_cell_summaries(run_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not run_dir.is_dir():
        return summaries
    for cell_dir in sorted(run_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        result_path = cell_dir / "trial_0" / "result.json"
        if not result_path.is_file():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append(
            {
                "cell_name": data.get("cell_name", cell_dir.name),
                "verdict": data.get("verdict"),
                "failure_detectors_fired": data.get("failure_detectors_fired") or [],
                "warn_detectors_fired": data.get("warn_detectors_fired") or [],
                "capture": data.get("capture") or {},
                "exit_code": data.get("exit_code"),
            }
        )
    return summaries


def _baseline_passed(summaries: list[dict[str, Any]]) -> bool:
    for row in summaries:
        if row.get("cell_name") == _BASELINE_CELL and row.get("verdict") == "pass":
            return True
    return False


def _resolve_stop_outcome(
    step: AgentStep,
    summaries: list[dict[str, Any]],
) -> tuple[str, str]:
    """Map a proposer stop step to outcome label + operator-facing message."""
    reason: StopReason | None = step.stop_reason
    if reason is None and step.stop:
        if _baseline_passed(summaries):
            reason = "baseline_pass"
        elif not step.next_mitigations and "No remaining" in step.hypothesis:
            reason = "exhausted_candidates"
        else:
            reason = "agent_requested"

    if reason == "baseline_pass":
        return (
            "baseline_pass",
            "Baseline cell (none-none) passed. The repro succeeds without "
            "mitigations; no search was run.",
        )
    if reason == "exhausted_candidates":
        return (
            "exhausted_candidates",
            "No further registered mitigations to try (already attempted or "
            "not in the allowlist). Inspect failure detectors in "
            "agent_report.md or run a manual probe matrix.",
        )
    return (
        "agent_stop",
        step.hypothesis
        or (
            "Agent stopped search. Inspect failure detectors and extend "
            "mitigations allowlist or run a manual probe matrix."
        ),
    )


def _find_winning_mitigation(summaries: list[dict[str, Any]]) -> str | None:
    for row in summaries:
        cell = row.get("cell_name") or ""
        if cell == _BASELINE_CELL:
            continue
        if row.get("verdict") == "pass" and "-" in cell:
            return cell.split("-", 1)[0]
    return None


def _execute_probe_matrix(
    config: AgentConfig,
    mitigation_axis: list[str],
    recipe_template: dict[str, Any],
) -> Path:
    """Run (or dry-run) probe recipe; return ticket run directory."""
    recipe_dict = _build_probe_recipe_dict(config, mitigation_axis, recipe_template)
    sidecar = config.policy.sidecar_files or None
    recipe = build_probe_recipe_from_dict(
        recipe_dict,
        sidecar_files=sidecar,
        source_path=None,
        source_sha256=None,
    )
    return run_recipe(
        recipe,
        output_dir=config.output_dir,
        dry_run=config.dry_run,
        layout="flat_resume",
        resume_existing=True,
        subprocess_argv=config.subprocess_argv,
    )


def run_agent_loop(
    config: AgentConfig,
    *,
    proposer: LLMProposer | None = None,
) -> AgentLoopResult:
    """Run the closed-loop mitigation search."""
    recipe_template = _recipe_template_dict(config)
    if config.ticket is None and recipe_template.get("ticket"):
        ticket_slug = safe_slug(str(recipe_template["ticket"]))
    else:
        ticket_slug = _ticket_slug(config.ticket)
    run_dir = config.output_dir / ticket_slug
    state = wake(run_dir, ticket=ticket_slug)
    if proposer is None:
        proposer = make_proposer(config.llm_backend, model=config.llm_model)

    candidates = _list_candidate_mitigations(config, recipe_template)
    mitigation_axis: list[str] = [_BASELINE_MITIGATION]
    for m in state.tried_mitigations:
        if m != _BASELINE_MITIGATION and m not in mitigation_axis:
            mitigation_axis.append(m)

    start_time = time.monotonic()
    append_log_event(
        run_dir,
        "session_start",
        {
            "ticket": ticket_slug,
            "argv": list(config.subprocess_argv),
            "symptom": config.symptom,
            "llm_backend": config.llm_backend,
        },
    )

    outcome = "in_progress"
    recommended = "Review agent_report.md and probe cell artifacts."

    try:
        while True:
            if config.policy.max_walltime_sec is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= config.policy.max_walltime_sec:
                    outcome = "walltime_exhausted"
                    recommended = (
                        "Wall-time budget exhausted. Resume with the same ticket "
                        "to continue from flat_resume checkpoints."
                    )
                    break

            config.policy.check_iteration_budget(state.iterations_completed)

            run_dir = _execute_probe_matrix(config, mitigation_axis, recipe_template)
            summaries = _read_cell_summaries(run_dir)
            winner = _find_winning_mitigation(summaries)
            if winner:
                state.winning_mitigation = winner
                state.converged = True
                outcome = "converged"
                recommended = (
                    f"Re-run the repro with mitigation `{winner}` applied "
                    f"(see cell `{winner}-{_BASELINE_DIAGNOSTIC}` probe.env or matrix)."
                )
                append_log_event(
                    run_dir,
                    "converged",
                    {"winning_mitigation": winner},
                )
                break

            step = proposer.propose(
                symptom=config.symptom,
                cell_summaries=summaries,
                candidates=candidates,
                tried=state.tried_mitigations,
            )
            step = config.policy.validate_step(step)
            state.last_category = step.category
            state.last_hypothesis = step.hypothesis
            append_log_event(
                run_dir,
                "llm_step",
                {
                    "category": step.category,
                    "hypothesis": step.hypothesis,
                    "next_mitigations": step.next_mitigations,
                    "confidence": step.confidence,
                    "stop": step.stop,
                    "stop_reason": step.stop_reason,
                },
            )

            if step.stop or not step.next_mitigations:
                outcome, recommended = _resolve_stop_outcome(step, summaries)
                append_log_event(
                    run_dir,
                    "search_stopped",
                    {"outcome": outcome, "stop_reason": step.stop_reason},
                )
                break

            pending = config.policy.pending_approvals(step.next_mitigations)
            if pending:
                outcome = "approval_required"
                recommended = (
                    f"Approval required for mitigations: {pending}. "
                    "Re-run without --require-approval after operator ack."
                )
                append_log_event(
                    run_dir,
                    "approval_required",
                    {"mitigations": pending},
                )
                break

            for mitigation in step.next_mitigations:
                if mitigation in mitigation_axis:
                    continue
                mitigation_axis.append(mitigation)
                state.tried_mitigations.append(mitigation)
                append_log_event(
                    run_dir,
                    "mitigation_tried",
                    {"mitigation": mitigation},
                )

            state.iterations_completed += 1
            append_log_event(
                run_dir,
                "iteration_complete",
                {"iteration": state.iterations_completed},
            )

    except PolicyViolation as exc:
        outcome = "policy_stop"
        recommended = str(exc)
        append_log_event(run_dir, "policy_stop", {"reason": str(exc)})
    except UnknownMitigationError as exc:
        outcome = "registry_error"
        recommended = str(exc)
        append_log_event(run_dir, "error", {"reason": str(exc)})

    report_path = None
    if not config.dry_run:
        summaries = _read_cell_summaries(run_dir)
        report_path = write_agent_report(
            run_dir,
            state=state,
            cell_summaries=summaries,
            outcome=outcome,
            recommended_action=recommended,
        )

    if config.run_bundle and not config.dry_run and report_path is not None:
        try:
            from aorta.bundle import bundle_run_dir
            from aorta.probe.bundle_hook import build_redactor_from_recipe
            from aorta.triage.recipe import load_recipe

            resolved = run_dir / "recipe.resolved.yaml"
            redactor = None
            if resolved.is_file():
                recipe = load_recipe(resolved)
                if recipe.probe_extras and recipe.probe_extras.redaction:
                    redactor = build_redactor_from_recipe(recipe.probe_extras.redaction)
            bundle_run_dir(run_dir, redactor=redactor)
            log.info("Wrote bundle for %s", run_dir)
        except Exception as exc:  # pragma: no cover - bundle is best-effort
            log.warning("Bundle step skipped: %s", exc)

    return AgentLoopResult(
        run_dir=run_dir,
        state=state,
        report_path=report_path,
        outcome=outcome,
        recommended_action=recommended,
    )


__all__ = ["AgentConfig", "AgentLoopResult", "run_agent_loop"]
