"""Closed-loop orchestration: grow probe matrix, call ``run_recipe``, repeat."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aorta.agent.llm import _BASELINE_CELL, AgentStep, LLMProposer, StopReason, make_proposer
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.agent.report import write_agent_report
from aorta.agent.state import (
    AgentState,
    aggregate_cell_verdict,
    append_log_event,
    read_trial_results,
    wake,
    winning_mitigation,
)
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
    # Set when --bundle was requested but bundling failed; surfaced to the
    # operator by the CLI so a silent log.warning can't imply a bundle exists.
    bundle_error: str | None = None


def _ticket_slug(ticket: str | None) -> str:
    if ticket is None or not str(ticket).strip():
        return NO_TICKET_SLUG
    return safe_slug(ticket)


def _resolve_raw_ticket(
    config: AgentConfig, recipe_template: dict[str, Any]
) -> str | None:
    """Operator ticket ID (CLI wins, then recipe), or None when absent.

    Whitespace-only / empty values normalise to None so the raw ID and the
    filesystem slug agree EVERYWHERE: ``run_recipe`` slugs the recipe's
    ``ticket`` via ``resolve_run_dir`` while the agent slugs this same value
    for its log/report dir, so passing the normalised raw ID to both keeps
    them pointed at one directory. Slugging is a filesystem concern, never an
    ID rewrite -- logs, the audit trail, the report, and the bundle manifest
    all keep the raw ID.
    """
    for candidate in (config.ticket, recipe_template.get("ticket")):
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    return None


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
        # Carried forward so the generated probe recipe (and its
        # recipe.resolved.yaml) keeps redaction config for the --bundle step.
        "redaction",
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
    ticket: str | None,
    mitigation_axis: list[str],
    recipe_template: dict[str, Any],
) -> dict[str, Any]:
    # ``ticket`` is the already-resolved raw operator ID (see
    # _resolve_raw_ticket). Passing it verbatim keeps the probe's
    # resolve_run_dir slug aligned with the agent's log/report dir.
    # Always include the baseline diagnostic so a canonical none-none baseline
    # cell exists; baseline-pass / winner detection both key off it. A recipe
    # diagnostic_axis that omits "none" would otherwise have no no-op baseline.
    diagnostic_axis = list(recipe_template.get("diagnostic_axis", [_BASELINE_DIAGNOSTIC]))
    if _BASELINE_DIAGNOSTIC not in diagnostic_axis:
        diagnostic_axis = [_BASELINE_DIAGNOSTIC, *diagnostic_axis]
    data: dict[str, Any] = {
        "schema_version": 1,
        "mode": "probe",
        "ticket": ticket,
        "trials": recipe_template.get("trials", 1),
        "mitigation_axis": mitigation_axis,
        "diagnostic_axis": diagnostic_axis,
    }
    for key in (
        "timeout_per_trial",
        "env_passthrough_mode",
        "step_time_regex",
        "collect_paths",
        "custom_patterns",
        "hang_window_sec",
        "hang_grace_period_at_start",
        "redaction",
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
        trial_results = read_trial_results(cell_dir)
        if not trial_results:
            continue
        # Union detectors across trials and surface the first failing trial
        # for capture/exit evidence so a single bad trial in an otherwise
        # passing cell is not hidden behind trial_0.
        failure_detectors: list[str] = []
        warn_detectors: list[str] = []
        for data in trial_results:
            for det in data.get("failure_detectors_fired") or []:
                if det not in failure_detectors:
                    failure_detectors.append(det)
            for det in data.get("warn_detectors_fired") or []:
                if det not in warn_detectors:
                    warn_detectors.append(det)
        evidence = next(
            (d for d in trial_results if d.get("verdict") not in (None, "pass")),
            trial_results[0],
        )
        summaries.append(
            {
                "cell_name": trial_results[0].get("cell_name", cell_dir.name),
                "verdict": aggregate_cell_verdict(trial_results),
                "failure_detectors_fired": failure_detectors,
                "warn_detectors_fired": warn_detectors,
                "capture": evidence.get("capture") or {},
                "exit_code": evidence.get("exit_code"),
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
) -> tuple[str, str, StopReason]:
    """Map a proposer stop step to (outcome label, operator message, resolved reason).

    The third element is the inferred ``StopReason`` -- the proposer may set
    ``stop=True`` without a ``stop_reason``, so we derive one here and return
    it so the audit log records the same reason that drove ``outcome``
    instead of a bare ``None``.
    """
    reason: StopReason | None = step.stop_reason
    if reason is None:
        if _baseline_passed(summaries):
            reason = "baseline_pass"
        elif not step.next_mitigations and "No remaining" in step.hypothesis:
            reason = "exhausted_candidates"
        else:
            reason = "agent_requested"
    elif reason == "baseline_pass" and not _baseline_passed(summaries):
        # baseline_pass is a deterministic probe verdict, not the proposer's
        # call. run_agent_loop short-circuits a genuinely passing baseline
        # before the proposer is consulted, so reaching here with an
        # unconfirmed baseline_pass means none-none did NOT pass. Downgrade to
        # an agent-requested stop rather than letting a misbehaving proposer
        # falsely report success and end the search early.
        reason = "agent_requested"

    if reason == "baseline_pass":
        return (
            "baseline_pass",
            "Baseline cell (none-none) passed. The repro succeeds without "
            "mitigations; no search was run.",
            reason,
        )
    if reason == "exhausted_candidates":
        return (
            "exhausted_candidates",
            "No further registered mitigations to try (already attempted or "
            "not in the allowlist). Inspect failure detectors in "
            "agent_report.md or run a manual probe matrix.",
            reason,
        )
    return (
        "agent_stop",
        step.hypothesis
        or (
            "Agent stopped search. Inspect failure detectors and extend "
            "mitigations allowlist or run a manual probe matrix."
        ),
        reason or "agent_requested",
    )


def _find_winning_mitigation(summaries: list[dict[str, Any]]) -> str | None:
    for row in summaries:
        win = winning_mitigation(row.get("cell_name") or "", row.get("verdict"))
        if win is not None:
            return win
    return None


def _execute_probe_matrix(
    config: AgentConfig,
    ticket: str | None,
    mitigation_axis: list[str],
    recipe_template: dict[str, Any],
) -> Path:
    """Run (or dry-run) probe recipe; return ticket run directory."""
    recipe_dict = _build_probe_recipe_dict(ticket, mitigation_axis, recipe_template)
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
    raw_ticket = _resolve_raw_ticket(config, recipe_template)
    ticket_slug = _ticket_slug(raw_ticket)
    run_dir = config.output_dir / ticket_slug
    # Record the raw operator ID (not the slug) so reports/logs show the real
    # ticket; fall back to the slug only for the no-ticket case.
    state = wake(run_dir, ticket=raw_ticket or ticket_slug)
    if proposer is None:
        proposer = make_proposer(config.llm_backend, model=config.llm_model)

    candidates = _list_candidate_mitigations(config, recipe_template)
    mitigation_axis: list[str] = [_BASELINE_MITIGATION]
    for m in state.tried_mitigations:
        if m != _BASELINE_MITIGATION and m not in mitigation_axis:
            mitigation_axis.append(m)

    if config.dry_run:
        # Dry-run is filesystem-free: print the planned probe matrix and return
        # without writing any logs or report. run_recipe(dry_run=True) returns a
        # sentinel Path("."); discard it rather than treating it as a real run
        # dir (otherwise log/report writes land in the caller's cwd). run_dir
        # below is the planned path, kept only for the result -- nothing is
        # written there.
        _execute_probe_matrix(config, raw_ticket, mitigation_axis, recipe_template)
        return AgentLoopResult(
            run_dir=run_dir,
            state=state,
            report_path=None,
            outcome="dry_run",
            recommended_action=(
                "Dry-run only: planned probe cells printed above; "
                "no probe cells executed and no artifacts written."
            ),
        )

    start_time = time.monotonic()
    append_log_event(
        run_dir,
        "session_start",
        {
            "ticket": raw_ticket,
            "ticket_slug": ticket_slug,
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

            run_dir = _execute_probe_matrix(
                config, raw_ticket, mitigation_axis, recipe_template
            )
            summaries = _read_cell_summaries(run_dir)

            # Baseline-pass is fully deterministic from probe results: if the
            # none-none baseline cell passes, the repro succeeds without any
            # mitigation and the search is moot. Short-circuit here -- BEFORE
            # winner detection (a passing {mitigation}-none cell is not a "fix"
            # when nothing was broken), the iteration-budget check, and the
            # proposer call. Otherwise an LLM backend would spend tokens, and
            # an exhausted budget would mis-report policy_stop instead of
            # baseline_pass.
            if _baseline_passed(summaries):
                outcome = "baseline_pass"
                recommended = (
                    "Baseline cell (none-none) passed. The repro succeeds "
                    "without mitigations; no search was run."
                )
                append_log_event(run_dir, "baseline_pass", {})
                break

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

            # Budget bounds the number of PROPOSAL cycles. Checked here -- after
            # executing the current axis and winner detection, before proposing
            # the next mitigation -- so the most recently appended mitigation
            # always gets a chance to run (and converge) before we stop.
            config.policy.check_iteration_budget(state.iterations_completed)

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
                outcome, recommended, resolved_reason = _resolve_stop_outcome(
                    step, summaries
                )
                append_log_event(
                    run_dir,
                    "search_stopped",
                    {"outcome": outcome, "stop_reason": resolved_reason},
                )
                break

            # validate_step() only enforces registry membership + category. The
            # operator's candidate/allowlist (--mitigation, recipe order) is a
            # loop-level guardrail, so enforce it here: a custom/buggy proposer
            # must not run a registered-but-not-allowlisted mitigation. Fail
            # fast -- PolicyViolation is caught below and surfaces as
            # policy_stop -- instead of silently widening the search.
            disallowed = [m for m in step.next_mitigations if m not in candidates]
            if disallowed:
                raise PolicyViolation(
                    f"proposer returned mitigations outside the allowed "
                    f"candidate set: {sorted(disallowed)}. "
                    f"Allowed: {sorted(candidates)}."
                )

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
        append_log_event(run_dir, "registry_error", {"reason": str(exc)})
    except Exception as exc:  # noqa: BLE001 - audit trail must survive any backend error
        # A proposer backend (e.g. LiteLLM network/provider error) or any
        # other unexpected failure must NOT skip the report write below --
        # losing agent_report.md/agent_log.jsonl would erase the audit trail
        # for the run. Convert to a safe stop outcome and fall through.
        outcome = "error"
        recommended = (
            f"Agent loop aborted on an unexpected error ({type(exc).__name__}): "
            f"{exc}. Inspect agent_log.jsonl and probe cell artifacts; the "
            "report below was still written."
        )
        log.exception(
            "agent loop aborted; recording 'error' outcome and emitting report"
        )
        append_log_event(
            run_dir,
            "error",
            {"reason": str(exc), "error_type": type(exc).__name__},
        )

    # Dry-run returns early above, so the search loop always wrote a real
    # run_dir by here -- write the autopsy report unconditionally.
    summaries = _read_cell_summaries(run_dir)
    report_path = write_agent_report(
        run_dir,
        state=state,
        cell_summaries=summaries,
        outcome=outcome,
        recommended_action=recommended,
    )

    bundle_error: str | None = None
    if config.run_bundle and report_path is not None:
        try:
            from aorta.bundle import bundle_run_dir
            from aorta.probe.bundle_hook import build_redactor_from_recipe

            # build_redactor_from_recipe resolves the redaction: block via the
            # parse-only loader (no axis/registry validation), so it works for
            # sidecar-defined mitigations. It prefers the explicit recipe path
            # and falls back to run_dir/recipe.resolved.yaml, returning an
            # IdentityRedactor only when neither carries redaction.
            redactor = build_redactor_from_recipe(config.recipe_path, run_dir)
            # Pass the raw ticket so the manifest records the operator ID;
            # without it bundle_run_dir infers the slug from run_dir.name. When
            # raw_ticket is None it falls back to that inference (no-ticket).
            bundle_run_dir(run_dir, ticket=raw_ticket, redactor=redactor)
            log.info("Wrote bundle for %s", run_dir)
        except Exception as exc:  # noqa: BLE001 - bundle is best-effort, but must be surfaced
            # Don't let a bundle failure pass silently: log.warning is muted by
            # default, so without this the command exits cleanly and the
            # operator assumes the bundle exists. Capture the error on the
            # result so the CLI can tell them it did NOT.
            bundle_error = str(exc)
            log.warning("Bundle step failed: %s", exc)

    return AgentLoopResult(
        run_dir=run_dir,
        state=state,
        report_path=report_path,
        outcome=outcome,
        recommended_action=recommended,
        bundle_error=bundle_error,
    )


__all__ = ["AgentConfig", "AgentLoopResult", "run_agent_loop"]
