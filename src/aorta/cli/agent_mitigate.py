"""``aorta agent mitigate`` -- closed-loop mitigation search on opaque user commands.

Thin Click shim: validate argv, call :func:`aorta.agent.loop.run_agent_loop`,
map policy errors to ``ClickException``.

Registered as the ``mitigate`` built-in of the ``aorta.agents`` registry (see
:mod:`aorta.registry.agents`), which is what ``aorta agent`` dispatches through.
This module is reached only via that registry, never imported by
:mod:`aorta.cli.agent` directly.
"""

from __future__ import annotations

from pathlib import Path

import click

from aorta.agent.loop import AgentConfig, run_agent_loop
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.probe.cli_helpers import (
    ProbeUsageError,
    help_token_in_option_zone,
    reject_flag_shaped_value,
    require_double_dash_separator,
    validate_trailing_argv,
)
from aorta.registry import RegistryError
from aorta.run.cli_helpers import configure_verbose_logging

#: Front-door name used in usage hints. The registry dispatches this command as
#: ``mitigate`` under the ``agent`` group, so Click never derives the full path.
COMMAND_LABEL = "aorta agent mitigate"

# Human-readable one-liners keyed by :attr:`AgentLoopResult.outcome`.
_OUTCOME_HEADLINES: dict[str, str] = {
    "baseline_pass": "Baseline passed — no mitigation search needed.",
    "converged": "Mitigation found — repro passes with a non-baseline cell.",
    "exhausted_candidates": "Search stopped — no more mitigations to try.",
    "agent_stop": "Search stopped by the agent.",
    "approval_required": "Paused — operator approval required.",
    "walltime_exhausted": "Stopped — wall-time budget exhausted.",
    "policy_stop": "Stopped — policy limit hit.",
    "registry_error": "Stopped — registry error.",
    "error": "Aborted — unexpected error; see report and agent_log.jsonl.",
}


def _echo_agent_result(result: object) -> None:
    from aorta.agent.loop import AgentLoopResult

    assert isinstance(result, AgentLoopResult)
    headline = _OUTCOME_HEADLINES.get(
        result.outcome,
        f"Finished with outcome: {result.outcome}",
    )
    click.echo(f"Agent outcome: {result.outcome} — {headline}")
    if result.report_path:
        click.echo(f"Wrote {result.report_path}")
    click.echo(result.recommended_action)
    if result.bundle_error:
        # --bundle was requested but failed; the loop only logs a warning
        # (muted by default), so say so explicitly -- the bundle does NOT exist.
        click.echo(
            f"WARNING: --bundle requested but bundling failed: {result.bundle_error}. "
            "No bundle was written.",
            err=True,
        )


def _retarget_probe_usage(message: str) -> str:
    """Rewrite shared probe-helper usage strings for the ``mitigate`` command.

    ``require_double_dash_separator`` lives in ``aorta.probe.cli_helpers``
    and hard-codes ``Usage: aorta probe ...``. Reused verbatim, ``aorta
    agent mitigate`` would print the wrong command name, so we translate at
    the CLI boundary rather than forking that helper.
    (``validate_trailing_argv`` takes an explicit ``command_label``, so
    ``aorta agent mitigate`` is passed there directly; this fallback still
    covers any other probe-helper strings.) The deprecated bare ``aorta
    agent -- <cmd>`` form is rewritten to this command before parsing, so it
    is also pointed at the replacement spelling rather than at itself.
    """
    return message.replace("aorta probe", COMMAND_LABEL)


def _reject_flag_shaped_callback(
    ctx: click.Context, param: click.Parameter, value: object
) -> object:
    if value is None or not isinstance(value, (str, Path)):
        return value
    option_name = f"--{param.name.replace('_', '-')}" if param.name else "<option>"
    try:
        reject_flag_shaped_value(option_name, str(value))
    except ProbeUsageError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc
    return value


class _MitigateCommand(click.Command):
    """Require ``--`` before the user command (same rule as ``aorta probe``)."""

    _BYPASS_TOKENS: frozenset[str] = frozenset({"--help", "-h"})

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not ctx.resilient_parsing and not help_token_in_option_zone(
            args, self._value_taking_option_tokens(), self._BYPASS_TOKENS
        ):
            try:
                require_double_dash_separator(args)
            except ProbeUsageError as exc:
                raise click.UsageError(_retarget_probe_usage(str(exc)), ctx=ctx) from exc
        return super().parse_args(ctx, args)

    def _value_taking_option_tokens(self) -> frozenset[str]:
        tokens: set[str] = set()
        for param in self.params:
            if not isinstance(param, click.Option):
                continue
            if param.is_flag or param.count:
                continue
            for opt in param.opts:
                if opt.startswith("--"):
                    tokens.add(opt)
        return frozenset(tokens)


@click.command(
    name="mitigate",
    cls=_MitigateCommand,
    context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False},
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("agent_results"),
    show_default=True,
    callback=_reject_flag_shaped_callback,
    help="Top-level output directory for probe cells and agent_log.jsonl.",
)
@click.option(
    "--ticket",
    default=None,
    callback=_reject_flag_shaped_callback,
    help="Ticket ID for output grouping (recommended for bundle handoff).",
)
@click.option(
    "--recipe",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    callback=_reject_flag_shaped_callback,
    help=(
        "Probe-mode recipe YAML: supplies custom_patterns, timeout, "
        "diagnostic_axis, and mitigation search order. Mitigation axis "
        "still grows iteratively."
    ),
)
@click.option(
    "--symptom",
    default=None,
    help="Optional free-text symptom hint for the agent proposer.",
)
@click.option(
    "--max-iterations",
    default=8,
    show_default=True,
    type=int,
    help="Maximum agent loop iterations (mitigation proposals).",
)
@click.option(
    "--max-walltime-sec",
    default=None,
    type=float,
    help="Optional wall-clock budget for the whole loop.",
)
@click.option(
    "--llm-backend",
    # Hard-coded rather than read from aorta.agent.llm: this decorator runs at
    # import time, and `aorta --help` must not pay for that module's imports.
    # tests/agent/test_llm_providers.py fails if this list and
    # CHAT_PROVIDER_BACKENDS drift apart.
    type=click.Choice(["fake", "litellm", "openai", "vllm"]),
    default="fake",
    show_default=True,
    help=(
        "Proposer backend: fake (offline, no network). litellm, openai and vllm "
        "are configured by the shared chat provider settings and need "
        "amd-aorta[chat-cli]; litellm also still works on amd-aorta[agent] alone."
    ),
)
@click.option(
    "--llm-model",
    default=None,
    help=(
        "Override the model name for the selected backend. Defaults to whatever "
        "the chat provider settings configure (gpt-4o-mini on the standalone "
        "litellm path)."
    ),
)
@click.option(
    "--mitigation",
    "mitigation_allowlist",
    multiple=True,
    help="Restrict search to these registered mitigation names (repeatable).",
)
@click.option(
    "--mitigations-file",
    "mitigation_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="JSON sidecar with extra mitigations (repeatable).",
)
@click.option(
    "--require-approval",
    is_flag=True,
    help="Pause before running mitigations flagged as needing operator ack.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate and plan cells without executing subprocess trials.",
)
@click.option(
    "--bundle",
    "run_bundle",
    is_flag=True,
    help="Run aorta bundle on the ticket dir after the loop (needs redaction: in recipe).",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Stream progress to stderr. -v = INFO, -vv = DEBUG.",
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def mitigate(
    output: Path,
    ticket: str | None,
    recipe: Path | None,
    symptom: str | None,
    max_iterations: int,
    max_walltime_sec: float | None,
    llm_backend: str,
    llm_model: str | None,
    mitigation_allowlist: tuple[str, ...],
    mitigation_files: tuple[Path, ...],
    require_approval: bool,
    dry_run: bool,
    run_bundle: bool,
    verbose: int,
    argv: tuple[str, ...],
) -> None:
    """Closed-loop mitigation search via the probe engine."""
    configure_verbose_logging(verbose)
    try:
        clean_argv = validate_trailing_argv(argv, command_label=COMMAND_LABEL)
        policy = AgentPolicy(
            max_iterations=max_iterations,
            max_walltime_sec=max_walltime_sec,
            require_approval=require_approval,
            sidecar_files=tuple(mitigation_files),
        )
        config = AgentConfig(
            output_dir=output,
            ticket=ticket,
            subprocess_argv=clean_argv,
            symptom=symptom,
            policy=policy,
            llm_backend=llm_backend,
            llm_model=llm_model,
            mitigations_allowlist=mitigation_allowlist or None,
            recipe_path=recipe,
            dry_run=dry_run,
            run_bundle=run_bundle,
        )
        result = run_agent_loop(config)
    except (
        ProbeUsageError,
        PolicyViolation,
        ValueError,
        LookupError,
        RegistryError,
        RuntimeError,
    ) as exc:
        # ProbeUsageError/PolicyViolation -- argv + policy guardrails.
        # ValueError    -- non-probe/non-mapping --recipe, RecipeSchemaError/RecipeCellError.
        # LookupError   -- UnknownMitigationError/UnknownEnvironmentError (KeyError subclasses).
        # RegistryError -- malformed sidecar / collision with built-ins or plugins.
        # RuntimeError  -- launch-mode validation failure.
        raise click.ClickException(_retarget_probe_usage(str(exc))) from exc

    if dry_run:
        click.echo("Dry-run complete: planned probe cells printed above; no artifacts written.")
        return
    _echo_agent_result(result)


__all__ = ["COMMAND_LABEL", "mitigate"]
