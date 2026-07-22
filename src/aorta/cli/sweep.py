"""``aorta sweep`` -- unified matrix runner (merges ``triage`` + ``probe``).

Both legacy commands drove the same engine
(:func:`aorta.triage.runner.run_recipe`); they differed only in how a
trial body is produced -- a built-in **workload** run in-process
(``triage``) vs. an opaque **user command** wrapped as a subprocess
(``probe``). ``aorta sweep`` is the single front door for both flows.

Flow selection is automatic, with no new mandatory flag:

* **Subprocess flow** when a trailing ``-- <command>`` is supplied *or*
  the loaded recipe is ``mode: probe``. Engine knobs:
  ``layout="flat_resume"``, ``resume_existing=True``, ``subprocess_argv=...``.
* **Workload flow** otherwise (``mode: triage`` recipe, default, or the
  flag shim ``--workload ... --mitigation-axis ...``). Engine defaults.

Consistency guards reject mismatches up-front (probe-mode recipe with no
command; a trailing command with a workload recipe) instead of failing
deep in the engine. The mandatory ``--`` separator from ``aorta probe``
is preserved so a stray positional can never be silently swept into the
user command.

The actual flow bodies live in :func:`aorta.cli.triage.execute_triage_run`
and :func:`aorta.cli.probe.execute_probe` -- ``aorta sweep`` and the
deprecated ``aorta triage`` / ``aorta probe`` aliases all reach the same
code, so there is no behavioural drift between front doors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click

from aorta.cli.probe import (
    _print_list_patterns,
    _reject_flag_shaped_callback,
    execute_probe,
)
from aorta.cli.triage import (
    execute_list_environments,
    execute_list_mitigations,
    execute_triage_run,
)
from aorta.run.cli_helpers import configure_verbose_logging, parse_csv
from aorta.triage.recipe import RecipeSchemaError, load_recipe_mapping

_BYPASS_TOKENS: frozenset[str] = frozenset({"--help", "-h"})


def _peek_recipe_mode(path: Path) -> str | None:
    """Best-effort read of a recipe's top-level ``mode`` for flow dispatch.

    Returns ``"probe"``, ``"triage"``, or ``None`` when the file can't be
    parsed / isn't a mapping. ``None`` defers to the trailing-argv signal
    and lets the real loader (inside the chosen flow) raise the canonical,
    fully-validated error -- this peek never becomes the error surface.
    """
    try:
        data = load_recipe_mapping(path)
    except (RecipeSchemaError, OSError):
        return None
    if isinstance(data, dict):
        mode = data.get("mode", "triage")
        if mode == "probe":
            return "probe"
        if mode == "triage":
            return "triage"
        # Unknown/malformed mode (typo, null, non-str): don't guess "triage"
        # and mis-dispatch -- return None so the real loader raises the
        # canonical RecipeSchemaError instead of a misleading sweep usage error.
        return None
    return None


def _bare_positional_before_separator(
    args: Sequence[str], value_taking_options: frozenset[str]
) -> bool:
    """True iff a non-option positional appears before any ``--`` separator.

    Walks ``--opt value`` / ``--opt=value`` / flag tokens the same way
    :func:`aorta.probe.cli_helpers.help_token_in_option_zone` does. The
    first bare positional with no preceding ``--`` is the ambiguous case
    ``aorta probe`` guards against (a stray token silently becoming the
    user command); ``aorta sweep run`` rejects it and tells the user to
    introduce the command with ``--``. A ``--help``/``-h`` in the option
    zone short-circuits so help still renders.
    """
    i = 0
    while i < len(args):
        token = args[i]
        if token in _BYPASS_TOKENS:
            return False
        if token == "--":
            return False
        if token.startswith("--"):
            opt = token.split("=", 1)[0]
            if "=" in token or opt not in value_taking_options:
                i += 1
                continue
            i += 2  # consume the option's value too
            continue
        if token.startswith("-"):
            i += 1
            continue
        return True  # bare positional, no preceding '--'
    return False


class _SweepRunCommand(click.Command):
    """``sweep run`` command requiring ``--`` *only* when a user command is given.

    The workload flow (``aorta sweep run --recipe r.yaml``) has no
    trailing command and needs no separator. The subprocess flow
    (``aorta sweep run --recipe r.yaml -- python repro.py``) does, and a
    bare positional without ``--`` is rejected so it can't be misparsed
    as the user command. This mirrors ``aorta.cli.probe._ProbeCommand``
    but relaxes the always-required separator to the conditional rule.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not ctx.resilient_parsing and _bare_positional_before_separator(
            args, self._value_taking_option_tokens()
        ):
            raise click.UsageError(
                "missing '--' separator before the user command. A workload "
                "run takes no trailing command; a probe run must place the "
                "command after a literal '--'. "
                "Usage: aorta sweep run [options] -- <command> [args...]",
                ctx=ctx,
            )
        return super().parse_args(ctx, args)

    def _value_taking_option_tokens(self) -> frozenset[str]:
        """Long-form options that consume a value (derived from ``self.params``)."""
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


def _reject_triage_only_flags_in_probe_flow(
    *,
    workload: str | None,
    mitigation_axis: str | None,
    environment_axis: str | None,
    trials: int | None,
    steps: int | None,
    baseline_cell: str | None,
    confound_threshold: float | None,
) -> None:
    """Reject workload-flow-only knobs when a subprocess command is supplied."""
    offenders = {
        "--workload": workload,
        "--mitigation-axis": mitigation_axis,
        "--environment-axis": environment_axis,
        "--trials": trials,
        "--steps": steps,
        "--baseline-cell": baseline_cell,
        "--confound-threshold": confound_threshold,
    }
    set_flags = [k for k, v in offenders.items() if v not in (None, "")]
    if set_flags:
        raise click.UsageError(
            f"{', '.join(set_flags)} only apply to the workload flow and "
            "cannot be combined with a '-- <command>' (subprocess) run."
        )


@click.group()
def sweep() -> None:
    """Unified matrix runner: sweep mitigations x {environments|diagnostics} x trials.

    Runs both flows from one command -- a built-in workload, or your own
    command after '--'. Replaces the deprecated 'aorta triage' and
    'aorta probe'.
    """


@sweep.command(
    name="run",
    cls=_SweepRunCommand,
    context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False},
)
@click.option(
    "--recipe",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    callback=_reject_flag_shaped_callback,
    help="Path to a YAML or JSON recipe file ('mode: triage' or 'mode: probe').",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate and print the resolved plan (cells; argv for a probe run) without executing.",
)
@click.option(
    "--mode",
    type=click.Choice(["matrix"]),
    default="matrix",
    show_default=True,
    help="matrix = full contingency table. 'optimize' deferred to a future release. "
    "Applies to the workload/matrix flow only; ignored for probe/subprocess "
    "runs (recipe 'mode: probe' or a '-- <command>' invocation).",
)
@click.option(
    "--workload",
    default=None,
    help="Workload name (workload flow, flag mode; from the aorta.workloads entry-point group).",
)
@click.option(
    "--mitigation-axis",
    default=None,
    help=(
        "Comma-separated mitigation names for the matrix row axis. "
        "Include 'none' for the baseline row. Workload flow, flag mode."
    ),
)
@click.option(
    "--environment-axis",
    default=None,
    help=(
        "Comma-separated environment names for the matrix column axis. "
        "Bare names resolve via the registry; 'image:<ref>' items declare an "
        "inline docker cell. Workload flow, flag mode."
    ),
)
@click.option(
    "--trials",
    type=int,
    default=None,
    help="Trials per cell (workload flow, flag mode; recipe mode takes this from the file).",
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Steps per trial (workload flow, flag mode; recipe mode takes this from the file).",
)
@click.option(
    "--ticket",
    default=None,
    callback=_reject_flag_shaped_callback,
    help=(
        "Ticket ID for output-dir grouping. Absence routes to '_no_ticket_'. "
        "Recipe mode takes this from the file's 'ticket' key (workload flow "
        "rejects passing it together with --recipe; probe flow lets it override)."
    ),
)
@click.option(
    "--baseline-cell",
    default=None,
    help=(
        "Override the auto-resolved baseline cell (workload flow). Recipe mode "
        "takes this from 'confound.baseline_cell'; passing it with --recipe is rejected."
    ),
)
@click.option(
    "--confound-threshold",
    type=float,
    default=None,
    help=(
        "cell_step_time / baseline_step_time above this -> 'speed (+N%)' flag "
        "(workload flow). Flag-mode default 1.15. Recipe mode takes this from "
        "'confound.threshold'; passing it with --recipe is rejected."
    ),
)
@click.option(
    "--output",
    "--output-dir",
    "output",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    callback=_reject_flag_shaped_callback,
    help=(
        "Top-level output directory. Defaults to 'triage_results' for a "
        "workload run and 'probe_results' for a subprocess run. "
        "('--output-dir' is accepted as a back-compat alias.)"
    ),
)
@click.option(
    "--env-passthrough-mode",
    type=click.Choice(["inherit", "file"]),
    default=None,
    help=(
        "Probe flow only. How per-cell env vars reach the user command: "
        "'inherit' stamps them on os.environ (the child inherits); 'file' "
        "additionally writes a chmod-0600 probe.env and exports AORTA_ENV_FILE. "
        "When omitted, the recipe's 'env_passthrough_mode:' is used "
        "(default 'inherit'); when present, this flag overrides the recipe."
    ),
)
@click.option(
    "--mitigations-file",
    "mitigation_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "JSON sidecar file with ad-hoc mitigations and/or environments "
        "(repeatable). Merged with built-ins, plugins, and inline-docker envs "
        "at recipe load time."
    ),
)
@click.option(
    "--stop-after-events",
    type=int,
    default=None,
    metavar="K",
    help=(
        "Probe flow only. Stop each cell once K trials match the event "
        "verdict (default verdict: fail), instead of running a fixed count. "
        "The loop needs a hard cap: pass --max-trials unless the recipe's "
        "'stop_after:' block already supplies one. Overlays that block (#232)."
    ),
)
@click.option(
    "--max-trials",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Probe flow only. Hard cap on trials per cell when stop-after is "
        "active (always honored). Pair with --stop-after-events; either flag "
        "may be omitted when the recipe's 'stop_after:' block already supplies "
        "that half (so --max-trials alone overrides just the recipe's cap)."
    ),
)
@click.option(
    "--disable-detector",
    "disable_detectors",
    multiple=True,
    metavar="TIER[:ID]",
    help=(
        "Probe flow only. Silence a detector or whole tier (repeatable). Pass "
        "a tier name ('tier3') to skip the entire tier, or a '<tier>:<id>' "
        "token ('tier2:hang') to skip one detector. A disabled detector is "
        "not evaluated and never counts toward the verdict. Unioned with any "
        "'disable_detectors:' / 'disable_detector_tiers:' set in the recipe."
    ),
)
@click.option(
    "--collect",
    default="",
    help=(
        "Comma-separated collector recipe names to attach to every cell "
        "(e.g. 'layer_numerics' for the per-layer NaN logger). Cross-cutting "
        "capture, not a matrix axis -- allowed together with --recipe, where "
        "it overrides any recipe-pinned 'collect:'. Workload flow only."
    ),
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Workload flow only. Exit non-zero if any cell errored or never ran "
        "(every trial did_not_run, e.g. a setup failure). A cell that RAN but "
        "reported a failure (a real bug repro) does NOT trip this. The matrix "
        "is still written. Useful in CI to catch a cell that silently didn't "
        "run (e.g. a rejected LD_PRELOAD)."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Stream live per-cell progress to stderr while the matrix runs. "
        "-v = INFO, -vv = DEBUG (aorta.* logger). A concise end-of-run "
        "summary (which cells failed + where their artifacts are) prints "
        "to stdout regardless of this flag."
    ),
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def sweep_run(
    recipe: Path | None,
    dry_run: bool,
    mode: str,
    workload: str | None,
    mitigation_axis: str | None,
    environment_axis: str | None,
    trials: int | None,
    steps: int | None,
    ticket: str | None,
    baseline_cell: str | None,
    confound_threshold: float | None,
    output: Path | None,
    env_passthrough_mode: str | None,
    stop_after_events: int | None,
    max_trials: int | None,
    disable_detectors: tuple[str, ...],
    mitigation_files: tuple[Path, ...],
    collect: str,
    strict: bool,
    verbose: int,
    argv: tuple[str, ...],
) -> None:
    """Run a matrix sweep. Built-in workload, or your own command after '--'.

    Examples:

      aorta sweep run --recipe triage.yaml

      aorta sweep run --workload fsdp --mitigation-axis none,foo
          --environment-axis local --trials 3 --steps 50

      aorta sweep run --recipe probe.yaml --ticket ROCM-1234 -- python repro.py
    """
    configure_verbose_logging(verbose)
    has_command = bool(argv)
    recipe_mode = _peek_recipe_mode(recipe) if recipe is not None else None

    # --- consistency guards (clear up-front errors, no silent fallback) ---
    if recipe_mode == "probe" and not has_command:
        raise click.UsageError(
            f"recipe {recipe} is a probe-mode recipe; it requires a user "
            "command after '--'. "
            "Usage: aorta sweep run --recipe <file> -- <command> [args...]"
        )
    if has_command and recipe_mode == "triage":
        raise click.UsageError(
            "a trailing '-- <command>' is only valid for a probe-mode run. "
            "Drop the command to run the workload flow, or set 'mode: probe' "
            "in the recipe."
        )

    is_probe_flow = has_command or recipe_mode == "probe"
    if is_probe_flow:
        if parse_csv(collect):
            raise click.UsageError(
                "--collect applies to the workload flow only; it has no effect "
                "on a probe/subprocess run (a user command after '--' or a "
                "'mode: probe' recipe)."
            )
        if strict:
            raise click.UsageError(
                "--strict applies to the workload flow only; a probe/subprocess "
                "run (a user command after '--' or a 'mode: probe' recipe) "
                "reports per-cell verdicts differently."
            )
        _dispatch_probe_flow(
            recipe=recipe,
            output=output,
            ticket=ticket,
            dry_run=dry_run,
            env_passthrough_mode=env_passthrough_mode,
            stop_after_events=stop_after_events,
            max_trials=max_trials,
            disable_detectors=disable_detectors,
            mitigation_files=mitigation_files,
            argv=argv,
            workload=workload,
            mitigation_axis=mitigation_axis,
            environment_axis=environment_axis,
            trials=trials,
            steps=steps,
            baseline_cell=baseline_cell,
            confound_threshold=confound_threshold,
        )
    else:
        _dispatch_workload_flow(
            recipe=recipe,
            dry_run=dry_run,
            mode=mode,
            workload=workload,
            mitigation_axis=mitigation_axis,
            environment_axis=environment_axis,
            trials=trials,
            steps=steps,
            ticket=ticket,
            baseline_cell=baseline_cell,
            confound_threshold=confound_threshold,
            output=output,
            mitigation_files=mitigation_files,
            env_passthrough_mode=env_passthrough_mode,
            stop_after_events=stop_after_events,
            max_trials=max_trials,
            disable_detectors=disable_detectors,
            collect=collect,
            strict=strict,
        )


def _dispatch_probe_flow(
    *,
    recipe: Path | None,
    output: Path | None,
    ticket: str | None,
    dry_run: bool,
    env_passthrough_mode: str | None,
    stop_after_events: int | None,
    max_trials: int | None,
    disable_detectors: tuple[str, ...],
    mitigation_files: tuple[Path, ...],
    argv: tuple[str, ...],
    workload: str | None,
    mitigation_axis: str | None,
    environment_axis: str | None,
    trials: int | None,
    steps: int | None,
    baseline_cell: str | None,
    confound_threshold: float | None,
) -> None:
    """Validate probe-flow-specific preconditions, then run the subprocess flow."""
    _reject_triage_only_flags_in_probe_flow(
        workload=workload,
        mitigation_axis=mitigation_axis,
        environment_axis=environment_axis,
        trials=trials,
        steps=steps,
        baseline_cell=baseline_cell,
        confound_threshold=confound_threshold,
    )
    if recipe is None:
        raise click.UsageError(
            "a probe run needs a recipe. Pass --recipe <path> (a 'mode: probe' "
            "file) together with the user command after '--'."
        )
    execute_probe(
        recipe=recipe,
        output=output if output is not None else Path("probe_results"),
        ticket=ticket,
        dry_run=dry_run,
        env_passthrough_mode=env_passthrough_mode,
        stop_after_events=stop_after_events,
        max_trials=max_trials,
        disable_detectors=disable_detectors,
        mitigation_files=mitigation_files,
        argv=argv,
        command_label="aorta sweep run",
    )


def _dispatch_workload_flow(
    *,
    recipe: Path | None,
    dry_run: bool,
    mode: str,
    workload: str | None,
    mitigation_axis: str | None,
    environment_axis: str | None,
    trials: int | None,
    steps: int | None,
    ticket: str | None,
    baseline_cell: str | None,
    confound_threshold: float | None,
    output: Path | None,
    mitigation_files: tuple[Path, ...],
    env_passthrough_mode: str | None,
    stop_after_events: int | None,
    max_trials: int | None,
    disable_detectors: tuple[str, ...],
    collect: str = "",
    strict: bool = False,
) -> None:
    """Validate workload-flow-specific preconditions, then run the workload flow."""
    if env_passthrough_mode is not None:
        raise click.UsageError(
            "--env-passthrough-mode applies to the probe flow only (a user "
            "command after '--'); it has no effect on a workload run."
        )
    probe_only_flags = {
        "--stop-after-events": stop_after_events,
        "--max-trials": max_trials,
        "--disable-detector": disable_detectors,
    }
    set_probe_only = [k for k, v in probe_only_flags.items() if v not in (None, ())]
    if set_probe_only:
        raise click.UsageError(
            f"{', '.join(set_probe_only)} only apply to the probe/subprocess "
            "flow (a user command after '--' or a 'mode: probe' recipe); they "
            "have no effect on a workload run."
        )
    execute_triage_run(
        recipe=recipe,
        dry_run=dry_run,
        mode=mode,
        workload=workload,
        mitigation_axis=mitigation_axis,
        environment_axis=environment_axis,
        trials=trials,
        steps=steps,
        ticket=ticket,
        baseline_cell=baseline_cell,
        confound_threshold=confound_threshold,
        output_dir=output if output is not None else Path("triage_results"),
        mitigation_files=mitigation_files,
        collect=collect,
        strict=strict,
    )


@sweep.command(name="list-mitigations")
@click.option(
    "--mitigations-file",
    "files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="JSON sidecar to merge into the listing (repeatable).",
)
def sweep_list_mitigations(files: tuple[Path, ...]) -> None:
    """List every registered mitigation with its source_package and env-var bundle."""
    execute_list_mitigations(files)


@sweep.command(name="list-environments")
@click.option(
    "--mitigations-file",
    "files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="JSON sidecar to merge into the listing (repeatable).",
)
def sweep_list_environments(files: tuple[Path, ...]) -> None:
    """List every registered environment and its baseline recipe."""
    execute_list_environments(files)


@sweep.command(name="list-patterns")
@click.option(
    "--version",
    "show_version",
    is_flag=True,
    help="Print 'aorta sweep pattern library v<N> (aorta <pkg-version>)' and exit.",
)
def sweep_list_patterns(show_version: bool) -> None:
    """Print the built-in Tier-4 failure-signature pattern catalogue and exit."""
    _print_list_patterns(show_version=show_version, banner_prefix="aorta sweep")


__all__ = ["sweep"]
