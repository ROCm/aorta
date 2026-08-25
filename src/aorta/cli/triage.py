"""`aorta triage` - DEPRECATED alias for the workload flow of `aorta sweep`.

`aorta triage` has been merged into the unified `aorta sweep` command
(issue #248). It keeps working as a thin deprecation shim: every command
here prints a stderr notice naming the `aorta sweep` replacement and then
delegates to the shared ``execute_*`` functions that `aorta sweep` also
calls -- so behaviour stays byte-identical.

The shared engine entry points (both converge on
:func:`aorta.triage.run_recipe`):

* ``--recipe <file>`` -- primary mode. Recipe file is the source of truth;
  validated at load time.
* ``--mode matrix --workload ... --mitigation-axis ... --environment-axis
  ...`` -- flag shim. Internally builds a :class:`Recipe` via
  :func:`aorta.triage.recipe.build_recipe_from_flags` and dispatches.

Discovery subcommands delegate to B3's resolver so users can see which
mitigations / environments come from ``aorta`` vs a plugin / sidecar.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import click

from aorta.cli._deprecation import emit_deprecation
from aorta.cli._env_display import format_env_bundle
from aorta.registry import (
    RegistryError,
    load_environments,
    load_mitigations,
)
from aorta.run.cli_helpers import configure_verbose_logging, parse_csv
from aorta.triage.recipe import (
    RecipeCellError,
    RecipeSchemaError,
    _parse_collect,
    build_recipe_from_flags,
    load_recipe,
)
from aorta.triage.runner import (
    MatrixIncompleteError,
    MatrixStrictError,
    _is_rank_zero,
    run_recipe,
)


@click.group()
def triage() -> None:
    """Triage matrix runner for mitigation x environment x trials sweeps."""


@triage.command(name="run")
# Recipe-mode options
@click.option(
    "--recipe",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a YAML or JSON recipe file (primary mode).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the recipe and print the resolved cell list without executing.",
)
# Flag-mode options
@click.option(
    "--mode",
    type=click.Choice(["matrix"]),
    default="matrix",
    show_default=True,
    help="matrix = full contingency table. 'optimize' deferred to P1 per D11.",
)
@click.option(
    "--workload",
    default=None,
    help="Workload name (required in flag mode; from the aorta.workloads entry-point group).",
)
@click.option(
    "--mitigation-axis",
    default=None,
    help=(
        "Comma-separated mitigation names for the matrix row axis. "
        "Include 'none' for the baseline row. Required in flag mode."
    ),
)
@click.option(
    "--environment-axis",
    default=None,
    help=(
        "Comma-separated environment names for the matrix column axis. "
        "Bare names resolve via the registry; 'image:<ref>' items declare an "
        "inline docker cell (auto-named _inline_<hash>). Required in flag mode."
    ),
)
@click.option(
    "--trials",
    type=int,
    default=None,
    help="Trials per cell (required in flag mode; recipe mode takes this from the file).",
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Steps per trial (required in flag mode; recipe mode takes this from the file).",
)
@click.option(
    "--ticket",
    default=None,
    help=(
        "Ticket ID for output-dir grouping "
        "(triage_results/<ticket>/<workload>/<timestamp>/). Absence routes to '_no_ticket_'. "
        "Recipe mode takes this from the file's 'ticket' key; passing this flag "
        "together with --recipe is rejected."
    ),
)
@click.option(
    "--baseline-cell",
    default=None,
    help=(
        "Override the auto-resolved baseline cell. Must match a cell name "
        "(flag mode: '<mitigation>-<environment>' or "
        "'<mitigation>-_inline_<hash>' for inline docker). Recipe mode takes "
        "this from the file's 'confound.baseline_cell' key; passing this flag "
        "together with --recipe is rejected."
    ),
)
@click.option(
    "--confound-threshold",
    type=float,
    default=None,
    help=(
        "cell_step_time / baseline_step_time above this -> 'speed (+N%)' flag. "
        "Flag-mode default is 1.15. Recipe mode takes this from the file's "
        "'confound.threshold' key; passing this flag together with --recipe is rejected."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("triage_results"),
    show_default=True,
    help="Top-level directory for matrix.json + matrix.md output.",
)
@click.option(
    "--mitigations-file",
    "mitigation_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "JSON sidecar file with ad-hoc mitigations and/or environments "
        "(repeatable). Forwarded to the registry; sidecar entries are merged "
        "with built-ins, entry-point plugins, and auto-registered inline-docker "
        "envs with the same name-collision rules (B3.1)."
    ),
)
@click.option(
    "--collect",
    default="",
    help=(
        "Comma-separated collector recipe names to attach to every cell "
        "(e.g. 'layer_numerics' for the per-layer NaN logger). Cross-cutting "
        "capture, not a matrix axis -- allowed together with --recipe, where "
        "it overrides any recipe-pinned 'collect:'. Names are validated "
        "against the known collector recipes."
    ),
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Exit non-zero if any cell errored or never ran (every trial "
        "did_not_run, e.g. a setup failure). A cell that RAN but reported a "
        "failure (a real bug repro) does NOT trip this -- that's an expected "
        "matrix outcome. The matrix is still written. Useful in CI to catch a "
        "cell that silently didn't run (e.g. a rejected LD_PRELOAD)."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Stream live per-cell progress to stderr while the matrix runs. "
        "-v = INFO (matrix preamble, per-cell start/finish, timings, "
        "trials passed). -vv = DEBUG (aorta platform internals). "
        "Scope is the aorta.* logger hierarchy; workload code in "
        "sibling packages is unaffected. A concise end-of-run summary "
        "(which cells failed + where their artifacts are) and the final "
        "'Wrote matrix to ...' line print to stdout regardless of this "
        "flag; -v adds the live blow-by-blow for long runs."
    ),
)
def triage_run(
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
    output_dir: Path,
    mitigation_files: tuple[Path, ...],
    collect: str,
    strict: bool,
    verbose: int,
) -> None:
    """Run the triage matrix: sweep mitigations x environments x trials, write matrix.md + matrix.json."""
    emit_deprecation("aorta triage run", "aorta sweep run")
    configure_verbose_logging(verbose)
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
        output_dir=output_dir,
        mitigation_files=mitigation_files,
        collect=collect,
        strict=strict,
    )


def execute_triage_run(
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
    output_dir: Path,
    mitigation_files: tuple[Path, ...],
    collect: str = "",
    strict: bool = False,
) -> None:
    """Workload-flow body shared by ``aorta sweep run`` and ``aorta triage run``.

    Logging is configured by the Click handler before this is called;
    keeping it out of here lets ``aorta sweep run`` own verbose setup once
    regardless of which flow it dispatches to.

    ``collect`` is a comma-separated list of collector recipe names
    (cross-cutting capture, e.g. ``layer_numerics``). It is allowed
    alongside ``--recipe`` -- unlike the flag-mode axis args -- because it
    is not a matrix axis. When passed with ``--recipe`` it OVERRIDES any
    ``collect:`` the recipe pins; when the flag is empty a recipe-pinned
    value is preserved.
    """
    # Defence-in-depth: Click's ``Choice(["matrix"])`` already enforces this,
    # but the CLI advertises ``--mode optimize`` as a future P1 addition (per
    # D11) and the dispatch site for that branch does not exist yet. Assert
    # here so whoever wires the new value can't silently slip past every
    # downstream code path that assumes "matrix is the only mode".
    assert mode == "matrix", f"unexpected --mode {mode!r}; only 'matrix' is implemented"
    if recipe is not None:
        _reject_flag_mode_args(
            workload=workload,
            mitigation_axis=mitigation_axis,
            environment_axis=environment_axis,
            trials=trials,
            steps=steps,
            ticket=ticket,
            baseline_cell=baseline_cell,
            confound_threshold=confound_threshold,
        )
        try:
            r = load_recipe(recipe, sidecar_files=mitigation_files or None)
            # --collect is not a flag-mode axis arg, so it's allowed with
            # --recipe. When provided, it overrides a recipe-pinned collect
            # NAME list; validate via the same loader path the recipe uses.
            # Per-collector options are recipe-file-only (the mapping form).
            # When the CLI overrides the collector names, keep only options
            # for collectors that are still enabled and clear per-cell
            # overrides so the operator-requested collectors apply everywhere.
            cli_collect = parse_csv(collect)
            if cli_collect:
                # Hand the loader its mapping form, carrying the recipe options
                # of the collectors that survived: validating the bare name
                # list would check them against their defaults, which both
                # misses a conflict the recipe's options create and invents one
                # they resolve.
                collect_names, collect_options = _parse_collect(
                    "--collect",
                    {
                        name: dict(r.collect_options.get(name) or {}) or None
                        for name in dict.fromkeys(cli_collect)
                    },
                )
                r = dataclasses.replace(
                    r,
                    collect=collect_names,
                    collect_options=collect_options,
                    cells=tuple(
                        dataclasses.replace(cell, collect=None, collect_options=None)
                        for cell in r.cells
                    ),
                )
        except (RecipeSchemaError, RecipeCellError, RegistryError) as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        missing = [
            name
            for name, val in (
                ("--workload", workload),
                ("--mitigation-axis", mitigation_axis),
                ("--environment-axis", environment_axis),
                ("--trials", trials),
                ("--steps", steps),
            )
            if val in (None, "")
        ]
        if missing:
            raise click.UsageError(
                f"Flag mode requires: {', '.join(missing)}. Alternatively, pass --recipe <file>."
            )
        try:
            r = build_recipe_from_flags(
                workload=workload,  # type: ignore[arg-type]
                mitigation_axis=mitigation_axis,  # type: ignore[arg-type]
                environment_axis=environment_axis,  # type: ignore[arg-type]
                trials=trials,  # type: ignore[arg-type]
                steps=steps,
                ticket=ticket,
                baseline_cell=baseline_cell,
                confound_threshold=1.15 if confound_threshold is None else confound_threshold,
                sidecar_files=mitigation_files or None,
                collect=parse_csv(collect),
            )
        except (RecipeSchemaError, RecipeCellError, RegistryError) as exc:
            raise click.ClickException(str(exc)) from exc

    # ``run_recipe`` runs ``_preflight_validate`` (env-slug collisions and
    # baseline resolution) before the per-cell ``try/except`` in
    # ``_run_one_cell`` swallows anything, so those errors bubble out here.
    # Wrap them to match the one-line ``ClickException`` shape the load path
    # produces, so the exit shape doesn't depend on which validator caught
    # the error. ``mitigation_files`` is NOT re-passed via
    # ``extra_sidecar_files``: ``load_recipe`` / ``build_recipe_from_flags``
    # already wrote it to ``r.sidecar_files`` and the runner picks it up
    # from there.
    try:
        run_dir = run_recipe(
            r,
            output_dir=output_dir,
            dry_run=dry_run,
            strict=strict,
        )
    except (MatrixIncompleteError, MatrixStrictError) as exc:
        # Artifacts ARE written for inspection -- print where they
        # landed first, then raise ClickException so the CLI exits
        # non-zero with the degradation reason. This is distinct from
        # RecipeCellError below: that's pre-execution validation
        # failure (no artifacts), this is post-execution degradation
        # (matrix.md / matrix.json present but classification couldn't
        # anchor, or --strict caught an errored/did_not_run cell). Only
        # rank 0 wrote artifacts, so only rank 0 reports.
        if _is_rank_zero():
            click.echo(f"Wrote matrix to {exc.run_dir}")
        raise click.ClickException(str(exc)) from exc
    except (
        RegistryError,
        RecipeCellError,
        RecipeSchemaError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    # Under torchrun only rank 0 writes the matrix; non-zero ranks ran the
    # trials into a scratch dir that no longer exists, so they must not claim
    # to have written anything.
    if not dry_run and _is_rank_zero():
        click.echo(f"Wrote matrix to {run_dir}")


def _reject_flag_mode_args(
    *,
    workload: str | None,
    mitigation_axis: str | None,
    environment_axis: str | None,
    trials: int | None,
    steps: int | None,
    ticket: str | None,
    baseline_cell: str | None,
    confound_threshold: float | None,
) -> None:
    """Surface a clear error if a user mixes --recipe with any flag-mode knob.

    Recipe-mode takes every matrix parameter from the file; silently ignoring
    a flag would mask a misconfiguration (e.g. the user thought
    ``--confound-threshold`` would override the recipe). Every flag that
    affects the resolved Recipe -- workload, both axes, trials, steps,
    ticket, baseline cell, and the confound threshold -- is checked here.
    ``--mode``, ``--output-dir``, ``--dry-run``, and ``--mitigations-file``
    are intentionally NOT in the list: they configure the runner, not the
    recipe content, so passing them with ``--recipe`` is meaningful.
    """
    conflicts = {
        "--workload": workload,
        "--mitigation-axis": mitigation_axis,
        "--environment-axis": environment_axis,
        "--trials": trials,
        "--steps": steps,
        "--ticket": ticket,
        "--baseline-cell": baseline_cell,
        "--confound-threshold": confound_threshold,
    }
    set_flags = [k for k, v in conflicts.items() if v not in (None, "")]
    if set_flags:
        raise click.UsageError(
            f"--recipe conflicts with {', '.join(set_flags)}. "
            "Use either --recipe OR the flag-mode args, not both."
        )


@triage.command(name="list-mitigations")
@click.option(
    "--mitigations-file",
    "files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="JSON sidecar to merge into the listing (repeatable).",
)
def list_mitigations(files: tuple[Path, ...]) -> None:
    """List every registered mitigation with its source_package and env-var bundle."""
    emit_deprecation("aorta triage list-mitigations", "aorta sweep list-mitigations")
    execute_list_mitigations(files)


def execute_list_mitigations(files: tuple[Path, ...]) -> None:
    """Print the mitigation registry; shared by ``aorta sweep`` + ``aorta triage``."""
    # Wrap RegistryError the same way `triage run` and `aorta run` do: a
    # malformed or colliding --mitigations-file should surface as a one-line
    # ClickException, not a Python traceback.
    try:
        registry = load_mitigations(extra_files=list(files) or None)
    except RegistryError as exc:
        raise click.ClickException(str(exc)) from exc
    name_w = max(len("NAME"), *(len(n) for n in registry))
    src_w = max(len("SOURCE"), *(len(m.source_package) for m in registry.values()))
    click.echo(f"{'NAME'.ljust(name_w)}  {'SOURCE'.ljust(src_w)}  ENV")
    for name in sorted(registry):
        m = registry[name]
        env_str = format_env_bundle(m.env)
        click.echo(f"{name.ljust(name_w)}  {m.source_package.ljust(src_w)}  {env_str}")


@triage.command(name="list-environments")
@click.option(
    "--mitigations-file",
    "files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="JSON sidecar to merge into the listing (repeatable).",
)
def list_environments(files: tuple[Path, ...]) -> None:
    """List every registered environment and its baseline recipe."""
    emit_deprecation("aorta triage list-environments", "aorta sweep list-environments")
    execute_list_environments(files)


def execute_list_environments(files: tuple[Path, ...]) -> None:
    """Print the environment registry; shared by ``aorta sweep`` + ``aorta triage``."""
    try:
        registry = load_environments(extra_files=list(files) or None)
    except RegistryError as exc:
        raise click.ClickException(str(exc)) from exc
    name_w = max(len("NAME"), *(len(n) for n in registry))
    src_w = max(len("SOURCE"), *(len(e.source_package) for e in registry.values()))
    docker_w = max(len("DOCKER"), *(len(e.docker or "-") for e in registry.values()))
    venv_w = max(len("VENV"), *(len(e.venv or "-") for e in registry.values()))
    click.echo(
        f"{'NAME'.ljust(name_w)}  {'SOURCE'.ljust(src_w)}  "
        f"{'DOCKER'.ljust(docker_w)}  {'VENV'.ljust(venv_w)}  ENV"
    )
    for name in sorted(registry):
        e = registry[name]
        env_str = format_env_bundle(e.env)
        click.echo(
            f"{name.ljust(name_w)}  {e.source_package.ljust(src_w)}  "
            f"{(e.docker or '-').ljust(docker_w)}  "
            f"{(e.venv or '-').ljust(venv_w)}  {env_str}"
        )
