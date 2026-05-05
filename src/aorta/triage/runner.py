"""Triage-matrix orchestration.

:func:`run_recipe` is the single entry point both the recipe-file mode and
the flag-mode CLI funnel into. Given a validated :class:`Recipe`, it:

1. Resolves the per-run output directory
   (``<output-dir>/<ticket>/<workload>/<timestamp>``) and creates it.
2. Writes an inline-docker sidecar JSON (if the recipe references any
   ``_inline_<hash>`` envs) so B1's registry resolver picks them up.
3. Captures the host :func:`aorta.instrumentation.environment.collect_env`
   snapshot once -> ``host_env.json``.
4. For each unique environment in ``recipe.cells``, captures a
   per-environment ``collect_env`` snapshot once, *right before that env's
   first cell runs* -> ``environments/<name>/env.json``.
5. Builds a :class:`aorta.run.RunRequest` per cell and calls
   :func:`aorta.run.run_trials` **in-process**. Per-cell exceptions are
   caught and surfaced as an ``error`` row so other cells still run.
6. Aggregates each cell via :func:`aorta.triage.matrix.aggregate_cell`.
7. Resolves the baseline cell and classifies every cell via
   :mod:`aorta.triage.confound`.
8. Writes ``matrix.md``, ``matrix.json``, ``recipe.resolved.yaml``.

Per the acceptance criteria in issue #151, this module MUST NOT use
``subprocess`` -- every cell runs as a plain Python call to
:func:`run_trials`. A grep-test under ``tests/triage/`` enforces that.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from aorta.instrumentation.environment import EnvSnapshot, collect_env
from aorta.registry import get_mitigation
from aorta.run import RunRequest, TrialResult, run_trials
from aorta.triage.confound import (
    classify_all,
    resolve_baseline,
)
from aorta.triage.matrix import CellStats, aggregate_cell
from aorta.triage.output import (
    format_timestamp,
    resolve_run_dir,
    write_matrix_json,
    write_matrix_md,
    write_resolved_recipe,
)
from aorta.triage.recipe import InlineEnv, Recipe

log = logging.getLogger(__name__)

_INLINE_SIDECAR_NAME = "inline_environments.sidecar.json"


def _write_inline_sidecar(run_dir: Path, inline_envs: tuple[InlineEnv, ...]) -> Path | None:
    """Persist inline-docker envs as a B3 sidecar so B1 can resolve them.

    Returns the sidecar path (``None`` when the recipe has no inline envs).
    The sidecar lives inside ``run_dir`` so it's preserved for audit --
    anyone inspecting the run directory can see exactly what inline env
    registrations were in effect.
    """
    if not inline_envs:
        return None
    path = run_dir / _INLINE_SIDECAR_NAME
    doc = {
        "version": 1,
        "environments": {env.name: {"docker": env.docker} for env in inline_envs},
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _capture_env(
    target: Path,
    scope: str,
    warnings: list[str],
) -> EnvSnapshot:
    """Call collect_env and persist to ``target``, appending a warning if partial.

    :func:`collect_env` is contractually fail-soft (A1), so this wrapper
    never re-raises -- probe failure never aborts the matrix.
    """
    snapshot = collect_env()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    if snapshot.partial:
        reasons = ", ".join(snapshot.partial_reasons) or "(no reasons reported)"
        warnings.append(
            f"env probe for scope {scope!r} is partial: {reasons}. "
            "See the scope's env.json for details."
        )
    return snapshot


def _resolve_cell_env_vars(
    cell_mitigations: tuple[str, ...],
    cell_extra_env: dict[str, str],
    sidecar_files: tuple[Path, ...] | None,
) -> dict[str, str]:
    """Compute the unioned env-var bundle B1 will apply for a cell.

    B1 also unions internally; we duplicate the computation here so
    matrix.json can record the resolved env-var set alongside the aggregated
    stats, without having to rely on B1 threading them through
    TrialResult.
    """
    extra = list(sidecar_files) if sidecar_files else None
    env: dict[str, str] = {}
    for name in cell_mitigations:
        env.update(get_mitigation(name, extra_files=extra))
    env.update(cell_extra_env)
    return env


def _cells_dir(run_dir: Path) -> Path:
    d = run_dir / "cells"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collect_trial_paths(results_dir: Path) -> list[str]:
    """Return the trial_*.json paths B1 wrote, sorted by trial index.

    B1's dispatcher writes to ``<results_dir>/<workload>/trial_<N>.json``
    (the dispatcher appends the workload subdir; that's a B1 contract B2
    currently honours without surgery). We glob the workload subdir so the
    matrix.json ``trial_paths`` field matches reality on disk.
    """
    if not results_dir.exists():
        return []
    found: list[Path] = []
    for candidate in sorted(results_dir.rglob("trial_*.json")):
        found.append(candidate)
    return [str(p) for p in found]


def _run_one_cell(
    cell,
    recipe: Recipe,
    run_dir: Path,
    sidecar_files: tuple[Path, ...],
) -> tuple[list[TrialResult], str | None, dict[str, str], list[str]]:
    """Execute a single cell through B1 and return (trials, error, env_vars, trial_paths).

    Exception handling scope is deliberately wide: any failure originating
    from B1 (unknown mitigation, workload crash in ``setup``, docker pull
    failure from a future docker-aware environment) should flag the cell as
    errored without bringing down the whole matrix. The full traceback is
    logged at WARNING so operators can diagnose, but the returned ``error``
    string stays short -- it's the text shown in matrix.md.
    """
    cell_dir = _cells_dir(run_dir) / cell.name
    cell_dir.mkdir(parents=True, exist_ok=True)

    resolved_env_vars = _resolve_cell_env_vars(cell.mitigations, cell.extra_env, sidecar_files)

    request = RunRequest(
        workload=recipe.workload,
        trials=cell.effective_trials(recipe.trials),
        environment=cell.environment,
        mitigations=tuple(cell.mitigations),
        extra_env=dict(cell.extra_env),
        steps=cell.effective_steps(recipe.steps),
        results_dir=cell_dir,
        sidecar_files=sidecar_files,
    )

    try:
        trials = run_trials(request)
    except Exception as exc:
        log.warning("cell %r failed with %s: %s", cell.name, type(exc).__name__, exc, exc_info=True)
        return [], f"{type(exc).__name__}: {exc}", resolved_env_vars, []

    trial_paths = _collect_trial_paths(cell_dir)
    return trials, None, resolved_env_vars, trial_paths


def _print_dry_run(recipe: Recipe) -> None:
    """Write the resolved cell list to stdout without touching the filesystem."""
    click.echo(f"Dry run: {recipe.workload} / ticket={recipe.ticket or '(none)'}")
    click.echo(f"Cells ({len(recipe.cells)}):")
    for cell in recipe.cells:
        click.echo(
            f"  - {cell.name}: mitigations={list(cell.mitigations)} "
            f"environment={cell.environment} "
            f"trials={cell.effective_trials(recipe.trials)} "
            f"steps={cell.effective_steps(recipe.steps)}"
            + (f" extra_env={cell.extra_env}" if cell.extra_env else "")
        )
    if recipe.inline_environments:
        click.echo("Inline docker environments:")
        for env in recipe.inline_environments:
            click.echo(f"  - {env.name} -> {env.docker}")
    click.echo(
        f"Baseline rule: " f"{recipe.confound.baseline_cell or '(auto-resolve at run time)'}"
    )
    click.echo(f"Confound threshold: {recipe.confound.threshold}")


def run_recipe(
    recipe: Recipe,
    output_dir: Path,
    dry_run: bool = False,
    extra_sidecar_files: tuple[Path, ...] = (),
    timestamp: str | None = None,
) -> Path:
    """Execute a recipe and write matrix.md / matrix.json / recipe.resolved.yaml.

    Args:
        recipe: Pre-validated recipe (from :func:`aorta.triage.recipe.load_recipe`
            or :func:`aorta.triage.recipe.build_recipe_from_flags`).
        output_dir: Top-level output directory (the CLI's ``--output-dir``).
        dry_run: When True, validates and prints the resolved cell list to
            stdout without touching the filesystem and returns a sentinel
            ``Path(".")``.
        extra_sidecar_files: Operator-supplied sidecar JSONs (from
            ``--mitigations-file``). These are threaded through to B1's
            registry resolver alongside the runner-generated inline sidecar.
        timestamp: Override for the run-dir timestamp component (test hook).

    Returns:
        The run directory path (``<output-dir>/<ticket>/<workload>/<timestamp>``).
    """
    if dry_run:
        _print_dry_run(recipe)
        return Path(".")

    ts = timestamp or format_timestamp()
    run_dir = resolve_run_dir(output_dir, recipe, timestamp=ts)

    inline_sidecar_path = _write_inline_sidecar(run_dir, recipe.inline_environments)
    sidecar_files: tuple[Path, ...] = tuple(extra_sidecar_files)
    if inline_sidecar_path is not None:
        sidecar_files = sidecar_files + (inline_sidecar_path,)

    warnings: list[str] = []

    _capture_env(run_dir / "host_env.json", scope="host", warnings=warnings)

    # Per-environment probes, captured once per unique env in the order
    # cells reference them. ``seen`` preserves first-use ordering so the
    # probe lands right before the env's first cell runs (matches the
    # "captured once per unique --environment-axis value" acceptance
    # criterion).
    seen_envs: set[str] = set()

    env_dir = run_dir / "environments"

    cell_stats: list[CellStats] = []
    for cell in recipe.cells:
        if cell.environment not in seen_envs:
            _capture_env(
                env_dir / cell.environment / "env.json",
                scope=f"environment:{cell.environment}",
                warnings=warnings,
            )
            seen_envs.add(cell.environment)

        trials, error, resolved_env_vars, trial_paths = _run_one_cell(
            cell, recipe, run_dir, sidecar_files
        )

        stats = aggregate_cell(
            name=cell.name,
            mitigations=tuple(cell.mitigations),
            environment=cell.environment,
            extra_env=dict(cell.extra_env),
            resolved_env_vars=resolved_env_vars,
            trials=trials,
            effective_steps=cell.effective_steps(recipe.steps),
            trial_paths=trial_paths,
            error=error,
        )
        cell_stats.append(stats)

    baseline_cell = resolve_baseline(recipe.cells, recipe.confound.baseline_cell)
    confound_tags = classify_all(cell_stats, baseline_cell.name, recipe.confound.threshold)
    baseline_stats = next(c for c in cell_stats if c.name == baseline_cell.name)

    if baseline_stats.error is not None:
        warnings.append(
            f"baseline cell {baseline_cell.name!r} errored "
            f"({baseline_stats.error}); step-time ratios for non-baseline "
            "cells are reported as n/a."
        )

    write_matrix_md(
        run_dir / "matrix.md",
        recipe=recipe,
        cell_stats=cell_stats,
        baseline=baseline_stats,
        confound_tags=confound_tags,
        warnings=warnings,
        run_timestamp=ts,
    )
    write_matrix_json(
        run_dir / "matrix.json",
        recipe=recipe,
        cell_stats=cell_stats,
        baseline_name=baseline_cell.name,
        confound_tags=confound_tags,
        run_timestamp=ts,
        warnings=warnings,
    )
    write_resolved_recipe(
        run_dir / "recipe.resolved.yaml",
        recipe=recipe,
        sidecar_files=sidecar_files,
    )

    return run_dir


__all__ = ["run_recipe"]
