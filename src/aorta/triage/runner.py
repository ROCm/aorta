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
import shutil
from pathlib import Path
from typing import Any

import click

from aorta.instrumentation.environment import EnvSnapshot, collect_env
from aorta.registry import get_environment, get_mitigation
from aorta.registry.errors import RegistryError
from aorta.run import RunRequest, TrialResult, run_trials
from aorta.triage.confound import (
    classify_all,
    resolve_baseline,
)
from aorta.triage.matrix import CellStats, aggregate_cell
from aorta.triage.output import (
    format_timestamp,
    resolve_run_dir,
    safe_slug,
    write_matrix_json,
    write_matrix_md,
    write_resolved_recipe,
)
from aorta.triage.recipe import InlineEnv, Recipe

log = logging.getLogger(__name__)

_INLINE_SIDECAR_NAME = "inline_environments.sidecar.json"
_OPERATOR_SIDECAR_DIR = "sidecars"


def _merge_sidecar_files(
    recipe_files: tuple[Path, ...],
    extra: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Union ``recipe.sidecar_files`` with caller-supplied extras, deduped.

    ``Recipe`` carries the sidecars that ``load_recipe`` /
    ``build_recipe_from_flags`` validated against, so a programmatic
    ``load_recipe(path, sidecar_files=...) -> run_recipe(recipe)`` flow
    Just Works without re-passing them. ``extra_sidecar_files`` stays as a
    runner-level escape hatch (tests construct a ``Recipe`` directly and
    add sidecars at execute time). Dedup by resolved path so the CLI's
    ``load_recipe(... sidecar_files=files) + run_recipe(... extra_sidecar_files=files)``
    pattern doesn't double-copy the same files into ``<run_dir>/sidecars``.
    """
    seen: set[Path] = set()
    merged: list[Path] = []
    for p in (*recipe_files, *extra):
        # ``resolve(strict=False)`` so missing files still dedup (Click's
        # ``exists=True`` validates the CLI path; programmatic callers may
        # legitimately pass an absolute path that hasn't been touched).
        key = Path(p).resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return tuple(merged)


def _copy_operator_sidecars(run_dir: Path, sidecar_files: tuple[Path, ...]) -> tuple[Path, ...]:
    """Snapshot operator-supplied sidecars into the run dir for replay.

    ``recipe.resolved.yaml`` advertises itself as the rerun artifact, but it
    intentionally references mitigation / environment names by *name* -- so a
    recipe whose names only resolve via a ``--mitigations-file`` would be
    unreplayable from the resolved YAML alone. Copy each sidecar into
    ``<run_dir>/sidecars/<basename>`` (B3 already enforces unique basenames
    via :func:`aorta.registry.sidecar.check_sidecar_basenames`, so the
    target paths cannot collide). The README documents the replay command:
    ``aorta triage run --recipe recipe.resolved.yaml --mitigations-file
    sidecars/<basename>``.
    """
    if not sidecar_files:
        return ()
    target_dir = run_dir / _OPERATOR_SIDECAR_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sidecar_files:
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return tuple(copied)


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


def _check_env_slug_collisions(cells: tuple) -> None:
    """Reject recipes whose distinct env names slug to the same dir component.

    ``environments/<safe_slug(name)>/env.json`` would silently overwrite when
    two different registered names slug-collapse to the same string (e.g.
    ``"a/b"`` and ``"a:b"`` both become ``"a_b"``). Callers treat them as
    separate envs in memory, so the on-disk artifact would contradict the
    in-memory state. Cell-name collisions are already prevented by the
    cell-name regex in :mod:`aorta.triage.recipe`; environment names come
    from the registry and are not constrained, so we enforce the parallel
    invariant at run setup time and raise with the offending pair so the
    operator can fix it.
    """
    seen: dict[str, str] = {}
    for cell in cells:
        slug = safe_slug(cell.environment)
        prev = seen.get(slug)
        if prev is not None and prev != cell.environment:
            raise RegistryError(
                f"environment names {prev!r} and {cell.environment!r} both map "
                f"to filesystem component {slug!r}; rename one in the registry "
                "/ sidecar so per-environment artifacts can be distinguished"
            )
        seen[slug] = cell.environment


def _is_isolated_environment(
    env_name: str,
    inline_envs: tuple[InlineEnv, ...],
    sidecar_files: tuple[Path, ...] = (),
) -> bool:
    """Return True iff the env would isolate the trial from the runner process.

    Inline-docker envs always count as isolated (the cell shorthand
    explicitly declares a docker ref). Registered envs count as isolated if
    their :class:`aorta.registry.Environment` descriptor sets ``docker`` or
    ``venv`` -- in either case, a runner-process ``collect_env()`` call would
    record the host's state, not the trial's, and therefore the resulting
    ``environments/<name>/env.json`` would be misleading. B1 doesn't actually
    perform docker / venv isolation today (in-process execution); the fix
    when it does is to capture inside the isolated env. Until then, gate the
    runner-process probe on this predicate.

    ``sidecar_files`` MUST be threaded through so envs defined only in a
    ``--mitigations-file`` JSON are visible to the registry resolver. Without
    it, sidecar-defined docker/venv envs would mis-classify as "local" and
    pick up a misleading host-state snapshot under their name. Lookup
    failures still fall back to "treat as local" so probe behaviour is
    unchanged for envs we genuinely don't know about; the pre-flight
    :func:`_validate_names_resolve` would already have failed if the name
    were truly unknown.
    """
    if any(env_name == e.name for e in inline_envs):
        return True
    extra = list(sidecar_files) if sidecar_files else None
    try:
        descriptor = get_environment(env_name, extra_files=extra)
    except RegistryError:
        return False
    return bool(descriptor.docker or descriptor.venv)


def _write_isolated_env_placeholder(
    target: Path,
    env_name: str,
    inline_envs: tuple[InlineEnv, ...],
    warnings: list[str],
    sidecar_files: tuple[Path, ...] = (),
) -> None:
    """Write a non-misleading placeholder for envs the runner cannot probe.

    For docker / venv environments the only honest thing we can record is the
    descriptor itself plus a note explaining why no live snapshot is captured.
    Pretending otherwise (calling ``collect_env`` in the runner process) would
    log host state under a docker label and silently mislead anyone reading
    the artifact.

    ``sidecar_files`` is forwarded to the registry lookup so the descriptor
    written here matches the env that B1 will actually use, even when the env
    only exists in an operator-supplied ``--mitigations-file``.
    """
    inline_match = next((e for e in inline_envs if e.name == env_name), None)
    descriptor: dict[str, Any] = {"name": env_name}
    if inline_match is not None:
        descriptor["docker"] = inline_match.docker
        descriptor["source"] = "inline"
    else:
        extra = list(sidecar_files) if sidecar_files else None
        try:
            env_desc = get_environment(env_name, extra_files=extra)
            descriptor["docker"] = env_desc.docker
            descriptor["venv"] = env_desc.venv
            descriptor["source_package"] = env_desc.source_package
        except RegistryError as exc:  # pragma: no cover - guarded by predicate
            descriptor["_lookup_error"] = f"{type(exc).__name__}: {exc}"
    skip_reason = (
        "B1 currently runs trials in the runner process, so a runner-time "
        "collect_env() snapshot would record the host's state instead of the "
        "isolated docker/venv environment the descriptor advertises. The "
        "snapshot is intentionally skipped to avoid a misleading artifact; "
        "host_env.json next to this file captures the runner's view."
    )
    placeholder = {
        "name": env_name,
        "snapshot_captured": False,
        "skip_reason": skip_reason,
        "descriptor": descriptor,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
    warnings.append(
        f"environment {env_name!r}: per-env probe skipped (isolated env, B1 in-process). "
        "See the env's env.json for the descriptor and the host-level snapshot in host_env.json."
    )


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

    Sort by the integer ``N`` extracted from the filename, NOT
    lexicographically -- a lex sort would put ``trial_10.json`` before
    ``trial_2.json`` and the recorded order would diverge from execution
    order once a cell has 10+ trials. Files whose names don't parse as
    ``trial_<int>.json`` sort last (alphabetically), which keeps any future
    sibling artifacts visible without breaking the contract for the common
    case.
    """
    if not results_dir.exists():
        return []

    def _key(path: Path) -> tuple[int, str]:
        stem = path.stem  # "trial_3"
        if stem.startswith("trial_"):
            try:
                return (int(stem[len("trial_") :]), "")
            except ValueError:
                pass
        # Sentinel: push non-conforming names after every trial_N entry.
        return (10**12, str(path))

    found = sorted(results_dir.rglob("trial_*.json"), key=_key)
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
    cell_dir = _cells_dir(run_dir) / safe_slug(cell.name)
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


def _preflight_validate(recipe: Recipe) -> None:
    """Run every fail-fast check that does NOT need the filesystem.

    Centralised so dry-run and real-run share the same validation surface:
    a recipe that ``--dry-run`` accepts must always be acceptable to
    ``run_recipe(..., dry_run=False)``. Previously dry-run skipped these
    checks and printed a clean summary for recipes that the real run then
    rejected -- exactly the kind of "validation-only execution" footgun
    the dry-run flag is supposed to prevent.

    Currently this covers:

    * environment-name slug collisions across cells (would otherwise raise
      mid-run, after the host_env probe + sidecar copies have already
      written to disk).
    * baseline resolution (the configured / auto-resolved baseline must
      exist; an unresolvable baseline turns ``classify_all`` into a
      ``RecipeCellError`` deep inside the run loop).

    Any callable added here MUST be pure (no FS / network / subprocess
    side-effects) so dry-run truly stays read-only on the host.
    """
    _check_env_slug_collisions(recipe.cells)
    resolve_baseline(recipe.cells, recipe.confound.baseline_cell)


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
    click.echo(f"Baseline rule: {recipe.confound.baseline_cell or '(auto-resolve at run time)'}")
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
            or :func:`aorta.triage.recipe.build_recipe_from_flags`). Any
            sidecar files passed to those constructors are carried on
            ``recipe.sidecar_files`` and used here automatically -- callers
            do **not** need to re-pass them via ``extra_sidecar_files``.
        output_dir: Top-level output directory (the CLI's ``--output-dir``).
        dry_run: When True, validates and prints the resolved cell list to
            stdout without touching the filesystem and returns a sentinel
            ``Path(".")``.
        extra_sidecar_files: Additional sidecar JSONs to thread to B1's
            registry resolver and snapshot into ``<run_dir>/sidecars/``.
            Unioned with ``recipe.sidecar_files`` (deduped by resolved
            path), so it's safe -- though redundant -- for the CLI to pass
            the same files here too. The arg exists for runner-level
            callers that build a ``Recipe`` directly (tests, in-process
            embedders) and want to add sidecars at execute time.
        timestamp: Override for the run-dir timestamp component (test hook).

    Returns:
        The run directory path (``<output-dir>/<ticket>/<workload>/<timestamp>``).
    """
    # Preflight first, BEFORE the dry-run early-return: dry-run is documented
    # as "validation without execution", so it must reject everything the
    # real run would reject. Otherwise CI / pre-submit checks happily pass on
    # recipes that fail the moment they actually run.
    _preflight_validate(recipe)

    if dry_run:
        _print_dry_run(recipe)
        return Path(".")

    ts = timestamp or format_timestamp()
    run_dir = resolve_run_dir(output_dir, recipe, timestamp=ts)

    # Operator sidecars come from two places: ones the Recipe was built
    # against (``recipe.sidecar_files``, populated by ``load_recipe`` /
    # ``build_recipe_from_flags``) and ones the caller hands in directly at
    # execute time (``extra_sidecar_files``). Merge with dedup so the CLI's
    # belt-and-suspenders pattern of passing the same files at both layers
    # doesn't produce duplicate <run_dir>/sidecars/ copies.
    all_operator_sidecars = _merge_sidecar_files(recipe.sidecar_files, tuple(extra_sidecar_files))

    # Snapshot operator-supplied sidecars (--mitigations-file) into run_dir
    # FIRST so the recipe.resolved.yaml + the copies form a self-contained
    # replay bundle. Use the in-run-dir copies as the resolver's source of
    # truth from here on, so what gets executed and what gets archived for
    # replay are byte-identical.
    operator_sidecar_paths = _copy_operator_sidecars(run_dir, all_operator_sidecars)

    inline_sidecar_path = _write_inline_sidecar(run_dir, recipe.inline_environments)
    sidecar_files: tuple[Path, ...] = operator_sidecar_paths
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

    # Env-slug collision + baseline resolution were already enforced by
    # _preflight_validate at the very top of run_recipe (so dry-run sees the
    # same errors). No need to re-check here.

    cell_stats: list[CellStats] = []
    for cell in recipe.cells:
        if cell.environment not in seen_envs:
            env_json_path = env_dir / safe_slug(cell.environment) / "env.json"
            if _is_isolated_environment(
                cell.environment, recipe.inline_environments, sidecar_files
            ):
                _write_isolated_env_placeholder(
                    env_json_path,
                    cell.environment,
                    recipe.inline_environments,
                    warnings,
                    sidecar_files,
                )
            else:
                _capture_env(
                    env_json_path,
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
        sidecar_files=sidecar_files,
    )
    write_resolved_recipe(
        run_dir / "recipe.resolved.yaml",
        recipe=recipe,
        sidecar_files=sidecar_files,
    )

    if operator_sidecar_paths:
        # Operator-supplied sidecars were snapshotted into the run dir so the
        # archived recipe.resolved.yaml + sidecar copies form a self-contained
        # replay bundle. Print the exact rerun command so the operator does
        # not have to reconstruct the --mitigations-file flags by hand.
        flags = " ".join(
            f"--mitigations-file {Path(_OPERATOR_SIDECAR_DIR) / p.name}"
            for p in operator_sidecar_paths
        )
        click.echo(
            f"to rerun: cd {run_dir} && aorta triage run --recipe recipe.resolved.yaml {flags}"
        )

    return run_dir


__all__ = ["run_recipe"]
