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

import contextlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import click

from aorta.instrumentation.environment import EnvSnapshot, collect_env
from aorta.registry import get_environment, get_mitigation
from aorta.registry.errors import RegistryError
from aorta.run import RunRequest, TrialResult, run_trials
from aorta.triage.confound import (
    classify_all,
    is_did_not_run_cell,
    resolve_baseline,
)
from aorta.triage.matrix import (
    OUTCOME_COMPLETED,
    OUTCOME_CRASHED_AFTER_ITERATIONS,
    OUTCOME_DID_NOT_RUN,
    OUTCOME_UNKNOWN,
    CellStats,
    aggregate_cell,
)
from aorta.triage.output import (
    acquire_flat_resume_lock,
    format_timestamp,
    resolve_run_dir,
    safe_slug,
    write_matrix_json,
    write_matrix_md,
    write_resolved_recipe,
)
from aorta.triage.recipe import InlineEnv, Recipe, RecipeCellError

log = logging.getLogger(__name__)

_INLINE_SIDECAR_NAME = "inline_environments.sidecar.json"
_OPERATOR_SIDECAR_DIR = "sidecars"


def _is_rank_zero() -> bool:
    """True if this process should write run artifacts.

    Mirrors the dispatcher's per-trial write gate (``RANK == 0``): under a
    distributed launcher every rank runs ``run_recipe`` so the workload's
    collectives complete on all ranks, but only rank 0 owns the output tree.
    ``RANK`` is unset for single-process ``aorta triage run`` (treated as
    rank 0). A non-integer ``RANK`` is treated defensively as rank 0 so a
    misconfigured launcher writes rather than silently dropping artifacts.
    """
    raw_rank = os.environ.get("RANK", "0")
    try:
        return int(raw_rank) == 0
    except ValueError:
        log.warning("Ignoring non-integer RANK=%r; treating this process as rank 0.", raw_rank)
        return True


class MatrixIncompleteError(Exception):
    """Raised AFTER ``run_recipe`` successfully writes matrix.md /
    matrix.json when the run completed but classification could not
    anchor -- e.g. the explicit ``baseline_cell`` produced only
    did_not_run trials, or every auto-baseline candidate did. The
    matrix artifacts ARE present for inspection (the exception carries
    ``run_dir``); the exception itself signals the degradation to the
    CLI so it can exit non-zero. This is distinct from
    :class:`aorta.triage.recipe.RecipeCellError`, which is for
    pre-execution validation failures (no artifacts written).
    """

    def __init__(self, message: str, run_dir: Path) -> None:
        super().__init__(message)
        self.run_dir = run_dir


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


_DISPATCHER_TRIAL_RE = re.compile(r"^trial_d\d+_m\d+_t(\d+)$")
_LEGACY_TRIAL_RE = re.compile(r"^trial_(\d+)$")


def _collect_trial_paths(results_dir: Path) -> list[str]:
    """Return the per-trial JSON paths the dispatcher wrote, sorted by trial index.

    The dispatcher writes to
    ``<results_dir>/<workload>/trial_d<dataset>_m<mitigation>_t<trial>.json``
    (the dispatcher appends the workload subdir; that's a B1 contract
    B2 currently honours without surgery). Older fixtures and tests
    use the simpler ``trial_<N>.json`` shape; both are supported here
    so this helper is the single sorting authority across the matrix /
    resume code paths.

    Sort by the integer ``N`` extracted from the filename, NOT
    lexicographically -- a lex sort would put ``..._t10.json`` before
    ``..._t2.json`` and the recorded order would diverge from
    execution order once a cell has 10+ trials. This silently broke
    the resume short-circuit's hydration order for any
    trials >= 10 because ``_hydrate_trials_from_paths`` walks the
    list in order. Files whose names don't parse under either regex
    sort last (alphabetically), which keeps any future sibling
    artifacts visible without breaking the contract for the common
    case.
    """
    if not results_dir.exists():
        return []

    def _key(path: Path) -> tuple[int, str]:
        stem = path.stem
        m = _DISPATCHER_TRIAL_RE.match(stem) or _LEGACY_TRIAL_RE.match(stem)
        if m is not None:
            return (int(m.group(1)), "")
        # Sentinel: push non-conforming names after every trial_N entry.
        return (10**12, str(path))

    found = sorted(results_dir.rglob("trial_*.json"), key=_key)
    return [str(p) for p in found]


@dataclass(frozen=True)
class _HydratedTrial:
    """One successfully hydrated dispatcher JSON, keyed by parsed trial index.

    Carries the dispatcher's source path alongside the body so the
    resume short-circuit can return ``trial_paths`` aligned with
    ``hydrated`` (same index -> same position in both lists) without
    re-walking the directory.
    """

    index: int
    path: str
    trial: TrialResult


def _hydrate_trials_by_index(trial_paths: list[str]) -> dict[int, _HydratedTrial]:
    """Hydrate dispatcher ``trial_*.json`` files into an index-keyed map.

    Each entry appears in the returned dict iff ALL THREE of:

    1. The filename's trial-index suffix parses under the dispatcher
       regex (``trial_d<d>_m<m>_t<idx>``) or the legacy
       ``trial_<idx>`` shape.
    2. ``Path.read_text`` + ``json.loads`` succeed.
    3. :meth:`TrialResult.from_dict` accepts the body.

    Any failure at steps 2 or 3 logs at WARNING and the entry is
    silently dropped -- the caller's
    ``required_indices.issubset(hydrated_by_index.keys())`` check
    then catches the missing index and falls back to a full re-run.

    Index-keyed (rather than the previous "list + side-channel index
    set" shape) so a corrupted required ``_t0.json`` co-existing with
    a stale extra ``_t<N>.json`` cannot pass the resume validation.
    The previous implementation validated indices via the *filename
    set* (from :func:`_collect_trial_paths`) while hydration silently
    skipped the unreadable required file; the count check then passed
    on the stale extra and the slice returned the wrong trial bodies.
    The fix lives in the key set of this dict: a body that does not
    hydrate cannot be in the keys, so the subset check is the load-
    bearing invariant rather than the filename walk.
    """
    by_index: dict[int, _HydratedTrial] = {}
    for raw in trial_paths:
        path = Path(raw)
        m = _DISPATCHER_TRIAL_RE.match(path.stem) or _LEGACY_TRIAL_RE.match(path.stem)
        if m is None:
            # Filename doesn't parse; ignored just as
            # ``_collect_trial_paths`` already shoves these to the
            # end of the sorted list.
            continue
        index = int(m.group(1))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("resume: unreadable trial JSON %s (%s)", path, exc)
            continue
        try:
            trial = TrialResult.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("resume: trial JSON %s violates schema (%s)", path, exc)
            continue
        by_index[index] = _HydratedTrial(index=index, path=raw, trial=trial)
    return by_index


def _trial_passed_for_log(trial: Any) -> bool:
    """Mirror of ``aorta.triage.matrix._trial_passed`` for the per-cell
    log line.

    Kept local rather than imported from the matrix module's private
    namespace so the runner doesn't depend on a leading-underscore
    callable (Python convention: private to defining module). The
    semantic is identical -- a trial passes iff ``exit_status == "ok"``
    AND the wrapped ``WorkloadResult.passed`` is not False -- so the
    log line and the matrix.md row agree on what counts as a pass.
    """
    if getattr(trial, "exit_status", None) != "ok":
        return False
    result = getattr(trial, "result", None)
    if isinstance(result, dict) and result.get("passed") is False:
        return False
    return True


_DID_NOT_RUN_CELL_SUFFIX = (
    " WARNING: workload never reached main work phase; matrix step-time/"
    "confound for this cell will be marked n/a"
)


def _format_cell_summary(trials: list[TrialResult], passed_count: int) -> tuple[str, str]:
    """Format the bracketed cell-summary tail for the per-cell log line.

    Returns ``(bracket, suffix)`` -- ``bracket`` is the
    ``[outcome: ... | iters: ...]`` (or legacy ``[N/M trials passed]``)
    chunk, ``suffix`` is either an empty string or the WARNING tail used
    for did-not-run cells (issue #173).

    Falls back to today's ``[N/M trials passed]`` shape when no trial in
    the cell populated the new ``main_work_started`` /
    ``executed_iterations`` / ``configured_iterations`` fields, so old
    workloads keep their familiar terminal output exactly. The new shape
    only kicks in once the workload speaks the new contract.
    """
    total = len(trials)
    # Mirror aggregate_cell's outcome rules without importing the helper
    # (which expects WorkloadResult-shaped objects vs. TrialResult here)
    # to avoid coupling the runner's log line to the matrix module's
    # internal helper signature.
    #
    # Switch to the new bracket grammar whenever the cell carries any
    # signal the new grammar surfaces:
    #   (a) any trial explicitly populates one of the new contract
    #       fields (so matrix.md will show the Iters column for this
    #       cell -- terminal log must agree), OR
    #   (b) platform inference classifies any trial as non-UNKNOWN
    #       (e.g. a legacy workload that crashed in setup -> inferred
    #       did_not_run -> the matrix.md row will carry the
    #       did_not_run tag).
    # Otherwise stay in legacy ``[N/M trials passed]`` so plain-vanilla
    # success runs keep their familiar terminal output exactly.
    outcomes: list[str] = []
    for trial in trials:
        result = getattr(trial, "result", {})
        outcomes.append(_outcome_for_log(trial, result if isinstance(result, dict) else {}))

    explicit_contract = any(
        isinstance(getattr(t, "result", None), dict)
        and (
            t.result.get("main_work_started") is not None
            or t.result.get("configured_iterations") is not None
            or t.result.get("executed_iterations") is not None
        )
        for t in trials
    )
    if not explicit_contract and all(o == OUTCOME_UNKNOWN for o in outcomes):
        return f"[{passed_count}/{total} trials passed]", ""

    counter: dict[str, int] = {}
    for o in outcomes:
        counter[o] = counter.get(o, 0) + 1
    if len(counter) == 1:
        only_outcome = next(iter(counter))
        outcome_part = f"{counter[only_outcome]}/{total} {only_outcome}"
    else:
        outcome_part = "mixed"

    iters_part = _iters_for_log(trials)
    bracket = f"[outcome: {outcome_part} | iters: {iters_part}]"
    suffix = _DID_NOT_RUN_CELL_SUFFIX if counter == {OUTCOME_DID_NOT_RUN: total} else ""
    return bracket, suffix


def _outcome_for_log(trial: Any, result: dict) -> str:
    """Trial-outcome classifier mirroring aorta.triage.matrix._trial_outcome.

    Two signal sources, in priority order:
    1. Explicit contract via ``main_work_started`` /
       ``executed_iterations`` / ``configured_iterations``.
    2. Platform inference for legacy workloads (``main_work_started``
       is None) -- non-ok exit_status with no per-step times and no
       elapsed time => ``did_not_run``. Mirrors
       ``aorta.triage.matrix._looks_like_did_not_run``; the two helpers
       are kept in lockstep so the terminal log and matrix.json agree
       on which trials get the inferred tag.
    """
    started = result.get("main_work_started")
    if started is False:
        return OUTCOME_DID_NOT_RUN
    if started is True:
        executed = result.get("executed_iterations")
        configured = result.get("configured_iterations")
        if not isinstance(executed, int) or not isinstance(configured, int):
            return OUTCOME_UNKNOWN
        if executed >= configured:
            return OUTCOME_COMPLETED
        if result.get("passed") is False:
            return OUTCOME_CRASHED_AFTER_ITERATIONS
        return OUTCOME_UNKNOWN
    # main_work_started is None -- platform inference branch.
    if _looks_like_did_not_run_for_log(trial, result):
        return OUTCOME_DID_NOT_RUN
    return OUTCOME_UNKNOWN


def _looks_like_did_not_run_for_log(trial: Any, result: dict) -> bool:
    """Mirror of ``aorta.triage.matrix._looks_like_did_not_run`` for the
    runner's per-cell log line. Three signals must agree: non-ok
    exit_status, no measured per-step times, and zero elapsed_sec.
    Kept here (rather than imported) so the runner doesn't depend on a
    matrix-private helper.
    """
    exit_status = getattr(trial, "exit_status", None)
    if exit_status is None or exit_status == "ok":
        return False
    step_times = result.get("step_times_ms")
    if isinstance(step_times, list) and any(
        isinstance(t, (int, float)) and t > 0 for t in step_times
    ):
        return False
    elapsed = result.get("elapsed_sec")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        return False
    return True


def _iters_for_log(trials: list[TrialResult]) -> str:
    """Pre-render the ``iters: N/M`` fragment for the cell-summary log.

    Same rules as ``aorta.triage.matrix._aggregate_iter_counts`` but
    operating on dicts. Returns ``"—"`` only for true legacy cells (no
    trial populated ``configured_iterations``); ``"?/?"`` when trials
    disagreed on the configured count; ``"—/<configured>"`` when the
    budget is known but ``executed_iterations`` is missing for some
    trial. The terminal log line and the matrix.md row should agree on
    what each cell looked like, so this helper mirrors the matrix-side
    rules byte-for-byte.
    """
    configured: list[int] = []
    executed: list[int | None] = []
    for trial in trials:
        result = getattr(trial, "result", {})
        if not isinstance(result, dict):
            executed.append(None)
            continue
        cfg = result.get("configured_iterations")
        if isinstance(cfg, int):
            configured.append(cfg)
        ex = result.get("executed_iterations")
        executed.append(ex if isinstance(ex, int) else None)

    if not configured:
        return "—"
    if len(set(configured)) > 1:
        return "?/?"
    cfg_value = configured[0]
    populated = [e for e in executed if e is not None]
    if len(populated) != len(executed) or not populated:
        return f"—/{cfg_value}"
    lo, hi = min(populated), max(populated)
    return f"{lo}/{cfg_value}" if lo == hi else f"{lo}..{hi}/{cfg_value}"


def _run_one_cell(
    cell,
    recipe: Recipe,
    run_dir: Path,
    sidecar_files: tuple[Path, ...],
    *,
    layout: Literal["timestamped", "flat_resume"] = "timestamped",
    resume_existing: bool = False,
    subprocess_argv: tuple[str, ...] | None = None,
) -> tuple[list[TrialResult], str | None, dict[str, str], list[str]]:
    """Execute a single cell through B1 and return (trials, error, env_vars, trial_paths).

    Exception handling scope is deliberately wide: any failure originating
    from B1 (unknown mitigation, workload crash in ``setup``, docker pull
    failure from a future docker-aware environment) should flag the cell as
    errored without bringing down the whole matrix. The full traceback is
    logged at WARNING so operators can diagnose, but the returned ``error``
    string stays short -- it's the text shown in matrix.md.

    ``layout`` / ``resume_existing`` / ``subprocess_argv`` activate the
    ``aorta probe`` (issue #188) code paths:

    * ``layout="flat_resume"`` puts the cell artifacts at
      ``<run_dir>/<safe_slug(cell.name)>/`` (NO ``cells/`` segment) so
      the artifact tree matches the rubric's
      ``<output>/<ticket>/<cell>/trial_<n>/...`` layout exactly.
    * ``resume_existing=True`` consults
      :func:`aorta.probe.resume.is_trial_complete` for every trial
      directory under the cell BEFORE invoking the dispatcher. When
      every trial is complete, the dispatcher call is skipped and the
      trial JSONs the dispatcher had written previously are surfaced
      as the cell's trials.
    * ``subprocess_argv`` (not None) flips the request to
      ``save_logs=True`` and routes the opaque user argv to the
      reserved ``_aorta_subprocess_argv`` slot -- the only legal
      channel for delivering argv to :class:`SubprocessWorkload`.
    """
    if layout == "flat_resume":
        cell_dir = run_dir / safe_slug(cell.name)
    else:
        cell_dir = _cells_dir(run_dir) / safe_slug(cell.name)
    cell_dir.mkdir(parents=True, exist_ok=True)

    resolved_env_vars = _resolve_cell_env_vars(cell.mitigations, cell.extra_env, sidecar_files)
    effective_trials = cell.effective_trials(recipe.trials)

    # Resume short-circuit: if every trial directory under this cell
    # carries a valid result.json + non-empty verdict, the cell is
    # already done. Hydrate the dispatcher's per-trial JSONs into
    # ``TrialResult`` objects so ``aggregate_cell`` sees the real
    # trial bodies (counts, timings, exit_status). Returning ``[]``
    # here would make matrix.md report ``trials=0/passed=0`` for a
    # skipped-but-complete cell -- the resume contract is "no extra
    # work, identical artifacts".
    #
    # Fall through to a full re-run when hydration cannot reconstruct
    # the full trial set (dispatcher JSON missing because the prior
    # run was killed between writing the probe ``result.json`` and
    # the dispatcher's ``trial_<N>.json``; or schema drift). Re-running
    # a complete cell is wasteful but preserves matrix correctness;
    # silently returning [] would not.
    if resume_existing and layout == "flat_resume":
        from aorta.probe.resume import is_trial_complete

        completed = sum(
            1 for i in range(effective_trials) if is_trial_complete(cell_dir / f"trial_{i}")
        )
        if completed >= effective_trials:
            trial_paths = _collect_trial_paths(cell_dir)
            # Hydrate into an index-keyed map. The key set is the
            # load-bearing invariant for the resume short-circuit: a
            # dispatcher JSON that fails to read or schema-validate
            # is absent from the keys, so a corrupted required
            # ``_t0.json`` co-existing with a stale extra
            # ``_t<N>.json`` (where N >= effective_trials) is caught
            # by the subset check below -- the previous
            # filename-set-based validation would have admitted it.
            # See ``_hydrate_trials_by_index`` for the failure modes
            # silently dropped on the floor.
            hydrated_by_index = _hydrate_trials_by_index(trial_paths)
            required_indices = set(range(effective_trials))
            if required_indices.issubset(hydrated_by_index.keys()):
                # Build the canonical (hydrated, trial_paths) pair from
                # the dict so positions line up by index: position i
                # carries the body and path for trial i. ``aggregate_cell``
                # records the path list in matrix.json::cells[*].trial_paths
                # in the same order as the trial bodies, so misalignment
                # here would silently scramble downstream cross-references.
                # Stale on-disk extras with index >= effective_trials are
                # dropped here (not deleted) so manual inspection of the
                # prior run is still possible; downstream aggregation
                # only sees what the current recipe asked for.
                hydrated = [hydrated_by_index[i].trial for i in range(effective_trials)]
                trial_paths = [hydrated_by_index[i].path for i in range(effective_trials)]
                log.info(
                    "cell %r: all %d trial(s) already complete -- skipping per resume mode",
                    cell.name,
                    effective_trials,
                )
                return hydrated, None, resolved_env_vars, trial_paths
            # Subset check failed -> at least one required index is missing.
            # The single-branch log subsumes the previous two-branch shape
            # (separate "indices missing" vs "count short" cases): with the
            # index-keyed map, an index that didn't hydrate IS by definition
            # both a missing index and a missing body, so distinguishing the
            # two on the operator side adds no diagnostic value.
            missing = sorted(required_indices - hydrated_by_index.keys())
            log.info(
                "cell %r: result.json marks %d trial(s) complete but "
                "successful hydration is missing trial indices %s "
                "(hydrated %s); re-running the cell so matrix counts "
                "stay consistent",
                cell.name,
                completed,
                missing,
                sorted(hydrated_by_index.keys()),
            )

    # Recipe-scope workload_config is the base; cell-scope merges over it so
    # the cell wins on key collision and non-collision keys union. Empty
    # dicts on both sides collapse to ``{}``, which RunRequest.__post_init__
    # deep-copies and the dispatcher spreads into ``config`` -- so omitting
    # workload_config from the recipe stays byte-equivalent to today.
    merged_workload_config = {**recipe.workload_config, **cell.workload_config}

    # Probe-mode extras delivered to ``SubprocessWorkload`` via the
    # typed ``RunRequest.probe_extras`` field. The runner is the only
    # producer of this dict; ``SubprocessWorkload`` reads it from the
    # ``_aorta_probe_extras`` slot the dispatcher injects.
    probe_extras_payload: dict[str, Any] | None = None
    probe_extras = recipe.probe_extras
    if subprocess_argv is not None and probe_extras is not None:
        probe_extras_payload = {
            "cell_name": cell.name,
            "env_passthrough_mode": probe_extras.env_passthrough_mode,
            "timeout_per_trial": probe_extras.timeout_per_trial,
            "cell_env_vars": dict(resolved_env_vars),
            "step_time_regex": probe_extras.step_time_regex,
            "collect_paths": list(probe_extras.collect_paths),
            # Phase 2 (issue #188): pre-compiled custom_patterns and
            # hang knobs forwarded to SubprocessWorkload. The
            # compiled CompiledPattern objects ride as a tuple --
            # ``RunRequest.__post_init__`` deep-copies the dict
            # (and therefore traverses this tuple), but the
            # CompiledPattern fields are all treated as immutable
            # post-compile (re.Pattern, CodeType, str, bool) so the
            # traversal is benign: nothing the dispatcher hands to
            # the workload can be mutated from outside RunRequest.
            "custom_patterns": tuple(probe_extras.custom_patterns),
            "hang_window_sec": probe_extras.hang_window_sec,
            "hang_grace_period_at_start": probe_extras.hang_grace_period_at_start,
            "tier3_vram_growth": probe_extras.tier3_vram_growth,
        }

    # save_logs is forced True for probe-mode cells because
    # SubprocessWorkload reads ``_aorta_log_prefix`` to derive its
    # per-trial directory; the dispatcher only injects that key when
    # save_logs=True.
    save_logs = recipe.save_logs or (subprocess_argv is not None)

    request = RunRequest(
        workload=recipe.workload,
        trials=effective_trials,
        environment=cell.environment,
        mitigations=tuple(cell.mitigations),
        extra_env=dict(cell.extra_env),
        steps=cell.effective_steps(recipe.steps),
        config_overrides=merged_workload_config,
        results_dir=cell_dir,
        sidecar_files=sidecar_files,
        save_logs=save_logs,
        subprocess_argv=subprocess_argv,
        probe_extras=probe_extras_payload,
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


def _print_dry_run(
    recipe: Recipe,
    subprocess_argv: tuple[str, ...] | None = None,
) -> None:
    """Write the resolved cell list to stdout without touching the filesystem.

    For probe-mode (``recipe.probe_extras is not None``), each line shows
    the cell's planned env-var bundle and the literal trailing argv -- the
    rubric's FR 1.2 contract. Triage-mode dry-run shows the existing
    cell summary (mitigations / environment / trials / steps) for back-compat.
    """
    if recipe.probe_extras is not None:
        # Probe-mode dry-run: cell name, resolved env bundle, literal argv.
        # No FS writes, no execution -- matches FR 1.2 byte-for-byte.
        argv_display = list(subprocess_argv) if subprocess_argv is not None else []
        click.echo(
            f"Dry run (probe): ticket={recipe.ticket or '(none)'} "
            f"cells={len(recipe.cells)} trials/cell={recipe.trials}"
        )
        click.echo(f"Argv (forwarded byte-for-byte): {argv_display}")
        click.echo(f"Env-passthrough mode: {recipe.probe_extras.env_passthrough_mode}")
        for cell in recipe.cells:
            env_bundle = recipe.probe_extras.cell_envs.get(cell.name, {})
            env_str = " ".join(f"{k}={v}" for k, v in sorted(env_bundle.items())) or "(no env)"
            click.echo(f"  {cell.name}: env=[{env_str}] argv={argv_display}")
        return

    click.echo(f"Dry run: {recipe.workload} / ticket={recipe.ticket or '(none)'}")
    if recipe.workload_config:
        click.echo(f"Recipe workload_config: {recipe.workload_config}")
    click.echo(f"Cells ({len(recipe.cells)}):")
    for cell in recipe.cells:
        # Mirror the runner's merge so the dry-run shows the EFFECTIVE config
        # the workload would be constructed with -- cell-scope wins on key
        # collision, non-collision keys union with recipe-scope.
        effective_workload_config = {**recipe.workload_config, **cell.workload_config}
        click.echo(
            f"  - {cell.name}: mitigations={list(cell.mitigations)} "
            f"environment={cell.environment} "
            f"trials={cell.effective_trials(recipe.trials)} "
            f"steps={cell.effective_steps(recipe.steps)}"
            + (f" extra_env={cell.extra_env}" if cell.extra_env else "")
            + (f" workload_config={effective_workload_config}" if effective_workload_config else "")
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
    *,
    layout: Literal["timestamped", "flat_resume"] = "timestamped",
    resume_existing: bool = False,
    subprocess_argv: tuple[str, ...] | None = None,
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
        layout: Output-tree layout (issue #188). ``"timestamped"`` (default)
            preserves byte-equivalence with ``aorta triage run``;
            ``"flat_resume"`` produces ``<output_dir>/<ticket>/`` (no
            timestamp, no ``<workload>`` segment) for ``aorta probe``'s
            resume semantics.
        resume_existing: When True (probe-mode only), per-cell
            invocations consult :func:`aorta.probe.resume.is_trial_complete`
            and skip cells whose every trial is already complete.
        subprocess_argv: Opaque argv forwarded byte-for-byte to every
            cell's :class:`SubprocessWorkload` via the typed
            ``RunRequest.subprocess_argv`` field. Required for probe-mode;
            ignored in triage-mode.

    Returns:
        The run directory path (``<output-dir>/<ticket>/<workload>/<timestamp>``
        for triage-mode, ``<output-dir>/<ticket>/`` for probe-mode).
    """
    # Preflight first, BEFORE the dry-run early-return: dry-run is documented
    # as "validation without execution", so it must reject everything the
    # real run would reject. Otherwise CI / pre-submit checks happily pass on
    # recipes that fail the moment they actually run.
    _preflight_validate(recipe)

    if dry_run:
        _print_dry_run(recipe, subprocess_argv=subprocess_argv)
        return Path(".")

    ts = timestamp or format_timestamp()

    # Under a distributed launcher (torchrun) EVERY rank runs this function:
    # the cells must execute on all ranks so the collectives inside the
    # workload actually complete. But only RANK 0 should write artifacts.
    # The per-trial JSON is already rank-0-gated in the dispatcher; here we
    # gate the matrix layer too. Without this, every non-zero rank calls
    # ``resolve_run_dir`` (mkdir(exist_ok=False)) and collides into a
    # redundant ``<timestamp>-2`` / ``-3`` / ... directory, then writes its
    # own matrix.md / matrix.json there. Non-zero ranks get a throwaway temp
    # run_dir (auto-removed) so every write in ``_run_recipe_locked`` lands
    # in scratch and is discarded; rank 0 gets the real one. Every write site
    # is rooted at ``run_dir``, so a scratch root is sufficient -- no
    # per-site gating needed.
    should_write = _is_rank_zero()

    with contextlib.ExitStack() as stack:
        if should_write:
            run_dir = resolve_run_dir(output_dir, recipe, timestamp=ts, layout=layout)
            # ``flat_resume`` reuses a stable ``<output>/<ticket>/`` directory
            # across invocations so resume can detect per-trial ``result.json``
            # files as already complete. That same property means two concurrent
            # ``aorta probe`` invocations against the same ``--output``/
            # ``--ticket`` would race on ``matrix.json``, ``matrix.md``, and
            # per-cell artifacts. Take an advisory PID+host lock for the
            # duration of the matrix run; the ``"timestamped"`` layout (``aorta
            # triage run``) is unaffected because ``resolve_run_dir`` already
            # gives each invocation a fresh leaf via the ``-N`` suffix loop.
            if layout == "flat_resume":
                stack.enter_context(acquire_flat_resume_lock(run_dir))
        else:
            # Non-rank-0: scratch root, removed on exit. No lock (we never
            # touch the shared tree).
            run_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="aorta-rank-")))

        return _run_recipe_locked(
            recipe=recipe,
            extra_sidecar_files=extra_sidecar_files,
            ts=ts,
            run_dir=run_dir,
            layout=layout,
            resume_existing=resume_existing,
            subprocess_argv=subprocess_argv,
        )


def _run_recipe_locked(
    *,
    recipe: Recipe,
    extra_sidecar_files: tuple[Path, ...],
    ts: str,
    run_dir: Path,
    layout: Literal["timestamped", "flat_resume"],
    resume_existing: bool,
    subprocess_argv: tuple[str, ...] | None,
) -> Path:
    """Body of :func:`run_recipe` after the flat_resume lock is held.

    Extracted so the lock-acquisition is a single
    ``contextlib.ExitStack`` block at the top of :func:`run_recipe` and
    the matrix-loop body doesn't need to be re-indented inside a ``with``.
    Not part of the public API -- callers go through :func:`run_recipe`.
    """
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
    total_cells = len(recipe.cells)
    log.info(
        "matrix run started: workload=%s ticket=%s cells=%d trials/cell=%d steps/trial=%d output=%s",
        recipe.workload,
        recipe.ticket or "(none)",
        total_cells,
        recipe.trials,
        recipe.steps,
        run_dir,
    )
    for cell_idx, cell in enumerate(recipe.cells, start=1):
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

        log.info(
            "cell %d/%d: %s (mitigations=%s environment=%s) -- starting %d trial(s)",
            cell_idx,
            total_cells,
            cell.name,
            list(cell.mitigations) or ["(none)"],
            cell.environment,
            cell.effective_trials(recipe.trials),
        )
        cell_t0 = time.perf_counter()
        trials, error, resolved_env_vars, trial_paths = _run_one_cell(
            cell,
            recipe,
            run_dir,
            sidecar_files,
            layout=layout,
            resume_existing=resume_existing,
            subprocess_argv=subprocess_argv,
        )
        cell_elapsed = time.perf_counter() - cell_t0
        if error is not None:
            log.info(
                "cell %d/%d: %s -- ERROR (%s) in %.1fs",
                cell_idx,
                total_cells,
                cell.name,
                error,
                cell_elapsed,
            )
        else:
            # Use the canonical pass/fail predicate from
            # ``aorta.triage.matrix._trial_passed`` so the per-cell log line
            # agrees with the matrix-level passed_count: a trial is "passed"
            # iff its exit_status is "ok" AND its WorkloadResult.passed is
            # not False. Reading only ``result.get("passed")`` ignores the
            # exit_status side -- ``aorta.run.dispatcher`` always populates
            # a WorkloadResult, but on infrastructure exceptions sets
            # exit_status="infrastructure_failed" with passed=False, so
            # both halves of the predicate are needed.
            passed = sum(1 for t in trials if _trial_passed_for_log(t))
            summary, suffix = _format_cell_summary(trials, passed)
            log.info(
                "cell %d/%d: %s -- done in %.1fs %s%s",
                cell_idx,
                total_cells,
                cell.name,
                cell_elapsed,
                summary,
                suffix,
            )

        # Mirror _run_one_cell's merge so CellStats.workload_config records
        # the effective user-supplied ``config_overrides`` for the cell
        # (recipe-scope + cell-scope workload_config merged; cell wins on
        # collision; non-collision keys union). NOT the full runtime config
        # the dispatcher constructs -- the dispatcher additionally injects
        # ``steps`` and ``_aorta_*`` keys (e.g. ``_aorta_environment``,
        # ``_aorta_save_logs``) which are runtime/platform-supplied and
        # deliberately not surfaced in the matrix. Stays {} when neither
        # scope sets workload_config.
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
            workload_config={**recipe.workload_config, **cell.workload_config},
        )
        cell_stats.append(stats)

    # Did-not-run baseline disqualification (issue #173). Three cases,
    # all of them produce matrix.md / matrix.json so the operator can
    # inspect the run; the runner raises ``MatrixIncompleteError`` at
    # the very end (after artifacts are written) for the two degraded
    # cases so the CLI can exit non-zero:
    #   1. Explicit ``confound.baseline_cell`` -> resolves first; if
    #      that cell ended up all-did_not_run, append a loud warning
    #      and mark the run incomplete. The matrix is still rendered
    #      (the explicit cell shows its did_not_run tag, others fall
    #      through to n/a) so the operator can see WHAT went wrong
    #      from artifacts alone.
    #   2. Auto-resolution -> SKIP all-did_not_run cells when applying
    #      the resolution rules. If a usable candidate remains,
    #      classify against that one normally -- no warning, no
    #      degradation flag. This matches the issue's "auto-selection
    #      skips all-did_not_run cells" rule.
    #   3. Auto-resolution exhausted -> soft warning + every non-
    #      baseline cell falls through to CONFOUND_NA via the existing
    #      ``baseline.mean_step_time_ms <= 0`` branch in
    #      confound.classify (the suppressed wall_clock_total fallback
    #      in aggregate_cell ensures the mean lands at 0). Run is
    #      marked incomplete.
    # Legacy workloads (empty outcome_counts) never enter the skip set,
    # so old runs keep their existing classification path unchanged.
    #
    # matrix.json::baseline_cell records the resolved name in cases 1
    # and 3 (rather than null). Preserving the name keeps a record of
    # which cell was attempted as the baseline -- useful for an
    # operator diagnosing why every other row collapsed to n/a -- while
    # the accompanying warning carries the "no usable comparison" signal.
    disqualified_names = {s.name for s in cell_stats if is_did_not_run_cell(s)}
    incomplete_reason: str | None = None

    if recipe.confound.baseline_cell is not None:
        baseline_cell = resolve_baseline(recipe.cells, recipe.confound.baseline_cell)
        if baseline_cell.name in disqualified_names:
            baseline_stats_ref = next(c for c in cell_stats if c.name == baseline_cell.name)
            msg = (
                f"explicit baseline_cell {baseline_cell.name!r} produced only "
                f"did_not_run trials "
                f"({baseline_stats_ref.outcome_counts.get(OUTCOME_DID_NOT_RUN, 0)}/"
                f"{baseline_stats_ref.trials} trials, workload never reached "
                "its main work phase). Confound classification disabled; "
                f"non-baseline cells render n/a. Inspect "
                f"cells/{safe_slug(baseline_cell.name)}/ for the failure "
                "cause, or remove confound.baseline_cell from the recipe to "
                "fall back to auto-resolution."
            )
            warnings.append(msg)
            incomplete_reason = msg
    else:
        try:
            baseline_cell = resolve_baseline(recipe.cells, None, skip_names=disqualified_names)
        except RecipeCellError:
            # Every candidate was disqualified -- fall back to the original
            # auto-resolution (without the skip set) so matrix.json still
            # records WHO would have been the baseline, then warn loudly.
            baseline_cell = resolve_baseline(recipe.cells, None)
            baseline_stats_ref = next(c for c in cell_stats if c.name == baseline_cell.name)
            msg = (
                f"Auto-baseline {baseline_cell.name!r} had "
                f"{baseline_stats_ref.outcome_counts.get(OUTCOME_DID_NOT_RUN, 0)}/"
                f"{baseline_stats_ref.trials} trials in did_not_run, and no "
                "other cell in the recipe survived auto-baseline resolution "
                "either. No usable baseline -- confound classification "
                "disabled. Inspect "
                f"cells/{safe_slug(baseline_cell.name)}/ for the failure cause."
            )
            warnings.append(msg)
            incomplete_reason = msg

    baseline_stats = next(c for c in cell_stats if c.name == baseline_cell.name)
    confound_tags = classify_all(cell_stats, baseline_cell.name, recipe.confound.threshold)

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

    if incomplete_reason is not None:
        # Artifacts are written; raise so the CLI can print the failure
        # message and exit non-zero. Tests / programmatic callers can
        # ``except MatrixIncompleteError`` to inspect ``run_dir``.
        raise MatrixIncompleteError(incomplete_reason, run_dir)

    return run_dir


__all__ = ["MatrixIncompleteError", "run_recipe"]
