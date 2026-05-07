"""Output layout + writers for the triage matrix.

Three artifacts per run, all in the same ``<output-dir>/<ticket>/<workload>/
<run-timestamp>/`` directory:

* ``matrix.md`` -- human-readable table matching the §"matrix.md target
  format" block in issue #151.
* ``matrix.json`` -- full machine-readable per-cell data: step-time stats,
  the env-var bundle as actually applied (``resolved_env_vars``), the
  resolved :class:`aorta.registry.Environment` descriptor
  (``resolved_environment``), trial JSON paths, exit-status histogram, and
  cell-level errors.
* ``recipe.resolved.yaml`` -- a strict, reloadable recipe snapshot. **Named
  mitigations and environments are deliberately NOT expanded** -- the file
  is the rerunnable artifact, not a drift-pinning artifact. Inline-docker
  cells are re-emitted as ``{docker: <ref>}`` so the same ``_inline_<hash>``
  is re-derived without a sidecar; named registry entries stay by name.
  See :func:`write_resolved_recipe` for the rationale and ``recipes/README.md``
  for replay caveats. To detect registry drift across runs, compare each
  run's ``matrix.json::cells[*].resolved_env_vars``.

Sibling files / directories (written by :mod:`aorta.triage.runner`):

* ``host_env.json`` -- one collect_env() snapshot taken at runner start.
* ``environments/<env-name>/env.json`` -- one collect_env() snapshot per
  unique environment, captured right before that env's first cell runs.
* ``cells/<cell-name>/`` -- B1's per-trial JSON output for that cell.
* ``sidecars/<basename>`` -- byte-identical copy of every operator-supplied
  ``--mitigations-file`` so the run dir is self-contained for replay.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from aorta.registry import get_environment
from aorta.triage.confound import ConfoundTag
from aorta.triage.matrix import CellStats
from aorta.triage.recipe import Recipe

NO_TICKET_SLUG = "_no_ticket_"

# Filesystem-safe slug: replace anything that isn't [A-Za-z0-9_.-] with '_'.
# Ticket IDs like "PROJ-123" pass through unchanged; spaces, slashes, ':'
# etc. get sanitised so we never create surprise subdirectories.
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")
# Even after character-class scrubbing, "." and ".." are still meaningful path
# components on every filesystem we care about.  Keep them out of the output
# tree so a ticket like ".." can't move the run directory up a level.
_RESERVED_SLUGS = frozenset({".", ".."})


def safe_slug(value: str) -> str:
    """Turn a ticket / workload / env name into a safe directory component.

    Replaces anything outside ``[A-Za-z0-9_.-]`` with ``_`` and rewrites the
    reserved ``.`` / ``..`` components so the result can never refer to the
    current or parent directory. Empty input also rewrites to ``_``.
    """
    cleaned = _SAFE_RE.sub("_", value)
    if not cleaned or cleaned in _RESERVED_SLUGS:
        return "_"
    return cleaned


def format_timestamp(now: _dt.datetime | None = None) -> str:
    """Return an ISO-8601-ish timestamp suitable as a directory name.

    ``2026-04-28T14-12-03`` matches the layout shown in issue #151 §"Output
    layout". Colons are replaced with dashes because Windows filesystems
    reject them and it makes the path easier to copy-paste from a shell.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%S")


def resolve_run_dir(
    output_dir: Path,
    recipe: Recipe,
    timestamp: str | None = None,
) -> Path:
    """Return ``<output-dir>/<ticket>/<workload>/<timestamp>[-N]/``.

    Creates parents as needed. **Never overwrites an existing directory.**
    The base candidate is ``<timestamp>``; if that already exists (two runs
    in the same wall-clock second for the same ``(ticket, workload)`` --
    common in CI loops or concurrent jobs), a numeric suffix ``-2``, ``-3``,
    ... is appended until ``mkdir(exist_ok=False)`` succeeds. The race
    between two parallel processes is resolved by ``mkdir`` itself: only one
    can win for a given suffix, the loser bumps and retries.

    The base directory ``<output_dir>/<ticket>/<workload>/`` IS created with
    ``exist_ok=True`` -- it's a shared parent across runs and the
    "no-overwrite" guarantee only applies to the per-run leaf.
    """
    ticket_slug = safe_slug(recipe.ticket) if recipe.ticket else NO_TICKET_SLUG
    workload_slug = safe_slug(recipe.workload)
    ts = timestamp or format_timestamp()
    parent = Path(output_dir) / ticket_slug / workload_slug
    parent.mkdir(parents=True, exist_ok=True)

    # Try the bare timestamp first, then -2, -3, ... -- bounded so a buggy
    # caller can't spin forever. The cap is generous (10k runs in the same
    # second-bucket would already imply something is very wrong upstream)
    # but finite so failures surface as a clean error rather than a hang.
    for suffix in range(1, 10_001):
        candidate = parent / (ts if suffix == 1 else f"{ts}-{suffix}")
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(
        f"resolve_run_dir: exhausted 10000 suffixes for {parent / ts!r}; "
        "something is wedged upstream (clock not advancing? runaway loop?)"
    )


def _format_mitigations(mitigations: Iterable[str]) -> str:
    items = list(mitigations)
    return ", ".join(items) if items else "-"


def _format_failure_rate(cell: CellStats) -> str:
    if cell.error is not None:
        return "n/a"
    pct = int(round(cell.failure_rate * 100))
    return f"{pct}%"


def _format_failures(cell: CellStats) -> str:
    """Render the Failures column as ``failed / trials`` (e.g. ``3 / 8``).

    Both numerator and denominator are spelled out so the column header
    ("Failures") and the value together read as "3 failures out of 8 trials"
    -- the previous "Trials" header on this same value confused readers
    into reading 3/8 as a trial-count column (issue #160 review round 6).
    """
    if cell.error is not None:
        return "n/a"
    return f"{cell.failed_count} / {cell.trials}"


def _format_step_ms(cell: CellStats) -> str:
    if cell.error is not None or cell.mean_step_time_ms <= 0:
        return "n/a"
    return f"{cell.mean_step_time_ms:.1f}"


def _format_confound(tag: ConfoundTag) -> str:
    return str(tag)


def write_matrix_md(
    path: Path,
    recipe: Recipe,
    cell_stats: list[CellStats],
    baseline: CellStats,
    confound_tags: dict[str, tuple[ConfoundTag, float | None]],
    warnings: list[str],
    run_timestamp: str,
) -> None:
    """Render matrix.md in the format from issue #151 §"matrix.md target format"."""
    lines: list[str] = []
    lines.append(f"# Triage Matrix - {recipe.workload}")
    lines.append("")
    if warnings:
        lines.append("> [!WARNING]")
        for w in warnings:
            lines.append(f"> {w}")
        lines.append("")

    lines.append(f"**Ticket**: {recipe.ticket or '(none)'}  ")
    lines.append(f"**Workload**: {recipe.workload}  ")
    recipe_line = "**Recipe**: "
    if recipe.source_path is not None:
        sha = (recipe.source_sha256 or "")[:10]
        recipe_line += f"{recipe.source_path} (sha256:{sha})"
    else:
        recipe_line += "(flag-mode; in-memory)"
    lines.append(recipe_line + "  ")
    lines.append(f"**Trials per cell**: {recipe.trials}  ")
    lines.append(f"**Steps per trial**: {recipe.steps}  ")
    lines.append(f"**Run timestamp**: {run_timestamp}  ")
    baseline_step = (
        f"{baseline.mean_step_time_ms:.1f} ms"
        if baseline.error is None and baseline.mean_step_time_ms > 0
        else "n/a"
    )
    lines.append(f"**Baseline cell**: {baseline.name} (mean step time = {baseline_step})")
    lines.append("")
    lines.append("## Reproduction Summary")
    lines.append("")

    header = (
        "Cell",
        "Mitigations",
        "Environment",
        "Failure rate",
        "Failures",
        "Mean step (ms)",
        "Confound",
    )
    rows: list[tuple[str, ...]] = [header]
    for cell in cell_stats:
        tag, _ = confound_tags.get(cell.name, (cell.error and "error" or "-", None))
        rows.append(
            (
                cell.name,
                _format_mitigations(cell.mitigations),
                cell.environment,
                _format_failure_rate(cell),
                _format_failures(cell),
                _format_step_ms(cell),
                _format_confound(tag),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

    def _row(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines.append(_row(rows[0]))
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows[1:]:
        lines.append(_row(row))

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Cell name comes from the recipe; mitigations + environment columns "
        "disambiguate when names get terse."
    )
    lines.append("- Confound column legend:")
    lines.append("  - `(baseline)` -- the cell against which all step-time ratios are computed.")
    lines.append("  - `-` -- the mitigation appears to work without a speed cost. Trust this cell.")
    lines.append(
        "  - `speed (+N%)` -- the mitigation may be suppressing failure via slower "
        "iteration rather than a real fix. Verify with `rocprofv3` dispatch comparison "
        "before drawing causal conclusions."
    )
    lines.append(
        "  - `no effect` -- the mitigation neither changed the failure rate nor slowed "
        "iteration; it likely doesn't apply to this workload."
    )
    lines.append(
        "  - `n/a` -- the cell could not be compared against the baseline. Possible "
        "reasons: the baseline errored, the baseline produced no usable timing, the "
        "cell itself produced no usable timing, or the cell and baseline derived "
        "their step-time from different fallback branches (e.g. `per_step` vs "
        "`wall_clock_total`) so the ratio would mix fundamentally different "
        "signals. Distinct from `-`: these cells are **unclassified**, not "
        "trustworthy. Check `matrix.json::cells[*].step_time_source` to see which "
        "branch each row landed on."
    )
    lines.append("  - `error` -- the whole cell failed; row preserved so the matrix is complete.")
    lines.append(
        "- `Failures` is `failed_count / trial_count` (e.g. `3 / 8` = three failed out "
        "of eight). `Failure rate` is the same data as a percentage and counts every "
        "trial whose `exit_status != ok` or whose `WorkloadResult.passed` is False; "
        "neither is NaN-specific. Use `matrix.json::cells[*].exit_status_counts` to "
        "break failures down by mode (`workload_failed` vs `infrastructure_failed` "
        "vs `unknown`, etc.)."
    )
    lines.append(
        "- Only `mean step (ms)` is shown here. Per-cell `std`, `min`, `max`, "
        "`p50`, `p90`, `p99`, raw step-time arrays, exit-status histogram, and "
        "per-trial JSON paths are in `matrix.json`."
    )
    lines.append(
        "- `recipe.resolved.yaml` (alongside this file) is a strict, reloadable "
        "recipe: inline-docker cells are pinned via `{docker: <ref>}`, but named "
        "mitigations / environments are NOT expanded -- a later registry change "
        "can produce a different run. Compare `matrix.json::cells[*].resolved_env_vars` "
        "between runs to detect drift. See `recipes/README.md` for full replay caveats."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_json(
    path: Path,
    recipe: Recipe,
    cell_stats: list[CellStats],
    baseline_name: str,
    confound_tags: dict[str, tuple[ConfoundTag, float | None]],
    run_timestamp: str,
    warnings: list[str],
    sidecar_files: tuple[Path, ...] | None = None,
) -> None:
    """Serialise the full per-cell matrix as JSON.

    Each cell entry carries (a) the aggregated stats from
    :class:`aorta.triage.matrix.CellStats`, (b) the resolved
    :class:`aorta.registry.Environment` descriptor (formerly written into
    ``recipe.resolved.yaml`` -- moved here so that file can stay schema-valid
    and re-loadable by :func:`aorta.triage.recipe.load_recipe`), and (c) the
    confound tag + step-time ratio.
    """
    doc: dict[str, Any] = {
        "schema_version": 1,
        "workload": recipe.workload,
        "ticket": recipe.ticket,
        "trials_per_cell": recipe.trials,
        "steps_per_trial": recipe.steps,
        "run_timestamp": run_timestamp,
        "baseline_cell": baseline_name,
        "confound": {
            "threshold": recipe.confound.threshold,
            "baseline_cell_configured": recipe.confound.baseline_cell,
        },
        "warnings": list(warnings),
        "recipe_source": {
            "path": str(recipe.source_path) if recipe.source_path else None,
            "sha256": recipe.source_sha256,
        },
        "cells": [],
    }
    for cell in cell_stats:
        tag, ratio = confound_tags.get(cell.name, ("-", None))
        entry = asdict(cell)
        entry["failure_rate"] = cell.failure_rate
        entry["confound"] = tag
        entry["step_time_ratio"] = ratio
        try:
            entry["resolved_environment"] = resolved_cell_environment(
                cell.environment,
                inline_environments=recipe.inline_environments,
                sidecar_files=sidecar_files,
            )
        except Exception as exc:  # pragma: no cover - belt-and-suspenders
            entry["resolved_environment"] = {
                "name": cell.environment,
                "_resolution_error": f"{type(exc).__name__}: {exc}",
            }
        doc["cells"].append(entry)

    path.write_text(json.dumps(doc, indent=2, sort_keys=False), encoding="utf-8")


def write_resolved_recipe(
    path: Path,
    recipe: Recipe,
    sidecar_files: tuple[Path, ...] | None = None,
) -> None:
    """Write a schema-valid ``recipe.resolved.yaml`` that can be re-loaded as-is.

    The output is a strict :func:`aorta.triage.recipe.load_recipe` input -- no
    debug-only fields, no per-cell ``resolved_mitigation_env`` block, no
    top-level ``inline_environments`` key. The "resolved" property comes from
    inline-docker cells being re-emitted in the ``{docker: <ref>}`` shorthand
    form (so re-loading on another machine reproduces the same
    ``_inline_<hash>`` env without needing a sidecar JSON next to the file).

    Per-cell debug expansions (the ``Environment`` descriptor and the unioned
    mitigation env-var bundle) live in ``matrix.json`` instead, where they
    belong as run-time state.

    For runs that used ``--mitigations-file``, the resolved YAML still
    references those mitigation/environment names by name -- it is **not**
    self-contained on its own. The runner snapshots the operator-supplied
    sidecars into ``<run_dir>/sidecars/<basename>`` (see
    :func:`aorta.triage.runner._copy_operator_sidecars`) so the rerun
    command is::

        aorta triage run --recipe recipe.resolved.yaml \\
            --mitigations-file sidecars/<basename>  # repeat per sidecar

    The resolved YAML schema is intentionally kept strict (no metadata
    namespace) -- the README documents the replay command, and the runner
    echoes a one-line "to rerun:" hint on stdout when sidecars are present.
    ``sidecar_files`` is intentionally unused at write time; it is preserved
    on the signature so the runner can keep its single
    "everything-needed-for-replay" call site.
    """
    del sidecar_files  # see docstring; kept for caller-stable signature

    inline_docker = {e.name: e.docker for e in recipe.inline_environments}

    resolved_cells: list[dict[str, Any]] = []
    for cell in recipe.cells:
        if cell.environment in inline_docker:
            cell_env: Any = {"docker": inline_docker[cell.environment]}
        else:
            cell_env = cell.environment
        cell_doc: dict[str, Any] = {
            "name": cell.name,
            "mitigations": list(cell.mitigations),
            "environment": cell_env,
        }
        if cell.extra_env:
            cell_doc["extra_env"] = dict(cell.extra_env)
        if cell.trials is not None:
            cell_doc["trials"] = cell.trials
        if cell.steps is not None:
            cell_doc["steps"] = cell.steps
        resolved_cells.append(cell_doc)

    doc: dict[str, Any] = {
        "schema_version": recipe.schema_version,
        "workload": recipe.workload,
        "trials": recipe.trials,
        "steps": recipe.steps,
    }
    if recipe.ticket is not None:
        doc["ticket"] = recipe.ticket
    doc["confound"] = {
        "threshold": recipe.confound.threshold,
        "baseline_cell": recipe.confound.baseline_cell,
    }
    doc["cells"] = resolved_cells

    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def resolved_cell_environment(
    cell_environment: str,
    inline_environments: tuple,
    sidecar_files: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Return the resolved environment descriptor for a cell.

    Centralised so both ``write_matrix_json`` and any future audit / debug
    tooling agree on the shape: registered envs project their full
    :class:`aorta.registry.Environment` descriptor; inline-docker cells emit
    the shorthand needed to re-derive the same ``_inline_<hash>``.
    """
    extra = list(sidecar_files) if sidecar_files else None
    inline_docker = {e.name: e.docker for e in inline_environments}
    if cell_environment in inline_docker:
        return {
            "name": cell_environment,
            "docker": inline_docker[cell_environment],
            "venv": None,
            "source_package": "_inline_",
            "inline": True,
        }
    env_desc = get_environment(cell_environment, extra_files=extra)
    return {
        "name": env_desc.name,
        "docker": env_desc.docker,
        "venv": env_desc.venv,
        "source_package": env_desc.source_package,
        "inline": False,
    }


__all__ = [
    "NO_TICKET_SLUG",
    "format_timestamp",
    "resolve_run_dir",
    "resolved_cell_environment",
    "safe_slug",
    "write_matrix_json",
    "write_matrix_md",
    "write_resolved_recipe",
]
