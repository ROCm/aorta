"""Output layout + writers for the triage matrix.

Three artifacts per run, all in the same ``<output-dir>/<ticket>/<workload>/
<run-timestamp>/`` directory:

* ``matrix.md`` -- human-readable table matching the §"matrix.md target
  format" block in issue #151.
* ``matrix.json`` -- full machine-readable per-cell data (step-time arrays,
  env-var bundles, trial JSON paths, cell-level errors, etc.).
* ``recipe.resolved.yaml`` -- the recipe AS EXECUTED, with every registry
  name expanded to its underlying env-var bundle + docker ref. This is the
  reproducibility artifact: re-running it on a different machine produces
  the same matrix even if the registries drift in the interim.

Sibling files / directories (written by :mod:`aorta.triage.runner`):

* ``host_env.json`` -- one collect_env() snapshot taken at runner start.
* ``environments/<env-name>/env.json`` -- one collect_env() snapshot per
  unique environment, captured right before that env's first cell runs.
* ``cells/<cell-name>/`` -- B1's per-trial JSON output for that cell.
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

from aorta.registry import get_environment, get_mitigation
from aorta.triage.confound import ConfoundTag
from aorta.triage.matrix import CellStats
from aorta.triage.recipe import Recipe

NO_TICKET_SLUG = "_no_ticket_"

# Filesystem-safe slug: replace anything that isn't [A-Za-z0-9_.-] with '_'.
# Ticket IDs like "PROJ-123" pass through unchanged; spaces, slashes, ':'
# etc. get sanitised so we never create surprise subdirectories.
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def safe_slug(value: str) -> str:
    """Turn a ticket / workload / env name into a safe directory component."""
    cleaned = _SAFE_RE.sub("_", value)
    return cleaned or "_"


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
    """Return ``<output-dir>/<ticket>/<workload>/<timestamp>/``.

    Creates parents as needed. Never overwrites: the timestamp component is
    unique per invocation, so re-running against the same ticket always
    produces a fresh directory.
    """
    ticket_slug = safe_slug(recipe.ticket) if recipe.ticket else NO_TICKET_SLUG
    workload_slug = safe_slug(recipe.workload)
    ts = timestamp or format_timestamp()
    run_dir = Path(output_dir) / ticket_slug / workload_slug / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _format_mitigations(mitigations: Iterable[str]) -> str:
    items = list(mitigations)
    return ", ".join(items) if items else "-"


def _format_nan_rate(cell: CellStats) -> str:
    if cell.error is not None:
        return "n/a"
    pct = int(round(cell.nan_rate * 100))
    return f"{pct}%"


def _format_fail_count(cell: CellStats) -> str:
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
        "NaN rate",
        "Trials",
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
                _format_nan_rate(cell),
                _format_fail_count(cell),
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
    lines.append("  - `error` -- the whole cell failed; row preserved so the matrix is complete.")
    lines.append(
        "- Only `mean step (ms)` is shown here. Per-cell `std`, `p50`, `p99`, raw "
        "step-time arrays, and per-trial JSON paths are in `matrix.json`."
    )
    lines.append(
        "- `recipe.resolved.yaml` (alongside this file) captures the registry state "
        "at run time -- re-run it to reproduce."
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
) -> None:
    """Serialise the full per-cell matrix as JSON."""
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
        entry["nan_rate"] = cell.nan_rate
        entry["confound"] = tag
        entry["step_time_ratio"] = ratio
        doc["cells"].append(entry)

    path.write_text(json.dumps(doc, indent=2, sort_keys=False), encoding="utf-8")


def write_resolved_recipe(
    path: Path,
    recipe: Recipe,
    sidecar_files: tuple[Path, ...] | None = None,
) -> None:
    """Write recipe.resolved.yaml with every registry name expanded.

    Expansion rules:

    * Each cell gets a ``resolved_mitigation_env`` block containing the
      unioned env-var bundle from its mitigations (same order semantics as
      the runner uses at execution time) plus the cell's ``extra_env``
      overlay -- this is the exact env-var set applied to the trials.
    * Each cell gets a ``resolved_environment`` block containing the
      registry :class:`Environment` descriptor (or, for inline docker, the
      ``{name, docker}`` pair the auto-registration produced).
    * The top-level ``schema_version`` is preserved so a reader can tell
      this is a triage recipe snapshot, not an arbitrary YAML file.
    """
    extra = list(sidecar_files) if sidecar_files else None

    inline_envs = {e.name: e.docker for e in recipe.inline_environments}

    resolved_cells: list[dict[str, Any]] = []
    for cell in recipe.cells:
        mit_union: dict[str, str] = {}
        mit_contributions: list[dict[str, Any]] = []
        for name in cell.mitigations:
            bundle = get_mitigation(name, extra_files=extra)
            mit_contributions.append({"name": name, "env": dict(bundle)})
            mit_union.update(bundle)
        mit_union.update(cell.extra_env)

        if cell.environment in inline_envs:
            resolved_env: dict[str, Any] = {
                "name": cell.environment,
                "docker": inline_envs[cell.environment],
                "inline": True,
            }
        else:
            env_desc = get_environment(cell.environment, extra_files=extra)
            resolved_env = {
                "name": env_desc.name,
                "docker": env_desc.docker,
                "venv": env_desc.venv,
                "source_package": env_desc.source_package,
                "inline": False,
            }

        resolved_cells.append(
            {
                "name": cell.name,
                "mitigations": list(cell.mitigations),
                "mitigation_contributions": mit_contributions,
                "environment": cell.environment,
                "resolved_environment": resolved_env,
                "extra_env": dict(cell.extra_env),
                "resolved_mitigation_env": mit_union,
                "trials": cell.effective_trials(recipe.trials),
                "steps": cell.effective_steps(recipe.steps),
            }
        )

    doc = {
        "schema_version": recipe.schema_version,
        "ticket": recipe.ticket,
        "workload": recipe.workload,
        "trials": recipe.trials,
        "steps": recipe.steps,
        "confound": {
            "threshold": recipe.confound.threshold,
            "baseline_cell": recipe.confound.baseline_cell,
        },
        "inline_environments": [
            {"name": e.name, "docker": e.docker} for e in recipe.inline_environments
        ],
        "cells": resolved_cells,
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


__all__ = [
    "NO_TICKET_SLUG",
    "format_timestamp",
    "resolve_run_dir",
    "safe_slug",
    "write_matrix_json",
    "write_matrix_md",
    "write_resolved_recipe",
]
