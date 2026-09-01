"""Find the user's aorta run directories and render their artifacts as text.

Sits between :mod:`aorta.artifacts` and the two things that consume it -- the
chat tools in :mod:`aorta.chat.tools.artifacts` and the run-artifact RAG
collection in :mod:`aorta.chat.rag.runs`. Both need the same discovery walk and
the same rendering, and a second copy of either would drift.

Rendering is prose-shaped rather than raw JSON on purpose. A raw ``env.json`` is
tens of kilobytes of mostly-null probe blocks, which chunks into noise and
crowds out the source retrieval; the fields a failure question actually turns
on are few and worth naming explicitly.

The one rule every renderer here holds to is the reader's own: **an absent field
is never rendered as a zero.** ``aorta.artifacts`` reports it as ``None`` and
lists it in ``missing_fields``, and these renderers print ``unknown`` plus an
explicit note. Rendering it as ``0`` would hand the model a clean bill of health
for a run whose results were never recorded -- the exact silent
misclassification the typed readers exist to prevent.

Stdlib plus :mod:`aorta.artifacts` only. No langchain here, so the tools layer
can import it without the RAG stack.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from aorta.artifacts import (
    ENV_FILENAME,
    HOST_ENV_FILENAME,
    MATRIX_FILENAME,
    ArtifactReadError,
    EnvArtifact,
    MatrixArtifact,
    MatrixCell,
    read_env,
    read_matrix,
)

#: Artifact filenames worth indexing, mapped to the kind tag they carry into
#: retrieval metadata. ``result.json`` is deliberately absent: nothing consumes
#: it, and one per trial would swamp the collection with near-identical chunks.
ARTIFACT_KINDS: dict[str, str] = {
    MATRIX_FILENAME: "matrix",
    ENV_FILENAME: "env",
    HOST_ENV_FILENAME: "env",
}

#: Directory names never descended into while looking for run artifacts. Run
#: trees are shallow, but they are routinely created inside a source checkout.
_PRUNED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "site-packages",
    }
)

#: Depth limit on the walk, counted in path components below the root. Deep
#: enough for ``<root>/<run>/environments/<name>/env.json``, shallow enough that
#: pointing the setting at a home directory does not become a full filesystem
#: scan.
_MAX_DEPTH = 4



def _sorted_children(directory: Path) -> tuple[list[Path], list[Path]]:
    """Return ``(subdirectories, files)`` of *directory*, or empties if unreadable."""
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return [], []
    dirs = [e for e in entries if e.is_dir() and not e.is_symlink()]
    files = [e for e in entries if e.is_file()]
    return dirs, files


def iter_artifacts(root: Path, max_depth: int = _MAX_DEPTH) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, kind)`` for every run artifact under *root*.

    Breadth-first and depth-capped, so the common case of a handful of run
    directories beside the recipe costs a handful of ``iterdir`` calls.
    Symlinked directories are not followed: a run directory pointing at ``/``
    would otherwise turn this into a filesystem crawl.
    """
    if not root.is_dir():
        return
    frontier = [(root, 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        dirs, files = _sorted_children(directory)
        for path in files:
            kind = ARTIFACT_KINDS.get(path.name)
            if kind is not None:
                yield path, kind
        if depth < max_depth:
            frontier.extend(
                (child, depth + 1) for child in dirs if child.name not in _PRUNED_DIRS
            )


def find_run_dirs(root: Path, max_depth: int = _MAX_DEPTH) -> list[Path]:
    """Directories holding at least one run artifact, nearest the root first."""
    seen: dict[Path, None] = {}
    for path, _ in iter_artifacts(root, max_depth=max_depth):
        seen.setdefault(path.parent, None)
    return list(seen)


# ── rendering ─────────────────────────────────────────────────────────────

#: Printed in place of a value the artifact did not carry. Not ``0`` and not
#: ``-``: it has to read as "not recorded" to a model that will quote it.
_UNKNOWN = "unknown"


def _value(value: object | None) -> str:
    return _UNKNOWN if value is None else str(value)


def _schema_note(schema_version: object, status: str, note: str | None) -> list[str]:
    """Render the schema banner, loudly when the version is not the known one."""
    if status == "supported":
        return [f"schema_version: {schema_version}"]
    return [
        f"schema_version: {schema_version} ({status.upper()})",
        f"  note: {note}" if note else f"  note: unrecognised schema ({status})",
    ]


def _missing_note(missing: tuple[str, ...], subject: str) -> list[str]:
    if not missing:
        return []
    return [
        f"NOT RECORDED in this {subject}: {', '.join(missing)}",
        "  (these are absent from the artifact, which is not the same as zero)",
    ]


def _render_names(values: tuple[str, ...] | None) -> str:
    """Render a name tuple, keeping absent and empty distinguishable."""
    if values is None:
        return _UNKNOWN
    return ", ".join(values) if values else "none"


def render_matrix_cell(cell: MatrixCell) -> str:
    """Render one ``cells[]`` entry as a short block."""
    rate = cell.failure_rate
    lines = [
        f"cell: {_value(cell.name)}",
        f"  mitigations: {_render_names(cell.mitigations)}",
        f"  trials: {_value(cell.trials)}",
        f"  passed: {_value(cell.passed_count)}",
        f"  failed: {_value(cell.failed_count)}",
        f"  failure_rate: {_UNKNOWN if rate is None else format(rate, '.3f')}",
        f"  workload_failed trials: {_value(cell.workload_failed_count)}",
    ]

    if cell.failure_hints is None:
        lines.append(f"  failure_hints: {_UNKNOWN}")
    elif cell.failure_hints:
        lines.append("  failure_hints:")
        for hint in cell.failure_hints:
            count = "" if hint.trials is None else f" (x{hint.trials})"
            lines.append(f"    - {hint.text}{count}")
    else:
        lines.append("  failure_hints: none")

    if cell.error:
        lines.append(f"  cell error: {cell.error}")
    lines.extend(f"  {line}" for line in _missing_note(cell.missing_fields, "cell"))
    return "\n".join(lines)


def render_matrix(matrix: MatrixArtifact, *, include_cells: bool = True) -> str:
    """Render a ``matrix.json`` as the text a model reads or the index stores."""
    source = matrix.source_path
    lines = [f"AORTA triage matrix: {source}" if source else "AORTA triage matrix"]
    lines.extend(_schema_note(matrix.schema_version, matrix.schema_status, matrix.schema_note))
    lines.append(f"workload: {_value(matrix.workload)}")
    lines.append(f"ticket: {_value(matrix.ticket)}")
    lines.append(f"run_timestamp: {_value(matrix.run_timestamp)}")
    lines.append(f"steps_per_trial: {_value(matrix.steps_per_trial)}")
    lines.append(f"trials_per_cell: {_value(matrix.trials_per_cell)}")
    lines.append(f"cells: {len(matrix.cells)}")
    lines.extend(_missing_note(matrix.missing_fields, "matrix"))

    if include_cells and matrix.cells:
        lines.append("")
        lines.extend(render_matrix_cell(cell) for cell in matrix.cells)
    return "\n".join(lines)


def render_env(env: EnvArtifact) -> str:
    """Render an ``env.json`` / ``host_env.json`` down to the triage-relevant facts."""
    source = env.source_path
    lines = [f"AORTA environment snapshot: {source}" if source else "AORTA environment snapshot"]
    lines.extend(_schema_note(env.schema_version, env.schema_status, env.schema_note))
    lines.append(f"captured_at: {_value(env.captured_at)}")
    lines.append(f"rocm.version: {_value(env.rocm_version)}")

    if env.partial is None:
        # The probe's honesty signal is itself missing, so nothing here can be
        # read as "the probe was clean".
        lines.append(f"probe complete: {_UNKNOWN}")
    elif env.partial:
        reasons = env.partial_reasons
        lines.append("probe complete: NO -- this snapshot is partial")
        if reasons is None:
            lines.append(f"  partial_reasons: {_UNKNOWN}")
        else:
            lines.extend(f"  - {reason}" for reason in reasons)
    else:
        lines.append("probe complete: yes")

    lines.extend(_missing_note(env.missing_fields, "snapshot"))
    return "\n".join(lines)


def render_artifact(path: Path, kind: str) -> str:
    """Read and render one artifact, by kind.

    Raises :class:`~aorta.artifacts.ArtifactReadError` when there is no
    readable artifact at *path*; a partial one renders with its gaps named.
    """
    if kind == "matrix":
        return render_matrix(read_matrix(path))
    if kind == "env":
        return render_env(read_env(path))
    raise ValueError(f"unknown artifact kind: {kind!r}")


def describe_run_dir(directory: Path, root: Path) -> str:
    """One line per run directory, for the listing tool."""
    names = sorted({p.name for p, _ in iter_artifacts(directory, max_depth=1)})
    try:
        label = str(directory.relative_to(root))
    except ValueError:
        label = str(directory)
    return f"{label or '.'}: {', '.join(names)}"


__all__ = [
    "ARTIFACT_KINDS",
    "ArtifactReadError",
    "describe_run_dir",
    "find_run_dirs",
    "iter_artifacts",
    "render_artifact",
    "render_env",
    "render_matrix",
    "render_matrix_cell",
]
