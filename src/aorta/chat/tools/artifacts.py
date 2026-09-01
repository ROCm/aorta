"""Run-artifact tools: read this user's own ``matrix.json`` / ``env.json``.

The adapter that turns :mod:`aorta.artifacts` into something the graph can call.
It holds no parsing of its own -- the typed readers in core own that, and are
importable without the chat extra so an external consumer can use them too. What
this module adds is the three things a tool needs and a library must not
assume: a sandbox root, a string return for every failure mode, and rendering.

The sandbox is ``settings.runs_root`` rather than ``settings.aorta_root``. The
sibling tools read the *codebase*, which is the installed package by default;
run artifacts are output, they live wherever the operator pointed
``--output``, and the two roots are unrelated. Configure it with
``AORTA_CHAT_RUNS_PATH`` when runs are not below the working directory.

Errors come back as ``Error: ...`` strings, matching ``files.py`` and
``search.py``: a raised exception aborts the whole graph run, while a string
lets the model correct its own argument and try again.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from aorta.chat.config import settings
from aorta.chat.runs import (
    ArtifactReadError,
    describe_run_dir,
    find_run_dirs,
    iter_artifacts,
    render_artifact,
)

#: Cap on one tool's return, mirroring ``read_file``'s 8000. A full matrix for a
#: wide sweep renders longer than any answer needs, and the untruncated text is
#: what the run-artifact collection is for.
_MAX_CHARS = 8000

#: Listing cap. A long-lived run root accumulates; the model can narrow by path.
_MAX_RUNS_LISTED = 40


def _resolve_safe(path: str) -> Path:
    """Resolve *path* under the runs root, refusing anything that escapes it.

    ``relative_to`` rather than a string prefix test: ``/runs-old`` starts with
    the characters of ``/runs`` without being inside it.
    """
    root = settings.runs_root
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"path escapes the run root: {path}") from None
    return resolved


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CHARS:
        return text
    return text[:_MAX_CHARS] + f"\n\n... (truncated, {len(text)} total chars)"


@tool
def list_runs(path: str = ".") -> str:
    """List AORTA run directories and the artifacts each one holds.

    Use this first when the user asks about "the run", "my sweep" or "the
    failure" without naming a directory.

    Args:
        path: Relative path within the run root to look under. Defaults to the
            whole run root.

    Returns:
        One line per run directory, with the artifact filenames it contains.
    """
    try:
        target = _resolve_safe(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not target.exists():
        return f"Error: path '{path}' does not exist."
    if not target.is_dir():
        return f"Error: '{path}' is not a directory."

    root = settings.runs_root
    run_dirs = find_run_dirs(target)
    if not run_dirs:
        return (
            f"No AORTA run artifacts found under {target}. Looked for "
            f"matrix.json, env.json and host_env.json. If runs live elsewhere, "
            f"set AORTA_CHAT_RUNS_PATH to their parent directory."
        )

    lines = [describe_run_dir(directory, root) for directory in run_dirs[:_MAX_RUNS_LISTED]]
    if len(run_dirs) > _MAX_RUNS_LISTED:
        lines.append(f"... and {len(run_dirs) - _MAX_RUNS_LISTED} more run directories")
    return f"Run artifacts under {target}:\n" + "\n".join(lines)


def _read_one(path: str, kind: str, label: str) -> str:
    """Shared body of the two readers: resolve, locate, render."""
    try:
        target = _resolve_safe(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not target.exists():
        return f"Error: path '{path}' does not exist."

    if target.is_dir():
        # Given a run directory, find the artifact in it rather than making the
        # model guess the filename -- env.json and host_env.json are the same
        # shape written under different names depending on how aorta was run.
        candidates = [p for p, k in iter_artifacts(target, max_depth=1) if k == kind]
        if not candidates:
            return (
                f"Error: no {label} artifact in '{path}'. "
                "Use list_runs to see which directories have one."
            )
        target = candidates[0]

    try:
        return _truncate(render_artifact(target, kind))
    except ArtifactReadError as exc:
        # Not a partial artifact but an absent or unparseable one, which is
        # worth reporting as-is: a truncated matrix.json usually means the run
        # died mid-write, and that is itself the answer to the question.
        return f"Error: {exc}"


@tool
def read_run_matrix(path: str = ".") -> str:
    """Read a triage run's matrix.json: per-cell pass/fail counts and hints.

    This is the artifact that says whether a failure reproduced and which
    mitigation cleared it.

    Fields the run did not record are shown as 'unknown' and listed under
    'NOT RECORDED'. Treat those as unknown, never as zero -- a cell whose
    failure_rate is unknown has not been shown to pass.

    Args:
        path: A run directory, or a path to matrix.json itself, relative to
            the run root.

    Returns:
        The run summary followed by one block per cell.
    """
    return _read_one(path, "matrix", "matrix.json")


@tool
def read_run_env(path: str = ".") -> str:
    """Read a run's environment snapshot: ROCm version and probe completeness.

    Reports whether the probe itself was partial, which distinguishes "this
    environment fact is absent" from "the probe could not read it".

    Args:
        path: A run directory, or a path to env.json / host_env.json itself,
            relative to the run root.

    Returns:
        The triage-relevant fields, plus any reasons the probe was partial.
    """
    return _read_one(path, "env", "env.json / host_env.json")


@tool
def search_run_artifacts(query: str, k: int | None = None) -> str:
    """Search this user's indexed run artifacts semantically.

    Separate from search_code: that searches the AORTA source, this searches
    the outcomes of runs on this machine. Use it to find which run showed a
    symptom when the user has not said which directory to look in.

    Args:
        query: Natural language description of the failure or environment.
        k: Number of results (default from settings).

    Returns:
        Matching artifact excerpts with their source paths.
    """
    if k is None:
        k = settings.search_tool_k
    elif k <= 0:
        return "Error: k must be a positive integer."

    # Deferred: the RAG stack is heavier than the readers above, and the two
    # direct readers must stay usable when no run collection has been built.
    from aorta.chat.rag.runs import RunCollectionMissingError, search_run_docs

    try:
        docs = search_run_docs(query, k)
    except (RunCollectionMissingError, FileNotFoundError) as exc:
        return f"Error: {exc}"

    if not docs:
        return "No matching run artifacts found in the index."

    results: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        kind = doc.metadata.get("artifact_kind", "?")
        results.append(f"--- Result {i}: {source} ({kind}) ---\n{doc.page_content}")
    return _truncate("\n\n".join(results))


__all__ = [
    "list_runs",
    "read_run_env",
    "read_run_matrix",
    "search_run_artifacts",
]
