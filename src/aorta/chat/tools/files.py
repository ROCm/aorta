"""File-system tools scoped to the AORTA codebase directory."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from langchain_core.tools import tool

from aorta.chat.config import settings
from aorta.chat.tools._sandbox import AORTA_ROOT_LABEL, resolve_within

_GITIGNORE_PATTERNS: list[str] = [
    "__pycache__",
    "*.pyc",
    ".git",
    ".git/*",
    "node_modules",
    ".venv",
    "venv",
    "*.egg-info",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
]


def _resolve_safe(path: str) -> Path:
    """Resolve *path* under ``aorta_path``, refusing anything that escapes it."""
    return resolve_within(settings.aorta_root, path, AORTA_ROOT_LABEL)


def _is_ignored(rel_path: str) -> bool:
    parts = rel_path.split("/")
    for pattern in _GITIGNORE_PATTERNS:
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


@tool
def list_files(path: str = ".") -> str:
    """List files and directories under the given path inside the AORTA codebase.

    Args:
        path: Relative path within the AORTA codebase. Defaults to root.

    Returns:
        A newline-separated listing of files and directories.
    """
    try:
        target = _resolve_safe(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not target.exists():
        return f"Error: path '{path}' does not exist."
    if not target.is_dir():
        return f"Error: '{path}' is not a directory."

    root = settings.aorta_root
    lines: list[str] = []
    for item in sorted(target.iterdir()):
        rel = str(item.relative_to(root))
        if _is_ignored(rel):
            continue
        suffix = "/" if item.is_dir() else ""
        lines.append(f"{rel}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file inside the AORTA codebase.

    Args:
        file_path: Relative path to the file within the AORTA codebase.

    Returns:
        The file contents (truncated to 8000 chars if very large).
    """
    try:
        target = _resolve_safe(file_path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not target.exists():
        return f"Error: file '{file_path}' does not exist."
    if not target.is_file():
        return f"Error: '{file_path}' is not a file."

    max_chars = 8000
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated, {len(text)} total chars)"
    return text
