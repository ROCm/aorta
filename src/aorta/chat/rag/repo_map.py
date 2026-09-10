"""Generate an Aider-style repository map (file tree + function/class signatures)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from aorta.chat.config import settings
from aorta.chat.rag.walk import is_symlink, should_skip_dir

logger = logging.getLogger(__name__)


def _extract_python_signatures(file_path: Path) -> list[str]:
    """Parse a Python file with the ast module and extract top-level signatures."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    sigs: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            method_str = ", ".join(methods) if methods else ""
            sigs.append(f"class {node.name}({method_str})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args if a.arg != "self"]
            sigs.append(f"def {node.name}({', '.join(args)})")
    return sigs


def _build_tree(root: Path, prefix: str = "") -> list[str]:
    """Recursively build a tree listing with Python signatures."""
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    lines: list[str] = []

    # Unlike the ``os.walk`` consumers, this descent is hand-rolled, so nothing
    # declines to follow a link for it: a symlinked directory would be recursed
    # into outside the configured root, and a cycle would recurse until the
    # interpreter's own limit stopped it.
    entries = [entry for entry in entries if not is_symlink(entry)]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = prefix + ("    " if is_last else "│   ")

        if entry.is_dir():
            if should_skip_dir(entry):
                continue
            lines.append(f"{prefix}{connector}{entry.name}/")
            lines.extend(_build_tree(entry, child_prefix))
        else:
            sig_comment = ""
            if entry.suffix == ".py":
                sigs = _extract_python_signatures(entry)
                if sigs:
                    sig_comment = f"  # {'; '.join(sigs)}"
            lines.append(f"{prefix}{connector}{entry.name}{sig_comment}")

    return lines


def generate_repo_map(
    codebase_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a repository map and optionally write it to disk.

    Returns:
        The repo map as a string.
    """
    root = Path(codebase_path or settings.aorta_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Codebase path does not exist: {root}")

    header = f"# Repository Map: {root.name}\n\n```\n{root.name}/\n"
    tree_lines = _build_tree(root)
    body = "\n".join(tree_lines)
    footer = "\n```\n"
    repo_map = header + body + footer

    out = Path(output_path or settings.repo_map_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(repo_map, encoding="utf-8")
    logger.info("Repo map written to %s (%d lines)", out, len(tree_lines))

    return repo_map


_TRUNCATION_NOTICE = (
    "\n\n[Repository map truncated at {shown:,} of {total:,} characters. "
    "Use the search_repo_map tool to look up any file, function or class that "
    "is not listed above.]"
)


def load_repo_map(max_chars: int | None = None) -> str:
    """Load the repo map from disk, capped to a promptable size.

    The cap is not cosmetic. ``plan_node`` injects this straight into a system
    message, and the map for a real codebase runs to megabytes -- AORTA's is
    around 3 MB, which is several hundred thousand tokens. Uncapped, that
    overflows any context window, and on a metered endpoint it does so at a
    price. Nothing is lost by truncating: ``search_repo_map`` queries the full
    file on disk, so the model can still reach every entry.

    Pass ``max_chars=0`` for the whole file, e.g. for the search tool.
    """
    path = Path(settings.repo_map_path)
    if not path.exists():
        return "(Repository map not yet generated. Run indexing first.)"

    text = path.read_text(encoding="utf-8")
    limit = settings.repo_map_prompt_max_chars if max_chars is None else max_chars
    if limit <= 0 or len(text) <= limit:
        return text

    logger.warning(
        "Repo map is %d chars; truncating to %d for the prompt. "
        "search_repo_map still sees all of it.",
        len(text),
        limit,
    )
    return text[:limit] + _TRUNCATION_NOTICE.format(shown=limit, total=len(text))
