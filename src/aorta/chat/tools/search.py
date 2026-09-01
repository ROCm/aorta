"""Code search tools: semantic (ChromaDB), regex (grep), and repo map search."""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.tools import tool

from aorta.chat.config import settings
from aorta.chat.rag.walk import prune_dirnames


def _search_docs(query: str, k: int) -> list:
    """Lazy-load the ChromaDB vectorstore and retrieve exactly k results."""
    from aorta.chat.rag.retriever import search_docs

    return search_docs(query, k)


@tool
def search_code(query: str, k: int | None = None) -> str:
    """Search the AORTA codebase for code semantically related to the query.

    Args:
        query: Natural language description of what to find.
        k: Number of results to return (default from settings).

    Returns:
        Matching code chunks with file paths and line numbers.
    """
    if k is None:
        k = settings.search_tool_k
    elif k <= 0:
        raise ValueError("k must be a positive integer.")
    docs = _search_docs(query, k)

    if not docs:
        return "No matching code found in the AORTA codebase."

    results: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        start = doc.metadata.get("start_line", "?")
        end = doc.metadata.get("end_line", "?")
        header = f"--- Result {i}: {source} (lines {start}-{end}) ---"
        results.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(results)


_SEARCH_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java",
    ".cpp", ".c", ".h", ".rs", ".rb", ".sh", ".bash",
    ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
    ".md", ".rst", ".txt",
}


@tool
def grep_code(pattern: str, path: str = ".", max_results: int = 20) -> str:
    """Search for a regex pattern across files in the AORTA codebase.

    Use this for exact or pattern-based searches (e.g. function names,
    class definitions, imports). Complements search_code which does
    semantic/meaning-based search.

    Args:
        pattern: Regex pattern to search for (e.g. "def.*config", "class.*Config").
        path: Subdirectory to search within (default: entire codebase root).
        max_results: Maximum number of matching lines to return.

    Returns:
        Matching lines with file paths and line numbers.
    """
    root = settings.aorta_root
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return f"Error: path escapes AORTA root: {path}"
    if not target.exists():
        return f"Error: path '{path}' does not exist."

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Error: invalid regex pattern: {exc}"

    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(target):
        current = Path(dirpath)
        prune_dirnames(current, dirnames)

        for filename in sorted(filenames):
            fpath = current / filename
            if fpath.suffix not in _SEARCH_EXTENSIONS:
                continue

            rel = str(fpath.relative_to(root))
            try:
                with fpath.open(mode="r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if compiled.search(line):
                            matches.append(f"{rel}:{line_num}: {line.strip()}")
                        if len(matches) >= max_results:
                            break
            except OSError:
                continue
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches found for pattern '{pattern}'."

    header = f"Found {len(matches)} match(es):\n"
    return header + "\n".join(matches)


@tool
def search_repo_map(query: str) -> str:
    """Search the repository function/class index for matching entries.

    The repo map contains file paths and function/class signatures for the
    entire AORTA codebase. Use this to quickly find where specific functions
    or classes are defined without reading files.

    Args:
        query: Keyword to search for (e.g. "config", "parse", "training").

    Returns:
        Matching lines from the repository map.
    """
    from aorta.chat.rag.repo_map import load_repo_map

    # max_chars=0 asks for the whole file. This tool is what makes the prompt
    # truncation in plan_node safe, so it must never see a truncated map.
    repo_map = load_repo_map(max_chars=0)
    lower_query = query.lower()
    matches = [
        line for line in repo_map.splitlines()
        if lower_query in line.lower()
    ]

    if not matches:
        return f"No entries matching '{query}' found in the repository map."

    matches = matches[:30]
    return f"Found {len(matches)} matching entries:\n" + "\n".join(matches)
