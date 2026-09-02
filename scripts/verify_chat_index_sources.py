#!/usr/bin/env python3
"""Assert every chunk in a chat index came from a git-tracked file of this tree.

A belt-and-braces check on the public-tree guard in
``aorta.chat.rag.corpus``. The build already restricts its corpus to
``git ls-files``, so this can only fail if that enforcement broke -- which is
exactly the failure worth a red CI run rather than a silent publish.

The stakes are asymmetric and that is why the check is duplicated here. A
published index stores every chunk's source text verbatim, so an index built
over an internal reproducer or a customer bundle republishes that source, and a
release asset cannot be recalled. The cost of one more subprocess and one sqlite
scan is nothing against that.

Usage:
    python scripts/verify_chat_index_sources.py <index.sqlite> [--repo <path>]

Exits 0 when clean, 1 with the offending paths listed otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: Reported before bailing out, so a truncated failure list still shows the
#: shape of the problem.
_MAX_REPORTED = 40


def tracked_paths(repo: Path) -> set[str]:
    """Every path git tracks in ``repo``, as repo-relative POSIX strings."""
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {entry for entry in completed.stdout.split("\0") if entry}


def index_sources(index: Path) -> set[str]:
    """The distinct ``source`` metadata values in every chunk table of ``index``.

    Reads the sqlite file directly rather than through ``SqliteVecStore``: this
    script must work even if the chat extra failed to install, and it has no
    business loading the sqlite-vec extension just to read a text column.
    """
    import sqlite3

    sources: set[str] = set()
    with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as conn:
        tables = [
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'chunks_%'"
            )
        ]
        if not tables:
            raise SystemExit(f"error: {index} holds no chunk tables; is it a chat index?")
        for table in tables:
            # Table name interpolated because sqlite cannot bind an identifier;
            # the value came from sqlite_master, not from user input.
            for (metadata,) in conn.execute(f'SELECT metadata FROM "{table}"'):  # noqa: S608
                # Fail closed. This is the last gate before publication, so a
                # chunk whose provenance cannot be read is the one most worth
                # stopping for -- skipping it let an index holding unverifiable
                # source text still print OK.
                try:
                    parsed = json.loads(metadata)
                except ValueError as exc:
                    raise SystemExit(
                        f"error: a chunk in {table} has metadata that is not "
                        f"JSON ({exc}); its provenance cannot be verified."
                    ) from exc
                source = parsed.get("source") if isinstance(parsed, dict) else None
                if not isinstance(source, str) or not source:
                    raise SystemExit(
                        f"error: a chunk in {table} has no usable 'source' in "
                        f"its metadata (got {source!r}); its provenance cannot "
                        "be verified."
                    )
                sources.add(source.replace("\\", "/"))
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="Path to the built .sqlite index.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="Repository the index was built from. Defaults to the working directory.",
    )
    args = parser.parse_args(argv)

    if not args.index.exists():
        print(f"error: no index at {args.index}", file=sys.stderr)
        return 1

    tracked = tracked_paths(args.repo)
    sources = index_sources(args.index)
    if not sources:
        # "All zero paths are tracked" is a pass this guard must not be able to
        # print: an index with nothing readable in it has not been verified.
        print(
            f"::error::{args.index} yielded no indexed source paths, so nothing "
            "was verified. Refusing to report it as publishable.",
            file=sys.stderr,
        )
        return 1
    untracked = sorted(source for source in sources if source not in tracked)

    if untracked:
        print(
            f"::error::{len(untracked)} of {len(sources)} indexed source paths are not "
            "tracked by git. The index embeds source text verbatim, so this must not "
            "be published.",
            file=sys.stderr,
        )
        for source in untracked[:_MAX_REPORTED]:
            print(f"  untracked: {source}", file=sys.stderr)
        if len(untracked) > _MAX_REPORTED:
            print(f"  ... and {len(untracked) - _MAX_REPORTED} more", file=sys.stderr)
        return 1

    print(f"OK: all {len(sources)} indexed source paths are tracked by git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
