#!/usr/bin/env python3
"""Bump the project version in ``pyproject.toml``.

Used by the release workflow (and runnable locally) to compute the next version
without hard-coding it anywhere. Supports semantic-version bumps
(``major``/``minor``/``patch``) or setting an explicit version, and rewrites
only the single ``version = "..."`` line inside the ``[project]`` table so the
rest of the file is left byte-for-byte untouched.

Examples:
    python scripts/bump_version.py patch        # 0.2.0 -> 0.2.1
    python scripts/bump_version.py minor        # 0.2.0 -> 0.3.0
    python scripts/bump_version.py --set 1.4.2  # set an explicit version

Prints the new version to stdout so callers (e.g. CI) can capture it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_VERSION_LINE_RE = re.compile(r'^(\s*version\s*=\s*")([^"]*)(".*)$')


def _table_header(line: str) -> str | None:
    """Return the bracketed TOML table name on ``line``, or ``None``.

    Tolerates a trailing inline comment so a valid header like
    ``[project]  # note`` is still recognized (TOML allows comments after a
    table header). Only the comment is dropped; nothing else is rewritten.
    """
    stripped = line.strip()
    if not stripped.startswith("["):
        return None
    without_comment = stripped.split("#", 1)[0].strip()
    if without_comment.endswith("]"):
        return without_comment
    return None


def bump_version(current: str, level: str) -> str:
    """Return ``current`` bumped by ``level`` (``major``/``minor``/``patch``)."""
    match = _SEMVER_RE.match(current)
    if match is None:
        raise ValueError(
            f"cannot bump non-semver version {current!r}; expected MAJOR.MINOR.PATCH"
        )
    major, minor, patch = (int(part) for part in match.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump level {level!r}; expected major/minor/patch")


def read_version(text: str) -> str:
    """Return the version declared in the ``[project]`` table of ``text``."""
    in_project = False
    for line in text.splitlines():
        header = _table_header(line)
        if header is not None:
            in_project = header == "[project]"
            continue
        if in_project:
            match = _VERSION_LINE_RE.match(line)
            if match is not None:
                return match.group(2)
    raise ValueError("no project.version found in pyproject text")


def set_version(text: str, new_version: str) -> str:
    """Return ``text`` with the ``[project]`` version line set to ``new_version``."""
    out: list[str] = []
    in_project = False
    replaced = False
    for line in text.splitlines(keepends=True):
        header = _table_header(line)
        if header is not None:
            in_project = header == "[project]"
            out.append(line)
            continue
        if in_project and not replaced:
            # Split off the exact trailing newline sequence ("\r\n", "\r",
            # "\n" or "") and match against the bare content, so a CRLF/CR
            # checkout keeps its line endings byte-for-byte (the "untouched"
            # guarantee) instead of relying on the regex's "." swallowing the
            # stray "\r".
            content = line.rstrip("\r\n")
            terminator = line[len(content):]
            match = _VERSION_LINE_RE.match(content)
            if match is not None:
                out.append(f"{match.group(1)}{new_version}{match.group(3)}{terminator}")
                replaced = True
                continue
        out.append(line)
    if not replaced:
        raise ValueError("no project.version line to replace in pyproject text")
    return "".join(out)


def resolve_new_version(current: str, level: str | None, explicit: str | None) -> str:
    """Resolve the target version from a bump ``level`` or an ``explicit`` value."""
    if explicit is not None:
        if _SEMVER_RE.match(explicit) is None:
            raise ValueError(f"explicit version {explicit!r} is not MAJOR.MINOR.PATCH")
        return explicit
    if level is not None:
        return bump_version(current, level)
    raise ValueError("either a bump level (major/minor/patch) or --set VERSION is required")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level",
        nargs="?",
        choices=("major", "minor", "patch"),
        help="semantic-version component to bump",
    )
    parser.add_argument(
        "--set",
        dest="explicit",
        help="set an explicit MAJOR.MINOR.PATCH version (overrides the bump level)",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="path to pyproject.toml (default: ./pyproject.toml)",
    )
    args = parser.parse_args(argv)

    # newline="" disables universal-newline translation on both read and write
    # so a CRLF/CR pyproject.toml round-trips byte-for-byte (Path.read_text /
    # write_text would normalize "\r\n" -> "\n" on read and back to os.linesep
    # on write, defeating set_version's line-ending preservation). encoding is
    # pinned so the result doesn't depend on the platform default.
    with open(args.pyproject, encoding="utf-8", newline="") as fh:
        text = fh.read()
    current = read_version(text)
    new_version = resolve_new_version(current, args.level, args.explicit)
    # Render the full updated text BEFORE opening for write: open(..., "w")
    # truncates immediately, so if set_version raised here (e.g. malformed
    # [project]) an in-place write would leave pyproject.toml emptied.
    new_text = set_version(text, new_version)
    with open(args.pyproject, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_text)
    print(new_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
