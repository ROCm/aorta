#!/usr/bin/env python3
"""Compute the next release version from the project's git tags.

The version is no longer stored in ``pyproject.toml``; it is derived from the
``vX.Y.Z`` git tags by ``setuptools_scm`` at build time (see
``[tool.setuptools_scm]`` in ``pyproject.toml``). This script computes the *next*
version to release/stamp from the latest such tag, without writing any file --
the release/nightly workflows capture its stdout and feed it to
``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AORTA`` (release) or use it to create the
new tag. Keeping the math here (rather than inline in YAML) keeps it unit-tested.

Examples:
    python scripts/bump_version.py patch                 # latest v0.2.0 -> 0.2.1
    python scripts/bump_version.py minor                 # latest v0.2.0 -> 0.3.0
    python scripts/bump_version.py --set 1.4.2           # an explicit version
    python scripts/bump_version.py patch --suffix rc20260619
                                                         # 0.2.0 -> 0.2.1rc20260619
    python scripts/bump_version.py patch --current 0.2.0 # override git lookup (tests/CI)

Prints the resolved version to stdout so callers (e.g. CI) can capture it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# Match a MAJOR.MINOR.PATCH base, but only when the patch is not followed by
# another dot or digit -- so a malformed 4-segment value like "0.2.0.1" is
# rejected instead of being silently truncated to "0.2.0". A trailing
# pre-release part (e.g. "0.2.0rc20260101") is still allowed for re-stamping.
_SEMVER_PREFIX_RE = re.compile(r"^(\d+\.\d+\.\d+)(?![\d.])")
# A suffix is concatenated straight onto the base version, so it must be a
# single PEP 440-style token: non-empty and limited to alphanumerics plus . + -
# (no quotes, whitespace, or newlines that could corrupt the emitted version).
_SUFFIX_RE = re.compile(r"^[A-Za-z0-9.+-]+$")

# When no version tag exists yet, releases start from this base (so a first
# ``patch`` release resolves to 0.0.1, ``minor`` to 0.1.0, ``major`` to 1.0.0).
_INITIAL_VERSION = "0.0.0"


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


def apply_suffix(current: str, suffix: str) -> str:
    """Return the ``MAJOR.MINOR.PATCH`` base of ``current`` with ``suffix`` appended.

    ``current`` is matched against an anchored ``MAJOR.MINOR.PATCH`` prefix
    (``_SEMVER_PREFIX_RE``); only a suffix that begins on a *non-numeric,
    non-dot* boundary is stripped, so re-stamping
    ``0.2.0rc20260101`` -> ``0.2.0rc20260619`` is idempotent on the base.
    Inputs whose extra part is dot-prefixed (PEP 440 ``.dev0`` / ``.post1``)
    or extends the release number (``0.2.0.1``) are intentionally rejected
    with ``ValueError`` rather than silently truncated. Used to mint nightly
    release-candidate versions such as ``0.2.1rc20260619``.
    """
    if _SUFFIX_RE.match(suffix) is None:
        raise ValueError(
            f"invalid suffix {suffix!r}; expected a non-empty token of "
            "alphanumerics and . + - (no quotes/whitespace/newlines)"
        )
    match = _SEMVER_PREFIX_RE.match(current)
    if match is None:
        raise ValueError(
            f"cannot suffix non-semver version {current!r}; expected a MAJOR.MINOR.PATCH prefix"
        )
    return f"{match.group(1)}{suffix}"


def current_version_from_git() -> str:
    """Return the latest ``vX.Y.Z`` release tag as ``X.Y.Z`` (or ``0.0.0``).

    Only annotated/lightweight tags of the form ``v<MAJOR>.<MINOR>.<PATCH>`` are
    considered, so the rolling ``dev-wheels`` nightly tag (and any other
    non-release tag) is ignored. When the repository has no release tag yet, the
    initial ``0.0.0`` base is returned so a first ``patch`` release resolves to
    ``0.0.1``.
    """
    try:
        out = subprocess.run(
            ["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - env-specific
        raise RuntimeError(f"could not list git tags: {exc}") from exc

    versions: list[tuple[int, int, int]] = []
    for line in out.splitlines():
        tag = line.strip()
        match = _SEMVER_RE.match(tag[1:] if tag.startswith("v") else tag)
        if match is not None:
            versions.append(tuple(int(p) for p in match.groups()))  # type: ignore[arg-type]
    if not versions:
        return _INITIAL_VERSION
    major, minor, patch = max(versions)
    return f"{major}.{minor}.{patch}"


def resolve_new_version(
    current: str,
    level: str | None,
    explicit: str | None,
    suffix: str | None = None,
) -> str:
    """Resolve the target version from ``current`` plus a bump/explicit/suffix.

    At least one of ``level``, ``explicit``, or ``suffix`` is required: this tool
    computes the *next* version, so returning ``current`` unchanged on empty input
    would be a misleading "success" (e.g. a blank bump input in CI). The base is
    chosen by precedence ``explicit`` (``--set``) > ``level`` bump > ``current``
    as-is (suffix-only). A ``suffix`` (e.g. ``rc20260619``) is then appended to
    that base, so ``level='patch'`` + ``suffix='rc...'`` yields the *next*
    release's rc (``0.2.0`` -> ``0.2.1rc...``) rather than an rc of the
    already-released base.
    """
    if explicit is not None:
        if _SEMVER_RE.match(explicit) is None:
            raise ValueError(f"explicit version {explicit!r} is not MAJOR.MINOR.PATCH")
        base = explicit
    elif level is not None:
        base = bump_version(current, level)
    elif suffix is not None:
        # suffix-only: stamp onto the current base (e.g. re-stamping an rc date).
        if _SEMVER_RE.match(current) is None:
            raise ValueError(
                f"current version {current!r} is not MAJOR.MINOR.PATCH; "
                "pass a bump level or --set VERSION"
            )
        base = current
    else:
        raise ValueError(
            "one of a bump level (major/minor/patch), --set VERSION, or --suffix SUFFIX is required"
        )
    if suffix is not None:
        return apply_suffix(base, suffix)
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level",
        nargs="?",
        choices=("major", "minor", "patch"),
        help="semantic-version component to bump above the latest release tag",
    )
    parser.add_argument(
        "--set",
        dest="explicit",
        help="set an explicit MAJOR.MINOR.PATCH version (overrides the bump level)",
    )
    parser.add_argument(
        "--suffix",
        help="append SUFFIX to the resolved base version, e.g. 'rc20260619' "
        "(used for nightly release candidates)",
    )
    parser.add_argument(
        "--current",
        help="override the latest-release base instead of reading it from git "
        "tags (mainly for tests/CI)",
    )
    args = parser.parse_args(argv)

    current = args.current if args.current is not None else current_version_from_git()
    new_version = resolve_new_version(current, args.level, args.explicit, args.suffix)
    print(new_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
