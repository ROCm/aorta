#!/usr/bin/env python3
"""Verify the built wheel in ``dist/`` carries an expected version.

The release/nightly workflows build a wheel whose version is pinned via
``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AORTA`` and must fail fast if the
artifact does not actually carry that version (e.g. a setuptools_scm
misconfiguration), so a mis-versioned wheel is never released or uploaded.

The version is read from the wheel's embedded ``*.dist-info/METADATA`` -- the
same metadata ``pip`` installs -- rather than by pattern-matching a single
filename, so a future build/platform tag (or an unexpected extra wheel) can't
make the check pass or fail on the wrong signal. Comparison is on the PEP
440-normalized version (via ``packaging`` when available) so cosmetic spelling
differences (e.g. a ``-``/``_`` separator before a pre-release) don't cause a
false mismatch; without ``packaging`` it falls back to exact string equality.

Usage:
    python scripts/check_wheel_version.py <expected-version> [dist-dir]

Exits non-zero with a ``::error::`` line (GitHub Actions annotation) on any
mismatch or ambiguity; prints a confirmation and exits 0 on success.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from email.parser import BytesParser
from zipfile import BadZipFile, ZipFile

_METADATA_RE = re.compile(r"[^/]+\.dist-info/METADATA$")


def _versions_match(built: str, expected: str) -> bool:
    """Return whether two version strings are equal under PEP 440.

    Uses ``packaging`` for a proper PEP 440 comparison (so ``0.2.1-rc1`` and
    ``0.2.1rc1`` are equal) when it is importable -- it is a dependency of
    ``build``, so it is present in the workflows after ``pip install build``.
    Without it (or on an unparseable value), fall back to exact, stripped string
    equality, which still matches for the already-normalized versions these
    workflows produce.
    """
    a, b = built.strip(), expected.strip()
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - packaging is present in CI
        return a == b
    try:
        return Version(a) == Version(b)
    except InvalidVersion:
        return a == b


def wheel_metadata_version(wheel_path: str) -> str:
    """Return the ``Version`` field from a wheel's ``dist-info/METADATA``."""
    with ZipFile(wheel_path) as zf:
        names = [n for n in zf.namelist() if _METADATA_RE.match(n)]
        if len(names) != 1:
            raise ValueError(
                f"expected exactly one dist-info/METADATA in {wheel_path}, found {names or 'none'}"
            )
        version = BytesParser().parsebytes(zf.read(names[0])).get("Version")
    if not version:
        raise ValueError(f"no Version field in {wheel_path} METADATA")
    return version


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not 1 <= len(args) <= 2:
        print(f"::error::usage: {os.path.basename(__file__)} <expected-version> [dist-dir]")
        return 2
    expected = args[0]
    dist_dir = args[1] if len(args) == 2 else "dist"

    wheels = sorted(glob.glob(os.path.join(dist_dir, "*.whl")))
    if len(wheels) != 1:
        listing = ", ".join(wheels) if wheels else "none"
        print(f"::error::expected exactly one built wheel in {dist_dir}/, found: {listing}")
        return 1

    try:
        built = wheel_metadata_version(wheels[0])
    except (OSError, BadZipFile, ValueError) as exc:
        print(f"::error::could not read version from {wheels[0]}: {exc}")
        return 1

    if not _versions_match(built, expected):
        print(
            f"::error::built wheel version {built!r} does not match resolved "
            f"version {expected!r} (wheel: {wheels[0]})"
        )
        return 1

    print(f"Verified built wheel version {built} matches resolved {expected}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
