#!/usr/bin/env python3
"""Fail an image build when ROCm is not readable, in EITHER install layout.

Run from ``Dockerfile.ci-gpu`` and ``Dockerfile.rocm-latest`` immediately after
``FROM``. Before issue #381 this guard demanded the classic ``/opt/rocm``
layout, because that was the only layout the repo could read. Discovery is now
layout-agnostic, so the guard accepts a classic *or* a wheel (TheRock) install
and fails only when neither is found.

What must not regress is **fail-closed**, not the classic layout. Everything
we use to read a ROCm install fails SOFT -- on an unreadable install the env
probe reports ``null`` and the env-knob audit finds no libraries, so a bad
base-image digest would quietly gut the evidence the NaN escalations depend on
while CI still looked green. Assert here instead, where it is loud, cheap, and
precedes any test run. Do not delete this to make a bump pass.

Two assertions, each mirroring a real consumer rather than a proxy for one:

* a readable ``.info/version`` or ``.info/version-dev`` under the core root --
  ``scripts/ci/nightly_eval.py`` reads exactly these for the dashboard's
  ``rocm`` column;
* a ``lib/`` directory under the libraries root -- ``scripts/audit_env_knobs.py``
  exits 2 when its ``--rocm-lib`` is not a directory.

**This duplicates the resolution rules in
``src/aorta/instrumentation/rocm_paths.py`` and that is deliberate.** The docker
build context is this ``docker/`` directory (see ``docker-compose.build.yaml``),
so ``src/`` is not reachable at build time, and the repo is a *runtime* mount at
``/workspace/aorta``. ``tests/docker/test_rocm_layout_guard.py`` pins the two
implementations to each other against shared synthetic trees, so the duplication
is caught when it drifts rather than discovered on a red runner.

Stdlib only, and it must stay that way: it runs before any pip install.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

CLASSIC_ROCM_ROOT = Path("/opt/rocm")
ROOT_ENV_VARS = ("ROCM_PATH", "ROCM_HOME")
WHEEL_PACKAGE_PREFIX = "_rocm_sdk_"
WHEEL_CORE_PACKAGE = "_rocm_sdk_core"
WHEEL_LIBRARIES_PACKAGE = "_rocm_sdk_libraries"
WHEEL_DEVEL_PACKAGE = "_rocm_sdk_devel"
ROOT_MARKERS = (".info", "bin", "lib")
USABLE_VERSION_MARKERS = (".info/version", ".info/version-dev")
VERSION_MARKER_PROBE_BYTES = 64

LAYOUT_CLASSIC = "classic"
LAYOUT_WHEEL = "wheel"


def _absolute(path: Path) -> Path | None:
    """Anchor a relative override without resolving symlinks.

    Mirrors rocm_paths._absolute: a relative ROCM_PATH/ROCM_HOME would make the
    reported roots depend on the working directory. Lexical on purpose, so a
    /opt/rocm symlink is still reported as /opt/rocm.

    None when a relative path cannot be anchored -- abspath calls getcwd() for
    one, which raises if the working directory was deleted, and this must not
    escape as a traceback. An absolute path never calls getcwd.
    """
    try:
        return Path(os.path.abspath(path))
    except OSError:
        return None


def _safe_is_dir(path: Path) -> bool:
    """``is_dir()`` that never raises. Mirrors rocm_paths._safe_is_dir.

    Every probe here routes through this so an unreadable mount makes the guard
    print its own diagnostics instead of dying with a traceback -- a traceback
    fails the build with no statement of what it looked for.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def _has_readable_version(path: Path) -> bool:
    """Readable, non-empty .info/version{,-dev}. Mirrors rocm_paths.

    A real read, so an empty file, a whitespace-only file, a *directory* named
    .info/version and an unreadable file all collapse to the same "no".
    """
    for marker in USABLE_VERSION_MARKERS:
        candidate = path / marker
        try:
            with candidate.open("rb") as handle:
                if handle.read(VERSION_MARKER_PROBE_BYTES).strip():
                    return True
        except OSError:
            continue
    return False


def _is_rocm_root(path: Path) -> bool:
    """Loose test, for an explicit ROCM_PATH / ROCM_HOME override."""
    try:
        return _safe_is_dir(path) and any(
            (path / marker).exists() for marker in ROOT_MARKERS
        )
    except OSError:
        return False


def _is_usable_rocm_root(path: Path) -> bool:
    """Stricter test, for autodetected /opt/rocm only.

    Autodetection outranks an importable wheel, so it needs more than "the
    directory exists": a bin-only compat shim (which a wheel-based image may
    ship to keep hipcc on PATH) would otherwise be treated as the install and
    make this guard fail a perfectly good image. Mirrors
    rocm_paths._is_usable_rocm_root.
    """
    if not _safe_is_dir(path):
        return False
    if _has_readable_version(path):
        return True
    return _safe_is_dir(path / "lib")


def _wheel_roots(site_dir: Path, source: str) -> tuple[Path, Path, Path, str, str] | None:
    core = site_dir / WHEEL_CORE_PACKAGE
    if not _safe_is_dir(core):
        return None
    libraries = site_dir / WHEEL_LIBRARIES_PACKAGE
    include = site_dir / WHEEL_DEVEL_PACKAGE
    return (
        core,
        libraries if _safe_is_dir(libraries) else core,
        include if _safe_is_dir(include) else core,
        LAYOUT_WHEEL,
        source,
    )


def _roots_from_candidate(candidate: Path, source: str):
    if candidate.name.startswith(WHEEL_PACKAGE_PREFIX):
        roots = _wheel_roots(candidate.parent, source)
        if roots is not None:
            return roots
        if _safe_is_dir(candidate):
            return (candidate, candidate, candidate, LAYOUT_WHEEL, source)
        return None
    if _is_rocm_root(candidate):
        return (candidate, candidate, candidate, LAYOUT_CLASSIC, source)
    return None


def resolve():
    """Mirror ``rocm_paths.resolve_rocm_roots``: env vars, /opt/rocm, then import."""
    for name in ROOT_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        candidate = _absolute(Path(value))
        if candidate is None:
            continue
        roots = _roots_from_candidate(candidate, name)
        if roots is not None:
            return roots

    if _is_usable_rocm_root(CLASSIC_ROCM_ROOT):
        return (
            CLASSIC_ROCM_ROOT,
            CLASSIC_ROCM_ROOT,
            CLASSIC_ROCM_ROOT,
            LAYOUT_CLASSIC,
            "opt_rocm",
        )

    # Interpreter-scoped on purpose. A wheel install is only usable by the
    # interpreter it was installed for, so the guard must run under the same
    # `python` that will later run aorta -- on rocm/pytorch that is
    # /opt/venv/bin/python via PATH, not /usr/bin/python3.
    try:
        spec = importlib.util.find_spec(WHEEL_CORE_PACKAGE)
    except Exception:  # noqa: BLE001 -- mirrors rocm_paths: resolution never raises
        spec = None
    if spec is not None:
        core = None
        if spec.origin:
            core = Path(spec.origin).parent
        else:
            for location in spec.submodule_search_locations or ():
                core = Path(location)
                break
        if core is not None:
            roots = _wheel_roots(core.parent, f"import:{WHEEL_CORE_PACKAGE}")
            if roots is not None:
                return roots

    return (CLASSIC_ROCM_ROOT, CLASSIC_ROCM_ROOT, CLASSIC_ROCM_ROOT, LAYOUT_CLASSIC, "none")


def main() -> int:
    core, libraries, include, layout, source = resolve()

    print(f"ROCm layout : {layout}")
    print(f"resolved via: {source}")
    print(f"core root   : {core}")
    print(f"lib root    : {libraries}")
    print(f"include root: {include}")
    print(f"interpreter : {sys.executable}")

    failures = []

    version_files = [core / ".info" / "version", core / ".info" / "version-dev"]
    version = None
    for path in version_files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        # UnicodeDecodeError is NOT an OSError: a binary blob sitting at the
        # marker path would otherwise crash the guard with a traceback instead of
        # reporting which paths it tried. environment.py's _read_text_file treats
        # non-UTF8 the same way, so the two agree on what "unreadable" means.
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            version = text
            print(f"ROCm version: {text}  (from {path})")
            break
    if version is None:
        failures.append(
            "no readable "
            + " or ".join(str(path) for path in version_files)
            + "; scripts/ci/nightly_eval.py would report rocm: null"
        )

    lib_dir = libraries / "lib"
    if _safe_is_dir(lib_dir):
        print(f"lib dir     : {lib_dir}")
    else:
        failures.append(
            f"{lib_dir} is not a directory; "
            "scripts/audit_env_knobs.py --rocm-lib would find no libraries"
        )

    if not failures:
        return 0

    # Two distinct failures, and the message has to say which one happened.
    # "neither layout was found" is only true when discovery came up empty; when
    # it resolved a tree that is merely incomplete, saying that contradicts the
    # resolved layout/source printed directly above it and sends the reader
    # looking for the wrong problem.
    nothing_found = source == "none"
    print("", file=sys.stderr)
    if nothing_found:
        print("ERROR: no ROCm install found in either layout.", file=sys.stderr)
    else:
        print(
            f"ERROR: found a {layout} ROCm install (via {source}) but it is not "
            "readable.",
            file=sys.stderr,
        )
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    print(
        f"  resolved layout={layout} source={source} core={core} lib_root={libraries}",
        file=sys.stderr,
    )
    for name in ROOT_ENV_VARS:
        print(f"  {name}={os.environ.get(name, '<unset>')}", file=sys.stderr)
    if nothing_found:
        print(
            "  Both the classic (/opt/rocm) and wheel (TheRock, under\n"
            "  site-packages) layouts are accepted; neither was found.",
            file=sys.stderr,
        )
    else:
        print(
            "  The layout was recognised, so this is an incomplete or damaged\n"
            "  install rather than an unsupported one -- check the paths above.",
            file=sys.stderr,
        )
    print(
        "  Fix the base image or the digest pin -- do not delete this guard to\n"
        "  make a bump pass (#381).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
