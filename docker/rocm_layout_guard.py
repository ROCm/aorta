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

LAYOUT_CLASSIC = "classic"
LAYOUT_WHEEL = "wheel"


def _is_rocm_root(path: Path) -> bool:
    try:
        return path.is_dir() and any((path / marker).exists() for marker in ROOT_MARKERS)
    except OSError:
        return False


def _wheel_roots(site_dir: Path, source: str) -> tuple[Path, Path, Path, str, str] | None:
    core = site_dir / WHEEL_CORE_PACKAGE
    if not core.is_dir():
        return None
    libraries = site_dir / WHEEL_LIBRARIES_PACKAGE
    include = site_dir / WHEEL_DEVEL_PACKAGE
    return (
        core,
        libraries if libraries.is_dir() else core,
        include if include.is_dir() else core,
        LAYOUT_WHEEL,
        source,
    )


def _roots_from_candidate(candidate: Path, source: str):
    if candidate.name.startswith(WHEEL_PACKAGE_PREFIX):
        roots = _wheel_roots(candidate.parent, source)
        if roots is not None:
            return roots
        if candidate.is_dir():
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
        roots = _roots_from_candidate(Path(value), name)
        if roots is not None:
            return roots

    if _is_rocm_root(CLASSIC_ROCM_ROOT):
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
    except (ImportError, ValueError):
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
        except OSError:
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
    if lib_dir.is_dir():
        print(f"lib dir     : {lib_dir}")
    else:
        failures.append(
            f"{lib_dir} is not a directory; "
            "scripts/audit_env_knobs.py --rocm-lib would find no libraries"
        )

    if not failures:
        return 0

    print("", file=sys.stderr)
    print("ERROR: base image has no readable ROCm install.", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    print(
        f"  resolved layout={layout} source={source} core={core} lib_root={libraries}",
        file=sys.stderr,
    )
    for name in ROOT_ENV_VARS:
        print(f"  {name}={os.environ.get(name, '<unset>')}", file=sys.stderr)
    print(
        "  Both the classic (/opt/rocm) and wheel (TheRock, under site-packages)\n"
        "  layouts are accepted; neither was found. Fix the base image or the\n"
        "  digest pin -- do not delete this guard to make a bump pass (#381).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
