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
* the GEMM sonames under the libraries root's ``lib/`` --
  ``scripts/audit_env_knobs.py`` resolves exactly these, and reports every knob
  as uncovered without them.

The second assertion used to check only that ``lib/`` *was a directory*, on the
grounds that the audit exits 2 when its ``--rocm-lib`` is not one. That mirrored
the audit's argument validation rather than what it consumes, and the two
stopped being the same thing once resolution could hand the check a *different
component's* ``lib/``: ``_wheel_roots`` falls back ``libraries -> core`` when
``_rocm_sdk_libraries`` is absent, and ``core/lib`` exists regardless because the
LLVM toolchain lives at ``core/lib/llvm/bin``. So a wheel image carrying neither
``libhipblaslt`` nor ``librocblas`` passed the guard and broke later in the
audit -- the "a bad digest quietly guts the evidence" case above, arriving
through the fallback rather than around it. Under the classic layout the two
roots coincide, so the directory proxy held and the gap was invisible.

Deliberately NOT implemented as "reject any wheel install whose libraries root
fell back to core": that condition is also true for an explicit
``$ROCM_PATH``/``$ROCM_HOME`` override naming a lone component (see
``_roots_from_candidate``), and it is true by construction on every classic
install, so it would reject working images while still not proving the
libraries are readable. The sonames are what the consumer actually needs.

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

import codecs
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
# Mirrors scripts/audit_env_knobs.py DEFAULT_SONAMES -- the libraries that
# script resolves, and without which every knob reports as uncovered. Kept in
# step by test_guard_requires_the_sonames_the_audit_consumes, which reads the
# tuple out of the audit rather than restating it.
AUDIT_SONAMES = ("libhipblaslt.so", "librocblas.so")
# Mirrors rocm_paths._VERSION_MARKER_BYTES. ONE limit for both probing that a
# marker is usable and reading its value: a larger read sees strictly more bytes
# and can reject content a smaller probe accepted, which is how discovery came
# to accept a marker main() then rejected.
VERSION_MARKER_BYTES = 4096

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
    """``is_dir()`` that never raises. Mirrors rocm_paths.safe_is_dir.

    Every probe here routes through this so an unreadable mount makes the guard
    print its own diagnostics instead of dying with a traceback -- a traceback
    fails the build with no statement of what it looked for.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


def read_version_marker(path: Path, limit: int = VERSION_MARKER_BYTES) -> str | None:
    """The stripped contents of a version marker, or None. Mirrors rocm_paths.

    ONE reader AT ONE LIMIT for both users in this file -- the discovery
    predicate below and main()'s verification. They diverged twice: first on the
    predicate, then on the byte limit, and both times discovery accepted a
    marker main() then rejected, so resolve() selected an install and the build
    failed with "no readable version" about the very file discovery had just
    read. Sharing the function is not enough; they must share the limit too.

    Every unusable case collapses to None: absent, a *directory* named
    .info/version, unreadable (all OSError), empty or whitespace-only content,
    and non-UTF-8 bytes. That last one is deliberately not salvaged with
    errors="replace" -- environment.py reports such a marker as null, and
    main()'s failure text promises exactly that.

    Decoded incrementally because the read is bounded and can split a
    multi-byte character at the limit; a plain decode would report that as
    corruption. Reads limit + 1 bytes so a file that ENDED is distinguishable
    from one the limit CUT, and finalises the decoder unless it was actually
    cut -- with an unconditional final=False, `b"7.2.4\\xc3"` decoded to
    "7.2.4" and a corrupt marker read as a valid version.
    """
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError:
        return None
    truncated = len(raw) > limit
    if truncated:
        raw = raw[:limit]
    try:
        text = codecs.getincrementaldecoder("utf-8")().decode(raw, not truncated)
    except UnicodeDecodeError:
        return None
    return text.strip() or None


def _has_readable_version(path: Path) -> bool:
    """Readable, non-empty .info/version{,-dev}. Mirrors rocm_paths."""
    return any(
        read_version_marker(path / marker) is not None for marker in USABLE_VERSION_MARKERS
    )


def _has_soname(lib_dir: Path, soname: str) -> bool:
    """Whether audit_env_knobs.resolve_library would find ``soname``.

    Mirrors that function's ACCEPTANCE RULE, not merely its shape: a bare
    ``<soname>`` that is a FILE (the devel package link), else a
    ``<soname>.<major>`` that is a file whose suffix is all digits (what a
    runtime-only tree ships). The guard only needs presence, so it does not
    reproduce the pick-the-highest-major part -- which major wins cannot turn a
    found library into a missing one.

    Being more permissive than the audit is not a harmless approximation, it is
    the bug this whole check replaced, one level down. An earlier version used
    exists() and an unrestricted ``{soname}.*`` glob, so a directory named
    ``libhipblaslt.so``, a stray ``librocblas.so.debug`` from a separate-debug
    -info tree, or a directory named ``libhipblaslt.so.1`` all passed the guard
    while the audit found nothing -- guard exits 0, audit exits 2, and the
    build-time check that exists to catch exactly that hands it downstream.

    Fail-soft like every other probe here: an unreadable mount answers False
    and the caller reports it as missing, rather than the build dying with a
    traceback that says nothing about what was looked for.
    """
    try:
        if (lib_dir / soname).is_file():
            return True
        for candidate in lib_dir.glob(f"{soname}.*"):
            suffix = candidate.name[len(soname) + 1 :]
            if suffix.isdigit() and candidate.is_file():
                return True
        return False
    except OSError:
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
        # Same reader discovery used, so a marker good enough to select this
        # root cannot be one this loop then reports as missing.
        text = read_version_marker(path)
        if text is not None:
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
    if not _safe_is_dir(lib_dir):
        failures.append(
            f"{lib_dir} is not a directory; "
            "scripts/audit_env_knobs.py --rocm-lib would find no libraries"
        )
    else:
        print(f"lib dir     : {lib_dir}")
        # Not just "the directory exists" -- see the module docstring. On the
        # wheel layout this directory may be core/lib, which exists for the LLVM
        # toolchain alone, so the directory check cannot tell a libraries
        # component from its absence.
        missing = [soname for soname in AUDIT_SONAMES if not _has_soname(lib_dir, soname)]
        if missing:
            failures.append(
                f"{lib_dir} has no " + " and no ".join(missing) + "; "
                "scripts/audit_env_knobs.py would report every knob uncovered"
                + (
                    " (the libraries root fell back to the core component, so "
                    f"{WHEEL_LIBRARIES_PACKAGE} is probably not installed)"
                    if layout == LAYOUT_WHEEL and libraries == core
                    else ""
                )
            )
        else:
            print(f"GEMM libs   : {', '.join(AUDIT_SONAMES)}")

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
