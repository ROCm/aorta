"""Layout-agnostic discovery of the installed ROCm tree (issue #381).

ROCm ships in two on-disk shapes and we must read both:

* **classic** -- a system install rooted at ``/opt/rocm`` (DEB/RPM packages
  and SDK tarballs). One root holds ``bin/``, ``include/``, ``lib/`` and
  ``share/``. This is what customers run and it is *not* deprecated.
* **wheel** -- TheRock's Python distribution (ROCm 7.14+, and what
  ``rocm/pytorch:latest`` is). There is no ``/opt/rocm`` at all. The tree is
  split across sibling packages under ``site-packages``:
  ``_rocm_sdk_core`` (``bin/``, ``include/``, ``.info/version``,
  ``share/therock/``), ``_rocm_sdk_libraries`` (the math libraries, their
  Tensile kernel databases and ``share/miopen/db``), and optionally
  ``_rocm_sdk_devel`` (the development headers).

Because those two shapes disagree about which root holds what, resolution
yields **three** roots rather than one. On a classic install all three are
the same directory, so every path derived from them is byte-identical to the
hardcoded ``/opt/rocm/...`` constants this replaced -- the classic path does
not change behaviour.

Measured, not assumed. On the wheel image ``aorta-rocm_7.15.0a20260716``:
``/opt/rocm`` is genuinely absent, ``ROCM_PATH`` and ``ROCM_HOME`` are
**both unset** (so neither is a usable layout discriminator -- see the
warning in ``docs/ci-testing-plan.md``), ``.info/version`` exists at
``_rocm_sdk_core/.info/version``, ``libhipblaslt.so.1`` lives in
``_rocm_sdk_libraries/lib``, and the Tensile database is one level deeper
than on classic (``lib/hipblaslt/library/gfx950/`` vs a flat
``lib/hipblaslt/library/``).

This module deliberately imports nothing beyond the standard library and
performs no subprocess calls, so the standalone CI scripts
(``scripts/audit_env_knobs.py``, ``scripts/ci/nightly_eval.py``,
``scripts/sanitizers/prepare_gemm_isa.py``) can use the same resolver as the
env probe without pulling in torch.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# The conventional system install. ``/opt/rocm`` is normally a symlink to the
# active versioned tree (``/opt/rocm-7.2.4``); it is deliberately NOT resolved,
# so a snapshot reports the stable path an operator would type.
CLASSIC_ROCM_ROOT = Path("/opt/rocm")

# Checked in order. Both are honoured because images disagree about which one
# they set: our own Dockerfiles historically set ROCM_HOME, while TheRock's
# images set ROCM_PATH (when they set anything at all).
ROOT_ENV_VARS: tuple[str, ...] = ("ROCM_PATH", "ROCM_HOME")

# TheRock wheel component packages, and the prefix that identifies one when an
# env var points directly at a component directory.
WHEEL_PACKAGE_PREFIX = "_rocm_sdk_"
WHEEL_CORE_PACKAGE = "_rocm_sdk_core"
WHEEL_LIBRARIES_PACKAGE = "_rocm_sdk_libraries"
WHEEL_DEVEL_PACKAGE = "_rocm_sdk_devel"

# Build provenance dropped by TheRock, relative to the core root. Richer than
# anything the classic layout offers: full 40-char submodule pins, the TheRock
# commit, and the CI run that produced the build.
THEROCK_MANIFEST_RELPATH = Path("share") / "therock" / "therock_manifest.json"

# A directory only counts as a ROCm root if it looks like one. Without this an
# empty leftover ``/opt/rocm`` would shadow a perfectly good wheel install.
_ROOT_MARKERS: tuple[str, ...] = (".info", "bin", "lib")

LAYOUT_CLASSIC = "classic"
LAYOUT_WHEEL = "wheel"


@dataclass(frozen=True)
class RocmRoots:
    """Resolved ROCm roots plus how they were found.

    ``source`` is recorded so that a ``null`` in a snapshot is attributable:
    "no version file under /opt/rocm" and "no ROCm install found at all" are
    very different operator problems, and before #381 they looked identical.
    """

    core: Path
    """Holds ``bin/``, ``.info/version`` and ``share/therock/``."""

    libraries: Path
    """Holds ``lib/`` (math libraries, Tensile databases) and ``share/miopen``."""

    include: Path
    """Holds ``include/``. Separate because the wheel layout puts the
    development headers in ``_rocm_sdk_devel``, which runtime-only images
    such as ``aorta-rocm_7.15.0a20260716`` do not install at all."""

    layout: str
    """``"classic"`` or ``"wheel"``."""

    source: str
    """Which mechanism answered: an env var name (``"ROCM_PATH"`` /
    ``"ROCM_HOME"``), ``"opt_rocm"`` for a validated classic install,
    ``"import:<package>"`` for wheel detection, or ``"none"`` when nothing
    was found and the classic root is being assumed. ``"none"`` is the
    value that makes a null version attributable: it means no ROCm install
    was located at all, as opposed to one that lacks a version file."""

    @property
    def bin_dir(self) -> Path:
        return self.core / "bin"

    @property
    def version_file(self) -> Path:
        return self.core / ".info" / "version"

    @property
    def version_dev_file(self) -> Path:
        return self.core / ".info" / "version-dev"

    @property
    def manifest_file(self) -> Path:
        return self.core / THEROCK_MANIFEST_RELPATH

    @property
    def lib_dir(self) -> Path:
        return self.libraries / "lib"

    @property
    def include_dir(self) -> Path:
        return self.include / "include"


def _classic_roots(source: str) -> RocmRoots:
    """The classic single-root layout, where all three roots coincide."""
    return RocmRoots(
        core=CLASSIC_ROCM_ROOT,
        libraries=CLASSIC_ROCM_ROOT,
        include=CLASSIC_ROCM_ROOT,
        layout=LAYOUT_CLASSIC,
        source=source,
    )


def _is_rocm_root(path: Path) -> bool:
    """True when ``path`` is a directory that looks like a ROCm install."""
    try:
        if not path.is_dir():
            return False
        return any((path / marker).exists() for marker in _ROOT_MARKERS)
    except OSError as exc:  # unreadable mount, permission denied
        log.debug("cannot inspect candidate ROCm root %s: %s", path, exc)
        return False


def _wheel_component(site_dir: Path, package: str) -> Path | None:
    candidate = site_dir / package
    return candidate if candidate.is_dir() else None


def _installed_wheel_component(package: str) -> Path | None:
    """Locate an installed TheRock component package via the import system.

    Pure ``importlib`` -- no subprocess, and no ``rocm-sdk`` CLI (which is
    absent on runtime-only images because it ships in ``rocm-sdk-devel``).
    Only finds components installed for the *running* interpreter, which is
    the case that matters: a wheel-based image installs torch and the ROCm
    wheels into the same environment the probe runs in.
    """
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError) as exc:
        # A half-installed or shadowed package raises rather than returning
        # None; treat it as absent rather than letting it escape into a probe.
        log.debug("find_spec(%s) failed: %s", package, exc)
        return None
    if spec is None:
        return None
    if spec.origin:
        return Path(spec.origin).parent
    # Namespace-package form: no __init__.py, so origin is None.
    for location in spec.submodule_search_locations or ():
        return Path(location)
    return None


def _wheel_roots(site_dir: Path, source: str) -> RocmRoots | None:
    """Assemble the three roots from a directory holding the components."""
    core = _wheel_component(site_dir, WHEEL_CORE_PACKAGE)
    if core is None:
        return None
    libraries = _wheel_component(site_dir, WHEEL_LIBRARIES_PACKAGE) or core
    # Headers ship in the devel component; fall back to core so include paths
    # stay absolute and simply do not exist on a runtime-only install.
    include = _wheel_component(site_dir, WHEEL_DEVEL_PACKAGE) or core
    return RocmRoots(
        core=core,
        libraries=libraries,
        include=include,
        layout=LAYOUT_WHEEL,
        source=source,
    )


def _roots_from_candidate(candidate: Path, source: str) -> RocmRoots | None:
    """Interpret one candidate path as either a wheel component or a classic root."""
    if candidate.name.startswith(WHEEL_PACKAGE_PREFIX):
        # An env var pointing at a component directory (rocm/pytorch:latest
        # sets ROCM_PATH to its ``_rocm_sdk_devel``). The siblings are the
        # rest of the install.
        roots = _wheel_roots(candidate.parent, source)
        if roots is not None:
            return roots
        # A lone component with no _rocm_sdk_core beside it: use it for
        # everything rather than discarding an explicit operator override.
        if candidate.is_dir():
            return RocmRoots(
                core=candidate,
                libraries=candidate,
                include=candidate,
                layout=LAYOUT_WHEEL,
                source=source,
            )
        return None
    if _is_rocm_root(candidate):
        return RocmRoots(
            core=candidate,
            libraries=candidate,
            include=candidate,
            layout=LAYOUT_CLASSIC,
            source=source,
        )
    return None


def resolve_rocm_roots(environ: dict[str, str] | None = None) -> RocmRoots:
    """Resolve the ROCm roots, first hit wins.

    Order: ``$ROCM_PATH`` -> ``$ROCM_HOME`` -> ``/opt/rocm`` -> an installed
    TheRock wheel. An explicit operator override therefore always beats
    autodetection, and the classic system path keeps priority over a wheel
    that merely happens to be importable.

    Never raises and never returns ``None``: when nothing is found the
    classic root is returned with ``source="none"``, so every derived
    constant stays an absolute path and the probe's fail-soft contract holds
    (a missing file yields ``None``, it does not crash).
    """
    env = os.environ if environ is None else environ

    for name in ROOT_ENV_VARS:
        value = env.get(name)
        if not value:
            continue
        roots = _roots_from_candidate(Path(value), name)
        if roots is not None:
            return roots
        log.debug("%s=%s does not look like a ROCm install; continuing", name, value)

    if _is_rocm_root(CLASSIC_ROCM_ROOT):
        return _classic_roots("opt_rocm")

    core = _installed_wheel_component(WHEEL_CORE_PACKAGE)
    if core is not None:
        roots = _wheel_roots(core.parent, f"import:{WHEEL_CORE_PACKAGE}")
        if roots is not None:
            return roots

    # Nothing found. Return the classic root anyway so every derived path
    # stays absolute and the probe reads it as simply missing, but record
    # that this is an assumption rather than a discovery.
    return _classic_roots("none")
