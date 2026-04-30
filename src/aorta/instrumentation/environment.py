"""``aorta env probe`` implementation (issue #147).

Library-first per the updated A1 spec: the primary deliverable is

* ``collect_env() -> EnvSnapshot``

which B1 (per-trial runner) and B2 (matrix runner) call **in-process** so
every trial / matrix run records its environment without shelling out.
``aorta env probe`` (CLI) is a thin wrapper around it.

Captured blocks:

* ``system_health`` -- verbatim ``rdhc --quick --json`` output (or null).
* ``rocm`` -- explicit reads of ``/opt/rocm/.info/version{,_dev}`` and
  ``/sys/module/amdgpu/version``.
* ``hip`` -- ``hipconfig`` toolchain outputs.
* ``hipblaslt`` -- commit + library hash + Tensile fingerprint + applied
  PR flags.
* ``runtime_context`` -- container runtime + Python env detection.
* ``docker`` -- image + digest when in a container.
* ``env_vars`` -- canonical list of HSA / RCCL / FBGEMM / PyTorch vars.
* ``python_version``, ``pytorch_version``.

Fail-soft contract: ``collect_env()`` NEVER raises. Every probe that falls
back to ``None`` appends a human-readable reason to ``partial_reasons``;
the snapshot is then marked ``partial=True``. This keeps a triage matrix
running when the probe hits something missing (no rdhc, no sudo,
restricted dmesg) instead of aborting the whole run.

No GPU compute. No tensor allocations. Target wall time: <15 s with rdhc
present, <5 s without. Every capture function returns a fully-shaped dict
with ``None`` for missing values, so the schema is stable: keys never go
missing across environments.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# RDHC subprocess budget. The issue caps at 30 s; we keep that to stay
# inside the 30 s worst-case env probe budget.
RDHC_TIMEOUT_SEC = 30.0

# Pointer appended to install-related rdhc partial_reasons so operators
# who hit ``system_health: null`` find the install + sudo recipe right
# away. Not appended to timeout / parse-failure reasons -- those are
# rdhc-side runtime issues, not install issues.
_RDHC_INSTALL_HINT = "see docs/env-probe.md#installing-rdhc"

# Generic per-subprocess budget for hipconfig, dpkg, etc. None of these
# should take more than a second on a healthy host.
SHORT_TIMEOUT_SEC = 5.0

# Canonical env var list -- explicit, NOT prefix matching. Workload
# config (AMP_DTYPE, MODEL_DTYPE, SHAMPOO_PRECONDITIONER_DTYPE) belongs
# in the trial result emitted by ``aorta run`` (Task B1), so it is
# deliberately absent here. Asserted by tests.
CANONICAL_ENV_VARS: tuple[str, ...] = (
    # HSA / runtime
    "HSA_XNACK",
    "HSA_KERNARG_POOL_SIZE",
    "HSA_NO_SCRATCH_RECLAIM",
    # GPU queue / codegen
    "GPU_MAX_HW_QUEUES",
    "AMDGCN_USE_BUFFER_OPS",
    "DISABLE_TF32",
    # RCCL / NCCL
    "NCCL_MAX_NCHANNELS",
    # FBGEMM
    "FBGEMM_NO_JK",
    "FBGEMM_TBE_V2",
    "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL",
    "FBGEMM_BOUNDS_CHECK_INDICES_V2",
    # PyTorch / inductor
    "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
    "PYTORCH_CUDA_ALLOC_CONF",
)

# Filesystem locations -- collected here so tests can monkeypatch them.
# Each constant is paired with a short note on the source so future
# editors don't have to rediscover where the data comes from.
#
# All paths verified against:
#   - host: ROCm 7.2.1 baremetal install
#   - rocm/pytorch:7.2.0 docker image
#   - rocm/pytorch:7.0.2.1 docker image (`version_dev` legitimately absent)
# Each path is absolute; the structural test in
# tests/instrumentation/test_environment.py::TestPathConstants
# enforces this so a future relative-path typo fails fast.

# Per #147 schema: "Explicit ROCm version files".
# /opt/rocm is conventionally a symlink to the active versioned install
# (e.g., /opt/rocm-7.2.1), so /opt/rocm/.info/version always points at the
# active release.
ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")            # release tag, e.g. "7.2.1"
ROCM_VERSION_DEV_FILE = Path("/opt/rocm/.info/version-dev")    # full build, e.g. "7.2.1-43"
# Linux kernel-side AMDGPU module version (KMD = kernel-mode driver).
# Provided by the kernel since the amdgpu module exposes a sysfs `version`.
KMD_VERSION_FILE = Path("/sys/module/amdgpu/version")          # e.g. "6.16.13"

# hipBLASLt build identity sources. The version header ships in the
# hipblaslt-dev package; on hosts without it, _capture_hipblaslt's commit
# / package_version fields fall back to None with a recorded reason.
HIPBLASLT_VERSION_HEADER = Path(
    "/opt/rocm/include/hipblaslt/hipblaslt-version.h"
)
HIPBLASLT_LIB_DIR = Path("/opt/rocm/lib")  # libhipblaslt.so* lives here
HIPBLASLT_TENSILE_DIR = Path(
    "/opt/rocm/lib/hipblaslt/library"
)  # Tensile kernel database (.dat on modern builds, .yaml on older)

# Container runtime detection markers.
# /.dockerenv -- created by Docker since the early days.
# /run/.containerenv -- created by Podman.
# /proc/1/cgroup -- the init process's cgroup; last-resort cgroup token
#   sniff for the runtime *type* (docker / podman / singularity), per
#   OCI conventions.
# /proc/self/cgroup -- the current process's cgroup, parsed for the
#   container *ID* (a 12-64 hex SHA segment). Distinct file, distinct
#   purpose -- both go in the constants block so tests can monkeypatch
#   either without touching the real /proc.
DOCKERENV_MARKER = Path("/.dockerenv")
PODMAN_CONTAINERENV_MARKER = Path("/run/.containerenv")
CGROUP_FILE = Path("/proc/1/cgroup")
SELF_CGROUP_FILE = Path("/proc/self/cgroup")


# ---------------------------------------------------------------------------
# EnvSnapshot dataclass -- the public typed object B1/B2/CLI consume
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSnapshot:
    """Wraps the env.json schema as a typed object.

    Attributes mirror the env.json keys 1-to-1; ``to_dict()`` / ``from_dict()``
    round-trip losslessly. The dataclass is ``frozen=True``, which prevents
    *attribute rebinding* on the snapshot itself (``snap.rocm = ...`` raises),
    but it does NOT deep-freeze the nested ``dict`` / ``list`` containers --
    callers can still mutate ``snap.rocm["version"] = ...`` or
    ``snap.partial_reasons.append(...)`` in place. Treat embedded snapshots
    as read-only; if you need to modify, deep-copy first via
    ``EnvSnapshot.from_dict(copy.deepcopy(snap.to_dict()))``.

    The two fail-soft fields make the snapshot honest about partial captures:

    * ``partial`` -- True if at least one probe fell back to None when it was
      expected to populate. False on a clean probe.
    * ``partial_reasons`` -- one human-readable string per fallback. The list
      is empty when ``partial`` is False. Each entry names the field plus a
      short cause (e.g. ``"system_health: rdhc not on PATH"``).

    "Documented absences" do NOT trigger partial:

    * ``docker == None`` on baremetal (no container, nothing to record)
    * ``env_vars[X] == None`` for an unset env var (the documented contract)
    * ``runtime_context.venv_path == None`` outside a venv

    "Probe fell back" cases that DO trigger partial:

    * ``system_health == None`` (rdhc unavailable / no sudo / timeout)
    * any field in ``rocm`` / ``hip`` / ``hipblaslt`` is None
    * ``pytorch_version == None`` (torch not installed)
    * ``docker.image`` / ``docker.digest`` is None when inside a container
    """

    schema_version: str
    captured_at: str
    system_health: dict | None
    rocm: dict
    hip: dict
    hipblaslt: dict
    runtime_context: dict
    docker: dict | None
    env_vars: dict[str, str | None]
    python_version: str
    pytorch_version: str | None
    partial: bool
    partial_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the env.json shape. Round-trip pair with from_dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvSnapshot:
        """Reconstruct from a previously serialised env.json dict.

        Tolerates extra unknown keys (forward-compat) and missing optional
        ``partial_reasons`` (defaults to empty list).
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs.setdefault("partial_reasons", [])
        return cls(**kwargs)

    def summary(self) -> str:
        """Human-friendly multi-line summary for CLI / logs.

        Six lines, fixed width labels. Used by ``aorta env probe`` to print
        a brief after writing the JSON.
        """
        rt = self.runtime_context or {}
        rocm = self.rocm or {}
        hip = self.hip or {}
        hipblaslt = self.hipblaslt or {}
        # Use ``is not None`` -- RDHC may return an empty dict on a healthy
        # host with nothing to report, which is still a successful capture
        # and must NOT be summarised as "unavailable".
        sysh = (
            "present"
            if self.system_health is not None
            else "unavailable (system_health=null)"
        )
        partial_marker = (
            f" [PARTIAL, {len(self.partial_reasons)} reason(s)]"
            if self.partial
            else ""
        )
        return "\n".join(
            (
                f"  runtime:  {rt.get('type', '?')} / python={rt.get('python_env', '?')}{partial_marker}",
                f"  rocm:     {rocm.get('version', '?')} (dev: {rocm.get('version_dev', '?')})",
                f"  hip:      {hip.get('version', '?')} ({hip.get('platform', '?')})",
                f"  hipblaslt: commit={hipblaslt.get('commit', '?')}",
                f"  rdhc:     {sysh}",
                f"  python:   {self.python_version} | pytorch: {self.pytorch_version}",
            )
        )


# ---------------------------------------------------------------------------
# collect_env -- the public entrypoint B1 / B2 / CLI all call
# ---------------------------------------------------------------------------


def collect_env() -> EnvSnapshot:
    """Capture the current process environment as an :class:`EnvSnapshot`.

    NEVER raises. The promise is enforced two ways:

    * Every probe is individually fail-soft (returns None or a shaped
      dict; appends a human-readable reason to a shared list on fallback).
    * The orchestrator body is wrapped in a top-level ``try / except
      Exception``. If anything genuinely unexpected raises (a stdlib call
      misbehaves, a probe is buggy, etc.), the disaster-recovery helper
      :func:`_disaster_snapshot` constructs a minimally-shaped
      :class:`EnvSnapshot` with ``partial=True`` and the original
      exception captured in ``partial_reasons``. Callers (B1 dispatcher,
      B2 matrix runner, CLI) always get back a valid object.

    No GPU compute. No tensor allocations. The optional ``import torch``
    for the version probe does NOT initialise CUDA / HIP context.
    """
    reasons: list[str] = []
    try:
        runtime_context = _detect_runtime_context()  # never partial; always populates
        system_health = _run_rdhc(reasons)
        rocm = _capture_rocm_version_files(reasons)
        hip = _capture_hip_toolchain(reasons)
        hipblaslt = _capture_hipblaslt(reasons)
        docker = _capture_docker_metadata(runtime_context, reasons)
        env_vars = _capture_env_vars()  # individual nulls are documented, not partial
        pytorch_version = _capture_pytorch_version(reasons)

        return EnvSnapshot(
            schema_version=SCHEMA_VERSION,
            captured_at=_utc_now_iso(),
            system_health=system_health,
            rocm=rocm,
            hip=hip,
            hipblaslt=hipblaslt,
            runtime_context=runtime_context,
            docker=docker,
            env_vars=env_vars,
            python_version=platform.python_version(),
            pytorch_version=pytorch_version,
            partial=bool(reasons),
            partial_reasons=reasons,
        )
    except Exception as exc:  # noqa: BLE001 -- this is the never-raises gate
        log.info("collect_env() hit unexpected exception", exc_info=True)
        return _disaster_snapshot(
            preceding_reasons=reasons,
            unexpected_reason=(
                f"collect_env: unexpected failure "
                f"({type(exc).__name__}: {exc})"
            ),
        )


def _disaster_snapshot(
    preceding_reasons: list[str], unexpected_reason: str
) -> EnvSnapshot:
    """Return a minimally-shaped EnvSnapshot when collect_env crashes.

    Used by the never-raises top-level guard. Every field gets a sane
    null/empty default so downstream consumers (B1, B2, jq pipelines)
    still see the schema they expect, with ``partial=True`` and the
    triggering exception in ``partial_reasons``.

    Even helpers used here are guarded -- if ``_utc_now_iso`` or
    ``platform.python_version`` themselves raise, we fall back to empty
    strings rather than re-throw.
    """
    try:
        captured_at = _utc_now_iso()
    except Exception:  # noqa: BLE001
        captured_at = ""
    try:
        python_version = platform.python_version()
    except Exception:  # noqa: BLE001
        python_version = ""

    return EnvSnapshot(
        schema_version=SCHEMA_VERSION,
        captured_at=captured_at,
        system_health=None,
        rocm={"version": None, "version_dev": None, "kmd_version": None},
        hip={
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        },
        hipblaslt={
            "commit": None,
            "package_version": None,
            "lib_hash": None,
            "tensile_yaml_revision": None,
            "applied_prs": {},
        },
        runtime_context={
            "type": "baremetal",
            "python_env": "system",
            "venv_path": None,
            "conda_env_name": None,
        },
        docker=None,
        env_vars=dict.fromkeys(CANONICAL_ENV_VARS),
        python_version=python_version,
        pytorch_version=None,
        partial=True,
        partial_reasons=[*preceding_reasons, unexpected_reason],
    )


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with trailing 'Z' (per #147 schema example)."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# RDHC wrapper
# ---------------------------------------------------------------------------


def _run_rdhc(reasons: list[str]) -> dict | None:
    """Run ``sudo -n -E rdhc --quick --json <tmp>`` and return parsed dict.

    Manages its own temp file via :mod:`tempfile` -- nothing leaks into
    the env probe's output directory.

    Returns ``None`` on any of:
    * RDHC not installed (``shutil.which`` returns nothing for ``rdhc.py``
      *and* ``rdhc``).
    * ``sudo -n`` would prompt for a password (return code != 0).
    * RDHC takes longer than ``RDHC_TIMEOUT_SEC``.
    * RDHC exits non-zero or produces malformed JSON.

    Each failure mode appends one human-readable entry to ``reasons`` AND
    logs a single INFO line. Never raises.
    """
    rdhc = shutil.which("rdhc.py") or shutil.which("rdhc")
    if rdhc is None:
        msg = f"system_health: rdhc not on PATH ({_RDHC_INSTALL_HINT})"
        log.info(msg)
        reasons.append(msg)
        return None

    # Tempfile creation is part of the never-raises surface: a read-only or
    # full /tmp would otherwise abort collect_env(). Treat as "rdhc
    # unavailable" with a clear reason.
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json", prefix="rdhc_quick_", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
    except OSError as exc:
        msg = f"system_health: failed to create rdhc temp file ({exc})"
        log.info(msg)
        reasons.append(msg)
        return None

    try:
        cmd = ["sudo", "-n", "-E", rdhc, "--quick", "--json", str(tmp_path)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RDHC_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired:
            msg = f"system_health: rdhc exceeded {RDHC_TIMEOUT_SEC:.0f}s timeout"
            log.info(msg)
            reasons.append(msg)
            return None
        except (FileNotFoundError, OSError) as exc:
            # Same actionable hint as the not-on-PATH branch -- if exec
            # itself fails (e.g. broken interpreter shebang in a stripped
            # image), reinstalling rocm-systems is usually the fix.
            msg = (
                f"system_health: failed to invoke rdhc ({exc}) "
                f"({_RDHC_INSTALL_HINT})"
            )
            log.info(msg)
            reasons.append(msg)
            return None

        if result.returncode != 0:
            # Pull the last non-empty line of stderr; truncate to keep
            # partial_reasons readable in CLI output and JSON. Falls back
            # to "(no stderr; likely sudo-n unavailable)" when the child
            # printed nothing -- which is what the no-password case does.
            # The install hint is only appended for the no-stderr case
            # (where sudo config is the likely fix); when rdhc DOES print
            # to stderr the operator should debug from that, not from a
            # generic install link.
            stderr_lines = (result.stderr or "").splitlines()
            stderr_tail = next(
                (line.strip() for line in reversed(stderr_lines) if line.strip()),
                "",
            )
            if stderr_tail:
                detail = f"stderr: {stderr_tail[:200]}"
            else:
                detail = (
                    f"no stderr; likely sudo-n unavailable ({_RDHC_INSTALL_HINT})"
                )
            msg = (
                f"system_health: rdhc exited {result.returncode} ({detail})"
            )
            log.info(msg)
            reasons.append(msg)
            return None

        try:
            return json.loads(tmp_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            msg = f"system_health: rdhc output not parseable ({exc})"
            log.info(msg)
            reasons.append(msg)
            return None
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ROCm version files
# ---------------------------------------------------------------------------


def _capture_rocm_version_files(reasons: list[str]) -> dict[str, str | None]:
    """Read ROCm version markers directly from disk.

    These are explicit reads (not via RDHC) so that ``rocm.version`` is
    populated even when RDHC is unavailable. All three keys are always
    present; missing files yield ``None`` and append a reason.
    """
    block = {
        "version": _read_text_file(ROCM_VERSION_FILE),
        "version_dev": _read_text_file(ROCM_VERSION_DEV_FILE),
        "kmd_version": _read_text_file(KMD_VERSION_FILE),
    }
    paths = {
        "version": ROCM_VERSION_FILE,
        "version_dev": ROCM_VERSION_DEV_FILE,
        "kmd_version": KMD_VERSION_FILE,
    }
    # Note: _read_text_file returns None for missing, empty, permission
    # denied, AND non-utf8 cases. Reason wording covers all four so the
    # operator does not assume "missing" when the file is just empty.
    for key, value in block.items():
        if value is None:
            reasons.append(
                f"rocm.{key}: {paths[key]} missing, empty, or unreadable"
            )
    return block


def _read_text_file(path: Path) -> str | None:
    """Read a small text file; return its stripped contents or ``None``.

    Part of the ``never raises`` surface: catches everything that can come
    out of ``Path.read_text``, including ``UnicodeDecodeError`` from files
    with non-UTF8 bytes (e.g. a corrupt ``/sys/module/amdgpu/version``
    or a locale-mismatched ``/opt/rocm/.info/*``). Returns ``None`` for
    every error path so the caller can record a partial reason instead of
    crashing.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    except UnicodeDecodeError as exc:
        log.debug("non-utf8 contents in %s: %s", path, exc)
        return None
    except OSError as exc:
        log.debug("read failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# HIP toolchain
# ---------------------------------------------------------------------------


def _capture_hip_toolchain(reasons: list[str]) -> dict[str, str | None]:
    """Run ``hipconfig --<flag>`` for each toolchain field.

    Issued as separate invocations because hipconfig prints results
    concatenated when multiple flags are passed, with no delimiter -- a
    single ``hipconfig --version --platform`` produces ``"7.2.5amd"``,
    which is unparseable. Five short subprocesses still finish in <100 ms.
    """
    if shutil.which("hipconfig") is None:
        msg = "hip: hipconfig not on PATH; all hip.* fields = null"
        log.info(msg)
        reasons.append(msg)
        return {
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        }

    block = {
        "version": _hipconfig("--version"),
        "platform": _hipconfig("--platform"),
        "compiler": _hipconfig("--compiler"),
        "runtime": _hipconfig("--runtime"),
        "cpp_config": _hipconfig("--cpp_config"),
    }
    for key, value in block.items():
        if value is None:
            reasons.append(f"hip.{key}: hipconfig --{key} returned no usable value")
    return block


def _hipconfig(flag: str) -> str | None:
    """Run ``hipconfig <flag>`` and return stripped stdout (or None)."""
    try:
        result = subprocess.run(
            ["hipconfig", flag],
            capture_output=True,
            text=True,
            timeout=SHORT_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


# ---------------------------------------------------------------------------
# hipBLASLt introspection
# ---------------------------------------------------------------------------


# The version-tweak field in hipblaslt-version.h is the canonical source
# for the build's git SHA (typically a 7-12 char short hash).
_HIPBLASLT_TWEAK_RE = re.compile(
    r"#define\s+HIPBLASLT_VERSION_TWEAK\s+([A-Za-z0-9_.+-]+)"
)
_HIPBLASLT_VERSION_RE = re.compile(
    r"#define\s+HIPBLASLT_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)"
)


def _capture_hipblaslt(reasons: list[str]) -> dict[str, Any]:
    """Capture hipBLASLt build identity.

    Goal: catch GEMM kernel library drift across docker images / conda
    envs / venvs. See issue #147 motivation.

    The ``applied_prs`` block is intentionally empty in this first cut.
    Adding ``pr_<id>_applied`` keys later is additive and does not bump
    ``schema_version``. Each PR detector needs a unique signature
    (symbol via ``nm``, string via ``strings``, or Tensile YAML revision
    bump) -- those will land in a follow-up alongside the first PR we
    care to track.
    """
    header_text = _read_text_file(HIPBLASLT_VERSION_HEADER)
    commit, package_version = _parse_hipblaslt_header(header_text)
    lib_hash = _hash_hipblaslt_library()
    tensile_yaml_revision = _tensile_fingerprint()

    block: dict[str, Any] = {
        "commit": commit,
        "package_version": package_version,
        "lib_hash": lib_hash,
        "tensile_yaml_revision": tensile_yaml_revision,
        "applied_prs": {},
    }
    # Distinguish "header file unreadable" from "header readable but the
    # specific define is missing/unparseable" so partial_reasons points
    # callers at the right thing to investigate.
    header_unreadable = header_text is None
    if commit is None:
        if header_unreadable:
            reasons.append(
                f"hipblaslt.commit: {HIPBLASLT_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"hipblaslt.commit: {HIPBLASLT_VERSION_HEADER} did not "
                "contain a readable HIPBLASLT_VERSION_TWEAK define"
            )
    if package_version is None:
        if header_unreadable:
            reasons.append(
                f"hipblaslt.package_version: {HIPBLASLT_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"hipblaslt.package_version: {HIPBLASLT_VERSION_HEADER} did not "
                "contain MAJOR/MINOR/PATCH defines"
            )
    if lib_hash is None:
        reasons.append(
            f"hipblaslt.lib_hash: {HIPBLASLT_LIB_DIR}/libhipblaslt.so "
            "missing or unreadable"
        )
    if tensile_yaml_revision is None:
        # _tensile_fingerprint returns None for both "directory missing /
        # unlistable" AND "directory present but no kernel files" --
        # the wording covers both so partial_reasons is honest.
        reasons.append(
            "hipblaslt.tensile_yaml_revision: directory missing/unreadable "
            f"or no kernel files under {HIPBLASLT_TENSILE_DIR}"
        )
    return block


def _parse_hipblaslt_header(text: str | None) -> tuple[str | None, str | None]:
    """Extract commit (TWEAK) and package_version (MAJOR.MINOR.PATCH).

    Returns (commit, package_version), each ``None`` if the header was
    missing or did not contain the expected defines.
    """
    if not text:
        return (None, None)
    tweak_match = _HIPBLASLT_TWEAK_RE.search(text)
    commit = tweak_match.group(1) if tweak_match else None
    parts: dict[str, str] = {}
    for match in _HIPBLASLT_VERSION_RE.finditer(text):
        parts[match.group(1)] = match.group(2)
    if {"MAJOR", "MINOR", "PATCH"}.issubset(parts):
        package_version = f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"
    else:
        package_version = None
    return (commit, package_version)


def _hash_hipblaslt_library() -> str | None:
    """SHA-256 the canonical (resolved) ``libhipblaslt.so``.

    Resolves through symlinks so e.g. ``libhipblaslt.so`` ->
    ``libhipblaslt.so.1`` -> ``libhipblaslt.so.1.2.70201`` collapses to one
    hash regardless of which name the consumer linked against.
    Returns ``"sha256:<hex>"`` or ``None``.
    """
    candidate = HIPBLASLT_LIB_DIR / "libhipblaslt.so"
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    try:
        digest = hashlib.sha256()
        with resolved.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except OSError as exc:
        log.debug("hash failed for %s: %s", resolved, exc)
        return None


def _tensile_fingerprint() -> str | None:
    """Fingerprint the Tensile kernel database.

    Modern hipBLASLt ships ``.dat`` files (binary), older builds shipped
    ``.yaml``. We hash the **sorted filenames** of all kernel files in
    the library dir -- a fast, deterministic fingerprint that changes
    whenever the kernel set changes (new gfx target, new operation
    layout, removed kernel). Hashing the contents would be GB of work
    and add seconds; the filename set already tracks the meaningful
    drift.
    """
    if not HIPBLASLT_TENSILE_DIR.is_dir():
        return None
    try:
        names = sorted(
            p.name
            for p in HIPBLASLT_TENSILE_DIR.iterdir()
            if p.is_file() and p.suffix in (".yaml", ".dat", ".co")
        )
    except OSError as exc:
        log.debug("tensile dir listing failed: %s", exc)
        return None
    if not names:
        return None
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return f"filenames-sha256:{digest}"


# ---------------------------------------------------------------------------
# Runtime context: container + Python env detection
# ---------------------------------------------------------------------------


def _detect_runtime_context() -> dict[str, str | None]:
    """Detect container runtime + Python environment.

    The schema's allowed values for ``runtime_context.type`` are
    ``docker | podman | singularity | baremetal`` (per #147). To keep
    strict consumers safe, this function only ever returns one of those
    four; runtimes outside the documented set (e.g. containerd-managed
    Kubernetes pods) currently fall through to ``baremetal``. Adding new
    values is a schema change and would bump ``schema_version``.

    Container precedence (first match wins):
        1. ``/.dockerenv`` -> docker
        2. ``/run/.containerenv`` -> podman
        3. ``SINGULARITY_NAME`` env var or ``singularity`` in
           ``/proc/1/cgroup`` -> singularity
        4. ``docker`` / ``podman`` token in ``/proc/1/cgroup``
           -> matched runtime (cgroup fallback for stripped containers)
        5. otherwise -> baremetal

    Python env precedence:
        1. ``$CONDA_DEFAULT_ENV`` -> conda
        2. ``sys.prefix != sys.base_prefix`` -> venv
        3. otherwise -> system

    Never partial: every field is either populated or has a documented
    null reason (e.g. ``venv_path`` is null outside a venv -- that is the
    contract, not a fallback).
    """
    container_type = _detect_container_type()
    python_env = _detect_python_env()
    return {
        "type": container_type,
        "python_env": python_env,
        "venv_path": str(sys.prefix) if python_env == "venv" else None,
        "conda_env_name": (
            os.environ.get("CONDA_DEFAULT_ENV") if python_env == "conda" else None
        ),
    }


def _detect_container_type() -> str:
    """Resolve the container runtime; ``"baremetal"`` if none matched."""
    if DOCKERENV_MARKER.exists():
        return "docker"
    if PODMAN_CONTAINERENV_MARKER.exists():
        return "podman"
    if os.environ.get("SINGULARITY_NAME"):
        return "singularity"

    cgroup = _read_text_file(CGROUP_FILE)
    if cgroup:
        # cgroup lines look like '12:freezer:/docker/<id>' or
        # '0::/system.slice/docker-<id>.scope'. Apply the documented
        # precedence: singularity wins over docker/podman so a
        # Singularity environment that happens to inherit a docker-shaped
        # cgroup path is not misclassified. Limited to schema-documented
        # values.
        for runtime in ("singularity", "docker", "podman"):
            if runtime in cgroup:
                return runtime
    return "baremetal"


def _detect_python_env() -> str:
    """Return ``"conda"``, ``"venv"``, or ``"system"``."""
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return "conda"
    # sys.base_prefix differs from sys.prefix inside a venv (PEP 405).
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return "venv"
    return "system"


# ---------------------------------------------------------------------------
# Docker metadata
# ---------------------------------------------------------------------------


def _capture_docker_metadata(
    runtime_context: dict[str, str | None],
    reasons: list[str],
) -> dict[str, str | None] | None:
    """Capture image + digest when running inside a container.

    Returns ``None`` for baremetal -- there is no image to record. This
    documented absence does NOT trigger ``partial=True``.

    For containerised runs we emit the block with best-effort values; the
    aorta-side launcher can populate them via the env vars below before
    invoking ``aorta env probe`` (which is the only reliable way to know
    image+digest from inside a container without privileged docker access):

    * ``AORTA_DOCKER_IMAGE``  -> ``docker.image``
    * ``AORTA_DOCKER_DIGEST`` -> ``docker.digest``

    Always also emits ``container_id`` parsed from ``/proc/self/cgroup``,
    which is recoverable from inside the container.

    When inside a container but the launcher did not set the env vars,
    appends a reason -- the snapshot can still be useful but cross-image
    comparison loses fidelity.
    """
    if runtime_context.get("type") == "baremetal":
        return None

    block = {
        "image": os.environ.get("AORTA_DOCKER_IMAGE"),
        "digest": os.environ.get("AORTA_DOCKER_DIGEST"),
        "container_id": _read_container_id(),
    }
    if block["image"] is None:
        reasons.append(
            "docker.image: AORTA_DOCKER_IMAGE env var not set by the launcher"
        )
    if block["digest"] is None:
        reasons.append(
            "docker.digest: AORTA_DOCKER_DIGEST env var not set by the launcher"
        )
    return block


_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{12,64}")


def _read_container_id() -> str | None:
    """Pull the container ID out of ``SELF_CGROUP_FILE`` if present.

    Reads the *current* process's cgroup (not init's) -- the container
    ID lives there as a 12-64 hex segment (e.g. ``/docker/<id>/`` or
    ``/system.slice/docker-<id>.scope``).
    """
    text = _read_text_file(SELF_CGROUP_FILE)
    if not text:
        return None
    for line in text.splitlines():
        match = _CONTAINER_ID_RE.search(line)
        if match:
            return match.group(0)
    return None


# ---------------------------------------------------------------------------
# Env vars + Python/PyTorch
# ---------------------------------------------------------------------------


def _capture_env_vars() -> dict[str, str | None]:
    """Capture canonical env vars (explicit list, not prefix matching).

    Individual ``None`` values are the documented contract (env var unset)
    and DO NOT trigger ``partial=True``.
    """
    return {name: os.environ.get(name) for name in CANONICAL_ENV_VARS}


def _capture_pytorch_version(reasons: list[str]) -> str | None:
    """Best-effort import of torch to read its version. No GPU touched.

    ``import torch`` does NOT initialise CUDA / HIP context; it only
    populates Python objects. Acceptance criterion "no GPU compute" is
    preserved.

    Returns the version as a string when available, or ``None`` when torch
    is not installed OR is installed without a ``__version__`` attribute.
    Either fallback path appends a reason. Never returns the string
    ``"None"`` -- that would break consumers doing strict null checks
    against the JSON.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        reasons.append("pytorch_version: torch not importable")
        return None
    except Exception as exc:  # noqa: BLE001 -- defensive; never let env probe fail
        log.debug("torch import for version probe failed: %s", exc)
        reasons.append(f"pytorch_version: torch import raised ({type(exc).__name__})")
        return None

    version = getattr(torch, "__version__", None)
    if version is None:
        reasons.append("pytorch_version: torch lacks __version__ attribute")
        return None
    return str(version)
