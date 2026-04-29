"""``aorta env probe`` implementation (issue #147).

Captures a versioned, schema-stable snapshot of the trial environment:

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
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# RDHC subprocess budget. The issue caps at 30 s; we keep that to stay
# inside the 30 s worst-case env probe budget.
RDHC_TIMEOUT_SEC = 30.0

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
ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")
ROCM_VERSION_DEV_FILE = Path("/opt/rocm/.info/version-dev")
KMD_VERSION_FILE = Path("/sys/module/amdgpu/version")

HIPBLASLT_VERSION_HEADER = Path("/opt/rocm/include/hipblaslt/hipblaslt-version.h")
HIPBLASLT_LIB_DIR = Path("/opt/rocm/lib")
HIPBLASLT_TENSILE_DIR = Path("/opt/rocm/lib/hipblaslt/library")

DOCKERENV_MARKER = Path("/.dockerenv")
PODMAN_CONTAINERENV_MARKER = Path("/run/.containerenv")
CGROUP_FILE = Path("/proc/1/cgroup")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def capture_environment(output_path: str | os.PathLike[str] = "env.json") -> dict[str, Any]:
    """Capture the env probe snapshot and write it to ``output_path``.

    Args:
        output_path: File to write the JSON snapshot to. Parent dirs
            created.

    Returns:
        The same dict that was written to disk.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_context = _detect_runtime_context()
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _utc_now_iso(),
        "system_health": _run_rdhc(),
        "rocm": _capture_rocm_version_files(),
        "hip": _capture_hip_toolchain(),
        "hipblaslt": _capture_hipblaslt(),
        "runtime_context": runtime_context,
        "docker": _capture_docker_metadata(runtime_context),
        "env_vars": _capture_env_vars(),
        "python_version": platform.python_version(),
        "pytorch_version": _capture_pytorch_version(),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str, sort_keys=False)

    log.info("Wrote env probe to %s (schema_version=%s)", output_path, SCHEMA_VERSION)
    return snapshot


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


def _run_rdhc() -> dict | None:
    """Run ``sudo -n -E rdhc --quick --json <tmp>`` and return parsed dict.

    Manages its own temp file via :mod:`tempfile` -- nothing leaks into
    the env probe's output directory.

    Returns None on any of:
    * RDHC not installed (``shutil.which`` returns nothing for ``rdhc.py``
      *and* ``rdhc``).
    * ``sudo -n`` would prompt for a password (return code != 0).
    * RDHC takes longer than ``RDHC_TIMEOUT_SEC``.
    * RDHC exits non-zero or produces malformed JSON.

    All failure modes log a single INFO line so users understand why
    ``system_health`` is null. Never raises.
    """
    rdhc = shutil.which("rdhc.py") or shutil.which("rdhc")
    if rdhc is None:
        log.info("system_health=null: rdhc not on PATH")
        return None

    # NamedTemporaryFile with delete=False so we control cleanup ourselves
    # in the finally block (sudo'd subprocess writes to this path; the
    # default delete=True closes the fd before subprocess can use it on
    # some platforms).
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="rdhc_quick_", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

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
            log.info("system_health=null: rdhc exceeded %.0fs timeout", RDHC_TIMEOUT_SEC)
            return None
        except (FileNotFoundError, OSError) as exc:
            log.info("system_health=null: failed to invoke rdhc (%s)", exc)
            return None

        if result.returncode != 0:
            # Most common cause: sudo -n requires a password.
            log.info(
                "system_health=null: rdhc exited %s (likely sudo-n unavailable)",
                result.returncode,
            )
            return None

        try:
            return json.loads(tmp_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            log.info("system_health=null: rdhc output not parseable (%s)", exc)
            return None
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ROCm version files
# ---------------------------------------------------------------------------


def _capture_rocm_version_files() -> dict[str, str | None]:
    """Read ROCm version markers directly from disk.

    These are explicit reads (not via RDHC) so that ``rocm.version`` is
    populated even when RDHC is unavailable. All three keys are always
    present; missing files yield ``None``.
    """
    return {
        "version": _read_text_file(ROCM_VERSION_FILE),
        "version_dev": _read_text_file(ROCM_VERSION_DEV_FILE),
        "kmd_version": _read_text_file(KMD_VERSION_FILE),
    }


def _read_text_file(path: Path) -> str | None:
    """Read a small text file; return its stripped contents or ``None``."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    except OSError as exc:
        log.debug("read failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# HIP toolchain
# ---------------------------------------------------------------------------


def _capture_hip_toolchain() -> dict[str, str | None]:
    """Run ``hipconfig --<flag>`` for each toolchain field.

    Issued as separate invocations because hipconfig prints results
    concatenated when multiple flags are passed, with no delimiter -- a
    single ``hipconfig --version --platform`` produces ``"7.2.5amd"``,
    which is unparseable. Five short subprocesses still finish in <100 ms.
    """
    if shutil.which("hipconfig") is None:
        log.info("hip block: hipconfig not on PATH; all hip.* fields = null")
        return {
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        }

    return {
        "version": _hipconfig("--version"),
        "platform": _hipconfig("--platform"),
        "compiler": _hipconfig("--compiler"),
        "runtime": _hipconfig("--runtime"),
        "cpp_config": _hipconfig("--cpp_config"),
    }


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


def _capture_hipblaslt() -> dict[str, Any]:
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
    return {
        "commit": commit,
        "package_version": package_version,
        "lib_hash": _hash_hipblaslt_library(),
        "tensile_yaml_revision": _tensile_fingerprint(),
        "applied_prs": {},  # filled in once specific PRs are configured
    }


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
        # '0::/system.slice/docker-<id>.scope'. Detect the first runtime
        # name that appears. Limited to schema-documented values.
        for runtime in ("docker", "podman", "singularity"):
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
) -> dict[str, str | None] | None:
    """Capture image + digest when running inside a container.

    Returns ``None`` for baremetal -- there is no image to record.
    For containerised runs we emit the block with best-effort values; the
    aorta-side launcher can populate them via the env vars below before
    invoking ``aorta env probe`` (which is the only reliable way to know
    image+digest from inside a container without privileged docker access):

    * ``AORTA_DOCKER_IMAGE``  -> ``docker.image``
    * ``AORTA_DOCKER_DIGEST`` -> ``docker.digest``

    Always also emits ``container_id`` parsed from ``/proc/self/cgroup``,
    which is recoverable from inside the container.
    """
    if runtime_context.get("type") == "baremetal":
        return None

    return {
        "image": os.environ.get("AORTA_DOCKER_IMAGE"),
        "digest": os.environ.get("AORTA_DOCKER_DIGEST"),
        "container_id": _read_container_id(),
    }


_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{12,64}")


def _read_container_id() -> str | None:
    """Pull the container ID out of ``/proc/self/cgroup`` if present."""
    text = _read_text_file(Path("/proc/self/cgroup"))
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
    """Capture canonical env vars (explicit list, not prefix matching)."""
    return {name: os.environ.get(name) for name in CANONICAL_ENV_VARS}


def _capture_pytorch_version() -> str | None:
    """Best-effort import of torch to read its version. No GPU touched.

    ``import torch`` does NOT initialise CUDA / HIP context; it only
    populates Python objects. Acceptance criterion "no GPU compute" is
    preserved.

    Returns the version as a string when available, or ``None`` when torch
    is not installed OR is installed without a ``__version__`` attribute.
    Never returns the string ``"None"`` -- that would break consumers
    doing strict null checks against the JSON.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 -- defensive; never let env probe fail
        log.debug("torch import for version probe failed: %s", exc)
        return None

    version = getattr(torch, "__version__", None)
    if version is None:
        return None
    return str(version)
