"""Build-system detection for ``aorta env probe`` (issue #163, A1.2a).

Detects whether the probe is running inside a Buck2 build environment.
A1's existing ROCm/PyTorch introspection assumes libraries are
discoverable via system package managers (apt / dnf / pkg-config) or
Docker image digests. Inside a Buck2 monorepo those signals are absent
-- libraries are Buck targets, the runtime root is a Buck repo, and
source revisions live in hg or git via Buck's repo root, not in /etc.

This module captures the metadata block; A1.2b wires Buck-aware library
introspection on top of it.

Public API (one function):

* ``detect_build_system() -> dict`` -- always returns a dict. Shape:
  - ``{"kind": "buck2", "buck2_version": str, "repo_root": str, "revision": str | None}``
    when buck2 is on PATH AND both ``buck2 --version`` and ``buck2
    root`` succeed (i.e., we are demonstrably running inside a
    functional Buck2 checkout). The ``buck2_version`` and ``repo_root``
    fields are guaranteed populated; only ``revision`` may be ``None``
    (e.g., a Buck2 sample repo with no VCS, or hg/git absent).
  - ``{"kind": "none"}`` in every other case, including the common
    "buck2 is installed but the current directory is not inside a
    Buck repo" scenario where ``buck2 root`` exits non-zero.

Per the wider env-probe contract this function NEVER raises. Every
subprocess call is wrapped; timeouts and non-zero exits degrade
silently to the ``{"kind": "none"}`` shape.

Per aorta's external-tool policy (wrap, don't vendor): we shell out to
``buck2``, ``hg``, and ``git``. No buck2 binaries, rules, or macros are
vendored.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

log = logging.getLogger(__name__)

# Per-subprocess budget. buck2 / hg / git are expected to answer in well
# under a second on a healthy host; the cap exists to bound the worst
# case (lock contention, NFS-backed repo root). Matches SHORT_TIMEOUT_SEC
# in environment.py but is kept independent so build_system can be
# imported without pulling in the rest of the probe.
_BUCK_TIMEOUT_SEC = 5.0


def detect_build_system() -> dict:
    """Detect the active build system. Returns a fully-shaped dict.

    Always returns one of the two documented shapes; never raises.

    Detection sequence:

    1. ``buck2`` not on PATH -> ``{"kind": "none"}`` immediately, no
       subprocess work.
    2. ``buck2 --version`` fails -> ``{"kind": "none"}``. The binary is
       on PATH but non-functional (broken symlink, missing toolchain).
    3. ``buck2 root`` fails -> ``{"kind": "none"}``. This is the
       dominant "buck2 is installed but cwd is not inside a Buck repo"
       case; we deliberately refuse to claim ``kind=buck2`` here
       because (a) it would misclassify a non-Buck environment, and
       (b) the revision lookup that follows would run against the
       current working directory rather than a real Buck checkout,
       producing a misleading SHA.
    4. Both buck2 calls succeed -> ``{"kind": "buck2", ...}`` with
       ``buck2_version`` and ``repo_root`` guaranteed populated. The
       VCS lookup runs inside ``repo_root``; on failure ``revision``
       is ``None`` but the buck2 dict still stands.
    """
    buck2 = shutil.which("buck2")
    if buck2 is None:
        return {"kind": "none"}

    version = _run_capture([buck2, "--version"])
    repo_root = _run_capture([buck2, "root"])

    # Both core buck2 calls must succeed before we claim kind=buck2.
    # The asymmetric-failure branches (one succeeded, one didn't) are
    # collapsed into kind=none on purpose: emitting a buck2 dict with
    # one field None would lie about the env (we're not actually in a
    # functional Buck checkout) and would push the half-shaped dict
    # into downstream consumers (B1/B2, recipe emitter in A1.2c).
    if version is None or repo_root is None:
        log.info(
            "build_system: buck2 on PATH but version=%r repo_root=%r; "
            "reporting kind=none (need both to claim kind=buck2)",
            version,
            repo_root,
        )
        return {"kind": "none"}

    return {
        "kind": "buck2",
        "buck2_version": version,
        "repo_root": repo_root,
        "revision": _detect_revision(repo_root),
    }


def _detect_revision(repo_root: str) -> str | None:
    """Resolve the source revision of ``repo_root``.

    Prefers Mercurial (``hg id -i``) since several large Buck2
    deployments are hg-backed; falls back to git
    (``git rev-parse HEAD``). Returns ``None`` if neither resolves --
    e.g., a fresh Buck2 demo repo with no VCS, or a repo where the VCS
    binary is absent.

    Always runs inside ``repo_root`` so the answer is unambiguous.
    Caller (``detect_build_system``) guarantees a non-empty
    ``repo_root`` before this is invoked: if ``buck2 root`` failed we
    have already short-circuited to ``{"kind": "none"}`` rather than
    fall through to a cwd-relative VCS lookup.
    """
    if shutil.which("hg") is not None:
        rev = _run_capture(["hg", "id", "-i"], cwd=repo_root)
        if rev:
            # `hg id -i` appends a "+" suffix when the working copy has
            # uncommitted changes. Keep the suffix -- it's load-bearing
            # for triage ("the env was captured against a dirty tree").
            return rev

    if shutil.which("git") is not None:
        rev = _run_capture(["git", "rev-parse", "HEAD"], cwd=repo_root)
        if rev:
            return rev

    return None


def _run_capture(cmd: list[str], cwd: str | None = None) -> str | None:
    """Run ``cmd`` and return stripped stdout, or ``None`` on any failure.

    Failures captured: command not found (FileNotFoundError), non-zero
    exit, timeout, OSError. Each path logs at INFO and returns ``None``;
    callers decide how to surface the absence.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_BUCK_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        log.info("build_system: %s failed (%s)", cmd[0], exc)
        return None
    except subprocess.TimeoutExpired:
        log.info("build_system: %s timed out after %.1fs", cmd[0], _BUCK_TIMEOUT_SEC)
        return None

    if result.returncode != 0:
        log.info(
            "build_system: %s exited %d (stderr: %s)",
            cmd[0],
            result.returncode,
            (result.stderr or "").strip()[:200],
        )
        return None

    out = (result.stdout or "").strip()
    return out or None
