"""Buck-aware library introspection for ``aorta env probe`` (issue #163, A1.2b).

A1's existing per-library blocks (``hipblaslt``, ``rocblas``,
``miopen``, ``rccl``, ...) work by reading headers, hashing libraries
on disk, and parsing pkg-config / Docker image digests. Inside a
Buck2 monorepo none of those signals exist -- libraries are Buck
*targets*, not files at well-known FHS paths.

This module wraps ``buck2 audit dependencies <target> --transitive
--json`` and matches each transitive dep label against a small
known-pattern list (hipblaslt, pytorch, rccl, rocm). For matches it
returns one ``library_introspection`` entry per library with
``source: "buck"`` plus the Buck-specific ``target`` field. A1's
existing per-library blocks remain populated independently; the
caller (``collect_env``) merges by name with Buck winning ties.

Per aorta's external-tool policy: subprocess wrapper around the
open-source ``buck2`` binary; no Buck rules / macros / libraries
vendored.

Per the env-probe never-raises contract: every subprocess call is
guarded; timeouts and OS errors degrade silently. Callers receive a
``(entries, reasons)`` tuple -- one human-readable string per
documented failure goes into ``reasons`` so the snapshot's
``partial_reasons`` field surfaces it for the operator.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess

log = logging.getLogger(__name__)

# Library identifier -> regex patterns matched against the string form
# of `buck2 audit dependencies` labels. Order doesn't matter; any
# match wins. Patterns are anchored loosely so internal project-
# specific prefixes (e.g. //third-party/rocm:hipblaslt vs
# //rocm:hipblaslt-lib) match.
#
# To add a new library, append a new key + patterns. No other code
# changes are needed; the merge logic in environment.py handles
# arbitrary library names. Document the addition in the PR.
KNOWN_LIBRARY_PATTERNS: dict[str, list[re.Pattern]] = {
    "hipblaslt": [
        re.compile(r":hipblaslt(_lib|-lib)?$"),
        re.compile(r"^hipblaslt-"),  # cell-name prefix
    ],
    "pytorch": [
        re.compile(r":(py)?torch$"),
        re.compile(r":torch_(cuda|hip|rocm)$"),
    ],
    "rccl": [
        re.compile(r":rccl(_lib|-lib)?$"),
    ],
    "rocm": [
        re.compile(r":(hip|rocm)_runtime$"),  # libamdhip64 / librocm-runtime
    ],
}


def introspect_libraries_via_buck(
    target: str,
    repo_revision: str | None,
    timeout: int = 10,
    cwd: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Run ``buck2 audit dependencies`` and return matched library entries.

    Returns ``(entries, reasons)``:

    * ``entries`` -- one dict per matched library, in the unified
      ``library_introspection`` shape: ``{name, source: "buck",
      revision, target}``. Empty when no matches or when the audit
      fails. ``revision`` is set to ``repo_revision`` (the build-system-
      wide revision captured by ``detect_build_system``); per-target
      revisions can be derived from a follow-up ``buck2 audit
      providers`` call but are out of scope for A1.2b.
    * ``reasons`` -- one human-readable string per documented failure
      (buck2 absent, target not found, audit timeout, JSON parse
      error). Suitable for appending to ``partial_reasons``. Empty on
      a clean audit run -- including the case where the audit
      succeeded but matched nothing (an empty match set is not a
      failure).

    Never raises. Callers get a fully-shaped tuple even on failure.

    The function does NOT validate ``target`` syntax; an invalid label
    will surface as a ``buck2 audit`` non-zero exit captured in
    ``reasons``.
    """
    buck2 = shutil.which("buck2")
    if buck2 is None:
        # Defensive: collect_env() only calls this when build_system.kind
        # == "buck2", which already implies buck2 is on PATH. If that
        # invariant is violated we still degrade cleanly.
        return [], [
            f"library_introspection: buck2 not on PATH; "
            f"--buck-target {target} ignored"
        ]

    cmd = [buck2, "audit", "dependencies", target, "--transitive", "--json"]
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], [
            f"library_introspection: buck2 audit dependencies {target} "
            f"timed out after {timeout}s (raise --buck-timeout if needed)"
        ]
    except (FileNotFoundError, OSError) as exc:
        return [], [
            f"library_introspection: buck2 audit dependencies {target} "
            f"failed to launch ({exc})"
        ]

    if result.returncode != 0:
        # Common cause: target not found, or buck2 misconfiguration.
        # Surface the stderr (truncated) so the operator can debug
        # without re-running by hand.
        stderr = (result.stderr or "").strip()[:300]
        return [], [
            f"library_introspection: buck2 audit dependencies {target} "
            f"exited {result.returncode} (stderr: {stderr or '(empty)'})"
        ]

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return [], [
            f"library_introspection: buck2 audit dependencies {target} "
            f"returned non-JSON output ({exc})"
        ]

    # `buck2 audit dependencies --json` returns a dict mapping the
    # queried target(s) to a list of dep labels. Flatten across all
    # queried targets (we pass exactly one, but the shape is generic).
    dep_labels: list[str] = []
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                dep_labels.extend(str(x) for x in v)
    else:
        return [], [
            f"library_introspection: buck2 audit dependencies {target} "
            f"returned unexpected JSON shape ({type(payload).__name__})"
        ]

    entries: list[dict] = []
    seen_names: set[str] = set()
    for label in dep_labels:
        match = _match_library(label)
        if match is None:
            continue
        if match in seen_names:
            # Duplicate match: keep the first label (deterministic by
            # `buck2 audit dependencies` alphabetical ordering). The
            # second-match label would be lost; record nothing.
            continue
        seen_names.add(match)
        entries.append(
            {
                "name": match,
                "source": "buck",
                "revision": repo_revision,
                "target": label,
            }
        )

    return entries, []


def _match_library(label: str) -> str | None:
    """Return the library name a Buck target label resolves to, or None."""
    for name, patterns in KNOWN_LIBRARY_PATTERNS.items():
        for pat in patterns:
            if pat.search(label):
                return name
    return None
