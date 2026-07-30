"""Buck-aware library introspection for ``aorta env probe`` (issue #163, A1.2b).

A1's existing per-library blocks (``hipblaslt``, ``rocblas``,
``miopen``, ``rccl``, ...) work by reading headers, hashing libraries
on disk, and parsing pkg-config / Docker image digests. Inside a
Buck2 monorepo none of those signals exist -- libraries are Buck
*targets*, not files at well-known FHS paths.

This module wraps a bound ``buck2 cquery 'deps(%s)' <target> --json`` and
matches each transitive dep label against a small known-pattern
list (hipblaslt, pytorch, rccl, rocm). For matches it returns one
``library_introspection`` entry per library with ``source: "buck"``
plus the Buck-specific ``target`` field. A1's existing per-library
blocks remain populated independently; the caller (``collect_env``)
merges by name with Buck winning ties.

The original A1.2b design (#163) planned to use ``buck2 audit
dependencies --transitive --json``. That subcommand was removed in
open-source buck2 before this code ever ran end-to-end. ``buck2
cquery 'deps(...)' --json`` is the documented replacement: it
returns the transitive deps of the configured target as a flat JSON
list of stringified labels. The dogfood test that flagged this is
issue #183 (the Buck-buildable CLI PR) -- caught only when the
introspection ran against a real //:aorta target for the first time.

Per aorta's external-tool policy: subprocess wrapper around the
open-source ``buck2`` binary; no Buck rules / macros / libraries
vendored.

Per the env-probe never-raises contract: every subprocess call is
guarded; timeouts and OS errors degrade silently. Callers receive a
``BuckIntrospectionResult`` dataclass. It remains iterable as
``(entries, reasons)`` for compatibility; each human-readable reason
can be surfaced in the snapshot's ``partial_reasons`` field.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass

from aorta.instrumentation.buck_invocation import BuckInvocationContext

log = logging.getLogger(__name__)

# Library identifier -> regex patterns matched against the string form
# of `buck2 cquery 'deps(...)'` labels. Order doesn't matter; any
# match wins. Patterns are anchored loosely so internal project-
# specific prefixes (e.g. //third-party/rocm:hipblaslt vs
# //rocm:hipblaslt-lib) match.
#
# Label format from `buck2 cquery --json` is
# ``<cell>//<package>:<name> (<platform-config-suffix>)``. The
# stringified suffix is stripped (see _strip_config_suffix) before
# matching so a target like ``//third-party/rocm:hipblaslt-lib
# (prelude//platforms:default#abc123)`` matches the same pattern
# regardless of which configuration buck2 picked.
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
    "ainic": [
        # AMD-ANP / RCCL net-plugin Buck targets (issue #202). The plugin
        # ships as :rccl-anp(-lib) / :rccl-net(-lib) across cells.
        re.compile(r":rccl[_-]anp(_lib|-lib)?$"),
        re.compile(r":rccl[_-]net(_lib|-lib)?$"),
    ],
    "rocm": [
        re.compile(r":(hip|rocm)_runtime$"),  # libamdhip64 / librocm-runtime
    ],
}

# `buck2 cquery --json` emits labels with a trailing parenthesised
# configuration suffix, e.g. ``//path:name (prelude//platforms:default#hash)``.
# Strip it before regex matching so pattern authors don't have to
# account for the suffix in every entry of KNOWN_LIBRARY_PATTERNS.
_CONFIG_SUFFIX_RE = re.compile(r"\s+\([^)]*\)\s*$")


@dataclass(frozen=True)
class BuckIntrospectionResult:
    """Fail-soft cquery result plus configured-root metadata."""

    entries: list[dict]
    reasons: list[str]
    succeeded: bool
    configured_root_target: str | None

    def __iter__(self):
        """Preserve the historical ``entries, reasons = result`` API."""

        yield self.entries
        yield self.reasons


def introspect_libraries_via_buck(
    target: str,
    repo_revision: str | None,
    timeout: int = 10,
    cwd: str | None = None,
    context: BuckInvocationContext | None = None,
) -> BuckIntrospectionResult:
    """Run bound ``buck2 cquery 'deps(%s)' <target>`` introspection.

    Returns a :class:`BuckIntrospectionResult`. It remains two-value iterable
    as ``(entries, reasons)`` for compatibility with existing callers:

    * ``entries`` -- one dict per matched library, in the unified
      ``library_introspection`` shape: ``{name, source: "buck",
      revision, target, configured_target}`` (schema 1.6). ``target``
      is the canonical Buck label with the cquery configuration
      suffix stripped (round-trips cleanly into another
      ``buck2 query`` / ``buck2 build`` without depending on which
      daemon configured the graph). ``configured_target`` preserves
      the raw cquery output -- ``//pkg:lib (prelude//platforms:default#<hash>)``
      -- for forensics when reconciling two probes that diverged on
      the same source tree. Empty list when no matches or when the
      cquery fails. ``revision`` is set to ``repo_revision`` (the
      build-system-wide revision captured by ``detect_build_system``);
      per-target revisions can be derived from a follow-up
      ``buck2 audit providers`` call but are out of scope for A1.2b.
    * ``reasons`` -- one human-readable string per documented failure
      (buck2 absent, target not found, cquery timeout, JSON parse
      error). Suitable for appending to ``partial_reasons``. Empty on
      a clean cquery run -- including the case where the cquery
      succeeded but matched nothing (an empty match set is not a
      failure).

    ``succeeded`` distinguishes a clean cquery (including no library matches)
    from every fail-soft path. ``configured_root_target`` retains the root
    label including its cquery configuration suffix when that label can be
    identified in the returned dependency closure.

    Never raises. Callers get a fully-shaped result even on failure.

    The function does NOT validate ``target`` syntax; an invalid label
    will surface as a ``buck2 cquery`` non-zero exit captured in
    ``reasons``.
    """
    invocation_context = context or BuckInvocationContext()
    buck2 = shutil.which("buck2")
    if buck2 is None:
        # Defensive: collect_env() only calls this when build_system.kind
        # == "buck2", which already implies buck2 is on PATH. If that
        # invariant is violated we still degrade cleanly.
        return BuckIntrospectionResult(
            entries=[],
            reasons=[
                f"library_introspection: buck2 not on PATH; " f"--buck-target {target} ignored"
            ],
            succeeded=False,
            configured_root_target=None,
        )

    # Use cquery (not uquery) so the deps reflect the configured graph
    # the build would actually use, matching the semantics the prior
    # `audit dependencies --transitive` provided.
    # Keep every user-controlled value in its own argv element. In particular,
    # bind the target through Buck's multi-query ``%s`` parameter instead of
    # interpolating it into the query language, where query syntax in a target
    # string could alter the expression.
    cmd = [
        buck2,
        "cquery",
        *invocation_context.to_buck_args(),
        "deps(%s)",
        target,
        "--json",
    ]
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
        return BuckIntrospectionResult(
            entries=[],
            reasons=[
                f"library_introspection: buck2 cquery deps({target}) "
                f"timed out after {timeout}s (raise --buck-timeout if needed)"
            ],
            succeeded=False,
            configured_root_target=None,
        )
    except (FileNotFoundError, OSError) as exc:
        safe_exc = invocation_context.redact_config_overrides(str(exc))
        return BuckIntrospectionResult(
            entries=[],
            reasons=[
                f"library_introspection: buck2 cquery deps({target}) "
                f"failed to launch ({safe_exc})"
            ],
            succeeded=False,
            configured_root_target=None,
        )

    if result.returncode != 0:
        # Common cause: target not found, or buck2 misconfiguration.
        # Surface the stderr (truncated) so the operator can debug
        # without re-running by hand.
        # Redact before truncation so a config token crossing the 300-character
        # boundary cannot leave a partial raw value in env.json.
        stderr = invocation_context.redact_config_overrides((result.stderr or "").strip())[:300]
        return BuckIntrospectionResult(
            entries=[],
            reasons=[
                f"library_introspection: buck2 cquery deps({target}) "
                f"exited {result.returncode} (stderr: {stderr or '(empty)'})"
            ],
            succeeded=False,
            configured_root_target=None,
        )

    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        return BuckIntrospectionResult(
            entries=[],
            reasons=[
                f"library_introspection: buck2 cquery deps({target}) "
                f"returned non-JSON output ({exc})"
            ],
            succeeded=False,
            configured_root_target=None,
        )

    # `buck2 cquery 'deps(...)' --json` returns a flat list of
    # stringified labels (with a parenthesised configuration suffix per
    # entry, stripped by _strip_config_suffix during matching). Older
    # buck2 versions returned a dict mapping the queried target -> dep
    # list; accept both shapes so this code survives a buck2 upgrade
    # that brings back the dict shape.
    dep_labels: list[str] = []
    if isinstance(payload, list):
        dep_labels = [str(x) for x in payload]
    elif isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                dep_labels.extend(str(x) for x in v)
    else:
        return BuckIntrospectionResult(
            entries=[],
            reasons=[
                f"library_introspection: buck2 cquery deps({target}) "
                f"returned unexpected JSON shape ({type(payload).__name__})"
            ],
            succeeded=False,
            configured_root_target=None,
        )

    configured_root_target = _find_configured_root_target(target, dep_labels)

    entries: list[dict] = []
    seen_names: set[str] = set()
    for label in dep_labels:
        match = _match_library(label)
        if match is None:
            continue
        if match in seen_names:
            # Duplicate match: keep the first label (deterministic by
            # `buck2 cquery` topological ordering). The second-match
            # label would be lost; record nothing.
            continue
        seen_names.add(match)
        # Store both the canonical (stripped) Buck label and the raw
        # cquery output. ``target`` is the form that round-trips into
        # another ``buck2 query`` / ``buck2 build`` invocation without
        # the per-run configuration hash changing between probes;
        # ``configured_target`` preserves the configuration suffix
        # (``(prelude//platforms:default#<hash>)``) for forensics when
        # someone is debugging why two probes against the same source
        # tree yielded different configured graphs. Schema 1.6 (#187
        # review): keeping only ``configured_target`` -- as the
        # original a1.2b draft did -- destabilises env.json diffs
        # because the hash changes across buck2 daemon restarts.
        canonical = _strip_config_suffix(label)
        entries.append(
            {
                "name": match,
                "source": "buck",
                "revision": repo_revision,
                "target": canonical,
                "configured_target": label,
            }
        )

    return BuckIntrospectionResult(
        entries=entries,
        reasons=[],
        succeeded=True,
        configured_root_target=configured_root_target,
    )


def _strip_config_suffix(label: str) -> str:
    """Strip the parenthesised configuration suffix buck2 cquery emits."""
    return _CONFIG_SUFFIX_RE.sub("", label).strip()


def _find_configured_root_target(target: str, dep_labels: list[str]) -> str | None:
    """Find the requested target in a configured ``deps(%s)`` closure."""

    requested = _strip_config_suffix(target)
    for label in dep_labels:
        bare = _strip_config_suffix(label)
        if bare == requested:
            return label
        # Cquery commonly qualifies a root-cell ``//pkg:name`` input as
        # ``root//pkg:name`` in configured output.
        if requested.startswith("//") and bare.endswith(requested):
            return label
    return None


def _match_library(label: str) -> str | None:
    """Return the library name a Buck target label resolves to, or None."""
    bare = _strip_config_suffix(label)
    for name, patterns in KNOWN_LIBRARY_PATTERNS.items():
        for pat in patterns:
            if pat.search(bare):
                return name
    return None


__all__ = [
    "BuckIntrospectionResult",
    "KNOWN_LIBRARY_PATTERNS",
    "introspect_libraries_via_buck",
]
