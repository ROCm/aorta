"""Tier 4 built-in pattern library for ``aorta probe`` (issue #188).

A small, versioned set of regex patterns covering the failure modes
the AORTA team has seen often enough to bake into the platform.
Every pattern has a stable detector ID (the value persisted in
``result.json::failure_detectors_fired``); the regexes themselves
may evolve between versions, but the IDs are part of the public
contract.

Version bump policy (rubric §X.1 row 5):

* Adding a pattern: increment :data:`BUILTIN_PATTERN_VERSION`.
* Renaming a pattern's regex (refining recall): increment.
* Removing a pattern: increment, and document the rename of any
  downstream user reading the old ID.

Each pattern ships a sibling fixture log under
``tests/probe/fixtures/tier4_logs/<detector-suffix>.txt`` (``.txt``
keeps the fixtures out of the project's ``.gitignore`` ``*.log``
rule; the basename is the detector ID with the ``tier4:`` prefix
stripped, e.g. ``cuda_error.txt`` for ``tier4:cuda_error``). The
parametrised Tier 4 test reads each fixture and asserts the
corresponding ID fires.

The detectors run AGAINST a 10-MiB-capped window of the trial's
stdout/stderr concatenation (see :data:`MAX_LOG_BYTES` in
:mod:`aorta.probe.sandbox`). A log larger than the cap is scanned
in successive windows that **overlap** by
:data:`_WINDOW_OVERLAP_BYTES` (see :func:`_iter_windows`) so a
multi-line match straddling a seam still fires; each
``re.search`` invocation still touches at most ``MAX_LOG_BYTES``
of input, preserving the catastrophic-backtracking bound that's
the only defence against operator-supplied regex without
swapping the regex engine (rubric §X.2 R2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aorta.probe.sandbox import MAX_LOG_BYTES

# Version string surfaced by ``aorta probe --list-patterns --version``.
# Bumped per the policy above; consumers MAY pin to a known version
# in their CI to detect platform-side pattern changes that would
# alter ``failure_detectors_fired`` for already-archived runs.
BUILTIN_PATTERN_VERSION = "1"

# Detector IDs. The prefix is uniformly ``tier4:`` so a reader of
# ``failure_detectors_fired`` can route by prefix (``tier1:`` for
# process-level, ``tier2:`` for hang, ``tier3:`` for kernel,
# ``tier4:`` for built-in patterns, ``custom:<id>`` for
# user-provided patterns from the recipe's ``custom_patterns:``).
DETECTOR_PYTHON_TRACEBACK = "tier4:python_traceback"
DETECTOR_HIP_ERROR = "tier4:hip_error"
DETECTOR_CUDA_ERROR = "tier4:cuda_error"
DETECTOR_ROCM_ERROR = "tier4:rocm_error"
DETECTOR_NCCL_RCCL_ERROR = "tier4:nccl_rccl_error"
DETECTOR_COLLECTIVE_TIMEOUT = "tier4:collective_timeout"
DETECTOR_NAN_SIGNATURE = "tier4:nan_signature"


@dataclass(frozen=True)
class BuiltinPattern:
    """One entry in the Tier 4 catalogue.

    ``regex`` is a pre-compiled :class:`re.Pattern` so the scanner
    does not re-compile per trial; ``sample`` is a one-line example
    of a log fragment that would match (shown by
    ``aorta probe --list-patterns`` so an operator can recognise
    the shape without reading the regex).
    """

    detector_id: str
    description: str
    regex: re.Pattern[str]
    sample: str


# Pattern order is the order ``aorta probe --list-patterns`` prints
# (rubric §2.B FR 2.5). It is also the order :func:`scan` walks, but
# every match contributes to ``fired`` independently so order is
# UI-only — the returned list reflects the patterns that matched, in
# this catalogue order. The verdict resolver further preserves the
# encounter order it sees, but Tier 4 firing order is deterministic
# because the catalogue is fixed.
_BUILTIN_PATTERNS: tuple[BuiltinPattern, ...] = (
    BuiltinPattern(
        detector_id=DETECTOR_PYTHON_TRACEBACK,
        description="Python traceback ('Traceback (most recent call last):')",
        # MULTILINE so '^' matches at the start of each line; DOTALL
        # NOT set so '.' does not gobble across the traceback frames.
        regex=re.compile(
            r"^Traceback \(most recent call last\):",
            re.MULTILINE,
        ),
        sample="Traceback (most recent call last):",
    ),
    BuiltinPattern(
        detector_id=DETECTOR_HIP_ERROR,
        description="HIP error code (hipError_* or HIP error: ...)",
        # Both shapes seen in the wild: the enum-style identifier
        # printed by ROCm/HIP and the prose 'HIP error:' prefix.
        regex=re.compile(r"hipError_[A-Za-z0-9_]+|HIP error:"),
        sample="hipError_OutOfMemory",
    ),
    BuiltinPattern(
        detector_id=DETECTOR_CUDA_ERROR,
        description="CUDA error code (cudaError_* or CUDA error: ...)",
        regex=re.compile(r"cudaError[A-Za-z0-9_]*|CUDA error:"),
        sample="cudaErrorIllegalAddress",
    ),
    BuiltinPattern(
        detector_id=DETECTOR_ROCM_ERROR,
        description="ROCm error code marker ('Error code: <n>')",
        regex=re.compile(r"Error code: \d+"),
        sample="Error code: 1",
    ),
    BuiltinPattern(
        detector_id=DETECTOR_NCCL_RCCL_ERROR,
        description="NCCL or RCCL collective error",
        regex=re.compile(r"NCCL error|RCCL ERROR"),
        sample="NCCL error: unhandled system error",
    ),
    BuiltinPattern(
        detector_id=DETECTOR_COLLECTIVE_TIMEOUT,
        description="Torch distributed watchdog collective-operation timeout",
        regex=re.compile(r"Watchdog caught collective operation timeout"),
        sample="Watchdog caught collective operation timeout: WorkNCCL ...",
    ),
    BuiltinPattern(
        detector_id=DETECTOR_NAN_SIGNATURE,
        description="Training-loss NaN signature",
        # Case-insensitive 'loss is NaN' / 'loss=nan' / bare 'loss
        # NaN'. Common shapes across PyTorch, TF, JAX trainers.
        regex=re.compile(r"loss(?: is)? NaN|loss=nan", re.IGNORECASE),
        sample="loss is NaN",
    ),
)


def all_patterns() -> tuple[BuiltinPattern, ...]:
    """Return the immutable Tier 4 pattern catalogue."""
    return _BUILTIN_PATTERNS


def scan(log_text: str) -> list[str]:
    """Return Tier 4 detector IDs that fired against ``log_text``.

    Each pattern runs at most once over the input; patterns are
    independent (an HIP error and a Python traceback in the same
    log fire both detectors). The scanner enforces the
    :data:`MAX_LOG_BYTES` per-scan window cap: logs longer than
    the cap are split into successive windows that **overlap by
    :data:`_WINDOW_OVERLAP_BYTES`** so a multi-line match
    straddling a seam still fires (see :func:`_iter_windows`).
    Each pattern runs against each window. This bounds
    catastrophic backtracking on operator-supplied logs without
    changing the regex engine (rubric §X.2 R2). Per Copilot's PR
    #197 review — the doc previously claimed "non-overlapping",
    which has been wrong since the Sonbol-round straddling-match
    fix.

    Returns the IDs in catalogue order — Tier 4 is internally
    deterministic, but ``failure_detectors_fired`` overall preserves
    encounter order via the verdict resolver.
    """
    if not log_text:
        return []
    fired: list[str] = []
    for window in _iter_windows(log_text, MAX_LOG_BYTES):
        for pattern in _BUILTIN_PATTERNS:
            if pattern.detector_id in fired:
                continue
            if pattern.regex.search(window):
                fired.append(pattern.detector_id)
        if len(fired) == len(_BUILTIN_PATTERNS):
            break
    return fired


# Overlap between adjacent windows in :func:`_iter_windows`. Sized to
# comfortably exceed the longest match the Tier 4 catalogue can
# produce -- the widest built-in pattern today matches a Python
# traceback header plus a few continuation lines (a few hundred
# bytes), so 4 KiB is ~10x safety margin. If a future pattern needs
# more, raise this constant; the cost is bounded ``re.search``
# re-work on the overlap region, never a missed match.
_WINDOW_OVERLAP_BYTES = 4096


def _iter_windows(text: str, window: int):
    """Yield ``window``-sized slices of ``text`` with a small overlap.

    Used to bound each ``re.search`` invocation's worst-case input
    size. The previous shape sliced ``text`` into back-to-back
    non-overlapping chunks; a pattern whose match straddled a
    chunk boundary (``Traceback (most recent call last):`` ending
    one window with the body starting the next) was silently
    missed -- the very class of multi-line failure signature the
    Tier 4 detectors are meant to catch on the long-log path.
    Overlap by :data:`_WINDOW_OVERLAP_BYTES` (capped at half the
    window so each step still advances by at least ``window // 2``,
    avoiding an infinite loop when a test or future caller sets a
    pathologically small window). Cost is at most one extra
    overlap-sized scan per seam, bounded by ``2 * len(text) /
    window``. Per Sonbol's PR #197 review.
    """
    if len(text) <= window:
        yield text
        return
    # Cap overlap so step >= window // 2 -- guarantees forward progress.
    overlap = min(_WINDOW_OVERLAP_BYTES, max(window // 2, 0))
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        yield text[start:end]
        if end == len(text):
            return
        start = end - overlap


__all__ = [
    "BUILTIN_PATTERN_VERSION",
    "BuiltinPattern",
    "DETECTOR_COLLECTIVE_TIMEOUT",
    "DETECTOR_CUDA_ERROR",
    "DETECTOR_HIP_ERROR",
    "DETECTOR_NAN_SIGNATURE",
    "DETECTOR_NCCL_RCCL_ERROR",
    "DETECTOR_PYTHON_TRACEBACK",
    "DETECTOR_ROCM_ERROR",
    "all_patterns",
    "scan",
]
