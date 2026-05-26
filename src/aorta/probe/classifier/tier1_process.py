"""Tier 1 process-level failure detectors for ``aorta probe`` (issue #188).

Tier 1 is the only tier that does not need to inspect logs or system
state — every signal is derived from how the subprocess itself
exited:

* ``tier1:exit_nonzero`` -- ``proc.returncode != 0`` and no signal
  fired. Standalone ``exit 1``.
* ``tier1:sigsegv`` -- ``returncode == -signal.SIGSEGV``.
* ``tier1:sigabrt`` -- ``returncode == -signal.SIGABRT``.
* ``tier1:sigbus``  -- ``returncode == -signal.SIGBUS``.
* ``tier1:timeout`` -- the workload was killed via
  ``Popen.wait(timeout=...)`` raising ``TimeoutExpired`` after
  ``recipe.timeout_per_trial`` seconds.
* ``tier1:coredump`` -- any file matching ``core.*`` exists in the
  trial directory post-exit. This only fires when ``/proc/sys/
  kernel/core_pattern`` writes into the trial dir; the dispatcher
  does NOT chdir the workload (the user's ``--`` command often uses
  relative paths and a forced ``cwd`` would silently break it), so
  with the typical relative-path ``core`` / ``core.<pid>`` pattern
  the core file lands in ``aorta probe``'s invocation cwd, not the
  trial dir. Operators who care about coredump detection set an
  absolute ``core_pattern`` that templates the trial dir in.

The classifier consumes a small :class:`Tier1Context` value and
returns an ordered list of detector IDs (in encounter order — the
verdict resolver in :mod:`aorta.probe.classifier.verdict` cares
about order because ``failure_detectors_fired`` is required to
preserve it). Tier 1 is fully unit-testable without spawning a real
subprocess: construct a ``Tier1Context`` with the desired exit code
and trial directory and call :func:`detect`.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from pathlib import Path

# Detector IDs — exported so Tier 4 / 5 cross-references and the
# verdict resolver name them without typos. ID strings are stable
# (part of the public ``result.json`` contract); changing one is a
# breaking change for downstream tooling that parses
# ``failure_detectors_fired``.
DETECTOR_EXIT_NONZERO = "tier1:exit_nonzero"
DETECTOR_SIGSEGV = "tier1:sigsegv"
DETECTOR_SIGABRT = "tier1:sigabrt"
DETECTOR_SIGBUS = "tier1:sigbus"
DETECTOR_TIMEOUT = "tier1:timeout"
DETECTOR_COREDUMP = "tier1:coredump"

# Map ``returncode`` (negative when the process died via signal) to
# the corresponding detector ID. ``signal`` constants are platform-
# dependent on the surface but stable on Linux (the only platform
# aorta probe ships against today).
_SIGNAL_DETECTORS: dict[int, str] = {
    -int(signal.SIGSEGV): DETECTOR_SIGSEGV,
    -int(signal.SIGABRT): DETECTOR_SIGABRT,
    -int(signal.SIGBUS): DETECTOR_SIGBUS,
}


@dataclass(frozen=True)
class Tier1Context:
    """Inputs for :func:`detect`.

    ``trial_dir`` is the per-trial output directory; the coredump
    scan looks for ``core.*`` in this dir only (NOT recursive). The
    dispatcher does NOT pass ``cwd=trial_dir`` to ``Popen`` -- the
    user's ``--`` command is allowed to depend on the launcher's
    cwd, so forcing a new one would break repros that reference
    files by relative path. As a result the detector only sees core
    files when ``/proc/sys/kernel/core_pattern`` is configured to
    write into the trial dir (e.g. an absolute template that
    interpolates the trial path). With the kernel default
    (``core`` next to the process cwd) the detector stays silent
    even on a real segfault.

    ``timed_out`` flips the verdict to ``tier1:timeout`` and
    suppresses the ``exit_nonzero`` detector (a process killed by
    ``proc.kill()`` after a timeout always exits with a non-zero
    code; we want the more informative detector ID to fire alone).
    """

    exit_code: int
    timed_out: bool
    trial_dir: Path


def detect(ctx: Tier1Context) -> list[str]:
    """Return the ordered list of Tier 1 detector IDs that fired.

    Encounter order is fixed to match how a human reading
    ``result.json`` expects to scan the failure list: most specific
    cause first (timeout > signal > exit_nonzero), then ancillary
    artifacts (coredump) regardless of the primary cause.

    Returns ``[]`` when ``exit_code == 0`` and no timeout/coredump
    fired (the trial's Tier 1 status is "pass — exit zero").
    """
    fired: list[str] = []

    if ctx.timed_out:
        fired.append(DETECTOR_TIMEOUT)
    elif ctx.exit_code in _SIGNAL_DETECTORS:
        fired.append(_SIGNAL_DETECTORS[ctx.exit_code])
    elif ctx.exit_code != 0:
        fired.append(DETECTOR_EXIT_NONZERO)

    if _has_coredump(ctx.trial_dir):
        fired.append(DETECTOR_COREDUMP)

    return fired


def _has_coredump(trial_dir: Path) -> bool:
    """Return True iff a ``core.*`` file exists directly under ``trial_dir``.

    Non-recursive. Tier 1 cannot reason about cores from grandchild
    processes; nested subdirectories would also create false
    positives if the workload itself touches a ``core/`` dir. The
    detector relies on the operator's ``core_pattern`` writing into
    the trial dir -- see :class:`Tier1Context` for why the
    dispatcher does not chdir the workload. ``trial_dir.glob``
    returns an iterator; short-circuit on the first match to avoid
    scanning the whole directory for large trials.
    """
    if not trial_dir.is_dir():
        return False
    for _ in trial_dir.glob("core.*"):
        return True
    # Also tolerate the bare ``core`` filename (the default kernel
    # ``core_pattern`` on some distros writes ``core`` with no PID
    # suffix). Cheap second check; keeps the detector accurate
    # across distros.
    return (trial_dir / "core").is_file()


ALL_DETECTOR_IDS = (
    DETECTOR_TIMEOUT,
    DETECTOR_SIGSEGV,
    DETECTOR_SIGABRT,
    DETECTOR_SIGBUS,
    DETECTOR_EXIT_NONZERO,
    DETECTOR_COREDUMP,
)


__all__ = [
    "ALL_DETECTOR_IDS",
    "DETECTOR_COREDUMP",
    "DETECTOR_EXIT_NONZERO",
    "DETECTOR_SIGABRT",
    "DETECTOR_SIGBUS",
    "DETECTOR_SIGSEGV",
    "DETECTOR_TIMEOUT",
    "Tier1Context",
    "detect",
]
