"""Reserved platform-internal workload wrapping an opaque user subprocess.

Wired into the ``aorta.workloads`` entry-point group as the leading-
underscored name ``_subprocess`` so it cannot collide with any user-
facing workload name. Consumed by ``aorta probe`` (issue #188) to wrap
arbitrary launch commands -- ``bash launch.sh``, ``torchrun ...``,
``buck2 run //path:target -- ...``, ``docker run ...`` -- without
parsing or modifying the user's argv.

The argv is delivered as a reserved config key, ``_aorta_subprocess_argv``,
injected by :class:`aorta.run.dispatcher.RunRequest` after
``config_overrides`` is merged. The reserved-prefix block at the top of
``run_trials`` rejects any user-supplied ``_aorta_*`` key, so users
cannot smuggle an argv via ``config_overrides`` -- the typed
``RunRequest.subprocess_argv`` field is the only legal channel.

Per-trial output layout (Phase 1):

* ``<cell_dir>/trial_<N>/stdout.log`` -- captured stdout written as
  RAW BYTES (the file handle is opened in binary ``"wb"`` mode so the
  child process's output lands on disk byte-for-byte; no decode,
  encoding-error handling, or line buffering is performed in the
  parent). Downstream readers should treat the file as bytes and
  decode lazily.
* ``<cell_dir>/trial_<N>/stderr.log`` -- captured stderr, same raw-
  bytes contract as stdout.log.
* ``<cell_dir>/trial_<N>/result.json`` -- Tier-1 verdict + metadata.
* ``<cell_dir>/trial_<N>/probe.env`` -- only when
  ``env_passthrough_mode == "file"`` (chmod 0600).

Verdict (Phase 2): the five-tier classifier in
:mod:`aorta.probe.classifier` runs post-exit. A trial whose process
exit is non-zero, timed out, hung, hit a dmesg / amd-smi signal, hit
a built-in Tier-4 pattern, or hit a user ``custom_patterns`` entry
with ``on_match: fail`` resolves to ``"fail"``; otherwise ``"pass"``.
Phase 1's exit-code-only rule remains a strict subset (``exit_code
== 0 and no detector fires`` -> pass), so any tooling that read the
Phase-1 minimum shape keeps working.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from aorta.probe.classifier import TrialContext, classify_trial
from aorta.probe.classifier.tier1_process import (
    DETECTOR_EXIT_NONZERO,
    Tier1Context,
)
from aorta.probe.classifier.tier1_process import detect as tier1_detect
from aorta.probe.classifier.tier2_hang import (
    DEFAULT_HANG_GRACE_SEC,
    DEFAULT_HANG_WINDOW_SEC,
    HangMonitor,
)
from aorta.probe.classifier.tier3_kernel import (
    AmdSmiSnapshot,
    Tier3State,
    gpu_idle_probe_from_state,
    poll_amd_smi,
    scan_dmesg,
)
from aorta.probe.classifier.verdict import (
    Verdict,
    partition_detectors,
    verdict_from_detectors,
)
from aorta.run.retention import RetentionOutcome, apply_retention
from aorta.workloads._base import Workload, WorkloadResult

log = logging.getLogger(__name__)

# Process-wide Tier 3 state. Shared across every SubprocessWorkload
# instance the dispatcher constructs over one ``aorta probe`` invocation
# so the rubric's "tier3 disabled: <reason>" warning is logged at most
# ONCE per invocation (FR 2.11), regardless of how many cells x trials
# the matrix produces. ``Tier3State`` is mutable and the dispatcher's
# deep-copy semantics for ``probe_extras`` would defeat that guarantee
# if we tried to plumb it through the config dict -- a module-level
# singleton is the smallest correct alternative and lives only for the
# lifetime of the ``aorta probe`` process (probe-mode is single-process
# by design; the rubric forbids subprocess-launched workloads).
_TIER3_STATE = Tier3State()

# Pad added to the dmesg ``--since`` window to cover the small wall-
# clock drift between ``time.perf_counter`` and the kernel's monotonic
# clock and to catch messages logged a few seconds after the child
# crashed (e.g. amdgpu reset messages often arrive on the next tick).
_DMESG_SINCE_PAD_SEC = 5.0

# Config key the dispatcher uses to deliver the opaque user argv. The
# leading ``_aorta_`` prefix is reserved by the dispatcher and rejected
# in ``config_overrides`` -- the only legal producer is
# :class:`aorta.run.dispatcher.RunRequest.subprocess_argv`.
CONFIG_KEY_SUBPROCESS_ARGV = "_aorta_subprocess_argv"

# Config keys the dispatcher injects when ``RunRequest.save_logs`` is
# True. ``_aorta_log_prefix`` is the only reliable in-process channel
# carrying the per-trial coordinate (the ``_t<N>`` suffix) and is what
# this workload uses to compute its per-trial output directory.
CONFIG_KEY_LOG_PREFIX = "_aorta_log_prefix"

# Probe-extras-derived config keys. The runner attaches these to the
# per-cell request config_overrides BEFORE the dispatcher's
# reserved-prefix injection so the workload can pick them up.
CONFIG_KEY_PROBE_EXTRAS = "_aorta_probe_extras"

# Match the ``trial_d<d>_m<m>_t<idx>`` suffix the dispatcher uses for
# ``_aorta_log_prefix``. Captured group is the trial index; we use it
# to compute ``trial_<idx>/`` per the probe-mode artifact layout.
_LOG_PREFIX_TRIAL_RE = re.compile(r"trial_d\d+_m\d+_t(\d+)$")


# Grace period (seconds) between the SIGTERM and the SIGKILL escalation in
# ``_terminate_process_tree``. Short enough that an interrupted operator isn't
# left waiting, long enough that a foreground ``docker run`` can forward the
# SIGTERM to the container's PID 1 and let it stop cleanly.
_TERMINATE_GRACE_SEC = 10.0

# Reap budget (seconds) for the ``proc.wait()`` *after* a SIGKILL has gone out.
# SIGKILL is uncatchable, so the group is torn down by the kernel almost
# immediately; this wait only exists to reap zombies, not to give the child a
# chance to react. Capping it (rather than reusing the full SIGTERM grace) keeps
# Ctrl-C aborts and timeout handling responsive instead of stalling up to
# ``grace_sec`` on a process that is already dead.
_REAP_AFTER_KILL_SEC = 1.0


def _terminate_process_tree(
    proc: subprocess.Popen, grace_sec: float = _TERMINATE_GRACE_SEC
) -> None:
    """Best-effort teardown of the child's *entire* process group.

    The child is launched with ``start_new_session=True`` so it leads its own
    process group; signalling that group (``os.killpg``) rather than just
    ``proc.pid`` reaps grandchildren too -- e.g. a
    ``sudo -> bash -> docker run -> python3`` tree that would otherwise survive
    an interrupted or timed-out trial and keep its GPUs pinned (#220).

    Escalation: ``SIGTERM`` first (a foreground ``docker run`` forwards it to
    the container's PID 1, giving the container a chance to stop cleanly), then
    ``SIGKILL`` after ``grace_sec`` for anything that ignored it. Note that a
    *detached* ``docker run -d`` container is reparented to the docker daemon
    and is not in this group; tearing it down needs an explicit ``docker kill``
    in the workload wrapper.

    A race where the group already exited (``ESRCH`` /
    ``ProcessLookupError``) is the expected common case, not an error, and is
    swallowed silently. Any *other* signal-delivery failure (e.g. ``EPERM``)
    is logged at WARNING -- it means the group could not be torn down and the
    tree may leak, which the operator needs to see. As a safety net it refuses
    to signal the caller's own process group -- which can only happen if the
    child was not started in a new session -- so a misconfiguration can never
    take down the ``aorta`` process itself.

    The only exception it propagates is ``KeyboardInterrupt``: a second
    ``Ctrl-C`` landing while we wait out the grace period must not abandon the
    teardown mid-escalation and re-orphan the tree (#220). In that case we
    force ``SIGKILL`` on the group, reap best-effort, and *then* re-raise so
    the operator's interrupt still aborts the run.
    """
    pgid: int | None
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None
    if pgid is not None and pgid == os.getpgrp():
        # Defensive: child not in its own session (should never happen in
        # production). Fall back to signalling just the direct child so we
        # never nuke our own process group.
        pgid = None

    def _send(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                # Expected ESRCH race: the group exited between the decision
                # to signal and delivery. Not an error -- fall through to the
                # direct-child attempt (also a likely ESRCH no-op).
                pass
            except OSError as exc:
                # Non-ESRCH failure (e.g. EPERM): the group could NOT be
                # signalled, so the child tree may survive and keep its GPUs
                # pinned (#220). Surface it instead of leaking silently, then
                # still try the direct child as a best-effort fallback.
                log.warning(
                    "killpg(%d, %s) failed: %s; the child process group may "
                    "not be fully reaped (orphaned grandchildren can keep "
                    "their GPUs pinned, #220)",
                    pgid,
                    signal.Signals(sig).name,
                    exc,
                )
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass
        except OSError as exc:
            log.warning(
                "send_signal(%s) to child pid %s failed: %s; the child may "
                "not be reaped (#220)",
                signal.Signals(sig).name,
                getattr(proc, "pid", "?"),
                exc,
            )

    reap_sec = min(grace_sec, _REAP_AFTER_KILL_SEC)

    def _reap_after_kill() -> None:
        # The group has already been SIGKILLed; give it a brief, fully
        # interrupt-tolerant chance to be reaped so we don't leave zombies.
        try:
            proc.wait(timeout=reap_sec)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            pass
        except KeyboardInterrupt:
            pass

    _send(signal.SIGTERM)
    try:
        proc.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    except (ProcessLookupError, OSError):
        return
    except KeyboardInterrupt:
        # Don't abandon teardown mid-escalation: force-kill the group and
        # reap before propagating the operator's interrupt.
        _send(signal.SIGKILL)
        _reap_after_kill()
        raise
    _send(signal.SIGKILL)
    try:
        proc.wait(timeout=reap_sec)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        pass
    except KeyboardInterrupt:
        # Group already got SIGKILL; reap what we can, then propagate.
        _reap_after_kill()
        raise


class SubprocessWorkload(Workload):
    """Workload that forks the user's opaque launch command.

    Distinct from triage-mode workloads in three ways:

    1. The "config" it cares about is delivered via reserved
       ``_aorta_*`` keys (argv, log prefix, probe-extras) rather than
       user ``config_overrides``. The class is platform-internal and
       only ``aorta probe`` wires it up.
    2. Per-trial artifacts land in the ``flat_resume`` layout
       (``<cell_dir>/trial_<N>/...``) rather than the dispatcher's
       default ``<results_dir>/<workload>/trial_d<d>_m<m>_t<n>.json``.
       The dispatcher's per-trial JSON is still written -- as a sibling
       under ``_subprocess/`` -- and serves as the "I-ran" marker for
       triage-mode tooling; the probe-mode ``result.json`` is the one
       resume / classifier code consults.
    3. Verdict is the union of Tier 1-5 detectors (Phase 2). Phase 1's
       ``exit_code == 0 -> pass`` rule is a strict subset: a trial
       that exits zero AND fires no other tier still resolves to
       ``"pass"``. See :mod:`aorta.probe.classifier`.
    """

    launch_mode = "single_process"
    min_world_size = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._argv: tuple[str, ...] | None = None
        self._trial_dir: Path | None = None
        self._trial_index: int | None = None

    def setup(self) -> None:
        """Resolve argv + trial directory from the reserved config keys."""
        argv = self.config.get(CONFIG_KEY_SUBPROCESS_ARGV)
        if argv is None:
            raise RuntimeError(
                "SubprocessWorkload requires the platform-supplied "
                f"{CONFIG_KEY_SUBPROCESS_ARGV!r} config key. This workload is "
                "wired internally by 'aorta probe'; do not invoke it "
                "directly via 'aorta run'."
            )
        if not isinstance(argv, list) or not argv:
            raise RuntimeError(
                f"{CONFIG_KEY_SUBPROCESS_ARGV} must be a non-empty list[str], "
                f"got {type(argv).__name__} ({argv!r})"
            )
        if not all(isinstance(a, str) for a in argv):
            raise RuntimeError(
                f"{CONFIG_KEY_SUBPROCESS_ARGV} entries must be str, got "
                f"{[type(a).__name__ for a in argv]}"
            )
        self._argv = tuple(argv)

        log_prefix = self.config.get(CONFIG_KEY_LOG_PREFIX)
        if not isinstance(log_prefix, str) or not log_prefix:
            raise RuntimeError(
                "SubprocessWorkload requires the platform-supplied "
                f"{CONFIG_KEY_LOG_PREFIX!r} config key. The dispatcher "
                "only injects it on the rank-0 ('should_write') path when "
                "save_logs=True. The most likely root causes are: (1) "
                "the workload was invoked under a launcher with RANK!=0 "
                "(probe-mode is single-rank by design; multi-rank wrapping "
                "is not supported in Phase 1), (2) the runner forgot to "
                "set save_logs=True (this is a runner bug if the cell is "
                "probe-mode), or (3) the workload was invoked outside "
                "'aorta probe' altogether (not supported -- this workload "
                "is platform-internal)."
            )
        match = _LOG_PREFIX_TRIAL_RE.search(Path(log_prefix).name)
        if match is None:
            raise RuntimeError(
                f"{CONFIG_KEY_LOG_PREFIX} {log_prefix!r} does not match the "
                "documented 'trial_d<d>_m<m>_t<idx>' shape; cannot derive "
                "per-trial directory."
            )
        self._trial_index = int(match.group(1))
        # ``Path(log_prefix).parent`` is ``<results_dir>/<workload>/``
        # (the dispatcher's per-workload subdir); the probe-mode cell
        # dir is its parent. trial_<N>/ lives directly under the cell
        # dir so the artifact tree matches the rubric's
        # ``<cell>/trial_<n>/{stdout,stderr,result}`` layout.
        cell_dir = Path(log_prefix).parent.parent
        self._trial_dir = cell_dir / f"trial_{self._trial_index}"
        self._trial_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> WorkloadResult:
        """Fork the user command and write the Phase-2 ``result.json``.

        Phase 2 hangs the five-tier classifier off this method
        post-exit. The Tier 1 verdict (Phase 1 contract) is a
        subset of the Phase 2 verdict — a trial that exits 0 with
        no Tier 2/3/4/5 detector firing still resolves to
        ``verdict = "pass"``, matching Phase 1 byte-for-byte. The
        Phase 1 minimum-shape test in
        ``tests/probe/test_subprocess_workload.py`` continues to
        pass without modification because Phase 2 only ADDS keys
        to ``result.json``.
        """
        if self._argv is None or self._trial_dir is None or self._trial_index is None:
            raise RuntimeError("SubprocessWorkload.run() called before setup()")

        argv = self._argv
        trial_dir = self._trial_dir
        stdout_path = trial_dir / "stdout.log"
        stderr_path = trial_dir / "stderr.log"
        result_path = trial_dir / "result.json"

        probe_extras = self.config.get(CONFIG_KEY_PROBE_EXTRAS) or {}
        env_mode = probe_extras.get("env_passthrough_mode", "inherit")
        timeout = probe_extras.get("timeout_per_trial")
        custom_patterns = tuple(probe_extras.get("custom_patterns") or ())
        # ``... or DEFAULT`` collapses a recipe-configured ``0.0`` (a
        # legitimate "disable grace" / "disable window" value validated
        # by the recipe-builder) into the default. Use explicit ``is
        # None`` so an opt-in zero survives the runtime extraction.
        _hang_window_raw = probe_extras.get("hang_window_sec")
        hang_window_sec = (
            DEFAULT_HANG_WINDOW_SEC if _hang_window_raw is None else float(_hang_window_raw)
        )
        _hang_grace_raw = probe_extras.get("hang_grace_period_at_start")
        hang_grace_sec = (
            DEFAULT_HANG_GRACE_SEC if _hang_grace_raw is None else float(_hang_grace_raw)
        )
        # ``tier3_vram_growth`` is a validated boolean on the
        # recipe-builder side (RecipeSchemaError on non-bool) and is
        # serialized from a typed ``ProbeExtras`` field, so the payload
        # is contractually a bool. Validate-and-fail-fast rather than
        # ``bool(...)``: truthiness coercion would turn a malformed
        # payload (e.g. the string ``"false"``) into ``True`` and
        # silently re-enable the detector. This mirrors the sibling
        # ``float(...)`` knobs above, which also surface a bad payload
        # instead of swallowing it.
        _vram_growth_raw = probe_extras.get("tier3_vram_growth", True)
        if not isinstance(_vram_growth_raw, bool):
            raise TypeError(
                "probe_extras['tier3_vram_growth'] must be a bool, got "
                f"{type(_vram_growth_raw).__name__} ({_vram_growth_raw!r})"
            )
        tier3_vram_growth = _vram_growth_raw

        # Issue #229: detector-disable knobs. Validated on the
        # recipe-builder side; default to empty so non-probe / legacy
        # payloads are the no-op (every detector active).
        disabled_detectors = _coerce_disable_tokens(
            probe_extras.get("disable_detectors"), "disable_detectors"
        )
        disabled_tiers = _coerce_disable_tokens(
            probe_extras.get("disable_detector_tiers"), "disable_detector_tiers"
        )
        # Disabling a whole tier means "not evaluated at all" -- for Tier 3
        # that includes the side-effecting amd-smi / dmesg probes, not just
        # their contribution to the verdict. Gate the collection itself so
        # the knob actually avoids the probe overhead + permission noise.
        tier3_enabled = "tier3" not in disabled_tiers

        # ``inherit`` mode: the dispatcher has already stamped the
        # cell's mitigation + diagnostic env vars onto os.environ in
        # _run_single_trial's pre-run overlay (it restores them in the
        # finally block after run()). We pass a snapshot to Popen so
        # the child inherits exactly what the parent had at fork.
        #
        # ``file`` mode: ALSO write a 0600 KEY=VALUE\n env file in the
        # trial dir and export AORTA_ENV_FILE so the user's argv can
        # reference it (``docker run --env-file $AORTA_ENV_FILE ...``).
        # See F6 in the rubric for the no-parse-argv rationale.
        cell_env_snapshot = self._capture_cell_env(probe_extras)
        child_env = os.environ.copy()
        env_file_path = trial_dir / "probe.env"
        if env_mode == "file":
            try:
                _write_env_file(env_file_path, cell_env_snapshot)
            except ValueError as exc:
                # ``_write_env_file`` rejects hostile/malformed mitigation
                # keys/values (newlines, '=' in key, etc.) by raising
                # ``ValueError``. Without this handler the exception would
                # escape ``run()`` and the dispatcher would record an
                # ``infrastructure_failed`` TrialResult -- but with NO
                # per-trial ``result.json``. That breaks two contracts at
                # once:
                #
                # 1. The artifact-tree contract (every probe trial leaves
                #    ``trial_<n>/result.json``).
                # 2. ``flat_resume``: ``is_trial_complete`` keys off the
                #    presence of ``result.json``, so a missing file makes
                #    every subsequent ``aorta probe`` invocation re-run
                #    the same broken cell forever.
                #
                # Treat it like the exec-time ``Popen`` failures below
                # (FileNotFoundError, PermissionError): synthesize a
                # Tier-1 ``fail`` ``result.json``, write the error message
                # to ``stderr.log``, and return a ``WorkloadResult`` with
                # ``launched=False`` / ``main_work_started=False`` so the
                # matrix classifier doesn't conflate this with a real
                # subprocess that exited non-zero.
                return self._write_env_file_failure_result(
                    exc=exc,
                    result_path=result_path,
                    stderr_path=stderr_path,
                    env_file_path=env_file_path,
                    argv=argv,
                    probe_extras=probe_extras,
                    env_mode=env_mode,
                    cell_env_snapshot=cell_env_snapshot,
                )
            child_env["AORTA_ENV_FILE"] = str(env_file_path.absolute())
        else:
            # ``inherit`` mode: scrub any probe.env left over from a
            # prior run that used ``file`` mode (resume scenarios under
            # ``flat_resume`` re-use the same ``trial_<n>/`` directory
            # when a previous attempt's ``result.json`` was truncated).
            # Without this cleanup the on-disk artifacts would
            # contradict the current invocation -- the operator would
            # see a ``probe.env`` even though the child never had
            # ``AORTA_ENV_FILE`` exported, which is at best confusing
            # and at worst points downstream tooling at stale env
            # contents.
            env_file_path.unlink(missing_ok=True)

        # Tier 3 pre-snapshot. Fail-soft: returns None when ``amd-smi``
        # is missing or polling fails; ``scan_amd_smi`` then accepts
        # ``None`` and contributes nothing without aborting the trial.
        # The shared ``_TIER3_STATE`` ensures the "amd-smi disabled"
        # log fires at most once across the full probe invocation.
        # Skipped entirely when Tier 3 is operator-disabled.
        amd_smi_pre = poll_amd_smi(_TIER3_STATE) if tier3_enabled else None

        t0 = time.perf_counter()
        exit_code: int
        timed_out = False
        # ``launched`` distinguishes "subprocess ran (maybe poorly)"
        # from "subprocess never started" (exec-time Popen failure --
        # ENOENT / EACCES / ENOEXEC). The artifact-tree contract from
        # PR #194 round 4 still applies (``result.json`` is written
        # either way), but the matrix outcome classifier needs the
        # signal to avoid counting a command-not-found as a completed
        # 1/1 trial. See the ``return WorkloadResult(...)`` block
        # below for the propagation into ``main_work_started`` /
        # ``executed_iterations``.
        launched = False
        hang_monitor: HangMonitor | None = None
        try:
            with open(stdout_path, "wb") as out_fh, open(stderr_path, "wb") as err_fh:
                proc = subprocess.Popen(
                    list(argv),
                    stdout=out_fh,
                    stderr=err_fh,
                    env=child_env,
                    # Lead a new session/process group so the *whole* child
                    # tree (sudo -> bash -> docker run -> python3, etc.) can
                    # be reaped via os.killpg on timeout / interrupt instead
                    # of orphaning grandchildren that keep their GPUs pinned
                    # (#220). See _terminate_process_tree.
                    start_new_session=True,
                )
                launched = True
                # Everything from the monitor wiring through proc.wait() is in
                # one try so a Ctrl-C landing *anywhere* after Popen -- e.g.
                # during HangMonitor.start() -- still reaches the
                # ``except KeyboardInterrupt`` that reaps the child tree, and
                # the ``finally`` always stops the monitor thread. Otherwise an
                # interrupt between Popen and the wait would orphan the
                # sudo/docker/python grandchildren this PR is meant to reap
                # (#220) and leak the monitor thread.
                try:
                    # Wire the third leg of the two-of-three Tier 2
                    # predicate. The closure spawns one ``amd-smi monitor``
                    # call per HangMonitor poll (~12/min at the default 5s
                    # poll cadence) and returns True iff the busiest GPU
                    # reports < GPU_IDLE_UTILIZATION_THRESHOLD_PCT. When
                    # amd-smi is missing or unparseable the closure returns
                    # False, so the predicate gracefully degrades to the
                    # 2-of-2 ``stdout_silent`` + ``io_idle`` shape that the
                    # round-1 wiring was already covering (rubric §2.B FR
                    # 2.11 fail-soft policy).
                    hang_monitor = HangMonitor(
                        pid=proc.pid,
                        stdout_path=stdout_path,
                        hang_window_sec=hang_window_sec,
                        hang_grace_period_at_start=hang_grace_sec,
                        gpu_idle_probe=gpu_idle_probe_from_state(_TIER3_STATE),
                    )
                    hang_monitor.start()
                    exit_code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    # Tear down the whole child process group, not just the
                    # direct child: a timed-out trial may have spawned a
                    # sudo/docker/python grandchild tree that would otherwise
                    # survive and keep its GPUs pinned (#220).
                    # ``_terminate_process_tree`` is race-safe -- it swallows
                    # the ESRCH that occurs when the child exits between the
                    # timeout firing and the signal landing -- so the trial
                    # deterministically records ``timed_out=True`` and
                    # ``exit_code=-1`` with a ``result.json`` on disk.
                    _terminate_process_tree(proc)
                    exit_code = -1
                except KeyboardInterrupt:
                    # Operator Ctrl-C (SIGINT delivered to the aorta parent;
                    # the child is in its own session and does NOT receive the
                    # terminal's SIGINT). Reap the whole child tree before the
                    # interrupt unwinds the run, otherwise sudo/docker/python
                    # grandchildren survive and exhaust the GPUs (#220), then
                    # re-raise so the run aborts as the operator intended. We
                    # do NOT set ``timed_out`` here: an operator abort isn't a
                    # timeout, and the re-raise skips ``result.json`` anyway.
                    _terminate_process_tree(proc)
                    raise
                finally:
                    if hang_monitor is not None:
                        hang_monitor.stop()
        except (FileNotFoundError, PermissionError, OSError) as exc:
            # Exec-time ``Popen`` failures all become Tier-1 fails with
            # the artifact tree intact (stderr.log + result.json). The
            # error families we capture:
            #
            # * ``FileNotFoundError`` (ENOENT, exit 127) -- argv[0]
            #   doesn't resolve to a binary on $PATH;
            # * ``PermissionError`` (EACCES, exit 126) -- argv[0]
            #   exists but isn't executable (chmod missing the +x bit);
            # * other ``OSError`` (e.g. ENOEXEC "Exec format error"
            #   for a shebang-less script, ELOOP for a symlink loop) --
            #   the operator gets the diagnostic via stderr.log but
            #   the trial still occupies its slot in the matrix.
            #
            # Without this guard a non-executable user command would
            # escape to the dispatcher as ``infrastructure_failed``,
            # leaving the per-trial directory without a
            # ``result.json`` and breaking the documented "every probe
            # trial leaves an artifact" contract. ``FileNotFoundError``
            # and ``PermissionError`` are ``OSError`` subclasses so the
            # three-way tuple is logically a single OSError catch; we
            # name the subclasses explicitly so the exit-code mapping
            # below stays grep-able.
            if isinstance(exc, FileNotFoundError):
                exit_code = 127
            elif isinstance(exc, PermissionError):
                exit_code = 126
            else:
                exit_code = 1
            try:
                stderr_path.write_text(f"{exc}\n", encoding="utf-8")
            except OSError:
                pass
        walltime_sec = time.perf_counter() - t0

        # Tier 2-5 classifier post-exit. Reads the captured logs
        # back from disk -- the file handles above are closed by
        # the ``with`` block. Errors here MUST NOT propagate (the
        # workload already succeeded or failed; classifier crashes
        # are bugs, not trial outcomes).
        log_text = _read_log_text(stdout_path, stderr_path)
        hang_detected = bool(hang_monitor and hang_monitor.hang_detected)
        # Reconcile the latched hang flag against the actual exit.
        #
        # The in-flight HangMonitor watches three signals on the
        # *wrapper* PID: stdout silence on this trial's stdout.log, I/O
        # idleness on /proc/<wrapper_pid>/io, and GPU idleness. For a
        # workload that delegates its real work to a child process tree
        # the parent can't see (``sudo`` -> ``bash run.sh`` ->
        # ``docker run`` -> container -> ``python3``), the wrapper goes
        # quiet and does almost no I/O of its own while the descendants
        # do all the work. That trips ``stdout_silent`` + ``io_idle``
        # (two of three) and latches ``hang_detected`` even though the
        # workload is alive and busy. Those two legs are structurally
        # blind to delegated work, so a mid-run latch is advisory only.
        #
        # A process that voluntarily exited 0 within its timeout was, by
        # definition, not hung -- a real hang would have either kept the
        # process alive until ``proc.wait(timeout=...)`` fired
        # (``timed_out=True``) or surfaced as a non-zero exit. Drop the
        # latched flag in that case so a clean run is never classified
        # ``tier2:hang``. Genuine hangs (``timed_out=True``) and crashes
        # (``exit_code != 0``) keep the flag and still classify as fail.
        hang_reconciled_away = False
        if hang_detected and exit_code == 0 and not timed_out:
            log.info(
                "tier2:hang latched mid-run for argv[0]=%r but the process "
                "exited 0 within the timeout; discarding the advisory flag "
                "(the monitor's stdout/io legs are blind to work delegated "
                "to a child process tree).",
                argv[0] if argv else "<unknown>",
            )
            hang_detected = False
            hang_reconciled_away = True

        # Best-effort ``peak_vram_mib`` from the Tier-3 amd-smi
        # snapshots. We only have two samples (pre + post Popen) so
        # this is a coarse high-water-mark, not a true peak -- a
        # short-lived spike in the middle of the trial is invisible
        # to a 2-point sampler. We surface it anyway because the
        # alternative is leaving the field permanently ``None`` and
        # rendering Tier-5 sandbox conditions like
        # ``peak_vram_mib > 70000`` unusable on real hosts, which is
        # the bot-flagged gap. Both snapshots are fail-soft (either
        # can be ``None`` when amd-smi is missing / unparseable) --
        # if neither is available we fall back to ``None`` and the
        # sandbox's existing ``peak_vram_mib is None -> 0`` shim in
        # :func:`aorta.probe.sandbox.build_sandbox_env` keeps
        # conditions deterministic.

        # Tier 3 post-snapshot + dmesg scan. ``since_seconds`` covers
        # the trial walltime plus a small pad so kernel messages
        # logged shortly after the child crashed (amdgpu reset etc.)
        # still land in the window. Both helpers are fail-soft and
        # share ``_TIER3_STATE`` with the pre-snapshot above so the
        # one-warning-per-invocation contract holds.
        #
        # Skipped when the subprocess never launched (exec-time Popen
        # failure -- ENOENT / EACCES / ENOEXEC): a command-not-found
        # trial did no GPU work, so the post-snapshot + dmesg scan add
        # only overhead and permission noise, and any amdgpu/kernel
        # message in the window belongs to an unrelated process -- not
        # this trial. Gating on ``launched`` keeps those events from
        # being misattributed to a trial that never ran. (Copilot review)
        amd_smi_post: AmdSmiSnapshot | None = None
        dmesg_text: str | None = None
        fired_kernel_ids: list[str] = []
        if tier3_enabled and launched:
            amd_smi_post = poll_amd_smi(_TIER3_STATE)
            try:
                fired_kernel_ids = scan_dmesg(
                    _TIER3_STATE,
                    since_seconds=walltime_sec + _DMESG_SINCE_PAD_SEC,
                )
                # ``scan_dmesg`` returns the fired detector IDs directly;
                # rebuild a synthetic text blob so the classifier's
                # ``scan_dmesg_text`` second pass is a no-op (it would
                # otherwise re-scan ``None`` and emit []). The empty
                # string here keeps Tier 3's text path inert; the
                # already-fired IDs are surfaced via ``tier3_extra``.
                dmesg_text = "" if fired_kernel_ids else None
            except Exception:
                fired_kernel_ids = []
                dmesg_text = None

        peak_vram_mib: int | None
        if amd_smi_pre is not None and amd_smi_post is not None:
            peak_vram_mib = max(amd_smi_pre.vram_used_mib, amd_smi_post.vram_used_mib)
        elif amd_smi_pre is not None:
            peak_vram_mib = amd_smi_pre.vram_used_mib
        elif amd_smi_post is not None:
            peak_vram_mib = amd_smi_post.vram_used_mib
        else:
            peak_vram_mib = None

        # Classifier crash containment (rubric §2.B FR 2.11 fail-soft
        # policy applied to the classifier itself). The trial has
        # already run end-to-end -- we have its exit_code,
        # walltime_sec, captured logs, and Tier 1 inputs all in hand.
        # If a tier classifier raises (regex catastrophe, schema
        # surprise from a future refactor, anything), we MUST still
        # write a ``result.json`` so the trial doesn't silently
        # disappear from the matrix. Fall back to a Tier-1-only
        # verdict derived from the same Tier 1 inputs, record the
        # classifier exception under ``capture['classifier_error']``
        # for the operator, and continue.
        # A subprocess that never launched (exec-time Popen failure --
        # ENOENT / EACCES / ENOEXEC) has only Tier 1 to speak to its
        # outcome. Honouring a Tier-1 disable on that path would let a
        # command-not-found resolve to a green verdict -- a run that did
        # no real work yet looks like a pass. Force Tier 1 (and its
        # ``exit_nonzero`` detector) back on for the exec-failure path so
        # the failure always surfaces; the disable still applies on every
        # launched trial. (oyazdanb review)
        effective_disabled_tiers = disabled_tiers
        effective_disabled_detectors = disabled_detectors
        if not launched:
            effective_disabled_tiers = disabled_tiers - {"tier1"}
            effective_disabled_detectors = disabled_detectors - {DETECTOR_EXIT_NONZERO}
        try:
            verdict_obj, tier_durations_ms = classify_trial(
                TrialContext(
                    exit_code=exit_code,
                    timed_out=timed_out,
                    walltime_sec=walltime_sec,
                    trial_dir=trial_dir,
                    log_text=log_text,
                    custom_patterns=custom_patterns,
                    hang_detected=hang_detected,
                    exec_failed=not launched,
                    peak_vram_mib=peak_vram_mib,
                    dmesg_text=dmesg_text,
                    amd_smi_pre=amd_smi_pre,
                    amd_smi_post=amd_smi_post,
                    tier3_extra=tuple(fired_kernel_ids),
                    tier3_state=_TIER3_STATE,
                    tier3_vram_growth=tier3_vram_growth,
                    disabled_tiers=effective_disabled_tiers,
                    disabled_detectors=effective_disabled_detectors,
                )
            )
        except Exception as classifier_exc:  # noqa: BLE001 -- classifier crash containment
            verdict_obj, tier_durations_ms = _tier1_only_fallback_verdict(
                exit_code=exit_code,
                timed_out=timed_out,
                trial_dir=trial_dir,
                exec_failed=not launched,
                classifier_exc=classifier_exc,
                disabled_tiers=effective_disabled_tiers,
                disabled_detectors=effective_disabled_detectors,
            )

        result_doc: dict[str, Any] = {
            "verdict": verdict_obj.verdict,
            "exit_code": exit_code,
            "walltime_sec": walltime_sec,
            "peak_vram_mib": peak_vram_mib,
            "argv": list(argv),
            "cell_name": probe_extras.get("cell_name", "_unknown_"),
            "trial_index": self._trial_index,
            "failure_detectors_fired": list(verdict_obj.failure_detectors_fired),
            # Issue #230: infra-error signals (timeout-without-hang,
            # exec-failed) kept separate from genuine failures so the
            # matrix can exclude them from the event-rate denominator.
            "error_detectors_fired": list(verdict_obj.error_detectors_fired),
            "warn_detectors_fired": list(verdict_obj.warn_detectors_fired),
            "capture": dict(verdict_obj.capture),
            "tier_durations_ms": dict(tier_durations_ms),
            # Phase 1 keys preserved for back-compat with any
            # downstream tool that already parses them. The Phase 1
            # minimum-shape test in tests/probe/test_subprocess_workload.py
            # asserts these continue to exist; the Phase 2 shape
            # extends the doc by ADDING keys, never by removing
            # them (rubric §2.B FR 2.9).
            "env_passthrough_mode": env_mode,
            "timed_out": timed_out,
            "env": dict(cell_env_snapshot),
        }
        # Durable per-trial breadcrumb when a latched tier2:hang was
        # reconciled away on a clean exit (exit 0, not timed out). Only the
        # tier2:hang signal is dropped -- the final verdict still reflects the
        # other tiers, so a clean exit can still fail on Tier-3/4/5. The #224
        # follow-ups (descendant-tree-aware hang predicate) need to study these
        # wrapper-delegated false positives, so leave a trace in result.json
        # rather than only the log.info above.
        if hang_reconciled_away:
            result_doc["capture"]["tier2_hang_latched_but_reconciled"] = True
        # Issue #229: surface the operator's disable knobs in the trial
        # artifact so a reader knows a detector was intentionally silenced
        # rather than simply not firing. Record the *effective* set the
        # classifier actually honoured -- on the exec-failure path Tier 1
        # is forced back on, so echoing the requested set here would make
        # the artifact self-contradictory (claim ``tier1`` disabled while
        # ``tier1:exit_nonzero`` shows as fired). (Copilot review)
        if effective_disabled_detectors:
            result_doc["capture"]["disabled_detectors"] = sorted(
                effective_disabled_detectors
            )
        if effective_disabled_tiers:
            result_doc["capture"]["disabled_detector_tiers"] = sorted(
                effective_disabled_tiers
            )
        result_path.write_text(
            json.dumps(result_doc, indent=2, sort_keys=False),
            encoding="utf-8",
        )

        # Issue #231: prune heavy per-trial artifacts now that the verdict
        # is known, keeping the level mapped from this trial's verdict.
        # Runs AFTER result.json is written so the trial record (which
        # retention never deletes) always survives for resume + the matrix.
        # When retention ran, stamp its outcome back into result.json so the
        # applied level + pruned-artifact list are auditable (oyazdanb).
        retention_outcome = self._apply_retention(
            trial_dir, verdict_obj.verdict, probe_extras
        )
        if retention_outcome is not None:
            self._record_retention(result_doc, retention_outcome)
            result_path.write_text(
                json.dumps(result_doc, indent=2, sort_keys=False),
                encoding="utf-8",
            )

        # ``main_work_started`` / ``executed_iterations`` mirror
        # whether ``Popen`` actually launched a child. A normal
        # non-zero child exit is still ``launched=True`` (the
        # subprocess ran and exited); exec-time ``Popen`` failures
        # (ENOENT / EACCES / ENOEXEC) leave both fields at 0/False so
        # the matrix outcome classifier doesn't conflate a
        # command-not-found with a completed 1/1 trial. The
        # ``result.json`` is still written either way -- the artifact
        # contract from PR #194 round 4 is independent of the
        # matrix-side semantic.
        passed = verdict_obj.verdict == "pass"
        return WorkloadResult(
            passed=passed,
            failure_count=0 if passed else 1,
            failure_details=(
                []
                if passed
                else [
                    {
                        "exit_code": exit_code,
                        "timed_out": timed_out,
                        "type": _failure_detail_type(
                            launched=launched,
                            timed_out=timed_out,
                            exit_code=exit_code,
                        ),
                        "failure_detectors_fired": list(verdict_obj.failure_detectors_fired),
                    }
                ]
            ),
            main_work_started=launched,
            executed_iterations=1 if launched else 0,
            configured_iterations=1,
            elapsed_sec=walltime_sec,
            metrics={
                "verdict": verdict_obj.verdict,
                "exit_code": exit_code,
                "result_json_path": str(result_path),
                "failure_detectors_fired": list(verdict_obj.failure_detectors_fired),
                "error_detectors_fired": list(verdict_obj.error_detectors_fired),
                "warn_detectors_fired": list(verdict_obj.warn_detectors_fired),
            },
        )

    def _write_env_file_failure_result(
        self,
        *,
        exc: ValueError,
        result_path: Path,
        stderr_path: Path,
        env_file_path: Path,
        argv: tuple[str, ...],
        probe_extras: dict[str, Any],
        env_mode: str,
        cell_env_snapshot: dict[str, str],
    ) -> WorkloadResult:
        """Persist an ``error``-verdict artifact when probe.env validation rejects.

        Mirrors the exec-failed bookkeeping in :meth:`run` so the per-trial
        directory always contains both ``stderr.log`` (with the validation
        error message) and ``result.json``. Without this, ``flat_resume``'s
        ``is_trial_complete`` predicate keys off a missing ``result.json``
        and re-runs the same broken cell on every subsequent
        ``aorta probe`` invocation.

        Also unlinks ``probe.env`` (best-effort) before writing the
        ``error``-verdict result so the trial directory cannot leave
        behind a misleading artifact: ``_write_env_file`` is now
        validation-first (atomic-on-failure), but a stale probe.env from
        a PRIOR run of the same ``trial_<n>/`` directory (resume +
        flat_resume reuse the same dir) would otherwise survive the
        validation rejection and contradict
        ``result.json::failure_type==env_file_validation_failed``.
        """
        env_file_path.unlink(missing_ok=True)
        try:
            stderr_path.write_text(f"{exc}\n", encoding="utf-8")
        except OSError:
            pass
        result_doc: dict[str, Any] = {
            # Issue #230: a rejected probe.env is an infrastructure/config
            # failure -- the subprocess never launched, so the trial made
            # no valid observation of the thing under test. It resolves to
            # ``error`` (excluded from the matrix event-rate denominator),
            # not ``fail``. ``meta:env_file_validation_failed`` names the
            # error reason in the dedicated error list.
            "verdict": "error",
            # Exit-code 2 mirrors the ``_setup_validation`` convention used
            # elsewhere in the codebase for "config rejected before
            # subprocess could start" -- distinct from 126/127 which we
            # reserve for chmod/PATH problems on argv[0].
            "exit_code": 2,
            # Walltime is 0 because the subprocess never launched; we
            # spent only the env-file-validation pass which is bounded
            # to a few hundred microseconds. Reporting 0.0 keeps the
            # matrix's step-time aggregates from being polluted by a
            # cell that never produced step times.
            "walltime_sec": 0.0,
            # No subprocess and no Tier-3 amd-smi probe ran, so VRAM was
            # never measured -- ``None`` (the normal path's "unavailable"
            # value), not 0.
            "peak_vram_mib": None,
            "argv": list(argv),
            "cell_name": probe_extras.get("cell_name", "_unknown_"),
            "trial_index": self._trial_index,
            "failure_detectors_fired": [],
            "error_detectors_fired": ["meta:env_file_validation_failed"],
            # The remaining detector / capture / tier-timing keys are empty
            # here but PRESENT so this error artifact is schema-identical to
            # the normal probe path -- downstream parsers (and the documented
            # result.json schema) never have to special-case env-file errors.
            "warn_detectors_fired": [],
            "capture": {},
            "tier_durations_ms": {},
            "env_passthrough_mode": env_mode,
            "timed_out": False,
            # Mirror the normal-path result.json so every trial's env is
            # auditable/redactable, even when probe.env validation rejected
            # before the subprocess launched.
            "env": dict(cell_env_snapshot),
            "failure_type": "env_file_validation_failed",
            "error_message": str(exc),
        }
        result_path.write_text(
            json.dumps(result_doc, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        # Issue #231: honor ``retain.on_error`` for this infra-error trial
        # too (mirrors the main run() path). The probe.env was already
        # unlinked above; this prunes any other heavy artifacts a prior
        # resume left in the reused trial dir. Stamp the outcome back into
        # result.json so the applied level is auditable here too (oyazdanb).
        retention_outcome = self._apply_retention(
            result_path.parent, "error", probe_extras
        )
        if retention_outcome is not None:
            self._record_retention(result_doc, retention_outcome)
            result_path.write_text(
                json.dumps(result_doc, indent=2, sort_keys=False),
                encoding="utf-8",
            )
        return WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[
                {
                    "exit_code": 2,
                    "timed_out": False,
                    "type": "env_file_validation_failed",
                }
            ],
            # ``launched`` / ``main_work_started`` mirror the Popen-exec-
            # failed branch in :meth:`run`: the subprocess never started,
            # so neither flag flips. This keeps the matrix outcome
            # classifier from counting this as a completed 1/1 trial.
            main_work_started=False,
            executed_iterations=0,
            configured_iterations=1,
            elapsed_sec=0.0,
            metrics={
                "verdict": "error",
                "exit_code": 2,
                "result_json_path": str(result_path),
                "failure_detectors_fired": [],
                "error_detectors_fired": ["meta:env_file_validation_failed"],
                "warn_detectors_fired": [],
            },
        )

    def _apply_retention(
        self, trial_dir: Path, verdict: str, probe_extras: dict[str, Any]
    ) -> RetentionOutcome | None:
        """Prune this trial's heavy artifacts per ``retain`` (issue #231).

        Returns the :class:`~aorta.run.retention.RetentionOutcome` so the
        caller can record the applied level + deleted-artifact list into
        ``result.json`` for post-hoc auditability (a reader of a bundled or
        resumed run can tell a heavy artifact was *pruned* rather than never
        produced). Returns ``None`` when retention did not run -- the recipe
        omits ``retain`` (keep-everything default), the payload is malformed,
        or the engine raised (best-effort: a retention error must never sink
        an already-classified trial).

        ``aorta.run.retention`` never deletes the trial record
        (``result.json``), so resume and the matrix are unaffected
        regardless of level.
        """
        retain = probe_extras.get("retain")
        if not retain:
            return None
        # Retention is documented best-effort -- a malformed payload (e.g. a
        # string or RetainPolicy passed via programmatic config) must not
        # sink the trial with an AttributeError. Mirror the isinstance(dict)
        # guard in _capture_cell_env and warn+skip when it isn't a mapping.
        if not isinstance(retain, dict):
            log.warning(
                "retention: probe_extras['retain'] is %s, expected a mapping; "
                "skipping retention for %s",
                type(retain).__name__,
                trial_dir,
            )
            return None
        level = retain.get(f"on_{verdict}", "full")
        try:
            outcome = apply_retention(trial_dir, level)
        except Exception as exc:  # noqa: BLE001 -- retention is best-effort
            log.warning(
                "retention: pruning %s at level %r failed (%s); artifacts kept",
                trial_dir,
                level,
                exc,
            )
            return None
        if outcome.deleted:
            log.info(
                "retention[%s]: pruned %d artifact(s) (~%d bytes) from %s",
                level,
                len(outcome.deleted),
                outcome.freed_bytes,
                trial_dir,
            )
        return outcome

    @staticmethod
    def _record_retention(result_doc: dict[str, Any], outcome: RetentionOutcome) -> None:
        """Stamp the retention outcome into ``result_doc["capture"]``.

        Makes the applied level + pruned-artifact list auditable from the
        trial record alone, so a reader of a bundled/resumed run can tell a
        missing heavy artifact was *pruned by policy* rather than never
        produced (oyazdanb review). The trial record is itself never pruned
        by retention, so this stays readable at every level.
        """
        capture = result_doc.setdefault("capture", {})
        if isinstance(capture, dict):
            capture["retention"] = {
                "level": outcome.level,
                "deleted": list(outcome.deleted),
                "freed_bytes": outcome.freed_bytes,
            }

    def _capture_cell_env(self, probe_extras: dict[str, Any]) -> dict[str, str]:
        """Compute the cell's mitigation+diagnostic env-var bundle.

        Phase 1 takes the bundle from the probe_extras dict the runner
        attaches per-cell -- the dispatcher has already stamped these
        on os.environ for ``inherit`` mode, but ``file`` mode needs
        the raw bundle to write into ``probe.env`` (without scooping
        unrelated host env vars).
        """
        bundle = probe_extras.get("cell_env_vars") or {}
        if not isinstance(bundle, dict):
            return {}
        return {str(k): str(v) for k, v in bundle.items()}


def _coerce_disable_tokens(raw: object, key: str) -> frozenset[str]:
    """Build a disable-token set from a ``probe_extras`` payload, fail-fast.

    ``probe_extras`` is an untyped dict at runtime, so a malformed payload
    where the value is a bare string (e.g. ``"tier3"``) would otherwise
    iterate per-character and yield ``{'t','i','e','r','3'}``. Reject
    strings explicitly so a bad payload surfaces instead of silently
    mis-disabling. Each element must itself be a ``str`` -- a non-string
    token (e.g. ``["tier3", 5]``) would otherwise survive into the
    set and later crash ``sorted(disabled_detectors)`` (TypeError on
    mixed types) or silently no-op against the string detector IDs.
    ``None`` is the documented no-op (every detector active). Mirrors
    the fail-fast posture of the sibling ``tier3_vram_growth`` knob.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, str) or not isinstance(raw, (list, tuple, set, frozenset)):
        raise TypeError(
            f"probe_extras[{key!r}] must be a list/tuple/set/frozenset of string "
            f"tokens (a non-string sequence), got {type(raw).__name__} ({raw!r})"
        )
    tokens: list[str] = []
    for tok in raw:
        if not isinstance(tok, str):
            raise TypeError(
                f"probe_extras[{key!r}] tokens must all be strings, got "
                f"{type(tok).__name__} ({tok!r})"
            )
        tokens.append(tok)
    return frozenset(tokens)


def _validate_env_file_entries(env: dict[str, str]) -> None:
    """Reject hostile/malformed env keys+values without touching the filesystem.

    Two row-injection vectors that the sidecar-mitigation loader does
    NOT catch (it only enforces ``isinstance(key, str)``):

    * ``\\n``, ``\\r`` or ``=`` in a *key* would let a hostile sidecar
      inject extra ``KEY=VALUE`` rows or rebind a later key.
    * ``\\n``/``\\r`` in a *value* would corrupt the bare-KEY=VALUE
      file shape; a downstream reader would silently see a truncated
      value.

    Run a full-map validation pass BEFORE the caller opens / truncates
    ``probe.env`` so a rejection on row 5 does not leave rows 1..4 on
    disk (per the round-6 review on ``_write_env_file``: a partial
    file is "valid-looking but incomplete" -- worse than no file at
    all, because downstream tools cannot tell the difference between
    "the cell ran with these vars" and "the cell rejected the
    bundle but only after writing these four").
    """
    for key in sorted(env):
        if "\n" in key or "\r" in key or "=" in key:
            raise ValueError(
                f"env key {key!r} contains a newline, carriage "
                "return, or '=' character; probe.env uses bare "
                "KEY=VALUE format and rejects these to prevent "
                "row-injection via a hostile mitigation sidecar"
            )
        value = env[key]
        if "\n" in value or "\r" in value:
            raise ValueError(
                f"env value for {key!r} contains a newline; "
                "probe.env uses bare KEY=VALUE format and cannot "
                "encode multi-line values"
            )


def _failure_detail_type(*, launched: bool, timed_out: bool, exit_code: int) -> str:
    """Map a failing trial's process outcome to a ``failure_details[].type``.

    The type must agree with the trial's actual exit state. The previous
    inline expression hard-coded ``subprocess_nonzero_exit`` whenever the
    child *launched*, which contradicted the artifact for a trial that
    exited ``0`` but was failed by a non-Tier-1 detector (e.g. a Tier-4 log
    pattern) -- ``{"exit_code": 0, "type": "subprocess_nonzero_exit"}`` is
    impossible to reconcile by anyone reading the JSON (issue #229).

    Precedence (most-specific first):

    * ``not launched`` -> ``subprocess_exec_failed`` -- exec-time ``Popen``
      failure (ENOENT / EACCES / ENOEXEC); the child never started.
    * ``timed_out``    -> ``subprocess_timeout`` -- killed by the per-trial
      timeout (``exit_code`` is the synthetic ``-1`` in this case, so it is
      neither a "nonzero exit" the operator chose nor a clean exit).
    * ``exit_code != 0`` -> ``subprocess_nonzero_exit`` -- the child ran and
      exited non-zero (the original, still-correct case).
    * otherwise (``exit_code == 0``) -> ``detector_failure`` -- the child
      exited cleanly; the failure came purely from a classifier detector.
    """
    if not launched:
        return "subprocess_exec_failed"
    if timed_out:
        return "subprocess_timeout"
    if exit_code != 0:
        return "subprocess_nonzero_exit"
    return "detector_failure"


def _read_log_text(stdout_path: Path, stderr_path: Path) -> str:
    """Read stdout + stderr back as a single text blob for the classifier.

    Reads are bounded to :data:`aorta.probe.sandbox.MAX_LOG_BYTES`
    per stream as a hard regex-DoS cap; the per-tier scanners further
    bound individual ``re.search`` invocations.

    Errors decoded with ``errors="replace"`` (U+FFFD per invalid
    byte, 1:1 byte→char) rather than ``backslashreplace`` (up to
    4 chars per invalid byte for ``\\xff`` etc.). The cheaper
    expansion matters because ``MAX_LOG_BYTES`` is meant to be the
    upper bound on the regex-input length; with
    ``backslashreplace`` a binary-heavy stdout could quadruple the
    decoded string and let a runaway log inflate regex CPU/memory
    past the documented cap. The replacement char loses the
    underlying byte value but the classifier scanners don't depend
    on the exact byte -- they pattern-match on textual error
    messages, where invalid UTF-8 is noise that just needs a
    placeholder so the surrounding text stays in line.
    """
    from aorta.probe.sandbox import MAX_LOG_BYTES

    parts: list[str] = []
    for path in (stdout_path, stderr_path):
        try:
            data = path.read_bytes()[:MAX_LOG_BYTES]
            parts.append(data.decode("utf-8", errors="replace"))
        except FileNotFoundError:
            parts.append("")
        except OSError:
            parts.append("")
    return "\n".join(parts)


def _tier1_only_fallback_verdict(
    *,
    exit_code: int,
    timed_out: bool,
    trial_dir: Path,
    exec_failed: bool,
    classifier_exc: BaseException,
    disabled_tiers: frozenset[str] = frozenset(),
    disabled_detectors: frozenset[str] = frozenset(),
) -> tuple[Verdict, dict[str, float]]:
    """Build a deterministic verdict when :func:`classify_trial` raises.

    Tier 1 alone is enough to give the trial a sensible verdict: it
    is the only tier that's a pure function of the subprocess exit
    state and the trial dir, so it can't itself crash on regex /
    capture / dmesg edge cases. We re-run :func:`tier1_detect`
    here (cheap, no FS work beyond a glob in ``trial_dir``) and
    encode the original classifier exception under
    ``capture['classifier_error']`` so operators see WHY the full
    classifier was bypassed instead of a silent Tier-1-only
    result. ``tier_durations_ms`` records the fallback in
    ``capture`` rather than the per-tier breakdown -- the other
    four tiers genuinely did not run.

    The operator's detector-disable knobs apply here too, so this
    fallback honours the same contract as :func:`classify_trial`: a
    silenced ``tier1`` / ``tier1:<id>`` doesn't fire. The caller passes
    the *effective* set (Tier 1 forced back on for an exec-failure trial)
    so a command-not-found still fails even with Tier 1 disabled.

    Verdict rule: the same fail > error > pass precedence the full
    resolver applies (issue #230) -- a Tier-1 ``tier1:timeout`` with
    no recognised hang or a ``tier1:exec_failed`` resolves to
    ``error``, a genuine Tier-1 signal to ``fail``, nothing to
    ``pass``. Matches the resolver so a classifier crash doesn't
    silently flip an infra error back into a fail.

    ``exec_failed`` (the workload's ``not launched`` flag) is
    forwarded into :class:`Tier1Context` so the fallback can still
    emit ``tier1:exec_failed`` (and suppress the misleading
    ``exit_nonzero`` from the synthetic 127/126/1 exit code) even
    when the full classifier is the thing that crashed.
    """
    if "tier1" in disabled_tiers:
        fired: list[str] = []
    else:
        fired = [
            d
            for d in tier1_detect(
                Tier1Context(
                    exit_code=exit_code,
                    timed_out=timed_out,
                    trial_dir=trial_dir,
                    exec_failed=exec_failed,
                )
            )
            if d not in disabled_detectors
        ]
    failures, errors = partition_detectors(list(fired))
    capture: dict[str, str | float | int] = {
        "classifier_error": f"{type(classifier_exc).__name__}: {classifier_exc}",
    }
    verdict = Verdict(
        verdict=verdict_from_detectors(failures, errors),
        failure_detectors_fired=failures,
        warn_detectors_fired=[],
        capture=capture,
        error_detectors_fired=errors,
    )
    # Per-tier durations: Tier 1 ran, everything else was skipped.
    tier_durations_ms = {
        "tier1": 0.0,
        "tier2": 0.0,
        "tier3": 0.0,
        "tier4": 0.0,
        "tier5": 0.0,
    }
    return verdict, tier_durations_ms


def _write_env_file(path: Path, env: dict[str, str]) -> None:
    """Write a POSIX KEY=VALUE\\n env file at ``chmod 0600``.

    Atomic with respect to validation failure: ``_validate_env_file_entries``
    runs the full key/value check BEFORE we ``os.open(..., O_TRUNC)``, so
    a hostile or malformed bundle either produces a complete + correct
    file or leaves the path untouched. The previous interleaved shape
    (open-truncate first, validate per-row while writing) could leave a
    partial probe.env from rows 1..N-1 when row N was rejected. That
    file looked legitimate (0600, KEY=VALUE shape) but contained only
    a subset of the cell's bundle -- exactly the misleading artifact
    the round-6 review flagged.

    The 0600 mode is set via ``os.O_CREAT`` mode bits on platforms
    that honour them, and chmod'd after as a belt-and-suspenders for
    filesystems that ignore the open-mode (NFS without root squash,
    some FUSE backends).

    Per R5 in the rubric, the env file is the leakage surface for
    secrets in ``file`` mode; Phase 3 redaction scrubs these from the
    bundle, but Phase 1 ships the 0600 guard as the only mitigation.
    """
    _validate_env_file_entries(env)

    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for key in sorted(env):
                fh.write(f"{key}={env[key]}\n")
    finally:
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass


__all__ = [
    "CONFIG_KEY_LOG_PREFIX",
    "CONFIG_KEY_PROBE_EXTRAS",
    "CONFIG_KEY_SUBPROCESS_ARGV",
    "SubprocessWorkload",
]
