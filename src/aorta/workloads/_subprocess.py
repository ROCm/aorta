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

Verdict rule (Phase 1, Tier 1 only): ``exit_code == 0`` -> pass, else
fail. Pattern matching, hang detection, dmesg scanning, and the
``custom_patterns`` runner are deferred to Phase 2.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from aorta.workloads._base import Workload, WorkloadResult

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
    3. Verdict is Tier 1 only in Phase 1: ``exit_code == 0`` -> pass.
       The full classifier and the sandbox land in Phase 2.
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
        """Fork the user command and write the Tier-1 ``result.json``."""
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
        try:
            with open(stdout_path, "wb") as out_fh, open(stderr_path, "wb") as err_fh:
                proc = subprocess.Popen(
                    list(argv),
                    stdout=out_fh,
                    stderr=err_fh,
                    env=child_env,
                )
                launched = True
                try:
                    exit_code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    # Race-safe shutdown: the child can exit between
                    # the ``wait()`` timeout and our ``kill()`` (e.g.
                    # the workload finished while the kernel was
                    # delivering the SIGALRM the timeout uses), which
                    # makes ``Popen.kill()`` raise
                    # ``ProcessLookupError`` (ESRCH) on Linux. Swallow
                    # that one specific case so the trial deterministically
                    # records ``timed_out=True`` and ``exit_code=-1``
                    # rather than crashing the workload and leaving the
                    # trial with no ``result.json``. Any other OSError
                    # from kill() (EPERM etc.) re-raises so we don't
                    # silently swallow a genuine bug.
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait()
                    except ProcessLookupError:
                        pass
                    exit_code = -1
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

        verdict = "pass" if exit_code == 0 and not timed_out else "fail"

        result_doc: dict[str, Any] = {
            "verdict": verdict,
            "exit_code": exit_code,
            "walltime_sec": walltime_sec,
            "argv": list(argv),
            "cell_name": probe_extras.get("cell_name", "_unknown_"),
            "trial_index": self._trial_index,
            "env_passthrough_mode": env_mode,
            "timed_out": timed_out,
        }
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
        return WorkloadResult(
            passed=(verdict == "pass"),
            failure_count=0 if verdict == "pass" else 1,
            failure_details=(
                []
                if verdict == "pass"
                else [
                    {
                        "exit_code": exit_code,
                        "timed_out": timed_out,
                        "type": (
                            "subprocess_nonzero_exit" if launched else "subprocess_exec_failed"
                        ),
                    }
                ]
            ),
            main_work_started=launched,
            executed_iterations=1 if launched else 0,
            configured_iterations=1,
            elapsed_sec=walltime_sec,
            metrics={
                "verdict": verdict,
                "exit_code": exit_code,
                "result_json_path": str(result_path),
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
    ) -> WorkloadResult:
        """Persist a Tier-1 ``fail`` artifact when probe.env validation rejects.

        Mirrors the exec-failed bookkeeping in :meth:`run` so the per-trial
        directory always contains both ``stderr.log`` (with the validation
        error message) and ``result.json``. Without this, ``flat_resume``'s
        ``is_trial_complete`` predicate keys off a missing ``result.json``
        and re-runs the same broken cell on every subsequent
        ``aorta probe`` invocation.

        Also unlinks ``probe.env`` (best-effort) before writing the fail
        result so the trial directory cannot leave behind a misleading
        artifact: ``_write_env_file`` is now validation-first
        (atomic-on-failure), but a stale probe.env from a PRIOR run of
        the same ``trial_<n>/`` directory (resume + flat_resume reuse
        the same dir) would otherwise survive the validation rejection
        and contradict ``result.json::failure_type==env_file_validation_failed``.
        """
        env_file_path.unlink(missing_ok=True)
        try:
            stderr_path.write_text(f"{exc}\n", encoding="utf-8")
        except OSError:
            pass
        result_doc: dict[str, Any] = {
            "verdict": "fail",
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
            "argv": list(argv),
            "cell_name": probe_extras.get("cell_name", "_unknown_"),
            "trial_index": self._trial_index,
            "env_passthrough_mode": env_mode,
            "timed_out": False,
            "failure_type": "env_file_validation_failed",
            "error_message": str(exc),
        }
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
                "verdict": "fail",
                "exit_code": 2,
                "result_json_path": str(result_path),
            },
        )

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
