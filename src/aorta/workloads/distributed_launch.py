"""Spur and Monarch launchers for the tiny-transformer workload.

Neither launcher is an install dependency. Spur is a subprocess of
``spur_bin`` (default ``spur``). Monarch is imported only when
:func:`launch_monarch` runs.

The spur controller runs one worker, :func:`run_worker_rank`, with the
already resolved training config. It does not replay the parent ``aorta``
command: a triage matrix would otherwise train every cell inside each
launch, and a library caller would otherwise keep only ``steps``. Each
worker maps ``SLURM_PROCID`` to ``RANK`` and ``SLURM_LOCALID`` to
``LOCAL_RANK`` (``RANK % gpus`` is only the block-placement fallback).
``MASTER_ADDR`` is the training step's own rank 0. That rank publishes its
IPv4 under ``AORTA_SHARED_DIR``; the other ranks of the same step read it.
``AORTA_RANK0_IP`` overrides the published address. Rank 0 also writes
``rank0.json`` there, a path every node can see.
The controller's verdict passes only when the worker exit code is 0 and
that rank-0 result passed.

The monarch controller follows this sequence: TCP transport on the
controller IPv4, one worker per node, a proc mesh, and
``setup_torch_elastic_env_async``. The workers start as one step of the
current allocation: ``srun --jobid <id> --overlap`` under Slurm, or
``spur srun`` from inside a Spur allocation (``scheduler: spur``). The actor
calls :func:`run_training_rank` instead of an inlined model. A missing
``monarch`` import raises an error that says Monarch is not installed.
"""

from __future__ import annotations

import asyncio
import errno
import inspect
import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aorta.workloads._base import WorkloadResult

log = logging.getLogger(__name__)

# Grace before SIGKILL, then a short reap. SIGKILL does not need the SIGTERM
# budget; a second Ctrl-C during the grace still force-kills and re-raises.
_TERMINATE_GRACE_SEC = 10.0
_REAP_AFTER_KILL_SEC = 1.0

_IDENTITY_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
)

# Worker-side mapping. spur sets SLURM_PROCID per task; RANK has to be set
# before the training process starts because the loop reads it to pick a device
# and to decide who publishes rank0.json.
SPUR_WORKER_SCRIPT = """\
set -euo pipefail
if [ -z "${SLURM_PROCID:-}" ]; then
  echo "spur worker: SLURM_PROCID is required to set RANK" >&2
  exit 2
fi
export RANK="$SLURM_PROCID"
gpus="${AORTA_GPUS_PER_NODE:?AORTA_GPUS_PER_NODE is required}"
# SLURM_LOCALID is this task's index on the node. RANK % gpus matches that
# only for block placement. Cyclic placement puts global ranks 0 and 2 on
# one node, and both would select GPU 0.
if [ -n "${SLURM_LOCALID:-}" ]; then
  case "$SLURM_LOCALID" in
    *[!0-9]*)
      echo "spur worker: SLURM_LOCALID must be an integer, got ${SLURM_LOCALID}" >&2
      exit 2
      ;;
  esac
  export LOCAL_RANK="$SLURM_LOCALID"
else
  export LOCAL_RANK=$(( RANK % gpus ))
fi
: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${AORTA_INNER:?AORTA_INNER is required}"
# Rank 0 of this step publishes the address. A separate spur srun would be
# another allocation, and its rank 0 can be a different host.
if [ -z "${MASTER_ADDR:-}" ]; then
  : "${AORTA_RESULTS_DIR:?AORTA_RESULTS_DIR is required to learn the rank 0 address}"
  _aorta_addr="${AORTA_RESULTS_DIR}/rank0.addr"
  if [ "$RANK" = "0" ]; then
    _aorta_ip=$(python3 -c 'import socket; print(socket.gethostbyname(socket.getfqdn()))')
    case "$_aorta_ip" in
      *.*.*.*) ;;
      *)
        echo "spur worker: rank 0 address ${_aorta_ip} is not IPv4" >&2
        exit 2
        ;;
    esac
    printf '%s\\n' "$_aorta_ip" > "${_aorta_addr}.tmp"
    mv "${_aorta_addr}.tmp" "$_aorta_addr"
  else
    _aorta_deadline=$((SECONDS + 60))
    while [ ! -s "$_aorta_addr" ]; do
      if [ "$SECONDS" -ge "$_aorta_deadline" ]; then
        echo "spur worker: rank 0 did not publish its address" >&2
        exit 2
      fi
      sleep 0.2
    done
  fi
  MASTER_ADDR=$(cat "$_aorta_addr")
  export MASTER_ADDR
fi
"""

# Background the task and stop it when the controller publishes the cancel
# file. Spur rejects ``scancel job.step``; this is what ends the remote task.
# ``set -m`` puts the task in its own process group, so a signal reaches the
# processes it started, not only the first one. That group is not the one
# the scheduler signals: without cgroups, Spur's step supervisor signals the
# wrapper's group only. The trap passes TERM, INT, and HUP on to the task's
# group and remembers 128 plus that signal; the task itself only ever sees
# TERM. If that signal lands before ``wait``, a task that ignores TERM would
# block there and never reach the reap, so the wrapper arms a two-second
# SIGKILL once the child pid is saved, and disarms it once ``wait`` returns.
# The timer is not started from the trap: a background job there would
# overwrite ``$!`` and the cancel watcher would be tracked as the timer.
# The wrapper exits with the remembered signal when one was caught, expanded
# in the ``exit`` command, ahead of whatever status the task itself returned.
# The trap is set before the task starts, so an early signal cannot kill the
# wrapper and orphan the task. State variables are cleared first so a value
# inherited from the environment is not a pid the trap or the final kill
# will signal.
_STEP_WATCH = """\
_aorta_signal=
_aorta_child=
_aorta_escalator=
_aorta_wait_done=
_aorta_watcher=
# Bound ``wait`` when TERM was already forwarded. The timer is its own
# process group so disarming it does not signal the wrapper or the task.
# Call this only after ``$!`` has been saved: it backgrounds a job.
_aorta_arm_kill() {
  if [ -n "${_aorta_wait_done}" ] || [ -n "${_aorta_escalator}" ]; then
    return
  fi
  set -m
  (
    trap '' HUP INT TERM
    sleep 2
    kill -KILL -- "-$_aorta_child" 2>/dev/null || true
  ) >/dev/null 2>&1 &
  _aorta_escalator=$!
  set +m
}
_aorta_stop() {
  _aorta_signal=$1
  if [ -z "${_aorta_child}" ]; then
    return
  fi
  kill -TERM -- "-$_aorta_child" 2>/dev/null || true
}
# Test-only. Production leaves both paths unset. Touch the ready file and
# block until the release file exists, or 30s, so a test can signal while
# this shell is parked outside ``wait``.
_aorta_gate() {
  if [ -z "$1" ] || [ -z "$2" ]; then
    return
  fi
  : > "$1"
  _aorta_gate_deadline=$((SECONDS + 30))
  while [ ! -e "$2" ]; do
    if [ "$SECONDS" -ge "$_aorta_gate_deadline" ]; then
      break
    fi
    sleep 0.05 || true
  done
}
trap '_aorta_stop 129' HUP
trap '_aorta_stop 130' INT
trap '_aorta_stop 143' TERM
# Before the task, so a test can signal while an inherited child pid would
# still be in effect. Production leaves the gate unset.
_aorta_gate "${AORTA_STEP_BEFORE_TASK_READY:-}" "${AORTA_STEP_BEFORE_TASK_RELEASE:-}"
set -m
"$@" &
_aorta_child=$!
set +m
# The trap could not forward a signal that arrived before the pid existed.
if [ -n "$_aorta_signal" ]; then
  kill -TERM -- "-$_aorta_child" 2>/dev/null || true
  _aorta_arm_kill
fi
_aorta_status=0
_aorta_cancel=${AORTA_STEP_CANCEL:-}
if [ -z "$_aorta_cancel" ] && [ -n "${AORTA_RESULTS_DIR:-}" ]; then
  _aorta_cancel="${AORTA_RESULTS_DIR}/aorta-step-cancel"
fi
if [ -n "$_aorta_cancel" ]; then
  (
    while [ ! -e "$_aorta_cancel" ]; do
      sleep 0.2 || true
    done
    kill -TERM -- "-$_aorta_child" 2>/dev/null || true
    sleep 1 || true
    kill -KILL -- "-$_aorta_child" 2>/dev/null || true
  ) >/dev/null 2>&1 &
  _aorta_watcher=$!
fi
# Last pause before wait, so a signal during watcher setup is included.
# Arm here, not from the trap, so a signal between the watcher's ``&`` and
# the ``$!`` assignment above cannot replace the watcher pid with the timer.
_aorta_gate "${AORTA_STEP_BEFORE_WAIT_READY:-}" "${AORTA_STEP_BEFORE_WAIT_RELEASE:-}"
if [ -n "$_aorta_signal" ]; then
  _aorta_arm_kill
  # The timer's SIGKILL makes bash print "Killed" on this wait.
  wait "$_aorta_child" 2>/dev/null || _aorta_status=$?
else
  wait "$_aorta_child" || _aorta_status=$?
fi
_aorta_wait_done=1
if [ -n "${_aorta_escalator:-}" ]; then
  kill -KILL -- "-$_aorta_escalator" 2>/dev/null || true
  wait "$_aorta_escalator" 2>/dev/null || true
fi
# The first process can exit while its children run on. Stop the rest of
# the group here; the watcher is killed below, before its own SIGKILL.
if kill -0 -- "-$_aorta_child" 2>/dev/null; then
  _aorta_gate "${AORTA_STEP_REAP_READY:-}" "${AORTA_STEP_REAP_RELEASE:-}"
  kill -TERM -- "-$_aorta_child" 2>/dev/null || true
  _aorta_tries=0
  while [ "$_aorta_tries" -lt 10 ] && kill -0 -- "-$_aorta_child" 2>/dev/null; do
    # Spur's worker script is ``set -e``. A signal to this group kills sleep
    # too; without ``|| true`` the shell would exit before the SIGKILL below.
    sleep 0.2 || true
    _aorta_tries=$((_aorta_tries + 1))
  done
  kill -KILL -- "-$_aorta_child" 2>/dev/null || true
fi
if [ -n "${_aorta_watcher:-}" ]; then
  kill "$_aorta_watcher" 2>/dev/null || true
  wait "$_aorta_watcher" 2>/dev/null || true
fi
exit "${_aorta_signal:-$_aorta_status}"
"""

SPUR_WORKER_SCRIPT += _STEP_WATCH

# First line of a launched command, so Spur's ``scontrol show step`` can see
# which step this process created. The marker stays inside the 256-byte step
# name. Slurm names a step after its executable, so it gets ``--job-name``.
_OWNED_STEP_MARK = "aorta-owned-step"
_STEP_CANCEL_NAME = "aorta-step-cancel"
_STEP_TOKEN_RE = re.compile(r"\Aaorta-\d+-[0-9a-f]+\Z")
_STEP_ID_RE = re.compile(r"StepId=(\d+)\.(\d+)\b")
_STEP_GONE_TIMEOUT_SEC = 15.0
_STEP_POLL_SEC = 0.2
_LOG_TEXT = {"encoding": "utf-8", "errors": "backslashreplace"}

_MONARCH_MISSING = (
    "Monarch is not installed. launcher 'monarch' needs a pytorch_monarch "
    "build that imports on this host (published torchmonarch wheels are "
    "CUDA-only); import failed: {exc!r}"
)

# Launcher knobs. They never reach the training loop.
_CONTROLLER_CONFIG_KEYS = frozenset(
    {"launcher", "nodes", "gpus_per_node", "spur_bin", "scheduler"}
)

# One rank, one config. Importing the function keeps the worker from
# rebuilding ``sys.argv`` (a triage parent would rerun every cell).
_WORKER_BOOTSTRAP = """\
import json
import os

from aorta.workloads.distributed_launch import run_worker_rank

with open(os.environ["AORTA_WORKER_CONFIG"], encoding="utf-8") as handle:
    loaded = json.load(handle)
if not isinstance(loaded, dict):
    raise SystemExit("AORTA_WORKER_CONFIG must contain a JSON object")
run_worker_rank(loaded)
"""

_LOCAL_RESULT_PREFIXES = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))


def is_ipv4(value: object) -> bool:
    """True for a dotted decimal IPv4 address without leading zeros."""
    if not isinstance(value, str):
        return False
    parts = value.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        if part != "0" and part.startswith("0"):
            return False
        if int(part) > 255:
            return False
    return True


def controller_ipv4() -> str:
    """IPv4 address workers should dial.

    ``AORTA_CONTROLLER_IP`` wins, then ``MONARCH_CONTROLLER_IP``. Otherwise
    the host's own name is resolved with ``gethostbyname``, which returns
    IPv4. A non-IPv4 result is an error: link-local IPv6 is not usable as
    ``MASTER_ADDR`` here.
    """
    for key in ("AORTA_CONTROLLER_IP", "MONARCH_CONTROLLER_IP"):
        raw = os.environ.get(key)
        if raw:
            if not is_ipv4(raw):
                raise RuntimeError(f"{key} must be an IPv4 address, got {raw!r}")
            return raw
    try:
        ip = socket.gethostbyname(socket.getfqdn())
    except OSError as exc:
        raise RuntimeError(
            "Could not resolve the controller IPv4 address. "
            f"Set AORTA_CONTROLLER_IP. Hostname lookup failed: {exc}"
        ) from exc
    if not is_ipv4(ip):
        raise RuntimeError(
            f"Controller address {ip!r} is not IPv4. Set AORTA_CONTROLLER_IP "
            "to the controller's IPv4 address."
        )
    return ip


def is_launched_rank() -> bool:
    """True when this process should train instead of spawning workers.

    Spur workers export ``AORTA_INNER=1``. Torchrun and Monarch's elastic
    agent set ``TORCHELASTIC_RUN_ID`` together with ``RANK`` and
    ``WORLD_SIZE``. A controller that only inherited ``RANK`` and
    ``WORLD_SIZE`` from the shell still spawns.
    """
    if os.environ.get("AORTA_INNER") == "1":
        return True
    if not os.environ.get("TORCHELASTIC_RUN_ID"):
        return False
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def spawning_controller(workload: str, *configs: Mapping[str, Any] | None) -> bool:
    """True when this process launches workers and must keep the trial output.

    A shell can inherit ``RANK`` from an earlier allocation. Spur and Monarch
    still launch from that process. The dispatcher and the triage runner
    otherwise treat a nonzero ``RANK`` as a peer and discard the result.
    An in-process rank, including one under torchrun or ``AORTA_INNER``,
    stays on the rank-0 write gate.
    """
    if workload != "tiny_transformer" or is_launched_rank():
        return False
    for config in configs:
        if isinstance(config, dict) and config.get("launcher") in {"spur", "monarch"}:
            return True
    return False


def run_training_rank(config: dict[str, Any]) -> dict[str, Any]:
    """Run one training rank and return its :class:`WorkloadResult` as a dict.

    Shared by the in-process path, spur workers, and the Monarch actor.
    Cleanup still runs when setup or run raises, but that path does not
    enter ``dist.barrier``. A cleanup error does not replace the original
    exception. When the run itself finished, a cleanup error is stored on
    ``metrics["cleanup_error"]``. ``KeyboardInterrupt`` is not caught.
    """
    if not isinstance(config, dict):
        raise TypeError(f"training config must be a dict, got {type(config).__name__}")
    # Imported here so a controller that only classifies a launcher does not
    # load PyTorch. ``run_trials`` imports this module for every workload.
    from aorta.workloads.training import TrainingWorkload

    workload = TrainingWorkload(config)
    cleanup_error: str | None = None
    finished = False
    try:
        workload.setup()
        result = workload.run()
        finished = True
    finally:
        try:
            workload.cleanup(collective=finished)
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            log.warning("training rank cleanup failed", exc_info=True)
    payload = asdict(result)
    if cleanup_error is not None:
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            metrics["cleanup_error"] = cleanup_error
        else:
            payload["metrics"] = {"cleanup_error": cleanup_error}
    return payload


def worker_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Training config for one worker, without controller-only keys.

    ``launcher``, mesh shape, and ``spur_bin`` tell the controller how to
    spawn. Keys starting with ``_aorta_`` are controller metadata. Everything
    else, including model, optimizer, device, and checks, is the job the
    caller asked for.
    """
    if not isinstance(config, dict):
        raise TypeError(f"training config must be a dict, got {type(config).__name__}")
    kept: dict[str, Any] = {}
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        if key in _CONTROLLER_CONFIG_KEYS or key.startswith("_aorta_"):
            continue
        kept[key] = value
    return kept


def _explicit_rank0_ipv4() -> str:
    """``AORTA_RANK0_IP`` when set, else ``""`` so this step's rank 0 publishes it."""
    raw = os.environ.get("AORTA_RANK0_IP", "").strip()
    if not raw:
        return ""
    if not is_ipv4(raw):
        raise RuntimeError(f"AORTA_RANK0_IP must be an IPv4 address, got {raw!r}")
    return raw


def _requests_gpus(device: object) -> bool:
    """Reserve GPUs unless the recipe asked for CPU.

    ``device: auto`` still requests them. Without a GPU request a standalone
    ``spur srun`` can leave the workers on unreserved devices, and ``auto``
    then trains on CPU while ``expected_world_size`` still passes.
    """
    return device != "cpu"


def run_worker_rank(config: dict[str, Any]) -> None:
    """Train this rank and, on rank 0, publish the result where the controller reads it.

    Every rank trains. Only rank 0 writes ``rank0.json``. The file is replaced
    into ``AORTA_RESULTS_DIR`` after an fsync so a controller on another host
    sees the finished object, not a partial write.
    """
    result = run_training_rank(worker_training_config(config))
    if os.environ.get("RANK") != "0":
        return
    raw_dir = os.environ.get("AORTA_RESULTS_DIR", "").strip()
    if not raw_dir:
        raise RuntimeError("rank 0 needs AORTA_RESULTS_DIR to publish its result")
    dest = Path(raw_dir) / "rank0.json"
    blob = json.dumps({"result": result})
    temporary = dest.with_suffix(".json.tmp")
    temporary.write_text(blob, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(dest)


def build_spur_launch(
    *,
    spur_bin: str,
    nodes: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    config_path: Path,
    results_dir: Path,
    step_token: str,
    device: object = "auto",
) -> tuple[list[str], dict[str, str]]:
    """Argv and env for one ``spur srun`` of :func:`run_worker_rank`.

    An empty ``master_addr`` is left unset so rank 0 of this step publishes
    it. ``device`` other than ``cpu`` adds ``--gpus-per-node``.
    """
    if master_addr and not is_ipv4(master_addr):
        raise ValueError(f"MASTER_ADDR must be an IPv4 address, got {master_addr!r}")
    if not isinstance(master_port, int) or isinstance(master_port, bool):
        raise TypeError(f"master_port must be an int, got {type(master_port).__name__}")
    script = _marked_worker_script(step_token)
    argv = [
        spur_bin,
        "srun",
        "-N",
        str(nodes),
        "--ntasks-per-node",
        str(gpus_per_node),
        "--ntasks",
        str(nodes * gpus_per_node),
    ]
    if _requests_gpus(device):
        argv.extend(["--gpus-per-node", str(gpus_per_node)])
    argv.extend(
        [
            "bash",
            "-c",
            script,
            "aorta-inner",
            sys.executable,
            "-c",
            _WORKER_BOOTSTRAP,
        ]
    )
    env = _worker_env(
        master_addr=master_addr,
        master_port=master_port,
        world_size=nodes * gpus_per_node,
        gpus_per_node=gpus_per_node,
    )
    env["AORTA_WORKER_CONFIG"] = str(config_path)
    env["AORTA_RESULTS_DIR"] = str(results_dir)
    return argv, env


def _open_subprocess_log(config: Mapping[str, Any], target: Path) -> Any:
    """Open one sibling log without following a link out of the results root.

    The dispatcher threads ``_aorta_results_root`` and its inode. A plain
    ``open`` of ``<prefix>.subprocess.stdout.log`` follows a symlink planted
    at the workload directory or at the filename and truncates whatever it
    names. Descent from that frozen anchor refuses both, the same way the
    trial logs do. A caller that did not thread the anchor keeps a plain
    ``open``; there is no operator boundary to pin.
    """
    from aorta.run import _fsafe
    from aorta.run.collectors import trusted_collector_anchor

    root = config.get("_aorta_results_root")
    if not _fsafe.HAVE_FD_TRAVERSAL or not isinstance(root, str) or not root:
        return open(target, "w", **_LOG_TEXT)
    anchor = trusted_collector_anchor(config, target.parent)
    components = _fsafe.relative_components(anchor.path, target.parent)
    if components is None:
        raise _fsafe.UnsafePathError(
            errno.EINVAL,
            f"subprocess log {target} is outside trusted root {anchor.path}",
        )
    with _fsafe.open_dir_nofollow(anchor, components) as dir_fd:
        return _fsafe.open_write_nofollow(dir_fd, target.name, **_LOG_TEXT)


def _open_subprocess_logs(config: Mapping[str, Any]) -> tuple[Any, Any]:
    """Open sibling logs for a launcher child, or ``(None, None)``.

    The dispatcher already holds ``<prefix>.stdout.log`` and
    ``<prefix>.stderr.log`` for the Python process. A child inherits the
    OS streams, so those files miss its output. The sibling files are
    ``<prefix>.subprocess.stdout.log`` and ``<prefix>.subprocess.stderr.log``.
    Callers keep the handles open until the child and its process group
    are gone, then close them.
    """
    if not config.get("_aorta_save_logs"):
        return None, None
    prefix = config.get("_aorta_log_prefix")
    if not isinstance(prefix, str) or not prefix:
        return None, None
    opened: list[Any] = []
    try:
        for suffix in ("stdout", "stderr"):
            opened.append(_open_subprocess_log(config, Path(f"{prefix}.subprocess.{suffix}.log")))
    except OSError as exc:
        for handle in opened:
            handle.close()
        log.warning("could not open subprocess logs under %s: %s", prefix, exc)
        return None, None
    return opened[0], opened[1]


def _close_subprocess_logs(*handles: Any) -> None:
    """Close launcher log files. A close error is logged and does not replace the result."""
    for handle in handles:
        if handle is None:
            continue
        try:
            handle.close()
        except OSError as exc:
            log.warning("could not close subprocess log: %s", exc)


@dataclass(frozen=True)
class _OwnedStep:
    """The scheduler step this process created, never the enclosing allocation."""

    scheduler: str
    job_id: str
    token: str
    spur_bin: str


def _new_step_token() -> str:
    """A token that is safe to embed in a shell command and unique to this launch."""
    return f"aorta-{os.getpid()}-{secrets.token_hex(4)}"


def _marked_worker_script(token: str) -> str:
    """Worker script whose first line names the step this launch owns."""
    if _STEP_TOKEN_RE.fullmatch(token) is None:
        raise ValueError(f"step token must match {_STEP_TOKEN_RE.pattern}, got {token!r}")
    return f"# {_OWNED_STEP_MARK} {token}\n{SPUR_WORKER_SCRIPT}"


def _marked_command(token: str, command: list[str]) -> list[str]:
    """Run ``command`` under a shell line that carries the owned-step token."""
    if _STEP_TOKEN_RE.fullmatch(token) is None:
        raise ValueError(f"step token must match {_STEP_TOKEN_RE.pattern}, got {token!r}")
    return [
        "bash",
        "-c",
        f"# {_OWNED_STEP_MARK} {token}\n{_STEP_WATCH}",
        "aorta-owned",
        *command,
    ]


def _numeric_job_id(scheduler: str) -> str | None:
    """Allocation id for ``scheduler``, or ``None`` when this process has none.

    A Spur step belongs to ``SPUR_JOB_ID``. A Slurm step belongs to
    ``MONARCH_SLURM_JOB`` or ``SLURM_JOB_ID``. Anything else is refused so a
    cancel cannot be aimed at a flag or at the wrong allocation.
    """
    if scheduler == "spur":
        raw = os.environ.get("SPUR_JOB_ID", "").strip()
    else:
        raw = os.environ.get("MONARCH_SLURM_JOB", "").strip() or os.environ.get(
            "SLURM_JOB_ID", ""
        ).strip()
    if raw.isdigit():
        return raw
    return None


def _owned_step(scheduler: str, *, spur_bin: str, token: str) -> _OwnedStep | None:
    job_id = _numeric_job_id(scheduler)
    if job_id is None:
        return None
    return _OwnedStep(scheduler=scheduler, job_id=job_id, token=token, spur_bin=spur_bin)


def _matching_step_ids(text: str, *, job_id: str, token: str) -> list[str]:
    """Step numbers of ``job_id`` whose ``scontrol`` block contains ``token``.

    ``StepId=17.batch`` is not a user step and does not match. A step of
    another job does not match, even when the token appears beside it.
    """
    found: list[str] = []
    matches = list(_STEP_ID_RE.finditer(text))
    token_at = re.compile(rf"(?<![A-Za-z0-9-]){re.escape(token)}(?![A-Za-z0-9-])")
    for index, match in enumerate(matches):
        if match.group(1) != job_id:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if token_at.search(text[match.start() : end]) is None:
            continue
        found.append(match.group(2))
    return found


def _run_scheduler(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run one scheduler query. A missing tool or a timeout is logged, not raised."""
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("scheduler command %s failed: %s", argv, exc)
        return None


def _show_step_argv(step: _OwnedStep) -> list[str]:
    if step.scheduler == "spur":
        return [step.spur_bin, "scontrol", "show", "step", step.job_id]
    return ["scontrol", "show", "step", step.job_id]


def _scancel_argv(step: _OwnedStep, target: str) -> list[str]:
    """Cancel ``job.step`` only. A bare job id would cancel the allocation."""
    if "." not in target or target.split(".", 1)[0] != step.job_id:
        raise ValueError(f"refusing to cancel {target!r}; expected {step.job_id}.<step>")
    if step.scheduler == "spur":
        # Spur's scancel takes whole job ids. ``job.step`` is rejected by its
        # parser, which is safer than cancelling the allocation. Slurm accepts
        # the step target and stops only that step.
        return [step.spur_bin, "scancel", "--quiet", target]
    return ["scancel", "--quiet", target]


def _cancel_scheduler_step(step: _OwnedStep) -> None:
    """Cancel the step whose command carries ``step.token``.

    The lookup is ``scontrol show step <job>``. Only a block that contains
    the token is cancelled, so another step of the same allocation is left
    alone. Spur may reject the step target; the cancel file is what stops
    those tasks.
    """
    shown = _run_scheduler(_show_step_argv(step))
    if shown is None or shown.returncode != 0:
        detail = "" if shown is None else (shown.stderr or shown.stdout).strip()
        log.warning(
            "could not list steps of job %s (%s); remote tasks may still be running",
            step.job_id,
            detail or "no output",
        )
        return
    step_ids = _matching_step_ids(
        f"{shown.stdout}\n{shown.stderr}",
        job_id=step.job_id,
        token=step.token,
    )
    if not step_ids:
        log.warning(
            "no step of job %s carries %s; remote tasks may still be running",
            step.job_id,
            step.token,
        )
        return
    for step_id in step_ids:
        target = f"{step.job_id}.{step_id}"
        done = _run_scheduler(_scancel_argv(step, target))
        if done is None or done.returncode != 0:
            detail = "" if done is None else (done.stderr or done.stdout).strip()
            log.warning("could not cancel scheduler step %s (%s)", target, detail or "no output")


def _owned_step_listed(step: _OwnedStep) -> bool | None:
    """Whether ``scontrol`` still lists this step.

    ``False`` means the listing succeeded and the token is absent. ``None``
    means the listing could not be read, which is not proof the step ended.
    """
    shown = _run_scheduler(_show_step_argv(step))
    if shown is None or shown.returncode != 0:
        return None
    ids = _matching_step_ids(
        f"{shown.stdout}\n{shown.stderr}",
        job_id=step.job_id,
        token=step.token,
    )
    return bool(ids)


def _wait_owned_step_gone(step: _OwnedStep | None) -> bool:
    """True once the owned step is no longer listed.

    A missing scheduler or a step that stays listed returns ``False``. The
    caller then keeps the cancel file, so a worker still between polls can
    observe it.
    """
    if step is None:
        return False
    deadline = time.monotonic() + _STEP_GONE_TIMEOUT_SEC
    while True:
        listed = _owned_step_listed(step)
        if listed is False:
            return True
        if listed is None or time.monotonic() >= deadline:
            return False
        time.sleep(_STEP_POLL_SEC)


def _stop_owned_step(step: _OwnedStep | None, cancel_file: Path | None) -> None:
    """Ask the owned step to exit, before its files are deleted.

    The cancel file is on the shared work directory, which every task of a
    Spur step polls. ``scancel`` then names the same step. Neither call
    uses the allocation id alone.
    """
    if cancel_file is not None:
        try:
            cancel_file.write_text("1\n", encoding="utf-8")
        except OSError as exc:
            log.warning("could not publish %s: %s", cancel_file, exc)
    if step is not None:
        _cancel_scheduler_step(step)


def _client_abandoned(proc: subprocess.Popen[Any], pgid: int | None) -> bool:
    """True when the local client is gone or dying and the remote step may not be.

    A zero exit means the step finished. A signalled client, or ``spur srun``
    exiting 130 on Ctrl-C, leaves a buffered step running.
    """
    if _needs_group_cleanup(proc, pgid):
        return True
    code = proc.returncode
    return isinstance(code, int) and not isinstance(code, bool) and (code < 0 or code == 130)


def _step_may_outlive_client(
    proc: subprocess.Popen[Any], pgid: int | None, step: _OwnedStep | None
) -> bool:
    """True when the owned step may still run after the local client returned.

    An abandoned client is one case. A nonzero exit is the other: ``spur srun``
    also exits nonzero when it loses the controller, and the buffered step
    keeps running. A step the scheduler no longer lists has finished. A
    listing that cannot be read counts as still running, and so does a
    standalone launch with no step identity to look up.
    """
    if _client_abandoned(proc, pgid):
        return True
    code = proc.returncode
    if not isinstance(code, int) or isinstance(code, bool) or code == 0:
        return False
    if step is None:
        return True
    return _owned_step_listed(step) is not False


async def _await_mesh_shutdown(hosts: Any) -> None:
    """Shut down an owned host mesh. ``KeyboardInterrupt`` propagates.

    Monarch's ``shutdown`` may return a Future or a coroutine. A mesh with
    no ``shutdown`` raises ``AttributeError``, which the caller treats as a
    failed teardown and cancels the step instead.
    """
    result = hosts.shutdown()
    if inspect.isawaitable(result):
        await result


def launch_spur(
    config: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
    spur_bin: str,
) -> WorkloadResult:
    """Spawn ``spur srun`` and return rank 0's verdict gated on the exit code.

    The worker command is one rank of the resolved config. ``MASTER_ADDR`` is
    this step's rank 0, published into ``AORTA_SHARED_DIR`` unless
    ``AORTA_RANK0_IP`` is set. If the local client is abandoned, or exits
    nonzero while the scheduler still lists the owned step or there is no
    step identity (no ``SPUR_JOB_ID``), the cancel file is published and the
    step is cancelled as ``job.step`` when it has one. The shared directory
    stays until that step is confirmed gone, including when the wait is
    interrupted. The allocation id is never a cancel target.
    """
    raw_port = os.environ.get("AORTA_MASTER_PORT")
    if raw_port is None or raw_port == "":
        raise RuntimeError(
            "launcher 'spur' needs AORTA_MASTER_PORT set to a free TCP port. "
            "An ephemeral port cannot be held until the workers bind it."
        )
    master_port = _tcp_port(raw_port, name="AORTA_MASTER_PORT")
    master_addr = _explicit_rank0_ipv4()
    controller = os.environ.get("AORTA_CONTROLLER_IP", "").strip()
    if master_addr and controller and controller != master_addr:
        log.info(
            "MASTER_ADDR is rank 0 at %s; AORTA_CONTROLLER_IP %s is the controller",
            master_addr,
            controller,
        )
    work = _shared_workdir()
    step_token = _new_step_token()
    owned = _owned_step("spur", spur_bin=spur_bin, token=step_token)
    stdout_log, stderr_log = _open_subprocess_logs(config)
    keep_work = False
    try:
        payload = worker_training_config(config)
        config_path = work / "worker_config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        argv, env = build_spur_launch(
            spur_bin=spur_bin,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            master_addr=master_addr,
            master_port=master_port,
            config_path=config_path,
            results_dir=work,
            step_token=step_token,
            device=config.get("device", "auto"),
        )
        log.info(
            "spur launch MASTER_ADDR=%s WORLD_SIZE=%s argv=%s",
            env.get("MASTER_ADDR"),
            env.get("WORLD_SIZE"),
            argv,
        )
        proc: subprocess.Popen[str] | None = None
        pgid: int | None = None
        try:
            proc = subprocess.Popen(
                argv,
                env=env,
                start_new_session=True,
                stdout=stdout_log,
                stderr=stderr_log,
            )
            pgid = _session_pgid(proc)
            proc.wait()
        except KeyboardInterrupt:
            raise
        finally:
            # Publish the cancel file and name the step before signalling the
            # local client. Killing that client does not stop a buffered step.
            # The directory is kept from here on, and released only once the
            # step is no longer listed. A second Ctrl-C during the wait keeps it.
            try:
                if proc is not None and _step_may_outlive_client(proc, pgid, owned):
                    keep_work = True
                    _stop_owned_step(owned, work / _STEP_CANCEL_NAME)
                    if _wait_owned_step_gone(owned):
                        keep_work = False
                    else:
                        log.warning(
                            "keeping %s with its cancel file; the owned step was not "
                            "confirmed stopped. Remove it once no task of this launch runs.",
                            work,
                        )
            finally:
                if proc is not None and _needs_group_cleanup(proc, pgid):
                    _terminate_process_tree(proc, pgid=pgid)
        exit_code = proc.returncode if proc is not None and isinstance(proc.returncode, int) else 1
        rank0, read_error = _read_rank0_result(work)
        result = verdict_from_workers(exit_code, rank0, launcher="spur")
        if read_error is not None:
            result.failure_details.append(
                {"problems": ["worker_result_unreadable"], "error": read_error}
            )
            result.passed = False
            result.failure_count += 1
        return result
    finally:
        _close_subprocess_logs(stdout_log, stderr_log)
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)


def monarch_step_prefix(*, scheduler: str, nodes: int, spur_bin: str = "spur") -> list[str]:
    """``srun`` arguments for one task on each of ``nodes`` hosts of the current allocation.

    ``slurm`` attaches to ``MONARCH_SLURM_JOB`` or ``SLURM_JOB_ID`` with
    ``srun --jobid <id> --overlap``. ``spur`` runs ``<spur_bin> srun`` from a
    shell inside a Spur allocation (``SPUR_JOB_ID`` set), where it becomes a
    step of that allocation. Spur's ``--jobid --overlap`` would start one task
    on one node, so it is not used. ``--ntasks`` is explicit in both: a step
    otherwise inherits the allocation's task count.
    """
    counts = [f"-N{nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1"]
    if scheduler == "spur":
        spur_job = os.environ.get("SPUR_JOB_ID", "").strip()
        if not spur_job:
            raise RuntimeError(
                "scheduler 'spur' runs Monarch inside a Spur allocation. Run aorta from "
                "the shell that `spur salloc` opens, so SPUR_JOB_ID is set. "
                "This workload does not request the allocation."
            )
        return [spur_bin, "srun", *counts]
    if scheduler != "slurm":
        raise ValueError(f"scheduler must be 'slurm' or 'spur', got {scheduler!r}")
    explicit = os.environ.get("MONARCH_SLURM_JOB", "").strip()
    job_id = explicit or os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RuntimeError(
            "launcher 'monarch' attaches to an existing allocation. "
            "Set SLURM_JOB_ID or MONARCH_SLURM_JOB. This workload does not submit Slurm jobs."
        )
    if not explicit and os.environ.get("SPUR_JOB_ID", "").strip() == job_id:
        raise RuntimeError(
            f"SLURM_JOB_ID {job_id} is a Spur allocation, not a Slurm one. "
            "Set scheduler: spur in the recipe, or set MONARCH_SLURM_JOB to the Slurm job id."
        )
    return ["srun", "--jobid", job_id, "--overlap", *counts]


def launch_monarch(
    config: dict[str, Any],
    *,
    nodes: int,
    gpus_per_node: int,
    scheduler: str = "slurm",
    spur_bin: str = "spur",
) -> WorkloadResult:
    """Run the training actor under a Monarch proc mesh on an existing allocation.

    Does not request an allocation. With ``scheduler="slurm"``,
    ``SLURM_JOB_ID`` or ``MONARCH_SLURM_JOB`` names the Slurm allocation. With
    ``scheduler="spur"``, the controller runs inside a Spur allocation and the
    workers are ``spur srun`` steps of it. Those workers watch
    ``AORTA_STEP_CANCEL`` under ``AORTA_SHARED_DIR``, because Spur rejects
    ``scancel job.step``. The host mesh is shut down on the training event
    loop. The step is cancelled only when that shutdown does not finish, and
    only as ``job.step``. The cancel directory stays until the step is
    confirmed gone.
    """
    imported = _import_monarch()
    actor_mod, spmd_mod, bootstrap, service_env, alloc_ids, serialize_ids, proc_addrs = imported
    # Import after Monarch itself imports, so a missing install still raises
    # the not-installed error. The class lives in its own module so workers
    # unpickle ``aorta.workloads.monarch_trainer.Trainer``.
    from aorta.workloads.monarch_trainer import Trainer

    prefix = monarch_step_prefix(scheduler=scheduler, nodes=nodes, spur_bin=spur_bin)
    ip = controller_ipv4()
    actor_mod.enable_transport(f"tcp://{ip}:0")
    port = _tcp_port(os.environ.get("MONARCH_PORT", "22222"), name="MONARCH_PORT")
    placed = _monarch_nodes(prefix, nodes)
    hostnames = [host for _node, host in placed]
    proc_ids = alloc_ids(nodes)
    env = {**os.environ, service_env: serialize_ids(proc_ids)}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    step_token = _new_step_token()
    owned = _owned_step(scheduler, spur_bin=spur_bin, token=step_token)
    cancel_dir: Path | None = None
    cancel_file: Path | None = None
    if scheduler == "spur":
        # Spur has no step-scoped scancel. The workers watch this file, which
        # has to live on a directory every node can see.
        cancel_dir = _shared_workdir()
        cancel_file = cancel_dir / _STEP_CANCEL_NAME
        env["AORTA_STEP_CANCEL"] = str(cancel_file)
    worker_cmd = _marked_command(step_token, [sys.executable, "-c", bootstrap(port, None)])
    # Spur parses --kill-on-bad-exit but does not implement it.
    kill_flag = ["--kill-on-bad-exit=1"] if scheduler == "slurm" else []
    # Slurm lists a step as ``Name=<executable>``, which would be ``bash``.
    # The token has to be the step name for the cancel lookup to find it.
    name_flag = [f"--job-name={step_token}"] if scheduler == "slurm" else []
    # The probe is its own step. In an allocation larger than ``nodes`` the
    # worker step could land elsewhere, so it is pinned to the probed nodes.
    node_flag = ["-w", ",".join(node for node, _host in placed)]
    argv = [*prefix, *kill_flag, *name_flag, *node_flag, *worker_cmd]
    log.info("monarch controller tcp://%s scheduler=%s hosts=%s", ip, scheduler, hostnames)
    hosts: Any = None
    workers: subprocess.Popen[str] | None = None
    worker_pgid: int | None = None
    # True until a mesh exists and its shutdown does not finish. An interrupt
    # during shutdown leaves this false so the step is still cancelled.
    mesh_ok = True
    keep_cancel = False
    stdout_log, stderr_log = _open_subprocess_logs(config)
    try:
        workers = subprocess.Popen(
            argv,
            env=env,
            start_new_session=True,
            stdout=stdout_log,
            stderr=stderr_log,
        )
        worker_pgid = _session_pgid(workers)
        hosts = actor_mod.attach_to_workers(
            name="trainers",
            ca="trust_all_connections",
            workers=proc_addrs([f"tcp://{host}:{port}" for host in hostnames], proc_ids),
        )
        actor_config = worker_training_config(config)

        async def _train_and_close() -> Any:
            nonlocal mesh_ok
            try:
                return await _monarch_train(
                    hosts, spmd_mod, Trainer, gpus_per_node, actor_config
                )
            finally:
                # Same loop as training. A second ``asyncio.run`` would build
                # a new runtime after Monarch's services were bound to this one.
                closed = False
                try:
                    await _await_mesh_shutdown(hosts)
                    closed = True
                except Exception as exc:
                    log.warning("monarch host mesh shutdown failed: %s", exc)
                finally:
                    mesh_ok = closed

        raw = asyncio.run(_train_and_close())
        return _verdict_from_monarch(raw)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        log.warning("monarch launch failed: %s", exc, exc_info=True)
        return WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[
                {
                    "problems": ["monarch_launch"],
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            ],
            metrics={"launcher": "monarch", "world_size": None, "final_loss": None},
        )
    finally:
        # A failed shutdown, or a step that started before the mesh existed,
        # cancels only that step. The allocation is not a cancel target.
        try:
            if not mesh_ok or (hosts is None and workers is not None):
                # Kept until the step is confirmed gone, including when a
                # second Ctrl-C interrupts the wait below.
                keep_cancel = cancel_file is not None
                _stop_owned_step(owned, cancel_file)
                if cancel_file is not None:
                    if _wait_owned_step_gone(owned):
                        keep_cancel = False
                    else:
                        log.warning(
                            "keeping %s with its cancel file; the owned step was not "
                            "confirmed stopped. Remove it once no task of this launch runs.",
                            cancel_dir,
                        )
        finally:
            if workers is not None:
                _terminate_process_tree(workers, pgid=worker_pgid)
            _close_subprocess_logs(stdout_log, stderr_log)
            if cancel_dir is not None and not keep_cancel:
                shutil.rmtree(cancel_dir, ignore_errors=True)


def verdict_from_workers(
    exit_code: int,
    rank0: dict[str, Any] | None,
    *,
    launcher: str,
) -> WorkloadResult:
    """Pass only when the workers exited 0 and rank 0's result passed.

    Metrics keep rank 0's ``world_size`` and ``final_loss`` and always
    record ``launcher``. A missing rank-0 file fails the trial.
    """
    if isinstance(rank0, dict):
        result = workload_result_from_mapping(rank0, launcher=launcher)
    else:
        result = WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[{"problems": ["missing_rank0_result"]}],
            metrics={"launcher": launcher, "world_size": None, "final_loss": None},
        )
    if exit_code == 0:
        return result
    details = list(result.failure_details)
    details.append({"problems": ["worker_exit"], "exit_code": exit_code})
    metrics = dict(result.metrics)
    metrics["launcher"] = launcher
    return WorkloadResult(
        passed=False,
        failure_count=result.failure_count + 1,
        first_failure_iteration=result.first_failure_iteration,
        failure_details=details,
        total_iterations=result.total_iterations,
        step_times_ms=list(result.step_times_ms),
        elapsed_sec=result.elapsed_sec,
        metrics=metrics,
        main_work_started=result.main_work_started,
        executed_iterations=result.executed_iterations,
        configured_iterations=result.configured_iterations,
    )


def workload_result_from_mapping(data: dict[str, Any], *, launcher: str) -> WorkloadResult:
    """Rebuild a :class:`WorkloadResult` from a rank's serialized result."""
    metrics_raw = data.get("metrics")
    metrics = dict(metrics_raw) if isinstance(metrics_raw, dict) else {}
    metrics["launcher"] = launcher
    if "world_size" not in metrics:
        metrics["world_size"] = None
    if "final_loss" not in metrics:
        metrics["final_loss"] = None
    details_raw = data.get("failure_details")
    if isinstance(details_raw, list):
        details = [item for item in details_raw if isinstance(item, dict)]
    else:
        details = []
    failure_count = data.get("failure_count")
    if isinstance(failure_count, bool) or not isinstance(failure_count, int):
        failure_count = 0 if data.get("passed") is True else 1
    first = data.get("first_failure_iteration")
    if first is not None and (isinstance(first, bool) or not isinstance(first, int)):
        first = None
    step_times_raw = data.get("step_times_ms")
    step_times: list[float] = []
    if isinstance(step_times_raw, list):
        for item in step_times_raw:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                continue
            step_times.append(float(item))
    elapsed = data.get("elapsed_sec")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        elapsed = 0.0
    total = data.get("total_iterations")
    if isinstance(total, bool) or not isinstance(total, int):
        total = 0
    executed = _optional_int(data.get("executed_iterations"))
    configured = _optional_int(data.get("configured_iterations"))
    main_work = data.get("main_work_started")
    if not isinstance(main_work, bool):
        main_work = None
    return WorkloadResult(
        passed=data.get("passed") is True,
        failure_count=failure_count,
        first_failure_iteration=first,
        failure_details=details,
        total_iterations=total,
        step_times_ms=step_times,
        elapsed_sec=float(elapsed),
        metrics=metrics,
        main_work_started=main_work,
        executed_iterations=executed,
        configured_iterations=configured,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _worker_env(
    *,
    master_addr: str,
    master_port: int,
    world_size: int,
    gpus_per_node: int,
) -> dict[str, str]:
    env = dict(os.environ)
    for key in _IDENTITY_ENV:
        env.pop(key, None)
    env["AORTA_INNER"] = "1"
    if master_addr:
        env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = str(world_size)
    env["AORTA_GPUS_PER_NODE"] = str(gpus_per_node)
    return env


def _tcp_port(raw: str, *, name: str) -> int:
    """Return ``raw`` as a TCP port.

    Callers pass an explicit port. Probing an ephemeral port and closing the
    socket leaves a gap before workers bind it, so this function does not
    pick one.
    """
    if not raw.isdigit():
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    port = int(raw)
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be in 1..65535, got {port}")
    return port


def _shared_workdir() -> Path:
    """A fresh directory under ``AORTA_SHARED_DIR`` for this launch's result.

    The directory is explicit so rank 0 and the controller name the same
    path. A node-local default such as ``tempfile``'s ``/tmp`` would make a
    successful rank-0 write look like a missing result.
    """
    raw = os.environ.get("AORTA_SHARED_DIR", "").strip()
    if not raw:
        raise RuntimeError(
            "this launch needs AORTA_SHARED_DIR set to a directory that "
            "every node in the allocation can read and write. Rank 0 writes "
            "rank0.json there. Node-local directories such as /tmp are not "
            "visible to a controller on another host."
        )
    root = Path(raw)
    if not root.is_dir():
        raise RuntimeError(f"AORTA_SHARED_DIR {raw!r} is not a directory")
    resolved = root.resolve()
    for prefix in _LOCAL_RESULT_PREFIXES:
        try:
            resolved.relative_to(prefix)
        except ValueError:
            continue
        log.warning(
            "AORTA_SHARED_DIR %s is node-local; rank 0's result will be "
            "invisible to a controller on another host",
            resolved,
        )
        break
    return Path(tempfile.mkdtemp(prefix="aorta-tiny-transformer-", dir=resolved))


def _read_rank0_result(output_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return rank 0's published result, and a read error when the file is unusable.

    Only ``rank0.json`` counts. A directory of earlier trial files must not
    supply another cell's verdict.
    """
    path = output_dir / "rank0.json"
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read worker result %s: %s", path, exc)
        return None, f"{path.name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path.name}: expected a JSON object"
    result = payload.get("result")
    if isinstance(result, dict):
        return result, None
    if isinstance(payload.get("passed"), bool):
        return payload, None
    return None, f"{path.name}: no rank-0 result"


def _import_monarch() -> tuple[Any, ...]:
    """Import Monarch. Any failure says it is not installed."""
    try:
        import monarch.actor as actor_mod
        import monarch.spmd as spmd_mod
        from monarch._src.job._slurm_batch import _worker_bootstrap
        from monarch._src.job.service_identity import (
            SERVICE_PROC_IDS_ENV,
            allocate_service_proc_ids,
            serialize_service_proc_ids,
            service_proc_addrs,
        )
    except Exception as exc:
        raise RuntimeError(_MONARCH_MISSING.format(exc=exc)) from exc
    else:
        return (
            actor_mod,
            spmd_mod,
            _worker_bootstrap,
            SERVICE_PROC_IDS_ENV,
            allocate_service_proc_ids,
            serialize_service_proc_ids,
            service_proc_addrs,
        )


async def _monarch_train(
    hosts: Any,
    spmd_mod: Any,
    trainer_cls: type,
    gpus_per_node: int,
    config: dict[str, Any],
) -> Any:
    await hosts.initialized
    procs = hosts.spawn_procs(per_host={"gpus": gpus_per_node})
    await procs.logging_option(stream_to_client=True)
    await spmd_mod.setup_torch_elastic_env_async(procs)
    trainers = procs.spawn("trainer", trainer_cls)
    return await trainers.train.call(config)


def _verdict_from_monarch(raw: Any) -> WorkloadResult:
    rows = _rank_rows(raw)
    if not rows:
        return WorkloadResult(
            passed=False,
            failure_count=1,
            failure_details=[{"problems": ["missing_rank0_result"]}],
            metrics={"launcher": "monarch", "world_size": None, "final_loss": None},
        )
    rank0 = rows[0]
    for item in rows:
        metrics = item.get("metrics")
        if isinstance(metrics, dict) and metrics.get("rank") == 0:
            rank0 = item
            break
    result = workload_result_from_mapping(rank0, launcher="monarch")
    if any(item.get("passed") is not True for item in rows):
        result = WorkloadResult(
            passed=False,
            failure_count=max(result.failure_count, 1),
            first_failure_iteration=result.first_failure_iteration,
            failure_details=list(result.failure_details) or [{"problems": ["rank_result_failed"]}],
            total_iterations=result.total_iterations,
            step_times_ms=list(result.step_times_ms),
            elapsed_sec=result.elapsed_sec,
            metrics=dict(result.metrics),
            main_work_started=result.main_work_started,
            executed_iterations=result.executed_iterations,
            configured_iterations=result.configured_iterations,
        )
    return result


def _rank_rows(raw: Any) -> list[dict[str, Any]]:
    items = raw.items if isinstance(raw, dict) else getattr(raw, "items", None)
    if not callable(items):
        raise TypeError(
            f"Monarch train() returned {type(raw).__name__}, expected a mapping of rank results"
        )
    rows: list[dict[str, Any]] = []
    for _point, value in items():
        if not isinstance(value, dict):
            raise TypeError(f"Monarch rank result must be a dict, got {type(value).__name__}")
        rows.append(value)
    return rows


def _monarch_nodes(prefix: list[str], nodes: int) -> list[tuple[str, str]]:
    """Scheduler node name and advertised hostname of each probed node.

    ``prefix`` comes from :func:`monarch_step_prefix`. Order follows
    ``SLURM_NODEID``, which Spur also sets. The node name is
    ``SLURMD_NODENAME``, which both schedulers set per task, so the worker
    step can be pinned with ``-w``. This runs a step of the current
    allocation; it does not request a new one.
    """
    script = (
        'echo "$SLURM_NODEID ${SLURMD_NODENAME:-} '
        '$(python3 -c "import socket; print(socket.gethostname())")"'
    )
    argv = [*prefix, "bash", "-c", script]
    try:
        out = subprocess.check_output(argv, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not list hostnames with {' '.join(prefix)}: {exc}") from exc
    rows: list[tuple[int, str, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[0].isdigit() or "," in parts[1]:
            continue
        rows.append((int(parts[0]), parts[1], parts[2]))
    rows.sort()
    if len(rows) != nodes or len({node for _, node, _ in rows}) != nodes:
        raise RuntimeError(
            f"srun returned {len(rows)} distinct nodes with SLURMD_NODENAME for "
            f"{nodes} nodes: {out!r}"
        )
    return [(node, host) for _, node, host in rows]


def _session_pgid(proc: subprocess.Popen[Any]) -> int | None:
    """Process group of a ``start_new_session`` child, saved while the leader is alive.

    After the leader exits, ``getpgid`` on its pid fails even when children
    remain in the group. The caller keeps this value. A pid that is not its
    own process-group leader is not our session (a recycled pid, or a test
    double) and is ignored so cleanup cannot signal an unrelated process.
    """
    pid = getattr(proc, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return None
    except OSError as exc:
        log.warning("getpgid(%s) failed: %s", pid, exc)
        return None
    if pgid != pid or pgid == os.getpgrp():
        return None
    return pgid


def _group_has_members(pgid: int) -> bool:
    """True when any process in ``pgid`` still exists, including after the leader exits."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        log.warning("killpg(%s, 0) failed: %s", pgid, exc)
        return False
    return True


def _needs_group_cleanup(proc: subprocess.Popen[Any], pgid: int | None) -> bool:
    """True when the leader or any of its group is still running."""
    if proc.poll() is None:
        return True
    return pgid is not None and _group_has_members(pgid)


def _wait_for_group(proc: subprocess.Popen[Any], pgid: int | None, timeout: float) -> None:
    """Block until the leader and its group are gone, or ``timeout`` elapses.

    ``Popen.wait`` returns when the leader exits. A child that ignores
    SIGTERM stays in the group, so the wait continues until ``killpg``
    reports that the group is empty.
    """
    deadline = time.monotonic() + timeout
    while _needs_group_cleanup(proc, pgid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        slice_s = min(0.2, remaining)
        if proc.poll() is None:
            try:
                proc.wait(timeout=slice_s)
            except subprocess.TimeoutExpired:
                pass
            except OSError as exc:
                log.warning(
                    "waiting for child pid %s failed: %s",
                    getattr(proc, "pid", "?"),
                    exc,
                )
                return
        else:
            time.sleep(slice_s)


def _terminate_process_tree(proc: subprocess.Popen[Any], pgid: int | None = None) -> None:
    """SIGTERM the child's process group, then SIGKILL, then reap.

    ``pgid`` is the group captured while the leader was alive. Waiting only
    on the leader would return while a child that ignores SIGTERM still holds
    GPUs or ports. ``KeyboardInterrupt`` during the grace period still
    SIGKILLs the group and is re-raised. ``ProcessLookupError`` is the group
    already being gone. Any other ``OSError`` is logged: the tree may still
    be alive.
    """
    if pgid is None:
        pgid = _session_pgid(proc)
    elif pgid == os.getpgrp():
        pgid = None

    def _send(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                pass
            except OSError as exc:
                log.warning(
                    "killpg(%s, %s) failed: %s; the child process group may still be running",
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
                "send_signal(%s) to child pid %s failed: %s",
                signal.Signals(sig).name,
                getattr(proc, "pid", "?"),
                exc,
            )

    def _force_kill() -> None:
        if proc.poll() is not None and pgid is not None and _group_has_members(pgid):
            log.warning(
                "process group %s still has members after the leader exited; sending SIGKILL",
                pgid,
            )
        _send(signal.SIGKILL)
        try:
            _wait_for_group(proc, pgid, _REAP_AFTER_KILL_SEC)
        except KeyboardInterrupt:
            pass
        if _needs_group_cleanup(proc, pgid):
            log.warning("process group %s still has members after SIGKILL", pgid)

    _send(signal.SIGTERM)
    try:
        _wait_for_group(proc, pgid, _TERMINATE_GRACE_SEC)
    except KeyboardInterrupt:
        _force_kill()
        raise
    if not _needs_group_cleanup(proc, pgid):
        return
    _force_kill()


__all__ = [
    "SPUR_WORKER_SCRIPT",
    "build_spur_launch",
    "controller_ipv4",
    "is_ipv4",
    "is_launched_rank",
    "spawning_controller",
    "launch_monarch",
    "launch_spur",
    "monarch_step_prefix",
    "run_training_rank",
    "run_worker_rank",
    "verdict_from_workers",
    "worker_training_config",
]
