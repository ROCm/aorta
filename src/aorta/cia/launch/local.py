"""Run a CIA job as a detached process on the current workstation.

The local backend deliberately preserves the Slurm backend's observable
contract: one launch script, one combined log, a durable terminal state, and a
native job handle that cancellation can use.  The handle includes the Linux
process start time as well as its PID so a later cancellation never signals an
unrelated process after PID reuse.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

from aorta.cia.launch.cluster import (
    _ENV_NAME,
    containerize,
    forwarded_env,
    venv_activate_path,
)

LOCAL_JOB_PREFIX = "local:"
LOCAL_STATUS_FILE = "local.status.json"
LOCAL_PROCESS_FILE = "local.process.json"
ALREADY_FINISHED = "local process finished before cancellation"


def _status_path(job_dir: Path, launch_token: str = "") -> Path:
    name = f"local.status.{launch_token}.json" if launch_token else LOCAL_STATUS_FILE
    return job_dir / name


def _process_path(job_dir: Path, launch_token: str = "") -> Path:
    name = f"local.process.{launch_token}.json" if launch_token else LOCAL_PROCESS_FILE
    return job_dir / name


def _process_stat(pid: int) -> tuple[str, int, int, str] | None:
    """Return start ticks, process group, session, and state from Linux procfs."""
    try:
        # The command name is parenthesized and may contain spaces.  Splitting
        # only after its final ')' keeps pgrp/session/starttime at 2/3/19.
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        fields = tail.split()
        return fields[19], int(fields[2]), int(fields[3]), fields[0]
    except (OSError, IndexError, ValueError):
        return None


def _process_start_ticks(pid: int) -> str:
    """Return Linux's process start-time field, or ``""`` when it is gone."""
    stat = _process_stat(pid)
    if stat is None:
        return ""
    return stat[0]


def _parse_job_id(job_id: str) -> tuple[int, str, str] | None:
    if not job_id.startswith(LOCAL_JOB_PREFIX):
        return None
    try:
        parts = job_id.split(":", 3)
        if len(parts) == 3:
            _prefix, pid, started = parts
            launch_token = ""
        elif len(parts) == 4:
            _prefix, pid, started, launch_token = parts
        else:
            return None
        return int(pid), started, launch_token
    except (TypeError, ValueError):
        return None


def is_local_job_id(job_id: str) -> bool:
    return _parse_job_id(job_id) is not None


def _same_process(pid: int, started: str) -> bool:
    current = _process_start_ticks(pid)
    return bool(current and current == started)


def _group_members(process_group: int, session: int) -> dict[int, str]:
    """Live members of one process group, keyed by PID and start ticks."""
    members: dict[int, str] = {}
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        stat = _process_stat(pid)
        if stat is None:
            continue
        started, group, process_session, state = stat
        if group == process_group and process_session == session and state != "Z":
            members[pid] = started
    return members


def _signal_group_members(process_group: int, session: int, signum: int) -> str:
    """Signal current members after rechecking each stable process identity."""
    for pid, started in _group_members(process_group, session).items():
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        except OSError as exc:
            return f"could not open local process {pid}: {exc}"
        try:
            # Opening the pidfd pins one process even if its numeric PID is
            # immediately reused. Validate identity after opening, then signal
            # through the fd so there is no check-to-kill reuse window.
            stat = _process_stat(pid)
            if stat is None:
                continue
            current_started, group, process_session, _state = stat
            if (current_started, group, process_session) != (
                started,
                process_group,
                session,
            ):
                continue
            signal.pidfd_send_signal(pidfd, signum)
        except OSError as exc:
            return f"could not signal local process {pid}: {exc}"
        finally:
            os.close(pidfd)
    return ""


def _local_node_requested(node: str) -> bool:
    if not node:
        return True
    local_names = {
        "local",
        "localhost",
        "127.0.0.1",
        socket.gethostname(),
        socket.getfqdn(),
    }
    return node in local_names


def build_local_script(
    *,
    command: str,
    status_path: Path,
    launch_token: str,
    working_dir: str = "",
    env_vars: dict[str, str] | None = None,
    tolerate_nonzero: bool = False,
) -> str:
    """Render a self-recording shell script for a workstation job."""
    resolved_env = {
        **forwarded_env(),
        **{key: str(value) for key, value in (env_vars or {}).items()},
    }
    for key in resolved_env:
        if _ENV_NAME.fullmatch(key) is None:
            raise ValueError(f"invalid environment variable name: {key!r}")

    status = shlex.quote(str(status_path))
    body = [
        "set -uo pipefail",
        f"_cia_status={status}",
        "_cia_status_written=0",
        "_cia_write_status() {",
        "  _cia_rc=$1",
        '  _cia_state="$2"',
        '  _cia_tmp="${_cia_status}.$$"',
        (
            r"""  printf '{"launch_token":"%s","state":"%s","exit_code":%s}\n' """
            f"{shlex.quote(launch_token)} "
            r'''"$_cia_state" "$_cia_rc" >"$_cia_tmp"'''
        ),
        '  mv -f "$_cia_tmp" "$_cia_status"',
        "  _cia_status_written=1",
        "}",
        "_cia_on_exit() {",
        "  _cia_rc=$?",
        '  [ "$_cia_status_written" -eq 1 ] && return',
        '  if [ "$_cia_rc" -eq 0 ]; then',
        '    _cia_write_status "$_cia_rc" COMPLETED',
        "  else",
        '    _cia_write_status "$_cia_rc" FAILED',
        "  fi",
        "}",
        "trap _cia_on_exit EXIT",
        "trap 'exit 143' TERM INT HUP",
        'echo "[cia] node=$(hostname) local_pid=$$"',
    ]

    activate = venv_activate_path()
    if activate:
        body.append(f"source {shlex.quote(activate)}")
        body.append('echo "[cia] venv=$VIRTUAL_ENV aorta=$(command -v aorta || echo MISSING)"')
    if working_dir:
        body.append(f"cd {shlex.quote(working_dir)} || exit 1")
    for key, value in resolved_env.items():
        body.append(f"export {key}={shlex.quote(value)}")

    body.extend(
        [
            # A raw command may itself be ``exit N``. Keep that from bypassing
            # the wrapper's durable status write and tolerate-nonzero policy.
            # Separate delimiter lines also preserve trailing comments and
            # heredoc terminators in the command exactly as the caller wrote them.
            "(",
            containerize(command, working_dir, resolved_env),
            ")",
            "_cia_workload_rc=$?",
            'echo "[cia] workload exit=$_cia_workload_rc"',
        ]
    )
    if tolerate_nonzero:
        body.extend(
            [
                '[ "$_cia_workload_rc" -ne 0 ] && '
                'echo "[cia] non-zero tolerated (expected by caller)"',
                "exit 0",
            ]
        )
    else:
        body.append("exit $_cia_workload_rc")

    return "\n".join(["#!/bin/bash", *body, ""])


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Reap a child while allowing it to outlive the launching request."""
    try:
        process.wait()
    except Exception:
        pass


def submit_local(
    *,
    command: str,
    job_name: str,
    log_path: str,
    script_path: Path,
    working_dir: str = "",
    env_vars: dict[str, str] | None = None,
    node: str = "",
    tolerate_nonzero: bool = False,
) -> tuple[str, str]:
    """Start *command* locally. Returns ``(local_job_id, error_message)``."""
    if not _local_node_requested(node):
        return (
            "",
            f"local execution cannot pin work to {node!r}; "
            "remove --node or select the Slurm backend",
        )

    # The script changes directory before running the workload. Resolve every
    # runner-owned path first so that neither execution nor sidecar placement
    # depends on that later cd. An absolute argv also makes Popen execute this
    # file rather than search PATH when the caller supplied just "launch.sh".
    script_path = script_path.expanduser().resolve()
    log_file_path = Path(log_path).expanduser().resolve()
    launch_token = uuid.uuid4().hex
    status_path = _status_path(script_path.parent, launch_token)
    process_path = _process_path(script_path.parent, launch_token)
    try:
        script = build_local_script(
            command=command,
            status_path=status_path,
            launch_token=launch_token,
            working_dir=working_dir,
            env_vars=env_vars,
            tolerate_nonzero=tolerate_nonzero,
        )
        script_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.unlink(missing_ok=True)
        process_path.unlink(missing_ok=True)
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)
    except Exception as exc:
        return "", f"could not write local launch script {script_path}: {exc}"

    try:
        with log_file_path.open("ab", buffering=0) as log_file:
            process = subprocess.Popen(
                [str(script_path)],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except Exception as exc:
        return "", f"local process launch failed: {exc}"

    started = ""
    for _ in range(50):
        started = _process_start_ticks(process.pid)
        if started:
            break
        if process.poll() is not None:
            break
        time.sleep(0.002)
    if not started:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return "", f"could not record identity for local process {process.pid}"

    job_id = f"{LOCAL_JOB_PREFIX}{process.pid}:{started}:{launch_token}"
    temporary = process_path.with_name(f".{process_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "pid": process.pid,
                    "process_group_id": process.pid,
                    "session_id": process.pid,
                    "process_start_ticks": started,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, process_path)
    except OSError as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return "", f"could not persist identity for local process {process.pid}: {exc}"
    finally:
        temporary.unlink(missing_ok=True)

    threading.Thread(target=_reap, args=(process,), daemon=True).start()
    return job_id, ""


def local_state(job_id: str, job_dir: Path) -> str:
    """Return a Slurm-shaped state for a local process."""
    identity = _parse_job_id(job_id)
    if identity is None:
        return "UNKNOWN"
    pid, started, launch_token = identity
    try:
        payload = json.loads(_status_path(job_dir, launch_token).read_text(encoding="utf-8"))
        if launch_token and payload.get("launch_token") != launch_token:
            raise ValueError("local status belongs to another launch")
        state = str(payload.get("state") or "").upper()
        if state in {"COMPLETED", "FAILED", "CANCELLED"}:
            return state
    except (OSError, ValueError):
        pass

    return "RUNNING" if _same_process(pid, started) else "FAILED"


def record_local_cancellation(
    job_id: str,
    job_dir: Path,
    *,
    preserve_terminal: bool = True,
) -> None:
    """Persist cancellation after the exact local process group has stopped."""
    identity = _parse_job_id(job_id)
    if identity is None:
        return
    _pid, _started, launch_token = identity
    path = _status_path(job_dir, launch_token)
    if preserve_terminal:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (not launch_token or existing.get("launch_token") == launch_token) and str(
                existing.get("state") or ""
            ).upper() in {"COMPLETED", "FAILED"}:
                return
        except (OSError, ValueError):
            pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "launch_token": launch_token,
                    "state": "CANCELLED",
                    "exit_code": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        # Cancellation already succeeded. A damaged sidecar must not turn that
        # into an exception before triage can mark job.json terminal too.
        pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def cancel_local(job_id: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Terminate one local process group without risking a reused PID."""
    identity = _parse_job_id(job_id)
    if identity is None:
        return False, f"invalid local job id: {job_id!r}"
    pid, started, _launch_token = identity
    leader = _process_stat(pid)
    if leader is not None and leader[0] != started:
        return False, f"local process identity {pid} has been reused"
    if not _group_members(pid, pid):
        return True, ALREADY_FINISHED

    error = _signal_group_members(pid, pid, signal.SIGTERM)
    if error:
        return False, error

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_members(pid, pid):
            return True, ""
        error = _signal_group_members(pid, pid, signal.SIGTERM)
        if error:
            return False, error
        time.sleep(0.05)

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _group_members(pid, pid):
            return True, ""
        error = _signal_group_members(pid, pid, signal.SIGKILL)
        if error:
            return False, error
        time.sleep(0.05)
    survivors = sorted(_group_members(pid, pid))
    return False, f"local process group {pid} still has members after SIGKILL: {survivors}"
