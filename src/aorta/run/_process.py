"""Process launcher for isolated trial workers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from aorta.run._worker_protocol import (
    PROTOCOL_VERSION,
    WorkerProtocolError,
    read_envelope,
    validate_identity,
    write_envelope_atomic,
)


class TrialWorkerError(RuntimeError):
    """Raised when an isolated trial worker cannot produce a valid result."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.completed_results: tuple[Any, ...] = ()


def _terminate_worker(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            proc.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            try:
                proc.kill()
            except OSError:
                pass
            return
    killpg = os.killpg
    sigkill = signal.SIGKILL
    try:
        killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            killpg(proc.pid, sigkill)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _launch_trial_worker(
    payload: dict[str, Any],
    *,
    child_env: dict[str, str],
    trial_id: str,
) -> dict[str, Any]:
    """Launch one fresh interpreter and return its validated response."""
    nonce = uuid.uuid4().hex
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "nonce": nonce,
        "trial_id": trial_id,
        **payload,
    }
    with tempfile.TemporaryDirectory(prefix="aorta-trial-worker-") as tmp_dir:
        root = Path(tmp_dir)
        request_path = root / "request.json"
        response_path = root / "response.json"
        write_envelope_atomic(request_path, request)
        try:
            request_path.chmod(0o600)
        except OSError:
            pass

        creationflags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "aorta.run._trial_worker",
                    str(request_path),
                    str(response_path),
                ],
                env=child_env,
                start_new_session=start_new_session,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise TrialWorkerError(f"could not start trial worker: {exc}") from exc
        previous_handlers: dict[int, Any] = {}

        def _forward_termination(signum: int, _frame: Any) -> None:
            _terminate_worker(proc)
            raise SystemExit(128 + signum)

        for sig_name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    previous_handlers[sig] = signal.signal(
                        sig,
                        _forward_termination,
                    )
                except (ValueError, OSError):
                    pass
        try:
            try:
                returncode = proc.wait()
            except BaseException:
                _terminate_worker(proc)
                raise
        finally:
            for sig, handler in previous_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

        try:
            response = read_envelope(response_path)
            validate_identity(response, nonce=nonce, trial_id=trial_id)
        except WorkerProtocolError as exc:
            raise TrialWorkerError(
                f"trial worker exited {returncode} without a valid response: {exc}"
            ) from exc

        if returncode != 0 or response.get("kind") != "result":
            error = response.get("error")
            if isinstance(error, dict):
                detail = (
                    f"{error.get('phase', 'worker')} "
                    f"{error.get('type', 'Error')}: {error.get('message', '')}"
                )
            else:
                detail = repr(error)
            raise TrialWorkerError(f"trial worker failed with exit code {returncode}: {detail}")
        result = response.get("trial_result")
        if not isinstance(result, dict):
            raise TrialWorkerError("trial worker response lacks a trial_result object")
        return result


def launch_trial_worker(
    payload: dict[str, Any],
    *,
    child_env: dict[str, str],
    trial_id: str,
) -> dict[str, Any]:
    try:
        return _launch_trial_worker(
            payload,
            child_env=child_env,
            trial_id=trial_id,
        )
    except TrialWorkerError:
        raise
    except OSError as exc:
        raise TrialWorkerError(f"could not prepare trial worker request: {exc}") from exc


__all__ = ["TrialWorkerError", "launch_trial_worker"]
