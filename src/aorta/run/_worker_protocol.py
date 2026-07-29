"""Versioned JSON protocol for fresh-process trial workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1


class WorkerProtocolError(RuntimeError):
    """Raised when a trial-worker request or response is malformed."""


def read_envelope(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError(f"cannot read worker envelope {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkerProtocolError(
            f"worker envelope {path} must be a JSON object, got {type(data).__name__}"
        )
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise WorkerProtocolError(
            f"worker envelope {path} has protocol_version="
            f"{data.get('protocol_version')!r}; expected {PROTOCOL_VERSION}"
        )
    return data


def write_envelope_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def validate_identity(
    data: dict[str, Any],
    *,
    nonce: str,
    trial_id: str,
) -> None:
    if data.get("nonce") != nonce:
        raise WorkerProtocolError("trial-worker response nonce does not match request")
    if data.get("trial_id") != trial_id:
        raise WorkerProtocolError(
            f"trial-worker response trial_id={data.get('trial_id')!r}; expected {trial_id!r}"
        )


__all__ = [
    "PROTOCOL_VERSION",
    "WorkerProtocolError",
    "read_envelope",
    "validate_identity",
    "write_envelope_atomic",
]
