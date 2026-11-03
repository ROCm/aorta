"""Truthful, versioned recording contracts for future live rocjitsu tools."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from aorta.instrumentation.rocjitsu_sanitizers import kernel_inventory

RECORDING_SCHEMA = "aorta.rocjitsu_recording/0.1"
PINNED_ROCJITSU_COMMIT = "46ef3f0438eb6d9eccb59e1f06beaff1b4dd02d1"


class OutputCheckState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"
    CRASH = "crash"


@dataclass(frozen=True)
class OutputCheck:
    state: OutputCheckState
    oracle: str = ""
    checked_items: int = 0
    mismatches: int = 0
    implementation: str = ""

    def __post_init__(self) -> None:
        if self.checked_items < 0 or self.mismatches < 0:
            raise ValueError("output-check counts must be non-negative")
        if self.mismatches > self.checked_items:
            raise ValueError("output-check mismatches exceed checked items")
        if self.state is OutputCheckState.PASS and self.checked_items == 0:
            raise ValueError("an output check with no checked items cannot pass")
        if self.state is OutputCheckState.FAIL and self.mismatches == 0:
            raise ValueError("a failed output check must record a mismatch")


@dataclass(frozen=True)
class CodeObjectFacts:
    path: str
    sha256: str
    requested_arch: str
    observed_arch: str
    kernels: tuple[str, ...]
    applicable: bool
    reason: str = ""


@dataclass(frozen=True)
class OperationObservation:
    operation: str
    arguments: dict[str, Any]
    machine_result: Any
    rendered_result: str
    output_check: OutputCheck
    elapsed_seconds: float


@dataclass(frozen=True)
class LiveRecording:
    code_object: CodeObjectFacts
    operations: tuple[OperationObservation, ...] = ()
    schema: str = RECORDING_SCHEMA
    rocjitsu_commit: str = PINNED_ROCJITSU_COMMIT
    limitations: tuple[str, ...] = field(default_factory=tuple)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_code_object(
    path: Path,
    *,
    requested_arch: str = "",
    max_kernels: int = 64,
) -> CodeObjectFacts:
    """Read architecture and kernel facts from the object, never from defaults."""
    if max_kernels < 1:
        raise ValueError("max_kernels must be positive")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    inventory = kernel_inventory(resolved)
    kernels = tuple(
        dict.fromkeys(kernel for code_object in inventory.code_objects for kernel in code_object.kernels)
    )
    processors = {code_object.processor for code_object in inventory.code_objects}
    reason = ""
    observed = ""
    if inventory.unreadable:
        reason = f"unreadable_code_object:{inventory.unreadable}"
    elif inventory.compressed_bundles:
        reason = "unreadable_compressed_bundle"
    elif not processors:
        reason = "no_amdgpu_code_object"
    elif len(processors) != 1:
        reason = "ambiguous_architecture"
    else:
        observed = next(iter(processors))
        if observed != "gfx950":
            reason = f"unsupported_architecture:{observed}"
        elif len(kernels) > max_kernels:
            reason = f"kernel_limit_exceeded:{len(kernels)}>{max_kernels}"
    return CodeObjectFacts(
        path=str(resolved),
        sha256=_digest(resolved),
        requested_arch=requested_arch,
        observed_arch=observed,
        kernels=kernels,
        applicable=not reason,
        reason=reason,
    )


def render_output_check(check: OutputCheck) -> str:
    """Render output truth without turning an absent oracle into PASS."""
    if check.state in (OutputCheckState.NOT_RUN, OutputCheckState.UNAVAILABLE):
        return "no output check"
    if check.state is OutputCheckState.CRASH:
        return "output check crashed"
    if check.state is OutputCheckState.PASS:
        return f"output check PASS ({check.checked_items} checked)"
    return (
        f"output check FAIL ({check.mismatches} mismatches "
        f"of {check.checked_items} checked)"
    )


def write_recording(path: Path, recording: LiveRecording) -> None:
    """Write one complete recording atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(recording), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PINNED_ROCJITSU_COMMIT",
    "RECORDING_SCHEMA",
    "CodeObjectFacts",
    "LiveRecording",
    "OperationObservation",
    "OutputCheck",
    "OutputCheckState",
    "inspect_code_object",
    "render_output_check",
    "write_recording",
]
