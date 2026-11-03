"""Capture live kernel facts beside a completed CIA job without touching Autopsy."""

from __future__ import annotations

from pathlib import Path

from aorta.rocjitsu.recording import LiveRecording, inspect_code_object, write_recording


def capture_shadow_recording(
    *,
    job_dir: Path,
    code_object: Path,
    requested_arch: str,
    max_kernels: int,
) -> Path:
    """Write factual code-object metadata for a later expert run."""
    job_dir = job_dir.resolve()
    expert_dir = job_dir / "rj_expert"
    if expert_dir.is_symlink():
        raise ValueError("rocjitsu expert directory must not be a symlink")
    expert_dir.mkdir(mode=0o700, exist_ok=True)
    resolved_expert = expert_dir.resolve()
    if resolved_expert.parent != job_dir:
        raise ValueError("rocjitsu expert directory escaped the CIA job")
    path = resolved_expert / "live_recording.json"
    if path.exists():
        raise FileExistsError(f"live rocjitsu recording already exists: {path}")
    facts = inspect_code_object(
        code_object,
        requested_arch=requested_arch,
        max_kernels=max_kernels,
    )
    recording = LiveRecording(
        code_object=facts,
        limitations=(
            "no expert model was invoked",
            "no simulator operation was recorded",
            "no live output oracle was available",
        ),
    )
    write_recording(path, recording)
    return path


__all__ = ["capture_shadow_recording"]
