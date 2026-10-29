"""One execution-backend seam for CIA submit, wait, cancel, and quick work."""

from pathlib import Path

from aorta.cia.launch.backend import (
    AUTO_SLURM_BACKEND,
    LOCAL_BACKEND,
    SLURM_BACKEND,
    JobBackend,
    LocalBackend,
    SlurmBackend,
    backend_for,
    backend_from_record,
    resolve_backend,
    select_backend,
)
from aorta.cia.launch.local import ALREADY_FINISHED


def cancel(job_id: str) -> tuple[bool, str]:
    """Stop the scheduler allocation or local process represented by *job_id*."""
    return backend_for(job_id).cancel(job_id)


def backend_for_job(job_id: str) -> str:
    """Compatibility spelling for callers that only need the persisted name."""
    return backend_for(job_id).name


def state(job_id: str, job_dir: Path) -> str:
    """Return the state of whichever backend owns *job_id*."""
    return backend_for(job_id).state(job_id, job_dir)


def node(job_id: str) -> str:
    """Return the node reported by whichever backend owns *job_id*."""
    return backend_for(job_id).node(job_id)


def record_cancellation(
    job_id: str,
    job_dir: Path,
    *,
    preserve_terminal: bool = True,
) -> None:
    """Persist backend-specific cancellation state when needed."""
    backend_for(job_id).record_cancellation(
        job_id,
        job_dir,
        preserve_terminal=preserve_terminal,
    )


def launch(
    *,
    command: str,
    job_name: str,
    log_path: str,
    script_path: Path,
    working_dir: str = "",
    env_vars: dict[str, str] | None = None,
    node: str = "",
    tolerate_nonzero: bool = False,
    backend: str | JobBackend | None = None,
) -> tuple[str, str]:
    """Submit *command*. Returns ``(job_id, error)``; one of the two is empty.

    A failed launch must surface its error rather than be reported as a run
    that never happened.
    """
    try:
        selected = select_backend(backend)
    except ValueError as exc:
        return "", str(exc)
    return selected.submit(
        command=command,
        job_name=job_name,
        log_path=log_path,
        script_path=script_path,
        working_dir=working_dir,
        env_vars=env_vars,
        node=node,
        tolerate_nonzero=tolerate_nonzero,
    )


__all__ = [
    "ALREADY_FINISHED",
    "AUTO_SLURM_BACKEND",
    "LOCAL_BACKEND",
    "SLURM_BACKEND",
    "JobBackend",
    "LocalBackend",
    "SlurmBackend",
    "backend_for",
    "backend_for_job",
    "backend_from_record",
    "cancel",
    "launch",
    "node",
    "record_cancellation",
    "resolve_backend",
    "select_backend",
    "state",
]
