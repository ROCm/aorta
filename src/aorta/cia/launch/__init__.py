"""Launch a workload through Slurm or on the current workstation.

``launch`` is the seam.  ``auto`` uses Slurm when ``sbatch`` is installed and
otherwise starts a detached local process; callers do not need scheduler
branches of their own.
"""

import os
from pathlib import Path


def cancel(job_id: str) -> tuple[bool, str]:
    """Stop the scheduler allocation or local process represented by *job_id*.

    Returns ``(cancelled, error)``. Cancellation of a job that has already
    ended is successful.
    """
    from aorta.cia.launch.local import LOCAL_JOB_PREFIX, cancel_local

    if job_id.startswith(LOCAL_JOB_PREFIX):
        return cancel_local(job_id)

    from aorta.cia.launch.cluster import cancel_sbatch

    return cancel_sbatch(job_id)


def resolve_backend(requested: str | None = None) -> str:
    """Resolve ``auto|slurm|local`` to the backend this launch should use."""
    from aorta.cia.launch.cluster import sbatch_available

    backend = (requested or os.environ.get("CIA_JOB_BACKEND") or "auto").strip().lower()
    if backend not in {"auto", "slurm", "local"}:
        raise ValueError(f"unknown CIA job backend {backend!r}; expected auto, slurm, or local")
    if backend == "auto":
        return "slurm" if sbatch_available() else "local"
    return backend


def backend_for_job(job_id: str) -> str:
    """Identify the backend from its native job handle."""
    from aorta.cia.launch.local import LOCAL_JOB_PREFIX

    return "local" if job_id.startswith(LOCAL_JOB_PREFIX) else "slurm"


def state(job_id: str, job_dir: Path) -> str:
    """Return the state of a local job.

    Slurm state remains queried by :mod:`aorta.cia.triage` so its established
    accounting behavior and compatibility helpers stay intact.
    """
    from aorta.cia.launch.local import LOCAL_JOB_PREFIX, local_state

    if not job_id.startswith(LOCAL_JOB_PREFIX):
        return "UNKNOWN"
    return local_state(job_id, job_dir)


def record_cancellation(
    job_id: str,
    job_dir: Path,
    *,
    preserve_terminal: bool = True,
) -> None:
    """Persist a local terminal state after cancellation."""
    from aorta.cia.launch.local import record_local_cancellation

    record_local_cancellation(
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
    backend: str | None = None,
) -> tuple[str, str]:
    """Submit *command*. Returns ``(job_id, error)``; one of the two is empty.

    A failed launch must surface its error rather than be reported as a run
    that never happened.
    """
    try:
        selected = resolve_backend(backend)
    except ValueError as exc:
        return "", str(exc)

    if selected == "local":
        from aorta.cia.launch.local import submit_local

        return submit_local(
            command=command,
            job_name=job_name,
            log_path=log_path,
            script_path=script_path,
            working_dir=working_dir,
            env_vars=env_vars,
            node=node,
            tolerate_nonzero=tolerate_nonzero,
        )

    from aorta.cia.launch.cluster import submit_sbatch

    return submit_sbatch(
        command=command,
        job_name=job_name,
        log_path=log_path,
        script_path=script_path,
        working_dir=working_dir,
        env_vars=env_vars,
        node=node,
        tolerate_nonzero=tolerate_nonzero,
    )
