"""Launch: put a workload on a node and record where its log will be.

``launch`` is the seam. It is Slurm today; a scheduler-less backend branches
here rather than at every call site, which is why callers should not reach for
``submit_sbatch`` directly.
"""

from pathlib import Path


def launch(
    *,
    command: str,
    job_name: str,
    log_path: str,
    script_path: Path,
    working_dir: str = "",
    env_vars: dict[str, str] | None = None,
    node: str = "",
) -> tuple[str, str]:
    """Submit *command*. Returns ``(job_id, error)``; one of the two is empty.

    A failed launch must surface its error rather than be reported as a run
    that never happened.
    """
    from aorta.cia.launch.cluster import submit_sbatch

    return submit_sbatch(
        command=command,
        job_name=job_name,
        log_path=log_path,
        script_path=script_path,
        working_dir=working_dir,
        env_vars=env_vars,
        node=node,
    )
