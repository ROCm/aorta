"""Scheduler backend contract for CIA jobs."""

from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aorta.cia.launch.cluster as cluster_impl
import aorta.cia.launch.local as local_impl


class JobBackend(Protocol):
    """Everything a caller may need to know about one execution backend."""

    @property
    def name(self) -> str: ...

    @property
    def script_name(self) -> str: ...

    @property
    def launcher(self) -> str: ...

    @property
    def poll_interval(self) -> float: ...

    @property
    def wait_subject(self) -> str: ...

    @property
    def supports_remote_probe(self) -> bool: ...

    @property
    def already_finished_reason(self) -> str: ...

    def submit(
        self,
        *,
        command: str,
        job_name: str,
        log_path: str,
        script_path: Path,
        working_dir: str = "",
        env_vars: dict[str, str] | None = None,
        node: str = "",
        tolerate_nonzero: bool = False,
    ) -> tuple[str, str]: ...

    def cancel(self, job_id: str) -> tuple[bool, str]: ...

    def state(self, job_id: str, job_dir: Path) -> str: ...

    def node(self, job_id: str) -> str: ...

    def record_node(self, job_id: str, requested: str) -> str: ...

    def run_quick(
        self, argv: list[str], timeout: int | float
    ) -> subprocess.CompletedProcess[str]: ...

    def record_cancellation(
        self,
        job_id: str,
        job_dir: Path,
        *,
        preserve_terminal: bool = True,
    ) -> None: ...

    def result_fields(self, job_id: str, state: str = "") -> dict[str, str]: ...

    def display_job_id(self, job_id: str) -> str: ...

    def launch_summary(self, job_id: str, node: str) -> str: ...

    def should_reconcile(self, other: JobBackend) -> bool: ...

    def cancel_was_noop(self, message: str) -> bool: ...


@dataclass(frozen=True)
class LocalBackend:
    name: str = "local"
    script_name: str = "launch.sh"
    launcher: str = "aorta_direct"
    poll_interval: float = 0.2
    wait_subject: str = "local process"
    supports_remote_probe: bool = False
    already_finished_reason: str = local_impl.ALREADY_FINISHED

    def submit(self, **kwargs) -> tuple[str, str]:
        return local_impl.submit_local(**kwargs)

    def cancel(self, job_id: str) -> tuple[bool, str]:
        return local_impl.cancel_local(job_id)

    def state(self, job_id: str, job_dir: Path) -> str:
        return local_impl.local_state(job_id, job_dir)

    def node(self, job_id: str) -> str:
        return socket.gethostname()

    def record_node(self, job_id: str, requested: str) -> str:
        return requested or self.node(job_id)

    def run_quick(self, argv: list[str], timeout: int | float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )

    def record_cancellation(
        self,
        job_id: str,
        job_dir: Path,
        *,
        preserve_terminal: bool = True,
    ) -> None:
        local_impl.record_local_cancellation(
            job_id,
            job_dir,
            preserve_terminal=preserve_terminal,
        )

    def result_fields(self, job_id: str, state: str = "") -> dict[str, str]:
        return {}

    def display_job_id(self, job_id: str) -> str:
        return job_id.split(":", 2)[1] if job_id.startswith(local_impl.LOCAL_JOB_PREFIX) else job_id

    def launch_summary(self, job_id: str, node: str) -> str:
        where = f" on {node}" if node else " on this workstation"
        return f"started local process {self.display_job_id(job_id)}{where}"

    def should_reconcile(self, other: JobBackend) -> bool:
        return other.name == self.name

    def cancel_was_noop(self, message: str) -> bool:
        return message == self.already_finished_reason


@dataclass(frozen=True)
class SlurmBackend:
    quick_fallback: JobBackend | None = None
    name: str = "slurm"
    script_name: str = "launch.sbatch"
    launcher: str = "sbatch"
    poll_interval: float = 5.0
    wait_subject: str = "slurm"
    supports_remote_probe: bool = True
    already_finished_reason: str = ""

    def submit(self, **kwargs) -> tuple[str, str]:
        return cluster_impl.submit_sbatch(**kwargs)

    def cancel(self, job_id: str) -> tuple[bool, str]:
        return cluster_impl.cancel_sbatch(job_id)

    def state(self, job_id: str, job_dir: Path) -> str:
        return cluster_impl.sacct_state(job_id)

    def node(self, job_id: str) -> str:
        return cluster_impl.sacct_nodelist(job_id)

    def record_node(self, job_id: str, requested: str) -> str:
        return requested

    def run_quick(self, argv: list[str], timeout: int | float) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["srun", "--nodes=1", "-t", "5", *argv],
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            if self.quick_fallback is None:
                raise
            return self.quick_fallback.run_quick(argv, timeout)

    def record_cancellation(
        self,
        job_id: str,
        job_dir: Path,
        *,
        preserve_terminal: bool = True,
    ) -> None:
        return None

    def result_fields(self, job_id: str, state: str = "") -> dict[str, str]:
        fields = {"slurm_job_id": job_id}
        if state:
            fields["slurm_state"] = state
        return fields

    def display_job_id(self, job_id: str) -> str:
        return job_id

    def launch_summary(self, job_id: str, node: str) -> str:
        where = f" to {node}" if node else " to a node the scheduler chose"
        return f"submitted slurm job {job_id}{where}"

    def should_reconcile(self, other: JobBackend) -> bool:
        return True

    def cancel_was_noop(self, message: str) -> bool:
        return False


LOCAL_BACKEND = LocalBackend()
SLURM_BACKEND = SlurmBackend()
AUTO_SLURM_BACKEND = SlurmBackend(quick_fallback=LOCAL_BACKEND)


def resolve_backend(requested: str | None = None) -> JobBackend:
    """Resolve ``auto|slurm|local`` to a backend instance."""
    name = (requested or os.environ.get("CIA_JOB_BACKEND") or "auto").strip().lower()
    if name == "auto":
        return AUTO_SLURM_BACKEND if cluster_impl.sbatch_available() else LOCAL_BACKEND
    if name == "slurm":
        return SLURM_BACKEND
    if name == "local":
        return LOCAL_BACKEND
    raise ValueError(f"unknown CIA job backend {name!r}; expected auto, slurm, or local")


def backend_for(job_id: str) -> JobBackend:
    """Identify the backend from its native job handle."""
    return LOCAL_BACKEND if job_id.startswith(local_impl.LOCAL_JOB_PREFIX) else SLURM_BACKEND


def select_backend(
    selected: JobBackend | str | None = None,
    *,
    job_id: str = "",
) -> JobBackend:
    """Normalize a backend object, configured name, or native job handle."""
    if selected is None:
        return backend_for(job_id) if job_id else resolve_backend()
    if isinstance(selected, str):
        return resolve_backend(selected)
    return selected


def backend_from_record(scheduler: str, job_id: str) -> JobBackend:
    """Resolve a persisted scheduler name, falling back to its native handle."""
    return resolve_backend(scheduler) if scheduler else backend_for(job_id)
