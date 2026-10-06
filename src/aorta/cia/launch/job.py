from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class JobRecord:
    job_id: str
    node: str
    recipe: str
    launched_at: str
    log_path: str
    aorta_output: str
    status: str = "running"         # running | failed | completed
    schema_version: str = SCHEMA_VERSION
    watch_files: list[str] = field(default_factory=list)
    launch_command: str = ""
    working_dir: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    estimated_runtime_min: int = 0
    scheduler: str = ""             # discovered: slurm | kubernetes | bare_metal
    launcher: str = ""              # discovered: torchrun | primus | aorta_direct | sbatch
    scheduler_job_id: str = ""      # native job ID (Slurm JobId, K8s pod name) for log discovery
    head_node: str = ""             # SSH host for scheduler queries; see CIA_SSH_HOST
    #: The recipe file a re-run would use, when the launcher wrote one.
    #: ``recipe`` above is a label for a reader; this is a path for a tool.
    #: Empty means the job did not come from a recipe file, so a sweep that
    #: needs one has nothing to run rather than something to guess at.
    recipe_path: str = ""
    sidecar_path: str = ""          # mitigations sidecar, for recipes that accept one

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"cia-{ts}-{short}"


def write_job_json(record: JobRecord, jobs_root: Path) -> Path:
    job_dir = jobs_root / record.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "job.json"
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def read_job_json(path: Path) -> JobRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("schema_version", None)
    return JobRecord(**{k: v for k, v in data.items() if k in JobRecord.__dataclass_fields__})


def _write_atomic(path: Path, payload: dict) -> None:
    """Replace *path* with *payload* in one step, or not at all.

    Watch re-reads job.json every round, and on a shared filesystem that read
    can land in the middle of a write. A truncated record does not read as a
    damaged job, it reads as no job: the record fails to parse, the job drops
    out of the active scan, and monitoring stops for a run that is still going.

    The temporary file is made in the same directory so the replace is a rename
    within one filesystem, which is the part that makes it atomic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _update_job_field(jobs_root: Path, job_id: str, **fields: object) -> None:
    """Change *fields* on a job record, leaving the rest as found.

    Read-modify-write rather than dumping an in-memory record: the copy on
    disk may have been changed by another process since this one loaded it,
    and rewriting the whole thing would put those changes back to what this
    process last saw.
    """
    path = jobs_root / job_id / "job.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    data.update(fields)
    _write_atomic(path, data)


def update_job_status(jobs_root: Path, job_id: str, status: str) -> None:
    _update_job_field(jobs_root, job_id, status=status)


def record_watch_files(jobs_root: Path, job_id: str, files: list[str]) -> None:
    """Remember which files Watch resolved for this job.

    Discovery runs once per job rather than once per round. The record was
    only ever updated in memory, and every round loads a fresh one from disk,
    so a job whose log path had to be discovered re-ran that discovery for the
    life of the run -- including the model call inside it.
    """
    if not files:
        return
    _update_job_field(jobs_root, job_id, watch_files=list(files))
