from __future__ import annotations

import json
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
    head_node: str = ""             # SSH host for scheduler queries (e.g. 149.28.124.225)

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


def update_job_status(jobs_root: Path, job_id: str, status: str) -> None:
    path = jobs_root / job_id / "job.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = status
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
