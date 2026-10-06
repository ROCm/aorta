from __future__ import annotations

from pathlib import Path

from aorta.cia.launch.job import JobRecord, read_job_json


def scan_active_jobs(jobs_root: Path) -> list[JobRecord]:
    """Return all JobRecords with status=running found under jobs_root."""
    jobs_root = Path(jobs_root)
    records: list[JobRecord] = []
    for job_json in sorted(jobs_root.glob("*/job.json")):
        try:
            record = read_job_json(job_json)
            if record.status == "running":
                records.append(record)
        except Exception:
            pass
    return records


def scan_all_jobs(jobs_root: Path) -> list[JobRecord]:
    """Return all JobRecords regardless of status."""
    jobs_root = Path(jobs_root)
    records: list[JobRecord] = []
    for job_json in sorted(jobs_root.glob("*/job.json")):
        try:
            records.append(read_job_json(job_json))
        except Exception:
            pass
    return records
