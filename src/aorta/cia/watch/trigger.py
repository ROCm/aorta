from __future__ import annotations

import json
from pathlib import Path

from aorta.cia.launch.job import JobRecord, update_job_status


def trigger_autopsy(bundle_root: Path, job: JobRecord, jobs_root: Path) -> dict:
    """Run Autopsy on the assembled bundle and write report.json.

    Imports autopsy lazily to keep watchdog startup fast.
    """
    from aorta.cia.autopsy.orchestrator import run_autopsy

    print(f"[autopsy] starting on bundle {bundle_root}")
    report = run_autopsy(bundle_root, kb_version="kb-static-poc")

    report_path = bundle_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    category = report.get("category", "unknown")
    confidence = report.get("confidence", 0.0)
    print(f"[autopsy] category={category} confidence={confidence:.2f}")
    print(f"[autopsy] report → {report_path}")

    # Mark job as failed in registry
    update_job_status(jobs_root, job.job_id, "failed")

    return report
