from __future__ import annotations

import json
from pathlib import Path

from aorta.cia.cancellation import Stop
from aorta.cia.launch.job import JobRecord, update_job_status


def trigger_autopsy(
    bundle_root: Path, job: JobRecord, jobs_root: Path, *, stop: Stop = None
) -> dict:
    """Run Autopsy on the assembled bundle and write report.json.

    Imports autopsy lazily to keep watchdog startup fast.
    """
    from aorta.cia.autopsy.orchestrator import run_autopsy

    print(f"[autopsy] starting on bundle {bundle_root}")
    # The job goes through. Without it run_autopsy cannot escalate -- its
    # production sweep is guarded on `job is not None` -- so every autopsy Watch
    # triggered could recommend `aorta sweep run` and none could ever run one.
    # The recommendation was reaching the report while the path that acts on it
    # was unreachable from the only caller that produces those reports.
    report = run_autopsy(
        bundle_root,
        kb_version="kb-static-poc",
        job=job,
        head_node=job.head_node,
        stop=stop,
    )

    report_path = bundle_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    category = report.get("category", "unknown")
    confidence = report.get("confidence", 0.0)
    print(f"[autopsy] category={category} confidence={confidence:.2f}")
    print(f"[autopsy] report → {report_path}")

    # Mark job as failed in registry
    update_job_status(jobs_root, job.job_id, "failed")

    return report
