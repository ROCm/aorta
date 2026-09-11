from __future__ import annotations

from pathlib import Path

import yaml

from aorta.cia.launch.job import JobRecord


def write_bundle(job: JobRecord, job_dir: Path, alert_evidence: str, signal: str) -> Path:
    """Assemble the Autopsy bundle directory from job record + log context.

    Creates:
      <job_dir>/bundle/
        manifest.yaml
        logs/watch.stderr.log   ← copy of log content around the alert
    """
    bundle = job_dir / "bundle"
    logs_dir = bundle / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Write the relevant log excerpt as watch.stderr.log
    watch_log = logs_dir / "watch.stderr.log"
    watch_log.write_text(alert_evidence, encoding="utf-8")

    # Derive aorta matrix path (may not exist yet — that's fine)
    aorta_matrix_rel = "aorta/matrix.json"

    paths: dict[str, str] = {
        "stderr": "logs/watch.stderr.log",
        "aorta_matrix": aorta_matrix_rel,
    }
    # A `mode: sanitizer` sweep writes sanitizer_report.json at its --output root,
    # which deploy already points inside the bundle, so it needs publishing rather
    # than copying. Only advertise it when present: the sanitizer adapter treats a
    # declared-but-missing report differently from an absent one.
    sanitizer_rel = "aorta/sanitizer_report.json"
    if (bundle / sanitizer_rel).is_file():
        paths["sanitizer_report"] = sanitizer_rel

    # A workload that traps a device-side fault writes its debugger session into
    # the bundle before Watch alerts. Publish it on the same terms as the
    # sanitizer report, or the rocgdb adapter has nothing to read and the verdict
    # loses the only evidence that names a line and a register.
    rocgdb_rel = "rocgdb/session.log"
    if (bundle / rocgdb_rel).is_file():
        paths["rocgdb_session"] = rocgdb_rel

    manifest: dict = {
        "schema_version": "0.1",
        "job_id": job.job_id,
        "failure_at": _utc_now(),
        "nodes": [{"hostname": job.node, "rank": 0}],
        "paths": paths,
        "metadata": {
            "recipe": job.recipe,
            "framework": "pytorch",
            "source_run": job.aorta_output,
            "watch_signal": signal,
            "scheduler": job.scheduler,
            "launcher": job.launcher,
        },
    }

    (bundle / "manifest.yaml").write_text(yaml.dump(manifest, default_flow_style=False))
    return bundle


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
