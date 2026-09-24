from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


#: How much of one delta the archive below keeps.
#:
#: The same 4000 ``write_bundle`` is called with -- ``evidence or
#: new_content[:4000]`` in poll.py -- and that is the whole reason the number
#: is repeated rather than rounded to something tidier. The two files are the
#: two classes of one corpus, and a length difference between the classes is a
#: feature a classifier can win on without reading a word.
CLEAN_DELTA_CHARS = 4000

#: Where the archive goes: beside ``events.jsonl``, inside the job directory,
#: and outside ``bundle/`` on purpose. A bundle is what Autopsy reads about a
#: failure, and nothing here is about a failure.
CLEAN_DELTA_FILE = "local_classifier_clean_deltas.jsonl"


def archive_clean_delta(
    job_dir: Path,
    *,
    job_id: str,
    delta: str,
    healthy: bool,
    signal: str,
    confidence: float,
    source: str = "",
    local_classifier: dict[str, Any] | None = None,
    limit_bytes: int,
) -> bool:
    """Keep one delta that did *not* alert. Returns whether anything was written.

    ``write_bundle`` above is the reason this exists. It runs only after Watch
    alerts, so every full-length log delta anywhere on disk belongs to a
    failure, and the only healthy deltas recorded anywhere are the
    ``watchdog_ok`` excerpts the events file caps at 500 characters. A corpus
    built from those two has a positive class of up to 4000 characters and a
    negative class of at most 500: a classifier scores well on it by measuring
    length, and the clean-gate this feeds cannot be measured from the artifacts
    the system currently produces. This is the other half of the corpus, and it
    costs no model to collect -- which matters, because the measurement that
    gates ``watch.local_classifier.enabled`` is blocked on data rather than on weights.

    Every non-alerting delta is kept, healthy or not, with the verdict that was
    reached beside it. Watch can decide a delta is unhealthy and stay quiet
    because it is not sure enough, and those are the most interesting states in
    the corpus; filtering to ``healthy`` here would throw away the examples
    nearest the boundary and leave the label to be re-derived from nothing.

    **Bounds, all of them.**

    * Off unless *limit_bytes* is positive. It is 0 in the shipped config.
    * *limit_bytes* per job, checked before each append, so the file stops at
      the cap plus at most one record rather than growing with the run.
    * :data:`CLEAN_DELTA_CHARS` per delta, truncated before redaction so the
      kept prefix is byte-for-byte the prefix ``write_bundle`` would have kept.
    * One record per poll that produced new bytes and did not alert; a job that
      alerts stops producing them, because it stops being watched.

    **Redaction, and one honest caveat about it.** The text is scrubbed by
    ``aorta.cia.llm.redact`` -- the same gate Watch's outbound LLM traffic goes
    through, rewriting filesystem paths and IP addresses. That is right for a
    new artifact that will be copied off the node to train something. It also
    introduces an asymmetry, because ``write_bundle`` does *not* redact: a
    corpus that pairs these records against raw bundles has one class carrying
    redaction markers and the other not, which is the same kind of free signal
    as the length difference this file exists to remove. So every record says
    ``"redacted": true``, and a consumer joining the two classes has to scrub
    the bundle side to match before scoring anything.
    """
    if limit_bytes <= 0:
        return False

    path = Path(job_dir) / CLEAN_DELTA_FILE
    try:
        if path.exists() and path.stat().st_size >= limit_bytes:
            return False
    except OSError:
        return False

    try:
        # Imported here rather than at module scope so that a failure to load
        # the scrubber costs this archive and not ``write_bundle``, which is on
        # the alert path and must never be blocked by an optional corpus.
        from aorta.cia.llm import redact

        kept = redact(delta[:CLEAN_DELTA_CHARS])
    except Exception as exc:  # noqa: BLE001 - never at the cost of the poll
        print(
            f"[watch] {job_id}: could not scrub a clean delta, so it was not "
            f"archived: {type(exc).__name__}: {exc}"
        )
        return False

    record: dict[str, Any] = {
        "schema_version": "0.1",
        "ts": _utc_now(),
        "job_id": job_id,
        "healthy": bool(healthy),
        "signal": signal,
        "confidence": confidence,
        "source": source,
        "redacted": True,
        # The pre-truncation length, so a reader can tell a short delta from
        # the head of a long one without guessing from what survived.
        "delta_chars": len(delta),
        "truncated": len(delta) > CLEAN_DELTA_CHARS,
        "delta": kept,
    }
    if local_classifier is not None:
        record["local_classifier"] = local_classifier

    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:
        print(
            f"[watch] {job_id}: could not append to {path.name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False
    return True


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
