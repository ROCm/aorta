from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text
from aorta.cia.launch.job import JobRecord
from aorta.cia.launch.registry import scan_active_jobs
from aorta.cia.watch.cursors import load_cursors, read_new_bytes, save_cursors
from aorta.cia.watch.log_finder import LogFinder
from aorta.cia.watch.watcher import LogWatcher


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_watch_config(config_path: Path | None = None) -> dict:
    default = Path(__file__).parent / "watch_config.yaml"
    path = config_path or default
    if path.is_file():
        return yaml.safe_load(path.read_text()) or {}
    return {}


def should_alert(healthy: bool, confidence: float, threshold: float) -> bool:
    """Whether Watch escalates: unhealthy *and* sure enough to say so.

    The threshold is the whole reason Watch is trustworthy. Reporting every
    suspicion would train its readers to ignore it, and the model is genuinely
    unsure early in a run -- before a job has written anything, "stalled" and
    "still starting" look identical.
    """
    return not healthy and confidence >= threshold


def elapsed_seconds(launched_at: str) -> int | None:
    """Seconds since *launched_at*, or None when it cannot be read.

    This field used to be ``int(time.time())`` -- the Unix epoch, about 1.7
    billion. The model was told on every poll that the job had been running for
    fifty-four years, which is not merely wrong but backwards: the whole use of
    the number is telling "still starting up" from "stalled", and a constant
    1.7e9 says neither while looking like it says something.

    None rather than 0 when the timestamp is unreadable, so the caller can
    leave the field out. A job that has been running for no time at all and a
    job whose launch time is unknown are different things.
    """
    if not launched_at:
        return None
    try:
        started = datetime.fromisoformat(launched_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - started).total_seconds()))


#: How many polls a chunk is retried for before it is given up on. Three is
#: enough for a rate limit or a dropped connection and short enough that a job
#: whose assessment cannot succeed is not stuck on the same bytes for ever.
MAX_ASSESS_ATTEMPTS = 3


def _emit_skipped(events_path: Path, job, content: str, error: str) -> None:
    """Record a chunk that could not be assessed, where the verdicts go."""
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "schema_version": "0.1",
                    "event_id": str(uuid.uuid4()),
                    "ts": _utc_now(),
                    "phase": "watchdog",
                    "event_type": "watchdog_skipped",
                    "job_id": job.job_id,
                    "signal": "WATCH_ASSESSMENT_FAILED",
                    "confidence": 0.0,
                    "excerpt": content[:500],
                    "assessment": (
                        f"{MAX_ASSESS_ATTEMPTS} assessment attempts failed; this "
                        f"log chunk was not examined. Last error: {error}"
                    ),
                    "source": job.watch_files[0] if job.watch_files else "",
                }
            )
            + "\n"
        )


def poll_jobs(
    jobs_root: Path,
    *,
    config_path: Path | None = None,
    max_rounds: int | None = None,
) -> None:
    """Main watch loop: discover active jobs and monitor their logs with LLM."""
    cfg = _load_watch_config(config_path)
    watch_cfg = cfg.get("watch", {})
    finder_cfg = cfg.get("log_finder", {})

    interval = float(watch_cfg.get("poll_interval_sec", 30))
    confidence_threshold = float(watch_cfg.get("confidence_threshold", 0.70))
    expectations = "\n".join(
        f"- {e}" for e in watch_cfg.get("expectations", [
            "Training loss should be decreasing or stable — not NaN or diverging",
            "Training steps should be advancing — not stuck on the same step",
            "No out-of-memory errors or GPU faults",
        ])
    )

    finder = LogFinder(config=finder_cfg)
    watcher = LogWatcher()
    jobs_root = Path(jobs_root)
    rounds = 0
    #: Jobs that have already alerted. Autopsy is expensive and its verdict is
    #: about the failure, not about the bytes that arrived after it, so a job
    #: is diagnosed once per session however much more it goes on to write.
    alerted: set[str] = set()
    #: Consecutive assessment failures per job, so a chunk is retried rather
    #: than dropped, and a job whose assessment always fails is eventually let
    #: go rather than blocking its own progress for ever.
    failures: dict[str, int] = {}

    print(f"[watch] polling {jobs_root} every {interval}s")

    while max_rounds is None or rounds < max_rounds:
        rounds += 1
        active = scan_active_jobs(jobs_root)

        for job in active:
            if job.job_id in alerted:
                continue
            job_dir = jobs_root / job.job_id
            events_path = job_dir / "events.jsonl"
            job_context = (
                f"job_id={job.job_id} node={job.node} recipe={job.recipe} "
                f"launched={job.launched_at}"
            )

            # Discover or refresh watch files
            if not job.watch_files:
                declared = Path(job.log_path) if job.log_path else None
                if declared is not None and not declared.is_file():
                    # Declared but not written yet: the job is still starting.
                    # Resolving now would cache a guess for the life of the run,
                    # and discovery has nowhere good to look this early -- it
                    # falls back to trawling the working directory, which for a
                    # sanitizer sweep is a source checkout whose test fixtures
                    # are *designed* to read like failing logs. Watch then
                    # faithfully diagnoses somebody's test data.
                    continue
                if declared is not None:
                    # The job told us where it writes. Nothing discovery finds
                    # can be more authoritative than that.
                    job.watch_files = [str(declared)]
                else:
                    job.watch_files = [
                        str(p)
                        for p in finder.find(
                            job_dir,
                            job_context=job_context,
                            scheduler=job.scheduler,
                            scheduler_job_id=job.scheduler_job_id,
                            head_node=getattr(job, "head_node", ""),
                        )
                    ]
                if job.watch_files:
                    print(f"[watch] {job.job_id}: watching {[Path(p).name for p in job.watch_files]}")

            if not job.watch_files:
                continue

            # Read new bytes from each watched file since last cursor
            cursors = load_cursors(job_dir)
            new_parts: list[str] = []
            total_new = 0
            for p_str in job.watch_files:
                p = Path(p_str)
                cursor = cursors.get(p_str, 0)
                text, new_cursor = read_new_bytes(p, cursor)
                cursors[p_str] = new_cursor
                if text:
                    new_parts.append(f"=== {p.name} ===\n{text}")
                    total_new += len(text)

            if not new_parts:
                # Nothing read, so nothing to lose by committing: this keeps a
                # file that shrank or was rotated from being re-read forever.
                save_cursors(job_dir, cursors)
                continue

            new_content = "\n\n".join(new_parts)
            total_bytes = sum(cursors.values())
            elapsed = elapsed_seconds(job.launched_at)
            elapsed_ctx = f" elapsed_sec={elapsed}" if elapsed is not None else ""
            job_ctx = f"{job_context}{elapsed_ctx} total_bytes_seen={total_bytes}"

            # LLM assessment — every poll with new content
            try:
                pred = watcher.forward(
                    new_content=new_content,
                    job_context=job_ctx,
                    expectations=expectations,
                    # The file tools are the model's to aim, so they are bound
                    # to this job: its own directory, and the logs Watch already
                    # resolved for it, which may sit outside that directory
                    # because the launcher chose where they go.
                    allowed_roots=[job_dir, *(Path(p).parent for p in job.watch_files)],
                )
            except Exception as e:
                # The cursors for this chunk are deliberately not saved, so the
                # next poll reads these bytes again. Saving before assessing
                # meant a single transient model failure discarded the chunk for
                # good -- and a NaN, a fault or an OOM is usually printed once.
                attempts = failures.get(job.job_id, 0) + 1
                failures[job.job_id] = attempts
                print(
                    f"[watch] {job.job_id}: watcher error "
                    f"(attempt {attempts}/{MAX_ASSESS_ATTEMPTS}): {e}"
                )
                if attempts < MAX_ASSESS_ATTEMPTS:
                    continue

                # Out of attempts. Retrying for ever would stop this job ever
                # being watched again, so the chunk is given up on -- visibly,
                # in the same events file the verdict would have gone to, rather
                # than by advancing a cursor and saying nothing.
                _emit_skipped(events_path, job, new_content, str(e))
                print(
                    f"[watch] {job.job_id}: giving up on {total_new} bytes after "
                    f"{attempts} attempts; recorded as WATCH_ASSESSMENT_FAILED"
                )
                failures.pop(job.job_id, None)
                save_cursors(job_dir, cursors)
                continue

            failures.pop(job.job_id, None)

            signal = getattr(pred, "signal", "WATCH_CLEAN")
            healthy = getattr(pred, "healthy", True)
            confidence = float(getattr(pred, "confidence", 0.0))
            evidence = getattr(pred, "evidence", "")
            assessment = getattr(pred, "assessment", "")

            print(f"[watch] {job.job_id}: {signal} confidence={confidence:.2f} — {assessment[:120]}")

            # Emit event
            with events_path.open("a", encoding="utf-8") as fh:
                ev = {
                    "schema_version": "0.1",
                    "event_id": str(uuid.uuid4()),
                    "ts": _utc_now(),
                    "phase": "watchdog",
                    "event_type": "watchdog_alert" if not healthy else "watchdog_ok",
                    "job_id": job.job_id,
                    "signal": signal,
                    "confidence": confidence,
                    "excerpt": (evidence or "")[:500],
                    "assessment": assessment[:300],
                    "source": job.watch_files[0] if job.watch_files else "",
                }
                fh.write(json.dumps(ev) + "\n")

            # Assessed, and the event is on disk. Only now is this chunk done.
            save_cursors(job_dir, cursors)

            if should_alert(healthy, confidence, confidence_threshold):
                print(f"[watch] {job.job_id}: ALERT {signal} — triggering autopsy")
                from aorta.cia.watch.bundle_writer import write_bundle
                from aorta.cia.watch.trigger import trigger_autopsy
                bundle = write_bundle(job, job_dir, evidence or new_content[:4000], signal)
                trigger_autopsy(bundle, job, jobs_root)
                # continue, not break: break left the whole round, so a job that
                # alerted every round starved every job listed after it. And the
                # set above is what actually makes this once per job -- the
                # comment here used to claim that on its own.
                alerted.add(job.job_id)
                continue

        time.sleep(interval)


@dataclass
class WatchEvent:
    schema_version: str
    event_id: str
    ts: str
    phase: str
    event_type: str
    job_id: str
    signal: str
    excerpt: str
    line: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def emit_event(
    fh: TextIO,
    *,
    job_id: str,
    signal: str,
    excerpt: str,
    line: int,
    source: str,
    event_type: str = "watchdog_alert",
) -> WatchEvent:
    ev = WatchEvent(
        schema_version="0.1",
        event_id=str(uuid.uuid4()),
        ts=_utc_now(),
        phase="watchdog",
        event_type=event_type,
        job_id=job_id,
        signal=signal,
        excerpt=excerpt[:500],
        line=line,
        source=source,
    )
    fh.write(json.dumps(ev.to_dict()) + "\n")
    fh.flush()
    return ev


def poll_file(
    path: Path,
    *,
    job_id: str,
    events_out: Path,
    interval_sec: float = 5.0,
    max_rounds: int | None = None,
    from_offset: int = 0,
) -> int:
    """Tail a growing log file; emit watchdog_alert on new NaN lines."""
    path = path.resolve()
    events_out.parent.mkdir(parents=True, exist_ok=True)
    offset = from_offset
    alerts = 0
    rounds = 0

    with events_out.open("a", encoding="utf-8") as evfh:
        while max_rounds is None or rounds < max_rounds:
            rounds += 1
            if not path.is_file():
                time.sleep(interval_sec)
                continue

            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) <= offset:
                time.sleep(interval_sec)
                continue

            new_chunk = text[offset:]
            offset = len(text)
            base_line = text[:offset].count("\n") - new_chunk.count("\n")

            scan = scan_stderr_text(new_chunk)
            if scan.alert:
                for line_no, excerpt in scan.hits:
                    emit_event(
                        evfh,
                        job_id=job_id,
                        signal=scan.signal,
                        excerpt=excerpt,
                        line=base_line + line_no,
                        source=str(path),
                    )
                    alerts += 1

            time.sleep(interval_sec)

    return alerts


def scan_once(path: Path, *, job_id: str, events_out: Path | None = None) -> list[WatchEvent]:
    text = path.read_text(encoding="utf-8", errors="replace")
    scan = scan_stderr_text(text)
    events: list[WatchEvent] = []
    if not scan.hits:
        return events

    sink = None
    if events_out:
        events_out.parent.mkdir(parents=True, exist_ok=True)
        sink = events_out.open("w", encoding="utf-8")
    try:
        for line_no, excerpt in scan.hits:
            ev = WatchEvent(
                schema_version="0.1",
                event_id=str(uuid.uuid4()),
                ts=_utc_now(),
                phase="watchdog",
                event_type="watchdog_alert",
                job_id=job_id,
                signal=scan.signal,
                excerpt=excerpt[:500],
                line=line_no,
                source=str(path.resolve()),
            )
            events.append(ev)
            if sink:
                sink.write(json.dumps(ev.to_dict()) + "\n")
    finally:
        if sink:
            sink.close()
    return events
