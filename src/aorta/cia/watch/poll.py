from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text
from aorta.cia.cancellation import Stop, pause, stopped
from aorta.cia.launch.job import JobRecord, record_watch_files
from aorta.cia.launch.registry import scan_active_jobs
from aorta.cia.watch.cursors import load_cursors, read_new_bytes, save_cursors
from aorta.cia.watch.log_finder import LogFinder
from aorta.cia.watch.watcher import LocalClassifierObservation, LogWatcher


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


def _shadow_event(job: JobRecord, observation: LocalClassifierObservation, used: dict) -> dict:
    """One ``watchdog_shadow`` line: what the local classifier said, beside what Watch used.

    A distinct ``event_type`` so nothing that reads this file for verdicts can
    mistake it for one. ``watchdog_alert`` and ``watchdog_ok`` are what the
    Autopsy trigger, the corpus builder and an operator all read; a shadow line
    is a measurement of a tier that changed nothing, and it has to be
    unmistakable at a glance and at a grep.

    The verdict that was actually used is repeated here rather than left to be
    joined against the line above. The join is by position in an append-only
    file that two processes can write to, and a comparison that silently
    mispairs is worse than one that costs a few bytes.
    """
    return {
        "schema_version": "0.1",
        "event_id": str(uuid.uuid4()),
        "ts": _utc_now(),
        "phase": "watchdog",
        "event_type": "watchdog_shadow",
        "job_id": job.job_id,
        # The classifier's own answers occupy the fields a scorer already knows how to
        # read: the slug it chose, and p(healthy) as the confidence in it.
        "signal": observation.signal,
        "confidence": round(observation.clean_probability, 4),
        # Nothing was read out of the log to quote, and inventing an excerpt
        # from a delta the shadow tier merely scored would put unbounded log
        # text into the events file for no reader.
        "excerpt": "",
        "assessment": (
            f"Shadow only. The local classifier put p(healthy) at "
            f"{observation.clean_probability:.2f} against a clean threshold of "
            f"{observation.clean_threshold:.2f}. No control flow depended on it."
        ),
        "source": job.watch_files[0] if job.watch_files else "",
        **observation.as_event_fields(),
        **{f"watch_{key}": value for key, value in used.items()},
    }


def _emit_log_read_error(
    events_path: Path,
    job: JobRecord,
    source: Path,
    cursor: int,
    error: OSError,
) -> None:
    """Record one unreadable watched file without losing every other job.

    The stderr line is immediately visible to the Watch operator; the event
    makes the same failure available to bundle readers and Autopsy. Failure to
    write that event must not recreate the original outage, so it is reported
    and contained here too.
    """
    kind = type(error).__name__
    assessment = (
        f"Could not read watched log {source}: {kind}: {error}. "
        f"Its cursor remains at byte {cursor} so Watch will retry it."
    )
    print(f"[watch] {job.job_id}: {assessment}")
    payload = {
        "schema_version": "0.1",
        "event_id": str(uuid.uuid4()),
        "ts": _utc_now(),
        "phase": "watchdog",
        "event_type": "watchdog_error",
        "job_id": job.job_id,
        "signal": "WATCH_LOG_READ_FAILED",
        "confidence": 1.0,
        "excerpt": "",
        "assessment": assessment,
        "source": str(source),
    }
    try:
        with events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError as exc:
        print(
            f"[watch] {job.job_id}: could not persist the log-read error "
            f"for {source}: {type(exc).__name__}: {exc}"
        )


#: How many Autopsies may run at once. Bounded because each is an LLM ReAct
#: loop that can escalate to a production sweep, and an unbounded pool would
#: let one bad round start one per job on the cluster at the same time.
AUTOPSY_WORKERS = 2

#: Total accepted work, including running workers. Equal to the worker count
#: deliberately: ThreadPoolExecutor's own queue is unbounded, and permitting
#: more here would recreate the invisible backlog this gate exists to remove.
#: Jobs beyond this capacity are persisted as ``deferred`` and reconsidered on
#: a later round instead of occupying process memory behind four-hour work.
AUTOPSY_CAPACITY = AUTOPSY_WORKERS

#: How long a queued or running Autopsy may sit before a later round treats it
#: as lost. Longer than the production sweep's own four-hour limit, because
#: finishing slowly is not the same as dying, and re-queueing a live one wastes
#: a node.
AUTOPSY_STALE_AFTER_SEC = 5 * 60 * 60

#: A queued task has not started external work, so it does not inherit the
#: running lease. With capacity equal to the worker count it should become
#: ``running`` promptly; surviving this long means the process died between
#: persisting the admission and starting the worker. A new Watch may reclaim it
#: in minutes rather than waiting five hours.
AUTOPSY_QUEUED_STALE_AFTER_SEC = 5 * 60

#: How many times a job may be sent for Autopsy before Watch stops trying. A
#: crash loop that re-queues for ever would spend every round on one job.
AUTOPSY_MAX_ATTEMPTS = 2

#: States that mean this job is settled and must not be queued again.
#:
#: "failed" is deliberately not among them. A failed Autopsy is the case the
#: attempt counter exists for -- a transient LLM or network error should be
#: retried, and a persistent one becomes "gave_up" on its own once the
#: attempts run out. Treating failure as terminal would have made the counter
#: decorative.
_AUTOPSY_TERMINAL = frozenset({"done", "gave_up"})

#: Written beside the job the moment it alerts, before the work is queued.
#: "Diagnosed once" used to live in a set inside poll_jobs, so a watcher that
#: restarted re-diagnosed everything it had already alerted on, and nothing
#: outside the process could see an Autopsy was in flight.
_AUTOPSY_STATE = "autopsy.state.json"

#: One file per attempt, created exclusively. See :func:`claim_autopsy_attempt`.
_AUTOPSY_CLAIM = ".autopsy.claim"


def autopsy_state(job_dir: Path) -> dict:
    """What is known about this job's Autopsy, or ``{}`` if it has not alerted."""
    try:
        return json.loads((job_dir / _AUTOPSY_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _stale(recorded: dict, max_age: float = AUTOPSY_STALE_AFTER_SEC) -> bool:
    """Whether a queued/running record is old enough to have been lost.

    Time rather than a liveness check: the worker is a thread in a process that
    may no longer exist, and a pid on a shared filesystem says nothing about
    whether *this* host still runs it.
    """
    stamped = recorded.get("ts")
    if not isinstance(stamped, str):
        return True
    try:
        when = datetime.fromisoformat(stamped.replace("Z", "+00:00"))
    except ValueError:
        return True
    age = (datetime.now(timezone.utc) - when).total_seconds()
    return age > max_age


def autopsy_is_settled(job_dir: Path) -> bool:
    """Whether this job needs no further Autopsy.

    Settled means finished, failed, or given up on. A record still claiming
    "queued" or "running" long after anything could still be running it is a
    watcher that died holding it, and the job goes back in the queue -- the
    persistence exists to stop duplicate work, not to suppress work that never
    happened.
    """
    return _autopsy_state_is_settled(autopsy_state(job_dir))


def _autopsy_state_is_settled(recorded: dict) -> bool:
    """Whether one already-read state record means no work is eligible."""
    state = recorded.get("state")
    if not state:
        return False
    if state in _AUTOPSY_TERMINAL:
        return True
    if state == "queued":
        return not _stale(recorded, AUTOPSY_QUEUED_STALE_AFTER_SEC)
    if state == "running":
        return not _stale(recorded)
    return False


def autopsy_attempts(job_dir: Path) -> int:
    """How many times this job has been sent for Autopsy."""
    try:
        return int(autopsy_state(job_dir).get("attempts") or 0)
    except (TypeError, ValueError):
        return 0


def record_autopsy_state(job_dir: Path, state: str, **fields: object) -> bool:
    """Note that this job is deferred, queued, running, or finished with Autopsy.

    Written before the work is enqueued rather than after it finishes: the
    point is to survive a crash *during* an Autopsy, which is exactly when the
    in-memory version forgot. A job already carrying a state is skipped, so a
    four-hour escalation is started once however many rounds run over it.

    Replaced atomically so a failed update leaves the previous recoverable
    state intact. The return value is part of admission: callers must not claim
    work that neither a worker nor this durable record owns.
    """
    payload = {"state": state, "ts": _utc_now(), **fields}
    target = job_dir / _AUTOPSY_STATE
    temporary = job_dir / f".{_AUTOPSY_STATE}.{uuid.uuid4().hex}.tmp"
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        print(f"[watch] could not record autopsy state for {job_dir.name}: {exc}")
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return True


def _submit_autopsy(
    *,
    pool: ThreadPoolExecutor,
    capacity: BoundedSemaphore,
    queued: dict[str, Future],
    bundle: Path,
    job: JobRecord,
    jobs_root: Path,
    job_dir: Path,
    attempt: int,
    stop: Stop,
    signal: str = "",
    confidence: float | None = None,
) -> bool:
    """Admit one Autopsy without using the executor's unbounded queue.

    Returns True only after a worker has accepted the work, another Watch owns
    this attempt, or ``deferred`` has been durably recorded. If both workers
    are occupied, the completed-attempt count stays unchanged; the persisted
    bundle is the durable queue, and the next Watch round or process restart
    can submit it without waiting for fresh log bytes or spending a retry.
    False means nobody owns the diagnosis and the alerting log bytes must
    remain uncommitted for another round.

    ``queued`` is written before ``submit``. If the process dies in that gap,
    its short queue lease makes the orphan promptly recoverable. A worker
    changes it to ``running`` before any external Autopsy work begins.
    """
    previous_attempts = max(0, attempt - 1)
    fields: dict[str, object] = {
        "job_id": job.job_id,
        "attempts": previous_attempts,
        "next_attempt": attempt,
    }
    if signal:
        fields["signal"] = signal
    if confidence is not None:
        fields["confidence"] = confidence

    if not capacity.acquire(blocking=False):
        persisted = record_autopsy_state(
            job_dir,
            "deferred",
            **fields,
            reason="watch capacity is full",
        )
        if persisted:
            print(
                f"[watch] {job.job_id}: Autopsy deferred; "
                "all workers are occupied"
            )
        return persisted

    # Capacity and ownership are separate claims. Capacity prevents an
    # unbounded local backlog; the exclusive file prevents another Watch
    # process from submitting this same attempt at the same time.
    if not claim_autopsy_attempt(job_dir, attempt):
        capacity.release()
        # The exclusive marker means another Watch owns this exact attempt.
        # That is an accepted owner for the caller's cursor/alert claim just as
        # surely as a worker in this process.
        return True

    record_autopsy_state(
        job_dir,
        "queued",
        **{**fields, "attempts": attempt},
    )
    try:
        future = pool.submit(
            _run_autopsy_off_the_loop, bundle, job, jobs_root, job_dir, stop
        )
    except RuntimeError as exc:
        capacity.release()
        persisted = record_autopsy_state(
            job_dir,
            "deferred",
            **fields,
            reason=f"executor rejected submission: {exc}",
        )
        if persisted:
            print(
                f"[watch] {job.job_id}: Autopsy deferred; "
                "the executor rejected submission"
            )
        return persisted

    queued[job.job_id] = future
    future.add_done_callback(lambda _done: capacity.release())
    return True


def claim_autopsy_attempt(job_dir: Path, attempt: int) -> bool:
    """Take the exclusive right to run *attempt* for this job. True if we got it.

    ``autopsy_is_settled`` reads the state and the caller writes "queued" some
    lines later, and nothing holds the gap. Two watchers over the same job both
    read "not settled", both write, and the second write wins a file it was
    never told it was racing for -- so the job is diagnosed twice, which means
    two LLM ReAct loops and two escalations to a four-hour sweep.

    ``O_CREAT | O_EXCL`` is the smallest thing that decides it, and the kernel
    decides rather than the reader. Keyed by attempt number because a retry is a
    legitimate second claim: attempts only advance through the paths that
    already reason about staleness and AUTOPSY_MAX_ATTEMPTS, so a new number
    means that reasoning has happened and this is a new race to win.

    Nothing to unlock, so nothing leaks when a watcher dies holding it -- the
    marker is a fact about an attempt that was started, not a lease.
    """
    marker = job_dir / f"{_AUTOPSY_CLAIM}.{attempt}"
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError as exc:
        # Same trade as record_autopsy_state below: a filesystem that cannot
        # take the marker should cost a duplicate Autopsy, not a missed one.
        print(f"[watch] could not claim autopsy for {job_dir.name}: {exc}")
        return True
    try:
        os.write(fd, (_utc_now() + "\n").encode())
    finally:
        os.close(fd)
    return True


def _run_autopsy_off_the_loop(
    bundle: Path, job: JobRecord, jobs_root: Path, job_dir: Path, stop: Stop
) -> None:
    """Run one Autopsy and keep its state on disk. Never raises into the pool."""
    from aorta.cia.watch.trigger import trigger_autopsy

    attempts = autopsy_attempts(job_dir)
    record_autopsy_state(job_dir, "running", job_id=job.job_id, attempts=attempts)
    try:
        trigger_autopsy(bundle, job, jobs_root, stop=stop)
    except Exception as exc:  # noqa: BLE001 - a worker that raises is silent
        print(f"[watch] autopsy for {job.job_id} failed: {type(exc).__name__}: {exc}")
        record_autopsy_state(
            job_dir, "failed", job_id=job.job_id, attempts=attempts,
            error=str(exc)[:200],
        )
    else:
        record_autopsy_state(job_dir, "done", job_id=job.job_id, attempts=attempts)


def poll_jobs(
    jobs_root: Path,
    *,
    config_path: Path | None = None,
    max_rounds: int | None = None,
    stop: Stop = None,
    only: str = "",
) -> None:
    """Main watch loop: discover active jobs and monitor their logs with LLM.

    *only* narrows the loop to a single job id. The standalone watcher wants
    every active job, which is what it is for. A triage does not: it starts a
    watcher of its own, so with four triages running there were four watchers
    over all four jobs, each paying for its own model call on every log chunk
    and each free to alert on a job it did not submit.
    """
    cfg = _load_watch_config(config_path)
    watch_cfg = cfg.get("watch", {})
    finder_cfg = cfg.get("log_finder", {})

    interval = float(watch_cfg.get("poll_interval_sec", 30))
    confidence_threshold = float(watch_cfg.get("confidence_threshold", 0.70))
    # Read here and not inside LogWatcher, unlike the rest of ``watch.local_classifier``,
    # because the archive is about a job directory and the watcher has never
    # been told where one is. Keeping it that way is what lets the local-classifier tier be
    # a pure function of the delta.
    tier_cfg = watch_cfg.get("local_classifier") or {}
    archive_bytes = int(tier_cfg.get("shadow_archive_bytes", 0) or 0)
    expectations = "\n".join(
        f"- {e}" for e in watch_cfg.get("expectations", [
            "Training loss should be decreasing or stable — not NaN or diverging",
            "Training steps should be advancing — not stuck on the same step",
            "No out-of-memory errors or GPU faults",
        ])
    )

    finder = LogFinder(config=finder_cfg)
    # The whole ``watch`` block, not just ``watch.local_classifier``: the watcher picks the
    # keys it owns, which is the shape ``LogFinder`` already has, and a second
    # place that knows the nesting is a second place to get it wrong.
    watcher = LogWatcher(watch_cfg)
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

    pool = ThreadPoolExecutor(
        max_workers=AUTOPSY_WORKERS, thread_name_prefix="cia-autopsy"
    )
    capacity = BoundedSemaphore(AUTOPSY_CAPACITY)
    queued: dict[str, Future] = {}
    try:
        _poll_rounds(
            pool=pool, capacity=capacity, queued=queued, jobs_root=jobs_root,
            finder=finder, watcher=watcher,
            interval=interval, confidence_threshold=confidence_threshold,
            expectations=expectations, max_rounds=max_rounds, stop=stop,
            alerted=alerted, failures=failures, rounds=rounds, only=only,
            archive_bytes=archive_bytes,
        )
    finally:
        # Why the loop ended decides what happens to work still queued.
        #
        # Cancelled: the caller set the stop flag, which is it saying it is not
        # waiting for an answer. Holding the interpreter open for a four-hour
        # Autopsy nobody will read is what that flag exists to prevent. What
        # must not follow is a record left saying "queued" for a job nothing
        # will pick up, so anything cancelled before it started is marked
        # abandoned and becomes eligible again.
        #
        # Waited for: the rounds simply ran out, which is not a request to drop
        # work already accepted.
        if stopped(stop):
            pool.shutdown(wait=False, cancel_futures=True)
            for job_id, future in queued.items():
                if future.cancelled():
                    record_autopsy_state(
                        jobs_root / job_id, "abandoned", job_id=job_id
                    )
        else:
            pool.shutdown(wait=True)


def _poll_rounds(*, pool, capacity, queued, jobs_root, finder, watcher, interval,
                 confidence_threshold, expectations, max_rounds, stop,
                 alerted, failures, rounds, only="", archive_bytes=0) -> None:
    """The rounds themselves, so the pool above owns its own lifetime."""
    while max_rounds is None or rounds < max_rounds:
        if stopped(stop):
            print("[watch] caller gave up; ending the poll loop")
            return
        rounds += 1
        active = scan_active_jobs(jobs_root)
        if only:
            active = [job for job in active if job.job_id == only]

        for job in active:
            job_state_dir = jobs_root / job.job_id
            recorded = autopsy_state(job_state_dir)

            # The persisted state is authoritative. ``alerted`` is only the
            # same-process claim for work a worker (here or in another Watch)
            # accepted before its state could be written. Checking the set
            # first suppressed a worker that had already recorded ``failed``,
            # so the bounded retry path below was unreachable until this whole
            # Watch process restarted.
            #
            # The same rule recovers an abandoned attempt and a stale
            # queued/running attempt. A fresh queued/running one and every
            # terminal state remain settled, so they keep the in-memory claim.
            if not recorded.get("state") and job.job_id in alerted:
                continue
            if _autopsy_state_is_settled(recorded):
                alerted.add(job.job_id)
                continue
            alerted.discard(job.job_id)

            # A retry does not wait for the job to say something new. Alerting
            # is driven by fresh log bytes, which is right for deciding whether
            # a job is in trouble and wrong for re-running an Autopsy that
            # failed: the failure was in the diagnosis, not in the log. A job
            # that alerted and then went quiet -- which a crashed one does --
            # would otherwise keep its failed state for ever while the counter
            # that was meant to retry it never advanced.
            pending = recorded
            if pending.get("state") in {
                "deferred",
                "failed",
                "abandoned",
                "queued",
                "running",
            }:
                bundle = job_state_dir / "bundle"
                if bundle.exists():
                    attempts = autopsy_attempts(job_state_dir) + 1
                    if attempts > AUTOPSY_MAX_ATTEMPTS:
                        print(
                            f"[watch] {job.job_id}: autopsy failed "
                            f"{AUTOPSY_MAX_ATTEMPTS} times; not trying again"
                        )
                        claimed = record_autopsy_state(
                            job_state_dir, "gave_up", job_id=job.job_id,
                            attempts=attempts - 1,
                        )
                    else:
                        print(f"[watch] {job.job_id}: retrying autopsy ({attempts})")
                        claimed = _submit_autopsy(
                            pool=pool,
                            capacity=capacity,
                            queued=queued,
                            bundle=bundle,
                            job=job,
                            jobs_root=jobs_root,
                            job_dir=job_state_dir,
                            attempt=attempts,
                            stop=stop,
                            signal=str(pending.get("signal") or ""),
                            confidence=pending.get("confidence"),
                        )
                    if claimed:
                        alerted.add(job.job_id)
                    else:
                        print(
                            f"[watch] {job.job_id}: Autopsy was not admitted "
                            "or persisted; will retry"
                        )
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
                    # Written back, because the record this loop reads is
                    # reloaded from disk every round. Without this the
                    # discovery above ran again on every pass for the whole
                    # life of the job, model call included, and each pass was
                    # free to land somewhere different from the last.
                    record_watch_files(jobs_root, job.job_id, job.watch_files)

            if not job.watch_files:
                continue

            # Read new bytes from each watched file since last cursor
            cursors = load_cursors(job_dir)
            new_parts: list[str] = []
            total_new = 0
            for p_str in job.watch_files:
                p = Path(p_str)
                cursor = cursors.get(p_str, 0)
                try:
                    text, new_cursor = read_new_bytes(p, cursor)
                except OSError as exc:
                    # Per file, not around the job or the round. A permission
                    # error or transient NFS failure in one sidecar must not
                    # stop the primary log, the next job, or every later
                    # polling round. Do not update this cursor: the same bytes
                    # are owed another attempt once the file is readable.
                    _emit_log_read_error(events_path, job, p, cursor, exc)
                    continue
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
            # Present only when the local-classifier tier ran, which is off by default.
            # Read with getattr for the same reason every other field here is:
            # this loop does not know which tier answered and must not start
            # caring.
            observation = getattr(pred, "local_classifier", None)

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
                if observation is not None:
                    # Written after the verdict rather than before it, so a
                    # reader scanning for what Watch decided meets the decision
                    # first and the measurement second.
                    fh.write(
                        json.dumps(
                            _shadow_event(
                                job,
                                observation,
                                {
                                    "signal": signal,
                                    "healthy": healthy,
                                    "confidence": confidence,
                                },
                            )
                        )
                        + "\n"
                    )

            if should_alert(healthy, confidence, confidence_threshold):
                print(f"[watch] {job.job_id}: ALERT {signal} — triggering autopsy")
                from aorta.cia.watch.bundle_writer import write_bundle
                bundle = write_bundle(job, job_dir, evidence or new_content[:4000], signal)
                attempts = autopsy_attempts(job_dir) + 1
                if attempts > AUTOPSY_MAX_ATTEMPTS:
                    # Said out loud rather than quietly skipped: a job that
                    # cannot be diagnosed is a thing the operator should know,
                    # and a silent give-up looks like a job nobody alerted on.
                    print(
                        f"[watch] {job.job_id}: autopsy failed "
                        f"{AUTOPSY_MAX_ATTEMPTS} times; not trying again"
                    )
                    claimed = record_autopsy_state(
                        job_dir, "gave_up", job_id=job.job_id,
                        attempts=attempts - 1, signal=signal,
                    )
                    if claimed:
                        # The terminal decision is durable, so these alerting
                        # bytes no longer need to be replayed.
                        save_cursors(job_dir, cursors)
                        alerted.add(job.job_id)
                    else:
                        print(
                            f"[watch] {job.job_id}: could not persist the "
                            "terminal Autopsy state; will retry"
                        )
                    continue
                # Off this thread. Autopsy is an LLM ReAct loop that can
                # escalate to a production sweep with a four-hour limit, and it
                # ran here -- inside the serial job loop -- so one alert stopped
                # every other active job being watched for as long as it took.
                # The jobs that most need watching are the ones running beside
                # a failure.
                claimed = _submit_autopsy(
                    pool=pool,
                    capacity=capacity,
                    queued=queued,
                    bundle=bundle,
                    job=job,
                    jobs_root=jobs_root,
                    job_dir=job_dir,
                    attempt=attempts,
                    stop=stop,
                    signal=signal,
                    confidence=confidence,
                )
                if claimed:
                    # Committing the cursor is itself part of admission. Before
                    # this ordering, a failed deferred-state write consumed the
                    # only alert bytes even though no worker owned the bundle.
                    save_cursors(job_dir, cursors)
                    alerted.add(job.job_id)
                else:
                    print(
                        f"[watch] {job.job_id}: Autopsy was not admitted or "
                        "persisted; leaving the alert bytes for another round"
                    )
                # continue, not break: break left the whole round, so a job that
                # alerted every round starved every job listed after it. And the
                # set above is what actually makes this once per job -- the
                # comment here used to claim that on its own.
                continue

            # A healthy assessment has no downstream admission to secure. The
            # event is on disk, so this chunk is done.
            #
            # It is also the only place in Watch where a full-length delta that
            # nobody alerted on is still in memory. ``write_bundle`` runs on the
            # branch above, so every long delta ever persisted belongs to a
            # failure and the clean-gate has no negative class to be measured
            # against. Archiving here is off unless an operator asked for it
            # and is bounded by ``archive_clean_delta``; the cursor is saved
            # either way, so a full archive costs nothing but the archive.
            if archive_bytes > 0:
                from aorta.cia.watch.bundle_writer import archive_clean_delta

                archive_clean_delta(
                    job_dir,
                    job_id=job.job_id,
                    delta=new_content,
                    healthy=healthy,
                    signal=signal,
                    confidence=confidence,
                    source=job.watch_files[0] if job.watch_files else "",
                    local_classifier=(
                        observation.as_event_fields() if observation is not None else None
                    ),
                    limit_bytes=archive_bytes,
                )
            save_cursors(job_dir, cursors)

        if pause(stop, interval):
            print("[watch] caller gave up; ending the poll loop")
            return


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
