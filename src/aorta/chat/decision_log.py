"""Opt-in, privacy-preserving records of chatbot decisions.

This is not a transcript. Summary mode stores counts and digests for user/model
content while retaining the decisions needed to explain a turn: route,
selector ranking and scrubbed rationale, tool order, CIA outcomes, and critic
acceptance. Nothing here reads a record back or sends one anywhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aorta._user_paths import state_home
from aorta.probe.redaction import scrub_text

logger = logging.getLogger(__name__)

SESSION_LOG_ENV = "AORTA_CHAT_SESSION_LOG"
SCHEMA_VERSION = "0.1"

_NATIVE_TRACE = re.compile(
    r"^TOOL RESULT from (?P<name>[^:\n]+):\n(?P<output>.*)$", re.S
)
_TEXT_TRACE = re.compile(
    r"^\[(?P<name>[A-Za-z_]\w*)\((?P<args>.*?)\)\]\s*→\n(?P<output>.*)$",
    re.S,
)
_JOB_ID = re.compile(
    # A bare id at the start of a line is how ``list_cluster_jobs`` prints one,
    # and that tool needs no extra permission, so it is the one most likely to
    # have run. Still anchored rather than free-floating: the bundle path on
    # the following line repeats the same id, and an unanchored match would
    # read it as a second job.
    r"(?:^[ \t]*|\bJob\s+|[\"']job_id[\"']\s*:\s*[\"'])"
    r"(?P<id>(?:cia|nan)-[A-Za-z0-9._-]+)",
    re.M,
)
_CATEGORY = re.compile(
    r"(?:^\s*category:\s*|[\"']category[\"']\s*:\s*[\"'])"
    r"(?P<value>[A-Za-z_][A-Za-z0-9_-]*)",
    re.I | re.M,
)
_CONFIDENCE = re.compile(
    r"(?:^\s*confidence:\s*|[\"']confidence[\"']\s*:\s*)"
    r"(?P<value>0(?:\.\d+)?|1(?:\.0+)?)",
    re.I | re.M,
)

_warned_full: set[str] = set()
_warning_lock = threading.Lock()
_tool_calls: ContextVar[tuple[str, list[dict[str, Any]]] | None] = ContextVar(
    "aorta_decision_tool_calls", default=None
)


def new_session_id() -> str:
    """A non-identifying ID shared by all turns in one front-door session."""
    return uuid.uuid4().hex


def session_log_mode() -> str | None:
    """``summary``, ``full``, or None from :data:`SESSION_LOG_ENV`."""
    raw = os.environ.get(SESSION_LOG_ENV, "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return None
    if raw in {"1", "true", "yes", "on", "summary"}:
        return "summary"
    if raw == "full":
        return "full"
    logger.warning(
        "Ignoring %s=%r; expected 1, summary, full, or 0.",
        SESSION_LOG_ENV,
        raw,
    )
    return None


def decision_log_path(session_id: str) -> Path:
    """The private JSONL path for *session_id*.

    The filename is a digest rather than caller-controlled text, so even an
    externally supplied session ID cannot name another path.
    """
    component = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return state_home() / "aorta" / "chat" / "sessions" / f"{component}.jsonl"


def summarize_text(value: Any) -> dict[str, Any]:
    """Non-reversible shape and digest of arbitrary text."""
    text = _as_text(value)
    encoded = text.encode("utf-8")
    return {
        "characters": len(text),
        "bytes": len(encoded),
        "lines": len(text.splitlines()) if text else 0,
        "fences": len(re.findall(r"`{3,}", text)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _content(value: Any, mode: str) -> Any:
    text = _as_text(value)
    return text if mode == "full" else summarize_text(text)


@contextmanager
def capture_tool_calls(mode: str | None):
    """Collect privacy-safe tool-call metadata in this graph context."""
    calls: list[dict[str, Any]] = []
    if mode is None:
        yield calls
        return
    bound = _tool_calls.set((mode, calls))
    try:
        yield calls
    finally:
        _tool_calls.reset(bound)


def note_tool_call(tool: str, arguments: Any) -> None:
    """Record one call's order and arguments without touching the stream."""
    capture = _tool_calls.get()
    if capture is None:
        return
    mode, calls = capture
    try:
        serialized = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(arguments)
    calls.append(
        {
            "tool": str(tool),
            "arguments": _content(serialized, mode),
        }
    )


def _scrub_reason(reason: Any) -> str:
    """Always scrub selector prose, even when normal redaction is disabled."""
    scrubbed, _paths, _ipv4, _ipv6 = scrub_text(
        _as_text(reason),
        scrub_paths=True,
        scrub_ip_addresses=True,
    )
    return scrubbed


def _parse_tool_trace(entry: Any) -> tuple[str, str, str]:
    text = _as_text(entry)
    native = _NATIVE_TRACE.match(text)
    if native:
        return native.group("name").strip(), "", native.group("output")
    action = _TEXT_TRACE.match(text)
    if action:
        return (
            action.group("name").strip(),
            action.group("args"),
            action.group("output"),
        )
    return "unknown", "", text


def _cia_results(output: str) -> list[dict[str, Any]]:
    """One entry per job named in *output*, each carrying its own verdict.

    Read per job rather than per result: the category and confidence are taken
    from the span between one job id and the next, so a result naming two jobs
    no longer gives the second one the first one's verdict. That direction of
    error is the one worth spending code on -- a job whose verdict is missing
    is absent from whatever reads this, while a job wearing another job's
    category is an answer, and a wrong one.

    A verdict that is genuinely absent stays ``None``. The job id is what makes
    the row joinable; ``jobs_root`` on the same event is what the id resolves
    against, and the report on disk is a better source for a verdict than a
    rendered string in any case.
    """
    matches = list(_JOB_ID.finditer(output))
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, match in enumerate(matches):
        job_id = match.group("id")
        if job_id in seen:
            continue
        seen.add(job_id)
        following = (
            matches[position + 1].start()
            if position + 1 < len(matches)
            else len(output)
        )
        segment = output[match.end() : following]
        category = _CATEGORY.search(segment)
        confidence = _CONFIDENCE.search(segment)
        results.append(
            {
                "job_id": job_id,
                "category": category.group("value") if category else None,
                "confidence": (
                    float(confidence.group("value")) if confidence else None
                ),
            }
        )
    return results


def _jobs_root() -> str | None:
    """Where a recorded job id resolves on this machine.

    Without it the ids in a record point at nothing: the bundle, the autopsy
    report and the probe cells that would say what actually happened all live
    under this root, and nothing else in a record names it. That is what the
    empty ``resolution`` field is waiting on, so a log that cannot be resolved
    can never acquire one.

    ``None`` rather than a guess when the cia extra is absent, since the
    property reaches the agents' own default to answer.
    """
    try:
        from aorta.chat.config import settings

        return str(settings.jobs_root)
    except Exception:  # noqa: BLE001 - no cia extra, or an unreadable setting
        logger.debug("Could not resolve the jobs root for the decision log.")
        return None


def _base(
    session_id: str,
    turn: int,
    event: str,
    mode: str,
    front_door: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "session_id": session_id,
        "turn": turn,
        "event": event,
        "mode": mode,
        # Which entry point the turn came through, so a browser demo and a
        # scripted CLI run are told apart rather than averaged together. On
        # every event, like the rest of the envelope, so a line still reads
        # alone. ``None`` when the caller did not say.
        "front_door": front_door,
        # Stable attachment point for a later verified outcome.
        "resolution": None,
    }


def turn_events(
    *,
    session_id: str,
    turn: int,
    query: str,
    reply: str,
    state: dict[str, Any],
    mode: str,
    front_door: str | None = None,
    duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Event records for one completed turn."""
    events = [
        {
            **_base(session_id, turn, "input", mode, front_door),
            "question": _content(query, mode),
        },
        {
            **_base(session_id, turn, "route", mode, front_door),
            "route": state.get("route"),
        },
        {
            **_base(session_id, turn, "selection", mode, front_door),
            "ranked_tools": list(state.get("candidate_tools") or []),
            # This sentence is the point of the decision log, but filesystem
            # paths and addresses are never part of that point.
            "reason": _scrub_reason(state.get("selection_rationale")),
        },
        {
            **_base(session_id, turn, "plan", mode, front_door),
            "plan": _content(state.get("plan"), mode),
        },
    ]

    traces = list(state.get("tool_trace") or [])
    calls = list(state.get("_decision_tool_calls") or [])
    for position in range(1, max(len(traces), len(calls)) + 1):
        trace = traces[position - 1] if position <= len(traces) else ""
        call = calls[position - 1] if position <= len(calls) else {}
        parsed_name, parsed_arguments, output = _parse_tool_trace(trace)
        name = str(call.get("tool") or parsed_name)
        arguments = call.get("arguments")
        if arguments is None:
            arguments = _content(parsed_arguments, mode)
        cia_results = _cia_results(output)
        event = {
            **_base(session_id, turn, "tool", mode, front_door),
            "position": position,
            "tool": name,
            "arguments": arguments,
            "output": _content(output, mode),
            "cia_results": cia_results,
        }
        if cia_results:
            # Beside the ids rather than on every event, because it is only
            # meaningful where there is an id to resolve.
            event["jobs_root"] = _jobs_root()
        events.append(event)

    route = state.get("route")
    feedback = state.get("critic_feedback")
    accepted = (
        feedback is None
        if route == "action" and bool(state.get("command_output"))
        else None
    )
    events.extend(
        [
            {
                **_base(session_id, turn, "critic", mode, front_door),
                "accepted": accepted,
                # What acceptance cost. ``accepted`` says the answer was taken
                # in the end; this says whether it was taken first time or on
                # the third attempt, which is the difference between a cheap
                # decision and an expensive one.
                "iterations": state.get("iteration"),
                "feedback": _content(feedback, mode),
            },
            {
                **_base(session_id, turn, "answer", mode, front_door),
                "answer": _content(reply, mode),
                # Wall clock for the whole turn, on the event that ends it. A
                # turn that took four minutes and one that took four seconds
                # are different decisions even when they record the same ones.
                "duration_seconds": duration_seconds,
            },
        ]
    )
    return events


def failure_event(
    *,
    session_id: str,
    turn: int,
    query: str,
    error: BaseException,
    mode: str,
    front_door: str | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """A failed turn, without persisting its question in summary mode."""
    return {
        **_base(session_id, turn, "failure", mode, front_door),
        "question": _content(query, mode),
        "error_type": type(error).__name__,
        "error": _content(str(error), mode),
        # A turn that failed after four minutes of cluster job and one that
        # failed on the first call are different failures.
        "duration_seconds": duration_seconds,
    }


def append_events(
    session_id: str, events: list[dict[str, Any]], mode: str
) -> Path | None:
    """Append *events* with private permissions; logging failure is non-fatal."""
    if not events:
        return None
    path = decision_log_path(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            payload = "".join(
                json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n"
                for event in events
            ).encode("utf-8")
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                remaining = remaining[written:]
        finally:
            os.close(descriptor)
    except OSError as exc:
        logger.warning("Could not append the chat decision log at %s: %s", path, exc)
        return None

    if mode == "full":
        with _warning_lock:
            first = session_id not in _warned_full
            _warned_full.add(session_id)
        if first:
            logger.warning(
                "Full chat session logging is enabled; prompts, plans, tool "
                "results, and answers are stored verbatim in %s",
                path,
            )
    return path


def record_turn(
    *,
    session_id: str,
    turn: int,
    query: str,
    reply: str,
    state: dict[str, Any],
    front_door: str | None = None,
    duration_seconds: float | None = None,
) -> Path | None:
    """Record one successful turn when session logging is enabled."""
    mode = session_log_mode()
    if mode is None:
        return None
    try:
        events = turn_events(
            session_id=session_id,
            turn=turn,
            query=query,
            reply=reply,
            state=state,
            mode=mode,
            front_door=front_door,
            duration_seconds=duration_seconds,
        )
        return append_events(session_id, events, mode)
    except Exception as exc:  # noqa: BLE001 - optional telemetry cannot break a turn
        logger.warning("Could not build the chat decision record: %s", exc)
        return None


def record_failure(
    *,
    session_id: str,
    turn: int,
    query: str,
    error: BaseException,
    front_door: str | None = None,
    duration_seconds: float | None = None,
) -> Path | None:
    """Record a failed turn when session logging is enabled."""
    mode = session_log_mode()
    if mode is None:
        return None
    try:
        event = failure_event(
                session_id=session_id,
                turn=turn,
                query=query,
                error=error,
                mode=mode,
                front_door=front_door,
                duration_seconds=duration_seconds,
            )
        return append_events(session_id, [event], mode)
    except Exception as exc:  # noqa: BLE001 - preserve the original failure
        logger.warning("Could not build the failed chat decision record: %s", exc)
        return None
