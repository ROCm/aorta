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
    r"(?:\bJob\s+|[\"']job_id[\"']\s*:\s*[\"'])"
    r"(?P<id>(?:cia|nan)-[A-Za-z0-9._-]+)"
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
    ids = list(dict.fromkeys(match.group("id") for match in _JOB_ID.finditer(output)))
    if not ids:
        return []
    category_match = _CATEGORY.search(output)
    confidence_match = _CONFIDENCE.search(output)
    category = category_match.group("value") if category_match else None
    confidence = (
        float(confidence_match.group("value")) if confidence_match else None
    )
    return [
        {
            "job_id": job_id,
            "category": category,
            "confidence": confidence,
        }
        for job_id in ids
    ]


def _base(session_id: str, turn: int, event: str, mode: str) -> dict[str, Any]:
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
) -> list[dict[str, Any]]:
    """Event records for one completed turn."""
    events = [
        {
            **_base(session_id, turn, "input", mode),
            "question": _content(query, mode),
        },
        {
            **_base(session_id, turn, "route", mode),
            "route": state.get("route"),
        },
        {
            **_base(session_id, turn, "selection", mode),
            "ranked_tools": list(state.get("candidate_tools") or []),
            # This sentence is the point of the decision log, but filesystem
            # paths and addresses are never part of that point.
            "reason": _scrub_reason(state.get("selection_rationale")),
        },
        {
            **_base(session_id, turn, "plan", mode),
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
        events.append(
            {
                **_base(session_id, turn, "tool", mode),
                "position": position,
                "tool": name,
                "arguments": arguments,
                "output": _content(output, mode),
                "cia_results": _cia_results(output),
            }
        )

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
                **_base(session_id, turn, "critic", mode),
                "accepted": accepted,
                "feedback": _content(feedback, mode),
            },
            {
                **_base(session_id, turn, "answer", mode),
                "answer": _content(reply, mode),
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
) -> dict[str, Any]:
    """A failed turn, without persisting its question in summary mode."""
    return {
        **_base(session_id, turn, "failure", mode),
        "question": _content(query, mode),
        "error_type": type(error).__name__,
        "error": _content(str(error), mode),
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
            )
        return append_events(session_id, [event], mode)
    except Exception as exc:  # noqa: BLE001 - preserve the original failure
        logger.warning("Could not build the failed chat decision record: %s", exc)
        return None
