"""Opt-in, privacy-preserving records of chatbot decisions.

This is not a transcript. Summary mode stores counts and digests for user/model
content while retaining the decisions needed to explain a turn: route,
selector ranking and rationale digest, tool order, CIA outcomes, and critic
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
    #
    # The bare branch additionally requires the full `new_job_id()` shape --
    # `cia-%Y%m%d-%H%M%S-<6 hex>` -- because at a line start the loose form is
    # not a job id, it is any hyphenated word. `nan-trap value mean_sq=0`,
    # which `cia/autopsy/adapters/rocgdb.py` emits, parsed as a job called
    # `nan-trap`, and the lines under it supplied a category and a confidence:
    # a phantom job wearing a real verdict, which is the error direction this
    # function exists to prevent. It also put `jobs_root` on a tool event that
    # named no job, so the "only where there are ids" rule the configuration
    # note relies on stopped holding.
    #
    # The prefixed and quoted-key branches stay loose on purpose. There the
    # surrounding text is the evidence that an id is what follows, so a
    # shortened or hand-written id is still a job, and nothing else is going
    # to appear after `"job_id":`.
    r"(?:^[ \t]*(?=(?:cia|nan)-\d{8}-\d{6}-[0-9a-f]{6}\b)"
    r"|\bJob\s+|[\"']job_id[\"']\s*:\s*[\"'])"
    r"(?P<id>(?:cia|nan)-[A-Za-z0-9._-]+)",
    re.M,
)
#: The value halves of the patterns below, for the structured read, which has
#: the values already and needs to check them rather than find them. Written
#: once and reused so the two reads cannot come to accept different vocabularies
#: for the same field.
_JOB_ID_VALUE = re.compile(r"(?:cia|nan)-[A-Za-z0-9._-]+")
_CATEGORY_VALUE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")

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

    A job named more than once gets **every** span that follows one of its own
    mentions, not only the first. Deduplicating to the first mention loses the
    verdict whenever a tool announces a job before it has one -- ``Job X
    started`` then ``Job X done: category: ...`` is the ordinary shape of a
    progress log, and the second line is where the answer is. The same thing
    happens with no repeated *announcement* at all: ``_JOB_ID`` also matches a
    bare id at the start of a line, which is what the bundle path under a
    listing looks like, so a single reported job could truncate its own
    evidence.

    Spanning to the next mention of ANY job is what keeps that safe, and it is
    a strictly wider read rather than a looser one: a span only ever begins at
    a mention of this job and ends before the next job is named, so no text
    this change newly consults can belong to a different job. Bounding at the
    next *distinct* id instead would fix the announce-then-finish case and
    still lose an interleaved one (``Job A`` / ``Job B`` / ``Job A done``),
    where the answer sits after a mention of A that is not A's first.

    A verdict that is genuinely absent stays ``None``. The job id is what makes
    the row joinable; ``jobs_root`` on the same event is what the id resolves
    against, and the report on disk is a better source for a verdict than a
    rendered string in any case.

    **A verdict rendered before its own id is the case spans cannot see.**
    A span begins at an id, so nothing ahead of the first one is read, and
    ``json.dumps(..., sort_keys=True)`` puts ``category`` and ``confidence``
    ahead of ``job_id`` -- the rendering the quoted-key alternatives in all
    three patterns exist for. With one job that lost a verdict the old
    whole-output search found; with two it gave the first job the second's
    category, which is worse and is the exact failure this function was
    written to remove.

    Both are answered before the regex read, by parsing *output* as JSON when
    it is JSON and taking each object's own keys. That is not a heuristic
    about where a verdict sits relative to an id -- in an object they have no
    order to reason about -- so it is right for any key ordering rather than
    for the two this rendering happens to produce. Anything that is not JSON,
    or is JSON without job ids in it, falls through to the spans below
    unchanged.

    The one-job whole-output fallback stays for the renderings that are not
    JSON at all: a plain-text tool that prints a category above its id is the
    same shape, and with one job there is nothing else the verdict could
    belong to. It is restricted to that case for the same reason it is safe
    there.
    """
    structured = _cia_results_from_json(output)
    if structured is not None:
        return structured

    matches = list(_JOB_ID.finditer(output))
    # Insertion-ordered, so each job keeps one row at the position it was first
    # named -- the deduplication the `seen` set used to do, without the cost.
    spans: dict[str, list[str]] = {}
    for position, match in enumerate(matches):
        following = (
            matches[position + 1].start()
            if position + 1 < len(matches)
            else len(output)
        )
        spans.setdefault(match.group("id"), []).append(
            output[match.end() : following]
        )
    results: list[dict[str, Any]] = []
    for job_id, job_spans in spans.items():
        if len(spans) == 1 and not _any_value(job_spans):
            job_spans = [output]
        confidence = _first_value(_CONFIDENCE, job_spans)
        results.append(
            {
                "job_id": job_id,
                "category": _first_value(_CATEGORY, job_spans),
                "confidence": float(confidence) if confidence else None,
            }
        )
    return results


def _cia_results_from_json(output: str) -> list[dict[str, Any]] | None:
    """One entry per object carrying a ``job_id``, or ``None`` if that is not what
    *output* is.

    ``None`` rather than ``[]`` for the not-applicable answer, so the caller can
    tell "this is not JSON with job ids in it" from "this is, and it names no
    jobs" -- collapsing those would make any JSON output suppress the regex read
    that is the only one most tools need.

    Only objects that carry a ``job_id`` are read, and only their own
    ``category`` and ``confidence``. Nesting is walked because a tool wrapping
    results in ``{"results": [...]}`` is the ordinary shape, but an inner
    object's keys never leak outward: the walk descends past a container and an
    object with an id is taken whole rather than merged with its parent's. It
    also descends *into* an object with an id, after taking it, so a job nested
    under another job -- ``{"job_id": ..., "children": [{"job_id": ...}]}`` --
    gets its own row in document order, parent first.

    Validated to the same vocabulary the patterns accept, rather than taken on
    trust -- ``{"job_id": 3}`` or ``{"confidence": "high"}`` is not a verdict
    this record can carry, and writing it through would put a shape into the
    log that nothing downstream expects.

    **A job named more than once keeps its verdict, half by half**, which is
    the rule the text path in :func:`_cia_results` already follows and this
    side did not. Announce-then-finish is the ordinary shape of a progress log in JSON
    as much as in text -- a ``results`` array carrying a running row for
    ``cia-a1`` and its completed row after it -- and taking the first object
    and discarding the rest recorded the row with ``category=None``, which is
    the record asserting that a job which reached a verdict reached none.

    First *valid* value wins, per half and not per object, exactly as
    :func:`_first_value` does across text spans: a later object fills a half
    the earlier one left empty and can never replace one it filled. So
    widening the read cannot change a verdict that was already being reported,
    and a stale row appearing after a final one cannot overwrite it. The row
    keeps the position of the object that introduced the id, because document
    order is the order the jobs were announced in.

    Merging across objects that share an id is not the merging
    :func:`test_a_nested_object_does_not_inherit_a_parent_verdict` forbids.
    That one is a parent's verdict reaching a *different* job nested under it,
    which is misattribution; this is one job's own two mentions. The locator
    check below runs before the merge for that reason -- a pointer at a job is
    not a mention of its verdict, whether or not the id is already known.
    """
    try:
        doc = json.loads(output)
    except (ValueError, TypeError, RecursionError):
        return None

    results: list[dict[str, Any]] = []
    rows: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        raw_id = node.get("job_id")
        if not isinstance(raw_id, str) or not _JOB_ID_VALUE.fullmatch(raw_id):
            for value in node.values():
                walk(value)
            return
        # A *locator* is not a result row. ``build_report`` writes
        # ``{"bundle": {"job_id": ..., "root": ...}, "category": ...,
        # "confidence": ...}`` -- the id under a child object and the verdict
        # beside it, on the parent -- so walking the child as the row recorded
        # the id, left both halves ``None``, claimed the output, and never fell
        # through. A raw dump of the in-tree autopsy report was therefore
        # joinable and wrong, which is worse than not joinable: the record
        # asserts "this job reached no verdict" about a report whose whole
        # purpose is to carry one. Prefixed ``read_autopsy_report`` output
        # escaped it only because its header makes ``json.loads`` fail.
        #
        # Declining rather than merging, and the distinction is the one
        # ``test_a_nested_object_does_not_inherit_a_parent_verdict`` pins:
        # taking a parent's verdict for a nested id is how a summary label gets
        # copied onto every job under it. So the id stays unclaimed here and
        # the walk continues, leaving the answer to a later object that does
        # carry a verdict, or to the line reader.
        #
        # Both conjuncts are load-bearing. ``root`` is what makes this a
        # pointer at a job rather than a report about one, and without it a
        # bare ``{"job_id": ...}`` inside a results list -- which must be taken,
        # verdict-less and unmerged -- would be declined too. Requiring neither
        # verdict key means a locator that grew one is read as the row it has
        # become.
        if "root" in node and "category" not in node and "confidence" not in node:
            for value in node.values():
                walk(value)
            return
        category = node.get("category")
        confidence = node.get("confidence")
        verdict = {
            "category": (
                category
                if isinstance(category, str) and _CATEGORY_VALUE.fullmatch(category)
                else None
            ),
            "confidence": (
                float(confidence)
                if type(confidence) in (int, float) and 0 <= confidence <= 1
                else None
            ),
        }
        known = rows.get(raw_id)
        if known is not None:
            # Fill, never replace: the earlier object's answer is the reported
            # one, and only a half it left empty is still open.
            for half, value in verdict.items():
                if known[half] is None:
                    known[half] = value
        else:
            rows[raw_id] = row = {"job_id": raw_id, **verdict}
            results.append(row)
        # Then descend, because a job's own object can carry other jobs -- a
        # sweep with ``children``, a retry chain -- and stopping here dropped
        # every one of them while still claiming the output, so the line reader
        # never ran either. Descending cannot leak this object's verdict into
        # them: each nested object is read only for its own keys, by the same
        # rules as a top-level one.
        for value in node.values():
            walk(value)

    walk(doc)
    return results or None


def _any_value(spans: list[str]) -> bool:
    """Whether *spans* carry either half of a verdict.

    Either, not both: a result that names a category and no confidence is a
    verdict the spans did find, and re-reading the whole output for the missing
    half would take it from wherever it happened to sit -- including from
    before the id, which is the text the span deliberately excludes.
    """
    return bool(
        _first_value(_CATEGORY, spans) or _first_value(_CONFIDENCE, spans)
    )


def _first_value(pattern: re.Pattern[str], spans: list[str]) -> str | None:
    """The first ``value`` group ``pattern`` finds across *spans*, in order.

    First rather than last: the spans are in document order, and a rendering
    that states a verdict twice states the same one twice. Taking the first
    keeps the answer identical to the single-mention case, so widening the
    read cannot change a verdict that was already being reported.
    """
    for span in spans:
        found = pattern.search(span)
        if found:
            return found.group("value")
    return None


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
            # The ranking is the structured decision. The model's explanation
            # saw the full conversation and may quote any part of it, so summary
            # mode gives it the same content boundary as prompts and plans.
            "reason": _content(state.get("selection_rationale"), mode),
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
