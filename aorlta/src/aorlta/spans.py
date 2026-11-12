"""Span recorder (design Section 5 item 5; Section 6.9, Timeline).

Every phase emits a span `{name, iteration, t_start, t_end, attrs}`. `t_start` and `t_end` are UTC
wall times in seconds since the Unix epoch, so spans from different processes and nodes can be laid
on one timeline. Within one recorder they are read from `time.monotonic()` against a single
(wall, monotonic) anchor taken at start, so durations are monotonic and a wall-clock step cannot
make a span negative or shift spans against each other.

Phases overlap, so spans may be open concurrently, from any thread. Iteration 0 holds run set-up
(engine start, self-test, the start-of-run round trip); training iterations count from 1.

Attributes carry ids, hashes, counts and timings only (Section 5 item 8): values are scalars or
flat lists of scalars, and strings are capped at MAX_ATTR_STR characters, so a prompt, a feedback
text or a source file cannot be attached to a span by accident.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import atomic_write_bytes, read_text
from .config import ConfigError, strict_decode
from .ids import canonical_json

MAX_ATTR_STR = 512
NAME_RE = re.compile(r"[A-Za-z0-9_.:<>-]+")
SPAN_KEYS = ("name", "iteration", "t_start", "t_end", "attrs")


class SpanError(ValueError):
    """A span or a span file breaks the span rules."""


@dataclass(frozen=True)
class Span:
    name: str
    iteration: int
    t_start: float  # UTC, seconds since the epoch
    t_end: float
    attrs: dict[str, Any]

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "iteration": self.iteration, "t_start": self.t_start, "t_end": self.t_end,
                "attrs": dict(self.attrs)}


def _check_name(name: Any, iteration: Any) -> None:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise SpanError(f"span name must match {NAME_RE.pattern}, got {name!r}")
    if type(iteration) is not int or iteration < 0:
        raise SpanError(f"span iteration must be an int >= 0, got {iteration!r}")


def _check_scalar(key: str, value: Any) -> Any:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise SpanError(f"attr {key!r}: non-finite number {value!r}")
    if type(value) is str:
        if len(value) > MAX_ATTR_STR:
            raise SpanError(f"attr {key!r}: a {len(value)}-character string; attrs carry ids and hashes, not text")
        return value
    raise SpanError(f"attr {key!r}: values must be scalars or flat lists of scalars, got {type(value).__name__}")


def validate_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if not isinstance(key, str) or not key:
            raise SpanError(f"attr keys must be non-empty str, got {key!r}")
        if isinstance(value, (list, tuple)):
            out[key] = [_check_scalar(key, v) for v in value]
        else:
            out[key] = _check_scalar(key, value)
    return out


class SpanHandle:
    """An open span. `end()` closes it exactly once."""

    def __init__(self, recorder: SpanRecorder, name: str, iteration: int, mono_start: float, attrs: dict[str, Any]):
        self._recorder = recorder
        self.name, self.iteration = name, iteration
        self._mono_start = mono_start
        self._attrs = attrs
        self._span: Span | None = None

    @property
    def closed(self) -> bool:
        return self._span is not None

    def set(self, **attrs: Any) -> None:
        self._recorder._set(self, validate_attrs(attrs))

    def end(self, **attrs: Any) -> Span:
        return self._recorder._end(self, validate_attrs(attrs))


class SpanRecorder:
    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self._monotonic = monotonic
        self._mono0 = monotonic()
        self._wall0 = wall()
        self._lock = threading.Lock()
        self._spans: list[Span] = []
        self._open: dict[int, SpanHandle] = {}

    def _wall_at(self, mono: float) -> float:
        return self._wall0 + (mono - self._mono0)

    def now(self) -> float:
        """The current UTC wall time on this recorder's timeline."""
        return self._wall_at(self._monotonic())

    def start(self, name: str, iteration: int, **attrs: Any) -> SpanHandle:
        _check_name(name, iteration)
        handle = SpanHandle(self, name, iteration, self._monotonic(), validate_attrs(attrs))
        with self._lock:
            self._open[id(handle)] = handle
        return handle

    def _set(self, handle: SpanHandle, attrs: dict[str, Any]) -> None:
        with self._lock:
            if handle._span is not None:
                raise SpanError(f"span {handle.name!r} is already closed")
            handle._attrs.update(attrs)

    def _end(self, handle: SpanHandle, attrs: dict[str, Any]) -> Span:
        mono_end = self._monotonic()
        with self._lock:
            if handle._span is not None or self._open.pop(id(handle), None) is None:
                raise SpanError(f"span {handle.name!r} (iteration {handle.iteration}) was already ended")
            merged = {**handle._attrs, **attrs}
            span = Span(handle.name, handle.iteration, self._wall_at(handle._mono_start), self._wall_at(mono_end), merged)
            handle._span = span
            self._spans.append(span)
        return span

    @contextmanager
    def span(self, name: str, iteration: int, **attrs: Any) -> Iterator[SpanHandle]:
        """Record the enclosed block. If it raises, the span still ends, with `error` = the
        exception's class name, and the exception propagates."""
        handle = self.start(name, iteration, **attrs)
        try:
            yield handle
        except BaseException as exc:
            handle.end(error=type(exc).__name__)
            raise
        handle.end()

    def record(self, name: str, iteration: int, t_start: float, t_end: float, **attrs: Any) -> Span:
        """Record a span timed elsewhere (for example by a worker on another node), in UTC seconds."""
        _check_name(name, iteration)
        for label, t in (("t_start", t_start), ("t_end", t_end)):
            if type(t) not in (int, float) or not math.isfinite(t):
                raise SpanError(f"{label} must be a finite number, got {t!r}")
        if t_end < t_start:
            raise SpanError(f"span {name!r} ends before it starts ({t_end} < {t_start})")
        span = Span(name, iteration, float(t_start), float(t_end), validate_attrs(attrs))
        with self._lock:
            self._spans.append(span)
        return span

    def spans(self, iteration: int | None = None) -> list[Span]:
        with self._lock:
            return [s for s in self._spans if iteration is None or s.iteration == iteration]

    def open_spans(self, iteration: int | None = None) -> list[tuple[str, int]]:
        with self._lock:
            return [(h.name, h.iteration) for h in self._open.values() if iteration is None or h.iteration == iteration]

    def close_iteration(self, iteration: int) -> list[Span]:
        """The iteration's spans; refuses while any of its spans is still open, since a missing end
        would shorten the timeline without saying so."""
        still_open = self.open_spans(iteration)
        if still_open:
            raise SpanError(f"iteration {iteration} still has open spans: {sorted(n for n, _ in still_open)}")
        return self.spans(iteration)

    def write_jsonl(self, path: str | Path, iteration: int | None = None) -> Path:
        """One canonical-JSON span per line, written atomically."""
        lines = [canonical_json(s.to_json()) + b"\n" for s in self.spans(iteration)]
        return atomic_write_bytes(path, b"".join(lines))


def read_jsonl(path: str | Path) -> list[Span]:
    out = []
    for n, line in enumerate(read_text(path).splitlines(), 1):
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict) or tuple(sorted(raw)) != tuple(sorted(SPAN_KEYS)):
                raise SpanError(f"keys must be exactly {SPAN_KEYS}")
            span = strict_decode(Span, raw, "span")
            _check_name(span.name, span.iteration)
            if span.t_end < span.t_start:
                raise SpanError("t_end < t_start")
            validate_attrs(span.attrs)
        except (ValueError, ConfigError) as exc:
            raise SpanError(f"{path}:{n}: {exc}") from None
        out.append(span)
    return out
