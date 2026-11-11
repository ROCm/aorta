"""C5 step 4: reload every engine from a slot (design Section 6.5).

`update_weights_from_disk(path, flush_cache=True)` replies `(success, message, num_paused_requests)`, and on
success the message ends "(N checkpoint tensors read)", N counting the tensors the loader yielded (A5). A reply
counts only if success is exactly True and N is exactly the slot's tensor count (248 partial, 738 full).

A flushing reload is refused while a request pins the prefix cache (A7), so reloads happen only behind the
generation barrier; a reload attempted with requests in flight is a programming error and stops the run.
The engine's own `weight_version` never changes after a reload (A9), so the driver records which version each
engine holds.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aorlta.atomic import PARTIAL_SUFFIX
from aorlta.sync.writer import INDEX_NAME, MANIFEST_NAME, SyncError

_COUNT_RE = re.compile(r"\((\d+) checkpoint tensors read\)")


class ReloadError(SyncError):
    """An engine did not take the slot after the retry and the restart; the run stops."""


class BarrierViolation(SyncError):
    """A reload was attempted while requests were in flight."""


class Engine(Protocol):
    engine_id: int

    def update_weights_from_disk(self, model_path: str, flush_cache: bool = True) -> Any: ...


@dataclass(frozen=True)
class ReloadReply:
    ok: bool
    tensors_read: int | None
    message: str


def parse_reload_reply(reply: Any, expected: int) -> ReloadReply:
    """Fail closed: anything but `(True, '... (<expected> checkpoint tensors read)', ...)` is a failure."""
    if not isinstance(reply, (tuple, list)) or len(reply) < 2:
        return ReloadReply(False, None, f"malformed reply {reply!r:.300}")
    success, message = reply[0], reply[1]
    message = message if isinstance(message, str) else repr(message)
    if success is not True:
        return ReloadReply(False, None, f"engine refused: {message[:500]}")
    counts = _COUNT_RE.findall(message)
    if len(counts) != 1:
        return ReloadReply(False, None, f"no single tensor count in {message[:300]!r}")
    n = int(counts[0])
    if n != expected:
        return ReloadReply(False, n, f"{n} checkpoint tensors read, expected {expected}")
    return ReloadReply(True, n, message)


@dataclass(frozen=True)
class ReloadResult:
    engine_id: int
    ok: bool
    tensors_read: int | None
    message: str
    seconds: float
    attempts: int
    restarted: bool


def _check_slot(slot: Path) -> None:
    if slot.name.endswith(PARTIAL_SUFFIX):
        raise ReloadError(f"refusing to reload from an unfinished slot {slot}")
    if not (slot / INDEX_NAME).is_file() or not (slot / MANIFEST_NAME).is_file():
        raise ReloadError(f"{slot} is not a completed slot (no index or manifest)")


def _reload_one(engine: Engine, slot: Path, expected: int, retries: int, restart) -> ReloadResult:
    t0 = time.monotonic()
    attempts = 0
    restarted = False
    reply = ReloadReply(False, None, "not attempted")
    for phase in ("try", "restart"):
        if phase == "restart":
            if restart is None:
                break
            try:
                restart(engine)
            except Exception as exc:  # noqa: BLE001 - an engine that cannot be restarted did not take the slot
                return ReloadResult(engine.engine_id, False, None, f"restart failed: {type(exc).__name__}: {exc}"[:500],
                                    time.monotonic() - t0, attempts, True)
            restarted = True
        for _ in range(1 + (retries if phase == "try" else 0)):
            attempts += 1
            try:
                raw = engine.update_weights_from_disk(str(slot), flush_cache=True)
            except Exception as exc:  # noqa: BLE001 - an exception is a failed reload, never a success
                reply = ReloadReply(False, None, f"{type(exc).__name__}: {exc}"[:500])
                continue
            reply = parse_reload_reply(raw, expected)
            if reply.ok:
                return ReloadResult(engine.engine_id, True, reply.tensors_read, reply.message,
                                    time.monotonic() - t0, attempts, restarted)
    return ReloadResult(engine.engine_id, False, reply.tensors_read, reply.message, time.monotonic() - t0,
                        attempts, restarted)


def reload_all(
    engines: Sequence[Engine],
    slot: str | Path,
    expected: int,
    *,
    in_flight: Callable[[], int],
    retries: int = 1,
    restart: Callable[[Engine], None] | None = None,
) -> list[ReloadResult]:
    """Reload every engine from `slot` in parallel. Retry once; then restart that engine (if a restart hook is
    given) and reload; then raise `ReloadError`. `in_flight()` must report 0 (the generation barrier)."""
    busy = in_flight()
    if busy != 0:
        raise BarrierViolation(f"reload with {busy} requests in flight: the generation barrier was not respected")
    if not engines:
        raise ReloadError("no engines to reload")
    slot = Path(slot)
    _check_slot(slot)
    results: list[ReloadResult | None] = [None] * len(engines)

    def run(i: int) -> None:
        results[i] = _reload_one(engines[i], slot, expected, retries, restart)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(engines))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = [r for r in results if r is not None]
    failed = [r for r in final if not r.ok]
    if len(final) != len(engines) or failed:
        raise ReloadError(
            f"{len(failed)} of {len(engines)} engines did not take {slot}: "
            + "; ".join(f"engine {r.engine_id}: {r.message}" for r in failed)
        )
    return final


class VersionBook:
    """Which policy version (and slot) each engine holds, recorded by the driver."""

    def __init__(self) -> None:
        self._held: dict[int, tuple[int, str]] = {}
        self._lock = threading.Lock()

    def record(self, engine_id: int, version: int, slot: str | Path) -> None:
        with self._lock:
            self._held[engine_id] = (version, str(slot))

    def version(self, engine_id: int) -> int | None:
        held = self._held.get(engine_id)
        return held[0] if held else None

    def all_at(self, version: int) -> bool:
        return bool(self._held) and all(v == version for v, _ in self._held.values())

    def as_dict(self) -> dict[int, dict]:
        return {e: {"version": v, "slot": s} for e, (v, s) in sorted(self._held.items())}
