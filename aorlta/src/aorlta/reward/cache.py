"""The two content-addressed caches (design Section 6.2, Cache).

A leg's key is `sha256(canonical JSON {source_sha256, task_sha256, leg, leg_settings, checker_name,
checker_version, requires, bracketed})`: `requires` maps each leg it requires to that leg's key, so a
change upstream (the compile's flags, say) changes the key of every leg that reads what it built, and
`bracketed` is whether its pool needs a clean control bracket. A candidate's reward key adds the reward
function's name, version, code hash and settings, the profile's clean repeats, and everything else its
reward reads (the whole TaskRef, the token count and the token budget).

Only settled legs (`clean`, `finding`, `fail`) are cached, and only with a clean control bracket
where the leg is bracketed. `unsettled`, `skipped_by_plan`, `not_run` and anything with a dirty or
missing bracket never are, and a lookup holds an entry to the same rule. Entries live at
`<root>/legs/<key[:2]>/<key>.json` (and `rewards/`), written atomically, with an in-memory LRU in
front. A read checks the schema, the key and the key's fields again; any mismatch, or a file that does
not parse, is a miss, and is logged.

The LRU is updated at once; the file is written by a background thread, because an atomic write
fsyncs twice and on the NFS-mounted run area that is tens of milliseconds, too long to spend on the
service's coordinator for every leg. A crash can lose the last few files, which only costs a
recomputation. `flush()` waits for every pending write; `close()` also stops the thread.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import queue
import threading
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, read_text
from ..ids import sha256_json
from ..schemas import SETTLED_STATES, Candidate, LegResult, RewardResult, SchemaError, from_dict, to_dict
from .adapters.base import inconsistency

log = logging.getLogger(__name__)

LEG_SCHEMA = "aorlta.reward.leg_cache/1"
REWARD_SCHEMA = "aorlta.reward.reward_cache/1"


def leg_key_fields(*, source_sha256: str, task_sha256: str, leg: str, leg_settings: dict, checker_name: str,
                   checker_version: str, requires: dict[str, str], bracketed: bool) -> dict:
    return {"source_sha256": source_sha256, "task_sha256": task_sha256, "leg": leg,
            "leg_settings": leg_settings, "checker_name": checker_name, "checker_version": checker_version,
            "requires": dict(sorted(requires.items())), "bracketed": bool(bracketed)}


def leg_key(**fields: Any) -> str:
    return sha256_json(leg_key_fields(**fields))


def reward_key_fields(cand: Candidate, leg_keys: dict[str, str], reward_fn: dict, profile: str, *,
                      reward_settings: dict, clean_repeats: int, bracketed: Iterable[str]) -> dict:
    return {"profile": profile, "clean_repeats": clean_repeats, "legs": dict(sorted(leg_keys.items())),
            "bracketed": sorted(bracketed), "reward_fn": dict(reward_fn), "reward_settings": dict(reward_settings),
            "task": to_dict(cand.task), "response_tokens": cand.response_tokens, "max_new_tokens": cand.max_new_tokens}


def cacheable_leg(result: LegResult, *, bracketed: bool) -> bool:
    """Settled, its state consistent with its findings (as an adapter's result must be), and with a clean bracket
    if the leg is bracketed."""
    if result.state not in SETTLED_STATES or inconsistency(result) is not None:
        return False
    if bracketed:
        return isinstance(result.bracket, dict) and result.bracket.get("clean") is True
    return True


def _any_unsettled(result: RewardResult) -> bool:
    return any(getattr(leg, "state", None) == "unsettled" for leg in (result.legs or {}).values())


def _legs_acceptable(result: RewardResult, fields: dict) -> bool:
    """Every leg as the leg cache would take it: consistent with its findings, and a settled bracketed leg with a
    clean bracket (a reward hit does not pass through get_leg)."""
    bracketed = set(fields.get("bracketed") or ())
    for name, leg in (result.legs or {}).items():
        if not isinstance(leg, LegResult) or inconsistency(leg) is not None:
            return False
        if name in bracketed and leg.state in SETTLED_STATES and not cacheable_leg(leg, bracketed=True):
            return False
    return True


class _Writer:
    """Writes cache files off the caller's thread, in order; once closed, on the caller's thread."""

    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.thread = threading.Thread(target=self._main, name="reward-cache-writer", daemon=True)
        self.thread.start()

    def _main(self) -> None:
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                self._write(*item)
            finally:
                self.q.task_done()

    @staticmethod
    def _write(path: Path, entry: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, entry)
        except (OSError, ValueError, TypeError) as exc:  # a full disk, or what JSON cannot hold (a NaN in a finding)
            log.warning("cache: could not write %s (%s)", path.name, type(exc).__name__)

    def put(self, path: Path, entry: dict) -> None:
        with self.lock:  # nothing is queued behind the stop sentinel
            if not self.closed:
                self.q.put((path, entry))
                return
        self._write(path, entry)

    def flush(self) -> None:
        self.q.join()

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.q.put(None)
        self.thread.join()


class _Store:
    def __init__(self, root: Path | None, kind: str, schema: str, lru_size: int, writer: _Writer | None):
        self.root = None if root is None else Path(root) / kind
        self.schema = schema
        self.lru: OrderedDict[str, dict] = OrderedDict()
        self.lru_size = lru_size
        self.lock = threading.Lock()
        self.writer = writer
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "rejected": 0, "corrupt": 0}

    def path(self, key: str) -> Path | None:
        return None if self.root is None else self.root / key[:2] / f"{key}.json"

    def get(self, key: str, fields: dict) -> dict | None:
        with self.lock:
            entry = self.lru.get(key)
            if entry is not None:
                self.lru.move_to_end(key)
        if entry is None:
            entry = self._read(key, fields)
        if entry is None or entry.get("fields") != fields:
            with self.lock:
                self.stats["misses"] += 1
            return None
        with self.lock:
            self.stats["hits"] += 1
            self._remember(key, entry)
        return entry

    def _read(self, key: str, fields: dict) -> dict | None:
        path = self.path(key)
        if path is None or not path.exists():
            return None
        try:
            entry = json.loads(read_text(path))
            ok = (isinstance(entry, dict) and entry.get("schema") == self.schema and entry.get("key") == key
                  and entry.get("fields") == fields and sha256_json(entry.get("fields")) == key)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            log.warning("cache: unreadable entry %s (%s): a miss", path.name, type(exc).__name__)
            ok = False
        if not ok:
            with self.lock:
                self.stats["corrupt"] += 1
            log.warning("cache: entry %s does not match its key or schema: a miss", path.name)
            return None
        return entry

    def put(self, key: str, fields: dict, payload: dict) -> None:
        entry = {"schema": self.schema, "key": key, "fields": fields, **payload}
        path = self.path(key)
        if path is not None:
            self.writer.put(path, entry)
        with self.lock:
            self.stats["writes"] += 1
            self._remember(key, entry)

    def _remember(self, key: str, entry: dict) -> None:
        self.lru[key] = entry
        self.lru.move_to_end(key)
        while len(self.lru) > self.lru_size:
            self.lru.popitem(last=False)


class RewardCache:
    """Leg cache and candidate-level reward cache. `root` None keeps them in memory only; `enabled`
    False turns both off (every lookup is a miss and nothing is stored)."""

    def __init__(self, root: str | Path | None, *, enabled: bool = True, lru_size: int = 4096):
        self.enabled = enabled
        self._writer = _Writer() if root is not None else None
        self.legs = _Store(None if root is None else Path(root), "legs", LEG_SCHEMA, lru_size, self._writer)
        self.rewards = _Store(None if root is None else Path(root), "rewards", REWARD_SCHEMA, lru_size, self._writer)

    def flush(self) -> None:
        """Wait until every entry put so far is on disk."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Write every entry put so far and stop the writer thread; an entry put later is written at once."""
        if self._writer is not None:
            self._writer.close()

    # legs

    def get_leg(self, key: str, fields: dict) -> LegResult | None:
        if not self.enabled:
            return None
        entry = self.legs.get(key, fields)
        if entry is None:
            return None
        try:
            result = from_dict(LegResult, entry.get("result"))
        except SchemaError as exc:
            log.warning("cache: leg entry %s fails its schema (%s): a miss", key[:12], exc)
            with self.legs.lock:
                self.legs.stats["corrupt"] += 1
            return None
        if not cacheable_leg(result, bracketed=fields.get("bracketed") is True) or result.leg != fields["leg"]:
            with self.legs.lock:
                self.legs.stats["corrupt"] += 1
            return None
        return result

    def put_leg(self, key: str, fields: dict, result: LegResult, *, bracketed: bool, artifacts: dict | None = None) -> bool:
        """Store a leg result if it may be cached; returns whether it was stored."""
        if not self.enabled:
            return False
        if not cacheable_leg(result, bracketed=bracketed):
            with self.legs.lock:
                self.legs.stats["rejected"] += 1
            return False
        stored = dataclasses.replace(result, cache_hit=False)
        self.legs.put(key, fields, {"result": to_dict(stored), "artifacts": dict(artifacts or {})})
        return True

    def leg_artifacts(self, key: str) -> dict:
        with self.legs.lock:
            entry = self.legs.lru.get(key)
        return dict(entry.get("artifacts") or {}) if entry else {}

    # rewards

    def get_reward(self, key: str, fields: dict) -> RewardResult | None:
        if not self.enabled:
            return None
        entry = self.rewards.get(key, fields)
        if entry is None:
            return None
        try:
            result = from_dict(RewardResult, entry.get("result"))
        except SchemaError as exc:
            log.warning("cache: reward entry %s fails its schema (%s): a miss", key[:12], exc)
            return None
        if result.reward is None or result.status != "scored" or _any_unsettled(result):
            return None
        if not _legs_acceptable(result, fields):
            with self.rewards.lock:
                self.rewards.stats["corrupt"] += 1
            return None
        return result

    def put_reward(self, key: str, fields: dict, result: RewardResult) -> bool:
        """Only scored results are cached, and none with an unsettled leg: a finding can score a candidate while
        another leg timed out, and that leg must run again for the next identical candidate."""
        if (not self.enabled or result.reward is None or result.status != "scored" or not result.full_check
                or _any_unsettled(result) or not _legs_acceptable(result, fields)):
            return False
        self.rewards.put(key, fields, {"result": to_dict(result)})
        return True

    def stats(self) -> dict:
        return {"legs": dict(self.legs.stats), "rewards": dict(self.rewards.stats), "enabled": self.enabled}
