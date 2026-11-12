"""Write semantics of the run's records (design Section 6.8; Section 5 item 6).

- **Per-iteration files** (`episodes.jsonl`, `summary.json`, `spans.jsonl`) are written into
  `iterations/iter-<n>.partial/` and the directory is renamed to `iterations/iter-<n>/` when the iteration
  completes. Readers list and open only renamed directories, so a partial iteration is invisible to them.
- **Ledgers** (`ledgers/*.jsonl`: spans, ASan, score batches, controls, eval events) are append-only JSONL, one
  canonical-JSON record per line, fsynced after each batch. Every record names its iteration, so a resume can cut
  a ledger back to the iterations that were committed.
- **The evidence store** (`evidence/`) holds raw stderr and sanitizer logs, content-addressed by sha256, with a
  sha256 manifest. It is created owner-only (0700 / 0600): it may hold partner data, which never goes to logs.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..atomic import PARTIAL_SUFFIX, _fsync_dir, atomic_write_bytes, atomic_write_json, read_text
from ..ids import canonical_json
from ..spans import Span
from .schema import EpisodeError, validate_episode, validate_summary

ITERATIONS_DIR, LEDGERS_DIR, EVIDENCE_DIR = "iterations", "ledgers", "evidence"
EPISODES_NAME, SUMMARY_NAME, SPANS_NAME = "episodes.jsonl", "summary.json", "spans.jsonl"
ITER_RE = re.compile(r"iter-([0-9]{6})")


def iteration_dir_name(n: int) -> str:
    if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 999_999):
        raise ValueError(f"iteration must be an int in [1, 999999], got {n!r}")
    return f"iter-{n:06d}"


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        # Some filesystems do not support fsync on a directory; any other failure is a failed sync.
        if not (exc.errno in (errno.EINVAL, errno.ENOTSUP) and path.is_dir()):
            raise
    finally:
        os.close(fd)


def jsonl_bytes(records: Iterable[Any]) -> bytes:
    return b"".join(canonical_json(r) + b"\n" for r in records)


class IterationOutputs:
    """The outputs of one iteration, invisible until `commit`. A leftover `.partial` (from a crash) is refused, not
    reused: resume removes it first, so a re-run iteration can never add to what a dead process wrote."""

    def __init__(self, run_dir: Path, n: int):
        self.n = n
        self.root = Path(run_dir) / ITERATIONS_DIR
        name = iteration_dir_name(n)
        self.final = self.root / name
        self.partial = self.root / (name + PARTIAL_SUFFIX)
        if self.final.exists():
            raise FileExistsError(f"{self.final} exists: iteration {n} is already committed")
        self.root.mkdir(parents=True, exist_ok=True)
        self.partial.mkdir()
        self.committed = False

    def write_episodes(self, records: Sequence[dict]) -> Path:
        ids = [r.get("chain_id") for r in records]
        if len(set(ids)) != len(ids):
            raise EpisodeError(f"iteration {self.n}: a chain id appears twice")
        for r in records:
            validate_episode(r)
            if r["iteration"] != self.n:
                raise EpisodeError(f"{r['chain_id']}: iteration {r['iteration']} written into iteration {self.n}")
        return atomic_write_bytes(self.partial / EPISODES_NAME, jsonl_bytes(sorted(records, key=lambda r: r["chain_id"])))

    def write_summary(self, summary: dict) -> Path:
        validate_summary(summary)
        return atomic_write_json(self.partial / SUMMARY_NAME, summary)

    def write_spans(self, spans: Sequence[Span]) -> Path:
        return atomic_write_bytes(self.partial / SPANS_NAME, jsonl_bytes(s.to_json() for s in spans))

    def write_json(self, name: str, obj: Any) -> Path:
        return atomic_write_json(self.partial / name, obj)

    def commit(self) -> Path:
        for p in self.partial.iterdir():
            _fsync(p)
        _fsync(self.partial)
        os.replace(self.partial, self.final)
        _fsync(self.root)
        self.committed = True
        return self.final


def complete_iterations(run_dir: Path) -> list[int]:
    root = Path(run_dir) / ITERATIONS_DIR
    if not root.is_dir():
        return []
    return sorted(int(m.group(1)) for p in root.iterdir() if p.is_dir() and (m := ITER_RE.fullmatch(p.name)))


def _iteration_path(run_dir: Path, n: int) -> Path:
    path = Path(run_dir) / ITERATIONS_DIR / iteration_dir_name(n)
    if not path.is_dir():
        raise FileNotFoundError(f"iteration {n} is not complete: no {path}")
    return path


def read_episodes(run_dir: Path, n: int) -> list[dict]:
    out = []
    for i, line in enumerate(read_text(_iteration_path(run_dir, n) / EPISODES_NAME).splitlines(), 1):
        rec = json.loads(line)
        validate_episode(rec)
        out.append(rec)
    return out


def read_summary(run_dir: Path, n: int) -> dict:
    summary = json.loads(read_text(_iteration_path(run_dir, n) / SUMMARY_NAME))
    validate_summary(summary)
    return summary


# --- ledgers ---------------------------------------------------------------------------------------------------


class LedgerError(ValueError):
    pass


def _iteration_of(rec: Any) -> int | None:
    it = rec.get("iteration") if isinstance(rec, dict) else None
    return it if isinstance(it, int) and not isinstance(it, bool) and it >= 0 else None


class AppendLedger:
    """Append-only JSONL. `append` writes one batch and fsyncs it. `read` skips a torn final line (a write the crash
    cut short) and refuses any other line that is not a record naming its iteration. `truncate_after(n)` keeps the
    records of iterations <= n."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._dir_synced = False

    def append(self, records: Sequence[dict]) -> None:
        if not records:
            return
        for r in records:
            if _iteration_of(r) is None:
                raise LedgerError(f"{self.path.name}: every record names its iteration (an int >= 0), got {r!r:.120}")
        data = jsonl_bytes(records)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            # The file's own entry in its directory, without which a power loss can drop it. Once per ledger
            # object, not only on creation: a process that died before this line left an entry never synced.
            if not self._dir_synced:
                _fsync_dir(self.path.parent)
                self._dir_synced = True

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        torn = lines.pop() if lines else b""  # the bytes after the last newline: empty unless a write was cut
        out = []
        for i, line in enumerate(lines + ([torn] if torn else []), 1):
            try:
                rec = json.loads(line)
            except ValueError:
                if i > len(lines):  # the torn final line
                    continue
                raise LedgerError(f"{self.path}:{i}: not a JSON record") from None
            if _iteration_of(rec) is None:
                raise LedgerError(f"{self.path}:{i}: a record with no iteration")
            out.append(rec)
        return out

    def truncate_after(self, n: int) -> int:
        """Keep the records of iterations <= n (and drop a torn final line); return how many were dropped. A bad line
        anywhere else is refused, as `read` refuses it, and the file is left as it was: a resume never rewrites
        committed history it cannot read. A kept last record that lacks its newline gets it back, or the next append
        would run on into the same line."""
        if not self.path.exists():
            return 0
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        torn = lines.pop() if lines else b""  # the bytes after the last newline: empty unless a write was cut
        kept, dropped = [], 0
        for i, line in enumerate(lines + ([torn] if torn else []), 1):
            try:
                rec = json.loads(line)
            except ValueError:
                if i > len(lines):  # the torn final line
                    dropped += 1
                    continue
                raise LedgerError(f"{self.path}:{i}: not a JSON record") from None
            it = _iteration_of(rec)
            if it is None:
                raise LedgerError(f"{self.path}:{i}: a record with no iteration")
            if it <= n:
                kept.append(line + b"\n")
            else:
                dropped += 1
        data = b"".join(kept)
        if data != raw:
            with self._lock:
                atomic_write_bytes(self.path, data)
        return dropped


# --- the sealed evidence store -----------------------------------------------------------------------------------


class EvidenceStore:
    """Raw artifacts, content-addressed: `blobs/<sha[:2]>/<sha>`, with `MANIFEST.jsonl` (sha256, size, kind,
    iteration per put). Owner-only permissions."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.manifest = AppendLedger(self.root / "MANIFEST.jsonl")
        self._lock = threading.Lock()
        self._synced_shards: set[str] = set()
        old = os.umask(0o077)
        try:
            self.blobs.mkdir(parents=True, exist_ok=True)
        finally:
            os.umask(old)
        os.chmod(self.root, 0o700)
        os.chmod(self.blobs, 0o700)

    def path_of(self, sha: str) -> Path:
        return self.blobs / sha[:2] / sha

    def put(self, data: bytes, *, kind: str, iteration: int) -> str:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("evidence is bytes")
        sha = hashlib.sha256(data).hexdigest()
        path = self.path_of(sha)
        with self._lock:
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                    raise LedgerError(f"evidence blob {sha} does not match its name")
            else:
                old = os.umask(0o077)
                try:
                    path.parent.mkdir(exist_ok=True)
                    atomic_write_bytes(path, bytes(data))
                finally:
                    os.umask(old)
            # The shard's entry in `blobs/` and the blob's in the shard, before a manifest can name the blob. Once per
            # shard per store object, not only on creation: a process that died before this left them unsynced.
            if path.parent.name not in self._synced_shards:
                _fsync_dir(path.parent)
                _fsync_dir(self.blobs)
                self._synced_shards.add(path.parent.name)
            # listed on every put, stored once: a resume keeps a blob while a committed iteration lists it
            self.manifest.append([{"iteration": iteration, "sha256": sha, "size": len(data), "kind": kind}])
        return sha

    def get(self, sha: str) -> bytes:
        data = self.path_of(sha).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise LedgerError(f"evidence blob {sha} does not match its name")
        return data

    def prune_unlisted(self) -> list[str]:
        """Remove blobs no manifest entry lists: after a resume cut the manifest back, they belong to no committed
        iteration (a blob also listed by an earlier iteration stays)."""
        listed = {r["sha256"] for r in self.manifest.read()}
        removed = []
        for p in sorted(self.blobs.glob("*/*")):
            if p.is_file() and p.name not in listed:
                p.unlink()
                removed.append(p.name)
        return removed

    def missing_listed(self, upto: int) -> list[str]:
        """The blobs that a record of an iteration <= `upto` lists and that are not on disk at the size it records."""
        missing = set()
        for r in self.manifest.read():
            if r["iteration"] <= upto:
                path = self.path_of(r["sha256"])
                if not path.is_file() or path.stat().st_size != r["size"]:
                    missing.add(r["sha256"])
        return sorted(missing)

    def verify(self) -> list[str]:
        problems = []
        listed = {r["sha256"]: r for r in self.manifest.read()}
        on_disk = {p.name for p in self.blobs.glob("*/*") if p.is_file() and not p.name.endswith(PARTIAL_SUFFIX)}
        for sha in sorted(set(listed) - on_disk):
            problems.append(f"listed but missing: {sha}")
        for sha in sorted(on_disk - set(listed)):
            problems.append(f"not in the manifest: {sha}")
        for sha in sorted(on_disk & set(listed)):
            if hashlib.sha256(self.path_of(sha).read_bytes()).hexdigest() != sha:
                problems.append(f"content does not match its name: {sha}")
        return problems
