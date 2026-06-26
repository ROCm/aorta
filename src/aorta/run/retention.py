"""Verdict-keyed per-trial artifact retention (issue #231).

A probe / triage sweep runs a workload many times. Some trials emit
**large** per-trial artifacts (profiler traces, per-layer numeric dumps --
hundreds of MB each). Keeping them for *every* trial fills the disk on a
big sweep (~900 trials x ~100 MB = tens of GB of mostly-useless
passing-run data). This module prunes those heavy files per trial as a
function of the trial's verdict, keeping the expensive output only where
it has diagnostic value (the failures).

The policy itself (the verdict -> level mapping) is a recipe-schema
concept and lives in :class:`aorta.triage.recipe.RetainPolicy`. This
module is the *engine*: given a trial directory and an already-resolved
retention **level** string, it classifies each artifact and deletes the
ones the level does not keep. It is deliberately dependency-free (stdlib
only) and knows nothing about verdicts, recipes, or the classifier, so it
imports cleanly from the probe workload without any cycle.

Retention levels form a monotonic ladder -- each keeps everything its
predecessors keep, plus its own class:

====================  ================================================
``none``              the trial JSON record only (bookkeeping)
``log``               + the stdout/stderr trial log (and ``probe.env``)
``summary``           + small collector "summary" roll-ups
``full``              + heavy collector outputs (everything; the default)
====================  ================================================

**The trial JSON record is never deleted at any level.** Deletion is of
the *artifact files*, not the per-trial record the matrix / rate
bookkeeping (and probe resume) read -- those always survive regardless of
retention level (issue #231 acceptance criterion).

Collectors declare which of their outputs are heavy vs summary via an
optional ``artifacts.json`` manifest in the trial directory (see
:data:`RETENTION_MANIFEST_NAME`); absent a manifest, classification falls
back to filename convention (``*.summary.*`` is a summary; anything that
is not the record or a known log is heavy).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Retention levels, lowest -> highest. ``full`` is the default (and the
# legacy behaviour when no ``retain`` block is present): keep everything.
RETAIN_LEVELS: tuple[str, ...] = ("none", "log", "summary", "full")
_LEVEL_RANK: dict[str, int] = {name: i for i, name in enumerate(RETAIN_LEVELS)}

# Artifact classes, ranked low -> high. A file of class C is kept at
# retention level L iff ``_CLASS_RANK[C] <= _LEVEL_RANK[L]``. RECORD sits
# at rank 0 so it is kept at every level (and is additionally hard-guarded
# against deletion below -- a buggy collector manifest can never drop it).
RECORD = "record"  # trial JSON / probe resume marker -- never deleted
LOG = "log"  # stdout/stderr trial log, probe.env
SUMMARY = "summary"  # small collector roll-ups
HEAVY = "heavy"  # big collector outputs (traces, per-layer dumps)
_CLASS_RANK: dict[str, int] = {RECORD: 0, LOG: 1, SUMMARY: 2, HEAVY: 3}
# Classes a collector manifest is allowed to assign. RECORD is excluded on
# purpose: only the engine decides what the immutable trial record is.
_MANIFEST_CLASSES: frozenset[str] = frozenset({LOG, SUMMARY, HEAVY})

# Optional per-trial manifest a collector drops to declare its outputs'
# classes. Shape: ``{"artifacts": [{"path": "trace.pb", "class": "heavy"},
# {"path": "rollup.json", "class": "summary"}]}``. ``path`` is relative to
# the trial directory (POSIX-style). A missing / malformed manifest is a
# warning, not an error -- classification falls back to convention so a
# bad manifest never aborts a run or silently keeps tens of GB.
RETENTION_MANIFEST_NAME = "artifacts.json"

# Trial-dir-relative POSIX paths that ARE the trial record -- never
# deleted, regardless of level or any manifest entry. ``result.json`` is
# the probe resume marker (``aorta.probe.resume.is_trial_complete``, which
# keys off ``trial_dir/result.json`` specifically) and the matrix's
# per-trial source of truth. Matched by exact relative path, NOT basename:
# a heavy collector output that happens to be named ``result.json`` under a
# subdirectory is prunable like any other artifact.
_RECORD_PATHS: frozenset[str] = frozenset({"result.json"})
# Known small operational artifacts that ride at the ``log`` tier.
_LOG_NAMES: frozenset[str] = frozenset({"stdout.log", "stderr.log", "probe.env"})


@dataclass(frozen=True)
class RetentionOutcome:
    """What :func:`apply_retention` did to one trial directory.

    Returned for logging / testing. ``deleted`` and ``kept`` are
    trial-dir-relative POSIX paths; ``freed_bytes`` is the summed size of
    the deleted files (best-effort -- a file that vanishes mid-sweep
    contributes 0).
    """

    level: str
    deleted: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()
    freed_bytes: int = 0
    # True when ``level`` was ``full`` (or unset) so nothing was scanned.
    no_op: bool = False


def classify_artifact(rel_path: str, manifest: dict[str, str] | None = None) -> str:
    """Return the retention class of one trial artifact.

    Precedence: the record guard wins first (so a collector can never mark
    the trial record deletable), then an explicit manifest entry, then the
    log / summary filename conventions, then a default of :data:`HEAVY`
    (an unrecognised file is assumed to be a big collector output -- the
    conservative choice for disk).

    The record guard matches the exact trial-dir-relative path (so only
    the top-level ``result.json`` is protected); the log / summary
    conventions match on basename.
    """
    if rel_path in _RECORD_PATHS:
        return RECORD
    name = Path(rel_path).name
    if manifest and rel_path in manifest:
        return manifest[rel_path]
    if name == RETENTION_MANIFEST_NAME:
        return SUMMARY
    if name in _LOG_NAMES:
        return LOG
    if name.endswith(".summary.json") or ".summary." in name:
        return SUMMARY
    return HEAVY


def _load_manifest(trial_dir: Path) -> dict[str, str]:
    """Read the optional ``artifacts.json`` collector manifest.

    Returns a ``{relative_posix_path: class}`` map. Tolerant by design,
    never aborts the run:

    * A missing file, or a malformed file (bad JSON, no ``artifacts``
      key, or an ``artifacts`` value that is not a list), yields ``{}``
      with a warning -- the whole trial falls back to filename
      convention.
    * A malformed entry (missing ``path`` / ``class``) is logged and
      skipped, so that one path falls back to convention.
    * An entry with an *unknown* class is logged and kept in the map
      pinned to :data:`HEAVY` (the conservative choice for disk), not
      skipped -- so it is retained/pruned as a heavy artifact rather
      than deferring to convention.
    """
    manifest_path = trial_dir / RETENTION_MANIFEST_NAME
    # A symlinked manifest could point read_text() at an arbitrary file
    # outside the trial tree (is_file()/read_text() dereference symlinks).
    # Mirror the deletion-side symlink/containment guards: treat it as
    # malformed and fall back to filename convention.
    if manifest_path.is_symlink():
        log.warning(
            "retention: %s in %s is a symlink; treating as malformed and "
            "falling back to filename convention",
            RETENTION_MANIFEST_NAME,
            trial_dir,
        )
        return {}
    if not manifest_path.is_file():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = raw["artifacts"]
        if not isinstance(entries, list):
            raise TypeError(
                f"'artifacts' must be a list, got {type(entries).__name__}"
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning(
            "retention: ignoring malformed %s in %s (%s); "
            "falling back to filename convention",
            RETENTION_MANIFEST_NAME,
            trial_dir,
            exc,
        )
        return {}
    mapping: dict[str, str] = {}
    for entry in entries:
        try:
            path = str(entry["path"])
            cls = str(entry["class"])
        except (TypeError, KeyError):
            log.warning("retention: skipping malformed manifest entry %r", entry)
            continue
        if cls not in _MANIFEST_CLASSES:
            log.warning(
                "retention: manifest entry %r has unknown class %r (allowed: %s); "
                "treating as heavy",
                path,
                cls,
                sorted(_MANIFEST_CLASSES),
            )
            cls = HEAVY
        mapping[path] = cls
    return mapping


def apply_retention(trial_dir: Path, level: str) -> RetentionOutcome:
    """Prune ``trial_dir`` to the artifacts kept by retention ``level``.

    Deletes every file whose class outranks ``level`` -- except the trial
    record, which is hard-guarded. Empty directories left behind are
    removed. ``level == "full"`` (the default) is a fast no-op: nothing is
    scanned or deleted, preserving today's keep-everything behaviour.

    Unknown ``level`` values are treated as ``full`` (fail-safe: never
    delete more than asked) with a warning.

    Symlinks are never unlinked, and any path that dereferences outside
    ``trial_dir`` is skipped (kept) with a warning -- a buggy or hostile
    collector symlink can never make retention delete files outside the
    trial output tree. Enumeration uses ``os.walk(..., followlinks=False)``
    so retention never even *descends* into a symlinked subdirectory (a
    collector that drops a symlinked dir pointing at, say, ``/`` could
    otherwise make a naive ``rglob`` walk an arbitrary, huge external tree).
    """
    if level not in _LEVEL_RANK:
        log.warning("retention: unknown level %r; keeping everything (full)", level)
        return RetentionOutcome(level="full", no_op=True)
    if level == "full":
        return RetentionOutcome(level="full", no_op=True)
    if not trial_dir.is_dir():
        return RetentionOutcome(level=level, no_op=True)

    manifest = _load_manifest(trial_dir)
    level_rank = _LEVEL_RANK[level]
    root = trial_dir.resolve()
    deleted: list[str] = []
    kept: list[str] = []
    freed = 0

    for path in _iter_trial_files(trial_dir, kept):
        # ``_iter_trial_files`` already excluded symlinks (files and
        # directories) and never descended into symlinked dirs. Defence in
        # depth: still refuse to delete a path that dereferences outside
        # ``trial_dir``. Mirrors the containment guard in
        # ``aorta.bundle.writer``, but retention is tolerant -- it keeps the
        # offending entry and warns rather than aborting the sweep.
        if not path.is_file():
            continue
        rel = path.relative_to(trial_dir).as_posix()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            log.warning(
                "retention: skipping %s; resolves outside trial dir (%s); keeping it",
                rel,
                resolved,
            )
            kept.append(rel)
            continue
        cls = classify_artifact(rel, manifest)
        if cls == RECORD or _CLASS_RANK[cls] <= level_rank:
            kept.append(rel)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        try:
            path.unlink()
        except OSError as exc:
            log.warning("retention: could not delete %s (%s); keeping it", path, exc)
            kept.append(rel)
            continue
        freed += size
        deleted.append(rel)

    _prune_empty_dirs(trial_dir)
    return RetentionOutcome(
        level=level,
        deleted=tuple(deleted),
        kept=tuple(kept),
        freed_bytes=freed,
    )


def _iter_trial_files(trial_dir: Path, kept: list[str]) -> list[Path]:
    """Enumerate the regular files under ``trial_dir``, symlink-safe.

    Uses ``os.walk(..., followlinks=False)`` so retention never descends
    into a symlinked subdirectory -- a collector that drops a symlink to an
    external tree (e.g. ``/``) can't make retention walk an arbitrary,
    possibly huge filesystem. (The ``resolve()`` containment check in
    :func:`apply_retention` blocks *deletion* outside the tree, but not the
    traversal itself, which is the DoS this avoids.)

    Symlinks -- whether to files or directories -- are recorded in ``kept``
    with a warning and never returned, so the deletion side never unlinks a
    link (which would reclaim no disk and could escape the trial tree).
    Returns the regular files in a deterministic sorted order.
    """
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(trial_dir, followlinks=False):
        base = Path(dirpath)
        # Symlinked subdirs appear in ``dirnames``; os.walk won't recurse
        # into them (followlinks=False). Record + drop them so they are
        # neither descended into nor later treated as prunable.
        symlinked_dirs = {d for d in dirnames if (base / d).is_symlink()}
        for name in symlinked_dirs:
            rel = (base / name).relative_to(trial_dir).as_posix()
            log.warning("retention: skipping symlink %s; not pruning it", rel)
            kept.append(rel)
        dirnames[:] = [d for d in dirnames if d not in symlinked_dirs]
        for name in filenames:
            path = base / name
            if path.is_symlink():
                rel = path.relative_to(trial_dir).as_posix()
                log.warning("retention: skipping symlink %s; not pruning it", rel)
                kept.append(rel)
                continue
            files.append(path)
    return sorted(files)


def _prune_empty_dirs(trial_dir: Path) -> None:
    """Remove now-empty subdirectories left after deleting heavy files.

    Walks bottom-up (``os.walk(topdown=False)``) so a directory emptied
    only because its children were pruned is itself removed. As in
    :func:`_iter_trial_files`, ``followlinks=False`` keeps the walk from
    descending into symlinked dirs, and symlinked dirs are never
    ``rmdir``-ed. The trial directory root is never removed.
    """
    for dirpath, _dirnames, _filenames in os.walk(
        trial_dir, topdown=False, followlinks=False
    ):
        path = Path(dirpath)
        if path == trial_dir or path.is_symlink():
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            try:
                path.rmdir()
            except OSError:
                pass
        except OSError:
            pass


__all__ = [
    "HEAVY",
    "LOG",
    "RECORD",
    "RETAIN_LEVELS",
    "RETENTION_MANIFEST_NAME",
    "SUMMARY",
    "RetentionOutcome",
    "apply_retention",
    "classify_artifact",
]
