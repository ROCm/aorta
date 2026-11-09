"""Run state, prompt selection and checkpoints (design Section 6.6, "Checkpoint and resume").

Prompt selection is a pure function of (pool, run_seed, iteration, P): each epoch is a permutation of the pool,
ordered by sha256 of (order seed, prompt id), and cut into floor(M / P) blocks of P prompts; iteration n takes block
(n - 1) mod floor(M / P) of epoch (n - 1) div floor(M / P). No prompt repeats within an iteration, the M mod P
prompts at the end of an epoch's permutation wait for the next epoch, and a resumed run selects exactly what an
uninterrupted run would. The cursor (iteration, epoch, block, order seed, pool hash) is still saved in every
checkpoint and checked at resume, so a changed pool or P is refused rather than silently reshuffled.

A checkpoint is the state at the end of iteration n: the trainer's files (LoRA latent, Adam moments, RNG states),
`state.json` (the data cursor, the policy version, the slot the engines hold, the config hash, the phase history
the deadlines read, the NOOP streak) and `MANIFEST.json` (sha256 of every file). It is written into
`ckpt/iter-<n>.partial/`, fsynced, and renamed to `ckpt/iter-<n>/` just before the iteration's own directory is
renamed; that second rename commits iteration n (resume.py).
"""

from __future__ import annotations

import os
import random
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..atomic import PARTIAL_SUFFIX, atomic_write_json, read_text
from ..config import ConfigError, plain, strict_decode
from ..ids import sha256_file, sha256_json
from ..schemas import PromptRecord

PROMPT_ORDER_SCHEMA = "aorlta.prompt_order/1"
RUN_STATE_SCHEMA = "aorlta.run_state/1"
MANIFEST_SCHEMA = "aorlta.ckpt_manifest/1"
STATE_NAME, MANIFEST_NAME, TRAINER_DIR = "state.json", "MANIFEST.json", "trainer"
CKPT_RE = re.compile(r"iter-([0-9]{6})")


class CheckpointError(RuntimeError):
    """A checkpoint is missing, incomplete or does not match its manifest; the run does not resume from it."""


def iteration_dir_name(n: int) -> str:
    if not (isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= 999_999):
        raise ValueError(f"iteration must be an int in [0, 999999], got {n!r}")
    return f"iter-{n:06d}"


# --- prompt selection ---------------------------------------------------------------------------------------


def pool_digest(records: Sequence[PromptRecord]) -> str:
    """sha256 of the pool's prompts, independent of their order; a prompt id may appear only once."""
    ids = [r.prompt_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("a prompt id appears twice in the pool")
    rows = sorted([r.prompt_id, r.prompt_ids_sha256, r.task.task_sha256, r.source] for r in records)
    return sha256_json(rows)


def order_seed(run_seed: int, epoch: int) -> str:
    return sha256_json([PROMPT_ORDER_SCHEMA, run_seed, epoch])


def epoch_order(records: Sequence[PromptRecord], run_seed: int, epoch: int) -> list[PromptRecord]:
    seed_hex = order_seed(run_seed, epoch)
    return sorted(records, key=lambda r: sha256_json([seed_hex, r.prompt_id]))


def cursor_for(iteration: int, records: Sequence[PromptRecord], run_seed: int, P: int) -> dict:
    """The data cursor of `iteration` (from 1): where its prompts come from."""
    if not (isinstance(iteration, int) and iteration >= 1):
        raise ValueError(f"iteration must be an int >= 1, got {iteration!r}")
    M = len(records)
    if M < P:
        raise ValueError(f"the prompt pool has {M} prompts, fewer than P = {P}")
    per_epoch = M // P
    epoch, block = divmod(iteration - 1, per_epoch)
    return {"iteration": iteration, "epoch": epoch, "block": block, "iterations_per_epoch": per_epoch,
            "order_seed": order_seed(run_seed, epoch), "pool_sha256": pool_digest(records), "pool_size": M,
            "P": P, "curriculum": []}


def select_prompts(records: Sequence[PromptRecord], iteration: int, run_seed: int, P: int
                   ) -> tuple[list[PromptRecord], dict]:
    cursor = cursor_for(iteration, records, run_seed, P)
    order = epoch_order(records, run_seed, cursor["epoch"])
    b = cursor["block"]
    return order[b * P:(b + 1) * P], cursor


# --- run state ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunState:
    schema: str
    run_id: str
    source: str
    iteration: int  # the last completed iteration (0: none yet)
    policy_version: str  # the weights after `iteration` (v0 is the merged base)
    config_sha256: str
    next_cursor: dict  # the data cursor of iteration `iteration` + 1
    engines_slot: str | None  # the slot the engines hold
    history: dict  # {"generation_s": [...], "scoring_s": [...]}, for the rolling medians
    noop_streak: int
    trainer_digest: str  # the trainer's state digest when this state was saved
    base_sha256: str | None  # the merged base's file manifest hash (run_plan.json)
    rng: dict = field(default_factory=dict)  # {"python_random": [...]}

    def to_json(self) -> dict:
        return plain(self)

    @classmethod
    def from_json(cls, data: dict) -> RunState:
        try:
            state = strict_decode(cls, data, "run_state")
        except ConfigError as exc:
            raise CheckpointError(f"state.json: {exc}") from None
        if state.schema != RUN_STATE_SCHEMA:
            raise CheckpointError(f"state.json: schema {state.schema!r}, expected {RUN_STATE_SCHEMA!r}")
        return state


def python_rng_state() -> list:
    version, internal, gauss = random.getstate()
    return [version, list(internal), gauss]


def restore_python_rng(saved: list) -> None:
    version, internal, gauss = saved
    random.setstate((version, tuple(internal), gauss))


# --- checkpoint files ----------------------------------------------------------------------------------------


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        if not path.is_dir():
            raise
    finally:
        os.close(fd)


def fsync_tree(root: Path) -> None:
    for dirpath, _, files in os.walk(root):
        for name in files:
            _fsync_path(Path(dirpath) / name)
        _fsync_path(Path(dirpath))


def file_manifest(root: Path, *, skip: Sequence[str] = ()) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and rel not in skip:
            out[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    return out


@dataclass
class PendingCheckpoint:
    """A checkpoint written and fsynced under its `.partial` name; `commit()` renames it into place."""

    tmp: Path
    final: Path

    def commit(self) -> Path:
        os.replace(self.tmp, self.final)
        _fsync_path(self.final.parent)
        return self.final


def write_checkpoint(ckpt_root: Path, state: RunState, save_trainer: Callable[[Path], None], *,
                     point: Callable[[str], None] = lambda name: None) -> PendingCheckpoint:
    """Write iteration `state.iteration`'s checkpoint under `ckpt/iter-<n>.partial/` (not yet committed). A
    committed checkpoint is never overwritten, and a leftover `.partial` is refused (resume removes it)."""
    name = iteration_dir_name(state.iteration)
    final, tmp = ckpt_root / name, ckpt_root / (name + PARTIAL_SUFFIX)
    if final.exists():
        raise CheckpointError(f"{final} already exists: iteration {state.iteration} is already committed")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    tmp.mkdir()  # FileExistsError on a leftover: resume must have removed it
    (tmp / TRAINER_DIR).mkdir()
    save_trainer(tmp / TRAINER_DIR)
    point("ckpt.trainer_saved")
    atomic_write_json(tmp / STATE_NAME, state.to_json(), canonical=True)
    point("ckpt.state_written")
    files = file_manifest(tmp)
    atomic_write_json(tmp / MANIFEST_NAME, {"schema": MANIFEST_SCHEMA, "iteration": state.iteration, "files": files,
                                            "files_sha256": sha256_json(files)}, canonical=True)
    fsync_tree(tmp)
    point("ckpt.partial_written")
    return PendingCheckpoint(tmp, final)


def complete_checkpoints(ckpt_root: Path) -> list[int]:
    if not ckpt_root.is_dir():
        return []
    return sorted(int(m.group(1)) for p in ckpt_root.iterdir() if p.is_dir() and (m := CKPT_RE.fullmatch(p.name)))


def verify_checkpoint(path: Path) -> RunState:
    import json

    manifest_path = path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CheckpointError(f"{path}: no {MANIFEST_NAME}")
    try:
        manifest = json.loads(read_text(manifest_path))
    except ValueError as exc:
        raise CheckpointError(f"{manifest_path}: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise CheckpointError(f"{manifest_path}: not a {MANIFEST_SCHEMA} manifest")
    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or manifest.get("files_sha256") != sha256_json(recorded):
        raise CheckpointError(f"{manifest_path}: the file list does not match its hash")
    on_disk = file_manifest(path, skip=(MANIFEST_NAME,))
    if on_disk != recorded:
        missing = sorted(set(recorded) - set(on_disk))
        extra = sorted(set(on_disk) - set(recorded))
        changed = sorted(k for k in set(recorded) & set(on_disk) if recorded[k] != on_disk[k])
        raise CheckpointError(f"{path}: files differ from the manifest (missing {missing}, extra {extra}, "
                              f"changed {changed})")
    state = RunState.from_json(json.loads(read_text(path / STATE_NAME)))
    m = CKPT_RE.fullmatch(path.name)
    if m is None or int(m.group(1)) != state.iteration or manifest.get("iteration") != state.iteration:
        raise CheckpointError(f"{path}: holds iteration {state.iteration}")
    return state


def prune_checkpoints(ckpt_root: Path, keep: int) -> list[str]:
    """Delete all but the newest `keep` committed checkpoints. Each is first renamed to a `.partial` name (one
    atomic step), so a crash mid-delete leaves only something resume removes."""
    done = complete_checkpoints(ckpt_root)
    removed = []
    for n in done[:-keep] if keep < len(done) else []:
        path = ckpt_root / iteration_dir_name(n)
        doomed = ckpt_root / (iteration_dir_name(n) + ".pruned" + PARTIAL_SUFFIX)
        os.replace(path, doomed)
        shutil.rmtree(doomed)
        removed.append(path.name)
    return removed
