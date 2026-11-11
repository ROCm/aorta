"""Resume from the last complete checkpoint (design Section 6.6, "Checkpoint and resume").

    load the last complete checkpoint; delete every `.partial` file and any outputs of the incomplete iteration;
    start fresh engines from the base; round trip; merge the checkpoint's LoRA latent, write it and reload;
    landing check; redo iteration n from step 1.

An iteration is committed when its directory `iterations/iter-<n>/` appears; its checkpoint is renamed into place
just before that (loop.py, step 10). `prepare_resume` is the file half of a resume:

- the last committed iteration n_c is the newest n with both `ckpt/iter-<n>/` and `iterations/iter-<n>/`; that
  checkpoint is verified against its manifest, and a committed checkpoint that fails the check stops the resume;
- every `.partial` file or directory under the run directory is removed (Section 5 item 6: readers never open
  them, and a new process cannot finish what a dead one started), except in the external evaluator's outbox, which
  that evaluator may still be writing;
- a checkpoint newer than n_c is removed: it was renamed, but its iteration never committed;
- records of iterations after n_c are cut from the append-only ledgers and the evidence manifest (a ledger with a
  bad line before its end stops the resume before anything is removed; only a torn final line is dropped); TensorBoard
  files after n_c are removed, and a committed iteration's file that the crash prevented is written from its
  committed summary;
- checkpoints beyond `keep_ckpts` are pruned (a crash between a commit and its prune leaves one too many).

The reward cache is kept: its entries are keyed by content and checker version, so legs that settled before the
crash make the re-run cheap. The engine half (fresh engines, round trip, restore, landing check) is the loop's.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..atomic import PARTIAL_SUFFIX
from ..episodes import tensorboard as tb
from ..episodes.writer import (EVIDENCE_DIR, LEDGERS_DIR, AppendLedger, EvidenceStore, complete_iterations,
                               read_summary)
from .state import RunState, complete_checkpoints, iteration_dir_name, prune_checkpoints, verify_checkpoint


@dataclass
class ResumePoint:
    iteration: int  # the last committed iteration (0: none)
    state: RunState | None
    checkpoint: Path | None
    removed: list[str] = field(default_factory=list)


# Directories another process writes while the run restarts: the external evaluator publishes its results through
# `.partial` files in its outbox, and may still be running.
NOT_THE_RUNS = (Path("eval") / "outbox",)


def remove_partials(root: Path, *, skip: tuple[Path, ...] = ()) -> list[str]:
    """Delete every `.partial` entry under `root`, without descending into the ones it deletes or into `skip`
    (paths relative to `root`)."""
    removed = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if (here / d).relative_to(root) not in skip]
        for d in list(dirnames):
            if d.endswith(PARTIAL_SUFFIX):
                shutil.rmtree(here / d)
                dirnames.remove(d)
                removed.append(str((here / d).relative_to(root)))
        for f in filenames:
            if f.endswith(PARTIAL_SUFFIX):
                (here / f).unlink()
                removed.append(str((here / f).relative_to(root)))
    return removed


def last_committed(run_dir: Path) -> int:
    outputs = set(complete_iterations(run_dir))
    return max((n for n in complete_checkpoints(run_dir / "ckpt") if n in outputs), default=0)


def prepare_resume(run_dir: Path, keep_ckpts: int) -> ResumePoint:
    run_dir = Path(run_dir)
    ckpt_root = run_dir / "ckpt"
    n_c = last_committed(run_dir)
    state = path = None
    if n_c:
        path = ckpt_root / iteration_dir_name(n_c)
        state = verify_checkpoint(path)  # raises CheckpointError: never fall back past a committed iteration
    ledgers = [AppendLedger(p) for p in sorted((run_dir / LEDGERS_DIR).glob("*.jsonl"))]
    store = EvidenceStore(run_dir / EVIDENCE_DIR) if (run_dir / EVIDENCE_DIR).is_dir() else None
    for ledger in ledgers + ([store.manifest] if store else []):
        ledger.read()  # raises LedgerError on a bad line before a ledger's end, before anything is removed
    removed = remove_partials(run_dir, skip=NOT_THE_RUNS)
    for n in complete_checkpoints(ckpt_root):
        if n > n_c:
            shutil.rmtree(ckpt_root / iteration_dir_name(n))
            removed.append(f"ckpt/{iteration_dir_name(n)}")
    later = [n for n in complete_iterations(run_dir) if n > n_c]
    if later:  # an iteration directory is only renamed after its checkpoint, so this means the checkpoint is gone
        raise RuntimeError(f"iterations {later} are committed but their checkpoints are missing; refusing to resume")
    removed += [f"tb/{name}" for name in tb.remove_after(run_dir / "tb", n_c)]
    have_tb = {n for n, _ in tb.tb_files(run_dir / "tb")}
    for n in complete_iterations(run_dir):
        if n not in have_tb:
            summary = read_summary(run_dir, n)
            tb.write_iteration(run_dir / "tb", n, tb.scalars_from_summary(summary), summary["timing"]["t_landed"])
            removed.append(f"tb: rewrote iteration {n} from its summary")
    for ledger in ledgers:
        dropped = ledger.truncate_after(n_c)
        if dropped:
            removed.append(f"{LEDGERS_DIR}/{ledger.path.name}: {dropped} records")
    if store is not None:
        dropped = store.manifest.truncate_after(n_c)
        if dropped:
            removed.append(f"{EVIDENCE_DIR}/MANIFEST.jsonl: {dropped} records")
        removed += [f"{EVIDENCE_DIR}/blobs/{sha}" for sha in store.prune_unlisted()]
    removed += [f"ckpt/{name}" for name in prune_checkpoints(ckpt_root, keep_ckpts)]
    return ResumePoint(n_c, state, path, removed)
