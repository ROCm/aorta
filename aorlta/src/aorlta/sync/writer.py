"""C5 steps 2, 3 and 7: the name map, the slot writer, slot cleanup (design Section 6.5).

TokenSpeed's Qwen3.5 `load_weights` silently skips a tensor name it does not know, yet counts it as read
(Appendix A, A5/A6). So every name written must be a base-index key, and the written set must be exactly the
target keys (partial checkpoint, D11) or exactly every base key (full checkpoint). Anything else stops the run
before a byte is written.

A slot is `<sync.dir>/<run_id>/w{n mod slots}`: safetensors shards of at most 4 GiB, a
`model.safetensors.index.json` for those names, the base directory's config and tokenizer files, and
`manifest.json` with `{name: shape, dtype, xxh64}`. It is written as `w{s}.partial`, fsynced, then renamed, so an
engine never sees a half-written slot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from aorlta.atomic import PARTIAL_SUFFIX, atomic_write_json, read_text
from aorlta.ids import sha256_json
from aorlta.sync.merge import TARGET_KEY_RE

INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "aorlta.sync.manifest/1"
SHARD_LIMIT = 4 * 2**30
COPY_SUFFIXES = (".json", ".jinja", ".txt")
DEFAULT_MARGIN = 1 << 30  # bytes kept free beyond two slots
_LAYER_RE = re.compile(r"layers\.(\d+)\.(.+)$")
_DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


class SyncError(RuntimeError):
    """A weight-sync step failed; the run stops (Section 6.5, failure modes)."""


class NameMapError(SyncError):
    """A name would be written that is not exactly one of the base checkpoint's keys."""


class ShmFullError(SyncError):
    """Not enough free space for the slot."""


# --- the name map (step 2) ---------------------------------------------------------------------------


def read_base_index(base_dir: str | Path) -> dict[str, str]:
    """key -> shard file, from the base's `model.safetensors.index.json` (required)."""
    path = Path(base_dir) / INDEX_NAME
    if not path.is_file():
        raise SyncError(f"{path}: the base checkpoint has no index; the name map needs one")
    data = json.loads(read_text(path))
    wm = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(wm, dict) or not wm or not all(isinstance(k, str) and isinstance(v, str) for k, v in wm.items()):
        raise SyncError(f"{path}: weight_map missing or malformed")
    return dict(wm)


def target_keys(base_keys: Iterable[str]) -> frozenset[str]:
    return frozenset(k for k in base_keys if TARGET_KEY_RE.fullmatch(k))


def module_key(module_name: str, targets: frozenset[str]) -> str:
    """The prototype's `ckpt_key`: the one target key ending in `layers.<i>.<rest>.weight`."""
    m = _LAYER_RE.search(module_name)
    if not m:
        raise NameMapError(f"module {module_name!r} has no layers.<i>.<name> path")
    suffix = f"layers.{m.group(1)}.{m.group(2)}.weight"
    hits = [k for k in targets if k == suffix or k.endswith("." + suffix)]
    if len(hits) != 1:
        raise NameMapError(f"module {module_name!r} maps to {len(hits)} base keys {sorted(hits)[:3]}, not exactly 1")
    return hits[0]


def check_names(written: Iterable[str], base_keys: Iterable[str], *, partial: bool = True) -> None:
    """The written set must equal the base's target keys (partial) or all its keys (full), exactly."""
    written = set(written)
    base = set(base_keys)
    unknown = sorted(written - base)
    expected = target_keys(base) if partial else frozenset(base)
    missing = sorted(expected - written)
    extra = sorted(written - expected)
    if unknown or missing or extra:
        raise NameMapError(
            f"the names to write are not exactly the {len(expected)} {'target' if partial else 'base'} keys: "
            f"{len(unknown)} not in the base index {unknown[:3]}, {len(missing)} missing {missing[:3]}, "
            f"{len(extra)} not targets {extra[:3]} (TokenSpeed would skip an unknown name and count it as read)"
        )


def name_map(module_names: Iterable[str], base_keys: Iterable[str]) -> dict[str, str]:
    """module name -> checkpoint key, one to one, covering every target key of the base index."""
    base_keys = list(base_keys)
    targets = target_keys(base_keys)
    mapping = {name: module_key(name, targets) for name in module_names}
    if len(set(mapping.values())) != len(mapping):
        raise NameMapError("two modules map to the same checkpoint key")
    check_names(mapping.values(), base_keys, partial=True)
    return mapping


def expected_count(base_keys: Iterable[str], *, partial: bool) -> int:
    """The tensor count a reload of a slot must report: 248 (partial) or 738 (full) for Qwen3.5-4B."""
    base_keys = list(base_keys)
    return len(target_keys(base_keys)) if partial else len(set(base_keys))


# --- the slot (steps 3 and 7) ------------------------------------------------------------------------


def slot_path(sync_dir: str | Path, run_id: str, version: int, slots: int = 2) -> Path:
    if slots < 2:
        raise SyncError(f"slots must be >= 2, got {slots}")
    if not isinstance(version, int) or version < 0:
        raise SyncError(f"version must be an int >= 0, got {version!r}")
    return Path(sync_dir) / run_id / f"w{version % slots}"


def tensor_bytes(t: torch.Tensor) -> memoryview:
    """The tensor's bytes as safetensors stores them (little-endian, C order)."""
    return memoryview(t.detach().contiguous().view(torch.uint8).numpy()).cast("B")


def tensor_xxh64(t: torch.Tensor) -> str:
    import xxhash

    return xxhash.xxh64(tensor_bytes(t)).hexdigest()


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def free_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def _plan_shards(sizes: Mapping[str, int]) -> list[list[str]]:
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_size = 0
    for name in sorted(sizes):
        n = sizes[name]
        if n > SHARD_LIMIT:
            raise SyncError(f"{name}: {n} bytes is more than one 4 GiB shard")
        if cur and cur_size + n > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += n
    if cur:
        shards.append(cur)
    return shards


@dataclass(frozen=True)
class SlotInfo:
    path: Path
    version: int
    slot: int
    partial: bool
    tensors: int
    bytes: int
    seconds: float
    manifest_sha256: str


def write_slot(
    weights: Mapping[str, torch.Tensor],
    *,
    base_dir: str | Path,
    sync_dir: str | Path,
    run_id: str,
    version: int,
    slots: int = 2,
    partial: bool = True,
    others: Mapping[str, torch.Tensor] | None = None,
    margin: int = DEFAULT_MARGIN,
) -> SlotInfo:
    """Write W_eff (keyed by checkpoint key) as slot `version mod slots`. A full checkpoint (`partial=False`)
    also writes `others`, the base's remaining tensors."""
    t0 = time.monotonic()
    base_dir = Path(base_dir)
    base_keys = list(read_base_index(base_dir))
    tensors = dict(weights)
    if not partial:
        if others is None:
            raise SyncError("a full checkpoint needs the base's other tensors")
        overlap = sorted(set(tensors) & set(others))
        if overlap:
            raise SyncError(f"W_eff and the other base tensors overlap: {overlap[:3]}")
        tensors.update(others)
    check_names(tensors, base_keys, partial=partial)  # stop before writing anything

    for name, t in tensors.items():
        if t.dtype not in _DTYPE_NAMES:
            raise SyncError(f"{name}: dtype {t.dtype} is not one the writer knows")
    sizes = {name: t.numel() * t.element_size() for name, t in tensors.items()}
    total = sum(sizes.values())
    final = slot_path(sync_dir, run_id, version, slots)
    # the served slot already takes its space, and a slot being replaced keeps its own until the rename below
    need = total + margin
    free = free_bytes(final.parent)
    if free < need:
        raise ShmFullError(f"{final.parent}: {free} bytes free, need {need} ({total} for the slot + margin {margin})")

    tmp = final.with_name(final.name + PARTIAL_SUFFIX)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    from safetensors.torch import save_file

    plan = _plan_shards(sizes)
    weight_map: dict[str, str] = {}
    entries: dict[str, dict] = {}
    for i, names in enumerate(plan, start=1):
        fn = f"model-{i:05d}-of-{len(plan):05d}.safetensors"
        shard = {}
        for name in names:
            t = tensors[name].detach().to("cpu").contiguous()
            shard[name] = t
            entries[name] = {"shape": list(t.shape), "dtype": _DTYPE_NAMES[t.dtype], "xxh64": tensor_xxh64(t)}
            weight_map[name] = fn
        save_file(shard, str(tmp / fn), metadata={"format": "pt"})
        _fsync_path(tmp / fn)
        del shard
    atomic_write_json(tmp / INDEX_NAME, {"metadata": {"total_size": total}, "weight_map": weight_map})
    for f in sorted(base_dir.iterdir()):
        if f.is_file() and f.suffix in COPY_SUFFIXES and f.name not in (INDEX_NAME, MANIFEST_NAME):
            shutil.copyfile(f, tmp / f.name)
            _fsync_path(tmp / f.name)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "version": version,
        "partial": partial,
        "hash": "xxh64",
        "tensors": entries,
    }
    atomic_write_json(tmp / MANIFEST_NAME, manifest)
    _fsync_path(tmp)
    if final.exists():  # slot n-2, never the one served (that is slot n-1)
        shutil.rmtree(final)
    os.rename(tmp, final)
    _fsync_path(final.parent)
    return SlotInfo(
        path=final,
        version=version,
        slot=version % slots,
        partial=partial,
        tensors=len(entries),
        bytes=total,
        seconds=time.monotonic() - t0,
        manifest_sha256=sha256_json(manifest),
    )


def read_manifest(slot: str | Path) -> dict:
    slot = Path(slot)
    if slot.name.endswith(PARTIAL_SUFFIX):
        raise SyncError(f"refusing to read a slot that was never completed: {slot}")
    data = json.loads(read_text(slot / MANIFEST_NAME))
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise SyncError(f"{slot}: not a {MANIFEST_SCHEMA} manifest")
    return data


def load_slot_tensors(slot: str | Path, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    """The slot's tensors, read back from its shards on the CPU (for checks, not for the engine)."""
    from safetensors import safe_open

    slot = Path(slot)
    read_manifest(slot)
    wm = json.loads(read_text(slot / INDEX_NAME))["weight_map"]
    want = set(wm) if names is None else set(names)
    out: dict[str, torch.Tensor] = {}
    for fn in sorted(set(wm.values())):
        with safe_open(str(slot / fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in want:
                    out[k] = f.get_tensor(k).clone()  # no view into the file: the slot may be deleted later
    if set(out) != want:
        raise SyncError(f"{slot}: {len(want - set(out))} requested tensors not found")
    return out


def verify_slot(slot: str | Path) -> dict:
    """Re-read every tensor and compare shape, dtype and xxh64 with the manifest, and the index with the files: each
    shard must hold exactly the tensors the index assigns to it, as an engine reads every tensor of every shard."""
    from safetensors import safe_open

    slot = Path(slot)
    manifest = read_manifest(slot)
    entries = manifest["tensors"]
    wm = json.loads(read_text(slot / INDEX_NAME))["weight_map"]
    bad = []
    for fn in sorted(set(wm.values())):
        with safe_open(str(slot / fn), framework="pt", device="cpu") as f:
            held = set(f.keys())
        assigned = {k for k, v in wm.items() if v == fn}
        if held != assigned:
            bad.append(f"{fn}: holds {len(held - assigned)} tensors the index does not assign to it "
                       f"{sorted(held - assigned)[:3]}, lacks {len(assigned - held)} {sorted(assigned - held)[:3]}")
    if bad:
        return {"ok": False, "tensors": len(wm), "problems": bad[:10]}
    tensors = load_slot_tensors(slot)
    if set(tensors) != set(entries):
        bad.append(f"index names {len(tensors)} != manifest names {len(entries)}")
    for name, t in tensors.items():
        e = entries.get(name)
        got = {"shape": list(t.shape), "dtype": _DTYPE_NAMES.get(t.dtype), "xxh64": tensor_xxh64(t)}
        if e != got:
            bad.append(f"{name}: manifest {e} != file {got}")
    return {"ok": not bad, "tensors": len(tensors), "problems": bad[:10]}


def delete_slot(slot: str | Path) -> None:
    shutil.rmtree(slot, ignore_errors=True)


def prune_slots(sync_dir: str | Path, run_id: str, keep: str | Path) -> list[Path]:
    """Step 7: after the landing check passes, delete every slot of the run except `keep` (the one served),
    and any `.partial` left by an interrupted write."""
    keep = Path(keep).resolve()
    root = Path(sync_dir) / run_id
    removed = []
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_dir() and re.fullmatch(r"w\d+(\.partial)?", p.name) and p.resolve() != keep:
                shutil.rmtree(p)
                removed.append(p)
    return removed


def load_base_tensors(base_dir: str | Path, keys: Iterable[str]) -> dict[str, torch.Tensor]:
    """The named base tensors on the CPU, as stored (for a full checkpoint's untouched part)."""
    from safetensors import safe_open

    base_dir = Path(base_dir)
    wm = read_base_index(base_dir)
    want = set(keys)
    unknown = sorted(want - set(wm))
    if unknown:
        raise NameMapError(f"not in the base index: {unknown[:3]}")
    out: dict[str, torch.Tensor] = {}
    for fn in sorted({wm[k] for k in want}):
        with safe_open(str(base_dir / fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in want:
                    out[k] = f.get_tensor(k).clone()
    return out
