"""The slot's own checks (design Section 6.5, steps 3 and 7): `verify_slot` against shards that disagree with the
index, and the free-space rule on a filesystem sized for two slots and the margin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aorlta.sync import writer as W

MARGIN = 1 << 16


def targets(tiny_base) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    with safe_open(str(tiny_base / "model.safetensors"), framework="pt") as f:
        return {k: f.get_tensor(k) for k in W.target_keys(f.keys())}


def rewrite(shard: Path, change) -> None:
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(shard))
    change(tensors)
    save_file(tensors, str(shard), metadata={"format": "pt"})


def test_verify_slot_refuses_a_shard_holding_a_tensor_the_index_does_not_name(tiny_base, tmp_path):
    info = W.write_slot(targets(tiny_base), base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=0)
    assert W.verify_slot(info.path)["ok"]
    (shard,) = set(json.loads((info.path / W.INDEX_NAME).read_text())["weight_map"].values())
    embed = "model.language_model.embed_tokens.weight"  # a real parameter, not a target
    rewrite(info.path / shard, lambda t: t.update({embed: torch.zeros(320, 64, dtype=torch.bfloat16)}))
    res = W.verify_slot(info.path)
    assert not res["ok"] and embed in " ".join(res["problems"])


def test_verify_slot_refuses_a_tensor_in_another_shard_than_the_index_says(tiny_base, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "SHARD_LIMIT", 40_000)
    info = W.write_slot(targets(tiny_base), base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=0)
    assert W.verify_slot(info.path)["ok"]
    wm = json.loads((info.path / W.INDEX_NAME).read_text())["weight_map"]
    first, second = sorted(set(wm.values()))[:2]
    name = min(k for k, fn in wm.items() if fn == first)
    moved = {}
    rewrite(info.path / first, lambda t: moved.update({name: t.pop(name)}))
    rewrite(info.path / second, lambda t: t.update(moved))
    res = W.verify_slot(info.path)
    assert not res["ok"] and name in " ".join(res["problems"])


def used(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def slot_bytes(tiny_base, tmp_path, weights) -> int:
    info = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path / "probe", run_id="r", version=0,
                        margin=MARGIN)
    return used(info.path)


def test_two_slots_and_the_margin_are_enough(tiny_base, tmp_path, monkeypatch):
    weights = targets(tiny_base)
    shm = tmp_path / "shm"
    capacity = 2 * slot_bytes(tiny_base, tmp_path, weights) + MARGIN
    monkeypatch.setattr(W, "free_bytes", lambda path: capacity - used(shm))
    for version in range(4):  # the served slot stays while the next is written; the older goes once it is
        info = W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=version, margin=MARGIN)
        assert W.free_bytes(shm) >= MARGIN
        W.prune_slots(shm, "r", keep=info.path)


def test_a_slot_that_would_leave_less_than_the_margin_free_is_refused(tiny_base, tmp_path, monkeypatch):
    weights = targets(tiny_base)
    total = sum(t.numel() * t.element_size() for t in weights.values())
    shm = tmp_path / "shm"
    capacity = slot_bytes(tiny_base, tmp_path, weights) + total + MARGIN - 1
    monkeypatch.setattr(W, "free_bytes", lambda path: capacity - used(shm))
    W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=0, margin=MARGIN)
    with pytest.raises(W.ShmFullError, match="free"):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=1, margin=MARGIN)
    assert not (shm / "r" / "w1.partial").exists()


def test_the_slot_being_replaced_still_counts_until_the_new_one_is_complete(tiny_base, tmp_path, monkeypatch):
    weights = targets(tiny_base)
    shm = tmp_path / "shm"
    capacity = 2 * slot_bytes(tiny_base, tmp_path, weights) + MARGIN
    for version in (0, 1):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=version, margin=MARGIN)
    monkeypatch.setattr(W, "free_bytes", lambda path: capacity - used(shm))
    with pytest.raises(W.ShmFullError, match="free"):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=2, margin=MARGIN)
    assert W.verify_slot(shm / "r" / "w0")["ok"] and not (shm / "r" / "w0.partial").exists()


def test_a_stale_partial_of_the_slot_is_removed_before_the_space_is_measured(tiny_base, tmp_path, monkeypatch):
    import shutil

    weights = targets(tiny_base)
    shm = tmp_path / "shm"
    capacity = 2 * slot_bytes(tiny_base, tmp_path, weights) + MARGIN
    monkeypatch.setattr(W, "free_bytes", lambda path: capacity - used(shm))
    served = W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=0, margin=MARGIN)
    stale = shm / "r" / "w1.partial"
    shutil.copytree(served.path, stale)  # a write of version 1 interrupted after its last shard
    info = W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=1, margin=MARGIN)
    assert W.verify_slot(info.path)["ok"] and W.verify_slot(served.path)["ok"] and not stale.exists()


def test_the_copied_files_count_toward_the_space_a_slot_needs(tiny_base, tmp_path, monkeypatch):
    import shutil

    base = tmp_path / "base"
    shutil.copytree(tiny_base, base)
    (base / "tokenizer.json").write_bytes(b"{}" + b" " * (4 * MARGIN))  # copied into every slot
    weights = targets(base)
    total = sum(t.numel() * t.element_size() for t in weights.values())
    shm = tmp_path / "shm"
    monkeypatch.setattr(W, "free_bytes", lambda path: total + 2 * MARGIN - used(shm))
    with pytest.raises(W.ShmFullError, match="for the copied files"):
        W.write_slot(weights, base_dir=base, sync_dir=shm, run_id="r", version=0, margin=MARGIN)
    assert not (shm / "r" / "w0.partial").exists() and not (shm / "r" / "w0").exists()


def test_a_slot_whose_metadata_leaves_less_than_the_margin_is_not_published(tiny_base, tmp_path, monkeypatch):
    weights = targets(tiny_base)
    shm = tmp_path / "shm"
    reads = iter([1 << 40])  # room at the start; then nothing left
    monkeypatch.setattr(W, "free_bytes", lambda path: next(reads, MARGIN - 1))
    with pytest.raises(W.ShmFullError, match="once slot 0 is written"):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=shm, run_id="r", version=0, margin=MARGIN)
    assert not (shm / "r" / "w0.partial").exists() and not (shm / "r" / "w0").exists()


@pytest.mark.parametrize("name", ["/abs/model.safetensors", "../model.safetensors", "sub/model.safetensors", ".."])
def test_a_base_index_naming_a_shard_outside_the_model_directory_is_refused(tiny_base, tmp_path, name):
    import shutil

    base = tmp_path / "base"
    shutil.copytree(tiny_base, base)
    index = json.loads((base / W.INDEX_NAME).read_text())
    key = next(iter(index["weight_map"]))
    index["weight_map"][key] = name
    (base / W.INDEX_NAME).write_text(json.dumps(index))
    with pytest.raises(W.ShardNameError, match="not file names in the directory") as exc:
        W.read_base_index(base)
    assert exc.value.names == [name]


def test_a_base_saved_as_one_file_without_an_index_is_read_from_its_header(tiny_base, tmp_path):
    """Hugging Face saves a checkpoint small enough for one file as `model.safetensors`, with no index."""
    import shutil

    from safetensors.torch import load_file, save_file

    base = tmp_path / "one"
    shutil.copytree(tiny_base, base)
    tensors = {}
    for shard in sorted(set(W.read_base_index(base).values())):
        tensors.update(load_file(str(base / shard)))
        (base / shard).unlink()
    (base / W.INDEX_NAME).unlink()
    save_file(tensors, str(base / W.SINGLE_FILE), metadata={"format": "pt"})
    assert W.read_base_index(base) == {k: W.SINGLE_FILE for k in tensors}
    weights = {k: v for k, v in tensors.items() if k in W.target_keys(tensors)}
    info = W.write_slot(weights, base_dir=base, sync_dir=tmp_path / "shm", run_id="r", version=0, margin=MARGIN)
    assert W.verify_slot(info.path)["ok"]
    assert set(W.load_base_tensors(base, list(tensors)[:3])) == set(list(tensors)[:3])


@pytest.mark.parametrize("damage", ["index", "single"])
def test_a_base_index_or_single_file_that_cannot_be_read_is_a_sync_error(tiny_base, tmp_path, damage):
    import shutil

    base = tmp_path / "bad"
    shutil.copytree(tiny_base, base)
    if damage == "index":
        (base / W.INDEX_NAME).write_text('{"weight_map":')
    else:
        (base / W.INDEX_NAME).unlink()
        (base / W.SINGLE_FILE).write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{broken}")
    with pytest.raises(W.SyncError):
        W.read_base_index(base)
