"""Read a safetensors checkpoint tree tensor by tensor, with the standard library only.

Shared by ``verify_checkpoint_delta.py``, which pairs two trees tensor by
tensor, and ``eval_episodes.py``, which needs to know whether two trees hold
the same tensors. Kept apart from the verifier because ``eval_episodes`` is
part of what ``eval_episodes.scorer_files`` digests, and the verifier is not.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """(metadata, offset of the data buffer) from a safetensors file."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        meta = json.loads(handle.read(length))
    meta.pop("__metadata__", None)
    return meta, 8 + length


def tensor_map(root: Path) -> dict[str, Path]:
    """Every tensor name in a checkpoint directory, mapped to the file holding it.

    Reads ``model.safetensors.index.json`` when the tree is sharded, else the
    single ``model.safetensors`` a small model is saved as. Each tree is read
    through its *own* index, so a PRE and POST sharded differently still pair
    tensor by tensor.
    """
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: root / shard for name, shard in weight_map.items()}
    single = root / "model.safetensors"
    if single.is_file():
        meta, _ = read_header(single)
        return dict.fromkeys(meta, single)
    raise FileNotFoundError(f"{root}: no model.safetensors.index.json or model.safetensors")


def tensor_identity(root: Path, chunk: int = 1 << 24) -> dict[str, Any]:
    """A sha256 over every tensor in a tree: name, dtype and shape, then raw bytes.

    In name order, through the tree's own index, so it depends on the tensors
    alone: not on how they are sharded, on the ``__metadata__`` a writer
    stamps, or on any README, config or tokenizer file saved beside them.
    """
    digest = hashlib.sha256()
    headers: dict[Path, tuple[dict[str, Any], int]] = {}
    count = total = 0
    for name, path in sorted(tensor_map(root).items()):
        if path not in headers:
            headers[path] = read_header(path)
        meta, base = headers[path]
        info = meta[name]
        begin, end = info["data_offsets"]
        digest.update(json.dumps([name, info["dtype"], info["shape"]]).encode() + b"\0")
        with open(path, "rb") as handle:
            handle.seek(base + begin)
            remaining = end - begin
            while remaining:
                block = handle.read(min(chunk, remaining))
                if not block:
                    raise ValueError(f"{path}: tensor {name!r} ends past the end of the file")
                digest.update(block)
                remaining -= len(block)
        digest.update(b"\0")
        count += 1
        total += end - begin
    return {"sha256": digest.hexdigest(), "tensors": count, "bytes": total}
