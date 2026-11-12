#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base checkpoint.

For every adapter module, W' = bf16(W.float() + (alpha/r) * B.float() @ A.float()), in fp32 on the CPU, cast with
round-to-nearest-even. That is `aorlta.sync.merge.merge_all`, the formula the trainer and the weight sync use. The
modules go through WP3's name map (`aorlta.sync.writer.name_map`), so the merged set is exactly the base's LoRA target
keys (248 for Qwen3.5-4B). On top of it, an adapter module must be named `model.layers.N.<name>` (the text-only
model) or `model.language_model.layers.N.<name>` (the checkpoint), and nothing but lora_A/lora_B pairs may be in the
adapter: an embedding, head or other tensor is refused, as is any adapter_config field the plain formula does not
cover (DoRA, rsLoRA, rank or alpha patterns, modules_to_save, ...).

The output keeps the base's layout byte for byte. Every base file is copied, and only the target tensors' data ranges
inside the shards are rewritten, so the shard names, the safetensors headers, the tensor offsets and the index are
the base's. Each copy is hashed from the bytes it reads (`base.read_sha256`), and a target's copied bytes must be
the tensor the merge read. The result is then checked twice: byte ranges against the base (everything but the
targets identical, each target equal to the merge), and an independent safetensors read. `merge-manifest.json`
records the inputs' hashes, the recipe, the timings, the checks and the sha256 of every output file and tensor. With
--sensitivity it
also counts the bf16 elements that a different fp32 summation order could flip, and checks that the CPU merge does
not depend on the thread count.

  merge_base_adapter.py merge  --base DIR --adapter DIR --out DIR [--expect-scale 2.0] [--expect-modules 248]
                               [--cross-check-adapter DIR] [--base-pins FILE] [--sensitivity] [--threads N]
  merge_base_adapter.py verify --base DIR --adapter DIR --merged DIR [--expect-scale 2.0] [--expect-modules 248]

Exit codes: 0 done and every check passed; 1 a check failed; 2 the inputs were refused before anything was written.
Only configs, shapes, counts, hashes and timings are printed or recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import platform
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aorlta.atomic import atomic_write_json  # noqa: E402
from aorlta.ids import sha256_json  # noqa: E402
from aorlta.sync.merge import bits, merge_all  # noqa: E402
from aorlta.sync.writer import INDEX_NAME, SyncError, name_map, read_base_index, target_keys, tensor_bytes  # noqa: E402

SCHEMA = "aorlta.wp9.merge-manifest/1"
MANIFEST_NAME = "merge-manifest.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
PEFT_PREFIX = "base_model.model."
LORA_KEY_RE = re.compile(r"(?P<module>.+)\.lora_(?P<ab>[AB])\.weight")
MODULE_RE = re.compile(r"model\.(?P<lm>language_model\.)?(?P<tail>layers\.\d+\.[A-Za-z0-9_.]+)")
FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
HASH_CHUNK = 1 << 24

# adapter_config.json fields that change what the adapter computes, and the values the plain formula covers.
NEUTRAL = {
    "peft_type": ("LORA",),
    "bias": ("none",),
    "lora_bias": (False,),
    "use_dora": (False,),
    "use_rslora": (False,),
    "use_qalora": (False,),
    "use_bdlora": (None, False),
    "fan_in_fan_out": (False,),
    "modules_to_save": (None, []),
    "rank_pattern": (None, {}),
    "alpha_pattern": (None, {}),
    "layers_to_transform": (None, []),
    "layer_replication": (None, []),
    "trainable_token_indices": (None, [], {}),
    "target_parameters": (None, []),
    "alora_invocation_tokens": (None, []),
    "ensure_weight_tying": (None, False),
    "init_lora_weights": (True, False, "gaussian"),
}
# Fields that do not change the merge (bookkeeping, or which modules exist, which the tensors decide).
BOOKKEEPING = {"base_model_name_or_path", "revision", "peft_version", "task_type", "inference_mode", "target_modules",
               "r", "lora_alpha", "lora_dropout", "auto_mapping", "layers_pattern", "exclude_modules",
               "megatron_core", "qalora_group_size"}


class Refused(ValueError):
    """The inputs do not fit the plain merge formula; nothing was written."""


class CheckFailed(RuntimeError):
    """The written output does not match what the merge should have produced."""


# --- inputs --------------------------------------------------------------------------------------------------


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def check_adapter_config(cfg: dict) -> dict:
    """Refuse any field outside what W + (alpha/r) B A covers; return the merge-relevant fields."""
    problems = []
    for key, allowed in NEUTRAL.items():
        if key in cfg and not any(cfg[key] == a and type(cfg[key]) is type(a) for a in allowed):
            problems.append(f"{key}={cfg[key]!r}")
    for key, value in cfg.items():
        if key in NEUTRAL or key in BOOKKEEPING:
            continue
        if key.endswith("_config"):
            if value not in (None, {}):
                problems.append(f"{key}={str(value)[:80]}")
        elif value not in (None, False, {}, [], ""):
            problems.append(f"unknown field {key}={str(value)[:80]}")
    if problems:
        raise Refused("adapter_config.json has fields the plain merge does not cover: " + ", ".join(problems))
    r, alpha = cfg.get("r"), cfg.get("lora_alpha")
    if not isinstance(r, int) or isinstance(r, bool) or r <= 0:
        raise Refused(f"adapter_config.json r must be a positive int, got {r!r}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not alpha > 0:
        raise Refused(f"adapter_config.json lora_alpha must be > 0, got {alpha!r}")
    return {k: cfg.get(k) for k in ("peft_type", "r", "lora_alpha", "lora_dropout", "bias", "use_rslora", "use_dora",
                                    "target_modules", "modules_to_save", "base_model_name_or_path", "revision",
                                    "peft_version", "task_type")}


@dataclass(frozen=True)
class Adapter:
    path: Path
    config: dict
    rank: int
    alpha: float
    scale: float
    modules: dict[str, tuple[torch.Tensor, torch.Tensor]]  # module name without the PEFT prefix -> (A, B)
    naming: str  # "model.layers.N" or "model.language_model.layers.N"
    dtypes: tuple[str, ...]
    tensors: int
    weights_sha256: str
    config_sha256: str


def load_adapter(path: str | Path) -> Adapter:
    """Each file is read once, and its hash and its contents come from the same bytes: a file rewritten while the
    tool runs cannot be hashed as one adapter and merged as another."""
    from safetensors import SafetensorError
    from safetensors.torch import load as load_safetensors

    path = Path(path)
    cfg_path, w_path = path / ADAPTER_CONFIG, path / ADAPTER_WEIGHTS
    if not cfg_path.is_file() or not w_path.is_file():
        raise Refused(f"{path}: needs {ADAPTER_CONFIG} and {ADAPTER_WEIGHTS}")
    try:
        cfg_bytes = cfg_path.read_bytes()
        cfg = json.loads(cfg_bytes)
    except (OSError, ValueError) as exc:  # unreadable, or not JSON
        raise Refused(_unreadable(str(cfg_path), exc)) from None
    if not isinstance(cfg, dict):
        raise Refused(f"{cfg_path}: not a JSON object")
    fields = check_adapter_config(cfg)
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    targets = cfg.get("target_modules")
    listed = [t for t in targets if isinstance(t, str)] if isinstance(targets, list) else None
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    namings = set()
    try:
        w_bytes = w_path.read_bytes()
        weights = load_safetensors(w_bytes)
    except (OSError, SafetensorError) as exc:  # unreadable, or not a safetensors file
        raise Refused(_unreadable(str(w_path), exc)) from None
    keys = sorted(weights)
    for key in keys:
        m = LORA_KEY_RE.fullmatch(key)
        if m is None:
            raise Refused(f"{key}: not a lora_A/lora_B weight (an embedding, head or bias change is not merged "
                          f"by this formula)")
        module = m.group("module")
        if not module.startswith(PEFT_PREFIX):
            raise Refused(f"{key}: does not start with {PEFT_PREFIX!r}")
        module = module[len(PEFT_PREFIX):]
        mm = MODULE_RE.fullmatch(module)
        if mm is None:
            raise Refused(f"{key}: not a text decoder layer (model.layers.N.* or model.language_model.layers.N.*)")
        namings.add("model.language_model.layers.N" if mm.group("lm") else "model.layers.N")
        # PEFT's rule for a list: the module's full name is an entry, or ends with "." + an entry
        if listed is not None and not any(module == t or module.endswith("." + t) for t in listed):
            raise Refused(f"{key}: module {module!r} is not in target_modules")
        t = weights[key]
        if t.dtype not in FLOAT_DTYPES or t.dim() != 2:
            raise Refused(f"{key}: expected a 2-D float tensor, got {t.dtype} {tuple(t.shape)}")
        slot = pairs.setdefault(module, {})
        if m.group("ab") in slot:
            raise Refused(f"{key}: lora_{m.group('ab')} appears twice for {module}")
        slot[m.group("ab")] = t
    if not keys:
        raise Refused(f"{w_path}: no tensors")
    if len(namings) != 1:
        raise Refused(f"the adapter mixes module namings {sorted(namings)}")
    modules = {}
    for module, slot in sorted(pairs.items()):
        if set(slot) != {"A", "B"}:
            raise Refused(f"{module}: has only lora_{''.join(sorted(slot))}")
        a, b = slot["A"], slot["B"]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise Refused(f"{module}: A {tuple(a.shape)} and B {tuple(b.shape)} are not rank r={rank}")
        modules[module] = (a, b)
    return Adapter(path=path, config=fields, rank=rank, alpha=float(alpha), scale=alpha / rank, modules=modules,
                   naming=namings.pop(), dtypes=tuple(sorted({str(t.dtype) for ab in modules.values() for t in ab})),
                   tensors=len(keys), weights_sha256=hashlib.sha256(w_bytes).hexdigest(),
                   config_sha256=hashlib.sha256(cfg_bytes).hexdigest())


def checkpoint_keys(adapter: Adapter, base_keys: list[str]) -> dict[str, str]:
    """module -> base checkpoint key, through WP3's name map (exactly the base's target keys, one to one), and the
    stricter rule that the key is `model.language_model.` + the module's `layers.N.<name>` + `.weight`."""
    try:
        mapping = name_map(adapter.modules, base_keys)
    except SyncError as exc:
        raise Refused(f"name map: {exc}") from exc
    for module, key in mapping.items():
        tail = MODULE_RE.fullmatch(module).group("tail")
        if key != f"model.language_model.{tail}.weight":
            raise Refused(f"{module} mapped to {key}, not model.language_model.{tail}.weight")
    return mapping


def factors_for(adapter: Adapter, base_keys: list[str]) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Base checkpoint key -> (A, B, alpha/r), the input `merge_all` takes."""
    mapping = checkpoint_keys(adapter, base_keys)
    return {mapping[m]: (a, b, adapter.scale) for m, (a, b) in adapter.modules.items()}


def cross_check(primary: Adapter, other: Adapter, base_keys: list[str]) -> dict:
    """Two adapter directories hold the same LoRA: the same config, the same checkpoint keys, the same A and B bits."""
    fa, fb = factors_for(primary, base_keys), factors_for(other, base_keys)
    same_keys = set(fa) == set(fb)
    differing = [k for k in sorted(set(fa) & set(fb))
                 if not (_same_bits(fa[k][0], fb[k][0]) and _same_bits(fa[k][1], fb[k][1]))]
    out = {"dir": str(other.path), "naming": other.naming, "weights_sha256": other.weights_sha256,
           "config_identical": json.loads((primary.path / ADAPTER_CONFIG).read_text())
           == json.loads((other.path / ADAPTER_CONFIG).read_text()),
           "same_checkpoint_keys": same_keys, "modules": len(fb), "modules_differing": len(differing)}
    out["pass"] = out["config_identical"] and same_keys and not differing
    return out


def base_revision(base: Path) -> dict:
    out = {"snapshot": base.name if base.parent.name == "snapshots" else None, "refs_main": None}
    ref = base.parent.parent / "refs" / "main"
    if base.parent.name == "snapshots" and ref.is_file():
        out["refs_main"] = ref.read_text().strip()
    return out


def check_pins(base: Path, pins_file: Path, files: dict[str, dict], revision: dict) -> dict:
    """The base files against a pin list ({"base_model": {"revision", "files": [{"name", "bytes", "sha256"}]}})."""
    try:
        pins = json.loads(pins_file.read_text())["base_model"]
        rows = [(row["name"], row["bytes"], row["sha256"]) for row in pins["files"]]
        pins.get("revision")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:  # unreadable, or not a pin list
        raise Refused(_unreadable(str(pins_file), exc)) from None
    bad = []
    for name, size, sha in rows:
        got = files.get(name)
        if got is None:
            bad.append(f"{name}: missing")
        elif got["bytes"] != size or got["sha256"] != sha:
            bad.append(f"{name}: differs")
    if revision["snapshot"] is not None and pins.get("revision") != revision["snapshot"]:
        bad.append(f"revision {pins.get('revision')} != snapshot {revision['snapshot']}")
    return {"file": str(pins_file), "sha256": sha256_path(pins_file), "revision": pins.get("revision"),
            "files": len(pins["files"]), "problems": bad, "pass": not bad}


# --- safetensors layout ---------------------------------------------------------------------------------------


def read_header(path: Path) -> tuple[int, dict[str, dict]]:
    """(header length n, {key: dtype, shape, begin, end}), offsets relative to the data start at 8 + n."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise Refused(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if not 0 < n < 1 << 28:
            raise Refused(f"{path}: implausible header length {n}")
        raw = f.read(n)
    try:
        header = json.loads(raw)
        header.pop("__metadata__", None)
        entries = {k: {"dtype": v["dtype"], "shape": list(v["shape"]), "begin": int(v["data_offsets"][0]),
                       "end": int(v["data_offsets"][1])} for k, v in header.items()}
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:  # not a safetensors header
        raise Refused(_unreadable(f"{path}'s header", exc)) from None
    return n, entries


def base_index(base: Path) -> dict[str, str]:
    """The base's weight map (WP3's reader, which refuses a shard name that is not a file of the directory)."""
    try:
        return read_base_index(base)
    except SyncError as exc:
        raise Refused(f"base index: {exc}") from exc


def shard_files(base: Path) -> dict[str, list[str]]:
    wm = base_index(base)
    out: dict[str, list[str]] = {}
    for key, fn in wm.items():
        out.setdefault(fn, []).append(key)
    for fn, keys in out.items():
        _, entries = read_header(base / fn)
        if set(entries) != set(keys):
            raise Refused(f"{fn}: the index names {len(keys)} tensors, the header {len(entries)}")
    return out


def base_files(base: Path) -> list[str]:
    names = sorted(p.name for p in base.iterdir() if p.is_file())
    if MANIFEST_NAME in names:
        raise Refused(f"{base} already has a {MANIFEST_NAME}: it is a merge output, not a base")
    return names


# --- the merge ------------------------------------------------------------------------------------------------


def rne_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """IEEE round-to-nearest-even of finite fp32 values to bf16, by integer arithmetic, as uint16 values in int32."""
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).to(torch.int32) & 0xFFFF


def as_u16(t: torch.Tensor) -> torch.Tensor:
    return bits(t).to(torch.int32) & 0xFFFF


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def fp32_sum(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """The fp32 value before the cast, computed exactly as `aorlta.sync.merge._merge_one` does."""
    return w.float() + scale * (b.float() @ a.float())


def check_rounding(base: dict, factors: dict, merged: dict) -> dict:
    """Every merged element is the round-to-nearest-even of the fp32 sum, checked by integer arithmetic, and every
    sum and every merged element is finite (a finite sum above bf16's largest value rounds to infinity)."""
    bad, nonfinite, elements = [], 0, 0
    for key in sorted(merged):
        a, b, scale = factors[key]
        s = fp32_sum(base[key], a, b, scale)
        nonfinite += int((~(torch.isfinite(s) & torch.isfinite(merged[key]))).sum())
        got = as_u16(merged[key])
        if not (torch.equal(as_u16(s.to(torch.bfloat16)), got) and torch.equal(rne_bf16_bits(s), got)):
            bad.append(key)
        elements += s.numel()
    return {"tensors": len(merged), "elements": elements, "not_rne": len(bad), "not_rne_keys": bad[:5],
            "nonfinite": nonfinite, "pass": not bad and nonfinite == 0}


def bf16_order(t: torch.Tensor) -> torch.Tensor:
    """bf16 values as integers in number-line order: neighbouring values are one apart, and -0 and +0 are both 0."""
    u = as_u16(t)
    return torch.where(u >= 0x8000, -(u & 0x7FFF), u)


def compare_bf16(x: torch.Tensor, y: torch.Tensor) -> dict:
    """Elements where two bf16 tensors differ: neighbours on the bf16 number line (a rounding flip), or further apart,
    with the largest absolute difference and, for the further-apart ones, the largest magnitude involved."""
    ox, oy = bf16_order(x), bf16_order(y)
    d = ox != oy
    n = int(d.sum())
    if not n:
        return {"differing": 0, "neighbours": 0, "further": 0, "max_abs_difference": 0.0,
                "max_abs_value_where_further": 0.0}
    further = (ox[d] - oy[d]).abs() > 1
    xd, yd = x[d].float(), y[d].float()
    big = torch.maximum(xd.abs(), yd.abs())[further]
    return {"differing": n, "neighbours": int((~further).sum()), "further": int(further.sum()),
            "max_abs_difference": float((xd - yd).abs().max()),
            "max_abs_value_where_further": float(big.max()) if big.numel() else 0.0}


def add_comparison(total: dict, part: dict) -> dict:
    out = dict(total)
    for k in ("differing", "neighbours", "further"):
        out[k] += part[k]
    for k in ("max_abs_difference", "max_abs_value_where_further"):
        out[k] = max(out[k], part[k])
    return out


def sensitivity(base: dict, factors: dict, merged: dict, threads: int) -> dict:
    """How many bf16 elements depend on the fp32 summation order: the recipe against B @ A accumulated in fp64 and
    rounded once to fp32. A differing element is a rounding flip to the neighbouring bf16 value, or, where W and the
    update cancel to almost zero, further apart in steps but tiny in value. And whether one CPU thread gives the same
    bits as `threads` (one tensor per group)."""
    from aorlta.sync.merge import group_of

    empty = torch.zeros(0, dtype=torch.bfloat16)
    total = compare_bf16(empty, empty)
    tensors = 0
    per_group: dict[str, int] = {}
    for key in sorted(merged):
        a, b, scale = factors[key]
        alt = (base[key].double() + scale * (b.double() @ a.double())).float().to(torch.bfloat16)
        c = compare_bf16(alt, merged[key])
        if c["differing"]:
            tensors += 1
            per_group[group_of(key)] = per_group.get(group_of(key), 0) + c["differing"]
        total = add_comparison(total, c)
    sample = {}
    for key in sorted(merged):
        sample.setdefault(group_of(key), key)
    torch.set_num_threads(1)
    try:
        single = [k for k in sample.values()
                  if not torch.equal(as_u16(fp32_sum(base[k], *factors[k]).to(torch.bfloat16)), as_u16(merged[k]))]
    finally:
        torch.set_num_threads(threads)
    return {"vs_fp64_accumulation": {"elements_differing": total["differing"], "tensors_differing": tensors,
                                     **{k: v for k, v in total.items() if k != "differing"},
                                     "by_group": dict(sorted(per_group.items()))},
            "one_thread_vs_threads": {"threads": threads, "tensors_checked": len(sample),
                                      "tensors_differing": len(single), "pass": not single}}


@dataclass
class Merged:
    base_keys: list[str]
    targets: list[str]
    weights: dict[str, torch.Tensor]
    base_targets: dict[str, torch.Tensor]
    factors: dict[str, tuple[torch.Tensor, torch.Tensor, float]]
    stats: dict
    seconds: dict = field(default_factory=dict)


def merge_in_memory(base: Path, adapter: Adapter, *, expect_scale: float | None, expect_modules: int | None) -> Merged:
    from safetensors import SafetensorError

    from aorlta.sync.writer import load_base_tensors

    if expect_scale is not None and adapter.scale != expect_scale:
        raise Refused(f"lora_alpha / r = {adapter.alpha:g} / {adapter.rank} = {adapter.scale}, expected {expect_scale}")
    base_keys = list(base_index(base))
    factors = factors_for(adapter, base_keys)
    targets = sorted(target_keys(base_keys))
    if expect_modules is not None and len(factors) != expect_modules:
        raise Refused(f"{len(factors)} modules, expected {expect_modules}")
    t = time.monotonic()
    try:
        base_t = load_base_tensors(base, targets)
    except (OSError, SafetensorError, SyncError) as exc:  # a shard that is missing or not a safetensors file
        raise Refused(_unreadable("the base shards", exc)) from None
    load_s = time.monotonic() - t
    try:
        res = merge_all(base_t, factors, base_t, version=0)
    except ValueError as exc:  # MergeError: dtype or shape
        raise Refused(f"merge: {exc}") from exc
    return Merged(base_keys, targets, res.weights, base_t, factors, res.stats.as_dict(),
                  {"load_base_targets": load_s, "merge": res.stats.seconds})


def copy_hashed(src: Path, dst: Path) -> str:
    """Copy `src` to a new `dst`; the sha256 of the bytes copied, from the same read."""
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "xb") as fo:
        for block in iter(lambda: fi.read(HASH_CHUNK), b""):
            h.update(block)
            fo.write(block)
    return h.hexdigest()


def write_output(base: Path, out: Path, merged: Merged) -> dict[str, str]:
    """Copy every base file into `out` (which must not exist), then rewrite the target ranges in the shards. Returns
    the sha256 of the bytes each copy read."""
    shards = shard_files(base)
    names = base_files(base)
    out.mkdir()
    read = {name: copy_hashed(base / name, out / name) for name in names}
    for fn, keys in sorted(shards.items()):
        n, entries = read_header(base / fn)
        with open(out / fn, "r+b") as f:
            for key in sorted((k for k in keys if k in merged.weights), key=lambda k: entries[k]["begin"]):
                e, t = entries[key], merged.weights[key]
                data = tensor_bytes(t)
                if e["dtype"] != "BF16" or e["shape"] != list(t.shape) or len(data) != e["end"] - e["begin"]:
                    raise CheckFailed(f"{fn}:{key}: header {e} does not fit the merged {t.dtype} {tuple(t.shape)}")
                # The merge read the targets before the copy: what it merged must be what was copied and hashed.
                f.seek(8 + n + e["begin"])
                if f.read(len(data)) != tensor_bytes(merged.base_targets[key]):
                    raise CheckFailed(f"{fn}:{key}: the copy is not the tensor the merge read; "
                                      "the base changed while it was merged")
                f.seek(8 + n + e["begin"])
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
    return read


# --- the checks -----------------------------------------------------------------------------------------------


def _ranges(n: int, entries: dict[str, dict], size: int) -> list[tuple[str | None, int, int]]:
    """Every byte range of the file in order: (None, ...) for the header and any gap, (key, ...) for a tensor."""
    out, pos = [(None, 0, 8 + n)], 8 + n
    for key, e in sorted(entries.items(), key=lambda kv: kv[1]["begin"]):
        a, z = 8 + n + e["begin"], 8 + n + e["end"]
        if a < pos:
            raise CheckFailed(f"tensor {key} overlaps the previous range")
        if a > pos:
            out.append((None, pos, a))
        out.append((key, a, z))
        pos = z
    if pos < size:
        out.append((None, pos, size))
    elif pos > size:
        raise CheckFailed(f"tensors end at {pos}, past the file size {size}")
    return out


def _compare_range(mo: mmap.mmap, mb: mmap.mmap, a: int, z: int, fh, expected: torch.Tensor | None):
    """(output equals the base, output equals `expected` or None, sha256 of the range, changed uint16 elements).
    The views into the maps die with this frame, so the maps can be closed afterwards."""
    vo, vb = memoryview(mo)[a:z], memoryview(mb)[a:z]
    for i in range(0, z - a, HASH_CHUNK):
        fh.update(vo[i:i + HASH_CHUNK])
    ob, bb = np.frombuffer(vo, np.uint8), np.frombuffer(vb, np.uint8)
    same_as_base = bool(np.array_equal(ob, bb))
    if expected is None:
        return same_as_base, None, hashlib.sha256(vo).hexdigest(), None
    matches = bool(np.array_equal(ob, np.frombuffer(tensor_bytes(expected), np.uint8)))
    changed = int(np.count_nonzero(ob.view(np.uint16) != bb.view(np.uint16)))
    return same_as_base, matches, hashlib.sha256(vo).hexdigest(), changed


def compare_bytes(base: Path, out: Path, merged: Merged) -> dict:
    """Byte ranges against the base: headers, gaps and non-target tensors identical, each target equal to the merge.
    Also hashes every output file and tensor in the same pass."""
    shards = shard_files(base)
    targets = set(merged.targets)
    files, tensors, problems = {}, {}, []
    changed_targets = 0
    for name in base_files(base):
        if not (out / name).is_file():
            problems.append(f"{name}: missing from the output")
            continue
        if name not in shards:
            hb, ho = sha256_path(base / name), sha256_path(out / name)
            files[name] = {"bytes": (out / name).stat().st_size, "sha256": ho}
            if hb != ho:
                problems.append(f"{name}: differs from the base")
            continue
        n, entries = read_header(base / name)
        size = (base / name).stat().st_size
        if (out / name).stat().st_size != size:
            problems.append(f"{name}: size differs from the base")
            continue
        fh = hashlib.sha256()
        with open(base / name, "rb") as fb, open(out / name, "rb") as fo:
            mb = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)
            mo = mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                for key, a, z in _ranges(n, entries, size):
                    is_target = key is not None and key in targets
                    same, matches, digest, changed = _compare_range(
                        mo, mb, a, z, fh, merged.weights[key] if is_target else None)
                    if key is None:
                        if not same:
                            problems.append(f"{name}: header or gap bytes {a}-{z} differ from the base")
                        continue
                    e = entries[key]
                    row = {"file": name, "dtype": e["dtype"], "shape": e["shape"], "target": is_target,
                           "sha256": digest}
                    if is_target:
                        if not matches:
                            problems.append(f"{key}: the written bytes are not the merge")
                        row["changed_elements"] = changed
                        changed_targets += changed > 0
                    elif not same:
                        problems.append(f"{key}: a non-target tensor differs from the base")
                    tensors[key] = row
            finally:
                mo.close()
                mb.close()
        files[name] = {"bytes": size, "sha256": fh.hexdigest()}
    missing = sorted(set(merged.base_keys) - set(tensors))
    if missing:
        problems.append(f"{len(missing)} base tensors not found in the output, e.g. {missing[:3]}")
    return {"files": files, "tensors": tensors, "targets_changed": changed_targets, "targets": len(targets),
            "non_targets": len(tensors) - len(targets), "problems": problems[:20], "pass": not problems}


def reread(base: Path, out: Path, merged: Merged) -> dict:
    """An independent read with safetensors: the output's keys, dtypes and shapes are the base's; each target is the
    merge bit for bit, and every other tensor is the base's bit for bit."""
    from safetensors import safe_open

    shards = shard_files(base)
    targets = set(merged.targets)
    problems, seen, unchanged = [], 0, 0
    for fn in sorted(shards):
        try:
            fo_ctx = safe_open(str(out / fn), framework="pt")
        except Exception as exc:  # noqa: BLE001 - an unreadable output is a failed check, not a crash
            problems.append(f"{fn}: safetensors cannot read the output: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        with safe_open(str(base / fn), framework="pt") as fb, fo_ctx as fo:
            if set(fb.keys()) != set(fo.keys()):
                problems.append(f"{fn}: the output's keys differ from the base's")
                continue
            for key in fb.keys():
                tb, to = fb.get_tensor(key), fo.get_tensor(key)
                seen += 1
                if tb.dtype != to.dtype or tb.shape != to.shape:
                    problems.append(f"{key}: {to.dtype} {tuple(to.shape)} != base {tb.dtype} {tuple(tb.shape)}")
                elif key in targets:
                    if not _same_bits(to, merged.weights[key]):
                        problems.append(f"{key}: not the merge")
                elif _same_bits(to, tb):
                    unchanged += 1
                else:
                    problems.append(f"{key}: a non-target tensor changed")
    if seen != len(merged.base_keys):
        problems.append(f"read {seen} tensors, the base index has {len(merged.base_keys)}")
    return {"tensors_read": seen, "non_targets_bitwise_unchanged": unchanged, "problems": problems[:20],
            "pass": not problems}


def _unreadable(name: str, exc: Exception) -> str:
    return f"{name}: cannot be read: {type(exc).__name__}: {str(exc)[:120]}"


def check_index(base: Path, out: Path) -> dict:
    if not (base / INDEX_NAME).is_file():  # a base saved as one model.safetensors has no index, and nor has its merge
        same = not (out / INDEX_NAME).exists()
        return {"weight_map_identical": same, "pass": same}
    want = json.loads((base / INDEX_NAME).read_text())
    try:
        same = json.loads((out / INDEX_NAME).read_text()) == want
    except (OSError, ValueError) as exc:  # missing, or not JSON
        return {"weight_map_identical": False, "problem": _unreadable(INDEX_NAME, exc), "pass": False}
    return {"weight_map_identical": same, "pass": same}


def run_checks(base: Path, out: Path, merged: Merged) -> dict:
    t = time.monotonic()
    names = {p.name for p in out.iterdir() if p.is_file()} - {MANIFEST_NAME} if out.is_dir() else set()
    file_set = {"pass": names == set(base_files(base)), "extra": sorted(names - set(base_files(base))),
                "missing": sorted(set(base_files(base)) - names)}
    checks = {"file_set": file_set, "index": check_index(base, out)}
    checks["bytes"] = compare_bytes(base, out, merged)
    checks["bytes_seconds"] = time.monotonic() - t
    t = time.monotonic()
    checks["reread"] = reread(base, out, merged)
    checks["reread_seconds"] = time.monotonic() - t
    checks["pass"] = all(checks[k]["pass"] for k in ("file_set", "index", "bytes", "reread"))
    return checks


# --- the commands ---------------------------------------------------------------------------------------------


def _git_state() -> dict:
    for git in ("/usr/bin/git", "git"):
        try:
            head = subprocess.run([git, "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
            dirty = subprocess.run([git, "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, check=True).stdout.strip()
            return {"commit": head, "tracked_files_modified": len(dirty.splitlines()) if dirty else 0}
        except (OSError, subprocess.CalledProcessError):
            continue
    return {"commit": None, "tracked_files_modified": None}


def _versions() -> dict:
    import safetensors

    return {"python": platform.python_version(), "torch": torch.__version__, "safetensors": safetensors.__version__,
            "numpy": np.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "host": platform.node()}


def _adapter_record(adapter: Adapter) -> dict:
    return {"dir": str(adapter.path), "config": adapter.config, "config_sha256": adapter.config_sha256,
            "weights_sha256": adapter.weights_sha256, "tensors": adapter.tensors, "modules": len(adapter.modules),
            "rank": adapter.rank, "lora_alpha": adapter.alpha, "scale": adapter.scale, "naming": adapter.naming,
            "dtypes": list(adapter.dtypes)}


def _print_summary(manifest: dict) -> None:
    ch = manifest["checks"]
    print(json.dumps({
        "pass": manifest["pass"],
        "targets": ch["bytes"]["targets"], "targets_changed": ch["bytes"]["targets_changed"],
        "non_targets": ch["bytes"]["non_targets"],
        "non_targets_bitwise_unchanged": ch["reread"]["non_targets_bitwise_unchanged"],
        "rounding": {k: ch["rounding"][k] for k in ("tensors", "not_rne", "nonfinite", "pass")},
        "files": manifest["output"]["files"], "tensor_digest": manifest["output"]["tensor_digest"],
        "seconds": {k: round(v, 2) for k, v in manifest["seconds"].items()},
        "problems": ch["bytes"]["problems"][:3] + ch["reread"]["problems"][:3],
    }, indent=1))


def cmd_merge(args) -> int:
    t0 = time.monotonic()
    torch.set_num_threads(args.threads)
    base, out = Path(args.base).resolve(), Path(args.out).resolve()
    if out.exists():
        raise Refused(f"{out} exists; the tool never overwrites a merge")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise Refused(f"{partial} exists (an interrupted run); remove it first")
    seconds: dict[str, float] = {}
    t = time.monotonic()
    names = base_files(base)
    files = {n: {"bytes": (base / n).stat().st_size, "sha256": sha256_path(base / n)} for n in names}
    seconds["hash_base"] = time.monotonic() - t
    revision = base_revision(base)
    pins = check_pins(base, Path(args.base_pins), files, revision) if args.base_pins else None
    if pins is not None and not pins["pass"]:
        raise Refused(f"the base does not match {args.base_pins}: {pins['problems'][:3]}")
    adapter = load_adapter(args.adapter)
    other = load_adapter(args.cross_check_adapter) if args.cross_check_adapter else None
    merged = merge_in_memory(base, adapter, expect_scale=args.expect_scale, expect_modules=args.expect_modules)
    seconds.update(merged.seconds)
    xcheck = cross_check(adapter, other, merged.base_keys) if other is not None else None
    if xcheck is not None and not xcheck["pass"]:
        raise Refused(f"the cross-check adapter differs: {xcheck}")
    t = time.monotonic()
    rounding = check_rounding(merged.base_targets, merged.factors, merged.weights)
    seconds["check_rounding"] = time.monotonic() - t
    if rounding["nonfinite"]:
        raise Refused(f"the merge holds {rounding['nonfinite']} non-finite values; nothing was written")
    sens = None
    if args.sensitivity:
        t = time.monotonic()
        sens = sensitivity(merged.base_targets, merged.factors, merged.weights, args.threads)
        seconds["sensitivity"] = time.monotonic() - t
    t = time.monotonic()
    read = write_output(base, partial, merged)
    seconds["write"] = time.monotonic() - t
    checks = run_checks(base, partial, merged)
    seconds["check_bytes"], seconds["check_reread"] = checks.pop("bytes_seconds"), checks.pop("reread_seconds")
    checks["rounding"] = rounding
    ok = checks["pass"] and rounding["pass"] and (sens is None or sens["one_thread_vs_threads"]["pass"])
    tensors = checks["bytes"].pop("tensors")
    out_files = checks["bytes"].pop("files")
    seconds["total"] = time.monotonic() - t0
    manifest = {
        "schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pass": ok,
        "tool": {"path": "tools/merge_base_adapter.py", **_git_state(), "argv": sys.argv[1:]},
        "versions": _versions(), "torch_threads": args.threads,
        "base": {"dir": str(base), "revision": revision, "files": files, "read_sha256": read,
                 "keys": len(merged.base_keys), "targets": len(merged.targets), "pins": pins},
        "adapter": _adapter_record(adapter), "cross_check": xcheck,
        "recipe": {"formula": "W' = bf16(W.float() + (lora_alpha / r) * (B.float() @ A.float()))",
                   "scale": adapter.scale, "device": "cpu", "accumulation": "float32",
                   "cast": "torch .to(torch.bfloat16), checked against an integer round-to-nearest-even",
                   "layout": "every base file copied; only the target tensors' data ranges rewritten"},
        "merge_stats": merged.stats, "checks": checks, "sensitivity": sens, "seconds": seconds,
        "output": {"dir": str(out), "files": out_files, "tensors": tensors,
                   "tensor_digest": sha256_json({k: v["sha256"] for k, v in tensors.items()}),
                   "target_digest": sha256_json({k: v["sha256"] for k, v in tensors.items() if v["target"]})},
    }
    atomic_write_json(partial / MANIFEST_NAME, manifest)
    if ok:
        os.rename(partial, out)
        manifest["output"]["dir"] = str(out)
    else:
        print(f"CHECK FAILED; the output stays at {partial} for inspection", file=sys.stderr)
    _print_summary(manifest)
    return 0 if ok else 1


def cmd_verify(args) -> int:
    torch.set_num_threads(args.threads)
    base, out = Path(args.base).resolve(), Path(args.merged).resolve()
    adapter = load_adapter(args.adapter)
    merged = merge_in_memory(base, adapter, expect_scale=args.expect_scale, expect_modules=args.expect_modules)
    checks = run_checks(base, out, merged)
    checks["rounding"] = check_rounding(merged.base_targets, merged.factors, merged.weights)
    problems = [checks["index"]["problem"]] if "problem" in checks["index"] else []
    try:
        recorded = json.loads((out / MANIFEST_NAME).read_text())["output"]["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:  # missing, not JSON, or not a merge manifest
        recorded = None
        problems.append(_unreadable(MANIFEST_NAME, exc))
    files = checks["bytes"]["files"]
    same_as_manifest = recorded is not None and recorded == files
    ok = checks["pass"] and checks["rounding"]["pass"] and same_as_manifest
    print(json.dumps({"pass": ok, "checks_pass": checks["pass"], "rounding_pass": checks["rounding"]["pass"],
                      "files_match_manifest": same_as_manifest, "files": files,
                      "targets_changed": checks["bytes"]["targets_changed"],
                      "problems": problems + checks["bytes"]["problems"][:3] + checks["reread"]["problems"][:3]},
                     indent=1))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("merge", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--base", required=True, help="base checkpoint directory: model.safetensors.index.json "
                       "and its shards, or one model.safetensors")
        p.add_argument("--adapter", required=True, help="PEFT adapter directory")
        p.add_argument("--expect-scale", type=float, default=None, help="refuse unless lora_alpha / r is this")
        p.add_argument("--expect-modules", type=int, default=None, help="refuse unless the adapter has this many")
        p.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
        if name == "merge":
            p.add_argument("--out", required=True, help="output directory (must not exist)")
            p.add_argument("--cross-check-adapter", help="a second adapter directory that must hold the same LoRA")
            p.add_argument("--base-pins", help="JSON: base_model.revision and base_model.files[name, bytes, sha256]")
            p.add_argument("--sensitivity", action="store_true", help="count elements a summation order could flip")
        else:
            p.add_argument("--merged", required=True, help="a merge output directory")
    args = ap.parse_args(argv)
    try:
        return cmd_merge(args) if args.cmd == "merge" else cmd_verify(args)
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
