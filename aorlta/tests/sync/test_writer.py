"""C5.T2 and the slot writer (design Section 6.5, steps 2, 3 and 7).

C5.T2 uses Qwen3.5-4B's own `model.safetensors.index.json` and `config.json` (copied from the HF snapshot
851bf6e8 into tests/sync/fixtures) and the module names PEFT gives the real model class, built on the meta
device: no weights are loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aorlta.sync import writer as W
from aorlta.sync.merge import TARGET_MODULE_PATTERN, TARGET_KEY_RE
from aorlta.trainer.merged_lora import lora_layers

FIXTURES = Path(__file__).parent / "fixtures" / "qwen3_5_4b"


@pytest.fixture(scope="module")
def real_index_keys() -> list[str]:
    return list(json.loads((FIXTURES / "model.safetensors.index.json").read_text())["weight_map"])


@pytest.fixture(scope="module")
def real_module_names() -> list[str]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    cfg = AutoConfig.from_pretrained(str(FIXTURES))
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(cfg)
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                          target_modules=TARGET_MODULE_PATTERN, bias="none"))
    assert sum(p.numel() for p in pm.parameters() if p.requires_grad) == 32_464_896  # F1 / F13 count
    return list(lora_layers(pm))


# --- C5.T2 -----------------------------------------------------------------------------------------------


def test_name_map_produces_exactly_the_248_base_keys(real_module_names, real_index_keys):
    mapping = W.name_map(real_module_names, real_index_keys)
    targets = {k for k in real_index_keys if TARGET_KEY_RE.fullmatch(k)}
    assert len(real_module_names) == 248
    assert len(mapping) == 248 and len(targets) == 248
    assert set(mapping.values()) == targets
    assert all(v in set(real_index_keys) for v in mapping.values())
    assert mapping["base_model.model.model.language_model.layers.3.self_attn.q_proj"] == \
        "model.language_model.layers.3.self_attn.q_proj.weight"
    assert mapping["base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv"] == \
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert W.expected_count(real_index_keys, partial=True) == 248
    assert W.expected_count(real_index_keys, partial=False) == 738


@pytest.mark.parametrize("plant", ["renamed", "missing", "duplicate", "non_target", "wrong_layer"])
def test_a_planted_wrong_module_name_stops_the_run(real_module_names, real_index_keys, plant):
    names = list(real_module_names)
    i = names.index("base_model.model.model.language_model.layers.0.mlp.gate_proj")
    if plant == "renamed":
        names[i] = names[i] + "x"
    elif plant == "missing":
        del names[i]
    elif plant == "duplicate":
        names.append("other.prefix.model.language_model.layers.0.mlp.gate_proj")
    elif plant == "non_target":
        names.append("base_model.model.model.language_model.layers.0.linear_attn.conv1d")
    elif plant == "wrong_layer":
        names[i] = "base_model.model.model.language_model.layers.32.mlp.gate_proj"
    with pytest.raises(W.NameMapError):
        W.name_map(names, real_index_keys)


@pytest.mark.parametrize("plant", ["not_in_index", "missing", "non_target", "text_only_name"])
def test_check_names_refuses_anything_but_the_target_set(real_index_keys, plant):
    written = sorted(k for k in real_index_keys if TARGET_KEY_RE.fullmatch(k))
    if plant == "not_in_index":
        written[0] = written[0].replace(".weight", ".weights")
    elif plant == "missing":
        written = written[1:]
    elif plant == "non_target":
        written.append("model.language_model.layers.0.linear_attn.conv1d.weight")
    elif plant == "text_only_name":  # the text-only class names its modules model.layers.N...
        written[0] = written[0].replace("model.language_model.", "model.")
    with pytest.raises(W.NameMapError):
        W.check_names(written, real_index_keys, partial=True)


def test_check_names_accepts_the_exact_sets(real_index_keys):
    W.check_names([k for k in real_index_keys if TARGET_KEY_RE.fullmatch(k)], real_index_keys, partial=True)
    W.check_names(real_index_keys, real_index_keys, partial=False)
    with pytest.raises(W.NameMapError):
        W.check_names(real_index_keys, real_index_keys, partial=True)


# --- the writer on the tiny checkpoint ------------------------------------------------------------------


def _tiny_targets(tiny_base) -> dict[str, torch.Tensor]:
    keys = W.read_base_index(tiny_base)
    from safetensors import safe_open

    out = {}
    with safe_open(str(tiny_base / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if TARGET_KEY_RE.fullmatch(k):
                out[k] = f.get_tensor(k) + 0  # a copy
    assert len(out) == 31 and set(out) <= set(keys)
    return out


def test_write_slot_refuses_a_wrong_name_before_writing(tiny_base, tmp_path):
    weights = _tiny_targets(tiny_base)
    k = sorted(weights)[0]
    weights[k.replace(".weight", ".wieght")] = weights.pop(k)
    with pytest.raises(W.NameMapError):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=1)
    assert not (tmp_path / "r" / "w1").exists() and not (tmp_path / "r" / "w1.partial").exists()


def test_write_slot_refuses_a_missing_name(tiny_base, tmp_path):
    weights = _tiny_targets(tiny_base)
    weights.pop(sorted(weights)[3])
    with pytest.raises(W.NameMapError):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=1)
    assert not (tmp_path / "r").exists() or not any((tmp_path / "r").iterdir())


def test_partial_slot_layout_and_manifest(tiny_base, tmp_path):
    weights = _tiny_targets(tiny_base)
    info = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="run1", version=3)
    assert info.path == tmp_path / "run1" / "w1" and info.slot == 1 and info.tensors == 31 and info.partial
    assert not (tmp_path / "run1" / "w1.partial").exists()
    names = {p.name for p in info.path.iterdir()}
    assert {"model.safetensors.index.json", "manifest.json", "config.json", "tokenizer_config.json",
            "chat_template.jinja"} <= names
    index = json.loads((info.path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(weights)
    manifest = W.read_manifest(info.path)
    assert manifest["version"] == 3 and manifest["hash"] == "xxh64" and set(manifest["tensors"]) == set(weights)
    for name, t in weights.items():
        e = manifest["tensors"][name]
        assert e == {"shape": list(t.shape), "dtype": "BF16", "xxh64": W.tensor_xxh64(t)}
    back = W.load_slot_tensors(info.path)
    assert all(torch.equal(back[k].view(torch.int16), weights[k].view(torch.int16)) for k in weights)
    assert W.verify_slot(info.path)["ok"]


def test_verify_slot_notices_a_changed_byte(tiny_base, tmp_path):
    weights = _tiny_targets(tiny_base)
    info = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=0)
    m = json.loads((info.path / "manifest.json").read_text())
    name = sorted(m["tensors"])[0]
    m["tensors"][name]["xxh64"] = "0" * 16
    (info.path / "manifest.json").write_text(json.dumps(m))
    assert not W.verify_slot(info.path)["ok"]


def test_full_slot_holds_every_base_key(tiny_base, tmp_path):
    weights = _tiny_targets(tiny_base)
    keys = W.read_base_index(tiny_base)
    others = W.load_base_tensors(tiny_base, [k for k in keys if k not in weights])
    info = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=2, partial=False,
                        others=others)
    assert info.tensors == len(keys) == W.expected_count(keys, partial=False)
    with pytest.raises(W.SyncError):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=2, partial=False)


def test_shards_never_exceed_the_limit(tiny_base, tmp_path, monkeypatch):
    weights = _tiny_targets(tiny_base)
    monkeypatch.setattr(W, "SHARD_LIMIT", 40_000)
    info = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=1)
    index = json.loads((info.path / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    assert len(shards) > 1
    sizes = {}
    for name, fn in index.items():
        sizes[fn] = sizes.get(fn, 0) + weights[name].numel() * 2
    assert max(sizes.values()) <= 40_000
    assert all(fn.endswith(f"-of-{len(shards):05d}.safetensors") for fn in shards)
    assert W.verify_slot(info.path)["ok"]


def test_shm_full_stops_with_a_clear_error(tiny_base, tmp_path, monkeypatch):
    weights = _tiny_targets(tiny_base)
    monkeypatch.setattr(W, "free_bytes", lambda path: 10)
    with pytest.raises(W.ShmFullError, match="free"):
        W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=1)
    assert not (tmp_path / "r" / "w1.partial").exists()


def test_slots_alternate_and_prune_keeps_the_served_one(tiny_base, tmp_path):
    weights = _tiny_targets(tiny_base)
    a = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=4)
    b = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=5)
    assert (a.path.name, b.path.name) == ("w0", "w1")
    (tmp_path / "r" / "w0.partial").mkdir()
    removed = W.prune_slots(tmp_path, "r", keep=b.path)
    assert {p.name for p in removed} == {"w0", "w0.partial"}
    assert b.path.is_dir() and not a.path.exists()


def test_read_manifest_refuses_a_partial_slot(tmp_path):
    p = tmp_path / "w0.partial"
    p.mkdir()
    with pytest.raises(W.SyncError):
        W.read_manifest(p)
