"""tools/merge_base_adapter.py on a tiny Qwen3.5 checkpoint and a tiny synthetic r4/alpha8 adapter (alpha/r = 2, as
in an r16/alpha32 adapter): the merge equals PEFT's own fp32 merge bit for bit, keeps the base's layout, changes
exactly the target tensors, rounds to nearest even, hashes what it wrote, and refuses every adapter the plain formula
does not cover without writing anything."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
# A Qwen3.5 LoRA adapter's target_modules, in its adapter_config.json's order.
TARGETS = ["in_proj_qkv", "out_proj", "in_proj_z", "down_proj", "o_proj", "in_proj_a", "v_proj", "in_proj_b", "q_proj",
           "gate_proj", "up_proj", "k_proj"]
SHARDS = ("model.safetensors-00001-of-00002.safetensors", "model.safetensors-00002-of-00002.safetensors")
LM_PREFIX, TEXT_PREFIX = "base_model.model.model.language_model.layers.", "base_model.model.model.layers."


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("merge_base_adapter", ROOT / "tools" / "merge_base_adapter.py")
TINY = _load("_wp9_tiny_qwen3_5", ROOT / "tests" / "sync" / "conftest.py")  # tiny_config() and TINY_TARGETS only


def u16(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def read_all(d: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    out = {}
    for fn in SHARDS:
        out.update(load_file(str(d / fn)))
    return out


def header_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return raw[: 8 + struct.unpack("<Q", raw[:8])[0]]


def run(*argv: str) -> int:
    return M.main([*argv, "--threads", str(torch.get_num_threads())])


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> SimpleNamespace:
    """A tiny Qwen3.5 base in the real base's layout (two shards with its file names, an index, fp32 A_log tensors,
    MTP tensors whose names end like targets), and one r4/alpha8 adapter in both namings an adapter may come in: the
    text-only model's `model.layers.N` and the checkpoint's `model.language_model.layers.N`."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3_5ForConditionalGeneration

    root = tmp_path_factory.mktemp("wp9-merge")
    torch.manual_seed(1234)
    saved = root / "saved"
    Qwen3_5ForConditionalGeneration(TINY.tiny_config()).to(torch.bfloat16).save_pretrained(str(saved))
    tensors = {}
    for f in sorted(saved.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
    a_logs = [k for k in tensors if k.endswith(".A_log")]
    assert a_logs
    for k in a_logs:
        tensors[k] = tensors[k].float()
    g = torch.Generator().manual_seed(5)
    for k, shape in {"mtp.fc.weight": (64, 128), "mtp.layers.0.mlp.gate_proj.weight": (96, 64),
                     "mtp.layers.0.self_attn.q_proj.weight": (128, 64)}.items():
        tensors[k] = (torch.randn(shape, generator=g) * 0.02).to(torch.bfloat16)
    base = root / "base"
    base.mkdir()
    keys = sorted(tensors)
    wm = {}
    for fn, part in zip(SHARDS, (keys[::2], keys[1::2]), strict=True):
        save_file({k: tensors[k].contiguous() for k in part}, str(base / fn))
        wm.update({k: fn for k in part})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                   "weight_map": wm}, indent=2))
    shutil.copy(saved / "config.json", base / "config.json")
    (base / "tokenizer_config.json").write_text("{}")
    (base / "chat_template.jinja").write_text("{# tiny #}")
    (base / "LICENSE").write_text("tiny\n")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(base), dtype=torch.float32)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS, bias="none"))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_A" in n:
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) / p.shape[1] ** 0.5)
            elif "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    lm = root / "adapter-lm"
    pm.save_pretrained(str(lm))
    sd = load_file(str(lm / M.ADAPTER_WEIGHTS))
    assert len(sd) == 2 * TINY.TINY_TARGETS and all(k.startswith(LM_PREFIX) for k in sd)
    text = root / "adapter-text"
    text.mkdir()
    shutil.copy(lm / M.ADAPTER_CONFIG, text / M.ADAPTER_CONFIG)
    save_file({k.replace(LM_PREFIX, TEXT_PREFIX): v for k, v in sd.items()}, str(text / M.ADAPTER_WEIGHTS),
              metadata={"format": "pt"})
    return SimpleNamespace(root=root, base=base, lm=lm, text=text, keys=keys)


@pytest.fixture(scope="module")
def merged(tiny) -> Path:
    out = tiny.root / "merged"
    rc = run("merge", "--base", str(tiny.base), "--adapter", str(tiny.text), "--out", str(out), "--expect-scale", "2.0",
             "--expect-modules", str(TINY.TINY_TARGETS), "--cross-check-adapter", str(tiny.lm), "--sensitivity")
    assert rc == 0
    return out


def target_names(tiny) -> list[str]:
    from aorlta.sync.writer import read_base_index, target_keys

    return sorted(target_keys(read_base_index(tiny.base)))


def test_the_merge_equals_peft_merge_and_unload_in_fp32(tiny, merged):
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(tiny.base), dtype=torch.float32)
    ref = PeftModel.from_pretrained(model, str(tiny.lm)).merge_and_unload().state_dict()
    out, base = read_all(merged), read_all(tiny.base)
    targets = target_names(tiny)
    assert len(targets) == TINY.TINY_TARGETS
    for k in targets:
        assert torch.equal(u16(ref[k].to(torch.bfloat16)), u16(out[k])), k
        assert not torch.equal(u16(out[k]), u16(base[k])), k


def test_the_output_keeps_the_base_layout_and_only_the_targets_change(tiny, merged):
    base_files = sorted(p.name for p in tiny.base.iterdir())
    assert sorted(p.name for p in merged.iterdir()) == sorted(base_files + [M.MANIFEST_NAME])
    for name in base_files:
        if name not in SHARDS:
            assert (merged / name).read_bytes() == (tiny.base / name).read_bytes(), name
    for fn in SHARDS:
        assert header_bytes(merged / fn) == header_bytes(tiny.base / fn)
        assert (merged / fn).stat().st_size == (tiny.base / fn).stat().st_size
    out, base = read_all(merged), read_all(tiny.base)
    targets = set(target_names(tiny))
    assert set(out) == set(base) == set(tiny.keys)
    changed = {k for k in base if out[k].dtype != base[k].dtype or
               not torch.equal(out[k].contiguous().view(torch.uint8), base[k].contiguous().view(torch.uint8))}
    assert changed == targets
    assert {"mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.self_attn.q_proj.weight"} <= set(base) - changed
    assert all(out[k].dtype == torch.float32 for k in base if k.endswith(".A_log"))


def test_the_manifest_hashes_are_the_files_and_tensors_written(tiny, merged):
    from aorlta.ids import sha256_json

    man = json.loads((merged / M.MANIFEST_NAME).read_text())
    assert man["pass"] is True and man["schema"] == M.SCHEMA
    files = man["output"]["files"]
    assert sorted(files) == sorted(p.name for p in tiny.base.iterdir())
    for name, row in files.items():
        assert row["sha256"] == hashlib.sha256((merged / name).read_bytes()).hexdigest(), name
    rows = man["output"]["tensors"]
    for fn in SHARDS:
        raw = (merged / fn).read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + n])
        header.pop("__metadata__", None)
        for k, v in header.items():
            a, z = v["data_offsets"]
            assert rows[k]["sha256"] == hashlib.sha256(raw[8 + n + a:8 + n + z]).hexdigest(), k
    assert man["output"]["tensor_digest"] == sha256_json({k: v["sha256"] for k, v in rows.items()})
    assert sum(v["target"] for v in rows.values()) == TINY.TINY_TARGETS
    assert all(v["changed_elements"] > 0 for v in rows.values() if v["target"])
    ad = man["adapter"]
    assert (ad["rank"], ad["lora_alpha"], ad["scale"], ad["modules"], ad["tensors"], ad["naming"]) == (
        4, 8.0, 2.0, TINY.TINY_TARGETS, 2 * TINY.TINY_TARGETS, "model.layers.N")
    assert man["cross_check"]["pass"] and man["cross_check"]["naming"] == "model.language_model.layers.N"
    assert man["checks"]["rounding"]["pass"] and man["checks"]["rounding"]["not_rne"] == 0
    assert man["sensitivity"]["one_thread_vs_threads"]["pass"]
    s = man["sensitivity"]["vs_fp64_accumulation"]
    assert s["neighbours"] + s["further"] == s["elements_differing"]
    assert man["checks"]["bytes"]["targets_changed"] == TINY.TINY_TARGETS


def test_both_adapter_namings_merge_to_the_same_bytes(tiny, merged):
    out = tiny.root / "merged-lm"
    assert run("merge", "--base", str(tiny.base), "--adapter", str(tiny.lm), "--out", str(out)) == 0
    for fn in SHARDS:
        assert (out / fn).read_bytes() == (merged / fn).read_bytes()
    shutil.rmtree(out)


def test_the_cast_is_round_to_nearest_even_and_the_check_has_teeth():
    from aorlta.sync.merge import merge_all

    w = torch.ones(3, 4, dtype=torch.bfloat16)
    a = torch.ones(1, 4)
    # scale 2: deltas 2^-8 (a tie between 1 and 1 + 2^-7), 3 * 2^-8 (a tie between 1 + 2^-7 and 1 + 2^-6), and just
    # above the first tie. Nearest-even gives 1, 1 + 2^-6, 1 + 2^-7; truncation would give 1, 1 + 2^-7, 1; rounding
    # half up would give 1 + 2^-7, 1 + 2^-6, 1 + 2^-7.
    b = torch.tensor([[2.0**-9], [3 * 2.0**-9], [2.0**-9 + 2.0**-20]])
    base, factors = {"k": w}, {"k": (a, b, 2.0)}
    got = merge_all(base, factors, base, version=0).weights["k"]
    want = torch.tensor([1.0, 1 + 2.0**-6, 1 + 2.0**-7]).to(torch.bfloat16)
    assert torch.equal(u16(got[:, 0]), u16(want))
    s = M.fp32_sum(w, a, b, 2.0)
    assert torch.equal(M.rne_bf16_bits(s), u16(got))
    assert M.check_rounding(base, factors, {"k": got})["pass"]
    truncated = (s.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
    assert not M.check_rounding(base, factors, {"k": truncated})["pass"]
    neg = torch.tensor([-(1 + 2.0**-8), -(1 + 3 * 2.0**-8)])
    assert torch.equal(M.rne_bf16_bits(neg), u16(torch.tensor([-1.0, -(1 + 2.0**-6)]).to(torch.bfloat16)))


@pytest.mark.parametrize(("b", "sum_finite"), [(1.7e38, True), (3e38, False)], ids=["cast_overflows", "sum_overflows"])
def test_an_element_whose_sum_or_cast_is_not_finite_counts_once(b, sum_finite):
    from aorlta.sync.merge import merge_all

    w = torch.zeros(2, 4, dtype=torch.bfloat16)
    base, factors = {"k": w}, {"k": (torch.ones(1, 4), torch.full((2, 1), b), 2.0)}
    got = merge_all(base, factors, base, version=0).weights["k"]
    assert bool(torch.isfinite(M.fp32_sum(w, *factors["k"])).all()) is sum_finite
    rounding = M.check_rounding(base, factors, {"k": got})
    assert (rounding["nonfinite"], rounding["pass"]) == (8, False)


def test_compare_bf16_tells_a_rounding_flip_from_a_value_further_away():
    one, step = 1.0, 2.0**-7
    tiny = 2.0**-133  # the smallest bf16 subnormal
    x = torch.tensor([one, one, 0.0, -0.0, tiny, 1e-30, 2.0]).to(torch.bfloat16)
    y = torch.tensor([one, one + step, -tiny, 0.0, -tiny, 0.0, 2.0 + 4 * step]).to(torch.bfloat16)
    c = M.compare_bf16(x, y)
    # equal; neighbours; neighbours across zero; -0 == +0; two apart across zero; far apart near zero; far apart
    assert (c["differing"], c["neighbours"], c["further"]) == (5, 2, 3)
    assert c["max_abs_difference"] == 4 * step
    assert c["max_abs_value_where_further"] == 2.0 + 4 * step
    total = M.add_comparison(M.compare_bf16(x[:0], y[:0]), c)
    assert total == c


def _adapter_variant(tiny, dst: Path, *, src: Path | None = None, config=None, tensors=None) -> Path:
    from safetensors.torch import load_file, save_file

    src = tiny.text if src is None else src
    dst.mkdir()
    cfg = json.loads((src / M.ADAPTER_CONFIG).read_text())
    if config:
        config(cfg)
    (dst / M.ADAPTER_CONFIG).write_text(json.dumps(cfg))
    sd = load_file(str(src / M.ADAPTER_WEIGHTS))
    if tensors:
        tensors(sd)
    save_file(sd, str(dst / M.ADAPTER_WEIGHTS))
    return dst


def _first(sd: dict, suffix: str) -> str:
    return next(k for k in sorted(sd) if k.endswith(suffix))


def _rename_module(sd: dict, new_module: str) -> None:
    key = _first(sd, "mlp.gate_proj.lora_A.weight")
    old = key[: -len(".lora_A.weight")]
    for ab in ("A", "B"):
        sd[f"{new_module}.lora_{ab}.weight"] = sd.pop(f"{old}.lora_{ab}.weight")


REFUSALS = {
    "alpha_r_not_2": dict(config=lambda c: c.update(lora_alpha=16)),
    "rslora": dict(config=lambda c: c.update(use_rslora=True)),
    "dora": dict(config=lambda c: c.update(use_dora=True)),
    "modules_to_save": dict(config=lambda c: c.update(modules_to_save=["lm_head"])),
    "rank_pattern": dict(config=lambda c: c.update(rank_pattern={"q_proj": 8})),
    "unknown_field": dict(config=lambda c: c.update(use_something_new=True)),
    "embedding_change": dict(tensors=lambda sd: sd.update(
        {"base_model.model.model.embed_tokens.weight": torch.zeros(320, 64)})),
    "missing_lora_b": dict(tensors=lambda sd: sd.pop(_first(sd, "self_attn.q_proj.lora_B.weight"))),
    "missing_module": dict(tensors=lambda sd: [sd.pop(_first(sd, f"self_attn.q_proj.lora_{ab}.weight"))
                                               for ab in "AB"]),
    "mtp_module": dict(tensors=lambda sd: _rename_module(sd, "base_model.model.mtp.layers.0.mlp.gate_proj")),
    "vision_module": dict(tensors=lambda sd: _rename_module(sd, "base_model.model.model.visual.blocks.0.attn.qkv")),
    "mixed_namings": dict(tensors=lambda sd: _rename_module(
        sd, _first(sd, "mlp.gate_proj.lora_A.weight")[: -len(".lora_A.weight")].replace(TEXT_PREFIX, LM_PREFIX))),
    "wrong_rank": dict(tensors=lambda sd: sd.update(
        {_first(sd, "self_attn.q_proj.lora_A.weight"): torch.zeros(5, 64)})),
    "overflowing_merge": dict(tensors=lambda sd: sd.update(
        {k: torch.full_like(v, 1e30) for k, v in sd.items() if ".self_attn.q_proj.lora_" in k})),
    "overflowing_cast": dict(tensors=lambda sd: sd.update(  # fp32 sums of 3.4e38: finite, but bf16 infinity
        {k: torch.full_like(v, 1.0 if ".lora_A." in k else 4.25e37) for k, v in sd.items()
         if ".self_attn.q_proj.lora_" in k})),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_adapters_the_formula_does_not_cover_are_refused_before_writing(tiny, tmp_path, capsys, case):
    adapter = _adapter_variant(tiny, tmp_path / "adapter", **REFUSALS[case])
    out = tmp_path / "out"
    rc = run("merge", "--base", str(tiny.base), "--adapter", str(adapter), "--out", str(out), "--expect-scale", "2.0",
             "--expect-modules", str(TINY.TINY_TARGETS))
    assert rc == 2, capsys.readouterr().err
    assert "REFUSED" in capsys.readouterr().err
    assert not out.exists() and not out.with_name("out.partial").exists()


@pytest.mark.parametrize("text", ['{"r": 4', "[]"])
def test_an_adapter_config_that_is_not_a_json_object_is_refused(tiny, tmp_path, capsys, text):
    adapter = _adapter_variant(tiny, tmp_path / "adapter")
    (adapter / M.ADAPTER_CONFIG).write_text(text)
    out = tmp_path / "out"
    assert run("merge", "--base", str(tiny.base), "--adapter", str(adapter), "--out", str(out)) == 2
    assert "REFUSED" in capsys.readouterr().err and not out.exists()
    assert run("verify", "--base", str(tiny.base), "--adapter", str(adapter), "--merged", str(out)) == 2
    assert "REFUSED" in capsys.readouterr().err


@pytest.mark.parametrize("text", ['{"base_model": ', '{"base_model": []}', "[]"])
def test_a_pin_file_that_is_not_a_pin_list_is_refused(tiny, tmp_path, capsys, text):
    pins = tmp_path / "pins.json"
    pins.write_text(text)
    out = tmp_path / "out"
    assert run("merge", "--base", str(tiny.base), "--adapter", str(tiny.text), "--out", str(out),
               "--base-pins", str(pins)) == 2
    assert "REFUSED" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("header", [b"{", b"[]", b'{"x": {"dtype": "BF16"}}'])
def test_a_base_shard_with_a_malformed_header_is_refused(tiny, tmp_path, capsys, header):
    base = tmp_path / "base"
    shutil.copytree(tiny.base, base)
    (base / SHARDS[0]).write_bytes(struct.pack("<Q", len(header)) + header)
    out = tmp_path / "out"
    assert run("merge", "--base", str(base), "--adapter", str(tiny.text), "--out", str(out)) == 2
    assert "REFUSED" in capsys.readouterr().err and not out.exists()


def test_a_base_index_that_is_not_json_is_refused(tiny, tmp_path, capsys):
    base = tmp_path / "base"
    shutil.copytree(tiny.base, base)
    (base / M.INDEX_NAME).write_text('{"weight_map":')
    out = tmp_path / "out"
    assert run("merge", "--base", str(base), "--adapter", str(tiny.text), "--out", str(out)) == 2
    assert "REFUSED" in capsys.readouterr().err and not out.exists()
    assert run("verify", "--base", str(base), "--adapter", str(tiny.text), "--merged", str(out)) == 2


def _modules(adapter: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(adapter / M.ADAPTER_WEIGHTS), framework="pt") as f:
        return sorted(k[len(M.PEFT_PREFIX):-len(".lora_A.weight")] for k in f.keys() if k.endswith(".lora_A.weight"))


def _suffixes(modules: list[str]) -> list[str]:
    return sorted({".".join(m.split(".")[-2:]) for m in modules})


@pytest.mark.parametrize("form", ["dotted_suffixes", "qualified_names"])
def test_target_modules_may_be_qualified_names_or_dotted_suffixes(tiny, merged, tmp_path, form):
    """PEFT targets a module whose full name is an entry or ends with "." + an entry: the tool takes the adapter
    PEFT builds from such a list, and merges it to the same bytes."""
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration

    modules = _modules(tiny.lm)
    names = _suffixes(modules) if form == "dotted_suffixes" else modules
    adapter = _adapter_variant(tiny, tmp_path / "adapter", src=tiny.lm,
                               config=lambda c: c.update(target_modules=names))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(tiny.base), dtype=torch.float32)
    pm = PeftModel.from_pretrained(model, str(adapter))
    assert sum(n.endswith(".lora_A") for n, _ in pm.named_modules()) == TINY.TINY_TARGETS
    out = tmp_path / "out"
    assert run("merge", "--base", str(tiny.base), "--adapter", str(adapter), "--out", str(out)) == 0
    for fn in SHARDS:
        assert (out / fn).read_bytes() == (merged / fn).read_bytes()


@pytest.mark.parametrize("form", ["suffix_without_a_dot", "other_naming"])
def test_target_modules_peft_would_not_match_are_refused(tiny, tmp_path, form):
    modules = _modules(tiny.lm)
    if form == "suffix_without_a_dot":  # "proj" ends every mlp module's name, but not after a dot
        names = [s for s in _suffixes(modules) if not s.startswith("mlp.")] + ["proj"]
    else:  # the text-only model's names for modules the adapter names model.language_model.layers.N
        names = [m.replace("model.language_model.", "model.") for m in modules]
    adapter = _adapter_variant(tiny, tmp_path / "adapter", src=tiny.lm,
                               config=lambda c: c.update(target_modules=names))
    out = tmp_path / "out"
    assert run("merge", "--base", str(tiny.base), "--adapter", str(adapter), "--out", str(out)) == 2
    assert not out.exists() and not out.with_name("out.partial").exists()


@pytest.mark.parametrize("change", ["one_tensor", "config"])
def test_a_cross_check_adapter_that_differs_is_refused(tiny, tmp_path, change):
    def bump(sd):
        k = _first(sd, "mlp.up_proj.lora_B.weight")
        sd[k] = sd[k] + 1e-3

    other = _adapter_variant(tiny, tmp_path / "other", **({"tensors": bump} if change == "one_tensor" else
                                                          {"config": lambda c: c.update(lora_dropout=0.05)}))
    out = tmp_path / "out"
    assert run("merge", "--base", str(tiny.base), "--adapter", str(tiny.text), "--out", str(out),
               "--cross-check-adapter", str(other)) == 2
    assert not out.exists() and not out.with_name("out.partial").exists()


def test_an_existing_output_and_wrong_base_pins_are_refused(tiny, merged, tmp_path, capsys):
    assert run("merge", "--base", str(tiny.base), "--adapter", str(tiny.text), "--out", str(merged)) == 2
    pins = tmp_path / "pins.json"
    files = [{"name": n, "bytes": (tiny.base / n).stat().st_size, "sha256": "0" * 64} for n in SHARDS]
    pins.write_text(json.dumps({"base_model": {"revision": "x", "files": files}}))
    out = tmp_path / "out"
    assert run("merge", "--base", str(tiny.base), "--adapter", str(tiny.text), "--out", str(out),
               "--base-pins", str(pins)) == 2
    assert not out.exists()
    files = [{"name": n, "bytes": (tiny.base / n).stat().st_size,
              "sha256": hashlib.sha256((tiny.base / n).read_bytes()).hexdigest()} for n in SHARDS]
    pins.write_text(json.dumps({"base_model": {"revision": "x", "files": files}}))
    assert run("merge", "--base", str(tiny.base), "--adapter", str(tiny.text), "--out", str(out),
               "--base-pins", str(pins)) == 0
    assert json.loads((out / M.MANIFEST_NAME).read_text())["base"]["pins"]["pass"] is True


def _flip_byte(path: Path, offset: int) -> None:
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ 0x01]))


@pytest.mark.parametrize("where", ["none", "non_target", "target", "header"])
def test_verify_catches_any_changed_byte(tiny, merged, tmp_path, where):
    copy = tmp_path / "copy"
    shutil.copytree(merged, copy)
    raw = (copy / SHARDS[1]).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    targets = set(target_names(tiny))
    if where == "header":
        _flip_byte(copy / SHARDS[1], 9)
    elif where != "none":
        key = next(k for k in sorted(header) if k != "__metadata__" and (k in targets) == (where == "target"))
        _flip_byte(copy / SHARDS[1], 8 + n + header[key]["data_offsets"][0])
    rc = run("verify", "--base", str(tiny.base), "--adapter", str(tiny.text), "--merged", str(copy))
    assert rc == (0 if where == "none" else 1)


OUTPUT_DAMAGE = {
    "index_missing": lambda d: (d / "model.safetensors.index.json").unlink(),
    "index_not_json": lambda d: (d / "model.safetensors.index.json").write_text("{\"weight_map\": "),
    "shard_missing": lambda d: (d / SHARDS[0]).unlink(),
    "config_missing": lambda d: (d / "config.json").unlink(),
    "manifest_missing": lambda d: (d / M.MANIFEST_NAME).unlink(),
    "manifest_not_json": lambda d: (d / M.MANIFEST_NAME).write_text("{\"schema\": "),
    "manifest_without_output": lambda d: (d / M.MANIFEST_NAME).write_text("{}"),
    "directory_missing": lambda d: shutil.rmtree(d),
}


@pytest.mark.parametrize("damage", sorted(OUTPUT_DAMAGE))
def test_verify_reports_missing_or_malformed_output_files_as_a_failed_check(tiny, merged, tmp_path, capsys, damage):
    copy = tmp_path / "copy"
    shutil.copytree(merged, copy)
    OUTPUT_DAMAGE[damage](copy)
    capsys.readouterr()
    assert run("verify", "--base", str(tiny.base), "--adapter", str(tiny.text), "--merged", str(copy)) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["pass"] is False and summary["problems"]


@pytest.mark.parametrize("where", ["absolute", "parent"])
def test_an_index_naming_a_shard_outside_the_model_directory_is_refused_before_writing(tiny, tmp_path, capsys, where):
    """`out / name` with such a name is the base's own shard, which the merge would then rewrite in place."""
    base = tmp_path / "base"
    shutil.copytree(tiny.base, base)
    index = json.loads((base / M.INDEX_NAME).read_text())
    shard = sorted(set(index["weight_map"].values()))[0]
    moved = str((base / shard).resolve()) if where == "absolute" else f"../base/{shard}"
    index["weight_map"] = {k: moved if fn == shard else fn for k, fn in index["weight_map"].items()}
    (base / M.INDEX_NAME).write_text(json.dumps(index))
    before = hashlib.sha256((base / shard).read_bytes()).hexdigest()
    out = tmp_path / "out"
    rc = run("merge", "--base", str(base), "--adapter", str(tiny.text), "--out", str(out), "--expect-scale", "2.0",
             "--expect-modules", str(TINY.TINY_TARGETS))
    assert rc == 2 and "REFUSED" in capsys.readouterr().err
    assert hashlib.sha256((base / shard).read_bytes()).hexdigest() == before
    assert not out.exists() and not out.with_name("out.partial").exists()


def test_a_base_saved_as_one_file_without_an_index_merges_to_the_same_tensors(tiny, merged, tmp_path):
    """A checkpoint small enough for one file is saved as `model.safetensors` with no index; its merge holds the same
    tensors as the two-shard base's, in one file, with no index either."""
    from safetensors.torch import load_file, save_file

    base = tmp_path / "one"
    shutil.copytree(tiny.base, base)
    tensors = read_all(base)
    for shard in SHARDS:
        (base / shard).unlink()
    (base / M.INDEX_NAME).unlink()
    save_file(tensors, str(base / "model.safetensors"), metadata={"format": "pt"})
    out = tmp_path / "out"
    rc = run("merge", "--base", str(base), "--adapter", str(tiny.text), "--out", str(out), "--expect-scale", "2.0",
             "--expect-modules", str(TINY.TINY_TARGETS))
    assert rc == 0
    assert not (out / M.INDEX_NAME).exists() and (out / "model.safetensors").is_file()
    got, want = load_file(str(out / "model.safetensors")), read_all(merged)
    assert sorted(got) == sorted(want)
    assert all(got[k].dtype == want[k].dtype and torch.equal(got[k].view(torch.uint8), want[k].view(torch.uint8))
               for k in want)
    assert run("verify", "--base", str(base), "--adapter", str(tiny.text), "--merged", str(out)) == 0
