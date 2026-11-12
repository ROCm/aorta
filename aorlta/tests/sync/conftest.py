"""Shared fixtures for the C5 tests: a tiny Qwen3.5 checkpoint on disk and a CPU stand-in for a TokenSpeed engine.

The tiny model has the real model's module and checkpoint names (`model.language_model.layers.N....`), so the
name map, the writer and the reload see the same names they see on Qwen3.5-4B. `TinyEngine` behaves like
TokenSpeed where WP3 depends on it: `update_weights_from_disk` reads a slot's shards, copies every tensor whose
name it knows, silently skips a name it does not know but still counts it as read (A5, A6), and replies
`(True, 'Succeeded ... (N checkpoint tensors read)', 0)`; `generate` returns prompt log-probs as
`(None|lp, id, None)` triples with `cached_tokens` 0, as a fresh prefill does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

TINY_TEXT = dict(
    hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
    head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"], vocab_size=320,
    full_attention_interval=4, max_position_embeddings=4096, tie_word_embeddings=True,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default",
                     "rope_theta": 10000000, "partial_rotary_factor": 0.25},
    rms_norm_eps=1e-6, attn_output_gate=True, dtype="bfloat16",
)
TINY_VISION = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=64, patch_size=16,
                   spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)
TINY_TARGETS = 31  # 3 linear-attention layers x 8 + 1 full-attention layer x 7


def tiny_config():
    from transformers import Qwen3_5Config

    return Qwen3_5Config(text_config=TINY_TEXT, vision_config=TINY_VISION, tie_word_embeddings=True)


def load_tiny(base_dir: Path):
    from transformers import Qwen3_5ForConditionalGeneration

    m = Qwen3_5ForConditionalGeneration.from_pretrained(str(base_dir), dtype=torch.bfloat16)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory) -> Path:
    """A tiny Qwen3.5 checkpoint in bf16, with a `model.safetensors.index.json` like the real one."""
    from transformers import Qwen3_5ForConditionalGeneration

    d = tmp_path_factory.mktemp("tiny-base")
    torch.manual_seed(1234)
    m = Qwen3_5ForConditionalGeneration(tiny_config()).to(torch.bfloat16)
    m.save_pretrained(str(d), safe_serialization=True)
    from safetensors import safe_open

    files = sorted(p.name for p in d.glob("*.safetensors"))
    wm = {}
    for fn in files:
        with safe_open(str(d / fn), framework="pt") as f:
            wm.update({k: fn for k in f.keys()})
    if not (d / "model.safetensors.index.json").exists():
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "tokenizer_config.json").write_text("{}")
    (d / "chat_template.jinja").write_text("{# tiny #}")
    return d


def make_peft(base_dir: Path, *, r: int = 4, alpha: float = 8.0, seed: int = 0):
    from peft import LoraConfig, get_peft_model

    from aorlta.sync.merge import TARGET_MODULE_PATTERN

    torch.manual_seed(seed)
    m = load_tiny(base_dir)
    return get_peft_model(m, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                                        target_modules=TARGET_MODULE_PATTERN, bias="none"))


@pytest.fixture
def peft_factory(tiny_base):
    return lambda **kw: make_peft(tiny_base, **kw)


class TinyEngine:
    """A CPU engine with TokenSpeed's reply shapes, backed by the tiny model."""

    def __init__(self, base_dir: Path, engine_id: int = 0):
        self.engine_id = engine_id
        self.model = load_tiny(base_dir)
        self.params = dict(self.model.named_parameters())
        self.reloads: list[str] = []
        self.skipped: list[str] = []
        self.busy = 0
        self.fail_next: list = []

    def update_weights_from_disk(self, model_path: str, flush_cache: bool = True):
        from safetensors import safe_open

        if self.fail_next:
            return self.fail_next.pop(0)
        path = Path(model_path)
        wm = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        consumed = 0
        with torch.no_grad():
            for fn in sorted(set(wm.values())):
                with safe_open(str(path / fn), framework="pt") as f:
                    for name in f.keys():
                        consumed += 1
                        p = self.params.get(name)
                        if p is None:  # A6: an unknown name is skipped, yet it was read
                            self.skipped.append(name)
                            continue
                        p.copy_(f.get_tensor(name))
        self.reloads.append(str(path))
        if consumed == 0:
            return (False, f"no checkpoint tensors found at {model_path!r}; nothing was updated", 0)
        return (True, f"Succeeded to update model weights from {model_path} ({consumed} checkpoint tensors read)", 0)

    def generate(self, input_ids, sampling_params, return_logprob=True, logprob_start_len=-1):
        ids = list(input_ids)
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = self.model(input_ids=x).logits[0]
        lp = torch.log_softmax(logits.float(), dim=-1)
        meta = {"cached_tokens": 0, "finish_reason": {"type": "length"}}
        if logprob_start_len == 0:
            vals = lp[:-1].gather(-1, x[0, 1:, None])[:, 0].tolist()
            meta["input_token_logprobs"] = [(None, ids[0], None)] + [(v, t, None) for v, t in zip(vals, ids[1:])]
        out_ids = []
        cur = ids
        for _ in range(int(sampling_params.get("max_new_tokens", 1))):
            nxt = int(torch.argmax(lp[-1]))
            out_ids.append(nxt)
            cur = cur + [nxt]
            with torch.no_grad():
                lp = torch.log_softmax(self.model(input_ids=torch.tensor([cur])).logits[0].float(), dim=-1)
        return {"output_ids": out_ids, "meta_info": meta}


@pytest.fixture
def engine_factory(tiny_base):
    return lambda engine_id=0: TinyEngine(tiny_base, engine_id)


@pytest.fixture
def tiny_probes():
    from aorlta.sync.landing import ProbeSet

    g = torch.Generator().manual_seed(99)
    return ProbeSet(tuple(tuple(int(t) for t in torch.randint(0, 320, (48,), generator=g)) for _ in range(3)))
