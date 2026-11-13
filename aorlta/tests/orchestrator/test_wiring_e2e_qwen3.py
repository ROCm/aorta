"""The wired components end to end on CPU on a tiny dense Qwen3, as test_wiring_e2e runs them on a tiny Qwen3.5.

What a Qwen3 changes, and this run goes through:
  - the tokenizer's control ids (Qwen3's 151643-151668) and its chat template, whose thinking-on generation prompt
    opens no think block: the rollout's stop token, the repair suffix and the replies' stop stripping come from it;
  - the trainer's loader: the real one (`wiring.load_policy_model`), which picks AutoModelForCausalLM from
    config.json, with `trainer.require_fast_path` on and the linear-attention check not applying;
  - the decoder's `model.layers.N` names, through the name map, the merge, the slot writer and the reload;
  - tied embeddings stored as Qwen3-0.6B stores them: one `model.safetensors` with no index, holding
    `lm_head.weight` beside `model.embed_tokens.weight`. A full slot carries it unchanged; the engine replica skips
    it (TokenSpeed's Qwen3 does, with tied embeddings) and counts it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

ORCH = Path(__file__).resolve().parent
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

import test_wiring_e2e as E  # noqa: E402

from aorlta.config import config_from_dict  # noqa: E402
from aorlta.orchestrator import wiring  # noqa: E402
from aorlta.rollout.compile_check import FakeCompiler  # noqa: E402
from aorlta.rollout.engine_proc import InProcessEngine  # noqa: E402
from aorlta.rollout.fake_engine import BYTE_TOKEN_BASE  # noqa: E402
from aorlta.rollout.templates import ChatFormat  # noqa: E402
from aorlta.sync.writer import expected_count, read_base_index  # noqa: E402

pytestmark = pytest.mark.heavy

VOCAB = 151936  # Qwen3's: the byte ids (10000+) and the control ids are real rows
SPECIAL = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645, "<think>": 151667,
           "</think>": 151668}
_SPECIAL_RE = re.compile("|".join(re.escape(s) for s in SPECIAL))
_SPECIAL_TEXT = {v: k for k, v in SPECIAL.items()}
IM_END = SPECIAL["<|im_end|>"]

dataset = E.dataset


class Qwen3ByteTokenizer:
    """The fake engine's byte tokenizer plus Qwen3's control tokens, with the system/user part of Qwen3's chat
    template: thinking off closes an empty think block, thinking on leaves the block to the model. Its format is
    read from it as from a real tokenizer (`ChatFormat.derive`)."""

    def __init__(self):
        self.chat_format = ChatFormat.derive(self)

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        for m in _SPECIAL_RE.finditer(text):
            out += [BYTE_TOKEN_BASE + b for b in text[i:m.start()].encode("utf-8")]
            out.append(SPECIAL[m.group(0)])
            i = m.end()
        return out + [BYTE_TOKEN_BASE + b for b in text[i:].encode("utf-8")]

    def decode(self, ids) -> str:
        parts, buf = [], bytearray()
        for t in ids:
            if t in _SPECIAL_TEXT:
                parts += [buf.decode("utf-8", errors="replace"), _SPECIAL_TEXT[t]]
                buf.clear()
            else:
                buf.append(t - BYTE_TOKEN_BASE)
        return "".join(parts + [buf.decode("utf-8", errors="replace")])

    def render(self, messages, thinking, add_generation_prompt=True) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in messages)
        if not add_generation_prompt:
            return text
        return text + "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if thinking == "off" else "")


def load_engine_model(base_dir) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class RecordingTwin(E.TwinEngine):
    """The twin, recording each slot it reloads: its tensors' names, its lm_head.weight if it holds one, and the
    reply's count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slots: list[dict] = []

    def update_weights_from_disk(self, model_path: str, flush_cache: bool = True):
        from aorlta.sync.writer import load_slot_tensors

        held = load_slot_tensors(model_path)
        reply = super().update_weights_from_disk(model_path, flush_cache)
        self.slots.append({"names": set(held), "lm_head": held.get("lm_head.weight"), "reply": reply})
        return reply


@pytest.fixture(scope="module")
def tiny_qwen3(tmp_path_factory) -> Path:
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny-qwen3-real-vocab")
    torch.manual_seed(1234)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=32, vocab_size=VOCAB, max_position_embeddings=4096,
                      tie_word_embeddings=True, rms_norm_eps=1e-6)
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(str(d), safe_serialization=True)
    path = d / "model.safetensors"
    tensors = load_file(str(path))
    assert "lm_head.weight" not in tensors and not (d / "model.safetensors.index.json").exists()
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, str(path), metadata={"format": "pt"})
    return d


def run_config(tmp_path, base, ds, *, partial: bool, thinking: str):
    return config_from_dict({
        "run": {"run_id": "wired-qwen3", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(base), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(ds / "listings"), "source": "standin"}]},
        "rollout": {"engines": [0, 1], "K": 4, "P": 4, "max_new_tokens": 512, "request_timeout_s": 300,
                    "compile_adapter": "fake.compile", "thinking": thinking},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"iterations": 2, "run_seed": 7},
        "trainer": {"ranks": 1, "lr": 1e-2, "micro_batch_tokens": 2048, "gap_gate": "enforce"},
        "sync": {"dir": str(tmp_path / "shm"), "partial": partial},
    })


@pytest.mark.parametrize("partial,thinking", [(True, "off"), (False, "on")])
def test_two_wired_iterations_on_a_tiny_qwen3_train_publish_and_land(tmp_path, tiny_qwen3, dataset, partial,
                                                                      thinking):
    cfg = run_config(tmp_path, tiny_qwen3, dataset, partial=partial, thinking=thinking)
    engines = [RecordingTwin(tiny_qwen3, E.FAKE, i, loader=load_engine_model) for i in range(2)]
    handles = [InProcessEngine(i, e) for i, e in enumerate(engines)]
    parts = wiring.Parts(tokenizer=Qwen3ByteTokenizer(), handles=handles, compiler=FakeCompiler(), trainer_device="cpu",
                         probes=E.PROBES)  # no load_model: the real loader, and the config's require_fast_path
    orch = wiring.build(cfg, parts=parts)
    result = orch.run()
    assert result["status"] == "completed", result
    assert result["iterations_completed"] == 2

    keys = list(read_base_index(tiny_qwen3))
    assert len(keys) == 25 and "lm_head.weight" in keys
    trained_multi_turn = 0
    for n in (1, 2):
        s = E.summary(cfg, n)
        assert s["samples"]["total"] == 16 and s["scored_fraction"] >= 0.9, s["samples"]
        assert s["trainer"]["step_taken"] and s["samples"]["trained"] > 0
        assert s["trainer"]["gap_gate"]["mode"] == "enforce" and s["trainer"]["gap_p99"] < 0.5
        assert s["sync"]["landing"] == "LANDED" and s["sync"]["changed"] > 0
        eps = E.episodes(cfg, n)
        trained_multi_turn += sum(1 for e in eps if e["trained"] and len(e["turns"]) > 1)
    assert trained_multi_turn > 0

    merged = orch.trainer.merged
    targets = {k for k in keys if re.fullmatch(r"model\.layers\.\d\.(self_attn\.[qkvo]|mlp\.(gate|up|down))_proj\.weight", k)}
    assert sorted(merged.weights) == sorted(targets) and len(targets) == 14  # 2 layers x 7 projections
    assert all(E.serves(e, merged.weights) for e in engines)
    assert not any(torch.equal(merged.weights[k], merged.base[k]) for k in merged.weights)
    # the trainer's base was loaded by the real loader, as a causal LM, with the fast-path check required
    assert type(orch.trainer.model.get_base_model()).__name__ == "Qwen3ForCausalLM"
    # every request stopped on Qwen3's <|im_end|>
    reqs = [r for e in engines for r in e.fake.requests if r.outcome == "ok"]
    assert reqs and all(r.sampling_params["stop_token_ids"] == [IM_END] for r in reqs)
    # each published slot holds what its reload must count: the targets, or every base key, lm_head.weight unchanged
    from safetensors.torch import load_file

    base_head = load_file(str(tiny_qwen3 / "model.safetensors"))["lm_head.weight"]
    published = engines[0].slots[2:]  # after the round trip's two
    assert len(published) == 2
    for rec in published:
        assert rec["names"] == (targets if partial else set(keys))
        n = expected_count(keys, partial=partial)
        assert n == (14 if partial else 25) and rec["reply"][1].endswith(f"({n} checkpoint tensors read)")
        assert (rec["lm_head"] is None) if partial else torch.equal(rec["lm_head"], base_head)
