"""Shared builders for the C4 tests: a tiny Qwen3.5 with PEFT LoRA, synthetic iterations, an in-process world.

The tiny model has the real model's module tree (WP3's tiny config: 3 linear-attention layers and 1 full-attention
layer), so PEFT selects the same kinds of target modules. On CPU transformers runs the linear-attention layers on
its torch path, which ignores `cu_seq_lens_q` and `seq_idx` (only the fast path honours them, WP1). The tests that
pack therefore swap in `segment_conv` and `segment_chunk`, which run the same torch code one sequence at a time:
the fast path's semantics, the torch path's arithmetic. They are test code, never used to train.
"""

from __future__ import annotations

import math
import os
import random
import threading

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from aorlta.ids import chain_id  # noqa: E402
from aorlta.schemas import RewardResult, TrainingSample  # noqa: E402

TINY_TEXT = dict(
    hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
    head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
    linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"], vocab_size=320,
    full_attention_interval=4, max_position_embeddings=4096, tie_word_embeddings=True,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default",
                     "rope_theta": 10000000, "partial_rotary_factor": 0.25},
    rms_norm_eps=1e-6, attn_output_gate=True,
)
TINY_VISION = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=64, patch_size=16,
                   spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)
TARGETS = (r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)"
           r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))")
VOCAB = 320
RUN = "wp4test"


# --- the tiny model --------------------------------------------------------------------------------


def segment_conv(x, weight, bias=None, activation=None, seq_idx=None):
    """causal_conv1d_fn's contract (x [B, C, L], seq_idx [B, L]) in torch, one sequence at a time."""
    _, c, n = x.shape
    k = weight.shape[-1]
    sid = seq_idx[0].tolist() if seq_idx is not None else [0] * n
    starts = [0] + [i for i in range(1, n) if sid[i] != sid[i - 1]] + [n]
    out = torch.cat([F.conv1d(F.pad(x[:, :, a:b], (k - 1, 0)), weight.unsqueeze(1), bias, groups=c)
                     for a, b in zip(starts[:-1], starts[1:])], dim=-1)
    return F.silu(out) if activation in ("silu", "swish") else out


def segment_chunk(query, key, value, g=None, beta=None, cu_seqlens=None, **kw):
    """chunk_gated_delta_rule's contract with `cu_seqlens`, in transformers' torch code, one sequence at a time."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    if cu_seqlens is None:
        return M.torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, **kw)
    edges = cu_seqlens.tolist()
    outs = [M.torch_chunk_gated_delta_rule(query[:, a:b], key[:, a:b], value[:, a:b], g=g[:, a:b], beta=beta[:, a:b],
                                           **kw)[0] for a, b in zip(edges[:-1], edges[1:])]
    return torch.cat(outs, dim=1), None


def honour_packing(model) -> None:
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    for m in model.modules():
        if isinstance(m, M.Qwen3_5GatedDeltaNet):
            m.causal_conv1d_fn = segment_conv
            m.chunk_gated_delta_rule = segment_chunk


def tiny_model(dtype=torch.float32, seed: int = 0, packing: bool = True):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from aorlta.trainer.batching import prepare_model

    torch.manual_seed(seed)
    cfg = Qwen3_5Config(text_config=dict(TINY_TEXT, dtype=str(dtype).replace("torch.", "")),
                        vision_config=TINY_VISION, tie_word_embeddings=True)
    m = Qwen3_5ForConditionalGeneration(cfg).to(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    if packing:
        honour_packing(m)
    prepare_model(m, require_fast_path=False)
    return m


def tiny_peft(dtype=torch.float32, seed: int = 0, *, b_std: float = 0.0, r: int = 4, alpha: float = 8.0,
              packing: bool = True):
    """The tiny model with PEFT LoRA on the target modules. b_std > 0 draws B from N(0, b_std) so that A gets a
    gradient at the first step (PEFT's default B = 0 gives A none)."""
    from peft import LoraConfig, get_peft_model

    m = tiny_model(dtype, seed, packing)
    pm = get_peft_model(m, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS, bias="none"))
    if b_std > 0:
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for mod in pm.modules():
                if hasattr(mod, "lora_B") and isinstance(mod.lora_B, torch.nn.ModuleDict) and "default" in mod.lora_B:
                    w = mod.lora_B["default"].weight
                    w.copy_(torch.randn(w.shape, generator=g) * b_std)
    return pm


def base_keys(pm) -> list[str]:
    """The tiny model's checkpoint names (what `save_pretrained` writes), read from a PEFT model: WP3's
    `MergedLoRA` maps each LoRA layer to one of them."""
    return [name.replace(".base_layer.", ".") for name, _ in pm.base_model.model.named_parameters()
            if "lora_" not in name]


def merged_forward(pm, **kwargs):
    """WP3's merged-weight forward on a PEFT model, installed (`kwargs` go to `MergedLoRA`, e.g. the rounding).
    Build it after `broadcast_lora`: it caches W_eff from the latent it sees."""
    from aorlta.trainer.merged_lora import MergedLoRA

    return MergedLoRA(pm, base_keys(pm), **kwargs).install()


class NoMerge:
    """For tests that check batching only: the plain PEFT forward, and a refresh that only counts versions."""

    def __init__(self):
        self.version = 0

    def refresh(self):
        self.version += 1
        return None

    def install(self):
        return self


# --- a saved tiny base, for rank pools (spawned ranks load it by path, with these module-level loaders) ----------


def save_tiny_base(path, *, dtype=torch.bfloat16, seed: int = 1234):
    """The tiny Qwen3.5 written as a Hugging Face directory with a safetensors index (what `read_base_index` reads)."""
    import json
    from pathlib import Path

    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    d = Path(path)
    torch.manual_seed(seed)
    cfg = Qwen3_5Config(text_config=dict(TINY_TEXT, dtype=str(dtype).replace("torch.", "")), vision_config=TINY_VISION,
                        tie_word_embeddings=True)
    Qwen3_5ForConditionalGeneration(cfg).to(dtype).save_pretrained(str(d), safe_serialization=True)
    if not (d / "model.safetensors.index.json").exists():
        wm = {}
        for p in sorted(d.glob("*.safetensors")):
            with safe_open(str(p), framework="pt") as f:
                wm.update({k: p.name for k in f.keys()})
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return d


def _load_tiny(base_dir, dtype):
    from transformers import Qwen3_5ForConditionalGeneration

    m = Qwen3_5ForConditionalGeneration.from_pretrained(str(base_dir), dtype=dtype)
    m.eval()
    honour_packing(m)
    return m


def load_tiny_bf16(base_dir, device):
    return _load_tiny(base_dir, torch.bfloat16).to(device), None


def load_tiny_fp32(base_dir, device):
    return _load_tiny(base_dir, torch.float32).to(device), None


def load_tiny_bf16_apart(base_dir, device):
    """Rank 2 (its process is named for it) loads a base that differs in one element of embed_tokens."""
    import multiprocessing as mp

    m, w = load_tiny_bf16(base_dir, device)
    if mp.current_process().name.endswith("rank2"):
        with torch.no_grad():
            next(p for n, p in m.named_parameters() if n.endswith("embed_tokens.weight")).view(-1)[0] += 1.0
    return m, w


def load_tiny_bf16_target_apart(base_dir, device):
    """Rank 2 loads a base that differs in one element of the second LoRA target (in sorted order), which the base
    control's sampled targets leave out: only every rank's W_eff, compared at the start, can see it."""
    import multiprocessing as mp
    import re

    m, w = load_tiny_bf16(base_dir, device)
    if mp.current_process().name.endswith("rank2"):
        targets = sorted(n for n, _ in m.named_modules() if re.fullmatch(TARGETS, n))
        with torch.no_grad():
            dict(m.named_modules())[targets[1]].weight.view(-1)[0] += 0.5
    return m, w


def no_merge(model, keys, **kwargs):
    """A RankSpec merged factory for the fp32 tests: the plain PEFT forward (`NoMerge`)."""
    return NoMerge()


# --- synthetic iterations -----------------------------------------------------------------------------


def synthetic_iteration(*, groups: int = 6, k: int = 4, seed: int = 0, iteration: int = 1, prompt=(9, 30),
                        response=(3, 20), unscored: float = 0.15, quarantined: float = 0.05, one_scored_group=True,
                        scored_without_sample: int = 1):
    """TrainingSamples and RewardResults for one iteration. Engine log-probs are placeholders (filled by
    `engine_logprobs_from`). One group is left with a single scored sample, and `scored_without_sample` scored
    results get no TrainingSample (as for a logprob_error chain)."""
    rnd = random.Random(seed)
    samples, results = [], []
    for gi in range(groups):
        pid = f"standin:p{gi}"
        for j in range(k):
            cid = chain_id(RUN, iteration, pid, j)
            p = [rnd.randrange(VOCAB) for _ in range(rnd.randint(*prompt))]
            r = [rnd.randrange(VOCAB) for _ in range(rnd.randint(*response))]
            samples.append(TrainingSample(cid, iteration, pid, p, r, [-1.0] * len(r)))
            u = rnd.random()
            if one_scored_group and gi == groups - 1 and j > 0:
                results.append(_result(cid, None, "unscored", "timeout"))
            elif u < quarantined:
                results.append(_result(cid, None, "quarantined", "leak_check"))
            elif u < quarantined + unscored:
                results.append(_result(cid, None, "unscored", "sanitizer_not_settled"))
            else:
                results.append(_result(cid, round(rnd.uniform(-1.55, 1.0), 4), "scored", None))
    dropped = 0
    keep = []
    for s in samples:
        r = next(x for x in results if x.candidate_id == s.chain_id)
        if dropped < scored_without_sample and r.status == "scored" and not s.prompt_id.endswith(str(groups - 1)):
            dropped += 1
            continue
        keep.append(s)
    return keep, results


def _result(cid, reward, status, reason):
    return RewardResult(candidate_id=cid, reward=reward, status=status, reason=reason, parts=None,
                        reward_fn={"name": "test", "version": "0", "code_sha256": "0"}, legs={}, full_check=True,
                        plan=None, audit=False, cache_key="0", timings={})


def engine_logprobs_from(model, samples, *, device="cpu", amp=False, shift: dict | None = None):
    """Each sample's raw log-probs scored on its own by `model`: an engine that agrees with the trainer up to
    rounding. `shift` maps chain_id -> (token index, nats added) to plant a gap."""
    from aorlta.trainer.batching import raw_logprobs_at

    out = []
    for s in samples:
        ids = [*s.prompt_ids, *s.response_ids]
        pos = list(range(len(s.prompt_ids), len(ids)))
        lp = [float(x) for x in raw_logprobs_at(model, ids, pos, device=device, amp=amp)]
        if shift and s.chain_id in shift:
            t, d = shift[s.chain_id]
            lp[t] = min(0.0, lp[t] - d)
        out.append(TrainingSample(s.chain_id, s.iteration, s.prompt_id, s.prompt_ids, s.response_ids, lp))
    return out


# --- an in-process world: R ranks on R threads, the real collective protocol --------------------------


class SharedCollective:
    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=600)
        self.slots: list = [None] * size

    def world(self, rank: int) -> "ThreadWorld":
        return ThreadWorld(self, rank)


class ThreadWorld:
    def __init__(self, shared: SharedCollective, rank: int):
        self.shared, self.rank, self.size = shared, rank, shared.size

    def all_reduce_sum_(self, t: torch.Tensor) -> None:
        sh = self.shared
        sh.slots[self.rank] = t.detach().clone()
        sh.barrier.wait()
        total = sh.slots[0].clone()
        for x in sh.slots[1:]:
            total += x
        sh.barrier.wait()
        t.copy_(total)

    def broadcast_(self, t: torch.Tensor, src: int = 0) -> None:
        sh = self.shared
        if self.rank == src:
            sh.slots[src] = t.detach().clone()
        sh.barrier.wait()
        value = sh.slots[src].clone()
        sh.barrier.wait()
        t.copy_(value)

    def all_gather_object(self, obj):
        sh = self.shared
        sh.slots[self.rank] = obj
        sh.barrier.wait()
        out = list(sh.slots)
        sh.barrier.wait()
        return out


def run_ranks(fns, shared: SharedCollective | None = None):
    """Run one callable per rank on its own thread; re-raise the first error (a failing rank breaks the
    barrier, so the others stop instead of waiting for it)."""
    results, errors = [None] * len(fns), []

    def go(i):
        try:
            results[i] = fns[i]()
        except threading.BrokenBarrierError as exc:
            errors.append(exc)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.insert(0, exc)
            if shared is not None:
                shared.barrier.abort()

    threads = [threading.Thread(target=go, args=(i,)) for i in range(len(fns))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def gloo_iteration(agg: str, world=None, seed: int = 11):
    """The iteration test_ranks.py runs over gloo, and over LocalWorld for comparison."""
    from aorlta.trainer.loss import LossSettings, build_groups
    from aorlta.trainer.ranks import LocalWorld, OptimSettings, RankTrainer, broadcast_lora

    model = tiny_peft(torch.float32, seed=seed, b_std=0.05)
    if world is not None:
        broadcast_lora(model, world)
    samples, results = synthetic_iteration(groups=8, k=4, seed=seed)
    groups = build_groups(engine_logprobs_from(model, samples), results)
    t = RankTrainer(model, NoMerge(), LossSettings(aggregation=agg, loss_temperature=0.7),
                    OptimSettings(lr=1e-3, grad_clip=0.0), world=world or LocalWorld(), micro_batch_tokens=96,
                    amp=False)
    (rep,) = t.train_iteration(groups)
    return {"grad": t.last_flat_grad.clone(), "params": [p.detach().clone() for p in t.params], "loss": rep["loss"],
            "checks": {k: v["passed"] for k, v in rep["checks"].items()}, "ranks": rep["ranks"],
            "microbatches": rep["counts"]["microbatches"]}


def gloo_worker(rank: int, world_size: int, store: str, out_dir: str, agg: str) -> None:
    """One rank of the real collective: gloo over a file store, the trainer's DistWorld."""
    import torch.distributed as dist

    from aorlta.trainer.ranks import DistWorld

    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world_size)
    try:
        torch.save(gloo_iteration(agg, DistWorld()), f"{out_dir}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(a @ b) / (float(a.norm()) * float(b.norm()) + 1e-300)


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def flat_grads(params) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double() for p in params])


def isclose(a, b, tol=1e-12):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
