"""C4.T10's measurement: the on-policy trainer-engine log-prob gap, by turn (design Section 6.4,
Checks; Section 8, WP4, "Added 2 Nov after WP1"; risk R15).

A chain's final sequence holds every turn's reply (Appendix C: ids_{t+1} starts with ids_t + reply_t), so one trainer
forward over it scores every turn: turn t's reply sits at [input_len_t, input_len_t + len_t). The engine's log-probs
for that reply are the ones its turn-t request returned (fp32 log-softmax of the raw logits, temperature 1; A3), so
both sides compare raw with raw, as the gap gate does.

What is scored, on the same samples:
  - `score_positions`: one sequence alone, exactly as `batching.raw_logprobs_at` computes it (the landing check's
    and WP4's trainer-side function), plus the log-prob at the loss temperature and the raw entropy;
  - `score_packed`: the gap gate's own view, each group packed as the streaming step packs it;
  - `fp32_linear_attention`: fix 1, the linear-attention core on fp32 inputs. fla 0.6.0 stores the per-chunk state in
    its inputs' dtype and casts the state to that dtype for the w.h product, so the state is fp32 only when the
    kernel's inputs are (WP1's "f32in").
Nothing here trains or changes a weight.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from aorlta.trainer import kernels
from aorlta.trainer.batching import (DEFAULT_CHUNK_TOKENS, _head_logits, amp_context, backbone_and_head,
                                     forward_microbatch, pack)
from aorlta.trainer.loss import TrainerInputError, TurnSpan, turn_spans

# Per-sample statistics a gap gate could use, each a function of one reply's per-token gaps (see `reply_stats`).
STATISTICS = ("max", "mean", "median", "p90", "p99", "frac_gt_0p5", "frac_gt_1", "abs_mean_signed", "k3_mean")


@dataclass(frozen=True)
class ChainItem:
    """One chain with engine log-probs on every turn: its final sequence and where each turn's reply sits in it."""

    chain_id: str
    prompt_id: str
    k: int
    end_reason: str
    ids: tuple[int, ...]  # the final turn's input ids + its reply
    spans: tuple[TurnSpan, ...]  # every turn's reply, in turn order, with the engine's log-probs
    meta: tuple[dict, ...]  # per turn: input_len, n_reply, cached_tokens, ideal_cached_tokens, finish

    @property
    def positions(self) -> list[int]:
        return [p for s in self.spans for p in range(s.start, s.start + s.length)]

    @property
    def final_turn(self) -> int:
        return self.spans[-1].turn


def chain_items(chains: Iterable, samples: Iterable, *, rerendered: bool = False) -> list[ChainItem]:
    """Every chain that has a TrainingSample (so its engine log-probs line up on every turn), in chain order.
    `turn_spans` refuses a chain whose turn records do not reproduce the sample's ids; a `rerendered` chain keeps the
    turns whose reply is in the final sequence (see `turn_spans`), and `meta` holds the same turns."""
    by_id = {s.chain_id: s for s in samples}
    out = []
    for rec in chains:
        s = by_id.get(rec.chain_id)
        if s is None:
            continue  # engine_error, quarantined, logprob_error: nothing to compare
        spans = turn_spans(s, rec, rerendered=rerendered)
        kept = {sp.turn for sp in spans}
        meta = tuple({"turn": tr.turn, "input_len": tr.input_len, "n_reply": len(tr.response_ids),
                      "cached_tokens": tr.cached_tokens, "ideal_cached_tokens": tr.ideal_cached_tokens,
                      "finish": str((tr.finish_reason or {}).get("type"))} for tr in rec.turns if tr.turn in kept)
        out.append(ChainItem(rec.chain_id, rec.prompt_id, rec.k, rec.end_reason,
                             tuple(s.prompt_ids) + tuple(s.response_ids), spans, meta))
    return out


# --- the trainer's side ---------------------------------------------------------------------------------


def score_positions(model, ids: Sequence[int], positions: Sequence[int], *, device, temperature: float,
                    amp: bool = True, lm_head_fp32: bool = False, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                    extras: bool = True) -> dict[str, torch.Tensor]:
    """One sequence alone, no grad: the raw log-prob of ids[p] for each p in `positions`, computed exactly as
    `batching.raw_logprobs_at` computes it (same backbone call, same head chunks), plus, with `extras`, the log-prob
    at `temperature` and the entropy of the raw distribution at the same positions. Tensors are float32 on the CPU."""
    if not positions or min(positions) < 1 or max(positions) >= len(ids):
        raise TrainerInputError("positions must lie in [1, len(ids))")
    backbone, head = backbone_and_head(model)
    n = len(ids)
    x = torch.tensor([list(ids)], dtype=torch.long, device=device)
    pos = torch.tensor(list(positions), dtype=torch.long, device=device)
    raws, temps, ents = [], [], []
    with torch.no_grad(), amp_context(device, amp):
        cu = torch.tensor([0, n], dtype=torch.int32, device=device)
        hidden = backbone(input_ids=x, use_cache=False, position_ids=torch.arange(n, device=device)[None],
                          cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=n, max_length_k=n,
                          seq_idx=torch.zeros((1, n), dtype=torch.int32, device=device))
        h = hidden.last_hidden_state[0].index_select(0, pos - 1)
        targets = x[0].index_select(0, pos)
        for a in range(0, h.shape[0], chunk_tokens):
            z = _head_logits(head, h[a:a + chunk_tokens], lm_head_fp32).float()
            t = targets[a:a + chunk_tokens, None]
            ls = torch.log_softmax(z, dim=-1)
            raws.append(ls.gather(-1, t)[:, 0])
            if extras:
                temps.append(torch.log_softmax(z / temperature, dim=-1).gather(-1, t)[:, 0])
                ents.append(-(ls.exp() * ls).sum(-1))
    out = {"raw": torch.cat(raws).float().cpu()}
    if extras:
        out["temp"] = torch.cat(temps).float().cpu()
        out["entropy"] = torch.cat(ents).float().cpu()
    return out


def score_items(model, items: Sequence[ChainItem], *, device, temperature: float, amp: bool = True,
                lm_head_fp32: bool = False, extras: bool = True) -> list[dict[str, torch.Tensor]]:
    """`score_positions` for every item over every turn's reply, in turn order."""
    model.eval()
    return [score_positions(model, it.ids, it.positions, device=device, temperature=temperature, amp=amp,
                            lm_head_fp32=lm_head_fp32, extras=extras) for it in items]


def score_packed(model, plans: Sequence, *, micro_batch_tokens: int, device, temperature: float, amp: bool = True,
                 lm_head_fp32: bool = False, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> dict[str, torch.Tensor]:
    """The gap gate's view: each group's samples packed as the streaming step packs them (`RankTrainer._add` with
    across_groups=False: one `pack` per group, one sequence per row where the linear attention would run a packed row
    as one sequence), forwarded by `forward_microbatch` without grad. Returns the raw log-probs of each sample's
    response (its final turn), by chain id."""
    model.eval()
    max_sequences = None if kernels.honours_packing(model) else 1
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for g in plans:
            if not g.samples:
                continue
            for mb in pack(list(g.samples), micro_batch_tokens, max_sequences=max_sequences):
                _, raw, _ = forward_microbatch(model, mb, device=device, temperature=temperature, amp=amp,
                                               lm_head_fp32=lm_head_fp32, chunk_tokens=chunk_tokens)
                for s, sl in zip(mb.samples, mb.sample_slices()):
                    out[s.chain_id] = raw[sl].detach().float().cpu()
    return out


def _fp32_inputs(fn):
    def chunk_fp32(query, key, value, g=None, beta=None, **kwargs):
        o, state = fn(query.float(), key.float(), value.float(), g=g.float(), beta=beta.float(), **kwargs)
        return o.to(query.dtype), state

    chunk_fp32.aorlta_fp32 = True
    return chunk_fp32


def linear_attention_layers(model) -> list:
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    return [m for _, m in model.named_modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]


@contextlib.contextmanager
def fp32_linear_attention(model):
    """Fix 1 on the trainer side: every linear-attention layer's chunked gated-delta rule runs on fp32 inputs, so
    its state, its WY factors and its output are fp32 (the output goes back to the input dtype, as before). Undone on
    exit. Refuses a model with no linear-attention layers, so the fix can never silently do nothing."""
    layers = linear_attention_layers(model)
    if not layers:
        raise TrainerInputError("the model has no Qwen3.5 linear-attention layers to run in fp32")
    saved = [(m, m.chunk_gated_delta_rule) for m in layers]
    try:
        for m, fn in saved:
            m.chunk_gated_delta_rule = _fp32_inputs(fn)
        yield len(saved)
    finally:
        for m, fn in saved:
            m.chunk_gated_delta_rule = fn


# --- statistics ----------------------------------------------------------------------------------------


def reply_stats(delta: np.ndarray) -> dict[str, float]:
    """Candidate per-sample gate statistics of one reply's per-token gaps delta = trainer - engine (raw log-probs).
    `k3_mean` is the mean of exp(delta) - 1 - delta, Schulman's k3 estimate of KL(engine || trainer) from the engine's
    own samples; `abs_mean_signed` is |mean delta|, the per-token log importance ratio's size."""
    d = np.asarray(delta, dtype=np.float64)
    if d.size == 0:
        raise ValueError("a reply with no tokens has no gap")
    a = np.abs(d)
    return {"tokens": int(d.size), "max": float(a.max()), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)), "p99": float(np.quantile(a, 0.99)),
            "frac_gt_0p5": float((a > 0.5).mean()), "frac_gt_1": float((a > 1.0).mean()),
            "mean_signed": float(d.mean()), "abs_mean_signed": float(abs(d.mean())),
            "k3_mean": float(np.mean(np.expm1(d) - d))}


def quantiles(values: Iterable[float], qs: Sequence[float] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size)}
    for q in qs:
        out[f"p{q * 100:g}" if q != 0.5 else "median"] = float(np.quantile(v, q))
    out["max"] = float(v.max())
    out["mean"] = float(v.mean())
    return out


def token_quantiles(delta: np.ndarray) -> dict[str, float]:
    """Per-token |delta|: median, p90, p99, p99.9 and max."""
    return quantiles(np.abs(np.asarray(delta, dtype=np.float64)), (0.5, 0.9, 0.99, 0.999))


def exclusions(stats: Sequence[Mapping[str, float]], statistic: str, limits: Sequence[float]) -> dict[str, int]:
    """How many samples a gate on `statistic` would exclude at each limit (a sample is excluded when its statistic
    exceeds the limit, as `loss.gap_keeps` does)."""
    v = np.asarray([s[statistic] for s in stats], dtype=np.float64)
    return {f"{lim:g}": int((v > lim).sum()) for lim in limits}
