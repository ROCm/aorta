"""C4 batching (design Section 6.4, Batching): packing, the packed forward, log-probs at loss positions.

- Samples are sorted by length and packed first-fit-decreasing into micro-batches of at most
  `micro_batch_tokens`; a sample longer than that gets a micro-batch of its own.
- Padding-free packing. WP1 showed it is correct on the fast path only when every packed row carries
  restarting `position_ids` *and* `cu_seq_lens_q`/`_k` *and* `seq_idx`; dropping either signal leaks across
  sequences. So every row carries all three, a single-sequence row too.
- Full attention on a packed row. Under `sdpa`, transformers turns restarting position ids into an explicit
  block-diagonal mask, and SDPA with a mask cannot use its flash kernel (WP1). `PACKED_ATTENTION` instead runs one
  causal SDPA call per sequence, so each sequence's attention is the same call a solo forward makes. Its name is
  not a transformers mask type, so transformers builds no mask at all.
- Logits only at loss positions (WP1: a 19k sequence next to an engine at 45% fits only this way). The hidden
  states are gathered at the positions that predict response tokens, and the LM head runs on those in chunks,
  each under activation checkpointing, so at most one chunk of full-vocabulary logits is alive at a time.
- Masking. The loss positions are the final turn's response tokens, the stop token included when it was sampled;
  the prompt (system, earlier turns, feedback, assistant prefix) is context only (Section 6.4, Masking).
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from aorlta.trainer.loss import TrainSample, TrainerInputError

PACKED_ATTENTION = "aorlta_packed_sdpa"
DEFAULT_CHUNK_TOKENS = 2048


# --- packing ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MicroBatch:
    """One packed row. Loss token j predicts `targets[j]` from the hidden state at `loss_positions[j]`, and
    belongs to `samples[token_sample[j]]`."""

    samples: tuple[TrainSample, ...]
    input_ids: torch.Tensor  # [1, L] int64
    position_ids: torch.Tensor  # [1, L] int64, restarting at 0 for every sequence
    cu_seqlens: torch.Tensor  # [n + 1] int32
    seq_idx: torch.Tensor  # [1, L] int32: the sequence each token belongs to
    max_length: int
    loss_positions: torch.Tensor  # [T] int64
    targets: torch.Tensor  # [T] int64
    token_sample: torch.Tensor  # [T] int64

    @property
    def tokens(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def loss_tokens(self) -> int:
        return int(self.targets.shape[0])

    def sample_slices(self) -> list[slice]:
        """The loss tokens of each sample, in order: samples are laid out one after another."""
        out, start = [], 0
        for s in self.samples:
            out.append(slice(start, start + s.n_response))
            start += s.n_response
        return out

    def model_kwargs(self, device) -> dict:
        cu = self.cu_seqlens.to(device)
        return {"position_ids": self.position_ids.to(device), "cu_seq_lens_q": cu, "cu_seq_lens_k": cu,
                "max_length_q": self.max_length, "max_length_k": self.max_length,
                "seq_idx": self.seq_idx.to(device)}


def loss_span(sample: TrainSample) -> tuple[int, int]:
    """(first, end) of the token positions that carry loss: the final turn's response, nothing before it."""
    return len(sample.prompt_ids), len(sample.prompt_ids) + len(sample.response_ids)


def make_microbatch(samples: Sequence[TrainSample]) -> MicroBatch:
    if not samples:
        raise TrainerInputError("an empty micro-batch")
    ids, pos, sid, cu, lpos, tgt, tsam = [], [], [], [0], [], [], []
    offset = 0
    for j, s in enumerate(samples):
        seq = s.ids()
        ids.extend(seq)
        pos.extend(range(len(seq)))
        sid.extend([j] * len(seq))
        cu.append(cu[-1] + len(seq))
        first, end = loss_span(s)
        lpos.extend(range(offset + first - 1, offset + end - 1))
        tgt.extend(seq[first:end])
        tsam.extend([j] * (end - first))
        offset += len(seq)
    return MicroBatch(
        samples=tuple(samples),
        input_ids=torch.tensor([ids], dtype=torch.long),
        position_ids=torch.tensor([pos], dtype=torch.long),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32),
        seq_idx=torch.tensor([sid], dtype=torch.int32),
        max_length=max(s.length for s in samples),
        loss_positions=torch.tensor(lpos, dtype=torch.long),
        targets=torch.tensor(tgt, dtype=torch.long),
        token_sample=torch.tensor(tsam, dtype=torch.long),
    )


def pack(samples: Sequence[TrainSample], micro_batch_tokens: int) -> list[MicroBatch]:
    """First-fit-decreasing by length (ties by chain id, so the result is deterministic)."""
    if micro_batch_tokens < 1:
        raise TrainerInputError(f"micro_batch_tokens must be >= 1, got {micro_batch_tokens}")
    bins: list[list[TrainSample]] = []
    room: list[int] = []
    for s in sorted(samples, key=lambda s: (-s.length, s.chain_id)):
        for b, free in enumerate(room):
            if s.length <= free:
                bins[b].append(s)
                room[b] -= s.length
                break
        else:
            bins.append([s])
            room.append(micro_batch_tokens - s.length)  # negative when one sample alone overflows
    return [make_microbatch(b) for b in bins]


# --- the packed attention -----------------------------------------------------------------------------


def packed_sdpa_attention(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None, **kwargs):
    """Causal SDPA per packed sequence. query [B, H, L, D], key/value [B, Hkv, L, D]; returns [B, L, H, D]."""
    if attention_mask is not None:
        raise TrainerInputError("packed attention got an attention mask; packed rows carry no padding")
    cu = kwargs.get("cu_seq_lens_q")
    if cu is None:
        bounds = [(0, query.shape[2])]
    else:
        if query.shape[0] != 1:
            raise TrainerInputError("a packed row must be batch size 1")
        edges = cu.tolist()
        if edges[0] != 0 or edges[-1] != query.shape[2] or any(b <= a for a, b in zip(edges, edges[1:])):
            raise TrainerInputError(f"cu_seq_lens_q {edges[:4]}... does not tile the row of {query.shape[2]}")
        bounds = list(zip(edges[:-1], edges[1:]))
    outs = [F.scaled_dot_product_attention(query[:, :, a:b], key[:, :, a:b], value[:, :, a:b], dropout_p=dropout,
                                           is_causal=True, scale=scaling, enable_gqa=True) for a, b in bounds]
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    return out.transpose(1, 2).contiguous(), None


def register_packed_attention() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(PACKED_ATTENTION, packed_sdpa_attention)


def unwrap(model):
    """The transformers model under a PEFT wrapper (or the model itself)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def backbone_and_head(model):
    base = unwrap(model)
    return base.model, base.lm_head


def prepare_model(model, *, require_fast_path: bool = True, watcher=None, modeling=None,
                  grad_checkpointing: bool = False):
    """Refuse to train without the fast path (WP1's `kernels.ensure_fast_path`), switch full attention to
    `PACKED_ATTENTION` and read it back, and turn off the KV cache. Returns the fast-path report."""
    from aorlta.trainer import kernels

    base = unwrap(model)
    report = kernels.ensure_fast_path(require_fast_path, base, watcher=watcher, modeling=modeling)
    register_packed_attention()
    base.set_attn_implementation(PACKED_ATTENTION)
    text_cfg = getattr(base.config, "text_config", base.config)
    if text_cfg._attn_implementation != PACKED_ATTENTION:
        raise TrainerInputError(f"attention implementation is {text_cfg._attn_implementation!r} after setting "
                                f"{PACKED_ATTENTION!r}")
    base.config.use_cache = False
    if grad_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return report


# --- log-probs at loss positions -------------------------------------------------------------------


def _head_logits(head, h: torch.Tensor, fp32: bool) -> torch.Tensor:
    if not fp32:
        return head(h)
    w32 = getattr(head, "_aorlta_weight_fp32", None)
    if w32 is None or w32.device != head.weight.device:
        w32 = head.weight.detach().float()
        head._aorlta_weight_fp32 = w32  # the head is frozen (tied to the frozen embeddings)
    with torch.autocast(h.device.type, enabled=False):
        return F.linear(h.float(), w32)


def _chunk_logprobs(head, temperature: float, fp32: bool, h: torch.Tensor, targets: torch.Tensor):
    z = _head_logits(head, h, fp32).float()
    lp_t = torch.log_softmax(z / temperature, dim=-1).gather(-1, targets[:, None])[:, 0]
    with torch.no_grad():
        lp_raw = torch.log_softmax(z, dim=-1).gather(-1, targets[:, None])[:, 0]
    return lp_t, lp_raw


def target_logprobs(head, h: torch.Tensor, targets: torch.Tensor, *, temperature: float, fp32: bool = False,
                    chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> tuple[torch.Tensor, torch.Tensor]:
    """(log_softmax(z / T)[target], log_softmax(z)[target]) for hidden states h [T, d]; the first carries the
    gradient, the second (raw, temperature 1, as the engine reports it, A3) is detached."""
    lps, raws = [], []
    for a in range(0, h.shape[0], chunk_tokens):
        hc, tc = h[a:a + chunk_tokens], targets[a:a + chunk_tokens]
        if torch.is_grad_enabled() and hc.requires_grad:
            lp_t, lp_raw = checkpoint(_chunk_logprobs, head, temperature, fp32, hc, tc, use_reentrant=False)
        else:
            lp_t, lp_raw = _chunk_logprobs(head, temperature, fp32, hc, tc)
        lps.append(lp_t)
        raws.append(lp_raw)
    return torch.cat(lps), torch.cat(raws)


def amp_context(device, amp: bool):
    dev = torch.device(device).type
    return torch.autocast(dev, dtype=torch.bfloat16) if amp else contextlib.nullcontext()


def forward_microbatch(model, mb: MicroBatch, *, device, temperature: float, amp: bool = True,
                       lm_head_fp32: bool = False, chunk_tokens: int = DEFAULT_CHUNK_TOKENS):
    """Run one packed row; returns (lp_T, lp_raw, h_loss): per loss token, in `mb`'s order."""
    backbone, head = backbone_and_head(model)
    with amp_context(device, amp):
        hidden = backbone(input_ids=mb.input_ids.to(device), use_cache=False, **mb.model_kwargs(device))
        h = hidden.last_hidden_state[0].index_select(0, mb.loss_positions.to(device))
        lp_t, lp_raw = target_logprobs(head, h, mb.targets.to(device), temperature=temperature, fp32=lm_head_fp32,
                                       chunk_tokens=chunk_tokens)
    return lp_t, lp_raw, h


def head_logprobs(model, h: torch.Tensor, targets: torch.Tensor, *, device, temperature: float, amp: bool = True,
                  lm_head_fp32: bool = False) -> torch.Tensor:
    """log_softmax(z / T)[target] recomputed from hidden states, without the chunking or checkpointing: the
    trainer's independent check that the loss used T_loss."""
    _, head = backbone_and_head(model)
    with torch.no_grad(), amp_context(device, amp):
        z = _head_logits(head, h, lm_head_fp32).float()
        return torch.log_softmax(z / temperature, dim=-1).gather(-1, targets.to(device)[:, None])[:, 0]


def raw_logprobs_at(model, ids: Sequence[int], positions: Sequence[int], *, device, amp: bool = True,
                    lm_head_fp32: bool = False, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> torch.Tensor:
    """No-grad raw log-probs of ids[p] for each p in `positions` (p >= 1), one sequence on its own: what the
    engine's response log-probs are compared with, turn by turn (C4.T10)."""
    if not positions or min(positions) < 1 or max(positions) >= len(ids):
        raise TrainerInputError("positions must lie in [1, len(ids))")
    backbone, head = backbone_and_head(model)
    x = torch.tensor([list(ids)], dtype=torch.long, device=device)
    pos = torch.tensor(list(positions), dtype=torch.long, device=device)
    with torch.no_grad(), amp_context(device, amp):
        n = len(ids)
        cu = torch.tensor([0, n], dtype=torch.int32, device=device)
        hidden = backbone(input_ids=x, use_cache=False, position_ids=torch.arange(n, device=device)[None],
                          cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=n, max_length_k=n,
                          seq_idx=torch.zeros((1, n), dtype=torch.int32, device=device))
        h = hidden.last_hidden_state[0].index_select(0, pos - 1)
        _, raw = target_logprobs(head, h, x[0].index_select(0, pos), temperature=1.0, fp32=lm_head_fp32,
                                 chunk_tokens=chunk_tokens)
    return raw
