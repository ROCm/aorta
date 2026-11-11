"""Packing, the packed forward and masking (C4.T4) on CPU."""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from c4_helpers import VOCAB, tiny_model
from aorlta.trainer import kernels
from aorlta.trainer.batching import (
    PACKED_ATTENTION,
    backbone_and_head,
    forward_microbatch,
    loss_span,
    make_microbatch,
    pack,
    packed_sdpa_attention,
    prepare_model,
    raw_logprobs_at,
)
from aorlta.trainer.loss import TrainSample, TrainerInputError

IM_START, IM_END, THINK, END_THINK, NL = 300, 301, 302, 303, 304  # tiny-vocabulary stand-ins for Appendix C ids


def _ts(cid, n_prompt, n_resp, seed=0, adv=1.0):
    rnd = random.Random(seed)
    return TrainSample(cid, (1, "standin:p"), tuple(rnd.randrange(VOCAB) for _ in range(n_prompt)),
                       tuple(rnd.randrange(VOCAB) for _ in range(n_resp)), tuple([-1.0] * n_resp), 0.0, adv)


# --- packing ----------------------------------------------------------------------------------------------


def test_pack_is_first_fit_decreasing_within_capacity():
    samples = [_ts(f"c{i}", n, 5, seed=i) for i, n in enumerate([20, 3, 50, 7, 31, 12, 44, 9])]
    mbs = pack(samples, 64)
    seen = [s.chain_id for mb in mbs for s in mb.samples]
    assert sorted(seen) == sorted(s.chain_id for s in samples)  # every sample exactly once
    assert all(mb.tokens <= 64 for mb in mbs)
    lengths = [s.length for s in mbs[0].samples]
    assert lengths[0] == max(s.length for s in samples)  # longest first
    assert pack(list(reversed(samples)), 64)[0].samples == mbs[0].samples  # deterministic in input order


def test_a_sample_longer_than_the_capacity_gets_its_own_microbatch():
    big, small = _ts("big", 90, 10), _ts("small", 5, 5)
    mbs = pack([small, big], 64)
    assert [[s.chain_id for s in mb.samples] for mb in mbs] == [["big"], ["small"]]
    with pytest.raises(TrainerInputError):
        pack([small], 0)


def test_pack_with_one_sequence_a_row_gives_every_sample_its_own_microbatch():
    samples = [_ts(f"c{i}", n, 5, seed=i) for i, n in enumerate([20, 3, 50, 7])]
    assert [[s.chain_id for s in mb.samples] for mb in pack(samples, 64)] == [["c2", "c1"], ["c0", "c3"]]
    assert [[s.chain_id for s in mb.samples] for mb in pack(samples, 64, max_sequences=1)] == [
        ["c2"], ["c0"], ["c3"], ["c1"]]
    with pytest.raises(TrainerInputError):
        pack(samples, 64, max_sequences=0)


def test_only_the_per_sequence_kernels_make_the_tiny_model_honour_packing():
    """The C4 tests' tiny model on CPU runs transformers' torch path for the linear attention unless
    `honour_packing` swaps in the per-sequence kernels."""
    assert kernels.honours_packing(tiny_model())
    assert not kernels.honours_packing(tiny_model(packing=False))


def test_microbatch_tensors_restart_positions_and_mark_boundaries():
    a, b = _ts("a", 6, 3, seed=1), _ts("b", 4, 2, seed=2)
    mb = make_microbatch([a, b])
    assert mb.position_ids[0].tolist() == list(range(9)) + list(range(6))
    assert mb.cu_seqlens.tolist() == [0, 9, 15] and mb.cu_seqlens.dtype == torch.int32
    assert mb.seq_idx[0].tolist() == [0] * 9 + [1] * 6 and mb.seq_idx.dtype == torch.int32
    assert mb.input_ids[0].tolist() == list(a.ids()) + list(b.ids())
    # loss positions predict exactly the responses: logits at p predict token p + 1
    assert mb.loss_positions.tolist() == [5, 6, 7, 9 + 3, 9 + 4]
    assert mb.targets.tolist() == list(a.response_ids) + list(b.response_ids)
    assert mb.token_sample.tolist() == [0, 0, 0, 1, 1]
    assert mb.max_length == 9
    assert [s for s in mb.sample_slices()] == [slice(0, 3), slice(3, 5)]
    kw = mb.model_kwargs("cpu")
    assert kw["cu_seq_lens_q"] is not None and kw["seq_idx"] is not None and kw["position_ids"] is not None


# --- the packed attention ---------------------------------------------------------------------------------


def test_packed_attention_is_sdpa_per_sequence():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 11, 8), torch.randn(1, 2, 11, 8), torch.randn(1, 2, 11, 8)
    cu = torch.tensor([0, 4, 11], dtype=torch.int32)
    out, _ = packed_sdpa_attention(None, q, k, v, None, scaling=0.3, cu_seq_lens_q=cu)
    for a, b in ((0, 4), (4, 11)):
        want = F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], is_causal=True, scale=0.3,
                                              enable_gqa=True).transpose(1, 2)
        assert torch.equal(out[:, a:b], want)
    with pytest.raises(TrainerInputError, match="mask"):
        packed_sdpa_attention(None, q, k, v, torch.ones(1, 1, 11, 11))
    with pytest.raises(TrainerInputError, match="tile"):
        packed_sdpa_attention(None, q, k, v, None, cu_seq_lens_q=torch.tensor([0, 4, 10], dtype=torch.int32))


def test_prepare_model_sets_the_packed_attention_and_refuses_without_the_fast_path():
    m = tiny_model()
    assert m.config.text_config._attn_implementation == PACKED_ATTENTION
    with pytest.raises(kernels.FastPathUnavailable):  # CPU: no fla, no causal-conv1d
        prepare_model(m, require_fast_path=True)


# --- the packed forward equals each sequence alone ------------------------------------------------------------


def test_packed_forward_equals_each_sequence_alone():
    m = tiny_model(torch.float32, seed=3)
    samples = [_ts("a", 37, 9, seed=1), _ts("b", 70, 21, seed=2), _ts("c", 5, 64, seed=3)]
    lp_p, raw_p, _ = forward_microbatch(m, make_microbatch(samples), device="cpu", temperature=0.7, amp=False)
    for s, sl in zip(samples, make_microbatch(samples).sample_slices()):
        lp_s, raw_s, _ = forward_microbatch(m, make_microbatch([s]), device="cpu", temperature=0.7, amp=False)
        assert torch.allclose(lp_p[sl], lp_s, atol=2e-5, rtol=0)
        assert torch.allclose(raw_p[sl], raw_s, atol=2e-5, rtol=0)
        ids = list(s.ids())
        alone = raw_logprobs_at(m, ids, list(range(len(s.prompt_ids), len(ids))), device="cpu", amp=False)
        assert torch.allclose(alone, raw_s, atol=2e-5, rtol=0)


def test_without_boundaries_a_packed_row_leaks_so_the_test_can_see_a_leak():
    m = tiny_model(torch.float32, seed=3)
    a, b = _ts("a", 37, 9, seed=1), _ts("b", 70, 21, seed=2)
    backbone, _ = backbone_and_head(m)
    row = make_microbatch([a, b])
    with torch.no_grad():
        h_pack = backbone(input_ids=row.input_ids, use_cache=False, **row.model_kwargs("cpu")).last_hidden_state[0]
        h_none = backbone(input_ids=row.input_ids, use_cache=False).last_hidden_state[0]
        h_b = backbone(input_ids=torch.tensor([list(b.ids())]), use_cache=False).last_hidden_state[0]
    n = a.length
    assert float((h_pack[n:] - h_b).abs().max()) < 1e-5
    assert float((h_none[n:] - h_b).abs().max()) > 1e-2


# --- C4.T4: masking ------------------------------------------------------------------------------------------


def _chain_layout():
    """A two-repair chain laid out as Appendix C builds it (thinking off), with each token's role:
    system/user0, the turn-0 assistant prefix, reply 0, the appended <|im_end|>, feedback 1 (user turn), prefix,
    reply 1, feedback 2, prefix, reply 2 (the final answer, ending in the sampled <|im_end|>)."""
    rnd = random.Random(7)
    toks, roles = [], []

    def add(ids, role):
        toks.extend(ids)
        roles.extend([role] * len(ids))

    def text(n):
        return [rnd.randrange(0, 290) for _ in range(n)]

    add([IM_START] + text(6) + [IM_END, NL, IM_START] + text(9) + [IM_END, NL], "context")
    add([IM_START] + text(1) + [NL, THINK, NL, NL, END_THINK, NL, NL], "prefix")
    add(text(12), "reply0")  # truncated: no stop token sampled
    add([IM_END], "appended_im_end")
    for t in (1, 2):
        add([NL, IM_START] + text(1) + [NL] + text(10) + [IM_END, NL], "feedback")
        add([IM_START] + text(1) + [NL, THINK, NL, NL, END_THINK, NL, NL], "prefix")
        if t == 1:
            add(text(8) + [IM_END], "reply1")
    final = text(14) + [IM_END]
    return toks, roles, final


def _chain_sample(adv: float = 1.0, *, vocab_mod: int | None = None):
    """The layout above as the trainer receives it: a TrainingSample and a ChainRecord, joined by build_groups
    (which also locates every turn's reply in the final sequence)."""
    from c4_helpers import RUN, _result
    from aorlta.ids import chain_id
    from aorlta.schemas import ChainRecord, TrainingSample, TurnRecord
    from aorlta.trainer.loss import build_groups

    toks, roles, final = _chain_layout()
    if vocab_mod:
        toks, final = [t % vocab_mod for t in toks], [t % vocab_mod for t in final]
    replies = {r: [t for t, x in zip(toks, roles) if x == r] for r in ("reply0", "reply1")}
    turns = []
    for t, name in enumerate(("reply0", "reply1", "final")):
        ids = replies[name] if name != "final" else final
        start = roles.index(name) if name != "final" else len(toks)
        turns.append(TurnRecord(t, 1, "0", start, ids, [-1.0] * len(ids), {"type": "stop"}, 0, 0, 0.0, 0.0, None,
                                None, "pass"))
    cid = chain_id(RUN, 1, "standin:m", 0)
    record = ChainRecord(cid, "standin:m", 0, 0, f"{RUN}:v0", turns, 2, "compiled")
    other = chain_id(RUN, 1, "standin:m", 1)
    samples = [TrainingSample(cid, 1, "standin:m", toks, final, [-1.0] * len(final)),
               TrainingSample(other, 1, "standin:m", toks[:5], final[:3], [-1.0] * 3)]
    results = [_result(cid, adv, "scored", None), _result(other, 0.0, "scored", None)]
    (group,) = build_groups(samples, results, chains={cid: record})
    s = next(x for x in group.samples if x.chain_id == cid)
    assert s.advantage == adv / 2 and [t.start for t in s.turns] == [roles.index("reply0"), roles.index("reply1"),
                                                                       len(toks)]
    return s, toks, roles, final


def test_c4_t4_the_loss_span_is_the_final_reply_only():
    s, toks, roles, final = _chain_sample()
    first, end = loss_span(s)
    assert (first, end) == (len(toks), len(toks) + len(final))
    mb = make_microbatch([s])
    assert mb.targets.tolist() == final  # the stop token the policy sampled included
    covered = set((mb.loss_positions + 1).tolist())  # the token positions that carry loss
    for p, role in enumerate(roles):
        assert p not in covered, f"{role} token at {p} carries loss"
    assert covered == set(range(len(toks), len(toks) + len(final)))


@pytest.mark.parametrize("adv", [1.0, -0.5])
def test_c4_t4_no_gradient_from_any_token_but_the_final_reply(adv):
    """dL/dh at the last layer is non-zero only where a loss term sits: there the final hidden state feeds the
    head and nothing else. Every earlier turn's token, the feedback included, gets exactly zero."""
    s, toks, roles, final = _chain_sample(adv, vocab_mod=300)
    m = tiny_model(torch.float32, seed=5)
    backbone, _ = backbone_and_head(m)
    mb = make_microbatch([s])
    captured = {}

    def hook(_mod, _args, out):
        out.last_hidden_state.retain_grad()
        captured["h"] = out.last_hidden_state

    handle = backbone.register_forward_hook(hook)
    try:
        for p in m.parameters():
            p.requires_grad_(True)
        lp, _, _ = forward_microbatch(m, mb, device="cpu", temperature=0.7, amp=False)
        (-(s.advantage) * lp).sum().backward()
    finally:
        handle.remove()
    g = captured["h"].grad[0].abs().sum(-1)
    # positions whose hidden state predicts a final-reply token: the last prompt token through the
    # second-to-last reply token
    expected = set(range(len(toks) - 1, len(toks) + len(final) - 1))
    for p, role in enumerate(roles + ["final"] * len(final)):
        if p in expected:
            assert float(g[p]) > 0, f"loss position {p} got no gradient"
        else:
            assert float(g[p]) == 0.0, f"{role} at {p} got gradient {float(g[p])}"
    assert roles.count("feedback") > 20


def test_a_rerendered_chain_leaves_out_an_earlier_reply_it_does_not_hold_and_an_appended_one_is_checked():
    """`repair_turn_ids: rerender` drops earlier think blocks and trims earlier replies, so an earlier reply is not
    in the final sequence as sampled. A rerendered chain leaves that turn out and keeps the rest, the final turn still
    checked as the sample's response; an appended chain with the same mismatch is refused."""
    from c4_helpers import RUN, _result
    from aorlta.ids import chain_id
    from aorlta.schemas import ChainRecord, TrainingSample, TurnRecord
    from aorlta.trainer.loss import TrainerInputError, build_groups

    toks, roles, final = _chain_layout()
    replies = {r: [t for t, x in zip(toks, roles) if x == r] for r in ("reply0", "reply1")}
    turns = [TurnRecord(t, 1, "0", roles.index(name) if name != "final" else len(toks),
                        replies[name] if name != "final" else final, [-1.0] * len(replies.get(name, final)),
                        {"type": "stop"}, 0, 0, 0.0, 0.0, None, None, "pass")
             for t, name in enumerate(("reply0", "reply1", "final"))]
    rendered = [t if r != "reply0" else (t + 1) % 290 for t, r in zip(toks, roles)]  # reply 0 as re-rendered
    cid = chain_id(RUN, 1, "standin:m", 0)
    chains = {cid: ChainRecord(cid, "standin:m", 0, 0, f"{RUN}:v0", turns, 2, "compiled")}
    other = chain_id(RUN, 1, "standin:m", 1)
    samples = [TrainingSample(cid, 1, "standin:m", rendered, final, [-1.0] * len(final)),
               TrainingSample(other, 1, "standin:m", rendered[:5], final[:3], [-1.0] * 3)]
    results = [_result(cid, 1.0, "scored", None), _result(other, 0.0, "scored", None)]
    with pytest.raises(TrainerInputError, match="turn 0's reply"):
        build_groups(samples, results, chains=chains)
    (group,) = build_groups(samples, results, chains=chains, rerendered=True)
    s = next(x for x in group.samples if x.chain_id == cid)
    assert [t.turn for t in s.turns] == [1, 2] and s.turns[-1].start == len(rendered)  # reply 1 still lines up
