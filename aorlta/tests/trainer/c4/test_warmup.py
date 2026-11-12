"""The start-up warm-up (`trainer.startup_warmup`): one streamed step's forward and backward on a synthetic
group shaped like a step's micro-batches, with the gradient dropped. The trainer must come out exactly as it went in:
the steps after a warm-up are bit for bit the steps an identical trainer takes without one, before any step (no Adam
moments yet) and between steps (with them)."""

from __future__ import annotations

import pytest
import torch

from c4_helpers import engine_logprobs_from, merged_forward, synthetic_iteration, tiny_peft
from aorlta.trainer.batching import pack
from aorlta.trainer.loss import LossSettings, TrainerInputError, build_groups
from aorlta.trainer.ranks import LocalWorld, OptimSettings, RankTrainer, warmup_group

MB = 256


def _trainer():
    model = tiny_peft(torch.bfloat16, seed=0, b_std=0.02)  # B != 0, so both LoRA factors get a gradient
    t = RankTrainer(model, merged_forward(model), LossSettings(loss_temperature=0.7),
                    OptimSettings(lr=1e-3, grad_clip=1.0), world=LocalWorld(), micro_batch_tokens=MB, amp=True,
                    gap_gate=False)
    return model, t


def _plans(model, seed: int, iteration: int):
    samples, results = synthetic_iteration(groups=4, k=4, seed=seed, iteration=iteration, unscored=0.0,
                                           quarantined=0.0, one_scored_group=False, scored_without_sample=0)
    return build_groups(engine_logprobs_from(model, samples, amp=True), results)


def _step(t: RankTrainer, plans) -> dict:
    t.stream_begin()
    for g in plans:
        t.stream_add(g)
    return t.stream_finish()


def _same_state(a: RankTrainer, b: RankTrainer) -> bool:
    sa, sb = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    moments = sorted(sa) == sorted(sb) and all(
        sorted(sa[i]) == sorted(sb[i]) and all(torch.equal(sa[i][k], sb[i][k]) for k in sa[i]) for i in sa)
    return (moments and a.steps_done == b.steps_done and a.merged.version == b.merged.version
            and all(torch.equal(p, q) for p, q in zip(a.params, b.params))
            and all(torch.equal(a.merged.weights[k], b.merged.weights[k]) for k in a.merged.weights))


@pytest.mark.parametrize("mb", [256, 300, 2048, 20480])
def test_the_warmup_group_packs_like_a_step(mb):
    """One packed row of three sequences whose length is not a multiple of 16, then one row alone whose length is."""
    pool = [[5, 6, 7], [8, 9]]
    g = warmup_group(pool, mb, max_new_tokens=4096)
    rows = pack(g.samples, mb)
    assert [len(r.samples) for r in rows] == [3, 1] and all(r.tokens <= mb for r in rows)
    assert rows[0].tokens % 16 != 0 and rows[1].tokens % 16 == 0
    assert len({s.chain_id for s in g.samples}) == 4 and g.dropped is None
    for s in g.samples:
        assert set(s.ids()) <= {5, 6, 7, 8, 9}  # only the given tokens
        assert len(s.prompt_ids) >= 1 and 1 <= s.n_response <= 4096 and len(s.engine_logprobs) == s.n_response
    with pytest.raises(TrainerInputError, match="needs token ids"):
        warmup_group([[], []], mb, 4096)


def test_a_warmup_reply_is_half_its_sample_and_at_most_max_new_tokens():
    full, capped = warmup_group([[5, 6, 7]], 20480, 4096), warmup_group([[5, 6, 7]], 20480, 512)
    assert [s.length for s in full.samples] == [s.length for s in capped.samples]
    assert [s.n_response for s in full.samples] == [min(4096, s.length // 2) for s in full.samples]
    assert [s.n_response for s in capped.samples] == [min(512, s.length // 2) for s in capped.samples]


def test_a_warmup_runs_a_steps_forward_and_backward_and_the_steps_after_it_are_as_without_it():
    ma, a = _trainer()
    _, b = _trainer()
    plans1, plans2 = _plans(ma, 100, 1), _plans(ma, 200, 2)
    group = warmup_group([s.ids() for g in plans1 for s in g.samples], MB, max_new_tokens=64)
    w = a.warmup(group)  # before any step: no Adam moments yet
    assert w["counts"]["microbatches"] == 2 and w["counts"]["samples"] == 4
    assert w["counts"]["forward_tokens"] == sum(s.length for s in group.samples)
    assert w["counts"]["loss_tokens"] == sum(s.n_response for s in group.samples)
    assert w["seconds"]["forward"] > 0 and w["seconds"]["backward"] > 0
    assert all(p.grad is None for p in a.params) and _same_state(a, b)
    r1 = (_step(a, plans1), _step(b, plans1))
    a.warmup(group)  # between steps, with Adam moments
    assert _same_state(a, b)
    r2 = (_step(a, plans2), _step(b, plans2))
    for ra, rb in (r1, r2):
        assert (ra["loss"], ra["grad_norm"], ra["counts"]) == (rb["loss"], rb["grad_norm"], rb["counts"])
    assert _same_state(a, b) and a.steps_done == 2


def test_a_warmup_refuses_an_open_streamed_step():
    _, t = _trainer()
    group = warmup_group([[5, 6, 7]], MB, 64)
    t.stream_begin()
    with pytest.raises(TrainerInputError, match="a streamed step is open"):
        t.warmup(group)


def test_a_warmup_runs_its_loss_on_every_token_with_the_gap_gate_enforced():
    """The warm-up group's engine log-probs are made up, so an enforced gap gate would take every sample out of the
    objective and the loss would see no token."""
    model = tiny_peft(torch.bfloat16, seed=0, b_std=0.02)
    t = RankTrainer(model, merged_forward(model), LossSettings(loss_temperature=0.7, max_generation_gap=1e-3),
                    OptimSettings(lr=1e-3, grad_clip=1.0), world=LocalWorld(), micro_batch_tokens=MB, amp=True)
    group = warmup_group([[5, 6, 7, 8]], MB, max_new_tokens=64)
    w = t.warmup(group)
    assert w["counts"]["excluded"] == 0
    assert w["counts"]["loss_tokens"] == sum(s.n_response for s in group.samples) > 0
    assert t.gap_gate  # enforced again for the real steps
