"""C4.T1 and C4.T7 on CPU, on the tiny Qwen3.5 in fp32 (Section 6.4's GPU tests, with the same pass limits).

C4.T1: the batched, packed, 4-rank gradient and loss equal the per-sample reference, for each aggregation.
C4.T7: streaming accumulation, groups added in a random order across 4 ranks, equals one batched step.

The ranks are 4 replicas on 4 threads talking through `ThreadWorld`, which implements the same all-reduce and
all-gather the trainer calls on torch.distributed (test_ranks.py runs the real gloo collective in processes).
"""

from __future__ import annotations

import random

import pytest
import torch

from c4_helpers import (
    NoMerge,
    SharedCollective,
    cosine,
    engine_logprobs_from,
    flat_grads,
    rel,
    run_ranks,
    synthetic_iteration,
    tiny_peft,
)
from aorlta.trainer.batching import forward_microbatch, make_microbatch
from aorlta.trainer.loss import LossSettings, build_groups, reference_loss
from aorlta.trainer.ranks import Assignment, LocalWorld, OnlineAssigner, OptimSettings, RankTrainer

AGGREGATIONS = ["sample_mean", "token_mean", "prompt_token_mean", "constant(32)", "sample_sum"]
SEED = 11
LOSS_REL = 1e-6  # Section 6.4, C4.T1
GRAD_COS = 0.99999


def _iteration(seed=SEED):
    model = tiny_peft(torch.float32, seed=SEED, b_std=0.05)
    samples, results = synthetic_iteration(groups=8, k=4, seed=seed)
    samples = engine_logprobs_from(model, samples)
    return build_groups(samples, results)


def _reference(groups, settings):
    model = tiny_peft(torch.float32, seed=SEED, b_std=0.05)
    model.train()

    def logprobs_of(s):
        lp, raw, _ = forward_microbatch(model, make_microbatch([s]), device="cpu",
                                        temperature=settings.loss_temperature, amp=False)
        return lp, raw

    loss = reference_loss(settings, groups, logprobs_of)
    return loss, flat_grads(_ordered(model))


def _ordered(model):
    from aorlta.trainer.checks import LoraTensors

    return LoraTensors.of(model).params()


def _trainers(n, settings, micro_batch_tokens):
    shared = SharedCollective(n)
    models = [tiny_peft(torch.float32, seed=SEED, b_std=0.05) for _ in range(n)]  # built in order: same weights
    # The constructor checks that every rank starts from the same latent (a collective), so it runs on the ranks.
    trainers = run_ranks([lambda r=r: RankTrainer(models[r], NoMerge(), settings, OptimSettings(lr=1e-3, grad_clip=0.0),
                                                  world=shared.world(r), micro_batch_tokens=micro_batch_tokens,
                                                  amp=False) for r in range(n)], shared)
    return shared, trainers


@pytest.mark.parametrize("agg", AGGREGATIONS)
def test_c4_t1_batched_packed_four_ranks_equal_the_per_sample_reference(agg):
    groups = _iteration()
    settings = LossSettings(aggregation=agg, loss_temperature=0.7)
    ref_loss, ref_grad = _reference(groups, settings)
    shared, trainers = _trainers(4, settings, micro_batch_tokens=96)
    reports = run_ranks([lambda t=t: t.train_iteration(groups) for t in trainers], shared)
    for t, rep in zip(trainers, reports):
        (step,) = rep
        assert step["ranks"] == 4
        assert rel(step["loss"], ref_loss) <= LOSS_REL, (step["loss"], ref_loss)
        got = t.last_flat_grad.double()
        assert cosine(got, ref_grad) >= GRAD_COS
        assert float((got - ref_grad).norm() / ref_grad.norm()) < 1e-5
    counts = reports[0][0]["counts"]  # every rank reports the all-gathered totals
    assert counts["microbatches"] < counts["samples"] == sum(len(g.samples) for g in groups)  # rows were packed
    assert counts["excluded"] == 0 and counts["compared"] == counts["samples"]


def test_a_gap_excluded_sample_contributes_no_gradient_and_stays_in_the_normaliser():
    model = tiny_peft(torch.float32, seed=SEED, b_std=0.05)
    samples, results = synthetic_iteration(groups=8, k=4, seed=SEED + 3)
    plans = build_groups(engine_logprobs_from(model, samples), results)
    victim = next(s.chain_id for g in plans for s in g.samples)
    groups = build_groups(engine_logprobs_from(model, samples, shift={victim: (0, 5.0)}), results)
    settings = LossSettings(loss_temperature=0.7, max_off_policy_fraction=0.1)
    t = RankTrainer(tiny_peft(torch.float32, seed=SEED, b_std=0.05), NoMerge(), settings,
                    OptimSettings(lr=1e-3, grad_clip=0.0), micro_batch_tokens=96, amp=False)
    (rep,) = t.train_iteration(groups)
    assert rep["excluded"] == [victim] and rep["totals"]["samples"] == sum(len(g.samples) for g in groups)
    ref_model = tiny_peft(torch.float32, seed=SEED, b_std=0.05)
    ref_model.train()

    def logprobs_of(s):
        lp, raw, _ = forward_microbatch(ref_model, make_microbatch([s]), device="cpu", temperature=0.7, amp=False)
        return lp, raw

    without = reference_loss(settings, groups, logprobs_of, keep=lambda s: s.chain_id != victim)
    g_without = flat_grads(_ordered(ref_model))
    assert rel(rep["loss"], without) <= LOSS_REL
    assert cosine(t.last_flat_grad, g_without) >= GRAD_COS
    for p in _ordered(ref_model):
        p.grad = None
    reference_loss(settings, groups, logprobs_of)
    assert cosine(t.last_flat_grad, flat_grads(_ordered(ref_model))) < 0.9999  # the victim's gradient is visible


def test_gradient_checkpointing_gives_the_same_step():
    from aorlta.trainer.batching import prepare_model, unwrap

    groups = _iteration(seed=SEED + 2)
    settings = LossSettings(loss_temperature=0.7)
    out = []
    for ckpt in (False, True):
        model = tiny_peft(torch.float32, seed=SEED, b_std=0.05)
        prepare_model(model, require_fast_path=False, grad_checkpointing=ckpt)
        assert unwrap(model).is_gradient_checkpointing == ckpt
        t = RankTrainer(model, NoMerge(), settings, OptimSettings(lr=1e-3, grad_clip=0.0), micro_batch_tokens=128,
                        amp=False)
        (rep,) = t.train_iteration(groups)
        out.append((rep["loss"], t.last_flat_grad))
    assert rel(out[1][0], out[0][0]) <= LOSS_REL
    assert cosine(out[1][1], out[0][1]) >= GRAD_COS


@pytest.mark.parametrize("agg", ["sample_mean", "prompt_token_mean", "token_mean"])
def test_c4_t7_streaming_in_random_group_order_equals_one_batched_step(agg):
    groups = _iteration(seed=SEED + 1)
    settings = LossSettings(aggregation=agg, loss_temperature=0.7)
    batched = RankTrainer(tiny_peft(torch.float32, seed=SEED, b_std=0.05), NoMerge(), settings,
                          OptimSettings(lr=1e-3, grad_clip=0.0), world=LocalWorld(), micro_batch_tokens=128,
                          amp=False)
    (b_rep,) = batched.train_iteration(groups)
    order = list(groups)
    random.Random(5).shuffle(order)
    shared, trainers = _trainers(4, settings, micro_batch_tokens=128)

    def stream(t):  # every rank runs the same online assignment over the same order: the broadcast group ids
        assigner, ledger = OnlineAssigner(4), []
        t.stream_begin()
        for g in order:
            if not g.samples:
                continue
            ledger.append(Assignment.of(len(ledger), g, assigner.assign(g)))
            if ledger[-1].rank == t.world.rank:
                t.stream_add(g)
        return t.stream_finish(ledger)

    reports = run_ranks([lambda t=t: stream(t) for t in trainers], shared)
    for t, rep in zip(trainers, reports):
        assert rel(rep["loss"], b_rep["loss"]) <= LOSS_REL
        assert cosine(t.last_flat_grad, batched.last_flat_grad) >= GRAD_COS
        assert rep["totals"] == b_rep["totals"]
    assert b_rep["totals"]["groups"] == sum(1 for g in groups if g.samples)
