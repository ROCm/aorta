"""C4 over 4 ranks in 4 processes (`aorlta.trainer.rank_pool`): rank 0 in the test's process, ranks 1-3 spawned, the
gradient all-reduced over gloo. The tiny Qwen3.5 is saved once and every rank loads it by path, as the run's ranks
load Qwen3.5-4B.

- 4 ranks take the step 1 rank takes on the same data, with WP3's merged forward in bf16, over two iterations
  (loss within 1e-6, gradient cosine >= 0.99999, the same totals, every rank's latent and W_eff identical);
- C4.T1 and C4.T7 over 4 processes in fp32: the streamed step equals the per-sample reference and one batched step;
- the failure paths: ranks that start apart, a rank whose latent or W_eff drifts, a group that fails on a rank, a rank
  that dies, discard, and a checkpoint loaded on every rank.
"""

from __future__ import annotations

import random

import pytest
import torch

from c4_helpers import (
    TARGETS,
    NoMerge,
    cosine,
    engine_logprobs_from,
    load_tiny_bf16,
    load_tiny_bf16_apart,
    load_tiny_bf16_target_apart,
    load_tiny_fp32,
    rel,
    save_tiny_base,
    synthetic_iteration,
)
from aorlta.spans import SpanRecorder
from aorlta.trainer import checks as C
from aorlta.trainer.batching import forward_microbatch, make_microbatch
from aorlta.trainer.loss import LossSettings, build_groups, reference_loss
from aorlta.trainer.rank_pool import RankError, RankPool, RankSpec, file_init_method
from aorlta.trainer.ranks import LocalWorld, OptimSettings, RankTrainer, rank_agreement_cosine

LOSS_REL = 1e-6  # C4.T1's loss limit
GRAD_COS_BF16, GRAD_COS_FP32 = rank_agreement_cosine(torch.bfloat16), rank_agreement_cosine(torch.float32)
B_STD = 0.05


@pytest.fixture(scope="module")
def base_bf16(tmp_path_factory):
    return save_tiny_base(tmp_path_factory.mktemp("tiny-bf16"), dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def base_fp32(tmp_path_factory):
    return save_tiny_base(tmp_path_factory.mktemp("tiny-fp32"), dtype=torch.float32)


@pytest.fixture(autouse=True)
def two_threads():
    """Rank 0 runs in this process; give it the workers' thread count, so a group computes alike on any rank."""
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def b_drawn(model, keys, **kwargs):
    """fp32 tests' merged factory: the plain PEFT forward, with B drawn from N(0, B_STD) by a fixed generator (the
    same on every rank), so that A gets a gradient at the first step."""
    draw_b(model)
    return NoMerge()


def draw_b(model):
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for a, b in C.LoraTensors.of(model).modules.values():
            b.copy_(torch.randn(b.shape, generator=g) * B_STD)


def spec(base, ranks, *, fp32=False, agg="sample_mean", load=None, mb=96, lr=1e-3):
    return RankSpec(
        base_dir=str(base), devices=("cpu",) * ranks, memory_fractions=(None,) * ranks,
        loss=LossSettings(aggregation=agg, loss_temperature=0.7, max_generation_gap=50.0),
        optim=OptimSettings(lr=lr, grad_clip=0.0), micro_batch_tokens=mb,
        load_model=load or (load_tiny_fp32 if fp32 else load_tiny_bf16),
        merged_factory=b_drawn if fp32 else None, lora_r=4, lora_alpha=8, target_modules=TARGETS, seed=0,
        amp=not fp32, gap_gate=True, require_fast_path=False, backend="gloo",
        init_method=file_init_method(base.parent / "stores") if ranks > 1 else "", worker_threads=2)


def iteration(base, n, *, fp32=False, seed=11):
    """Iteration n's groups, with engine log-probs from the base itself (the policy at B = 0, or near it)."""
    model = (load_tiny_fp32 if fp32 else load_tiny_bf16)(base, "cpu")[0]
    samples, results = synthetic_iteration(groups=8, k=4, seed=seed + n, iteration=n)
    return [g for g in build_groups(engine_logprobs_from(model, samples, amp=not fp32), results) if g.samples]


def step(pool, n, plans, order_seed=None):
    pool.begin(n)
    order = list(plans)
    if order_seed is not None:
        random.Random(order_seed).shuffle(order)
    for g in order:
        pool.assign(g, {"prompt_id": g.key[1], "members": g.members})
    return pool.finish()


def latent(pool) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).double() for p in pool.trainer.params])


# --- 4 ranks against 1 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("agg", ["sample_mean", "token_mean", "prompt_token_mean"])
def test_four_ranks_take_the_step_one_rank_takes(base_bf16, agg):
    plans = {n: iteration(base_bf16, n) for n in (1, 2)}
    one = RankPool(spec(base_bf16, 1, agg=agg), call_timeout_s=120).start()
    four = RankPool(spec(base_bf16, 4, agg=agg), call_timeout_s=120).start()
    try:
        for n in (1, 2):  # step 1 moves only B (PEFT's B = 0); step 2 moves A too, on Adam's moments from step 1
            r1 = step(one, n, plans[n])
            g1 = one.trainer.last_flat_grad.clone()
            r4 = step(four, n, plans[n], order_seed=n)
            g4 = four.trainer.last_flat_grad.clone()
            assert r4["ranks"] == 4 and r1["ranks"] == 1
            assert rel(r4["loss"], r1["loss"]) <= LOSS_REL, (n, r4["loss"], r1["loss"])
            assert cosine(g4, g1) >= GRAD_COS_BF16 and float((g4 - g1).norm() / g1.norm()) < 1e-5
            assert r4["totals"] == r1["totals"] and r4["normaliser"] == r1["normaliser"]
            once = r4["checks"]["groups_trained_once"]
            assert once["passed"] and once["groups"] == len(plans[n]) and min(once["per_rank"]) >= 1, once
            assert sum(r["groups"] for r in r4["by_rank"]) == len(plans[n])
            rep = r4["checks"]["replicas_identical"]
            assert rep["passed"] and rep["latent"] and rep["w_eff"] and rep["w_eff_compared"], rep
            assert all(c["passed"] for k, c in r4["checks"].items() if k in C.GATING_PRE + C.GATING_POST)
        sums = four.weights_checksums()
        assert all(s == sums[0] for s in sums)  # every rank's forward runs on rank 0's W_eff, bit for bit
        # The latents differ by summation order only, which Adam amplifies where a gradient element is near zero.
        a, b = latent(four), latent(one)
        assert cosine(a, b) >= 0.9999 and float((a - b).norm() / b.norm()) < 1e-3
        w1, w4 = one.merged.weights, four.merged.weights
        differ = sum(int((w1[k].view(torch.int16) != w4[k].view(torch.int16)).sum()) for k in w1)
        assert differ <= 1e-3 * sum(w.numel() for w in w1.values()), differ
    finally:
        one.shutdown()
        four.shutdown()


# --- C4.T1 and C4.T7 over 4 processes -------------------------------------------------------------------------


def _reference_model(base):
    from peft import LoraConfig, get_peft_model

    from aorlta.trainer.batching import prepare_model

    m = load_tiny_fp32(base, "cpu")[0]
    for p in m.parameters():
        p.requires_grad_(False)
    prepare_model(m, require_fast_path=False)
    torch.manual_seed(0)  # the pool's spec.seed: the same A as every rank
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS, bias="none"))
    draw_b(m)
    return m


@pytest.mark.parametrize("agg", ["sample_mean", "token_mean", "prompt_token_mean", "constant(32)"])
def test_c4_t1_four_rank_streaming_equals_the_per_sample_reference(base_fp32, agg):
    plans = iteration(base_fp32, 1, fp32=True)
    settings = LossSettings(aggregation=agg, loss_temperature=0.7, max_generation_gap=50.0)
    ref = _reference_model(base_fp32)
    ref.train()

    def logprobs_of(s):
        lp, raw, _ = forward_microbatch(ref, make_microbatch([s]), device="cpu", temperature=0.7, amp=False)
        return lp, raw

    ref_loss = reference_loss(settings, plans, logprobs_of)
    ref_grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double()
                          for p in C.LoraTensors.of(ref).params()])
    pool = RankPool(spec(base_fp32, 4, fp32=True, agg=agg), call_timeout_s=120).start()
    try:
        rep = step(pool, 1, plans, order_seed=3)
        got = pool.trainer.last_flat_grad.double()
        assert rel(rep["loss"], ref_loss) <= LOSS_REL, (rep["loss"], ref_loss)
        assert cosine(got, ref_grad) >= GRAD_COS_FP32 and float((got - ref_grad).norm() / ref_grad.norm()) < 1e-5
        assert rep["counts"]["microbatches"] < rep["counts"]["samples"]  # rows were packed
        assert rep["checks"]["groups_trained_once"]["passed"]
    finally:
        pool.shutdown()


@pytest.mark.parametrize("agg", ["sample_mean", "token_mean"])
def test_c4_t7_four_ranks_in_random_group_order_equal_one_batched_step(base_fp32, agg):
    plans = iteration(base_fp32, 1, fp32=True, seed=23)
    settings = LossSettings(aggregation=agg, loss_temperature=0.7, max_generation_gap=50.0)
    batched = RankTrainer(_reference_model(base_fp32), NoMerge(), settings, OptimSettings(lr=1e-3, grad_clip=0.0),
                          world=LocalWorld(), micro_batch_tokens=96, amp=False)
    (b_rep,) = batched.train_iteration(plans)
    pool = RankPool(spec(base_fp32, 4, fp32=True, agg=agg), call_timeout_s=120).start()
    try:
        rep = step(pool, 1, plans, order_seed=11)
        assert rel(rep["loss"], b_rep["loss"]) <= LOSS_REL
        assert cosine(pool.trainer.last_flat_grad, batched.last_flat_grad) >= GRAD_COS_FP32
        assert rep["totals"] == b_rep["totals"]
        assert cosine(latent(pool), torch.cat([p.detach().reshape(-1).double() for p in batched.params])) >= GRAD_COS_FP32
    finally:
        pool.shutdown()


# --- failures ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["control", "target"])
def test_ranks_that_start_apart_are_refused(base_bf16, where):
    """Rank 2's base differs in one element: of embed_tokens, which the base control samples, or of a LoRA target the
    control leaves out, which only the W_eff comparison at the start sees."""
    load = load_tiny_bf16_apart if where == "control" else load_tiny_bf16_target_apart
    pool = RankPool(spec(base_bf16, 4, load=load), call_timeout_s=120)
    message = "same LoRA latent and base weights" if where == "control" else "same merged weights W_eff"
    with pytest.raises(Exception, match=message):
        pool.start()
    assert pool.remote == []  # every worker was stopped


def test_an_assignment_a_worker_did_not_receive_in_order_stops_the_step(base_bf16):
    """A lost or reordered broadcast: the worker refuses the assignment at once, and the drain stops on it."""
    plans = iteration(base_bf16, 1)
    pool = RankPool(spec(base_bf16, 4), call_timeout_s=120).start()
    try:
        pool.begin(1)
        pool.assign(plans[0])
        pool.remote[2].send("assign", (5, (1, "standin:x"), 3, 1, 1), None)  # rank 3 sees seq 5 after seq 0
        with pytest.raises(RankError, match="rank 3, assign: .*assignment 5 arrived after 1 others"):
            pool.finish()
    finally:
        pool.shutdown()


@pytest.mark.parametrize("what", ["latent", "w_eff"])
def test_a_rank_that_drifts_apart_fails_the_step(base_bf16, what):
    plans = iteration(base_bf16, 1)
    pool = RankPool(spec(base_bf16, 4), call_timeout_s=120).start()
    try:
        pool.tamper(2, what)  # after the start check, before the step
        with pytest.raises(C.TrainerCheckFailed) as exc:
            step(pool, 1, plans)
        assert "replicas_identical" in exc.value.failed
        assert exc.value.report["checks"]["replicas_identical"][what if what == "latent" else "w_eff"] is False
    finally:
        pool.shutdown()


def test_a_group_that_fails_on_a_worker_stops_the_step_before_any_collective(base_bf16):
    import dataclasses

    plans = iteration(base_bf16, 1)
    bad = dataclasses.replace(plans[1], samples=tuple(dataclasses.replace(s, response_ids=(10 ** 6,) * s.n_response)
                                                      for s in plans[1].samples))  # ids beyond the vocabulary
    pool = RankPool(spec(base_bf16, 4), call_timeout_s=120).start()
    try:
        pool.begin(1)
        assert pool.assign(plans[0]) == 0
        assert pool.assign(bad) == 1  # the next group goes to the first rank with nothing queued
        with pytest.raises(RankError, match="rank 1, group 1"):
            pool.finish()
        pool.discard()
        assert step(pool, 1, plans)["checks"]["groups_trained_once"]["passed"]  # and the pool is usable again
    finally:
        pool.shutdown()


def test_a_rank_that_dies_stops_the_step_at_the_drain(base_bf16):
    plans = iteration(base_bf16, 1)
    pool = RankPool(spec(base_bf16, 4), call_timeout_s=120).start()
    try:
        pool.begin(1)
        pool.remote[1].proc.kill()  # rank 2
        pool.remote[1].proc.join(30)
        with pytest.raises(RankError, match="rank 2"):
            for g in plans:
                pool.assign(g)
            pool.finish()
        with pytest.raises(RankError, match="rank 2"):
            pool.begin(2)
    finally:
        pool.shutdown()


def test_each_group_is_one_span_on_the_rank_that_trained_it_and_discard_drops_the_step(base_bf16):
    spans = SpanRecorder()
    plans = iteration(base_bf16, 1)
    pool = RankPool(spec(base_bf16, 4), spans=spans, call_timeout_s=120).start()
    try:
        before = [p.detach().clone() for p in pool.trainer.params]
        pool.begin(1)
        for g in plans:
            pool.assign(g, {"prompt_id": g.key[1], "members": g.members})
        pool.discard()  # nothing stepped, on any rank
        assert all(torch.equal(a, b) for a, b in zip(before, pool.trainer.params))
        t0 = spans.now()
        rep = step(pool, 1, plans)
        owner = {a.key: a.rank for a in pool.ledger}
        got = [s for s in spans.spans(1) if s.name == "train.stream" and s.t_start >= t0]
        assert sorted(s.attrs["prompt_id"] for s in got) == sorted(g.key[1] for g in plans)
        assert all(s.attrs["rank"] == owner[(1, s.attrs["prompt_id"])] for s in got)
        assert {s.attrs["rank"] for s in got} == {0, 1, 2, 3}
        assert all(s.t_start <= s.t_end <= spans.now() for s in got)
        assert all(s.attrs["forward_s"] > 0 and s.attrs["samples"] > 0 for s in got)
        assert rep["drain_sec"] >= 0
    finally:
        pool.shutdown()


def test_stopping_rank_0_drops_its_queued_groups_and_finishes_only_the_one_in_hand():
    """A pool shut down mid-step (the run is stopping) must not leave rank 0's thread training its queue past the
    process's exit: a stand-in run once hung at exit that way."""
    import time

    from aorlta.trainer.rank_pool import _LocalRank

    class Slow:
        seconds = {"forward": 0.0, "backward": 0.0}
        trained: list = []

        def stream_add(self, plan):
            time.sleep(0.5)
            self.trained.append(plan)

    done = []
    rank = _LocalRank(Slow(), "cpu", lambda *a: done.append(a))
    for seq in range(10):
        rank.put(seq, f"group-{seq}")
    time.sleep(0.1)  # the first group is in hand
    t0 = time.monotonic()
    rank.stop(timeout_s=30)
    assert time.monotonic() - t0 < 2.0 and not rank._thread.is_alive()
    assert Slow.trained == ["group-0"] and len(done) == 10  # every group reported: one trained, nine dropped
    assert [d[2] is None for d in done] == [False] + [True] * 9


def test_a_checkpoint_loads_on_every_rank_and_re_merges_to_the_same_weights(base_bf16, tmp_path):
    from aorlta.trainer.rank_pool import save_state

    plans = iteration(base_bf16, 1)
    first = RankPool(spec(base_bf16, 4), call_timeout_s=120).start()
    try:
        step(first, 1, plans)
        save_state(first.trainer, tmp_path)
        digest, sums = first.state_digest(), first.weights_checksums()
    finally:
        first.shutdown()
    again = RankPool(spec(base_bf16, 4), call_timeout_s=120).start()
    try:
        assert again.load_state(tmp_path) == digest
        again.remerge(1)
        assert again.weights_checksums() == sums  # every rank re-merged to the W_eff the first pool served
        assert again.state_digest() == digest
        again.tamper(3, "latent")
        with pytest.raises(RankError, match="trained states differ"):
            again.state_digest()
    finally:
        again.shutdown()
