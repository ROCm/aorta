"""Ranks: group assignment, mini-batches, pi_old for later mini-batches, and the real all-reduce over 4 gloo
processes (the hand-written all-reduce of Section 6.4, Ranks)."""

from __future__ import annotations

import pytest
import torch

from c4_helpers import cosine, engine_logprobs_from, gloo_iteration, gloo_worker, rel, synthetic_iteration, tiny_peft
from c4_helpers import NoMerge
from aorlta.trainer import checks as C
from aorlta.trainer import ranks as R
from aorlta.trainer.loss import GroupPlan, LossSettings, TrainSample, TrainerInputError, build_groups
from aorlta.trainer.ranks import (
    LocalWorld,
    OnlineAssigner,
    OptimSettings,
    RankTrainer,
    assign_groups,
    clip_coefficient,
    rank_agreement_cosine,
    ranks_agree,
    split_minibatches,
)


def _g(name, lengths):
    samples = tuple(TrainSample(f"{name}/{i}", (1, name), (1,) * (n - 2), (2, 3), (-1.0, -1.0), 0.0, 0.0)
                    for i, n in enumerate(lengths))
    return GroupPlan((1, name), len(lengths), len(lengths), 0.0, samples, None)


def test_assign_groups_balances_tokens_and_is_deterministic():
    groups = [_g(f"g{i}", [n] * 4) for i, n in enumerate([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])]
    out = assign_groups(groups, 4)
    assert sorted(g.key for r in out for g in r) == sorted(g.key for g in groups)
    loads = [sum(R.group_tokens(g) for g in r) for r in out]
    assert max(loads) - min(loads) <= 4 * 30
    assert assign_groups(list(reversed(groups)), 4) == out
    assert assign_groups([GroupPlan((1, "x"), 1, 1, None, (), "fewer_than_2_scored")], 2) == [[], []]


def test_online_assignment_goes_to_the_least_loaded_rank():
    a = OnlineAssigner(3)
    assert [a.assign(_g(f"g{i}", [n])) for i, n in enumerate([50, 10, 10, 10, 30])] == [0, 1, 2, 1, 2]


def test_online_assignment_prefers_the_rank_with_the_least_work_still_queued():
    """A rank that has trained what it was given is idle, however much it was given: it takes the next group before
    a rank that is still busy, even one that was given fewer tokens."""
    a = OnlineAssigner(3)
    g = [_g(f"g{i}", [n]) for i, n in enumerate([50, 10, 10, 10])]
    assert [a.assign(x) for x in g[:3]] == [0, 1, 2]
    a.done(0, R.group_tokens(g[0]))  # rank 0 finished its 50 tokens; ranks 1 and 2 still hold 10 each
    assert a.assign(g[3]) == 0
    assert (a.queued, a.load) == ([10, 10, 10], [60, 10, 10])


# --- the assignment ledger: every group trained exactly once, on its rank -------------------------------------


def _ledger_plans():
    model = tiny_peft(torch.float32, seed=5, b_std=0.05)
    samples, results = synthetic_iteration(groups=4, k=4, seed=5, unscored=0.0, quarantined=0.0,
                                           one_scored_group=False, scored_without_sample=0)
    return [g for g in build_groups(engine_logprobs_from(model, samples), results) if g.samples]


def _two_rank_step(plans, trained_by, ledger_of):
    """Two thread ranks: rank r adds `trained_by[r]` and finishes with `ledger_of(r)`."""
    from c4_helpers import SharedCollective, run_ranks

    shared = SharedCollective(2)
    models = [tiny_peft(torch.float32, seed=5, b_std=0.05) for _ in range(2)]
    trainers = run_ranks([lambda r=r: RankTrainer(models[r], NoMerge(), LossSettings(loss_temperature=0.7),
                                                  OptimSettings(lr=1e-3, grad_clip=0.0), world=shared.world(r),
                                                  micro_batch_tokens=128, amp=False) for r in range(2)], shared)

    def rank(r):
        t = trainers[r]
        t.stream_begin()
        for g in trained_by[r]:
            t.stream_add(g)
        return t.stream_finish(ledger_of(r))

    return run_ranks([lambda r=r: rank(r) for r in range(2)], shared)


def _ledger(plans, ranks):
    return [R.Assignment.of(i, g, r) for i, (g, r) in enumerate(zip(plans, ranks))]


def test_a_step_whose_ranks_trained_exactly_the_ledger_passes():
    plans = _ledger_plans()
    led = _ledger(plans, [0, 1, 0, 1])
    reps = _two_rank_step(plans, [[plans[0], plans[2]], [plans[1], plans[3]]], lambda r: led)
    c = reps[0]["checks"]["groups_trained_once"]
    assert c["passed"] and c["per_rank"] == [2, 2] and c["missing"] == c["extra"] == []
    assert c["totals"] == c["ledger_totals"] == reps[0]["totals"]


@pytest.mark.parametrize("case", ["dropped", "trained_twice", "wrong_rank", "ledgers_differ", "seq_gap", "key_twice"])
def test_a_step_is_refused_unless_every_ledger_group_was_trained_once_on_its_rank(case):
    plans = _ledger_plans()
    led = _ledger(plans, [0, 1, 0, 1])
    trained = [[plans[0], plans[2]], [plans[1], plans[3]]]
    ledger_of = lambda r: led  # noqa: E731
    if case == "dropped":
        trained[1] = [plans[1]]  # the ledger's group 3 was trained by nobody
    elif case == "trained_twice":
        trained[0] = [plans[0], plans[2], plans[3]]  # group 3 by both ranks
    elif case == "wrong_rank":
        trained = [[plans[0]], [plans[1], plans[2], plans[3]]]  # group 2 by rank 1, not rank 0
    elif case == "ledgers_differ":
        other = _ledger(plans, [0, 1, 1, 0])
        ledger_of = lambda r: led if r == 0 else other  # noqa: E731
    elif case == "seq_gap":
        ledger_of = lambda r: [a._replace(seq=a.seq * 2) for a in led]  # noqa: E731
    elif case == "key_twice":
        twice = led + [led[0]._replace(seq=4)]
        trained[0] = [plans[0], plans[2], plans[0]]
        ledger_of = lambda r: twice  # noqa: E731
    with pytest.raises(C.StepRefused) as exc:
        _two_rank_step(plans, trained, ledger_of)
    assert exc.value.failed == ["groups_trained_once"]


def test_a_streamed_step_over_several_ranks_needs_its_ledger():
    from c4_helpers import SharedCollective

    t = RankTrainer(tiny_peft(torch.float32, seed=5), NoMerge(), LossSettings(loss_temperature=0.7), OptimSettings(),
                    micro_batch_tokens=128, amp=False)
    t.world = SharedCollective(2).world(0)  # a second rank, after the start check
    t.stream_begin()
    with pytest.raises(TrainerInputError, match="needs the assignment ledger"):
        t.stream_finish()


def test_the_weights_checksum_sees_one_bit():
    from c4_helpers import merged_forward

    model = tiny_peft(torch.bfloat16, seed=0)
    merged = merged_forward(model)
    a = R.weights_checksum(merged)
    assert a == R.weights_checksum(merged) and len(a) == len(merged.weights)
    key = sorted(merged.weights)[3]
    w = merged._weights[key]
    w.view(torch.int16).view(-1)[w.numel() // 2] ^= 1
    b = R.weights_checksum(merged)
    assert [x[0] for x in a if x != b[a.index(x)]] == [key]  # that tensor's checksum, and only it, moved
    assert R.weights_checksum(NoMerge()) is None


def test_split_minibatches_is_contiguous_and_balanced_in_samples():
    groups = [_g(f"g{i}", [10] * k) for i, k in enumerate([4, 4, 4, 4, 2, 6])]
    m = split_minibatches(groups, 2)
    assert [g.key for x in m for g in x] == [g.key for g in groups]
    assert [sum(len(g.samples) for g in x) for x in m] == [12, 12]
    with pytest.raises(TrainerInputError):
        split_minibatches(groups[:1], 2)


def test_clip_coefficient_is_megatrons():
    assert clip_coefficient(0.5, 1.0) == 1.0
    assert clip_coefficient(4.0, 1.0) == pytest.approx(1.0 / (4.0 + 1e-6))
    assert clip_coefficient(4.0, 0.0) == 1.0


def test_streaming_needs_one_step_per_iteration():
    model = tiny_peft(torch.float32, seed=0)
    t = RankTrainer(model, NoMerge(), LossSettings(loss_temperature=0.7, steps_per_iteration=2), OptimSettings())
    with pytest.raises(TrainerInputError, match="steps_per_iteration 1"):
        t.stream_begin()


def test_two_steps_per_iteration_ratio_against_the_sampling_weights(monkeypatch):
    """With steps_per_iteration 2, step 2's pi_old is the pre-step-1 policy, so its ratio is not 1."""
    model = tiny_peft(torch.float32, seed=4, b_std=0.05)
    samples, results = synthetic_iteration(groups=6, k=4, seed=4, unscored=0.0, quarantined=0.0,
                                           one_scored_group=False, scored_without_sample=0)
    groups = build_groups(engine_logprobs_from(model, samples), results)
    t = RankTrainer(model, NoMerge(), LossSettings(loss_temperature=0.7, steps_per_iteration=2),
                    OptimSettings(lr=5e-2, grad_clip=0.0), micro_batch_tokens=4096, amp=False)
    seen = []
    real = R.token_loss

    def spy(settings, lp, old_lp, adv, **kw):
        seen.append((t.steps_done, (lp.detach() - old_lp).abs().max().item()))
        return real(settings, lp, old_lp, adv, **kw)

    monkeypatch.setattr(R, "token_loss", spy)
    reps = t.train_iteration(groups)
    assert [r["step"] for r in reps] == [1, 2]
    assert sum(r["totals"]["samples"] for r in reps) == sum(len(g.samples) for g in groups)
    step1 = [d for s, d in seen if s == 0]
    step2 = [d for s, d in seen if s == 1]
    assert step1 and max(step1) == 0.0  # old = lp.detach(): ratio exactly 1
    assert step2 and max(step2) > 1e-4  # the weights moved, pi_old did not


def test_the_old_logprobs_of_step_2_are_the_initial_policys():
    model = tiny_peft(torch.float32, seed=4, b_std=0.05)
    samples, results = synthetic_iteration(groups=6, k=4, seed=4, unscored=0.0, quarantined=0.0,
                                           one_scored_group=False, scored_without_sample=0)
    groups = build_groups(engine_logprobs_from(model, samples), results)
    t = RankTrainer(model, NoMerge(), LossSettings(loss_temperature=0.7, steps_per_iteration=2),
                    OptimSettings(lr=5e-2, grad_clip=0.0), micro_batch_tokens=4096, amp=False)
    old = t._old_logprobs(split_minibatches(groups, 2)[1])
    fresh = tiny_peft(torch.float32, seed=4, b_std=0.05)
    from aorlta.trainer.batching import forward_microbatch, make_microbatch

    for g in split_minibatches(groups, 2)[1]:
        for s in g.samples:
            with torch.no_grad():
                lp, raw, _ = forward_microbatch(fresh, make_microbatch([s]), device="cpu", temperature=0.7,
                                                amp=False)
            assert torch.allclose(old[s.chain_id][0], lp, atol=2e-5) and torch.allclose(old[s.chain_id][1], raw,
                                                                                          atol=2e-5)


# --- the real collective -------------------------------------------------------------------------------------


@pytest.mark.parametrize("agg", ["sample_mean", "token_mean"])
def test_four_gloo_ranks_equal_one_rank_and_stay_identical(tmp_path, agg):
    import torch.multiprocessing as mp

    store = tmp_path / "store"
    mp.spawn(gloo_worker, args=(4, str(store), str(tmp_path), agg), nprocs=4, join=True)
    ranks = [torch.load(tmp_path / f"rank{r}.pt") for r in range(4)]
    single = gloo_iteration(agg)
    assert single["ranks"] == 1 and all(r["ranks"] == 4 for r in ranks)
    for r in ranks:
        assert all(r["checks"][n] for n in C.GATING_PRE + C.GATING_POST)
        assert torch.equal(r["grad"], ranks[0]["grad"])  # the all-reduce gives every rank the same sum
        assert all(C.bit_identical(a, b) for a, b in zip(r["params"], ranks[0]["params"]))
        assert rel(r["loss"], single["loss"]) <= 1e-6
        assert cosine(r["grad"], single["grad"]) >= rank_agreement_cosine(torch.float32)


# --- the agreement limit, R ranks against one -----------------------------------------------------------------


def test_the_rank_agreement_limit_is_c4_t1s_in_fp32_and_0_99998_in_bf16():
    """C4.T1's 0.99999 was set for fp32; in bf16 the limit is 0.99998. A torch dtype and its name give the same limit,
    and a dtype with no agreed limit is refused rather than given one."""
    assert rank_agreement_cosine(torch.float32) == rank_agreement_cosine("float32") == 0.99999
    assert rank_agreement_cosine(torch.bfloat16) == rank_agreement_cosine("bfloat16") == 0.99998
    for dtype in (torch.float16, torch.float64, "int8"):
        with pytest.raises(ValueError, match="no rank agreement limit"):
            rank_agreement_cosine(dtype)


@pytest.mark.parametrize("cos,bf16,fp32", [(1.0, True, True), (0.99999, True, True), (0.999985, True, False),
                                           (0.99998, True, False), (0.9999799, False, False), (0.99987, False, False)])
def test_a_bf16_gradient_agrees_at_0_99998_and_an_fp32_one_only_at_0_99999(cos, bf16, fp32):
    assert ranks_agree(cos, torch.bfloat16) is bf16
    assert ranks_agree(cos, torch.float32) is fp32
