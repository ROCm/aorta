"""C4's objective on CPU, without a model: advantages (C4.T6), unscored samples (C4.T5), the normalisers of every
aggregation against their definitions, the three policy losses, TIS, KL and the loss temperature."""

from __future__ import annotations

import math

import pytest
import torch

from c4_helpers import RUN, _result, synthetic_iteration
from aorlta.config import TrainerSection
from aorlta.ids import chain_id
from aorlta.schemas import TrainingSample
from aorlta.trainer import batching
from aorlta.trainer.batching import target_logprobs
from aorlta.trainer.loss import (
    LossSettings,
    StepTotals,
    TrainerInputError,
    build_groups,
    gap_keeps,
    kl_k3,
    normaliser,
    pg_token_loss,
    sample_coefficient,
    sample_gap,
    tis_weights,
    token_loss,
)


def _sample(pid: str, k: int, n_resp: int = 3, iteration: int = 1, n_prompt: int = 4) -> TrainingSample:
    return TrainingSample(chain_id(RUN, iteration, pid, k), iteration, pid, list(range(n_prompt)),
                          list(range(10, 10 + n_resp)), [-0.5] * n_resp)


def _group(pid, rewards, n_resp=None):
    """One group: rewards[k] is a float (scored), None (unscored) or 'q' (quarantined)."""
    samples, results = [], []
    for k, r in enumerate(rewards):
        samples.append(_sample(pid, k, (n_resp or [3] * len(rewards))[k]))
        cid = samples[-1].chain_id
        if r is None:
            results.append(_result(cid, None, "unscored", "sanitizer_not_settled"))
        elif r == "q":
            results.append(_result(cid, None, "quarantined", "leak_check"))
        else:
            results.append(_result(cid, float(r), "scored", None))
    return samples, results


# --- C4.T6: advantages without std normalisation ----------------------------------------------------------


def test_c4_t6_advantages_have_no_std_division():
    s, r = _group("standin:a", [1.0, 0.0, 0.0, -1.0])
    (g,) = build_groups(s, r)
    assert [x.advantage for x in g.samples] == [1.0, 0.0, 0.0, -1.0]
    assert g.mean_reward == 0.0


def test_advantage_is_reward_minus_the_scored_mean():
    s, r = _group("standin:b", [0.5, -0.25, 1.0])
    (g,) = build_groups(s, r)
    mean = (0.5 - 0.25 + 1.0) / 3
    assert [x.advantage for x in g.samples] == [0.5 - mean, -0.25 - mean, 1.0 - mean]
    assert math.fsum(x.advantage for x in g.samples) == pytest.approx(0.0, abs=1e-15)


# --- C4.T5: unscored samples, and groups with one scored sample ------------------------------------------


def test_c4_t5_an_unscored_sample_changes_neither_the_mean_nor_any_normaliser():
    base_s, base_r = _group("standin:c", [1.0, 0.0, -1.0], n_resp=[3, 5, 7])
    with_s, with_r = _group("standin:c", [1.0, 0.0, -1.0, None, "q"], n_resp=[3, 5, 7, 11, 13])
    (g0,) = build_groups(base_s, base_r)
    (g1,) = build_groups(with_s, with_r)
    assert g1.mean_reward == g0.mean_reward
    assert [x.advantage for x in g1.samples] == [x.advantage for x in g0.samples]
    assert [x.chain_id for x in g1.samples] == [x.chain_id for x in g0.samples]
    assert StepTotals.of([g1]) == StepTotals.of([g0]) == StepTotals(3, 15, 1)
    assert g1.excluded == {"unscored": 1, "quarantined": 1}
    for agg in ("sample_mean", "token_mean", "prompt_token_mean", "constant(4096)", "sample_sum"):
        st = LossSettings(aggregation=agg)
        assert normaliser(st, StepTotals.of([g1])) == normaliser(st, StepTotals.of([g0]))
        assert [sample_coefficient(st, x, g1.n_response_tokens) for x in g1.samples] == \
               [sample_coefficient(st, x, g0.n_response_tokens) for x in g0.samples]


def test_c4_t5_a_group_with_one_scored_sample_is_dropped():
    s1, r1 = _group("standin:d", [0.7, None, None, "q"])
    s2, r2 = _group("standin:e", [1.0, -1.0])
    groups = build_groups(s1 + s2, r1 + r2)
    d = next(g for g in groups if g.key == (1, "standin:d"))
    assert d.samples == () and d.dropped == "fewer_than_2_scored" and d.scored == 1
    assert StepTotals.of(groups) == StepTotals(2, 6, 1)


def test_a_sample_without_a_result_is_unscored_not_a_default_reward():
    s, r = _group("standin:f", [1.0, -1.0, 0.0])
    groups = build_groups(s, r[:2])  # the third sample's result never arrived
    (g,) = groups
    assert g.excluded == {"no_reward_result": 1}
    assert [x.chain_id for x in g.samples] == [s[0].chain_id, s[1].chain_id]
    assert g.mean_reward == 0.0


def test_a_scored_chain_without_a_sample_counts_in_the_mean_but_is_not_trained():
    s, r = _group("standin:g", [1.0, 0.0, -0.5])
    (g,) = build_groups(s[:2], r)  # chain 2 is scored but has no TrainingSample (logprob_error)
    assert g.mean_reward == pytest.approx(0.5 / 3)
    assert len(g.samples) == 2 and g.excluded == {"scored_without_sample": 1}
    assert StepTotals.of([g]).samples == 2


def test_contradictory_results_stop_the_run():
    s, r = _group("standin:h", [1.0, 0.0])
    bad = [r[0], _result(r[1].candidate_id, None, "scored", None)]
    with pytest.raises(TrainerInputError, match="scored"):
        build_groups(s, bad)
    with pytest.raises(TrainerInputError, match="outside|not a number"):
        build_groups(s, [r[0], _result(r[1].candidate_id, 1.5, "scored", None)])
    with pytest.raises(TrainerInputError, match="unscored"):
        build_groups(s, [r[0], _result(r[1].candidate_id, 0.3, "unscored", "x")])
    with pytest.raises(TrainerInputError, match="two"):
        build_groups(s + [s[0]], r)
    with pytest.raises(TrainerInputError, match="two"):
        build_groups(s, r + [r[0]])
    with pytest.raises(TrainerInputError, match="group key"):
        build_groups([TrainingSample(s[0].chain_id, 1, "standin:other", [1], [2], [-1.0])], r[:1])
    with pytest.raises(TrainerInputError, match="empty response"):
        build_groups([TrainingSample(s[0].chain_id, 1, "standin:h", [1], [], [])], r[:1])


def test_groups_from_a_synthetic_iteration():
    s, r = synthetic_iteration(groups=8, k=4, seed=3)
    groups = build_groups(s, r)
    for g in groups:
        scored = [x for x in r if x.status == "scored" and x.candidate_id.split("/")[2] == g.key[1]]
        if len(scored) >= 2:
            mean = math.fsum(x.reward for x in scored) / len(scored)
            assert g.mean_reward == mean
            for x in g.samples:
                assert x.advantage == x.reward - mean
        else:
            assert g.samples == ()


# --- the normalisers, against their definitions ----------------------------------------------------------


def _two_groups():
    s1, r1 = _group("standin:x", [1.0, 0.0, -1.0], n_resp=[2, 4, 6])
    s2, r2 = _group("standin:y", [0.5, -0.5], n_resp=[10, 30])
    return build_groups(s1 + s2, r1 + r2)


@pytest.mark.parametrize("agg", ["sample_mean", "token_mean", "prompt_token_mean", "constant(64)", "sample_sum"])
def test_every_aggregation_matches_its_definition(agg):
    groups = _two_groups()
    st = LossSettings(aggregation=agg)
    d = normaliser(st, StepTotals.of(groups))
    got = {s.chain_id: sample_coefficient(st, s, g.n_response_tokens) / d for g in groups for s in g.samples}
    n_samples, n_tokens, n_groups = 5, 52, 2
    want = {}
    for g in groups:
        for s in g.samples:
            want[s.chain_id] = {
                "sample_mean": 1 / (n_samples * s.n_response),
                "token_mean": 1 / n_tokens,
                "prompt_token_mean": 1 / (n_groups * g.n_response_tokens),
                "constant(64)": 1 / (n_samples * 64),
                "sample_sum": 1 / n_samples,
            }[agg]
    assert got == pytest.approx(want, rel=1e-15)


def test_an_empty_step_has_no_normaliser():
    with pytest.raises(TrainerInputError, match="nothing to train"):
        normaliser(LossSettings(), StepTotals(0, 0, 0))


# --- settings --------------------------------------------------------------------------------------------


def test_settings_come_from_the_trainer_section():
    t = TrainerSection(pg_loss="cispo", aggregation="constant(4096)", loss_temperature=0.7, tis=2.0, kl_beta=1e-3)
    st = LossSettings.from_trainer(t)
    assert (st.pg_loss, st.aggregation_kind(), st.loss_temperature, st.tis, st.kl_beta) == \
           ("cispo", ("constant", 4096), 0.7, 2.0, 1e-3)
    with pytest.raises(TrainerInputError, match="loss_temperature is unset"):
        LossSettings.from_trainer(TrainerSection())


def test_engine_logprobs_cannot_be_pi_old_at_a_loss_temperature_other_than_one():
    with pytest.raises(TrainerInputError, match="temperature 1"):
        LossSettings(old_logprobs="engine", loss_temperature=0.7)
    LossSettings(old_logprobs="engine", loss_temperature=1.0)


@pytest.mark.parametrize("bad", [dict(pg_loss="dapo"), dict(aggregation="constant(0)"), dict(aggregation="mean"),
                                 dict(tis=0.0), dict(kl_beta=-1.0), dict(steps_per_iteration=0),
                                 dict(eps_clip_low=1.5), dict(max_generation_gap=float("nan")),
                                 dict(min_scored_per_group=1), dict(loss_temperature=0.0)])
def test_bad_settings_are_refused(bad):
    with pytest.raises(TrainerInputError):
        LossSettings(**bad)


# --- the policy-gradient losses ----------------------------------------------------------------------------


@pytest.mark.parametrize("pg", ["ppo_clip", "reinforce", "cispo"])
def test_every_pg_loss_gives_minus_a_dlp_when_old_equals_new(pg):
    st = LossSettings(pg_loss=pg)
    lp = torch.tensor([-0.3, -1.2, -2.0, -0.01], requires_grad=True)
    adv = torch.tensor([1.0, -0.5, 0.25, 2.0])
    loss, aux = pg_token_loss(st, lp, lp.detach(), adv)
    loss.sum().backward()
    assert torch.equal(lp.grad, -adv)
    assert aux.sum() == 0


def test_ppo_clip_is_miles_compute_policy_loss():
    st = LossSettings(pg_loss="ppo_clip", eps_clip_low=0.2, eps_clip_high=0.28)
    old = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    lp = old + torch.log(torch.tensor([1.5, 0.5, 1.1, 0.7]))
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss, clipped = pg_token_loss(st, lp, old, adv)
    ratio = torch.tensor([1.5, 0.5, 1.1, 0.7])
    want = torch.maximum(-ratio * adv, -ratio.clamp(0.8, 1.28) * adv)
    assert torch.allclose(loss, want, atol=1e-6)
    assert clipped.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_cispo_weights_reinforce_by_a_clipped_detached_ratio():
    st = LossSettings(pg_loss="cispo", eps_clip_low=1.0, eps_clip_high=3.0)
    old = torch.zeros(3)
    lp = torch.log(torch.tensor([0.5, 2.0, 6.0])).requires_grad_(True)
    adv = torch.tensor([1.0, 1.0, 1.0])
    loss, clipped = pg_token_loss(st, lp, old, adv)
    loss.sum().backward()
    assert torch.allclose(lp.grad, -torch.tensor([0.5, 2.0, 4.0]), atol=1e-6)  # capped at 1 + 3
    assert clipped.tolist() == [0.0, 0.0, 1.0]


def test_tis_and_kl():
    st = LossSettings(tis=2.0, kl_beta=0.1)
    old = torch.tensor([-1.0, -1.0, -2.0])
    eng = torch.tensor([-1.5, -0.2, -2.0])
    w = tis_weights(st, old, eng)
    assert torch.allclose(w, torch.tensor([min(math.exp(0.5), 2.0), math.exp(-0.8), 1.0]))
    lp = torch.tensor([-1.0, -0.5, -3.0], requires_grad=True)
    ref = torch.tensor([-1.0, -1.5, -2.0])
    k3 = kl_k3(lp, ref)
    r = ref - lp.detach()
    assert torch.allclose(k3, torch.exp(r) - r - 1) and bool((k3 >= 0).all())
    loss, aux = token_loss(st, lp, lp.detach(), torch.ones(3), tis_w=w, ref_lp=ref)
    assert torch.allclose(loss, -w * 1.0 + 0.1 * k3)
    with pytest.raises(TrainerInputError):
        token_loss(st, lp, lp.detach(), torch.ones(3), ref_lp=ref)
    with pytest.raises(TrainerInputError):
        token_loss(st, lp, lp.detach(), torch.ones(3), tis_w=w)


# --- the loss temperature ------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_loss_logprobs_are_at_the_loss_temperature_and_gap_logprobs_raw(chunk):
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 50, bias=False)
    h = torch.randn(7, 8, requires_grad=True)
    tgt = torch.randint(0, 50, (7,))
    lp, raw = target_logprobs(head, h, tgt, temperature=0.7, chunk_tokens=chunk)
    z = head(h)
    want = torch.log_softmax(z / 0.7, -1).gather(-1, tgt[:, None])[:, 0]
    assert torch.allclose(lp, want, atol=1e-6)
    assert torch.allclose(raw, torch.log_softmax(z, -1).gather(-1, tgt[:, None])[:, 0], atol=1e-6)
    assert not raw.requires_grad and not torch.allclose(lp, raw)
    g1 = torch.autograd.grad(lp.sum(), h)[0]
    g2 = torch.autograd.grad(want.sum(), h)[0]
    assert torch.allclose(g1, g2, atol=1e-6)


def test_the_fp32_head_is_an_fp32_matmul():
    torch.manual_seed(1)
    head = torch.nn.Linear(8, 50, bias=False).to(torch.bfloat16)
    h = torch.randn(5, 8).to(torch.bfloat16)
    tgt = torch.randint(0, 50, (5,))
    _, raw = target_logprobs(head, h, tgt, temperature=1.0, fp32=True)
    z = h.float() @ head.weight.float().t()
    assert torch.equal(raw, torch.log_softmax(z, -1).gather(-1, tgt[:, None])[:, 0])


@pytest.mark.parametrize("weights", [torch.bfloat16, torch.float32])
def test_a_bf16_head_on_cpu_is_an_fp32_gemm_with_the_bf16_gemms_result(weights, monkeypatch):
    """PyTorch's bf16 GEMM on a CPU without AVX-512 is slow, so the CPU head multiplies the bf16-rounded inputs
    in fp32 and rounds the result to bf16: the bf16 GEMM's arithmetic (fp32 accumulation), up to summation order."""
    torch.manual_seed(2)
    head = torch.nn.Linear(64, 4096, bias=False).to(weights)
    h0 = torch.randn(33, 64).to(weights)
    tgt = torch.randint(0, 4096, (33,))
    seen = []

    def spy(x, w, b=None):  # what the GEMM gets: fp32 tensors holding bf16 values
        seen.append((x.dtype, w.dtype, all(torch.equal(t, t.bfloat16().float()) for t in (x, w))))
        return linear(x, w, b)

    linear = torch.nn.functional.linear
    monkeypatch.setattr(batching.F, "linear", spy)
    h = h0.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = batching._head_logits(head, h, False)
        _, raw = target_logprobs(head, h, tgt, temperature=1.0)
    assert seen and set(seen) == {(torch.float32, torch.float32, True)} and z.dtype == torch.bfloat16
    monkeypatch.undo()
    ref_h = h0.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref = head(ref_h)
    assert (z != ref).float().mean() < 1e-3
    torch.testing.assert_close(z.float(), ref.float(), rtol=2 ** -7, atol=1e-4)
    want = torch.log_softmax(ref.float(), -1).gather(-1, tgt[:, None])[:, 0]
    torch.testing.assert_close(raw, want, rtol=0, atol=2e-2)
    grad = torch.autograd.grad(torch.log_softmax(z.float(), -1).sum(), h)[0]
    ref_grad = torch.autograd.grad(torch.log_softmax(ref.float(), -1).sum(), ref_h)[0]
    torch.testing.assert_close(grad.float(), ref_grad.float(), rtol=2 ** -6, atol=1e-3)


def test_an_fp32_head_on_cpu_without_autocast_is_the_head_itself():
    torch.manual_seed(3)
    head = torch.nn.Linear(8, 50, bias=False)
    h = torch.randn(5, 8)
    assert torch.equal(batching._head_logits(head, h, False), head(h))


# --- the gap -------------------------------------------------------------------------------------------------


def test_sample_gap_statistics_and_the_gate():
    st = LossSettings(max_generation_gap=2.0)
    g = sample_gap("c", torch.tensor([-1.0, -2.0, -0.5, -3.0]), [-1.0, -1.5, -0.5, -0.5])
    assert (g.tokens, g.max_abs, g.mean_abs, g.frac_gt_1) == (4, 2.5, 0.75, 0.25)
    assert g.mean_signed == pytest.approx(-0.75)
    assert not gap_keeps(st, g)
    assert gap_keeps(st, sample_gap("c", torch.tensor([-1.0, -2.0]), [-1.0, -0.5]))
    assert not gap_keeps(st, None)
    with pytest.raises(TrainerInputError):
        sample_gap("c", torch.tensor([-1.0]), [-1.0, -2.0])
