"""The step checks (C4.T8 on CPU) and the gap gate, on the tiny Qwen3.5 in bf16 with the merged-weight forward.

The merged-weight forward is WP3's (`aorlta.trainer.merged_lora.MergedLoRA`).
"""

from __future__ import annotations

import pytest
import torch

from c4_helpers import (
    NoMerge, SharedCollective, engine_logprobs_from, merged_forward, run_ranks, synthetic_iteration, tiny_peft,
)
from aorlta.sync.merge import merge_weight
from aorlta.trainer import checks as C
from aorlta.trainer import ranks as R
from aorlta.trainer.loss import LossSettings, TrainerInputError, build_groups
from aorlta.trainer.ranks import LocalWorld, OptimSettings, RankTrainer


def _parts(*, seed=0, b_std=0.0, groups=6, k=4, shift=None, merged_kw=None):
    model = tiny_peft(torch.bfloat16, seed=seed, b_std=b_std)
    merged = merged_forward(model, **(merged_kw or {}))
    samples, results = synthetic_iteration(groups=groups, k=k, seed=seed + 100, unscored=0.0, quarantined=0.0,
                                           one_scored_group=False, scored_without_sample=0)
    samples = engine_logprobs_from(model, samples, amp=True, shift=shift)
    return model, merged, build_groups(samples, results)


def _trainer(model, merged, *, world=None, settings=None, wd=0.0, lr=1e-3):
    st = settings or LossSettings(loss_temperature=0.7)
    return RankTrainer(model, merged, st, OptimSettings(lr=lr, grad_clip=1.0, weight_decay=wd),
                       world=world or LocalWorld(), micro_batch_tokens=256, amp=True)


def _setup(*, seed=0, b_std=0.0, groups=6, k=4, shift=None, settings=None, wd=0.0, lr=1e-3, merged_kw=None):
    model, merged, plans = _parts(seed=seed, b_std=b_std, groups=groups, k=k, shift=shift, merged_kw=merged_kw)
    return model, merged, plans, _trainer(model, merged, settings=settings, wd=wd, lr=lr)


def _ranks(n, perturb=None):
    """n replicas built in order (the same weights), their trainers constructed on the rank threads."""
    shared = SharedCollective(n)
    parts = [_parts() for _ in range(n)]
    if perturb:
        perturb(parts)
    return shared, parts


def _swap_two(t):
    """Swap the first entry of `t` with the first one that differs from it: every moment of `t` is kept."""
    flat = t.detach().view(-1)
    j = int((flat != flat[0]).nonzero()[0])
    with torch.no_grad():
        flat[0], flat[j] = flat[j].clone(), flat[0].clone()


def test_lora_latent_is_fp32_and_only_it_trains():
    model, merged, plans, t = _setup()
    assert {p.dtype for p in t.params} == {torch.float32}
    assert len(t.params) == 2 * len(t.lora.modules) == 2 * 31
    assert C.structural_problems(model, t.lora, t.optimizer) == []


# --- C4.T8 ---------------------------------------------------------------------------------------------------


def test_c4_t8_a_unchanged_at_step_1_is_accepted_and_every_b_moves():
    model, merged, plans, t = _setup()
    a_before = {m: a.detach().clone() for m, (a, _) in t.lora.modules.items()}
    assert all(not bool(b.ne(0).any()) for _, b in t.lora.modules.values())  # PEFT's B = 0 init
    (rep,) = t.train_iteration(plans)
    moved = rep["checks"]["every_trained_tensor_moved"]
    assert moved["passed"] and moved["a_expected_unchanged"] == 31 and moved["unmoved"] == 0
    for m, (a, b) in t.lora.modules.items():
        assert C.bit_identical(a, a_before[m])  # dL/dA = 0 exactly while B = 0
        assert bool(b.ne(0).any())
    assert all(rep["checks"][n]["passed"] for n in C.GATING_PRE + C.GATING_POST)
    (rep2,) = t.train_iteration(plans)  # step 2: B != 0, so every A moves too
    m2 = rep2["checks"]["every_trained_tensor_moved"]
    assert m2["passed"] and m2["a_expected_unchanged"] == 0
    assert all(not C.bit_identical(a, a_before[m]) for m, (a, _) in t.lora.modules.items())


def test_c4_t8_a_planted_non_moving_b_is_caught():
    model, merged, plans, t = _setup()
    name, (_, b) = next(iter(t.lora.modules.items()))
    b.register_hook(lambda g: torch.zeros_like(g))  # this B never gets a gradient
    with pytest.raises(C.TrainerCheckFailed) as exc:
        t.train_iteration(plans)
    assert exc.value.failed == ["every_trained_tensor_moved"]
    assert exc.value.report["checks"]["every_trained_tensor_moved"]["unmoved_examples"] == [f"{name}.lora_B"]


@pytest.mark.parametrize("which", ["embed_tokens", "target"])
def test_c4_t8_a_planted_base_tensor_change_is_caught(which):
    """The change is planted inside the optimizer step, as a frozen tensor reaching the optimizer would be."""
    model, merged, plans, t = _setup()
    key = next(k for k in t.control.tensors if (which == "embed_tokens") == k.endswith("embed_tokens.weight"))
    step = t.optimizer.step

    def step_and_touch(*a, **kw):
        out = step(*a, **kw)
        with torch.no_grad():
            t.control.tensors[key].view(-1)[7] += 1.0
        return out

    t.optimizer.step = step_and_touch
    with pytest.raises(C.TrainerCheckFailed) as exc:
        t.train_iteration(plans)
    assert exc.value.failed == ["frozen_control_unchanged"]
    assert exc.value.report["checks"]["frozen_control_unchanged"]["moved"] == [key]


def test_c4_t8_an_a_permuted_while_its_b_was_zero_is_caught():
    model, merged, plans, t = _setup()
    name, (a, _) = next(iter(t.lora.modules.items()))
    step = t.optimizer.step

    def step_and_permute(*args, **kw):
        out = step(*args, **kw)
        _swap_two(a)
        return out

    t.optimizer.step = step_and_permute
    with pytest.raises(C.TrainerCheckFailed) as exc:
        t.train_iteration(plans)
    assert exc.value.failed == ["every_trained_tensor_moved"]
    assert exc.value.report["checks"]["every_trained_tensor_moved"]["a_moved_while_b_was_zero"] == [f"{name}.lora_A"]


def test_a_base_change_before_the_forward_is_caught_by_the_gap_gate_first():
    """embed_tokens is also the (tied) LM head: changed before the forward, it moves the trainer's log-probs off
    the engine's, and the gap gate refuses the step before any weight changes."""
    model, merged, plans, t = _setup()
    key = next(k for k in t.control.tensors if k.endswith("embed_tokens.weight"))
    tok = plans[0].samples[0].response_ids[0]
    with torch.no_grad():
        t.control.tensors[key][tok] += 3.0
    with pytest.raises(C.StepRefused) as exc:
        t.train_iteration(plans)
    assert exc.value.failed == ["off_policy_fraction"]


def test_the_control_is_embed_tokens_and_8_targets_spanning_depth():
    model, merged, plans, t = _setup()
    keys = sorted(t.control.tensors)
    assert sum(k.endswith("embed_tokens.weight") for k in keys) == 1
    targets = [k for k in keys if k.endswith("base_layer.weight")]
    assert len(targets) == 8 and len({k.split(".layers.")[1].split(".")[0] for k in targets}) >= 3


def test_a_with_zero_b_is_not_checked_under_weight_decay():
    model, merged, plans, t = _setup(wd=0.1)
    (rep,) = t.train_iteration(plans)
    m = rep["checks"]["every_trained_tensor_moved"]
    assert m["passed"] and m["a_not_checked"] == 31 and m["a_expected_unchanged"] == 0


# --- W_eff after the step --------------------------------------------------------------------------------------


def test_the_step_refreshes_w_eff_to_the_merge_of_the_new_latent():
    model, merged, plans, t = _setup(b_std=0.02)
    before = merged.weights
    (rep,) = t.train_iteration(plans)
    assert merged.version == 1 and rep["checks"]["w_eff_refreshed"]["passed"]
    for name, layer in merged.layers.items():
        want = merge_weight(layer.base_layer.weight, layer.lora_A["default"].weight, layer.lora_B["default"].weight,
                            float(layer.scaling["default"]))
        assert C.bit_identical(merged.weights[name], want)
    assert any(not torch.equal(before[n], merged.weights[n]) for n in before)


def test_the_step_refreshes_w_eff_to_the_stochastic_merge_of_the_new_latent():
    """`sync.stochastic_rounding` on: the refreshed W_eff is the stochastic merge for version 1 (the bits the
    writer gets), not the nearest-rounded one."""
    model, merged, plans, t = _setup(b_std=0.02, merged_kw=dict(rounding="stochastic", run_seed=7))
    (rep,) = t.train_iteration(plans)
    assert merged.version == 1 and rep["checks"]["w_eff_refreshed"]["passed"]
    assert merged.last_stats.rounding == "stochastic"
    differs_from_nearest = 0
    for name, layer in merged.layers.items():
        args = (layer.base_layer.weight, layer.lora_A["default"].weight, layer.lora_B["default"].weight,
                float(layer.scaling["default"]), name)
        assert C.bit_identical(merged.weights[name], merge_weight(*args, rounding="stochastic", run_seed=7, version=1))
        differs_from_nearest += not C.bit_identical(merged.weights[name], merge_weight(*args))
    assert differs_from_nearest == len(merged.layers) == 31


def test_a_step_without_the_refresh_is_caught(monkeypatch):
    model, merged, plans, t = _setup(b_std=0.02)
    monkeypatch.setattr(merged, "refresh", lambda: None)
    with pytest.raises(C.TrainerCheckFailed) as exc:
        t.train_iteration(plans)
    assert exc.value.failed == ["w_eff_refreshed"]


# --- the gap gate ---------------------------------------------------------------------------------------------


def _first_chain(plans):
    return next(s.chain_id for g in plans for s in g.samples)


def test_a_sample_over_the_gap_bound_is_excluded_and_too_many_refuse_the_step():
    model, _, plans, _ = _setup()
    ids = [s.chain_id for g in plans for s in g.samples]
    shift = {ids[0]: (0, 5.0), ids[5]: (1, 5.0)}  # 2 of 24 samples, 8% > 2%
    model, merged, plans, t = _setup(shift=shift)
    before = [p.detach().clone() for p in t.params]
    with pytest.raises(C.StepRefused) as exc:
        t.train_iteration(plans)
    assert exc.value.failed == ["off_policy_fraction"]
    off = exc.value.report["checks"]["off_policy_fraction"]
    assert (off["compared"], off["excluded"]) == (24, 2)
    assert sorted(exc.value.report["excluded"]) == sorted([ids[0], ids[5]])
    assert all(C.bit_identical(p, q) for p, q in zip(t.params, before))  # refused: nothing stepped


def test_an_excluded_sample_loses_its_gradient_but_keeps_its_counts():
    model, _, plans, _ = _setup(groups=13, k=4)
    first = _first_chain(plans)  # 1 of 52 samples, under 2%
    model, merged, plans, t = _setup(groups=13, k=4, shift={first: (0, 5.0)})
    (rep,) = t.train_iteration(plans)
    assert rep["counts"]["excluded"] == 1 and rep["excluded"] == [first]
    assert rep["totals"]["samples"] == 52  # still in the normaliser (Section 6.4; module docstring of loss.py)
    assert rep["checks"]["off_policy_fraction"]["passed"]


def test_missing_engine_logprobs_exclude_a_sample():
    from aorlta.schemas import TrainingSample

    model, merged, plans, t = _setup()
    g0 = plans[0]
    s = g0.samples[0]
    bad = type(s)(**{**s.__dict__, "engine_logprobs": s.engine_logprobs[:-1]})
    plans = [type(g0)(**{**g0.__dict__, "samples": (bad,) + g0.samples[1:]})] + plans[1:]
    with pytest.raises(C.StepRefused) as exc:  # 1 of 24 > 2%
        t.train_iteration(plans)
    assert exc.value.report["counts"]["logprob_error"] == 1
    del TrainingSample


def test_a_non_finite_gradient_refuses_the_step():
    model, merged, plans, t = _setup()
    _, b = next(iter(t.lora.modules.values()))
    b.register_hook(lambda g: torch.full_like(g, float("nan")))
    with pytest.raises(C.StepRefused) as exc:
        t.train_iteration(plans)
    assert "gradient_is_finite_and_nonzero" in exc.value.failed


# --- the loss temperature ------------------------------------------------------------------------------------


def test_the_temperature_check_passes_and_sees_the_raw_logprobs_differ():
    model, merged, plans, t = _setup()
    (rep,) = t.train_iteration(plans)
    tc = rep["checks"]["loss_temperature_applied"]
    assert tc["passed"] and tc["mismatched"] == 0 and tc["compared"] > 0 and tc["max_abs_vs_raw"] > 0


def test_a_loss_at_temperature_1_when_0_7_is_configured_refuses_the_step(monkeypatch):
    model, merged, plans, t = _setup()
    real = R.forward_microbatch

    def at_t1(model, mb, **kw):
        return real(model, mb, **dict(kw, temperature=1.0))

    monkeypatch.setattr(R, "forward_microbatch", at_t1)
    with pytest.raises(C.StepRefused) as exc:
        t.train_iteration(plans)
    assert exc.value.failed == ["loss_temperature_applied"]


# --- replicas, structure ---------------------------------------------------------------------------------------


def _first_a(model):
    return next(iter(C.LoraTensors.of(model).modules.values()))[0]


def test_replicas_that_drift_apart_during_the_run_are_caught():
    shared, parts = _ranks(2)

    def rank(r):
        model, merged, plans = parts[r]
        t = _trainer(model, merged, world=shared.world(r))
        if r == 1:
            with torch.no_grad():
                _first_a(model).view(-1)[0] += 1e-3  # rank 1's A drifts after the start-of-run check
        return t.train_iteration(plans)

    with pytest.raises(C.TrainerCheckFailed) as exc:
        run_ranks([lambda r=r: rank(r) for r in range(2)], shared)
    assert "replicas_identical" in exc.value.failed


def test_replicas_whose_latents_differ_by_a_permutation_are_caught():
    """The latent comparison on its own (this merged forward holds no W_eff): rank 1's step ends with two entries
    of one B swapped."""
    shared, parts = _ranks(2)

    def rank(r):
        model, _, plans = parts[r]
        t = _trainer(model, NoMerge(), world=shared.world(r))
        if r == 1:
            _, b = next(iter(t.lora.modules.values()))
            step = t.optimizer.step

            def step_and_permute(*args, **kw):
                out = step(*args, **kw)
                _swap_two(b)
                return out

            t.optimizer.step = step_and_permute
        return t.train_iteration(plans)

    with pytest.raises(C.TrainerCheckFailed) as exc:
        run_ranks([lambda r=r: rank(r) for r in range(2)], shared)
    assert exc.value.failed == ["replicas_identical"]
    assert exc.value.report["checks"]["replicas_identical"]["latent"] is False


@pytest.mark.parametrize("change", ["shifted", "permuted"])
def test_ranks_that_do_not_start_from_the_same_latent_are_refused(change):
    """B is PEFT's zero, so W_eff = W on both ranks and only the latent can show that their A differ."""
    def drift(parts):
        a = _first_a(parts[1][0])
        if change == "permuted":
            _swap_two(a)
        else:
            with torch.no_grad():
                a.view(-1)[0] += 1e-3

    shared, parts = _ranks(2, drift)
    with pytest.raises(TrainerInputError, match="same LoRA latent"):
        run_ranks([lambda r=r: _trainer(parts[r][0], parts[r][1], world=shared.world(r)) for r in range(2)], shared)


def test_broadcast_lora_gives_every_rank_rank_0s_latent():
    """The setup order: PEFT, then broadcast_lora, then the merged forward (which caches W_eff from the latent)."""
    shared = SharedCollective(2)
    models = [tiny_peft(torch.bfloat16, seed=0, b_std=0.02) for _ in range(2)]
    with torch.no_grad():  # rank 1's PEFT drew a different latent (the base weights are the same)
        for a, b in C.LoraTensors.of(models[1]).modules.values():
            a.add_(0.01)
            b.mul_(-1.0)
    _, _, plans = _parts()

    def rank(r):
        R.broadcast_lora(models[r], shared.world(r))
        merged = merged_forward(models[r])
        return _trainer(models[r], merged, world=shared.world(r), settings=LossSettings(
            loss_temperature=0.7, max_generation_gap=50.0)).train_iteration(plans)

    reps = run_ranks([lambda r=r: rank(r) for r in range(2)], shared)
    assert all(rep[0]["checks"]["replicas_identical"]["passed"] for rep in reps)
    p0, p1 = (C.LoraTensors.of(m).params() for m in models)
    assert all(C.bit_identical(a, b) for a, b in zip(p0, p1))


def test_a_trainable_base_weight_is_refused():
    model = tiny_peft(torch.bfloat16, seed=0)
    next(p for n, p in model.named_parameters() if "embed_tokens" in n).requires_grad_(True)
    with pytest.raises(TrainerInputError, match="not a LoRA weight"):
        RankTrainer(model, merged_forward(model), LossSettings(loss_temperature=0.7), OptimSettings())


def test_the_trainer_needs_the_merged_forward():
    model = tiny_peft(torch.bfloat16, seed=0)
    with pytest.raises(TrainerInputError, match="merged"):
        RankTrainer(model, None, LossSettings(loss_temperature=0.7), OptimSettings())


def test_the_cosine_is_advisory():
    model, merged, plans, t = _setup()
    (rep,) = t.train_iteration(plans)
    c = rep["checks"]["step_descends_the_gradient"]
    assert c["advisory"] and "step_descends_the_gradient" not in C.GATING_POST
    assert c["cosine_delta_vs_negative_grad"] > 0  # step 1 of Adam: always positive


def test_an_engine_logprob_the_rollout_accepts_is_usable_for_the_gap():
    """An engine's log-probability of 1 can round slightly above 0: the gap gate takes what the rollout accepted."""
    from types import SimpleNamespace

    from aorlta.rollout.engine_pool import MAX_LOGPROB, logprobs_aligned
    from aorlta.trainer.loss import logprobs_usable

    for v, ok in ((-0.5, True), (0.0, True), (MAX_LOGPROB / 2, True), (MAX_LOGPROB, True), (2 * MAX_LOGPROB, False)):
        assert logprobs_usable(SimpleNamespace(engine_logprobs=[-1.0, v], response_ids=[1, 2])) is ok
        assert logprobs_aligned({"output_ids": [1, 2], "lp_ids": [1, 2], "lp_vals": [-1.0, v]}) is ok
