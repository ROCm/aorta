"""Stochastic rounding in the merge (`sync.stochastic_rounding`; design Section 6.5): the seed, the cast and
`merge_all`. The switch changes only how the fp32 -> bf16 cast rounds. What must hold: the rounding is unbiased; its
random bits are a deterministic function of (run seed, version, tensor key), and `merge_weight` gives the bits
`merge_all` gives; a value already in bf16 rounds as the plain cast; and the version-0 merge of an untrained latent is
still the base. That the trainer's forward runs on the tensors the writer puts in the slot is in the trainer's PR
(`test_stochastic_rounding.py`).
"""

from __future__ import annotations

import hashlib
import math

import pytest
import torch

from aorlta.sync.merge import MergeError, bits, merge_all, merge_weight, rounding_seed, stochastic_bf16

K = "model.language_model.layers.0.mlp.gate_proj.weight"
K2 = "model.language_model.layers.3.self_attn.q_proj.weight"
STEP = 2.0**-7  # the bf16 spacing in [1, 2)
SEED = 20261102


def _w(shape=(64, 48), seed=0, scale=0.02):
    return (torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale).to(torch.bfloat16)


def _neighbours(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The bf16 values just below and just above |x| in magnitude (by truncating the fp32 pattern), with x's sign."""
    lo = (x.view(torch.int32) & -65536).view(torch.float32)
    hi = ((x.view(torch.int32) & -65536) + 65536).view(torch.float32)
    return lo, hi


# --- the seed ---------------------------------------------------------------------------------------------------


def test_the_seed_is_a_fixed_function_of_run_seed_version_and_key():
    s = rounding_seed(SEED, 3, K)
    assert s == rounding_seed(SEED, 3, K)
    assert 0 <= s < 2**63
    others = {rounding_seed(SEED + 1, 3, K), rounding_seed(SEED, 4, K), rounding_seed(SEED, 3, K2)}
    assert s not in others and len(others) == 3
    digest = hashlib.sha256(b'["aorlta.stochastic_rounding/1",20261102,3,"' + K.encode() + b'"]').digest()
    assert s == int.from_bytes(digest[:8], "little") & (2**63 - 1)


@pytest.mark.parametrize("args", [(1.5, 1, K), (True, 1, K), (SEED, -1, K), (SEED, True, K), (SEED, 1, ""),
                                  (SEED, 1, None)])
def test_the_seed_refuses_bad_inputs(args):
    with pytest.raises(MergeError):
        rounding_seed(*args)


# --- the cast ---------------------------------------------------------------------------------------------------


def test_each_element_lands_on_one_of_its_two_bf16_neighbours():
    x = torch.randn(4096, generator=torch.Generator().manual_seed(3)) * 0.05
    y = stochastic_bf16(x, 11).float()
    lo, hi = _neighbours(x)
    assert torch.all((y == lo) | (y == hi))
    assert bool((y == hi).any()) and bool((y == lo).any())


def test_values_already_in_bf16_and_non_finite_values_round_as_the_plain_cast():
    x = torch.tensor([1.0, -1.0, 0.0, -0.0, 0.5, -3.75, float("nan"), float("inf"), -float("inf")])
    for seed in range(20):
        assert torch.equal(bits(stochastic_bf16(x, seed)), bits(x.to(torch.bfloat16)))


def test_the_same_seed_gives_the_same_bits_and_another_seed_others():
    x = torch.randn(4096, generator=torch.Generator().manual_seed(4)) * 0.05
    a, b, c = stochastic_bf16(x, 5), stochastic_bf16(x, 5), stochastic_bf16(x, 6)
    assert torch.equal(bits(a), bits(b))
    assert not torch.equal(bits(a), bits(c))


@pytest.mark.parametrize("frac", [0.1, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rounding_up_happens_with_the_probability_of_the_distance(frac, sign):
    n = 1 << 20
    x = torch.full((n,), sign * (1.0 + frac * STEP))
    y = stochastic_bf16(x, rounding_seed(SEED, 1, f"frac{frac}{sign}")).float()
    up = float((y.abs() > 1.0).double().mean())
    sigma = math.sqrt(frac * (1 - frac) / n)
    assert abs(up - frac) < 6 * sigma
    # unbiased: the mean error is within the same few standard errors of zero
    assert abs(float((y.double() - x.double()).mean()) / STEP) < 6 * sigma


def test_the_cast_refuses_anything_but_fp32():
    with pytest.raises(MergeError):
        stochastic_bf16(torch.zeros(4, dtype=torch.float64), 1)


# --- merge_all --------------------------------------------------------------------------------------------------


def _factors(seed=1, scale=1e-3, shape=(64, 48), r=16):
    a = torch.randn(r, shape[1], generator=torch.Generator().manual_seed(seed)) * scale
    b = torch.randn(shape[0], r, generator=torch.Generator().manual_seed(seed + 1)) * scale
    return a, b


def test_nearest_stays_the_default_and_the_plain_cast():
    w = _w()
    a, b = _factors(scale=0.05)
    r = merge_all({K: w}, {K: (a, b, 2.0)}, {K: w}, 1)
    assert r.stats.rounding == "nearest" and r.stats.as_dict()["rounding"] == "nearest"
    assert torch.equal(bits(r.weights[K]), bits((w.float() + 2.0 * (b @ a)).to(torch.bfloat16)))


def test_stochastic_merge_is_reproducible_per_version_and_key_and_not_the_nearest_cast():
    w = _w()
    a, b = _factors(scale=0.05)
    base, f = {K: w, K2: w.clone()}, {K: (a, b, 2.0), K2: (a, b, 2.0)}
    r1 = merge_all(base, f, base, 1, rounding="stochastic", run_seed=SEED)
    again = merge_all(base, f, base, 1, rounding="stochastic", run_seed=SEED)
    r2 = merge_all(base, f, base, 2, rounding="stochastic", run_seed=SEED)
    near = merge_all(base, f, base, 1)
    assert r1.stats.rounding == "stochastic"
    for k in (K, K2):
        assert torch.equal(bits(r1.weights[k]), bits(again.weights[k]))
        assert not torch.equal(bits(r1.weights[k]), bits(r2.weights[k]))
        assert not torch.equal(bits(r1.weights[k]), bits(near.weights[k]))
        assert torch.equal(bits(r1.weights[k]),
                           bits(merge_weight(w, a, b, 2.0, k, rounding="stochastic", run_seed=SEED, version=1)))
    # same inputs under two keys: different bits
    assert not torch.equal(bits(r1.weights[K]), bits(r1.weights[K2]))


def test_a_zero_update_is_the_base_under_stochastic_rounding():
    w = _w()
    a, _ = _factors()
    r = merge_all({K: w}, {K: (a, torch.zeros(64, 16), 2.0)}, {K: w}, 0, rounding="stochastic", run_seed=SEED)
    assert torch.equal(bits(r.weights[K]), bits(w)) and r.stats.changed_vs_base == 0


def test_an_update_far_below_bf16_resolution_survives_in_expectation_only_when_stochastic():
    """WP3's reason for the switch, in miniature: at a tiny lr nearest rounding serves almost none of the update;
    stochastic rounding serves all of it in the update's own direction (aligned close to 1), plus noise."""
    w = _w(shape=(512, 512), scale=0.02)
    a, b = _factors(scale=5e-4, shape=(512, 512))  # (alpha/r) B A ~ 4e-6 per element, bf16 step ~ 1e-4 at |W| 0.02
    near = merge_all({K: w}, {K: (a, b, 2.0)}, {K: w}, 1).stats
    sr = merge_all({K: w}, {K: (a, b, 2.0)}, {K: w}, 1, rounding="stochastic", run_seed=SEED).stats
    assert near.aligned < 0.15  # measured 0.06
    assert abs(sr.aligned - 1.0) < 0.05  # measured 1.001
    assert sr.coverage > 1.0 and sr.changed_frac > near.changed_frac


@pytest.mark.parametrize("kw", [dict(rounding="up"), dict(rounding="stochastic"),
                                dict(rounding="stochastic", run_seed=None)])
def test_merge_refuses_an_unknown_rounding_or_stochastic_without_a_seed(kw):
    w = _w()
    a, b = _factors()
    with pytest.raises(MergeError):
        merge_all({K: w}, {K: (a, b, 2.0)}, {K: w}, 1, **kw)
    with pytest.raises(MergeError):
        merge_weight(w, a, b, 2.0, K, **kw)
