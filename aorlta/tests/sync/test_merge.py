"""C5 step 1: the merge, changed(n) and coverage (design Section 6.5)."""

from __future__ import annotations

import math

import pytest
import torch

from aorlta.sync.merge import MergeError, bits, group_of, is_target_key, merge_all, merge_weight

K = "model.language_model.layers.0.mlp.gate_proj.weight"
K2 = "model.language_model.layers.3.self_attn.q_proj.weight"


def _w(shape=(64, 48), seed=0, scale=0.02):
    return (torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale).to(torch.bfloat16)


def test_merge_is_the_section_6_5_formula():
    w = _w()
    a = torch.randn(16, 48, generator=torch.Generator().manual_seed(1)) * 0.01
    b = torch.randn(64, 16, generator=torch.Generator().manual_seed(2)) * 0.01
    want = (w.float() + 2.0 * (b.float() @ a.float())).to(torch.bfloat16)
    assert torch.equal(bits(merge_weight(w, a, b, 2.0)), bits(want))


def test_b_zero_is_the_base_bit_for_bit():
    w = _w()
    a = torch.randn(16, 48)
    assert torch.equal(bits(merge_weight(w, a, torch.zeros(64, 16), 2.0)), bits(w))


def test_merge_refuses_mismatched_inputs():
    w = _w()
    with pytest.raises(MergeError):
        merge_weight(w.float(), torch.zeros(16, 48), torch.zeros(64, 16), 2.0)
    with pytest.raises(MergeError):
        merge_weight(w, torch.zeros(16, 47), torch.zeros(64, 16), 2.0)
    with pytest.raises(MergeError):
        merge_all({K: w}, {K: (torch.zeros(16, 48), torch.zeros(64, 16), 2.0)}, {}, 1)


def test_changed_counts_against_the_previous_step_not_the_base():
    w = _w()
    a = torch.randn(16, 48, generator=torch.Generator().manual_seed(1)) * 0.1
    b1 = torch.randn(64, 16, generator=torch.Generator().manual_seed(2)) * 0.1
    r1 = merge_all({K: w}, {K: (a, b1, 2.0)}, {K: w}, 1)
    assert r1.stats.changed == r1.stats.changed_vs_base > 0
    # Step 2 makes no change: changed(2) is 0 although W_eff(2) is still far from the base.
    r2 = merge_all({K: w}, {K: (a, b1, 2.0)}, r1.weights, 2)
    assert r2.stats.changed == 0
    assert r2.stats.changed_vs_base == r1.stats.changed_vs_base > 0
    # Step 3 moves a few elements.
    b3 = b1.clone()
    b3[0] += 1.0
    r3 = merge_all({K: w}, {K: (a, b3, 2.0)}, r2.weights, 3)
    assert 0 < r3.stats.changed <= 48
    assert r3.stats.changed == int((bits(r3.weights[K]) != bits(r2.weights[K])).sum())


def test_an_update_below_bf16_resolution_changes_nothing():
    w = torch.full((8, 8), 0.5, dtype=torch.bfloat16)  # ulp(0.5) = 2**-8; half-ulp 2**-9
    a = torch.ones(1, 8)
    b = torch.full((8, 1), 2**-12)  # 2 * 2**-12 = 2**-11 per element, below half an ulp
    r = merge_all({K: w}, {K: (a, b, 2.0)}, {K: w}, 1)
    assert r.stats.changed == 0 and r.stats.coverage == 0.0
    b_big = torch.full((8, 1), 2**-8)  # 2**-7 per element: two ulps, exactly representable
    r = merge_all({K: w}, {K: (a, b_big, 2.0)}, {K: w}, 1)
    assert r.stats.changed == 64 and r.stats.changed_frac == 1.0
    assert (r.stats.coverage, r.stats.aligned, r.stats.residual) == (1.0, 1.0, 0.0)


def test_aligned_share_separates_rounding_inflation_from_the_update():
    """One element of four changes, by a whole ulp (2**-8) where the update was 0.75 ulp everywhere: coverage
    counts the ulp jump, the aligned share counts only the update's own direction."""
    w = torch.tensor([[0.5, 0.75, 1.0, 1.5]], dtype=torch.bfloat16)  # ulps 2**-8, 2**-8, 2**-7, 2**-7
    a = torch.ones(1, 4)
    b = torch.full((1, 1), 0.75 * 2**-8 / 2.0)  # scale 2.0: update 0.75 * 2**-8 on every element
    r = merge_all({K: w}, {K: (a, b, 2.0)}, {K: w}, 1)
    upd = 0.75 * 2**-8
    assert r.stats.changed == 2  # 0.5 and 0.75 move by one ulp; 1.0 and 1.5 (ulp 2**-7) do not
    assert r.stats.coverage == pytest.approx(math.sqrt(2 * (2**-8) ** 2 / (4 * upd**2)))
    assert r.stats.aligned == pytest.approx(2 * 2**-8 * upd / (4 * upd**2))
    assert r.stats.residual == pytest.approx(math.sqrt((2 * (2**-8 - upd) ** 2 + 2 * upd**2) / (4 * upd**2)))


def test_coverage_is_the_visible_share_of_the_update():
    w = _w((128, 96), scale=0.02)
    a = torch.randn(16, 96, generator=torch.Generator().manual_seed(4)) * 0.05
    b = torch.randn(128, 16, generator=torch.Generator().manual_seed(5)) * 0.05
    r = merge_all({K: w}, {K: (a, b, 2.0)}, {K: w}, 1)
    lora = 2.0 * (b @ a)
    want = torch.linalg.norm(r.weights[K].float() - w.float()) / torch.linalg.norm(lora)
    assert math.isclose(r.stats.coverage, float(want), rel_tol=1e-5)
    assert r.stats.groups["mlp.gate_proj"]["coverage"] == pytest.approx(r.stats.coverage)
    zero = merge_all({K: w}, {K: (a, torch.zeros(128, 16), 2.0)}, {K: w}, 1)
    assert zero.stats.coverage is None and zero.stats.changed == 0


def test_groups_and_totals_add_up():
    base = {K: _w((32, 16), seed=1), K2: _w((24, 16), seed=2)}
    f = {k: (torch.randn(4, 16) * 0.2, torch.randn(v.shape[0], 4) * 0.2, 2.0) for k, v in base.items()}
    r = merge_all(base, f, base, 1)
    assert set(r.stats.groups) == {"mlp.gate_proj", "self_attn.q_proj"}
    assert r.stats.total == 32 * 16 + 24 * 16
    assert sum(g["changed"] for g in r.stats.groups.values()) == r.stats.changed


def test_names():
    assert is_target_key(K) and is_target_key(K2)
    assert is_target_key("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
    assert not is_target_key("model.language_model.layers.0.linear_attn.conv1d.weight")
    assert not is_target_key("mtp.layers.0.mlp.gate_proj.weight")
    assert not is_target_key("model.visual.blocks.0.mlp.linear_fc1.weight")
    assert group_of(K) == "mlp.gate_proj"
