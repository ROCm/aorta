"""C4.T2: the merged-weight forward (design Section 6.4).

- Forward: `MergedLoRALinear` equals `F.linear(x, W_eff)` bitwise, in fp32, in bf16 and under autocast.
- Backward with rounding disabled: its gradients equal autograd's through W + (alpha/r) B A, checked by finite
  differences (fp64 gradcheck) and against autograd itself (fp32).
- On a tiny Qwen3.5 model with PEFT: the model with `MergedLoRA` installed computes exactly the model whose
  target weights are W_eff, and after a step the forward runs on the refreshed W_eff.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from aorlta.sync.merge import TARGET_MODULE_PATTERN, merge_weight
from aorlta.trainer.merged_lora import MergedLoRA, MergedLoRAError, MergedLoRALinear, merged_lora_linear

S = 2.0  # alpha / r = 32 / 16


def _inputs(dtype=torch.float32, shape=(2, 5, 24), out_f=20, r=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g).to(dtype)
    w = (torch.randn(out_f, shape[-1], generator=g) * 0.05).to(dtype)
    a = torch.randn(r, shape[-1], generator=g) * 0.1
    b = torch.randn(out_f, r, generator=g) * 0.1
    return x, w, a, b


@pytest.mark.parametrize("shape", [(7, 24), (2, 5, 24)])
@pytest.mark.parametrize("mode", ["fp32", "bf16", "autocast_bf16_from_fp32", "autocast_bf16_from_bf16"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_forward_is_f_linear_bitwise(shape, mode, with_bias):
    x, w, a, b = _inputs(shape=shape)
    w_eff = merge_weight(w.to(torch.bfloat16), a, b, S)
    bias = (torch.randn(w.shape[0]) * 0.01) if with_bias else None
    if mode == "fp32":
        w_use, x_use, ctx = w_eff.float(), x, torch.autocast("cpu", enabled=False)
        bias = bias
    elif mode == "bf16":
        w_use, x_use, ctx = w_eff, x.to(torch.bfloat16), torch.autocast("cpu", enabled=False)
        bias = bias.to(torch.bfloat16) if bias is not None else None
    else:
        w_use = w_eff
        x_use = x if mode == "autocast_bf16_from_fp32" else x.to(torch.bfloat16)
        ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    with ctx:
        got = merged_lora_linear(x_use, w_use, a, b, S, bias)
        want = F.linear(x_use, w_use, bias)
    assert got.dtype == want.dtype
    assert torch.equal(got, want)


def test_gradients_match_finite_differences_without_rounding():
    """fp64 throughout, so nothing rounds: W_eff = W + s B A exactly, and the straight-through gradient is the
    true gradient. gradcheck compares the analytic backward with central finite differences."""
    x, w, a, b = (t.double() for t in _inputs(shape=(3, 6), out_f=5, r=2))
    for t in (x, a, b):
        t.requires_grad_(True)

    def f(x, a, b):
        return MergedLoRALinear.apply(x, (w + S * b @ a).detach(), a, b, S, None)

    assert torch.autograd.gradcheck(f, (x, a, b), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_gradients_equal_autograd_through_w_plus_ba_fp32():
    x, w, a, b = _inputs(shape=(4, 9, 24))
    g_out = torch.randn(4, 9, 20, generator=torch.Generator().manual_seed(5))
    xs, as_, bs = (t.clone().requires_grad_(True) for t in (x, a, b))
    (merged_lora_linear(xs, (w + S * b @ a).detach(), as_, bs, S) * g_out).sum().backward()
    xr, ar, br = (t.clone().requires_grad_(True) for t in (x, a, b))
    (F.linear(xr, w + S * br @ ar) * g_out).sum().backward()
    for got, want in ((xs.grad, xr.grad), (as_.grad, ar.grad), (bs.grad, br.grad)):
        # the two sum in different orders: fp32 rounding of terms up to about 10, so an atol of a few ulps of those
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got - want).abs().max()


def test_low_rank_products_use_the_autocast_dtype_and_return_param_dtypes():
    x, w, a, b = _inputs()
    w_eff = merge_weight(w.to(torch.bfloat16), a, b, S)
    xs, as_, bs = x.clone().requires_grad_(True), a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = merged_lora_linear(xs, w_eff, as_, bs, S)
    y.float().sum().backward()
    assert xs.grad.dtype == torch.float32 and as_.grad.dtype == torch.float32 and bs.grad.dtype == torch.float32
    g = torch.ones(x.shape[:-1] + (w.shape[0],), dtype=torch.bfloat16).reshape(-1, w.shape[0])
    xb = x.to(torch.bfloat16).reshape(-1, x.shape[-1])
    want_db = (g.t() @ (xb @ a.to(torch.bfloat16).t())).float() * S
    want_da = ((g @ b.to(torch.bfloat16)).t() @ xb).float() * S
    assert torch.equal(bs.grad, want_db)
    assert torch.equal(as_.grad, want_da)


# --- on a tiny Qwen3.5 model -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    text = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                linear_value_head_dim=16, linear_conv_kernel_dim=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"], vocab_size=320, full_attention_interval=4,
                max_position_embeddings=4096, tie_word_embeddings=True,
                rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default",
                                 "rope_theta": 10000000, "partial_rotary_factor": 0.25},
                rms_norm_eps=1e-6, attn_output_gate=True, dtype="bfloat16")
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=64, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64)
    d = tmp_path_factory.mktemp("tiny-c4t2")
    torch.manual_seed(7)
    Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision,
                                                  tie_word_embeddings=True)).to(torch.bfloat16).save_pretrained(str(d))
    return d


def _load(tiny_dir):
    from transformers import Qwen3_5ForConditionalGeneration

    m = Qwen3_5ForConditionalGeneration.from_pretrained(str(tiny_dir), dtype=torch.bfloat16).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _peft(tiny_dir, dropout=0.0):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    return get_peft_model(_load(tiny_dir), LoraConfig(r=4, lora_alpha=8, lora_dropout=dropout,
                                                      target_modules=TARGET_MODULE_PATTERN, bias="none"))


def _keys(model):
    return [k for k, _ in model.named_parameters()] + [k for k, _ in model.named_buffers()]


def _base_keys(tiny_dir):
    from safetensors import safe_open

    with safe_open(str(tiny_dir / "model.safetensors"), framework="pt") as f:
        return list(f.keys())


def _randomize_b(pm, seed=3, std=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=g) * std)


def test_tiny_model_runs_exactly_the_model_with_w_eff_weights(tiny_dir):
    pm = _peft(tiny_dir)
    _randomize_b(pm)
    merged = MergedLoRA(pm, _base_keys(tiny_dir))
    merged.refresh()
    assert merged.last_stats.changed == 0 and merged.last_stats.changed_vs_base > 0  # B was set before v0
    merged.install()
    plain = _load(tiny_dir)
    params = dict(plain.named_parameters())
    with torch.no_grad():
        for key, w_eff in merged.weights.items():
            params[key].copy_(w_eff)
    x = torch.randint(0, 320, (2, 33), generator=torch.Generator().manual_seed(1))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        got = pm(input_ids=x).logits
        want = plain(input_ids=x).logits
    assert torch.equal(got, want)


def test_disable_adapter_runs_the_base(tiny_dir):
    pm = _peft(tiny_dir)
    _randomize_b(pm)
    merged = MergedLoRA(pm, _base_keys(tiny_dir))
    merged.refresh()
    merged.install()
    x = torch.randint(0, 320, (1, 20), generator=torch.Generator().manual_seed(2))
    base = _load(tiny_dir)
    with torch.no_grad():
        with pm.disable_adapter():
            got = pm(input_ids=x).logits
        want = base(input_ids=x).logits
        lora = pm(input_ids=x).logits
    assert torch.equal(got, want)
    assert not torch.equal(lora, want)


def test_after_a_step_the_forward_uses_the_refreshed_w_eff(tiny_dir):
    """C4.T3 in miniature: the W_eff each layer's forward uses after `step` is exactly the merge of the new
    latent, which is also what `step` hands to C5."""
    pm = _peft(tiny_dir)
    merged = MergedLoRA(pm, _base_keys(tiny_dir)).install()
    v0 = {k: t.clone() for k, t in merged.weights.items()}
    opt = torch.optim.Adam([p for p in pm.parameters() if p.requires_grad], lr=5e-2)
    x = torch.randint(0, 320, (1, 24), generator=torch.Generator().manual_seed(4))
    for _ in range(2):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = pm(input_ids=x, labels=x).loss
        loss.backward()
        result = merged.step(opt)
        opt.zero_grad(set_to_none=True)
    assert result.stats.version == 2 and merged.version == 2
    with merged.capture() as used, torch.no_grad():
        pm(input_ids=x)
    assert set(used) == set(merged.layers)
    factors = merged.factors()
    changed_any = False
    for key, t in used.items():
        a, b, s = factors[key]
        expect = merge_weight(merged.base[key], a, b, s)
        assert torch.equal(t, expect), key
        assert torch.equal(t, result.weights[key]), key
        changed_any |= not torch.equal(t, v0[key])
    assert changed_any


def test_step_one_moves_b_and_leaves_a_unchanged(tiny_dir):
    """With B = 0, dL/dA is exactly 0 at step 1 (Section 6.4, checks every step)."""
    pm = _peft(tiny_dir)
    merged = MergedLoRA(pm, _base_keys(tiny_dir)).install()
    a0 = {k: v[0].clone() for k, v in merged.factors().items()}
    b0 = {k: v[1].clone() for k, v in merged.factors().items()}
    opt = torch.optim.Adam([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randint(0, 320, (1, 24), generator=torch.Generator().manual_seed(6))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pm(input_ids=x, labels=x).loss.backward()
    for key, layer in merged.layers.items():
        assert torch.count_nonzero(layer.lora_A["default"].weight.grad) == 0, key
    merged.step(opt)
    for key, (a, b, _) in merged.factors().items():
        assert torch.equal(a, a0[key]), key
        assert not torch.equal(b, b0[key]), key


def test_v0_is_the_base_bit_for_bit(tiny_dir):
    pm = _peft(tiny_dir)
    merged = MergedLoRA(pm, _base_keys(tiny_dir))
    assert merged.version == 0 and merged.last_stats.changed_vs_base == 0
    for key, t in merged.weights.items():
        assert torch.equal(t.view(torch.int16), merged.base[key].view(torch.int16))


def test_dropout_is_refused(tiny_dir):
    pm = _peft(tiny_dir, dropout=0.1)
    with pytest.raises(MergedLoRAError, match="dropout"):
        MergedLoRA(pm, _base_keys(tiny_dir))


def test_a_layer_without_a_base_key_is_refused(tiny_dir):
    from aorlta.sync.writer import NameMapError

    pm = _peft(tiny_dir)
    keys = [k for k in _base_keys(tiny_dir) if not k.endswith("layers.3.self_attn.o_proj.weight")]
    with pytest.raises(NameMapError):
        MergedLoRA(pm, keys)
