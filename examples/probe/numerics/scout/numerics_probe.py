"""Scout the primitives a public TF32 NaN reproducer would need.

Four independent questions, one process each, because hipBLASLt reads its env
at first use and a knob flipped mid-process proves nothing:

  wiring  -- is DISABLE_TF32 actually observed on this stack, and does
             torch's allow_tf32 knob change fp32 matmul numerics at all?
             Everything downstream is vacuous if the answer is no.
  cancel  -- the one-pass variance identity E[x^2]-E[x]^2 on near-constant
             data. A real and common antipattern, and precision decides
             whether the computed variance goes negative and rsqrt returns
             NaN. Reports the threshold, so "TF32 moved it" is a number.
  attn    -- fp16 attention with large logits, to see whether the SDPA
             backend pin (fa_prefer_ck / fa_prefer_aotriton) is a live
             numerics axis on this hardware.
  train   -- a real optimizer loop at an aggressive learning rate, to see
             whether TF32 precision decides divergence rather than just
             perturbing it.

Nothing here injects a fault. Every mode measures what the stack does with a
configuration and reports it; a mode that finds nothing prints that it found
nothing.
"""

from __future__ import annotations

import json
import os
import sys

import torch


def _emit(mode: str, doc: dict) -> None:
    doc["mode"] = mode
    doc["allow_tf32_env"] = os.environ.get("ALLOW_TF32")
    doc["DISABLE_TF32"] = os.environ.get("DISABLE_TF32")
    doc["HIPBLASLT_ALLOW_TF32"] = os.environ.get("HIPBLASLT_ALLOW_TF32")
    doc["TORCH_ROCM_FA_PREFER_CK"] = os.environ.get("TORCH_ROCM_FA_PREFER_CK")
    print("PROBE_JSON " + json.dumps(doc), flush=True)


def _apply_allow_tf32() -> dict:
    """Apply the ALLOW_TF32 request and report every knob we can read back."""
    want = os.environ.get("ALLOW_TF32")
    before = {}
    after = {}
    try:
        before["matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    except Exception as exc:  # noqa: BLE001
        before["matmul.allow_tf32"] = f"<{exc}>"
    try:
        before["matmul.fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
    except Exception as exc:  # noqa: BLE001
        before["matmul.fp32_precision"] = f"<{exc}>"
    try:
        before["cudnn.allow_tf32"] = torch.backends.cudnn.allow_tf32
    except Exception as exc:  # noqa: BLE001
        before["cudnn.allow_tf32"] = f"<{exc}>"

    if want is not None:
        flag = want == "1"
        try:
            torch.backends.cuda.matmul.allow_tf32 = flag
        except Exception as exc:  # noqa: BLE001
            after["set_error"] = str(exc)
    try:
        after["matmul.allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        after["matmul.fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
    except Exception as exc:  # noqa: BLE001
        after["read_error"] = str(exc)
    return {"knobs_before": before, "knobs_after": after}


# --------------------------------------------------------------------------
def mode_wiring() -> int:
    """Relative error of an fp32 matmul against a float64 reference.

    TF32 keeps fp32's exponent and drops the mantissa to ~10 bits, so the
    signature is a jump in relative error of roughly three orders of
    magnitude with the exponent range untouched. That is what makes this a
    precision axis and not an overflow one.
    """
    doc = _apply_allow_tf32()
    dev = "cuda"
    torch.manual_seed(0)
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    ref = (a.double() @ b.double())
    got = (a @ b).double()
    err = ((got - ref).abs().max() / ref.abs().max()).item()
    rel = ((got - ref).norm() / ref.norm()).item()
    doc.update({"max_rel_err": err, "fro_rel_err": rel, "n": n})
    _emit("wiring", doc)
    return 0


# --------------------------------------------------------------------------
def mode_cancel() -> int:
    """One-pass variance on near-constant data: where does rsqrt go NaN?

    x = mu + sigma*z with sigma << mu. The identity E[x^2]-E[x]^2 subtracts
    two numbers that agree to ~log10(mu/sigma)^2 digits, so the answer is
    pure rounding error once that exceeds the mantissa. Sweeping mu/sigma
    finds the ratio at which the computed variance first goes negative --
    which is a *precision* threshold, so TF32 should move it.
    """
    doc = _apply_allow_tf32()
    dev = "cuda"
    torch.manual_seed(0)
    d = 4096
    rows = 256
    results = []
    first_nan = None
    for exponent in range(1, 9):
        ratio = 10.0**exponent
        mu = ratio
        z = torch.randn(rows, d, device=dev, dtype=torch.float32)
        x = mu + z  # sigma = 1, so mu/sigma == ratio
        ones = torch.ones(d, 1, device=dev, dtype=torch.float32)
        # Reductions routed through matmul on purpose: that is the operation
        # DISABLE_TF32 governs. A torch.mean() reduction is not a GEMM and
        # would not be affected, which is exactly the trap to avoid here.
        mean = (x @ ones) / d
        mean_sq = ((x * x) @ ones) / d
        var = mean_sq - mean * mean
        neg = int((var < 0).sum().item())
        inv = torch.rsqrt(var)
        nan = int((~torch.isfinite(inv)).sum().item())
        results.append(
            {
                "mu_over_sigma": ratio,
                "negative_var_rows": neg,
                "nonfinite_rsqrt_rows": nan,
                "rows": rows,
                "var_min": float(var.min().item()),
            }
        )
        if nan and first_nan is None:
            first_nan = ratio
    doc.update({"sweep": results, "first_nan_ratio": first_nan})
    _emit("cancel", doc)
    return 0


# --------------------------------------------------------------------------
def mode_attn() -> int:
    """fp16 attention with logits pushed toward the fp16 ceiling."""
    doc = _apply_allow_tf32()
    dev = "cuda"
    torch.manual_seed(0)
    out = []
    for scale in (1.0, 8.0, 32.0, 128.0):
        b, h, s, e = 2, 8, 1024, 128
        q = torch.randn(b, h, s, e, device=dev, dtype=torch.float16) * scale
        k = torch.randn(b, h, s, e, device=dev, dtype=torch.float16) * scale
        v = torch.randn(b, h, s, e, device=dev, dtype=torch.float16)
        try:
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            nonfinite = int((~torch.isfinite(o)).sum().item())
            out.append({"scale": scale, "nonfinite": nonfinite, "error": None})
        except Exception as exc:  # noqa: BLE001
            out.append({"scale": scale, "nonfinite": None, "error": str(exc)[:200]})
    doc.update({"sweep": out})
    _emit("attn", doc)
    return 0


# --------------------------------------------------------------------------
def mode_train() -> int:
    """A real optimizer loop at an aggressive lr; does precision decide divergence?

    Sweeps seeds so the answer is a rate rather than one anecdote: a chaotic
    divergence that TF32 merely perturbs would show a similar rate on both
    sides, and only a causal one shows a clean split.
    """
    doc = _apply_allow_tf32()
    dev = "cuda"
    lr = float(os.environ.get("PROBE_LR", "0.5"))
    steps = int(os.environ.get("PROBE_STEPS", "40"))
    seeds = int(os.environ.get("PROBE_SEEDS", "8"))
    width = int(os.environ.get("PROBE_WIDTH", "2048"))
    depth = int(os.environ.get("PROBE_DEPTH", "8"))

    diverged = 0
    first_bad_steps = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        layers = []
        for _ in range(depth):
            layers += [torch.nn.Linear(width, width), torch.nn.GELU()]
        model = torch.nn.Sequential(*layers).to(dev).float()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        x = torch.randn(64, width, device=dev)
        y = torch.randn(64, width, device=dev)
        bad = None
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                bad = step
                break
            loss.backward()
            opt.step()
        if bad is not None:
            diverged += 1
            first_bad_steps.append(bad)
    doc.update(
        {
            "lr": lr, "steps": steps, "seeds": seeds,
            "width": width, "depth": depth,
            "diverged": diverged,
            "first_bad_steps": first_bad_steps,
        }
    )
    _emit("train", doc)
    return 0


MODES = {
    "wiring": mode_wiring,
    "cancel": mode_cancel,
    "attn": mode_attn,
    "train": mode_train,
}


if __name__ == "__main__":
    name = sys.argv[1]
    print(f"[probe] mode={name} torch={torch.__version__} hip={torch.version.hip}", flush=True)
    sys.exit(MODES[name]())
