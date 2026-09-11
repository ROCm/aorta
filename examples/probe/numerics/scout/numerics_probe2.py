"""Round 2: three genuine non-finite mechanisms, and the whole registry against them.

Round 1 established that `DISABLE_TF32=1` — the env var aorta's `tf32_off`
mitigation sets — does not change fp32 matmul numerics on this stack at all.
That removes the mitigation the plan was built around, so the question becomes
the more basic one: does the registry contain *any* live lever over a
non-finite outcome here?

The mechanisms are chosen to be real failure modes rather than contrivances,
and each is labelled with how honest a "scenario" it would be:

  fp16_overflow -- fp16 activations grow past 65504 in a deep stack. The
                   single most common real NaN in mixed-precision training,
                   and the reason loss scaling and bf16 exist. Overflow is an
                   *exponent* problem, so no precision knob can address it;
                   included because a reproducer nothing fixes is still the
                   right thing to measure the registry against.
  onepass_var   -- E[x^2]-E[x]^2 on near-constant data, reductions routed
                   through GEMM. A real antipattern; round 1 showed it goes
                   non-finite at mu/sigma >= 1e4 identically with TF32 on and
                   off, i.e. precision does not decide it here.
  uninit_read   -- reading a torch.empty buffer that a real bug forgot to
                   fill. Genuinely common, and the one mechanism with a
                   plausible registered lever, because the caching allocator
                   is what makes the garbage a previous tensor rather than a
                   fresh zeroed page.

`calib` is separate: it asks whether the `tf32` setting is engaging anything
resembling a 10-bit mantissa, since round 1 measured only a ~4x error change
where true TF32 would give ~1000x.
"""

from __future__ import annotations

import json
import os
import sys

import torch

DEV = "cuda"


def _emit(doc: dict) -> None:
    doc["env_seen"] = {
        k: v for k, v in os.environ.items()
        if k in {
            "DISABLE_TF32", "HSA_XNACK", "GPU_MAX_HW_QUEUES", "DEBUG_HIP_DYNAMIC_QUEUES",
            "ROC_AQL_QUEUE_SIZE", "HSA_ENABLE_SDMA", "HSA_NO_SCRATCH_RECLAIM",
            "HSA_DISABLE_CACHE", "ROC_SIGNAL_POOL_SIZE", "GPU_FORCE_BLIT_COPY_SIZE",
            "DEBUG_CLR_BATCH_CPU_SYNC_SIZE", "NCCL_LAUNCH_ORDER_IMPLICIT",
            "RCCL_GFX942_CHEAP_FENCE_OFF", "PYTORCH_NO_CUDA_MEMORY_CACHING",
            "PYTORCH_CUDA_ALLOC_CONF", "HIP_LAUNCH_BLOCKING", "AMD_LOG_LEVEL",
            "TORCH_ROCM_FA_PREFER_CK", "ALLOW_TF32",
        }
    }
    print("PROBE_JSON " + json.dumps(doc), flush=True)


def _allow_tf32() -> None:
    want = os.environ.get("ALLOW_TF32")
    if want is not None:
        torch.backends.cuda.matmul.allow_tf32 = want == "1"


# --------------------------------------------------------------------------
def mode_calib() -> int:
    """Is the `tf32` fp32_precision setting engaging a reduced mantissa at all?

    Reference points: an exact float64 product, and the same product with A
    and B pre-truncated to a 10-bit mantissa in software. If `tf32` were
    engaging real TF32 hardware, its error would sit near the truncated
    reference; if it sits near IEEE fp32, the setting is nominal here.
    """
    results = {}
    torch.manual_seed(0)
    n = 2048
    a = torch.randn(n, n, device=DEV, dtype=torch.float32)
    b = torch.randn(n, n, device=DEV, dtype=torch.float32)
    ref = a.double() @ b.double()

    def rel(x):
        return ((x.double() - ref).norm() / ref.norm()).item()

    for setting in ("ieee", "tf32", "bf16"):
        try:
            torch.backends.cuda.matmul.fp32_precision = setting
            results[setting] = rel(a @ b)
        except Exception as exc:  # noqa: BLE001
            results[setting] = f"<{exc}>"

    # Software 10-bit-mantissa truncation: what TF32 inputs would look like.
    def trunc10(t):
        bits = t.view(torch.int32)
        return (bits & torch.tensor(-8192, dtype=torch.int32, device=DEV)).view(torch.float32)

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    results["software_tf32_reference"] = rel(trunc10(a) @ trunc10(b))
    results["bf16_cast_reference"] = rel((a.bfloat16() @ b.bfloat16()).float())
    _emit({"mode": "calib", "fro_rel_err": results, "n": n})
    return 0


# --------------------------------------------------------------------------
def mode_fp16_overflow() -> int:
    """fp16 forward whose activations pass 65504. Real, and deterministic."""
    _allow_tf32()
    torch.manual_seed(int(os.environ.get("PROBE_SEED", "0")))
    width = int(os.environ.get("PROBE_WIDTH", "4096"))
    depth = int(os.environ.get("PROBE_DEPTH", "12"))
    gain = float(os.environ.get("PROBE_GAIN", "2.5"))

    layers = []
    for _ in range(depth):
        lin = torch.nn.Linear(width, width, bias=False)
        # A deliberately hot init: this is what an unscaled residual stack or a
        # bad init actually looks like, and it is the growth per layer -- not
        # any single op -- that reaches the fp16 ceiling.
        torch.nn.init.normal_(lin.weight, std=gain / (width**0.5))
        layers.append(lin)
    model = torch.nn.Sequential(*layers).to(DEV).half()
    x = torch.randn(32, width, device=DEV, dtype=torch.float16)

    peak = []
    with torch.no_grad():
        h = x
        for i, layer in enumerate(model):
            h = layer(h)
            m = h.abs().max().item()
            peak.append(m)
            if not torch.isfinite(h).all():
                break
    loss = h.float().pow(2).mean().item()
    finite = bool(torch.isfinite(torch.tensor(loss)))
    if not finite:
        print(f"loss=nan at depth {len(peak)} (fp16 overflow)", flush=True)
    _emit({
        "mode": "fp16_overflow", "finite": finite, "loss": loss,
        "peak_abs_per_layer": [round(p, 1) if p == p else None for p in peak],
        "layers_run": len(peak), "depth": depth, "width": width, "gain": gain,
    })
    return 0 if finite else 1


# --------------------------------------------------------------------------
def mode_onepass_var() -> int:
    _allow_tf32()
    torch.manual_seed(int(os.environ.get("PROBE_SEED", "0")))
    ratio = float(os.environ.get("PROBE_RATIO", "1e5"))
    d, rows = 4096, 256
    z = torch.randn(rows, d, device=DEV, dtype=torch.float32)
    x = ratio + z
    ones = torch.ones(d, 1, device=DEV, dtype=torch.float32)
    mean = (x @ ones) / d
    var = ((x * x) @ ones) / d - mean * mean
    inv = torch.rsqrt(var)
    bad = int((~torch.isfinite(inv)).sum().item())
    if bad:
        print(f"loss=nan ({bad}/{rows} rows non-finite after rsqrt)", flush=True)
    _emit({"mode": "onepass_var", "finite": bad == 0, "nonfinite_rows": bad,
           "rows": rows, "ratio": ratio, "var_min": float(var.min().item())})
    return 0 if bad == 0 else 1


# --------------------------------------------------------------------------
def mode_uninit_read() -> int:
    """Read a torch.empty buffer a bug forgot to fill.

    First dirty the allocator with a large non-finite tensor and free it, then
    take a same-sized torch.empty. Under the caching allocator that block is
    handed straight back with the old contents; with caching disabled the
    allocation goes to hipMalloc, which need not preserve it. Whether that
    difference exists on ROCm is the question -- it is not assumed.
    """
    _allow_tf32()
    n = 1 << 22
    dirty = torch.full((n,), float("inf"), device=DEV, dtype=torch.float32)
    dirty[::2] = float("nan")
    del dirty
    buf = torch.empty(n, device=DEV, dtype=torch.float32)
    bad = int((~torch.isfinite(buf)).sum().item())
    loss = buf.mean().item()
    finite = bad == 0
    if not finite:
        print(f"loss=nan ({bad}/{n} elements non-finite from uninitialised buffer)", flush=True)
    _emit({"mode": "uninit_read", "finite": finite, "nonfinite_elems": bad,
           "elems": n, "mean": loss})
    return 0 if finite else 1


MODES = {
    "calib": mode_calib,
    "fp16_overflow": mode_fp16_overflow,
    "onepass_var": mode_onepass_var,
    "uninit_read": mode_uninit_read,
}

if __name__ == "__main__":
    sys.exit(MODES[sys.argv[1]]())
