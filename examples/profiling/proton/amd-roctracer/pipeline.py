"""Three-kernel Triton pipeline, captured by Proton's ``roctracer`` backend.

The kernels are original to this repository: an elementwise scale, a fused
bias+GELU, and a row-sum reduction, launched in that order so the hatchet tree
carries a real launch sequence rather than a single kernel.

Unlike the ``triton-vecadd`` / ``triton-softmax`` payloads this one drives
Proton itself, because pinning ``roctracer`` only works from inside a live HIP
runtime -- see the ``import torch`` comment below and ``recipe.yaml``. That is
specific to this backend: ``rocprofiler`` needs the opposite load order, and
``instrumentation`` needs no particular one.

Constexpr kernel parameters are spelled lowercase (``block_size`` rather than
Triton's conventional ``BLOCK_SIZE``) to satisfy this repository's lint
configuration; Triton treats the name as opaque.

Standalone:
    python pipeline.py --rows 4096 --cols 1024 --iters 20
    python pipeline.py --backend roctracer

Requires Triton and PyTorch built for ROCm.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# ``torch`` is imported at module scope, not lazily inside main(), and that
# ordering is load-bearing: roctracer records nothing when it attaches before
# the HIP runtime it is meant to trace, and importing torch is what brings that
# runtime up. The same constraint is why this example is driven by ``mode: env``
# -- Triton's Proton CLI initialises the driver only on the path where ``-b`` is
# absent, so pinning a backend through ``mode: cli`` produces a hatchet holding
# nothing but a bare ROOT frame, and still exits 0.
import torch
import triton
import triton.language as tl
import triton.profiler as proton

#: Prefix of the variables aorta's ``proton`` collector exports in ``mode: env``.
_ENV_PREFIX = "AORTA_PROTON_"

#: Session name for a run aorta did not name, so a standalone capture still
#: lands somewhere predictable (``./pipeline.hatchet``).
_DEFAULT_SESSION = "pipeline"

#: Elements per program in the flat elementwise kernel. A tuning constant, not
#: a size knob -- the problem size comes from ``--rows`` / ``--cols``.
_SCALE_BLOCK_SIZE = 1024

#: Relative-error ceiling for the composed pipeline against its torch
#: equivalent. Not exact equality: the last kernel reduces ``--cols`` float32
#: values in a different order than ``Tensor.sum``, so the low bits differ
#: legitimately, and the discrepancy grows with the reduced extent.
_MAX_REL_ERR = 1e-5


@triton.jit
def scale_kernel(in_ptr, out_ptr, n_elements, scale, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    values = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, values * scale, mask=mask)


@triton.jit
def bias_gelu_kernel(in_ptr, bias_ptr, out_ptr, row_stride, n_cols, block_size: tl.constexpr):
    row = tl.program_id(axis=0)
    col_offsets = tl.arange(0, block_size)
    mask = col_offsets < n_cols
    values = tl.load(in_ptr + row * row_stride + col_offsets, mask=mask)
    values += tl.load(bias_ptr + col_offsets, mask=mask)
    # The exact erf formulation, which is what torch's ``approximate="none"``
    # default computes. The cheaper tanh approximation would need a ``tanh``
    # this Triton's ``tl.math`` does not expose, and would then have to be
    # checked against a torch call that opts into the same approximation.
    activated = 0.5 * values * (1.0 + tl.math.erf(values * 0.7071067811865476))
    tl.store(out_ptr + row * row_stride + col_offsets, activated, mask=mask)


@triton.jit
def row_sum_kernel(in_ptr, out_ptr, row_stride, n_cols, block_size: tl.constexpr):
    row = tl.program_id(axis=0)
    col_offsets = tl.arange(0, block_size)
    mask = col_offsets < n_cols
    values = tl.load(in_ptr + row * row_stride + col_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(values, axis=0))


def pipeline(x: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """Run the three kernels in sequence and return the per-row totals."""
    n_rows, n_cols = x.shape
    # One program per row for the activation and the reduction, so a whole row
    # fits in one program. ``tl.arange`` requires a power-of-two extent.
    row_block = triton.next_power_of_2(n_cols)

    scaled = torch.empty_like(x)
    scale_kernel[(triton.cdiv(x.numel(), _SCALE_BLOCK_SIZE),)](
        x, scaled, x.numel(), scale, block_size=_SCALE_BLOCK_SIZE
    )

    activated = torch.empty_like(x)
    bias_gelu_kernel[(n_rows,)](scaled, bias, activated, x.stride(0), n_cols, block_size=row_block)

    totals = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    row_sum_kernel[(n_rows,)](activated, totals, x.stride(0), n_cols, block_size=row_block)
    return totals


def reference(x: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.nn.functional.gelu(x * scale + bias).sum(dim=-1)


def proton_kwargs(backend: str | None) -> dict[str, str | None]:
    """Translate aorta's ``AORTA_PROTON_*`` bundle into ``proton.start()`` keywords.

    A variable aorta did not export falls back to the collector's own default,
    which keeps a standalone run and a trial configured the same way spelled
    identically. ``backend`` is the one knob a standalone run needs on the
    command line, since it is what this example exists to pin. Every variable
    ``build_env`` can export is read, including ``AORTA_PROTON_HOOK``: a knob
    the recipe sets and the payload drops would be configured and silently
    absent from the capture, which is the failure this example exists to
    demonstrate the absence of.
    """
    environ = os.environ
    return {
        "name": environ.get(f"{_ENV_PREFIX}NAME") or _DEFAULT_SESSION,
        "context": environ.get(f"{_ENV_PREFIX}CONTEXT", "shadow"),
        "data": environ.get(f"{_ENV_PREFIX}DATA", "tree"),
        "backend": environ.get(f"{_ENV_PREFIX}BACKEND") or backend,
        "mode": environ.get(f"{_ENV_PREFIX}MODE"),
        "hook": environ.get(f"{_ENV_PREFIX}HOOK"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-kernel Triton pipeline for Proton examples")
    parser.add_argument("--rows", type=int, default=4096, help="rows in the input matrix")
    parser.add_argument("--cols", type=int, default=1024, help="columns (reduced dimension)")
    parser.add_argument("--iters", type=int, default=20, help="profiled pipeline iterations")
    # Only the whole-kernel AMD backends. ``instrumentation`` is excluded
    # deliberately: it records intra-kernel scopes, which these kernels do not
    # carry, so it yields the same 160-byte empty tree this example exists to
    # warn about (verified). Use ../amd-instrumentation for that backend.
    parser.add_argument(
        "--backend",
        default=None,
        # ``roctracer`` only. This module imports torch before Proton, which is
        # what roctracer needs and what rocprofiler cannot tolerate -- it is
        # configured from a ``libproton.so`` constructor and wants to land before
        # HSA. Use ``../amd-rocprofiler/gelu.py``, whose imports are the other
        # way round, for that backend.
        choices=("roctracer",),
        help="Proton backend for a standalone capture; ignored when $AORTA_PROTON_BACKEND is set",
    )
    args = parser.parse_args(argv)
    for name in ("rows", "cols", "iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("pipeline: no GPU visible to torch; refusing to run on CPU", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    print(f"pipeline: device={torch.cuda.get_device_name(device)}")
    print(f"pipeline: rows={args.rows} cols={args.cols} iters={args.iters}")

    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32, generator=generator)
    bias = torch.randn(args.cols, device=device, dtype=torch.float32, generator=generator)
    # The 1/sqrt(d) scaling an attention block applies before its activation --
    # a reason for the first kernel to exist rather than be folded away.
    scale = 1.0 / math.sqrt(args.cols)

    # The correctness pass runs unprofiled, which also absorbs the JIT compile
    # of all three kernels. The capture then holds exactly ``--iters`` launches
    # of each, so proton_kernel_count is a number you can predict and check
    # rather than one you have to trust.
    totals = pipeline(x, bias, scale)
    torch.cuda.synchronize(device)
    expected = reference(x, bias, scale)
    max_rel_err = ((totals - expected).abs() / expected.abs().clamp(min=1.0)).max().item()

    # No capture at all when nothing asked for one: that is how an operator
    # separates "my payload is broken" from "my profiler is broken".
    profiling = args.backend is not None or f"{_ENV_PREFIX}NAME" in os.environ
    if profiling:
        capture = proton_kwargs(args.backend)
        print(f"pipeline: proton backend={capture['backend']} name={capture['name']}")
        proton.start(**capture)
    try:
        for _ in range(args.iters):
            pipeline(x, bias, scale)
        torch.cuda.synchronize(device)
    finally:
        if profiling:
            proton.finalize()

    print(f"pipeline: max_rel_err={max_rel_err:.3e}")
    # Negated bounded comparison, not ``max_rel_err > tol``: every comparison
    # with NaN is false, so the direct form would let a non-finite result print
    # PASS -- the exact condition this check exists to catch.
    if not (max_rel_err <= _MAX_REL_ERR):
        print("pipeline: FAIL result differs from the torch equivalent", file=sys.stderr)
        return 1
    print("pipeline: PASS")
    return 0


if __name__ == "__main__":
    # Only exit non-zero, never SystemExit(0): Proton's execute_as_main on
    # Triton 3.6.0 catches Exception, not BaseException, so a SystemExit on the
    # success path escapes its CLI before finalize() writes the .hatchet.
    code = main()
    if code:
        raise SystemExit(code)
