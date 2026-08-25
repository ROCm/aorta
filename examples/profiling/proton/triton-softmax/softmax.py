"""Triton fused row-softmax, sized for Proton profiling.

Adapted from the Triton tutorial ``02-fused-softmax.py`` in
https://github.com/triton-lang/triton (MIT License). Reduced to the
one-program-per-row kernel plus a correctness check against
``torch.softmax``, with CLI/env knobs so the profiled region stays small.

Constexpr kernel parameters are spelled lowercase (``block_size`` rather
than the tutorial's ``BLOCK_SIZE``) to satisfy this repository's lint
configuration; Triton treats the name as opaque.

Standalone:
    python softmax.py --rows 4096 --cols 1024 --iters 20

Requires Triton and PyTorch built for ROCm.
"""

from __future__ import annotations

import argparse
import sys

import torch
import triton
import triton.language as tl

#: Absolute-error ceiling against ``torch.softmax``. Not exact equality (unlike
#: the vector-add example) because the reduction reassociates.
_MAX_ABS_ERR = 1e-6


@triton.jit
def softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    block_size: tl.constexpr,
):
    row = tl.program_id(axis=0)
    col_offsets = tl.arange(0, block_size)
    mask = col_offsets < n_cols

    row_values = tl.load(in_ptr + row * in_row_stride + col_offsets, mask=mask, other=-float("inf"))
    # Subtract the row max before exponentiating: the numerically stable
    # formulation, and the whole reason a fused kernel beats three passes.
    shifted = row_values - tl.max(row_values, axis=0)
    numerator = tl.exp(shifted)
    softmax_out = numerator / tl.sum(numerator, axis=0)

    tl.store(out_ptr + row * out_row_stride + col_offsets, softmax_out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_cols,
        block_size=block_size,
    )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton fused softmax for Proton examples")
    parser.add_argument("--rows", type=int, default=4096, help="rows in the input matrix")
    parser.add_argument("--cols", type=int, default=1024, help="columns (reduced dimension)")
    parser.add_argument("--iters", type=int, default=20, help="timed kernel launches")
    args = parser.parse_args(argv)
    for name in ("rows", "cols", "iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("softmax: no GPU visible to torch; refusing to run on CPU", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    print(f"softmax: device={torch.cuda.get_device_name(device)}")
    print(f"softmax: rows={args.rows} cols={args.cols} iters={args.iters}")

    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32, generator=generator)

    out = softmax(x)
    torch.cuda.synchronize(device)
    # ``dim``, not ``axis``: torch accepts both (``axis`` is its numpy-compat
    # alias) but only ``dim`` is documented, and this file uses ``axis`` for
    # Triton's own reduction API a few lines up. Keeping the two spellings on
    # their own sides of the fence stops the mix reading as a bug.
    max_err = torch.max(torch.abs(out - torch.softmax(x, dim=-1))).item()

    for _ in range(args.iters):
        softmax(x)
    torch.cuda.synchronize(device)

    print(f"softmax: max_abs_err={max_err:.3e}")
    # Negated bounded comparison, not ``max_err > tol``: every comparison with
    # NaN is false, so the direct form lets a non-finite result print PASS --
    # the exact condition this check exists to catch. ``not (x <= tol)`` is
    # true for NaN.
    if not (max_err <= _MAX_ABS_ERR):
        print("softmax: FAIL result differs from torch.softmax", file=sys.stderr)
        return 1
    print("softmax: PASS")
    return 0


if __name__ == "__main__":
    # Only exit non-zero, never SystemExit(0): Proton's execute_as_main on
    # Triton 3.6.0 catches Exception, not BaseException, so a SystemExit on the
    # success path escapes its CLI before finalize() writes the .hatchet.
    code = main()
    if code:
        raise SystemExit(code)
