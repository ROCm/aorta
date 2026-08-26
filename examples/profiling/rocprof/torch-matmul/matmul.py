"""Repeated PyTorch matmul on the default accelerator, sized for profiling.

A minimal PyTorch payload for the rocprof collector examples: it dispatches
a handful of large GEMMs through whatever BLAS backend the installed
PyTorch uses (hipBLASLt / rocBLAS on ROCm), so a kernel trace shows real
library kernel names rather than a hand-written one.

Standalone:
    python matmul.py --size 2048 --iters 20 --dtype float16

Requires PyTorch built for ROCm. There is no CPU fallback on purpose --
a silent CPU run would produce an empty GPU profile and look like a
collector bug.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

#: Relative-error ceiling per operand dtype. The reference is computed from the
#: same already-rounded operands promoted to float32, so the only error being
#: bounded is the library's accumulation order -- not the input rounding. The
#: float32 ceiling matches ``hip-gemm``'s ``kRelTolerance``; a backend that
#: splits the K loop differently must not make a correct example look broken.
#: Measured against a CPU reference at size 2048: 3.9e-06 / 4.1e-04 / 2.9e-03.
_REL_TOLERANCE = {
    "float32": 1e-3,
    "float16": 2e-2,
    "bfloat16": 5e-2,
}

#: How many entries of the product to check against an independent reference.
_SAMPLES = 8


def max_sampled_rel_err(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, samples: int = _SAMPLES
) -> float:
    """Largest relative error over a few sampled entries of ``c``.

    A full reference GEMM would cost as much as the payload, so this checks
    ``c[i, j] == dot(a[i, :], b[:, j])`` at evenly spaced ``i``. Finiteness
    alone does not verify a product -- an all-zero or transposed result is
    perfectly finite -- and rule 2 of the examples contract asks each payload
    to verify its own output. Mirrors the CPU reference the ``hip-gemm``
    example computes for the same reason.

    The samples are deliberately **off-diagonal**: a transpose leaves the
    diagonal untouched (``c.T[i, i] == c[i, i]``), so probing ``c[i, i]``
    could never catch one.
    """
    edge = c.shape[0]
    step = max(1, edge // samples)
    # Any fixed non-zero offset breaks the row == column symmetry; a third of
    # the edge keeps the pairs spread out for a small ``samples``.
    offset = max(1, edge // 3)
    worst = 0.0
    for row in range(0, edge, step):
        col = (row + offset) % edge
        reference = torch.dot(a[row].to(torch.float32), b[:, col].to(torch.float32)).item()
        actual = float(c[row, col].item())
        # Guard against dividing by a near-zero reference, as hip-gemm does.
        scale = max(abs(reference), 1.0)
        worst = max(worst, abs(actual - reference) / scale)
    return worst


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=2048, help="square matrix edge length")
    parser.add_argument("--iters", type=int, default=20, help="timed matmul iterations")
    parser.add_argument("--warmup", type=int, default=3, help="untimed warmup iterations")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="float16", help="operand dtype")
    parser.add_argument("--device", default="cuda", help="torch device (ROCm reports as 'cuda')")
    args = parser.parse_args(argv)
    for name in ("size", "iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("matmul: no GPU visible to torch; refusing to run on CPU", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]
    print(f"matmul: device={torch.cuda.get_device_name(device)} dtype={args.dtype}")
    print(f"matmul: size={args.size} iters={args.iters} warmup={args.warmup}")

    generator = torch.Generator(device=device).manual_seed(1234)
    a = torch.randn(args.size, args.size, device=device, dtype=dtype, generator=generator)
    b = torch.randn(args.size, args.size, device=device, dtype=dtype, generator=generator)

    for _ in range(args.warmup):
        a @ b
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(args.iters):
        c = a @ b
    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start

    mean_ms = elapsed_s * 1e3 / args.iters
    gflops = 2.0 * args.size**3 / (mean_ms * 1e-3) / 1e9
    print(f"matmul: mean_kernel_ms={mean_ms:.4f}")
    print(f"matmul: gflops={gflops:.2f}")

    # Finiteness rules out the "profiled a NaN storm" reading of an otherwise
    # plausible trace, but it does not verify the product: an all-zero or
    # transposed result is finite too. The sampled reference does verify it.
    if not torch.isfinite(c).all():
        print("matmul: FAIL non-finite values in result", file=sys.stderr)
        return 1
    max_rel_err = max_sampled_rel_err(a, b, c)
    tolerance = _REL_TOLERANCE[args.dtype]
    print(f"matmul: max_rel_err={max_rel_err:.3e}")
    if not (max_rel_err < tolerance):
        print(
            f"matmul: FAIL correctness (max_rel_err={max_rel_err:.3e} >= {tolerance:.3e})",
            file=sys.stderr,
        )
        return 1
    print("matmul: PASS")
    return 0


if __name__ == "__main__":
    # Only exit non-zero, never SystemExit(0). rocprofv3 does not care, but
    # --collect proton can be pointed at this payload, and Proton's
    # execute_as_main on Triton 3.6.0 catches Exception, not BaseException, so a
    # SystemExit on the success path escapes before finalize() writes the data.
    code = main()
    if code:
        raise SystemExit(code)
