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

    # Cheap sanity check: a finite result rules out the "profiled a NaN
    # storm" reading of an otherwise plausible looking trace.
    if not torch.isfinite(c).all():
        print("matmul: FAIL non-finite values in result", file=sys.stderr)
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
