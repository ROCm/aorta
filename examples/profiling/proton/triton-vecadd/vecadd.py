"""Triton elementwise vector add, sized for Proton profiling.

Adapted from the Triton tutorial ``01-vector-add.py`` in
https://github.com/triton-lang/triton (MIT License). Trimmed to a single
kernel launch loop with a correctness check, and given CLI/env knobs so the
profiled region stays small.

Constexpr kernel parameters are spelled lowercase (``block_size`` rather
than the tutorial's ``BLOCK_SIZE``) to satisfy this repository's lint
configuration; Triton treats the name as opaque.

Standalone:
    python vecadd.py --size 1048576 --iters 20

Requires Triton and PyTorch built for ROCm.
"""

from __future__ import annotations

import argparse
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, out, n_elements, block_size=block_size)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton vector add for Proton examples")
    parser.add_argument("--size", type=int, default=1 << 20, help="number of elements")
    parser.add_argument("--iters", type=int, default=20, help="timed kernel launches")
    parser.add_argument("--block-size", type=int, default=1024, help="elements per program")
    args = parser.parse_args(argv)
    for name in ("size", "iters", "block_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    # ``tl.arange(0, block_size)`` requires a power-of-two extent. Rejecting it
    # here keeps the failure an argparse message instead of a Triton
    # compilation error raised from inside the kernel launch, which is what a
    # value like 1000 produced before.
    if args.block_size & (args.block_size - 1):
        parser.error(
            f"--block-size must be a power of two (tl.arange requires one); got "
            f"{args.block_size}, nearest is {triton.next_power_of_2(args.block_size)}"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("vecadd: no GPU visible to torch; refusing to run on CPU", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    print(f"vecadd: device={torch.cuda.get_device_name(device)}")
    print(f"vecadd: size={args.size} iters={args.iters} block_size={args.block_size}")

    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.rand(args.size, device=device, dtype=torch.float32, generator=generator)
    y = torch.rand(args.size, device=device, dtype=torch.float32, generator=generator)

    out = add(x, y, args.block_size)
    torch.cuda.synchronize(device)
    max_err = torch.max(torch.abs(out - (x + y))).item()

    for _ in range(args.iters):
        add(x, y, args.block_size)
    torch.cuda.synchronize(device)

    print(f"vecadd: max_abs_err={max_err:.3e}")
    if max_err != 0.0:
        print("vecadd: FAIL result differs from torch elementwise add", file=sys.stderr)
        return 1
    print("vecadd: PASS")
    return 0


if __name__ == "__main__":
    # Only exit non-zero, never SystemExit(0): Proton's execute_as_main on
    # Triton 3.6.0 catches Exception, not BaseException, so a SystemExit on the
    # success path escapes its CLI before finalize() writes the .hatchet.
    code = main()
    if code:
        raise SystemExit(code)
