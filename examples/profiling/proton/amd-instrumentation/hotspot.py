"""One deliberately unbalanced Triton kernel, measured from the inside.

The kernel is original to this repository. It does almost nothing in a scope
named ``cheap`` (a single multiply) and a short transcendental loop in a scope
named ``expensive``, so Proton's ``instrumentation`` backend has two regions
*within one kernel* to attribute cycles to -- attribution no queue-tracing
backend can give.

Constexpr kernel parameters are spelled lowercase (``block_size`` rather than
Triton's conventional ``BLOCK_SIZE``) to satisfy this repository's lint
configuration; Triton treats the name as opaque.

Standalone:
    python hotspot.py --size 65536 --steps 8
    python hotspot.py --backend instrumentation

Requires Triton and PyTorch built for ROCm.
"""

from __future__ import annotations

import argparse
import os
import sys

# Proton is imported BEFORE torch, and that ordering is load-bearing even
# though this backend intercepts no queues. `libproton.so` calls
# `rocprofiler_force_configure` from an `__attribute__((constructor))`, so the
# import registers Proton as a rocprofiler-sdk client whatever backend a session
# later selects; on Triton 3.8.0, registering after HSA is up (a torch import
# chain is enough) makes the atexit `registration::finalize` re-enter its own
# non-recursive registration mutex through Proton's `protonToolFini` and
# deadlock, so the process hangs forever after a perfectly good capture. See
# ROCm/aorta#434.
#
# ``torch`` still lands before ``proton.start()`` runs in main(), which is what
# the queue-intercepting backends need: ``roctracer`` records nothing unless the
# HIP runtime is already up. That constraint is about the START call rather than
# the import, so both orderings hold at once.
#
# The reason this example chooses ``mode: env`` is narrower and
# version-independent: Triton 3.7.1's CLI parses ``--mode`` and then calls
# ``start()`` without it, so ``instrumentation_mode`` would be dropped there.
# 3.8.0 forwards it, but this route reaches Proton on both -- see recipe.yaml.
import triton.profiler as proton  # isort: skip  # noqa: I001
import triton.profiler.language as pl  # isort: skip  # noqa: I001

import torch
import triton
import triton.language as tl

# Instrumenting a Triton-DSL kernel is opt-in: ``triton/profiler/language.py``
# enables only the Gluon semantic by default, because Triton's higher-level IR
# undergoes aggressive rewrites -- loop pipelining, instruction re-ordering, IR
# duplication -- that can invalidate naive instrumentation and report scope
# boundaries that mislead. Opting in is a deliberate acceptance of that risk,
# so the caveat travels with the numbers rather than being buried here.
pl.enable_semantic("triton")

#: Prefix of the variables aorta's ``proton`` collector exports in ``mode: env``.
_ENV_PREFIX = "AORTA_PROTON_"

#: Session name for a run aorta did not name, so a standalone capture still
#: lands somewhere predictable (``./hotspot.hatchet``).
_DEFAULT_SESSION = "hotspot"

#: Elements per program. A tuning constant, not a size knob -- the problem size
#: comes from ``--size``.
_BLOCK_SIZE = 1024

#: Absolute-error ceiling against the torch equivalent. Not exact equality: the
#: device ``erf`` and torch's differ in the last bits, and the loop applies it
#: ``--steps`` times. ``erf`` is contractive toward zero, so the discrepancy
#: shrinks rather than compounds across iterations.
_MAX_ABS_ERR = 1e-6


@triton.jit
def unbalanced_kernel(
    in_ptr, out_ptr, n_elements, loop_steps: tl.constexpr, block_size: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    values = tl.load(in_ptr + offsets, mask=mask)

    pl.enter_scope("cheap")
    scaled = values * 2.0
    pl.exit_scope("cheap")

    pl.enter_scope("expensive")
    activated = scaled
    for _ in range(loop_steps):
        activated = tl.math.erf(activated)
    pl.exit_scope("expensive")

    tl.store(out_ptr + offsets, activated, mask=mask)


def hotspot(x: torch.Tensor, steps: int) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    unbalanced_kernel[grid](x, out, n_elements, loop_steps=steps, block_size=_BLOCK_SIZE)
    return out


def reference(x: torch.Tensor, steps: int) -> torch.Tensor:
    activated = x * 2.0
    for _ in range(steps):
        activated = torch.erf(activated)
    return activated


def proton_kwargs(backend: str | None) -> dict[str, str | None]:
    """Translate aorta's ``AORTA_PROTON_*`` bundle into ``proton.start()`` keywords.

    A variable aorta did not export falls back to the collector's own default,
    which keeps a standalone run and a trial configured the same way spelled
    identically. ``AORTA_PROTON_MODE`` carries the recipe's
    ``instrumentation_mode``, which is the knob that reaches Proton only on this
    path -- see recipe.yaml. ``AORTA_PROTON_HOOK`` is read for the same reason:
    a knob the recipe sets and the payload drops would be configured and
    silently absent from the capture.
    """
    environ = os.environ
    return {
        "name": environ.get(f"{_ENV_PREFIX}NAME") or _DEFAULT_SESSION,
        "context": environ.get(f"{_ENV_PREFIX}CONTEXT", "shadow"),
        "data": environ.get(f"{_ENV_PREFIX}DATA", "tree"),
        "backend": environ.get(f"{_ENV_PREFIX}BACKEND") or backend,
        "hook": environ.get(f"{_ENV_PREFIX}HOOK"),
        "mode": environ.get(f"{_ENV_PREFIX}MODE"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unbalanced Triton kernel for Proton examples")
    parser.add_argument("--size", type=int, default=1 << 16, help="number of elements")
    parser.add_argument(
        "--steps", type=int, default=8, help="erf iterations in the expensive scope"
    )
    parser.add_argument("--iters", type=int, default=5, help="profiled kernel launches")
    # ``roctracer`` and ``rocprofiler`` stay on the list as the whole-kernel
    # control: the scope records compile away when the instrumentation backend
    # is not the one measuring, so the same payload gives one leaf and one
    # number, which is the comparison the README describes.
    parser.add_argument(
        "--backend",
        default=None,
        # ``rocprofiler`` is deliberately absent, not forgotten. This module
        # imports torch before Proton, which is right for the two backends left
        # here and is exactly the ordering that leaves rocprofiler with an empty
        # dispatch buffer. Offering it would hand out the trap this PR
        # documents; ``../amd-rocprofiler/gelu.py`` has the import order for it.
        choices=("instrumentation", "roctracer"),
        help="Proton backend for a standalone capture; ignored when $AORTA_PROTON_BACKEND is set",
    )
    args = parser.parse_args(argv)
    for name in ("size", "steps", "iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    # ``loop_steps`` is a tl.constexpr, so the loop is fully unrolled at compile
    # time. A large value turns into a long compile and a large kernel rather
    # than a slow one, which is not what this payload is demonstrating.
    if args.steps > 64:
        parser.error(
            f"--steps must be <= 64 (the loop is unrolled at compile time); got {args.steps}"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("hotspot: no GPU visible to torch; refusing to run on CPU", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    print(f"hotspot: device={torch.cuda.get_device_name(device)}")
    print(f"hotspot: size={args.size} steps={args.steps} iters={args.iters}")

    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(args.size, device=device, dtype=torch.float32, generator=generator)

    # No capture at all when nothing asked for one: that is how an operator
    # separates "my payload is broken" from "my profiler is broken".
    profiling = args.backend is not None or f"{_ENV_PREFIX}NAME" in os.environ
    if profiling:
        capture = proton_kwargs(args.backend)
        print(
            f"hotspot: proton backend={capture['backend']} "
            f"mode={capture['mode'] or '(unset)'} name={capture['name']}"
        )
        proton.start(**capture)
    # Every launch, including the first, happens inside the capture. The
    # instrumentation backend rewrites the kernel's IR to add the scope records,
    # so a warm-up launch outside the session would put an *uninstrumented*
    # binary in Triton's cache and the profiled launches would reuse it.
    try:
        out = hotspot(x, args.steps)
        for _ in range(args.iters - 1):
            hotspot(x, args.steps)
        torch.cuda.synchronize(device)
    finally:
        if profiling:
            proton.finalize()

    max_err = torch.max(torch.abs(out - reference(x, args.steps))).item()
    print(f"hotspot: max_abs_err={max_err:.3e}")
    # Negated bounded comparison, not ``max_err > tol``: every comparison with
    # NaN is false, so the direct form would let a non-finite result print PASS
    # -- the exact condition this check exists to catch.
    if not (max_err <= _MAX_ABS_ERR):
        print("hotspot: FAIL result differs from the torch equivalent", file=sys.stderr)
        return 1
    print("hotspot: PASS")
    return 0


if __name__ == "__main__":
    # Only exit non-zero, never SystemExit(0): Proton's execute_as_main on
    # Triton 3.6.0 catches Exception, not BaseException, so a SystemExit on the
    # success path escapes its CLI before finalize() writes the .hatchet.
    code = main()
    if code:
        raise SystemExit(code)
