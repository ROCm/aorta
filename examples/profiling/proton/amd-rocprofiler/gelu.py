"""Transcendental-heavy Triton GELU, for Proton's ``rocprofiler`` backend.

The kernel is original to this repository: one elementwise ``erf``-based GELU
launched in a loop, so PC sampling has a steady stream of instructions to
sample rather than a handful of launches.

**The ``rocprofiler`` backend exists only on Triton ``main``.** No released
Triton has it. The payload probes for the capability and returns 2 -- the same
"this environment cannot run this" code it uses for a missing GPU -- rather
than letting the request fail as a confusing traceback. It still runs the
kernel and self-checks when no capture was requested, so the payload itself
stays verifiable on a Triton that cannot profile it.

Constexpr kernel parameters are spelled lowercase (``block_size`` rather than
Triton's conventional ``BLOCK_SIZE``) to satisfy this repository's lint
configuration; Triton treats the name as opaque.

Standalone:
    python gelu.py --size 4194304 --iters 50
    python gelu.py --backend rocprofiler --backend-mode pcsampling

Requires Triton and PyTorch built for ROCm.
"""

from __future__ import annotations

import argparse
import os
import sys

# ``torch`` is imported at module scope, not lazily inside main(), and that
# ordering is load-bearing: a queue-tracing backend records nothing when it
# attaches before the HIP runtime it is tracing, and importing torch is what
# brings that runtime up. It is also why this example is driven by
# ``mode: env`` -- see recipe.yaml.
import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton._C.libproton import proton as libproton

#: Prefix of the variables aorta's ``proton`` collector exports in ``mode: env``.
_ENV_PREFIX = "AORTA_PROTON_"

#: Session name for a run aorta did not name, so a standalone capture still
#: lands somewhere predictable (``./gelu.hatchet``).
_DEFAULT_SESSION = "gelu"

#: Elements per program. A tuning constant, not a size knob -- the problem size
#: comes from ``--size``.
_BLOCK_SIZE = 1024

#: Proton's backend set on every released Triton -- 3.6.0, 3.7.0 and 3.7.1 all
#: offer exactly these. Used to answer the availability question on the Tritons
#: that cannot be asked, since ``rocprofiler`` is precisely what is missing.
_PRE_REGISTRY_BACKENDS = frozenset({"cupti", "roctracer", "instrumentation"})

#: Proton's ``--mode`` domain per backend, mirroring aorta's own
#: ``BACKEND_MODES``. Duplicated rather than imported because this payload has
#: to run standalone, with no aorta on the path -- but it has to agree, or the
#: standalone command would accept a pair the recipe path rejects and fail
#: inside Proton instead of at its own argument check.
_BACKEND_MODES = {
    "rocprofiler": frozenset({"pcsampling", "periodic_flushing"}),
    "roctracer": frozenset({"periodic_flushing"}),
}

#: Absolute-error ceiling against ``torch.nn.functional.gelu``. Not exact
#: equality: the device ``erf`` and torch's fused activation differ in the last
#: bits. Elementwise, so nothing accumulates and the ceiling stays tight.
_MAX_ABS_ERR = 1e-6


@triton.jit
def gelu_kernel(in_ptr, out_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    values = tl.load(in_ptr + offsets, mask=mask)
    # The exact erf formulation, which is what torch's ``approximate="none"``
    # default computes. It is also the reason this payload suits PC sampling:
    # the erf expansion is many instructions per element, so the samples land
    # somewhere interesting instead of all on a load.
    activated = 0.5 * values * (1.0 + tl.math.erf(values * 0.7071067811865476))
    tl.store(out_ptr + offsets, activated, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    gelu_kernel[(triton.cdiv(n_elements, _BLOCK_SIZE),)](x, out, n_elements, block_size=_BLOCK_SIZE)
    return out


def available_backends() -> list[str] | None:
    """Report the Proton backends this Triton offers, or ``None`` if unknowable.

    Upstream builds the CLI's backend choices from
    ``libproton.get_available_profilers()``. That function arrived with the
    backend registry it reports, so its absence is itself informative: the
    Triton in front of you predates the registry, and no argument spelling will
    get a post-registry backend out of it.
    """
    probe = getattr(libproton, "get_available_profilers", None)
    if probe is None:
        return None
    return list(probe())


def unavailable_reason(backend: str) -> str | None:
    """Explain why ``backend`` cannot be used here, or ``None`` if it can.

    Pre-registry Tritons cannot be asked, so they are answered from the set
    that has been present since Proton shipped. Guessing "unavailable" for all
    of them would refuse ``roctracer``, which works fine on 3.7.1 -- and the
    point of the message is to name the fix, not to be conservative.
    """
    backends = available_backends()
    if backends is not None:
        if backend in backends:
            return None
        return (
            f"Proton backend {backend!r} is not available in Triton "
            f"{triton.__version__}; it offers {sorted(backends)}."
        )
    if backend in _PRE_REGISTRY_BACKENDS:
        return None
    return (
        f"Triton {triton.__version__} has no libproton.get_available_profilers, "
        f"so its Proton predates the backend registry and cannot offer {backend!r} "
        f"(it has only {sorted(_PRE_REGISTRY_BACKENDS)}). This example needs "
        "Triton built from upstream main; no released wheel or container image "
        "ships the rocprofiler backend."
    )


def proton_kwargs(args: argparse.Namespace) -> dict[str, str | None]:
    """Translate aorta's ``AORTA_PROTON_*`` bundle into ``proton.start()`` keywords.

    A variable aorta did not export falls back to the collector's own default,
    which keeps a standalone run and a trial configured the same way spelled
    identically. ``AORTA_PROTON_MODE`` carries the recipe's ``backend_mode``
    (``pcsampling`` here), which is the knob that reaches Proton only on this
    path -- see recipe.yaml. ``AORTA_PROTON_HOOK`` is read for the same reason:
    a knob the recipe sets and the payload drops would be configured and
    silently absent from the capture.
    """
    environ = os.environ
    return {
        "name": environ.get(f"{_ENV_PREFIX}NAME") or _DEFAULT_SESSION,
        "context": environ.get(f"{_ENV_PREFIX}CONTEXT", "shadow"),
        "data": environ.get(f"{_ENV_PREFIX}DATA", "tree"),
        "backend": environ.get(f"{_ENV_PREFIX}BACKEND") or args.backend,
        "mode": environ.get(f"{_ENV_PREFIX}MODE") or args.backend_mode,
        "hook": environ.get(f"{_ENV_PREFIX}HOOK"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triton GELU for Proton examples")
    parser.add_argument("--size", type=int, default=1 << 22, help="number of elements")
    parser.add_argument("--iters", type=int, default=50, help="profiled kernel launches")
    # Only the whole-kernel AMD backends. ``instrumentation`` is excluded
    # deliberately: it records intra-kernel scopes, which this kernel does not
    # carry, so it would hand back an empty tree. ``roctracer`` is kept because
    # it is the control this example's README points at on a Triton without
    # rocprofiler.
    parser.add_argument(
        "--backend",
        default=None,
        choices=("rocprofiler", "roctracer"),
        help="Proton backend for a standalone capture; ignored when $AORTA_PROTON_BACKEND is set",
    )
    parser.add_argument(
        "--backend-mode",
        default=None,
        choices=("pcsampling", "periodic_flushing"),
        help="Proton --mode for the backend; ignored when $AORTA_PROTON_MODE is set",
    )
    args = parser.parse_args(argv)
    for name in ("size", "iters"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    if args.backend_mode is not None and args.backend is None:
        parser.error("--backend-mode needs --backend: Proton's --mode is per-backend")
    if args.backend_mode is not None:
        allowed = _BACKEND_MODES[args.backend]
        if args.backend_mode not in allowed:
            parser.error(
                f"--backend-mode {args.backend_mode} is not valid for "
                f"--backend {args.backend}; it takes {sorted(allowed)}"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("gelu: no GPU visible to torch; refusing to run on CPU", file=sys.stderr)
        return 2

    # No capture at all when nothing asked for one: that is how an operator
    # separates "my payload is broken" from "my profiler is broken", and it is
    # what keeps this file useful on the released Tritons that cannot offer the
    # backend the recipe asks for.
    profiling = args.backend is not None or f"{_ENV_PREFIX}NAME" in os.environ
    capture = proton_kwargs(args) if profiling else None
    # Checked before anything is measured, so an unobtainable backend reads as
    # "this environment cannot run this" (exit 2, the missing-GPU code) rather
    # than as a traceback out of Proton's internals.
    if capture is not None and capture["backend"] is not None:
        reason = unavailable_reason(capture["backend"])
        if reason is not None:
            print(f"gelu: {reason}", file=sys.stderr)
            return 2

    device = torch.device("cuda")
    print(f"gelu: device={torch.cuda.get_device_name(device)}")
    print(f"gelu: size={args.size} iters={args.iters} triton={triton.__version__}")

    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(args.size, device=device, dtype=torch.float32, generator=generator)

    # The correctness pass runs unprofiled, which also absorbs the JIT compile.
    # The capture then holds exactly ``--iters`` launches, so a kernel count is
    # a number you can predict and check rather than one you have to trust.
    out = gelu(x)
    torch.cuda.synchronize(device)
    max_err = torch.max(torch.abs(out - torch.nn.functional.gelu(x))).item()

    if capture is not None:
        print(
            f"gelu: proton backend={capture['backend']} "
            f"mode={capture['mode'] or '(unset)'} name={capture['name']}"
        )
        proton.start(**capture)
    try:
        for _ in range(args.iters):
            gelu(x)
        torch.cuda.synchronize(device)
    finally:
        if capture is not None:
            proton.finalize()

    print(f"gelu: max_abs_err={max_err:.3e}")
    # Negated bounded comparison, not ``max_err > tol``: every comparison with
    # NaN is false, so the direct form would let a non-finite result print PASS
    # -- the exact condition this check exists to catch.
    if not (max_err <= _MAX_ABS_ERR):
        print("gelu: FAIL result differs from torch.nn.functional.gelu", file=sys.stderr)
        return 1
    print("gelu: PASS")
    return 0


if __name__ == "__main__":
    # Only exit non-zero, never SystemExit(0): Proton's execute_as_main on
    # Triton 3.6.0 catches Exception, not BaseException, so a SystemExit on the
    # success path escapes its CLI before finalize() writes the .hatchet.
    code = main()
    if code:
        raise SystemExit(code)
