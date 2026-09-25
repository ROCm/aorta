#!/usr/bin/env python3
"""Verify a trained checkpoint against its starting point, on disk, on CPU.

What it checks
==============
Given the checkpoint a run started from (PRE) and the one it produced (POST),
both as safetensors trees, three things, each of which a real optimiser update
satisfies and a spurious write to the weights does not:

1. **every trained tensor moved.** A tensor the optimiser stepped and that is
   bit-identical afterwards means the update never reached it -- a severed
   graph, a frozen parameter that was meant to train, or a delta rounded away
   in a low-precision copy.
2. **the frozen control did not move at all.** The trainer freezes
   ``model.embed_tokens`` precisely so that "only what the optimiser touched
   changed" has something to contrast against. A control that moved, or that
   is absent from the tree, means the pair cannot be vouched for -- "the
   control did not move" and "there was no control" print the same way if you
   only test for movement.
3. **nothing moved further than the optimiser can move it**: no element's
   displacement exceeds ``4 x lr x steps`` (plus the storage dtype's rounding
   allowance, below). A delta beyond that did not come from Adam, so it came
   from something else writing to the weights, and a checkpoint carrying one
   is not the result of the training run it claims to be.

The floor (1) alone is not enough, and that is the reason for (3): a tensor
that was damaged has also "moved", so a check that only asks whether tensors
moved passes damage.

Why ``4 x lr x steps`` bounds Adam
==================================
Adam's update is ``lr * m_hat / (sqrt(v_hat) + eps)``, where ``m`` and ``v``
are exponential moving averages of the gradient and its square. The ratio
``|m_hat| / sqrt(v_hat)`` is bounded for *any* gradient sequence, by
Cauchy-Schwarz over the two EMAs::

    |m_t| <= (1 - b1) * sum_k b1^k |g_{t-k}|
          <= (1 - b1) / sqrt(1 - b2) * sqrt(sum_k (b1^2 / b2)^k) * sqrt(v_t)

and after bias correction the per-step displacement is at most ``lr * c(t)``
with ``c(t) = (1-b1)/sqrt(1-b2) * sqrt(sum_{k<t} (b1^2/b2)^k) *
sqrt(1-b2^t) / (1-b1^t)``. At the default betas (0.9, 0.999) ``c(1) = 1`` --
the first step moves every element by exactly ``lr`` whatever the gradient,
which is the normalisation Adam exists for -- and ``c`` grows slowly towards
its limit of about 7.3. The displacement over ``N`` steps is at most
``lr * sum_{t<=N} c(t)``, and :func:`adam_step_ceiling` computes that sum:
it stays under ``4 * N`` for every ``N`` up to 862 at the default betas. So
``4 x lr x steps`` is a sound ceiling for runs of that length, and this script
**refuses** a ``--steps`` / ``--betas`` combination where it is not, rather
than reporting a healthy tensor as damaged. ``eps > 0`` only shrinks the step,
and a chained run that restarts its moments at each link restarts ``t``, which
only shrinks ``c``; both make the bound looser, never tighter.

Two conditions it depends on and does not check: weight decay must be zero
(decoupled decay adds ``lr * wd * |w|`` per step) and the learning rate must be
the one given. The trainer runs Adam with ``weight_decay=0.0``.

The rounding allowance is per element: each in-place update is rounded to the
storage dtype, and a rounding moves an element by at most half an ulp of the
value it lands on, which never exceeds ``max(|pre|, |post|) + bound`` for that
element. So element ``i`` is allowed ``bound + steps * half_ulp * (max(|pre_i|,
|post_i|) + bound)``. An earlier version took the largest magnitude anywhere in
the tensor and granted it to every element, so one large element in a bf16
tensor let a zero elsewhere move by far more than Adam can. No separate
absolute floor is needed: near zero a rounding is below the smallest normal
number, orders of magnitude under any ``bound`` a real learning rate gives.

⚠ ``--steps`` is the number of optimiser steps across the *whole* interval
between the two checkpoints, which for a chained run is the sum over links and
not the iteration count of the last one. Passing the last link's count alone
gives a ceiling that is too tight. It is required, because without it the
ceiling cannot be computed and a floor-only pass is exactly the verdict this
script exists to stop issuing.

Why numpy rather than ``safetensors.safe_open``
===============================================
The container format is a little-endian u64 header length, that many bytes of
JSON giving each tensor's dtype, shape and byte span, then the raw buffer --
all of which numpy reads. ``safe_open(framework="pt")`` pulls in torch, and the
point of this script is that it runs anywhere, with no GPU and no training
stack, as the artifact a reader can repeat.

Exit codes
==========
0 every check passed. ``EXIT_FAILED`` (1) a tensor moved further than the
optimiser can move it, or holds a non-finite value: the POST tree is damaged.
``EXIT_INCOMPLETE`` (3) the pair cannot be vouched for -- no frozen control, a
control that moved, a trained tensor that did not move, trees that do not hold
the same tensors, or a ceiling that is not sound for the given steps. Distinct
from 1 because "damaged" and "cannot tell" send a reader to different places,
and from 2, which argparse uses for a usage error.

Usage
-----

    python examples/rl/verify_checkpoint_delta.py <pre-dir> <post-dir> \\
        --lr 1e-6 --steps 17
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

FROZEN = "model.embed_tokens.weight"
EXIT_FAILED = 1
EXIT_INCOMPLETE = 3

#: The multiple of ``lr * steps`` the ceiling allows.
SLACK = 4.0

# numpy has no bfloat16, so BF16 is read as raw u16 and widened by placing the
# bits in the high half of an f32 -- exact, because bf16 *is* the high half of
# an f32.
DTYPES = {"F32": np.float32, "F64": np.float64, "BF16": np.uint16, "F16": np.float16}

#: Half an ulp relative to magnitude, per storage dtype: the most one rounding
#: of an in-place update can move an element beyond the update itself. It is
#: 2^-p for a p-bit significand (implicit bit included): F32 24, F64 53, BF16 8,
#: F16 11.
HALF_ULP = {"F32": 2.0**-24, "F64": 2.0**-53, "BF16": 2.0**-8, "F16": 2.0**-11}


def adam_step_ceiling(steps: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    """``sum_{t<=steps} c(t)``: Adam's worst-case displacement in units of ``lr``.

    See the module docstring for the derivation. Returned in units of ``lr`` so
    it can be compared directly with ``SLACK * steps``.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ratio = beta1 * beta1 / beta2
    scale = (1.0 - beta1) / math.sqrt(1.0 - beta2)
    total = 0.0
    geometric = 0.0
    for t in range(1, steps + 1):
        geometric += ratio ** (t - 1)
        total += (
            scale * math.sqrt(geometric) * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
        )
    return total


def optimiser_bound(lr: float, steps: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    """``SLACK * lr * steps``, refused where it would not bound Adam.

    Raises ``ValueError`` when the worst-case Adam displacement over ``steps``
    exceeds the ceiling, because a ceiling below what the optimiser can
    legitimately do would report healthy tensors as damaged.
    """
    if not (math.isfinite(lr) and lr > 0):
        raise ValueError("lr must be a finite number > 0")
    if not (math.isfinite(beta1) and 0.0 <= beta1 < 1.0):
        raise ValueError("beta1 must be finite and in [0, 1)")
    if not (math.isfinite(beta2) and 0.0 < beta2 < 1.0):
        # The bound divides by beta2: it needs a positive second-moment decay.
        raise ValueError("beta2 must be finite and in (0, 1)")
    worst = adam_step_ceiling(steps, beta1, beta2)
    if worst > SLACK * steps:
        raise ValueError(
            f"{SLACK:g} x lr x steps is not a sound ceiling for {steps} Adam steps at "
            f"betas ({beta1}, {beta2}): the worst case is {worst / steps:.3f} x lr per "
            f"step. Verify shorter intervals, or chained links separately."
        )
    return SLACK * lr * steps


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """(metadata, offset of the data buffer) from a safetensors file."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        meta = json.loads(handle.read(length))
    meta.pop("__metadata__", None)
    return meta, 8 + length


def read_tensor(path: Path, info: dict[str, Any], base: int) -> Any:
    """One tensor as float32, or float64 when it is stored as F64.

    Every narrower dtype widens to float32 exactly. F64 does not narrow: a
    real update below float32's resolution would read as "unmoved", and damage
    below it would be invisible.
    """
    begin, end = info["data_offsets"]
    dtype = DTYPES[info["dtype"]]
    flat = np.fromfile(path, dtype=dtype, count=(end - begin) // np.dtype(dtype).itemsize,
                       offset=base + begin)
    if info["dtype"] == "BF16":
        flat = (flat.astype(np.uint32) << 16).view(np.float32)
    widest = np.float64 if info["dtype"] == "F64" else np.float32
    return flat.astype(widest).reshape(info["shape"])


def raw_identical(path_a: Path, info_a: dict[str, Any], base_a: int,
                  path_b: Path, info_b: dict[str, Any], base_b: int,
                  chunk: int = 1 << 24) -> bool:
    """Whether two stored tensors hold exactly the same bytes.

    The frozen control's claim is bit-identity, which a numeric delta cannot
    make: ``abs(-0.0 - 0.0)`` is zero. Read in chunks so the check costs
    ``chunk`` bytes of memory, not a second copy of the largest tensor.
    """
    begin_a, end_a = info_a["data_offsets"]
    begin_b, end_b = info_b["data_offsets"]
    if info_a["dtype"] != info_b["dtype"] or end_a - begin_a != end_b - begin_b:
        return False
    remaining = end_a - begin_a
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        fa.seek(base_a + begin_a)
        fb.seek(base_b + begin_b)
        while remaining:
            size = min(chunk, remaining)
            block_a, block_b = fa.read(size), fb.read(size)
            if len(block_a) != size or block_a != block_b:
                return False
            remaining -= size
    return True


def element_ceilings(a: Any, b: Any, diff: Any, bound: float, steps: int, half_ulp: float,
                     report: int = 8, chunk: int = 1 << 22) -> dict[str, Any]:
    """Each element's displacement against its own ceiling (see the module docstring).

    ``bound + steps * half_ulp * (max(|a_i|, |b_i|) + bound)`` per element, in
    chunks so the extra arrays cost ``chunk`` elements rather than a copy of
    the tensor. Returns the largest displacement-to-ceiling ratio, the
    ceiling at that element, how many elements exceed theirs, and the flat
    indices of the first ``report`` of them.
    """
    fa, fb, fd = a.reshape(-1), b.reshape(-1), diff.reshape(-1)
    best = {"ratio": 0.0, "ceiling": bound, "elements": 0, "where": []}
    for start in range(0, fd.size, chunk):
        end = start + chunk
        allowed = bound + steps * half_ulp * (
            np.maximum(np.abs(fa[start:end]), np.abs(fb[start:end])) + bound)
        ratio = fd[start:end] / allowed
        top = int(ratio.argmax()) if ratio.size else 0
        if ratio.size and float(ratio[top]) > best["ratio"]:
            best["ratio"], best["ceiling"] = float(ratio[top]), float(allowed[top])
        over = np.flatnonzero(ratio > 1.0)
        best["elements"] += int(over.size)
        room = report - len(best["where"])
        if room > 0:
            best["where"].extend(int(start + i) for i in over[:room])
    return best


def tensor_map(root: Path) -> dict[str, Path]:
    """Every tensor name in a checkpoint directory, mapped to the file holding it.

    Reads ``model.safetensors.index.json`` when the tree is sharded, else the
    single ``model.safetensors`` a small model is saved as. Each tree is read
    through its *own* index, so a PRE and POST sharded differently still pair
    tensor by tensor.
    """
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: root / shard for name, shard in weight_map.items()}
    single = root / "model.safetensors"
    if single.is_file():
        meta, _ = read_header(single)
        return dict.fromkeys(meta, single)
    raise FileNotFoundError(f"{root}: no model.safetensors.index.json or model.safetensors")


class _Reader:
    """Caches one header per file so a sharded tree is not re-parsed per tensor."""

    def __init__(self) -> None:
        self._headers: dict[Path, tuple[dict[str, Any], int]] = {}

    def info(self, path: Path, name: str) -> tuple[dict[str, Any], int]:
        if path not in self._headers:
            self._headers[path] = read_header(path)
        meta, base = self._headers[path]
        return meta[name], base


def compare(
    pre: Path,
    post: Path,
    *,
    bound: float,
    steps: int,
    frozen: str = FROZEN,
    report: int = 8,
    chunk: int = 1 << 24,
) -> dict[str, Any]:
    """Every tensor of PRE against POST. Pure apart from reading the two trees."""
    names_pre, names_post = tensor_map(pre), tensor_map(post)
    only_pre = sorted(set(names_pre) - set(names_post))
    only_post = sorted(set(names_post) - set(names_pre))
    reader = _Reader()
    result: dict[str, Any] = {
        "only_in_pre": only_pre,
        "only_in_post": only_post,
        "mismatched": [],
        "non_finite": [],
        "over": [],
        "moved": 0,
        "unmoved": [],
        "deltas": [],
        "frozen_delta": None,
        "frozen_identical": None,
        "max_ratio": 0.0,
        "dtypes": set(),
    }
    for name in sorted(set(names_pre) & set(names_post)):
        info_a, base_a = reader.info(names_pre[name], name)
        info_b, base_b = reader.info(names_post[name], name)
        if info_a["shape"] != info_b["shape"] or info_a["dtype"] != info_b["dtype"]:
            # Not a pair of the same model. Broadcasting would otherwise
            # compare two different shapes elementwise and report a number.
            result["mismatched"].append(
                f"{name}: {info_a['dtype']}{info_a['shape']} vs {info_b['dtype']}{info_b['shape']}"
            )
            continue
        result["dtypes"].add(info_a["dtype"])
        if name == frozen:
            result["frozen_identical"] = raw_identical(
                names_pre[name], info_a, base_a, names_post[name], info_b, base_b, chunk)
        a = read_tensor(names_pre[name], info_a, base_a)
        b = read_tensor(names_post[name], info_b, base_b)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            # Checked before any max: `max` of an array holding NaN is NaN,
            # and every comparison against NaN is False, so a NaN would read
            # as "did not move" and "within the ceiling" at the same time.
            result["non_finite"].append(name)
            continue
        diff = np.abs(b - a)
        top = float(diff.max()) if diff.size else 0.0
        if name == frozen:
            result["frozen_delta"] = top
            continue
        if top > 0:
            result["moved"] += 1
            result["deltas"].append(top)
        else:
            result["unmoved"].append(name)
        worst = element_ceilings(a, b, diff, bound, steps, HALF_ULP[info_a["dtype"]], report)
        result["max_ratio"] = max(result["max_ratio"], worst["ratio"])
        if worst["elements"]:
            result["over"].append({
                "tensor": name,
                "max_abs_delta": top,
                "ceiling": worst["ceiling"],
                "ratio": worst["ratio"],
                "max_abs_pre": float(np.abs(a).max()),
                "max_abs_post": float(np.abs(b).max()),
                "elements": worst["elements"],
                "where": [
                    [int(i) for i in np.unravel_index(flat, a.shape)] + [float(b.flat[flat])]
                    for flat in worst["where"]
                ],
            })
    result["deltas"].sort()
    return result


def verdict(result: dict[str, Any], frozen: str = FROZEN) -> tuple[int, str]:
    """(exit code, the one line a reader quotes). Fail-closed on every arm."""
    if result["over"]:
        return EXIT_FAILED, (
            f"DAMAGED -- {len(result['over'])} tensor(s) moved further than the "
            "optimiser can move them"
        )
    if result["non_finite"]:
        return EXIT_FAILED, f"DAMAGED -- {len(result['non_finite'])} tensor(s) hold a non-finite value"
    if result["only_in_pre"] or result["only_in_post"] or result["mismatched"]:
        return EXIT_INCOMPLETE, (
            "INCOMPLETE -- the two trees do not hold the same tensors, so they are "
            "not a before/after pair"
        )
    if result.get("frozen_identical") is None:
        return EXIT_INCOMPLETE, f"INCOMPLETE -- the frozen control {frozen} is not in the tree"
    if not result["frozen_identical"]:
        return EXIT_INCOMPLETE, "INCOMPLETE -- the frozen control moved, so the control is broken"
    if result["unmoved"]:
        return EXIT_INCOMPLETE, (
            f"INCOMPLETE -- {len(result['unmoved'])} trained tensor(s) did not move at all"
        )
    if not result["moved"]:
        return EXIT_INCOMPLETE, "INCOMPLETE -- no tensor moved; these may be the same checkpoint"
    return 0, (
        "every trained tensor moved, the frozen control did not, and nothing moved "
        "further than the optimiser can account for"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pre", type=Path)
    parser.add_argument("post", type=Path)
    parser.add_argument("--lr", type=float, required=True,
                        help="the trainer's learning rate")
    parser.add_argument("--steps", type=int, required=True,
                        help="optimiser steps between the two checkpoints, summed "
                             "over every chained link")
    parser.add_argument("--betas", default="0.9,0.999",
                        help="Adam's betas, for the soundness check on the ceiling")
    parser.add_argument("--frozen", default=FROZEN,
                        help="the tensor the trainer froze as a negative control")
    parser.add_argument("--report", type=int, default=8,
                        help="how many offending elements to locate per tensor")
    args = parser.parse_args(argv)

    try:
        beta1, beta2 = (float(x) for x in args.betas.split(","))
    except ValueError:
        parser.error(f"--betas must be two numbers, got {args.betas!r}")
    # Usage errors, not verdicts: a NaN ceiling makes every `top > ceiling`
    # comparison false and an infinite one permits any displacement, so a
    # non-finite input would turn the ceiling into a pass for everything.
    if not (math.isfinite(args.lr) and args.lr > 0):
        parser.error(f"--lr must be a finite number > 0, got {args.lr!r}")
    if args.steps < 1:
        parser.error(f"--steps must be >= 1, got {args.steps}")
    if not (math.isfinite(beta1) and 0.0 <= beta1 < 1.0):
        parser.error(f"--betas: beta1 must be finite and in [0, 1), got {beta1!r}")
    if not (math.isfinite(beta2) and 0.0 < beta2 < 1.0):
        # Open at 0 as well as 1: the ceiling's derivation divides by beta2.
        parser.error(f"--betas: beta2 must be finite and in (0, 1), got {beta2!r}")
    try:
        bound = optimiser_bound(args.lr, args.steps, beta1, beta2)
    except ValueError as exc:
        print(f"INCOMPLETE -- {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE

    result = compare(args.pre, args.post, bound=bound, steps=args.steps,
                     frozen=args.frozen, report=args.report)
    print(f"dtypes on disk            {sorted(result['dtypes'])}")
    print(f"trained tensors moved     {result['moved']}")
    print(f"trained tensors unmoved   {len(result['unmoved'])}")
    for label in ("only_in_pre", "only_in_post", "mismatched", "non_finite"):
        if result[label]:
            print(f"{label:<25} {result[label][:5]}")
    identical, frozen = result["frozen_identical"], result["frozen_delta"]
    if identical is None:
        state = "ABSENT"
    else:
        state = "BIT-IDENTICAL" if identical else "MOVED"
        if frozen is not None:
            state = f"max abs delta {frozen:.3e} ({state})"
    print(f"frozen {args.frozen}: {state}")
    deltas = result["deltas"]
    if deltas:
        print(f"delta min / median / max  {deltas[0]:.4e} / {deltas[len(deltas) // 2]:.4e} "
              f"/ {deltas[-1]:.4e}")
    print(f"optimiser ceiling         {bound:.4e}   ({SLACK:g} x lr {args.lr:g} x "
          f"{args.steps} steps, plus per-element rounding)")
    print(f"max delta / own ceiling   {result['max_ratio']:.4f}")
    for row in sorted(result["over"], key=lambda r: -r["ratio"]):
        print(f"  BEYOND THE CEILING: {row['tensor']}  {row['elements']} element(s), worst at "
              f"{row['ratio']:.4g} x its ceiling of {row['ceiling']:.4g}; max abs delta "
              f"{row['max_abs_delta']:.4g}, max |w| {row['max_abs_pre']:.4g} -> "
              f"{row['max_abs_post']:.4g}")
        for cell in row["where"]:
            print(f"      [{','.join(str(i) for i in cell[:-1])}] = {cell[-1]:+.6g}")
    code, line = verdict(result, args.frozen)
    print()
    print("VERDICT:", line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
