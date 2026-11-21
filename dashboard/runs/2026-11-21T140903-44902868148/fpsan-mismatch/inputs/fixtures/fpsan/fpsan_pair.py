#!/usr/bin/env python3
"""A tiny Triton reference/candidate pair for the daily FpSan controls.

Every kernel here computes ``out[row] = sum_k x[row, k] * w[k]``. The reference
reduces ``K`` in 16-wide tiles. The ``match`` candidate is ``row_dot`` re-tiled to
64: FpSan's payload algebra makes add and mul exactly associative and
commutative, so the re-tiled reduction gives the same sanitized output and the
control passes. The ``mismatch`` candidate, ``row_dot_no_accumulate``, overwrites
its accumulator each tile instead of adding to it, so only the last tile
survives and the digests differ.

``fpsan_pair.py match|mismatch`` runs each side in its own child process with an
empty ``TRITON_CACHE_DIR`` and ``TRITON_INSTRUMENTATION_MODE=fpsan``. A side is
instrumented only when every object in its cache records ``instrumentation_mode``
``fpsan`` and its TTGIR carries FpSan's ops; Triton releases that record the
mode without an AMD FpSan pass would otherwise read as instrumented. It then
prints exactly one ``AORTA_FPSAN_RESULT`` line and exits 0, mismatch included.
No GPU, a failed side, or sides compiled for different targets print no line
and exit 3, so aorta reports an error rather than a verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RESULT_MARKER = "AORTA_FPSAN_RESULT"
FPSAN_TTGIR_OP = "tti.experimental_fpsan_"
EXIT_NO_RESULT = 3
ROWS = 32
K = 128
SEED = 0
REFERENCE = ("row_dot", 16)
CANDIDATES = {
    "match": ("row_dot", 64),
    "mismatch": ("row_dot_no_accumulate", 64),
}
# An interpreter or kernel override would run something other than this source,
# and binary-only caching drops the TTGIR the instrumentation check reads.
_DROPPED_ENV = (
    "TRITON_INTERPRET",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_OVERRIDE_DIR",
    "TRITON_STORE_BINARY_ONLY",
    "TRITON_ALWAYS_COMPILE",
)


# A broken install raises more than ImportError; it still means no result.
try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None

# Module level: Triton resolves a kernel's names through its module globals.
if triton is not None:

    @triton.jit
    def row_dot(x_ptr, w_ptr, out_ptr, k_dim: tl.constexpr, block_k: tl.constexpr):
        row = tl.program_id(0)
        acc = tl.zeros((block_k,), dtype=tl.float32)
        for k in range(0, k_dim, block_k):
            offs = k + tl.arange(0, block_k)
            acc += tl.load(x_ptr + row * k_dim + offs) * tl.load(w_ptr + offs)
        tl.store(out_ptr + row, tl.sum(acc, axis=0))

    @triton.jit
    def row_dot_no_accumulate(x_ptr, w_ptr, out_ptr, k_dim: tl.constexpr, block_k: tl.constexpr):
        row = tl.program_id(0)
        acc = tl.zeros((block_k,), dtype=tl.float32)
        for k in range(0, k_dim, block_k):
            offs = k + tl.arange(0, block_k)
            acc = tl.load(x_ptr + row * k_dim + offs) * tl.load(w_ptr + offs)
        tl.store(out_ptr + row, tl.sum(acc, axis=0))


def run_side(kernel: str, block_k: int, out_json: Path) -> int:
    """Launch one kernel once on seeded inputs and record the SHA-256 of its output."""
    if triton is None:
        print("fpsan_pair: triton is not importable", file=sys.stderr)
        return EXIT_NO_RESULT
    import torch

    if not torch.cuda.is_available():
        print("fpsan_pair: no GPU visible to torch", file=sys.stderr)
        return EXIT_NO_RESULT
    kernels = {"row_dot": row_dot, "row_dot_no_accumulate": row_dot_no_accumulate}
    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(ROWS, K, generator=generator, dtype=torch.float32).cuda()
    w = torch.randn(K, generator=generator, dtype=torch.float32).cuda()
    out = torch.zeros(ROWS, dtype=torch.float32, device="cuda")
    kernels[kernel][(ROWS,)](x, w, out, k_dim=K, block_k=block_k)
    torch.cuda.synchronize()
    digest = hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
    out_json.write_text(json.dumps({"kernel": kernel, "digest": digest}) + "\n", encoding="utf-8")
    return 0


def instrumentation(cache: Path) -> tuple[bool, set[str]]:
    """Whether every object compiled into ``cache`` was built under FpSan, and for which arches."""
    instrumented = []
    arches = set()
    for obj in sorted(cache.rglob("*.hsaco")):
        try:
            meta = json.loads(obj.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        modes = {m.strip() for m in str(meta.get("instrumentation_mode") or "").split(",")}
        target = meta.get("target")
        arch = str(target.get("arch") if isinstance(target, dict) else "").split(":", 1)[0]
        arches.add(arch)
        try:
            ops = obj.with_suffix(".ttgir").read_text(encoding="utf-8").count(FPSAN_TTGIR_OP)
        except OSError:
            ops = 0
        instrumented.append("fpsan" in modes and ops > 0)
    return bool(instrumented) and all(instrumented), arches


def _side(side: str, kernel: str, block_k: int, work: Path) -> tuple[str, bool, set[str]]:
    side_dir = work / side
    cache = side_dir / "triton-cache"
    cache.mkdir(parents=True)
    out_json = side_dir / "side.json"
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED_ENV}
    env["TRITON_CACHE_DIR"] = str(cache)
    env["TRITON_INSTRUMENTATION_MODE"] = "fpsan"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "side",
            kernel,
            str(block_k),
            str(out_json),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        print(f"[{side}] {line}", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"the {side} side ({kernel}) exited {proc.returncode}")
    digest = json.loads(out_json.read_text(encoding="utf-8"))["digest"]
    instrumented, arches = instrumentation(cache)
    return digest, instrumented, arches


def compare(variant: str) -> int:
    with tempfile.TemporaryDirectory(prefix=f"fpsan-{variant}-") as tmp:
        work = Path(tmp)
        try:
            reference = _side("reference", *REFERENCE, work)
            candidate = _side("candidate", *CANDIDATES[variant], work)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            print(f"fpsan_pair: {exc}", file=sys.stderr)
            return EXIT_NO_RESULT
    arches = reference[2] | candidate[2]
    if len(arches) != 1 or not next(iter(arches)):
        print(
            f"fpsan_pair: sides compiled for {sorted(arches)}, expected one target", file=sys.stderr
        )
        return EXIT_NO_RESULT
    result = {
        "reference_digest": reference[0],
        "candidate_digest": candidate[0],
        "reference_instrumented": reference[1],
        "candidate_instrumented": candidate[1],
        "target": next(iter(arches)),
    }
    print(f"{RESULT_MARKER} {json.dumps(result, sort_keys=True)}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[0] == "side":
        return run_side(argv[1], int(argv[2]), Path(argv[3]))
    if len(argv) == 1 and argv[0] in CANDIDATES:
        return compare(argv[0])
    print(f"usage: fpsan_pair.py {{{'|'.join(CANDIDATES)}}}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
