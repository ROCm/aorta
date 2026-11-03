#!/usr/bin/env python3
"""Check that this host's Triton really instruments kernels under FpSan.

Run on a gfx950 host inside the CI image before trusting any ``fpsan-*`` row:

    python3 scripts/sanitizers/fpsan_probe.py --out fpsan-probe.json

Every kernel compiles and runs in its own child process with its own empty
``TRITON_CACHE_DIR``:

* ``mix`` (multiply, divide, add, ``exp``) uninstrumented and under
  ``TRITON_INSTRUMENTATION_MODE=fpsan``. The FpSan cache must pass the same
  instrumentation check ``fpsan_compare.py`` applies, the plain cache must
  fail it, and the two outputs must differ;
* ``rowsum`` (a chunked row reduction) and ``dot`` (a ``tl.dot`` matmul), each
  under FpSan at two reduction tile sizes. Their outputs must be bit-identical:
  the conservative reference config shrinks reduction tiles, which compares
  nothing unless FpSan's payload algebra ignores reduction order.

One JSON summary goes to stdout (and ``--out``): Triton version, device,
whether the build has the FpSan pass, each run's cache check, output digest
and first-launch seconds (compile included, so plain vs. FpSan compile time),
and the verdict. Exit status:

* 0 -- FpSan instruments, changes the output, and is reduction-order invariant;
* 1 -- FpSan is unavailable or did not instrument (bump the CI image);
* 2 -- no torch, Triton or GPU, so nothing was checked;
* 3 -- FpSan instruments, but a reduction-tiling pair disagreed;
* 4 -- the probe itself failed (a child timed out, an unreadable result), so
  it says nothing about FpSan.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

# A broken install fails with more than ImportError (a shared library that will
# not load), and that is still "no Triton", not an FpSan verdict.
try:
    import triton
    import triton.language as tl
except Exception as exc:  # noqa: BLE001
    triton = None
    _TRITON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _TRITON_IMPORT_ERROR = None

EXIT_PROBE_ERROR = 4

_HERE = Path(__file__).resolve().parent
# (kernel, variant, instrumentation mode)
RUNS = (
    ("mix", "a", ""),
    ("mix", "a", "fpsan"),
    ("rowsum", "a", "fpsan"),
    ("rowsum", "b", "fpsan"),
    ("dot", "a", "fpsan"),
    ("dot", "b", "fpsan"),
)
ROWS, COLS = 64, 1024
ROWSUM_BLOCK = {"a": 1024, "b": 64}
DOT_M, DOT_N, DOT_K = 128, 128, 256
DOT_BLOCK_K = {"a": 128, "b": 32}

if triton is not None:

    @triton.jit
    def _mix_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):  # noqa: N803
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x * y + x / (y + 1.0) + tl.exp(x), mask=mask)

    @triton.jit
    def _rowsum_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):  # noqa: N803
        row = tl.program_id(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, n_cols, BLOCK):
            acc += tl.load(x_ptr + row * n_cols + off + tl.arange(0, BLOCK))
        tl.store(out_ptr + row, tl.sum(acc, axis=0))

    @triton.jit
    def _dot_kernel(a_ptr, b_ptr, c_ptr, M, N, K,  # noqa: N803
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):  # noqa: N803
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc = tl.dot(a, b, acc)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))


def _sha256(tensor) -> str:
    import torch

    flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def child(kernel: str, variant: str) -> dict[str, object]:
    import torch

    torch.manual_seed(0)
    if kernel == "mix":
        n = 4096
        x, y = torch.randn(n, device="cuda"), torch.rand(n, device="cuda")
        out = torch.empty_like(x)
        launch = lambda: _mix_kernel[(n // 1024,)](x, y, out, n, BLOCK=1024)  # noqa: E731
    elif kernel == "rowsum":
        x = torch.randn(ROWS, COLS, device="cuda")
        out = torch.empty(ROWS, device="cuda")
        launch = lambda: _rowsum_kernel[(ROWS,)](x, out, COLS, BLOCK=ROWSUM_BLOCK[variant])  # noqa: E731
    else:
        a = torch.randn(DOT_M, DOT_K, device="cuda", dtype=torch.float16)
        b = torch.randn(DOT_K, DOT_N, device="cuda", dtype=torch.float16)
        out = torch.empty(DOT_M, DOT_N, device="cuda", dtype=torch.float16)
        launch = lambda: _dot_kernel[(DOT_M // 64, DOT_N // 64)](  # noqa: E731
            a, b, out, DOT_M, DOT_N, DOT_K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=DOT_BLOCK_K[variant]
        )
    torch.cuda.synchronize()
    start = time.monotonic()
    launch()
    torch.cuda.synchronize()
    return {
        "sha256": _sha256(out),
        "first_launch_seconds": round(time.monotonic() - start, 3),
        "device": torch.cuda.get_device_properties(0).gcnArchName,
    }


def _instrumentation():
    spec = importlib.util.spec_from_file_location("fpsan_compare", _HERE / "fpsan_compare.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.instrumentation


def _has_pass() -> bool | None:
    try:
        from triton._C.libtriton import passes
    except Exception:  # noqa: BLE001
        return None
    return hasattr(getattr(passes, "ttgpuir", None), "add_fp_sanitizer")


def probe(work: Path) -> tuple[int, dict[str, object]]:
    instrumentation = _instrumentation()
    runs: dict[str, dict[str, object]] = {}
    for kernel, variant, mode in RUNS:
        key = f"{kernel}/{variant}/{mode or 'plain'}"
        cache, out = work / key.replace("/", "-"), work / f"{key.replace('/', '-')}.json"
        env = {**os.environ, "TRITON_CACHE_DIR": str(cache), "TRITON_INSTRUMENTATION_MODE": mode}
        env.pop("TRITON_STORE_BINARY_ONLY", None)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "child",
             "--kernel", kernel, "--variant", variant, "--out", str(out)],
            env=env, capture_output=True, text=True, timeout=900,
        )
        run: dict[str, object] = {"returncode": proc.returncode, "cache": instrumentation(cache)}
        if proc.returncode == 0:
            run.update(json.loads(out.read_text(encoding="utf-8")))
        else:
            run["stderr_tail"] = proc.stderr.strip().splitlines()[-5:]
        runs[key] = run
    summary: dict[str, object] = {
        "triton_version": getattr(triton, "__version__", None),
        "fp_sanitizer_pass": _has_pass(),
        "runs": runs,
    }
    if any(r["returncode"] != 0 for k, r in runs.items() if k.endswith("/plain")):
        summary["verdict"] = "no_gpu: the uninstrumented kernel did not run"
        return 2, summary
    plain, fpsan = runs["mix/a/plain"], runs["mix/a/fpsan"]
    summary["device"] = plain.get("device")
    summary["compile_seconds"] = {
        "plain": plain.get("first_launch_seconds"), "fpsan": fpsan.get("first_launch_seconds")
    }
    instrumented = fpsan["returncode"] == 0 and fpsan["cache"]["instrumented"]
    summary["instrumented"] = instrumented
    summary["plain_instrumented"] = plain["cache"]["instrumented"]
    summary["output_differs"] = fpsan.get("sha256") not in (None, plain.get("sha256"))
    if not instrumented or plain["cache"]["instrumented"] or not summary["output_differs"]:
        summary["verdict"] = "fpsan_unavailable: the FpSan compile was not instrumented or did not change the output"
        return 1, summary
    invariant = {}
    for kernel in ("rowsum", "dot"):
        a, b = runs[f"{kernel}/a/fpsan"], runs[f"{kernel}/b/fpsan"]
        invariant[kernel] = a["returncode"] == b["returncode"] == 0 and a.get("sha256") == b.get("sha256")
    summary["tiling_invariant"] = invariant
    if not all(invariant.values()):
        summary["verdict"] = "fpsan_tiling_variant: a reduction-tiling pair disagreed under FpSan"
        return 3, summary
    summary["verdict"] = "fpsan_available"
    return 0, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    kid = sub.add_parser("child")
    kid.add_argument("--kernel", choices=("mix", "rowsum", "dot"), required=True)
    kid.add_argument("--variant", choices=("a", "b"), required=True)
    kid.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="also write the summary here")
    parser.add_argument("--work", type=Path, help="keep the caches here (default: a temp dir)")
    args = parser.parse_args(argv)
    if args.command == "child":
        args.out.write_text(json.dumps(child(args.kernel, args.variant)) + "\n", encoding="utf-8")
        return 0
    if triton is None or importlib.util.find_spec("torch") is None:
        code, summary = 2, {"verdict": "no_triton: torch or triton is not importable"}
        if _TRITON_IMPORT_ERROR:
            summary["triton_import_error"] = _TRITON_IMPORT_ERROR
    else:
        # Uncaught, a crash here exits 1, which reads as "FpSan unavailable".
        try:
            if args.work:
                args.work.mkdir(parents=True, exist_ok=True)
                code, summary = probe(args.work.resolve())
            else:
                with tempfile.TemporaryDirectory(prefix="fpsan-probe-") as tmp:
                    code, summary = probe(Path(tmp))
        except Exception as exc:  # noqa: BLE001
            code, summary = EXIT_PROBE_ERROR, {"verdict": f"probe_error: {type(exc).__name__}: {exc}"}
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
