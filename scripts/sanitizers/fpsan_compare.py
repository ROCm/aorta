#!/usr/bin/env python3
"""Run one asset kernel and its reference under Triton FpSan and print one result line.

``model_survey.py`` writes one shim per (kernel, scenario) that runs this as the
recipe's ``source.fpsan_command``. For that pair it:

1. writes a reference copy of the scenario spec. By default the reference is
   the same kernel and scenario at a conservative launch config:
   ``num_warps=4``, ``num_stages=1``, no AMD launch kwargs, plus any
   ``--reference-constexpr`` (smaller tiles) and ``--reference-grid``. With
   ``--reference-kernel`` it is another entry of the same spec at the same
   scenario (the Triton tutorial pairs);
2. runs the reference and then the candidate through the asset's own
   ``launch.py --mode run``, unchanged, each in a child process with its own
   empty ``TRITON_CACHE_DIR``, ``TRITON_INSTRUMENTATION_MODE=fpsan`` and the
   same ``--seed``. The child imports ``launch.py``, wraps its
   ``load_kernel``, records every ``fn[grid](...)`` launch, calls its
   ``main()``, and then SHA-256s every tensor argument of the one launch, in
   argument order: kernels write their outputs in place;
3. reads each cache: a side is instrumented only when every object it
   compiled records ``instrumentation_mode`` ``fpsan`` for an FpSan target and
   its TTGIR carries FpSan's embed/unembed ops. Triton 3.6 and 3.7 record the
   mode but have no AMD FpSan pass, so the mode alone proves nothing;
4. prints exactly one ``AORTA_FPSAN_RESULT`` line and exits 0, mismatch
   included.

Anything that makes the comparison untrustworthy prints no result line and
exits 3, so aorta reports an error and never a pass: ``load_kernel`` never
called, zero launches or more than one, a reference identical to the
candidate, a device-pointer table among the arguments, tensor arguments whose
dtypes or shapes differ between the sides, a side that failed or timed out.

    fpsan_compare.py --launch <model>/roofline/harness/launch.py \\
        --spec <model>/roofline/specs/moe.json --kernel _matmul_ogs --scenario P1 \\
        --flat-dir <model>/kernels --pkg-src <model>/pkg_src --work <dir>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

RESULT_MARKER = "AORTA_FPSAN_RESULT"
INSTRUMENTATION_MODE = "fpsan"
# The AMD targets Triton's backend runs the FpSan pass for.
FPSAN_ARCHES = frozenset({"gfx942", "gfx950", "gfx1250"})
# Ops the FpSan pass leaves in TTGIR; lowered away only in make_llir.
FPSAN_TTGIR_OP = "tti.experimental_fpsan_"
REFERENCE_NUM_WARPS = 4
REFERENCE_NUM_STAGES = 1
EXIT_NO_RESULT = 3
# A stale override or interpreter would compile something other than the
# kernel source, and binary-only caching drops the TTGIR the check reads.
_DROPPED_ENV = (
    "TRITON_INTERPRET",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_OVERRIDE_DIR",
    "TRITON_STORE_BINARY_ONLY",
    "TRITON_ALWAYS_COMPILE",
)
_SIDES = ("reference", "candidate")


class CompareError(RuntimeError):
    """The comparison cannot be trusted; no result line is printed."""


# ------------------------------------------------------------ reference spec


def _entry(entries: Sequence[object], kernel: str, scenario: str) -> dict:
    entry = next(
        (e for e in entries if isinstance(e, dict) and e.get("kernel") == kernel), None
    )
    if entry is None:
        raise CompareError(f"fpsan_spec_mismatch: kernel {kernel!r} is not in the spec")
    scenarios = entry.get("scenarios")
    if not isinstance(scenarios, dict) or not isinstance(scenarios.get(scenario), dict):
        raise CompareError(f"fpsan_spec_mismatch: {kernel!r} has no scenario {scenario!r}")
    if not isinstance(scenarios[scenario].get("launch"), dict):
        raise CompareError(f"fpsan_spec_mismatch: {kernel}/{scenario} has no launch")
    return entry


def _launch_args(launch: Mapping[str, object]) -> Iterator[dict]:
    for arg in launch.get("args") or []:
        if not isinstance(arg, dict):
            continue
        yield arg
        if arg.get("kind") == "tensor_tuple":
            yield from (v for v in arg.get("value") or [] if isinstance(v, dict))


def _pointer_tables(launch: Mapping[str, object]) -> list[str]:
    return [
        str(arg.get("name"))
        for arg in _launch_args(launch)
        if str(arg.get("init", "")).startswith("ptr_of:")
    ]


def reference_spec(
    entries: Sequence[object],
    *,
    kernel: str,
    scenario: str,
    reference_kernel: str | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    constexprs: Mapping[str, object] | None = None,
    grid: Sequence[int] | None = None,
) -> tuple[list[dict], str, list[str]]:
    """The one-entry reference spec, the kernel name it runs, and what changed.

    Raises ``CompareError`` when the reference would compare nothing (same
    kernel, same launch), names a constexpr the launch does not pass, keeps
    split-K on, or passes a device-pointer table whose targets are not hashed.
    """

    candidate = _entry(entries, kernel, scenario)
    cand_launch = candidate["scenarios"][scenario]["launch"]
    changes: list[str] = []
    if reference_kernel is not None and reference_kernel != kernel:
        base = _entry(entries, reference_kernel, scenario)
        changes.append(f"kernel {kernel} -> {reference_kernel}")
        warps, stages = num_warps, num_stages
    else:
        base, reference_kernel = candidate, kernel
        warps = REFERENCE_NUM_WARPS if num_warps is None else num_warps
        stages = REFERENCE_NUM_STAGES if num_stages is None else num_stages
    base_scenario = base["scenarios"][scenario]
    launch = copy.deepcopy(base_scenario["launch"])
    for key, value in (("num_warps", warps), ("num_stages", stages)):
        if value is not None and launch.get(key) != value:
            changes.append(f"{key} {launch.get(key)} -> {value}")
            launch[key] = value
    if reference_kernel == kernel:
        for key in ("amd_kwargs", "waves_per_eu"):
            if launch.pop(key, None):
                changes.append(f"{key} dropped")
    by_name = {
        arg.get("name"): arg for arg in launch.get("args") or [] if isinstance(arg, dict)
    }
    for name, value in (constexprs or {}).items():
        arg = by_name.get(name)
        if arg is None or arg.get("kind") != "constexpr":
            raise CompareError(
                f"fpsan_reference_invalid: {reference_kernel}/{scenario} passes no constexpr {name!r}"
            )
        if arg.get("value") != value:
            changes.append(f"{name} {arg.get('value')} -> {value}")
            arg["value"] = value
    if grid is not None and list(launch.get("grid") or []) != list(grid):
        changes.append(f"grid {launch.get('grid')} -> {list(grid)}")
        launch["grid"] = list(grid)
    split_k = by_name.get("SPLIT_K")
    if split_k is not None and split_k.get("value") not in (None, 1):
        raise CompareError(
            f"fpsan_reference_invalid: the reference keeps SPLIT_K={split_k.get('value')}; "
            "set SPLIT_K=1 with a matching grid"
        )
    if not changes:
        raise CompareError(
            f"fpsan_reference_identical: {kernel}/{scenario} already launches with the "
            "reference config, so the two sides would compare one kernel with itself; "
            "give it a reference override in the inventory"
        )
    tables = _pointer_tables(cand_launch) + _pointer_tables(launch)
    if tables:
        raise CompareError(
            f"fpsan_pointer_table: {kernel}/{scenario} passes device-pointer tables "
            f"{sorted(set(tables))}; outputs written through them are not hashed"
        )
    entry = {**base, "scenarios": {scenario: {**base_scenario, "launch": launch}}}
    return [entry], reference_kernel, changes


# ------------------------------------------------------------ instrumentation


def instrumentation(cache: Path) -> dict[str, object]:
    """Whether every object compiled into ``cache`` was built under FpSan."""

    objects = []
    for obj in sorted(cache.rglob("*.hsaco")):
        try:
            meta = json.loads(obj.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        modes = {m.strip() for m in str(meta.get("instrumentation_mode") or "").split(",")}
        target = meta.get("target")
        arch = target.get("arch") if isinstance(target, dict) else None
        ttgir = obj.with_suffix(".ttgir")
        try:
            ops = ttgir.read_text(encoding="utf-8").count(FPSAN_TTGIR_OP)
        except OSError:
            ops = 0
        objects.append({
            "object": obj.name,
            "name": meta.get("name"),
            "instrumentation_mode": meta.get("instrumentation_mode"),
            "arch": arch,
            "triton_version": meta.get("triton_version"),
            "fpsan_ttgir_ops": ops,
            "instrumented": INSTRUMENTATION_MODE in modes
            and str(arch).split(":", 1)[0] in FPSAN_ARCHES
            and ops > 0,
        })
    return {
        "objects": objects,
        "instrumented": bool(objects) and all(o["instrumented"] for o in objects),
    }


# ------------------------------------------------------------ child side


def _import_launch(launch: Path):
    spec = importlib.util.spec_from_file_location("aorta_fpsan_launch", launch)
    if spec is None or spec.loader is None:
        raise CompareError(f"fpsan_hook_missing: cannot import {launch}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load_kernel", "main"):
        if not callable(getattr(module, name, None)):
            raise CompareError(
                f"fpsan_hook_missing: {launch.name} defines no {name}(); the harness changed"
            )
    return module


def _tensors(args: Sequence[object], kwargs: Mapping[str, object]) -> Iterator[tuple[str, object]]:
    import torch

    def walk(name: str, value: object) -> Iterator[tuple[str, object]]:
        if isinstance(value, torch.Tensor):
            yield name, value
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from walk(f"{name}[{index}]", item)

    for index, value in enumerate(args):
        yield from walk(f"arg{index}", value)
    for name, value in kwargs.items():
        yield from walk(name, value)


def digest_tensors(args: Sequence[object], kwargs: Mapping[str, object]) -> tuple[str, list[dict]]:
    """One SHA-256 over every tensor argument, in order, and each tensor's own.

    Names are kept out of the combined digest: a reference kernel may call the
    same buffer by another parameter name.
    """

    import torch

    combined = hashlib.sha256()
    tensors = []
    for name, tensor in _tensors(args, kwargs):
        flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
        dtype, shape = str(tensor.dtype), list(tensor.shape)
        combined.update(f"{dtype}|{shape}|{digest}\n".encode())
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "sha256": digest})
    if not tensors:
        raise CompareError("fpsan_no_tensor_arguments: the launch passed no tensor to hash")
    return combined.hexdigest(), tensors


def run_side(args: argparse.Namespace) -> dict[str, object]:
    """Run ``launch.py`` in this process with both hooks and hash the one launch."""

    launch = Path(args.launch).resolve()
    module = _import_launch(launch)
    loads: list[str] = []
    original_load = module.load_kernel

    def load_kernel(*a: object, **kw: object) -> object:
        loaded = original_load(*a, **kw)
        loads.append(str(a[1] if len(a) > 1 else kw.get("name")))
        return loaded

    module.load_kernel = load_kernel
    # launch.py may replace the loaded kernel (triton_kernels.specialize) before
    # launching it, so launches are recorded at Triton's fn[grid] entry point.
    from triton.runtime import jit

    launches: list[tuple[tuple, dict]] = []
    original_getitem = jit.KernelInterface.__getitem__

    def getitem(self: object, grid: object) -> object:
        launcher = original_getitem(self, grid)

        def launch_and_record(*a: object, **kw: object) -> object:
            result = launcher(*a, **kw)
            launches.append((a, kw))
            return result

        return launch_and_record

    sys.argv = [
        str(launch), str(Path(args.spec).resolve()),
        "--kernel", args.kernel, "--scenario", args.scenario, "--mode", "run",
        "--arch", args.arch, "--warp-size", str(args.warp_size), "--iters", "1",
        "--out", str(Path(args.out).resolve()), "--flat-dir", str(Path(args.flat_dir).resolve()),
        "--seed", str(args.seed),
    ]
    jit.KernelInterface.__getitem__ = getitem
    try:
        module.main()
    finally:
        jit.KernelInterface.__getitem__ = original_getitem
    if not loads:
        raise CompareError(
            "fpsan_hook_not_fired: launch.py never called load_kernel; the harness changed"
        )
    if len(launches) != 1:
        raise CompareError(
            f"fpsan_launch_count: launch.py launched {len(launches)} kernels through "
            "fn[grid](...), expected exactly one"
        )
    import torch

    torch.cuda.synchronize()
    digest, tensors = digest_tensors(*launches[0])
    return {"load_kernel_calls": len(loads), "launches": len(launches), "digest": digest,
            "tensors": tensors}


def side_main(args: argparse.Namespace) -> int:
    try:
        record = run_side(args)
    except CompareError as exc:
        print(f"fpsan_compare: {exc}", file=sys.stderr)
        return EXIT_NO_RESULT
    Path(args.out_json).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------ parent


def _side_env(cache: Path, pkg_src: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED_ENV}
    env["TRITON_CACHE_DIR"] = str(cache)
    env["TRITON_INSTRUMENTATION_MODE"] = INSTRUMENTATION_MODE
    if pkg_src:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (pkg_src, env.get("PYTHONPATH")) if p)
    return env


def launch_side(
    side: str,
    *,
    spec: Path,
    kernel: str,
    args: argparse.Namespace,
    work: Path,
) -> dict[str, object]:
    """Run one side in a child process and return its hashes and cache check."""

    side_dir = work / side
    cache = side_dir / "triton-cache"
    out_json = side_dir / "side.json"
    side_dir.mkdir(parents=True)
    argv = [
        sys.executable, str(Path(__file__).resolve()), "side",
        "--launch", args.launch, "--spec", str(spec), "--kernel", kernel,
        "--scenario", args.scenario, "--flat-dir", args.flat_dir, "--arch", args.arch,
        "--warp-size", str(args.warp_size), "--seed", str(args.seed),
        "--out", str(side_dir / "run"), "--out-json", str(out_json),
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=Path(args.launch).resolve().parent.parent,
            env=_side_env(cache, args.pkg_src),
            capture_output=True,
            text=True,
            timeout=args.side_timeout or None,
        )
    except subprocess.TimeoutExpired as exc:
        _forward(side, exc.stdout, exc.stderr)
        raise CompareError(
            f"fpsan_{side}_timed_out: the {side} run exceeded {args.side_timeout}s"
        ) from exc
    _forward(side, proc.stdout, proc.stderr)
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        raise CompareError(f"fpsan_{side}_failed: exit {proc.returncode}: {tail[:300]}")
    try:
        record = json.loads(out_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompareError(f"fpsan_{side}_failed: no side record: {exc}") from exc
    return {**record, "kernel": kernel, "seconds": round(time.monotonic() - start, 1),
            "cache": instrumentation(cache)}


def _forward(side: str, *streams: object) -> None:
    """Child output goes to stderr: stdout carries the one result line only."""

    for stream in streams:
        text = stream.decode(errors="replace") if isinstance(stream, bytes) else stream
        for line in (text or "").splitlines():
            print(f"[{side}] {line}", file=sys.stderr)


def compare(args: argparse.Namespace) -> dict[str, object]:
    work = Path(args.work).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    spec = Path(args.spec).resolve()
    try:
        entries = json.loads(spec.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompareError(f"fpsan_spec_mismatch: {spec}: {exc}") from exc
    if not isinstance(entries, list):
        raise CompareError(f"fpsan_spec_mismatch: {spec} is not a JSON list")
    ref_entries, ref_kernel, changes = reference_spec(
        entries,
        kernel=args.kernel,
        scenario=args.scenario,
        reference_kernel=args.reference_kernel,
        num_warps=args.reference_num_warps,
        num_stages=args.reference_num_stages,
        constexprs=dict(args.reference_constexpr),
        grid=args.reference_grid,
    )
    ref_spec = work / "reference-spec.json"
    ref_spec.write_text(json.dumps(ref_entries, indent=1) + "\n", encoding="utf-8")
    print(f"fpsan_compare: reference {ref_kernel}: {'; '.join(changes)}", file=sys.stderr)
    sides = {
        "reference": launch_side("reference", spec=ref_spec, kernel=ref_kernel, args=args, work=work),
        "candidate": launch_side("candidate", spec=spec, kernel=args.kernel, args=args, work=work),
    }
    layouts = {side: [(t["dtype"], t["shape"]) for t in sides[side]["tensors"]] for side in _SIDES}
    if layouts["reference"] != layouts["candidate"]:
        raise CompareError(
            "fpsan_tensor_layout_mismatch: the reference and candidate launches pass "
            "tensors of different dtypes or shapes, so their digests are not comparable"
        )
    return {
        "reference": {"kernel": ref_kernel, "changes": changes},
        "sides": sides,
        "differing_tensors": [
            ref["name"]
            for ref, cand in zip(sides["reference"]["tensors"], sides["candidate"]["tensors"], strict=True)
            if ref["sha256"] != cand["sha256"]
        ],
        "result": {
            "reference_digest": sides["reference"]["digest"],
            "candidate_digest": sides["candidate"]["digest"],
            "reference_instrumented": sides["reference"]["cache"]["instrumented"],
            "candidate_instrumented": sides["candidate"]["cache"]["instrumented"],
            "target": args.arch,
        },
    }


def result_line(result: Mapping[str, object]) -> str:
    return f"{RESULT_MARKER} {json.dumps(dict(result), sort_keys=True)}"


def _constexpr(text: str) -> tuple[str, object]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected NAME=JSON, got {text!r}")
    try:
        return name, json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name}: {value!r} is not JSON") from exc


def _grid(text: str) -> list[int]:
    try:
        grid = [int(g) for g in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected X,Y,Z, got {text!r}") from exc
    if len(grid) != 3 or min(grid) <= 0:
        raise argparse.ArgumentTypeError(f"expected three positive sizes, got {text!r}")
    return grid


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--launch", required=True, help="the asset's roofline/harness/launch.py")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--flat-dir", required=True, help="the asset's <model>/kernels")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--warp-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["side"]:
        side = argparse.ArgumentParser(prog="fpsan_compare.py side")
        _common(side)
        side.add_argument("--out", required=True)
        side.add_argument("--out-json", required=True)
        return side_main(side.parse_args(argv[1:]))

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common(parser)
    parser.add_argument("--pkg-src", help="prepended to PYTHONPATH for both sides")
    parser.add_argument("--work", required=True, help="scratch dir; emptied first")
    parser.add_argument("--record", help="write what ran, both sides' hashes and cache checks here")
    parser.add_argument("--side-timeout", type=float, default=0.0,
                        help="seconds per side; 0 waits without a deadline")
    parser.add_argument("--reference-kernel", help="another entry of --spec to use as the reference")
    parser.add_argument("--reference-num-warps", type=int)
    parser.add_argument("--reference-num-stages", type=int)
    parser.add_argument("--reference-constexpr", type=_constexpr, action="append", default=[],
                        metavar="NAME=JSON", help="override one constexpr argument of the reference")
    parser.add_argument("--reference-grid", type=_grid, metavar="X,Y,Z")
    args = parser.parse_args(argv)

    start = time.monotonic()
    record: dict[str, object] = {"kernel": args.kernel, "scenario": args.scenario}
    try:
        record.update(compare(args))
    except CompareError as exc:
        record["error"] = str(exc)
        print(f"fpsan_compare: {exc}", file=sys.stderr)
    record["seconds"] = round(time.monotonic() - start, 1)
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if "error" in record:
        return EXIT_NO_RESULT
    if record["differing_tensors"]:
        print(f"fpsan_compare: digests differ in {record['differing_tensors']}", file=sys.stderr)
    sys.stdout.write(result_line(record["result"]) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
