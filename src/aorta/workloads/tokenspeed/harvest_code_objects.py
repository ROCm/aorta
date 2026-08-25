#!/usr/bin/env python3
"""Harvest TokenSpeed kernel code objects and emit a mode: sanitizer recipe.

TokenSpeed's performance kernels are Gluon/Triton, so they do not exist as
committed binaries -- they are JIT-compiled on first use into the Triton cache.
aorta's sanitizer pipeline, meanwhile, consumes code objects by path plus a
SHA-256 identity (``source.kind: kernel_list``). This script bridges the two:
run the kernel once with a clean cache, collect the resulting ``.hsaco``,
inventory it, and write a ready-to-run recipe pointing at what was collected.

The output is a *run area*, not something to commit. A harvested object is
specific to the image, the GPU target, and the shapes that were compiled, so
pinning one in git would assert a provenance it does not have. Re-harvest
instead; it takes seconds.

Compilation can be driven two ways. ``--kernel`` / ``--op`` use TokenSpeed's
benchmark harness, which is precise but only reaches ``gemm.mm`` -- every other
family lacks a registered input generator. ``--pytest-suite`` runs one of
TokenSpeed's own op test suites instead; those build their inputs directly, so
they are the only way to compile the attention, MoE, quantization, sampling and
transform kernels, and therefore the only way to get them into a Waitcheck
corpus today.

Usage (on the compute node, with the image already pulled):

  # gemm, via the benchmark harness
  python3 harvest_code_objects.py \
      --image lightseekorg/tokenspeed-amd:nightly-20260714 \
      --kernel gluon_mm_a16w16_gfx950 --dtype bf16 --dtype-role a \
      --dest /tmp/ts-work/sanitizer-run \
      --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck"

  # attention, via TokenSpeed's own tests
  python3 harvest_code_objects.py \
      --image lightseekorg/tokenspeed-amd:nightly-20260714 \
      --pytest-suite tokenspeed-kernel/test/ops/test_attention.py \
      --pytest-k mha_prefill \
      --dest /tmp/ts-work/attn-run \
      --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck"

Then:

  aorta sweep run --recipe /tmp/ts-work/sanitizer-run/waitcheck-<label>.yaml \
      --output-dir /tmp/ts-work/sanitizer-out
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _suite_root(suite: str) -> str:
    """The in-container directory a pytest suite must run from.

    Each suite is a package rooted at the parent of its ``test`` directory, with
    its own conftest and a top-level ``utils`` module; collection fails from
    anywhere else.
    """
    parts = Path(suite).parts
    if "test" in parts:
        return str(Path(*parts[: parts.index("test")]))
    return "."


def _driver(args: argparse.Namespace) -> tuple[list[str], str, str]:
    """The in-container command that compiles the kernels, plus workdir + label.

    Two ways in. The benchmark harness is precise but only reaches gemm.mm,
    because every other family lacks an input generator. TokenSpeed's own pytest
    suites build their inputs directly, so they compile the attention, MoE,
    quantization, sampling and transform kernels the harness cannot -- which is
    the only way to get those into a Waitcheck corpus today.
    """
    if args.pytest_suite:
        suite = args.pytest_suite
        # Absolute, because the workdir below is the suite's package root rather
        # than /workspace -- a path relative to /workspace would not resolve
        # from there.
        if not suite.startswith("/"):
            suite = f"/workspace/{suite}"
        cmd = [
            "python3",
            "-m",
            "pytest",
            suite,
            "-q",
            "--no-header",
            # The source tree is read-only for a non-root container user and the
            # cache plugin turns that into per-run warning noise.
            "-p",
            "no:cacheprovider",
        ]
        if args.pytest_k:
            cmd += ["-k", args.pytest_k]
        workdir = f"/workspace/{_suite_root(args.pytest_suite)}".rstrip("/")
        label = Path(args.pytest_suite).stem or "suite"
        return cmd, workdir, label

    selector = [args.kernel] if args.kernel else ["--op", args.op]
    cmd = [
        "python3",
        "-m",
        "tokenspeed_kernel.benchmark",
        *selector,
        "--dtype",
        args.dtype,
        "--dtype-role",
        args.dtype_role,
        # Minimal iterations: this run exists to trigger compilation, not to
        # measure anything. Verification stays off for the same reason.
        "--no-verify",
        "--warmup-iters",
        "1",
        "--bench-iters",
        "1",
    ]
    label = args.kernel or (args.op or "kernels").replace(".", "_")
    return cmd, "/workspace", label


def _assert_cache_writable(cache_dir: Path) -> None:
    """Fail early if the JIT cache is not writable by this user.

    Worth its own check because the downstream symptom is actively misleading.
    Triton initialises its AMD driver by compiling a small HIP utility into this
    directory; if that write fails, TokenSpeed catches the error with a bare
    ``except BaseException`` and re-raises "Triton is not supported on the
    current platform", which sends you looking at the GPU instead of at a
    permission bit. The usual cause is docker having auto-created the mount
    point as root because the path did not exist on this node.
    """
    if not cache_dir.is_dir():
        raise SystemExit(f"harvest: cache dir {cache_dir} was not created")
    probe = cache_dir / ".write-probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"harvest: cache dir {cache_dir} is not writable by uid {os.getuid()} "
            f"({exc}). If docker created it as root, remove it and re-run on the "
            "node that will execute the container -- /tmp is per-node."
        ) from exc


def _run_kernel(args: argparse.Namespace, cache_dir: Path) -> None:
    """Compile the kernels in-container so the JIT populates cache_dir."""
    _assert_cache_writable(cache_dir)
    driver, workdir, _ = _driver(args)
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        # Some MoE and attention tests move data over shared memory; the 64MB
        # docker default makes them die with an opaque bus error.
        "--ipc=host",
        "--shm-size=16g",
        "-w",
        workdir,
        # Run as the calling user, not the image's root. As root the JIT writes
        # a root-owned cache, and the caller then cannot delete its own run area
        # -- including the rmtree in main(), so a second harvest fails with
        # EPERM. USER is set because Triton calls getpass.getuser(), which falls
        # through to pwd.getpwuid() and raises for a uid that is absent from the
        # container's /etc/passwd. HOME points into the mount so anything else
        # wanting a home directory gets a writable one.
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"USER={getpass.getuser()}",
        "-e",
        "HOME=/triton-cache",
        "-e",
        f"HIP_VISIBLE_DEVICES={args.gpus}",
        "-e",
        "TRITON_CACHE_DIR=/triton-cache",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--group-add",
        "video",
        "--group-add",
        "render",
        "--security-opt",
        "seccomp=unconfined",
        "-v",
        f"{cache_dir}:/triton-cache",
        args.image,
        *driver,
    ]
    env = dict(os.environ)
    if args.docker_config:
        env["DOCKER_CONFIG"] = args.docker_config

    print(f"harvest: compiling via {' '.join(driver)}")
    proc = subprocess.run(docker_cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(
            f"harvest: compile run failed (rc={proc.returncode}). "
            "'No input generator registered' means the operator is not drivable "
            "through the benchmark harness -- only gemm.mm is; use "
            "--pytest-suite to drive it through TokenSpeed's own tests instead. "
            "'not supported on the current platform' is usually an unwritable "
            "TRITON_CACHE_DIR rather than a real platform problem."
        )


def _inventory(waitcheck: Path | None, obj: Path) -> list[dict]:
    """Ask rj_waitcheck to describe the kernels inside one code object.

    Using the sanitizer's own parser rather than readelf keeps the recorded
    target and entry offsets consistent with what the sanitizer will later
    select on.
    """
    if waitcheck is None or not waitcheck.exists():
        return []
    proc = subprocess.run(
        [str(waitcheck), "--list-kernels", str(obj)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    entries = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "kernel":
            entries.append(record)
    return entries


def _write_recipe(dest: Path, label: str, target: str, kernels: list[dict]) -> Path:
    """Emit a mode: sanitizer recipe over the harvested identities.

    Written by hand rather than via a YAML library so the file carries the same
    explanatory comments a committed recipe would, and so this script has no
    dependency beyond the standard library.
    """
    recipe_path = dest / f"waitcheck-{label}.yaml"
    lines = [
        "# GENERATED by harvest_code_objects.py -- do not commit.",
        "#",
        "# Waitcheck over TokenSpeed kernel code objects harvested from the Triton",
        "# JIT cache. Paths are absolute and the digests below pin the exact objects",
        "# that were compiled on this node; re-harvest after changing image, GPU",
        "# target, or shapes rather than editing this file.",
        "#",
        "# scope.kind: kernel keeps this to exact-entry analysis, which is the",
        "# supported path today. Whole-module ConSan is deliberately not requested:",
        "# aorta fails it closed because RocJITsu has no kernel allowlist and would",
        "# otherwise instrument every code object indiscriminately.",
        "schema_version: 1",
        "mode: sanitizer",
        f"ticket: TOKENSPEED-WAITCHECK-{label.upper().replace('_', '-')}",
        "",
        "sanitizer_plan:",
        f"  target: {target}",
        "  source:",
        "    kind: kernel_list",
        "    kernels:",
    ]
    for kernel in kernels:
        lines.extend(
            [
                f"      - name: {kernel['name']}",
                f"        code_object: {kernel['code_object']}",
                f"        code_object_sha256: {kernel['sha256']}",
                f"        code_object_index: {kernel['code_object_index']}",
            ]
        )
    lines.extend(
        [
            "  scope:",
            "    kind: kernel",
            "  selection:",
            "    requirement: top_dispatch_count",
            # kernel_list gives every entry dispatch_count 1, so top_n must cover
            # the whole list or harvested kernels are silently dropped.
            f"    top_n: {len(kernels)}",
            "  sanitizers:",
            "    - waitcheck",
            "  policy:",
            "    consan_policy: strict",
            "    on_missing_backend: fail",
            "  output:",
            "    report: sanitizer_report.json",
            "",
        ]
    )
    recipe_path.write_text("\n".join(lines))
    return recipe_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="TokenSpeed container image")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kernel", help="single kernel name, e.g. gluon_mm_a16w16_gfx950")
    group.add_argument("--op", help="operator family.mode, e.g. gemm.mm")
    group.add_argument(
        "--pytest-suite",
        help=(
            "drive compilation with a TokenSpeed test suite instead of the "
            "benchmark harness, e.g. tokenspeed-kernel/test/ops/test_attention.py. "
            "This is the only route to attention and MoE code objects, which the "
            "benchmark harness cannot compile."
        ),
    )
    parser.add_argument("--pytest-k", default=None, help="-k expression for --pytest-suite")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "fp8"])
    parser.add_argument("--dtype-role", default="a", help="a/b for gemm.mm")
    parser.add_argument("--dest", required=True, help="run area (node-local)")
    parser.add_argument("--gpus", default="0", help="HIP_VISIBLE_DEVICES")
    parser.add_argument("--waitcheck", default=None, help="path to rj_waitcheck")
    parser.add_argument("--docker-config", default=None, help="DOCKER_CONFIG override")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    if str(dest).startswith(("/home/", "/nfs/")):
        raise SystemExit(
            f"harvest: {dest} is on NFS; the docker daemon cannot bind-mount it. "
            "Use a node-local path such as /tmp/ts-work/sanitizer-run."
        )
    cache_dir = dest / "triton-cache"
    objects_dir = dest / "code_objects"
    # Start from an empty cache so the inventory contains only what this run
    # compiled, rather than whatever a previous harvest left behind.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    objects_dir.mkdir(parents=True, exist_ok=True)

    _run_kernel(args, cache_dir)

    waitcheck = Path(args.waitcheck).resolve() if args.waitcheck else None
    found = sorted(cache_dir.rglob("*.hsaco"))
    if not found:
        raise SystemExit(
            "harvest: the run produced no .hsaco. Either the selection compiled "
            "nothing (torch_* solutions call into rocBLAS and emit no code "
            "object; a --pytest-k that deselects everything does the same), or "
            "TRITON_CACHE_DIR did not reach the container."
        )

    kernels: list[dict] = []
    targets = set()
    seen: set[tuple[str, str, int]] = set()
    for obj in found:
        # Stage content-addressed. The Triton cache holds one directory per
        # shape specialization and they reuse file names -- a single attention
        # run emits ten distinct `_fwd_kernel.hsaco`. Staging by bare name
        # would overwrite them in turn, leaving a recipe whose digests match
        # nothing on disk except the last copy, and Waitcheck would then reject
        # every earlier entry for a digest mismatch.
        digest = _sha256(obj)
        staged = objects_dir / f"{obj.stem}.{digest[:12]}{obj.suffix}"
        if not staged.exists():
            shutil.copy2(obj, staged)
        entries = _inventory(waitcheck, staged)
        if not entries:
            # No inventory means no verified symbol name; fall back to the
            # original file stem, which the Triton cache names after the kernel.
            # (staged.stem now carries the digest suffix, so it is not usable.)
            candidates = [{"kernel_name": obj.stem, "code_object_index": 0}]
        else:
            candidates = entries

        for entry in candidates:
            if entry.get("target"):
                targets.add(str(entry["target"]))
            name = str(entry.get("kernel_name", obj.stem))
            index = int(entry.get("code_object_index", 0))
            # Byte-identical objects compiled twice are one identity, not two.
            key = (name, digest, index)
            if key in seen:
                continue
            seen.add(key)
            kernels.append(
                {
                    "name": name,
                    "code_object": str(staged),
                    "sha256": digest,
                    "code_object_index": index,
                }
            )

    target = next((t for t in sorted(targets) if t), "gfx950")
    _, _, label = _driver(args)
    inventory_path = dest / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "image": args.image,
                "target": target,
                "selector": {
                    "kernel": args.kernel,
                    "op": args.op,
                    "pytest_suite": args.pytest_suite,
                    "pytest_k": args.pytest_k,
                },
                "dtype": args.dtype,
                "dtype_role": args.dtype_role,
                "kernels": kernels,
            },
            indent=2,
            sort_keys=True,
        )
    )
    recipe_path = _write_recipe(dest, label, target, kernels)

    print(
        f"harvest: {len(kernels)} kernel identit{'y' if len(kernels) == 1 else 'ies'} "
        f"on {target}"
    )
    for kernel in kernels:
        print(f"  {kernel['name']}  {kernel['sha256'][:16]}...  {kernel['code_object']}")
    print(f"harvest: inventory -> {inventory_path}")
    print(f"harvest: recipe    -> {recipe_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
