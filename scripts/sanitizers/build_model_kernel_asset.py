#!/usr/bin/env python3
"""Pack the model-kernel survey's sources into one deterministic tarball.

The nightly ``sanitizers-model-survey`` and ``sanitizers-fpsan-survey`` jobs
download this tarball from a ROCm/aorta release, check it against the SHA-256
pinned in ``model_survey.py``, and run every (kernel, scenario) pair it lists
through waitcheck, ConSan and FpSan. It is a maintainer tool: run it once per
asset version, upload the result, and bump the pinned digest in the same PR.

What goes in, per model, is exactly what ``launch.py --mode run`` resolves at
run time and nothing else:

    <model>/roofline/harness/{launch.py,moe_routing.py}
    <model>/roofline/specs/<spec>.json
    <model>/pkg_src/...                      # triton_kernels + vLLM const table
    <model>/kernels/<flat file>.py           # only files a kept spec entry names

The harness locates ``pkg_src`` as ``parents[2]`` of ``launch.py``, which is
why each model keeps the ``roofline/harness`` nesting. Only kernels whose spec
``role`` is in ``--roles`` are packed, so a kernel the model never launches is
not published. Tooling, READMEs and ``__pycache__`` from the source trees are
never copied: they name the original authors' machines and are not needed to
launch a kernel.

From v2 the tarball also carries ``triton-tutorials/``: FpSan reference and
candidate pairs taken from upstream Triton's ``python/tutorials`` (persistent
vs. plain matmul, and three schedule variants), with a generated
``roofline/specs/tutorials.json`` and the Qwen3 ``launch.py`` as harness.
Each kernel file holds one tutorial kernel and the ``@triton.jit`` helpers it
calls, with ``@triton.autotune`` and ``launch_metadata`` removed so the spec
alone sets the launch config.

    build_model_kernel_asset.py --version 2 \\
        --gpt-oss-tree <gpt-oss-pffm-harness> \\
        --qwen3-tree <gpt-oss-qwen3-kernel-hw> --qwen3-launch <patched launch.py> \\
        --triton-tutorials-dir <triton checkout>/python/tutorials --triton-tutorials-rev <sha> \\
        --licenses-dir <dir with vllm-LICENSE, aiter-LICENSE, triton-LICENSE> \\
        --out aorta-model-kernels-v2.tar.gz
"""

from __future__ import annotations

import argparse
import ast
import builtins
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = "aorta-model-kernels"
DEFAULT_ROLES = ("launched", "conditional")
LICENSES = {
    "vllm": ("vllm-LICENSE", "vLLM", "Apache-2.0", "https://github.com/vllm-project/vllm"),
    "aiter": ("aiter-LICENSE", "AITER", "MIT", "https://github.com/ROCm/aiter"),
    "triton_kernels": (
        "triton-LICENSE",
        "Triton (python/triton_kernels)",
        "MIT",
        "https://github.com/triton-lang/triton",
    ),
    "triton_tutorials": (
        "triton-LICENSE",
        "Triton (python/tutorials)",
        "MIT",
        "https://github.com/triton-lang/triton",
    ),
}
_ORIGIN_RE = re.compile(r"^# origin: ([A-Za-z0-9_]+)/", re.MULTILINE)

TUTORIALS = "triton-tutorials"
TUTORIAL_SPEC = "tutorials.json"
# Spec role of an entry that is only ever a reference; the survey launches
# DEFAULT_ROLES entries, and fpsan_compare.py reaches these by name.
REFERENCE_ROLE = "reference"
# The tutorial each packed kernel function is taken from.
TUTORIAL_SOURCES = {
    "add_kernel": "01-vector-add.py",
    "softmax_kernel": "02-fused-softmax.py",
    "matmul_kernel": "03-matrix-multiplication.py",
    "_layer_norm_fwd_fused": "05-layer-norm.py",
    "matmul_kernel_persistent": "09-persistent-matmul.py",
}
# Names a tutorial kernel may read without defining them.
_KERNEL_GLOBALS = frozenset({"tl", "triton"}) | frozenset(dir(builtins))


def _tensor(name: str, dtype: str, shape: list[int], init: str = "zeros") -> dict:
    return {"name": name, "kind": "tensor", "dtype": dtype, "shape": shape, "init": init}


def _scalar(name: str, value: object, dtype: str = "i32") -> dict:
    return {"name": name, "kind": "scalar", "dtype": dtype, "value": value}


def _constexpr(name: str, value: object) -> dict:
    return {"name": name, "kind": "constexpr", "value": value}


def _launch(grid: int, warps: int, stages: int, args: list[dict]) -> dict:
    return {"launch": {"grid": [grid, 1, 1], "num_warps": warps, "num_stages": stages, "args": args}}


def _matmul_args(m: int, n: int, k: int, bm: int, bn: int, bk: int, last: dict) -> list[dict]:
    return [
        _tensor("a_ptr", "fp16", [m, k], "randn"), _tensor("b_ptr", "fp16", [k, n], "randn"),
        _tensor("c_ptr", "fp16", [m, n]),
        _scalar("M", m), _scalar("N", n), _scalar("K", k),
        _scalar("stride_am", k), _scalar("stride_ak", 1), _scalar("stride_bk", n),
        _scalar("stride_bn", 1), _scalar("stride_cm", n), _scalar("stride_cn", 1),
        _constexpr("BLOCK_SIZE_M", bm), _constexpr("BLOCK_SIZE_N", bn),
        _constexpr("BLOCK_SIZE_K", bk), _constexpr("GROUP_SIZE_M", 8), last,
    ]


def _softmax_args(rows: int, cols: int, stages: int) -> list[dict]:
    return [
        _tensor("output_ptr", "fp32", [rows, cols]), _tensor("input_ptr", "fp32", [rows, cols], "randn"),
        _scalar("input_row_stride", cols), _scalar("output_row_stride", cols),
        _scalar("n_rows", rows), _scalar("n_cols", cols),
        _constexpr("BLOCK_SIZE", cols), _constexpr("num_stages", stages),
    ]


def _layer_norm_args(rows: int, cols: int, block: int) -> list[dict]:
    return [
        _tensor("X", "fp16", [rows, cols], "randn"), _tensor("Y", "fp16", [rows, cols]),
        _tensor("W", "fp16", [cols], "randn"), _tensor("B", "fp16", [cols], "randn"),
        _tensor("Mean", "fp32", [rows]), _tensor("Rstd", "fp32", [rows]),
        _scalar("stride", cols), _scalar("N", cols), _scalar("eps", 1e-5, "fp32"),
        _constexpr("BLOCK_SIZE", block),
    ]


def _add_args(n: int, block: int) -> list[dict]:
    return [
        _tensor("x_ptr", "fp32", [n], "randn"), _tensor("y_ptr", "fp32", [n], "randn"),
        _tensor("output_ptr", "fp32", [n]), _scalar("n_elements", n), _constexpr("BLOCK_SIZE", block),
    ]


def tutorial_entries() -> list[dict]:
    """``tutorials.json``: each candidate beside the reference it is compared with.

    A pair shares scenario names and passes the same tensors in the same order,
    so ``launch.py`` draws the same inputs for both. Every problem size is a
    multiple of every tile that reduces over it: FpSan does not zero masked
    lanes, so padding that differs between the two would change the digest.
    """

    def entry(kernel: str, role: str, scenarios: dict) -> dict:
        source = TUTORIAL_SOURCES[kernel.split("@", 1)[0]]
        return {
            "kernel": kernel,
            "role": role,
            "flat_file": f"kernels/{tutorial_flat_file(kernel)}",
            "origin": f"triton_tutorials/{source}",
            "scenarios": scenarios,
        }

    matmul = {"S512": (512, 512, 512, 8), "S1024x768": (1024, 768, 512, 16)}
    return [
        entry("matmul_kernel_persistent", "launched", {
            name: _launch(sms, 8, 2, _matmul_args(m, n, k, 128, 128, 64, _constexpr("NUM_SMS", sms)))
            for name, (m, n, k, sms) in matmul.items()
        }),
        entry("matmul_kernel", REFERENCE_ROLE, {
            name: _launch((m // 64) * (n // 64), 4, 1, _matmul_args(m, n, k, 64, 64, 32, _constexpr("ACTIVATION", "")))
            for name, (m, n, k, _sms) in matmul.items()
        }),
        entry("softmax_kernel", "launched", {"R512xC1024": _launch(32, 8, 2, _softmax_args(512, 1024, 2))}),
        entry("softmax_kernel@row_per_program", REFERENCE_ROLE,
              {"R512xC1024": _launch(512, 4, 1, _softmax_args(512, 1024, 1))}),
        entry("_layer_norm_fwd_fused", "launched",
              {"M256xN4096": _launch(256, 8, 2, _layer_norm_args(256, 4096, 4096))}),
        entry("_layer_norm_fwd_fused@chunked", REFERENCE_ROLE,
              {"M256xN4096": _launch(256, 4, 1, _layer_norm_args(256, 4096, 512))}),
        entry("add_kernel", "launched", {"N98432": _launch(97, 4, 2, _add_args(98432, 1024))}),
        entry("add_kernel@block128", REFERENCE_ROLE, {"N98432": _launch(769, 4, 1, _add_args(98432, 128))}),
    ]


def tutorial_flat_file(kernel: str) -> str:
    function = kernel.split("@", 1)[0]
    return f"triton_tutorials__{Path(TUTORIAL_SOURCES[function]).stem.replace('-', '_')}__{function}.py"


def _is_jit(node: ast.FunctionDef) -> bool:
    return any("jit" in ast.unparse(d) for d in node.decorator_list)


def _free_names(node: ast.FunctionDef) -> set[str]:
    bound = {a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
    loaded: set[str] = set()
    for part in [node.args, *node.body]:
        for child in ast.walk(part):
            if isinstance(child, ast.Name):
                (bound if isinstance(child.ctx, ast.Store) else loaded).add(child.id)
    return loaded - bound


def extract_kernel(source: str, function: str, *, origin: str) -> str:
    """One standalone kernel file: ``function`` and the jit helpers it calls.

    Every decorator becomes a plain ``@triton.jit``. Raises ``SystemExit`` when
    the kernel reads a module name that is neither a helper nor ``tl``/``triton``.
    """

    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and _is_jit(n)}
    if function not in functions:
        raise SystemExit(f"{origin}: no @triton.jit function {function!r}")
    order: list[str] = []

    def visit(name: str) -> None:
        if name in order:
            return
        free = _free_names(functions[name])
        helpers = sorted(free & set(functions))
        unknown = sorted(free - set(functions) - _KERNEL_GLOBALS)
        if unknown:
            raise SystemExit(f"{origin}: {name} reads module names {unknown}")
        for helper in helpers:
            visit(helper)
        order.append(name)

    visit(function)
    lines = source.splitlines()
    bodies = [
        "@triton.jit\n" + "\n".join(lines[functions[name].lineno - 1 : functions[name].end_lineno])
        for name in order
    ]
    return (
        f"# kernel: {function}\n"
        f"# origin: {origin}:{functions[function].lineno}\n"
        "# decorators: @triton.autotune and launch_metadata removed; jit only\n"
        "import triton\nimport triton.language as tl\n\n\n" + "\n\n\n".join(bodies) + "\n"
    )


def tutorial_files(
    *, tutorials_dir: Path, revision: str, launch: Path, routing: Path
) -> tuple[dict[str, bytes], dict]:
    """Archive members for ``triton-tutorials``, plus its MANIFEST entry."""
    entries = tutorial_entries()
    files: dict[str, bytes] = {
        f"{TUTORIALS}/roofline/harness/launch.py": launch.read_bytes(),
        f"{TUTORIALS}/roofline/harness/moe_routing.py": routing.read_bytes(),
        f"{TUTORIALS}/roofline/specs/{TUTORIAL_SPEC}": (json.dumps(entries, indent=1) + "\n").encode(),
    }
    for entry in entries:
        function = entry["kernel"].split("@", 1)[0]
        source = (tutorials_dir / TUTORIAL_SOURCES[function]).read_text(encoding="utf-8")
        params = [a.arg for a in next(
            n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function
        ).args.args]
        for name, scenario in entry["scenarios"].items():
            names = [a["name"] for a in scenario["launch"]["args"]]
            if names != params:
                raise SystemExit(f"{entry['kernel']}/{name}: spec args {names} != {function}{params}")
        files[f"{TUTORIALS}/{entry['flat_file']}"] = extract_kernel(
            source, function, origin=entry["origin"]
        ).encode()
    launched = [e for e in entries if e["role"] in DEFAULT_ROLES]
    return files, {
        "specs": [TUTORIAL_SPEC],
        "roles": sorted(DEFAULT_ROLES),
        "pairs": sum(len(e["scenarios"]) for e in launched),
        "kernels": len(launched),
        "kernel_files": len({e["flat_file"] for e in entries}),
        "origins": {"triton_tutorials": len({e["flat_file"] for e in entries})},
        "upstream_revision": revision,
    }


def kept_entries(spec: Path, roles: frozenset[str]) -> list[dict]:
    entries = json.loads(spec.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"{spec}: expected a JSON list of kernel entries")
    return [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("kernel")
        and isinstance(entry.get("scenarios"), dict)
        and entry["scenarios"]
        and entry.get("role") in roles
    ]


def tree_files(src: Path) -> list[Path]:
    return sorted(
        p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )


def model_files(
    *,
    name: str,
    launch: Path,
    routing: Path,
    specs: list[Path],
    flat_dir: Path,
    pkg_src: list[Path],
    roles: frozenset[str],
) -> tuple[dict[str, bytes], dict]:
    """Archive members for one model, plus its MANIFEST entry."""
    files: dict[str, bytes] = {
        f"{name}/roofline/harness/launch.py": launch.read_bytes(),
        f"{name}/roofline/harness/moe_routing.py": routing.read_bytes(),
    }
    pairs = 0
    kernels: set[str] = set()
    flat: set[str] = set()
    for spec in specs:
        files[f"{name}/roofline/specs/{spec.name}"] = spec.read_bytes()
        for entry in kept_entries(spec, roles):
            pairs += len(entry["scenarios"])
            kernels.add(entry["kernel"])
            flat.add(Path(entry["flat_file"]).name)
    origins: dict[str, int] = {}
    for flat_name in sorted(flat):
        body = (flat_dir / flat_name).read_bytes()
        files[f"{name}/kernels/{flat_name}"] = body
        match = _ORIGIN_RE.search(body.decode("utf-8", errors="replace"))
        project = match.group(1) if match else "unknown"
        origins[project] = origins.get(project, 0) + 1
    for root in pkg_src:
        for path in tree_files(root):
            files[f"{name}/pkg_src/{path.relative_to(root.parent)}"] = path.read_bytes()
    unknown = set(origins) - set(LICENSES)
    if unknown:
        raise SystemExit(f"{name}: kernels from projects with no recorded license: {sorted(unknown)}")
    return files, {
        "specs": [spec.name for spec in specs],
        "roles": sorted(roles),
        "pairs": pairs,
        "kernels": len(kernels),
        "kernel_files": len(flat),
        "origins": dict(sorted(origins.items())),
    }


def notices(models: dict[str, dict]) -> str:
    used = sorted({project for model in models.values() for project in model["origins"]})
    lines = [
        "# Third-party notices",
        "",
        "The kernel files under `<model>/kernels/` and the `triton_kernels` and",
        "vLLM files under `<model>/pkg_src/` were taken from the projects below and",
        "flattened: each kernel's helpers were inlined into one standalone file, and",
        "each file's first comment line names its source as `# origin: <path>:<line>`.",
        "These are modified copies. The exact upstream revisions were not recorded",
        "when the kernels were harvested.",
        "",
        "| Project | License | Upstream | License text |",
        "|---|---|---|---|",
    ]
    for project in used:
        filename, title, spdx, url = LICENSES[project]
        lines.append(f"| {title} | {spdx} | {url} | `licenses/{filename}` |")
    lines += [
        "",
        "`<model>/roofline/harness/launch.py`, `moe_routing.py` and the",
        "`roofline/specs/*.json` launch specs come from the gpt-oss-rocm kernel study's",
        "hardware harness. The Qwen3 `launch.py` carries AORTA fixes on top of it:",
        "binding launch arguments by name, unwrapping `@staticmethod`, fp8 bit",
        "patterns, and a tiled device fill for buffers over 1 GiB.",
        "",
    ]
    tutorials = models.get(TUTORIALS)
    if tutorials:
        lines += [
            f"`{TUTORIALS}/kernels/` are taken from triton-lang/triton `python/tutorials`",
            f"at revision `{tutorials['upstream_revision']}`. Each file holds one tutorial",
            "kernel and the `@triton.jit` helpers it calls, copied unchanged except that",
            "`@triton.autotune` and `launch_metadata` were removed. Their launch spec",
            f"`{TUTORIALS}/roofline/specs/{TUTORIAL_SPEC}` was written for this asset, and",
            "their `launch.py` is the Qwen3 one.",
            "",
        ]
    return "\n".join(lines)


def readme(version: str, models: dict[str, dict]) -> str:
    rows = "\n".join(
        f"| {name} | {m['kernels']} | {m['pairs']} | {', '.join(m['specs'])} |"
        for name, m in models.items()
    )
    return f"""# AORTA model-kernel survey sources, v{version}

Input to the `sanitizers-model-survey` and `sanitizers-fpsan-survey` nightly
jobs in ROCm/aorta. Each (kernel, scenario) pair is launched once with
`roofline/harness/launch.py --mode run`; the object Triton compiles for that
launch is scanned by waitcheck and, for one scenario per kernel, dispatched
under ConSan. Every pair is also compared with a reference under Triton FpSan.
See `recipes/sanitizers/models/README.md` in ROCm/aorta.

| Model | Kernels | Pairs | Specs |
|---|--:|--:|---|
{rows}

Only kernels the model launches (spec `role` of {", ".join(f"`{r}`" for r in DEFAULT_ROLES)})
are included, plus, for `{TUTORIALS}`, the `{REFERENCE_ROLE}` entries FpSan compares
them with. Licenses: `THIRD_PARTY_NOTICES.md`. Per-file SHA-256: `MANIFEST.json`.
"""


def write_tarball(out: Path, files: dict[str, bytes]) -> str:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel in sorted(files):
            body = files[rel]
            info = tarfile.TarInfo(f"{ROOT}/{rel}")
            info.size = len(body)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(body))
    packed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=packed, mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    out.write_bytes(packed.getvalue())
    return hashlib.sha256(packed.getvalue()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--gpt-oss-tree", type=Path, required=True)
    parser.add_argument("--qwen3-tree", type=Path, required=True)
    parser.add_argument(
        "--qwen3-launch",
        type=Path,
        required=True,
        help="the patched launch.py Qwen3 runs with (see THIRD_PARTY_NOTICES.md)",
    )
    parser.add_argument("--triton-tutorials-dir", type=Path,
                        help="python/tutorials of a triton-lang/triton checkout (v2 and later)")
    parser.add_argument("--triton-tutorials-rev", help="the commit that checkout is at")
    parser.add_argument("--licenses-dir", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roles = frozenset(args.roles)
    if (args.triton_tutorials_dir is None) != (args.triton_tutorials_rev is None):
        parser.error("--triton-tutorials-dir and --triton-tutorials-rev go together")
    if args.triton_tutorials_rev is not None and not re.fullmatch(r"[0-9a-f]{40}", args.triton_tutorials_rev):
        parser.error("--triton-tutorials-rev must be a full commit SHA")

    gpt, qwen = args.gpt_oss_tree, args.qwen3_tree
    layouts = {
        "gpt-oss": {
            "launch": gpt / "roofline" / "harness" / "launch.py",
            "routing": gpt / "roofline" / "harness" / "moe_routing.py",
            "specs": [gpt / "roofline" / "specs" / f for f in ("attention.json", "misc.json", "moe.json")],
            "flat_dir": gpt / "kernels_gpt_oss",
            "pkg_src": [gpt / "pkg_src" / "vllm", qwen / "pkg_src" / "triton_kernels"],
        },
        "qwen3": {
            "launch": args.qwen3_launch,
            "routing": qwen / "roofline" / "harness" / "moe_routing.py",
            "specs": [qwen / "roofline" / "specs" / "qwen3.json"],
            "flat_dir": qwen / "kernels_qwen3",
            "pkg_src": [qwen / "pkg_src" / "vllm", qwen / "pkg_src" / "triton_kernels"],
        },
    }
    files: dict[str, bytes] = {}
    models: dict[str, dict] = {}
    for name, layout in layouts.items():
        member, manifest = model_files(name=name, roles=roles, **layout)
        files.update(member)
        models[name] = manifest
    if args.triton_tutorials_dir is not None:
        member, manifest = tutorial_files(
            tutorials_dir=args.triton_tutorials_dir,
            revision=args.triton_tutorials_rev,
            launch=args.qwen3_launch,
            routing=qwen / "roofline" / "harness" / "moe_routing.py",
        )
        files.update(member)
        models[TUTORIALS] = manifest

    used = {project for model in models.values() for project in model["origins"]}
    for project in sorted(used):
        filename = LICENSES[project][0]
        files[f"licenses/{filename}"] = (args.licenses_dir / filename).read_bytes()
    files["THIRD_PARTY_NOTICES.md"] = notices(models).encode()
    files["README.md"] = readme(args.version, models).encode()
    manifest = {
        "schema": "aorta.model_kernel_asset/1",
        "version": str(args.version),
        "models": models,
        "files": {rel: hashlib.sha256(body).hexdigest() for rel, body in sorted(files.items())},
    }
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    leaked = [rel for rel, body in files.items() if b"/home/" in body or b"/apps/" in body]
    if leaked:
        raise SystemExit(f"refusing to pack files that name a local path: {leaked}")
    digest = write_tarball(args.out, files)
    print(json.dumps({"out": str(args.out), "sha256": digest, "files": len(files), "models": models}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
