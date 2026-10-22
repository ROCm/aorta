#!/usr/bin/env python3
"""Pack the model-kernel survey's sources into one deterministic tarball.

The nightly ``sanitizers-model-survey`` job downloads this tarball from a
ROCm/aorta release, checks it against the SHA-256 pinned in the workflow, and
runs every (kernel, scenario) pair it lists through waitcheck and ConSan. It is
a maintainer tool: run it once per asset version, upload the result, and bump
the pinned digest in the same PR.

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

    build_model_kernel_asset.py --version 1 \\
        --gpt-oss-tree <gpt-oss-pffm-harness> \\
        --qwen3-tree <gpt-oss-qwen3-kernel-hw> --qwen3-launch <patched launch.py> \\
        --licenses-dir <dir with vllm-LICENSE, aiter-LICENSE, triton-LICENSE> \\
        --out aorta-model-kernels-v1.tar.gz
"""

from __future__ import annotations

import argparse
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
}
_ORIGIN_RE = re.compile(r"^# origin: ([A-Za-z0-9_]+)/", re.MULTILINE)


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
    return "\n".join(lines)


def readme(version: str, models: dict[str, dict]) -> str:
    rows = "\n".join(
        f"| {name} | {m['kernels']} | {m['pairs']} | {', '.join(m['specs'])} |"
        for name, m in models.items()
    )
    return f"""# AORTA model-kernel survey sources, v{version}

Input to the `sanitizers-model-survey` nightly job in ROCm/aorta. Each
(kernel, scenario) pair is launched once with
`roofline/harness/launch.py --mode run`; the object Triton compiles for that
launch is scanned by waitcheck and, for one scenario per kernel, dispatched
under ConSan. See `recipes/sanitizers/models/README.md` in ROCm/aorta.

| Model | Kernels | Pairs | Specs |
|---|--:|--:|---|
{rows}

Only kernels the model launches (spec `role` of {", ".join(f"`{r}`" for r in DEFAULT_ROLES)})
are included. Licenses: `THIRD_PARTY_NOTICES.md`. Per-file SHA-256: `MANIFEST.json`.
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
    parser.add_argument("--licenses-dir", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roles = frozenset(args.roles)

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
