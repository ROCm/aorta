#!/usr/bin/env python3
"""Prepare gfx950 GEMM code-object fixtures for sanitizer CI.

Extracts real **f32 (Type_SS)** hipBLASLt Tensile code objects for the shapes in the
synthetic fixture CSV. Each shape's (transA, transB) selects the matching transpose
layout's heavy SS Tensile bundle, whose gfx950 HIP code object is unbundled per shape.

This replaces the earlier behaviour that unbundled one ``*SB*`` (bf8) bundle and
``shutil.copy2``'d it to every ``sol_*.hsaco`` -- producing byte-identical blobs that
contained no f32 kernels. hipBLASLt Tensile libraries are open-source ROCm content;
the extracted objects contain no customer data.

With ``--consan-object PATH`` it also writes one representative heavy f32 SS object
(the NT / ``Ailk_Bjlk`` layout, ~16 MB / ~490 kernels) for driving ConSan over a real
code object via ``source.consan_command``.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

HIPBLASLT_LIBRARY = Path("/opt/rocm/lib/hipblaslt/library")
GFX = "gfx950"
TARGET = f"hipv4-amdgcn-amd-amdhsa--{GFX}"

# (transA, transB) -> Tensile contraction layout token in the bundle name.
_LAYOUT = {
    ("N", "T"): "Ailk_Bjlk",
    ("N", "N"): "Ailk_Bljk",
    ("T", "N"): "Alik_Bljk",
    ("T", "T"): "Alik_Bjlk",
}
# The heavy f32 GEMM library variant (unbundles to ~16 MB / ~490 kernels on gfx950).
_SS_HEAVY = "TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_{layout}_Cijk_Dijk_" + GFX + ".co"


def _read_rows(csv_path: Path, top_n: int) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.lstrip().startswith("#")))
    return sorted(rows, key=lambda row: int(row["count"]), reverse=True)[:top_n]


def _bundler() -> str:
    import shutil

    bundler = shutil.which("clang-offload-bundler")
    if bundler is None:
        raise SystemExit("clang-offload-bundler not found on PATH")
    return bundler


def _library_for(layout: str) -> Path:
    candidate = HIPBLASLT_LIBRARY / _SS_HEAVY.format(layout=layout)
    if not candidate.is_file():
        raise SystemExit(f"no gfx950 f32 SS Tensile bundle for layout {layout}: {candidate}")
    return candidate


def _unbundle(bundler: str, src: Path, dst: Path) -> None:
    subprocess.run(
        [bundler, "--type=o", "--unbundle", f"--input={src}", f"--output={dst}",
         f"--targets={TARGET}"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--consan-object",
        type=Path,
        default=None,
        help="also write one heavy f32 SS object (NT/Ailk_Bjlk) here for ConSan",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bundler = _bundler()

    for row in _read_rows(args.csv, args.top_n):
        layout = _LAYOUT.get((row["transA"].strip().upper(), row["transB"].strip().upper()))
        if layout is None:
            raise SystemExit(f"unsupported transpose {row['transA']}/{row['transB']}")
        _unbundle(bundler, _library_for(layout), args.out / f"sol_{row['top_solution_idx']}.hsaco")

    if args.consan_object is not None:
        args.consan_object.parent.mkdir(parents=True, exist_ok=True)
        _unbundle(bundler, _library_for("Ailk_Bjlk"), args.consan_object)


if __name__ == "__main__":
    main()
