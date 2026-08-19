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
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from aorta.instrumentation.rocm_paths import resolve_rocm_roots  # noqa: E402

# Resolved rather than hardcoded to /opt/rocm (issue #381) so the fixtures can
# be built from a TheRock wheel install, which keeps the Tensile database
# under site-packages.
_ROCM_ROOTS = resolve_rocm_roots()
HIPBLASLT_LIBRARY = _ROCM_ROOTS.lib_dir / "hipblaslt" / "library"
LLVM_BIN_DIR = _ROCM_ROOTS.llvm_bin_dir
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
    """Locate clang-offload-bundler, on PATH or in ROCm's LLVM bindir.

    The bindir is on PATH in neither layout -- the classic image exports only
    /opt/rocm/bin and the wheel image only the venv's bin -- so falling back to
    the resolved location is what lets this run without the caller having to
    prepend it first (issue #381).
    """
    import shutil

    bundler = shutil.which("clang-offload-bundler")
    if bundler is not None:
        return bundler
    resolved = LLVM_BIN_DIR / "clang-offload-bundler"
    if resolved.is_file():
        return str(resolved)
    raise SystemExit(
        f"clang-offload-bundler not found on PATH or in {LLVM_BIN_DIR}"
    )


def _library_for(layout: str) -> Path:
    bundle = _SS_HEAVY.format(layout=layout)
    # The classic layout is flat; the TheRock wheel layout nests the bundles
    # one level deeper under the gfx target (library/gfx950/...). Try both.
    candidates = [HIPBLASLT_LIBRARY / bundle, HIPBLASLT_LIBRARY / GFX / bundle]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = " or ".join(str(c) for c in candidates)
    raise SystemExit(f"no {GFX} f32 SS Tensile bundle for layout {layout}: {tried}")


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
