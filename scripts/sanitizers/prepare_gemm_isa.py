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
(the NT / ``Ailk_Bjlk`` layout) for driving ConSan over a real code object via
``source.consan_command``.

A (family, layout, arch) triple named exactly one bundle through ROCm 7.2.4. From
ROCm 7.14 the gfx950 libraries are split per device, so the layout alone no longer
identifies a file and the bundle is chosen for a specific ``--chip-id`` /
``--cu-count`` -- defaulting to the gate's MI350X. See ``_VARIANT_RE`` for how the
filename encodes exactly the predicates hipBLASLt gates each variant on.

Size is deliberately not quoted here: the extracted gfx950 object measures ~183 MiB
on the current CI base (ROCm 7.2.4), so the "~16 MB" this docstring used to claim
had already drifted by more than a factor of ten. It is a per-release property of
the shipped Tensile libraries -- the ROCm 7.14 variants measure ~156-168 MiB -- so
treat any figure as version-specific and measure rather than trust a comment.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

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
# The heavy f32 GEMM library family. Unbundles to a large gfx950 object -- ~183 MiB
# measured on ROCm 7.2.4; see the module docstring on why no fixed size is asserted.
_SS_HEAVY_STEM = "TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_{layout}_Cijk_Dijk"
# Through ROCm 7.2.4 that stem plus ``_gfx950.co`` named exactly one file.
_SS_HEAVY = _SS_HEAVY_STEM + "_" + GFX + ".co"

# Device the fixtures are built to represent: the gate's MI350X reports PCI chip id
# 0x75a0 with 256 CUs (measured via rocprofv3 agent info -- Device_Id 30112,
# Cu_Count 256). Fixed rather than probed so the digests recorded beside the
# artifacts reproduce off-GPU; override with --chip-id / --cu-count.
GATE_CHIP_ID = 0x75A0
GATE_CU_COUNT = 256

# ROCm 7.14 splits the gfx950 libraries per device, and the filename carries the
# very predicates hipBLASLt's own lazy master index (TensileLibrary_lazy_gfx950)
# gates each variant on:
#
#     _CU<n>                 CUCount == n
#     _ID<hex>[-<hex>...]    PciChipId in {...}
#
# so resolving by filename tracks the loader rather than guessing. For the gate's
# 0x75a0/256-CU part the index selects the CU256_ID75a0 variant; the same file
# ordering is reproduced by _select_variant's specificity rule below.
_VARIANT_RE = re.compile(r"^(?:_CU(?P<cu>[0-9]+))?(?:_ID(?P<ids>[0-9a-f]+(?:-[0-9a-f]+)*))?$")


class _Variant(NamedTuple):
    """One shipped bundle for a (family, layout, arch), with its device predicates.

    A NamedTuple rather than a dataclass because the tests load this file as a
    detached module (``module_from_spec`` without a ``sys.modules`` entry), which
    is enough to break ``@dataclass``'s annotation resolution.
    """

    path: Path
    cu_count: int | None
    chip_ids: frozenset[int]
    dir_rank: int

    def serves(self, chip_id: int, cu_count: int) -> bool:
        """Whether this bundle's predicates admit the requested device."""
        return (self.cu_count is None or self.cu_count == cu_count) and (
            not self.chip_ids or chip_id in self.chip_ids
        )

    @property
    def specificity(self) -> tuple[int, int, int, int, str]:
        """Sort key placing the narrowest admitting bundle first.

        Chip and CU precedence mirror the lazy index's row order, which lists a
        chip's CU-specialised libraries ahead of its unspecialised one and is
        resolved first-match. ``dir_rank`` and the trailing name make the choice
        total, so it never depends on the order a directory enumerates in.
        """
        return (
            0 if self.chip_ids else 1,
            0 if self.cu_count is not None else 1,
            len(self.chip_ids),
            self.dir_rank,
            self.path.name,
        )


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


def _library_dirs() -> list[Path]:
    """Directories that may hold the Tensile bundles, in precedence order.

    The classic layout is flat; ROCm 7.14 and the TheRock wheel layout nest the
    bundles one level deeper under the gfx target (library/gfx950/...).
    """
    return [HIPBLASLT_LIBRARY, HIPBLASLT_LIBRARY / GFX]


def _variants_for(layout: str) -> list[_Variant]:
    """Every shipped bundle for this layout, with its parsed device predicates."""
    prefix = _SS_HEAVY_STEM.format(layout=layout)
    suffix = f"_{GFX}.co"
    found: list[_Variant] = []
    for rank, directory in enumerate(_library_dirs()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"{prefix}*{suffix}")):
            match = _VARIANT_RE.match(path.name[len(prefix) : -len(suffix)])
            if match is None:
                # An unrecognised token block would silently widen what we treat
                # as eligible, so skip it rather than assume it is unconstrained.
                continue
            ids = match.group("ids")
            found.append(
                _Variant(
                    path=path,
                    cu_count=int(match.group("cu")) if match.group("cu") else None,
                    chip_ids=(
                        frozenset(int(part, 16) for part in ids.split("-")) if ids else frozenset()
                    ),
                    dir_rank=rank,
                )
            )
    return found


def _select_variant(variants: list[_Variant], chip_id: int, cu_count: int) -> _Variant | None:
    """The narrowest bundle whose predicates admit this device, or None."""
    admitting = [variant for variant in variants if variant.serves(chip_id, cu_count)]
    return min(admitting, key=lambda variant: variant.specificity, default=None)


def _library_for(layout: str, chip_id: int = GATE_CHIP_ID, cu_count: int = GATE_CU_COUNT) -> Path:
    variants = _variants_for(layout)
    if not variants:
        tried = " or ".join(str(d / _SS_HEAVY.format(layout=layout)) for d in _library_dirs())
        raise SystemExit(f"no {GFX} f32 SS Tensile bundle for layout {layout}: {tried}")
    selected = _select_variant(variants, chip_id, cu_count)
    if selected is None:
        offered = ", ".join(sorted(variant.path.name for variant in variants))
        raise SystemExit(
            f"no {GFX} f32 SS Tensile bundle for layout {layout} serves "
            f"chip 0x{chip_id:x} with {cu_count} CUs; shipped variants: {offered}"
        )
    return selected.path


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
    parser.add_argument(
        "--chip-id",
        type=lambda value: int(value, 0),
        default=GATE_CHIP_ID,
        help="PCI chip id to pick the ROCm 7.14+ per-device Tensile variant for "
        f"(default 0x{GATE_CHIP_ID:x}, the gate's MI350X)",
    )
    parser.add_argument(
        "--cu-count",
        type=int,
        default=GATE_CU_COUNT,
        help=f"CU count for the same choice (default {GATE_CU_COUNT})",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bundler = _bundler()

    def library(layout: str) -> Path:
        return _library_for(layout, args.chip_id, args.cu_count)

    for row in _read_rows(args.csv, args.top_n):
        layout = _LAYOUT.get((row["transA"].strip().upper(), row["transB"].strip().upper()))
        if layout is None:
            raise SystemExit(f"unsupported transpose {row['transA']}/{row['transB']}")
        _unbundle(bundler, library(layout), args.out / f"sol_{row['top_solution_idx']}.hsaco")

    if args.consan_object is not None:
        args.consan_object.parent.mkdir(parents=True, exist_ok=True)
        _unbundle(bundler, library("Ailk_Bjlk"), args.consan_object)


if __name__ == "__main__":
    main()
