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
filename encodes exactly the predicates hipBLASLt gates each variant on, and
``_select_variant`` for the resolution order it is matched in.

The selection rules here are transcribed from tensilelite rather than inferred
from the shipped file names, since a fixture built from a bundle the loader would
not use for the device would misrepresent what runs:

* ``include/Tensile/ExactLogicLibrary.hpp`` -- ``ExactLogicLibrary::findBestSolution``
  (exact match returns immediately, first fallback match is kept, and a matching
  row that yields nothing does not end the walk) and ``HardwarePredicate::isFallbackMatch``;
* ``include/Tensile/AMDGPUPredicates.hpp`` -- ``ChipIdRegistry`` (the chip-id
  fallback graph) and ``PciChipIdEqual``;
* the build-time hardware-row comparator, as documented in hipBLASLt's
  "PCI chip ID predicates walkthrough".

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
# so resolving by filename tracks the loader rather than guessing.
#
# Grammar note: both tokens are optional (ROCm <= 7.2.4 shipped neither), the
# order is fixed, and the ids are lowercase hex. Anything else is not parsed --
# see _variants_for on why that is a skip rather than a permissive read.
_VARIANT_RE = re.compile(r"^(?:_CU(?P<cu>[0-9]+))?(?:_ID(?P<ids>[0-9a-f]+(?:-[0-9a-f]+)*))?$")

# tensilelite's ChipIdRegistry, mirrored as data. A gfx950 device may use a
# solution built for one of its fallback ids, and PciChipIdEqual accepts that via
# ChipIdRegistry::canUseSolution -- so a bundle naming only 0x75a0 still serves an
# MI355X. Ignoring this made the selector fail closed on five real chip ids that
# hipBLASLt serves from the 0x75a0 bundles.
#     include/Tensile/AMDGPUPredicates.hpp, namespace ChipIdRegistry
_CHIP_ID_FALLBACKS: dict[int, tuple[int, ...]] = {
    0x75B0: (0x75A0,),
    0x75A2: (0x75A0,),
    0x75B2: (0x75A0,),
    0x75A3: (0x75A0,),
    0x75B3: (0x75A0,),
    0x75A8: (0x75A0,),
    0x75B8: (0x75A0,),
}

# How a bundle's predicates admit a device, in the loader's own terms.
_EXACT = "exact"
_FALLBACK = "fallback"


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

    def admits(self, chip_id: int, cu_count: int) -> str | None:
        """How this bundle admits the device: ``_EXACT``, ``_FALLBACK`` or None.

        Mirrors ``PciChipIdEqual::operator()`` plus
        ``HardwarePredicate::isFallbackMatch``: a bundle with no chip-id token has
        no chip constraint, so every match is exact; naming the device's own id is
        exact; naming one of its registry fallbacks is a fallback match.
        """
        if self.cu_count is not None and self.cu_count != cu_count:
            return None
        if not self.chip_ids or chip_id in self.chip_ids:
            return _EXACT
        if self.chip_ids.intersection(_CHIP_ID_FALLBACKS.get(chip_id, ())):
            return _FALLBACK
        return None

    @property
    def row_order(self) -> tuple[int, int, int, int, str]:
        """Sort key reproducing tensilelite's build-time hardware-row comparator.

        Documented order: rows with chip ids before rows without; then smaller
        chip-id sets ("exactness first"); then, when chip precedence does not
        decide, the higher CU count first. Rows with no CU constraint sort behind
        the constrained ones. ``dir_rank`` and the trailing name make the order
        total, so it never depends on the order a directory enumerates in.

        The documented rule between the set-size and CU steps -- ranking equal-size
        sets by fallback topology -- is deliberately not modelled: it can only
        reorder two bundles whose chip-id sets are the same size and different, and
        no gfx950 release ships such a pair for one layout. _select_variant's
        exact-before-fallback pass is what actually decides those cases.
        """
        return (
            0 if self.chip_ids else 1,
            len(self.chip_ids),
            -self.cu_count if self.cu_count is not None else 1,
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


class _Bundles(NamedTuple):
    """What a layout ships: the bundles we understood, and the names we did not."""

    variants: list[_Variant]
    unparsed: list[Path]


def _variants_for(layout: str) -> _Bundles:
    """Every shipped bundle for this layout, with its parsed device predicates.

    Names whose token block does not parse are returned separately rather than
    dropped. Reading one as unconstrained would make it eligible for every device
    -- the silent wrong answer this selector exists to avoid -- but discarding it
    without trace is how a new release's added token would quietly widen the
    choice to a broader bundle, so callers report them.
    """
    prefix = _SS_HEAVY_STEM.format(layout=layout)
    suffix = f"_{GFX}.co"
    found: list[_Variant] = []
    unparsed: list[Path] = []
    for rank, directory in enumerate(_library_dirs()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"{prefix}*{suffix}")):
            match = _VARIANT_RE.match(path.name[len(prefix) : -len(suffix)])
            if match is None:
                unparsed.append(path)
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
    return _Bundles(variants=found, unparsed=unparsed)


def _select_variant(variants: list[_Variant], chip_id: int, cu_count: int) -> _Variant | None:
    """The bundle the loader would resolve to for this device, or None.

    Mirrors ``ExactLogicLibrary::findBestSolution``: walk the rows in build order,
    return the first *exact* match, and otherwise keep only the first fallback
    match -- so an exact match anywhere beats a fallback that appeared earlier.

    The loader's ``if(rv) return rv;`` guard means a row whose predicate matches
    but which yields no solution does not end the walk. Here that case cannot
    arise: a variant *is* a shipped file, so a row offering nothing for this
    family simply contributes none and the walk continues past it.
    """
    fallback: _Variant | None = None
    for variant in sorted(variants, key=lambda candidate: candidate.row_order):
        admits = variant.admits(chip_id, cu_count)
        if admits == _EXACT:
            return variant
        if admits == _FALLBACK and fallback is None:
            fallback = variant
    return fallback


def _library_for(layout: str, chip_id: int = GATE_CHIP_ID, cu_count: int = GATE_CU_COUNT) -> Path:
    variants, unparsed = _variants_for(layout)
    if not variants:
        # Distinguish "hipBLASLt shipped nothing" from "it shipped bundles whose
        # names I did not understand" -- the latter is the expected failure mode of
        # a release that adds a token, and sends the reader somewhere very different.
        if unparsed:
            names = ", ".join(sorted(path.name for path in unparsed))
            raise SystemExit(
                f"no usable {GFX} f32 SS Tensile bundle for layout {layout}: "
                f"{len(unparsed)} shipped, none with a recognised device-token "
                f"block: {names}"
            )
        tried = " or ".join(str(d / _SS_HEAVY.format(layout=layout)) for d in _library_dirs())
        raise SystemExit(f"no {GFX} f32 SS Tensile bundle for layout {layout}: {tried}")
    if unparsed:
        # Usable bundles exist, so this is not fatal -- but the skipped name may
        # have been the more specific one, in which case the choice below is
        # broader than the loader's. Say so rather than widening silently.
        for path in unparsed:
            print(
                f"warning: ignoring {path.name}: unrecognised device-token block; "
                f"selection for layout {layout} may be broader than hipBLASLt's",
                file=sys.stderr,
            )
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
