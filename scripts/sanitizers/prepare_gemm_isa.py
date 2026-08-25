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
# order is fixed, and the ids are lowercase hex. Anything else does not parse,
# which is fatal rather than skipped -- see _library_for. Extending this regex is
# the intended response to a release that adds a token.
_VARIANT_RE = re.compile(r"^(?:_CU(?P<cu>[0-9]+))?(?:_ID(?P<ids>[0-9a-f]+(?:-[0-9a-f]+)*))?$")

# tensilelite's ChipIdRegistry, mirrored as data. A gfx950 device may use a
# solution built for one of its fallback ids, and PciChipIdEqual accepts that via
# ChipIdRegistry::canUseSolution -- so a bundle naming only 0x75a0 still serves an
# MI355X. Ignoring this made the selector fail closed on five real chip ids that
# hipBLASLt serves from the 0x75a0 bundles.
#
# Transcribed from ChipIdRegistry::chipIdFallbacks in
# rocm-libraries/projects/hipblaslt/tensilelite/include/Tensile/AMDGPUPredicates.hpp
# (develop, read 2026-08-25, matching the hipBLASLt 1.4.1 docs). Unlike the row
# table in tests/sanitizers/fixtures, this mirror cannot be pinned to an image:
# no ROCm image ships the header, so there is nothing on disk to diff against.
# Its drift story is therefore the error path rather than a test -- a chip id
# added upstream is one this table lacks, which fails closed naming the device and
# pointing here (see _library_for). Safe direction, and diagnosable, but it will
# not announce itself until someone runs on the new part.
_CHIP_ID_FALLBACKS: dict[int, tuple[int, ...]] = {
    0x75B0: (0x75A0,),
    0x75A2: (0x75A0,),
    0x75B2: (0x75A0,),
    0x75A3: (0x75A0,),
    0x75B3: (0x75A0,),
    0x75A8: (0x75A0,),
    0x75B8: (0x75A0,),
}
# Every entry above targets this id, and it has no entry of its own -- so it is a
# registered chip despite being absent from the table's keys.
_CHIP_ID_FALLBACK_ROOT = 0x75A0

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
    def row_order(self) -> tuple[int, int, int, str]:
        """Sort key reproducing tensilelite's build-time hardware-row comparator.

        Documented order: rows with chip ids before rows without; then smaller
        chip-id sets ("exactness first"); then, when chip precedence does not
        decide, the higher CU count first. Rows with no CU constraint sort behind
        the constrained ones. The trailing name makes the order total -- names are
        unique within the one directory these all come from -- so it never depends
        on the order that directory enumerates in.

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
    bundles one level deeper under the gfx target (library/gfx950/...). An install
    is one shape or the other, never a blend -- see ``_variants_for``, which takes
    the first of these that holds anything and never resolves across both.
    """
    return [HIPBLASLT_LIBRARY, HIPBLASLT_LIBRARY / GFX]


class _Bundles(NamedTuple):
    """What a layout ships: the bundles we understood, and the names we did not."""

    variants: list[_Variant]
    unparsed: list[Path]


def _variants_for(layout: str) -> _Bundles:
    """The bundles for this layout, from a single tree, with their predicates.

    Resolution stops at the first directory holding anything for this layout,
    because the two directories are alternative *layouts* rather than one
    catalogue split in half. Merging them would let an incidental or stale
    ``gfx950/`` tree outrank the flat install it sits beside, and silently: a
    tokenised nested bundle is more specific than an untokenised flat one, and
    specificity is settled before directory order is ever consulted. Neither
    real-image check could have caught it -- the classic base has no nested tree
    and the wheel image has nothing at the flat level.

    "Anything" deliberately includes names that do not parse. A stale tree of
    unrecognised names is precisely the case where falling through to the other
    layout would look like success, so it fails loudly instead.

    Names whose token block does not parse are returned separately rather than
    dropped. Reading one as unconstrained would make it eligible for every device
    -- the silent wrong answer this selector exists to avoid -- while discarding
    it without trace is how a new release's added token would quietly widen the
    choice to a broader bundle. ``_library_for`` treats any of them as fatal.
    """
    prefix = _SS_HEAVY_STEM.format(layout=layout)
    suffix = f"_{GFX}.co"
    for directory in _library_dirs():
        if not directory.is_dir():
            continue
        found: list[_Variant] = []
        unparsed: list[Path] = []
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
                )
            )
        if found or unparsed:
            return _Bundles(variants=found, unparsed=unparsed)
    return _Bundles(variants=[], unparsed=[])


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
    if unparsed:
        # Fatal even when other bundles parsed. Selecting among candidates only
        # means anything if every candidate's predicates are known: an
        # unrecognised token block could be the loader's exact choice for this
        # device, in which case resolution silently settles on a broader bundle.
        # Warning and continuing was not enough -- this artifact's digest gets
        # recorded and blessed into sanitizer baselines, so a plausible-but-wrong
        # object outlives the log line that mentioned it. Failing here costs a
        # loud nightly failure at the moment someone is already bumping ROCm,
        # which is the cheapest time to extend the grammar.
        names = ", ".join(sorted(path.name for path in unparsed))
        shipped = len(variants) + len(unparsed)
        detail = (
            f"none of {shipped} with a recognised device-token block"
            if not variants
            else f"{len(unparsed)} of {shipped} with an unrecognised device-token block"
        )
        raise SystemExit(
            f"no usable {GFX} f32 SS Tensile bundle for layout {layout}: {detail}, "
            f"so the loader's choice cannot be established: {names}"
        )
    if not variants:
        tried = " or ".join(str(d / _SS_HEAVY.format(layout=layout)) for d in _library_dirs())
        raise SystemExit(f"no {GFX} f32 SS Tensile bundle for layout {layout}: {tried}")
    selected = _select_variant(variants, chip_id, cu_count)
    if selected is None:
        offered = ", ".join(sorted(variant.path.name for variant in variants))
        # A chip id upstream has added to ChipIdRegistry but this mirror lacks
        # reaches here, and it is the one drift this table cannot detect any other
        # way, so name the cause rather than leaving it to be inferred.
        hint = (
            ""
            if chip_id in _CHIP_ID_FALLBACKS or chip_id == _CHIP_ID_FALLBACK_ROOT
            else f"; 0x{chip_id:x} is absent from this script's ChipIdRegistry mirror, "
            "so if it is a newer part the mirror needs updating from tensilelite"
        )
        raise SystemExit(
            f"no {GFX} f32 SS Tensile bundle for layout {layout} serves "
            f"chip 0x{chip_id:x} with {cu_count} CUs; shipped variants: {offered}{hint}"
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
