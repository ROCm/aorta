"""Unit tests for the sanitizer GEMM fixture extractor (pure logic; no ROCm needed)."""

from __future__ import annotations

import importlib.util
import itertools
import json
import shutil
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAZY_ROWS = Path(__file__).parent / "fixtures" / "tensile_lazy_gfx950_rows.json"


def _load():
    path = _REPO_ROOT / "scripts" / "sanitizers" / "prepare_gemm_isa.py"
    spec = importlib.util.spec_from_file_location("prepare_gemm_isa", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load()


def test_layout_map_covers_all_transposes():
    assert gen._LAYOUT[("N", "T")] == "Ailk_Bjlk"
    assert gen._LAYOUT[("N", "N")] == "Ailk_Bljk"
    assert gen._LAYOUT[("T", "N")] == "Alik_Bljk"
    assert gen._LAYOUT[("T", "T")] == "Alik_Bjlk"


def test_ss_heavy_name_formatting():
    assert gen._SS_HEAVY.format(layout="Ailk_Bjlk") == (
        "TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_"
        "Ailk_Bjlk_Cijk_Dijk_gfx950.co"
    )


def test_read_rows_skips_comments_sorts_and_respects_top_n(tmp_path):
    csv = tmp_path / "shapes.csv"
    csv.write_text(
        "# synthetic header comment\n"
        "rank,count,transA,transB,top_solution_idx\n"
        "1,10,N,T,111\n"
        "2,50,N,N,222\n"
        "3,30,T,N,333\n",
        encoding="utf-8",
    )
    # Sorted by count descending, capped at top_n.
    assert [r["top_solution_idx"] for r in gen._read_rows(csv, 2)] == ["222", "333"]
    # top-n 0 selects no shapes -> the consan-object-only extraction mode.
    assert gen._read_rows(csv, 0) == []


def test_library_for_missing_bundle_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)  # empty dir
    with pytest.raises(SystemExit, match="Ailk_Bjlk"):
        gen._library_for("Ailk_Bjlk")


def test_library_for_returns_matching_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    lib = tmp_path / gen._SS_HEAVY.format(layout="Ailk_Bjlk")
    lib.write_bytes(b"stub")
    assert gen._library_for("Ailk_Bjlk") == lib


def test_library_for_resolves_nested_gfx_subdir(monkeypatch, tmp_path):
    """The TheRock wheel layout nests bundles under ``library/<gfx>/`` (#381).

    Measured: classic ROCm 7.2.4 has a flat ``library/`` with ~3000 bundles,
    while the 7.14 wheel splits them per target and ``library/`` holds only
    the gfx directories. Looking only at the flat path finds nothing there,
    which is a hard failure -- the sanitizer GEMM fixtures cannot be built.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    nested = tmp_path / gen.GFX
    nested.mkdir()
    lib = nested / gen._SS_HEAVY.format(layout="Ailk_Bjlk")
    lib.write_bytes(b"stub")
    assert gen._library_for("Ailk_Bjlk") == lib


def test_library_for_prefers_the_flat_bundle_when_both_exist(monkeypatch, tmp_path):
    """Classic behaviour is unchanged when a per-gfx dir happens to sit beside it."""
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    bundle = gen._SS_HEAVY.format(layout="Ailk_Bjlk")
    flat = tmp_path / bundle
    flat.write_bytes(b"flat")
    nested = tmp_path / gen.GFX
    nested.mkdir()
    (nested / bundle).write_bytes(b"nested")
    assert gen._library_for("Ailk_Bjlk") == flat


def test_bundler_prefers_path(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/clang-offload-bundler")
    assert gen._bundler() == "/usr/bin/clang-offload-bundler"


def test_bundler_falls_back_to_the_resolved_llvm_bindir(monkeypatch, tmp_path):
    """ROCm's LLVM bindir is on PATH in neither install layout (#381).

    The classic image exports only /opt/rocm/bin and the wheel image only the
    venv's bin, so without this fallback the fixture build depends on the
    caller having prepended the bindir first.
    """
    monkeypatch.setattr(shutil, "which", lambda name: None)
    llvm_bin = tmp_path / "lib" / "llvm" / "bin"
    llvm_bin.mkdir(parents=True)
    bundler = llvm_bin / "clang-offload-bundler"
    bundler.write_bytes(b"stub")
    monkeypatch.setattr(gen, "LLVM_BIN_DIR", llvm_bin)
    assert gen._bundler() == str(bundler)


def test_bundler_missing_everywhere_names_the_bindir_it_tried(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(gen, "LLVM_BIN_DIR", tmp_path / "absent")
    with pytest.raises(SystemExit, match=str(tmp_path / "absent")):
        gen._bundler()


def test_library_for_error_names_both_candidate_paths(monkeypatch, tmp_path):
    """A miss must say where it looked, or the wheel case is undebuggable."""
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        gen._library_for("Ailk_Bjlk")
    message = str(excinfo.value)
    assert str(tmp_path / gen._SS_HEAVY.format(layout="Ailk_Bjlk")) in message
    assert str(tmp_path / gen.GFX) in message


# --------------------------------------------------------------------------
# Per-device variant selection (ROCm 7.14+).
#
# Through 7.2.4 a (family, layout, arch) triple named exactly one bundle. 7.14
# splits the gfx950 libraries per device and encodes the split in the filename
# using the same predicates hipBLASLt's lazy master index gates them on, so the
# tests below are named after real, measured file sets rather than invented ones.
# --------------------------------------------------------------------------

# The three variants ROCm 7.14 ships for every one of the four SS layouts.
_V_CU256_75A0 = "_CU256_ID75a0"
_V_75A0 = "_ID75a0"
_V_75A3_75A2 = "_ID75a3-75a2"
_SHIPPED_714 = (_V_CU256_75A0, _V_75A0, _V_75A3_75A2)


def _plant(directory: Path, layout: str, *variants: str) -> dict[str, Path]:
    """Create empty bundles for ``variants`` and return them by variant token."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = gen._SS_HEAVY_STEM.format(layout=layout)
    made = {}
    for variant in variants:
        path = directory / f"{stem}{variant}_{gen.GFX}.co"
        path.write_bytes(b"stub")
        made[variant] = path
    return made


def test_variant_tokens_parse_into_the_predicates_they_encode(monkeypatch, tmp_path):
    """``_CU<n>`` is a CU count and ``_ID<hex>[-<hex>]`` a set of PCI chip ids."""
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714, "")
    parsed = {
        variant.path.name: (variant.cu_count, sorted(variant.chip_ids))
        for variant in gen._variants_for("Ailk_Bjlk")
    }
    stem = gen._SS_HEAVY_STEM.format(layout="Ailk_Bjlk")
    assert parsed[f"{stem}_CU256_ID75a0_{gen.GFX}.co"] == (256, [0x75A0])
    assert parsed[f"{stem}_ID75a0_{gen.GFX}.co"] == (None, [0x75A0])
    assert parsed[f"{stem}_ID75a3-75a2_{gen.GFX}.co"] == (None, [0x75A2, 0x75A3])
    # The 7.2.4 name carries no tokens at all: unconstrained on both axes.
    assert parsed[f"{stem}_{gen.GFX}.co"] == (None, [])


def test_gate_defaults_name_the_measured_mi350x(monkeypatch, tmp_path):
    """rocprofv3 agent info on the gate reports Device_Id 30112 / Cu_Count 256."""
    assert (gen.GATE_CHIP_ID, gen.GATE_CU_COUNT) == (0x75A0, 256)


def test_classic_single_bundle_is_chosen_for_every_device(monkeypatch, tmp_path):
    """The 7.2.4 tree must stay byte-identical, and device-independent.

    Its one bundle per layout carries no predicates, so no device may change the
    answer -- verified end-to-end against the pinned 7.2.4 CI base, where the
    extracted objects' digests are unchanged by this selector.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", "")[""]
    for chip, cus in itertools.product((0x75A0, 0x75A2, 0x75A8, 0x1234), (256, 128, 32)):
        assert gen._library_for("Ailk_Bjlk", chip, cus) == planted


def test_gate_device_selects_the_cu_specialised_variant(monkeypatch, tmp_path):
    """0x75a0 + 256 CUs -> CU256_ID75a0, the lazy index's most specific row."""
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    assert gen._library_for("Ailk_Bjlk") == planted[_V_CU256_75A0]


def test_same_chip_with_a_different_cu_count_falls_to_the_generic_variant(
    monkeypatch, tmp_path
):
    """A CU predicate that does not hold excludes the bundle outright.

    Measured: 7.14 ships no CU128 bundle for this family, so a 128-CU 0x75a0
    part resolves to the chip's unspecialised library -- which is what the lazy
    index does too, its CU128 row offering nothing here.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    assert gen._library_for("Ailk_Bjlk", 0x75A0, 128) == planted[_V_75A0]


def test_a_multi_chip_variant_serves_every_chip_it_names(monkeypatch, tmp_path):
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    for chip in (0x75A2, 0x75A3):
        assert gen._library_for("Ailk_Bjlk", chip, 256) == planted[_V_75A3_75A2]


def test_a_device_no_variant_serves_fails_closed_and_lists_them(monkeypatch, tmp_path):
    """Silently handing back another device's kernels would be worse than failing.

    0x75a8 is a real gfx950 chip id the 7.14 index knows, yet no bundle in this
    family names it -- so there is no honest answer to give.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    with pytest.raises(SystemExit) as excinfo:
        gen._library_for("Ailk_Bjlk", 0x75A8, 256)
    message = str(excinfo.value)
    assert "0x75a8" in message and "256 CUs" in message
    # Naming what *was* shipped is what makes the failure actionable.
    for variant in _SHIPPED_714:
        assert f"{variant}_{gen.GFX}.co" in message


def test_selection_does_not_depend_on_directory_enumeration_order(
    monkeypatch, tmp_path
):
    """Two trees differing only in creation order must resolve identically."""
    stem = gen._SS_HEAVY_STEM.format(layout="Ailk_Bjlk")
    chosen = []
    for order in (_SHIPPED_714, tuple(reversed(_SHIPPED_714))):
        root = tmp_path / f"order{len(chosen)}"
        _plant(root, "Ailk_Bjlk", *order)
        monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", root)
        chosen.append(gen._library_for("Ailk_Bjlk").name)
    assert chosen[0] == chosen[1] == f"{stem}{_V_CU256_75A0}_{gen.GFX}.co"


def test_chip_specificity_outranks_cu_specificity(monkeypatch, tmp_path):
    """A chip match beats a CU match when only one of them can be had.

    Synthetic on purpose: no shipped 7.14 bundle carries ``_CU<n>`` without also
    carrying ``_ID<hex>``, so nothing on disk distinguishes the two orderings and
    a real-file test cannot pin this. The order still has to be *some* documented
    thing, and chip-first mirrors the lazy index, whose rows are chip-disjoint and
    only subdivide by CU count *within* a chip -- making the chip the outer
    discriminator there too.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", "_CU256", _V_75A0)
    assert gen._library_for("Ailk_Bjlk", 0x75A0, 256) == planted[_V_75A0]


def test_an_unrecognised_token_block_is_ignored_not_treated_as_generic(
    monkeypatch, tmp_path
):
    """A future token must not be mistaken for "no constraints".

    Reading an unparseable block as unconstrained would make it eligible for
    every device, which is exactly the silent-wrong-answer this selector exists
    to avoid; the only bundle here is therefore unusable and the call fails.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", "_XPACK7")
    assert gen._variants_for("Ailk_Bjlk") == []
    with pytest.raises(SystemExit, match="Ailk_Bjlk"):
        gen._library_for("Ailk_Bjlk")


def test_selection_reproduces_the_shipped_lazy_master_mapping(monkeypatch, tmp_path):
    """Drift check against hipBLASLt's own hardware->library table.

    ``fixtures/tensile_lazy_gfx950_rows.json`` records the predicate rows decoded
    from the ``TensileLibrary_lazy_gfx950`` index in the real 7.14 image, together
    with the library each row serves per layout. The filename tokens only restate
    those predicates, so a selector that picks by name has to land on the same
    file the loader would -- otherwise the fixtures stop representing what runs.

    Rows offering nothing for this family are the fall-through cases: the device
    must resolve to a *later* row's library, or to nothing at all.
    """
    truth = json.loads(_LAZY_ROWS.read_text(encoding="utf-8"))
    # Validate the recorded table's shape up front: a bare KeyError from deep in
    # the loop below would say far less about a mangled fixture than this does.
    assert truth.get("source", {}).get("library_type") == "Hardware", (
        "fixture must record a Hardware library, whose rows are resolved first-match"
    )
    rows = truth.get("rows")
    assert isinstance(rows, list) and rows, "fixture must record the predicate rows"
    for row in rows:
        assert row.get("processor") == gen.GFX, f"unexpected processor in row {row}"
        assert isinstance(row.get("chip_ids"), list), f"row {row} has no chip_ids"
        assert isinstance(row.get("libraries"), dict), f"row {row} has no libraries"
        assert row.get("cu_count") is None or isinstance(row["cu_count"], int), row

    def first_match(chip_id: int, cu_count: int, layout: str) -> str | None:
        """What the index resolves to: first row that matches *and* has a library."""
        for row in rows:
            if chip_id not in row["chip_ids"]:
                continue
            if row["cu_count"] is not None and row["cu_count"] != cu_count:
                continue
            if layout in row["libraries"]:
                return row["libraries"][layout]
        return None

    chips = sorted({chip for row in rows for chip in row["chip_ids"]})
    cu_counts = sorted({row["cu_count"] for row in rows if row["cu_count"]} | {256, 128, 32})
    layouts = sorted({layout for row in rows for layout in row["libraries"]})
    assert chips and cu_counts and layouts, "fixture must exercise something"

    checked = 0
    for chip_id, cu_count, layout in itertools.product(chips, cu_counts, layouts):
        root = tmp_path / f"{chip_id}-{cu_count}-{layout}"
        # Plant exactly what 7.14 ships for this layout.
        _plant(root, layout, *_SHIPPED_714)
        monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", root)
        expected = first_match(chip_id, cu_count, layout)
        if expected is None:
            # 0x75a8 is a real chip id the index knows but this family never
            # names, so there is no honest file to hand back.
            with pytest.raises(SystemExit):
                gen._library_for(layout, chip_id, cu_count)
        else:
            chosen = gen._library_for(layout, chip_id, cu_count)
            assert chosen.name == f"{expected}.co", f"chip 0x{chip_id:x} cu {cu_count} {layout}"
        checked += 1
    assert checked == len(chips) * len(cu_counts) * len(layouts)

    # The recorded table is the real one, not a hand-written stand-in: it must
    # still describe the device the fixtures are actually built for.
    gate = [
        row
        for row in rows
        if row["cu_count"] == gen.GATE_CU_COUNT and gen.GATE_CHIP_ID in row["chip_ids"]
    ]
    assert len(gate) == 1, "exactly one row pins the gate's chip id and CU count"
    assert all(
        name.endswith(f"{_V_CU256_75A0}_{gen.GFX}") for name in gate[0]["libraries"].values()
    )
