"""Unit tests for the sanitizer GEMM fixture extractor (pure logic; no ROCm needed)."""

from __future__ import annotations

import collections
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
    """Classic behaviour is unchanged when a per-gfx dir happens to sit beside it.

    Same name in both trees, so the two bundles have identical predicates. That
    makes this necessary but not sufficient on its own: it cannot distinguish
    "the flat tree wins" from "the tie-break happened to pick flat" -- which is
    what the next test is for.
    """
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

# The three variants ROCm 7.14 ships for every one of the four SS layouts. ROCm
# 10.0 ships the same three, re-measured against that image -- so this set is not
# 7.14-specific despite the name, and the ROCm 10 flip needed no change here.
# (ROCm 10 does ship _CU128_ID75a0 and _ID75a8 bundles for gfx950, as 7.14 already
# did, but for the fp16/bf16/f8 families rather than this f32 SS one.)
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
        for variant in gen._variants_for("Ailk_Bjlk").variants
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


# One tree per layout. The two library directories are alternative *layouts*, so
# resolution stops at the first that holds anything -- otherwise predicate
# specificity, which is settled first, lets a nested bundle outrank the flat
# install it sits beside. No real image can exercise the interaction: the classic
# base has no nested tree and the wheel image nothing at the flat level.


def test_a_more_specific_nested_bundle_does_not_outrank_the_flat_tree(monkeypatch, tmp_path):
    """The flat tree wins outright, not merely on a tie-break.

    This is the case that separates "one tree per layout" from "merge both and
    rank": a tokenised nested bundle is *more specific* than an untokenised flat
    one, so any merge -- however it orders directories -- resolves to the nested
    file and silently changes the extracted object and the digest recorded beside
    it. The same-name test above cannot see this, since equal predicates are
    exactly the case where a directory tie-break is reached.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    flat = _plant(tmp_path, "Ailk_Bjlk", _V_75A0)[_V_75A0]
    nested = _plant(tmp_path / gen.GFX, "Ailk_Bjlk", _V_CU256_75A0)[_V_CU256_75A0]
    bundles = gen._variants_for("Ailk_Bjlk")
    assert [v.path for v in bundles.variants] == [flat], "the nested tree must not contribute"
    assert gen._library_for("Ailk_Bjlk") == flat
    # The nested bundle really is what a merged ranking would have picked.
    assert nested.is_file()
    assert (
        gen._Variant(nested, 256, frozenset({0x75A0})).row_order
        < gen._Variant(flat, None, frozenset({0x75A0})).row_order
    )


def test_a_stale_tree_of_unparsed_names_does_not_fall_through_to_the_other_layout(
    monkeypatch, tmp_path
):
    """Stopping at the first *populated* tree counts names that do not parse.

    Otherwise a leftover tree whose tokens this script does not recognise would
    silently hand resolution to the other layout -- the one shape where falling
    through looks like success instead of failing.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    stale = _plant(tmp_path, "Ailk_Bjlk", "_XPACK7")["_XPACK7"]
    _plant(tmp_path / gen.GFX, "Ailk_Bjlk", *_SHIPPED_714)
    bundles = gen._variants_for("Ailk_Bjlk")
    assert bundles.variants == [] and bundles.unparsed == [stale]
    with pytest.raises(SystemExit, match="none of 1 with a recognised device-token block"):
        gen._library_for("Ailk_Bjlk")


def test_the_nested_tree_is_used_when_the_flat_one_holds_nothing_for_this_layout(
    monkeypatch, tmp_path
):
    """The real wheel shape: ``library/`` holds only the per-arch directories.

    A flat tree that exists but holds nothing *for this layout* must not count as
    populated, or the 7.14 layout would never be reached at all.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Alik_Bljk", _V_CU256_75A0)  # a different layout, flat
    nested = _plant(tmp_path / gen.GFX, "Ailk_Bjlk", *_SHIPPED_714)
    assert gen._library_for("Ailk_Bjlk") == nested[_V_CU256_75A0]


def test_classic_single_bundle_is_chosen_for_every_device(monkeypatch, tmp_path):
    """The classic single-bundle tree must stay byte-identical, and device-independent.

    Its one bundle per layout carries no predicates, so no device may change the
    answer -- verified end to end against a real classic tree, whose extracted
    objects' digests are unchanged by this selector. That tree was the CI base
    while ROCm 7.2.4 was pinned; the current ROCm 10.0 pin is a wheel, per-device
    tree, and the extraction has been re-run end to end against it as well
    (three real code objects, correct per-layout digests). Classic installs are
    what customers run, so this path stays covered here either way.
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


def test_same_chip_with_a_different_cu_count_falls_to_the_generic_variant(monkeypatch, tmp_path):
    """A CU predicate that does not hold excludes the bundle outright.

    Measured: 7.14 ships no CU128 bundle for this family, so a 128-CU 0x75a0
    part reaches the chip's unspecialised library. ``findBestSolution`` guards
    its return with ``if(rv)``, so a row whose predicate matches but which
    offers nothing does not end the walk -- which is why the CU128 row (fp16
    only for this layout) does not strand the device.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    assert gen._library_for("Ailk_Bjlk", 0x75A0, 128) == planted[_V_75A0]


def test_a_multi_chip_variant_serves_every_chip_it_names(monkeypatch, tmp_path):
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    for chip in (0x75A2, 0x75A3):
        assert gen._library_for("Ailk_Bjlk", chip, 256) == planted[_V_75A3_75A2]


def test_registry_mirrors_tensilelites_chip_id_fallback_graph():
    """Every gfx950 id tensilelite registers a fallback for, with its target.

    Transcribed from ``ChipIdRegistry::chipIdFallbacks``. Getting this wrong is
    not visible on the gate (0x75a0 needs no fallback), so it is pinned here.
    """
    assert gen._CHIP_ID_FALLBACKS == {
        0x75B0: (0x75A0,),
        0x75A2: (0x75A0,),
        0x75B2: (0x75A0,),
        0x75A3: (0x75A0,),
        0x75B3: (0x75A0,),
        0x75A8: (0x75A0,),
        0x75B8: (0x75A0,),
    }
    # 0x75a0 is the fallback root: it targets nothing and everything targets it.
    assert 0x75A0 not in gen._CHIP_ID_FALLBACKS
    assert {target for targets in gen._CHIP_ID_FALLBACKS.values() for target in targets} == {0x75A0}


@pytest.mark.parametrize("chip", [0x75A8, 0x75B8, 0x75B0, 0x75B2, 0x75B3])
def test_a_chip_with_no_bundle_of_its_own_uses_its_registry_fallback(monkeypatch, tmp_path, chip):
    """These ids resolve through the fallback graph, not to a failure.

    ``PciChipIdEqual::operator()`` accepts a solution via
    ``ChipIdRegistry::canUseSolution``, so a bundle naming only 0x75a0 does serve
    these parts. An earlier revision failed closed on all five, which would have
    been a fixture the loader could never have chosen -- and 0x75a8 is a real id
    the shipped index knows.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    assert gen._library_for("Ailk_Bjlk", chip, 256) == planted[_V_CU256_75A0]


def test_an_exact_match_beats_a_fallback_that_sorts_earlier(monkeypatch, tmp_path):
    """An exact match anywhere beats a fallback match that appeared earlier.

    For 0x75a2 the CU256_ID75a0 bundle sorts first (smaller chip set) and admits
    the device by fallback, while ID75a3-75a2 sorts last and admits it exactly.
    Taking the first admitting bundle would pick the fallback; the loader does not.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    order = sorted(gen._variants_for("Ailk_Bjlk").variants, key=lambda v: v.row_order)
    assert order[0].path == planted[_V_CU256_75A0], "the fallback candidate sorts first"
    assert order[0].admits(0x75A2, 256) == gen._FALLBACK
    assert gen._library_for("Ailk_Bjlk", 0x75A2, 256) == planted[_V_75A3_75A2]


def test_an_unregistered_chip_id_still_fails_closed(monkeypatch, tmp_path):
    """Fallback is a closed registry, not "anything gfx950 will do".

    A chip id tensilelite does not know has no fallback edge, so no bundle admits
    it and there is no honest answer to give.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    with pytest.raises(SystemExit) as excinfo:
        gen._library_for("Ailk_Bjlk", 0x1234, 256)
    message = str(excinfo.value)
    assert "0x1234" in message and "256 CUs" in message
    # Naming what *was* shipped is what makes the failure actionable.
    for variant in _SHIPPED_714:
        assert f"{variant}_{gen.GFX}.co" in message
    # And since the registry mirror cannot be diffed against any image, this
    # error is its only drift signal -- so it has to name that as the cause.
    assert "ChipIdRegistry mirror" in message


def test_a_registered_chip_that_finds_nothing_does_not_blame_the_registry(monkeypatch, tmp_path):
    """The mirror hint must not fire for an id the mirror already knows.

    0x75a3 is registered, so a miss here means the layout ships nothing usable --
    pointing at the registry would send the reader to the wrong place.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", "_CU999_ID75a3")
    with pytest.raises(SystemExit) as excinfo:
        gen._library_for("Ailk_Bjlk", 0x75A3, 256)
    message = str(excinfo.value)
    assert "0x75a3" in message
    assert "ChipIdRegistry mirror" not in message
    # The fallback root has no entry of its own, so it must not be blamed either.
    _plant(tmp_path, "Ailk_Bjlk", "_CU999_ID75a0")
    with pytest.raises(SystemExit) as excinfo:
        gen._library_for("Ailk_Bjlk", gen._CHIP_ID_FALLBACK_ROOT, 256)
    assert "ChipIdRegistry mirror" not in str(excinfo.value)


def test_selection_does_not_depend_on_directory_enumeration_order(monkeypatch, tmp_path):
    """Two trees differing only in creation order must resolve identically."""
    stem = gen._SS_HEAVY_STEM.format(layout="Ailk_Bjlk")
    chosen = []
    for order in (_SHIPPED_714, tuple(reversed(_SHIPPED_714))):
        root = tmp_path / f"order{len(chosen)}"
        _plant(root, "Ailk_Bjlk", *order)
        monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", root)
        chosen.append(gen._library_for("Ailk_Bjlk").name)
    assert chosen[0] == chosen[1] == f"{stem}{_V_CU256_75A0}_{gen.GFX}.co"


def test_row_order_follows_the_documented_build_time_comparator(monkeypatch, tmp_path):
    """Chip ids present first, then smaller chip sets, then higher CU count.

    Transcribed from hipBLASLt's "Build-time row ordering (fallback-aware)".
    Synthetic file set on purpose: the shipped 7.14 bundles never make the
    set-size and CU steps disagree (every bundle carries `_ID`, and the two
    single-chip ones differ only by CU), so no real tree can pin this order.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", "_CU256", _V_75A0, _V_CU256_75A0, _V_75A3_75A2, "")
    variants = gen._variants_for("Ailk_Bjlk").variants
    order = [v.path for v in sorted(variants, key=lambda v: v.row_order)]
    assert order == [
        planted[_V_CU256_75A0],  # {75a0}, CU 256
        planted[_V_75A0],  # {75a0}, no CU
        planted[_V_75A3_75A2],  # {75a2,75a3} -- larger set sorts after
        planted["_CU256"],  # no chip id, CU 256
        planted[""],  # no chip id, no CU
    ]
    assert [v.path for v in variants] != order, "name order must differ, or this proves nothing"
    assert gen._library_for("Ailk_Bjlk", 0x75A0, 256) == planted[_V_CU256_75A0]


def test_a_narrower_chip_set_wins_even_when_it_sorts_later_by_name(monkeypatch, tmp_path):
    """Exactness first: a smaller chip-ID set sorts first.

    This is the one case where walking bundles in directory order gives a
    different answer from walking them in the documented row order, so it is what
    proves the selector sorts at all. `-` (0x2d) sorts before `_` (0x5f), so
    `_ID75a0-75b0` precedes `_ID75a0` by name while being the broader set; both
    admit 0x75a0 exactly, so only the ordering rule separates them.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", "_ID75a0", "_ID75a0-75b0")
    by_name = [v.path for v in gen._variants_for("Ailk_Bjlk").variants]
    assert by_name[0] == planted["_ID75a0-75b0"], "the broader set must sort first by name"
    assert gen._library_for("Ailk_Bjlk", 0x75A0, 256) == planted["_ID75a0"]


def test_an_unrecognised_token_block_is_ignored_not_treated_as_generic(monkeypatch, tmp_path):
    """A future token must not be mistaken for "no constraints".

    Reading an unparseable block as unconstrained would make it eligible for
    every device, which is exactly the silent-wrong-answer this selector exists
    to avoid; the only bundle here is therefore unusable and the call fails.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", "_XPACK7")
    bundles = gen._variants_for("Ailk_Bjlk")
    assert bundles.variants == []
    assert bundles.unparsed == [planted["_XPACK7"]]
    with pytest.raises(SystemExit, match="Ailk_Bjlk"):
        gen._library_for("Ailk_Bjlk")


@pytest.mark.parametrize("token", ["_ID75A0", "_ID75a0_CU256", "_CU256_ID75a0_EXTRA"])
def test_the_token_grammar_is_exactly_as_strict_as_documented(monkeypatch, tmp_path, token):
    """Uppercase hex, reversed tokens and trailing tokens are all unparsed.

    Each is defensible on its own -- they fail closed -- but they are the shapes a
    future release could introduce, so pin which ones the grammar rejects rather
    than discovering it during a bump.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", token)
    bundles = gen._variants_for("Ailk_Bjlk")
    assert bundles.variants == [] and len(bundles.unparsed) == 1


def test_a_zero_padded_cu_token_parses(monkeypatch, tmp_path):
    """`_CU0256` is accepted and means 256 -- not silently a different device."""
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", "_CU0256_ID75a0")
    (variant,) = gen._variants_for("Ailk_Bjlk").variants
    assert variant.cu_count == 256


def test_bundles_that_ship_but_do_not_parse_are_named_not_reported_as_absent(
    monkeypatch, tmp_path, capsys
):
    """The message must not send the reader looking for files that are right there.

    Naming the untokenised 7.2.4 path reads as "hipBLASLt shipped nothing for this
    layout", when the truth is "it shipped bundles whose names I did not
    understand" -- and since skipping unknown tokens is deliberate, that is the
    *expected* failure mode of a release that adds one.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    _plant(tmp_path, "Ailk_Bjlk", "_XPACK7", "_CU256_ID75a0_XPACK7")
    with pytest.raises(SystemExit) as excinfo:
        gen._library_for("Ailk_Bjlk")
    message = str(excinfo.value)
    assert "none of 2 with a recognised device-token block" in message
    for name in ("_XPACK7_gfx950.co", "_CU256_ID75a0_XPACK7_gfx950.co"):
        assert name in message
    # And it must NOT claim the canonical 7.2.4 path was what was missing.
    assert str(tmp_path / gen._SS_HEAVY.format(layout="Ailk_Bjlk")) not in message


def test_an_unparsed_bundle_is_fatal_even_when_others_resolve(monkeypatch, tmp_path):
    """The widening case fails closed rather than warning and continuing.

    This is the shape that matters: a release tokenises the *specific* bundle and
    leaves a broader one parseable. Selection would settle on `_ID75a0` and exit
    0, even though the skipped `_CU256_ID75a0_*` may be the loader's exact choice
    for this device. Ordering only means something when every candidate's
    predicates are known, so with any of them unknown there is no answer to give.

    Warning and continuing was the earlier behaviour and was not enough: this
    artifact's digest is recorded and blessed into sanitizer baselines, so a
    plausible-but-wrong object outlives the log line that mentioned it.
    """
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", "_CU256_ID75a0_XPACK7", _V_75A0)
    with pytest.raises(SystemExit) as excinfo:
        gen._library_for("Ailk_Bjlk")
    message = str(excinfo.value)
    assert "_CU256_ID75a0_XPACK7_gfx950.co" in message
    assert "1 of 2 with an unrecognised device-token block" in message
    assert "the loader's choice cannot be established" in message
    # The broader bundle is right there and parses -- that is the point.
    assert planted[_V_75A0].is_file()


def test_every_bundle_parsing_is_what_makes_a_selection_possible(monkeypatch, tmp_path, capsys):
    """The normal case resolves and says nothing, so a failure means something."""
    monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", tmp_path)
    planted = _plant(tmp_path, "Ailk_Bjlk", *_SHIPPED_714)
    assert gen._library_for("Ailk_Bjlk") == planted[_V_CU256_75A0]
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""


def test_selection_reproduces_the_shipped_lazy_master_mapping(monkeypatch, tmp_path):
    """Drift check against hipBLASLt's own hardware->library table.

    ``fixtures/tensile_lazy_gfx950_rows.json`` records the predicate rows decoded
    from the ``TensileLibrary_lazy_gfx950`` index in the real 7.14 image, together
    with the library each row serves per layout. The filename tokens only restate
    those predicates, so a selector that picks by name has to land on the same
    file the loader would -- otherwise the fixtures stop representing what runs.

    The model below is ``ExactLogicLibrary::findBestSolution`` transcribed, not a
    guess at it: rows are walked in order, the first *exact* match returns, only
    the first *fallback* match is kept, and a row whose predicate matches but
    which offers nothing for this family does not end the walk -- the loader's
    ``return`` sits inside an ``if(rv)``. That last point is what makes the CU128
    row skippable, and it is the one thing here that cannot be read off the table.
    """
    truth = json.loads(_LAZY_ROWS.read_text(encoding="utf-8"))
    # Validate the recorded table's shape up front: a bare KeyError from deep in
    # the loop below would say far less about a mangled fixture than this does.
    assert (
        truth.get("source", {}).get("library_type") == "Hardware"
    ), "fixture must record a Hardware library, whose rows are walked in order"
    rows = truth.get("rows")
    assert isinstance(rows, list) and rows, "fixture must record the predicate rows"
    for row in rows:
        assert row.get("processor") == gen.GFX, f"unexpected processor in row {row}"
        assert isinstance(row.get("chip_ids"), list), f"row {row} has no chip_ids"
        assert isinstance(row.get("libraries"), dict), f"row {row} has no libraries"
        assert row.get("cu_count") is None or isinstance(row["cu_count"], int), row

    def find_best(chip_id: int, cu_count: int, layout: str) -> str | None:
        """``ExactLogicLibrary::findBestSolution`` over the recorded rows."""
        fallback = None
        for row in rows:
            if row["cu_count"] is not None and row["cu_count"] != cu_count:
                continue  # CUCount predicate rejects the row outright
            exact = chip_id in row["chip_ids"]
            usable = exact or bool(
                set(row["chip_ids"]).intersection(gen._CHIP_ID_FALLBACKS.get(chip_id, ()))
            )
            if not usable:
                continue
            offered = row["libraries"].get(layout)
            if exact:
                # `if(rv) return rv;` -- an empty row does not end the walk.
                if offered is not None:
                    return offered
            elif fallback is None and offered is not None:
                fallback = offered
        return fallback

    # Exercise every id tensilelite registers, not only the ones this build's
    # rows happen to name -- the fallback ids are exactly the ones with no row.
    chips = sorted({chip for row in rows for chip in row["chip_ids"]} | set(gen._CHIP_ID_FALLBACKS))
    cu_counts = sorted({row["cu_count"] for row in rows if row["cu_count"]} | {256, 128, 32})
    layouts = sorted({layout for row in rows for layout in row["libraries"]})
    assert chips and cu_counts and layouts, "fixture must exercise something"

    resolved = collections.Counter()
    for chip_id, cu_count, layout in itertools.product(chips, cu_counts, layouts):
        root = tmp_path / f"{chip_id}-{cu_count}-{layout}"
        # Plant exactly what 7.14 ships for this layout.
        _plant(root, layout, *_SHIPPED_714)
        monkeypatch.setattr(gen, "HIPBLASLT_LIBRARY", root)
        expected = find_best(chip_id, cu_count, layout)
        where = f"chip 0x{chip_id:x} cu {cu_count} {layout}"
        assert expected is not None, f"no library for {where}"
        chosen = gen._library_for(layout, chip_id, cu_count)
        assert chosen.name == f"{expected}.co", where
        resolved[expected.rsplit("Cijk_Dijk", 1)[-1]] += 1
    assert sum(resolved.values()) == len(chips) * len(cu_counts) * len(layouts)

    # Every registered gfx950 id resolves, because they all reach 0x75a0 -- which
    # is exactly why failing closed on 0x75a8 (an earlier revision did) was wrong.
    # All three shipped variants must be reachable, or the sweep is asserting one
    # answer over and over.
    assert set(resolved) == {
        f"{_V_CU256_75A0}_{gen.GFX}",
        f"{_V_75A0}_{gen.GFX}",
        f"{_V_75A3_75A2}_{gen.GFX}",
    }, resolved

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
