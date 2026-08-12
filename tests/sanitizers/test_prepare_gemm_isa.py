"""Unit tests for the sanitizer GEMM fixture extractor (pure logic; no ROCm needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
