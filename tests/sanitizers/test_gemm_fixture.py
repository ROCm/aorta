"""Provenance + determinism guard for the synthetic GEMM-shape fixture.

The committed fixture must be reproducible from the generator and its seed (so
its provenance is the generator, not any customer-derived trace) and must carry
the "no customer data" provenance header.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "recipes" / "sanitizers" / "fixtures" / "gemm_shapes_unique.csv"


def _load_generator():
    path = _REPO_ROOT / "scripts" / "sanitizers" / "gen_gemm_fixture.py"
    spec = importlib.util.spec_from_file_location("gen_gemm_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def test_fixture_matches_generator_output() -> None:
    rendered = gen.render(gen.DEFAULT_SEED, gen.DEFAULT_ROWS)
    assert _FIXTURE.read_text(encoding="utf-8") == rendered


def test_generator_is_deterministic() -> None:
    first = gen.render(gen.DEFAULT_SEED, gen.DEFAULT_ROWS)
    second = gen.render(gen.DEFAULT_SEED, gen.DEFAULT_ROWS)
    assert first == second


def test_fixture_declares_no_customer_provenance() -> None:
    header = _FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("#")
    assert "no customer data" in header


def test_fixture_has_no_batch_identifiers() -> None:
    # A crude scrub check over the data rows (not the provenance header): the
    # synthetic fixture must not carry names/paths/hashes.
    data = "\n".join(
        line
        for line in _FIXTURE.read_text(encoding="utf-8").lower().splitlines()
        if not line.lstrip().startswith("#")
    )
    for banned in ("customer", "/opt/", ".hsaco", "sha256", "http"):
        assert banned not in data
