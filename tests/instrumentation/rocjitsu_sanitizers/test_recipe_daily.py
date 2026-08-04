"""Tests for sanitizer recipe loader, resolvers, and parsers."""

from __future__ import annotations

import json
from pathlib import Path

from aorta.instrumentation.rocjitsu_sanitizers.backends import support
from aorta.instrumentation.rocjitsu_sanitizers.models import SelectionRequirement
from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe
from aorta.instrumentation.rocjitsu_sanitizers.selection import (
    observations_from_gemm_csv,
    observations_from_rocprof_trace,
    select_kernels,
)

_REPO = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO / "recipes" / "sanitizers" / "fixtures"


def test_gfx950_consan_support_is_full() -> None:
    policy = support("consan", "gfx950")
    assert policy["level"] == "full"


def test_load_daily_waitcheck_recipe() -> None:
    recipe = load_sanitizer_recipe(_REPO / "recipes" / "sanitizers" / "daily-waitcheck-gemm.yaml")
    assert recipe.target == "gfx950"
    assert recipe.source_kind == "gemm_csv"
    assert recipe.sanitizers == ("waitcheck",)
    assert recipe.top_n == 3


def test_gemm_csv_resolver_is_deterministic(tmp_path: Path) -> None:
    csv_path = _FIXTURES / "gemm_shapes_unique.csv"
    isa_dir = tmp_path / "isa"
    isa_dir.mkdir()
    for idx in (374829, 375437, 375583):
        blob = isa_dir / f"sol_{idx}.hsaco"
        blob.write_bytes(f"fake-{idx}".encode())
    observations = observations_from_gemm_csv(csv_path, target="gfx950", isa_dir=isa_dir)
    worklist = select_kernels(
        observations,
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=3,
    )
    assert len(worklist.kernels) == 3
    assert worklist.kernels[0].identity.name.startswith("gemm_")
    assert worklist.kernels[0].identity.code_object_scan


def test_rocprof_trace_adapter_drops_runtime_copies(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n"
        'KERNEL_DISPATCH,kernel_a,0,1000\n'
        'KERNEL_DISPATCH,kernel_a,0,2000\n'
        'KERNEL_DISPATCH,__amd_rocclr_copyBuffer,0,500\n',
        encoding="utf-8",
    )
    observations = observations_from_rocprof_trace(trace, target="gfx950")
    assert len(observations) == 1
    assert observations[0].identity.name == "kernel_a"
    assert observations[0].dispatch_count == 2


def test_verdict_baselines_fixture_present() -> None:
    baselines = json.loads((_FIXTURES / "expected" / "verdict_baselines.json").read_text())
    assert baselines["waitcheck_gemm"]["overall_verdict"] == "warn"
    assert baselines["consan_clean"]["overall_verdict"] == "pass"
    assert baselines["consan_racy"]["overall_verdict"] == "fail"
