from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import (
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    observations_from_dispatch_rows,
    observations_from_magpie_report,
    observations_from_magpie_workspace,
    select_kernels,
)


def test_magpie_top_time_uses_aggregate_time_not_input_order() -> None:
    observations = observations_from_magpie_report(
        {
            "kernel_summary": [
                {"name": "small", "time_ms": 2.0, "percent": 2, "calls": 100},
                {"name": "hot", "time_ms": 40.0, "percent": 40, "calls": 2},
                {"name": "medium", "time_ms": 10.0, "percent": 10, "calls": 5},
            ]
        },
        target="gfx950",
    )

    worklist = select_kernels(
        observations,
        requirement=SelectionRequirement.TOP_TIME,
        top_n=2,
    )

    assert [item.identity.name for item in worklist.kernels] == ["hot", "medium"]


def test_magpie_workspace_reuses_public_adapter(tmp_path: Path) -> None:
    workspace = tmp_path / "benchmark"
    workspace.mkdir()
    (workspace / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "kernel_summary": [{"name": "kernel", "time_ms": 3.0, "calls": 4}],
            }
        )
    )

    observations = observations_from_magpie_workspace(
        workspace,
        target="gfx950",
    )

    assert len(observations) == 1
    assert observations[0].identity.name == "kernel"
    assert observations[0].sources == ("magpie",)


def test_dispatch_count_ranking_sorts_unsorted_rows() -> None:
    observations = observations_from_dispatch_rows(
        [
            {"name": "third", "count": "3"},
            {"name": "first", "count": "100"},
            {"name": "second", "count": "20"},
        ],
        target="gfx950",
    )

    worklist = select_kernels(
        observations,
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=2,
    )

    assert [item.identity.name for item in worklist.kernels] == ["first", "second"]


def test_selection_deduplicates_exact_identity_and_merges_sources() -> None:
    identity = KernelIdentity(
        name="kernel",
        target="gfx950",
        code_object="/tmp/kernel.hsaco",
        code_object_sha256="0" * 64,
        entry_offset=32,
    )
    worklist = select_kernels(
        [
            KernelObservation(
                identity=identity,
                total_time_ms=2,
                dispatch_count=5,
                sources=("magpie",),
            ),
            KernelObservation(
                identity=identity,
                total_time_ms=2,
                dispatch_count=5,
                sources=("dispatch_csv",),
            ),
        ],
        requirement=SelectionRequirement.TOP_TIME,
        top_n=2,
    )

    assert len(worklist.kernels) == 1
    assert worklist.kernels[0].total_time_ms == 2
    assert worklist.kernels[0].dispatch_count == 5
    assert worklist.kernels[0].sources == ("dispatch_csv", "magpie")


def test_exact_identity_deduplicates_profiler_name_variants() -> None:
    common = {
        "target": "gfx950",
        "code_object": "/tmp/kernel.hsaco",
        "code_object_sha256": "0" * 64,
        "entry_offset": 32,
    }
    worklist = select_kernels(
        [
            KernelObservation(
                identity=KernelIdentity(name="z_demangled", **common),
                total_time_ms=2,
                sources=("magpie",),
            ),
            KernelObservation(
                identity=KernelIdentity(name="a_mangled", **common),
                total_time_ms=2,
                sources=("tracelens",),
            ),
        ],
        requirement=SelectionRequirement.TOP_TIME,
        top_n=2,
    )

    assert len(worklist.kernels) == 1
    assert worklist.kernels[0].identity.name == "a_mangled"


@pytest.mark.parametrize("top_n", [0, -1, True])
def test_selection_rejects_invalid_top_n(top_n: int) -> None:
    with pytest.raises(ValueError, match="top_n"):
        select_kernels(
            [],
            requirement=SelectionRequirement.TOP_TIME,
            top_n=top_n,
        )


def test_worklist_round_trip_is_strict() -> None:
    original = KernelWorklist(
        requirement=SelectionRequirement.TOP_TIME,
        top_n=1,
        kernels=(
            KernelObservation(
                identity=KernelIdentity(
                    name="kernel",
                    target="gfx950",
                    code_object="/tmp/kernel.hsaco",
                    code_object_sha256="0" * 64,
                    code_object_index=2,
                    entry_offset=0x120,
                ),
                total_time_ms=4.5,
                dispatch_count=7,
                sources=("magpie",),
            ),
        ),
    )

    rebuilt = KernelWorklist.from_dict(original.to_dict())

    assert rebuilt == original


def test_worklist_rejects_declared_count_mismatch() -> None:
    data = KernelWorklist(
        requirement=SelectionRequirement.TOP_TIME,
        top_n=1,
        kernels=(),
    ).to_dict()
    data["kernel_count"] = 1

    with pytest.raises(ValueError, match="kernel_count"):
        KernelWorklist.from_dict(data)


def test_selection_rejects_ambiguous_duplicate_metrics() -> None:
    identity = KernelIdentity(name="kernel", target="gfx950")
    with pytest.raises(ValueError, match="conflicting metrics"):
        select_kernels(
            [
                KernelObservation(
                    identity=identity,
                    total_time_ms=1,
                    sources=("one",),
                ),
                KernelObservation(
                    identity=identity,
                    total_time_ms=2,
                    sources=("two",),
                ),
            ],
            requirement=SelectionRequirement.TOP_TIME,
            top_n=1,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_observation_rejects_nonfinite_time(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        KernelObservation(
            identity=KernelIdentity(name="kernel", target="gfx950"),
            total_time_ms=value,
            sources=("test",),
        )


def test_dispatch_rows_require_count_and_string_name() -> None:
    with pytest.raises(ValueError, match="count"):
        observations_from_dispatch_rows(
            [{"name": "kernel"}],
            target="gfx950",
        )
    with pytest.raises(ValueError, match="string"):
        observations_from_dispatch_rows(
            [{"name": {"bad": "shape"}, "count": 1}],
            target="gfx950",
        )


def test_observation_requires_provenance_source() -> None:
    with pytest.raises(ValueError, match="sources"):
        KernelObservation(
            identity=KernelIdentity(name="kernel", target="gfx950"),
            total_time_ms=1,
        )
