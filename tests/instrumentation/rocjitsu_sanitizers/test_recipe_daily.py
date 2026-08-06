"""Tests for sanitizer recipe loader, resolvers, and parsers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.backends import support
from aorta.instrumentation.rocjitsu_sanitizers.models import SelectionRequirement, Verdict
from aorta.instrumentation.rocjitsu_sanitizers.recipe import (
    execute_sanitizer_run,
    load_sanitizer_recipe,
)
from aorta.instrumentation.rocjitsu_sanitizers.report import read_report
from aorta.instrumentation.rocjitsu_sanitizers.selection import (
    observations_from_consan_repro,
    observations_from_gemm_csv,
    observations_from_rocprof_trace,
    select_kernels,
)
from aorta.triage.recipe import RecipeSchemaError

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
    # Materialize a code object for every solution index the fixture references so
    # whichever top-N shapes are selected resolve to a real (scan) object.
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.lstrip().startswith("#")))
    for idx in {row["top_solution_idx"] for row in rows}:
        (isa_dir / f"sol_{idx}.hsaco").write_bytes(f"fake-{idx}".encode())
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


def test_execute_recipe_with_fewer_observations_than_top_n(tmp_path: Path) -> None:
    # A trace whose only dispatch is a dropped runtime copy yields zero
    # observations. top_n must not be clamped down to the observation count
    # (which would request top_n=0 and raise); the empty worklist is valid.
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "KERNEL_DISPATCH,__amd_rocclr_copyBuffer,0,500\n",
        encoding="utf-8",
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "schema_version: 1\n"
        "mode: sanitizer\n"
        "ticket: TEST-EMPTY\n"
        "sanitizer_plan:\n"
        "  target: gfx950\n"
        "  source:\n"
        "    kind: rocprof_trace\n"
        "    path: trace.csv\n"
        "  scope:\n"
        "    kind: kernel\n"
        "  selection:\n"
        "    requirement: top_dispatch_count\n"
        "    top_n: 8\n"
        "  sanitizers:\n"
        "    - waitcheck\n"
        "  policy:\n"
        "    consan_policy: strict\n"
        "    on_missing_backend: fail\n"
        "  output:\n"
        "    report: custom_report.json\n",
        encoding="utf-8",
    )

    report_path = execute_sanitizer_run(recipe, output_dir=tmp_path / "out", dry_run=True)

    assert report_path == tmp_path / "out" / "custom_report.json"
    assert report_path.is_file()


def _write_gemm_recipe(
    tmp_path: Path,
    *,
    scope: str,
    policy_lines: str,
    source_extra: str = "",
    include_isa_dir: bool = True,
) -> Path:
    isa_line = "    isa_dir: isa\n" if include_isa_dir else ""
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "schema_version: 1\n"
        "mode: sanitizer\n"
        "ticket: TEST\n"
        "sanitizer_plan:\n"
        "  target: gfx950\n"
        "  source:\n"
        "    kind: gemm_csv\n"
        "    path: shapes.csv\n"
        f"{isa_line}"
        f"{source_extra}"
        "  scope:\n"
        f"    kind: {scope}\n"
        "  selection:\n"
        "    requirement: top_dispatch_count\n"
        "    top_n: 3\n"
        "  sanitizers:\n"
        "    - waitcheck\n"
        "  policy:\n"
        f"{policy_lines}"
        "  output:\n"
        "    report: sanitizer_report.json\n",
        encoding="utf-8",
    )
    return recipe


_GOOD_POLICY = "    consan_policy: strict\n    on_missing_backend: fail\n"


def test_recipe_rejects_unknown_scope_kind(tmp_path: Path) -> None:
    recipe = _write_gemm_recipe(tmp_path, scope="module", policy_lines=_GOOD_POLICY)
    with pytest.raises(RecipeSchemaError, match="scope.kind"):
        load_sanitizer_recipe(recipe)


def test_recipe_rejects_unknown_consan_policy(tmp_path: Path) -> None:
    recipe = _write_gemm_recipe(
        tmp_path,
        scope="kernel",
        policy_lines="    consan_policy: strictt\n    on_missing_backend: fail\n",
    )
    with pytest.raises(RecipeSchemaError, match="consan_policy"):
        load_sanitizer_recipe(recipe)


def test_recipe_rejects_unknown_on_missing_backend(tmp_path: Path) -> None:
    recipe = _write_gemm_recipe(
        tmp_path,
        scope="kernel",
        policy_lines="    consan_policy: strict\n    on_missing_backend: skip\n",
    )
    with pytest.raises(RecipeSchemaError, match="on_missing_backend"):
        load_sanitizer_recipe(recipe)


def test_recipe_rejects_gemm_csv_without_isa_dir(tmp_path: Path) -> None:
    recipe = _write_gemm_recipe(
        tmp_path, scope="kernel", policy_lines=_GOOD_POLICY, include_isa_dir=False
    )
    with pytest.raises(RecipeSchemaError, match="isa_dir"):
        load_sanitizer_recipe(recipe)


def test_recipe_rejects_non_boolean_consan_log(tmp_path: Path) -> None:
    recipe = _write_gemm_recipe(
        tmp_path,
        scope="kernel",
        policy_lines=_GOOD_POLICY,
        source_extra='    consan_log: "false"\n',
    )
    with pytest.raises(RecipeSchemaError, match="consan_log"):
        load_sanitizer_recipe(recipe)


def _write_kernel_consan_recipe(
    tmp_path: Path,
    *,
    consan_command: str = "loaders/consan_app",
    ticket: str = "TEST-CONSAN-CMD",
) -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "schema_version: 1\n"
        "mode: sanitizer\n"
        f"ticket: {ticket}\n"
        "sanitizer_plan:\n"
        "  target: gfx950\n"
        "  source:\n"
        "    kind: kernel\n"
        "    kernel:\n"
        "      name: gemm_NT_M128_N128_K128\n"
        "      code_object: fixtures/isa/sol_0.hsaco\n"
        "      code_object_index: 0\n"
        f"    consan_command: {consan_command}\n"
        "  scope:\n"
        "    kind: kernel\n"
        "  selection:\n"
        "    requirement: top_dispatch_count\n"
        "    top_n: 1\n"
        "  sanitizers:\n"
        "    - consan\n"
        "  policy:\n"
        "    consan_policy: strict\n"
        "    on_missing_backend: fail\n"
        "  output:\n"
        "    report: sanitizer_report.json\n",
        encoding="utf-8",
    )
    return recipe


def test_loader_resolves_consan_command_for_kernel_source(tmp_path: Path) -> None:
    recipe = _write_kernel_consan_recipe(tmp_path)
    loaded = load_sanitizer_recipe(recipe)
    assert loaded.source_kind == "kernel"
    assert loaded.consan_command is not None
    assert loaded.consan_command.is_absolute()
    assert loaded.consan_command == (tmp_path / "loaders" / "consan_app").resolve()


def test_loader_resolves_consan_command_for_kernel_list_source(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "schema_version: 1\n"
        "mode: sanitizer\n"
        "ticket: TEST-CONSAN-LIST\n"
        "sanitizer_plan:\n"
        "  target: gfx950\n"
        "  source:\n"
        "    kind: kernel_list\n"
        "    kernels:\n"
        "      - gemm_a\n"
        "    consan_command: bin/loader\n"
        "  scope:\n"
        "    kind: kernel\n"
        "  selection:\n"
        "    requirement: top_dispatch_count\n"
        "    top_n: 1\n"
        "  sanitizers:\n"
        "    - consan\n"
        "  policy:\n"
        "    consan_policy: strict\n"
        "    on_missing_backend: fail\n"
        "  output:\n"
        "    report: sanitizer_report.json\n",
        encoding="utf-8",
    )
    loaded = load_sanitizer_recipe(recipe)
    assert loaded.source_kind == "kernel_list"
    assert loaded.consan_command == (tmp_path / "bin" / "loader").resolve()


def test_loader_resolves_consan_command_for_gemm_csv_source(tmp_path: Path) -> None:
    recipe = _write_gemm_recipe(
        tmp_path,
        scope="kernel",
        policy_lines=_GOOD_POLICY,
        source_extra="    consan_command: loaders/consan_app\n",
    )
    loaded = load_sanitizer_recipe(recipe)
    assert loaded.source_kind == "gemm_csv"
    assert loaded.consan_command == (tmp_path / "loaders" / "consan_app").resolve()


def test_loader_accepts_custom_consan_repro_variant(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "schema_version: 1\n"
        "mode: sanitizer\n"
        "ticket: TEST-CONSAN-VARIANT\n"
        "sanitizer_plan:\n"
        "  target: gfx950\n"
        "  source:\n"
        "    kind: consan_repro\n"
        "    variant: custom_race\n"
        "    hip: repro/custom.hip\n"
        "    command: bin/custom_repro\n"
        "  scope:\n"
        "    kind: kernel\n"
        "  selection:\n"
        "    requirement: top_dispatch_count\n"
        "    top_n: 1\n"
        "  sanitizers:\n"
        "    - consan\n"
        "  policy:\n"
        "    consan_policy: strict\n"
        "    on_missing_backend: fail\n"
        "  output:\n"
        "    report: sanitizer_report.json\n",
        encoding="utf-8",
    )
    loaded = load_sanitizer_recipe(recipe)
    assert loaded.repro_variant == "custom_race"
    assert loaded.consan_command == (tmp_path / "bin" / "custom_repro").resolve()
    observations = observations_from_consan_repro("custom_race", target="gfx950")
    assert len(observations) == 1
    assert observations[0].identity.name == "consan_custom_race"


def test_observations_from_consan_repro_keeps_builtin_labels() -> None:
    clean = observations_from_consan_repro("clean", target="gfx950")
    racy = observations_from_consan_repro("racy", target="gfx950")
    assert clean[0].identity.name == "consan_lds_race"
    assert racy[0].identity.name == "consan_lds_race_2wave"


def test_observations_from_consan_repro_rejects_empty_variant() -> None:
    with pytest.raises(ValueError):
        observations_from_consan_repro("  ", target="gfx950")


def test_loader_rejects_empty_consan_command(tmp_path: Path) -> None:
    recipe = _write_kernel_consan_recipe(tmp_path, consan_command='""')
    with pytest.raises(RecipeSchemaError, match="consan_command"):
        load_sanitizer_recipe(recipe)


def test_consan_command_run_is_fail_closed_when_command_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A kernel source + top_n: 1 with a consan_command that does not exist on
    # disk must fail closed: the ConSan check is not_checked, never pass. Without
    # ROCJITSU_BUILD the hook also fails to resolve; either is an acceptable
    # fail-closed outcome, so we assert the verdict is never pass.
    monkeypatch.delenv("ROCJITSU_BUILD", raising=False)
    recipe = _write_kernel_consan_recipe(tmp_path, consan_command="loaders/does_not_exist")
    report_path = execute_sanitizer_run(recipe, output_dir=tmp_path / "out")
    report = read_report(report_path)
    assert report.overall_verdict is not Verdict.PASS
    consan_checks = [check for check in report.checks if check.sanitizer == "consan"]
    assert consan_checks
    for check in consan_checks:
        assert check.verdict is Verdict.NOT_CHECKED


def test_verdict_baselines_fixture_present() -> None:
    baselines = json.loads((_FIXTURES / "expected" / "verdict_baselines.json").read_text())
    assert baselines["waitcheck_gemm"]["overall_verdict"] == "warn"
    assert baselines["consan_clean"]["overall_verdict"] == "pass"
    assert baselines["consan_racy"]["overall_verdict"] == "fail"
