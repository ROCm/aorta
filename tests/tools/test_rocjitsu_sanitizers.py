"""Tests for the built-in rocjitsu_sanitizers tool.

None of these need a GPU, a rocjitsu build, or the sanitizer artifacts: the
backends are exercised with monkeypatched subprocess results, kernel selection
is pure host code, and the tool-level runs use ``dry_run`` or mocked binaries.

The invocation model and support table are reconciled against
``rocm-systems/emulation/rocjitsu/docs/sanitizers.md`` (combined HSA-tools
hook: ``rj_waitcheck`` static tool + ``librocjitsu_dbi_hooks.so`` dynamic hook;
gfx950 = waitcheck Yes / ConSan Yes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import aorta.tools.rocjitsu_sanitizers as _pkg
from aorta.tools import Tool, get_tool, load_tools
from aorta.tools.rocjitsu_sanitizers import RocjitsuSanitizersTool, backends, runner
from aorta.tools.rocjitsu_sanitizers.select import select_kernels

_EXAMPLES = Path(_pkg.__file__).parent / "examples"
_GEMM_CSV = _EXAMPLES / "synthetic_gemm_shapes.csv"
_KERNELS_JSON = _EXAMPLES / "example_kernels.json"

# Doc example lines (verbatim shapes) the parsers must handle.
_WAITCHECK_OUT = (
    "rocjitsu-waitcheck: .text+0x40: missing s_wait_loadcnt <= 0 ... global_load_b32 ...\n"
    "rocjitsu-waitcheck:   consumer: v_mov_b32_e32 ...\n"
)
_CONSAN_OUT = (
    "[rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic ... kind=1 ... "
    "first_lds=[0,4) second_lds=[0,4) first_kind=2 second_kind=1 ...\n"
    "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=true "
    "analysis_complete=true static_complete=true dynamic_complete=true\n"
)


# --- discovery / registry ------------------------------------------------

def test_tool_discovered_via_entry_point() -> None:
    registry = load_tools()
    assert "rocjitsu_sanitizers" in registry
    assert registry["rocjitsu_sanitizers"] is RocjitsuSanitizersTool


def test_get_tool_returns_class_and_unknown_raises() -> None:
    assert get_tool("rocjitsu_sanitizers") is RocjitsuSanitizersTool
    with pytest.raises(Exception, match="unknown tool"):
        get_tool("does_not_exist")


def test_tool_conforms_to_protocol_and_name() -> None:
    assert isinstance(RocjitsuSanitizersTool(), Tool)
    assert RocjitsuSanitizersTool.name == "rocjitsu_sanitizers"


# --- backends: support policy matches the doc's target table -------------

def test_waitcheck_supported_on_gfx950_and_gfx1250() -> None:
    assert backends.support("waitcheck", "gfx950")["level"] == "full"
    assert backends.support("waitcheck", "gfx1250")["level"] == "full"


def test_consan_full_on_gfx950_gfx1100_gfx1250() -> None:
    sup = backends.support("consan", "gfx950")
    assert sup["level"] == "full"
    assert sup["runnable"] is True
    assert sup["requires"] == "hardware-or-simulator"
    assert backends.support("consan", "gfx1100")["level"] == "full"
    assert backends.support("consan", "gfx1250")["level"] == "full"


def test_consan_unsupported_on_waitcheck_only_target() -> None:
    sup = backends.support("consan", "gfx1200")
    assert sup["level"] == "unsupported"
    assert sup["runnable"] is False


def test_resolve_waitcheck_honours_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(backends.ENV_WAITCHECK_BIN, "/opt/rocjitsu/build/tools/rj_waitcheck")
    assert backends.resolve_waitcheck() == "/opt/rocjitsu/build/tools/rj_waitcheck"


# --- backends: output parsers --------------------------------------------

def test_parse_waitcheck_flags_missing_wait_with_context() -> None:
    result = backends.parse_waitcheck(
        "/tmp/k0.hsaco:gfx950[0]:.text+0x458: missing s_waitcnt lgkmcnt(0) before def of s45\n"
        "  producer .text+0x448: s_load_dwordx8 s[40:47], s[0:1], 0x50\n"
        "  consumer .text+0x458: s_load_dword s45, s[0:1], 0x8c\n"
    )
    assert result["counts"] == {"warning": 1}
    assert result["findings"][0]["context"]


def test_parse_consan_ignores_benign_capacity_lines() -> None:
    out = ("[rocjitsu-dbi-hooks] ConSan patch plan diagnostics=32768 capacity\n"
           "[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=true "
           "analysis_complete=true dynamic_complete=true\n")
    result = backends.parse_consan(out)
    assert result["counts"] == {}
    assert backends.verdict_for_counts(result["counts"]) == "pass"


def test_parse_consan_conflict_true_is_race() -> None:
    out = ("[rocjitsu-dbi-hooks] ConSan MOI auto replay reader=1 processed_access=2 "
           "diagnostics=1 conflict=true\n")
    result = backends.parse_consan(out)
    assert result["counts"].get("race") == 1
    assert backends.verdict_for_counts(result["counts"]) == "fail"


def test_consan_effective_complete_and_incomplete() -> None:
    complete = backends.consan_effective(
        {"analysis_complete": "true", "dynamic_complete": "true",
         "incomplete_code_objects": "0", "access": "2/2",
         "replay_unsupported_access": "0"})
    assert complete["complete"] is True
    incomplete = backends.consan_effective(
        {"analysis_complete": "true", "dynamic_complete": "false",
         "incomplete_code_objects": "1"})
    assert incomplete["complete"] is False
    assert backends.consan_effective({})["complete"] is None


def test_consan_env_recipe_and_bad_mode() -> None:
    env = backends.consan_env("/build/hook.so", mode="record-replay", log=True)
    assert env["HSA_TOOLS_LIB"] == "/build/hook.so"
    assert env["HSA_TOOLS_DISABLE_REGISTER"] == "1"
    assert env["RJ_CONSAN_LOG"] == "1"
    with pytest.raises(ValueError, match="RJ_CONSAN_MODE"):
        backends.consan_env("/build/hook.so", mode="bogus")


def test_consan_argv_simulator_wrapper() -> None:
    argv = backends.consan_argv(["./app", "-x"], simulator="rocjitsu", config="gfx1250.json")
    assert argv == ["rocjitsu", "--config", "gfx1250.json", "--", "./app", "-x"]


# --- select_kernels ------------------------------------------------------

def test_select_kernels_requires_a_source() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_kernels(magpie_report=None, gemm_csv=None, top_n=5)


def test_select_kernels_gemm_csv_carries_shape_and_solution() -> None:
    worklist = select_kernels(magpie_report=None, gemm_csv=_GEMM_CSV, top_n=5)
    assert worklist["schema"] == "rocjitsu_sanitizers.kernels/1"
    top = worklist["kernels"][0]
    assert top["source"] == "gemm_csv"
    assert top["shape"]["M"] and top["shape"]["K"]
    assert top["top_solution_idx"] == 100001


# --- runner: static waitcheck (mocked binary + exec) ---------------------

def _gemm_worklist() -> dict[str, Any]:
    return {"schema": "x", "top_n": 1, "sources": ["test"], "kernel_count": 1,
            "kernels": [{"source": "gemm_csv", "rank": 1, "name": "gemm_NT",
                         "top_solution_idx": 100001}]}


def test_run_waitcheck_reports_missing_wait_as_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / "sol_100001.hsaco").write_text("...")
    monkeypatch.setattr(backends, "resolve_waitcheck", lambda: "/fake/rj_waitcheck")
    monkeypatch.setattr(runner, "_exec",
                        lambda argv, timeout, env=None: (0, _WAITCHECK_OUT, ""))
    report = runner.run_waitcheck(
        _gemm_worklist(), "gfx950", tmp_path, dry_run=False, timeout=10)
    assert report["status"] == "ran"
    assert report["verdict"] == "warn"
    assert report["support"]["level"] == "full"


def test_run_waitcheck_skips_without_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "resolve_waitcheck", lambda: None)
    report = runner.run_waitcheck(
        _gemm_worklist(), "gfx950", None, dry_run=False, timeout=10)
    assert report["status"] == "skipped"
    assert "rj_waitcheck not found" in report["reason"]


# --- runner: dynamic ConSan gating ---------------------------------------

def test_run_consan_skips_on_unsupported_target() -> None:
    report = runner.run_consan(
        _gemm_worklist(), "gfx1200", ["python", "repro.py"], dry_run=False, timeout=10)
    assert report["verdict"] == "skipped"
    assert report["support"]["level"] == "unsupported"


def test_run_consan_skips_without_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "resolve_hook", lambda: "/fake/hook.so")
    report = runner.run_consan(
        _gemm_worklist(), "gfx950", None, dry_run=False, timeout=10)
    assert report["verdict"] == "skipped"
    assert "consan-command" in report["reason"]
    assert report["support"]["level"] == "full"


def test_run_consan_flags_race_with_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "resolve_hook", lambda: "/fake/hook.so")
    monkeypatch.setattr(runner, "_exec",
                        lambda argv, timeout, env=None: (0, _CONSAN_OUT, ""))
    report = runner.run_consan(
        _gemm_worklist(), "gfx950", ["python", "repro.py"], dry_run=False, timeout=10)
    assert report["status"] == "ran"
    assert report["verdict"] == "fail"


# --- runner + tool: orchestration ----------------------------------------

def test_run_sanitizers_dry_run_is_not_checked() -> None:
    report = runner.run_sanitizers(
        worklist=_gemm_worklist(), target="gfx950",
        checks=["waitcheck", "consan"], dry_run=True)
    assert report["overall_verdict"] == "not_checked"
    assert {c["check"] for c in report["checks"]} == {"waitcheck", "consan"}


def test_tool_invoke_from_worklist_dry_run(tmp_path: Path) -> None:
    result = RocjitsuSanitizersTool().invoke(
        inputs={"kernels": _KERNELS_JSON, "target": "gfx950",
                "checks": ["waitcheck", "consan"], "dry_run": True},
        output_dir=tmp_path,
    )
    assert result["overall_verdict"] == "not_checked"
    assert (tmp_path / "sanitizer_report.json").exists()
    assert result["report"]["kernel_count"] == 3
