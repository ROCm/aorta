"""Tests for the Phase-1 verdict-baseline comparator (fail-closed gate)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from aorta.instrumentation.rocjitsu_sanitizers import (
    CheckResult,
    ExecutionState,
    Finding,
    FindingSeverity,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    Verdict,
    build_report,
    write_report,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_comparator():
    path = _REPO_ROOT / "scripts" / "sanitizers" / "compare_verdict_baselines.py"
    spec = importlib.util.spec_from_file_location("compare_verdict_baselines", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparator = _load_comparator()


def _worklist() -> KernelWorklist:
    return KernelWorklist(
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=1,
        kernels=(
            KernelObservation(
                identity=KernelIdentity(name="gemm", target="gfx950"),
                total_time_ms=1,
                dispatch_count=1,
                sources=("test",),
            ),
        ),
    )


def _check(sanitizer: str, verdict: Verdict, *, message: str | None = None) -> CheckResult:
    findings: tuple[Finding, ...] = ()
    if message is not None:
        severity = FindingSeverity.RACE if sanitizer == "consan" else FindingSeverity.WARNING
        findings = (Finding(sanitizer=sanitizer, severity=severity, code="c", message=message),)
    return CheckResult(
        sanitizer=sanitizer,
        state=ExecutionState.RAN,
        verdict=verdict,
        findings=findings,
    )


def _write_case(root: Path, case_dir: str, check: CheckResult) -> None:
    report = build_report(target="gfx950", worklist=_worklist(), checks=(check,))
    write_report(report, root / case_dir / "sanitizer_report.json")


def _write_all_matching(root: Path) -> None:
    _write_case(root, "waitcheck", _check("waitcheck", Verdict.WARN, message="missing s_waitcnt"))
    _write_case(root, "consan-clean", _check("consan", Verdict.PASS))
    _write_case(root, "consan-racy", _check("consan", Verdict.FAIL, message="conflict=true"))


def test_comparator_passes_when_all_cases_match(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_all_matching(tmp_path)
    assert comparator.main(["prog", str(tmp_path)]) == 0


def test_comparator_rejects_schemaless_report(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_all_matching(tmp_path)
    (tmp_path / "consan-clean" / "sanitizer_report.json").write_text("{}", encoding="utf-8")
    assert comparator.main(["prog", str(tmp_path)]) == 1


def test_comparator_rejects_incomplete_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_all_matching(tmp_path)
    not_checked = CheckResult(
        sanitizer="consan",
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason="no backend",
    )
    _write_case(tmp_path, "consan-clean", not_checked)
    assert comparator.main(["prog", str(tmp_path)]) == 1


def test_comparator_rejects_missing_finding_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_all_matching(tmp_path)
    # FAIL verdict is correct, but the expected finding shape ("conflict=true")
    # is absent -> the gate must not accept it.
    _write_case(tmp_path, "consan-racy", _check("consan", Verdict.FAIL, message="something else"))
    assert comparator.main(["prog", str(tmp_path)]) == 1


def test_comparator_rejects_missing_report(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_case(tmp_path, "waitcheck", _check("waitcheck", Verdict.WARN, message="missing s_waitcnt"))
    _write_case(tmp_path, "consan-clean", _check("consan", Verdict.PASS))
    # consan-racy intentionally absent
    assert comparator.main(["prog", str(tmp_path)]) == 1
