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


_RACY_CONFLICT_MESSAGE = (
    "[rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic reader=1 index=0 kind=1 "
    "first_owner=0 second_owner=1 first_lds=[0,4) second_lds=[0,4) "
    "first_kind=2 second_kind=1"
)


def _write_all_matching(root: Path) -> None:
    _write_case(root, "waitcheck", _check("waitcheck", Verdict.WARN, message="missing s_waitcnt"))
    _write_case(root, "consan-clean", _check("consan", Verdict.PASS))
    _write_case(root, "consan-racy", _check("consan", Verdict.FAIL, message=_RACY_CONFLICT_MESSAGE))


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
    # FAIL verdict is correct, but the expected finding shape ("auto replay
    # diagnostic") is absent -> the gate must not accept it.
    _write_case(tmp_path, "consan-racy", _check("consan", Verdict.FAIL, message="something else"))
    assert comparator.main(["prog", str(tmp_path)]) == 1


def test_comparator_rejects_missing_report(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_case(tmp_path, "waitcheck", _check("waitcheck", Verdict.WARN, message="missing s_waitcnt"))
    _write_case(tmp_path, "consan-clean", _check("consan", Verdict.PASS))
    # consan-racy intentionally absent
    assert comparator.main(["prog", str(tmp_path)]) == 1


# --------------------------------------------------------------- vacuity sweep
# ROCm/aorta#450: two informational recipes paired a load-only driver with
# consan_policy: strict and so ended `error` with zero findings on every run.
# Nothing caught it, because the comparison above only covers the gated cases.


def _errored(sanitizer: str, reason: str) -> CheckResult:
    return CheckResult(
        sanitizer=sanitizer,
        state=ExecutionState.ERROR,
        verdict=Verdict.ERROR,
        reason=reason,
        returncode=86,
    )


def test_sweep_rejects_undeclared_vacuous_error(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # consan_gemm has no expected_error entry, so exit 86 with no findings is a
    # broken run rather than a result.
    _write_case(tmp_path, "informational/consan-gemm", _errored("consan", "combined_hook_exit_86"))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "consan_gemm" in out
    assert "combined_hook_exit_86" in out


def test_sweep_rejects_vacuous_error_in_full_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # Every gated case matches its baseline, so only the sweep can fail this.
    _write_all_matching(tmp_path)
    _write_case(tmp_path, "informational/consan-gemm", _errored("consan", "combined_hook_exit_86"))
    assert comparator.main(["prog", str(tmp_path)]) == 1


def test_sweep_accepts_declared_expected_error(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # consan_tiny declares expected_error reason combined_hook_exit_86 in the
    # committed baselines: an intended negative control, so it must not fire.
    _write_case(tmp_path, "informational/consan-tiny", _errored("consan", "combined_hook_exit_86"))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


def test_sweep_rejects_declared_case_erroring_for_another_reason(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The declaration names a reason, so consan_tiny is allowed to fail closed for
    # exit 86 and nothing else. A timeout there is a new failure.
    _write_case(tmp_path, "informational/consan-tiny", _errored("consan", "combined_hook_timeout"))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1
    assert "combined_hook_exit_86" in capsys.readouterr().out


def test_sweep_rejects_vacuous_not_checked(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    not_checked = CheckResult(
        sanitizer="consan",
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason="no backend",
    )
    _write_case(tmp_path, "informational/consan-gemm", not_checked)
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1


def test_sweep_accepts_clean_pass_with_zero_findings(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # daily-consan-lds-dispatch's real shape: a completed run that found nothing.
    # Zero findings alone must never be the trigger.
    _write_case(tmp_path, "informational/consan-lds-dispatch", _check("consan", Verdict.PASS))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


def test_sweep_accepts_error_that_still_produced_findings(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # An errored run that still reported something is not vacuous; the gated
    # baseline comparison is what judges whether the verdict is right.
    errored_with_finding = CheckResult(
        sanitizer="consan",
        state=ExecutionState.ERROR,
        verdict=Verdict.ERROR,
        reason="consan_coverage_incomplete: barrier patched/supported mismatch: 1/2",
        returncode=0,
        findings=(
            Finding(
                sanitizer="consan",
                severity=FindingSeverity.ERROR,
                code="c",
                message="partial coverage",
            ),
        ),
    )
    _write_case(tmp_path, "informational/consan-gemm", errored_with_finding)
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


def test_vacuous_only_ignores_absent_gated_cases(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The survey job's result tree contains none of the gated cases, and the job
    # may legitimately be killed early by the runner cap in #370. Absence is not
    # this check's business.
    _write_case(tmp_path, "informational/consan-lds-dispatch", _check("consan", Verdict.PASS))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


def test_vacuous_only_tolerates_empty_tree(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


def test_comparator_rejects_bad_usage(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    assert comparator.main(["prog"]) == 2
    assert comparator.main(["prog", "--vacuous-only"]) == 2
    assert comparator.main(["prog", str(tmp_path), "extra"]) == 2
