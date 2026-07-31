from __future__ import annotations

from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import (
    CheckResult,
    ExecutionState,
    ExecutionSummary,
    KernelCheckResult,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SanitizerReport,
    SelectionRequirement,
    Verdict,
    build_report,
    read_report,
    run_sanitizers,
    write_report,
)


def _worklist(*, target: str = "gfx950") -> KernelWorklist:
    return KernelWorklist(
        requirement=SelectionRequirement.TOP_TIME,
        top_n=1,
        kernels=(
            KernelObservation(
                identity=KernelIdentity(name="kernel", target=target),
                total_time_ms=1,
                dispatch_count=1,
                sources=("test",),
            ),
        ),
    )


def test_report_round_trip_and_fail_closed_precedence(tmp_path: Path) -> None:
    report = build_report(
        target="gfx950",
        worklist=_worklist(),
        checks=(
            CheckResult(
                sanitizer="waitcheck",
                state=ExecutionState.RAN,
                verdict=Verdict.PASS,
            ),
            CheckResult(
                sanitizer="consan",
                state=ExecutionState.NOT_CHECKED,
                verdict=Verdict.NOT_CHECKED,
                reason="worklist_scope_unsupported",
            ),
        ),
    )
    path = tmp_path / "sanitizer_report.json"

    write_report(report, path)
    rebuilt = read_report(path)

    assert rebuilt == report
    assert rebuilt.overall_verdict is Verdict.NOT_CHECKED
    assert rebuilt.execution_status is ExecutionSummary.PARTIAL
    assert path.stat().st_size > 0


def test_report_rejects_tampered_overall_verdict() -> None:
    report = build_report(
        target="gfx950",
        worklist=_worklist(),
        checks=(
            CheckResult(
                sanitizer="waitcheck",
                state=ExecutionState.RAN,
                verdict=Verdict.PASS,
            ),
        ),
    )
    data = report.to_dict()
    data["overall_verdict"] = "fail"

    with pytest.raises(ValueError, match="contradicts"):
        SanitizerReport.from_dict(data)


def test_pipeline_writes_not_checked_for_scoped_consan(tmp_path: Path) -> None:
    report = run_sanitizers(
        _worklist(),
        target="gfx950",
        sanitizers=("consan",),
        output_dir=tmp_path,
    )

    assert report.overall_verdict is Verdict.NOT_CHECKED
    assert report.checks[0].reason is not None
    assert (tmp_path / "sanitizer_report.json").is_file()


def test_report_rejects_target_mismatch() -> None:
    with pytest.raises(ValueError, match="target"):
        build_report(
            target="gfx942",
            worklist=_worklist(target="gfx950"),
            checks=(),
        )


def test_deserialized_report_rejects_target_mismatch() -> None:
    report = build_report(
        target="gfx950",
        worklist=_worklist(),
        checks=(),
    )
    data = report.to_dict()
    data["target"] = "gfx942"

    with pytest.raises(ValueError, match="target"):
        SanitizerReport.from_dict(data)


def test_warn_plus_not_checked_is_explicitly_partial() -> None:
    report = build_report(
        target="gfx950",
        worklist=_worklist(),
        checks=(
            CheckResult(
                sanitizer="waitcheck",
                state=ExecutionState.RAN,
                verdict=Verdict.WARN,
            ),
            CheckResult(
                sanitizer="consan",
                state=ExecutionState.NOT_CHECKED,
                verdict=Verdict.NOT_CHECKED,
                reason="worklist_scope_unsupported",
            ),
        ),
    )

    assert report.overall_verdict is Verdict.WARN
    assert report.execution_status is ExecutionSummary.PARTIAL


def test_nested_kernel_result_rejects_state_verdict_contradiction() -> None:
    with pytest.raises(ValueError, match="not_checked"):
        KernelCheckResult(
            identity=KernelIdentity(name="kernel", target="gfx950"),
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.PASS,
            reason="missing",
        )
