"""Tests for the Phase-1 verdict-baseline comparator (fail-closed gate)."""

from __future__ import annotations

import importlib.util
import json
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


def _write_case(root: Path, case_dir: str, *checks: CheckResult) -> None:
    report = build_report(target="gfx950", worklist=_worklist(), checks=checks)
    write_report(report, root / case_dir / "sanitizer_report.json")


def _use_baselines(monkeypatch, tmp_path: Path, data: dict) -> None:
    """Point the comparator at a throwaway baselines file."""
    path = tmp_path / "baselines.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(comparator, "_BASELINES", path)


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
    # The survey job's result tree contains none of the gated cases. Nothing is
    # required here, so nothing is missing.
    _write_case(tmp_path, "informational/consan-lds-dispatch", _check("consan", Verdict.PASS))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


def test_vacuous_only_tolerates_empty_tree_when_nothing_is_required(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


# ------------------------------------------------------- exact expected_error
# An expected_error narrows what one row is allowed to do. Anything about it
# that cannot be checked widens it instead, so a malformed declaration fails.


def _declaration(**overrides) -> dict:
    declared = {
        "execution_status": "error",
        "overall_verdict": "error",
        "reasons": ["combined_hook_exit_86"],
        "why": "intended negative control",
    }
    declared.update(overrides)
    return declared


def test_malformed_expected_error_fails_even_when_the_case_did_not_run(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # `reason` (singular) is the old spelling: silently ignored by a
    # minimum-fields check, which is exactly how a bypass gets in.
    _use_baselines(monkeypatch, tmp_path, {"consan_tiny": {"expected_error": {"reason": "x"}}})
    root = tmp_path / "results"
    root.mkdir()
    assert comparator.main(["prog", "--vacuous-only", str(root)]) == 2
    err = capsys.readouterr().err
    assert "unknown field(s) reason" in err
    assert "is missing execution_status, overall_verdict, reasons, why" in err


def test_expected_error_must_be_an_object(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _use_baselines(monkeypatch, tmp_path, {"consan_tiny": {"expected_error": True}})
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 2


def test_expected_error_rejects_unusable_reason_lists(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    for reasons in ("combined_hook_exit_86", [], [""], [86], ["a", "a"]):
        _use_baselines(
            monkeypatch,
            tmp_path,
            {"consan_tiny": {"expected_error": _declaration(reasons=reasons)}},
        )
        assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 2, reasons


def test_expected_error_rejects_a_status_it_could_never_match(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The sweep only looks at vacuous statuses, so declaring `complete` is dead
    # configuration that reads as protection.
    _use_baselines(
        monkeypatch,
        tmp_path,
        {"consan_tiny": {"expected_error": _declaration(execution_status="complete")}},
    )
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 2


def test_expected_error_rejects_a_verdict_it_could_never_match(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # Same argument as the status above: a vacuous report cannot carry `pass`,
    # so declaring it is a typo that reads as protection.
    _use_baselines(
        monkeypatch,
        tmp_path,
        {"consan_tiny": {"expected_error": _declaration(overall_verdict="pass")}},
    )
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 2


def test_expected_error_rejects_an_empty_why(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _use_baselines(
        monkeypatch, tmp_path, {"consan_tiny": {"expected_error": _declaration(why="   ")}}
    )
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 2


def test_sweep_rejects_a_reason_that_merely_contains_the_declared_token(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # combined_hook_exit_860 is a different exit code, not the signed-off one.
    _write_case(tmp_path, "informational/consan-tiny", _errored("consan", "combined_hook_exit_860"))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1
    assert "not the outcome that was signed off" in capsys.readouterr().out


def test_sweep_rejects_an_extra_undeclared_reason(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The declared failure happened, and so did something else nobody signed off.
    _write_case(
        tmp_path,
        "informational/consan-tiny",
        _errored("consan", "combined_hook_exit_86"),
        _errored("waitcheck_preflight", "consan_output_parse_error: truncated"),
    )
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1


def test_sweep_accepts_the_declared_reason_repeated_across_checks(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The real shape: the combined hook fails both checks with one reason, which
    # is one distinct reason and must still match.
    _write_case(
        tmp_path,
        "informational/consan-tiny",
        _errored("consan", "combined_hook_exit_86"),
        _errored("waitcheck_preflight", "combined_hook_exit_86"),
    )
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 0


def test_sweep_rejects_a_declared_case_that_was_never_checked(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # Same reason token, but the sanitizer never ran. `error` was reviewed;
    # `not_checked` was not.
    never_ran = CheckResult(
        sanitizer="consan",
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason="combined_hook_exit_86",
    )
    _write_case(tmp_path, "informational/consan-tiny", never_ran)
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1


def test_expected_error_is_rejected_on_a_gated_key(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # `waitcheck_gemm` is both a gated baseline key and what the survey directory
    # informational/waitcheck-gemm resolves to, so one exemption there would cover
    # two different recipes.
    _use_baselines(
        monkeypatch, tmp_path, {"waitcheck_gemm": {"expected_error": _declaration()}}
    )
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 2
    assert "not allowed on a gated case" in capsys.readouterr().err


# The flip side of the rejection above: because that declaration is impossible,
# the sweep must not advise it. Advice an operator cannot follow is the same
# defect shape as an option that is accepted and ignored.

_IMPOSSIBLE_ADVICE = "declare it as an 'expected_error' entry"


def test_vacuous_advice_is_followable_for_a_colliding_survey_key(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # informational/waitcheck-gemm resolves to the gated key waitcheck_gemm, so
    # test_expected_error_is_rejected_on_a_gated_key above is what an operator
    # who followed the old advice would have got -- exit 2, and for every row,
    # since the baselines then fail to load at all.
    _write_case(
        tmp_path, "informational/waitcheck-gemm", _errored("waitcheck", "worklist_not_fully_checked")
    )
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert _IMPOSSIBLE_ADVICE not in out
    assert "--known-vacuous informational/waitcheck-gemm" in out
    # And the advice it gives instead actually works.
    argv = [
        "prog",
        "--vacuous-only",
        "--known-vacuous",
        "informational/waitcheck-gemm",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 0


def test_vacuous_advice_for_a_gated_report_does_not_point_at_the_sweep(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The gated report itself, rather than the colliding survey directory.
    # --known-vacuous cannot rescue this one either: _compare_case judges it
    # against its own baseline and still fails it, so sending the operator to the
    # sweep would be a second piece of advice that does not work.
    _write_case(tmp_path, "waitcheck", _errored("waitcheck", "worklist_not_fully_checked"))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert _IMPOSSIBLE_ADVICE not in out
    assert "--known-vacuous" not in out
    assert "gated case with a baseline of its own" in out


def test_vacuous_advice_still_points_at_expected_error_for_an_ordinary_row(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The remedy above is scoped to colliding keys; every other row keeps the
    # declaration advice, which is correct and reachable for it.
    _write_case(tmp_path, "informational/consan-gemm", _errored("consan", "growth_limit"))
    assert comparator.main(["prog", "--vacuous-only", str(tmp_path)]) == 1
    assert _IMPOSSIBLE_ADVICE in capsys.readouterr().out


def test_committed_baselines_declarations_are_well_formed() -> None:
    # The check above only bites if the shipped file satisfies it.
    comparator._load_baselines(_REPO_ROOT / comparator._BASELINES)


# ------------------------------------------------------------ required reports
# ROCm/aorta#450 one level up: all six survey invocations run under `|| true`,
# so a recipe that dies before writing a report leaves nothing to judge.


def test_sweep_rejects_a_required_case_with_no_report(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_case(tmp_path, "informational/consan-lds-dispatch", _check("consan", Verdict.PASS))
    argv = [
        "prog",
        "--vacuous-only",
        "--require",
        "informational/consan-lds-dispatch",
        "--require",
        "informational/consan-gemm",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 1
    out = capsys.readouterr().out
    assert "informational/consan-gemm: no report" in out
    assert "informational/consan-lds-dispatch: no report" not in out


def test_sweep_rejects_an_entirely_empty_survey(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    argv = ["prog", "--vacuous-only", "--require", "informational/consan-gemm", str(tmp_path)]
    assert comparator.main(argv) == 1
    assert "no report" in capsys.readouterr().out


def test_sweep_accepts_required_cases_that_all_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_case(tmp_path, "informational/consan-lds-dispatch", _check("consan", Verdict.PASS))
    _write_case(tmp_path, "informational/consan-tiny", _errored("consan", "combined_hook_exit_86"))
    argv = [
        "prog",
        "--vacuous-only",
        "--require",
        "informational/consan-lds-dispatch",
        "--require",
        "informational/consan-tiny",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 0


def test_require_also_applies_to_the_gate_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_all_matching(tmp_path)
    assert comparator.main(["prog", str(tmp_path)]) == 0
    assert comparator.main(["prog", "--require", "informational/consan-gemm", str(tmp_path)]) == 1


# --------------------------------------------------------- known-vacuous rows
# The opposite of an expected_error: "this is wrong and being worked", not
# "this is fine". So it must not outlive the break it describes.


def test_known_vacuous_reports_without_failing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_case(tmp_path, "informational/consan-gemm", _errored("consan", "growth_limit"))
    argv = [
        "prog",
        "--vacuous-only",
        "--known-vacuous",
        "informational/consan-gemm",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 0
    out = capsys.readouterr().out
    assert "KNOWN VACUOUS, not failing" in out
    assert "1 known-vacuous" in out
    # It must not suggest declaring an expected_error, which would assert the
    # outcome is acceptable rather than broken-and-tracked.
    assert "expected_error" not in out


def test_known_vacuous_still_fails_the_other_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The whole point: suppressing one row must not suppress the next regression.
    _write_case(tmp_path, "informational/consan-gemm", _errored("consan", "growth_limit"))
    _write_case(tmp_path, "informational/consan-lds-dispatch", _errored("consan", "something_new"))
    argv = [
        "prog",
        "--vacuous-only",
        "--known-vacuous",
        "informational/consan-gemm",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 1


def test_known_vacuous_fails_once_the_row_produces_signal(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_case(tmp_path, "informational/consan-gemm", _check("consan", Verdict.PASS))
    argv = [
        "prog",
        "--vacuous-only",
        "--known-vacuous",
        "informational/consan-gemm",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 1
    assert "Remove informational/consan-gemm" in capsys.readouterr().out


def test_known_vacuous_fails_when_the_row_has_no_report(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    _write_case(tmp_path, "informational/consan-tiny", _errored("consan", "combined_hook_exit_86"))
    argv = [
        "prog",
        "--vacuous-only",
        "--known-vacuous",
        "informational/consan-gemm",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 1
    assert "no report was found for it" in capsys.readouterr().out


def test_known_vacuous_conflicts_with_a_declaration(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # consan_tiny is declared in the committed baselines, so calling it
    # known-vacuous asserts the opposite of what the file says.
    argv = [
        "prog",
        "--vacuous-only",
        "--known-vacuous",
        "informational/consan-tiny",
        str(tmp_path),
    ]
    assert comparator.main(argv) == 2
    assert "Pick one" in capsys.readouterr().err


def test_known_vacuous_also_applies_to_the_gate_mode(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # --vacuous-only picks which halves run; it does not change what the sweep
    # means. So a suppression the caller declared has to be honoured with or
    # without it -- the list used to be parsed, conflict-checked, and then
    # dropped here, which failed the row as an ordinary vacuous report.
    _write_all_matching(tmp_path)
    _write_case(tmp_path, "informational/consan-gemm", _errored("consan", "growth_limit"))
    assert comparator.main(["prog", str(tmp_path)]) == 1
    assert "expected_error" in capsys.readouterr().out
    argv = ["prog", "--known-vacuous", "informational/consan-gemm", str(tmp_path)]
    assert comparator.main(argv) == 0
    out = capsys.readouterr().out
    assert "KNOWN VACUOUS, not failing" in out
    assert "expected_error" not in out


def test_known_vacuous_conflict_is_caught_in_the_gate_mode_too(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    # The conflict check sits ahead of the mode branch, so forwarding the list
    # must not open a path on which a contradictory row is accepted.
    _write_all_matching(tmp_path)
    argv = ["prog", "--known-vacuous", "informational/consan-tiny", str(tmp_path)]
    assert comparator.main(argv) == 2
    assert "Pick one" in capsys.readouterr().err


def test_comparator_rejects_bad_usage(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)
    assert comparator.main(["prog"]) == 2
    assert comparator.main(["prog", "--vacuous-only"]) == 2
    assert comparator.main(["prog", str(tmp_path), "extra"]) == 2
    assert comparator.main(["prog", "--vacuous-only", "--require", str(tmp_path)]) == 2
    assert comparator.main(["prog", "--require"]) == 2
    assert comparator.main(["prog", "--known-vacuous"]) == 2
    assert comparator.main(["prog", "--nope", str(tmp_path)]) == 2
