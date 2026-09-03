#!/usr/bin/env python3
"""Compare sanitizer nightly reports against committed verdict baselines.

The comparator is the Phase-1 gate, so it must fail closed rather than trust a
single top-level field. Each report is *strictly* reloaded through
``read_report`` (schema + internal-consistency validation, e.g. that
``overall_verdict``/``execution_status`` actually agree with the checks), and
then compared against the committed baseline's ``overall_verdict``,
``execution_status`` (when declared), per-check verdicts, and finding shape.

On top of that per-case comparison the comparator sweeps *every* report under
the results root for vacuous outcomes -- a run that terminated ``error`` or
``not_checked`` and produced no findings at all. Such a run is a broken
configuration, not a result: the sanitizer never got far enough to have an
opinion, so any coverage claim derived from it is empty rather than positive.

That sweep exists because of ROCm/aorta#450. ``daily-consan-tiny`` and
``daily-consan-gemm`` paired a load-only driver with ``consan_policy: strict``,
which sets ``RJ_CONSAN_MOI_REQUIRE_RECORDS`` and so demands dynamic records a
non-dispatching driver can never produce. Both failed closed with
``combined_hook_exit_86`` and zero findings on every run for weeks, and nothing
noticed: the baseline comparison above only covers the three *gated* cases, and
these two are informational, so an ``error`` verdict there was nobody's problem.
The sweep is deliberately independent of whether a case is gated.

A case that is *legitimately* expected to end without findings declares it in
the baselines file as an ``expected_error`` entry naming the reason, e.g.
``daily-consan-tiny``, whose exit 86 is an intended negative control. Declaring
the reason rather than just the status keeps the assertion sharp: the case stays
allowed to fail closed for the documented reason and nothing else.

Usage::

    compare_verdict_baselines.py <results-root>                 # gate + sweep
    compare_verdict_baselines.py --vacuous-only <results-root>   # sweep only

``--vacuous-only`` is for result trees that do not contain the gated cases at
all -- notably the non-gating survey job, which runs in its own job with its own
output root and is where the two #450 recipes actually live.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport
from aorta.instrumentation.rocjitsu_sanitizers.report import read_report

_BASELINES = Path("recipes/sanitizers/fixtures/expected/verdict_baselines.json")
_CASES = {
    "waitcheck_gemm": Path("waitcheck") / "sanitizer_report.json",
    "consan_clean": Path("consan-clean") / "sanitizer_report.json",
    "consan_racy": Path("consan-racy") / "sanitizer_report.json",
}
_GATED_BY_PATH = {relative: name for name, relative in _CASES.items()}

# Execution summaries that carry no signal. ERROR is the #450 shape (the run died
# before concluding anything); NOT_CHECKED means the sanitizer never ran, which is
# equally uninformative when nothing declared it acceptable. COMPLETE and PARTIAL
# both mean the sanitizer reached a conclusion, so a finding-free run under either
# is a real clean result -- daily-consan-lds-dispatch passes with zero findings and
# must not trip this.
_VACUOUS_STATUSES = frozenset({"error", "not_checked"})


def _load_baselines(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"baselines file is empty or malformed: {path}")
    return data


def _check_by_name(report: SanitizerReport, sanitizer: str):
    for check in report.checks:
        if check.sanitizer == sanitizer:
            return check
    return None


def _compare_case(name: str, report: SanitizerReport, expected: dict[str, object]) -> list[str]:
    problems: list[str] = []
    want_verdict = expected.get("overall_verdict")
    if report.overall_verdict.value != want_verdict:
        problems.append(
            f"overall_verdict={report.overall_verdict.value!r}, expected {want_verdict!r}"
        )
    want_execution = expected.get("execution_status")
    if want_execution is not None and report.execution_status.value != want_execution:
        problems.append(
            f"execution_status={report.execution_status.value!r}, expected {want_execution!r}"
        )
    for sanitizer, want_check in (expected.get("checks") or {}).items():
        check = _check_by_name(report, sanitizer)
        if check is None:
            problems.append(f"missing check {sanitizer!r}")
        elif check.verdict.value != want_check:
            problems.append(
                f"check {sanitizer!r} verdict={check.verdict.value!r}, expected {want_check!r}"
            )
    for sanitizer, want_shape in (expected.get("finding_shape") or {}).items():
        check = _check_by_name(report, sanitizer)
        messages = [] if check is None else [finding.message for finding in check.findings]
        if not any(want_shape in message for message in messages):
            problems.append(
                f"check {sanitizer!r} findings do not contain expected shape {want_shape!r}"
            )
    return problems


def _case_key(report_path: Path, root: Path) -> str:
    """Baseline key for a report, whether or not it is one of the gated cases.

    Gated cases are looked up by their exact relative path. Everything else is
    keyed off its case directory, which the nightly names after the recipe:
    ``informational/consan-tiny/`` -> ``consan_tiny``.
    """
    relative = report_path.relative_to(root)
    gated = _GATED_BY_PATH.get(relative)
    if gated is not None:
        return gated
    return relative.parent.name.replace("-", "_")


def _reasons(report: SanitizerReport) -> list[str]:
    return sorted({check.reason for check in report.checks if check.reason})


def _vacuous_problems(report: SanitizerReport, expected: object) -> list[str]:
    """Reject a run that terminated without producing any signal.

    Returns problems when the report ended in a vacuous execution status with
    zero findings across every check, unless the baselines declare an
    ``expected_error`` whose ``reason`` the run actually matches.
    """
    status = report.execution_status.value
    if status not in _VACUOUS_STATUSES:
        return []
    if any(check.findings for check in report.checks):
        return []

    reasons = _reasons(report)
    shape = f"execution_status={status!r}, overall_verdict={report.overall_verdict.value!r}"
    if reasons:
        shape += f", reason {', '.join(repr(reason) for reason in reasons)}"
    shape += ", and zero findings"

    declared = expected.get("expected_error") if isinstance(expected, dict) else None
    if not isinstance(declared, dict):
        return [
            f"{shape} -- the run produced no sanitizer signal at all. Either the recipe "
            f"is misconfigured (see ROCm/aorta#450) or, if this outcome is intended, "
            f"declare it as an 'expected_error' entry in {_BASELINES}."
        ]

    want = declared.get("reason")
    if isinstance(want, str) and not any(want in reason for reason in reasons):
        return [
            f"{shape} -- declared expected_error reason {want!r} is absent, so this is "
            f"a different failure than the one that was signed off."
        ]
    return []


def _sweep_vacuous(root: Path, baselines: dict[str, object]) -> bool:
    """Check every report under ``root``, gated or not, for a vacuous outcome.

    Judges only the reports that are present. Absent reports are deliberately not
    an error here: the survey job is non-gating and may legitimately be killed by
    the runner cap tracked in #370, and the gated cases already have their own
    presence check in the comparison above.
    """
    failed = False
    reports = sorted(root.rglob("sanitizer_report.json"))
    if not reports:
        print(f"vacuity sweep: no reports found under {root}")
        return False
    for report_path in reports:
        name = _case_key(report_path, root)
        try:
            report = read_report(report_path)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{name}: report failed strict validation: {exc}")
            failed = True
            continue
        problems = _vacuous_problems(report, baselines.get(name))
        for problem in problems:
            print(f"{name}: {problem}")
        failed = failed or bool(problems)
    if not failed:
        print(f"vacuity sweep: ok ({len(reports)} report(s) carry signal or are declared)")
    return failed


def main(argv: list[str]) -> int:
    args = argv[1:]
    vacuous_only = False
    if args and args[0] == "--vacuous-only":
        vacuous_only = True
        args = args[1:]
    if len(args) != 1:
        print(
            "usage: compare_verdict_baselines.py [--vacuous-only] <results-root>",
            file=sys.stderr,
        )
        return 2
    root = Path(args[0])
    try:
        baselines = _load_baselines(_BASELINES)
    except (OSError, ValueError) as exc:
        print(f"error: could not read baselines {_BASELINES}: {exc}", file=sys.stderr)
        return 2

    if vacuous_only:
        return 1 if _sweep_vacuous(root, baselines) else 0

    failed = False
    for name, relative in _CASES.items():
        expected = baselines.get(name)
        if not isinstance(expected, dict):
            print(f"{name}: no baseline entry")
            failed = True
            continue
        report_path = root / relative
        if not report_path.is_file():
            print(f"{name}: missing report {report_path}")
            failed = True
            continue
        try:
            report = read_report(report_path)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{name}: report failed strict validation: {exc}")
            failed = True
            continue
        problems = _compare_case(name, report, expected)
        if problems:
            failed = True
            for problem in problems:
                print(f"{name}: {problem}")
        else:
            print(f"{name}: ok ({report.overall_verdict.value})")

    if _sweep_vacuous(root, baselines):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
