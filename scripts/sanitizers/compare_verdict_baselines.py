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
the baselines file as an ``expected_error`` entry, e.g. ``daily-consan-tiny``,
whose exit 86 is an intended negative control. The declaration is an exact
description of the one outcome that was signed off -- execution status, overall
verdict and the complete set of check reasons -- and the run must match it on
every field:

.. code-block:: json

    "consan_tiny": {
      "expected_error": {
        "execution_status": "error",
        "overall_verdict": "error",
        "reasons": ["combined_hook_exit_86"],
        "why": "..."
      }
    }

A declaration that is not well-formed is a **failure, not a bypass**. Every
``expected_error`` in the baselines file is validated when the file is loaded,
whether or not that case ran, because a mis-spelled or half-written declaration
would otherwise silently widen an exemption that exists to cover exactly one
case: the entry is meant to *narrow* what a row is allowed to do, so anything
about it that cannot be checked has to be rejected rather than skipped.

Absence is judged too, but only for cases the caller says it invoked. Pass
``--require <case-dir>`` for each one; a required case with no report is an
error. See ``_sweep_vacuous`` for why a job-level kill does not need tolerating
here.

A row that is vacuous today for a reason tracked *elsewhere* is spelled
``--known-vacuous <case-dir>`` at the call site, and it is the opposite of an
``expected_error``: the declaration says "this outcome is correct", whereas this
says "this outcome is wrong, it is being worked, and it must not mask the other
rows". Consequently it **fails when the case stops being vacuous**, so a fixed
row cannot leave a stale suppression behind, and a case may not be both.
Spelling it as an argument rather than a baselines entry is deliberate: it
belongs to one lane's current state, shows up in a workflow diff, and carries no
dashboard meaning.

Usage::

    compare_verdict_baselines.py <results-root>                  # gate + sweep
    compare_verdict_baselines.py --vacuous-only <results-root>   # sweep only
    compare_verdict_baselines.py --vacuous-only \
        --require informational/consan-gemm \
        --known-vacuous informational/consan-gemm <results-root>

``--vacuous-only`` is for result trees that do not contain the gated cases at
all -- notably the non-gating survey job, which runs in its own job with its own
output root and is where the two #450 recipes actually live. It selects which
*halves* run, nothing more: ``--require`` and ``--known-vacuous`` both describe
the sweep, so they mean the same thing with or without it. An option that is
accepted and then quietly ignored on one path is the very shape this script
exists to catch.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from aorta.instrumentation.rocjitsu_sanitizers.models import (
    ExecutionSummary,
    SanitizerReport,
    Verdict,
)
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
_VACUOUS_STATUSES = frozenset({ExecutionSummary.ERROR.value, ExecutionSummary.NOT_CHECKED.value})

# The verdicts those statuses can carry. Spelled out for the same reason as the
# statuses: a declaration is only protection if it could actually match, so
# `overall_verdict: "pass"` is a typo to reject, not a rule to file away.
_VACUOUS_VERDICTS = frozenset({Verdict.ERROR.value, Verdict.NOT_CHECKED.value})

# Every field an `expected_error` must carry, and no others. Spelled as an exact
# set rather than a minimum so that a typo -- `reason` for `reasons`, say -- is a
# loud failure instead of a field nobody compares. An exemption that is not
# checkable is indistinguishable from no exemption at all.
_EXPECTED_ERROR_FIELDS = frozenset({"execution_status", "overall_verdict", "reasons", "why"})


def _expected_error_problems(name: str, declared: object) -> list[str]:
    """Reject an ``expected_error`` that does not describe exactly one outcome.

    Checked at load time for every declared case, run or not: a malformed
    declaration is a defect in the committed baselines, and waiting for the case
    to run before saying so would leave the exemption unverified on every run
    where the recipe failed to produce a report at all.
    """
    where = f"{name}.expected_error"
    if not isinstance(declared, dict):
        return [f"{where} must be an object, got {type(declared).__name__}"]

    problems = []
    missing = sorted(_EXPECTED_ERROR_FIELDS - declared.keys())
    unknown = sorted(declared.keys() - _EXPECTED_ERROR_FIELDS)
    if missing:
        problems.append(f"{where} is missing {', '.join(missing)}")
    if unknown:
        problems.append(f"{where} has unknown field(s) {', '.join(unknown)}")

    status = declared.get("execution_status")
    if "execution_status" in declared and status not in _VACUOUS_STATUSES:
        # Only a vacuous status can ever reach this exemption, so declaring any
        # other one is dead configuration that reads as protection.
        problems.append(
            f"{where}.execution_status={status!r} is not one of "
            f"{sorted(_VACUOUS_STATUSES)}, so it could never match"
        )
    verdict = declared.get("overall_verdict")
    if "overall_verdict" in declared and verdict not in _VACUOUS_VERDICTS:
        problems.append(
            f"{where}.overall_verdict={verdict!r} is not one of "
            f"{sorted(_VACUOUS_VERDICTS)}, so it could never match"
        )
    why = declared.get("why")
    if "why" in declared and (not isinstance(why, str) or not why.strip()):
        problems.append(f"{where}.why must be a non-empty string explaining the sign-off")

    reasons = declared.get("reasons")
    if "reasons" in declared:
        if (
            not isinstance(reasons, list)
            or not reasons
            or not all(isinstance(reason, str) and reason.strip() for reason in reasons)
        ):
            problems.append(
                f"{where}.reasons must be a non-empty list of non-empty strings, got {reasons!r}"
            )
        elif len(set(reasons)) != len(reasons):
            problems.append(f"{where}.reasons contains duplicates: {reasons!r}")
    return problems


def _load_baselines(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"baselines file is empty or malformed: {path}")
    problems = [
        problem
        for name, expected in data.items()
        if isinstance(expected, dict) and "expected_error" in expected
        for problem in _expected_error_problems(name, expected["expected_error"])
    ]
    # A gated key may not also carry an exemption, because the two mechanisms
    # address different reports under the same name. `_case_key` maps the survey
    # directory `informational/waitcheck-gemm` to `waitcheck_gemm`, which is also
    # the gated key for `waitcheck/sanitizer_report.json` -- two different
    # recipes. An `expected_error` added for either one would silently excuse the
    # other, which is the cross-case bypass this exactness work exists to
    # prevent. Declaring one of them means renaming so the two stop colliding.
    problems.extend(
        f"{name}.expected_error is not allowed on a gated case: {name} is also the "
        f"baseline key for a survey case directory, so the exemption could not be "
        f"attributed to one report or the other"
        for name in sorted(data.keys() & _CASES.keys())
        if isinstance(data[name], dict) and "expected_error" in data[name]
    )
    if problems:
        raise ValueError("; ".join(problems))
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


def _case_key_for_dir(case_dir: str) -> str:
    """Baseline key for a case *directory*, as ``--known-vacuous`` names one.

    The same mapping as :func:`_case_key`, reached from the directory rather
    than from a report path, so a caller-supplied case can be looked up before
    any report exists.
    """
    return _case_key(Path(case_dir) / "sanitizer_report.json", Path("."))


def _vacuous_shape(report: SanitizerReport, reasons: Sequence[str]) -> str:
    """One-line description of what a report ended up as, without any advice."""
    shape = (
        f"execution_status={report.execution_status.value!r}, "
        f"overall_verdict={report.overall_verdict.value!r}"
    )
    if reasons:
        shape += f", reason {', '.join(repr(reason) for reason in reasons)}"
    return shape + ", and zero findings"


def _reasons(report: SanitizerReport) -> list[str]:
    return sorted({check.reason for check in report.checks if check.reason})


def _vacuous_problems(report: SanitizerReport, expected: object) -> list[str]:
    """Reject a run that terminated without producing any signal.

    Returns problems when the report ended in a vacuous execution status with
    zero findings across every check, unless the baselines declare an
    ``expected_error`` that the run matches *exactly* -- same execution status,
    same overall verdict, and the same complete set of check reasons.

    Exact rather than "contains", on all three fields, because the point of the
    declaration is to name one signed-off outcome. A substring test admits
    ``combined_hook_exit_860``; ignoring the reasons the run carries beyond the
    declared one admits a second, undeclared failure riding along with it; and
    ignoring the status admits a ``not_checked`` run that never started as though
    it were the ``error`` that was reviewed.

    ``expected_error`` is validated at load time, so anything reaching here is
    well-formed.
    """
    status = report.execution_status.value
    if status not in _VACUOUS_STATUSES:
        return []
    if any(check.findings for check in report.checks):
        return []

    reasons = _reasons(report)
    shape = _vacuous_shape(report, reasons)

    declared = expected.get("expected_error") if isinstance(expected, dict) else None
    if declared is None:
        return [
            f"{shape} -- the run produced no sanitizer signal at all. Either the recipe "
            f"is misconfigured (see ROCm/aorta#450) or, if this outcome is intended, "
            f"declare it as an 'expected_error' entry in {_BASELINES}."
        ]

    mismatches = []
    if status != declared["execution_status"]:
        mismatches.append(
            f"execution_status={status!r}, declared {declared['execution_status']!r}"
        )
    if report.overall_verdict.value != declared["overall_verdict"]:
        mismatches.append(
            f"overall_verdict={report.overall_verdict.value!r}, "
            f"declared {declared['overall_verdict']!r}"
        )
    if reasons != sorted(declared["reasons"]):
        mismatches.append(f"reasons={reasons!r}, declared {sorted(declared['reasons'])!r}")
    if mismatches:
        return [
            f"{shape} -- this is not the outcome that was signed off as an "
            f"expected_error ({'; '.join(mismatches)})."
        ]
    return []


def _sweep_vacuous(
    root: Path,
    baselines: dict[str, object],
    required: Sequence[str] = (),
    known_vacuous: Sequence[str] = (),
) -> bool:
    """Check every report under ``root``, gated or not, for a vacuous outcome.

    Judges every report present, plus the absence of any case named by
    ``--require``. A caller that knows what it invoked has to say so, because the
    sweep cannot infer it from a directory listing: an empty tree and a tree from
    a run that was never asked to produce anything look identical here.

    That distinction matters because the survey job runs all six of its recipes
    under ``|| true``, so a recipe that dies before writing a report leaves no
    trace at all -- the same fail-open shape this sweep exists to close, one level
    up. Requiring the invoked cases closes it.

    A *job-level* kill (the runner cap tracked in #370, a cancellation, a step
    timeout) needs no tolerance here and never did: it takes the whole payload
    down, so this comparator does not run and has no verdict to give. The only
    way to reach this function is for all six invocations to have returned.

    ``known_vacuous`` case directories are reported but do not fail the run, and
    a case listed there that turns out to carry signal *does* fail: the whole
    point is that the list has to shrink to nothing, so it cannot be left behind
    once the underlying break is fixed.
    """
    failed = False
    suppressed = {case.rstrip("/") for case in known_vacuous}
    for case in required:
        if not (root / case / "sanitizer_report.json").is_file():
            print(
                f"{case}: no report under {root / case} -- the recipe was invoked but "
                f"produced nothing, so there is no outcome to judge. A startup, recipe "
                f"or input failure looks exactly like this."
            )
            failed = True
    reports = sorted(root.rglob("sanitizer_report.json"))
    if not reports:
        print(f"vacuity sweep: no reports found under {root}")
        return failed
    seen_suppressed = set()
    for report_path in reports:
        name = _case_key(report_path, root)
        case_dir = report_path.parent.relative_to(root).as_posix()
        try:
            report = read_report(report_path)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{name}: report failed strict validation: {exc}")
            failed = True
            continue
        problems = _vacuous_problems(report, baselines.get(name))
        if case_dir in suppressed:
            seen_suppressed.add(case_dir)
            if problems:
                # Deliberately NOT the advice _vacuous_problems gives: a
                # known-vacuous row must not be talked into an expected_error,
                # which would assert the outcome is fine.
                print(
                    f"{name}: KNOWN VACUOUS, not failing -- "
                    f"{_vacuous_shape(report, _reasons(report))}. This row is on the "
                    f"caller's --known-vacuous list: the outcome is wrong and tracked "
                    f"elsewhere, not accepted."
                )
            else:
                print(
                    f"{name}: listed as --known-vacuous but this run produced signal. "
                    f"Remove {case_dir} from the caller's --known-vacuous list -- a "
                    f"suppression that is no longer true hides the next regression."
                )
                failed = True
            continue
        for problem in problems:
            print(f"{name}: {problem}")
        failed = failed or bool(problems)
    for case_dir in sorted(suppressed - seen_suppressed):
        print(
            f"{case_dir}: listed as --known-vacuous but no report was found for it. "
            f"Either it was not invoked or the suppression is stale."
        )
        failed = True
    if not failed:
        note = f" ({len(seen_suppressed)} known-vacuous)" if seen_suppressed else ""
        print(
            f"vacuity sweep: ok ({len(reports)} report(s) carry signal or are "
            f"declared){note}"
        )
    return failed


def main(argv: list[str]) -> int:
    args = argv[1:]
    vacuous_only = False
    required: list[str] = []
    known_vacuous: list[str] = []
    positional: list[str] = []
    while args:
        arg, args = args[0], args[1:]
        if arg == "--vacuous-only":
            vacuous_only = True
        elif arg in ("--require", "--known-vacuous"):
            if not args:
                print(f"error: {arg} needs a case directory", file=sys.stderr)
                return 2
            (required if arg == "--require" else known_vacuous).append(args[0])
            args = args[1:]
        elif arg.startswith("--"):
            print(f"error: unknown option {arg}", file=sys.stderr)
            return 2
        else:
            positional.append(arg)
    if len(positional) != 1:
        print(
            "usage: compare_verdict_baselines.py [--vacuous-only] "
            "[--require <case-dir>]... [--known-vacuous <case-dir>]... <results-root>",
            file=sys.stderr,
        )
        return 2
    root = Path(positional[0])
    try:
        baselines = _load_baselines(_BASELINES)
    except (OSError, ValueError) as exc:
        print(f"error: could not read baselines {_BASELINES}: {exc}", file=sys.stderr)
        return 2

    # The two mechanisms make opposite claims about the same row, so a case
    # cannot be both. Caught here rather than shrugged off, because whichever
    # one the reader believed would be the wrong one half the time.
    contradictory = sorted(
        case
        for case in known_vacuous
        if isinstance(entry := baselines.get(_case_key_for_dir(case)), dict)
        and "expected_error" in entry
    )
    if contradictory:
        print(
            f"error: {', '.join(contradictory)} is both --known-vacuous and declared as "
            f"an expected_error in {_BASELINES}. The first says the outcome is wrong and "
            f"being worked; the second says it is correct. Pick one.",
            file=sys.stderr,
        )
        return 2

    if vacuous_only:
        return 1 if _sweep_vacuous(root, baselines, required, known_vacuous) else 0

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

    if _sweep_vacuous(root, baselines, required, known_vacuous):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
