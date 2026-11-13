"""Optional diagnostic evidence validation and conservative aggregation."""
from dataclasses import replace

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.models import (
    CheckResult, CountMeaning, DiagnosticSummary, EvidenceStatus, ExecutionState,
    KernelCheckResult, KernelIdentity, aggregate_diagnostic_summaries, Verdict,
    KernelObservation, KernelWorklist, SanitizerReport, SelectionRequirement,
)


def entry(summary=None, state=ExecutionState.RAN):
    return KernelCheckResult(
        identity=KernelIdentity(name="test", target="gfx1250"), state=state,
        verdict=Verdict.WARN if state is ExecutionState.RAN else Verdict.ERROR,
        diagnostic_summary=summary,
    )


EXACT = DiagnosticSummary(reported_count=3, count_meaning=CountMeaning.EXACT,
                          emitted_records=1, requested_limit=1,
                          details=EvidenceStatus.INCOMPLETE, native_analysis=EvidenceStatus.COMPLETE)


def test_old_reports_and_constructors_have_no_affirmative_evidence():
    old = entry()
    assert "diagnostic_summary" not in old.to_dict()
    assert KernelCheckResult.from_dict(old.to_dict()) == old
    check = CheckResult("waitcheck", ExecutionState.RAN, Verdict.WARN, kernel_results=(old,))
    assert "diagnostic_summary" not in check.to_dict()
    assert CheckResult.from_dict(check.to_dict()) == check
    assert DiagnosticSummary.from_dict({}) == DiagnosticSummary()


def test_optional_metadata_round_trip():
    result = entry(EXACT)
    summary = aggregate_diagnostic_summaries((result,))
    check = CheckResult("waitcheck", ExecutionState.RAN, Verdict.WARN,
                        kernel_results=(result,), diagnostic_summary=summary)
    assert CheckResult.from_dict(check.to_dict()) == check


@pytest.mark.parametrize("field,value", [
    ("reported_count", True), ("reported_count", -1), ("reported_count", 1.0),
    ("emitted_records", "1"), ("emitted_records", -1), ("requested_limit", 0),
    ("requested_limit", False), ("count_meaning", "total"), ("details", "yes"),
    ("native_analysis", "success"), ("unknown", 1),
])
def test_malformed_metadata_is_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        DiagnosticSummary.from_dict({field: value})


@pytest.mark.parametrize("changes", [
    {"reported_count": None}, {"reported_count": 0},
    {"details": EvidenceStatus.COMPLETE},
    {"count_meaning": CountMeaning.LOWER_BOUND},
    {"native_analysis": EvidenceStatus.INCOMPLETE},
])
def test_contradictory_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(EXACT, **changes)


@pytest.mark.parametrize("other", [
    entry(), entry(DiagnosticSummary(reported_count=3, emitted_records=1)),
    entry(DiagnosticSummary(reported_count=3, count_meaning=CountMeaning.LOWER_BOUND)),
    entry(EXACT, ExecutionState.ERROR),
])
def test_mixed_unknown_missing_lower_bound_and_error_do_not_fabricate_totals(other):
    summary = aggregate_diagnostic_summaries((entry(EXACT), other))
    assert summary.reported_count is None
    assert summary.count_meaning is CountMeaning.UNKNOWN
    assert summary.native_analysis is not EvidenceStatus.COMPLETE


def test_exact_aggregate_and_explicit_early_stop():
    summary = aggregate_diagnostic_summaries((entry(EXACT), entry(EXACT)))
    assert (summary.reported_count, summary.emitted_records, summary.requested_limit) == (6, 2, 1)
    assert summary.native_analysis is EvidenceStatus.COMPLETE
    assert summary.details is EvidenceStatus.INCOMPLETE
    early = replace(EXACT, count_meaning=CountMeaning.LOWER_BOUND, native_analysis=EvidenceStatus.INCOMPLETE)
    assert aggregate_diagnostic_summaries((entry(EXACT), entry(early))).native_analysis is EvidenceStatus.INCOMPLETE


def test_invalid_aggregate_claim_is_rejected():
    with pytest.raises(ValueError, match="per-entry"):
        CheckResult("waitcheck", ExecutionState.RAN, Verdict.WARN,
                    kernel_results=(entry(),), diagnostic_summary=EXACT)


def test_exact_report_requires_results_for_the_full_worklist():
    result = entry(EXACT)
    check = CheckResult("waitcheck", ExecutionState.RAN, Verdict.WARN,
                        kernel_results=(result,), diagnostic_summary=EXACT)
    worklist = KernelWorklist(
        requirement=SelectionRequirement.TOP_TIME, top_n=2,
        kernels=tuple(KernelObservation(identity=KernelIdentity(name=name, target="gfx1250"),
                                        sources=("synthetic",)) for name in ("test", "missing")),
    )
    with pytest.raises(ValueError, match="missing selected"):
        SanitizerReport(target="gfx1250", worklist=worklist, checks=(check,))


def test_complete_details_and_native_analysis_stay_separate():
    summary = DiagnosticSummary(reported_count=1, count_meaning=CountMeaning.EXACT,
                                emitted_records=1, details=EvidenceStatus.COMPLETE)
    aggregate = aggregate_diagnostic_summaries((entry(summary),))
    assert aggregate.details is EvidenceStatus.COMPLETE
    assert aggregate.native_analysis is EvidenceStatus.UNKNOWN


def test_exact_report_cannot_include_unselected_results():
    first = entry(EXACT)
    foreign = replace(first, identity=KernelIdentity(name="foreign", target="gfx1250"))
    results = (first, foreign)
    check = CheckResult("waitcheck", ExecutionState.RAN, Verdict.WARN,
                        kernel_results=results, diagnostic_summary=aggregate_diagnostic_summaries(results))
    worklist = KernelWorklist(requirement=SelectionRequirement.TOP_TIME, top_n=1,
                             kernels=(KernelObservation(identity=first.identity, sources=("test",)),))
    with pytest.raises(ValueError, match="unselected"):
        SanitizerReport(target="gfx1250", worklist=worklist, checks=(check,))


def test_exact_aggregate_rejects_duplicate_whole_object_aliases():
    identity = KernelIdentity(name="first", target="gfx1250", code_object="/fixtures/object.co",
                              code_object_sha256="a" * 64, code_object_index=0)
    first = replace(entry(EXACT), identity=identity)
    alias = replace(first, identity=replace(identity, name="alias"))
    results = (first, alias)
    check = CheckResult("waitcheck", ExecutionState.RAN, Verdict.WARN, kernel_results=results,
                        diagnostic_summary=aggregate_diagnostic_summaries(results))
    worklist = KernelWorklist(requirement=SelectionRequirement.TOP_TIME, top_n=1,
                             kernels=(KernelObservation(identity=identity, sources=("test",)),))
    with pytest.raises(ValueError, match="duplicate results"):
        SanitizerReport(target="gfx1250", worklist=worklist, checks=(check,))


def test_one_whole_object_result_covers_selected_aliases_without_double_counting():
    identity = KernelIdentity(name="first", target="gfx1250", code_object="/fixtures/object.co",
                              code_object_sha256="a" * 64, code_object_index=0)
    result = replace(entry(EXACT), identity=identity)
    check = CheckResult("waitcheck", ExecutionState.RAN, Verdict.WARN, kernel_results=(result,),
                        diagnostic_summary=aggregate_diagnostic_summaries((result,)))
    worklist = KernelWorklist(requirement=SelectionRequirement.TOP_TIME, top_n=2,
                             kernels=tuple(KernelObservation(identity=item, sources=("test",))
                                           for item in (identity, replace(identity, name="alias"))))
    report = SanitizerReport(target="gfx1250", worklist=worklist, checks=(check,))
    assert report.checks[0].diagnostic_summary.reported_count == 3


@pytest.mark.parametrize('entry_flag', ['omitted', False])
def test_check_rejects_exact_summary_with_aggregate_lower_bound(entry_flag):
    result = entry(EXACT)
    data = CheckResult('waitcheck', ExecutionState.RAN, Verdict.WARN,
                       kernel_results=(result,), diagnostic_summary=EXACT).to_dict()
    data['diagnostics_truncated'] = True
    if entry_flag == 'omitted':
        data['kernel_results'][0].pop('diagnostics_truncated')
    with pytest.raises(ValueError, match='lower-bound marker contradicts exact count'):
        CheckResult.from_dict(data)


@pytest.mark.parametrize('summary', [None, DiagnosticSummary(),
                                   DiagnosticSummary(reported_count=3, count_meaning=CountMeaning.LOWER_BOUND)])
def test_check_keeps_legacy_truncation_with_nonexact_or_absent_summary(summary):
    result = entry(summary)
    aggregate = aggregate_diagnostic_summaries((result,))
    check = CheckResult('waitcheck', ExecutionState.RAN, Verdict.WARN,
                        kernel_results=(result,), diagnostic_summary=aggregate,
                        diagnostics_truncated=True)
    assert CheckResult.from_dict(check.to_dict()) == check


def test_multi_entry_check_can_exceed_the_per_invocation_limit():
    first = entry(EXACT)
    second = replace(first, identity=KernelIdentity(name='second', target='gfx1250'))
    results = (first, second)
    aggregate = aggregate_diagnostic_summaries(results)
    check = CheckResult('waitcheck', ExecutionState.RAN, Verdict.WARN,
                        kernel_results=results, diagnostic_summary=aggregate)
    assert check.diagnostic_summary.emitted_records == 2
    assert check.diagnostic_summary.requested_limit == 1
    assert CheckResult.from_dict(check.to_dict()) == check
