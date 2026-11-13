"""Offline reader tests: arbitrary synthetic entries; no native execution."""
import importlib.util
import json
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.models import (
    CheckResult, ExecutionState, KernelCheckResult, KernelIdentity, KernelObservation,
    KernelWorklist, SelectionRequirement, Verdict,
    aggregate_diagnostic_summaries)
from aorta.instrumentation.rocjitsu_sanitizers.report import build_report, write_report
from aorta.instrumentation.rocjitsu_sanitizers.waitcheck import parse_waitcheck_text

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "waitcheck_report", ROOT / '.agents/skills/rocm-sanitizer-triage/scripts/waitcheck_report.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def evidence(tmp_path):
    identity = KernelIdentity(name='user_attention_variant_42', target='gfx950',
                              code_object='/user/project/objects/attention.so',
                              code_object_sha256='a' * 64, code_object_index=7, entry_offset=0x240)
    worklist = KernelWorklist(requirement=SelectionRequirement.TOP_DISPATCH_COUNT, top_n=1,
                             kernels=(KernelObservation(identity=identity, dispatch_count=1, sources=('synthetic',)),))

    def create(kind='clean'):
        prefix = f'{identity.code_object}:gfx950[7]'
        hazard = kind in ('warn', 'cap')
        log = (f'rj_waitcheck: {prefix}:kernel=.text+0x240: instructions=4 memory-events=2 '
               f'diagnostics={">=" if kind == "cap" else ""}{1 if hazard else 0}\n')
        if hazard:
            log += f'{prefix}:.text+0x248: missing s_waitcnt lgkmcnt(0)\n'
        parsed = parse_waitcheck_text(log, expected=identity)
        verdict = Verdict.WARN if hazard else Verdict.PASS
        result = KernelCheckResult(identity=identity, state=ExecutionState.RAN, verdict=verdict,
                                   returncode=4 if hazard else 0, findings=parsed.findings,
                                   diagnostics_truncated=kind == 'cap')
        check = CheckResult(sanitizer='waitcheck', state=ExecutionState.RAN, verdict=verdict,
                            findings=result.findings, kernel_results=(result,),
                            diagnostics_truncated=kind == 'cap', backend=(('path', '/tools/rj_waitcheck'),))
        if kind == 'digest':
            result = KernelCheckResult(identity=identity, state=ExecutionState.ERROR,
                                       verdict=Verdict.ERROR, reason='code_object_digest_mismatch')
            check = CheckResult(sanitizer='waitcheck', state=ExecutionState.ERROR,
                                verdict=Verdict.ERROR, reason='worklist_not_fully_checked', kernel_results=(result,))
        elif kind == 'timeout':
            result = KernelCheckResult(identity=identity, state=ExecutionState.TIMED_OUT,
                                       verdict=Verdict.ERROR, reason='waitcheck_timeout')
            check = CheckResult(sanitizer='waitcheck', state=ExecutionState.TIMED_OUT,
                                verdict=Verdict.ERROR, reason='worklist_not_fully_checked', kernel_results=(result,))
            log = 'partial stdout before timeout\n'
        elif kind == 'dry':
            check = CheckResult(sanitizer='waitcheck', state=ExecutionState.NOT_CHECKED,
                                verdict=Verdict.NOT_CHECKED, reason='dry_run')
        report_path, log_path = tmp_path / 'report.json', tmp_path / 'native.log'
        write_report(build_report(target=identity.target, worklist=worklist, checks=(check,)), report_path)
        log_path.write_text(log, encoding='utf-8')
        return report_path, log_path
    return create


@pytest.mark.parametrize('kind', ['clean', 'warn', 'cap'])
def test_arbitrary_identity_and_reported_result_with_limits(evidence, kind):
    report, log = evidence(kind)
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'Raw summary agrees' in summary
    assert 'user_attention_variant_42' in summary
    assert 'not independently verified provenance' in summary
    assert 'AORTA CLI exit is not recorded' in summary
    if kind in ('warn', 'cap'):
        assert '"native_returncode": 4' in summary
        assert '"overall_verdict": "warn"' in summary
    if kind == 'cap':
        assert 'lower bound' in summary


def test_missing_raw_is_a_limit_not_evidence_of_clean(evidence):
    report, _ = evidence()
    summary, code = helper.summarize(report)
    assert code == 0
    assert 'No raw log supplied' in summary
    assert 'Raw findings/entry evidence was not compared' in summary


@pytest.mark.parametrize('kind,reason', [('digest', 'code_object_digest_mismatch'), ('dry', 'dry_run')])
def test_prelaunch_outcome_needs_no_raw(evidence, kind, reason):
    report, _ = evidence(kind)
    summary, code = helper.summarize(report)
    assert code == 0
    assert reason in summary
    assert 'No completed single-entry check is established' in summary


def test_timeout_keeps_partial_log_without_parse_pass(evidence):
    report, log = evidence('timeout')
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'waitcheck_timeout' in summary
    assert 'partial stdout before timeout' in summary
    assert 'not parsed as a completed scan' in summary


@pytest.mark.parametrize('mutation', ['path', 'entry', 'count', 'extra', 'unsupported', 'truncation'])
def test_raw_contradictions_and_unknown_lines_do_not_confirm_clean(evidence, mutation):
    report, log = evidence()
    raw = log.read_text()
    if mutation == 'path':
        raw = raw.replace('/user/project/', '/another/')
    elif mutation == 'entry':
        raw = raw.replace('0x240', '0x280')
    elif mutation == 'count':
        raw = raw.replace('diagnostics=0', 'diagnostics=2')
    elif mutation == 'extra':
        raw += 'ERROR: incomplete instruction support\n'
    elif mutation == 'truncation':
        raw = raw.replace('diagnostics=0', 'diagnostics=>=0')
    else:
        raw = 'new incompatible summary vocabulary\n'
    log.write_text(raw)
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'INCONSISTENT OR UNSUPPORTED' in summary
    assert 'Raw summary agrees' not in summary


@pytest.mark.parametrize('mutation', ['scope', 'identity', 'unknown', 'invalid'])
def test_invalid_or_unsupported_report_refused(evidence, mutation):
    report, _ = evidence()
    data = json.loads(report.read_text())
    if mutation == 'scope':
        data['checks'].append(data['checks'][0])
    elif mutation == 'identity':
        data['checks'][0]['kernel_results'][0]['identity']['entry_offset'] = 0
    elif mutation == 'unknown':
        data['checks'][0]['completion_status'] = 'unsupported-instructions'
    else:
        data['overall_verdict'] = 'error'
    report.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        helper.summarize(report)


def test_cli_output_exclusivity_and_reader_exit_two(evidence, tmp_path):
    report, log = evidence()
    output = tmp_path / 'summary.md'
    output.write_text('keep')
    argv = ['--report', str(report), '--log', str(log), '--output', str(output)]
    assert helper.main(argv) == 1
    assert output.read_text() == 'keep'
    output.unlink()
    log.write_text('unsupported output')
    assert helper.main(argv) == 2
    assert 'INCONSISTENT OR UNSUPPORTED' in output.read_text()


def test_bounded_report_input(evidence, monkeypatch):
    report, _ = evidence()
    monkeypatch.setattr(helper, 'LIMIT', 10)
    with pytest.raises(ValueError, match='exceeds'):
        helper.summarize(report)


def test_duplicate_json_field_refused(evidence):
    report, _ = evidence()
    raw = report.read_text()
    raw = raw.replace('"overall_verdict": "pass"', '"overall_verdict": "error", "overall_verdict": "pass"')
    report.write_text(raw)
    with pytest.raises(ValueError, match='duplicate JSON field'):
        helper.summarize(report)


def test_foreign_diagnostic_is_not_attributed_to_selected_entry(evidence):
    report, log = evidence('warn')
    original = log.read_text().splitlines()
    original[1] = original[1].replace('/user/project/', '/foreign/')
    log.write_text('\n'.join(original))
    data = json.loads(report.read_text())
    for findings in (data['checks'][0]['findings'], data['checks'][0]['kernel_results'][0]['findings']):
        findings[0]['message'] = original[1]
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'diagnostic object/target/image prefix' in summary


@pytest.mark.parametrize('mutation', ['verdict', 'returncode'])
def test_aggregate_result_conflict_preserved_and_flagged(evidence, mutation):
    report, log = evidence()
    data = json.loads(report.read_text())
    if mutation == 'verdict':
        data['overall_verdict'] = data['checks'][0]['verdict'] = 'warn'
    else:
        data['checks'][0]['returncode'] = 4
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'Report consistency: INCONSISTENT' in summary
    assert '"native_returncode": 0' in summary


@pytest.mark.parametrize('kind,native_exit', [('clean', 4), ('warn', 0), ('clean', None), ('warn', None)])
def test_report_only_native_completion_conflict(evidence, kind, native_exit):
    report, _ = evidence(kind)
    data = json.loads(report.read_text())
    data['checks'][0]['kernel_results'][0]['returncode'] = native_exit
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report)
    assert code == 2
    assert 'Report consistency: INCONSISTENT' in summary
    assert 'native exit is missing or contradicts' in summary
    assert f'"native_returncode": {json.dumps(native_exit)}' in summary
    assert f'"entry_verdict": "{"pass" if kind == "clean" else "warn"}"' in summary


def rewrite_evidence(report_path, log_path, text, *, include_metadata=False, requested_limit=None):
    """Portable synthetic derivatives; never execute a native program."""
    from dataclasses import replace
    from aorta.instrumentation.rocjitsu_sanitizers.report import read_report
    report = read_report(report_path)
    previous = report.checks[0].kernel_results[0]
    parsed = parse_waitcheck_text(text, expected=previous.identity)
    metadata = replace(parsed.diagnostic_summary, requested_limit=requested_limit) if include_metadata else None
    result = replace(previous, findings=parsed.findings, diagnostics_truncated=parsed.diagnostics_truncated,
                     diagnostic_summary=metadata)
    check = replace(report.checks[0], findings=result.findings, kernel_results=(result,),
                    diagnostics_truncated=result.diagnostics_truncated,
                    diagnostic_summary=aggregate_diagnostic_summaries((result,)) if include_metadata else None)
    write_report(build_report(target=report.target, worklist=report.worklist, checks=(check,)), report_path)
    log_path.write_text(text, encoding='utf-8')


def test_duplicate_records_are_not_omitted_details(evidence):
    report, log = evidence('warn')
    text = log.read_text().replace('diagnostics=1', 'diagnostics=2')
    text += text.splitlines()[1] + '\n'
    rewrite_evidence(report, log, text, include_metadata=True)
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'emitted records=2, retained unique findings=1' in summary
    assert 'compatibility gap' not in summary
    assert 'native analysis=unknown' in summary


@pytest.mark.parametrize('include_metadata', [False, True])
def test_plain_limited_details_remain_explicit_compatibility_gap(evidence, include_metadata):
    report, log = evidence('warn')
    text = log.read_text().replace('diagnostics=1', 'diagnostics=3')
    rewrite_evidence(report, log, text, include_metadata=include_metadata, requested_limit=1)
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'compatibility gap' in summary
    assert 'reported diagnostics=3, emitted records=1, retained unique findings=1' in summary
    assert 'count meaning=unknown' in summary
    assert 'native analysis=unknown' in summary
    assert '"native_returncode": 4' in summary
    assert '"diagnostics_truncated": false' in summary
    assert '"reported_count": 3' in summary
    assert '"emitted_records": 1' in summary


@pytest.mark.parametrize('include_metadata', [False, True])
def test_explicit_storage_omission_is_supported_separately_from_completion(evidence, include_metadata):
    report, log = evidence('warn')
    text = log.read_text().replace('diagnostics=1', 'diagnostics=3')
    text += 'omitted 2 diagnostic(s) after limit\n'
    rewrite_evidence(report, log, text, include_metadata=include_metadata, requested_limit=1)
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'Count meaning=exact; details=incomplete; native analysis=complete' in summary
    assert '"diagnostics_truncated": false' in summary


@pytest.mark.parametrize('signal', ['omitted at least 2 diagnostic(s) after limit',
                                    'analysis stopped early; additional diagnostics may exist'])
def test_explicit_early_stop_is_not_completed_analysis(evidence, signal):
    report, log = evidence('cap')
    text = log.read_text().replace('instructions=4', 'instructions=>=4').replace('memory-events=2', 'memory-events=>=2')
    if signal.startswith('omitted'):
        text = text.replace('diagnostics=>=1', 'diagnostics=>=3')
    text += signal + '\n'
    rewrite_evidence(report, log, text, include_metadata=True, requested_limit=1)
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'Count meaning=lower_bound' in summary
    assert 'native analysis=incomplete' in summary
    assert '"execution_status": "complete"' in summary


def test_reaching_requested_cap_does_not_establish_completeness(evidence):
    report, log = evidence('warn')
    rewrite_evidence(report, log, log.read_text(), include_metadata=True, requested_limit=1)
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert '"requested_limit": 1' in summary
    assert 'Count meaning=unknown; details=unknown; native analysis=unknown' in summary


def test_legacy_report_without_new_or_old_optional_metadata_is_readable(evidence):
    report, log = evidence('clean')
    data = json.loads(report.read_text())
    for item in (data['checks'][0], data['checks'][0]['kernel_results'][0]):
        item.pop('diagnostics_truncated')
        item.pop('diagnostic_summary', None)
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'Legacy report' in summary
    assert 'Count meaning=unknown; details=unknown; native analysis=unknown' in summary


def test_report_only_keeps_metadata_and_missing_raw_limit(evidence):
    report, log = evidence('warn')
    rewrite_evidence(report, log, log.read_text(), include_metadata=True, requested_limit=64)
    summary, code = helper.summarize(report)
    assert code == 0
    assert '"requested_limit": 64' in summary
    assert '"native_analysis": "unknown"' in summary
    assert 'Raw findings/entry evidence was not compared' in summary
    assert '"diagnostic_summary_raw": null' in summary


@pytest.mark.parametrize('key,value', [('reported_count', True), ('emitted_records', -1),
                                      ('requested_limit', 0), ('details', 'yes'),
                                      ('native_analysis', 'success'), ('unknown_key', 1)])
def test_malformed_new_metadata_is_rejected(evidence, key, value):
    report, log = evidence('warn')
    rewrite_evidence(report, log, log.read_text(), include_metadata=True, requested_limit=1)
    data = json.loads(report.read_text())
    data['checks'][0]['kernel_results'][0]['diagnostic_summary'][key] = value
    report.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        helper.summarize(report, log)


def test_new_metadata_cannot_claim_a_different_raw_count(evidence):
    report, log = evidence('warn')
    rewrite_evidence(report, log, log.read_text(), include_metadata=True, requested_limit=1)
    data = json.loads(report.read_text())
    data['checks'][0]['kernel_results'][0]['diagnostic_summary']['reported_count'] = 2
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'raw reported_count differs from report metadata' in summary


def test_report_completion_claim_requires_supporting_raw_signal(evidence):
    report, log = evidence('warn')
    rewrite_evidence(report, log, log.read_text(), include_metadata=True, requested_limit=1)
    data = json.loads(report.read_text())
    for item in (data['checks'][0], data['checks'][0]['kernel_results'][0]):
        item['diagnostic_summary'].update(reported_count=1, count_meaning='exact', native_analysis='complete')
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'raw count_meaning does not support report metadata' in summary


@pytest.mark.parametrize('terminal', ['omitted 9 diagnostic(s) after limit',
                                     'omitted some diagnostics after limit',
                                     'scan complete'])
def test_conflicting_or_unknown_terminal_lines_do_not_confirm_completion(evidence, terminal):
    report, log = evidence('warn')
    log.write_text(log.read_text() + terminal + '\n')
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'INCONSISTENT OR UNSUPPORTED' in summary


def test_empty_optional_metadata_stays_unknown(evidence):
    report, _ = evidence('clean')
    data = json.loads(report.read_text())
    for item in (data['checks'][0], data['checks'][0]['kernel_results'][0]):
        item['diagnostic_summary'] = {}
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report)
    assert code == 0
    assert '"reported_count": null' in summary
    assert '"native_analysis": "unknown"' in summary


@pytest.mark.parametrize('kind,count,records', [
    ('clean', 3, 0), ('clean', None, 1), ('warn', 0, None), ('warn', 1, 0),
    ('warn', 1, 2),
])
def test_report_only_metadata_count_records_and_native_exit_conflicts(evidence, kind, count, records):
    report, _ = evidence(kind)
    data = json.loads(report.read_text())
    data['checks'][0]['kernel_results'][0]['diagnostic_summary'] = {
        'reported_count': count, 'emitted_records': records,
    }
    if kind == 'warn' and records == 0:
        data['checks'][0]['findings'] = []
        data['checks'][0]['kernel_results'][0]['findings'] = []
    report.write_text(json.dumps(data))
    # The reader preserves original claims and flags them without needing raw.
    summary, code = helper.summarize(report)
    assert code == 2
    assert 'Report consistency: INCONSISTENT' in summary
    assert '"native_returncode": ' + ('0' if kind == 'clean' else '4') in summary


@pytest.mark.parametrize('cap,records,meaning,native', [
    (64, 3, 'unknown', 'unknown'), (1, 1, 'exact', 'complete')])
@pytest.mark.parametrize('include_metadata', [False, True])
def test_saved_text_derivatives_keep_the_real_omission_signal(tmp_path, cap, records, meaning, native, include_metadata):
    from dataclasses import replace
    text = (Path(__file__).parent / 'fixtures' / f'waitcheck-cap{cap}.txt').read_text()
    identity = KernelIdentity(name='waitcheck_multiple_missing', target='gfx1250',
                              code_object='/fixtures/multiple_missing_waits',
                              code_object_sha256='a' * 64, code_object_index=0, entry_offset=0)
    worklist = KernelWorklist(requirement=SelectionRequirement.TOP_TIME, top_n=1,
                             kernels=(KernelObservation(identity=identity, sources=('fixture',)),))
    parsed = parse_waitcheck_text(text, expected=identity)
    result = KernelCheckResult(identity=identity, state=ExecutionState.RAN, verdict=Verdict.WARN,
                               findings=parsed.findings, returncode=4,
                               diagnostic_summary=replace(parsed.diagnostic_summary, requested_limit=cap) if include_metadata else None)
    check = CheckResult(sanitizer='waitcheck', state=ExecutionState.RAN, verdict=Verdict.WARN,
                        findings=result.findings, kernel_results=(result,),
                        diagnostic_summary=aggregate_diagnostic_summaries((result,)) if include_metadata else None)
    report_path, log_path = tmp_path / 'report.json', tmp_path / 'native.txt'
    write_report(build_report(target='gfx1250', worklist=worklist, checks=(check,)), report_path)
    log_path.write_text(text)
    summary, code = helper.summarize(report_path, log_path)
    assert code == 0
    assert f'emitted records={records}, retained unique findings={records}' in summary
    assert f'Count meaning={meaning}' in summary
    assert f'native analysis={native}' in summary
    assert '"native_returncode": 4' in summary
    if cap == 1:
        assert 'Supported raw signal: omitted 2 diagnostic(s) after limit' in summary


@pytest.mark.parametrize('kind', ['warn', 'cap'])
@pytest.mark.parametrize('metadata', ['absent', 'empty'])
def test_report_only_hazard_exit_requires_retained_findings(evidence, kind, metadata):
    report, _ = evidence(kind)
    data = json.loads(report.read_text())
    for item in (data['checks'][0], data['checks'][0]['kernel_results'][0]):
        item['findings'] = []
        if metadata == 'empty':
            item['diagnostic_summary'] = {}
        else:
            item.pop('diagnostic_summary', None)
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report)
    assert code == 2
    assert 'native hazard exit has no retained findings' in summary
    assert '"native_returncode": 4' in summary
    assert '"entry_verdict": "warn"' in summary


@pytest.mark.parametrize('check_present,entry_present', [(False, False), (True, False), (False, True), (True, True)])
def test_legacy_truncation_presence_is_not_a_false_claim(evidence, check_present, entry_present):
    report, log = evidence('cap')
    data = json.loads(report.read_text())
    for item, present in ((data['checks'][0], check_present),
                          (data['checks'][0]['kernel_results'][0], entry_present)):
        if not present:
            item.pop('diagnostics_truncated')
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'Raw summary agrees' in summary
    assert 'diagnostics=at least 1' in summary
    assert '"diagnostics_truncated": ' + ('true' if check_present else 'null') in summary
    assert '"entry_diagnostics_truncated": ' + ('true' if entry_present else 'null') in summary


@pytest.mark.parametrize('location', ['check', 'entry'])
def test_explicit_false_truncation_claim_still_conflicts_with_raw(evidence, location):
    report, log = evidence('cap')
    data = json.loads(report.read_text())
    check, entry = data['checks'][0], data['checks'][0]['kernel_results'][0]
    check.pop('diagnostics_truncated')
    entry.pop('diagnostics_truncated')
    (check if location == 'check' else entry)['diagnostics_truncated'] = False
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'truncation differs' in summary


@pytest.mark.parametrize('location', ['check', 'entry'])
def test_explicit_true_truncation_claim_still_conflicts_with_raw(evidence, location):
    report, log = evidence('clean')
    data = json.loads(report.read_text())
    check, entry = data['checks'][0], data['checks'][0]['kernel_results'][0]
    check.pop('diagnostics_truncated')
    entry.pop('diagnostics_truncated')
    (check if location == 'check' else entry)['diagnostics_truncated'] = True
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'truncation differs' in summary


@pytest.mark.parametrize('check_null,entry_null', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('with_log', [False, True])
def test_supported_nullable_diagnostic_summary_is_missing_evidence(evidence, check_null, entry_null, with_log):
    report, log = evidence('warn')
    data = json.loads(report.read_text())
    for item, include in ((data['checks'][0], check_null),
                          (data['checks'][0]['kernel_results'][0], entry_null)):
        if include:
            item['diagnostic_summary'] = None
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report, log if with_log else None)
    assert code == 0
    assert '"entry_diagnostic_summary_reported": null' in summary
    assert '"check_diagnostic_summary_reported": null' in summary
    assert '"native_returncode": 4' in summary


@pytest.mark.parametrize('location', ['root', 'check', 'entry', 'identity'])
def test_unknown_null_fields_are_not_accepted_as_nullable_metadata(evidence, location):
    report, _ = evidence()
    data = json.loads(report.read_text())
    target = {'root': data, 'check': data['checks'][0],
              'entry': data['checks'][0]['kernel_results'][0],
              'identity': data['checks'][0]['kernel_results'][0]['identity']}[location]
    target['unsupported_null_field'] = None
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='unsupported fields'):
        helper.summarize(report)


@pytest.mark.parametrize('location', ['root', 'identity'])
def test_nullable_summary_is_not_allowed_at_unmodeled_locations(evidence, location):
    report, _ = evidence()
    data = json.loads(report.read_text())
    target = data if location == 'root' else data['checks'][0]['kernel_results'][0]['identity']
    target['diagnostic_summary'] = None
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='unsupported fields'):
        helper.summarize(report)


def test_early_stop_summary_preserves_all_lower_bound_counters(evidence):
    report, log = evidence('cap')
    text = log.read_text().replace('instructions=4', 'instructions=>=4').replace('memory-events=2', 'memory-events=>=2')
    text += 'analysis stopped early; additional diagnostics may exist\n'
    rewrite_evidence(report, log, text, include_metadata=True, requested_limit=1)
    summary, code = helper.summarize(report, log)
    assert code == 0
    assert 'instructions=at least 4, memory-events=at least 2, diagnostics=at least 1' in summary
    assert 'native analysis=incomplete' in summary


@pytest.mark.parametrize('nullable', [False, True])
@pytest.mark.parametrize('early_stop', [False, True])
def test_legacy_clean_exit_rejects_positive_raw_count_with_all_details_omitted(evidence, nullable, early_stop):
    report, log = evidence('clean')
    data = json.loads(report.read_text())
    for item in (data['checks'][0], data['checks'][0]['kernel_results'][0]):
        item.pop('diagnostics_truncated')
        if nullable:
            item['diagnostic_summary'] = None
    report.write_text(json.dumps(data))
    raw = log.read_text().replace('diagnostics=0', 'diagnostics=3')
    if early_stop:
        raw = raw.replace('instructions=4', 'instructions=>=4').replace('memory-events=2', 'memory-events=>=2')
        raw = raw.replace('diagnostics=3', 'diagnostics=>=3')
    raw += f'omitted {"at least " if early_stop else ""}3 diagnostic(s) after limit\n'
    log.write_text(raw)
    summary, code = helper.summarize(report, log)
    assert code == 2
    assert 'INCONSISTENT OR UNSUPPORTED RAW EVIDENCE' in summary
    assert 'native clean exit contradicts positive raw diagnostic count' in summary
    assert 'Raw summary agrees' not in summary
    assert '"entry_verdict": "pass"' in summary  # Preserve the supplied status.


@pytest.mark.parametrize('nullable', [False, True])
def test_report_only_native_zero_rejects_retained_findings_without_metadata(evidence, nullable):
    report, _ = evidence('warn')
    data = json.loads(report.read_text())
    check = data['checks'][0]
    entry = check['kernel_results'][0]
    # PASS/findings is already rejected by the model; FAIL is schema-valid and
    # must still honor Waitcheck's native exit/diagnostic contract.
    data['overall_verdict'] = check['verdict'] = entry['verdict'] = 'fail'
    entry['returncode'] = 0
    if nullable:
        check['diagnostic_summary'] = entry['diagnostic_summary'] = None
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report)
    assert code == 2
    assert 'native clean exit has retained findings' in summary
    assert '"entry_verdict": "fail"' in summary


@pytest.mark.parametrize('native_exit', [2, -9])
def test_completed_fail_rejects_unsupported_native_exit_without_raw(evidence, native_exit):
    report, _ = evidence('warn')
    data = json.loads(report.read_text())
    check = data['checks'][0]
    entry = check['kernel_results'][0]
    data['overall_verdict'] = check['verdict'] = entry['verdict'] = 'fail'
    check['returncode'] = entry['returncode'] = native_exit
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report)
    assert code == 2
    assert 'native exit is missing or contradicts' in summary
    assert '"entry_verdict": "fail"' in summary
    assert f'"native_returncode": {native_exit}' in summary


@pytest.mark.parametrize('kind,native_exit', [('clean', 0), ('warn', 4)])
def test_completed_fail_keeps_supported_policy_native_exit(evidence, kind, native_exit):
    report, _ = evidence(kind)
    data = json.loads(report.read_text())
    check = data['checks'][0]
    entry = check['kernel_results'][0]
    data['overall_verdict'] = check['verdict'] = entry['verdict'] = 'fail'
    check['returncode'] = entry['returncode'] = native_exit
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report)
    assert code == 0
    assert '"entry_verdict": "fail"' in summary


@pytest.mark.parametrize('records,limit,expected_exit', [
    (3, 1, 2), (1, 1, 0), (1, 3, 0), (3, None, 0), (None, 1, 0),
])
def test_report_only_checks_entry_record_limit(evidence, records, limit, expected_exit):
    report, _ = evidence('warn')
    data = json.loads(report.read_text())
    check = data['checks'][0]
    entry = check['kernel_results'][0]
    metadata = {'reported_count': 3, 'count_meaning': 'exact',
                'emitted_records': records, 'requested_limit': limit}
    check['diagnostic_summary'] = entry['diagnostic_summary'] = metadata
    report.write_text(json.dumps(data))
    summary, code = helper.summarize(report)
    assert code == expected_exit
    if expected_exit:
        assert 'reported emitted records exceed requested limit' in summary
    else:
        assert 'Report consistency: INCONSISTENT' not in summary
    assert '"entry_verdict": "warn"' in summary
    assert '"finding_count": 1' in summary


def test_report_only_rejects_exact_aggregate_with_omitted_entry_truncation(evidence):
    report, _ = evidence('warn')
    data = json.loads(report.read_text())
    check = data['checks'][0]
    entry = check['kernel_results'][0]
    check['diagnostics_truncated'] = True
    entry.pop('diagnostics_truncated')
    metadata = {'reported_count': 1, 'count_meaning': 'exact', 'emitted_records': 1,
                'requested_limit': 64}
    check['diagnostic_summary'] = entry['diagnostic_summary'] = metadata
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='lower-bound marker contradicts exact count'):
        helper.summarize(report)
