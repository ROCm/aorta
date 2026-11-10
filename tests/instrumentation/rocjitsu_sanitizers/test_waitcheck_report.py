"""Offline reader tests: arbitrary synthetic entries; no native execution."""
import importlib.util
import json
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.models import (
    CheckResult, ExecutionState, KernelCheckResult, KernelIdentity, KernelObservation,
    KernelWorklist, SelectionRequirement, Verdict)
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
