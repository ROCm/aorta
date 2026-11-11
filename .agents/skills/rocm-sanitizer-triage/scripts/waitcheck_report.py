#!/usr/bin/env python3
"""Summarize one saved AORTA Waitcheck entry offline; never execute a program.

Requires the AORTA Python environment. Exit 0: summary written (limits may remain).
Exit 2: summary written with report contradictions or inconsistent/unsupported raw evidence.
Exit 1: invalid report/scope/input/output, no summary written.
Argument usage errors also exit 2, before writing any summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

LIMIT = 4 * 1024 * 1024
HEADER = re.compile(
    r"^(?:rj_waitcheck: )?(?P<object>.+):(?P<target>gfx[0-9a-f]+)\[(?P<index>[0-9]+)\]:"
    r"kernel=\.text\+0x(?P<entry>[0-9a-f]+): instructions=(?P<instruction_cap>>=)?(?P<instructions>[0-9]+) "
    r"memory-events=(?P<event_cap>>=)?(?P<events>[0-9]+) diagnostics=(?P<cap>>=)?(?P<count>[0-9]+)$")
OMITTED = re.compile(r"omitted (?P<floor>at least )?(?P<count>[0-9]+) diagnostic\(s\) after limit")
EARLY_STOP = "analysis stopped early; additional diagnostics may exist"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded(path):
    require(path.is_file() and not path.is_symlink(), f"input must be a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    require(len(data) <= LIMIT, f"input exceeds {LIMIT} bytes: {path}")
    return data


def unknown_fields(raw, known, location="report"):
    if isinstance(raw, dict) and isinstance(known, dict):
        extra = raw.keys() - known.keys()
        require(not extra, f"unsupported fields at {location}: {sorted(extra)}")
        for key in raw:
            unknown_fields(raw[key], known[key], f"{location}.{key}")
    elif isinstance(raw, list) and isinstance(known, list):
        for index, (left, right) in enumerate(zip(raw, known)):
            unknown_fields(left, right, f"{location}[{index}]")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def raw_comparison(log, identity, result):
    """Compare native records before deduplication, keeping unknown semantics."""
    from aorta.instrumentation.rocjitsu_sanitizers.waitcheck import parse_waitcheck_text
    parsed = parse_waitcheck_text(log, expected=identity)
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    headers = [(line, HEADER.fullmatch(line)) for line in lines if HEADER.fullmatch(line)]
    require(len(headers) == 1, "need exactly one supported summary with an explicit entry marker")
    header_line, match = headers[0]
    require((match['object'], match['target'], int(match['index']), int(match['entry'], 16)) ==
            (identity.code_object, identity.target, identity.code_object_index, identity.entry_offset),
            "raw object/target/image/entry differs from report")
    evidence = parsed.diagnostic_summary
    require(evidence is not None, "parser does not expose diagnostic count evidence")
    require(evidence.emitted_records is not None, "raw diagnostic-record count is unavailable")
    require(int(match['count']) >= evidence.emitted_records, "raw count is smaller than emitted diagnostic records")
    require(parsed.findings == result.findings, "raw findings differ from reported per-entry findings")
    require(parsed.diagnostics_truncated == result.diagnostics_truncated, "raw truncation differs from report")
    declared = result.diagnostic_summary
    if declared is not None:
        for key in ('reported_count', 'emitted_records'):
            value = getattr(declared, key)
            require(value is None or value == getattr(evidence, key), f"raw {key} differs from report metadata")
        for key in ('count_meaning', 'details', 'native_analysis'):
            value = getattr(declared, key).value
            require(value == 'unknown' or value == getattr(evidence, key).value,
                    f"raw {key} does not support report metadata")
        require(declared.requested_limit is None or evidence.emitted_records <= declared.requested_limit,
                "raw emitted records exceed the reported requested limit")
    allowed = {header_line, EARLY_STOP}
    allowed.update(line for line in lines if OMITTED.fullmatch(line))
    diagnostic_prefix = re.compile(re.escape(f"{identity.code_object}:{identity.target}[{identity.code_object_index}]:")
                                   + r"\.text\+0x[0-9a-f]+: ")
    for finding in parsed.findings:
        require(diagnostic_prefix.match(finding.message), "diagnostic object/target/image prefix differs or is unsupported")
        allowed.add(finding.message)
        allowed.update(value for key, value in finding.metadata if key.startswith("context_"))
    require(all(line in allowed for line in lines), "raw contains unrecognized lines; completion/coverage is uncertain")
    require(result.returncode == (4 if parsed.findings else 0), "native exit contradicts raw finding result")
    require(result.verdict.value == ("warn" if parsed.findings else "pass"), "reported verdict contradicts raw findings")
    description = (f"Raw summary agrees with the reported entry and findings: instructions={match['instructions']}, "
                   f"memory-events={match['events']}, diagnostics={'at least ' if match['cap'] else ''}{match['count']}; "
                   f"emitted records={evidence.emitted_records}, retained unique findings={len(parsed.findings)}. "
                   f"Count meaning={evidence.count_meaning.value}; details={evidence.details.value}; "
                   f"native analysis={evidence.native_analysis.value}.")
    signals = [line for line in lines if OMITTED.fullmatch(line) or line == EARLY_STOP]
    if signals:
        description += " Supported raw signal: " + "; ".join(signals) + "."
    gap = int(match['count']) > evidence.emitted_records and evidence.details.value == 'unknown'
    if gap:
        description = ("INCONSISTENT OR UNSUPPORTED RAW EVIDENCE: compatibility gap: "
                       f"reported diagnostics={match['count']}, emitted records={evidence.emitted_records}, "
                       f"retained unique findings={len(parsed.findings)}. "
                       "No supported explicit omission signal explains the count/detail difference; "
                       f"count meaning={evidence.count_meaning.value}, detail completeness={evidence.details.value}, "
                       f"native analysis={evidence.native_analysis.value}. "
                       "This does not establish corruption or a checker crash.")
    return description, evidence, gap


def summarize(report_path, log_path=None):
    from aorta.instrumentation.rocjitsu_sanitizers.report import read_report
    report_bytes = bounded(report_path)
    raw = json.loads(report_bytes.decode("utf-8-sig"), object_pairs_hook=unique_object)
    # The official path-based reader gets a bounded captured copy, not a second
    # unbounded read of a source that could grow while it is being inspected.
    with tempfile.TemporaryDirectory(prefix="waitcheck-report-") as directory:
        snapshot = Path(directory) / "report.json"
        snapshot.write_bytes(report_bytes.removeprefix(b"\xef\xbb\xbf"))
        report = read_report(snapshot)
    unknown_fields(raw, report.to_dict())
    require(len(report.checks) == 1 and len(report.worklist.kernels) == 1,
            "only one check and one selected kernel are supported")
    check = report.checks[0]
    identity = report.worklist.kernels[0].identity
    require(check.sanitizer == "waitcheck" and not check.coverage, "only single-entry Waitcheck reports are supported")
    require(identity.exact and identity.code_object_index is not None, "explicit object/hash/image/entry identity required")
    require(len(check.kernel_results) <= 1, "multiple per-kernel results are unsupported")
    result = check.kernel_results[0] if check.kernel_results else None
    if result:
        require(result.identity == identity, "worklist and result identities disagree")
        require(result.findings == check.findings, "check and per-entry findings disagree")
        require(result.diagnostics_truncated == check.diagnostics_truncated, "check and entry truncation disagree")
    require(check.state.value != "ran" or result is not None, "ran check has no per-entry result")
    issues = []
    if result:
        if check.state.value == "ran" and (result.state != check.state or result.verdict != check.verdict):
            issues.append("reported single-entry aggregate state/verdict differs from its result")
        if check.returncode is not None and check.returncode != result.returncode:
            issues.append("reported aggregate returncode differs from native entry returncode")
        if result.state.value == "ran":
            expected_exit = {"pass": 0, "warn": 4}.get(result.verdict.value)
            if result.returncode is None or (expected_exit is not None and result.returncode != expected_exit):
                issues.append("native exit is missing or contradicts the reported completed Waitcheck verdict")
            metadata = result.diagnostic_summary
            if metadata is not None:
                count, records = metadata.reported_count, metadata.emitted_records
                if count is not None and records is not None and count < records:
                    issues.append("reported diagnostic count is smaller than emitted records")
                if count is not None and count < len(result.findings):
                    issues.append("reported diagnostic count is smaller than retained findings")
                if result.returncode == 0 and (count is not None and count > 0 or records is not None and records > 0):
                    issues.append("native clean exit contradicts reported diagnostic count/records")
                if result.returncode == 4 and (count == 0 or records == 0):
                    issues.append("native hazard exit contradicts zero reported count/records")
                if records is not None and records > 0 and not result.findings:
                    issues.append("reported emitted records have no retained findings")
    notes = ["Identity, object digest and backend details below are reported metadata, not independently verified provenance.",
             "The intended user artifact, current on-disk bytes, tool/version, recipe, command and authorization were not verified.",
             "AORTA CLI exit is not recorded in this report; this reader's exit is not the scanner or AORTA exit.",
             "Even matching evidence covers only supported checks on this selected compiled entry; it does not establish general correctness or qualify the route."]
    raw_check = raw['checks'][0]
    if 'diagnostics_truncated' not in raw_check or (result and 'diagnostics_truncated' not in raw_check['kernel_results'][0]):
        notes.append("Legacy report omits a truncation field: its default false is not independent proof of a complete diagnostic count.")
    if check.diagnostics_truncated:
        notes.append("The historical >= marker records a lower bound on native diagnostics; it does not count distinct bugs or establish native completion.")
    notes.append("AORTA execution_status complete describes adapter execution, not exhaustive native analysis. "
                 "diagnostics_truncated false, exit 0/4 and reaching a cap do not establish complete details.")
    if result is None or result.diagnostic_summary is None:
        notes.append("Legacy report has no diagnostic_summary metadata: reported count, requested limit, detail completeness "
                     "and native analysis completion remain unknown unless supported raw evidence supplies them.")
    else:
        notes.append("Diagnostic metadata in the report is a producing adapter's claim; raw comparison and tool provenance are separate evidence.")
    completed = check.state.value == "ran" and result is not None and result.state.value == "ran"
    if not completed:
        notes.append("No completed single-entry check is established. Preserve the stated failure/dry-run reason; do not interpret partial output as a pass.")
    raw_status, code, log_bytes, raw_summary = "No raw log supplied.", 2 if issues else 0, None, None
    if issues:
        notes.append("Reported statuses conflict; preserve these fields and resolve the contradiction before accepting a completed result.")
    if log_path is None:
        notes.append("Raw findings/entry evidence was not compared." if completed else
                     "A raw log may legitimately be absent for a dry run or a pre-launch refusal.")
    else:
        log_bytes = bounded(log_path)
        log = log_bytes.decode("utf-8-sig")
        if completed:
            try:
                raw_status, raw_summary, gap = raw_comparison(log, identity, result)
                if gap:
                    code = 2
                    notes.append("Preserve the warning and both counts. Do not silently rescan, raise limits or change scope. "
                                 "Use existing detailed evidence, or a separately authorized higher-limit run only if needed.")
            except (ValueError, TypeError) as exc:
                raw_status, code = f"INCONSISTENT OR UNSUPPORTED RAW EVIDENCE: {exc}", 2
                notes.append("Do not accept the reported completed result until this evidence gap is resolved.")
        else:
            raw_status = "Raw failure/partial log retained as evidence; not parsed as a completed scan."
    record = {
        "overall_verdict": report.overall_verdict.value, "execution_status": report.execution_status.value,
        "identity": identity.to_dict(), "check_state": check.state.value, "check_verdict": check.verdict.value,
        "check_reason": check.reason, "check_returncode": check.returncode,
        "entry_state": result.state.value if result else None, "entry_verdict": result.verdict.value if result else None,
        "entry_reason": result.reason if result else None, "native_returncode": result.returncode if result else None,
        "finding_count": len(check.findings), "diagnostics_truncated": check.diagnostics_truncated,
        "entry_diagnostic_summary_reported": result.diagnostic_summary.to_dict() if result and result.diagnostic_summary else None,
        "check_diagnostic_summary_reported": check.diagnostic_summary.to_dict() if check.diagnostic_summary else None,
        "diagnostic_summary_raw": raw_summary.to_dict() if raw_summary else None,
        "backend_reported": dict(check.backend), "findings": [item.to_dict() for item in check.findings],
    }
    inputs = {"report": {"path": str(report_path), "sha256": hashlib.sha256(report_bytes).hexdigest()}}
    if log_bytes is not None:
        inputs['log'] = {"path": str(log_path), "sha256": hashlib.sha256(log_bytes).hexdigest()}
    lines = ["# Saved Waitcheck evidence summary", "", raw_status,
             "", "Report consistency: " + ("INCONSISTENT: " + "; ".join(issues) if issues else "no supported-field contradiction identified."),
             "", "Reported result:",
             "```json", json.dumps(record, indent=2, ensure_ascii=True), "```", "", "Limits:", ""]
    lines += [f"- {note}" for note in notes]
    lines += ["", "Inputs read locally:", "```json", json.dumps(inputs, indent=2, ensure_ascii=True), "```", ""]
    if log_bytes is not None and not completed:
        lines += ["Failure/partial raw excerpt (JSON-escaped; at most 4096 characters):", "```json",
                  json.dumps(log[:4096], ensure_ascii=True), "```", ""]
    return "\n".join(lines), code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists() and not args.output.is_symlink(), "output already exists")
        summary, code = summarize(args.report, args.log)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(summary)
        print(f"Saved {args.output}; reader exit {code}, not a sanitizer verdict.")
        return code
    except (OSError, ValueError, TypeError, KeyError, RecursionError, ImportError) as exc:
        print(f"Cannot summarize Waitcheck evidence: {exc}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
