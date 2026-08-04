"""Fail-closed parsing for RocJITsu's combined Waitcheck + ConSan hook."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .consan_coverage import CoverageDecision, parse_coverage_decision
from .execution import ProcessResult, run_argv
from .models import (
    CheckResult,
    ExecutionState,
    Finding,
    FindingSeverity,
    KernelWorklist,
    ObjectCoverage,
    Verdict,
)

_WAITCHECK_PREFIX = "rocjitsu-waitcheck:"
_CONSAN_PREFIX = "[rocjitsu-dbi-hooks] ConSan"
_WAITCHECK_HAZARD = re.compile(r"missing\s+s_wait|hazard", re.IGNORECASE)
_WAITCHECK_CONTEXT = re.compile(r"^(producer|consumer)\b", re.IGNORECASE)
_AUTO_REPLAY_DIAGNOSTIC = re.compile(r"\bauto\s+replay\s+diagnostic(?:\s|$)", re.IGNORECASE)
_KV = re.compile(r"(\w+)=(\S+)")


class ConSanMode(str, Enum):
    """The only mode whose evidence PR #7 actually knows how to parse."""

    RECORD_REPLAY = "record-replay"


@dataclass(frozen=True)
class ParsedCombinedOutput:
    waitcheck_findings: tuple[Finding, ...]
    waitcheck_error: str | None
    consan_findings: tuple[Finding, ...]
    coverage: tuple[ObjectCoverage, ...]
    coverage_decision: CoverageDecision


def _kv(line: str) -> dict[str, str]:
    return dict(_KV.findall(line))


def _int(fields: dict[str, str], key: str) -> int:
    raw = fields.get(key)
    if raw is None:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"ConSan field {key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"ConSan field {key} must be non-negative, got {value}")
    return value


def _parse_waitcheck(
    lines: list[str],
) -> tuple[tuple[Finding, ...], str | None]:
    findings: list[Finding] = []
    hazard_summaries: list[dict[str, str]] = []
    analysis_errors: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        prefixed = line.startswith(_WAITCHECK_PREFIX)
        if prefixed:
            line = line[len(_WAITCHECK_PREFIX) :].strip()
        if _WAITCHECK_CONTEXT.match(line):
            if findings:
                previous = findings[-1]
                context_index = (
                    sum(1 for key, _value in previous.metadata if key.startswith("context_")) + 1
                )
                metadata = tuple(
                    sorted(
                        (
                            *previous.metadata,
                            (f"context_{context_index}", line),
                        )
                    )
                )
                findings[-1] = Finding(
                    sanitizer=previous.sanitizer,
                    severity=previous.severity,
                    code=previous.code,
                    message=previous.message,
                    kernel_name=previous.kernel_name,
                    code_object=previous.code_object,
                    entry_offset=previous.entry_offset,
                    metadata=metadata,
                )
            continue
        if not prefixed:
            continue
        fields = _kv(line)
        if fields.get("reason") == "analysis-failed":
            analysis_errors.append(line)
            continue
        if fields.get("reason") == "wait-hazard":
            hazard_summaries.append(fields)
            continue
        if _WAITCHECK_HAZARD.search(line):
            findings.append(
                Finding(
                    sanitizer="waitcheck",
                    severity=FindingSeverity.WARNING,
                    code=fields.get("code", "wait_hazard"),
                    message=line,
                    kernel_name=fields.get("kernel"),
                    code_object=fields.get("code_object"),
                    metadata=tuple(sorted(fields.items())),
                )
            )
    deduplicated = {finding.dedupe_key: finding for finding in findings}
    resolved = [deduplicated[key] for key in sorted(deduplicated, key=repr)]
    if not resolved and hazard_summaries:
        resolved.append(
            Finding(
                sanitizer="waitcheck",
                severity=FindingSeverity.WARNING,
                code="wait_hazard_summary",
                message="Waitcheck preflight reported one or more hazards",
                metadata=tuple(sorted(hazard_summaries[0].items())),
            )
        )
    return (
        tuple(resolved),
        "; ".join(analysis_errors) if analysis_errors else None,
    )


def parse_record_replay_output(output: str) -> ParsedCombinedOutput:
    """Parse one combined-hook stream without double-counting summaries."""

    lines = output.splitlines()
    detailed_findings: list[Finding] = []
    replay_summaries: list[dict[str, str]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if _CONSAN_PREFIX not in line:
            continue
        lowered = line.lower()
        fields = _kv(line)
        if _AUTO_REPLAY_DIAGNOSTIC.search(line):
            detailed_findings.append(
                Finding(
                    sanitizer="consan",
                    severity=FindingSeverity.RACE,
                    code=fields.get("kind", "record_replay_conflict"),
                    message=line,
                    kernel_name=fields.get("kernel"),
                    code_object=fields.get("code_object"),
                    metadata=tuple(sorted(fields.items())),
                )
            )
        elif "auto replay" in lowered:
            replay_summaries.append(fields)

    deduplicated = {finding.dedupe_key: finding for finding in detailed_findings}
    findings = [deduplicated[key] for key in sorted(deduplicated, key=repr)]
    detailed_by_reader: dict[str, int] = {}
    for finding in findings:
        reader = dict(finding.metadata).get("reader", "")
        detailed_by_reader[reader] = detailed_by_reader.get(reader, 0) + 1
    for summary in replay_summaries:
        diagnostics = _int(summary, "diagnostics")
        conflict = summary.get("conflict", "false").lower() == "true"
        reader = summary.get("reader", "")
        if (conflict or diagnostics > 0) and (detailed_by_reader.get(reader, 0) < diagnostics):
            findings.append(
                Finding(
                    sanitizer="consan",
                    severity=FindingSeverity.RACE,
                    code="record_replay_conflict_summary",
                    message=("ConSan Record/Replay reported additional " "summary-only conflicts"),
                    metadata=tuple(sorted(summary.items())),
                )
            )

    decision = parse_coverage_decision(output)
    object_coverage = tuple(
        ObjectCoverage(
            object_id=(
                f"reader={record.reader}" + ("" if record.load is None else f",load={record.load}")
            ),
            applicable=record.applicable,
            analysis_complete=record.analysis_complete,
            static_complete=record.analysis_complete,
            dynamic_complete=decision.verdict.dynamic_complete,
            incomplete_code_objects=0 if record.analysis_complete else 1,
            unsupported=sum(
                value
                for key, value in record.counts
                if key.endswith(
                    (
                        "_unsupported",
                        "_resource_failed",
                        "_placement_or_lowering_failed",
                        "_expert_limit_omitted",
                    )
                )
            ),
            access=(
                f"{record.count_map['access_patched']}/" f"{record.count_map['access_supported']}"
            ),
            fields=tuple(
                sorted(
                    (
                        ("reader", str(record.reader)),
                        ("load", "" if record.load is None else str(record.load)),
                        ("flavor", record.flavor),
                        ("engine", record.engine),
                        *((key, str(value)) for key, value in record.counts),
                    )
                )
            ),
        )
        for record in decision.coverage
    )
    waitcheck_findings, waitcheck_error = _parse_waitcheck(lines)
    return ParsedCombinedOutput(
        waitcheck_findings=waitcheck_findings,
        waitcheck_error=waitcheck_error,
        consan_findings=tuple(findings),
        coverage=object_coverage,
        coverage_decision=decision,
    )


def _error_result(
    sanitizer: str,
    *,
    state: ExecutionState,
    reason: str,
    process: ProcessResult,
    findings: tuple[Finding, ...],
    coverage: tuple[ObjectCoverage, ...] = (),
) -> CheckResult:
    return CheckResult(
        sanitizer=sanitizer,
        state=state,
        verdict=(
            Verdict.FAIL
            if any(finding.severity is FindingSeverity.RACE for finding in findings)
            else Verdict.ERROR
        ),
        reason=reason,
        returncode=process.returncode,
        findings=findings,
        coverage=coverage,
    )


def evaluate_record_replay(
    process: ProcessResult,
    *,
    strict: bool = False,
) -> tuple[CheckResult, CheckResult]:
    """Evaluate a future allowlisted combined-hook run.

    The first result is the hook's mandatory Waitcheck stage; the second is
    ConSan. This evaluator is usable before launch integration and guarantees
    that unhealthy instrumentation never becomes PASS.
    """

    if process.timed_out:
        return (
            _error_result(
                "waitcheck",
                state=ExecutionState.TIMED_OUT,
                reason="combined_hook_timeout",
                process=process,
                findings=(),
            ),
            _error_result(
                "consan",
                state=ExecutionState.TIMED_OUT,
                reason="combined_hook_timeout",
                process=process,
                findings=(),
            ),
        )
    if process.launch_error is not None:
        reason = f"combined_hook_launch_error: {process.launch_error}"
        return (
            _error_result(
                "waitcheck",
                state=ExecutionState.ERROR,
                reason=reason,
                process=process,
                findings=(),
            ),
            _error_result(
                "consan",
                state=ExecutionState.ERROR,
                reason=reason,
                process=process,
                findings=(),
            ),
        )
    if process.returncode != 0:
        reason = (
            "consan_strict_load_rejection"
            if process.returncode == 92
            else f"combined_hook_exit_{process.returncode}"
        )
        return (
            _error_result(
                "waitcheck",
                state=ExecutionState.ERROR,
                reason=reason,
                process=process,
                findings=(),
            ),
            _error_result(
                "consan",
                state=ExecutionState.ERROR,
                reason=reason,
                process=process,
                findings=(),
            ),
        )
    try:
        parsed = parse_record_replay_output(f"{process.stdout}\n{process.stderr}")
    except ValueError as exc:
        reason = f"consan_output_parse_error: {exc}"
        return (
            _error_result(
                "waitcheck",
                state=ExecutionState.ERROR,
                reason=reason,
                process=process,
                findings=(),
            ),
            _error_result(
                "consan",
                state=ExecutionState.ERROR,
                reason=reason,
                process=process,
                findings=(),
            ),
        )
    waitcheck = (
        _error_result(
            "waitcheck",
            state=ExecutionState.ERROR,
            reason=f"waitcheck_analysis_failed: {parsed.waitcheck_error}",
            process=process,
            findings=parsed.waitcheck_findings,
        )
        if parsed.waitcheck_error is not None
        else CheckResult(
            sanitizer="waitcheck",
            state=ExecutionState.RAN,
            verdict=(Verdict.WARN if parsed.waitcheck_findings else Verdict.PASS),
            returncode=process.returncode,
            findings=parsed.waitcheck_findings,
        )
    )
    if not parsed.coverage_decision.accepted:
        consan = _error_result(
            "consan",
            state=ExecutionState.ERROR,
            reason=("consan_coverage_incomplete: " + "; ".join(parsed.coverage_decision.reasons)),
            process=process,
            findings=parsed.consan_findings,
            coverage=parsed.coverage,
        )
        return waitcheck, consan

    _ = strict  # Strict policy failures are surfaced by exit code 92.

    consan = CheckResult(
        sanitizer="consan",
        state=ExecutionState.RAN,
        verdict=(Verdict.FAIL if parsed.consan_findings else Verdict.PASS),
        returncode=process.returncode,
        findings=parsed.consan_findings,
        coverage=parsed.coverage,
    )
    return waitcheck, consan


def scoped_consan_not_checked(worklist: KernelWorklist) -> CheckResult:
    """Fail closed when no targeted repro command is provisioned."""

    return CheckResult(
        sanitizer="consan",
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason=(
            "consan_command_not_provisioned: targeted repro required; refused to "
            f"wrap the whole application for {len(worklist.kernels)} selected kernels"
        ),
    )


def resolve_consan_hook(explicit: Path | None = None) -> Path | None:
    build_root = os.environ.get("ROCJITSU_BUILD", "").strip()
    if explicit is not None:
        return explicit
    if not build_root:
        return None
    candidate = (
        Path(build_root)
        / "lib"
        / "rocjitsu"
        / "src"
        / "rocjitsu"
        / "hooks"
        / "librocjitsu_dbi_hooks.so"
    )
    return candidate if candidate.is_file() else None


def run_consan(
    worklist: KernelWorklist,
    *,
    command: Path,
    hook_lib: Path | None = None,
    output_dir: Path,
    consan_log: bool = True,
    timeout_seconds: float = 900.0,
    strict: bool = False,
) -> CheckResult:
    """Run a targeted repro under the RocJITsu DBI hook."""

    resolved_hook = resolve_consan_hook(hook_lib)
    if resolved_hook is None or not resolved_hook.is_file():
        return CheckResult(
            sanitizer="consan",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason="consan_hook_not_found",
        )
    if not command.is_file():
        return CheckResult(
            sanitizer="consan",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason="consan_command_not_found",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HSA_TOOLS_LIB"] = str(resolved_hook)
    if consan_log:
        env["RJ_CONSAN_LOG"] = "1"
    process = run_argv(
        (str(command),),
        timeout_seconds=timeout_seconds,
        env=env,
    )
    ( _waitcheck, consan) = evaluate_record_replay(process, strict=strict)
    log_path = output_dir / "consan.log"
    log_path.write_text(f"{process.stdout}\n{process.stderr}", encoding="utf-8")
    return consan
