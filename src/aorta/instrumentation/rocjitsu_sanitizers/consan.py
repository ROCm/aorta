"""Fail-closed parsing for RocJITsu's combined Waitcheck + ConSan hook."""

from __future__ import annotations

import hashlib
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
    KernelCheckResult,
    KernelIdentity,
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
    """The only ConSan mode whose evidence this module can parse (Record/Replay)."""

    RECORD_REPLAY = "record-replay"


# The strict coverage cross-check (consan_coverage.parse_coverage_decision)
# requires per-site ``coverage_site`` records to reconcile against each
# aggregate ``coverage`` record's ``*_discovered`` counts. The hook only emits
# those per-site lines at its debug level (kLogDebug=3). A boolean-truthy
# ``RJ_CONSAN_LOG=1`` maps to kLogInfo, which prints the aggregate line but no
# per-site records -- so every kind would report as un-itemized and the
# cross-check would fail closed on an otherwise clean run. Request the debug
# level explicitly so the evidence is complete.
_CONSAN_LOG_DEBUG_LEVEL = "3"


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
            static_complete=decision.verdict.static_complete,
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
    """Locate the RocJITsu DBI hook, preferring an explicit override.

    Two provisioning layouts are supported. ``ROCJITSU_PREBUILT`` points at an
    unpacked prebuilt sanitizer bundle (published by ROCm/rocm-systems), whose
    flattened tree keeps the hook at ``lib/librocjitsu_dbi_hooks.so``.
    ``ROCJITSU_BUILD`` points at a raw CMake build tree, where the hook lives
    under ``lib/rocjitsu/src/rocjitsu/hooks/``. The prebuilt bundle wins when
    both are set.
    """

    if explicit is not None:
        return explicit
    prebuilt = os.environ.get("ROCJITSU_PREBUILT", "").strip()
    if prebuilt:
        candidate = Path(prebuilt) / "lib" / "librocjitsu_dbi_hooks.so"
        if candidate.is_file():
            return candidate
    build_root = os.environ.get("ROCJITSU_BUILD", "").strip()
    if build_root:
        candidate = (
            Path(build_root)
            / "lib"
            / "rocjitsu"
            / "src"
            / "rocjitsu"
            / "hooks"
            / "librocjitsu_dbi_hooks.so"
        )
        if candidate.is_file():
            return candidate
    return None


WAITCHECK_PREFLIGHT = "waitcheck_preflight"


@dataclass(frozen=True)
class ConSanRunResult:
    """A ConSan run and its mandatory combined-hook Waitcheck preflight.

    Both are surfaced as checks so an unhealthy preflight (WARN/ERROR) can never
    be masked by a clean ConSan verdict: the report's overall verdict is the max
    over all checks, so preflight evidence fails the run closed at report scope.
    """

    consan: CheckResult
    waitcheck_preflight: CheckResult


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_not_run(reason: str) -> CheckResult:
    return CheckResult(
        sanitizer=WAITCHECK_PREFLIGHT,
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason=reason,
    )


def _combined_not_checked(reason: str) -> ConSanRunResult:
    return ConSanRunResult(
        consan=CheckResult(
            sanitizer="consan",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason=reason,
        ),
        waitcheck_preflight=_preflight_not_run("consan_preflight_not_executed"),
    )


def _relabel(check: CheckResult, sanitizer: str) -> CheckResult:
    return CheckResult(
        sanitizer=sanitizer,
        state=check.state,
        verdict=check.verdict,
        reason=check.reason,
        returncode=check.returncode,
        findings=check.findings,
        coverage=check.coverage,
    )


def run_consan(
    worklist: KernelWorklist,
    *,
    command: Path,
    hook_lib: Path | None = None,
    output_dir: Path,
    consan_log: bool = True,
    timeout_seconds: float = 900.0,
    strict: bool = False,
    target: KernelIdentity | None = None,
) -> ConSanRunResult:
    """Run one targeted repro under the RocJITsu DBI hook.

    ConSan is only meaningful for a single, explicitly-selected kernel identity,
    so an empty or multi-kernel worklist fails closed rather than returning a
    vacuous PASS. When ``target`` is given it must match the selected identity,
    guarding against a recipe that names one repro while executing another. The
    returned result records the executed command/hook/identity digests and a
    per-kernel result so a PASS is always attributed to a real selection.
    """

    if len(worklist.kernels) != 1:
        return _combined_not_checked("consan_requires_one_targeted_repro")
    selected = worklist.kernels[0].identity
    if target is not None and target.stable_key != selected.stable_key:
        return _combined_not_checked("consan_target_does_not_match_worklist")

    resolved_hook = resolve_consan_hook(hook_lib)
    if resolved_hook is None or not resolved_hook.is_file():
        return _combined_not_checked("consan_hook_not_found")
    if not command.is_file():
        return _combined_not_checked("consan_command_not_found")

    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HSA_TOOLS_LIB"] = str(resolved_hook)
    # Pin the sanitizer contract so hostile inherited settings cannot weaken it:
    # no auto-registration, record/replay mode, and the requested policy.
    env["HSA_TOOLS_DISABLE_REGISTER"] = "1"
    env["RJ_CONSAN_MODE"] = ConSanMode.RECORD_REPLAY.value
    env["RJ_CONSAN_POLICY"] = "strict" if strict else "default"
    if consan_log:
        env["RJ_CONSAN_LOG"] = _CONSAN_LOG_DEBUG_LEVEL
    else:
        # Scrub any inherited RJ_CONSAN_LOG so disabling logging is deterministic
        # regardless of the parent environment (a stray value would otherwise
        # force hook logging and change coverage parsing strictness).
        env.pop("RJ_CONSAN_LOG", None)
    process = run_argv(
        (str(command),),
        timeout_seconds=timeout_seconds,
        env=env,
    )
    preflight, consan = evaluate_record_replay(process, strict=strict)
    log_path = output_dir / "consan.log"
    log_path.write_text(f"{process.stdout}\n{process.stderr}", encoding="utf-8")

    digests = {
        "command": str(command),
        "command_sha256": _sha256_file(command),
        "hook": str(resolved_hook),
        "hook_sha256": _sha256_file(resolved_hook),
        "selected_kernel": selected.name,
        "selected_identity_sha256": hashlib.sha256(
            selected.stable_key.encode("utf-8")
        ).hexdigest(),
    }
    kernel_result = KernelCheckResult(
        identity=selected,
        state=consan.state,
        verdict=consan.verdict,
        findings=consan.findings,
        reason=consan.reason,
        returncode=consan.returncode,
    )
    attributed_consan = CheckResult(
        sanitizer="consan",
        state=consan.state,
        verdict=consan.verdict,
        reason=consan.reason,
        returncode=consan.returncode,
        findings=consan.findings,
        kernel_results=(kernel_result,),
        coverage=consan.coverage,
        backend=tuple(sorted(digests.items())),
    )
    return ConSanRunResult(
        consan=attributed_consan,
        waitcheck_preflight=_relabel(preflight, WAITCHECK_PREFLIGHT),
    )
