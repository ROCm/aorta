"""Static Waitcheck backend for exact-entry and whole-code-object-scan identities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .execution import ProcessResult, run_argv
from .models import (
    CheckResult,
    CountMeaning,
    DiagnosticSummary,
    EvidenceStatus,
    ExecutionState,
    Finding,
    FindingSeverity,
    KernelCheckResult,
    KernelIdentity,
    KernelWorklist,
    Verdict,
    aggregate_diagnostic_summaries,
)

WAITCHECK_CLEAN_EXIT = 0
WAITCHECK_HAZARD_EXIT = 4
WAITCHECK_JSON_SCHEMA = "rj-waitcheck-diagnostic-v1"
_HEADER = re.compile(
    r"^(?:rj_waitcheck: )?(?P<object>.+):(?P<target>gfx[0-9a-f]+)\[(?P<index>[0-9]+)\]:"
    # An exact-entry run prefixes the summary with a kernel marker such as
    # ``kernel=.text+0x120:``; accept and capture it so a real exact-entry run is
    # not misread as a missing analysis summary.
    r"(?:\s*kernel=\.text\+0x(?P<entry>[0-9a-f]+):)?"
    r"\s+instructions=(?P<instruction_bound>>=)?(?P<instructions>[0-9]+) "
    r"memory-events=(?P<event_bound>>=)?(?P<events>[0-9]+) "
    r"diagnostics=(?P<truncated>>=)?(?P<diagnostics>[0-9]+)$"
)
_DIAGNOSTIC = re.compile(r"\bmissing\s+s_wait|\bhazard\b", re.IGNORECASE)
_CONTEXT = re.compile(r"^(producer|consumer)\b", re.IGNORECASE)
_OMITTED = re.compile(r"^omitted (?P<bound>at least )?(?P<count>[0-9]+) diagnostic\(s\) after limit$")
_STOPPED = "analysis stopped early; additional diagnostics may exist"


@dataclass(frozen=True)
class ParsedWaitcheckOutput:
    """Unique findings, historical >= marker, and optional native evidence.

    False truncation and successful parsing never prove exhaustive analysis.
    The default preserves existing two-argument constructors.
    """

    findings: tuple[Finding, ...]
    diagnostics_truncated: bool
    diagnostic_summary: DiagnosticSummary | None = None


class ProcessExecutor(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_waitcheck(explicit: Path | None = None) -> Path | None:
    """Resolve an explicit binary or a binary already available on PATH."""

    if explicit is not None:
        return explicit
    discovered = shutil.which("rj_waitcheck")
    return None if discovered is None else Path(discovered)


def _parse_non_negative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    text = str(value).strip()
    base = 16 if text.lower().startswith("0x") else 10
    try:
        parsed = int(text, base)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _optional_text(data: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


def _json_object(value: object, *, line_number: int) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Waitcheck JSONL line {line_number} must be an object")
    return value


def parse_waitcheck_jsonl(
    path: Path,
    *,
    expected: KernelIdentity,
) -> tuple[Finding, ...]:
    """Parse structured diagnostics and reject identity mismatches.

    The JSONL stream carries no analysis summary, so — unlike
    ``parse_waitcheck_text`` — it cannot tell a capped scan from a complete one;
    a corpus caller that needs that signal has to read the summary line too.
    """

    if not path.exists():
        return ()
    findings: list[Finding] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Waitcheck JSONL at line {line_number}: {exc}") from exc
        item = _json_object(raw, line_number=line_number)
        if item.get("schema") != WAITCHECK_JSON_SCHEMA:
            raise ValueError(
                f"Waitcheck JSONL line {line_number} has unsupported schema "
                f"{item.get('schema')!r}"
            )
        target = _optional_text(item, "target")
        if target is not None and target != expected.target:
            raise ValueError(
                f"Waitcheck target {target!r} does not match requested " f"{expected.target!r}"
            )
        kernel_name = _optional_text(item, "kernel_name", "kernel") or expected.name
        if kernel_name != expected.name:
            raise ValueError(
                f"Waitcheck kernel {kernel_name!r} does not match exact requested "
                f"kernel {expected.name!r}"
            )
        code_object_index = _parse_non_negative_int(
            item.get("code_object_index"),
            field="code_object_index",
        )
        expected_index = expected.code_object_index or 0
        # A missing/null index means a single-image code object (index 0), which
        # must not be rejected when the expected identity is also single-image.
        if (code_object_index or 0) != expected_index:
            raise ValueError(
                f"Waitcheck code-object index {code_object_index!r} does not match "
                f"requested {expected_index!r}"
            )
        entry_offset = _parse_non_negative_int(
            item.get("kernel_entry"),
            field="kernel_entry",
        )
        if (
            entry_offset is not None
            and expected.entry_offset is not None
            and entry_offset != expected.entry_offset
        ):
            raise ValueError(
                f"Waitcheck entry 0x{entry_offset:x} does not match requested "
                f"0x{expected.entry_offset:x}"
            )
        counter = _optional_text(item, "counter") or "unknown"
        access = _optional_text(item, "access") or "unknown"
        code = f"missing_{counter}_{access}"
        message = _optional_text(item, "message") or raw_line
        metadata: list[tuple[str, str]] = []
        for key in (
            "counter",
            "access",
            "required_count",
            "instruction",
            "producer_instruction",
            "section_offset_hex",
            "producer_section_offset_hex",
        ):
            value = _optional_text(item, key)
            if value is not None:
                metadata.append((key, value))
        findings.append(
            Finding(
                sanitizer="waitcheck",
                severity=FindingSeverity.WARNING,
                code=code,
                message=message,
                # Attribute to a kernel name only for exact-entry identities; a
                # whole-code-object scan reports at object scope (no kernel name).
                kernel_name=(kernel_name if expected.entry_offset is not None else None),
                code_object=expected.code_object,
                entry_offset=expected.entry_offset,
                metadata=tuple(sorted(metadata)),
            )
        )
    deduplicated = {finding.dedupe_key: finding for finding in findings}
    return tuple(deduplicated[key] for key in sorted(deduplicated, key=repr))


def parse_waitcheck_text(
    output: str,
    *,
    expected: KernelIdentity,
) -> ParsedWaitcheckOutput:
    """Preserve counts and raw records before deduplication, failing on conflicts.

    Bare counts have unknown semantics for an unidentified binary. Explicit
    omission/stop messages follow rocm-systems d03c6fe; see the reporting note.
    """

    findings: list[Finding] = []
    summary = None
    truncated = False
    signal = None
    omitted = None
    early = False
    expected_index = expected.code_object_index or 0
    for raw_line in output.splitlines():
        line = raw_line.strip()
        header = _HEADER.search(line)
        if header is not None:
            if summary is not None:
                raise ValueError("Waitcheck output contains multiple/conflicting analysis summaries")
            summary = header
            truncated = header.group("truncated") is not None
            bounds = (bool(header.group("instruction_bound")), bool(header.group("event_bound")))
            if any(bounds) and not (all(bounds) and truncated):
                raise ValueError("Waitcheck instruction/event bounds contradict diagnostic bound")
            if header.group("object") != expected.code_object:
                raise ValueError("Waitcheck object does not match requested code object")
            if header.group("target") != expected.target:
                raise ValueError(
                    f"Waitcheck target {header.group('target')!r} does not match "
                    f"requested {expected.target!r}"
                )
            if int(header.group("index")) != expected_index:
                raise ValueError(
                    f"Waitcheck code-object index {header.group('index')} does not "
                    f"match requested {expected_index}"
                )
            entry_group = header.group("entry")
            if (
                entry_group is not None
                and expected.entry_offset is not None
                and int(entry_group, 16) != expected.entry_offset
            ):
                raise ValueError(
                    f"Waitcheck entry 0x{int(entry_group, 16):x} does not match "
                    f"requested 0x{expected.entry_offset:x}"
                )
            continue
        if "instructions=" in line or "diagnostics=" in line:
            raise ValueError("Waitcheck output contains a malformed analysis summary")
        omission = _OMITTED.fullmatch(line)
        if omission is not None or line == _STOPPED:
            if summary is None or signal is not None:
                raise ValueError("Waitcheck completion signal missing summary or conflicting")
            signal = line
            early = line == _STOPPED or omission.group("bound") is not None
            omitted = int(omission.group("count")) if omission is not None else None
            if early != truncated:
                raise ValueError("Waitcheck completion signal contradicts count bound")
            if any(bool(summary.group(key)) != early for key in ("instruction_bound", "event_bound")):
                raise ValueError("Waitcheck explicit signal contradicts instruction/event bounds")
            continue
        context = _CONTEXT.match(line)
        if context is not None:
            if not findings or signal is not None:
                raise ValueError("Waitcheck context without an active diagnostic")
            if findings:
                previous = findings[-1]
                context_index = (
                    sum(1 for key, _value in previous.metadata if key.startswith("context_")) + 1
                )
                findings[-1] = Finding(
                    sanitizer=previous.sanitizer,
                    severity=previous.severity,
                    code=previous.code,
                    message=previous.message,
                    kernel_name=previous.kernel_name,
                    code_object=previous.code_object,
                    entry_offset=previous.entry_offset,
                    metadata=tuple(
                        sorted(
                            (
                                *previous.metadata,
                                (f"context_{context_index}", line),
                            )
                        )
                    ),
                )
            continue
        if _DIAGNOSTIC.search(line):
            prefix = re.escape(f"{expected.code_object}:{expected.target}[{expected_index}]:")
            if not re.match(prefix + r"\.text\+0x[0-9a-f]+: ", line):
                raise ValueError("Waitcheck diagnostic object/target/image prefix does not match requested identity")
            if summary is None or signal is not None:
                raise ValueError("Waitcheck diagnostic outside summary/detail sequence")
            findings.append(
                Finding(
                    sanitizer="waitcheck",
                    severity=FindingSeverity.WARNING,
                    code="wait_hazard",
                    message=line,
                    # Only attribute a finding to a kernel name for an exact-entry
                    # run; a whole-code-object scan must not fabricate per-kernel
                    # attribution to a synthetic name.
                    kernel_name=(expected.name if expected.entry_offset is not None else None),
                    code_object=expected.code_object,
                    entry_offset=expected.entry_offset,
                )
            )
            continue
        if line:
            if summary is None:
                raise ValueError("Waitcheck output did not contain an analysis summary; unrecognized line: " + line)
            raise ValueError(f"Waitcheck output contains an unrecognized line: {line}")
    if summary is None:
        raise ValueError("Waitcheck output did not contain an analysis summary")
    count = int(summary.group("diagnostics"))
    records = len(findings)
    if records > count:
        raise ValueError("Waitcheck summary count is smaller than emitted diagnostic records")
    if omitted is not None and (omitted < 1 or count - records != omitted):
        raise ValueError("Waitcheck omitted count contradicts summary and emitted records")
    if signal == _STOPPED and records != count:
        raise ValueError("Waitcheck early-stop signal does not account for omitted records")
    meaning = CountMeaning.LOWER_BOUND if truncated else CountMeaning.UNKNOWN
    details = EvidenceStatus.UNKNOWN
    native = EvidenceStatus.UNKNOWN
    if signal is not None:
        meaning = CountMeaning.LOWER_BOUND if early else CountMeaning.EXACT
        native = EvidenceStatus.INCOMPLETE if early else EvidenceStatus.COMPLETE
        details = EvidenceStatus.INCOMPLETE if omitted is not None else EvidenceStatus.COMPLETE
        # The early-stop sentence accounts for observed details only, not the
        # diagnostics that unvisited code could have produced.
        if early and omitted is None:
            details = EvidenceStatus.UNKNOWN
    deduplicated = {finding.dedupe_key: finding for finding in findings}
    return ParsedWaitcheckOutput(
        findings=tuple(deduplicated[key] for key in sorted(deduplicated, key=repr)),
        diagnostics_truncated=truncated,
        diagnostic_summary=DiagnosticSummary(
            reported_count=count, count_meaning=meaning, emitted_records=records,
            details=details, native_analysis=native,
        ),
    )


def _validate_max_diagnostics(max_diagnostics: int | None) -> None:
    if max_diagnostics is not None and (
        isinstance(max_diagnostics, bool)
        or not isinstance(max_diagnostics, int)
        or max_diagnostics < 1
    ):
        raise ValueError("max_diagnostics must be a positive integer")


def waitcheck_argv(
    binary: Path,
    identity: KernelIdentity,
    *,
    max_diagnostics: int | None = None,
) -> tuple[str, ...]:
    """Build argv for one code-object Waitcheck invocation."""

    if not identity.exact and not identity.code_object_scan:
        raise ValueError(
            "Waitcheck requires code_object and code_object_sha256, "
            "and optionally entry_offset for exact-entry mode"
        )
    _validate_max_diagnostics(max_diagnostics)
    argv = [
        str(binary),
        str(identity.code_object),
        "--target",
        identity.target,
    ]
    if identity.code_object_index is not None:
        argv.extend(["--code-object-index", str(identity.code_object_index)])
    if identity.entry_offset is not None:
        argv.extend(
            [
                "--kernel-entry",
                f"0x{identity.entry_offset:x}",
            ]
        )
    if max_diagnostics is not None:
        argv.extend(["--max-diagnostics", str(max_diagnostics)])
    return tuple(argv)


def _not_checked(
    identity: KernelIdentity, reason: str, max_diagnostics: int | None = None,
) -> KernelCheckResult:
    return KernelCheckResult(
        identity=identity,
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason=reason,
        diagnostic_summary=(DiagnosticSummary(requested_limit=max_diagnostics)
                            if max_diagnostics is not None else None),
    )


def _run_one(
    identity: KernelIdentity,
    *,
    binary: Path,
    log_path: Path,
    timeout_seconds: float,
    execute: ProcessExecutor,
    max_diagnostics: int | None = None,
) -> KernelCheckResult:
    requested = DiagnosticSummary(requested_limit=max_diagnostics) if max_diagnostics is not None else None
    if not identity.exact and not identity.code_object_scan:
        return _not_checked(identity, "code_object_identity_required", max_diagnostics)
    artifact = Path(str(identity.code_object))
    if not artifact.is_file():
        return _not_checked(identity, "code_object_not_found", max_diagnostics)
    if artifact.stat().st_size == 0:
        return _not_checked(identity, "code_object_empty", max_diagnostics)
    digest = _sha256_file(artifact)
    if digest != identity.code_object_sha256:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason="code_object_digest_mismatch",
            diagnostic_summary=requested,
        )
    process = execute(
        waitcheck_argv(binary, identity, max_diagnostics=max_diagnostics),
        timeout_seconds=timeout_seconds,
    )
    log_path.write_text(
        f"{process.stdout}\n{process.stderr}",
        encoding="utf-8",
    )
    if process.timed_out:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.TIMED_OUT,
            verdict=Verdict.ERROR,
            reason="waitcheck_timeout",
            diagnostic_summary=requested,
        )
    if process.launch_error is not None:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason=f"waitcheck_launch_error: {process.launch_error}",
            diagnostic_summary=requested,
        )
    # Parse defensively so an unexpected exit code can still be reported with the
    # backend's own message. When rj_waitcheck refuses an input it explains itself on
    # stderr ("failed to parse input executable or code object", exit 2) and prints no
    # analysis summary; returning the parse failure first replaced that explanation
    # with "Waitcheck output did not contain an analysis summary", which names the
    # symptom the refusal caused rather than the refusal. The exit-code branch below
    # is the only one that quotes stderr, so it has to be reached first.
    parse_error: Exception | None = None
    findings: tuple[Finding, ...] = ()
    diagnostics_truncated = False
    diagnostic_summary = requested
    try:
        parsed = parse_waitcheck_text(
            f"{process.stdout}\n{process.stderr}",
            expected=identity,
        )
        findings = parsed.findings
        diagnostics_truncated = parsed.diagnostics_truncated
        if parsed.diagnostic_summary is not None:
            diagnostic_summary = replace(parsed.diagnostic_summary, requested_limit=max_diagnostics)
    except Exception as exc:  # noqa: BLE001 - see below
        # Deliberately broad. The point of parsing here is that an unexpected exit code
        # reaches the branch below with the backend's own message; a parser raising
        # anything outside (OSError, ValueError) -- an IndexError or TypeError off a
        # malformed line -- would propagate and defeat exactly that. Nothing is
        # swallowed: on an expected exit code the error becomes a fail-closed
        # ERROR result quoting the exception, and KeyboardInterrupt / SystemExit are
        # BaseException and still propagate.
        parse_error = exc
    if process.returncode not in {WAITCHECK_CLEAN_EXIT, WAITCHECK_HAZARD_EXIT}:
        stderr_tail = process.stderr[-300:].strip()
        suffix = f": {stderr_tail}" if stderr_tail else ""
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason=f"waitcheck_backend_exit_{process.returncode}{suffix}",
            returncode=process.returncode,
            findings=findings,
            diagnostics_truncated=diagnostics_truncated,
            diagnostic_summary=diagnostic_summary,
        )
    if parse_error is not None:
        # An expected exit code with unparsable output is a real identity/format
        # mismatch, so it keeps reporting the parse failure verbatim.
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason=f"waitcheck_diagnostics_error: {parse_error}",
            returncode=process.returncode,
            diagnostic_summary=requested,
        )
    if process.returncode == WAITCHECK_HAZARD_EXIT and not findings:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason="waitcheck_hazard_exit_without_structured_diagnostics",
            returncode=process.returncode,
            diagnostics_truncated=diagnostics_truncated,
            diagnostic_summary=diagnostic_summary,
        )
    if process.returncode == WAITCHECK_CLEAN_EXIT and (
        findings or (diagnostic_summary is not None and diagnostic_summary.reported_count)
    ):
        return KernelCheckResult(
            identity=identity, state=ExecutionState.ERROR, verdict=Verdict.ERROR,
            reason="waitcheck_clean_exit_with_diagnostics", returncode=process.returncode,
            findings=findings, diagnostics_truncated=diagnostics_truncated,
            diagnostic_summary=diagnostic_summary,
        )
    return KernelCheckResult(
        identity=identity,
        state=ExecutionState.RAN,
        verdict=Verdict.WARN if findings else Verdict.PASS,
        returncode=process.returncode,
        findings=findings,
        diagnostics_truncated=diagnostics_truncated,
        diagnostic_summary=diagnostic_summary,
    )


def run_waitcheck(
    worklist: KernelWorklist,
    *,
    output_dir: Path,
    binary: Path | None = None,
    timeout_seconds: float = 900.0,
    execute: ProcessExecutor = run_argv,
    max_diagnostics: int | None = None,
) -> CheckResult:
    """Run exact Waitcheck for every worklist entry, failing closed on gaps.

    ``max_diagnostics`` requests a per-invocation detail limit. Leaving it unset
    keeps the backend default. Count/completion semantics depend on the format;
    ``diagnostics_truncated`` retains only the historical >= marker meaning.
    """

    # Validate at the public boundary so missing backends and empty/rejected
    # worklists cannot silently accept a malformed library-call argument.
    _validate_max_diagnostics(max_diagnostics)
    resolved = resolve_waitcheck(binary)
    if resolved is None or not resolved.is_file():
        return CheckResult(
            sanitizer="waitcheck",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason="rj_waitcheck_not_found",
            kernel_results=tuple(
                _not_checked(item.identity, "rj_waitcheck_not_found", max_diagnostics) for item in worklist.kernels
            ),
        )
    if resolved.stat().st_size == 0:
        return CheckResult(
            sanitizer="waitcheck",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason="rj_waitcheck_empty",
            kernel_results=tuple(
                _not_checked(item.identity, "rj_waitcheck_empty", max_diagnostics) for item in worklist.kernels
            ),
        )
    backend = (
        ("path", str(resolved)),
        ("sha256", _sha256_file(resolved)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scanned_objects: set[tuple[str | None, int | None]] = set()
    scan_tasks: list[tuple[int, KernelIdentity]] = []
    for index, observation in enumerate(worklist.kernels):
        identity = observation.identity
        # A whole-code-object scan (no entry offset) covers the entire object, so
        # multiple selected kernels that resolve to the same object must produce a
        # single object-level invocation rather than repeated scans attributed to
        # distinct synthetic names.
        if identity.entry_offset is None and identity.code_object_scan:
            object_key = (identity.code_object_sha256, identity.code_object_index)
            if object_key in scanned_objects:
                continue
            scanned_objects.add(object_key)
        scan_tasks.append((index, identity))

    def _scan(task: tuple[int, KernelIdentity]) -> KernelCheckResult:
        task_index, task_identity = task
        return _run_one(
            task_identity,
            binary=resolved,
            log_path=output_dir / f"waitcheck-{task_index}.log",
            timeout_seconds=timeout_seconds,
            execute=execute,
            max_diagnostics=max_diagnostics,
        )

    # Each selected code object is an independent rj_waitcheck subprocess doing
    # CPU-bound static disassembly (~a minute on a real ~16MB / ~245-kernel
    # unbundled Tensile object, #366), so scan distinct objects concurrently to cut
    # wall-clock time. This is coverage-preserving -- every selected object is
    # still scanned with its full per-scan timeout -- so the gate is unchanged;
    # it only removes the serialization penalty when the worklist spans more
    # than one object (the daily GEMM fixture spans the distinct NT/TT transpose
    # objects, so two scans run in parallel). Shapes that share a transpose
    # already dedup to one object above, and a single-object worklist keeps the
    # exact serial path. ``ThreadPoolExecutor.map`` preserves task order, so
    # results stay deterministic regardless of completion order.
    if len(scan_tasks) <= 1:
        kernel_results = tuple(_scan(task) for task in scan_tasks)
    else:
        max_workers = min(len(scan_tasks), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            kernel_results = tuple(pool.map(_scan, scan_tasks))
    findings = tuple(finding for result in kernel_results for finding in result.findings)
    # Preserve the historical marker independently from new native evidence.
    truncated = any(result.diagnostics_truncated for result in kernel_results)
    if not kernel_results:
        return CheckResult(
            sanitizer="waitcheck",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason="empty_worklist",
            backend=backend,
        )
    unhealthy = [result for result in kernel_results if result.state is not ExecutionState.RAN]
    if unhealthy:
        return CheckResult(
            sanitizer="waitcheck",
            state=(
                ExecutionState.TIMED_OUT
                if any(result.state is ExecutionState.TIMED_OUT for result in unhealthy)
                else ExecutionState.ERROR
            ),
            verdict=Verdict.ERROR,
            reason="worklist_not_fully_checked",
            findings=findings,
            kernel_results=kernel_results,
            backend=backend,
            diagnostics_truncated=truncated,
            diagnostic_summary=aggregate_diagnostic_summaries(kernel_results),
        )
    return CheckResult(
        sanitizer="waitcheck",
        state=ExecutionState.RAN,
        verdict=Verdict.WARN if findings else Verdict.PASS,
        findings=findings,
        kernel_results=kernel_results,
        backend=backend,
        diagnostics_truncated=truncated,
        diagnostic_summary=aggregate_diagnostic_summaries(kernel_results),
    )
