"""Static Waitcheck backend for exact-entry and whole-code-object-scan identities."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .execution import ProcessResult, run_argv
from .models import (
    CheckResult,
    ExecutionState,
    Finding,
    FindingSeverity,
    KernelCheckResult,
    KernelIdentity,
    KernelWorklist,
    Verdict,
)

WAITCHECK_CLEAN_EXIT = 0
WAITCHECK_HAZARD_EXIT = 4
WAITCHECK_JSON_SCHEMA = "rj-waitcheck-diagnostic-v1"
_HEADER = re.compile(
    r":(?P<target>gfx[0-9a-f]+)\[(?P<index>[0-9]+)\]:"
    r"\s+instructions=(?P<instructions>[0-9]+).*"
    r"diagnostics=(?:>=)?(?P<diagnostics>[0-9]+)"
)
_DIAGNOSTIC = re.compile(r"\bmissing\s+s_wait|\bhazard\b", re.IGNORECASE)
_CONTEXT = re.compile(r"^(producer|consumer)\b", re.IGNORECASE)


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
    """Parse structured diagnostics and reject identity mismatches."""

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
                kernel_name=kernel_name,
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
) -> tuple[Finding, ...]:
    """Parse exact-entry CLI output until structured single-entry output exists."""

    findings: list[Finding] = []
    header_seen = False
    expected_index = expected.code_object_index or 0
    for raw_line in output.splitlines():
        line = raw_line.strip()
        header = _HEADER.search(line)
        if header is not None:
            header_seen = True
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
            continue
        context = _CONTEXT.match(line)
        if context is not None:
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
            findings.append(
                Finding(
                    sanitizer="waitcheck",
                    severity=FindingSeverity.WARNING,
                    code="wait_hazard",
                    message=line,
                    kernel_name=expected.name,
                    code_object=expected.code_object,
                    entry_offset=expected.entry_offset,
                )
            )
    if not header_seen:
        raise ValueError("Waitcheck output did not contain an analysis summary")
    deduplicated = {finding.dedupe_key: finding for finding in findings}
    return tuple(deduplicated[key] for key in sorted(deduplicated, key=repr))


def waitcheck_argv(
    binary: Path,
    identity: KernelIdentity,
) -> tuple[str, ...]:
    """Build argv for one code-object Waitcheck invocation."""

    if not identity.exact and not identity.code_object_scan:
        raise ValueError(
            "Waitcheck requires code_object and code_object_sha256, "
            "and optionally entry_offset for exact-entry mode"
        )
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
    return tuple(argv)


def _not_checked(identity: KernelIdentity, reason: str) -> KernelCheckResult:
    return KernelCheckResult(
        identity=identity,
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason=reason,
    )


def _run_one(
    identity: KernelIdentity,
    *,
    binary: Path,
    log_path: Path,
    timeout_seconds: float,
    execute: ProcessExecutor,
) -> KernelCheckResult:
    if not identity.exact and not identity.code_object_scan:
        return _not_checked(identity, "code_object_identity_required")
    artifact = Path(str(identity.code_object))
    if not artifact.is_file():
        return _not_checked(identity, "code_object_not_found")
    if artifact.stat().st_size == 0:
        return _not_checked(identity, "code_object_empty")
    digest = _sha256_file(artifact)
    if digest != identity.code_object_sha256:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason="code_object_digest_mismatch",
        )
    process = execute(
        waitcheck_argv(binary, identity),
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
        )
    if process.launch_error is not None:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason=f"waitcheck_launch_error: {process.launch_error}",
        )
    try:
        findings = parse_waitcheck_text(
            f"{process.stdout}\n{process.stderr}",
            expected=identity,
        )
    except (OSError, ValueError) as exc:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason=f"waitcheck_diagnostics_error: {exc}",
            returncode=process.returncode,
        )
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
        )
    if process.returncode == WAITCHECK_HAZARD_EXIT and not findings:
        return KernelCheckResult(
            identity=identity,
            state=ExecutionState.ERROR,
            verdict=Verdict.ERROR,
            reason="waitcheck_hazard_exit_without_structured_diagnostics",
            returncode=process.returncode,
        )
    return KernelCheckResult(
        identity=identity,
        state=ExecutionState.RAN,
        verdict=Verdict.WARN if findings else Verdict.PASS,
        returncode=process.returncode,
        findings=findings,
    )


def run_waitcheck(
    worklist: KernelWorklist,
    *,
    output_dir: Path,
    binary: Path | None = None,
    timeout_seconds: float = 900.0,
    execute: ProcessExecutor = run_argv,
) -> CheckResult:
    """Run exact Waitcheck for every worklist entry, failing closed on gaps."""

    resolved = resolve_waitcheck(binary)
    if resolved is None or not resolved.is_file():
        return CheckResult(
            sanitizer="waitcheck",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason="rj_waitcheck_not_found",
            kernel_results=tuple(
                _not_checked(item.identity, "rj_waitcheck_not_found") for item in worklist.kernels
            ),
        )
    if resolved.stat().st_size == 0:
        return CheckResult(
            sanitizer="waitcheck",
            state=ExecutionState.NOT_CHECKED,
            verdict=Verdict.NOT_CHECKED,
            reason="rj_waitcheck_empty",
            kernel_results=tuple(
                _not_checked(item.identity, "rj_waitcheck_empty") for item in worklist.kernels
            ),
        )
    backend = (
        ("path", str(resolved)),
        ("sha256", _sha256_file(resolved)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_results = tuple(
        _run_one(
            observation.identity,
            binary=resolved,
            log_path=output_dir / f"waitcheck-{index}.log",
            timeout_seconds=timeout_seconds,
            execute=execute,
        )
        for index, observation in enumerate(worklist.kernels)
    )
    findings = tuple(finding for result in kernel_results for finding in result.findings)
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
        )
    return CheckResult(
        sanitizer="waitcheck",
        state=ExecutionState.RAN,
        verdict=Verdict.WARN if findings else Verdict.PASS,
        findings=findings,
        kernel_results=kernel_results,
        backend=backend,
    )
