"""Typed, JSON-stable models for kernel sanitizer runs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

WORKLIST_SCHEMA = "aorta.kernel_worklist/0.1"
REPORT_SCHEMA = "aorta.sanitizer_report/0.1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SelectionRequirement(str, Enum):
    """How observations are ranked before sanitizing."""

    TOP_TIME = "top_time"
    TOP_DISPATCH_COUNT = "top_dispatch_count"


class ExecutionState(str, Enum):
    """Whether a backend produced a trustworthy result."""

    RAN = "ran"
    NOT_CHECKED = "not_checked"
    ERROR = "error"
    TIMED_OUT = "timed_out"


class Verdict(str, Enum):
    """Guardrail outcome, kept separate from backend execution state."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_CHECKED = "not_checked"
    ERROR = "error"


class ExecutionSummary(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_CHECKED = "not_checked"
    ERROR = "error"


class FindingSeverity(str, Enum):
    WARNING = "warning"
    RACE = "race"
    ERROR = "error"


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer or null")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = _optional_int(data, key)
    if value is None:
        raise TypeError(f"{key} must be an integer")
    return value


def _required_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _required_bool(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _sequence(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


@dataclass(frozen=True)
class KernelIdentity:
    """Stable identity used to join profiling and sanitizer passes."""

    name: str
    target: str
    code_object: str | None = None
    code_object_sha256: str | None = None
    code_object_index: int | None = None
    entry_offset: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("kernel name must be non-empty")
        if not self.target:
            raise ValueError("kernel target must be non-empty")
        if self.code_object_sha256 is not None and not _SHA256.fullmatch(self.code_object_sha256):
            raise ValueError("code_object_sha256 must be lowercase SHA-256")
        for label, value in (
            ("code_object_index", self.code_object_index),
            ("entry_offset", self.entry_offset),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{label} must be non-negative")

    @property
    def exact(self) -> bool:
        """Whether Waitcheck can analyze exactly this kernel entry."""

        return (
            self.code_object is not None
            and self.code_object_sha256 is not None
            and self.entry_offset is not None
        )

    @property
    def code_object_scan(self) -> bool:
        """Whether Waitcheck can scan a whole code object without an entry offset."""

        return (
            self.code_object is not None
            and self.code_object_sha256 is not None
            and self.entry_offset is None
        )

    @property
    def stable_key(self) -> str:
        index_key = "" if self.code_object_index is None else str(self.code_object_index)
        if self.exact:
            entry_key = "" if self.entry_offset is None else f"{self.entry_offset:x}"
            return "\x1f".join(
                (
                    self.target,
                    self.code_object_sha256 or "",
                    index_key,
                    entry_key,
                )
            )
        if self.code_object_scan:
            # Whole-code-object scan has no entry offset to pin a kernel, so the
            # code object identity (sha/index) plus the kernel name keeps distinct
            # code objects and distinct kernels within one object from colliding.
            return "\x1f".join(
                (
                    self.target,
                    self.code_object_sha256 or "",
                    index_key,
                    self.name,
                )
            )
        return "\x1f".join((self.target, self.name))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target": self.target,
            "code_object": self.code_object,
            "code_object_sha256": self.code_object_sha256,
            "code_object_index": self.code_object_index,
            "entry_offset": self.entry_offset,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> KernelIdentity:
        return cls(
            name=_required_str(data, "name"),
            target=_required_str(data, "target"),
            code_object=_optional_str(data, "code_object"),
            code_object_sha256=_optional_str(data, "code_object_sha256"),
            code_object_index=_optional_int(data, "code_object_index"),
            entry_offset=_optional_int(data, "entry_offset"),
        )


@dataclass(frozen=True)
class KernelObservation:
    """One profiler observation before ranking."""

    identity: KernelIdentity
    total_time_ms: float = 0.0
    dispatch_count: int = 0
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_time_ms) or self.total_time_ms < 0:
            raise ValueError("total_time_ms must be finite and non-negative")
        if self.dispatch_count < 0:
            raise ValueError("dispatch_count must be non-negative")
        if not self.sources or not all(self.sources):
            raise ValueError("observation sources must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "total_time_ms": self.total_time_ms,
            "dispatch_count": self.dispatch_count,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> KernelObservation:
        source_values = _sequence(data.get("sources"), name="sources")
        if not all(isinstance(item, str) and item for item in source_values):
            raise TypeError("sources must be a list of non-empty strings")
        return cls(
            identity=KernelIdentity.from_dict(_mapping(data.get("identity"), name="identity")),
            total_time_ms=_required_float(data, "total_time_ms"),
            dispatch_count=_required_int(data, "dispatch_count"),
            sources=tuple(item for item in source_values if isinstance(item, str)),
        )


@dataclass(frozen=True)
class KernelWorklist:
    requirement: SelectionRequirement
    top_n: int
    kernels: tuple[KernelObservation, ...]
    schema: str = WORKLIST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORKLIST_SCHEMA:
            raise ValueError(
                f"unsupported worklist schema {self.schema!r}; " f"expected {WORKLIST_SCHEMA!r}"
            )
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        keys = [kernel.identity.stable_key for kernel in self.kernels]
        if len(keys) != len(set(keys)):
            raise ValueError("worklist contains duplicate kernel identities")
        if len(self.kernels) > self.top_n:
            raise ValueError("worklist contains more kernels than top_n")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "requirement": self.requirement.value,
            "top_n": self.top_n,
            "kernel_count": len(self.kernels),
            "kernels": [kernel.to_dict() for kernel in self.kernels],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> KernelWorklist:
        schema = _required_str(data, "schema")
        kernel_values = _sequence(data.get("kernels"), name="kernels")
        kernels = tuple(
            KernelObservation.from_dict(_mapping(item, name="kernels[]")) for item in kernel_values
        )
        declared_count = _required_int(data, "kernel_count")
        if declared_count != len(kernels):
            raise ValueError(f"kernel_count={declared_count} does not match {len(kernels)} entries")
        return cls(
            schema=schema,
            requirement=SelectionRequirement(_required_str(data, "requirement")),
            top_n=_required_int(data, "top_n"),
            kernels=kernels,
        )


@dataclass(frozen=True)
class Finding:
    sanitizer: str
    severity: FindingSeverity
    code: str
    message: str
    kernel_name: str | None = None
    code_object: str | None = None
    entry_offset: int | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.sanitizer or not self.code or not self.message:
            raise ValueError("finding sanitizer, code, and message must be non-empty")

    @property
    def dedupe_key(self) -> tuple[object, ...]:
        return (
            self.sanitizer,
            self.severity.value,
            self.code,
            self.message,
            self.kernel_name,
            self.code_object,
            self.entry_offset,
            self.metadata,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sanitizer": self.sanitizer,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "kernel_name": self.kernel_name,
            "code_object": self.code_object,
            "entry_offset": self.entry_offset,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Finding:
        metadata = _mapping(data.get("metadata"), name="metadata")
        if not all(isinstance(value, str) for value in metadata.values()):
            raise TypeError("finding metadata values must be strings")
        return cls(
            sanitizer=_required_str(data, "sanitizer"),
            severity=FindingSeverity(_required_str(data, "severity")),
            code=_required_str(data, "code"),
            message=_required_str(data, "message"),
            kernel_name=_optional_str(data, "kernel_name"),
            code_object=_optional_str(data, "code_object"),
            entry_offset=_optional_int(data, "entry_offset"),
            metadata=tuple(
                sorted((key, value) for key, value in metadata.items() if isinstance(value, str))
            ),
        )


@dataclass(frozen=True)
class ObjectCoverage:
    """ConSan health for one loaded code object."""

    object_id: str
    applicable: bool
    analysis_complete: bool
    static_complete: bool
    dynamic_complete: bool
    incomplete_code_objects: int = 0
    unsupported: int = 0
    access: str | None = None
    fields: tuple[tuple[str, str], ...] = ()

    @property
    def complete(self) -> bool:
        return (
            self.applicable
            and self.analysis_complete
            and self.static_complete
            and self.dynamic_complete
            and self.incomplete_code_objects == 0
            and self.unsupported == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "applicable": self.applicable,
            "analysis_complete": self.analysis_complete,
            "static_complete": self.static_complete,
            "dynamic_complete": self.dynamic_complete,
            "incomplete_code_objects": self.incomplete_code_objects,
            "unsupported": self.unsupported,
            "access": self.access,
            "complete": self.complete,
            "fields": dict(self.fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ObjectCoverage:
        fields = _mapping(data.get("fields"), name="fields")
        if not all(isinstance(value, str) for value in fields.values()):
            raise TypeError("coverage fields must have string values")
        coverage = cls(
            object_id=_required_str(data, "object_id"),
            applicable=_required_bool(data, "applicable"),
            analysis_complete=_required_bool(data, "analysis_complete"),
            static_complete=_required_bool(data, "static_complete"),
            dynamic_complete=_required_bool(data, "dynamic_complete"),
            incomplete_code_objects=_required_int(data, "incomplete_code_objects"),
            unsupported=_required_int(data, "unsupported"),
            access=_optional_str(data, "access"),
            fields=tuple(
                sorted((key, value) for key, value in fields.items() if isinstance(value, str))
            ),
        )
        declared_complete = data.get("complete")
        if declared_complete is not None and (
            not isinstance(declared_complete, bool) or declared_complete != coverage.complete
        ):
            raise ValueError("coverage complete field contradicts coverage data")
        return coverage


@dataclass(frozen=True)
class KernelCheckResult:
    identity: KernelIdentity
    state: ExecutionState
    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    reason: str | None = None
    returncode: int | None = None

    def __post_init__(self) -> None:
        if self.state is ExecutionState.NOT_CHECKED:
            if self.verdict is not Verdict.NOT_CHECKED or not self.reason:
                raise ValueError(
                    "not_checked kernel result requires not_checked verdict and reason"
                )
        elif self.state is ExecutionState.RAN:
            if self.verdict not in {Verdict.PASS, Verdict.WARN, Verdict.FAIL}:
                raise ValueError("ran kernel result has invalid verdict")
        elif self.verdict not in {Verdict.ERROR, Verdict.FAIL}:
            raise ValueError("failed kernel execution cannot be clean")
        if self.verdict is Verdict.PASS and self.findings:
            raise ValueError("a PASS kernel result cannot carry findings")

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "state": self.state.value,
            "verdict": self.verdict.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "reason": self.reason,
            "returncode": self.returncode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> KernelCheckResult:
        finding_values = _sequence(data.get("findings"), name="findings")
        return cls(
            identity=KernelIdentity.from_dict(_mapping(data.get("identity"), name="identity")),
            state=ExecutionState(_required_str(data, "state")),
            verdict=Verdict(_required_str(data, "verdict")),
            findings=tuple(
                Finding.from_dict(_mapping(item, name="findings[]")) for item in finding_values
            ),
            reason=_optional_str(data, "reason"),
            returncode=_optional_int(data, "returncode"),
        )


@dataclass(frozen=True)
class CheckResult:
    sanitizer: str
    state: ExecutionState
    verdict: Verdict
    reason: str | None = None
    returncode: int | None = None
    findings: tuple[Finding, ...] = ()
    kernel_results: tuple[KernelCheckResult, ...] = ()
    coverage: tuple[ObjectCoverage, ...] = ()
    backend: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.sanitizer:
            raise ValueError("sanitizer name must be non-empty")
        if self.state is ExecutionState.NOT_CHECKED and self.verdict is not Verdict.NOT_CHECKED:
            raise ValueError("not_checked state requires not_checked verdict")
        if self.state is ExecutionState.NOT_CHECKED and not self.reason:
            raise ValueError("not_checked state requires a reason")
        if self.state is ExecutionState.RAN and self.verdict not in {
            Verdict.PASS,
            Verdict.WARN,
            Verdict.FAIL,
        }:
            raise ValueError("ran check has invalid verdict")
        if self.state in {ExecutionState.ERROR, ExecutionState.TIMED_OUT} and self.verdict not in {
            Verdict.ERROR,
            Verdict.FAIL,
        }:
            raise ValueError("failed execution cannot have a clean verdict")
        if self.verdict is Verdict.PASS and self.findings:
            raise ValueError("a PASS check cannot carry findings")
        if self.kernel_results and _VERDICT_RANK[self.verdict] < max(
            _VERDICT_RANK[result.verdict] for result in self.kernel_results
        ):
            raise ValueError("a check verdict cannot be cleaner than its kernel results")

    def to_dict(self) -> dict[str, object]:
        return {
            "sanitizer": self.sanitizer,
            "state": self.state.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "returncode": self.returncode,
            "findings": [finding.to_dict() for finding in self.findings],
            "kernel_results": [result.to_dict() for result in self.kernel_results],
            "coverage": [item.to_dict() for item in self.coverage],
            "backend": dict(self.backend),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CheckResult:
        finding_values = _sequence(data.get("findings"), name="findings")
        kernel_values = _sequence(data.get("kernel_results"), name="kernel_results")
        coverage_values = _sequence(data.get("coverage"), name="coverage")
        backend = _mapping(data.get("backend"), name="backend")
        if not all(isinstance(value, str) for value in backend.values()):
            raise TypeError("backend metadata values must be strings")
        return cls(
            sanitizer=_required_str(data, "sanitizer"),
            state=ExecutionState(_required_str(data, "state")),
            verdict=Verdict(_required_str(data, "verdict")),
            reason=_optional_str(data, "reason"),
            returncode=_optional_int(data, "returncode"),
            findings=tuple(
                Finding.from_dict(_mapping(item, name="findings[]")) for item in finding_values
            ),
            kernel_results=tuple(
                KernelCheckResult.from_dict(_mapping(item, name="kernel_results[]"))
                for item in kernel_values
            ),
            coverage=tuple(
                ObjectCoverage.from_dict(_mapping(item, name="coverage[]"))
                for item in coverage_values
            ),
            backend=tuple(
                sorted((key, value) for key, value in backend.items() if isinstance(value, str))
            ),
        )


_VERDICT_RANK = {
    Verdict.PASS: 0,
    Verdict.NOT_CHECKED: 1,
    Verdict.WARN: 2,
    Verdict.ERROR: 3,
    Verdict.FAIL: 4,
}


@dataclass(frozen=True)
class SanitizerReport:
    target: str
    worklist: KernelWorklist
    checks: tuple[CheckResult, ...]
    overall_verdict: Verdict = field(init=False)
    execution_status: ExecutionSummary = field(init=False)
    schema: str = REPORT_SCHEMA

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("report target must be non-empty")
        if self.schema != REPORT_SCHEMA:
            raise ValueError(
                f"unsupported report schema {self.schema!r}; expected {REPORT_SCHEMA!r}"
            )
        if any(kernel.identity.target != self.target for kernel in self.worklist.kernels):
            raise ValueError("worklist target does not match report target")
        names = [check.sanitizer for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("report contains duplicate sanitizer results")
        verdict = (
            Verdict.NOT_CHECKED
            if not self.checks
            else max(
                (check.verdict for check in self.checks),
                key=lambda item: _VERDICT_RANK[item],
            )
        )
        object.__setattr__(self, "overall_verdict", verdict)
        if any(
            check.state in {ExecutionState.ERROR, ExecutionState.TIMED_OUT} for check in self.checks
        ):
            execution = ExecutionSummary.ERROR
        elif self.checks and all(
            check.state is ExecutionState.NOT_CHECKED for check in self.checks
        ):
            execution = ExecutionSummary.NOT_CHECKED
        elif any(check.state is ExecutionState.NOT_CHECKED for check in self.checks):
            execution = ExecutionSummary.PARTIAL
        elif self.checks:
            execution = ExecutionSummary.COMPLETE
        else:
            execution = ExecutionSummary.NOT_CHECKED
        object.__setattr__(self, "execution_status", execution)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "target": self.target,
            "overall_verdict": self.overall_verdict.value,
            "execution_status": self.execution_status.value,
            "worklist": self.worklist.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SanitizerReport:
        check_values = _sequence(data.get("checks"), name="checks")
        report = cls(
            schema=_required_str(data, "schema"),
            target=_required_str(data, "target"),
            worklist=KernelWorklist.from_dict(_mapping(data.get("worklist"), name="worklist")),
            checks=tuple(
                CheckResult.from_dict(_mapping(item, name="checks[]")) for item in check_values
            ),
        )
        declared = Verdict(_required_str(data, "overall_verdict"))
        if declared is not report.overall_verdict:
            raise ValueError(f"overall_verdict={declared.value!r} contradicts check results")
        declared_execution = ExecutionSummary(_required_str(data, "execution_status"))
        if declared_execution is not report.execution_status:
            raise ValueError("execution_status contradicts check execution states")
        return report
