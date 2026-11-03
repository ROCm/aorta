"""The ``AORTA_FPSAN_RESULT`` contract shared by every FpSan runner.

A recipe's FpSan command runs a reference kernel and a candidate kernel, both
under FpSan, and prints exactly one stdout line::

    AORTA_FPSAN_RESULT {"reference_digest": "<sha256>", "candidate_digest": "<sha256>",
                        "reference_instrumented": true, "candidate_instrumented": true,
                        "target": "gfx950"}

This module knows that line and nothing about how it was produced: no Triton
environment and no ``fpsan::Value`` check live here, so a later HIP-FpSan runner
evaluates its command's output with the same function.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .execution import ProcessResult
from .models import CheckResult, ExecutionState, Finding, FindingSeverity, Verdict

RESULT_MARKER = "AORTA_FPSAN_RESULT"
# AgentKernelArena's marker. Recognised only to say why it was refused.
_FOREIGN_MARKER = "AKA_FPSAN_RESULT"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# SHA-256 of zero bytes. Two empty output files hash to this on both sides and
# would otherwise compare equal, so a command that hashed nothing would pass.
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_REQUIRED_FIELDS = frozenset(
    {"reference_digest", "candidate_digest", "reference_instrumented", "candidate_instrumented"}
)
_OPTIONAL_FIELDS = frozenset({"target"})
MISMATCH_CODE = "fpsan_digest_mismatch"


@dataclass(frozen=True)
class FpsanResult:
    """One parsed ``AORTA_FPSAN_RESULT`` record."""

    reference_digest: str
    candidate_digest: str
    reference_instrumented: bool
    candidate_instrumented: bool
    target: str | None = None

    @property
    def instrumented(self) -> bool:
        return self.reference_instrumented and self.candidate_instrumented

    @property
    def digests_match(self) -> bool:
        return self.reference_digest == self.candidate_digest


def _marked(lines: list[str], marker: str) -> list[str]:
    """The payloads of ``lines`` that start with ``marker`` as a whole word.

    Any whitespace ends the marker, not only a space: a second result written
    with a tab must still be counted, or it would hide behind the first.
    """

    pattern = re.compile(rf"{re.escape(marker)}(?:\s|\Z)")
    payloads = []
    for line in lines:
        text = line.strip()
        if pattern.match(text):
            payloads.append(text[len(marker) :].strip())
    return payloads


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``json.loads`` object hook that refuses a repeated field.

    The stdlib keeps the last value of a duplicate key, so a second
    ``candidate_digest`` could overwrite a mismatch and the run would pass.
    """

    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"fpsan_result_unparsable: duplicate field {key!r}")
        record[key] = value
    return record


def _record(payload: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fpsan_result_unparsable: {exc}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("fpsan_result_unparsable: the result must be a JSON object")
    return value


def parse_fpsan_result(stdout: str) -> FpsanResult:
    """Parse the single ``AORTA_FPSAN_RESULT`` line out of a command's stdout.

    Raises ``ValueError`` whose message starts with the report reason: no line,
    more than one, a payload that is not a JSON object, a duplicate field, an
    unknown or missing field, a digest that is not 64 lowercase hex characters,
    the SHA-256 of empty input, or an instrumentation flag that is not a boolean.
    """

    lines = stdout.splitlines()
    payloads = _marked(lines, RESULT_MARKER)
    if not payloads:
        hint = (
            f"; an {_FOREIGN_MARKER} line was printed, but only {RESULT_MARKER} is accepted"
            if _marked(lines, _FOREIGN_MARKER)
            else ""
        )
        raise ValueError(f"fpsan_result_missing: no {RESULT_MARKER} line on stdout{hint}")
    if len(payloads) > 1:
        raise ValueError(
            f"fpsan_result_ambiguous: {len(payloads)} {RESULT_MARKER} lines on stdout, "
            "expected exactly one"
        )
    record = _record(payloads[0])
    unknown = sorted(set(record) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(f"fpsan_result_unparsable: unknown fields {unknown}")
    missing = sorted(_REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"fpsan_result_unparsable: missing fields {missing}")
    digests = {}
    for key in ("reference_digest", "candidate_digest"):
        value = record[key]
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"fpsan_result_unparsable: {key} must be 64 lowercase hex characters")
        if value == _EMPTY_SHA256:
            raise ValueError(
                "fpsan_result_empty_output: "
                f"{key} is the SHA-256 of empty input, so that side produced no bytes"
            )
        digests[key] = value
    flags = {}
    for key in ("reference_instrumented", "candidate_instrumented"):
        value = record[key]
        if not isinstance(value, bool):
            raise ValueError(f"fpsan_result_unparsable: {key} must be a boolean")
        flags[key] = value
    target = record.get("target")
    if "target" in record and (not isinstance(target, str) or not target.strip()):
        raise ValueError("fpsan_result_unparsable: target must be a non-empty string")
    return FpsanResult(
        reference_digest=digests["reference_digest"],
        candidate_digest=digests["candidate_digest"],
        reference_instrumented=flags["reference_instrumented"],
        candidate_instrumented=flags["candidate_instrumented"],
        target=None if target is None else target.strip(),
    )


def _error(
    sanitizer: str,
    reason: str,
    *,
    returncode: int | None = None,
    backend: tuple[tuple[str, str], ...] = (),
) -> CheckResult:
    return CheckResult(
        sanitizer=sanitizer,
        state=ExecutionState.ERROR,
        verdict=Verdict.ERROR,
        reason=reason,
        returncode=returncode,
        backend=backend,
    )


def _reported(result: FpsanResult) -> tuple[tuple[str, str], ...]:
    """What the command reported, kept in the check's backend on every verdict."""

    return (
        ("candidate_digest", result.candidate_digest),
        ("candidate_instrumented", str(result.candidate_instrumented).lower()),
        ("reference_digest", result.reference_digest),
        ("reference_instrumented", str(result.reference_instrumented).lower()),
        ("reported_target", result.target or "absent"),
    )


def evaluate_fpsan_output(
    process: ProcessResult,
    *,
    sanitizer: str,
    target: str,
) -> CheckResult:
    """Turn one FpSan command run into a check result.

    ``pass`` needs a clean exit and one result line with both sides instrumented
    and equal digests; equal digests with a side uninstrumented compared nothing.
    Unequal digests from two instrumented kernels are the finding, ``fail``.
    Everything else -- a timeout, a launch failure, a non-zero exit, a missing,
    repeated or malformed line, a duplicate field, the SHA-256 of empty input,
    an uninstrumented side, or a line for another target -- is ``error``: a
    harness problem, not evidence about the kernel.

    Process failures are reasoned under ``sanitizer`` (``hip_fpsan_timeout``);
    result-line failures keep the ``fpsan_`` prefix of the shared contract. Once
    the line parses, both digests and flags are kept in ``backend``, so a pass
    records what matched.
    """

    if process.timed_out:
        return CheckResult(
            sanitizer=sanitizer,
            state=ExecutionState.TIMED_OUT,
            verdict=Verdict.ERROR,
            reason=f"{sanitizer}_timeout",
        )
    if process.launch_error is not None:
        return _error(sanitizer, f"{sanitizer}_launch_error: {process.launch_error}")
    if process.returncode != 0:
        stderr_tail = process.stderr[-300:].strip()
        suffix = f": {stderr_tail}" if stderr_tail else ""
        return _error(
            sanitizer,
            f"{sanitizer}_command_exit_{process.returncode}{suffix}",
            returncode=process.returncode,
        )
    try:
        result = parse_fpsan_result(process.stdout)
    except ValueError as exc:
        return _error(sanitizer, str(exc), returncode=process.returncode)
    reported = _reported(result)
    if not result.instrumented:
        return _error(
            sanitizer,
            "fpsan_not_instrumented: the command reported "
            f"reference_instrumented={str(result.reference_instrumented).lower()}, "
            f"candidate_instrumented={str(result.candidate_instrumented).lower()}; "
            "both kernels must run under FpSan for their digests to be comparable",
            returncode=process.returncode,
            backend=reported,
        )
    if result.target is not None and result.target != target:
        return _error(
            sanitizer,
            f"fpsan_target_mismatch: the command reported target {result.target!r}, "
            f"the recipe requested {target!r}",
            returncode=process.returncode,
            backend=reported,
        )
    if result.digests_match:
        return CheckResult(
            sanitizer=sanitizer,
            state=ExecutionState.RAN,
            verdict=Verdict.PASS,
            returncode=process.returncode,
            backend=reported,
        )
    finding = Finding(
        sanitizer=sanitizer,
        severity=FindingSeverity.ERROR,
        code=MISMATCH_CODE,
        message=(
            "reference and candidate kernels disagree under FpSan: reference digest "
            f"{result.reference_digest[:16]}... != candidate digest "
            f"{result.candidate_digest[:16]}..."
        ),
        metadata=tuple(
            sorted(
                {
                    "reference_digest": result.reference_digest,
                    "candidate_digest": result.candidate_digest,
                    "target": target,
                }.items()
            )
        ),
    )
    return CheckResult(
        sanitizer=sanitizer,
        state=ExecutionState.RAN,
        verdict=Verdict.FAIL,
        reason=MISMATCH_CODE,
        returncode=process.returncode,
        findings=(finding,),
        backend=reported,
    )
