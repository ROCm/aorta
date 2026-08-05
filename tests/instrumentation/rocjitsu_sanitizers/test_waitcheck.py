from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import (
    ExecutionState,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    Verdict,
    parse_waitcheck_jsonl,
    run_waitcheck,
)
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocjitsu_sanitizers.waitcheck import parse_waitcheck_text

# Diagnostic fields mirror RocJITsu waitcheck_main.cpp at b4feaddd.


def _worklist(identity: KernelIdentity) -> KernelWorklist:
    return KernelWorklist(
        requirement=SelectionRequirement.TOP_TIME,
        top_n=1,
        kernels=(
            KernelObservation(
                identity=identity,
                total_time_ms=1,
                dispatch_count=1,
                sources=("test",),
            ),
        ),
    )


def _exact_identity(artifact: Path) -> KernelIdentity:
    return KernelIdentity(
        name="selected_kernel",
        target="gfx950",
        code_object=str(artifact),
        code_object_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        code_object_index=3,
        entry_offset=0x120,
    )


def test_waitcheck_hazard_uses_valid_exact_entry_cli(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "rj_waitcheck"
    binary.write_text("binary")
    artifact = tmp_path / "library.so"
    artifact.write_bytes(b"\x7fELF")
    captured: list[tuple[str, ...]] = []

    def execute(
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        assert timeout_seconds == 12
        assert env is None
        captured.append(tuple(argv))
        output = "\n".join(
            (
                f"{artifact}:gfx950[3]: instructions=10 memory-events=2 diagnostics=1",
                f"{artifact}:gfx950[3]:.text+0x20: missing s_waitcnt lgkmcnt(0)",
                "  producer .text+0x10: s_load_dword s4, s[0:1], 0",
                "  consumer .text+0x20: s_mov_b32 s5, s4",
            )
        )
        return ProcessResult(tuple(argv), 4, output, "")

    result = run_waitcheck(
        _worklist(_exact_identity(artifact)),
        output_dir=tmp_path / "out",
        binary=binary,
        timeout_seconds=12,
        execute=execute,
    )

    assert result.state is ExecutionState.RAN
    assert result.verdict is Verdict.WARN
    assert len(result.findings) == 1
    assert dict(result.backend)["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    argv = captured[0]
    assert argv[:2] == (str(binary), str(artifact))
    assert ("--target", "gfx950") == argv[2:4]
    assert "--code-object-index" in argv
    assert argv[argv.index("--kernel-entry") + 1] == "0x120"
    assert "--diagnostics-jsonl" not in argv


def test_waitcheck_backend_error_never_passes(tmp_path: Path) -> None:
    binary = tmp_path / "rj_waitcheck"
    binary.write_text("binary")
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")

    def execute(
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        return ProcessResult(tuple(argv), 2, "", "analysis failed")

    result = run_waitcheck(
        _worklist(_exact_identity(artifact)),
        output_dir=tmp_path / "out",
        binary=binary,
        execute=execute,
    )

    assert result.state is ExecutionState.ERROR
    assert result.verdict is Verdict.ERROR
    assert "worklist_not_fully_checked" in str(result.reason)
    assert result.kernel_results[0].returncode == 2


def test_waitcheck_timeout_never_passes(tmp_path: Path) -> None:
    binary = tmp_path / "rj_waitcheck"
    binary.write_text("binary")
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")

    def execute(
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        return ProcessResult(tuple(argv), None, "", "", timed_out=True)

    result = run_waitcheck(
        _worklist(_exact_identity(artifact)),
        output_dir=tmp_path / "out",
        binary=binary,
        execute=execute,
    )

    assert result.state is ExecutionState.TIMED_OUT
    assert result.verdict is Verdict.ERROR


def test_waitcheck_hazard_exit_without_json_is_error(tmp_path: Path) -> None:
    binary = tmp_path / "rj_waitcheck"
    binary.write_text("binary")
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")

    def execute(
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        return ProcessResult(
            tuple(argv),
            4,
            f"{artifact}:gfx950[3]: instructions=10 memory-events=2 diagnostics=>=1",
            "",
        )

    result = run_waitcheck(
        _worklist(_exact_identity(artifact)),
        output_dir=tmp_path / "out",
        binary=binary,
        execute=execute,
    )

    assert result.verdict is Verdict.ERROR
    assert "without_structured_diagnostics" in str(result.kernel_results[0].reason)


def test_waitcheck_requires_exact_identity(tmp_path: Path) -> None:
    binary = tmp_path / "rj_waitcheck"
    binary.write_text("binary")
    identity = KernelIdentity(name="kernel", target="gfx950")

    result = run_waitcheck(
        _worklist(identity),
        output_dir=tmp_path / "out",
        binary=binary,
    )

    assert result.verdict is Verdict.ERROR
    assert result.kernel_results[0].verdict is Verdict.NOT_CHECKED
    assert result.kernel_results[0].reason == "code_object_identity_required"


def test_waitcheck_missing_binary_is_not_checked(tmp_path: Path) -> None:
    identity = KernelIdentity(name="kernel", target="gfx950")

    result = run_waitcheck(
        _worklist(identity),
        output_dir=tmp_path / "out",
        binary=tmp_path / "missing",
    )

    assert result.state is ExecutionState.NOT_CHECKED
    assert result.verdict is Verdict.NOT_CHECKED


def test_waitcheck_rejects_structured_target_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")
    identity = _exact_identity(artifact)
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "schema": "rj-waitcheck-diagnostic-v1",
                "input": str(artifact),
                "message": "missing wait",
                "kernel_name": identity.name,
                "target": "gfx942",
                "code_object_index": identity.code_object_index,
                "kernel_entry": identity.entry_offset,
                "counter": "lgkmcnt",
                "access": "use",
            }
        )
        + "\n"
    )

    try:
        parse_waitcheck_jsonl(diagnostics, expected=identity)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("target mismatch was accepted")


def test_waitcheck_parses_upstream_diagnostic_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")
    identity = _exact_identity(artifact)
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "schema": "rj-waitcheck-diagnostic-v1",
                "input": str(artifact),
                "target": "gfx950",
                "code_object_index": 3,
                "kernel_name": "selected_kernel",
                "kernel_entry": 0x120,
                "kernel_entry_hex": "0x120",
                "counter": "lgkmcnt",
                "access": "use",
                "required_count": 0,
                "instruction": "s_mov_b32 s5, s4",
                "producer_instruction": "s_load_dword s4, s[0:1], 0",
                "section_offset_hex": "0x20",
                "producer_section_offset_hex": "0x10",
                "message": "missing s_waitcnt lgkmcnt(0)",
            }
        )
        + "\n"
    )

    findings = parse_waitcheck_jsonl(diagnostics, expected=identity)

    assert len(findings) == 1
    assert findings[0].code == "missing_lgkmcnt_use"


def _entry_identity() -> KernelIdentity:
    return KernelIdentity(
        name="selected_kernel",
        target="gfx950",
        code_object="/tmp/library.so",
        code_object_sha256="0" * 64,
        code_object_index=3,
        entry_offset=0x120,
    )


def test_parse_waitcheck_text_accepts_exact_entry_marker() -> None:
    output = "\n".join(
        (
            "/tmp/library.so:gfx950[3]:kernel=.text+0x120: "
            "instructions=10 memory-events=2 diagnostics=1",
            "/tmp/library.so:gfx950[3]:.text+0x20: missing s_waitcnt lgkmcnt(0)",
        )
    )

    findings = parse_waitcheck_text(output, expected=_entry_identity())

    assert len(findings) == 1


def test_parse_waitcheck_text_rejects_mismatched_entry() -> None:
    output = (
        "/tmp/library.so:gfx950[3]:kernel=.text+0x999: "
        "instructions=10 memory-events=2 diagnostics=1"
    )

    with pytest.raises(ValueError, match="entry"):
        parse_waitcheck_text(output, expected=_entry_identity())


def test_waitcheck_accepts_single_image_without_index(tmp_path: Path) -> None:
    # A single-image code object omits code_object_index in the JSONL; when the
    # expected identity is also single-image (index None -> 0), that must parse
    # rather than being rejected as a mismatch.
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")
    identity = KernelIdentity(
        name="selected_kernel",
        target="gfx950",
        code_object=str(artifact),
        code_object_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        entry_offset=0x120,
    )
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "schema": "rj-waitcheck-diagnostic-v1",
                "input": str(artifact),
                "target": "gfx950",
                "kernel_name": "selected_kernel",
                "kernel_entry": 0x120,
                "counter": "lgkmcnt",
                "access": "use",
                "message": "missing s_waitcnt lgkmcnt(0)",
            }
        )
        + "\n"
    )

    findings = parse_waitcheck_jsonl(diagnostics, expected=identity)

    assert len(findings) == 1
    assert findings[0].code == "missing_lgkmcnt_use"


def test_waitcheck_rejects_structured_entry_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")
    identity = _exact_identity(artifact)
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "schema": "rj-waitcheck-diagnostic-v1",
                "input": str(artifact),
                "message": "missing wait",
                "kernel_name": identity.name,
                "target": identity.target,
                "code_object_index": identity.code_object_index,
                "kernel_entry": identity.entry_offset + 4,
                "counter": "lgkmcnt",
                "access": "use",
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="entry"):
        parse_waitcheck_jsonl(diagnostics, expected=identity)


def test_waitcheck_scans_shared_object_once_without_kernel_attribution(tmp_path: Path) -> None:
    binary = tmp_path / "rj_waitcheck"
    binary.write_text("binary")
    artifact = tmp_path / "shared.hsaco"
    artifact.write_bytes(b"\x7fELFshared")
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    # Three GEMM shapes that all resolve to the same code object (scan identities,
    # no entry offset). They must trigger exactly one object-level invocation and
    # never attribute the hazard to a fabricated GEMM kernel name.
    worklist = KernelWorklist(
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=3,
        kernels=tuple(
            KernelObservation(
                identity=KernelIdentity(
                    name=f"gemm_{shape}",
                    target="gfx950",
                    code_object=str(artifact),
                    code_object_sha256=sha,
                    code_object_index=0,
                ),
                total_time_ms=0.0,
                dispatch_count=count,
                sources=("gemm_csv",),
            )
            for shape, count in (("a", 3), ("b", 2), ("c", 1))
        ),
    )
    calls: list[tuple[str, ...]] = []

    def execute(argv, *, timeout_seconds, env=None):
        calls.append(tuple(argv))
        output = "\n".join(
            (
                f"{artifact}:gfx950[0]: instructions=10 memory-events=2 diagnostics=1",
                f"{artifact}:gfx950[0]:.text+0x20: missing s_waitcnt lgkmcnt(0)",
            )
        )
        return ProcessResult(tuple(argv), 4, output, "")

    result = run_waitcheck(
        worklist, output_dir=tmp_path / "out", binary=binary, execute=execute
    )

    assert len(calls) == 1
    assert len(result.kernel_results) == 1
    assert result.verdict is Verdict.WARN
    assert result.findings
    assert all(finding.kernel_name is None for finding in result.findings)


def test_waitcheck_rejects_changed_code_object(tmp_path: Path) -> None:
    binary = tmp_path / "rj_waitcheck"
    binary.write_text("binary")
    artifact = tmp_path / "kernel.hsaco"
    artifact.write_bytes(b"\x7fELF")
    identity = _exact_identity(artifact)
    artifact.write_bytes(b"\x7fELFchanged")

    result = run_waitcheck(
        _worklist(identity),
        output_dir=tmp_path / "out",
        binary=binary,
    )

    assert result.verdict is Verdict.ERROR
    assert result.kernel_results[0].reason == "code_object_digest_mismatch"
