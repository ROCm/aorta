from __future__ import annotations

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import (
    ConSanMode,
    ExecutionState,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    Verdict,
    evaluate_record_replay,
    parse_record_replay_output,
    scoped_consan_not_checked,
)
from aorta.instrumentation.rocjitsu_sanitizers import consan as consan_module
from aorta.instrumentation.rocjitsu_sanitizers.consan import run_consan
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult

_PREFIX = "[rocjitsu-dbi-hooks] ConSan"

# Coverage and verdict fields mirror RocJITsu's coverage gate at b4feaddd.


def _zero_counts(kind: str) -> str:
    return (
        f"{kind}_discovered=0 {kind}_supported=0 {kind}_selected=0 "
        f"{kind}_patched=0 {kind}_unsupported=0 {kind}_resource_failed=0 "
        f"{kind}_placement_or_lowering_failed=0 "
        f"{kind}_expert_limit_omitted=0"
    )


def _healthy_evidence() -> str:
    coverage = (
        f"{_PREFIX} coverage reader=1 load=1 flavor=moi engine=record_replay "
        "analysis_complete=true expert_limit=false "
        "access_discovered=2 access_supported=2 access_selected=2 "
        "access_patched=2 access_unsupported=0 access_resource_failed=0 "
        "access_placement_or_lowering_failed=0 access_expert_limit_omitted=0 "
        f"{_zero_counts('barrier')} {_zero_counts('atomic')} {_zero_counts('fence')}"
    )
    sites = [
        (
            f"{_PREFIX} coverage_site reader=1 load=1 kind=access "
            "disposition=supported reason=none outcome=patched "
            "lowering_reason=none resource_reason=none container=k scope=kernel "
            f"text=0x{index:x} mnemonic=ds_read_b32"
        )
        for index in (4, 8)
    ]
    verdict = (
        f"{_PREFIX} analysis verdict applicable=true "
        "analysis_complete=true static_complete=true dynamic_complete=true "
        "applicable_code_objects=1 incomplete_code_objects=0 "
        "access=2/2 barrier=0/0 atomic=0/0 fence=0/0 "
        "visible_evidence=2 dynamic_incomplete=0 replay_unsupported_access=0 "
        "replay_unsupported_atomics=0 replay_unsupported_fences=0 "
        "replay_metadata_full=0"
    )
    return "\n".join((coverage, *sites, verdict))


def _worklist() -> KernelWorklist:
    return KernelWorklist(
        requirement=SelectionRequirement.TOP_TIME,
        top_n=1,
        kernels=(
            KernelObservation(
                identity=KernelIdentity(name="kernel", target="gfx950"),
                total_time_ms=1,
                dispatch_count=1,
                sources=("test",),
            ),
        ),
    )


def test_record_replay_detail_and_summary_count_once() -> None:
    output = "\n".join(
        [
            (
                f"{_PREFIX} MOI auto replay diagnostic kind=1 "
                "first_lds=[0,4) second_lds=[0,4) conflict=true diagnostics=1"
            ),
            f"{_PREFIX} MOI auto replay conflict=true diagnostics=1",
            _healthy_evidence(),
        ]
    )

    parsed = parse_record_replay_output(output)

    assert len(parsed.consan_findings) == 1


def test_summary_for_another_reader_is_not_suppressed() -> None:
    output = "\n".join(
        [
            f"{_PREFIX} MOI auto replay diagnostic reader=1 index=0 kind=1",
            f"{_PREFIX} MOI auto replay reader=1 diagnostics=1 conflict=true",
            f"{_PREFIX} MOI auto replay reader=2 diagnostics=1 conflict=true",
            _healthy_evidence(),
        ]
    )

    parsed = parse_record_replay_output(output)

    assert len(parsed.consan_findings) == 2


def test_benign_inventory_diagnostics_are_not_races() -> None:
    output = "\n".join(
        [
            f"{_PREFIX} MOI auto report plan reader=1 diagnostics=2 access_ranges=2",
            f"{_PREFIX} MOI auto replay diagnostics=0 conflict=false",
            _healthy_evidence(),
        ]
    )

    parsed = parse_record_replay_output(output)

    assert parsed.consan_findings == ()


def test_combined_waitcheck_is_reported_separately() -> None:
    output = "\n".join(
        [
            "rocjitsu-waitcheck: .text+0x40: missing s_wait_loadcnt <= 0",
            "rocjitsu-waitcheck: consumer: v_mov_b32",
            f"{_PREFIX} MOI auto replay diagnostics=0 conflict=false",
            _healthy_evidence(),
        ]
    )

    waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert waitcheck.verdict is Verdict.WARN
    assert len(waitcheck.findings) == 1
    assert consan.verdict is Verdict.PASS


def test_waitcheck_summary_and_detail_are_not_double_counted() -> None:
    output = "\n".join(
        [
            "rocjitsu-waitcheck: ConSan preflight reported reader=1 "
            "target=gfx950 reason=wait-hazard diagnostics=1 action=continue",
            "rocjitsu-waitcheck: .text+0x40: missing s_wait_loadcnt <= 0",
            "rocjitsu-waitcheck: consumer: v_mov_b32",
            _healthy_evidence(),
        ]
    )

    waitcheck, _consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert len(waitcheck.findings) == 1


def test_multiple_object_verdicts_are_preserved_and_reduced() -> None:
    output = (
        _healthy_evidence()
        .replace(
            "analysis_complete=true static_complete=true dynamic_complete=true",
            "analysis_complete=false static_complete=true dynamic_complete=false",
        )
        .replace(
            "visible_evidence=2 dynamic_incomplete=0",
            "visible_evidence=2 dynamic_incomplete=1",
        )
    )

    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert len(consan.coverage) == 1
    assert consan.state is ExecutionState.ERROR
    assert consan.verdict is Verdict.ERROR
    assert "consan_coverage_incomplete" in str(consan.reason)


def test_strict_rejection_never_passes() -> None:
    waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 92, "", "ConSan load rejection"),
        strict=True,
    )

    assert waitcheck.verdict is Verdict.ERROR
    assert consan.verdict is Verdict.ERROR
    assert consan.reason == "consan_strict_load_rejection"


def test_timeout_never_passes() -> None:
    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), None, "", "", timed_out=True)
    )

    assert consan.state is ExecutionState.TIMED_OUT
    assert consan.verdict is Verdict.ERROR


def test_missing_verdict_never_passes() -> None:
    output = f"{_PREFIX} MOI auto replay diagnostics=0 conflict=false"

    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert consan.state is ExecutionState.ERROR
    assert "missing ConSan coverage record" in str(consan.reason)


def test_malformed_coverage_never_passes() -> None:
    output = _healthy_evidence().replace(
        "replay_unsupported_access=0",
        "replay_unsupported_access=not-a-number",
    )

    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert consan.verdict is Verdict.ERROR
    assert "parse_error" in str(consan.reason)


def test_inconsistent_aggregate_coverage_never_passes() -> None:
    output = _healthy_evidence().replace("access=2/2", "access=1/2")

    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert consan.verdict is Verdict.ERROR
    assert "aggregate disagrees" in str(consan.reason)


def test_strict_mode_relies_on_backend_exit_and_coverage_gate() -> None:
    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 0, _healthy_evidence(), ""),
        strict=True,
    )

    assert consan.verdict is Verdict.PASS


def test_only_record_replay_is_exposed() -> None:
    with pytest.raises(ValueError):
        ConSanMode("inline-shadow")


def test_top_k_consan_is_fail_closed_without_command() -> None:
    result = scoped_consan_not_checked(_worklist())

    assert result.state is ExecutionState.NOT_CHECKED
    assert result.verdict is Verdict.NOT_CHECKED
    assert "consan_command_not_provisioned" in str(result.reason)


def _capture_consan_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    consan_log: bool,
) -> dict[str, str]:
    hook = tmp_path / "librocjitsu_dbi_hooks.so"
    hook.write_bytes(b"")
    command = tmp_path / "repro"
    command.write_bytes(b"")
    monkeypatch.delenv("RJ_CONSAN_LOG", raising=False)
    captured: dict[str, str] = {}

    def fake_run_argv(argv, *, timeout_seconds, env):
        captured.update(env)
        return ProcessResult(tuple(argv), 0, _healthy_evidence(), "")

    monkeypatch.setattr(consan_module, "run_argv", fake_run_argv)
    result = run_consan(
        _worklist(),
        command=command,
        hook_lib=hook,
        output_dir=tmp_path / "out",
        consan_log=consan_log,
    )
    assert result.state is ExecutionState.RAN
    return captured


def test_run_consan_requests_debug_log_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The strict coverage cross-check needs per-site coverage_site records, which
    # the hook only emits at its debug level (kLogDebug=3). A boolean-truthy
    # RJ_CONSAN_LOG=1 (kLogInfo) omits them and would fail closed on a clean run.
    env = _capture_consan_env(monkeypatch, tmp_path, consan_log=True)

    assert "RJ_CONSAN_LOG" in env
    assert env["RJ_CONSAN_LOG"] != "1"
    assert int(env["RJ_CONSAN_LOG"]) >= 3


def test_run_consan_omits_log_env_when_logging_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    env = _capture_consan_env(monkeypatch, tmp_path, consan_log=False)

    assert "RJ_CONSAN_LOG" not in env


def test_combined_waitcheck_analysis_failure_never_passes() -> None:
    output = "\n".join(
        (
            "rocjitsu-waitcheck: ConSan preflight reported reader=1 "
            "target=gfx950 reason=analysis-failed action=continue",
            _healthy_evidence(),
        )
    )

    waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert waitcheck.verdict is Verdict.ERROR
    assert "analysis_failed" in str(waitcheck.reason)
    assert consan.verdict is Verdict.PASS
