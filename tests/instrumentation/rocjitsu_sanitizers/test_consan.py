from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import (
    ConSanMode,
    ExecutionState,
    FindingSeverity,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    Verdict,
    evaluate_consan_output,
    evaluate_record_replay,
    parse_consan_output,
    parse_record_replay_output,
    scoped_consan_not_checked,
)
from aorta.instrumentation.rocjitsu_sanitizers import consan as consan_module
from aorta.instrumentation.rocjitsu_sanitizers.consan import run_consan
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocm_paths import (
    LAYOUT_WHEEL,
    WHEEL_CORE_PACKAGE,
    RocmRoots,
)

_PREFIX = "[rocjitsu-dbi-hooks] ConSan"

# Coverage and verdict fields mirror RocJITsu's coverage gate at b4feaddd.


def _zero_counts(kind: str) -> str:
    return (
        f"{kind}_discovered=0 {kind}_supported=0 {kind}_selected=0 "
        f"{kind}_patched=0 {kind}_unsupported=0 {kind}_resource_failed=0 "
        f"{kind}_placement_or_lowering_failed=0 "
        f"{kind}_expert_limit_omitted=0"
    )


def _healthy_evidence(*, engine: str = "record_replay") -> str:
    coverage = (
        f"{_PREFIX} coverage reader=1 load=1 flavor=moi engine={engine} "
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


# The three Sampled log shapes below are transcribed from RocJITsu's renderer
# (rj_hsa_dbi_moi_sampled_report_renderer.cpp at 164c20fae8c). Two of them look
# alike on purpose: the conflict record and the benign per-watchpoint evidence
# record share the "auto sampled" stem, and only the first is a race.


def _sampled_report(
    *,
    reader: int = 1,
    conflicts: int = 0,
    immediate: int = 0,
    examples: int = 0,
    pairs_without_example: int = 0,
) -> str:
    """One per-reader ``auto report`` line with the Sampled counters appended.

    Only the four conflict counters are load-bearing here; the rest of the line
    is carried so the parser is exercised against a realistically wide record.
    """
    return (
        f"{_PREFIX} MOI auto report reader={reader} addr=0x7f1200000000 bytes=65536 "
        "generation=3 code_object=b1946ac92492d234 event_counter=0 access_records=2 "
        "visible_records=2 dropped_records=0 capacity=1024 barrier_records=0 "
        "visible_barriers=0 dropped_barriers=0 barrier_capacity=256 atomic_records=0 "
        "visible_atomics=0 dropped_atomics=0 atomic_capacity=256 fence_records=0 "
        "visible_fences=0 dropped_fences=0 fence_capacity=256 diagnostics=0 "
        "visible_diagnostics=0 dropped_diagnostics=0 diagnostic_capacity=8 "
        "sampled_watchpoints=64 visible_sampled=2 sampled_sync_capacity=64 "
        f"sampled_conflicts={conflicts} sampled_immediate_conflicts={immediate} "
        "sampled_claimed_windows=2 sampled_dropped_windows=0 sampled_saturated_windows=0 "
        f"sampled_conflict_examples={examples} "
        f"sampled_conflict_pairs_without_example={pairs_without_example} fine_grained=false"
    )


def _sampled_conflict(*, reader: int = 1, first_index: int = 0, second_index: int = 1) -> str:
    return (
        f"{_PREFIX} MOI auto sampled conflict reader={reader} first_index={first_index} "
        f"second_index={second_index} first_kind=1 second_kind=2 first_owner=0 "
        "second_owner=1 epoch=2 generation=3 first_bytes=[0,4) second_bytes=[0,4) "
        "code_object=b1946ac92492d234 first_instruction=0x40 second_instruction=0x48 "
        "dispatch=0x1 workgroup=(0,0,0) cluster_workgroup=0 "
        "first_lanes=0x000000000000000f second_lanes=0x00000000000000f0"
    )


def _sampled_evidence(*, reader: int = 1, index: int = 0) -> str:
    """A benign retained-watchpoint record. Evidence of coverage, not a race."""
    return (
        f"{_PREFIX} MOI auto sampled reader={reader} index={index} kind=1 owner=0 "
        "epoch=2 generation=3 bytes=[0,4) consumed=false dispatch=0x1 "
        "workgroup=(0,0,0) instruction=0x40 trampoline=0x180 relocated_guest=0x200 "
        "scratch_vgpr=6 range=0 bank=0 mapped=true sync_class=0 sync_kind=0 "
        "sync_role=0 sync_scope=0 sync_outcome=0 sync_address=0x0 sync_bytes=0 "
        "sync_epochs=0/0"
    )


def _attention_evidence(*, access_sites: int = 3) -> str:
    """Evidence shaped like TokenSpeed's Gluon attention kernels (issue #405).

    Every discovered site is reported supported and then fails to lower, and the
    hook itemizes the access sites but never the barrier sites it counted.
    """

    coverage = (
        f"{_PREFIX} coverage reader=1 load=1 flavor=moi engine=record_replay "
        "analysis_complete=false expert_limit=false "
        "access_discovered=3 access_supported=3 access_selected=3 "
        "access_patched=0 access_unsupported=0 access_resource_failed=0 "
        "access_placement_or_lowering_failed=3 access_expert_limit_omitted=0 "
        "barrier_discovered=2 barrier_supported=2 barrier_selected=2 "
        "barrier_patched=0 barrier_unsupported=0 barrier_resource_failed=0 "
        "barrier_placement_or_lowering_failed=2 barrier_expert_limit_omitted=0 "
        f"{_zero_counts('atomic')} {_zero_counts('fence')}"
    )
    sites = [
        (
            f"{_PREFIX} coverage_site reader=1 load=1 kind=access "
            "disposition=supported reason=none "
            "outcome=placement_or_lowering_failed "
            "lowering_reason=instrumentation_patch_missing resource_reason=none "
            f"container=k scope=kernel text=0x{index:x} mnemonic=ds_read_b64_tr_b16"
        )
        for index in range(access_sites)
    ]
    verdict = (
        f"{_PREFIX} analysis verdict applicable=true "
        "analysis_complete=false static_complete=false dynamic_complete=true "
        "applicable_code_objects=1 incomplete_code_objects=1 "
        "access=0/3 barrier=0/2 atomic=0/0 fence=0/0 "
        "visible_evidence=0 dynamic_incomplete=0 replay_unsupported_access=0 "
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


def test_clean_sampled_evidence_is_not_a_race() -> None:
    # Retained-watchpoint records are how Sampled shows it was watching. They
    # share the "auto sampled" stem with the conflict record, so a parser that
    # keyed off the stem would report every clean run as a race.
    output = "\n".join(
        (
            _sampled_evidence(index=0),
            _sampled_evidence(index=1),
            _sampled_report(),
            _healthy_evidence(engine="sampled"),
        )
    )

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.findings == ()
    assert consan.verdict is Verdict.PASS


def test_sampled_log_limit_line_is_not_a_race() -> None:
    # Past 64 retained watchpoints the hook stops itemizing and says so. That
    # line is an omission notice about benign evidence, not a conflict.
    output = "\n".join(
        (
            f"{_PREFIX} MOI auto sampled reader=1 omitted=12 after log limit=64",
            _sampled_report(),
            _healthy_evidence(engine="sampled"),
        )
    )

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.findings == ()
    assert consan.verdict is Verdict.PASS


def test_sampled_conflict_records_are_races() -> None:
    output = "\n".join(
        (
            _sampled_conflict(first_index=0, second_index=1),
            _sampled_conflict(first_index=2, second_index=3),
            _sampled_report(conflicts=2, examples=2),
            _healthy_evidence(engine="sampled"),
        )
    )

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.verdict is Verdict.FAIL
    assert [finding.code for finding in consan.findings] == ["sampled_conflict"] * 2
    assert all(finding.severity is FindingSeverity.RACE for finding in consan.findings)
    assert all(finding.code_object == "b1946ac92492d234" for finding in consan.findings)


def test_sampled_summary_shortfall_adds_exactly_one_finding() -> None:
    # Three conflicts, one example logged: the other two are visible only as a
    # count, and must not be reported as two separate unexplained races either.
    output = "\n".join(
        (
            _sampled_conflict(),
            _sampled_report(conflicts=3, examples=1, pairs_without_example=2),
            _healthy_evidence(engine="sampled"),
        )
    )

    parsed = parse_consan_output(output)

    codes = [finding.code for finding in parsed.consan_findings]
    assert codes.count("sampled_conflict") == 1
    assert codes.count("sampled_conflict_summary") == 1
    assert len(parsed.consan_findings) == 2
    summary = next(
        finding for finding in parsed.consan_findings if finding.code == "sampled_conflict_summary"
    )
    # A dashboard reader has to be able to tell "3 races, 1 shown" from "1 race".
    assert "summary only" in summary.message
    metadata = dict(summary.metadata)
    assert metadata.get("reader") == "1"
    assert metadata.get("sampled_conflicts") == "3"
    assert metadata.get("sampled_conflict_pairs_without_example") == "2"
    assert metadata.get("itemized_conflicts") == "1"


def test_sampled_conflict_with_no_example_is_still_visible() -> None:
    output = "\n".join(
        (
            _sampled_report(conflicts=1, examples=0, pairs_without_example=1),
            _healthy_evidence(engine="sampled"),
        )
    )

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.verdict is Verdict.FAIL
    assert [finding.code for finding in consan.findings] == ["sampled_conflict_summary"]


def test_sampled_conflicts_accumulate_across_a_readers_reports() -> None:
    # One report per published snapshot, so a reader's conflict count is the sum.
    # Two conflicts counted, one example logged -> one summary-only finding,
    # even though each individual report claimed no shortfall of its own.
    output = "\n".join(
        (
            _sampled_conflict(),
            _sampled_report(conflicts=1, examples=1),
            _sampled_report(conflicts=1, examples=1),
            _healthy_evidence(engine="sampled"),
        )
    )

    parsed = parse_consan_output(output)

    codes = [finding.code for finding in parsed.consan_findings]
    assert codes.count("sampled_conflict_summary") == 1
    assert len(parsed.consan_findings) == 2


def test_sampled_immediate_conflicts_are_reported_separately() -> None:
    # A device-side counter, not a restatement of the host-side analysis: it is
    # what remains visible when the evidence window itself was dropped.
    output = "\n".join(
        (
            _sampled_report(conflicts=0, immediate=3),
            _healthy_evidence(engine="sampled"),
        )
    )

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.verdict is Verdict.FAIL
    assert [finding.code for finding in consan.findings] == ["sampled_immediate_conflict"]
    assert dict(consan.findings[0].metadata).get("sampled_immediate_conflicts") == "3"


def test_sampled_summary_and_immediate_conflicts_are_not_summed() -> None:
    # Two counters of different things: one finding each, distinctly coded, so
    # neither double-counts the other nor cancels it.
    output = "\n".join(
        (
            _sampled_report(conflicts=2, examples=0, pairs_without_example=2, immediate=1),
            _healthy_evidence(engine="sampled"),
        )
    )

    parsed = parse_consan_output(output)

    assert [finding.code for finding in parsed.consan_findings] == [
        "sampled_conflict_summary",
        "sampled_immediate_conflict",
    ]


def test_sampled_summary_for_another_reader_is_not_suppressed() -> None:
    output = "\n".join(
        (
            _sampled_conflict(reader=1),
            _sampled_report(reader=1, conflicts=1, examples=1),
            _sampled_report(reader=2, conflicts=1, examples=0, pairs_without_example=1),
            _healthy_evidence(engine="sampled"),
        )
    )

    parsed = parse_consan_output(output)

    summaries = [
        finding for finding in parsed.consan_findings if finding.code == "sampled_conflict_summary"
    ]
    assert [dict(finding.metadata)["reader"] for finding in summaries] == ["2"]
    assert len(parsed.consan_findings) == 2


def test_sampled_conflict_without_a_fingerprint_or_instruction_parses() -> None:
    # The Sampled renderer has no "missing" fallback for an empty fingerprint,
    # and reports an instruction it cannot pin down as a word rather than a
    # number. None of that is malformed output.
    conflict = (
        _sampled_conflict()
        .replace("code_object=b1946ac92492d234 ", "code_object= ")
        .replace("first_instruction=0x40", "first_instruction=unavailable")
        .replace("second_instruction=0x48", "second_instruction=ambiguous")
        .replace("first_lanes=0x000000000000000f", "first_lanes=unavailable")
    )
    output = "\n".join(
        (
            conflict,
            _sampled_report(conflicts=1, examples=1),
            _healthy_evidence(engine="sampled"),
        )
    )

    parsed = parse_consan_output(output)

    (finding,) = parsed.consan_findings
    assert finding.code_object is None
    assert dict(finding.metadata).get("first_instruction") == "unavailable"
    assert dict(finding.metadata).get("first_lanes") == "unavailable"


def test_malformed_sampled_counts_never_pass() -> None:
    output = "\n".join(
        (
            _sampled_report().replace("sampled_conflicts=0", "sampled_conflicts=abc"),
            _healthy_evidence(engine="sampled"),
        )
    )

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.state is ExecutionState.ERROR
    assert str(consan.reason).startswith("consan_output_parse_error:")


def test_truncated_sampled_summary_never_passes() -> None:
    # The counters come from one format string, so a line carrying some and not
    # others is a truncated log. Treating the absent ones as zero is the reading
    # that would buy a PASS.
    report = _sampled_report(conflicts=2)
    output = "\n".join(
        (
            report[: report.index(" sampled_conflict_examples=")],
            _healthy_evidence(engine="sampled"),
        )
    )

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.state is ExecutionState.ERROR
    assert str(consan.reason).startswith("consan_output_parse_error:")


@pytest.mark.parametrize(
    "broken",
    (
        _sampled_report(conflicts=1, examples=2),
        _sampled_report(conflicts=2, examples=1, pairs_without_example=0),
        _sampled_report().replace(" reader=1 ", " "),
    ),
)
def test_inconsistent_sampled_summary_never_passes(broken: str) -> None:
    output = "\n".join((broken, _healthy_evidence(engine="sampled")))

    _waitcheck, consan = evaluate_consan_output(ProcessResult(("app",), 0, output, ""))

    assert consan.state is ExecutionState.ERROR
    assert str(consan.reason).startswith("consan_output_parse_error:")


def test_legacy_and_sampled_evidence_do_not_cross_contaminate() -> None:
    # One log, both engines -- the shape of a saved log from before the
    # migration replayed next to a current one. The Sampled detail record must
    # not be credited against the Record/Replay summary's diagnostic count,
    # which would silently drop the legacy summary-only conflict.
    output = "\n".join(
        (
            f"{_PREFIX} MOI auto replay diagnostic reader=1 index=0 kind=1",
            f"{_PREFIX} MOI auto replay reader=1 diagnostics=2 conflict=true",
            _sampled_conflict(reader=1),
            _sampled_report(reader=1, conflicts=2, examples=1, pairs_without_example=1),
            _healthy_evidence(engine="sampled"),
        )
    )

    parsed = parse_consan_output(output)

    codes = [finding.code for finding in parsed.consan_findings]
    assert codes.count("record_replay_conflict_summary") == 1
    assert codes.count("sampled_conflict") == 1
    assert codes.count("sampled_conflict_summary") == 1
    assert len(parsed.consan_findings) == 4


def test_racy_baseline_finding_shape_matches_every_sampled_message() -> None:
    """The nightly gate matches ``finding_shape`` as a plain substring.

    So the declared shape is checked against messages the parser really emits
    rather than against hand-copied fixture text, and against all three ways
    Sampled states a race -- a racy run that happened to log no example record
    must still satisfy the gate instead of turning it red.
    """
    baselines = (
        Path(__file__).resolve().parents[3]
        / "recipes"
        / "sanitizers"
        / "fixtures"
        / "expected"
        / "verdict_baselines.json"
    )
    shape = json.loads(baselines.read_text(encoding="utf-8"))["consan_racy"]["finding_shape"][
        "consan"
    ]
    output = "\n".join(
        (
            _sampled_conflict(),
            _sampled_report(conflicts=3, examples=1, pairs_without_example=2, immediate=1),
            _healthy_evidence(engine="sampled"),
        )
    )

    findings = parse_consan_output(output).consan_findings

    assert len(findings) == 3
    assert [finding.message for finding in findings if shape not in finding.message] == []


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


def _loader_failure(soname: str = "libamdhip64.so.7") -> str:
    return (
        f"/repro: error while loading shared libraries: {soname}: "
        "cannot open shared object file: No such file or directory"
    )


_LOADER_FAILURE = _loader_failure()


def _pin_rocm_roots(
    monkeypatch: pytest.MonkeyPatch, core: Path, libraries: Path
) -> tuple[Path, Path]:
    """Pin the resolver to a wheel-layout tree rooted at ``core``/``libraries``.

    Returns the two lib dirs the diagnostic will report, core first.
    """
    monkeypatch.setattr(
        consan_module,
        "resolve_rocm_roots",
        lambda: RocmRoots(
            core=core,
            libraries=libraries,
            include=core,
            layout=LAYOUT_WHEEL,
            source=f"import:{WHEEL_CORE_PACKAGE}",
        ),
    )
    return core / "lib", libraries / "lib"


def _rocm_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path, *, holds: tuple[str, ...]
) -> tuple[Path, Path]:
    """Pin the resolver to a wheel-layout tree holding exactly ``holds``.

    The diagnostic now reads the filesystem to decide whether its remedy would
    work, so the branch under test must not depend on what ROCm the developer
    running pytest happens to have installed.
    """
    core = tmp_path / "_rocm_sdk_core"
    libraries = tmp_path / "_rocm_sdk_libraries"
    for lib_dir in (core / "lib", libraries / "lib"):
        lib_dir.mkdir(parents=True)
    for name in holds:
        (core / "lib" / name).write_bytes(b"")
    return _pin_rocm_roots(monkeypatch, core, libraries)


def test_loader_failure_is_reported_as_a_missing_library_not_a_bare_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Exit 127 plus the loader message means the repro never reached main.

    Measured on the wheel-layout ROCm base: hipcc output carries no RPATH or
    RUNPATH and the image sets no LD_LIBRARY_PATH, so a repro that builds fine
    dies before main. Without this the run reports only
    ``combined_hook_exit_127``, which reads like a sanitizer verdict and points
    at nothing.
    """
    core_lib, lib = _rocm_tree(monkeypatch, tmp_path, holds=("libamdhip64.so.7",))

    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 127, "", _LOADER_FAILURE)
    )

    reason = str(consan.reason)
    # The machine-readable token still leads, so consumers keying off it are
    # unaffected by the added prose.
    assert reason.startswith("combined_hook_exit_127")
    assert "libamdhip64.so.7" in reason
    # Both dirs, core first: the HIP runtime hangs off core, the math libraries
    # off libraries, and a hipcc-built repro can need either.
    assert f"append {core_lib}{os.pathsep}{lib} to LD_LIBRARY_PATH" in reason
    # And it says why aorta will not simply fix it, so the next reader does not
    # "fix" it by mutating the child environment.
    assert "environment of the process under test" in reason
    assert consan.verdict is Verdict.ERROR


def test_a_dependency_the_rocm_tree_does_not_hold_gets_no_rocm_remedy(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An unrelated missing library must not be answered with a ROCm lib path.

    ``libstdc++.so.6`` is not something appending the ROCm lib dirs will supply,
    so prescribing them would be authoritative and wrong -- it sends the reader
    after the wrong library. The soname is still named, because "the repro never
    started, and here is what it wanted" is the part that was missing from a
    bare ``combined_hook_exit_127``.
    """
    core_lib, _lib = _rocm_tree(monkeypatch, tmp_path, holds=("libamdhip64.so.7",))

    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 127, "", _loader_failure("libstdc++.so.6"))
    )

    reason = str(consan.reason)
    assert reason.startswith("combined_hook_exit_127")
    assert "libstdc++.so.6" in reason
    assert "missing dependency of the repro" in reason
    # The dirs are named as what was SEARCHED, never as a remedy to apply.
    assert f"not in the resolved ROCm lib dirs ({core_lib}" in reason
    assert "append" not in reason
    assert "wheel-layout" not in reason


def test_a_path_bearing_soname_gets_no_rocm_remedy_even_when_it_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A ``DT_NEEDED`` entry with a separator is a path, and paths are not searched.

    The loader resolves such an entry directly and never consults
    ``LD_LIBRARY_PATH`` for it, so the "append the ROCm lib dirs" remedy cannot
    possibly apply -- however present the file is. The HRX recipes link
    absolute-path libraries, so this is a shape the diagnostic really meets.

    The decoy has to EXIST for this test to bite. ``Path("/a/lib") / "/b/x.so"``
    is ``/b/x.so`` -- an absolute right-hand side replaces the root instead of
    joining onto it -- so without the separator check the probe walks straight
    out of the tree it was describing, lands on the real file, and prescribes
    the authoritative remedy for a name that remedy cannot reach.
    """
    absolute = tmp_path / "opt" / "hrx" / "lib" / "libamdhip64.so.7"
    absolute.parent.mkdir(parents=True)
    absolute.write_bytes(b"")
    # The ROCm tree itself holds nothing, so the only file that can satisfy the
    # probe is the decoy, reachable only via the root-replacing join.
    _rocm_tree(monkeypatch, tmp_path, holds=())

    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 127, "", _loader_failure(str(absolute)))
    )

    reason = str(consan.reason)
    assert str(absolute) in reason
    assert "missing dependency of the repro" in reason
    assert "append" not in reason


def test_lib_dirs_that_resolved_but_are_not_directories_are_still_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """"Nothing to search" must say WHAT was resolved, not just that it failed.

    The resolver answered here -- these are the paths it produced -- and neither
    is a directory; a stale mount is the realistic case, which
    ``rocm_paths.safe_is_dir`` collapses to False module-wide. Naming only the
    resolver source reads as "no ROCm install was found" when the truth is
    "found, and it is not there any more", and leaves the operator nothing to
    stat.
    """
    absent = tmp_path / "stale"
    core_lib, lib = _pin_rocm_roots(monkeypatch, absent / "core", absent / "libraries")

    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 127, "", _LOADER_FAILURE)
    )

    reason = str(consan.reason)
    assert str(core_lib) in reason
    assert str(lib) in reason
    assert f"resolver source=import:{WHEEL_CORE_PACKAGE}" in reason
    # Still the missing-dependency arm: an unreachable lib dir is not grounds
    # for the ROCm library-path remedy.
    assert "missing dependency of the repro" in reason
    assert "append" not in reason


def test_a_bare_exit_127_gets_no_invented_library_hint() -> None:
    """A repro is free to exit 127 for its own reasons.

    The diagnostic requires the loader message as well as the code, so an exit
    this module cannot explain keeps exactly the reason it always had.
    """
    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 127, "", "boom"))

    assert str(consan.reason) == "combined_hook_exit_127"


def test_the_loader_message_alone_does_not_trigger_the_hint() -> None:
    """The message on stderr without exit 127 is not a launch failure either."""
    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 3, "", _LOADER_FAILURE)
    )

    assert str(consan.reason) == "combined_hook_exit_3"


def test_the_loader_message_on_stdout_is_never_read_as_a_launch_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A repro that PRINTS the loader text and exits 127 gets no hint.

    ``run_argv`` captures the two streams separately and ld.so writes to stderr,
    so text on stdout is by construction the repro's own output. Reading it
    would infer a launch failure from program output -- the exact false positive
    ``_launch_diagnostic`` documents it must not admit -- and the remedy check
    does not save us here, because the soname named is one the ROCm tree really
    does hold.
    """
    _rocm_tree(monkeypatch, tmp_path, holds=("libamdhip64.so.7",))

    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 127, _LOADER_FAILURE, "")
    )

    assert str(consan.reason) == "combined_hook_exit_127"


def test_run_consan_never_injects_a_library_path_into_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The diagnostic is words only: the process under test keeps its own env.

    Prepending stock ROCm lib dirs here would hijack the LD_LIBRARY_PATH
    substitution the library-swap workflows in docs/ci-testing-plan.md depend
    on, so the repro must see exactly what aorta inherited.
    """
    monkeypatch.setenv("LD_LIBRARY_PATH", "/operator/substituted/lib")

    env = _capture_consan_env(monkeypatch, tmp_path, consan_log=True)

    assert env["LD_LIBRARY_PATH"] == "/operator/substituted/lib"


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


def test_unitemized_site_kind_is_a_coverage_gap_not_a_parse_error() -> None:
    # The hook counts 2 barrier sites and itemizes none of them. That is a hole
    # in the evidence, not malformed output, so it must not be reported as if
    # aorta failed to read the log.
    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 0, _attention_evidence(), "")
    )

    assert consan.verdict is Verdict.ERROR
    assert "parse_error" not in str(consan.reason)
    assert "consan_coverage_incomplete" in str(consan.reason)
    assert "reader 1 barrier sites not itemized: 0 of 2" in str(consan.reason)


def test_unitemized_coverage_still_reports_why_sites_failed() -> None:
    # The actionable numbers -- 0 of 3 access sites patched, and the lowering
    # reason every one of them reported -- have to survive into the check, since
    # that is the whole story of the run.
    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 0, _attention_evidence(), "")
    )

    assert "access placement_or_lowering_failed: 3 instrumentation_patch_missing" in str(
        consan.reason
    )
    assert [item.access for item in consan.coverage] == ["0/3"]


def test_unitemized_sites_never_pass_even_when_the_counts_look_healthy() -> None:
    # Every count reconciles and the verdict claims complete analysis -- except
    # that the 2 barrier sites the hook says it patched were never itemized, so
    # nothing corroborates them. This is the case where downgrading the parse
    # error could have bought a PASS, and it must not: coverage that was not
    # seen is not trusted, however healthy the aggregate looks.
    output = _healthy_evidence().replace(
        _zero_counts("barrier"),
        "barrier_discovered=2 barrier_supported=2 barrier_selected=2 "
        "barrier_patched=2 barrier_unsupported=0 barrier_resource_failed=0 "
        "barrier_placement_or_lowering_failed=0 barrier_expert_limit_omitted=0",
    ).replace("barrier=0/0", "barrier=2/2")

    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert consan.verdict is Verdict.ERROR
    assert "reader 1 barrier sites not itemized: 0 of 2" in str(consan.reason)


def test_partially_itemized_site_kind_is_still_a_parse_error() -> None:
    # One missing site out of three is lossy output, not a reportable gap: the
    # remaining records cannot be reconciled, so this must stay fail-closed on
    # the parse path.
    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 0, _attention_evidence(access_sites=2), "")
    )

    assert consan.verdict is Verdict.ERROR
    assert "parse_error" in str(consan.reason)
    assert "access site count mismatch" in str(consan.reason)


@pytest.mark.parametrize("kind", ("access", "barrier", "atomic", "fence"))
def test_not_applicable_site_is_excluded_from_discovered_count(kind: str) -> None:
    # The hook itemizes policy decisions that it deliberately excludes from
    # *_discovered. Those debug records must not make otherwise self-consistent
    # coverage look malformed (nightly run 35189763335).
    output = "\n".join(
        (
            _healthy_evidence(),
            (
                f"{_PREFIX} coverage_site reader=1 load=1 kind={kind} "
                "disposition=not_applicable reason=operation_kind_excluded "
                "outcome=not_applicable lowering_reason=semantic_not_applicable "
                "resource_reason=none container=k scope=kernel text=0xc "
                "mnemonic=unknown"
            ),
        )
    )

    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""), strict=True)

    assert consan.verdict is Verdict.PASS


def test_race_in_an_unitemized_run_is_not_discarded() -> None:
    # A race found while coverage was incomplete is still a race. The old parse
    # error threw the findings away with the rest of the parsed output.
    output = "\n".join(
        (
            f"{_PREFIX} MOI auto replay diagnostic kind=1 conflict=true diagnostics=1",
            _attention_evidence(),
        )
    )

    _waitcheck, consan = evaluate_record_replay(ProcessResult(("app",), 0, output, ""))

    assert consan.verdict is Verdict.FAIL
    assert len(consan.findings) == 1


def test_strict_mode_relies_on_backend_exit_and_coverage_gate() -> None:
    _waitcheck, consan = evaluate_record_replay(
        ProcessResult(("app",), 0, _healthy_evidence(), ""),
        strict=True,
    )

    assert consan.verdict is Verdict.PASS


def test_supported_consan_modes_are_exposed() -> None:
    assert ConSanMode.SAMPLED.value == "sampled"
    # Retained so a pre-migration bundle or saved log still names a mode this
    # module reads, even though aorta no longer requests it.
    assert ConSanMode.RECORD_REPLAY.value == "record-replay"
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
    assert result.consan.state is ExecutionState.RAN
    return captured


def test_run_consan_requests_debug_log_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The strict coverage cross-check needs per-site coverage_site records, which
    # the hook only emits at its debug level (kLogDebug=3). A boolean-truthy
    # RJ_CONSAN_LOG=1 (kLogInfo) omits them, so every kind would report as
    # un-itemized and an otherwise clean run would fail closed.
    env = _capture_consan_env(monkeypatch, tmp_path, consan_log=True)

    assert "RJ_CONSAN_LOG" in env
    assert env["RJ_CONSAN_LOG"] != "1"
    assert int(env["RJ_CONSAN_LOG"]) >= 3


def test_run_consan_omits_log_env_when_logging_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    env = _capture_consan_env(monkeypatch, tmp_path, consan_log=False)

    assert "RJ_CONSAN_LOG" not in env


def test_run_consan_scrubs_inherited_log_env_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # A stray RJ_CONSAN_LOG in the parent environment must not leak through when
    # the recipe disables logging; otherwise logging (and coverage strictness)
    # would be non-deterministic.
    hook = tmp_path / "librocjitsu_dbi_hooks.so"
    hook.write_bytes(b"")
    command = tmp_path / "repro"
    command.write_bytes(b"")
    monkeypatch.setenv("RJ_CONSAN_LOG", "1")
    captured: dict[str, str] = {}

    def fake_run_argv(argv, *, timeout_seconds, env):
        captured.update(env)
        return ProcessResult(tuple(argv), 0, _healthy_evidence(), "")

    monkeypatch.setattr(consan_module, "run_argv", fake_run_argv)
    run_consan(
        _worklist(),
        command=command,
        hook_lib=hook,
        output_dir=tmp_path / "out",
        consan_log=False,
    )

    assert "RJ_CONSAN_LOG" not in captured


def test_run_consan_pins_sampled_mode_and_the_max_preset(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The nightly racy case is a positive control, so the gate needs stride 1 on
    # both axes: at the hook's ordinary 1/256 default a known race can be missed
    # statistically and reported as a clean pass.
    env = _capture_consan_env(monkeypatch, tmp_path, consan_log=True)

    assert env.get("RJ_CONSAN_MODE") == "sampled"
    assert env.get("RJ_CONSAN_MOI_SAMPLED_PRESET") == "max"


# Every inherited setting that could weaken or invalidate the pinned gate.
# Spelled out rather than imported so dropping one from the source is a test
# failure, not a silent reopening of the weakening vector.
_SAMPLED_GATE_OVERRIDES = (
    "RJ_CONSAN_MOI_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES",
    "RJ_CONSAN_MOI_SAMPLE_STRIDE",
    "RJ_CONSAN_MOI_SAMPLE_OFFSET",
    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
    "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
    "RJ_CONSAN_MOI_WORKGROUP_SAMPLE_STRIDE",
    "RJ_CONSAN_MOI_WORKGROUP_SAMPLE_OFFSET",
    "RJ_CONSAN_MOI_CELL_SAMPLE_STRIDE",
    "RJ_CONSAN_MOI_CELL_SAMPLE_OFFSET",
    "RJ_CONSAN_MOI_SAMPLED_BANKS",
)


def test_run_consan_scrubs_inherited_sampled_gate_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # A preset supplies defaults only, so any one of these inherited from the
    # parent shell silently overrides max and thins the sampling back out --
    # and mixing the coupled with the per-axis selectors is a hard hook config
    # error. Pinning the preset without scrubbing leaves the gate
    # non-deterministic.
    for name in _SAMPLED_GATE_OVERRIDES:
        monkeypatch.setenv(name, "256")

    env = _capture_consan_env(monkeypatch, tmp_path, consan_log=True)

    assert [name for name in _SAMPLED_GATE_OVERRIDES if name in env] == []


def _multi_worklist(count: int) -> KernelWorklist:
    return KernelWorklist(
        requirement=SelectionRequirement.TOP_TIME,
        top_n=max(count, 1),
        kernels=tuple(
            KernelObservation(
                identity=KernelIdentity(name=f"kernel_{index}", target="gfx950"),
                total_time_ms=index + 1,
                dispatch_count=index + 1,
                sources=("test",),
            )
            for index in range(count)
        ),
    )


def _run_consan_with(monkeypatch, tmp_path, *, worklist, output, strict=False, target=None):
    hook = tmp_path / "librocjitsu_dbi_hooks.so"
    hook.write_bytes(b"")
    command = tmp_path / "repro"
    command.write_bytes(b"repro-binary")
    captured: dict[str, str] = {}

    def fake_run_argv(argv, *, timeout_seconds, env):
        captured.update(env)
        return ProcessResult(tuple(argv), 0, output, "")

    monkeypatch.setattr(consan_module, "run_argv", fake_run_argv)
    result = run_consan(
        worklist,
        command=command,
        hook_lib=hook,
        output_dir=tmp_path / "out",
        strict=strict,
        target=target,
    )
    return result, captured


def test_run_consan_empty_worklist_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    ran = False

    def fake_run_argv(argv, *, timeout_seconds, env):
        nonlocal ran
        ran = True
        return ProcessResult(tuple(argv), 0, _healthy_evidence(), "")

    monkeypatch.setattr(consan_module, "run_argv", fake_run_argv)
    command = tmp_path / "repro"
    command.write_bytes(b"")
    result = run_consan(
        _multi_worklist(0),
        command=command,
        hook_lib=tmp_path / "hook.so",
        output_dir=tmp_path / "out",
    )

    assert result.consan.state is ExecutionState.NOT_CHECKED
    assert "consan_requires_one_targeted_repro" in str(result.consan.reason)
    assert ran is False


def test_run_consan_multiple_kernels_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    result, _ = _run_consan_with(
        monkeypatch, tmp_path, worklist=_multi_worklist(2), output=_healthy_evidence()
    )
    assert result.consan.state is ExecutionState.NOT_CHECKED
    assert "consan_requires_one_targeted_repro" in str(result.consan.reason)


def test_run_consan_target_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    other = KernelIdentity(name="different", target="gfx950")
    result, _ = _run_consan_with(
        monkeypatch, tmp_path, worklist=_worklist(), output=_healthy_evidence(), target=other
    )
    assert result.consan.state is ExecutionState.NOT_CHECKED
    assert "consan_target_does_not_match_worklist" in str(result.consan.reason)


def test_run_consan_pins_policy_env_over_hostile_inheritance(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("RJ_CONSAN_POLICY", "off")
    monkeypatch.setenv("RJ_CONSAN_MODE", "inline-shadow")
    monkeypatch.delenv("HSA_TOOLS_DISABLE_REGISTER", raising=False)

    _, env = _run_consan_with(
        monkeypatch, tmp_path, worklist=_worklist(), output=_healthy_evidence(), strict=True
    )

    assert env.get("RJ_CONSAN_MODE") == ConSanMode.SAMPLED.value
    assert env["RJ_CONSAN_POLICY"] == "strict"
    assert env["HSA_TOOLS_DISABLE_REGISTER"] == "1"


def test_run_consan_default_policy_when_not_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _, env = _run_consan_with(
        monkeypatch, tmp_path, worklist=_worklist(), output=_healthy_evidence(), strict=False
    )
    assert env["RJ_CONSAN_POLICY"] == "default"


def test_run_consan_surfaces_preflight_and_attributes_kernel(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    output = "\n".join(
        (
            "rocjitsu-waitcheck: .text+0x40: missing s_wait_loadcnt <= 0",
            "rocjitsu-waitcheck: consumer: v_mov_b32",
            f"{_PREFIX} MOI auto replay diagnostics=0 conflict=false",
            _healthy_evidence(),
        )
    )
    result, _ = _run_consan_with(
        monkeypatch, tmp_path, worklist=_worklist(), output=output
    )

    assert result.consan.verdict is Verdict.PASS
    assert result.waitcheck_preflight.sanitizer == "waitcheck_preflight"
    assert result.waitcheck_preflight.verdict is Verdict.WARN
    # PASS is attributed to the single selected kernel, never vacuous.
    (kernel_result,) = result.consan.kernel_results
    assert kernel_result.identity.name == "kernel"
    backend = dict(result.consan.backend)
    assert "command_sha256" in backend
    assert "selected_identity_sha256" in backend


def test_object_coverage_static_complete_tracks_verdict() -> None:
    # The coverage record still reports analysis_complete=true, but the aggregate
    # analysis verdict is the authority for static completeness. A verdict that
    # reports static_complete=false must not surface a statically-complete object.
    output = _healthy_evidence().replace(
        "analysis_complete=true static_complete=true dynamic_complete=true",
        "analysis_complete=true static_complete=false dynamic_complete=true",
    )

    parsed = parse_record_replay_output(output)

    assert parsed.coverage
    assert parsed.coverage[0].static_complete is False
    assert parsed.coverage[0].analysis_complete is True


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
