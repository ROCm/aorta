from __future__ import annotations

import os
from pathlib import Path

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

    assert env["RJ_CONSAN_MODE"] == ConSanMode.RECORD_REPLAY.value
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
