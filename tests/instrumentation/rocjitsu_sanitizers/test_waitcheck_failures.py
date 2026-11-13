"""Recipe-to-report failure boundaries, using inert files and a fake executor."""
from __future__ import annotations

from functools import partial
import hashlib
from pathlib import Path

import pytest
import yaml

from aorta.instrumentation.rocjitsu_sanitizers import pipeline, waitcheck
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocjitsu_sanitizers.recipe import execute_sanitizer_run
from aorta.instrumentation.rocjitsu_sanitizers.report import read_report
from aorta.triage.recipe import RecipeSchemaError


@pytest.fixture
def setup_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Prevent discovery of the developer's installed tools in every test.
    monkeypatch.delenv("ROCJITSU_BUILD", raising=False)
    monkeypatch.setattr(waitcheck.shutil, "which", lambda name: None)
    bundle = tmp_path / "bundle"
    binary = bundle / "bin" / "rj_waitcheck"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"inert scanner; never execute")
    monkeypatch.setenv("ROCJITSU_PREBUILT", str(bundle))
    artifact = tmp_path / "control.co"
    artifact.write_bytes(b"inert code object; never execute")
    data = {
        "schema_version": 1, "mode": "sanitizer", "ticket": "FAILURE-BOUNDARY",
        "sanitizer_plan": {
            "target": "gfx1250", "source": {"kind": "kernel", "kernel": {
                "name": "control", "code_object": str(artifact),
                "code_object_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "code_object_index": 0, "entry_offset": 256}},
            "scope": {"kind": "kernel"},
            "selection": {"requirement": "top_dispatch_count", "top_n": 1},
            "sanitizers": ["waitcheck"],
            "policy": {"consan_policy": "strict", "on_missing_backend": "fail",
                       "timeout_seconds": 120, "waitcheck_max_diagnostics": 64},
            "output": {"report": "sanitizer_report.json"}}}
    calls = []

    def never_execute(*args, **kwargs):
        pytest.fail("native executor must not be reached")

    monkeypatch.setattr(pipeline, "run_waitcheck",
                        partial(waitcheck.run_waitcheck, execute=never_execute))

    def run(execute=None):
        if execute is not None:
            def capture(argv, *, timeout_seconds, **kwargs):
                calls.append(tuple(argv))
                assert timeout_seconds == 120
                assert argv[argv.index("--max-diagnostics") + 1] == "64"
                assert argv[argv.index("--kernel-entry") + 1] == "0x100"
                assert argv[argv.index("--code-object-index") + 1] == "0"
                return execute(argv)
            monkeypatch.setattr(pipeline, "run_waitcheck",
                                partial(waitcheck.run_waitcheck, execute=capture))
        recipe = tmp_path / "case.yaml"
        recipe.write_text(yaml.safe_dump(data), encoding="utf-8")
        path = execute_sanitizer_run(recipe, output_dir=tmp_path / "out")
        return read_report(path)

    return data, artifact, binary, run, calls


@pytest.mark.parametrize("problem,reason,kernel_state", [
    ("missing", "code_object_not_found", "not_checked"),
    ("empty", "code_object_empty", "not_checked"),
    ("changed", "code_object_digest_mismatch", "error"),
])
def test_object_problem_persists_nonclean_report(setup_case, tmp_path, problem, reason, kernel_state):
    _, artifact, _, run, calls = setup_case
    if problem == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"" if problem == "empty" else b"changed bytes")
    report = run()
    assert report.overall_verdict.value == "error"
    assert report.execution_status.value == "error"
    check = report.checks[0]
    assert check.reason == "worklist_not_fully_checked"
    result = check.kernel_results[0]
    assert (result.state.value, result.reason, result.returncode) == (kernel_state, reason, None)
    assert not calls
    assert not (tmp_path / "out/waitcheck/waitcheck-0.log").exists()


@pytest.mark.parametrize("problem,reason", [("missing", "rj_waitcheck_not_found"),
                                            ("empty", "rj_waitcheck_empty")])
def test_missing_backend_fail_policy_preserves_not_checked(setup_case, problem, reason):
    data, _, binary, run, calls = setup_case
    assert data["sanitizer_plan"]["policy"]["on_missing_backend"] == "fail"
    if problem == "missing":
        binary.unlink()
    else:
        binary.write_bytes(b"")
    report = run()
    assert (report.overall_verdict.value, report.execution_status.value) == ("not_checked", "not_checked")
    assert report.checks[0].reason == reason
    assert report.checks[0].kernel_results[0].reason == reason
    assert not calls


@pytest.mark.parametrize("field,value", [("entry_offset", -1), ("code_object_index", True)])
def test_invalid_selector_rejected_before_report_or_execution(setup_case, tmp_path, field, value):
    data, _, _, run, calls = setup_case
    data["sanitizer_plan"]["source"]["kernel"][field] = value
    with pytest.raises(RecipeSchemaError, match=field):
        run()
    assert not calls
    assert not (tmp_path / "out/sanitizer_report.json").exists()


@pytest.mark.parametrize("problem,reason,state", [
    ("timeout", "waitcheck_timeout", "timed_out"),
    ("invalid_entry", "waitcheck_backend_exit_2", "error"),
    ("unsupported_flag", "waitcheck_backend_exit_2", "error"),
    ("malformed_output", "waitcheck_diagnostics_error", "error"),
])
def test_backend_failure_survives_recipe_report_boundary(setup_case, tmp_path, problem, reason, state):
    _, _, _, run, calls = setup_case
    messages = {"invalid_entry": "synthetic: requested kernel entry not found",
                "unsupported_flag": "synthetic: unknown --kernel-entry option"}

    def execute(argv):
        if problem == "timeout":
            return ProcessResult(tuple(argv), None, "synthetic partial output", "", timed_out=True)
        if problem in messages:
            return ProcessResult(tuple(argv), 2, "", messages[problem])
        return ProcessResult(tuple(argv), 0, "synthetic incompatible output format", "")

    report = run(execute)
    assert (report.overall_verdict.value, report.execution_status.value) == ("error", "error")
    check = report.checks[0]
    assert check.state.value == state
    result = check.kernel_results[0]
    assert result.reason.startswith(reason)
    assert result.state.value == state
    if problem in messages:
        assert messages[problem] in result.reason
        assert result.returncode == 2
    if problem == "malformed_output":
        assert result.returncode == 0  # Zero native exit cannot hide an unreadable result.
    assert len(calls) == 1
    raw = (tmp_path / "out/waitcheck/waitcheck-0.log").read_text(encoding="utf-8")
    expected = ("synthetic partial output" if problem == "timeout" else
                "synthetic incompatible output format" if problem == "malformed_output" else messages[problem])
    assert expected in raw


def test_capped_finding_remains_warning_with_explicit_incomplete_count(setup_case):
    _, artifact, _, run, calls = setup_case
    prefix = f"{artifact}:gfx1250[0]"
    output = (f"rj_waitcheck: {prefix}:kernel=.text+0x100: "
              "instructions=4 memory-events=2 diagnostics=>=1\n"
              f"{prefix}:.text+0x108: missing s_wait_kmcnt <= 0 before use of s4\n")
    report = run(lambda argv: ProcessResult(tuple(argv), 4, output, ""))
    assert (report.overall_verdict.value, report.execution_status.value) == ("warn", "complete")
    check = report.checks[0]
    assert check.state.value == "ran"
    assert check.diagnostics_truncated is True
    assert check.kernel_results[0].diagnostics_truncated is True
    assert check.kernel_results[0].returncode == 4
    assert len(check.findings) == 1
    assert len(calls) == 1
