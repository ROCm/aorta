"""Triton FpSan backend: the result contract, the runner, recipes, and autopsy.

The stdout fixtures are hand-written in the ``AORTA_FPSAN_RESULT`` shape (see
fixtures/fpsan/README.md), so this file needs no GPU and no Triton.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from aorta.cia.autopsy.adapters.base import BundleContext
from aorta.cia.autopsy.adapters.sanitizer_report import (
    SIGNAL_FPSAN_MISMATCH,
    SIGNAL_NOT_CHECKED,
    SIGNAL_RACE,
    SanitizerReportAdapter,
    classify_sanitizer,
)
from aorta.instrumentation.rocjitsu_sanitizers import (
    CheckResult,
    ExecutionState,
    ExecutionSummary,
    FindingSeverity,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    Verdict,
    evaluate_fpsan_output,
    execute_sanitizer_run,
    parse_fpsan_result,
    read_report,
    run_fpsan,
    run_sanitizers,
    support,
)
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe
from aorta.triage.recipe import RecipeSchemaError

FIXTURES = Path(__file__).parent / "fixtures" / "fpsan"
DIGEST_A = "4dfe96464b850fae836ced6ef7177049a67ed5f67ad63f31c6cca93dbb2779b3"
DIGEST_B = "e986a6ed5ab5c6794224330f7f8caa9abac54a995ce4e3ee3dc28c2ee964a5c4"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _process(stdout: str = "", *, returncode: int | None = 0, stderr: str = "", **kw):
    return ProcessResult(
        argv=("fpsan_cmd",), returncode=returncode, stdout=stdout, stderr=stderr, **kw
    )


def _line(**fields: object) -> str:
    record = {
        "reference_digest": DIGEST_A,
        "candidate_digest": DIGEST_A,
        "reference_instrumented": True,
        "candidate_instrumented": True,
        **fields,
    }
    return "AORTA_FPSAN_RESULT " + json.dumps(record) + "\n"


def _evaluate(stdout: str, **kw) -> CheckResult:
    return evaluate_fpsan_output(_process(stdout, **kw), sanitizer="fpsan", target="gfx950")


def _worklist(target: str = "gfx950") -> KernelWorklist:
    return KernelWorklist(
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=1,
        kernels=(
            KernelObservation(
                identity=KernelIdentity(name="matmul_fused", target=target),
                dispatch_count=1,
                sources=("test",),
            ),
        ),
    )


def _script(tmp_path: Path, fixture: str, *, code: int = 0, name: str = "fpsan_cmd") -> Path:
    """An executable that prints a fixture on stdout and exits ``code``."""

    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\ncat '{FIXTURES / fixture}'\nexit {code}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# --------------------------------------------------------------------------- #
# Result contract
# --------------------------------------------------------------------------- #


class TestParse:
    def test_a_matching_line_parses(self):
        result = parse_fpsan_result(_fixture("match.txt"))
        assert (result.reference_digest, result.candidate_digest) == (DIGEST_A, DIGEST_A)
        assert result.instrumented and result.digests_match
        assert result.target == "gfx950"

    def test_target_is_optional(self):
        assert parse_fpsan_result(_line()).target is None

    @pytest.mark.parametrize(
        "stdout,reason",
        [
            ("", "fpsan_result_missing"),
            ("AORTA_FPSAN_RESULT [1, 2]\n", "fpsan_result_unparsable"),
            (_line(extra=1), "fpsan_result_unparsable: unknown fields"),
            (_line(reference_digest=DIGEST_A.upper()), "fpsan_result_unparsable"),
            (_line(candidate_digest="abc"), "fpsan_result_unparsable"),
            (_line(candidate_instrumented="true"), "fpsan_result_unparsable"),
            (_line(target=""), "fpsan_result_unparsable"),
        ],
    )
    def test_malformed_records_are_refused(self, stdout, reason):
        with pytest.raises(ValueError, match=reason):
            parse_fpsan_result(stdout)

    def test_a_missing_field_is_named(self):
        stdout = "AORTA_FPSAN_RESULT " + json.dumps(
            {"reference_digest": DIGEST_A, "candidate_digest": DIGEST_A}
        )
        with pytest.raises(ValueError, match="missing fields .*candidate_instrumented"):
            parse_fpsan_result(stdout)

    def test_a_marker_inside_a_word_is_not_a_result_line(self):
        with pytest.raises(ValueError, match="fpsan_result_missing"):
            parse_fpsan_result("AORTA_FPSAN_RESULTS " + _line()[len("AORTA_FPSAN_RESULT ") :])


class TestEvaluate:
    def test_equal_digests_from_two_instrumented_kernels_pass(self):
        check = _evaluate(_fixture("match.txt"))
        assert (check.state, check.verdict) == (ExecutionState.RAN, Verdict.PASS)
        assert check.findings == ()

    def test_different_digests_fail_with_one_mismatch_finding(self):
        check = _evaluate(_fixture("mismatch.txt"))
        assert (check.state, check.verdict) == (ExecutionState.RAN, Verdict.FAIL)
        assert check.reason == "fpsan_digest_mismatch"
        (finding,) = check.findings
        assert (finding.sanitizer, finding.severity, finding.code) == (
            "fpsan",
            FindingSeverity.ERROR,
            "fpsan_digest_mismatch",
        )
        metadata = dict(finding.metadata)
        assert (metadata["reference_digest"], metadata["candidate_digest"]) == (DIGEST_A, DIGEST_B)

    def test_an_uninstrumented_side_is_an_error_even_with_equal_digests(self):
        check = _evaluate(_fixture("candidate_not_instrumented.txt"))
        assert (check.state, check.verdict) == (ExecutionState.ERROR, Verdict.ERROR)
        assert check.reason.startswith("fpsan_not_instrumented:")
        assert "candidate_instrumented=false" in check.reason

    @pytest.mark.parametrize(
        "fixture,reason",
        [
            ("no_result_line.txt", "fpsan_result_missing"),
            ("two_result_lines.txt", "fpsan_result_ambiguous"),
            ("unparsable.txt", "fpsan_result_unparsable"),
        ],
    )
    def test_a_missing_repeated_or_broken_line_is_an_error(self, fixture, reason):
        check = _evaluate(_fixture(fixture))
        assert (check.state, check.verdict) == (ExecutionState.ERROR, Verdict.ERROR)
        assert check.reason.startswith(reason)
        assert check.findings == ()

    def test_the_agentkernelarena_marker_is_refused_and_named(self):
        check = _evaluate(_fixture("aka_marker.txt"))
        assert check.verdict is Verdict.ERROR
        assert check.reason.startswith("fpsan_result_missing")
        assert "AKA_FPSAN_RESULT" in check.reason

    def test_a_result_on_stderr_is_not_read(self):
        check = _evaluate("", stderr=_fixture("match.txt"))
        assert check.reason.startswith("fpsan_result_missing")

    def test_a_timeout_is_an_error_whatever_was_printed(self):
        check = _evaluate(_fixture("mismatch.txt"), returncode=None, timed_out=True)
        assert (check.state, check.verdict, check.reason) == (
            ExecutionState.TIMED_OUT,
            Verdict.ERROR,
            "fpsan_timeout",
        )

    def test_a_non_zero_exit_is_an_error_whatever_was_printed(self):
        check = _evaluate(_fixture("mismatch.txt"), returncode=1, stderr="Traceback: boom")
        assert (check.state, check.verdict) == (ExecutionState.ERROR, Verdict.ERROR)
        assert check.reason == "fpsan_command_exit_1: Traceback: boom"
        assert check.returncode == 1

    def test_a_launch_failure_is_an_error(self):
        check = _evaluate("", returncode=None, launch_error="PermissionError: denied")
        assert check.reason == "fpsan_launch_error: PermissionError: denied"

    def test_a_result_for_another_target_is_an_error(self):
        check = _evaluate(_line(target="gfx942"))
        assert check.verdict is Verdict.ERROR
        assert check.reason.startswith("fpsan_target_mismatch:")

    def test_the_sanitizer_id_is_the_callers(self):
        """A later hip_fpsan runner reuses the evaluator under its own id."""

        check = evaluate_fpsan_output(
            _process(_fixture("mismatch.txt")), sanitizer="hip_fpsan", target="gfx950"
        )
        assert check.sanitizer == "hip_fpsan"
        assert check.findings[0].sanitizer == "hip_fpsan"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class TestRunFpsan:
    def test_it_runs_the_command_with_fpsan_on_and_keeps_the_log(self, tmp_path):
        command = _script(tmp_path, "match.txt")
        seen = {}

        def execute(argv, *, timeout_seconds, env=None):
            seen.update(argv=argv, timeout=timeout_seconds, env=env)
            return _process(_fixture("match.txt"))

        check = run_fpsan(
            command=command,
            target="gfx950",
            output_dir=tmp_path / "out",
            timeout_seconds=12.0,
            execute=execute,
        )
        assert check.verdict is Verdict.PASS
        assert seen["argv"] == (str(command),)
        assert seen["timeout"] == 12.0
        assert seen["env"]["TRITON_INSTRUMENTATION_MODE"] == "fpsan"
        assert "AORTA_FPSAN_RESULT" in (tmp_path / "out" / "fpsan.log").read_text()
        backend = dict(check.backend)
        assert backend["command"] == str(command)
        assert len(backend["command_sha256"]) == 64
        assert backend["result_marker"] == "AORTA_FPSAN_RESULT"

    def test_a_missing_command_file_is_not_checked(self, tmp_path):
        check = run_fpsan(command=tmp_path / "absent", target="gfx950", output_dir=tmp_path)
        assert (check.state, check.reason) == (
            ExecutionState.NOT_CHECKED,
            "fpsan_command_not_found",
        )

    def test_an_unsupported_target_is_not_checked_and_nothing_runs(self, tmp_path):
        def execute(*_a, **_kw):
            raise AssertionError("must not run on an unsupported target")

        check = run_fpsan(
            command=_script(tmp_path, "match.txt"),
            target="gfx1100",
            output_dir=tmp_path / "out",
            execute=execute,
        )
        assert check.state is ExecutionState.NOT_CHECKED
        assert check.reason.startswith("fpsan_target_unsupported:")


def test_support_matrix():
    for target in ("gfx942", "gfx950", "gfx1250"):
        assert support("fpsan", target)["level"] == "full"
        assert support("fpsan", target)["runnable"] is True
    for target in ("gfx90a", "gfx1100", "gfx1201"):
        assert support("fpsan", target)["level"] == "none"
        assert support("fpsan", target)["runnable"] is False
    with pytest.raises(ValueError, match="unknown sanitizer"):
        support("hip_fpsan", "gfx950")


# --------------------------------------------------------------------------- #
# Pipeline and recipe
# --------------------------------------------------------------------------- #


class TestPipeline:
    def test_fpsan_without_a_command_is_not_checked(self, tmp_path):
        report = run_sanitizers(
            _worklist(), target="gfx950", sanitizers=("fpsan",), output_dir=tmp_path
        )
        (check,) = report.checks
        assert (check.sanitizer, check.state) == ("fpsan", ExecutionState.NOT_CHECKED)
        assert check.reason.startswith("fpsan_command_not_provisioned:")
        assert report.execution_status is ExecutionSummary.NOT_CHECKED

    def test_a_mismatch_fails_the_report(self, tmp_path):
        report = run_sanitizers(
            _worklist(),
            target="gfx950",
            sanitizers=("fpsan",),
            output_dir=tmp_path / "out",
            fpsan_command=_script(tmp_path, "mismatch.txt"),
        )
        assert report.overall_verdict is Verdict.FAIL
        assert report.execution_status is ExecutionSummary.COMPLETE
        stored = read_report(tmp_path / "out" / "sanitizer_report.json")
        assert stored.checks[0].findings[0].code == "fpsan_digest_mismatch"
        assert (tmp_path / "out" / "fpsan" / "fpsan.log").is_file()

    def test_a_match_passes_the_report(self, tmp_path):
        report = run_sanitizers(
            _worklist(),
            target="gfx950",
            sanitizers=("fpsan",),
            output_dir=tmp_path / "out",
            fpsan_command=_script(tmp_path, "match.txt"),
        )
        assert (report.overall_verdict, report.execution_status) == (
            Verdict.PASS,
            ExecutionSummary.COMPLETE,
        )

    def test_an_unsupported_target_is_not_checked(self, tmp_path):
        report = run_sanitizers(
            _worklist("gfx1100"),
            target="gfx1100",
            sanitizers=("fpsan",),
            output_dir=tmp_path / "out",
            fpsan_command=_script(tmp_path, "match.txt"),
        )
        (check,) = report.checks
        assert check.reason.startswith("fpsan_target_unsupported:")

    @pytest.mark.parametrize("name", ["hip_fpsan", "fp_san"])
    def test_other_names_are_still_unknown(self, tmp_path, name):
        with pytest.raises(ValueError, match="unknown sanitizers"):
            run_sanitizers(_worklist(), target="gfx950", sanitizers=(name,), output_dir=tmp_path)


_RECIPE = """\
schema_version: 1
mode: sanitizer
sanitizer_plan:
  target: {target}
  source:
    kind: kernel
    kernel: {{name: matmul_fused}}
{source_extra}
  scope: {{kind: kernel}}
  selection: {{requirement: top_dispatch_count, top_n: 1}}
  sanitizers: [{sanitizers}]
  policy: {{consan_policy: strict, on_missing_backend: fail}}
  output: {{report: sanitizer_report.json}}
"""


def _recipe(
    tmp_path: Path, *, source_extra: str = "", sanitizers: str = "fpsan", target: str = "gfx950"
) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(_RECIPE.format(source_extra=source_extra, sanitizers=sanitizers, target=target))
    return path


class TestRecipe:
    def test_fpsan_command_resolves_against_the_recipe(self, tmp_path):
        recipe = load_sanitizer_recipe(
            _recipe(tmp_path, source_extra="    fpsan_command: bin/fpsan_compare.sh")
        )
        assert recipe.sanitizers == ("fpsan",)
        assert recipe.fpsan_command == (tmp_path / "bin" / "fpsan_compare.sh").resolve()

    def test_an_absolute_fpsan_command_is_kept(self, tmp_path):
        recipe = load_sanitizer_recipe(
            _recipe(tmp_path, source_extra="    fpsan_command: /opt/fpsan/compare")
        )
        assert recipe.fpsan_command == Path("/opt/fpsan/compare")

    def test_fpsan_command_without_fpsan_requested_is_rejected(self, tmp_path):
        with pytest.raises(RecipeSchemaError, match="fpsan is not in"):
            load_sanitizer_recipe(
                _recipe(tmp_path, source_extra="    fpsan_command: bin/x", sanitizers="consan")
            )

    def test_an_empty_fpsan_command_is_rejected(self, tmp_path):
        with pytest.raises(RecipeSchemaError, match="fpsan_command must be a non-empty string"):
            load_sanitizer_recipe(_recipe(tmp_path, source_extra="    fpsan_command: ''"))

    def test_hip_fpsan_is_not_a_recipe_sanitizer_yet(self, tmp_path):
        with pytest.raises(RecipeSchemaError, match="unsupported sanitizers: \\['hip_fpsan'\\]"):
            load_sanitizer_recipe(_recipe(tmp_path, sanitizers="hip_fpsan"))

    def test_a_recipe_runs_end_to_end_and_fails_on_a_mismatch(self, tmp_path):
        _script(tmp_path, "mismatch.txt", name="bin/fpsan_compare.sh")
        report_path = execute_sanitizer_run(
            _recipe(tmp_path, source_extra="    fpsan_command: bin/fpsan_compare.sh"),
            output_dir=tmp_path / "out",
        )
        report = read_report(report_path)
        assert report.overall_verdict is Verdict.FAIL
        assert report.checks[0].sanitizer == "fpsan"

    def test_a_recipe_without_a_command_is_not_checked(self, tmp_path):
        report = read_report(execute_sanitizer_run(_recipe(tmp_path), output_dir=tmp_path / "out"))
        assert report.execution_status is ExecutionSummary.NOT_CHECKED


# --------------------------------------------------------------------------- #
# CIA classification
# --------------------------------------------------------------------------- #

_MISMATCH_FINDING = {
    "sanitizer": "fpsan",
    "severity": "error",
    "code": "fpsan_digest_mismatch",
    "message": "reference and candidate kernels disagree under FpSan",
    "metadata": {"reference_digest": DIGEST_A, "candidate_digest": DIGEST_B, "target": "gfx950"},
}


def _report(*checks: dict, verdict: str = "fail") -> dict:
    return {
        "schema": "aorta.sanitizer_report/0.1",
        "target": "gfx950",
        "overall_verdict": verdict,
        "checks": list(checks),
    }


def _collect(tmp_path: Path, report: dict):
    root = tmp_path / "bundle"
    (root / "aorta").mkdir(parents=True)
    (root / "aorta" / "sanitizer_report.json").write_text(json.dumps(report))
    return SanitizerReportAdapter().collect(
        BundleContext(
            root=root,
            manifest={"paths": {"sanitizer_report": "aorta/sanitizer_report.json"}},
            job_id="fpsan-test",
        )
    )


_FAILED = {"sanitizer": "fpsan", "state": "ran", "verdict": "fail", "findings": [_MISMATCH_FINDING]}


class TestClassification:
    def test_a_mismatch_is_silent_numerics_with_its_own_signal(self):
        verdict = classify_sanitizer(_report(_FAILED))
        assert (verdict.category, verdict.signals) == ("numeric_silent", [SIGNAL_FPSAN_MISMATCH])
        assert DIGEST_A in verdict.rationale and DIGEST_B in verdict.rationale

    def test_an_observed_race_outranks_the_mismatch_it_may_explain(self):
        race = {"sanitizer": "consan", "severity": "race", "code": "race", "message": "race"}
        verdict = classify_sanitizer(
            _report(_FAILED, {"sanitizer": "consan", "state": "ran", "findings": [race]})
        )
        assert verdict.signals == [SIGNAL_RACE]

    def test_the_adapter_carries_the_mismatch_as_evidence(self, tmp_path):
        artifact = _collect(tmp_path, _report(_FAILED))
        assert artifact.signals == [SIGNAL_FPSAN_MISMATCH]
        (evidence,) = artifact.evidence
        assert evidence["signal"] == SIGNAL_FPSAN_MISMATCH
        assert "fpsan_digest_mismatch" in evidence["excerpt"]
        assert artifact.tooling_gaps == []

    def test_a_check_that_did_not_run_is_not_checked_and_names_the_command(self, tmp_path):
        artifact = _collect(
            tmp_path,
            _report(
                {
                    "sanitizer": "fpsan",
                    "state": "not_checked",
                    "verdict": "not_checked",
                    "reason": "fpsan_command_not_provisioned: a command ...",
                    "findings": [],
                },
                verdict="not_checked",
            ),
        )
        assert artifact.signals == [SIGNAL_NOT_CHECKED]
        (gap,) = artifact.tooling_gaps
        assert "fpsan_command" in gap["description"]
        assert "AORTA_FPSAN_RESULT" in gap["description"]
        assert gap["missing_signal"] == SIGNAL_FPSAN_MISMATCH

    def test_a_harness_error_is_a_gap_not_a_mismatch(self, tmp_path):
        artifact = _collect(
            tmp_path,
            _report(
                {
                    "sanitizer": "fpsan",
                    "state": "error",
                    "verdict": "error",
                    "reason": "fpsan_not_instrumented: ...",
                    "findings": [],
                },
                verdict="error",
            ),
        )
        assert artifact.signals == [SIGNAL_NOT_CHECKED]
        assert (
            classify_sanitizer(
                _report({"sanitizer": "fpsan", "state": "error", "verdict": "error"})
            ).category
            == "tooling_gap"
        )
