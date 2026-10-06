"""A sanitizer that did not reach a verdict is never reported as clean.

The adapter named the states that counted as a gap -- the missing-backend
reasons and ``not_checked`` -- and let every other non-``ran`` state fall
through. ``error`` and ``timed_out`` are neither, so a sanitizer that crashed
contributed no evidence and no gap, and the clean branch then emitted
``SAN_CLEAN``.

That is not hypothetical. The survey reports committed to this repository say
``state: error, verdict: error, overall_verdict: error``, and the adapter read
them as "sanitizers ran clean on gfx950 (overall_verdict=error)" -- a sentence
that contradicts itself and would have gone to the router as evidence of a
healthy run.

The fix inverts the question: ``ran`` is the only state in which a sanitizer
reached a verdict, so every other one is a gap. A deny list lets a state added
upstream arrive as a pass; an allow list makes it arrive as "we do not know".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from aorta.cia.autopsy.adapters.base import BundleContext
from aorta.cia.autopsy.adapters.sanitizer_report import (
    SIGNAL_CLEAN,
    SIGNAL_NOT_CHECKED,
    SanitizerReportAdapter,
)

REPO = Path(__file__).resolve().parents[2]
SURVEY = REPO / "recipes" / "sanitizers" / "survey" / "reports"


def _collect(tmp_path: Path, report: dict):
    root = tmp_path / "bundle"
    (root / "aorta").mkdir(parents=True, exist_ok=True)
    (root / "aorta" / "sanitizer_report.json").write_text(json.dumps(report))
    ctx = BundleContext(
        root=root,
        manifest={"paths": {"sanitizer_report": "aorta/sanitizer_report.json"}},
        job_id="cia-test",
    )
    return SanitizerReportAdapter().collect(ctx)


def _report(**check) -> dict:
    base = {"sanitizer": "consan", "state": "ran", "verdict": "pass", "findings": []}
    return {
        "schema": "aorta.sanitizer_report/0.1",
        "target": "gfx950",
        "execution_status": "complete",
        "overall_verdict": check.get("verdict", "pass"),
        "checks": [{**base, **check}],
    }


class TestAStateThatIsNotRan:
    @pytest.mark.parametrize(
        "state,reason",
        [
            ("error", "consan_strict_load_rejection"),
            ("error", "combined_hook_exit_86"),
            ("timed_out", ""),
            ("error", ""),
            ("crashed", "segfault"),
            ("something_new_upstream", ""),
        ],
    )
    def test_is_a_gap_and_never_clean(self, state, reason, tmp_path):
        art = _collect(tmp_path, _report(state=state, verdict="error", reason=reason))
        assert SIGNAL_CLEAN not in art.signals
        assert SIGNAL_NOT_CHECKED in art.signals
        assert art.tooling_gaps, f"{state} produced no gap"

    def test_the_gap_names_what_happened(self, tmp_path):
        art = _collect(tmp_path, _report(state="error", verdict="error", reason="combined_hook_exit_86"))
        assert "combined_hook_exit_86" in art.tooling_gaps[0]["description"]

    def test_a_missing_backend_keeps_its_actionable_message(self, tmp_path):
        """Where the cause is known, say what to do about it."""
        art = _collect(tmp_path, _report(state="not_checked", reason="rj_waitcheck_not_found"))
        assert "ROCJITSU_BUILD" in art.tooling_gaps[0]["description"]

    def test_one_failed_sanitizer_is_enough(self, tmp_path):
        """A pass beside a crash is not a clean run."""
        report = _report(state="ran", verdict="pass")
        report["checks"].append(
            {"sanitizer": "waitcheck", "state": "error", "verdict": "error", "reason": "boom"}
        )
        art = _collect(tmp_path, report)
        assert SIGNAL_CLEAN not in art.signals
        assert SIGNAL_NOT_CHECKED in art.signals


class TestAStateThatIsRan:
    def test_a_genuine_pass_is_still_clean(self, tmp_path):
        art = _collect(tmp_path, _report(state="ran", verdict="pass"))
        assert SIGNAL_CLEAN in art.signals
        assert not art.tooling_gaps

    def test_findings_still_become_evidence(self, tmp_path):
        art = _collect(
            tmp_path,
            _report(
                state="ran",
                verdict="fail",
                findings=[{"severity": "error", "code": "race", "message": "wave 0 vs wave 3"}],
            ),
        )
        assert SIGNAL_CLEAN not in art.signals
        assert any("wave 0 vs wave 3" in e["excerpt"] for e in art.evidence)


class TestTheReportsCommittedToThisRepo:
    """The finding cites these by name, so they are the test."""

    @pytest.mark.parametrize(
        "name", ["gemm_f32_consan", "lds_reduce_consan", "tiny_vecadd_consan"]
    )
    def test_an_errored_survey_report_is_not_clean(self, name, tmp_path):
        source = SURVEY / name / "sanitizer_report.json"
        if not source.is_file():
            pytest.skip(f"{name} is not in this tree")
        root = tmp_path / "bundle"
        (root / "aorta").mkdir(parents=True)
        shutil.copy(source, root / "aorta" / "sanitizer_report.json")
        ctx = BundleContext(
            root=root,
            manifest={"paths": {"sanitizer_report": "aorta/sanitizer_report.json"}},
            job_id="cia-test",
        )
        art = SanitizerReportAdapter().collect(ctx)

        assert SIGNAL_CLEAN not in art.signals
        assert art.tooling_gaps

    @pytest.mark.parametrize("name", ["lds_reduce_waitcheck", "tiny_vecadd_waitcheck"])
    def test_a_passing_survey_report_still_reads_clean(self, name, tmp_path):
        source = SURVEY / name / "sanitizer_report.json"
        if not source.is_file():
            pytest.skip(f"{name} is not in this tree")
        root = tmp_path / "bundle"
        (root / "aorta").mkdir(parents=True)
        shutil.copy(source, root / "aorta" / "sanitizer_report.json")
        ctx = BundleContext(
            root=root,
            manifest={"paths": {"sanitizer_report": "aorta/sanitizer_report.json"}},
            job_id="cia-test",
        )
        art = SanitizerReportAdapter().collect(ctx)

        assert SIGNAL_CLEAN in art.signals
