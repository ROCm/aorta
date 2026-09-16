"""A sanitizer that never ran must not read as a sanitizer that found nothing.

The rendered tool result skipped the sanitizer section entirely when there was
no report, so "checked and clean" and "never checked" produced the same text.
Autopsy, handed an empty bundle, explained the absence of evidence, and the
answer that came out reported a clean static analysis of a kernel that nothing
had analysed.

That is not hypothetical. The interpreter for the job venv lived on an autofs
mount, so on a node where the automount had not triggered the shebang failed
with "Too many levels of symbolic links", no sanitizer ran, and the chat said
"no wait hazards found". A false clean from a diagnostic tool is worse than an
error, because it ends the investigation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")

from aorta.chat.tools.cluster import _format_result

_BARE = {"job_id": "cia-1", "bundle": "/jobs/cia-1/bundle"}
_AUTOPSY_ON_NOTHING = {
    **_BARE,
    "autopsy": {
        "category": "unknown",
        "confidence": 0.0,
        "rationale": "The evidence bundle contains only WATCH_CLEAN.",
    },
}
_REAL_PASS = {
    **_BARE,
    "sanitizer": {
        "overall_verdict": "pass",
        "execution_status": "complete",
        "target": "gfx950",
        "total_findings": 0,
        "checks": [{"sanitizer": "waitcheck", "verdict": "pass", "findings": []}],
    },
}


class TestARunThatProducedNoReport:
    def test_it_says_the_sanitizer_did_not_run(self):
        rendered = _format_result(_AUTOPSY_ON_NOTHING, "asm", expect_sanitizer=True)

        assert "DID NOT RUN" in rendered

    def test_it_says_not_to_call_that_clean(self):
        """The model is the reader here, and it had been calling it clean."""
        rendered = _format_result(_AUTOPSY_ON_NOTHING, "asm", expect_sanitizer=True)

        assert "must not be reported as one" in rendered

    def test_it_points_at_the_job_log(self):
        rendered = _format_result(_AUTOPSY_ON_NOTHING, "asm", expect_sanitizer=True)

        assert "job log" in rendered

    def test_the_word_pass_does_not_appear_for_it(self):
        """A stray 'pass' is the whole thing this is trying to avoid."""
        rendered = _format_result(_AUTOPSY_ON_NOTHING, "asm", expect_sanitizer=True)
        sanitizer_section = rendered[rendered.index("Sanitizer verdict") :]

        assert "pass" not in sanitizer_section.split("\n")[0]


class TestARunThatReallyWasClean:
    def test_it_is_not_accused_of_failing(self):
        rendered = _format_result(_REAL_PASS, "asm", expect_sanitizer=True)

        assert "DID NOT RUN" not in rendered

    def test_the_verdict_is_still_reported(self):
        rendered = _format_result(_REAL_PASS, "asm", expect_sanitizer=True)

        assert "pass" in rendered
        assert "gfx950" in rendered


class TestARunThatNeverAskedForOne:
    def test_a_workload_is_not_expected_to_produce_one(self):
        """triage_workload runs a program; there is no sanitizer owed."""
        rendered = _format_result(
            {**_BARE, "autopsy": {"category": "numeric_silent", "confidence": 0.62}},
            "workload",
            expect_sanitizer=False,
        )

        assert "DID NOT RUN" not in rendered

    def test_the_default_does_not_accuse(self):
        """Callers that predate the argument keep what they had."""
        assert "DID NOT RUN" not in _format_result(_AUTOPSY_ON_NOTHING, "x")


class TestTheExpectationComesFromTheRecipe:
    def test_a_recipe_run_owes_a_report(self):
        """write_asm_recipe asks for waitcheck with on_missing_backend=fail."""
        from pathlib import Path

        import aorta.chat.tools.cluster as cluster

        source = Path(cluster.__file__).read_text(encoding="utf-8")

        assert 'expect_sanitizer = "--recipe" in extra_args' in source
