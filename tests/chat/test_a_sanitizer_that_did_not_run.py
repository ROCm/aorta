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


class TestEverySweepOwesAReport:
    """Which arguments mean a sanitizer was asked for.

    This read --recipe alone, and --recipe is the one sweep argument the pasted
    kernel path does not use: triage_kernel_source passes --source. So the
    warning above never fired for the case it was written about -- a kernel
    nothing analysed, reported clean -- and the test that guarded it asserted
    the source line rather than the behaviour, so it agreed with the bug.
    """

    @staticmethod
    def _owed(argv: list[str]) -> bool:
        from aorta.chat.tools.cluster import _SWEEP_ARGS

        return any(arg in argv for arg in _SWEEP_ARGS)

    def test_a_recipe_run_owes_one(self):
        """write_asm_recipe asks for waitcheck with on_missing_backend=fail."""
        assert self._owed(["--recipe", "/jobs/r.yaml"]) is True

    def test_a_pasted_kernel_owes_one_too(self):
        """--source runs the same sweep, through the recipe triage.py writes."""
        assert self._owed(["--source", "/jobs/k.hip", "--kernel-name", "bump"]) is True

    def test_a_raw_workload_does_not(self):
        """The user's own program promises nothing about a sanitizer."""
        assert self._owed(["--command", "python train.py"]) is False

    def test_the_kernel_tool_passes_a_sweep_argument(self):
        """Pins the two together: if the tool stops using --source, this fails."""
        import inspect

        from aorta.chat.tools.cluster import _SWEEP_ARGS, triage_kernel_source

        source = inspect.getsource(triage_kernel_source.func)

        assert any(f'"{arg}"' in source for arg in _SWEEP_ARGS)


class TestARunThatDidNotCheckIsNotRemembered:
    """Caching it pins "we looked and found nothing" to the paste.

    The guard was an Autopsy verdict being present, and Autopsy will happily
    explain an empty bundle -- so a sweep whose sanitizer never ran carried one
    anyway and was cached as a completed diagnosis. The key is the paste, which
    has not changed, so every retry for the rest of the conversation replayed
    the failure instead of re-running it.
    """

    def test_the_marker_the_guard_reads_is_the_one_the_result_writes(self):
        """Two spellings of it would make the guard quietly useless."""
        from aorta.chat.tools.cluster import _DID_NOT_RUN

        assert _DID_NOT_RUN in _format_result(_AUTOPSY_ON_NOTHING, "x", expect_sanitizer=True)

    def test_a_result_that_ran_is_cacheable(self):
        """A report and a verdict together: the case worth remembering."""
        from aorta.chat.tools.cluster import _DID_NOT_RUN

        diagnosed = {**_REAL_PASS, "autopsy": {"category": "clean", "confidence": 0.9}}
        rendered = _format_result(diagnosed, "x", expect_sanitizer=True)

        assert "Autopsy verdict:" in rendered
        assert _DID_NOT_RUN not in rendered

    def test_both_cache_guards_require_it(self):
        """The kernel and assembly paths keep the same rule."""
        import inspect

        from aorta.chat.tools.cluster import triage_assembly_source, triage_kernel_source

        for tool in (triage_kernel_source, triage_assembly_source):
            source = inspect.getsource(tool.func)
            assert "_DID_NOT_RUN not in" in source, tool.name
