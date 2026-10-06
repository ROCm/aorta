"""A failed sanitizer run has to look failed.

The launch script takes ``rc=$?`` on the line after the workload command. The
workload command ends by echoing the sanitizer findings into the job log --
appended with ``;`` so it runs even when the guardrail exits non-zero, and
``|| true`` so a broken summary cannot fail the run. Both are right on their
own, and together they meant ``rc`` read the *echo's* status, which ``|| true``
pins at zero.

So every run reported ``[cia] workload exit=0``. A job whose interpreter did
not resolve on the compute node, which ran no sanitizer and produced no report,
reported exit=0 as well -- and Autopsy, handed an empty bundle, explained the
absence of evidence rather than the failure that caused it. The chat answer
that came out described a clean static analysis of a kernel that was never
analysed.
"""

from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("dspy", reason="the triage module needs the [cia] extra")

from aorta.cia import triage


def run_fragment(sweep: str) -> int:
    """Run the launcher's tail with *sweep* standing in for the real command."""
    script = f"{sweep} ; _cia_rc=$? ; echo findings || true ; (exit $_cia_rc)\nrc=$?\nexit $rc\n"
    return subprocess.run(["bash", "-c", script]).returncode


class TestTheStatusThatReachesTheLauncher:
    def test_a_failing_sweep_is_not_reported_as_zero(self):
        assert run_fragment("false") != 0

    def test_a_failing_sweep_keeps_its_own_code(self):
        """Sanitizer exit codes carry meaning; 86 is 'nothing was sampled'."""
        assert run_fragment("exit 86") == 86

    def test_a_passing_sweep_is_still_zero(self):
        assert run_fragment("true") == 0

    def test_a_broken_summary_cannot_fail_a_good_run(self):
        """The reason for '|| true' in the first place."""
        script = "true ; _cia_rc=$? ; false || true ; (exit $_cia_rc)\nrc=$?\nexit $rc\n"

        assert subprocess.run(["bash", "-c", script]).returncode == 0

    def test_the_summary_runs_even_when_the_sweep_failed(self):
        """The reason for ';' rather than '&&'."""
        script = "false ; _cia_rc=$? ; echo ran ; (exit $_cia_rc)"
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)

        assert "ran" in done.stdout


class TestTheCommandTheTriageBuilds:
    """Read off the source, because building one needs a cluster."""

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        return (
            Path(triage.__file__).resolve()
        ).read_text(encoding="utf-8")

    def test_it_saves_the_status_before_the_echo(self):
        assert "_cia_rc=$?" in self._source()

    def test_it_restores_the_status_after(self):
        assert "(exit $_cia_rc)" in self._source()

    def test_the_echo_still_cannot_fail_the_run(self):
        assert "|| true" in self._source()

    def test_the_echo_still_runs_after_a_non_zero_sweep(self):
        """';' not '&&': a guardrail that fires exits non-zero on purpose."""
        source = self._source()
        start = source.index("echo_findings = (")

        assert '"; _cia_rc=$?;' in source[start : start + 200]
