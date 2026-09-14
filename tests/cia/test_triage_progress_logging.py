"""Progress belongs to whoever is running the triage.

This module used to run as a console script in the agents' own virtualenv, and
its progress went straight to stderr because stderr was its own. It is now
imported and called inside the chat server, where that stream belongs to
something else -- a process with its own logging configuration, its own idea of
levels, and its own handlers.

So progress goes through logging and the caller decides where it lands. The
command line is still a caller, and still shows what it always showed: the
handler it configures for itself is what makes that true.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import pytest

from aorta.cia import triage as triage_mod



class TestTheLibraryCaseDefersToItsHost:
    def test_progress_goes_through_logging(self, caplog):
        with caplog.at_level(logging.INFO, logger="aorta.cia.triage"):
            triage_mod.log.info("slurm 123 state=RUNNING ...")

        assert "slurm 123 state=RUNNING ..." in caplog.text

    def test_it_is_a_logger_and_not_a_function(self):
        """It was ``def log(msg)``, which no host could configure."""
        assert isinstance(triage_mod.log, logging.Logger)
        assert triage_mod.log.name == "aorta.cia.triage"

    def test_nothing_is_written_when_the_host_has_not_asked_for_it(self):
        """A library that writes regardless is the thing being fixed.

        With the host's level above INFO the progress is simply not emitted --
        no handler of ours, no stream of ours.
        """
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        triage_mod.log.addHandler(handler)
        triage_mod.log.setLevel(logging.WARNING)
        try:
            triage_mod.log.info("chatty progress nobody asked for")
        finally:
            triage_mod.log.removeHandler(handler)
            triage_mod.log.setLevel(logging.NOTSET)

        assert records == []

    def test_the_host_s_own_handler_is_what_receives_it(self):
        """The point of the change: the caller decides where this lands."""
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        triage_mod.log.addHandler(handler)
        triage_mod.log.setLevel(logging.INFO)
        try:
            triage_mod.log.info("progress from inside the chat server")
        finally:
            triage_mod.log.removeHandler(handler)
            triage_mod.log.setLevel(logging.NOTSET)

        assert [r.getMessage() for r in records] == ["progress from inside the chat server"]
        assert records[0].name == "aorta.cia.triage"


class TestTheCommandLineStillLooksTheSame:
    """Routing through logging must not cost the CLI its running commentary."""

    def _run_cli(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "aorta.cia.triage", *args],
            capture_output=True,
            text=True,
            cwd="/apps/avsharma/aorta",
        )

    def test_the_result_is_json_on_stdout_and_nothing_else(self):
        done = self._run_cli("--source", "/nope/missing.hip")

        assert json.loads(done.stdout)["ok"] is False

    def test_progress_still_reaches_stderr_with_its_prefix(self):
        """main() configures this; without it the CLI would go quiet."""
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("[triage] %(message)s"))
        triage_mod.log.addHandler(handler)
        triage_mod.log.setLevel(logging.INFO)
        try:
            triage_mod.log.info("slurm 42 reached COMPLETED")
        finally:
            triage_mod.log.removeHandler(handler)
            triage_mod.log.setLevel(logging.NOTSET)

        assert stream.getvalue().strip() == "[triage] slurm 42 reached COMPLETED"

    def test_main_configures_that_handler_itself(self):
        """The format and stream the command line has always had."""
        import inspect

        source = inspect.getsource(triage_mod.main)

        assert "basicConfig" in source
        assert "[triage] %(message)s" in source
        assert "sys.stderr" in source

    def test_the_exit_code_still_reflects_the_result(self):
        assert self._run_cli("--source", "/nope/missing.hip").returncode == 1


class TestTheDocstringsDescribeWhatHappens:
    """Both said this shelled out to a separate virtualenv. Neither ever did."""

    def test_triage_does_not_claim_a_separate_virtualenv(self):
        """It may mention one to deny it; what it must not do is claim one."""
        doc = triage_mod.__doc__ or ""

        assert "Executed with the cluster-intelligence-agent virtualenv" not in doc
        assert "no separate agent virtualenv" in doc

    def test_triage_says_it_returns_a_dict_to_its_caller(self):
        doc = triage_mod.__doc__ or ""

        assert "returns a dict" in doc
        assert "run_triage" in doc

    def test_the_chat_tools_do_not_claim_to_shell_out(self):
        pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster

        doc = cluster.__doc__ or ""

        assert "shells out" not in doc
        assert "console scripts" not in doc

    def test_the_chat_tools_say_the_agents_run_in_this_process(self):
        pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster

        assert "in this process" in (cluster.__doc__ or "")

    def test_and_do_not_overclaim_about_the_rest_of_the_pipeline(self):
        """Watch and the probe still print; saying otherwise is the same defect."""
        pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster
        from aorta.cia.watch import poll

        doc = cluster.__doc__ or ""
        still_printing = "print(" in (poll.__file__ and open(poll.__file__).read())

        if still_printing:
            assert "still print" in doc
