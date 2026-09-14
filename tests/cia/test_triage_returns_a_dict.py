"""run_triage returns what its signature promises, on every path.

Three early exits printed JSON and ``return 1``. Callers read the result with
``.get()``, so an int arrived as ``AttributeError: 'int' object has no attribute
'get'`` — swallowed by the broad ``except Exception`` in the chat tool and shown
to the user as "triage failed". A missing source file became a mystery, and the
one thing the user could have fixed was the thing the message hid.

``main()`` had the same problem on its own return value.

They also printed to stdout, which was a subprocess pipe when this was a script
and is the chat server's stdout now that it is called in-process.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from aorta.cia.triage import main, run_triage

#: Each early exit, and what a reader should be told.
EARLY_EXITS = {
    "missing source": (["--source", "/nope/does-not-exist.hip"], "source", "source not found"),
    "missing recipe": (["--recipe", "/nope/does-not-exist.yaml"], "recipe", "recipe not found"),
}


def _run(argv: list[str]) -> tuple[dict, str]:
    """Run a triage, capturing anything it writes to stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run_triage(argv)
    return result, buffer.getvalue()


class TestEveryPathReturnsADict:
    @pytest.mark.parametrize("case", sorted(EARLY_EXITS))
    def test_the_result_is_a_dict(self, case):
        argv, _stage, _message = EARLY_EXITS[case]
        result, _ = _run(argv)
        assert isinstance(result, dict), f"{case} returned {type(result).__name__}"

    @pytest.mark.parametrize("case", sorted(EARLY_EXITS))
    def test_get_works_on_it(self, case):
        """The call that used to raise AttributeError."""
        argv, _stage, _message = EARLY_EXITS[case]
        result, _ = _run(argv)
        assert result.get("ok") is False

    @pytest.mark.parametrize("case", sorted(EARLY_EXITS))
    def test_it_says_which_stage_failed(self, case):
        argv, stage, _message = EARLY_EXITS[case]
        result, _ = _run(argv)
        assert result.get("stage") == stage

    @pytest.mark.parametrize("case", sorted(EARLY_EXITS))
    def test_the_error_names_the_actual_problem(self, case):
        argv, _stage, message = EARLY_EXITS[case]
        result, _ = _run(argv)
        assert message in result.get("error", "")

    def test_a_source_with_no_global_kernel_reports_that(self, tmp_path):
        source = tmp_path / "not_a_kernel.hip"
        source.write_text("int main(){return 0;}\n")
        result, _ = _run(["--source", str(source)])
        assert result.get("ok") is False
        assert "__global__" in result.get("error", "")


class TestNothingIsPrintedOnTheWay:
    @pytest.mark.parametrize("case", sorted(EARLY_EXITS))
    def test_stdout_stays_clean(self, case):
        """It is the chat server's stdout now, not a subprocess pipe."""
        argv, _stage, _message = EARLY_EXITS[case]
        _result, printed = _run(argv)
        assert printed == "", f"{case} wrote to stdout: {printed!r}"


class TestTheCallersThatBroke:
    def test_the_chat_tool_surfaces_the_real_message(self):
        """It used to read 'triage failed: AttributeError ...'."""
        from aorta.chat.tools.cluster import _run_triage

        with contextlib.redirect_stdout(io.StringIO()):
            out = _run_triage(["--source", "/nope/does-not-exist.hip"], "a label")

        assert "source not found" in out
        assert "AttributeError" not in out

    def test_main_still_exits_non_zero(self):
        import sys

        argv = sys.argv
        sys.argv = ["triage", "--source", "/nope/does-not-exist.hip"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = main()
        finally:
            sys.argv = argv
        assert code == 1

    def test_main_is_the_one_that_prints(self):
        """Reporting belongs to the CLI; run_triage returns."""
        import sys

        argv = sys.argv
        sys.argv = ["triage", "--source", "/nope/does-not-exist.hip"]
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                main()
        finally:
            sys.argv = argv
        assert "source not found" in buffer.getvalue()
