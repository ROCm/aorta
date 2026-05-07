"""Tests for ``aorta.run.cli_helpers``.

These pin the contract that lives between the CLI shim and the
library API: the helpers must produce the same shapes whether
called from Click or directly from B2 / a programmatic caller.
"""

import pytest

from aorta.run.cli_helpers import (
    RunSummary,
    parse_csv,
    parse_extra_env,
    parse_mitigations,
    summarize_results,
)
from aorta.run.results import TrialResult


def _result(trial_id: str, exit_status: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        workload="w",
        execution_env={},
        mitigations_applied=(),
        config={},
        env={},
        result={},
        wall_clock_sec=0.0,
        exit_status=exit_status,  # type: ignore[arg-type]
    )


class TestParseCsv:
    def test_empty_string(self):
        assert parse_csv("") == ()

    def test_only_commas_or_whitespace(self):
        assert parse_csv(", , ,") == ()

    def test_strips_and_drops_empty(self):
        assert parse_csv(" a , b ,, c ") == ("a", "b", "c")


class TestParseMitigations:
    def test_empty_defaults_to_none(self):
        """Empty CLI string must map to the documented baseline."""
        assert parse_mitigations("") == ("none",)

    def test_explicit_none(self):
        assert parse_mitigations("none") == ("none",)

    def test_multiple(self):
        assert parse_mitigations("tf32_off, hsa_xnack") == ("tf32_off", "hsa_xnack")


class TestParseExtraEnv:
    def test_empty(self):
        assert parse_extra_env("") == {}

    def test_simple(self):
        assert parse_extra_env("DEBUG=1,VERBOSE=true") == {"DEBUG": "1", "VERBOSE": "true"}

    def test_strips_whitespace(self):
        assert parse_extra_env(" DEBUG = 1 , VERBOSE = true ") == {
            "DEBUG": "1",
            "VERBOSE": "true",
        }

    def test_value_can_contain_equals(self):
        """``split('=', 1)`` keeps the rest of the value intact."""
        assert parse_extra_env("PYTHONPATH=/a/b=c:/d") == {"PYTHONPATH": "/a/b=c:/d"}

    def test_missing_equals_raises_valueerror(self):
        """Format errors raise ``ValueError`` -- the CLI bridges to ClickException."""
        with pytest.raises(ValueError, match="Invalid extra-env format"):
            parse_extra_env("NOEQUALS")

    def test_empty_key_raises_valueerror(self):
        with pytest.raises(ValueError, match="key is empty"):
            parse_extra_env("=value")

    def test_does_not_validate_key_shape(self):
        """Shape validation is ``run_trials``' job (parity with library callers).

        A bad key like ``1BAD`` parses fine here; ``run_trials``
        re-checks every ``extra_env`` key so callers that bypass this
        parser still hit the gate.
        """
        assert parse_extra_env("1BAD=ok") == {"1BAD": "ok"}


class TestSummarizeResults:
    def test_all_passed(self):
        rs = summarize_results([_result("w_d0_m0_t0", "ok"), _result("w_d0_m0_t1", "ok")])
        assert rs == RunSummary(total=2, passed=2, failed=0, failed_trial_ids=())

    def test_mixed(self):
        rs = summarize_results(
            [
                _result("w_d0_m0_t0", "ok"),
                _result("w_d0_m0_t1", "workload_failed"),
                _result("w_d0_m0_t2", "infrastructure_failed"),
            ]
        )
        assert rs.total == 3
        assert rs.passed == 1
        assert rs.failed == 2
        assert rs.failed_trial_ids == ("w_d0_m0_t1", "w_d0_m0_t2")

    def test_empty(self):
        rs = summarize_results([])
        assert rs == RunSummary(total=0, passed=0, failed=0, failed_trial_ids=())

    def test_accepts_any_iterable(self):
        """B2 may pass a generator; the helper must not require a list."""

        def gen():
            yield _result("w_d0_m0_t0", "ok")
            yield _result("w_d0_m0_t1", "ok")

        rs = summarize_results(gen())
        assert rs.total == 2
        assert rs.passed == 2
