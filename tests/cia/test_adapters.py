"""Each adapter turns one kind of artifact into a signal.

Two properties matter more than the mapping itself.

An adapter that cannot find its artifact must produce *no* signal, never a
clean one. "The sanitizer did not run" and "the sanitizer found nothing" look
identical in a report that conflates them, and the first has already been
mistaken for the second: a sweep reported not-checked because its backend was
absent, and every layer above read the absence of findings as a pass.

And an observed failure outranks an inferred one. A sanitizer that watched two
waves collide knows more than one that read the instruction stream and found a
hazard on a path that may never execute, and the confidence has to say so.
"""

from __future__ import annotations

from aorta.cia.autopsy.adapters.rocgdb import classify_rocgdb, parse_rocgdb_session
from aorta.cia.autopsy.adapters.sanitizer_report import classify_sanitizer
from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text


def _report(
    findings: list[dict],
    verdict: str = "warn",
    status: str = "complete",
    state: str = "ran",
    reason: str = "",
) -> dict:
    check: dict = {"findings": findings, "state": state, "sanitizer": "waitcheck"}
    if reason:
        check["reason"] = reason
    return {
        "schema": "sanitizer_report/v1",
        "target": "gfx950",
        "overall_verdict": verdict,
        "execution_status": status,
        "checks": [check],
    }


# ── stderr ────────────────────────────────────────────────────────────────


def test_a_non_finite_loss_is_a_signal():
    scan = scan_stderr_text("step 4 loss=6.40\nstep 5 loss=nan\naborting run\n")
    assert scan.signal == "WATCH_NUMERIC_NAN"
    assert scan.alert


def test_a_healthy_log_is_clean():
    scan = scan_stderr_text("step 4 loss=6.40\nstep 5 loss=6.31\n")
    assert scan.signal == "WATCH_CLEAN"
    assert not scan.alert


def test_the_word_inference_is_not_a_nan():
    """`inf` is a substring of `inference`, and a throughput note is not a fault.

    A matcher keyed on substrings fires here; that is the class of false
    positive the adapters must not reintroduce.
    """
    scan = scan_stderr_text("inference throughput dropped to 812 tok/s\n")
    assert scan.signal == "WATCH_CLEAN"


# ── sanitizer report ──────────────────────────────────────────────────────


def test_an_observed_race_is_reported_as_one():
    result = classify_sanitizer(
        _report(
            [{"code": "1", "sanitizer": "consan", "severity": "race",
              "message": "auto replay diagnostic conflict=true"}],
            verdict="fail",
        )
    )
    assert "SAN_CONSAN_RACE" in result.signals
    assert result.category == "gpu_race"


def test_a_static_hazard_is_reported_with_less_confidence_than_an_observed_race():
    """Both are gpu_race. Only one of them was watched happening."""
    observed = classify_sanitizer(
        _report([{"code": "1", "sanitizer": "consan", "severity": "race",
                  "message": "conflict=true"}], verdict="fail")
    )
    inferred = classify_sanitizer(
        _report([{"code": "wait_hazard", "sanitizer": "waitcheck", "severity": "warning",
                  "message": "missing s_waitcnt lgkmcnt(0) before def of s4"}])
    )
    assert "SAN_WAITCHECK_HAZARD" in inferred.signals
    assert inferred.confidence < observed.confidence, (
        "a hazard read from the instruction stream must not claim the "
        "confidence of a collision that was observed"
    )


def test_a_sanitizer_that_could_not_run_is_not_a_pass():
    """The regression that motivates this file."""
    result = classify_sanitizer(
        _report(
            [],
            verdict="not_checked",
            status="not_checked",
            state="not_checked",
            reason="rj_waitcheck_not_found",
        )
    )
    assert "SAN_CLEAN" not in result.signals, (
        "a sweep whose backend was missing reported itself as clean; "
        "'did not run' and 'found nothing' must not be the same signal"
    )
    assert "SAN_NOT_CHECKED" in result.signals


def test_a_clean_run_says_so():
    result = classify_sanitizer(_report([], verdict="pass", status="complete"))
    assert "SAN_CLEAN" in result.signals


# ── debugger session ──────────────────────────────────────────────────────


_SESSION = (
    'Thread 7 "probe_trap" received signal SIGABRT, Aborted.\n'
    '__assert_fail (assertion=0x7ffff5df0b3f <str> "isfinite(y)", '
    'file=0x7ffff5df0baf <str> "rmsnorm.hip", line=74, '
    'function=<__PRETTY_FUNCTION__.rmsnorm_forward>) at hip_assert.h:85\n'
    "\n"
    "===== DEVICE ASSERT TRAPPED =====\n"
    '* 7  AMDGPU Wave 1:1:1:2 (5,0,0)/0 "probe_trap"\n'
    "#1  0x00007fff in rmsnorm_forward(float const*, float*, int, int) at rmsnorm.hip:74\n"
    "nan-trap value mean_sq=0\n"
    "nan-trap value inv_rms=inf\n"
    "nan-trap value y=-nan\n"
)


def test_a_trapped_assert_names_the_source_line():
    session = parse_rocgdb_session(_SESSION)
    classification = classify_rocgdb(session)
    assert "DBG_DEVICE_ASSERT" in classification.signals
    assert classification.category == "numeric_silent"


def test_the_captured_registers_reach_the_classification():
    """Register state is the whole reason to attach a debugger.

    Without it the verdict is 'a bad number appeared somewhere'; with it the
    chain from mean_sq to y is in the report.
    """
    session = parse_rocgdb_session(_SESSION)
    classification = classify_rocgdb(session)
    assert "DBG_NAN_TRAP" in classification.signals
    assert classification.confidence >= 0.9


def test_an_empty_session_claims_nothing():
    """No debugger output is not evidence of a healthy run."""
    classification = classify_rocgdb(parse_rocgdb_session(""))
    assert classification.confidence < 0.5
    assert "DBG_NAN_TRAP" not in classification.signals
