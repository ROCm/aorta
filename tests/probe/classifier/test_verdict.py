"""Tests for the verdict-precedence resolver (FR 2.8)."""

from __future__ import annotations

import re

import pytest

from aorta.probe.classifier.tier5_custom import CompiledPattern, CustomScanResult
from aorta.probe.classifier.verdict import (
    DETECTOR_MISSING_PASS_SIGNAL,
    VALID_VERDICTS,
    VerdictInputs,
    resolve,
)


def test_valid_verdicts_is_the_three_way_vocabulary():
    """The canonical verdict set downstream layers validate against is exactly
    {pass, fail, error} -- no more, no less (issue #230)."""
    assert VALID_VERDICTS == {"pass", "fail", "error"}


def _required_pattern(raw_id: str) -> CompiledPattern:
    return CompiledPattern(
        detector_id=f"custom:{raw_id}",
        regex=re.compile("x"),
        condition_code=None,
        condition_source=None,
        on_match="fail",
        required_for_pass=True,
    )


def _inputs(**kw: object) -> VerdictInputs:
    base: dict[str, object] = {
        "tier1": [],
        "tier2": [],
        "tier3": [],
        "tier4": [],
        "custom_result": CustomScanResult(),
        "custom_required_patterns": (),
    }
    base.update(kw)
    return VerdictInputs(**base)  # type: ignore[arg-type]


def test_pass_when_nothing_fired():
    verdict = resolve(_inputs())
    assert verdict.verdict == "pass"
    assert verdict.failure_detectors_fired == []
    assert verdict.warn_detectors_fired == []


def test_tier1_fail_alone_is_enough():
    verdict = resolve(_inputs(tier1=["tier1:sigsegv"]))
    assert verdict.verdict == "fail"
    assert verdict.failure_detectors_fired == ["tier1:sigsegv"]


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({"tier1": ["tier1:exit_nonzero"]}, "tier1:exit_nonzero"),
        ({"tier2": ["tier2:hang"]}, "tier2:hang"),
        ({"tier3": ["tier3:amdgpu_reset"]}, "tier3:amdgpu_reset"),
        ({"tier4": ["tier4:hip_error"]}, "tier4:hip_error"),
    ],
)
def test_any_tier_failure_is_a_fail(kw, expected):
    verdict = resolve(_inputs(**kw))
    assert verdict.verdict == "fail"
    assert expected in verdict.failure_detectors_fired


def test_custom_fail_pattern_is_a_fail():
    custom = CustomScanResult(fail_detectors=["custom:my_pattern"])
    verdict = resolve(_inputs(custom_result=custom))
    assert verdict.verdict == "fail"
    assert "custom:my_pattern" in verdict.failure_detectors_fired


def test_warn_alone_is_pass():
    custom = CustomScanResult(warn_detectors=["custom:slow_iter"])
    verdict = resolve(_inputs(custom_result=custom))
    assert verdict.verdict == "pass"
    assert verdict.failure_detectors_fired == []
    assert verdict.warn_detectors_fired == ["custom:slow_iter"]


def test_tier3_warn_is_pass_not_fail():
    """Advisory Tier-3 detectors (e.g. vram_growth) warn, never fail."""
    verdict = resolve(_inputs(tier3_warn=["tier3:vram_growth"]))
    assert verdict.verdict == "pass"
    assert verdict.failure_detectors_fired == []
    assert verdict.warn_detectors_fired == ["tier3:vram_growth"]


def test_tier3_warn_does_not_mask_a_real_failure():
    """A warn-level tier3 ID coexists with a hard failure from another tier."""
    verdict = resolve(
        _inputs(tier3=["tier3:amdgpu_reset"], tier3_warn=["tier3:vram_growth"])
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_detectors_fired == ["tier3:amdgpu_reset"]
    assert verdict.warn_detectors_fired == ["tier3:vram_growth"]


def test_info_match_populates_capture_only():
    """Info-only patterns populate ``capture`` but never the verdict lists."""
    custom = CustomScanResult(capture={"bw": "240"})
    verdict = resolve(_inputs(custom_result=custom))
    assert verdict.verdict == "pass"
    assert verdict.capture == {"bw": "240"}


def test_encounter_order_preserved_across_tiers():
    """All tiers contribute in T1->T2->T3->T4->custom-fail order."""
    custom = CustomScanResult(fail_detectors=["custom:p"])
    verdict = resolve(
        _inputs(
            tier1=["tier1:exit_nonzero"],
            tier2=["tier2:hang"],
            tier3=["tier3:amdgpu_reset"],
            tier4=["tier4:hip_error", "tier4:nan_signature"],
            custom_result=custom,
        )
    )
    assert verdict.failure_detectors_fired == [
        "tier1:exit_nonzero",
        "tier2:hang",
        "tier3:amdgpu_reset",
        "tier4:hip_error",
        "tier4:nan_signature",
        "custom:p",
    ]


def test_required_for_pass_missing_marks_fail():
    """``required_for_pass`` not fired -> ``meta:missing_pass_signal``."""
    required = _required_pattern("benchmark_ok")
    custom = CustomScanResult()  # nothing fired
    verdict = resolve(
        _inputs(
            custom_result=custom,
            custom_required_patterns=(required,),
        )
    )
    assert verdict.verdict == "fail"
    assert DETECTOR_MISSING_PASS_SIGNAL in verdict.failure_detectors_fired


def test_required_for_pass_satisfied_no_meta():
    """When every required pattern fired, ``meta:missing_pass_signal`` is NOT injected."""
    required = _required_pattern("benchmark_ok")
    custom = CustomScanResult(
        fail_detectors=["custom:benchmark_ok"],
        fired_required_ids={"custom:benchmark_ok"},
    )
    verdict = resolve(
        _inputs(
            custom_result=custom,
            custom_required_patterns=(required,),
        )
    )
    assert DETECTOR_MISSING_PASS_SIGNAL not in verdict.failure_detectors_fired


def test_required_for_pass_no_required_patterns_no_meta():
    """When the recipe declares no required patterns, meta detector is never injected."""
    verdict = resolve(_inputs())
    assert DETECTOR_MISSING_PASS_SIGNAL not in verdict.failure_detectors_fired


# ---- Three-way verdict: error (issue #230) ------------------------------


def test_timeout_alone_is_error():
    """A timeout with no recognised hang is an ``error`` (no valid obs)."""
    verdict = resolve(_inputs(tier1=["tier1:timeout"]))
    assert verdict.verdict == "error"
    assert verdict.error_detectors_fired == ["tier1:timeout"]
    assert verdict.failure_detectors_fired == []


def test_exec_failed_is_error():
    """A launch failure (``tier1:exec_failed``) is an ``error``."""
    verdict = resolve(_inputs(tier1=["tier1:exec_failed"]))
    assert verdict.verdict == "error"
    assert verdict.error_detectors_fired == ["tier1:exec_failed"]
    assert verdict.failure_detectors_fired == []


def test_timeout_with_hang_is_fail_not_error():
    """fail > error: a recognised hang co-firing with a timeout wins."""
    verdict = resolve(_inputs(tier1=["tier1:timeout"], tier2=["tier2:hang"]))
    assert verdict.verdict == "fail"
    # The timeout still surfaces in the error list for the operator, but the
    # hang (a genuine failure) drives the verdict.
    assert verdict.failure_detectors_fired == ["tier2:hang"]
    assert verdict.error_detectors_fired == ["tier1:timeout"]


def test_fail_beats_error_across_tiers():
    """A real failure anywhere outranks a co-firing error detector."""
    verdict = resolve(
        _inputs(tier1=["tier1:exec_failed"], tier4=["tier4:hip_error"])
    )
    assert verdict.verdict == "fail"
    assert verdict.failure_detectors_fired == ["tier4:hip_error"]
    assert verdict.error_detectors_fired == ["tier1:exec_failed"]


def test_exit_nonzero_stays_fail_not_error():
    """A plain non-zero exit is a *valid* observation of a failure -> fail."""
    verdict = resolve(_inputs(tier1=["tier1:exit_nonzero"]))
    assert verdict.verdict == "fail"
    assert verdict.error_detectors_fired == []


def test_partition_helper_splits_by_error_set():
    from aorta.probe.classifier.verdict import partition_detectors

    failures, errors = partition_detectors(
        ["tier1:timeout", "tier1:exit_nonzero", "tier1:exec_failed", "tier4:nan"]
    )
    assert failures == ["tier1:exit_nonzero", "tier4:nan"]
    assert errors == ["tier1:timeout", "tier1:exec_failed"]


def test_returned_lists_are_fresh_copies():
    """Caller mutating verdict.failure_detectors_fired should not poison resolver state."""
    verdict = resolve(_inputs(tier1=["tier1:exit_nonzero"]))
    verdict.failure_detectors_fired.append("mutated")
    # The second resolve must NOT see "mutated"; the resolver builds a new list each call.
    verdict2 = resolve(_inputs(tier1=["tier1:exit_nonzero"]))
    assert "mutated" not in verdict2.failure_detectors_fired
