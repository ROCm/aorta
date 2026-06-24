"""Tests for the detector-disable knob (issue #229).

Covers the token-validation layer (:mod:`aorta.probe.classifier.disables`)
and its effect inside :func:`aorta.probe.classifier.classify_trial`: a
disabled detector must never flip the verdict nor appear in the fired
lists, and a disabled whole tier must skip evaluation entirely.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aorta.probe.classifier import TrialContext, classify_trial
from aorta.probe.classifier.disables import (
    DetectorSpecError,
    normalize_detector_id,
    normalize_detector_ids,
    normalize_tier,
    normalize_tiers,
)
from aorta.probe.classifier.tier3_kernel import (
    DETECTOR_VRAM_GROWTH,
    AmdSmiSnapshot,
    Tier3State,
)
from aorta.probe.classifier.tier5_custom import CompiledPattern


# --------------------------------------------------------------------------
# Token validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tok", ["tier1", "tier2", "tier3", "tier4", "tier5"])
def test_normalize_tier_accepts_known(tok: str) -> None:
    assert normalize_tier(tok) == tok


def test_normalize_tier_is_case_insensitive_and_trims() -> None:
    assert normalize_tier("  TIER3 ") == "tier3"


@pytest.mark.parametrize("tok", ["tier0", "tier6", "hang", "tier3:hang", ""])
def test_normalize_tier_rejects_unknown(tok: str) -> None:
    with pytest.raises(DetectorSpecError):
        normalize_tier(tok)


@pytest.mark.parametrize(
    "tok",
    ["tier2:hang", "tier3:vram_growth", "tier4:hip_error", "custom:my_id"],
)
def test_normalize_detector_id_accepts_known_prefixes(tok: str) -> None:
    assert normalize_detector_id(tok) == tok


def test_normalize_detector_id_lowercases_prefix_only() -> None:
    # Prefix canonicalised; the id half is preserved verbatim because
    # custom-pattern ids are user-named and case-sensitive. (Built-in
    # ids are validated against the catalogue below, so a custom id is
    # the right vehicle for the case-preservation contract.)
    assert normalize_detector_id("CUSTOM:My_Id") == "custom:My_Id"


@pytest.mark.parametrize(
    "tok",
    [
        "tier1:exit_nonzero",
        "tier2:hang",
        "tier3:vram_growth",
        "tier4:nan_signature",
    ],
)
def test_normalize_detector_id_accepts_known_builtin_ids(tok: str) -> None:
    # Built-in ids that exist in their tier's ALL_DETECTOR_IDS catalogue
    # pass validation unchanged.
    assert normalize_detector_id(tok) == tok


@pytest.mark.parametrize(
    "tok",
    [
        "tier1:nope",
        "tier3:vram_growht",  # typo
        "tier4:python_tracback",  # typo
        "tier2:Hang",  # built-in id is case-sensitive against catalogue
    ],
)
def test_normalize_detector_id_rejects_unknown_builtin_id(tok: str) -> None:
    # A built-in (tier1-4) id that is not in its tier's catalogue is a
    # typo and must fail at parse time instead of silently disabling
    # nothing.
    with pytest.raises(DetectorSpecError):
        normalize_detector_id(tok)


def test_normalize_detector_id_custom_stays_free_form() -> None:
    # custom:* ids are user-named -- any non-empty id is accepted.
    assert normalize_detector_id("custom:anything_goes") == "custom:anything_goes"


@pytest.mark.parametrize("tok", ["hang", "tier2:", ":hang", "tier9:foo", "tierX:y"])
def test_normalize_detector_id_rejects_malformed(tok: str) -> None:
    with pytest.raises(DetectorSpecError):
        normalize_detector_id(tok)


@pytest.mark.parametrize(
    ("tok", "expected"),
    [
        ("tier2: hang", "tier2:hang"),
        ("  tier2 : hang  ", "tier2:hang"),
        ("custom:  My_Id ", "custom:My_Id"),
    ],
)
def test_normalize_detector_id_strips_whitespace_around_id(tok: str, expected: str) -> None:
    # A copy/paste token with a space after the colon must canonicalise to
    # the bare id so it actually matches the fired detector id; case in the
    # id half is still preserved.
    assert normalize_detector_id(tok) == expected


@pytest.mark.parametrize("tok", ["tier2:   ", "custom: \t "])
def test_normalize_detector_id_rejects_whitespace_only_id(tok: str) -> None:
    # An id that is empty after trimming counts as missing.
    with pytest.raises(DetectorSpecError):
        normalize_detector_id(tok)


def test_normalize_lists_dedupe_preserve_order() -> None:
    assert normalize_tiers(["tier3", "tier2", "tier3"]) == ("tier3", "tier2")
    assert normalize_detector_ids(["tier2:hang", "tier2:hang"]) == ("tier2:hang",)


def test_normalize_lists_none_is_empty() -> None:
    assert normalize_tiers(None) == ()
    assert normalize_detector_ids(None) == ()


@pytest.mark.parametrize("bad", ["tier3", 5, {"a": 1}])
def test_normalize_lists_reject_non_list(bad: object) -> None:
    # A bare string is a common mistake (forgetting the YAML list dash);
    # it must be rejected rather than iterated character-by-character.
    with pytest.raises(DetectorSpecError):
        normalize_tiers(bad)


# --------------------------------------------------------------------------
# classify_trial integration
# --------------------------------------------------------------------------
def _ctx(tmp_path: Path, **kw: object) -> TrialContext:
    base: dict[str, object] = {
        "exit_code": 0,
        "timed_out": False,
        "walltime_sec": 1.0,
        "trial_dir": tmp_path,
        "log_text": "",
    }
    base.update(kw)
    return TrialContext(**base)  # type: ignore[arg-type]


def test_hang_fires_when_not_disabled(tmp_path: Path) -> None:
    verdict, _ = classify_trial(_ctx(tmp_path, hang_detected=True))
    assert verdict.verdict == "fail"
    assert "tier2:hang" in verdict.failure_detectors_fired


def test_disable_tier2_suppresses_hang(tmp_path: Path) -> None:
    verdict, _ = classify_trial(
        _ctx(tmp_path, hang_detected=True, disabled_tiers=frozenset({"tier2"}))
    )
    assert verdict.verdict == "pass"
    assert "tier2:hang" not in verdict.failure_detectors_fired


def test_disable_detector_id_suppresses_hang(tmp_path: Path) -> None:
    verdict, _ = classify_trial(
        _ctx(tmp_path, hang_detected=True, disabled_detectors=frozenset({"tier2:hang"}))
    )
    assert verdict.verdict == "pass"
    assert "tier2:hang" not in verdict.failure_detectors_fired


def test_disable_tier4_suppresses_log_pattern(tmp_path: Path) -> None:
    log = "Traceback... HIP error: out of memory\n"
    fired, _ = classify_trial(_ctx(tmp_path, log_text=log))
    assert fired.verdict == "fail"  # baseline: tier4 fires

    verdict, _ = classify_trial(
        _ctx(tmp_path, log_text=log, disabled_tiers=frozenset({"tier4"}))
    )
    assert "tier4:hip_error" not in verdict.failure_detectors_fired


def test_disable_specific_tier4_id_keeps_others(tmp_path: Path) -> None:
    log = "HIP error: boom\ncudaError_LaunchFailure\n"
    verdict, _ = classify_trial(
        _ctx(tmp_path, log_text=log, disabled_detectors=frozenset({"tier4:hip_error"}))
    )
    assert "tier4:hip_error" not in verdict.failure_detectors_fired
    assert "tier4:cuda_error" in verdict.failure_detectors_fired
    assert verdict.verdict == "fail"


def test_disable_tier3_skips_dmesg_scan(tmp_path: Path) -> None:
    dmesg = "amdgpu: GPU reset begin!\n"
    baseline, _ = classify_trial(
        _ctx(tmp_path, dmesg_text=dmesg, tier3_state=Tier3State())
    )
    assert "tier3:amdgpu_reset" in baseline.failure_detectors_fired

    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            dmesg_text=dmesg,
            tier3_state=Tier3State(),
            disabled_tiers=frozenset({"tier3"}),
        )
    )
    assert "tier3:amdgpu_reset" not in verdict.failure_detectors_fired
    assert verdict.verdict == "pass"


def test_disable_tier3_vram_growth_id_suppresses_warn(tmp_path: Path) -> None:
    # Disabling the warn-level ``tier3:vram_growth`` detector by id must
    # keep it out of the fired lists -- the classifier gates the VRAM
    # delta check on the disabled set before the scan, not only via the
    # post-hoc filter.
    pre = AmdSmiSnapshot(vram_used_mib=4000, thermal_throttle_count=0)
    post = AmdSmiSnapshot(vram_used_mib=71234, thermal_throttle_count=0)
    baseline, _ = classify_trial(
        _ctx(tmp_path, tier3_state=Tier3State(), amd_smi_pre=pre, amd_smi_post=post)
    )
    assert DETECTOR_VRAM_GROWTH in baseline.warn_detectors_fired

    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            tier3_state=Tier3State(),
            amd_smi_pre=pre,
            amd_smi_post=post,
            disabled_detectors=frozenset({DETECTOR_VRAM_GROWTH}),
        )
    )
    assert DETECTOR_VRAM_GROWTH not in verdict.warn_detectors_fired
    assert DETECTOR_VRAM_GROWTH not in verdict.failure_detectors_fired


def test_disable_custom_pattern(tmp_path: Path) -> None:
    pat = CompiledPattern(
        detector_id="custom:boom",
        regex=re.compile("boom"),
        condition_code=None,
        condition_source=None,
        on_match="fail",
        required_for_pass=False,
    )
    log = "...boom...\n"
    baseline, _ = classify_trial(_ctx(tmp_path, log_text=log, custom_patterns=(pat,)))
    assert "custom:boom" in baseline.failure_detectors_fired
    assert baseline.verdict == "fail"

    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            log_text=log,
            custom_patterns=(pat,),
            disabled_detectors=frozenset({"custom:boom"}),
        )
    )
    assert "custom:boom" not in verdict.failure_detectors_fired
    assert verdict.verdict == "pass"


def test_disable_custom_pattern_leaves_no_capture_side_effect(tmp_path: Path) -> None:
    # A disabled custom detector must be truly *not evaluated*: its
    # ``capture`` named groups must not surface (and, by the same token,
    # its sandbox ``condition`` must not run). Filtering happens before
    # the Tier-5 scan, so the silenced detector leaves no trace.
    pat = CompiledPattern(
        detector_id="custom:metric",
        regex=re.compile(r"loss=(?P<loss>[0-9.]+)"),
        condition_code=None,
        condition_source=None,
        on_match="info",
        required_for_pass=False,
    )
    log = "step 1 loss=3.14\n"
    baseline, _ = classify_trial(_ctx(tmp_path, log_text=log, custom_patterns=(pat,)))
    assert baseline.capture.get("loss") == "3.14"

    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            log_text=log,
            custom_patterns=(pat,),
            disabled_detectors=frozenset({"custom:metric"}),
        )
    )
    assert "loss" not in verdict.capture


def test_disable_required_custom_pattern_does_not_synthesize_missing_signal(
    tmp_path: Path,
) -> None:
    # A required-for-pass pattern that the operator disables must not
    # leave behind a meta:missing_pass_signal failure -- disabling means
    # "ignore this detector entirely", required-ness included.
    pat = CompiledPattern(
        detector_id="custom:must_see",
        regex=re.compile("never-matches-zzz"),
        condition_code=None,
        condition_source=None,
        on_match="fail",
        required_for_pass=True,
    )
    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            log_text="nothing here\n",
            custom_patterns=(pat,),
            disabled_detectors=frozenset({"custom:must_see"}),
        )
    )
    assert verdict.verdict == "pass"
    assert all("missing_pass_signal" not in d for d in verdict.failure_detectors_fired)


def test_classify_trial_passes_only_required_patterns_to_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Contract guard: ``VerdictInputs.custom_required_patterns`` is documented
    # to carry only ``required_for_pass=True`` patterns. ``classify_trial``
    # must narrow to that subset before building ``VerdictInputs`` -- even
    # though ``resolve`` re-filters -- so the API contract can't silently
    # drift back to the full active set.
    import aorta.probe.classifier as classifier_mod

    required = CompiledPattern(
        detector_id="custom:must_see",
        regex=re.compile("seen"),
        condition_code=None,
        condition_source=None,
        on_match="fail",
        required_for_pass=True,
    )
    optional = CompiledPattern(
        detector_id="custom:maybe",
        regex=re.compile("maybe"),
        condition_code=None,
        condition_source=None,
        on_match="warn",
        required_for_pass=False,
    )

    captured: dict[str, tuple[CompiledPattern, ...]] = {}
    real_resolve = classifier_mod.resolve

    def _spy_resolve(inputs):  # type: ignore[no-untyped-def]
        captured["required"] = inputs.custom_required_patterns
        return real_resolve(inputs)

    # raising=True (the default, explicit here): if ``resolve`` is ever
    # renamed/removed this guard should fail loudly rather than no-op.
    monkeypatch.setattr(classifier_mod, "resolve", _spy_resolve, raising=True)

    classify_trial(
        _ctx(tmp_path, log_text="seen\n", custom_patterns=(required, optional))
    )

    captured_required = captured.get("required")
    assert captured_required is not None, "resolve() was not invoked"
    assert {p.detector_id for p in captured_required} == {"custom:must_see"}
    assert all(p.required_for_pass for p in captured_required)
