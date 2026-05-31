"""Tests for Tier 5 ``custom_patterns`` runner (FR 2.5, 2.6, 2.7)."""

from __future__ import annotations

import re

import pytest

from aorta.probe.classifier.tier5_custom import (
    CompiledPattern,
    CustomScanResult,
    scan,
    validate_custom_patterns,
)
from aorta.probe.sandbox import SandboxError, validate_and_compile
from aorta.triage.recipe import RecipeSchemaError


def _make(
    *,
    raw_id: str,
    regex: str,
    on_match: str = "fail",
    condition: str | None = None,
    required_for_pass: bool = False,
) -> CompiledPattern:
    compiled_condition = validate_and_compile(condition) if condition else None
    return CompiledPattern(
        detector_id=f"custom:{raw_id}",
        regex=re.compile(regex),
        condition_code=compiled_condition,
        condition_source=condition,
        on_match=on_match,  # type: ignore[arg-type]
        required_for_pass=required_for_pass,
    )


# ---- scan() behaviour -----------------------------------------------------


def test_simple_match_fires_fail_detector():
    pattern = _make(raw_id="oom_killer", regex=r"oom_killer triggered")
    result = scan(
        "step 1\noom_killer triggered\nstep 2\n",
        (pattern,),
        exit_code=137,
        walltime_sec=12.0,
        peak_vram_mib=None,
    )
    assert result.fail_detectors == ["custom:oom_killer"]
    assert result.warn_detectors == []


def test_warn_match_does_not_fail():
    pattern = _make(raw_id="slow_iter", regex=r"slow iteration", on_match="warn")
    result = scan(
        "slow iteration detected\n",
        (pattern,),
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert result.fail_detectors == []
    assert result.warn_detectors == ["custom:slow_iter"]


def test_info_only_pattern_populates_capture():
    pattern = _make(
        raw_id="bw",
        regex=r"effective bandwidth: (?P<bw>[0-9]+) GB/s",
        on_match="info",
    )
    result = scan(
        "effective bandwidth: 240 GB/s\n",
        (pattern,),
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert result.fail_detectors == []
    assert result.warn_detectors == []
    assert result.capture.get("bw") == "240"


def test_condition_gates_match():
    """The condition must be true for the detector to fire."""
    pattern = _make(
        raw_id="oom_under_8gb",
        regex=r"out of memory",
        condition="peak_vram_mib < 8000",
    )
    result_low = scan(
        "out of memory\n",
        (pattern,),
        exit_code=1,
        walltime_sec=1.0,
        peak_vram_mib=4000,
    )
    assert result_low.fail_detectors == ["custom:oom_under_8gb"]
    result_high = scan(
        "out of memory\n",
        (pattern,),
        exit_code=1,
        walltime_sec=1.0,
        peak_vram_mib=64000,
    )
    assert result_high.fail_detectors == []


def test_required_for_pass_tracked():
    pattern = _make(
        raw_id="benchmark_ok",
        regex=r"benchmark passed",
        on_match="fail",  # only fail patterns may be required_for_pass
        required_for_pass=True,
    )
    result_hit = scan(
        "benchmark passed\n",
        (pattern,),
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert "custom:benchmark_ok" in result_hit.fired_required_ids
    result_miss = scan(
        "other output\n",
        (pattern,),
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert result_miss.fired_required_ids == set()


def test_no_match_no_capture():
    pattern = _make(
        raw_id="bw",
        regex=r"effective bandwidth: (?P<bw>[0-9]+) GB/s",
    )
    result = scan(
        "totally unrelated log\n",
        (pattern,),
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert result.fail_detectors == []
    assert result.capture == {}


def test_window_cap_does_not_explode_on_huge_log():
    """A 12 MiB log scans cleanly without OOM."""
    big = ("xx\n" * (2 * 1024 * 1024)) + "FATAL: kaboom\n"
    pattern = _make(raw_id="kaboom", regex=r"FATAL: kaboom")
    result = scan(
        big,
        (pattern,),
        exit_code=1,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert isinstance(result, CustomScanResult)


def test_window_overlap_catches_straddling_match(monkeypatch):
    """Regression for PR #197 review (Sonbol, sweep from Tier 4 to
    Tier 5): a custom_patterns match that straddles a window
    boundary must still fire.

    Operator-supplied custom patterns can legitimately match
    multi-line regions (stack frames, JSON payloads); the
    previous non-overlapping ``_iter_windows`` silently dropped
    those when the match seam fell on a chunk boundary. The fix
    mirrors Tier 4's overlap, defended below by shrinking the
    window and planting a match that crosses it.
    """
    from aorta.probe.classifier import tier5_custom

    monkeypatch.setattr(tier5_custom, "MAX_LOG_BYTES", 100)

    pad_left = "x" * 80
    pad_right = "y" * 80
    straddling = pad_left + "FATAL: payload boundary kaboom" + pad_right
    pattern = _make(raw_id="straddle", regex=r"FATAL: payload boundary kaboom")
    result = tier5_custom.scan(
        straddling,
        (pattern,),
        exit_code=1,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert "custom:straddle" in result.fail_detectors, (
        "straddling match must fire across the chunk seam; "
        "non-overlapping windows would silently miss it"
    )


def test_empty_patterns_returns_empty_result():
    result = scan(
        "anything",
        (),
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert result.fail_detectors == []
    assert result.warn_detectors == []
    assert result.capture == {}


def test_empty_log_returns_empty_result():
    pattern = _make(raw_id="x", regex=r"x")
    result = scan(
        "",
        (pattern,),
        exit_code=0,
        walltime_sec=1.0,
        peak_vram_mib=None,
    )
    assert result.fail_detectors == []


# ---- validate_custom_patterns() at recipe load -----------------------------


def test_validate_accepts_simple_recipe():
    raw = [
        {
            "id": "oom",
            "match": {"regex": "out of memory"},
            "on_match": "fail",
        }
    ]
    compiled = validate_custom_patterns(raw)
    assert len(compiled) == 1
    assert compiled[0].detector_id == "custom:oom"


def test_validate_none_returns_empty_tuple():
    assert validate_custom_patterns(None) == ()


def test_validate_rejects_empty_list():
    with pytest.raises(RecipeSchemaError, match="non-empty"):
        validate_custom_patterns([])


def test_validate_rejects_duplicate_ids():
    raw = [
        {"id": "x", "match": {"regex": "a"}},
        {"id": "x", "match": {"regex": "b"}},
    ]
    with pytest.raises(RecipeSchemaError, match="duplicate id"):
        validate_custom_patterns(raw)


def test_validate_rejects_invalid_regex():
    raw = [{"id": "broken", "match": {"regex": "(unclosed"}}]
    with pytest.raises(RecipeSchemaError, match="invalid regex"):
        validate_custom_patterns(raw)


def test_validate_rejects_hostile_condition():
    raw = [
        {
            "id": "p",
            "match": {
                "regex": "fatal",
                "condition": "__import__('os').system('id')",
            },
        }
    ]
    with pytest.raises(SandboxError):
        validate_custom_patterns(raw)


def test_sandbox_error_carries_recipe_path():
    """Regression for PR #197 review (Copilot): when a hostile
    ``condition`` survives parse, the raised ``SandboxError`` must
    name the offending ``custom_patterns[i]`` entry so a multi-entry
    recipe makes the bad entry findable without manual list-position
    counting. The wrapping preserves the original sandbox detail at
    the end of the message so no diagnostic is lost.
    """
    raw = [
        {"id": "ok", "match": {"regex": "ok"}},
        {"id": "ok2", "match": {"regex": "ok2"}},
        {
            "id": "bad",
            "match": {
                "regex": "fatal",
                "condition": "__import__('os').system('id')",
            },
        },
    ]
    with pytest.raises(SandboxError) as exc_info:
        validate_custom_patterns(raw)
    msg = str(exc_info.value)
    assert "recipe.custom_patterns[2].match.condition" in msg, (
        f"SandboxError must carry the recipe path; got: {msg!r}"
    )
    # The original sandbox detail (`forbidden`) survives the wrap.
    assert "forbidden" in msg.lower() or "__import__" in msg, (
        f"wrapped SandboxError dropped the original sandbox detail; got: {msg!r}"
    )


def test_validate_rejects_unknown_top_keys():
    raw = [{"id": "p", "match": {"regex": "x"}, "bogus": True}]
    with pytest.raises(RecipeSchemaError, match="unknown keys"):
        validate_custom_patterns(raw)


def test_validate_rejects_invalid_on_match():
    raw = [{"id": "p", "match": {"regex": "x"}, "on_match": "explode"}]
    with pytest.raises(RecipeSchemaError, match="on_match"):
        validate_custom_patterns(raw)


def test_validate_rejects_required_for_pass_on_warn():
    raw = [
        {
            "id": "p",
            "match": {"regex": "x"},
            "on_match": "warn",
            "required_for_pass": True,
        }
    ]
    with pytest.raises(RecipeSchemaError, match="required_for_pass"):
        validate_custom_patterns(raw)


def test_validate_accepts_condition_at_load_time():
    raw = [
        {
            "id": "vram_check",
            "match": {
                "regex": "vram=(?P<used>[0-9]+)",
                "condition": "int(capture['used']) > 1000",
            },
        }
    ]
    compiled = validate_custom_patterns(raw)
    assert compiled[0].condition_code is not None
    assert compiled[0].condition_source == "int(capture['used']) > 1000"
