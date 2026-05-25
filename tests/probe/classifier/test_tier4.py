"""Tests for Tier 4 built-in pattern library (FR 2.4).

Each fixture under ``tests/probe/fixtures/tier4_logs/`` holds a log
sample that should fire exactly one Tier-4 detector.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.probe.classifier import tier4_patterns
from aorta.probe.classifier.tier4_patterns import (
    BUILTIN_PATTERN_VERSION,
    all_patterns,
    scan,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tier4_logs"


@pytest.mark.parametrize(
    "fixture_name,detector_id",
    [
        ("python_traceback.txt", tier4_patterns.DETECTOR_PYTHON_TRACEBACK),
        ("hip_error.txt", tier4_patterns.DETECTOR_HIP_ERROR),
        ("cuda_error.txt", tier4_patterns.DETECTOR_CUDA_ERROR),
        ("rocm_error.txt", tier4_patterns.DETECTOR_ROCM_ERROR),
        ("nccl_rccl_error.txt", tier4_patterns.DETECTOR_NCCL_RCCL_ERROR),
        ("collective_timeout.txt", tier4_patterns.DETECTOR_COLLECTIVE_TIMEOUT),
        ("nan_signature.txt", tier4_patterns.DETECTOR_NAN_SIGNATURE),
    ],
)
def test_each_fixture_fires_its_detector(fixture_name, detector_id):
    text = (FIXTURES / fixture_name).read_text(encoding="utf-8")
    fired = scan(text)
    assert detector_id in fired, f"expected {detector_id} for fixture {fixture_name}; got {fired}"


def test_empty_log_fires_nothing():
    assert scan("") == []


def test_unrelated_log_fires_nothing():
    assert scan("everything is fine\nstep 100/100 done") == []


def test_multiple_patterns_in_one_log():
    """A log with both a traceback and an HIP error fires both detectors."""
    text = (
        "Traceback (most recent call last):\n"
        "  File 'x.py'\n"
        "RuntimeError: hipError_OutOfMemory\n"
    )
    fired = scan(text)
    assert tier4_patterns.DETECTOR_PYTHON_TRACEBACK in fired
    assert tier4_patterns.DETECTOR_HIP_ERROR in fired


def test_builtin_pattern_version_is_string_one():
    """The version starts at '1' for the Phase-2 PR (rubric §X.1)."""
    assert BUILTIN_PATTERN_VERSION == "1"


def test_catalogue_entries_have_compiled_regex():
    """Every entry exposes a pre-compiled regex (cost paid once)."""
    import re

    for pattern in all_patterns():
        assert isinstance(pattern.regex, re.Pattern)
        assert pattern.detector_id.startswith("tier4:")
        assert pattern.sample  # non-empty sample line
