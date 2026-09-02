"""Watch escalates only when it is unhealthy *and* sure.

The threshold is the whole reason Watch is worth listening to. Without it every
suspicion becomes an alert, and its readers learn to ignore all of them.

The 0.65 case is not hypothetical. On a real run Watch reported a possible hang
forty seconds in, because the job had not written anything yet and "stalled"
and "still starting" are indistinguishable at that point. It stayed quiet, kept
watching, and alerted on the second poll with the actual finding.
"""

from __future__ import annotations

import pytest
import yaml

from aorta.cia.watch.poll import should_alert

_CONFIG = "src/aorta/cia/watch/watch_config.yaml"


@pytest.mark.parametrize(
    "healthy, confidence, expected",
    [
        (False, 0.95, True),   # a sanitizer finding it is sure about
        (False, 0.99, True),   # a NaN in the log
        (False, 0.70, True),   # exactly at the threshold: sure enough
        (False, 0.6999, False),
        (False, 0.65, False),  # the real hang-vs-starting case
        (False, 0.0, False),
        (True, 0.99, False),   # confident that nothing is wrong
        (True, 0.10, False),
    ],
)
def test_escalation_requires_unhealthy_and_confident(healthy, confidence, expected):
    assert should_alert(healthy, confidence, 0.70) is expected


def test_a_healthy_verdict_never_escalates_however_confident():
    """Confidence qualifies the judgement; it does not override it."""
    assert not any(should_alert(True, c / 100, 0.70) for c in range(101))


def test_the_shipped_threshold_is_the_documented_one(repo_root):
    """A default nobody can find is a default nobody can trust."""
    config = yaml.safe_load((repo_root / _CONFIG).read_text(encoding="utf-8"))
    assert config["watch"]["confidence_threshold"] == 0.70


def test_expectations_are_prose_not_patterns(repo_root):
    """Watch is told what healthy looks like in English, deliberately.

    Every workload logs differently, so a regex would need writing per job. The
    model maps "loss should be decreasing" onto whatever that job calls loss,
    which is why adding an expectation is a line of YAML rather than code.
    """
    config = yaml.safe_load((repo_root / _CONFIG).read_text(encoding="utf-8"))
    expectations = config["watch"]["expectations"]
    assert len(expectations) >= 5
    for expectation in expectations:
        assert " " in expectation.strip(), f"{expectation!r} looks like a pattern"
