"""Deadlines and the scored-fraction policy (Section 6.6; D7)."""

import pytest

from aorlta.orchestrator import deadlines as dl


def test_the_generation_deadline_is_the_floor_or_three_rolling_medians():
    assert dl.generation_deadline_s(1800.0, []) == 1800.0
    assert dl.generation_deadline_s(1800.0, [100.0, 200.0, 300.0]) == 1800.0
    assert dl.generation_deadline_s(1800.0, [1000.0, 700.0, 900.0]) == 2700.0  # 3 x median 900


def test_the_scoring_deadline_is_the_floor_or_two_rolling_medians():
    assert dl.scoring_deadline_s(1800.0, []) == 1800.0
    assert dl.scoring_deadline_s(1800.0, [800.0, 900.0]) == 1800.0  # 2 x 850 < floor
    assert dl.scoring_deadline_s(1800.0, [1000.0, 1200.0, 5000.0]) == 2400.0  # 2 x median 1200: a stall barely moves it


def test_the_rolling_median_looks_at_the_last_ten_iterations():
    history = [10_000.0] * 5 + [100.0] * 10
    assert dl.rolling_median(history) == 100.0
    assert dl.push_history(list(range(10)), 99.0) == [float(x) for x in range(1, 10)] + [99.0]


def test_the_retry_deadline_is_one_and_a_half_times_the_scoring_deadline():
    assert dl.RETRY_DEADLINE_FACTOR == 1.5


@pytest.mark.parametrize("scored,total,met", [
    (230, 256, False), (231, 256, True), (9, 10, True), (8, 10, False), (90, 100, True), (89, 100, False),
    (256, 256, True), (0, 256, False),
])
def test_the_threshold_is_compared_exactly(scored, total, met):
    assert dl.fraction_met(scored, total, 0.9) is met


def test_an_iteration_at_or_above_the_threshold_trains():
    d = dl.decide(scored=9, total=10, retryable=1, min_fraction=0.9, retried=False)
    assert (d.decision, d.retried) == ("train", False)


def test_below_the_threshold_the_unsettled_legs_are_retried_once():
    d = dl.decide(scored=26, total=32, retryable=4, min_fraction=0.9, retried=False)
    assert d.decision == "retry"
    d = dl.decide(scored=26, total=32, retryable=4, min_fraction=0.9, retried=True)
    assert d.decision == "stop"


def test_still_below_the_threshold_after_the_retry_stops_the_run():
    assert dl.decide(scored=28, total=32, retryable=0, min_fraction=0.9, retried=True).decision == "stop"


def test_with_nothing_unsettled_there_is_nothing_to_retry():
    assert dl.decide(scored=20, total=32, retryable=0, min_fraction=0.9, retried=False).decision == "stop"


def test_a_retry_that_cannot_reach_the_threshold_is_skipped():
    # 20 scored + 4 retryable = 24 of 32 < 90% even if every retry scored
    d = dl.decide(scored=20, total=32, retryable=4, min_fraction=0.9, retried=False)
    assert d.decision == "stop" and "even if" in d.why


def test_inconsistent_counts_are_refused():
    with pytest.raises(ValueError):
        dl.decide(scored=30, total=32, retryable=3, min_fraction=0.9, retried=False)
