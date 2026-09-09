"""How long the job has been running is how long the job has been running.

The poll loop passed ``elapsed_sec=int(time.time())`` -- the Unix epoch, about
1.7 billion. Every poll told the model the job had been running for fifty-four
years. That is worse than omitting it: the reason the number is there at all is
to separate "still starting up" from "stalled", and a value that is enormous
and identical on every poll answers neither question while appearing to answer
both.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

import pytest

from aorta.cia.watch.poll import elapsed_seconds


def _ago(**kwargs) -> str:
    """An ISO timestamp that far in the past, in the format job.json uses."""
    moment = datetime.now(timezone.utc) - timedelta(**kwargs)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class TestItMeasuresFromLaunch:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"seconds": 0}, 0),
            ({"seconds": 90}, 90),
            ({"minutes": 5}, 300),
            ({"hours": 2}, 7200),
        ],
    )
    def test_it_returns_the_time_since_then(self, kwargs, expected):
        assert elapsed_seconds(_ago(**kwargs)) == pytest.approx(expected, abs=2)

    def test_it_is_nowhere_near_the_epoch(self):
        """The regression: 1.7e9 rather than a number of seconds."""
        assert elapsed_seconds(_ago(minutes=1)) < 1_000

    def test_it_grows_between_polls(self):
        """A constant cannot tell a stalled job from a starting one."""
        launched = _ago(seconds=10)
        first = elapsed_seconds(launched)
        time.sleep(1.1)
        assert elapsed_seconds(launched) > first


class TestOffsetsAndFormats:
    def test_it_reads_the_format_job_json_writes(self):
        assert elapsed_seconds("2026-09-09T00:52:09Z") is not None

    def test_it_reads_an_explicit_offset(self):
        moment = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert elapsed_seconds(moment.isoformat()) == pytest.approx(60, abs=2)

    def test_a_naive_timestamp_is_read_as_utc(self):
        """job.json writes UTC without saying so on some paths."""
        moment = datetime.now(timezone.utc) - timedelta(minutes=1)
        naive = moment.strftime("%Y-%m-%dT%H:%M:%S")
        assert elapsed_seconds(naive) == pytest.approx(60, abs=2)

    def test_a_clock_skewed_future_launch_is_not_negative(self):
        future = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        assert elapsed_seconds(future) == 0


class TestWhenItCannotTell:
    @pytest.mark.parametrize("value", ["", "not a time", "yesterday", "2026-13-45T99:99:99Z"])
    def test_an_unreadable_timestamp_gives_none(self, value):
        assert elapsed_seconds(value) is None

    def test_none_is_not_zero(self):
        """A job that just started and a job with no known start differ."""
        assert elapsed_seconds("") is not elapsed_seconds(_ago(seconds=0))


class TestWhatThePollLoopSends:
    """The helper is only useful if the loop actually asks it."""

    @staticmethod
    def _poll_source() -> str:
        import inspect

        import aorta.cia.watch.poll as poll_mod

        return inspect.getsource(poll_mod.poll_jobs)

    def test_the_loop_no_longer_sends_the_wall_clock(self):
        assert "elapsed_sec={int(time.time())}" not in self._poll_source()

    def test_the_loop_asks_the_helper(self):
        assert "elapsed_seconds(job.launched_at)" in self._poll_source()

    def test_an_unreadable_launch_time_omits_the_field(self):
        """None must drop the field, not render "elapsed_sec=None"."""
        elapsed = elapsed_seconds("not a time")
        rendered = f" elapsed_sec={elapsed}" if elapsed is not None else ""
        assert rendered == ""

    def test_a_readable_one_includes_it(self):
        elapsed = elapsed_seconds(_ago(minutes=1))
        rendered = f" elapsed_sec={elapsed}" if elapsed is not None else ""
        assert re.fullmatch(r" elapsed_sec=\d{1,3}", rendered), rendered
