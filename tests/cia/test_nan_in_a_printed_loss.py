"""The commonest way a training loop says its loss went non-finite.

``print(f"step {i} loss {loss}")`` writes ``step 2 loss nan``. There is no
operator in it -- no ``=``, no ``is`` -- and the pattern required one, so the
line that most training scripts actually emit did not match.

Found by running the assistant against a real workload. Watch read that log and
alerted at 0.98 confidence with an accurate assessment; Autopsy read the same
log, reported WATCH_CLEAN, and classified the run ``unknown`` at 0.0. The two
halves of the pipeline disagreed about a file they had both just read, and the
half that reaches the user was the one that was wrong.

The separator can be a bare space now. That is still adjacency: "checking loss
for nan" has a word in between and does not match, which is the false positive
the operator requirement was added to stop.
"""

from __future__ import annotations

import pytest

from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text


class TestAPrintedLossIsRead:
    @pytest.mark.parametrize(
        "line",
        [
            "step 2 loss nan",
            "step 0 loss inf",
            "epoch 3 train_loss nan",
            "iter 40 grad_norm inf",
            "step 12 loss -inf",
            "loss nan",
        ],
    )
    def test_the_space_separated_form_alerts(self, line):
        assert scan_stderr_text(line).alert is True

    @pytest.mark.parametrize(
        "line", ["loss = nan", "loss: nan", "loss is nan", "loss became inf"]
    )
    def test_the_forms_that_already_worked_still_do(self, line):
        assert scan_stderr_text(line).alert is True

    def test_the_log_from_the_run_that_found_this(self):
        log = "\n".join(f"step {i} loss {'inf' if i < 2 else 'nan'}" for i in range(6))

        assert scan_stderr_text(log).alert is True


class TestTheFalsePositivesStayClosed:
    """What the operator requirement was protecting against."""

    @pytest.mark.parametrize(
        "line",
        [
            "checking loss for nan",
            "no nan detected",
            "nan detection enabled",
            "scanning gradients for nan values",
            "loss 0.53",
            "loss nan-free",
            "loss nancy",
            "banana",
        ],
    )
    def test_it_does_not_alert(self, line):
        assert scan_stderr_text(line).alert is False

    def test_a_hyphenated_word_is_not_the_value(self):
        """``\\b`` matched before the hyphen, so "nan-free" read as "nan"."""
        assert not scan_stderr_text("loss nan-free after the fix").alert

    def test_a_clean_run_saying_so_is_still_clean(self):
        assert not scan_stderr_text("no non-finite values found in 400 steps").alert


class TestTheSignalReachesAutopsy:
    def test_an_alerting_log_is_signalled_as_nan(self):
        from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text as scan

        assert scan("step 2 loss nan").alert

    def test_a_clean_log_is_not(self):
        assert not scan_stderr_text("step 2 loss 0.41").alert
