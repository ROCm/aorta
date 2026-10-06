"""A NaN signature is a value, not a word.

The patterns were ``loss.*nan``, ``non-?finite`` and ``numeric_silent``, matched
anywhere in a line. So "checking loss for nan" read as a diverged run, "no
non-finite values found" read as a NaN, and a log echoing an autopsy verdict
read as a fresh failure -- ``numeric_silent`` matched the category's own name.

Each false hit costs a Watch alert and an autopsy run. The last one is a loop:
Autopsy writes ``numeric_silent`` into a report, the report is echoed into a
log, Watch reads it and calls Autopsy.
"""

from __future__ import annotations

import pytest

from aorta.cia.autopsy.adapters.stderr_watch import scan_stderr_text


def _alerts(line: str) -> bool:
    return scan_stderr_text(line).alert


class TestLinesThatSayItHappened:
    @pytest.mark.parametrize(
        "line",
        [
            "[train] step=5 loss=nan",
            "[train] step=5 loss=nan -- non-finite loss, aborting run",
            "[train] non-finite loss, aborting run",
            "loss: nan",
            "loss=-nan",
            "grad_norm=inf",
            "train_loss = NaN",
            "residual_norm=nan",
            "loss became nan at step 12",
            "loss diverged to inf",
            "NaN detected in layer 3",
            "detected nan in gradients",
            "non-finite gradients in backward pass",
        ],
    )
    def test_it_alerts(self, line):
        assert _alerts(line), f"missed a real NaN: {line}"


class TestLinesThatSayItDidNot:
    @pytest.mark.parametrize(
        "line",
        [
            "checking loss for nan",
            "check for nan every step",
            "no non-finite values found",
            "no nan in this run",
            "loss is finite, continuing",
            "residual norm ok, no nan",
            "NaN detection enabled",
            "nan checking disabled",
            "without nan guards",
        ],
    )
    def test_it_stays_quiet(self, line):
        assert not _alerts(line), f"false alert on: {line}"


class TestTheFeedbackLoop:
    """Autopsy's own output must not read as a fresh failure."""

    @pytest.mark.parametrize(
        "line",
        [
            "autopsy verdict: numeric_silent confidence=0.62",
            "[autopsy] category=numeric_silent confidence=0.62",
            "applying mitigation numeric_silent_guard",
            'rationale: "classified as numeric_silent from WATCH_NUMERIC_NAN"',
            "next_probe recommended for numeric_silent",
        ],
    )
    def test_an_echoed_verdict_is_not_a_new_nan(self, line):
        assert not _alerts(line)

    def test_the_category_name_is_no_longer_a_pattern(self):
        from aorta.cia.autopsy.adapters import stderr_watch

        joined = " ".join(p.pattern for p in stderr_watch.NAN_PATTERNS)
        assert "numeric_silent" not in joined


class TestWholeLogs:
    def test_the_nan_demo_still_alerts(self):
        """The log the demo actually produces."""
        log = (
            "[cia] node=x slurm_job=1\n"
            "[train] step=0 loss=6.43\n"
            "[train] step=4 loss=6.35\n"
            "[train] step=5 loss=nan\n"
            "[train] step=5 loss=nan -- non-finite loss, aborting run\n"
            "[cia] workload exit=1\n"
        )
        scan = scan_stderr_text(log)
        assert scan.alert
        assert scan.signal == "WATCH_NUMERIC_NAN"
        assert scan.hits[0][0] == 4

    def test_a_healthy_run_that_talks_about_nan_stays_clean(self):
        log = (
            "[train] step=0 loss=6.43\n"
            "NaN detection enabled\n"
            "no non-finite values found in this run\n"
            "[cia] workload exit=0\n"
        )
        scan = scan_stderr_text(log)
        assert not scan.alert
        assert scan.signal == "WATCH_CLEAN"


def test_the_next_probe_no_longer_names_a_canned_recipe():
    """It recommended the same demo recipe the escalation path stopped running."""
    import inspect

    from aorta.cia.autopsy.adapters import stderr_watch

    assert "Residual-NaN-Repro" not in inspect.getsource(stderr_watch)
