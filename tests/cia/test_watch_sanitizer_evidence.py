"""Watch should trust a sanitizer's structured state without asking an LLM."""

from __future__ import annotations

import aorta.cia.watch.watcher as watcher_mod
from aorta.cia.watch.watcher import LogWatcher, sanitizer_assessment


def test_a_ran_failure_is_an_alert() -> None:
    result = sanitizer_assessment(
        "[sanitizer] consan: verdict=fail state=ran findings=64"
    )

    assert result is not None
    assert result.healthy is False
    assert result.signal == "WATCH_UNKNOWN_ERROR"
    assert result.confidence == 1.0
    assert "state=ran" in result.evidence


def test_a_ran_pass_is_clean() -> None:
    result = sanitizer_assessment(
        "[sanitizer] waitcheck: verdict=pass state=ran findings=0"
    )

    assert result is not None
    assert result.healthy is True
    assert result.signal == "WATCH_CLEAN"


def test_not_checked_does_not_masquerade_as_machine_verified() -> None:
    assert (
        sanitizer_assessment(
            "[sanitizer] consan: verdict=not_checked "
            "state=not_checked findings=0"
        )
        is None
    )


def test_the_machine_verdict_bypasses_react(monkeypatch) -> None:
    class MustNotRun:
        def forward(self, **_kwargs):
            raise AssertionError("structured sanitizer evidence reached the LLM")

    def must_not_configure():
        raise AssertionError("structured sanitizer evidence configured an LLM")

    monkeypatch.setattr(watcher_mod, "ensure_configured", must_not_configure)
    watcher = LogWatcher()
    watcher.react = MustNotRun()

    result = watcher.forward(
        "[sanitizer] consan: verdict=fail state=ran findings=1",
        "job context",
        "no sanitizer failures",
    )

    assert result.healthy is False
