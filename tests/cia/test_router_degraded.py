"""Autopsy must say when it classified without its router.

This is the failure the port is most likely to reproduce, because nothing about
it looks wrong. The adapters are confident on their own, so a bundle still
returns a plausible category at a plausible confidence; the only thing missing
is the reasoning step that was supposed to weigh the evidence. It has already
happened once: after the agents moved from subprocesses to function calls,
nothing set the router's endpoint, every call failed, and correct-looking
verdicts kept arriving for days.

A verdict reached without the router is a weaker claim than one reached with
it, and the report has to carry that difference.
"""

from __future__ import annotations

import logging

import pytest

from aorta.cia.autopsy import orchestrator


@pytest.fixture
def evidenced_bundle(make_bundle):
    """A bundle with something in it: the router is only consulted when there
    is evidence to weigh, so an empty bundle never reaches the degraded path."""
    return make_bundle(stderr="step 5 loss=nan\nnon-finite loss, aborting run\n")


@pytest.fixture
def unreachable_router(monkeypatch):
    """Make constructing the router fail the way an unreachable model does."""

    class _Unreachable:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(
        "aorta.cia.autopsy.router.TriageRouter", _Unreachable, raising=False
    )
    return _Unreachable


def _run(bundle, caplog):
    with caplog.at_level(logging.WARNING, logger=orchestrator.log.name):
        return orchestrator.run_autopsy(bundle, kb_version="test"), caplog


def test_a_bundle_still_classifies_without_the_router(evidenced_bundle, unreachable_router, caplog):
    """Degrading is correct. Failing the run because the model is down is not.

    The adapters are not a fallback of last resort -- they reach a real verdict
    on their own, which is exactly why the degradation needs announcing.
    """
    from aorta.agent.llm import AUTOPSY_CATEGORIES

    report, _ = _run(evidenced_bundle, caplog)
    assert report["category"] in AUTOPSY_CATEGORIES
    assert report["category"] != "unknown", (
        "a NaN in the log should still classify without the router"
    )


def test_the_degradation_is_logged(evidenced_bundle, unreachable_router, caplog):
    _, caplog = _run(evidenced_bundle, caplog)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("router unavailable" in w.lower() for w in warnings), (
        "nothing warned that the router was unavailable; the only evidence "
        f"would be someone watching stderr. Saw: {warnings}"
    )


def test_the_degradation_reaches_the_report(evidenced_bundle, unreachable_router, caplog):
    """A log line is gone by the time anyone reads the bundle.

    The report is the durable artifact, so the gap has to be in it -- and a
    router that could not run is precisely a tooling gap, which the schema
    already models.
    """
    report, _ = _run(evidenced_bundle, caplog)
    gaps = report.get("tooling_gaps") or []
    assert any(g.get("missing_signal") == "LLM_ROUTER_REVIEW" for g in gaps), (
        "the report does not record that its verdict is adapter-only; a reader "
        f"cannot tell it was degraded. Gaps: {gaps}"
    )


def test_a_healthy_router_records_no_gap(evidenced_bundle, monkeypatch, caplog):
    """The counterpart: the marker must mean something when it is absent."""

    class _Prediction:
        category = "gpu_race"
        confidence = 0.95
        rationale = "SAN_CONSAN_RACE in aorta/sanitizer_report.json"
        next_probe = "none"
        next_probe_reason = ""

    class _Router:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, **kwargs):
            return _Prediction()

    monkeypatch.setattr("aorta.cia.autopsy.router.TriageRouter", _Router, raising=False)
    report, _ = _run(evidenced_bundle, caplog)
    gaps = report.get("tooling_gaps") or []
    assert not any(g.get("missing_signal") == "LLM_ROUTER_REVIEW" for g in gaps)
