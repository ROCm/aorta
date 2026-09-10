"""Watch's autopsy can reach the escalation it recommends.

``run_autopsy`` guards its production sweep on ``job is not None``, and
``trigger_autopsy`` — the only thing that produces a Watch-triggered report —
called it without the job it was holding. So every one of those autopsies could
put ``aorta sweep run`` in ``next_probe`` and none of them could ever run one.
The recommendation reached the report; the code that acts on it was unreachable
from the only caller that generates them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.cia.launch.job import JobRecord
from aorta.cia.watch import trigger as trigger_mod


def _job(**overrides) -> JobRecord:
    fields = {
        "job_id": "cia-test",
        "node": "node-01",
        "recipe": "a label, not a path",
        "launched_at": "2026-01-01T00:00:00Z",
        "log_path": "/tmp/cia-test/watch.log",
        "aorta_output": "/tmp/cia-test/aorta",
        "head_node": "head-01",
    }
    return JobRecord(**{**fields, **overrides})


@pytest.fixture
def called(monkeypatch, tmp_path):
    """Capture what trigger_autopsy hands to run_autopsy."""
    seen: dict = {}

    def fake_run_autopsy(bundle_root, **kwargs):
        seen.update(kwargs)
        seen["bundle_root"] = bundle_root
        return {"category": "gpu_race", "confidence": 0.55}

    monkeypatch.setattr(
        "aorta.cia.autopsy.orchestrator.run_autopsy", fake_run_autopsy
    )
    monkeypatch.setattr(trigger_mod, "update_job_status", lambda *a, **k: None)
    return seen


class TestTheJobGoesThrough:
    def test_run_autopsy_receives_it(self, called, tmp_path):
        job = _job()
        trigger_mod.trigger_autopsy(tmp_path, job, tmp_path)
        assert called["job"] is job

    def test_and_the_head_node_with_it(self, called, tmp_path):
        """run_aorta_probe needs somewhere to send the query."""
        trigger_mod.trigger_autopsy(tmp_path, _job(head_node="jump-99"), tmp_path)
        assert called["head_node"] == "jump-99"

    def test_the_escalation_guard_would_now_pass(self, called, tmp_path):
        """`job is not None` is what stood between the report and the sweep."""
        trigger_mod.trigger_autopsy(tmp_path, _job(), tmp_path)
        assert called["job"] is not None

    def test_the_rest_of_the_call_is_unchanged(self, called, tmp_path):
        trigger_mod.trigger_autopsy(tmp_path, _job(), tmp_path)
        assert called["kb_version"] == "kb-static-poc"
        assert called["bundle_root"] == tmp_path


class TestTheReportIsStillWritten:
    def test_report_json_lands_beside_the_bundle(self, called, tmp_path):
        trigger_mod.trigger_autopsy(tmp_path, _job(), tmp_path)
        assert (tmp_path / "report.json").is_file()

    def test_and_is_returned(self, called, tmp_path):
        report = trigger_mod.trigger_autopsy(tmp_path, _job(), tmp_path)
        assert report["category"] == "gpu_race"


def test_a_job_with_no_recipe_recorded_still_declines_safely(tmp_path):
    """Reaching the escalation is not the same as it running.

    ``resolve_recipe`` answers with nothing until the launcher records a recipe
    path, so the sweep declines rather than running somebody else's workload.
    Passing the job makes the path reachable; it does not make it reckless.
    """
    from aorta.cia.autopsy.probe import resolve_recipe

    assert resolve_recipe(tmp_path, _job()) == ("", "")
