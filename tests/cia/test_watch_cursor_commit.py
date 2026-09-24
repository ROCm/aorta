"""A chunk Watch could not assess is retried, not forgotten.

The cursor was saved immediately after reading, before ``watcher.forward()``.
A transient model failure — a rate limit, a dropped connection — therefore
advanced past bytes that were never examined, and the next poll read from after
them. A NaN, a fault or an OOM is usually printed once, so one failed call was
enough to lose the only evidence there was.

Cursors are now committed after the assessment and its event are on disk. That
alone would let a permanently failing model stop a job ever progressing, so a
chunk is retried a bounded number of times and then given up on visibly, in the
events file the verdict would have gone to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import aorta.cia.watch.poll as poll_mod
from aorta.cia.watch.poll import MAX_ASSESS_ATTEMPTS


class _Pred:
    signal = "WATCH_NUMERIC_NAN"
    healthy = False
    confidence = 0.99
    evidence = "loss=nan"
    assessment = "non-finite loss"


@pytest.fixture
def job_root(tmp_path, monkeypatch):
    """One job with one line of log, and the expensive parts stubbed out."""
    root = tmp_path / "jobs"
    job_dir = root / "cia-1"
    job_dir.mkdir(parents=True)
    log = job_dir / "watch.log"
    log.write_text("[train] step=5 loss=nan\n")
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "cia-1",
                "node": "n",
                "recipe": "r",
                "launched_at": "2026-01-01T00:00:00Z",
                "log_path": str(log),
                "aorta_output": str(job_dir / "aorta"),
                "status": "running",
                "schema_version": "0.1",
                "watch_files": [str(log)],
            }
        )
    )
    monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: object())
    monkeypatch.setattr(poll_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "aorta.cia.watch.bundle_writer.write_bundle",
        lambda job, jd, ev, sig: jd / "bundle",
    )
    monkeypatch.setattr(
        "aorta.cia.watch.trigger.trigger_autopsy", lambda b, j, r, stop=None: None
    )
    return root, job_dir


def _events(job_dir: Path) -> list[dict]:
    path = job_dir / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _install(monkeypatch, watcher) -> None:
    monkeypatch.setattr(poll_mod, "LogWatcher", lambda *a, **k: watcher)


class TestATransientFailureLosesNothing:
    def test_the_same_bytes_are_assessed_again(self, job_root, monkeypatch):
        root, job_dir = job_root
        seen: list[str] = []

        class Flaky:
            calls = 0

            def forward(self, new_content, **kwargs):
                Flaky.calls += 1
                seen.append(new_content)
                if Flaky.calls == 1:
                    raise RuntimeError("rate limited")
                return _Pred()

        _install(monkeypatch, Flaky())
        poll_mod.poll_jobs(root, max_rounds=2)

        assert len(seen) == 2
        assert seen[0] == seen[1], "the retry must see the chunk that failed"
        assert "loss=nan" in seen[0]

    def test_the_failure_is_eventually_assessed(self, job_root, monkeypatch):
        """The point: the NaN still produces its verdict."""
        root, job_dir = job_root

        class Flaky:
            calls = 0

            def forward(self, new_content, **kwargs):
                Flaky.calls += 1
                if Flaky.calls <= 2:
                    raise RuntimeError("rate limited")
                return _Pred()

        _install(monkeypatch, Flaky())
        poll_mod.poll_jobs(root, max_rounds=3)

        assert [e["signal"] for e in _events(job_dir)] == ["WATCH_NUMERIC_NAN"]

    def test_no_cursor_file_is_written_for_an_unassessed_chunk(self, job_root, monkeypatch):
        root, job_dir = job_root

        class Broken:
            def forward(self, new_content, **kwargs):
                raise RuntimeError("down")

        _install(monkeypatch, Broken())
        poll_mod.poll_jobs(root, max_rounds=1)

        cursors = json.loads((job_dir / "watch_cursors.json").read_text()) if (
            job_dir / "watch_cursors.json"
        ).is_file() else {}
        assert all(v == 0 for v in cursors.values()), cursors


class TestAPermanentFailureIsGivenUpOnVisibly:
    @pytest.fixture
    def broken(self, job_root, monkeypatch):
        root, job_dir = job_root
        seen: list[str] = []

        class Broken:
            def forward(self, new_content, **kwargs):
                seen.append(new_content)
                raise RuntimeError("model is down")

        _install(monkeypatch, Broken())
        return root, job_dir, seen

    def test_it_stops_after_the_attempt_budget(self, broken):
        root, _, seen = broken
        poll_mod.poll_jobs(root, max_rounds=MAX_ASSESS_ATTEMPTS + 3)
        assert len(seen) == MAX_ASSESS_ATTEMPTS

    def test_the_chunk_is_recorded_rather_than_dropped(self, broken):
        root, job_dir, _ = broken
        poll_mod.poll_jobs(root, max_rounds=MAX_ASSESS_ATTEMPTS + 1)

        events = _events(job_dir)
        assert [e["signal"] for e in events] == ["WATCH_ASSESSMENT_FAILED"]
        assert "loss=nan" in events[0]["excerpt"], "the bytes are kept for a reader"

    def test_it_says_why(self, broken):
        root, job_dir, _ = broken
        poll_mod.poll_jobs(root, max_rounds=MAX_ASSESS_ATTEMPTS + 1)
        assert "model is down" in _events(job_dir)[0]["assessment"]

    def test_the_job_is_not_stuck_on_those_bytes(self, broken):
        """Retrying for ever would stop this job ever being watched again."""
        root, job_dir, seen = broken
        poll_mod.poll_jobs(root, max_rounds=MAX_ASSESS_ATTEMPTS + 3)

        cursors = json.loads((job_dir / "watch_cursors.json").read_text())
        assert any(v > 0 for v in cursors.values()), "the cursor should have moved on"


class TestTheOrdinaryPathIsUnchanged:
    def test_a_successful_assessment_commits_its_cursor(self, job_root, monkeypatch):
        root, job_dir = job_root
        _install(monkeypatch, type("Ok", (), {"forward": lambda self, **k: _Pred()})())

        poll_mod.poll_jobs(root, max_rounds=1)

        cursors = json.loads((job_dir / "watch_cursors.json").read_text())
        assert any(v > 0 for v in cursors.values())

    def test_a_quiet_job_still_advances_past_nothing(self, job_root, monkeypatch):
        """An empty read commits, so a rotated file is not re-read for ever."""
        root, job_dir = job_root
        (job_dir / "watch.log").write_text("")
        _install(monkeypatch, type("Ok", (), {"forward": lambda self, **k: _Pred()})())

        poll_mod.poll_jobs(root, max_rounds=1)
        assert (job_dir / "watch_cursors.json").is_file()
