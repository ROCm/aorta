"""One alert per job, and one job's alert does not silence the others.

The alert branch ended in ``break`` under a comment reading "one alert per job
per session". Neither half held.

``break`` leaves ``for job in active``, so the moment any job alerted, every job
after it in that round went unexamined -- and since nothing recorded that a job
had alerted, the same job alerted again next round as soon as new bytes landed.
A job failing continuously therefore re-ran autopsy on every round while
starving every job listed behind it, indefinitely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import aorta.cia.watch.poll as poll_mod
from aorta.cia.watch.poll import poll_jobs


def _write_job(root: Path, job_id: str, content: str = "step=1 loss=nan\n") -> Path:
    """A job on disk, laid out the way scan_active_jobs expects."""
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    log = job_dir / "watch.log"
    log.write_text(content)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "node": "node1",
                "recipe": "a recipe",
                "launched_at": "2026-01-01T00:00:00Z",
                "log_path": str(log),
                "aorta_output": str(job_dir / "aorta"),
                "status": "running",
                "schema_version": "0.1",
                "watch_files": [str(log)],
            }
        )
    )
    return job_dir


@pytest.fixture
def always_alerts(monkeypatch):
    """Every assessment is an alert, and autopsy is recorded rather than run."""
    triggered: list[str] = []

    class Pred:
        signal = "WATCH_NUMERIC_NAN"
        healthy = False
        confidence = 0.99
        evidence = "loss=nan"
        assessment = "non-finite loss"

    class FakeWatcher:
        def forward(self, **kwargs):
            return Pred()

    monkeypatch.setattr(poll_mod, "LogWatcher", lambda *a, **k: FakeWatcher())
    monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: object())
    monkeypatch.setattr(poll_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "aorta.cia.watch.bundle_writer.write_bundle",
        lambda job, job_dir, evidence, signal: job_dir / "bundle",
    )
    monkeypatch.setattr(
        "aorta.cia.watch.trigger.trigger_autopsy",
        lambda bundle, job, jobs_root: triggered.append(job.job_id),
    )
    return triggered


class TestOneJobDoesNotSilenceTheOthers:
    def test_every_job_is_examined_in_a_round_where_the_first_alerts(
        self, tmp_path, always_alerts
    ):
        """break abandoned the round the moment any job alerted."""
        for job_id in ("cia-aaa", "cia-bbb", "cia-ccc"):
            _write_job(tmp_path, job_id)

        poll_jobs(tmp_path, max_rounds=1)

        assert sorted(always_alerts) == ["cia-aaa", "cia-bbb", "cia-ccc"]

    def test_a_job_that_keeps_writing_does_not_starve_the_rest(
        self, tmp_path, always_alerts
    ):
        """The starvation needs the first job to keep producing new bytes.

        With ``break``, a job that goes quiet after alerting lets the next one
        through on the following round -- which is why this only bites for a
        job still writing, and why that is the case worth pinning. A training
        job that has started printing NaN keeps printing.
        """
        first = _write_job(tmp_path, "cia-aaa")
        _write_job(tmp_path, "cia-zzz")

        def keep_writing(_):
            with (first / "watch.log").open("a") as fh:
                fh.write("step=n loss=nan\n")

        import aorta.cia.watch.poll as mod

        original = mod.time.sleep
        mod.time.sleep = keep_writing
        try:
            poll_jobs(tmp_path, max_rounds=4)
        finally:
            mod.time.sleep = original

        assert "cia-zzz" in always_alerts, "the noisy job starved the quiet one"


class TestOnePerJob:
    def test_a_job_is_diagnosed_once_however_much_more_it_writes(
        self, tmp_path, always_alerts
    ):
        """New bytes used to re-trigger autopsy on the same failure."""
        job_dir = _write_job(tmp_path, "cia-aaa")

        def grow(_):
            with (job_dir / "watch.log").open("a") as fh:
                fh.write("more output after the failure\n")

        import aorta.cia.watch.poll as mod

        # Each round the log grows, which is what used to re-arm the alert.
        original = mod.time.sleep
        mod.time.sleep = grow
        try:
            poll_jobs(tmp_path, max_rounds=4)
        finally:
            mod.time.sleep = original

        assert always_alerts.count("cia-aaa") == 1

    def test_the_guarantee_is_recorded_not_merely_commented(self):
        """It used to be a comment on a break, and nothing else."""
        import inspect

        source = inspect.getsource(poll_mod.poll_jobs)
        assert "alerted" in source
        assert "alerted.add" in source

    def test_the_alert_branch_no_longer_breaks_the_loop(self):
        import inspect

        source = inspect.getsource(poll_mod.poll_jobs)
        assert "break  # one alert per job per session" not in source
