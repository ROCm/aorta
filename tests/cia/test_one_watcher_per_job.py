"""Two watchers must not diagnose the same job.

Every triage started a watcher, and every watcher polled every active job. Four
chat turns meant four watchers over all four jobs: four model calls per log
chunk instead of one, and any watcher free to alert on a job it had not
submitted.

The guard against diagnosing twice was a read and a write with a gap between
them. ``autopsy_is_settled`` looks at the state file, the caller writes "queued"
some lines later, and nothing holds the interval. Two watchers both read "not
settled", both write, and the second wins a file it was never told it was
racing for -- so the job gets two Autopsies, which is two LLM ReAct loops and
two escalations to a sweep with a four-hour limit.

Both halves are fixed here. A triage's watcher is scoped to the job it
submitted, so the ordinary case has one watcher per job. The claim is exclusive,
so the case that remains -- a standalone `aorta cia watch` running beside a
triage, which is a supported setup -- is decided by the kernel rather than by
who wrote last.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the watch loop needs the cia extra")

from aorta.cia.launch.job import JobRecord, write_job_json
from aorta.cia.watch.poll import claim_autopsy_attempt, poll_jobs


class TestTheClaimIsExclusive:
    def test_one_caller_wins(self, tmp_path):
        job = tmp_path / "job-a"

        assert claim_autopsy_attempt(job, 1) is True
        assert claim_autopsy_attempt(job, 1) is False

    def test_exactly_one_of_many_racing_for_the_same_attempt(self, tmp_path):
        """The race the review describes, run as a race."""
        job = tmp_path / "job-a"
        ready = threading.Barrier(64, timeout=30)

        def race(_: int) -> bool:
            ready.wait()
            return claim_autopsy_attempt(job, 1)

        with ThreadPoolExecutor(max_workers=64) as pool:
            won = list(pool.map(race, range(64)))

        assert sum(won) == 1

    def test_it_holds_across_processes_too(self, tmp_path):
        """Watchers are threads in separate triage processes, not one pool."""
        job = tmp_path / "job-a"
        job.mkdir(parents=True)
        code = (
            "from pathlib import Path;"
            "from aorta.cia.watch.poll import claim_autopsy_attempt;"
            f"print(claim_autopsy_attempt(Path({str(job)!r}), 7))"
        )
        results = [
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True
            ).stdout.strip()
            for _ in range(2)
        ]

        assert results.count("True") == 1
        assert results.count("False") == 1

    def test_a_retry_is_a_new_claim(self, tmp_path):
        """Attempts only advance through the paths that reason about retrying."""
        job = tmp_path / "job-a"
        claim_autopsy_attempt(job, 1)

        assert claim_autopsy_attempt(job, 2) is True
        assert claim_autopsy_attempt(job, 2) is False

    def test_different_jobs_do_not_block_each_other(self, tmp_path):
        assert claim_autopsy_attempt(tmp_path / "job-a", 1) is True
        assert claim_autopsy_attempt(tmp_path / "job-b", 1) is True

    def test_it_creates_the_directory_it_claims_in(self, tmp_path):
        """A job whose directory is not there yet still gets a decision."""
        assert claim_autopsy_attempt(tmp_path / "not-yet" / "job-a", 1) is True

    def test_nothing_needs_unlocking(self, tmp_path):
        """A watcher that dies holding it leaves a fact, not a stuck lease.

        Whether to try again is already decided by staleness and the attempt
        ceiling, and that decision produces a new attempt number.
        """
        job = tmp_path / "job-a"
        claim_autopsy_attempt(job, 1)
        markers = [p.name for p in job.iterdir() if p.name.startswith(".autopsy.claim")]

        assert markers == [".autopsy.claim.1"]


def _job(jobs_root: Path, job_id: str) -> None:
    (jobs_root / job_id).mkdir(parents=True, exist_ok=True)
    write_job_json(
        JobRecord(
            job_id=job_id,
            node="node-1",
            recipe="r",
            launched_at="2026-01-01T00:00:00Z",
            log_path=str(jobs_root / job_id / "watch.log"),
            aorta_output=str(jobs_root / job_id / "out"),
            status="running",
        ),
        jobs_root,
    )


class TestATriageWatchesOnlyItsOwnJob:
    @staticmethod
    def _seen(jobs_root: Path, monkeypatch, **kwargs) -> list[str]:
        """Job ids the loop actually considered in one round."""
        seen: list[str] = []
        import aorta.cia.watch.poll as poll_mod

        real = poll_mod.autopsy_state

        def spy(job_dir):
            seen.append(Path(job_dir).name)
            return real(job_dir)

        # The merged #423 implementation reads the state once and classifies
        # that record, rather than calling autopsy_is_settled (which would read
        # it a second time). autopsy_state is therefore the stable observation
        # point after the per-job filter.
        monkeypatch.setattr(poll_mod, "autopsy_state", spy)
        poll_jobs(jobs_root, max_rounds=1, **kwargs)
        return seen

    def test_scoped_it_looks_at_one(self, tmp_path, monkeypatch):
        for name in ("job-a", "job-b", "job-c"):
            _job(tmp_path, name)

        assert self._seen(tmp_path, monkeypatch, only="job-b") == ["job-b"]

    def test_unscoped_it_still_looks_at_all_of_them(self, tmp_path, monkeypatch):
        """The standalone watcher is supposed to take every active job."""
        for name in ("job-a", "job-b", "job-c"):
            _job(tmp_path, name)

        assert sorted(self._seen(tmp_path, monkeypatch)) == ["job-a", "job-b", "job-c"]

    def test_an_unknown_id_watches_nothing(self, tmp_path, monkeypatch):
        _job(tmp_path, "job-a")

        assert self._seen(tmp_path, monkeypatch, only="job-missing") == []


class TestTheTriageAsksForItsOwnJob:
    def test_run_triage_scopes_the_watcher_it_starts(self, tmp_path, monkeypatch):
        from aorta.cia import triage as triage_mod

        started: dict = {}

        class FakeThread:
            def __init__(self, **kwargs):
                started.update(kwargs.get("kwargs", {}))

            def start(self):
                pass

            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        monkeypatch.setattr(triage_mod.threading, "Thread", FakeThread)
        monkeypatch.setattr(triage_mod, "launch", lambda **kw: ("99999", ""))
        monkeypatch.setattr(triage_mod, "cancel", lambda job_id: (True, ""))
        monkeypatch.setattr(triage_mod, "wait_for_job", lambda *a, **k: "ABANDONED(RUNNING)")

        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")
        result = triage_mod.run_triage(
            ["--source", str(source), "--jobs-root", str(tmp_path), "--kernel-name", "bump"]
        )

        assert started.get("only") == result["job_id"]


class TestTheClaimIsWiredIntoTheAlertPath:
    """The function being correct is not the same as it being consulted."""

    @staticmethod
    def _write_job(root: Path, job_id: str) -> Path:
        import json

        job_dir = root / job_id
        job_dir.mkdir(parents=True)
        log = job_dir / "watch.log"
        log.write_text("step=1 loss=nan\n")
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

    @pytest.fixture()
    def always_alerts(self, monkeypatch):
        """Every assessment alerts, and the Autopsy is recorded rather than run."""
        import aorta.cia.watch.poll as poll_mod

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
            lambda bundle, job, jobs_root, stop=None: triggered.append(job.job_id),
        )
        return triggered

    def test_an_alert_triggers_an_autopsy_normally(self, tmp_path, always_alerts):
        """The control: this is the path the claim sits in."""
        self._write_job(tmp_path, "cia-aaa")

        poll_jobs(tmp_path, max_rounds=1)

        assert always_alerts == ["cia-aaa"]

    def test_a_claim_another_watcher_holds_stops_this_one(self, tmp_path, always_alerts):
        """Exactly the race: the other watcher claimed between our read and write."""
        job_dir = self._write_job(tmp_path, "cia-aaa")
        claim_autopsy_attempt(job_dir, 1)

        poll_jobs(tmp_path, max_rounds=1)

        assert always_alerts == []
