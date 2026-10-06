"""What Watch discovered once, it does not discover again.

``job.watch_files`` was set on the in-memory record and never written back,
and every round loads a fresh record from disk. So a job whose log path had to
be discovered re-ran that discovery for the whole life of the run -- including
the model call inside it -- and each pass was free to land somewhere different
from the last.

The write is atomic because Watch re-reads ``job.json`` every round, possibly
from another process. A torn record does not read as a damaged job; it reads
as no job at all, because the parse fails, the job drops out of the active
scan, and monitoring stops for a run that is still going.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the watch loop needs the [cia] extra")

from aorta.cia.launch.job import (
    JobRecord,
    read_job_json,
    record_watch_files,
    update_job_status,
    write_job_json,
)


@pytest.fixture()
def jobs_root(tmp_path):
    record = JobRecord(
        job_id="cia-1",
        node="node1",
        recipe="r.yaml",
        launched_at="2026-01-01T00:00:00Z",
        log_path="",
        aorta_output=str(tmp_path / "cia-1" / "aorta"),
        status="running",
    )
    write_job_json(record, tmp_path)
    return tmp_path


class TestItIsWrittenBack:
    def test_the_files_land_on_disk(self, jobs_root):
        record_watch_files(jobs_root, "cia-1", ["/jobs/cia-1/train.log"])

        assert read_job_json(jobs_root / "cia-1" / "job.json").watch_files == [
            "/jobs/cia-1/train.log"
        ]

    def test_a_reloaded_record_already_knows_them(self, jobs_root):
        """Which is the point: the next round must not rediscover."""
        record_watch_files(jobs_root, "cia-1", ["/jobs/cia-1/train.log"])
        reloaded = read_job_json(jobs_root / "cia-1" / "job.json")

        assert reloaded.watch_files, "the next round would discover all over again"

    def test_several_files_survive(self, jobs_root):
        record_watch_files(jobs_root, "cia-1", ["/a.log", "/b.log"])

        assert read_job_json(jobs_root / "cia-1" / "job.json").watch_files == [
            "/a.log",
            "/b.log",
        ]

    def test_an_empty_result_is_not_recorded(self, jobs_root):
        """Nothing found is not a resolution, and must not stop the next try."""
        record_watch_files(jobs_root, "cia-1", [])

        assert read_job_json(jobs_root / "cia-1" / "job.json").watch_files == []

    def test_a_missing_record_is_not_an_error(self, jobs_root):
        record_watch_files(jobs_root, "cia-absent", ["/a.log"])  # must not raise


class TestTheRestOfTheRecordIsLeftAlone:
    def test_other_fields_survive(self, jobs_root):
        record_watch_files(jobs_root, "cia-1", ["/a.log"])
        back = read_job_json(jobs_root / "cia-1" / "job.json")

        assert back.status == "running"
        assert back.recipe == "r.yaml"
        assert back.node == "node1"

    def test_a_status_change_by_someone_else_is_not_undone(self, jobs_root):
        """Read-modify-write, not a dump of what this process last saw."""
        update_job_status(jobs_root, "cia-1", "completed")
        record_watch_files(jobs_root, "cia-1", ["/a.log"])

        assert read_job_json(jobs_root / "cia-1" / "job.json").status == "completed"


class TestTheWriteIsAtomic:
    def test_no_temporary_file_is_left_behind(self, jobs_root):
        record_watch_files(jobs_root, "cia-1", ["/a.log"])

        assert [p.name for p in (jobs_root / "cia-1").iterdir()] == ["job.json"]

    def test_the_record_is_always_parseable(self, jobs_root):
        """A torn write reads as no job rather than a damaged one."""
        for n in range(20):
            record_watch_files(jobs_root, "cia-1", [f"/log-{n}.log"])
            json.loads((jobs_root / "cia-1" / "job.json").read_text(encoding="utf-8"))

    def test_status_updates_are_atomic_too(self, jobs_root):
        """Same file, same risk: it was a bare write_text as well."""
        from pathlib import Path as _Path

        import aorta.cia.launch.job as job_mod

        source = _Path(job_mod.__file__).read_text(encoding="utf-8")
        body = source[source.index("def update_job_status") :]

        assert "_update_job_field" in body[:200]


class TestTheLoopPersistsWhatItFinds:
    def test_poll_writes_them_back(self):
        import aorta.cia.watch.poll as poll_mod

        source = Path(poll_mod.__file__).read_text(encoding="utf-8")

        assert "record_watch_files(jobs_root, job.job_id, job.watch_files)" in source

    def test_it_happens_where_they_are_resolved(self):
        """Not at the end of the round, where a `continue` would skip it."""
        import aorta.cia.watch.poll as poll_mod

        source = Path(poll_mod.__file__).read_text(encoding="utf-8")
        resolved = source.index("watching {[Path(p).name for p in job.watch_files]}")

        assert "record_watch_files" in source[resolved : resolved + 500]


class TestDiscoveryStopsRepeating:
    """Driven through poll_jobs, because that is where the cost was.

    A source check passes as soon as the call exists somewhere; it says
    nothing about whether a second round still pays for discovery.
    """

    @staticmethod
    def _job_without_a_log_path(root: Path, job_id: str = "cia-disc") -> Path:
        job_dir = root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        found = job_dir / "found.log"
        found.write_text("step=1 loss=0.5\n" * 20, encoding="utf-8")
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "node": "node1",
                    "recipe": "r.yaml",
                    "launched_at": "2026-01-01T00:00:00Z",
                    # Empty: this is the path that falls through to discovery.
                    "log_path": "",
                    "aorta_output": str(job_dir / "aorta"),
                    "status": "running",
                    "schema_version": "0.1",
                    "watch_files": [],
                }
            ),
            encoding="utf-8",
        )
        return job_dir

    def test_a_second_round_does_not_discover_again(self, tmp_path, monkeypatch):
        import aorta.cia.watch.poll as poll_mod

        job_dir = self._job_without_a_log_path(tmp_path)
        discoveries: list[str] = []

        class CountingFinder:
            def find(self, jd, **_kwargs):
                discoveries.append(str(jd))
                return [job_dir / "found.log"]

        class Pred:
            signal = "WATCH_CLEAN"
            healthy = True
            confidence = 0.1
            evidence = ""
            assessment = "fine"

        monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: CountingFinder())
        monkeypatch.setattr(
            poll_mod, "LogWatcher", lambda *a, **k: type("W", (), {"forward": lambda s, **k: Pred()})()
        )
        monkeypatch.setattr(poll_mod.time, "sleep", lambda _: None)

        poll_jobs = poll_mod.poll_jobs
        poll_jobs(tmp_path, max_rounds=1)
        poll_jobs(tmp_path, max_rounds=1)  # a fresh process would start here

        assert len(discoveries) == 1, (
            f"discovery ran {len(discoveries)} times; it should be remembered"
        )

    def test_the_discovered_path_is_the_one_used_afterwards(self, tmp_path, monkeypatch):
        import aorta.cia.watch.poll as poll_mod

        job_dir = self._job_without_a_log_path(tmp_path)

        class OnceFinder:
            def find(self, jd, **_kwargs):
                return [job_dir / "found.log"]

        class Pred:
            signal = "WATCH_CLEAN"
            healthy = True
            confidence = 0.1
            evidence = ""
            assessment = "fine"

        monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: OnceFinder())
        monkeypatch.setattr(
            poll_mod, "LogWatcher", lambda *a, **k: type("W", (), {"forward": lambda s, **k: Pred()})()
        )
        monkeypatch.setattr(poll_mod.time, "sleep", lambda _: None)

        poll_mod.poll_jobs(tmp_path, max_rounds=1)

        assert read_job_json(job_dir / "job.json").watch_files == [
            str(job_dir / "found.log")
        ]
