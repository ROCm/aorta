"""One unreadable log must not terminate monitoring for every job.

``read_new_bytes`` handled only ``FileNotFoundError`` from ``stat``. Permission
errors, stale NFS handles, and failures opening a file after a successful stat
escaped the per-job loop and ended ``poll_jobs``. One bad sidecar therefore
stopped the primary log, every other active job, and all later rounds.

Watch now catches ``OSError`` around each individual read, emits a visible
``WATCH_LOG_READ_FAILED`` event, leaves that path's cursor unchanged, and
continues with the remaining files and jobs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("dspy", reason="the watch loop needs the [cia] extra")

import aorta.cia.watch.poll as poll_mod
from aorta.cia.watch.cursors import read_new_bytes, save_cursors


class _Healthy:
    signal = "WATCH_CLEAN"
    healthy = True
    confidence = 0.99
    evidence = ""
    assessment = "healthy"


def _write_job(root: Path, job_id: str, watch_files: list[Path]) -> Path:
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "node": "node1",
                "recipe": "a recipe",
                "launched_at": "2026-01-01T00:00:00Z",
                "log_path": str(watch_files[0]),
                "aorta_output": str(job_dir / "aorta"),
                "status": "running",
                "schema_version": "0.1",
                "watch_files": [str(path) for path in watch_files],
            }
        ),
        encoding="utf-8",
    )
    return job_dir


def _events(job_dir: Path) -> list[dict]:
    path = job_dir / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


@pytest.fixture()
def healthy_watch(monkeypatch):
    seen: list[str] = []

    class Watcher:
        def forward(self, new_content, **_kwargs):
            seen.append(new_content)
            return _Healthy()

    monkeypatch.setattr(poll_mod, "LogWatcher", lambda *a, **k: Watcher())
    monkeypatch.setattr(poll_mod, "LogFinder", lambda *a, **k: object())
    monkeypatch.setattr(poll_mod.time, "sleep", lambda _seconds: None)
    return seen


class TestTheFailureIsContainedPerFile:
    @pytest.mark.parametrize(
        "failure",
        [
            PermissionError("permission denied"),
            OSError("stale NFS file handle"),
        ],
        ids=["permission", "nfs"],
    )
    def test_an_unreadable_first_job_does_not_stop_the_next(
        self, tmp_path, healthy_watch, monkeypatch, failure
    ):
        bad = tmp_path / "bad.log"
        good = tmp_path / "good.log"
        bad.write_text("unreadable evidence\n", encoding="utf-8")
        good.write_text("step=1 loss=1.0\n", encoding="utf-8")
        bad_job = _write_job(tmp_path, "cia-aaa", [bad])
        _write_job(tmp_path, "cia-bbb", [good])
        real_read = poll_mod.read_new_bytes

        def read(path, cursor):
            if path == bad:
                raise failure
            return real_read(path, cursor)

        monkeypatch.setattr(poll_mod, "read_new_bytes", read)

        poll_mod.poll_jobs(tmp_path, max_rounds=1)

        assert any("loss=1.0" in content for content in healthy_watch)
        assert _events(bad_job)[0]["signal"] == "WATCH_LOG_READ_FAILED"

    def test_a_bad_sidecar_does_not_hide_a_good_file_on_the_same_job(
        self, tmp_path, healthy_watch, monkeypatch
    ):
        bad = tmp_path / "sidecar.log"
        good = tmp_path / "primary.log"
        bad.write_text("sidecar\n", encoding="utf-8")
        good.write_text("step=2 loss=0.5\n", encoding="utf-8")
        _write_job(tmp_path, "cia-aaa", [bad, good])
        real_read = poll_mod.read_new_bytes

        def read(path, cursor):
            if path == bad:
                raise PermissionError("sidecar is unreadable")
            return real_read(path, cursor)

        monkeypatch.setattr(poll_mod, "read_new_bytes", read)

        poll_mod.poll_jobs(tmp_path, max_rounds=1)

        assert len(healthy_watch) == 1
        assert "primary.log" in healthy_watch[0]
        assert "loss=0.5" in healthy_watch[0]


class TestTheFailedCursorIsRetried:
    def test_an_existing_cursor_is_not_advanced(
        self, tmp_path, healthy_watch, monkeypatch
    ):
        bad = tmp_path / "bad.log"
        bad.write_text("already read|new bytes\n", encoding="utf-8")
        job_dir = _write_job(tmp_path, "cia-aaa", [bad])
        save_cursors(job_dir, {str(bad): 12})

        monkeypatch.setattr(
            poll_mod,
            "read_new_bytes",
            lambda path, cursor: (_ for _ in ()).throw(
                OSError("temporary read failure")
            ),
        )

        poll_mod.poll_jobs(tmp_path, max_rounds=1)

        saved = json.loads(
            (job_dir / "watch_cursors.json").read_text(encoding="utf-8")
        )
        assert saved[str(bad)] == 12

    def test_a_new_cursor_is_not_created_for_the_failed_path(
        self, tmp_path, healthy_watch, monkeypatch
    ):
        bad = tmp_path / "bad.log"
        bad.write_text("new bytes\n", encoding="utf-8")
        job_dir = _write_job(tmp_path, "cia-aaa", [bad])
        monkeypatch.setattr(
            poll_mod,
            "read_new_bytes",
            lambda path, cursor: (_ for _ in ()).throw(
                PermissionError("not readable")
            ),
        )

        poll_mod.poll_jobs(tmp_path, max_rounds=1)

        saved = json.loads(
            (job_dir / "watch_cursors.json").read_text(encoding="utf-8")
        )
        assert str(bad) not in saved


class TestTheErrorIsVisible:
    def test_the_event_names_the_path_and_exception(
        self, tmp_path, healthy_watch, monkeypatch
    ):
        bad = tmp_path / "bad.log"
        bad.write_text("evidence\n", encoding="utf-8")
        job_dir = _write_job(tmp_path, "cia-aaa", [bad])
        monkeypatch.setattr(
            poll_mod,
            "read_new_bytes",
            lambda path, cursor: (_ for _ in ()).throw(
                PermissionError("access denied")
            ),
        )

        poll_mod.poll_jobs(tmp_path, max_rounds=1)

        event = _events(job_dir)[0]
        assert event["event_type"] == "watchdog_error"
        assert event["source"] == str(bad)
        assert "PermissionError" in event["assessment"]
        assert "access denied" in event["assessment"]
        assert "retry" in event["assessment"].lower()

    def test_the_operator_sees_it_too(
        self, tmp_path, healthy_watch, monkeypatch, capsys
    ):
        bad = tmp_path / "bad.log"
        bad.write_text("evidence\n", encoding="utf-8")
        _write_job(tmp_path, "cia-aaa", [bad])
        monkeypatch.setattr(
            poll_mod,
            "read_new_bytes",
            lambda path, cursor: (_ for _ in ()).throw(
                OSError("NFS went away")
            ),
        )

        poll_mod.poll_jobs(tmp_path, max_rounds=1)

        output = capsys.readouterr().out
        assert "cia-aaa" in output
        assert "NFS went away" in output
        assert str(bad) in output


class TestCursorReaderStillDistinguishesMissingFromUnreadable:
    def test_a_missing_file_is_still_a_quiet_not_yet(self, tmp_path):
        missing = tmp_path / "not-created.log"

        assert read_new_bytes(missing, 7) == ("", 7)

    def test_a_stat_permission_error_propagates_to_the_per_file_boundary(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "log"
        path.write_text("data", encoding="utf-8")
        real_stat = Path.stat

        def stat(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError("stat denied")
            return real_stat(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", stat)

        with pytest.raises(PermissionError, match="stat denied"):
            read_new_bytes(path, 0)

    def test_an_open_error_propagates_to_the_per_file_boundary(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "log"
        path.write_text("data", encoding="utf-8")
        real_open = Path.open

        def open_file(candidate, *args, **kwargs):
            if candidate == path:
                raise OSError("open failed")
            return real_open(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_file)

        with pytest.raises(OSError, match="open failed"):
            read_new_bytes(path, 0)
