"""Tests for the ``flat_resume`` run-dir lock helper (PR #194 follow-up).

These cover the same-host live-PID rejection, same-host stale-PID takeover,
cross-host fail-closed semantics, and the corrupt-lock recovery path.
The matching reviewer comment is on
``src/aorta/triage/output.py::resolve_run_dir`` (PRRT_kwDOP5TFPc6E08d0).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.triage.output import (
    FLAT_RESUME_LOCKFILE,
    RunDirLockedError,
    acquire_flat_resume_lock,
)

# ---- Same-host, lock unheld ---------------------------------------------


def test_acquires_when_lock_absent(tmp_path: Path) -> None:
    """No existing lock -> context entered successfully, lockfile created."""
    with acquire_flat_resume_lock(tmp_path):
        lock_path = tmp_path / FLAT_RESUME_LOCKFILE
        assert lock_path.is_file(), "lock file must exist inside the context"
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["host"] == socket.gethostname()
        assert "started_at" in payload
    # On exit the lockfile is cleaned up.
    assert not (tmp_path / FLAT_RESUME_LOCKFILE).exists()


def test_releases_on_exception(tmp_path: Path) -> None:
    """Exception inside the context still cleans up the lockfile."""
    with pytest.raises(RuntimeError, match="boom"):
        with acquire_flat_resume_lock(tmp_path):
            raise RuntimeError("boom")
    assert not (tmp_path / FLAT_RESUME_LOCKFILE).exists()


def test_reentry_after_clean_release(tmp_path: Path) -> None:
    """Two sequential acquisitions succeed: the resume happy path."""
    with acquire_flat_resume_lock(tmp_path):
        pass
    with acquire_flat_resume_lock(tmp_path):
        pass


# ---- Same-host, live holder (the bug Copilot called out) ---------------


def test_rejects_when_same_host_live_pid(tmp_path: Path) -> None:
    """A second writer hitting the same run dir must fail fast, not stomp.

    The reviewer's concern: ``mkdir(exist_ok=True)`` made flat_resume
    silently race two concurrent ``aorta probe`` invocations. The lock
    converts that into a clean ``RunDirLockedError`` with the holder's
    PID + host in the message.
    """
    with acquire_flat_resume_lock(tmp_path):
        with pytest.raises(RunDirLockedError) as exc_info:
            with acquire_flat_resume_lock(tmp_path):
                pytest.fail("inner acquisition must not enter the body")
    err = exc_info.value
    assert err.holder_host == socket.gethostname()
    assert err.holder_pid == os.getpid()
    assert "still running" in str(err)


# ---- Same-host, stale lock from a crashed prior run --------------------


def _spawn_quickly_dying_child() -> int:
    """Return the PID of a process that has already exited.

    We can't use the current PID for the "dead" check (it's alive by
    definition). Spawning + waiting on a child gives us a PID that is
    guaranteed to be reaped (``Popen.wait`` reaps it) before we use it
    in the lock file.
    """
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    proc.wait()
    return proc.pid


def test_takes_over_stale_same_host_lock(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Same host + dead holder PID -> warn + overwrite, do not raise.

    This is the documented recovery path: ``flat_resume`` exists for the
    "previous run crashed, re-run to fill the gaps" workflow. A
    fail-closed semantic here would defeat the whole resume model.
    """
    dead_pid = _spawn_quickly_dying_child()
    lock_path = tmp_path / FLAT_RESUME_LOCKFILE
    lock_path.write_text(
        json.dumps(
            {
                "pid": dead_pid,
                "host": socket.gethostname(),
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="aorta.triage.output"):
        with acquire_flat_resume_lock(tmp_path):
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            assert payload["pid"] == os.getpid(), (
                "stale lock must have been overwritten with our identity"
            )

    assert any("stale" in record.message for record in caplog.records), (
        "stale-lock takeover must emit an operator-visible warning"
    )


# ---- Cross-host (NFS / shared FS): fail closed -------------------------


def test_rejects_cross_host_lock(tmp_path: Path) -> None:
    """Different host in the lockfile -> raise.

    Cross-host PID-liveness is not verifiable in Phase 1. Failing closed
    surfaces the situation to the operator rather than silently stomping
    a peer's in-flight run on a shared filesystem.
    """
    lock_path = tmp_path / FLAT_RESUME_LOCKFILE
    lock_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "host": "other-host.example.invalid",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunDirLockedError) as exc_info:
        with acquire_flat_resume_lock(tmp_path):
            pytest.fail("cross-host lock must block entry")
    err = exc_info.value
    assert err.holder_host == "other-host.example.invalid"
    assert "different host" in str(err)


# ---- Corrupt lockfile recovery ----------------------------------------


def test_recovers_from_corrupt_lockfile(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Unparseable lockfile -> warn + take over (rather than wedge resume)."""
    lock_path = tmp_path / FLAT_RESUME_LOCKFILE
    lock_path.write_text("this is not json {", encoding="utf-8")

    with caplog.at_level("WARNING", logger="aorta.triage.output"):
        with acquire_flat_resume_lock(tmp_path):
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            assert payload["pid"] == os.getpid()

    assert any("unreadable" in record.message for record in caplog.records)


# ---- Composition smoke: helper works on a resolve_run_dir leaf ---------


def test_composes_with_resolve_run_dir(tmp_path: Path) -> None:
    """Smoke: the helper accepts a freshly resolved flat_resume run dir.

    ``resolve_run_dir(layout='flat_resume')`` is the only call-site in
    ``run_recipe`` -- this asserts the directory it produces is a valid
    target for the lock (no hidden permission / path-type surprises).
    """
    from aorta.triage.output import resolve_run_dir
    from aorta.triage.recipe import Recipe

    recipe = Recipe(
        schema_version=1,
        workload="noop",
        trials=1,
        steps=1,
        cells=(),
        ticket="PROBE-LOCK-TEST",
    )
    run_dir = resolve_run_dir(tmp_path, recipe, layout="flat_resume")
    assert run_dir.is_dir()

    with acquire_flat_resume_lock(run_dir):
        assert (run_dir / FLAT_RESUME_LOCKFILE).is_file()
    assert not (run_dir / FLAT_RESUME_LOCKFILE).exists()
