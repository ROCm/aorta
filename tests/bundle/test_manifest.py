"""Manifest schema / round-trip tests (issue #196)."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from aorta.bundle.manifest import (
    MANIFEST_SCHEMA_VERSION,
    FileRecord,
    Manifest,
)


def _sample_files() -> list[FileRecord]:
    return [
        FileRecord(path="none-none/trial_0/stdout.log", bytes_in=6, bytes_out=6),
        FileRecord(path="none-none/trial_0/stderr.log", bytes_in=0, bytes_out=0),
    ]


def test_manifest_from_files_records_documented_shape(tmp_path):
    """Acceptance criterion 4: manifest in the documented shape."""
    pinned = _dt.datetime(2026, 5, 25, 10, 0, 0, tzinfo=_dt.timezone.utc)
    m = Manifest.from_files(
        ticket="TKT-1",
        source_run_dir=tmp_path,
        redactor_kind="identity",
        aorta_version="0.2.0",
        files=_sample_files(),
        now=pinned,
    )
    doc = json.loads(m.to_json())
    assert doc["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert doc["ticket"] == "TKT-1"
    assert doc["created_at"] == "2026-05-25T10:00:00Z"
    assert doc["aorta_version"] == "0.2.0"
    assert doc["redactor_kind"] == "identity"
    assert doc["redaction_applied"] is False
    assert isinstance(doc["files"], list) and len(doc["files"]) == 2
    first = doc["files"][0]
    for key in (
        "path",
        "env_keys_removed",
        "paths_rewritten",
        "ips_rewritten",
        "bytes_in",
        "bytes_out",
    ):
        assert key in first, f"missing {key} in file record"


def test_manifest_redaction_applied_rollup_true_when_any_count_nonzero(tmp_path):
    files = [
        FileRecord(path="a", bytes_in=1, bytes_out=1),
        FileRecord(path="b", env_keys_removed=2, bytes_in=10, bytes_out=8),
    ]
    m = Manifest.from_files(
        ticket="T",
        source_run_dir=tmp_path,
        redactor_kind="probe.v1",
        aorta_version="x",
        files=files,
    )
    assert m.redaction_applied is True


def test_manifest_round_trip_via_json(tmp_path):
    m = Manifest.from_files(
        ticket="TKT-2",
        source_run_dir=tmp_path,
        redactor_kind="identity",
        aorta_version="0.2.0",
        files=_sample_files(),
        now=_dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
    )
    reparsed = Manifest.from_json(m.to_json())
    assert reparsed == m


def test_manifest_rejects_unknown_schema_version():
    raw = json.dumps(
        {
            "schema_version": 99,
            "ticket": "X",
            "created_at": "2026",
            "aorta_version": "x",
            "source_run_dir": "/",
            "redaction_applied": False,
            "redactor_kind": "identity",
            "files": [],
        }
    )
    with pytest.raises(ValueError, match="unsupported schema_version"):
        Manifest.from_json(raw)


def test_manifest_from_files_normalises_naive_now_as_utc(tmp_path):
    """A naive datetime injection is treated as already-UTC --
    NOT silently stamped with the local clock + 'Z' suffix.

    Before this fix, a test pinning ``datetime(2026, 6, 1, 7, 0, 0)``
    on a +05:30 machine would land in the on-disk manifest as
    ``2026-06-01T07:00:00Z`` while operators read it as UTC -- a
    silent ~5h skew per Copilot's PR #199 review.
    """
    naive = _dt.datetime(2026, 6, 1, 7, 0, 0)
    m = Manifest.from_files(
        ticket="T",
        source_run_dir=tmp_path,
        redactor_kind="identity",
        aorta_version="x",
        files=[],
        now=naive,
    )
    assert m.created_at == "2026-06-01T07:00:00Z"


def test_manifest_from_files_normalises_non_utc_now_to_utc(tmp_path):
    """An aware datetime in +05:30 is converted to UTC before the
    'Z' suffix is applied -- so the on-disk timestamp matches what
    ``datetime.now(timezone.utc)`` would have produced at that
    wall-clock instant.
    """
    ist = _dt.timezone(_dt.timedelta(hours=5, minutes=30))
    local = _dt.datetime(2026, 6, 1, 12, 30, 0, tzinfo=ist)
    # 12:30 IST == 07:00 UTC
    m = Manifest.from_files(
        ticket="T",
        source_run_dir=tmp_path,
        redactor_kind="identity",
        aorta_version="x",
        files=[],
        now=local,
    )
    assert m.created_at == "2026-06-01T07:00:00Z"


def test_manifest_total_bytes_helpers():
    files = [
        FileRecord(path="a", bytes_in=10, bytes_out=8),
        FileRecord(path="b", bytes_in=20, bytes_out=18),
    ]
    m = Manifest.from_files(
        ticket="T",
        source_run_dir=Path("."),
        redactor_kind="identity",
        aorta_version="x",
        files=files,
    )
    assert m.total_bytes_in() == 30
    assert m.total_bytes_out() == 26
