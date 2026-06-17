"""Unit tests for the artifact-retention engine (issue #231).

Covers the pure :mod:`aorta.run.retention` classify/apply layer: the
level ladder, the record hard-guard, filename-convention classification,
the optional collector manifest (including malformed-manifest tolerance),
empty-dir pruning, and the ``full`` fast no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.run.retention import (
    HEAVY,
    LOG,
    RECORD,
    RETAIN_LEVELS,
    RETENTION_MANIFEST_NAME,
    SUMMARY,
    apply_retention,
    classify_artifact,
)


def _populate(trial_dir: Path) -> None:
    """Drop one artifact of every class (incl. a heavy file in a subdir)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text('{"verdict": "fail"}', encoding="utf-8")
    (trial_dir / "stdout.log").write_text("out", encoding="utf-8")
    (trial_dir / "stderr.log").write_text("err", encoding="utf-8")
    (trial_dir / "probe.env").write_text("K=V", encoding="utf-8")
    (trial_dir / "rollup.summary.json").write_text("{}", encoding="utf-8")
    (trial_dir / "trace.bin").write_text("x" * 1000, encoding="utf-8")
    sub = trial_dir / "prof"
    sub.mkdir()
    (sub / "big.pb").write_text("y" * 2000, encoding="utf-8")


def _names(trial_dir: Path) -> set[str]:
    return {p.relative_to(trial_dir).as_posix() for p in trial_dir.rglob("*") if p.is_file()}


# ---- classify_artifact ----------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("result.json", RECORD),
        ("stdout.log", LOG),
        ("stderr.log", LOG),
        ("probe.env", LOG),
        ("rollup.summary.json", SUMMARY),
        ("prof.summary.pb", SUMMARY),
        ("artifacts.json", SUMMARY),
        ("trace.bin", HEAVY),
        ("prof/big.pb", HEAVY),
        ("anything_else", HEAVY),
    ],
)
def test_classify_by_convention(rel: str, expected: str):
    assert classify_artifact(rel) == expected


def test_manifest_overrides_convention():
    manifest = {"data.json": HEAVY, "roll.json": SUMMARY}
    assert classify_artifact("data.json", manifest) == HEAVY
    assert classify_artifact("roll.json", manifest) == SUMMARY


def test_record_guard_beats_manifest():
    """A manifest can never reclassify the trial record as deletable."""
    assert classify_artifact("result.json", {"result.json": HEAVY}) == RECORD


# ---- apply_retention: the level ladder ------------------------------------


@pytest.mark.parametrize(
    ("level", "present", "gone"),
    [
        ("full", {"result.json", "stdout.log", "trace.bin", "rollup.summary.json", "prof/big.pb"}, set()),
        ("summary", {"result.json", "stdout.log", "rollup.summary.json"}, {"trace.bin", "prof/big.pb"}),
        ("log", {"result.json", "stdout.log", "stderr.log", "probe.env"}, {"trace.bin", "rollup.summary.json", "prof/big.pb"}),
        ("none", {"result.json"}, {"stdout.log", "trace.bin", "rollup.summary.json", "prof/big.pb"}),
    ],
)
def test_apply_levels(tmp_path: Path, level: str, present: set[str], gone: set[str]):
    d = tmp_path / "trial_0"
    _populate(d)
    apply_retention(d, level)
    survivors = _names(d)
    assert present <= survivors, (level, "missing", present - survivors)
    assert not (gone & survivors), (level, "should be gone", gone & survivors)
    # The trial record is sacrosanct at every level.
    assert (d / "result.json").is_file()


def test_full_is_a_noop(tmp_path: Path):
    d = tmp_path / "trial_0"
    _populate(d)
    before = _names(d)
    outcome = apply_retention(d, "full")
    assert outcome.no_op is True
    assert outcome.deleted == ()
    assert _names(d) == before


def test_unknown_level_keeps_everything(tmp_path: Path):
    d = tmp_path / "trial_0"
    _populate(d)
    before = _names(d)
    outcome = apply_retention(d, "bogus")
    assert outcome.no_op is True
    assert _names(d) == before


def test_freed_bytes_and_deleted_list(tmp_path: Path):
    d = tmp_path / "trial_0"
    _populate(d)
    outcome = apply_retention(d, "summary")
    assert set(outcome.deleted) == {"trace.bin", "prof/big.pb"}
    assert outcome.freed_bytes == 3000  # 1000 + 2000


def test_empty_subdirs_are_pruned(tmp_path: Path):
    d = tmp_path / "trial_0"
    _populate(d)
    apply_retention(d, "log")  # drops the only file under prof/
    assert not (d / "prof").exists()


def test_missing_trial_dir_is_noop(tmp_path: Path):
    outcome = apply_retention(tmp_path / "does_not_exist", "none")
    assert outcome.no_op is True


# ---- manifest behaviour ---------------------------------------------------


def test_apply_honors_manifest(tmp_path: Path):
    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")
    (d / "data.json").write_text("z" * 500, encoding="utf-8")  # heavy by convention
    (d / "roll.json").write_text("{}", encoding="utf-8")  # heavy by convention
    (d / RETENTION_MANIFEST_NAME).write_text(
        json.dumps(
            {"artifacts": [{"path": "data.json", "class": HEAVY}, {"path": "roll.json", "class": SUMMARY}]}
        ),
        encoding="utf-8",
    )
    apply_retention(d, "summary")
    survivors = _names(d)
    assert "data.json" not in survivors  # manifest heavy -> pruned
    assert "roll.json" in survivors  # manifest summary -> kept
    assert "result.json" in survivors


def test_malformed_manifest_falls_back_to_convention(tmp_path: Path):
    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")
    (d / RETENTION_MANIFEST_NAME).write_text("not json{", encoding="utf-8")
    (d / "huge.bin").write_text("q" * 100, encoding="utf-8")
    # Must not raise; heavy file pruned by convention; record kept.
    apply_retention(d, "none")
    survivors = _names(d)
    assert survivors == {"result.json"}


def test_unknown_manifest_class_treated_as_heavy(tmp_path: Path):
    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")
    (d / "mystery.dat").write_text("m" * 50, encoding="utf-8")
    (d / RETENTION_MANIFEST_NAME).write_text(
        json.dumps({"artifacts": [{"path": "mystery.dat", "class": "weird"}]}), encoding="utf-8"
    )
    apply_retention(d, "summary")  # heavy dropped at summary
    assert "mystery.dat" not in _names(d)


def test_levels_constant_is_the_documented_ladder():
    assert RETAIN_LEVELS == ("none", "log", "summary", "full")
