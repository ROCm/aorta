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


def test_record_guard_matches_exact_path_not_basename():
    """Only the top-level result.json is the record; a nested one is heavy.

    Resume / matrix completion keys off ``trial_dir/result.json``
    specifically, so a same-named heavy collector file under a subdir must
    stay prunable rather than being protected by basename.
    """
    assert classify_artifact("result.json") == RECORD
    assert classify_artifact("sub/result.json") == HEAVY
    assert classify_artifact("prof/nested/result.json") == HEAVY


# ---- apply_retention: the level ladder ------------------------------------


@pytest.mark.parametrize(
    ("level", "present", "gone"),
    [
        ("full", {"result.json", "stdout.log", "stderr.log", "probe.env", "trace.bin", "rollup.summary.json", "prof/big.pb"}, set()),
        ("summary", {"result.json", "stdout.log", "stderr.log", "probe.env", "rollup.summary.json"}, {"trace.bin", "prof/big.pb"}),
        ("log", {"result.json", "stdout.log", "stderr.log", "probe.env"}, {"trace.bin", "rollup.summary.json", "prof/big.pb"}),
        ("none", {"result.json"}, {"stdout.log", "stderr.log", "probe.env", "trace.bin", "rollup.summary.json", "prof/big.pb"}),
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


def test_nested_result_json_is_pruned(tmp_path: Path):
    """A heavy collector file named result.json under a subdir is prunable."""
    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")  # the real record
    sub = d / "collector"
    sub.mkdir()
    (sub / "result.json").write_text("z" * 500, encoding="utf-8")  # heavy
    apply_retention(d, "none")
    survivors = _names(d)
    assert survivors == {"result.json"}  # nested one pruned, real record kept


def test_non_list_artifacts_is_malformed(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A manifest whose ``artifacts`` is not a list warns + falls back."""
    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")
    (d / "huge.bin").write_text("q" * 100, encoding="utf-8")
    (d / RETENTION_MANIFEST_NAME).write_text(
        json.dumps({"artifacts": {"path": "huge.bin", "class": "summary"}}),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        apply_retention(d, "none")
    # Warned about the malformed manifest...
    assert any("malformed" in r.getMessage() for r in caplog.records)
    # ...and fell back to convention: huge.bin is heavy -> pruned at none.
    assert _names(d) == {"result.json"}


def test_symlinked_manifest_is_malformed(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A symlinked artifacts.json must not be dereferenced (it could read a
    file outside the trial tree). Treat it as malformed + fall back."""
    # A valid manifest living outside the trial dir that, if followed, would
    # keep huge.bin as a summary (i.e. survive pruning at level "none").
    outside = tmp_path / "outside"
    outside.mkdir()
    real_manifest = outside / "real.json"
    real_manifest.write_text(
        json.dumps({"artifacts": [{"path": "huge.bin", "class": "summary"}]}),
        encoding="utf-8",
    )

    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")
    (d / "huge.bin").write_text("q" * 100, encoding="utf-8")
    (d / RETENTION_MANIFEST_NAME).symlink_to(real_manifest)

    with caplog.at_level("WARNING"):
        apply_retention(d, "none")

    # The symlink was NOT followed: warned about symlink, fell back to
    # convention, and huge.bin (heavy by name) was pruned at level none.
    # The manifest symlink itself survives (deletion side skips symlinks).
    assert any("symlink" in r.getMessage() for r in caplog.records)
    survivors = _names(d)
    assert "huge.bin" not in survivors  # pruned by convention
    assert "result.json" in survivors  # record always kept


def test_symlink_escaping_trial_dir_is_not_deleted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A symlink pointing outside the trial dir is skipped, never unlinked."""
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "precious.bin"
    victim.write_text("do not delete", encoding="utf-8")

    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")
    link = d / "escape.bin"  # heavy name; would be pruned if treated as a file
    link.symlink_to(victim)

    with caplog.at_level("WARNING"):
        outcome = apply_retention(d, "none")

    assert link.is_symlink()  # the link itself survives
    assert victim.is_file() and victim.read_text() == "do not delete"
    assert "escape.bin" not in outcome.deleted
    assert any("symlink" in r.getMessage() for r in caplog.records)


def test_symlinked_subdir_is_not_descended(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A symlinked *directory* must be kept and never walked into.

    Enumeration uses ``os.walk(followlinks=False)`` so a collector that
    drops a symlinked subdir pointing at an external (possibly huge) tree
    can't make retention traverse it. The link is kept, its target is
    untouched, and nothing under it is pruned.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "external_heavy.bin").write_text("z" * 4096, encoding="utf-8")

    d = tmp_path / "trial_0"
    d.mkdir()
    (d / "result.json").write_text("{}", encoding="utf-8")
    (d / "trace.bin").write_text("x" * 1000, encoding="utf-8")  # heavy, real file
    (d / "linkdir").symlink_to(outside, target_is_directory=True)

    with caplog.at_level("WARNING"):
        outcome = apply_retention(d, "none")

    # The real heavy file was pruned; the symlinked dir + its target survive.
    assert "trace.bin" in outcome.deleted
    assert (d / "linkdir").is_symlink()
    assert (outside / "external_heavy.bin").is_file()  # never descended/pruned
    assert "linkdir" in outcome.kept
    # The external file was never even enumerated as a candidate.
    assert "linkdir/external_heavy.bin" not in outcome.deleted
    assert any("symlink" in r.getMessage() for r in caplog.records)


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
