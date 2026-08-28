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


# ---- fd-relative engine (trusted_root=) ---------------------------------

import contextlib  # noqa: E402 -- grouped with the fd-relative tests

from aorta.run import _fsafe  # noqa: E402

_needs_fd = pytest.mark.skipif(
    not _fsafe.HAVE_FD_TRAVERSAL, reason="fd-relative traversal unsupported here"
)


@_needs_fd
def test_fd_engine_prunes_in_tree_and_keeps_the_record(tmp_path: Path):
    """With a trusted root, the fd engine deletes heavy files but not the record."""
    results = tmp_path / "results"
    trial = results / "wl" / "trial_d0_m0_t0"
    _populate(trial)
    outcome = apply_retention(trial, "none", trusted_root=results.resolve())
    names = _names(trial)
    assert "result.json" in names  # record hard-guarded
    assert "trace.bin" not in names  # heavy pruned
    assert "prof/big.pb" in outcome.deleted


@_needs_fd
def test_fd_engine_refuses_a_symlinked_ancestor(tmp_path: Path, caplog):
    """A symlinked component above the trial dir makes the prune refuse + keep."""
    results = tmp_path / "results"
    real_trial = results / "wl_real" / "trial_d0_m0_t0"
    _populate(real_trial)
    outside = tmp_path / "outside"
    (outside / "trial_d0_m0_t0").mkdir(parents=True)
    victim = outside / "trial_d0_m0_t0" / "trace.bin"
    victim.write_text("keep me", encoding="utf-8")
    # <results>/wl is a symlink to the external tree.
    (results / "wl").symlink_to(outside, target_is_directory=True)

    with caplog.at_level("WARNING"):
        outcome = apply_retention(
            results / "wl" / "trial_d0_m0_t0", "none", trusted_root=results.resolve()
        )
    assert outcome.no_op
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert any("symlink" in r.getMessage() for r in caplog.records)


@_needs_fd
def test_fd_engine_ancestor_swap_after_open_is_inert(tmp_path: Path, monkeypatch):
    """A mid-prune ancestor swap cannot redirect the deletes (TOCTOU closed)."""
    results = tmp_path / "results"
    trial = results / "wl" / "trial_d0_m0_t0"
    _populate(trial)
    outside = tmp_path / "outside"
    planted = outside / "trial_d0_m0_t0"
    planted.mkdir(parents=True)
    victim = planted / "trace.bin"
    victim.write_text("keep me", encoding="utf-8")

    real_open = _fsafe.open_dir_nofollow
    swapped = {"done": False}

    @contextlib.contextmanager
    def swapping_open(trusted_root, components, **kwargs):
        with real_open(trusted_root, components, **kwargs) as fd:
            if not swapped["done"]:
                swapped["done"] = True
                (results / "wl").rename(results / "wl_real")
                (results / "wl").symlink_to(outside, target_is_directory=True)
            yield fd

    monkeypatch.setattr(_fsafe, "open_dir_nofollow", swapping_open)

    apply_retention(trial, "none", trusted_root=results.resolve())
    # The prune ran against the held fd; the planted external victim survives.
    assert victim.read_text(encoding="utf-8") == "keep me"


@_needs_fd
def test_fd_engine_refuses_a_results_root_replaced_by_a_real_directory(
    tmp_path: Path, caplog
):
    """A pinned anchor refuses the swap that no ``O_NOFOLLOW`` check can see.

    The payload renames the results directory aside and moves a real directory
    into its pathname. Nothing on the way down is a symlink, so only the frozen
    inode distinguishes the planted tree from the operator's -- without it the
    prune deletes the planted files.
    """
    results = tmp_path / "results"
    trial = results / "wl" / "trial_d0_m0_t0"
    _populate(trial)
    anchor = _fsafe.TrustedAnchor.freeze(results.resolve())
    assert anchor.identity is not None

    results.rename(tmp_path / "results.moved")
    planted = tmp_path / "planted"
    _populate(planted / "wl" / "trial_d0_m0_t0")
    planted.rename(results)
    victim = results / "wl" / "trial_d0_m0_t0" / "trace.bin"

    outcome = apply_retention(trial, "none", trusted_root=anchor)
    assert outcome.no_op
    assert victim.exists()

    # Unpinned, the same prune goes ahead against the planted tree -- the gap
    # the pin closes.
    assert not apply_retention(trial, "none", trusted_root=results.resolve()).no_op
    assert not victim.exists()


@_needs_fd
def test_fd_engine_treats_a_missing_tree_as_nothing_to_prune(tmp_path: Path, caplog):
    """An absent collector tree is a no-op, not a refusal.

    A validated-only collector writes no directory at all, and the pathname
    engine already treats a missing ``trial_dir`` as a no-op; warning about a
    symlink that is not there would send an operator hunting for one.
    """
    results = tmp_path / "results"
    results.mkdir()
    with caplog.at_level("WARNING"):
        outcome = apply_retention(
            results / "wl" / "trial_d0_m0_t0", "none", trusted_root=results.resolve()
        )
    assert outcome.no_op
    assert not any("symlink" in record.getMessage() for record in caplog.records)


@_needs_fd
def test_fd_engine_reads_manifest_no_follow(tmp_path: Path):
    """The fd engine honours a real manifest entry's class."""
    results = tmp_path / "results"
    trial = results / "wl" / "trial_d0_m0_t0"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text("{}", encoding="utf-8")
    (trial / "mystery.dat").write_text("m" * 50, encoding="utf-8")
    (trial / RETENTION_MANIFEST_NAME).write_text(
        json.dumps({"artifacts": [{"path": "mystery.dat", "class": "summary"}]}),
        encoding="utf-8",
    )
    # summary-classed mystery.dat is kept at level "summary".
    apply_retention(trial, "summary", trusted_root=results.resolve())
    assert "mystery.dat" in _names(trial)
