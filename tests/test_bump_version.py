"""Tests for scripts/bump_version.py (computes the next release version from git tags)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The two tests below shell out to the real ``git`` executable; skip (rather than
# error with FileNotFoundError) on minimal environments that lack it.
requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git executable not available"
)

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
try:
    from bump_version import (  # noqa: E402
        apply_suffix,
        bump_version,
        current_version_from_git,
        main,
        resolve_new_version,
    )
finally:
    # Keep the import-time path change local to bump_version so the rest of the
    # pytest session can't accidentally import the many top-level modules under
    # scripts/.
    sys.path.remove(_SCRIPTS_DIR)


@pytest.mark.parametrize(
    ("level", "expected"),
    [("patch", "0.2.1"), ("minor", "0.3.0"), ("major", "1.0.0")],
)
def test_bump_version_levels(level, expected):
    assert bump_version("0.2.0", level) == expected


def test_bump_version_rejects_non_semver():
    with pytest.raises(ValueError):
        bump_version("0.2", "patch")


def test_apply_suffix_appends_to_base():
    assert apply_suffix("0.2.0", "rc20260619") == "0.2.0rc20260619"


def test_apply_suffix_is_idempotent_on_base():
    # Re-stamping an already-suffixed version uses the base, not the old suffix.
    assert apply_suffix("0.2.0rc20260101", "rc20260619") == "0.2.0rc20260619"


def test_apply_suffix_rejects_non_semver():
    with pytest.raises(ValueError):
        apply_suffix("not-a-version", "rc20260619")


def test_apply_suffix_rejects_four_segment_version():
    # A malformed 4-segment value must not be silently truncated to its
    # MAJOR.MINOR.PATCH prefix (which would mint a misleading suffixed version).
    with pytest.raises(ValueError):
        apply_suffix("0.2.0.1", "rc20260619")


@pytest.mark.parametrize("current", ["0.2.0.dev0", "0.2.0.post1"])
def test_apply_suffix_rejects_dot_prefixed_prerelease(current):
    # The anchored prefix only strips a non-dot-prefixed suffix (e.g. rcN);
    # dot-prefixed PEP 440 segments are rejected, not silently dropped.
    with pytest.raises(ValueError):
        apply_suffix(current, "rc20260619")


@pytest.mark.parametrize(
    "suffix",
    [
        "",  # empty -> meaningless (base unchanged)
        'rc"20260619',  # embedded quote
        "rc 20260619",  # whitespace is not a valid version token
        "rc\n20260619",  # newline would inject extra content
        "rc;rm -rf",  # arbitrary punctuation outside the safe charset
    ],
)
def test_apply_suffix_rejects_unsafe_suffix(suffix):
    # The suffix is concatenated straight onto the version string, so anything
    # that isn't a clean PEP 440-style token must be rejected.
    with pytest.raises(ValueError):
        apply_suffix("0.2.0", suffix)


def test_resolve_new_version_explicit_overrides_level():
    assert resolve_new_version("0.2.0", "patch", "5.6.7") == "5.6.7"


def test_resolve_new_version_suffix_stamps_onto_bumped_base():
    # suffix combines WITH the bump: patch bumps 0.2.0 -> 0.2.1, then the rc
    # suffix is stamped onto that next base (so the rc precedes the eventual
    # 0.2.1 stable), rather than an rc of the already-released 0.2.0.
    assert resolve_new_version("0.2.0", "patch", None, "rc20260619") == "0.2.1rc20260619"


def test_resolve_new_version_suffix_without_level_uses_current_base():
    assert resolve_new_version("0.2.0", None, None, "rc20260619") == "0.2.0rc20260619"


def test_resolve_new_version_explicit_overrides_level_with_suffix():
    assert resolve_new_version("0.2.0", "patch", "5.6.7", "rc20260619") == "5.6.7rc20260619"


def test_resolve_new_version_rejects_bad_explicit():
    with pytest.raises(ValueError):
        resolve_new_version("0.2.0", None, "not-a-version")


def test_resolve_new_version_rejects_non_semver_current_without_bump():
    with pytest.raises(ValueError):
        resolve_new_version("0.2.0rc1", None, None)


def test_main_bump_uses_current_override(capsys):
    assert main(["patch", "--current", "0.2.0"]) == 0
    assert capsys.readouterr().out.strip() == "0.2.1"


def test_main_set_takes_precedence_over_level(capsys):
    assert main(["patch", "--set", "1.4.2", "--current", "0.2.0"]) == 0
    assert capsys.readouterr().out.strip() == "1.4.2"


def test_main_nightly_suffix_on_next_base(capsys):
    # Mirrors nightly.yml: bump to the next patch base, then stamp the rc date.
    assert main(["patch", "--suffix", "rc20260620", "--current", "0.2.0"]) == 0
    assert capsys.readouterr().out.strip() == "0.2.1rc20260620"


@requires_git
def test_current_version_from_git_picks_highest_release_tag(tmp_path, monkeypatch):
    """The helper returns the highest ``vX.Y.Z`` tag and ignores non-release
    tags such as the rolling ``dev-wheels`` nightly tag."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "f").write_text("x")
    git("add", "f")
    git("commit", "-qm", "c1")
    git("tag", "v0.2.0")
    git("tag", "dev-wheels")  # non-release tag must be ignored
    git("tag", "v0.10.0")  # highest by semver, not lexical
    monkeypatch.chdir(repo)
    assert current_version_from_git() == "0.10.0"


@requires_git
def test_current_version_from_git_defaults_to_initial(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    monkeypatch.chdir(repo)
    assert current_version_from_git() == "0.0.0"
