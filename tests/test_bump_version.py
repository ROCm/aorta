"""Tests for scripts/bump_version.py (the release version bumper)."""

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
try:
    from bump_version import (  # noqa: E402
        apply_suffix,
        bump_version,
        main,
        read_version,
        resolve_new_version,
        set_version,
    )
finally:
    # Keep the import-time path change local to bump_version so the rest of the
    # pytest session can't accidentally import the many top-level modules under
    # scripts/.
    sys.path.remove(_SCRIPTS_DIR)

SAMPLE = """\
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aorta"
version = "0.2.0"
requires-python = ">=3.10"
"""


@pytest.mark.parametrize(
    ("level", "expected"),
    [("patch", "0.2.1"), ("minor", "0.3.0"), ("major", "1.0.0")],
)
def test_bump_version_levels(level, expected):
    assert bump_version("0.2.0", level) == expected


def test_bump_version_rejects_non_semver():
    with pytest.raises(ValueError):
        bump_version("0.2", "patch")


def test_read_version_reads_project_table():
    assert read_version(SAMPLE) == "0.2.0"


def test_set_version_replaces_only_project_version():
    updated = set_version(SAMPLE, "0.3.0")
    assert read_version(updated) == "0.3.0"
    # Unrelated lines (including the build-system table) stay byte-for-byte intact.
    assert 'requires = ["setuptools>=61.0", "wheel"]' in updated
    assert 'build-backend = "setuptools.build_meta"' in updated
    assert updated.count('version = "') == 1


def test_set_version_preserves_trailing_content():
    text = '[project]\nversion = "0.2.0"  # keep me\n'
    assert set_version(text, "0.2.1") == '[project]\nversion = "0.2.1"  # keep me\n'


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
    # dot-prefixed PEP 440 segments are rejected, not silently dropped --
    # matches the documented contract (and the four-segment guard above).
    with pytest.raises(ValueError):
        apply_suffix(current, "rc20260619")


@pytest.mark.parametrize(
    "suffix",
    [
        "",  # empty -> would emit version = "0.2.0" unchanged / meaningless
        'rc"20260619',  # embedded quote would break/inject the TOML version line
        "rc 20260619",  # whitespace is not a valid version token
        "rc\n20260619",  # newline would inject extra TOML content
        "rc;rm -rf",  # arbitrary punctuation outside the safe charset
    ],
)
def test_apply_suffix_rejects_unsafe_suffix(suffix):
    # The suffix is concatenated straight into the quoted TOML version string,
    # so anything that isn't a clean PEP 440-style token must be rejected before
    # it can corrupt pyproject.toml.
    with pytest.raises(ValueError):
        apply_suffix("0.2.0", suffix)



@pytest.mark.parametrize("nl", ["\r\n", "\r", "\n"])
def test_set_version_preserves_line_endings(nl):
    """The "byte-for-byte untouched" promise must hold on CRLF/CR checkouts
    (e.g. a Windows ``pyproject.toml``), not just LF — only the version value
    changes, every line ending is preserved exactly.
    """
    text = nl.join(["[project]", 'name = "aorta"', 'version = "0.2.0"', ""])
    expected = nl.join(["[project]", 'name = "aorta"', 'version = "0.3.0"', ""])
    updated = set_version(text, "0.3.0")
    assert read_version(text) == "0.2.0"
    assert updated == expected


@pytest.mark.parametrize("nl", ["\r\n", "\r", "\n"])
def test_main_preserves_line_endings_end_to_end(tmp_path, capsys, nl):
    """End-to-end: running the CLI on a CRLF/CR file must not normalize line
    endings. ``read_text``/``write_text`` would translate newlines and silently
    rewrite the whole file; main() reads + writes with ``newline=""`` so only
    the version value changes on disk (compared as raw bytes).
    """
    raw = nl.join(["[project]", 'name = "aorta"', 'version = "0.2.0"', ""]).encode("utf-8")
    expected = nl.join(["[project]", 'name = "aorta"', 'version = "0.3.0"', ""]).encode("utf-8")
    p = tmp_path / "pyproject.toml"
    p.write_bytes(raw)
    rc = main(["--set", "0.3.0", "--pyproject", str(p)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "0.3.0"
    assert p.read_bytes() == expected


def test_main_suffix_end_to_end(tmp_path, capsys):
    """End-to-end CLI coverage for ``--suffix``: it stamps the suffix onto the
    base version, writes it back, and prints the new version (guards the
    argparse wiring + apply_suffix precedence, not just the helper in isolation).
    """
    p = tmp_path / "pyproject.toml"
    p.write_text('[project]\nname = "aorta"\nversion = "0.2.0"\n')
    rc = main(["--suffix", "rc20260620", "--pyproject", str(p)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "0.2.0rc20260620"
    assert 'version = "0.2.0rc20260620"' in p.read_text()


def test_main_set_takes_precedence_over_suffix(tmp_path, capsys):
    """``--set`` outranks ``--suffix`` (documented precedence: explicit > suffix
    > level); passing both must yield the explicit version verbatim.
    """
    p = tmp_path / "pyproject.toml"
    p.write_text('[project]\nname = "aorta"\nversion = "0.2.0"\n')
    rc = main(["--set", "1.4.2", "--suffix", "rc20260620", "--pyproject", str(p)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "1.4.2"
    assert 'version = "1.4.2"' in p.read_text()



# A trailing inline comment on the table header is valid TOML; the bumper must
# still recognize the [project] table (regression for the header parse).
COMMENTED_HEADER = '[project]  # the package\nname = "aorta"\nversion = "0.2.0"\n'


def test_read_version_handles_commented_table_header():
    assert read_version(COMMENTED_HEADER) == "0.2.0"


def test_set_version_handles_commented_table_header():
    updated = set_version(COMMENTED_HEADER, "0.3.0")
    assert read_version(updated) == "0.3.0"
    # The header (comment and all) is preserved verbatim.
    assert updated.startswith("[project]  # the package\n")


def test_resolve_new_version_explicit_overrides_level():
    assert resolve_new_version("0.2.0", "patch", "5.6.7") == "5.6.7"


def test_resolve_new_version_suffix_overrides_level():
    assert resolve_new_version("0.2.0", "patch", None, "rc20260619") == "0.2.0rc20260619"


def test_resolve_new_version_explicit_overrides_suffix():
    assert resolve_new_version("0.2.0", None, "5.6.7", "rc20260619") == "5.6.7"



def test_resolve_new_version_rejects_bad_explicit():
    with pytest.raises(ValueError):
        resolve_new_version("0.2.0", None, "not-a-version")


def test_resolve_new_version_requires_input():
    with pytest.raises(ValueError):
        resolve_new_version("0.2.0", None, None)
