"""Tests for scripts/check_wheel_version.py (verifies a built wheel's version)."""

import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
try:
    from check_wheel_version import main, wheel_metadata_version  # noqa: E402
finally:
    sys.path.remove(_SCRIPTS_DIR)


def _make_wheel(directory: Path, name: str, version: str) -> Path:
    """Write a minimal wheel (zip with a ``*.dist-info/METADATA``) and return it."""
    wheel = directory / name
    with ZipFile(wheel, "w") as zf:
        zf.writestr(
            f"amd_aorta-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: amd-aorta\nVersion: {version}\n",
        )
    return wheel


def test_wheel_metadata_version_reads_embedded_version(tmp_path):
    wheel = _make_wheel(tmp_path, "amd_aorta-1.2.3-py3-none-any.whl", "1.2.3")
    assert wheel_metadata_version(str(wheel)) == "1.2.3"


def test_main_passes_on_matching_version(tmp_path, capsys):
    _make_wheel(tmp_path, "amd_aorta-1.2.3-py3-none-any.whl", "1.2.3")
    assert main(["1.2.3", str(tmp_path)]) == 0
    assert "matches" in capsys.readouterr().out


def test_main_fails_on_version_mismatch(tmp_path, capsys):
    _make_wheel(tmp_path, "amd_aorta-1.2.3-py3-none-any.whl", "1.2.3")
    assert main(["9.9.9", str(tmp_path)]) == 1
    assert "::error::" in capsys.readouterr().out


def test_main_matches_regardless_of_filename(tmp_path):
    # The check reads METADATA, not the filename: a wheel whose name disagrees
    # with its embedded version still validates against the embedded version.
    _make_wheel(tmp_path, "amd_aorta-0.0.0-py3-none-any.whl", "2.5.0")
    assert main(["2.5.0", str(tmp_path)]) == 0


def test_main_matches_build_tagged_wheel(tmp_path):
    # A future build tag in the filename must not break the check.
    _make_wheel(tmp_path, "amd_aorta-1.2.3-1-py3-none-any.whl", "1.2.3")
    assert main(["1.2.3", str(tmp_path)]) == 0


def test_main_normalizes_rc_separator(tmp_path):
    # setuptools may spell an rc with a separator; normalized compare still matches.
    _make_wheel(tmp_path, "amd_aorta-0.2.1rc20260708-py3-none-any.whl", "0.2.1rc20260708")
    assert main(["0.2.1-rc20260708", str(tmp_path)]) == 0


def test_main_fails_when_no_wheel(tmp_path, capsys):
    assert main(["1.2.3", str(tmp_path)]) == 1
    assert "found: none" in capsys.readouterr().out


def test_main_fails_on_multiple_wheels(tmp_path, capsys):
    _make_wheel(tmp_path, "amd_aorta-1.2.3-py3-none-any.whl", "1.2.3")
    _make_wheel(tmp_path, "amd_aorta-1.2.4-py3-none-any.whl", "1.2.4")
    assert main(["1.2.3", str(tmp_path)]) == 1
    assert "exactly one built wheel" in capsys.readouterr().out


@pytest.mark.parametrize("argc", [0, 3])
def test_main_bad_arg_count_returns_usage(argc, capsys):
    assert main(["x"] * argc) == 2
    assert "usage:" in capsys.readouterr().out
