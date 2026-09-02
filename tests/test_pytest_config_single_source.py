"""Guard that pytest configuration has exactly one home: ``pytest.ini``.

A root ``pytest.ini`` outranks ``[tool.pytest.ini_options]`` in
``pyproject.toml`` unconditionally, so carrying both means one of them is dead
config that pytest silently ignores. Every other tool here (black, isort, mypy,
ruff) is configured in ``pyproject.toml``, which makes that the tempting place
to add pytest settings too -- these tests fail if someone does.

They also pin the settings that are load-bearing beyond taste, so a future
trim of ``pytest.ini`` cannot quietly drop them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every marker `pytest.ini` registers. CI selects on `gpu` / `rocm`
# (`-m "not gpu and not rocm"` and `-m "gpu or rocm"`); the rest must stay
# registered because `--strict-markers` turns an unregistered marker into a
# collection error.
EXPECTED_MARKERS = frozenset({"unit", "integration", "slow", "gemm", "hw_queue", "gpu", "rocm"})


def test_pyproject_declares_no_pytest_config() -> None:
    # Match a real TOML section header, not a mention of one: pyproject.toml
    # carries a comment naming the table to explain why it is absent.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = [
        line for line in pyproject.splitlines() if line.strip() == "[tool.pytest.ini_options]"
    ]
    assert not declared, (
        "pyproject.toml declares [tool.pytest.ini_options], which pytest ignores "
        "while pytest.ini exists at the repo root. Put pytest settings in "
        "pytest.ini instead."
    )


def test_pytest_ini_is_the_loaded_config(pytestconfig: pytest.Config) -> None:
    inipath = pytestconfig.inipath
    assert inipath is not None, "pytest loaded no config file at all"
    assert inipath == REPO_ROOT / "pytest.ini", f"unexpected config file: {inipath}"


def test_strict_marker_policy_is_configured(pytestconfig: pytest.Config) -> None:
    # tests/test_marker_partition.py clears addopts for its nested collection
    # and re-adds --strict-markers, describing it as "the repo's strict-marker
    # policy". That comment is only true while the repo config actually sets it.
    assert "--strict-markers" in pytestconfig.getini("addopts")


def test_every_marker_stays_registered(pytestconfig: pytest.Config) -> None:
    registered = {entry.split(":", 1)[0] for entry in pytestconfig.getini("markers")}
    assert (
        EXPECTED_MARKERS <= registered
    ), f"markers dropped from pytest.ini: {sorted(EXPECTED_MARKERS - registered)}"


def test_testpaths_still_points_at_tests(pytestconfig: pytest.Config) -> None:
    assert pytestconfig.getini("testpaths") == ["tests"]
