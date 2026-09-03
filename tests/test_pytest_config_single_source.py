"""Guard that pytest configuration has exactly one home: ``pytest.ini``.

A root ``pytest.ini`` outranks pytest configuration in ``pyproject.toml``
unconditionally, so carrying both means one of them is dead config. How loudly
that fails depends on the pytest in use: from pytest 9 the session header reads
``configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)``,
while pytest 8 -- still allowed by our ``pytest>=8.0.0`` floor -- drops it with
no diagnostic at all. Every other tool here (black, isort, mypy, ruff) is
configured in ``pyproject.toml``, which makes that the tempting place to add
pytest settings too -- these tests fail if someone does.

They also pin the settings that are load-bearing beyond taste, so a future
trim of ``pytest.ini`` cannot quietly drop them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest depends on tomli below 3.11, so this is available wherever we run.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every marker `pytest.ini` registers. CI selects on `gpu` / `rocm`
# (`-m "not gpu and not rocm"` and `-m "gpu or rocm"`); the rest must stay
# registered because `--strict-markers` turns an unregistered marker into a
# collection error.
EXPECTED_MARKERS = frozenset({"unit", "integration", "slow", "gemm", "hw_queue", "gpu", "rocm"})

# Spellings that all declare the same pytest configuration once TOML is parsed.
# A guard matching the canonical text would wave every one of these through.
EQUIVALENT_SPELLINGS = {
    "canonical": '[tool.pytest.ini_options]\naddopts = "-q"\n',
    "spaced-dotted-key": '[tool . pytest . ini_options]\naddopts = "-q"\n',
    "quoted-keys": '["tool"."pytest"."ini_options"]\naddopts = "-q"\n',
    "top-level-dotted-key": 'tool.pytest.ini_options.addopts = "-q"\n',
    "inline-table": '[tool.pytest]\nini_options = {addopts = "-q"}\n',
    # pytest 9 also reads `[tool.pytest]` directly, in native-TOML mode.
    "toml-mode-table": '[tool.pytest]\naddopts = ["-q"]\n',
}

# TOML that names the table without declaring it, or configures something else.
NON_DECLARATIONS = {
    "comment-only": "# pytest settings do not live here, see [tool.pytest.ini_options]\n",
    "other-tool": "[tool.black]\nline-length = 100\n",
    "empty-table": "[tool.pytest]\n",
    "string-value": 'addopts = "[tool.pytest.ini_options]"\n',
}


def _declared_pytest_tables(pyproject: str) -> list[str]:
    """Names of the pytest config tables ``pyproject.toml`` declares, if any.

    Follows pytest's own detection in ``_pytest.config.findpaths``, which reads
    ``[tool.pytest.ini_options]`` (ini mode) and, from pytest 9, any other key
    under ``[tool.pytest]`` (native-TOML mode). Parsing rather than matching
    text is what makes this immune to TOML spelling -- whitespace around the
    dots, quoted key parts and top-level dotted keys all reach the same table.

    Deliberately a superset, not an exact mirror: both tables are reported on
    every pytest version, so a bare ``[tool.pytest]`` cannot sit here dormant
    under pytest 8 and become live config on the next major bump.
    ``test_guard_agrees_with_pytests_own_config_detection`` checks the part
    that must not diverge -- that nothing pytest honours slips past.
    """
    tool = tomllib.loads(pyproject).get("tool", {})
    table = tool.get("pytest", {}) if isinstance(tool, dict) else {}
    if not isinstance(table, dict):
        return ["tool.pytest"]
    declared = ["tool.pytest.ini_options"] if "ini_options" in table else []
    if any(key != "ini_options" for key in table):
        declared.append("tool.pytest")
    return declared


def _pytest_reads_config(tmp_path: Path, pyproject: str) -> bool:
    """Whether pytest's own loader finds pytest config in this ``pyproject.toml``."""
    from _pytest.config.findpaths import load_config_dict_from_file

    path = tmp_path / "pyproject.toml"
    path.write_text(pyproject, encoding="utf-8")
    return load_config_dict_from_file(path) is not None


def test_pyproject_declares_no_pytest_config() -> None:
    # Parse the TOML rather than grep it: pyproject.toml carries a comment
    # naming the table to explain why it is absent, and the table has several
    # equally valid spellings that a textual match would miss.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = _declared_pytest_tables(pyproject)
    assert not declared, (
        f"pyproject.toml declares {', '.join(declared)}, which pytest ignores "
        "while pytest.ini exists at the repo root. Put pytest settings in "
        "pytest.ini instead."
    )


@pytest.mark.parametrize("spelling", sorted(EQUIVALENT_SPELLINGS))
def test_guard_catches_every_equivalent_spelling(spelling: str) -> None:
    assert _declared_pytest_tables(EQUIVALENT_SPELLINGS[spelling])


@pytest.mark.parametrize("case", sorted(NON_DECLARATIONS))
def test_guard_passes_toml_that_declares_nothing(case: str) -> None:
    assert not _declared_pytest_tables(NON_DECLARATIONS[case])


def test_guard_agrees_with_pytests_own_config_detection(tmp_path: Path) -> None:
    """Check the "mirrors pytest" claim above against pytest, rather than assert it.

    The guard must fire for every spelling pytest actually honours; missing one
    is exactly what lets the dead config back in. It is deliberately the
    stricter side of the mirror -- pytest 8 ignores a bare ``[tool.pytest]``
    that pytest 9 honours, and the guard rejects it on both, so a later pytest
    bump cannot quietly revive the table.
    """
    honoured = {name: _pytest_reads_config(tmp_path, t) for name, t in EQUIVALENT_SPELLINGS.items()}
    missed = [
        name
        for name, seen in honoured.items()
        if seen and not _declared_pytest_tables(EQUIVALENT_SPELLINGS[name])
    ]
    assert not missed, f"pytest reads config the guard does not flag: {missed}"
    # Without this the check above passes vacuously if the private loader moves.
    assert sum(honoured.values()) >= 5, f"pytest honoured too few spellings: {honoured}"

    for name, toml in NON_DECLARATIONS.items():
        assert not _pytest_reads_config(tmp_path, toml), f"pytest reads config from {name}"
        assert not _declared_pytest_tables(toml), f"guard false-positives on {name}"


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
