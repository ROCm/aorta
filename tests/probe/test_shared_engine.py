"""Shared-engine contract: both CLIs reach :func:`aorta.triage.runner.run_recipe`.

This test is FR 1.5 in the rubric -- it pins the "no parallel runner"
guarantee that issue #188 names as P0. Mirrors the mock-``run_trials``
pattern from ``tests/triage/test_runner_b1_api.py`` but at one level up:
we mock ``run_recipe`` itself and assert both ``aorta probe`` and
``aorta triage run`` reach it with a validated :class:`Recipe`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import aorta.cli.probe as probe_cli
import aorta.cli.triage as triage_cli
from aorta.cli.probe import probe
from aorta.cli.triage import triage
from aorta.triage.recipe import Recipe

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_run_recipe(monkeypatch):
    """Replace ``run_recipe`` in BOTH CLI modules with a single Mock.

    The probe and triage Click handlers each ``from aorta.triage.runner
    import run_recipe``, so the symbol is bound in each module's
    namespace; patching only ``aorta.triage.runner.run_recipe`` would
    miss them. Patching both binding sites gives us a single shared
    Mock either CLI must end up calling.
    """
    mock = MagicMock(return_value=Path("/tmp/mock-run-dir"))
    monkeypatch.setattr(probe_cli, "run_recipe", mock)
    monkeypatch.setattr(triage_cli, "run_recipe", mock)
    return mock


def test_probe_reaches_run_recipe(mock_run_recipe, tmp_path):
    """`aorta probe` calls run_recipe with a validated probe-mode Recipe."""
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            str(tmp_path / "out"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_run_recipe.assert_called_once()
    # First positional arg is the validated Recipe; keyword args carry
    # layout=flat_resume + resume_existing=True + subprocess_argv.
    args, kwargs = mock_run_recipe.call_args
    recipe_arg = args[0] if args else kwargs.get("recipe")
    assert isinstance(recipe_arg, Recipe)
    assert recipe_arg.probe_extras is not None
    assert kwargs.get("layout") == "flat_resume"
    assert kwargs.get("resume_existing") is True
    assert kwargs.get("subprocess_argv") == ("echo", "hi")


def test_triage_reaches_run_recipe(mock_run_recipe, tmp_path):
    """`aorta triage run --recipe ...` reaches the same run_recipe."""
    triage_recipe = tmp_path / "triage.yaml"
    triage_recipe.write_text(
        "schema_version: 1\n"
        "workload: fsdp\n"
        "trials: 1\n"
        "steps: 1\n"
        "cells:\n"
        "  - name: baseline-local\n"
        "    mitigations: [none]\n"
        "    environment: local\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        triage,
        ["run", "--recipe", str(triage_recipe), "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    mock_run_recipe.assert_called_once()
    args, kwargs = mock_run_recipe.call_args
    recipe_arg = args[0] if args else kwargs.get("recipe")
    assert isinstance(recipe_arg, Recipe)
    assert recipe_arg.probe_extras is None  # triage-mode never sets this.


def test_probe_and_triage_share_run_recipe(mock_run_recipe, tmp_path):
    """Both CLIs together: the SAME mocked run_recipe is reached twice."""
    triage_recipe = tmp_path / "triage.yaml"
    triage_recipe.write_text(
        "schema_version: 1\n"
        "workload: fsdp\n"
        "trials: 1\n"
        "steps: 1\n"
        "cells:\n"
        "  - name: baseline-local\n"
        "    mitigations: [none]\n"
        "    environment: local\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    r1 = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            str(tmp_path / "p"),
            "--",
            "echo",
            "hi",
        ],
    )
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(
        triage,
        ["run", "--recipe", str(triage_recipe), "--output-dir", str(tmp_path / "t")],
    )
    assert r2.exit_code == 0, r2.output
    assert mock_run_recipe.call_count == 2
