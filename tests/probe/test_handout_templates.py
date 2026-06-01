"""Handout template dry-run smoke tests (issue #188 Phase 3 FR 3.7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from aorta.cli import main

RECIPE_ROOT = Path(__file__).resolve().parents[2] / "recipes"

TEMPLATES = [
    "probe-template-torchrun.yaml",
    "probe-template-buck2.yaml",
    "probe-template-bash.yaml",
]


@pytest.mark.parametrize("template", TEMPLATES)
def test_handout_template_dry_run(template: str):
    recipe = RECIPE_ROOT / template
    assert recipe.is_file(), f"missing template {recipe}"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["probe", "--recipe", str(recipe), "--dry-run", "--", "echo", "hi"],
    )
    assert result.exit_code == 0, result.output
    assert "echo" in result.output
