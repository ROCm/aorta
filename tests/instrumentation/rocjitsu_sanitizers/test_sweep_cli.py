"""CLI guardrail-safety tests for ``aorta sweep run`` on mode: sanitizer recipes."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from aorta.cli.sweep import sweep_run

_REPO = Path(__file__).resolve().parents[3]
_WAITCHECK_RECIPE = _REPO / "recipes" / "sanitizers" / "daily-waitcheck-gemm.yaml"
_CONSAN_CLEAN_RECIPE = _REPO / "recipes" / "sanitizers" / "daily-consan-clean.yaml"


def test_sanitizer_dry_run_exits_zero(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        sweep_run,
        ["--recipe", str(_WAITCHECK_RECIPE), "--dry-run", "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "sanitizer report:" in result.output


def test_sanitizer_rejects_non_applicable_flag(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        sweep_run,
        ["--recipe", str(_WAITCHECK_RECIPE), "--dry-run", "--strict"],
    )
    assert result.exit_code != 0
    assert "not valid for a mode: sanitizer recipe" in result.output


def test_sanitizer_missing_backend_exits_nonzero(tmp_path: Path) -> None:
    # No RocJITsu hook / repro binary is provisioned locally, so the real run
    # fails closed to not_checked -> the CLI must exit non-zero, never zero.
    result = CliRunner().invoke(
        sweep_run,
        ["--recipe", str(_CONSAN_CLEAN_RECIPE), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code != 0
    assert "guardrail not clean" in result.output


def test_sanitizer_malformed_recipe_is_click_exception(tmp_path: Path) -> None:
    recipe = tmp_path / "bad.yaml"
    recipe.write_text(
        "schema_version: 1\n"
        "mode: sanitizer\n"
        "ticket: BAD\n"
        "sanitizer_plan:\n"
        "  target: gfx950\n"
        "  source:\n"
        "    kind: gemm_csv\n"
        "    path: shapes.csv\n"
        "    isa_dir: isa\n"
        "  scope:\n"
        "    kind: module\n"  # unsupported scope -> RecipeSchemaError
        "  selection:\n"
        "    requirement: top_dispatch_count\n"
        "    top_n: 3\n"
        "  sanitizers:\n"
        "    - waitcheck\n"
        "  policy:\n"
        "    consan_policy: strict\n"
        "    on_missing_backend: fail\n"
        "  output:\n"
        "    report: sanitizer_report.json\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(sweep_run, ["--recipe", str(recipe), "--output", str(tmp_path / "o")])
    assert result.exit_code != 0
    assert "scope.kind" in result.output
    # A ClickException prints a clean 'Error:' line, not a traceback.
    assert "Traceback" not in result.output
