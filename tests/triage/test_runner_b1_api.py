"""Tests asserting B2 drives B1 via the Python `run_trials` API, NOT subprocess.

Acceptance criteria (from issue #151 §"Plumbing"):

* No `subprocess` import anywhere under `src/aorta/triage/`.
* ``run_trials`` is called exactly once per cell with the expected
  :class:`aorta.run.RunRequest`.
* The `--mode matrix` flag shim and `--recipe` path both funnel through
  the same call site -- verified by driving both and asserting the same
  per-call shape.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import aorta.triage.runner as runner
from aorta.cli.triage import triage
from aorta.instrumentation.environment import EnvSnapshot
from aorta.run import RunRequest
from aorta.triage.recipe import build_recipe_from_flags

# ---- Fixtures -------------------------------------------------------------


class _FakeTrial:
    def __init__(self):
        self.exit_status = "ok"
        self.wall_clock_sec = 1.0
        self.result = {"passed": True, "step_times_ms": [100.0]}


def _clean_snapshot() -> EnvSnapshot:
    return EnvSnapshot(
        schema_version="1.0",
        captured_at="2026-04-28T14:12:03Z",
        system_health=None,
        rocm={},
        hip={},
        hipblaslt={},
        runtime_context={},
        docker=None,
        env_vars={},
        python_version="3.11.0",
        pytorch_version=None,
        partial=False,
        partial_reasons=[],
    )


@pytest.fixture
def patched_env(monkeypatch):
    monkeypatch.setattr(runner, "collect_env", MagicMock(return_value=_clean_snapshot()))


@pytest.fixture
def patched_run_trials(monkeypatch):
    mock = MagicMock(return_value=[_FakeTrial(), _FakeTrial()])
    monkeypatch.setattr(runner, "run_trials", mock)
    return mock


# ---- Source-level plumbing guard -----------------------------------------


def test_triage_package_has_no_subprocess_import():
    """Acceptance: grep confirms subprocess is never imported under src/aorta/triage."""
    triage_dir = Path(__file__).resolve().parents[2] / "src" / "aorta" / "triage"
    assert triage_dir.is_dir()
    for py in triage_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert (
            "import subprocess" not in text
        ), f"{py} imports subprocess; B2 must drive B1 via the Python API only."
        assert "from subprocess" not in text, f"{py} imports from subprocess."


# ---- Per-cell RunRequest shape -------------------------------------------


def test_run_trials_called_once_per_cell_with_expected_request(
    tmp_path, patched_env, patched_run_trials
):
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off,xnack",
        environment_axis="local",
        trials=3,
        steps=50,
        ticket="T-1",
    )
    runner.run_recipe(r, output_dir=tmp_path, timestamp="2026-01-01T00-00-00")
    assert patched_run_trials.call_count == 3

    calls = [c.args[0] for c in patched_run_trials.call_args_list]
    names = [(c.workload, c.mitigations, c.environment, c.trials, c.steps) for c in calls]
    assert names == [
        ("fsdp", ("none",), "local", 3, 50),
        ("fsdp", ("tf32_off",), "local", 3, 50),
        ("fsdp", ("xnack",), "local", 3, 50),
    ]


def test_per_cell_results_dir_points_into_cells_subdir(tmp_path, patched_env, patched_run_trials):
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="local",
        trials=1,
        steps=10,
        ticket="T-42",
    )
    runner.run_recipe(r, output_dir=tmp_path, timestamp="2026-02-02T02-02-02")
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    expected_tail = Path("T-42") / "fsdp" / "2026-02-02T02-02-02" / "cells" / "none-local"
    assert str(req.results_dir).endswith(str(expected_tail))


def test_cell_overrides_flow_through_effective_values(tmp_path, patched_env, patched_run_trials):
    """Per-cell trials/steps overrides take precedence over recipe-level values."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=2,
        steps=100,
        cells=(
            Cell(name="a", mitigations=("none",), environment="local"),
            Cell(name="b", mitigations=("tf32_off",), environment="local", trials=7, steps=999),
        ),
        ticket="T-1",
        confound=ConfoundCfg(baseline_cell="a"),
    )
    runner.run_recipe(r, output_dir=tmp_path)
    reqs = [c.args[0] for c in patched_run_trials.call_args_list]
    assert (reqs[0].trials, reqs[0].steps) == (2, 100)
    assert (reqs[1].trials, reqs[1].steps) == (7, 999)


def test_inline_docker_sidecar_written_and_passed_to_run_trials(
    tmp_path, patched_env, patched_run_trials
):
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="image:rocm/pytorch:nightly",
        trials=1,
        steps=10,
        ticket="INL-1",
    )
    run_dir = runner.run_recipe(r, output_dir=tmp_path, timestamp="2026-03-03T03-03-03")
    sidecar = run_dir / "inline_environments.sidecar.json"
    assert sidecar.exists()
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    assert sidecar in req.sidecar_files
    assert req.environment.startswith("_inline_")


def test_extra_sidecar_files_threaded_to_run_trials(tmp_path, patched_env, patched_run_trials):
    extra = tmp_path / "custom.json"
    extra.write_text('{"version": 1}', encoding="utf-8")
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="local",
        trials=1,
        steps=10,
        ticket="X-1",
    )
    runner.run_recipe(
        r,
        output_dir=tmp_path,
        extra_sidecar_files=(extra,),
    )
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    assert extra in req.sidecar_files


def test_dry_run_does_not_call_run_trials(tmp_path, patched_env, patched_run_trials):
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off",
        environment_axis="local",
        trials=1,
        steps=10,
    )
    runner.run_recipe(r, output_dir=tmp_path, dry_run=True)
    patched_run_trials.assert_not_called()


# ---- CLI end-to-end smoke ------------------------------------------------


def test_cli_flag_mode_smoke(tmp_path, patched_env, patched_run_trials):
    cli = CliRunner()
    result = cli.invoke(
        triage,
        [
            "run",
            "--mode",
            "matrix",
            "--workload",
            "fsdp",
            "--mitigation-axis",
            "none,tf32_off",
            "--environment-axis",
            "local",
            "--trials",
            "2",
            "--steps",
            "10",
            "--ticket",
            "CLI-1",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert patched_run_trials.call_count == 2


def test_cli_recipe_mode_smoke(tmp_path, patched_env, patched_run_trials):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
ticket: CLI-R-1
workload: fsdp
trials: 2
steps: 10
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
  - name: tf32-local
    mitigations: [tf32_off]
    environment: local
""",
        encoding="utf-8",
    )
    cli = CliRunner()
    result = cli.invoke(
        triage,
        [
            "run",
            "--recipe",
            str(recipe),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert patched_run_trials.call_count == 2


def test_cli_recipe_mode_dry_run(tmp_path, patched_env, patched_run_trials):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
ticket: CLI-R-2
workload: fsdp
trials: 1
steps: 5
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
""",
        encoding="utf-8",
    )
    cli = CliRunner()
    result = cli.invoke(
        triage,
        ["run", "--recipe", str(recipe), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "baseline-local" in result.output
    patched_run_trials.assert_not_called()


def test_cli_rejects_mixing_recipe_with_flag_mode_args(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
workload: fsdp
trials: 1
steps: 5
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
""",
        encoding="utf-8",
    )
    cli = CliRunner()
    result = cli.invoke(
        triage,
        [
            "run",
            "--recipe",
            str(recipe),
            "--workload",
            "other",
        ],
    )
    assert result.exit_code != 0
    assert "conflicts" in result.output


def test_cli_flag_mode_requires_workload(tmp_path):
    cli = CliRunner()
    result = cli.invoke(
        triage,
        [
            "run",
            "--mode",
            "matrix",
            "--mitigation-axis",
            "none",
            "--environment-axis",
            "local",
            "--trials",
            "1",
            "--steps",
            "10",
        ],
    )
    assert result.exit_code != 0
    assert "--workload" in result.output


def test_cli_list_mitigations():
    cli = CliRunner()
    result = cli.invoke(triage, ["list-mitigations"])
    assert result.exit_code == 0, result.output
    assert "tf32_off" in result.output
    assert "aorta" in result.output  # source_package column


def test_cli_list_environments():
    cli = CliRunner()
    result = cli.invoke(triage, ["list-environments"])
    assert result.exit_code == 0, result.output
    assert "local" in result.output
