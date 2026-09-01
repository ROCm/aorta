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

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import aorta.triage.runner as runner
from aorta.cli.triage import triage
from aorta.instrumentation.environment import EnvSnapshot
from aorta.run import RunRequest, TrialResult
from aorta.triage.recipe import build_recipe_from_flags

# ---- Fixtures -------------------------------------------------------------


class _FakeTrial:
    def __init__(self):
        self.exit_status = "ok"
        self.wall_clock_sec = 1.0
        self.result = {"passed": True, "step_times_ms": [100.0]}


def _clean_snapshot() -> EnvSnapshot:
    """Minimal non-partial EnvSnapshot for test isolation.

    Keep this in sync with the ``EnvSnapshot`` dataclass in
    ``aorta.instrumentation.environment``: env-probe v1.1 (PR #161)
    expanded the schema with rocblas / composable_kernel / tensile /
    triton / fbgemm / aiter / aotriton / miopen / rccl / gpu_arch /
    host / pytorch_build blocks.  We zero them out here so the
    triage runner sees a well-formed snapshot without tying the
    fixture to any host state.
    """
    return EnvSnapshot(
        schema_version="1.1",
        captured_at="2026-04-28T14:12:03Z",
        system_health=None,
        rocm={},
        hip={},
        hipblaslt={},
        rocblas={},
        composable_kernel={},
        tensile={},
        triton={},
        fbgemm={},
        aiter={},
        aotriton={},
        miopen={},
        rccl={},
        gpu_arch={},
        runtime_context={},
        host={},
        docker=None,
        env_vars={},
        python_version="3.11.0",
        pytorch_version=None,
        pytorch_build={},
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
        assert "import subprocess" not in text, (
            f"{py} imports subprocess; B2 must drive B1 via the Python API only."
        )
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


def test_workload_config_recipe_scope_reaches_dispatcher(
    tmp_path, patched_env, patched_run_trials
):
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(Cell(name="a", mitigations=("none",), environment="local"),),
        ticket="WC-1",
        confound=ConfoundCfg(baseline_cell="a"),
        workload_config={"shampoo_api": "new"},
    )
    runner.run_recipe(r, output_dir=tmp_path)
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    assert req.config_overrides == {"shampoo_api": "new"}


def test_workload_config_cell_overrides_recipe_and_merges(
    tmp_path, patched_env, patched_run_trials
):
    """Cell-scope wins on key collision; non-collision keys union (B2.2)."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(
            Cell(name="a", mitigations=("none",), environment="local"),
            Cell(
                name="b",
                mitigations=("none",),
                environment="local",
                workload_config={"shampoo_api": "old", "batch_size": 64},
            ),
        ),
        ticket="WC-2",
        confound=ConfoundCfg(baseline_cell="a"),
        workload_config={"shampoo_api": "new", "warmup": 5},
    )
    runner.run_recipe(r, output_dir=tmp_path)
    reqs = [c.args[0] for c in patched_run_trials.call_args_list]
    assert reqs[0].config_overrides == {"shampoo_api": "new", "warmup": 5}
    # Cell b: shampoo_api collides -> cell wins ('old'); warmup carries through
    # from recipe; batch_size unions in from cell.
    assert reqs[1].config_overrides == {
        "shampoo_api": "old",
        "warmup": 5,
        "batch_size": 64,
    }


def test_workload_config_absent_yields_empty_overrides(
    tmp_path, patched_env, patched_run_trials
):
    """Recipes without workload_config behave byte-equivalent to today."""
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="local",
        trials=1,
        steps=10,
    )
    runner.run_recipe(r, output_dir=tmp_path)
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    assert req.config_overrides == {}


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
    run_dir = runner.run_recipe(
        r,
        output_dir=tmp_path,
        extra_sidecar_files=(extra,),
    )
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    # The runner snapshots operator sidecars into <run_dir>/sidecars/<basename>
    # FIRST and uses that copy as the resolver source -- so what's executed and
    # what's archived for replay are byte-identical. Pin both halves of that
    # contract: the snapshot exists, and run_trials sees the snapshot path
    # (not the original).
    archived = run_dir / "sidecars" / "custom.json"
    assert archived.exists()
    assert archived in req.sidecar_files
    assert extra not in req.sidecar_files


def test_recipe_sidecar_files_alone_drives_run_trials(tmp_path, patched_env, patched_run_trials):
    """Programmatic ``load_recipe(... sidecar_files=...) -> run_recipe(recipe)``
    must reach run_trials with the sidecar plumbed in -- no need to also
    pass ``extra_sidecar_files`` at the runner.

    Pin the round-6 fix from the runner side: the per-cell ``RunRequest``
    that B1 receives carries the archived sidecar even when the runner was
    called with no ``extra_sidecar_files=``.
    """
    sidecar = tmp_path / "ops.sidecar.json"
    sidecar.write_text(
        '{"version": 1, "mitigations": {"my_local_mit": {"FOO": "BAR"}}}',
        encoding="utf-8",
    )
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="my_local_mit",
        environment_axis="local",
        trials=1,
        steps=10,
        ticket="X-1",
        sidecar_files=(sidecar,),
    )
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    archived = run_dir / "sidecars" / "ops.sidecar.json"
    assert archived.exists()
    assert archived in req.sidecar_files


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


def test_cli_collect_override_filters_recipe_collect_options(
    tmp_path, patched_env, patched_run_trials
):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
ticket: CLI-COLLECT-1
workload: fsdp
trials: 1
steps: 10
collect:
  layer_numerics:
    NANLOG_SAMPLE_EVERY: "1"
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
            "--collect",
            "rocprof",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    req: RunRequest = patched_run_trials.call_args.args[0]
    assert req.collect == ("rocprof",)
    assert req.collect_options == {}


def test_cli_collect_override_clears_cell_collect_overrides(
    tmp_path, patched_env, patched_run_trials
):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
ticket: CLI-COLLECT-2
workload: fsdp
trials: 1
steps: 10
collect: [rocprof]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect: []
  - name: tf32-local
    mitigations: [tf32_off]
    environment: local
    collect: [rocprof]
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
            "--collect",
            "layer_numerics",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    reqs = [call.args[0] for call in patched_run_trials.call_args_list]
    assert [req.collect for req in reqs] == [("layer_numerics",), ("layer_numerics",)]
    assert [req.collect_options for req in reqs] == [{}, {}]


def test_cli_collect_override_keeps_surviving_options_and_revalidates(
    tmp_path, patched_env, patched_run_trials
):
    """The CLI override validates against the options that SURVIVE it.

    ``rocprof`` + ``proton`` is only unrunnable on a queue-intercepting Proton
    backend, so restating both names against a recipe that pins the
    instrumentation backend has to be accepted -- checking the bare name list
    would test them against Proton's default backend and invent a conflict.
    """
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
ticket: CLI-COLLECT-3
workload: fsdp
trials: 1
steps: 10
collect:
  rocprof:
    trace: "kernel,hip"
  proton:
    backend: "instrumentation"
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        triage,
        [
            "run",
            "--recipe",
            str(recipe),
            "--collect",
            "rocprof,proton",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    req: RunRequest = patched_run_trials.call_args.args[0]
    assert req.collect == ("rocprof", "proton")
    assert req.collect_options == {
        "rocprof": {"trace": "kernel,hip"},
        "proton": {"backend": "instrumentation"},
    }


def test_cli_collect_override_rejects_a_conflicting_pair(tmp_path, patched_env, patched_run_trials):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
ticket: CLI-COLLECT-4
workload: fsdp
trials: 1
steps: 10
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        triage,
        [
            "run",
            "--recipe",
            str(recipe),
            "--collect",
            "rocprof,proton",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "queue interceptor" in result.output
    assert patched_run_trials.call_count == 0


def test_cli_exits_nonzero_when_baseline_did_not_run_but_writes_matrix(
    tmp_path, patched_env, monkeypatch
):
    """When the explicit baseline_cell collapses to did_not_run, the CLI
    must exit non-zero (so CI/scripts catch it) AND still write the
    matrix.md / matrix.json artifacts (so the operator can inspect what
    happened). Round-2 of the user's smoke-test feedback on PR #175:
    raising RecipeCellError before writing artifacts left the operator
    with nothing to look at; the soft-fallback path keeps both signals.
    """
    from types import SimpleNamespace

    def setup_crash(request):
        # Mimics recom_repro nan-repro: workload_failed exit, no
        # step_times_ms, zero elapsed_sec -> platform inference flags
        # both trials as did_not_run.
        return [
            SimpleNamespace(
                exit_status="workload_failed",
                wall_clock_sec=3.5,
                result={"passed": False, "step_times_ms": [], "elapsed_sec": 0.0},
            )
            for _ in range(2)
        ]

    import aorta.triage.runner as runner_mod

    monkeypatch.setattr(runner_mod, "run_trials", setup_crash)

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """\
schema_version: 1
ticket: CLI-INC-1
workload: fsdp
trials: 2
steps: 50
confound:
  baseline_cell: baseline-local
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
    out_dir = tmp_path / "out"
    result = cli.invoke(
        triage,
        ["run", "--recipe", str(recipe), "--output-dir", str(out_dir)],
    )
    # Loud failure signal for CI / scripts.
    assert result.exit_code != 0, result.output
    assert "explicit baseline_cell 'baseline-local'" in result.output
    # Artifacts present for the operator to inspect.
    assert "Wrote matrix to" in result.output
    matrix_files = list(out_dir.rglob("matrix.md"))
    assert len(matrix_files) == 1
    md = matrix_files[0].read_text()
    assert "> [!WARNING]" in md
    assert "did_not_run" in md


def _write_two_cell_recipe(tmp_path: Path, ticket: str) -> Path:
    """A baseline cell + one other cell, both local, 1 trial / 50 steps."""
    recipe = tmp_path / f"{ticket}.yaml"
    recipe.write_text(
        f"""\
schema_version: 1
ticket: {ticket}
workload: fsdp
trials: 1
steps: 50
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
  - name: bad-local
    mitigations: [tf32_off]
    environment: local
""",
        encoding="utf-8",
    )
    return recipe


def _ok_trials():
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            exit_status="ok",
            wall_clock_sec=1.0,
            result={"passed": True, "step_times_ms": [100.0], "elapsed_sec": 1.0},
        )
    ]


def _did_not_run_trials():
    """workload_failed with no step times + zero elapsed -> platform infers did_not_run."""
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            exit_status="workload_failed",
            wall_clock_sec=3.5,
            result={"passed": False, "step_times_ms": [], "elapsed_sec": 0.0},
        )
    ]


def test_strict_exits_nonzero_when_a_cell_did_not_run(tmp_path, patched_env, monkeypatch):
    """--strict must catch a cell that never ran (setup crash -> did_not_run),
    while the same recipe without --strict tolerates it (exit 0). The matrix is
    written either way so the operator can inspect the cause.
    """
    import aorta.triage.runner as runner_mod

    def per_cell(request):
        return _ok_trials() if request.mitigations == ("none",) else _did_not_run_trials()

    monkeypatch.setattr(runner_mod, "run_trials", per_cell)
    recipe = _write_two_cell_recipe(tmp_path, "STRICT-DNR")
    cli = CliRunner()
    out = tmp_path / "out"

    default = cli.invoke(triage, ["run", "--recipe", str(recipe), "--output-dir", str(out)])
    assert default.exit_code == 0, default.output

    strict = cli.invoke(
        triage, ["run", "--recipe", str(recipe), "--output-dir", str(out), "--strict"]
    )
    assert strict.exit_code != 0, strict.output
    assert "strict mode" in strict.output
    assert "1 of 2 cell(s)" in strict.output  # only the bad cell, not the baseline
    assert "bad-local" in strict.output
    assert "Wrote matrix to" in strict.output


def test_strict_exits_nonzero_when_a_cell_errored(tmp_path, patched_env, monkeypatch):
    """--strict must catch a whole-cell error (run_trials raised)."""
    import aorta.triage.runner as runner_mod

    def per_cell(request):
        if request.mitigations == ("none",):
            return _ok_trials()
        raise RuntimeError("cell blew up inside run_trials")

    monkeypatch.setattr(runner_mod, "run_trials", per_cell)
    recipe = _write_two_cell_recipe(tmp_path, "STRICT-ERR")
    cli = CliRunner()
    out = tmp_path / "out"

    strict = cli.invoke(
        triage, ["run", "--recipe", str(recipe), "--output-dir", str(out), "--strict"]
    )
    assert strict.exit_code != 0, strict.output
    assert "bad-local" in strict.output
    assert "Wrote matrix to" in strict.output


def test_strict_tolerates_a_real_bug_repro(tmp_path, patched_env, monkeypatch):
    """A cell that RAN but failed (a genuine bug repro) is an expected matrix
    outcome and must NOT trip --strict.
    """
    from types import SimpleNamespace

    import aorta.triage.runner as runner_mod

    def per_cell(request):
        if request.mitigations == ("none",):
            return _ok_trials()
        return [
            SimpleNamespace(
                exit_status="workload_failed",
                wall_clock_sec=2.0,
                result={
                    "passed": False,
                    "step_times_ms": [120.0],
                    "elapsed_sec": 2.0,
                    "main_work_started": True,
                },
            )
        ]

    monkeypatch.setattr(runner_mod, "run_trials", per_cell)
    recipe = _write_two_cell_recipe(tmp_path, "STRICT-REPRO")
    cli = CliRunner()
    out = tmp_path / "out"

    strict = cli.invoke(
        triage, ["run", "--recipe", str(recipe), "--output-dir", str(out), "--strict"]
    )
    assert strict.exit_code == 0, strict.output


def test_run_recipe_strict_raises_matrix_strict_error(tmp_path, patched_env, monkeypatch):
    """Programmatic run_recipe(strict=True) raises MatrixStrictError naming the
    offending cell(s), with artifacts already written.
    """
    import aorta.triage.runner as runner_mod
    from aorta.triage.runner import MatrixStrictError

    def per_cell(request):
        return _ok_trials() if request.mitigations == ("none",) else _did_not_run_trials()

    monkeypatch.setattr(runner_mod, "run_trials", per_cell)
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off",
        environment_axis="local",
        trials=1,
        steps=50,
        ticket="STRICT-API",
    )
    with pytest.raises(MatrixStrictError) as excinfo:
        runner.run_recipe(
            r, output_dir=tmp_path, timestamp="2026-01-01T00-00-00", strict=True
        )
    assert excinfo.value.cells == ["tf32_off-local"]
    # The message points at the concrete slugified per-cell artifact path
    # (cells/<safe_slug(cell)>/<workload>/trial_*.json), not a <cell>/<workload>
    # placeholder, so an operator can copy/paste straight to the trials (#274).
    msg = str(excinfo.value)
    assert "cells/tf32_off-local/fsdp/trial_*.json" in msg
    assert "<cell>" not in msg
    assert (excinfo.value.run_dir / "matrix.json").exists()


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


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--workload", "other"],
        ["--mitigation-axis", "tf32_off"],
        ["--environment-axis", "local"],
        ["--trials", "5"],
        ["--steps", "200"],
        ["--ticket", "OTHER-1"],
        ["--baseline-cell", "different-cell"],
        ["--confound-threshold", "1.5"],
    ],
    ids=[
        "workload",
        "mitigation-axis",
        "environment-axis",
        "trials",
        "steps",
        "ticket",
        "baseline-cell",
        "confound-threshold",
    ],
)
def test_cli_recipe_mode_rejects_every_flag_mode_knob(tmp_path, extra_args):
    """All flags that affect recipe content must be rejected in recipe mode (issue #160 c1)."""
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
    result = cli.invoke(triage, ["run", "--recipe", str(recipe), *extra_args])
    assert result.exit_code != 0
    assert "conflicts" in result.output
    assert extra_args[0] in result.output


def test_cli_recipe_mode_allows_runner_only_flags(tmp_path, patched_env, patched_run_trials):
    """--output-dir, --dry-run, --mode, --mitigations-file are runner-level, not recipe content."""
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
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output


def test_cli_flag_mode_rejects_non_positive_trials(tmp_path):
    """Flag-mode trials=0 rejected at recipe build, not deep in run_trials (issue #160 c6)."""
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
            "none",
            "--environment-axis",
            "local",
            "--trials",
            "0",
            "--steps",
            "10",
        ],
    )
    assert result.exit_code != 0
    assert "trials" in result.output.lower()


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


def test_cli_list_environments_shows_baseline_env(tmp_path):
    sidecar = tmp_path / "environment-env.json"
    sidecar.write_text(
        '{"version": 1, "environments": {'
        '"env-list-test": {"env": {"BASELINE_FLAG": "enabled"}}'
        "}}",
        encoding="utf-8",
    )

    cli = CliRunner()
    result = cli.invoke(
        triage,
        ["list-environments", "--mitigations-file", str(sidecar)],
    )

    assert result.exit_code == 0, result.output
    assert "ENV" in result.output
    assert "BASELINE_FLAG=enabled" in result.output


def test_cli_list_mitigations_wraps_registry_error_in_click_exception(tmp_path):
    """Malformed --mitigations-file -> clean ClickException, not a Python traceback.

    Regression for PR #160 second-round Copilot comment: `triage list-mitigations`
    let `RegistryError` escape uncaught, breaking the one-line-error CLI contract
    that `triage run` and `aorta run` already followed.
    """
    bad = tmp_path / "broken.sidecar.json"
    bad.write_text("not valid json {{{", encoding="utf-8")
    cli = CliRunner()
    result = cli.invoke(triage, ["list-mitigations", "--mitigations-file", str(bad)])
    assert result.exit_code != 0
    # Click renders ClickException as "Error: <msg>" -- pin that shape.
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_cli_list_environments_wraps_registry_error_in_click_exception(tmp_path):
    """Same fail-fast contract as list-mitigations."""
    bad = tmp_path / "broken.sidecar.json"
    bad.write_text("not valid json {{{", encoding="utf-8")
    cli = CliRunner()
    result = cli.invoke(triage, ["list-environments", "--mitigations-file", str(bad)])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "flag_name",
    [
        "--ticket",
        "--baseline-cell",
        "--confound-threshold",
    ],
)
def test_cli_run_help_documents_recipe_mode_rejection(flag_name):
    """Class-wide UX fix: every flag rejected in recipe mode must say so in help.

    Pre-fix only ``--confound-threshold`` mentioned the rejection in its
    help text. ``--ticket`` and ``--baseline-cell`` were also in the
    conflict set -- and ``--baseline-cell`` was the most confusing of the
    three because its summary line ("Override the auto-resolved baseline
    cell") reads like it should override the recipe's
    ``confound.baseline_cell`` too. Pin that every conflicting flag
    advertises the rejection so a user reading ``--help`` doesn't have to
    discover it by trial and error.
    """
    cli = CliRunner()
    result = cli.invoke(triage, ["run", "--help"])
    assert result.exit_code == 0, result.output
    # Find the flag's help block. Click wraps lines, so search by flag name
    # and assert the rejection sentence appears within a reasonable window.
    idx = result.output.find(flag_name)
    assert idx >= 0, f"{flag_name} missing from --help"
    window = result.output[idx : idx + 600]
    assert "rejected" in window, (
        f"{flag_name} help does not advertise the recipe-mode rejection; window was:\n{window!r}"
    )


def test_cli_run_wraps_run_recipe_errors_in_click_exception(tmp_path, patched_env):
    """Recipe-level errors raised from run_recipe (NOT load_recipe) must also
    surface as a one-line ClickException, matching ``aorta run`` and the
    list subcommands.

    Pre-fix: ``triage run`` only wrapped ``load_recipe`` /
    ``build_recipe_from_flags``, so anything raised later -- baseline
    resolution, env-slug collisions, etc. -- escaped as a Python traceback.
    The two flavours of error were the same shape but exited the CLI
    differently depending on which validator caught them.
    """
    recipe = tmp_path / "bad.yaml"
    # Multi-cell recipe with no auto-resolvable baseline -- load_recipe
    # accepts it (baseline resolution is run-time, by design), but
    # _preflight_validate inside run_recipe rejects it.
    recipe.write_text(
        """\
schema_version: 1
workload: fsdp
trials: 1
steps: 5
cells:
  - name: a-local
    mitigations: [tf32_off]
    environment: local
  - name: b-local
    mitigations: [xnack]
    environment: local
""",
        encoding="utf-8",
    )
    cli = CliRunner()
    result = cli.invoke(
        triage,
        ["run", "--recipe", str(recipe), "--output-dir", str(tmp_path / "out")],
    )
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "baseline" in result.output.lower()
    assert "Traceback" not in result.output


def test_cli_run_wraps_trial_worker_error_in_click_exception(
    tmp_path, monkeypatch
):
    import importlib

    from aorta.run._process import TrialWorkerError

    cli_module = importlib.import_module("aorta.cli.triage")

    def fail_worker(*_args, **_kwargs):
        raise TrialWorkerError("worker bootstrap failed")

    monkeypatch.setattr(cli_module, "run_recipe", fail_worker)
    recipe = tmp_path / "worker-error.yaml"
    recipe.write_text(
        """\
schema_version: 1
workload: race
trials: 1
steps: 1
confound:
  baseline_cell: baseline
cells:
  - name: baseline
    mitigations: [none]
    environment: local
""",
        encoding="utf-8",
    )
    result = CliRunner().invoke(triage, ["run", "--recipe", str(recipe)])
    assert result.exit_code != 0
    assert "Error: worker bootstrap failed" in result.output
    assert "Traceback" not in result.output
    assert result.exception is not None
    assert not isinstance(result.exception, TrialWorkerError)


# ---- Distributed rank-0 write gate ---------------------------------------


def _simple_recipe():
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    return Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=5,
        cells=(Cell(name="a", mitigations=("none",), environment="local"),),
        ticket="RANK-1",
        confound=ConfoundCfg(baseline_cell="a"),
    )


def test_rank_zero_writes_matrix(tmp_path, patched_env, patched_run_trials, monkeypatch):
    """RANK=0 (or unset) writes the matrix into the real output tree."""
    monkeypatch.setenv("RANK", "0")
    run_dir = runner.run_recipe(
        _simple_recipe(), output_dir=tmp_path, timestamp="2026-01-01T00-00-00"
    )
    assert (run_dir / "matrix.json").exists()
    assert (run_dir / "matrix.md").exists()
    # The real run dir lives under the requested output tree.
    assert tmp_path in run_dir.parents


def test_nonzero_rank_writes_nothing_to_output_tree(
    tmp_path, patched_env, patched_run_trials, monkeypatch
):
    """A non-zero rank still RUNS every cell (collectives must complete) but
    writes no artifacts into the shared output tree -- preventing the
    duplicate ``<timestamp>-2`` run dir torchrun produced before this gate."""
    out = tmp_path / "out"
    monkeypatch.setenv("RANK", "1")
    runner.run_recipe(_simple_recipe(), output_dir=out, timestamp="2026-01-01T00-00-00")

    # Trials still executed on this rank (the workload's collectives need it).
    assert patched_run_trials.call_count == 1
    # But nothing was written under the output dir -- the scratch run_dir was
    # a TemporaryDirectory that is removed on exit.
    assert not out.exists() or not any(out.rglob("matrix.json"))


def test_nonzero_rank_does_not_collide_with_rank_zero(
    tmp_path, patched_env, patched_run_trials, monkeypatch
):
    """Rank 0 and a non-zero rank pointed at the same output dir + timestamp
    must NOT both create a leaf -- rank 0 owns the real dir, the other goes to
    scratch. Before the gate this produced ``<ts>`` and ``<ts>-2``."""
    out = tmp_path / "out"
    monkeypatch.setenv("RANK", "0")
    runner.run_recipe(_simple_recipe(), output_dir=out, timestamp="2026-01-01T00-00-00")
    monkeypatch.setenv("RANK", "1")
    runner.run_recipe(_simple_recipe(), output_dir=out, timestamp="2026-01-01T00-00-00")

    leaves = list((out / "RANK-1" / "fsdp").iterdir())
    assert len(leaves) == 1
    assert leaves[0].name == "2026-01-01T00-00-00"


# ---- extra_env recipe-level + cell-level merge ----------------------------
#
# The runner merges ``{**recipe.extra_env, **cell.extra_env}`` into
# ``RunRequest.extra_env``, giving the cell-scope the win on key collision.
# This merge is the lowest-level seam of the env-precedence contract that
# the runner owns (above it: the dispatcher's mitigation and Environment.env
# layers; below it: nothing).  Pin all three observable behaviours: recipe
# only, cell only, and collision (cell wins).


def test_extra_env_recipe_scope_reaches_run_request(
    tmp_path, patched_env, patched_run_trials
):
    """A recipe-level ``extra_env`` with no per-cell override reaches every
    cell's ``RunRequest.extra_env`` unchanged."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(Cell(name="a", mitigations=("none",), environment="local"),),
        ticket="EE-1",
        confound=ConfoundCfg(baseline_cell="a"),
        extra_env={"GLOBAL_FLAG": "1", "NANLOG_TRACE": "on"},
    )
    runner.run_recipe(r, output_dir=tmp_path)
    req: RunRequest = patched_run_trials.call_args_list[0].args[0]
    assert req.extra_env == {"GLOBAL_FLAG": "1", "NANLOG_TRACE": "on"}


def test_extra_env_cell_scope_only_reaches_run_request(
    tmp_path, patched_env, patched_run_trials
):
    """A cell-level ``extra_env`` with no recipe-level extra_env reaches only
    that cell's ``RunRequest.extra_env``."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(
            Cell(name="a", mitigations=("none",), environment="local"),
            Cell(
                name="b",
                mitigations=("tf32_off",),
                environment="local",
                extra_env={"CELL_FLAG": "cell_only"},
            ),
        ),
        ticket="EE-2",
        confound=ConfoundCfg(baseline_cell="a"),
    )
    runner.run_recipe(r, output_dir=tmp_path)
    reqs = [c.args[0] for c in patched_run_trials.call_args_list]
    assert reqs[0].extra_env == {}
    assert reqs[1].extra_env == {"CELL_FLAG": "cell_only"}


def test_extra_env_cell_wins_on_collision_with_recipe(
    tmp_path, patched_env, patched_run_trials
):
    """Cell-scope wins on key collision; non-collision keys union (mirrors
    ``test_workload_config_cell_overrides_recipe_and_merges``)."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(
            Cell(name="a", mitigations=("none",), environment="local"),
            Cell(
                name="b",
                mitigations=("tf32_off",),
                environment="local",
                # SHARED_KEY collides; CELL_ONLY is new.
                extra_env={"SHARED_KEY": "cell_wins", "CELL_ONLY": "yes"},
            ),
        ),
        ticket="EE-3",
        confound=ConfoundCfg(baseline_cell="a"),
        # Recipe-level: SHARED_KEY also set, RECIPE_ONLY not in cell.
        extra_env={"SHARED_KEY": "recipe_value", "RECIPE_ONLY": "always"},
    )
    runner.run_recipe(r, output_dir=tmp_path)
    reqs = [c.args[0] for c in patched_run_trials.call_args_list]
    # Cell a has no extra_env; inherits recipe only.
    assert reqs[0].extra_env == {"SHARED_KEY": "recipe_value", "RECIPE_ONLY": "always"}
    # Cell b: SHARED_KEY -> cell wins; RECIPE_ONLY unions in; CELL_ONLY unions in.
    assert reqs[1].extra_env == {
        "SHARED_KEY": "cell_wins",
        "RECIPE_ONLY": "always",
        "CELL_ONLY": "yes",
    }


def test_race_workload_resolves_required_process_isolation(
    tmp_path, patched_env, patched_run_trials
):
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    recipe = Recipe(
        schema_version=1,
        workload="race",
        trials=1,
        steps=1,
        cells=(Cell(name="race", mitigations=("none",), environment="local"),),
        confound=ConfoundCfg(baseline_cell="race"),
    )
    runner.run_recipe(recipe, output_dir=tmp_path)
    request: RunRequest = patched_run_trials.call_args.args[0]
    assert request.trial_isolation == "process"


def test_process_isolation_dry_run_rejects_non_json_config(tmp_path):
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    recipe = Recipe(
        schema_version=1,
        workload="race",
        trials=1,
        steps=1,
        cells=(Cell(name="race", mitigations=("none",), environment="local"),),
        confound=ConfoundCfg(baseline_cell="race"),
        workload_config={"bad": float("nan")},
    )
    with pytest.raises(ValueError, match="non-finite"):
        runner.run_recipe(recipe, output_dir=tmp_path, dry_run=True)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_process_isolation_dry_run_rejects_workload_without_opt_in(
    tmp_path, monkeypatch
):
    from aorta.run.validation import IN_PROCESS_ONLY_POLICY
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    recipe = Recipe(
        schema_version=1,
        workload="legacy",
        trials=1,
        steps=1,
        cells=(Cell(name="legacy", mitigations=("none",), environment="local"),),
        confound=ConfoundCfg(baseline_cell="legacy"),
        trial_isolation="process",
    )
    monkeypatch.setattr(
        runner,
        "get_workload_policy",
        lambda _name: IN_PROCESS_ONLY_POLICY,
    )
    with pytest.raises(ValueError, match="does not support.*process"):
        runner.run_recipe(recipe, output_dir=tmp_path, dry_run=True)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_auto_dry_run_does_not_hide_malformed_workload_policy(
    tmp_path, monkeypatch
):
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    recipe = Recipe(
        schema_version=1,
        workload="broken",
        trials=1,
        steps=1,
        cells=(Cell(name="broken", mitigations=("none",), environment="local"),),
        confound=ConfoundCfg(baseline_cell="broken"),
    )

    def malformed_policy(_name):
        raise ValueError("unsupported isolation policy target")

    monkeypatch.setattr(runner, "get_workload_policy", malformed_policy)
    with pytest.raises(ValueError, match="unsupported isolation policy"):
        runner.run_recipe(recipe, output_dir=tmp_path, dry_run=True)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_distributed_process_worker_launch_error_is_fatal(
    tmp_path, patched_env, monkeypatch
):
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    recipe = Recipe(
        schema_version=1,
        workload="race",
        trials=1,
        steps=1,
        cells=(Cell(name="race", mitigations=("none",), environment="local"),),
        confound=ConfoundCfg(baseline_cell="race"),
    )
    monkeypatch.setenv("WORLD_SIZE", "2")

    def spawn_failure(_request):
        raise OSError("spawn failed")

    monkeypatch.setattr(runner, "run_trials", spawn_failure)
    with pytest.raises(OSError, match="spawn failed"):
        runner.run_recipe(recipe, output_dir=tmp_path)


def test_resolved_env_lookup_failure_warns_without_echoing_error_text(monkeypatch, caplog):
    """A fail-soft baseline lookup must not silently weaken the audit bundle."""
    from aorta.registry import RegistryError

    def fail_lookup(*args, **kwargs):
        raise RegistryError("secret-value-must-not-be-logged")

    monkeypatch.setattr(runner, "get_environment", fail_lookup)

    with caplog.at_level("WARNING", logger="aorta.triage.runner"):
        resolved = runner._resolve_cell_env_vars(
            (),
            {"EXTRA": "kept"},
            None,
            environment="broken-env",
        )

    assert resolved == {"EXTRA": "kept"}
    warning = "\n".join(record.getMessage() for record in caplog.records)
    assert "broken-env" in warning
    assert "RegistryError" in warning
    assert "omitting its baseline Environment.env layer" in warning
    assert "secret-value-must-not-be-logged" not in warning


def _persisted_trial(trial_id: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        workload="_subprocess",
        execution_env={},
        mitigations_applied=("none",),
        config={},
        env={},
        result={
            "passed": True,
            "failure_count": 0,
            "first_failure_iteration": None,
            "failure_details": [],
            "total_iterations": 1,
            "step_times_ms": [1.0],
            "elapsed_sec": 0.1,
            "metrics": {},
            "main_work_started": True,
            "executed_iterations": 1,
            "configured_iterations": 1,
        },
        wall_clock_sec=1.0,
        exit_status="ok",
        request_fingerprint="a" * 64,
    )


def test_resume_hydration_rejects_filename_body_identity_mismatch(tmp_path):
    path = tmp_path / "trial_d0_m0_t0.json"
    path.write_text(
        json.dumps(_persisted_trial("_subprocess_d0_m0_t1").to_dict()),
        encoding="utf-8",
    )

    hydrated = runner._hydrate_trials_by_index(
        [str(path)],
        expected_workload="_subprocess",
    )

    assert hydrated == {}


def test_resume_hydration_rejects_foreign_cell_coordinates(tmp_path):
    path = tmp_path / "trial_d9_m9_t0.json"
    path.write_text(
        json.dumps(_persisted_trial("_subprocess_d9_m9_t0").to_dict()),
        encoding="utf-8",
    )

    hydrated = runner._hydrate_trials_by_index(
        [str(path)],
        expected_workload="_subprocess",
    )

    assert hydrated == {}


def test_resume_hydration_rejects_duplicate_trial_indices(tmp_path):
    paths = [
        tmp_path / "trial_d0_m0_t0.json",
        tmp_path / "trial_0.json",
    ]
    body = json.dumps(_persisted_trial("_subprocess_d0_m0_t0").to_dict())
    for path in paths:
        path.write_text(body, encoding="utf-8")

    hydrated = runner._hydrate_trials_by_index(
        [str(path) for path in paths],
        expected_workload="_subprocess",
    )

    assert hydrated == {}


def test_resume_hydration_rejects_request_fingerprint_mismatch(tmp_path):
    path = tmp_path / "trial_d0_m0_t0.json"
    path.write_text(
        json.dumps(_persisted_trial("_subprocess_d0_m0_t0").to_dict()),
        encoding="utf-8",
    )

    hydrated = runner._hydrate_trials_by_index(
        [str(path)],
        expected_workload="_subprocess",
        expected_request_fingerprint="b" * 64,
    )

    assert hydrated == {}


# ---- Layer 2: DT_RPATH vs LD_LIBRARY_PATH (issue #413) --------------------
#
# The design property under test is precision, not detection. A warning that
# fires on every ROCm 10 run -- or on every run that sets LD_LIBRARY_PATH --
# trains operators to scroll past it, and it would then be scrolled past on
# the one run where a substitution was silently discarded. So the negative
# cases below carry as much weight as the positive one.


def _substitution_recipe(ld_library_path: str | None):
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    extra_env = {}
    if ld_library_path is not None:
        extra_env["LD_LIBRARY_PATH"] = ld_library_path
    return Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=1,
        cells=(
            Cell(
                name="patched-hipblaslt",
                mitigations=("none",),
                environment="local",
                extra_env=extra_env,
            ),
        ),
        confound=ConfoundCfg(baseline_cell="patched-hipblaslt"),
    )


def _matrix_warnings(tmp_path: Path) -> list[str]:
    matrix = next(tmp_path.rglob("matrix.json"))
    return json.loads(matrix.read_text(encoding="utf-8"))["warnings"]


def _snapshot_with_rocm_rpath(value: bool | None) -> EnvSnapshot:
    """A snapshot whose ``library_linkage`` reports *value*.

    Fed through ``collect_env`` rather than injected at the predicate, so
    the tests exercise the real path: the warning must read the linkage
    block out of the environment's own ``env.json``, which is the only
    place a truthful per-environment answer exists.
    """
    snapshot = _clean_snapshot()
    return dataclasses.replace(
        snapshot,
        library_linkage={
            **snapshot.library_linkage,
            "status": "ok" if value is not None else "unreadable",
            "rocm_rpath": value,
        },
    )


@pytest.fixture
def rpath_stack(monkeypatch, patched_env):
    """The local env's own probe reports DT_RPATH on its ROCm libraries."""
    monkeypatch.setattr(
        runner,
        "collect_env",
        MagicMock(return_value=_snapshot_with_rocm_rpath(True)),
    )


@pytest.fixture
def runpath_stack(monkeypatch, patched_env):
    """It reports DT_RUNPATH -- i.e. every ROCm 7.x host."""
    monkeypatch.setattr(
        runner,
        "collect_env",
        MagicMock(return_value=_snapshot_with_rocm_rpath(False)),
    )


def test_warns_when_ld_library_path_meets_an_rpath_stack(
    tmp_path, patched_env, patched_run_trials, rpath_stack, caplog
):
    """Both conditions hold: the substitution may be silently discarded."""
    with caplog.at_level("WARNING", logger="aorta.triage.runner"):
        runner.run_recipe(
            _substitution_recipe("/opt/patched/lib"),
            output_dir=tmp_path,
            timestamp="2026-01-01T00-00-00",
        )

    warnings = _matrix_warnings(tmp_path)
    hit = next(w for w in warnings if "DT_RPATH" in w)
    # Actionable: names the cell, the mechanism, and the way out.
    assert "patched-hipblaslt" in hit
    assert "LD_LIBRARY_PATH" in hit
    assert "BEFORE" in hit
    assert "LD_PRELOAD" in hit
    assert "recipes/hrx/" in hit
    # And it reaches a console reader, not only the artifact.
    assert "DT_RPATH" in "\n".join(r.getMessage() for r in caplog.records)


def test_silent_when_ld_library_path_meets_a_runpath_stack(
    tmp_path, patched_env, patched_run_trials, runpath_stack
):
    """Substitution on ROCm 7.x works. Warning here would be crying wolf.

    This is the case that rules out a version- or "is LD_LIBRARY_PATH set"
    trigger: it describes essentially every substitution run performed to
    date, all of which behaved exactly as the operator intended.
    """
    runner.run_recipe(
        _substitution_recipe("/opt/patched/lib"),
        output_dir=tmp_path,
        timestamp="2026-01-01T00-00-00",
    )
    assert not [w for w in _matrix_warnings(tmp_path) if "DT_RPATH" in w]


def test_silent_when_rpath_stack_has_no_substitution(
    tmp_path, patched_env, patched_run_trials, rpath_stack
):
    """A ROCm 10 run that substitutes nothing has nothing to lose.

    Rules out triggering on the ELF fact alone, which would fire on 100%
    of ROCm 10 runs and say nothing about any of them.
    """
    runner.run_recipe(
        _substitution_recipe(None),
        output_dir=tmp_path,
        timestamp="2026-01-01T00-00-00",
    )
    assert not [w for w in _matrix_warnings(tmp_path) if "DT_RPATH" in w]


def test_silent_when_the_link_tags_could_not_be_read(
    tmp_path, patched_env, patched_run_trials, monkeypatch
):
    """``rocm_rpath is None`` is "we don't know", which is not grounds to warn.

    Guards the tempting ``if not ...use_rpath()`` spelling, under which
    ``None`` would read as False (silent, correct here) but the inverse
    check elsewhere would read as True.
    """
    monkeypatch.setattr(
        runner,
        "collect_env",
        MagicMock(return_value=_snapshot_with_rocm_rpath(None)),
    )
    runner.run_recipe(
        _substitution_recipe("/opt/patched/lib"),
        output_dir=tmp_path,
        timestamp="2026-01-01T00-00-00",
    )
    assert not [w for w in _matrix_warnings(tmp_path) if "DT_RPATH" in w]


def test_operator_ld_library_path_reaches_the_child_unmodified(
    tmp_path, patched_env, patched_run_trials, rpath_stack
):
    """Layer 2 warns. It must never edit the environment under test.

    Prepending the stock ROCm directories would outrank nothing (DT_RPATH
    is consulted first regardless) while hijacking the very substitution
    the operator asked for -- turning a loud, diagnosable problem into a
    quiet, wrong one.
    """
    runner.run_recipe(
        _substitution_recipe("/opt/patched/lib"),
        output_dir=tmp_path,
        timestamp="2026-01-01T00-00-00",
    )
    request = patched_run_trials.call_args_list[0].args[0]
    assert request.extra_env["LD_LIBRARY_PATH"] == "/opt/patched/lib"
    # Byte-for-byte: no appended stock dir, no reordering, no separator fixup.
    assert "/opt/rocm" not in request.extra_env["LD_LIBRARY_PATH"]

    matrix = next(tmp_path.rglob("matrix.json"))
    cell = json.loads(matrix.read_text(encoding="utf-8"))["cells"][0]
    assert cell["resolved_env_vars"]["LD_LIBRARY_PATH"] == "/opt/patched/lib"


def test_warning_predicate_is_pure_and_does_not_touch_the_bundle():
    """The detector inspects; the caller decides what to do with the string."""
    bundle = {"LD_LIBRARY_PATH": "/opt/patched/lib", "HSA_XNACK": "1"}
    before = dict(bundle)
    message = runner._rpath_substitution_warning("cell-a", bundle, True)
    assert message is not None
    assert bundle == before


def test_silent_when_ld_library_path_is_empty():
    """An empty LD_LIBRARY_PATH contributes no directory to search.

    Set-but-empty is not a substitution attempt, so treating it as one
    would fire the warning on a run with nothing to lose.
    """
    assert (
        runner._rpath_substitution_warning("cell-a", {"LD_LIBRARY_PATH": ""}, True)
        is None
    )


# ---- Layer 2, per-environment sourcing (issue #413 follow-up) -------------
#
# The linkage fact belongs to the environment the cell RUNS in, not to the
# runner process. This module already refuses a runner-process collect_env()
# for isolated envs because host state under a docker label is misleading;
# a warning derived from that same host state breaks the rule less visibly.
# A ROCm 10 container launched from this ROCm 7 host is the exact case, and
# it is the one the whole feature exists for.


def _isolated_substitution_recipe(ld_library_path: str, image: str = "rocm/x:10"):
    from aorta.triage.recipe import Cell, ConfoundCfg, InlineEnv, Recipe

    env = InlineEnv(name="_inline_rocm10", docker=image)
    return Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=1,
        cells=(
            Cell(
                name="patched-in-container",
                mitigations=("none",),
                environment=env.name,
                extra_env={"LD_LIBRARY_PATH": ld_library_path},
            ),
        ),
        confound=ConfoundCfg(baseline_cell="patched-in-container"),
        inline_environments=(env,),
    )


def _wrapper_writing(rocm_rpath: bool | None):
    """A ``run_trials`` stand-in that plays the wrapper's half of the contract.

    Writes an in-container snapshot to the reserved ``_aorta_env_probe``
    output path, which is how an isolated env's real linkage reading gets
    back to the runner.
    """

    def fake_run_trials(request):
        probe = getattr(request, "env_probe", None)
        if probe:
            out = Path(probe["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(_snapshot_with_rocm_rpath(rocm_rpath).to_dict()),
                encoding="utf-8",
            )
        return [_FakeTrial(), _FakeTrial()]

    return MagicMock(side_effect=fake_run_trials)


def test_isolated_env_warning_comes_from_the_container_not_the_host(
    tmp_path, monkeypatch, runpath_stack
):
    """The regression: ROCm 10 container launched from a ROCm 7 host.

    ``collect_env`` in the runner process reports DT_RUNPATH -- true of the
    host, irrelevant to the cell -- while the container's own probe reports
    DT_RPATH. Sourcing the warning from the host would stay silent on
    precisely the substitution that gets silently discarded.
    """
    monkeypatch.setattr(runner, "run_trials", _wrapper_writing(True))
    runner.run_recipe(
        _isolated_substitution_recipe("/opt/patched/lib"),
        output_dir=tmp_path,
        timestamp="2026-01-01T00-00-00",
    )
    hits = [w for w in _matrix_warnings(tmp_path) if "DT_RPATH" in w]
    assert len(hits) == 1
    assert "patched-in-container" in hits[0]


def test_isolated_env_does_not_inherit_a_host_rpath_reading(
    tmp_path, monkeypatch, rpath_stack
):
    """The mirror image, and the reason this is not just a missed warning.

    Host libraries carry DT_RPATH; the container's do not. A host-sourced
    warning would fire on a cell whose substitution works perfectly, which
    is the crying-wolf the two-condition trigger exists to prevent.
    """
    monkeypatch.setattr(runner, "run_trials", _wrapper_writing(False))
    runner.run_recipe(
        _isolated_substitution_recipe("/opt/patched/lib"),
        output_dir=tmp_path,
        timestamp="2026-01-01T00-00-00",
    )
    assert not [w for w in _matrix_warnings(tmp_path) if "DT_RPATH" in w]


def test_isolated_env_with_no_container_snapshot_stays_silent(
    tmp_path, monkeypatch, rpath_stack, patched_run_trials
):
    """No per-environment truth available means no warning at all.

    The wrapper never opted into the probe contract, so nothing in this run
    knows what the container's libraries carry. Falling back to the host
    reading would be a confident claim about a stack we never looked at,
    and a warning sourced from the wrong environment is worse than none.
    """
    runner.run_recipe(
        _isolated_substitution_recipe("/opt/patched/lib"),
        output_dir=tmp_path,
        timestamp="2026-01-01T00-00-00",
    )
    assert not [w for w in _matrix_warnings(tmp_path) if "DT_RPATH" in w]


def test_each_environment_in_a_matrix_gets_its_own_reading(
    tmp_path, monkeypatch, runpath_stack
):
    """A process-global cache would answer for at most one of these cells.

    Two cells, two environments, one substitution attempt each: a local
    ROCm 7 env where the override works, and a ROCm 10 container where it
    does not. Exactly one warning, naming the container cell.
    """
    from aorta.triage.recipe import Cell, ConfoundCfg, InlineEnv, Recipe

    env = InlineEnv(name="_inline_rocm10", docker="rocm/x:10")
    recipe = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=1,
        cells=(
            Cell(
                name="host-substitution",
                mitigations=("none",),
                environment="local",
                extra_env={"LD_LIBRARY_PATH": "/opt/patched/lib"},
            ),
            Cell(
                name="container-substitution",
                mitigations=("none",),
                environment=env.name,
                extra_env={"LD_LIBRARY_PATH": "/opt/patched/lib"},
            ),
        ),
        confound=ConfoundCfg(baseline_cell="host-substitution"),
        inline_environments=(env,),
    )
    monkeypatch.setattr(runner, "run_trials", _wrapper_writing(True))

    runner.run_recipe(
        recipe, output_dir=tmp_path, timestamp="2026-01-01T00-00-00"
    )
    hits = [w for w in _matrix_warnings(tmp_path) if "DT_RPATH" in w]
    assert len(hits) == 1
    assert "container-substitution" in hits[0]
    assert "host-substitution" not in hits[0]


def test_reading_is_cached_per_environment_not_re_read_per_cell(tmp_path):
    """One env.json read per environment, not one per cell.

    The tags belong to the libraries installed in that environment and
    cannot change under a running matrix, so a 30-cell recipe must not put
    30 snapshot reads in the loop -- but the cache key has to be the
    environment, or the first cell's answer is imposed on every other stack
    in the matrix.
    """
    rocm7 = tmp_path / "rocm7.json"
    rocm10 = tmp_path / "rocm10.json"
    rocm7.write_text(json.dumps({"library_linkage": {"rocm_rpath": False}}))
    rocm10.write_text(json.dumps({"library_linkage": {"rocm_rpath": True}}))
    cache: dict[str, bool | None] = {}

    assert runner._cell_rocm_rpath("local", rocm7, cache) is False
    assert runner._cell_rocm_rpath("docker", rocm10, cache) is True
    assert cache == {"local": False, "docker": True}

    # Later cells answer from the cache, not the filesystem.
    rocm10.unlink()
    assert runner._cell_rocm_rpath("docker", rocm10, cache) is True


def test_a_missing_snapshot_is_not_cached_as_unknown(tmp_path):
    """An isolated env that answers late must still be allowed to answer.

    Its first cell may fail to produce an in-container snapshot while a
    later cell of the same env succeeds; caching the first "unknown" would
    silence the warning for the rest of the run.
    """
    target = tmp_path / "env.json"
    cache: dict[str, bool | None] = {}

    assert runner._cell_rocm_rpath("docker", target, cache) is None
    assert cache == {}

    target.write_text(json.dumps({"library_linkage": {"rocm_rpath": True}}))
    assert runner._cell_rocm_rpath("docker", target, cache) is True


def test_a_pre_117_snapshot_claims_nothing(tmp_path):
    """No ``library_linkage`` block means the producer never looked."""
    target = tmp_path / "env.json"
    target.write_text(json.dumps({"schema_version": "1.16"}))
    assert runner._cell_rocm_rpath("local", target, {}) is None


def test_a_placeholder_is_not_mistaken_for_a_reading(tmp_path):
    """The isolated-env placeholder describes the descriptor, not the stack."""
    target = tmp_path / "env.json"
    target.write_text(json.dumps({"snapshot_captured": False}))
    read, value = runner._snapshot_rocm_rpath(target)
    assert (read, value) == (False, None)


@pytest.mark.parametrize("written", ["true", "yes", 1, [], {"rocm_rpath": True}])
def test_only_a_real_boolean_counts_as_a_reading(tmp_path, written):
    """``rocm_rpath`` is a tri-state, so a non-boolean is the third state.

    env.json is a file on disk: a hand-edit, a producer from another branch, or
    a truncated write can all put a string there, and every value here except
    the empty list is TRUTHY. Passing one straight through would make the
    substitution warning fire off something nobody measured -- the field says
    what the ELF dynamic sections carried, and "true" is not that reading.

    The block still counts as read, so a caller may cache the answer instead of
    re-reading a file that will keep saying the same thing.
    """
    target = tmp_path / "env.json"
    target.write_text(json.dumps({"library_linkage": {"rocm_rpath": written}}))

    read, value = runner._snapshot_rocm_rpath(target)

    assert read is True
    assert value is None
