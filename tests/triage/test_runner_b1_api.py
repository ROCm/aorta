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
