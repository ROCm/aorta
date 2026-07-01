"""Tests for output layout + writers (src/aorta/triage/output.py) via run_recipe()."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

import aorta.triage.runner as runner
from aorta.instrumentation.environment import EnvSnapshot
from aorta.triage.output import NO_TICKET_SLUG, resolve_run_dir, safe_slug
from aorta.triage.recipe import Recipe, build_recipe_from_flags

# ---- Fixtures -------------------------------------------------------------


@dataclass
class _FakeTrial:
    exit_status: str = "ok"
    wall_clock_sec: float = 1.0
    result: dict | None = None


def _fake_trial(passed: bool = True, step_times_ms: list[float] | None = None) -> _FakeTrial:
    return _FakeTrial(
        result={
            "passed": passed,
            "step_times_ms": step_times_ms or [100.0],
        }
    )


def _fake_trial_did_not_run(configured_iterations: int = 50, wall_clock_sec: float = 3.5) -> _FakeTrial:
    """A trial that died before the workload's main work phase began.

    Mirrors the failure mode the issue (#173) is fighting: the workload
    crashed during import / setup so it produced wall-clock time but no
    iterations. The aggregator must refuse to surface a step-time number
    derived from this; the matrix must mark the cell ``did_not_run``.
    """
    return _FakeTrial(
        exit_status="workload_failed",
        wall_clock_sec=wall_clock_sec,
        result={
            "passed": False,
            "main_work_started": False,
            "executed_iterations": 0,
            "configured_iterations": configured_iterations,
        },
    )


def _fake_trial_completed(
    configured_iterations: int = 50,
    step_times_ms: list[float] | None = None,
) -> _FakeTrial:
    """A trial that fully completed its configured iteration budget."""
    return _FakeTrial(
        exit_status="ok",
        wall_clock_sec=1.0,
        result={
            "passed": True,
            "step_times_ms": step_times_ms or [100.0],
            "main_work_started": True,
            "executed_iterations": configured_iterations,
            "configured_iterations": configured_iterations,
        },
    )


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


def _partial_snapshot() -> EnvSnapshot:
    snap = _clean_snapshot()
    # EnvSnapshot is frozen, but list+bool fields mutate in place
    object.__setattr__(snap, "partial", True)
    snap.partial_reasons.append("rdhc: not installed")
    return snap


@pytest.fixture
def patched_env(monkeypatch):
    """Stub collect_env so tests don't hit the real host probe."""
    mock = MagicMock(return_value=_clean_snapshot())
    monkeypatch.setattr(runner, "collect_env", mock)
    return mock


@pytest.fixture
def patched_run_trials(monkeypatch):
    """Stub run_trials so no workloads are invoked."""
    mock = MagicMock(return_value=[_fake_trial(), _fake_trial()])
    monkeypatch.setattr(runner, "run_trials", mock)
    return mock


def _simple_recipe(ticket: str | None = "ABC-1", workload: str = "fsdp") -> Recipe:
    return build_recipe_from_flags(
        workload=workload,
        mitigation_axis="none,tf32_off",
        environment_axis="local",
        trials=2,
        steps=10,
        ticket=ticket,
    )


# ---- safe_slug / resolve_run_dir ------------------------------------------


def test_safe_slug_replaces_unsafe_chars():
    assert safe_slug("PROJ-123") == "PROJ-123"
    assert safe_slug("with space") == "with_space"
    assert safe_slug("a/b:c") == "a_b_c"
    assert safe_slug("") == "_"


def test_safe_slug_rejects_dot_components():
    """`.` and `..` are filesystem-meaningful even after char-class scrubbing."""
    assert safe_slug(".") == "_"
    assert safe_slug("..") == "_"


def test_resolve_run_dir_with_ticket(tmp_path):
    r = _simple_recipe(ticket="PROJ-1")
    run_dir = resolve_run_dir(tmp_path, r, timestamp="2026-01-01T00-00-00")
    assert run_dir == tmp_path / "PROJ-1" / "fsdp" / "2026-01-01T00-00-00"
    assert run_dir.exists()


def test_resolve_run_dir_without_ticket_routes_to_no_ticket(tmp_path):
    r = _simple_recipe(ticket=None)
    run_dir = resolve_run_dir(tmp_path, r, timestamp="2026-01-01T00-00-00")
    assert run_dir.parts[-3] == NO_TICKET_SLUG


# ---- layout="flat_resume" (issue #188 FR 1.13) ---------------------------


def test_flat_resume_layout(tmp_path):
    """`layout="flat_resume"` returns <output>/<ticket>/ (no timestamp, no workload)."""
    r = _simple_recipe(ticket="PROBE-1")
    run_dir = resolve_run_dir(tmp_path, r, layout="flat_resume")
    assert run_dir == tmp_path / "PROBE-1"
    assert run_dir.is_dir()


def test_flat_resume_layout_is_idempotent(tmp_path):
    """Re-invoking with the same args returns the same path (resume model)."""
    r = _simple_recipe(ticket="PROBE-1")
    a = resolve_run_dir(tmp_path, r, layout="flat_resume")
    b = resolve_run_dir(tmp_path, r, layout="flat_resume")
    assert a == b


def test_flat_resume_layout_no_ticket(tmp_path):
    r = _simple_recipe(ticket=None)
    run_dir = resolve_run_dir(tmp_path, r, layout="flat_resume")
    assert run_dir == tmp_path / NO_TICKET_SLUG


def test_resolve_run_dir_rejects_unknown_layout(tmp_path):
    """Regression for PR #194 review: an unknown ``layout`` value used
    to silently land in the timestamped branch (the type guard fires only
    under mypy --strict). The runtime check must reject misspellings so
    probe-mode callers cannot accidentally produce a timestamped tree.
    """
    r = _simple_recipe(ticket="PROJ-1")
    with pytest.raises(ValueError, match="layout must be"):
        resolve_run_dir(tmp_path, r, layout="flatresume")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layout must be"):
        resolve_run_dir(tmp_path, r, layout="")  # type: ignore[arg-type]


def test_cell_directory_rejects_unknown_layout():
    """Regression for PR #241 review: ``_cell_directory`` shares the
    ``layout`` contract with ``resolve_run_dir`` but used to treat it as a
    free-form str, so a typo (``"flatresume"``) silently rendered a
    triage-style ``cells/<slug>/`` link for a probe run. It must reject
    unknown values too.
    """
    from aorta.triage.output import _cell_directory

    assert _cell_directory("a-b", "flat_resume") == "a-b/"
    assert _cell_directory("a-b", "timestamped") == "cells/a-b/"
    with pytest.raises(ValueError, match="layout must be"):
        _cell_directory("a-b", "flatresume")
    with pytest.raises(ValueError, match="layout must be"):
        _cell_directory("a-b", "")


def test_default_layout_is_byte_equivalent_to_timestamped(tmp_path):
    """Existing callers see no behaviour change -- the default is 'timestamped'."""
    r = _simple_recipe(ticket="PROJ-1")
    a = resolve_run_dir(tmp_path, r, timestamp="2026-01-01T00-00-00")
    # Same call with explicit layout="timestamped" produces a sibling
    # timestamped dir (a -2 suffix appended because parent path matches).
    assert a == tmp_path / "PROJ-1" / "fsdp" / "2026-01-01T00-00-00"
    b = resolve_run_dir(
        tmp_path, r, timestamp="2026-01-01T00-00-00", layout="timestamped"
    )
    assert b == tmp_path / "PROJ-1" / "fsdp" / "2026-01-01T00-00-00-2"


# ---- End-to-end via run_recipe -------------------------------------------


def test_run_recipe_writes_expected_files(tmp_path, patched_env, patched_run_trials):
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path, timestamp="2026-02-03T04-05-06")

    assert (run_dir / "matrix.md").exists()
    assert (run_dir / "matrix.json").exists()
    assert (run_dir / "recipe.resolved.yaml").exists()
    assert (run_dir / "host_env.json").exists()
    assert (run_dir / "environments" / "local" / "env.json").exists()
    assert (run_dir / "cells").exists()
    assert (run_dir / "cells" / "none-local").exists()
    assert (run_dir / "cells" / "tf32_off-local").exists()


def test_host_env_collected_exactly_once(tmp_path, patched_env, patched_run_trials):
    """Probe is captured for the runner's host scope and any *non-isolated* env.

    Per the issue-3 fix: docker / venv envs (incl. inline-docker) skip the
    runner-process probe because B1 currently runs in-process and the probe
    would record host state under a docker label. So with one local env and
    one inline-docker env we expect: 1 host probe + 1 local-env probe = 2.
    """
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off",
        environment_axis="local,image:rocm/pytorch:nightly",
        trials=1,
        steps=10,
    )
    runner.run_recipe(r, output_dir=tmp_path)
    assert patched_env.call_count == 2


def test_isolated_env_writes_placeholder_not_runner_snapshot(
    tmp_path, patched_env, patched_run_trials
):
    """Docker / inline envs skip the runner-process probe and write a placeholder.

    Pinning the issue-3 fix: a runner-time `collect_env()` for an isolated
    env would record host state under the docker label and silently mislead
    anyone reading `environments/<name>/env.json`.
    """
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="image:rocm/pytorch:nightly",
        trials=1,
        steps=10,
    )
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    inline_name = r.cells[0].environment
    env_json = run_dir / "environments" / inline_name / "env.json"
    assert env_json.exists()
    placeholder = json.loads(env_json.read_text())
    assert placeholder["snapshot_captured"] is False
    assert "B1" in placeholder["skip_reason"]
    assert placeholder["descriptor"]["docker"] == "rocm/pytorch:nightly"
    md = (run_dir / "matrix.md").read_text()
    assert "skipped" in md.lower()


def test_per_env_probe_once_per_unique_env(tmp_path, patched_env, patched_run_trials):
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off,xnack",
        environment_axis="local",
        trials=1,
        steps=10,
    )
    runner.run_recipe(r, output_dir=tmp_path)
    # 1 host + 1 env (local) = 2, despite 3 cells.
    assert patched_env.call_count == 2


def test_rerun_creates_fresh_timestamp_dir(tmp_path, patched_env, patched_run_trials):
    r = _simple_recipe(ticket="T-1")
    first = runner.run_recipe(r, output_dir=tmp_path, timestamp="2026-01-01T00-00-00")
    second = runner.run_recipe(r, output_dir=tmp_path, timestamp="2026-01-02T00-00-00")
    assert first != second
    assert first.exists() and second.exists()


def test_different_workloads_dont_conflate(tmp_path, patched_env, patched_run_trials):
    r1 = _simple_recipe(workload="fsdp")
    # Build second recipe manually because build_recipe_from_flags validates
    # the workload. We're only asserting path layout, so a hand-built Recipe
    # with a different workload string is enough.
    r2 = Recipe(
        schema_version=1,
        workload="other_workload",
        trials=r1.trials,
        steps=r1.steps,
        cells=r1.cells,
        ticket=r1.ticket,
        confound=r1.confound,
        inline_environments=r1.inline_environments,
    )
    runner.run_recipe(r1, output_dir=tmp_path, timestamp="2026-01-01T00-00-00")
    runner.run_recipe(r2, output_dir=tmp_path, timestamp="2026-01-01T00-00-00")
    assert (tmp_path / "ABC-1" / "fsdp").exists()
    assert (tmp_path / "ABC-1" / "other_workload").exists()


# ---- matrix.md + matrix.json content -------------------------------------


def test_matrix_json_records_baseline_and_confound(tmp_path, patched_env, patched_run_trials):
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    assert doc["baseline_cell"] == "none-local"
    assert doc["confound"]["threshold"] == 1.15
    assert {c["name"] for c in doc["cells"]} == {"none-local", "tf32_off-local"}
    # Baseline cell must carry the baseline tag.
    base = next(c for c in doc["cells"] if c["name"] == "none-local")
    assert base["confound"] == "(baseline)"


def test_matrix_md_includes_headers(tmp_path, patched_env, patched_run_trials):
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert "# Triage Matrix - fsdp" in md
    assert "**Ticket**: T-1" in md
    assert "**Baseline cell**: none-local" in md
    assert "Cell" in md and "Confound" in md
    # Renamed column: was "NaN rate", now "Failure rate" (rate counts every
    # non-ok exit, not just NaNs).
    assert "Failure rate" in md
    assert "NaN rate" not in md
    # Round-6 rename: column was labelled "Trials" but rendered
    # `failed_count / trial_count` underneath, so readers parsed "3 / 8"
    # as a trial count rather than failures-out-of-trials. The header is
    # now "Failures"; pin both halves so a later edit can't drift them
    # apart again.
    assert "| Failures " in md  # header column with surrounding pipes
    assert "| Trials " not in md  # the misleading old header must not return
    assert "none-local" in md
    assert "tf32_off-local" in md


def test_matrix_md_failures_column_renders_failed_over_total(tmp_path, patched_env, monkeypatch):
    """Pin the `failed_count / trial_count` rendering under the new header.

    With the trials stub returning two passing trials, the cell row should
    read ``0 / 2`` under "Failures" -- zero failures out of two trials.
    Confirms the value semantics didn't drift along with the rename.
    """
    monkeypatch.setattr(
        runner, "run_trials", MagicMock(return_value=[_fake_trial(), _fake_trial()])
    )
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    # The baseline row's numeric cell: 0 failures of 2 valid trials.
    assert "0 / 2" in md
    # Legend line documents what "Failures" means (issue #230: denominator
    # is valid trials = passed + failed, errors excluded).
    assert "`Failures` is `failed_count / valid_trials`" in md


def test_resolved_recipe_is_loadable_by_load_recipe(tmp_path, patched_env, patched_run_trials):
    """`recipe.resolved.yaml` must round-trip through load_recipe() -- no debug fields."""
    from aorta.triage.recipe import load_recipe

    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    resolved_path = run_dir / "recipe.resolved.yaml"
    doc = yaml.safe_load(resolved_path.read_text())
    assert doc["workload"] == "fsdp"
    assert {c["name"] for c in doc["cells"]} == {"none-local", "tf32_off-local"}
    # Strict schema: no debug fields leaked into the rerunnable artifact.
    assert "inline_environments" not in doc
    for cell in doc["cells"]:
        assert "mitigation_contributions" not in cell
        assert "resolved_environment" not in cell
        assert "resolved_mitigation_env" not in cell
    # The whole point: load_recipe() accepts it.
    reloaded = load_recipe(resolved_path)
    assert reloaded.workload == r.workload
    assert {c.name for c in reloaded.cells} == {c.name for c in r.cells}


def test_resolved_recipe_inline_envs_emit_docker_shorthand(
    tmp_path, patched_env, patched_run_trials
):
    """Inline cells re-emit `{docker: <ref>}` so reload re-derives the same _inline_<hash>."""
    from aorta.triage.recipe import load_recipe

    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="image:rocm/pytorch:nightly",
        trials=1,
        steps=10,
    )
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    resolved_path = run_dir / "recipe.resolved.yaml"
    doc = yaml.safe_load(resolved_path.read_text())
    cell_env = doc["cells"][0]["environment"]
    assert isinstance(cell_env, dict)
    assert cell_env == {"docker": "rocm/pytorch:nightly"}
    reloaded = load_recipe(resolved_path)
    # Auto-name is reproducible from the ref, so a sidecar JSON isn't needed.
    assert reloaded.cells[0].environment == r.cells[0].environment


def test_resolved_recipe_round_trips_workload_config(
    tmp_path, patched_env, patched_run_trials
):
    """B2.2: workload_config at recipe + cell scope must survive load -> run -> reload."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe, load_recipe

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
                workload_config={"shampoo_api": "old"},
            ),
        ),
        ticket="WC-RT",
        confound=ConfoundCfg(baseline_cell="a"),
        workload_config={"shampoo_api": "new", "warmup": 5},
    )
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    reloaded = load_recipe(run_dir / "recipe.resolved.yaml")
    assert reloaded.workload_config == {"shampoo_api": "new", "warmup": 5}
    assert reloaded.cells[0].workload_config == {}
    assert reloaded.cells[1].workload_config == {"shampoo_api": "old"}


def test_resolved_recipe_round_trips_stop_after(
    tmp_path, patched_env, patched_run_trials
):
    """An active stop_after rule must survive load -> run -> reload, so a rerun
    from recipe.resolved.yaml keeps the stopping behaviour (issue #232)."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe, StopAfter, load_recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(Cell(name="a", mitigations=("none",), environment="local"),),
        ticket="SA-RT",
        confound=ConfoundCfg(baseline_cell="a"),
        stop_after=StopAfter(events=2, max_trials=5, event_verdict="fail"),
    )
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    reloaded = load_recipe(run_dir / "recipe.resolved.yaml")
    assert reloaded.stop_after == StopAfter(events=2, max_trials=5, event_verdict="fail")


# ---- Config column (diffs-only workload_config) --------------------------
#
# The column surfaces per-cell workload_config keys whose value varies
# across cells. Hidden when no cell sets workload_config OR when every
# cell agrees. Confound classification unchanged: this column is the
# disambiguation when two otherwise-identical rows differ only on a
# workload knob (e.g. `shampoo_api=old` selecting the V1 SHAMPOO entry).


def _cell(name: str, *, workload_config: dict | None = None):
    """One-line Cell builder for Config-column tests."""
    from aorta.triage.recipe import Cell

    return Cell(
        name=name,
        mitigations=("none",),
        environment="local",
        workload_config=workload_config or {},
    )


def _recipe_with_cells(*cells, recipe_workload_config: dict | None = None):
    from aorta.triage.recipe import ConfoundCfg, Recipe

    return Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=tuple(cells),
        ticket="CFG",
        confound=ConfoundCfg(baseline_cell=cells[0].name),
        workload_config=recipe_workload_config or {},
    )


def test_matrix_md_config_column_hidden_when_no_workload_config(
    tmp_path, patched_env, patched_run_trials
):
    r = _recipe_with_cells(_cell("a"), _cell("b"))
    md = (runner.run_recipe(r, output_dir=tmp_path) / "matrix.md").read_text(encoding="utf-8")
    assert "| Config " not in md


def test_matrix_md_config_column_hidden_when_all_cells_share_config(
    tmp_path, patched_env, patched_run_trials
):
    r = _recipe_with_cells(_cell("a"), _cell("b"),
                           recipe_workload_config={"shampoo_api": "new"})
    md = (runner.run_recipe(r, output_dir=tmp_path) / "matrix.md").read_text(encoding="utf-8")
    assert "| Config " not in md


def test_matrix_md_config_column_shows_only_varying_keys(
    tmp_path, patched_env, patched_run_trials
):
    """One cell flips shampoo_api -> Config column appears; only that key is rendered."""
    r = _recipe_with_cells(
        _cell("a"),
        _cell("b", workload_config={"shampoo_api": "old"}),
        recipe_workload_config={"warmup": 5},  # shared key: NOT rendered
    )
    md = (runner.run_recipe(r, output_dir=tmp_path) / "matrix.md").read_text(encoding="utf-8")
    assert "| Config " in md
    assert "shampoo_api=old" in md
    assert "warmup=5" not in md  # shared across cells -> hidden
    assert "- `Config` --" in md  # legend bullet present


def test_matrix_md_config_column_renders_dash_for_cell_with_no_varying_keys(
    tmp_path, patched_env, patched_run_trials
):
    r = _recipe_with_cells(
        _cell("a"),
        _cell("b", workload_config={"shampoo_api": "old"}),
    )
    md = (runner.run_recipe(r, output_dir=tmp_path) / "matrix.md").read_text(encoding="utf-8")
    # Cell "a" has no workload_config; under Config column it must show "—".
    a_row = next(line for line in md.splitlines() if line.startswith("| a "))
    assert "| — " in a_row


def test_matrix_json_records_workload_config_per_cell(
    tmp_path, patched_env, patched_run_trials
):
    """Persisted on CellStats so downstream consumers can re-render the column."""
    r = _recipe_with_cells(
        _cell("a"),
        _cell("b", workload_config={"shampoo_api": "old"}),
        recipe_workload_config={"warmup": 5},
    )
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    by_name = {c["name"]: c for c in doc["cells"]}
    assert by_name["a"]["workload_config"] == {"warmup": 5}
    assert by_name["b"]["workload_config"] == {"warmup": 5, "shampoo_api": "old"}


def test_matrix_json_records_resolved_environment_per_cell(
    tmp_path, patched_env, patched_run_trials
):
    """Per-cell debug expansion lives in matrix.json (moved out of recipe.resolved.yaml)."""
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    for cell in doc["cells"]:
        assert "resolved_environment" in cell
        env = cell["resolved_environment"]
        assert env["name"] == cell["environment"]
        # tf32_off cell still records its applied env vars via stats.resolved_env_vars
    tf32_cell = next(c for c in doc["cells"] if c["name"] == "tf32_off-local")
    assert tf32_cell["resolved_env_vars"] == {"DISABLE_TF32": "1"}


# ---- Fail-soft behaviour --------------------------------------------------


def test_partial_env_probe_emits_warning_but_writes_matrix(
    tmp_path, monkeypatch, patched_run_trials
):
    monkeypatch.setattr(runner, "collect_env", MagicMock(return_value=_partial_snapshot()))
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert (run_dir / "matrix.md").exists()
    assert "partial" in md.lower()
    doc = json.loads((run_dir / "matrix.json").read_text())
    assert any("partial" in w.lower() for w in doc["warnings"])


def test_cell_exception_preserves_matrix(tmp_path, patched_env, monkeypatch):
    call_count = {"n": 0}

    def flaky(request):
        call_count["n"] += 1
        if request.mitigations == ("tf32_off",):
            raise RuntimeError("synthetic docker failure")
        return [_fake_trial(), _fake_trial()]

    monkeypatch.setattr(runner, "run_trials", flaky)
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    error_cells = [c for c in doc["cells"] if c["error"]]
    ok_cells = [c for c in doc["cells"] if not c["error"]]
    assert len(error_cells) == 1 and error_cells[0]["name"] == "tf32_off-local"
    assert error_cells[0]["confound"] == "error"
    assert len(ok_cells) == 1 and ok_cells[0]["name"] == "none-local"
    # The happy cell still ran and classified:
    assert ok_cells[0]["confound"] == "(baseline)"


def test_baseline_cell_error_produces_top_of_file_warning(tmp_path, patched_env, monkeypatch):
    def broken_baseline(request):
        if request.mitigations == ("none",):
            raise RuntimeError("baseline crashed")
        return [_fake_trial()]

    monkeypatch.setattr(runner, "run_trials", broken_baseline)
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert "baseline" in md.lower() and "errored" in md.lower()


def test_baseline_error_marks_other_cells_unclassified_not_neutral(
    tmp_path, patched_env, monkeypatch
):
    """Round-6 fix: baseline error -> non-baseline rows render `n/a`, not `-`.

    Pre-fix, ``classify`` returned ``CONFOUND_NEUTRAL`` ('-') whenever the
    baseline had no usable timing, which made matrix.md silently advertise
    every other cell as 'mitigation works without a speed cost'. Pin the
    distinct ``n/a`` rendering and the matrix.json carry-through.
    """

    def broken_baseline(request):
        if request.mitigations == ("none",):
            raise RuntimeError("baseline crashed")
        return [_fake_trial(), _fake_trial()]

    monkeypatch.setattr(runner, "run_trials", broken_baseline)
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    tf32 = next(c for c in doc["cells"] if c["name"] == "tf32_off-local")
    assert tf32["confound"] == "n/a", (
        "non-baseline cell must surface as unclassified, not as the success "
        "tag '-' (which is 'mitigation works without a speed cost')."
    )
    assert tf32["step_time_ratio"] is None
    md = (run_dir / "matrix.md").read_text()
    # The legend explains the new tag, so any future renderer change that
    # emits the wrong glyph here will fail this assertion.
    assert "`n/a`" in md
    assert "the baseline errored" in md
    assert "**unclassified**, not trustworthy" in md


def test_wall_clock_only_cells_collapse_to_na_not_speed_confound(
    tmp_path, patched_env, monkeypatch
):
    """smoke-3: ``wall_clock_total`` on both sides must NOT render as ``speed (+N%)``.

    Reproduces the 2026-05-13 smoke matrix regression case: every cell has
    ``step_time_source == "wall_clock_total"`` (workload completed but never
    emitted per-step times, so the platform divided wall clock by configured
    steps). Pre-smoke-3 the classifier matched the two sources and emitted a
    bogus ``speed (+N%)`` tag whose numerator and denominator were both
    wall-clock-derived. The fix collapses every non-baseline cell to ``n/a``
    and updates the legend to name the new reason.
    """
    wall_only = _FakeTrial(
        exit_status="ok",
        wall_clock_sec=5.0,
        result={"passed": True},  # no step_times_ms, no elapsed/total -> wall_clock_total
    )
    monkeypatch.setattr(runner, "run_trials", MagicMock(return_value=[wall_only, wall_only]))
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)

    doc = json.loads((run_dir / "matrix.json").read_text())
    tf32 = next(c for c in doc["cells"] if c["name"] == "tf32_off-local")
    assert tf32["step_time_source"] == "wall_clock_total"
    assert tf32["confound"] == "n/a"
    assert tf32["step_time_ratio"] is None

    md = (run_dir / "matrix.md").read_text()
    # Legend must name the new reason so an operator reading "n/a" can find
    # the explanation without grepping source. Pin both halves so a future
    # legend rewrite can't quietly drop the per-step requirement.
    assert "lacks per-step instrumentation" in md
    assert "`step_time_source != per_step`" in md


# ---- Class D: matrix.json carries new aggregation fields -----------------


def test_matrix_json_records_min_max_p90_and_exit_status_histogram(
    tmp_path, patched_env, patched_run_trials
):
    """Pin the new CellStats fields requested in PR review (min/max/p90 + histogram).

    These were promised in the PR description but missing from the original
    aggregation model; without them callers cannot tell `workload_failed`
    from `infrastructure_failed` when triaging a cell.
    """
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    for cell in doc["cells"]:
        assert "min_step_time_ms" in cell
        assert "max_step_time_ms" in cell
        assert "p90_step_time_ms" in cell
        assert "exit_status_counts" in cell
        # Pin step_time_source so downstream tooling can detect when a cell
        # came from a different fallback branch than the baseline; this is
        # the lineage signal the confound classifier uses to refuse
        # apples-to-oranges ratios. Surfaced per Sonbol's review on #160.
        assert "step_time_source" in cell
        assert cell["step_time_source"] in {
            "per_step",
            "elapsed_per_iter",
            "wall_clock_total",
            "missing",
        }
        # Old field name must NOT have leaked back in.
        assert "nan_rate" not in cell
    # Failure rate is still surfaced under its new name.
    assert all("failure_rate" in c for c in doc["cells"])


# ---- Class C: env-name slug collision detection --------------------------


def test_runner_rejects_env_names_whose_slugs_collide(tmp_path, patched_env, patched_run_trials):
    """Distinct env names like 'a/b' and 'a:b' both slug to 'a_b'; reject early.

    Without this check, the second environment's env.json would silently
    overwrite the first while matrix.json kept them as distinct envs --
    on-disk artifacts would contradict the in-memory cell list.
    """
    from aorta.registry import RegistryError as _RegErr
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    cells = (
        Cell(name="c1", mitigations=("none",), environment="a/b"),
        Cell(name="c2", mitigations=("none",), environment="a:b"),
    )
    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=cells,
        ticket="T-1",
        confound=ConfoundCfg(),
        inline_environments=(),
    )
    with pytest.raises(_RegErr, match="filesystem component"):
        runner.run_recipe(r, output_dir=tmp_path)


# ---- Class E: natural sort of trial_*.json by trial index ----------------


def test_collect_trial_paths_sorts_numerically_not_lexicographically(tmp_path):
    """`trial_10.json` must come AFTER `trial_2.json`, not before.

    Pin the natural-sort fix; with lex sort the recorded order diverges from
    execution order once a cell has 10+ trials.
    """
    cell_dir = tmp_path / "cell"
    work_dir = cell_dir / "fsdp"
    work_dir.mkdir(parents=True)
    for i in [0, 1, 2, 3, 9, 10, 11, 100]:
        (work_dir / f"trial_{i}.json").write_text("{}")

    paths = runner._collect_trial_paths(cell_dir)
    indices = [int(p.rsplit("trial_", 1)[1].rsplit(".", 1)[0]) for p in paths]
    assert indices == [0, 1, 2, 3, 9, 10, 11, 100]


def test_collect_trial_paths_sorts_dispatcher_naming_by_trial_index(tmp_path):
    """Regression for PR #194 review: dispatcher writes
    ``trial_d<dataset>_m<mitigation>_t<trial>.json`` and the sort key
    must extract ``<trial>`` so ``..._t10.json`` doesn't lex-sort
    before ``..._t2.json``. Previously the helper only matched
    ``trial_<N>.json`` and fell into the alphabetical sentinel branch
    for dispatcher-shape files, mis-ordering hydration for any
    cell with >= 10 trials.
    """
    cell_dir = tmp_path / "cell"
    work_dir = cell_dir / "_subprocess"
    work_dir.mkdir(parents=True)
    for i in [0, 1, 2, 3, 9, 10, 11, 100]:
        (work_dir / f"trial_d0_m0_t{i}.json").write_text("{}")

    paths = runner._collect_trial_paths(cell_dir)

    def _t_index(p: str) -> int:
        # 'trial_d0_m0_t10' -> '10'
        return int(p.rsplit("_t", 1)[1].rsplit(".", 1)[0])

    indices = [_t_index(p) for p in paths]
    assert indices == [0, 1, 2, 3, 9, 10, 11, 100]


def test_collect_trial_paths_mixed_naming_keeps_each_shape_natural_sorted(tmp_path):
    """A directory that contains BOTH legacy ``trial_<N>.json`` and
    dispatcher ``trial_d<d>_m<m>_t<N>.json`` files should sort by the
    integer trial index across both, so resume-hydration order matches
    execution order regardless of how the file was written.
    """
    cell_dir = tmp_path / "cell"
    work_dir = cell_dir / "_subprocess"
    work_dir.mkdir(parents=True)
    (work_dir / "trial_d0_m0_t10.json").write_text("{}")
    (work_dir / "trial_2.json").write_text("{}")
    (work_dir / "trial_d0_m0_t1.json").write_text("{}")
    (work_dir / "trial_11.json").write_text("{}")

    paths = runner._collect_trial_paths(cell_dir)
    # By integer trial index: 1, 2, 10, 11.
    assert [Path(p).stem for p in paths] == [
        "trial_d0_m0_t1",
        "trial_2",
        "trial_d0_m0_t10",
        "trial_11",
    ]


# ---- Class B: --mitigations-file (sidecar) lifecycle ---------------------


def _write_sidecar(path: pytest.TempPathFactory | object, mitigations: dict) -> None:
    """Helper: write a minimal sidecar JSON with the given mitigation map."""
    path.write_text(  # type: ignore[attr-defined]
        json.dumps({"version": 1, "mitigations": mitigations}),
        encoding="utf-8",
    )


def test_operator_sidecars_copied_into_run_dir_for_replay(
    tmp_path, patched_env, patched_run_trials
):
    """--mitigations-file content must be archived alongside recipe.resolved.yaml.

    Without this, the resolved YAML references mitigation names by name and
    a rerun on a fresh checkout of the run dir would fail to resolve them.
    """
    sidecar = tmp_path / "ops.sidecar.json"
    _write_sidecar(sidecar, {"my_local_mit": {"FOO": "BAR"}})

    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,my_local_mit",
        environment_axis="local",
        trials=1,
        steps=10,
        sidecar_files=(sidecar,),
    )
    output_dir = tmp_path / "out"
    run_dir = runner.run_recipe(
        r,
        output_dir=output_dir,
        extra_sidecar_files=(sidecar,),
    )
    archived = run_dir / "sidecars" / "ops.sidecar.json"
    assert archived.exists()
    # Byte-identical: the snapshot is the file the operator passed in.
    assert archived.read_text() == sidecar.read_text()


def test_recipe_carries_sidecar_files_for_programmatic_run_recipe(
    tmp_path, patched_env, patched_run_trials
):
    """Round-6 fix: programmatic ``load_recipe -> run_recipe`` works without
    re-passing ``sidecar_files`` at the runner.

    Pre-fix, ``Recipe`` discarded the sidecar list ``load_recipe`` /
    ``build_recipe_from_flags`` validated against, so a caller that passed
    ``sidecar_files=(s,)`` to the loader and then called ``run_recipe(r)``
    (no ``extra_sidecar_files``) would hit unknown-mitigation errors at
    execute time -- despite the recipe being advertised as pre-validated.
    Pin (a) the recipe carries the list, and (b) the runner uses it.
    """
    sidecar = tmp_path / "ops.sidecar.json"
    _write_sidecar(sidecar, {"my_local_mit": {"FOO": "BAR"}})

    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,my_local_mit",
        environment_axis="local",
        trials=1,
        steps=10,
        sidecar_files=(sidecar,),
    )
    assert r.sidecar_files == (sidecar,), "Recipe must remember the sidecar list."

    output_dir = tmp_path / "out"
    # Crucially: NO extra_sidecar_files=. The recipe alone must drive
    # everything the runner needs (mitigation resolution + replay archive).
    run_dir = runner.run_recipe(r, output_dir=output_dir)
    archived = run_dir / "sidecars" / "ops.sidecar.json"
    assert archived.exists(), "operator sidecar must be archived for replay."
    assert archived.read_text() == sidecar.read_text()
    # And run_trials saw the archived copy in its sidecar_files so cells
    # using sidecar-only mitigations resolve.
    req = patched_run_trials.call_args_list[1].args[0]  # the my_local_mit cell
    assert archived in req.sidecar_files


def test_run_recipe_dedupes_recipe_and_extra_sidecar_files(
    tmp_path, patched_env, patched_run_trials
):
    """Belt-and-suspenders: passing the same sidecar via both layers must not
    double-copy or double-archive.

    The CLI used to pass ``--mitigations-file`` to both the loader and to
    ``run_recipe`` as ``extra_sidecar_files``. After the round-6 fix it only
    passes them through the recipe, but third-party callers may still
    duplicate -- pin that the runner deduplicates by resolved path.
    """
    sidecar = tmp_path / "ops.sidecar.json"
    _write_sidecar(sidecar, {"my_local_mit": {"FOO": "BAR"}})

    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,my_local_mit",
        environment_axis="local",
        trials=1,
        steps=10,
        sidecar_files=(sidecar,),
    )
    output_dir = tmp_path / "out"
    run_dir = runner.run_recipe(
        r,
        output_dir=output_dir,
        extra_sidecar_files=(sidecar,),  # same file as recipe.sidecar_files
    )
    sidecars_dir = run_dir / "sidecars"
    archived = list(sidecars_dir.iterdir())
    assert len(archived) == 1, f"expected one archived sidecar, got {archived}"
    assert archived[0].name == "ops.sidecar.json"


def test_output_module_docstring_does_not_overpromise_resolved_yaml():
    """Class A guard: the module docstring used to claim recipe.resolved.yaml
    expanded every registry name into env-var bundles + docker refs and was
    therefore drift-immune.

    The implementation deliberately does NOT expand named entries (so the
    file stays loadable as a strict recipe). Pin the corrected wording so
    the next docstring rewrite can't reintroduce the false claim.
    """
    import aorta.triage.output as output_mod

    doc = output_mod.__doc__ or ""
    # Old, overpromising sentences must be gone.
    assert "every registry name expanded" not in doc
    assert "even if the registries drift" not in doc
    # New wording explicitly flags the inline-vs-named asymmetry.
    assert "NOT expanded" in doc
    assert "resolved_env_vars" in doc


# ---- Class D: same-second collision -> -N suffix, never overwrite -------


def test_resolve_run_dir_appends_suffix_on_same_timestamp(
    tmp_path, patched_env, patched_run_trials
):
    """Two runs in the same wall-clock second must NOT overwrite each other.

    Pre-fix: ``mkdir(exist_ok=True)`` silently reused the directory and the
    second run clobbered the first run's matrix.{md,json}, contradicting the
    "never overwrites" docstring guarantee. Easy to hit in CI loops.
    """
    r = _simple_recipe(ticket="T-1")
    ts = "2026-02-03T04-05-06"
    first = runner.run_recipe(r, output_dir=tmp_path, timestamp=ts)
    second = runner.run_recipe(r, output_dir=tmp_path, timestamp=ts)
    assert first != second
    assert first.exists() and second.exists()
    # Disambiguator is the documented "-N" suffix on the leaf only.
    assert first.name == ts
    assert second.name == f"{ts}-2"
    # Both runs produced their own artifacts (no clobber).
    assert (first / "matrix.md").exists()
    assert (second / "matrix.md").exists()


def test_resolve_run_dir_handles_many_collisions(tmp_path, patched_env, patched_run_trials):
    """Suffix counter advances past -2 when -2 also exists."""
    r = _simple_recipe(ticket="T-1")
    ts = "2026-02-03T04-05-06"
    dirs = [runner.run_recipe(r, output_dir=tmp_path, timestamp=ts) for _ in range(5)]
    names = [d.name for d in dirs]
    assert names == [ts, f"{ts}-2", f"{ts}-3", f"{ts}-4", f"{ts}-5"]
    # No two paths collide.
    assert len(set(dirs)) == len(dirs)


# ---- Class B: dry-run runs the same preflight as real run ----------------


def test_dry_run_rejects_unresolvable_baseline():
    """Pre-fix: dry-run printed a clean summary for recipes a real run rejected.

    Pin the new contract: anything that ``run_recipe(..., dry_run=False)``
    rejects at preflight must also be rejected by ``--dry-run``, so CI
    pre-submit checks are honest.
    """
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe, RecipeCellError

    # Multi-cell recipe with no auto-resolvable baseline (no `baseline-*`
    # name, no `mitigations: [none]`, and no explicit baseline_cell).
    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(
            Cell(name="a-local", mitigations=("tf32_off",), environment="local"),
            Cell(name="b-local", mitigations=("xnack",), environment="local"),
        ),
        ticket="T-1",
        confound=ConfoundCfg(),
        inline_environments=(),
    )
    with pytest.raises(RecipeCellError, match="cannot resolve baseline cell"):
        runner.run_recipe(r, output_dir="ignored", dry_run=True)


def test_dry_run_rejects_env_slug_collision(tmp_path):
    """Dry-run must surface env-slug collisions before printing the summary."""
    from aorta.registry import RegistryError as _RegErr
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(
            Cell(name="c1", mitigations=("none",), environment="a/b"),
            Cell(name="c2", mitigations=("none",), environment="a:b"),
        ),
        ticket="T-1",
        confound=ConfoundCfg(),
        inline_environments=(),
    )
    with pytest.raises(_RegErr, match="filesystem component"):
        runner.run_recipe(r, output_dir="ignored", dry_run=True)


def test_dry_run_does_not_create_run_dir_on_validation_failure(tmp_path):
    """Preflight runs BEFORE resolve_run_dir, so a failed dry-run must not
    leave breadcrumbs on the filesystem."""
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe, RecipeCellError

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(
            Cell(name="a-local", mitigations=("tf32_off",), environment="local"),
            Cell(name="b-local", mitigations=("xnack",), environment="local"),
        ),
        ticket="T-1",
        confound=ConfoundCfg(),
        inline_environments=(),
    )
    with pytest.raises(RecipeCellError):
        runner.run_recipe(r, output_dir=tmp_path, dry_run=True)
    # No T-1/ subtree should have been created.
    assert not (tmp_path / "T-1").exists()


# ---- Class C: matrix.md reproducibility wording matches README -----------


def test_matrix_md_does_not_overpromise_reproducibility(tmp_path, patched_env, patched_run_trials):
    """The Notes footer used to claim recipe.resolved.yaml 'captures the
    registry state at run time', which is false: named mitigations and
    environments are not expanded, so a registry change between runs
    silently changes behaviour. Pin the corrected wording.
    """
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    # Old, overpromising sentence must be gone.
    assert "captures the registry state at run time" not in md
    # New wording explicitly flags the inline-vs-named asymmetry and points
    # operators at matrix.json::cells[*].resolved_env_vars for drift detection.
    assert "NOT expanded" in md
    assert "resolved_env_vars" in md


# ---- Failure hints surfacing in matrix.md / matrix.json ------------------
#
# When a workload returns `failure_details[*].hint`, both outputs must
# expose the hint at the cell level so a reader can tell "couldn't run"
# apart from "ran and produced wrong numbers" without opening per-trial
# JSON.


def _fake_trial_with_hint(hint: str) -> _FakeTrial:
    # step_times_ms populated so the platform did_not_run inference
    # doesn't fire -- this fixture represents "ran some iterations,
    # then failed with a hint", NOT "crashed before main work" (which
    # is what the inference flags). The two failure modes coexist.
    return _FakeTrial(
        exit_status="workload_failed",
        wall_clock_sec=1.0,
        result={
            "passed": False,
            "step_times_ms": [100.0],
            "failure_details": [{"status": "crash", "hint": hint}],
        },
    )


def test_matrix_md_omits_failure_hints_section_when_no_hints(
    tmp_path, patched_env, patched_run_trials
):
    """No cell carries a hint -> no header, not an empty `## Failure hints`."""
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert "## Failure hints" not in md


def test_matrix_md_renders_failure_hints_when_present(tmp_path, patched_env, monkeypatch):
    """A cell whose trials emit a hint surfaces under `## Failure hints`."""
    hint = "shampoo import failed: try shampoo_api='old'"
    monkeypatch.setattr(
        runner,
        "run_trials",
        MagicMock(return_value=[_fake_trial_with_hint(hint), _fake_trial_with_hint(hint)]),
    )
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert "## Failure hints" in md
    # One bullet per (cell, hint); both cells share the stub so both fire.
    assert f"**none-local** (2/2 trials): {hint}" in md
    assert f"**tf32_off-local** (2/2 trials): {hint}" in md
    # Section appears between the table and the Notes block.
    assert md.index("## Failure hints") < md.index("## Notes")


def test_matrix_json_records_failure_hints_per_cell(tmp_path, patched_env, monkeypatch):
    """matrix.json::cells[*].failure_hints exposes the same data structurally."""
    hint = "shampoo import failed"
    monkeypatch.setattr(
        runner,
        "run_trials",
        MagicMock(return_value=[_fake_trial_with_hint(hint), _fake_trial_with_hint(hint)]),
    )
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    for cell in doc["cells"]:
        # asdict preserves the tuple shape; json.dumps then serializes
        # tuples as JSON arrays, so consumers see [hint, count] pairs.
        assert cell["failure_hints"] == [[hint, 2]]


def test_matrix_json_failure_hints_empty_when_none_emitted(
    tmp_path, patched_env, patched_run_trials
):
    """No hints -> field is present-and-empty, not omitted, so consumers
    can rely on `cells[*].failure_hints` always being a list."""
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = json.loads((run_dir / "matrix.json").read_text())
    for cell in doc["cells"]:
        assert cell["failure_hints"] == []


def test_isolated_env_check_honors_sidecar_files(
    tmp_path, patched_env, patched_run_trials, monkeypatch
):
    """Sidecar-defined docker envs must classify as isolated, not local.

    Regression: `_is_isolated_environment` previously called `get_environment`
    without `extra_files`, so a sidecar-only docker env fell through to the
    "treat as local" branch and got a misleading host-state snapshot.
    """
    sidecar = tmp_path / "envs.sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "version": 1,
                "environments": {"sidecar_docker": {"docker": "rocm/pytorch:nightly"}},
            }
        ),
        encoding="utf-8",
    )

    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=1,
        steps=10,
        cells=(Cell(name="c1", mitigations=("none",), environment="sidecar_docker"),),
        ticket="T-1",
        confound=ConfoundCfg(),
        inline_environments=(),
    )
    output_dir = tmp_path / "out"
    run_dir = runner.run_recipe(r, output_dir=output_dir, extra_sidecar_files=(sidecar,))

    env_json = run_dir / "environments" / "sidecar_docker" / "env.json"
    placeholder = json.loads(env_json.read_text())
    # The honest placeholder, NOT a misleading collect_env() snapshot.
    assert placeholder["snapshot_captured"] is False
    assert placeholder["descriptor"]["docker"] == "rocm/pytorch:nightly"


# ---- did_not_run surfacing (issue #173) -----------------------------------
#
# Demo failure mode: nan-repro cells that crashed at import time were
# rendered with fake step times derived from setup-only wall clock,
# leading matrix.md readers to a coherent-looking but entirely false
# story. These tests pin the new behaviour:
#   * Iters column appears when at least one cell carries
#     ``configured_iterations``; hidden otherwise (legacy workloads).
#   * All-did_not_run cells render ``Iters: 0/<N>``,
#     ``Mean step (ms): n/a``, ``Confound: did_not_run``.
#   * Auto-baseline disqualified -> top-of-file `> [!WARNING]` block,
#     non-baseline cells render ``n/a``.
#   * Explicit ``baseline_cell:`` pointing to an all-did_not_run cell
#     raises ``RecipeCellError`` (loud failure for an operator's
#     deliberate choice).
#   * Notes legend describes ``did_not_run`` and ``Iters`` only when
#     they are actually displayed.


def test_iters_column_hidden_for_legacy_workloads(tmp_path, patched_env, patched_run_trials):
    """No cell populates configured_iterations -> column absent.

    The default ``patched_run_trials`` fixture returns trials that don't
    speak the new contract, exercising the backwards-compat path. Legacy
    runs must render exactly as today: no Iters column AND no
    did_not_run legend leakage (gated on outcome_counts being non-empty,
    which is itself gated on the new contract being in use).
    """
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert "| Iters " not in md
    assert "`Iters` -- iterations actually executed" not in md
    # New-contract legend entries must NOT leak into a legacy matrix.
    assert "`did_not_run`" not in md
    assert "primary code path began" not in md


def test_iters_column_shown_when_configured_disagrees(tmp_path, patched_env, monkeypatch):
    """Defensive ``?/?`` case: trials in one cell disagreed on the
    configured iteration count. ``configured_iters`` lands at None, but
    the contradiction itself is exactly what an operator needs to see --
    the column must remain visible so the ``?/?`` row surfaces.

    Pin the gating predicate (``iters_display != "—"``) by exercising
    the case it's specifically designed to keep visible.
    """

    def disagreeing(request):
        return [
            _fake_trial_completed(configured_iterations=50),
            _fake_trial_completed(configured_iterations=100),
        ]

    monkeypatch.setattr(runner, "run_trials", disagreeing)
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert "| Iters " in md
    assert "?/?" in md


def test_iters_column_appears_when_workload_populates_it(tmp_path, patched_env, monkeypatch):
    """At least one cell carries configured_iters -> column rendered, with
    the per-row pre-computed display string."""
    monkeypatch.setattr(
        runner,
        "run_trials",
        MagicMock(
            return_value=[_fake_trial_completed(configured_iterations=50) for _ in range(2)]
        ),
    )
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    assert "| Iters " in md
    assert "50/50" in md
    # Legend entry surfaces alongside the column.
    assert "`Iters` -- iterations actually executed" in md


def test_did_not_run_cell_renders_iters_zero_step_na_and_confound_tag(
    tmp_path, patched_env, monkeypatch
):
    """The full demo case: every trial dies in setup. Matrix.md must mark
    the cell honestly across all three columns. With every cell
    did_not_run there's no usable baseline -> MatrixIncompleteError
    raises after artifacts are written, but the artifacts ARE present
    for the inspection assertions below."""
    from aorta.triage.runner import MatrixIncompleteError

    def all_did_not_run(request):
        return [_fake_trial_did_not_run(configured_iterations=50) for _ in range(2)]

    monkeypatch.setattr(runner, "run_trials", all_did_not_run)
    r = _simple_recipe(ticket="T-1")
    with pytest.raises(MatrixIncompleteError) as ei:
        runner.run_recipe(r, output_dir=tmp_path)
    run_dir = ei.value.run_dir
    md = (run_dir / "matrix.md").read_text()
    assert "0/50" in md
    # The cell row should NOT carry a fake step-time number.
    # Find the row containing "none-local" and assert it has "n/a" for step.
    none_row = next(line for line in md.splitlines() if "| none-local " in line)
    assert "n/a" in none_row
    assert "did_not_run" in none_row
    # Legend entry for the new tag is included.
    assert "`did_not_run`" in md
    assert "primary code path began" in md

    # matrix.json carries the new fields and the tag.
    doc = json.loads((run_dir / "matrix.json").read_text())
    none_cell = next(c for c in doc["cells"] if c["name"] == "none-local")
    assert none_cell["confound"] == "did_not_run"
    assert none_cell["outcome_counts"] == {"did_not_run": 2}
    assert none_cell["configured_iters"] == 50
    assert none_cell["iters_display"] == "0/50"
    assert none_cell["mean_step_time_ms"] == 0.0


def test_auto_baseline_skips_did_not_run_cell_and_picks_survivor(
    tmp_path, patched_env, monkeypatch
):
    """Auto-resolution must SKIP all-did_not_run cells and try other
    candidates rather than collapsing the whole matrix to n/a on the
    first dead baseline.

    Setup: ``none-local`` (the natural auto-baseline via the
    "mitigations==[none]" rule) is all-did_not_run, but
    ``tf32_off-local`` ran fine. The runner must fall through to
    tf32_off-local as the baseline -- no warning, normal classification
    -- so the operator still gets a useful matrix. The dead cell still
    renders its ``did_not_run`` tag because classify() short-circuits
    on the cell itself.
    """

    def baseline_did_not_run(request):
        if request.mitigations == ("none",):
            return [_fake_trial_did_not_run(configured_iterations=50) for _ in range(2)]
        return [_fake_trial_completed(configured_iterations=50) for _ in range(2)]

    monkeypatch.setattr(runner, "run_trials", baseline_did_not_run)
    r = _simple_recipe(ticket="T-1")
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    md = (run_dir / "matrix.md").read_text()
    # No "no usable baseline" warning -- a candidate survived.
    assert "> [!WARNING]" not in md
    assert "No usable baseline" not in md
    # The dead cell still renders did_not_run.
    none_row = next(line for line in md.splitlines() if "| none-local " in line)
    assert "did_not_run" in none_row
    # The survivor became the baseline.
    doc = json.loads((run_dir / "matrix.json").read_text())
    assert doc["baseline_cell"] == "tf32_off-local"
    tf32 = next(c for c in doc["cells"] if c["name"] == "tf32_off-local")
    assert tf32["confound"] == "(baseline)"


def test_auto_baseline_warns_when_every_candidate_is_did_not_run(
    tmp_path, patched_env, monkeypatch
):
    """When skip-and-retry exhausts the candidate list (every cell is
    all-did_not_run), the runner emits the soft warning + writes
    matrix.md/.json + raises ``MatrixIncompleteError`` so the CLI exits
    non-zero. matrix.json still records WHO would have been the
    baseline so the operator can find the relevant trial logs.
    """
    from aorta.triage.runner import MatrixIncompleteError

    def all_did_not_run(request):
        return [_fake_trial_did_not_run(configured_iterations=50) for _ in range(2)]

    monkeypatch.setattr(runner, "run_trials", all_did_not_run)
    r = _simple_recipe(ticket="T-1")
    with pytest.raises(MatrixIncompleteError, match="No usable baseline") as ei:
        runner.run_recipe(r, output_dir=tmp_path)
    run_dir = ei.value.run_dir
    md = (run_dir / "matrix.md").read_text()
    assert "> [!WARNING]" in md
    assert "Auto-baseline 'none-local'" in md
    assert "no other cell in the recipe survived" in md
    doc = json.loads((run_dir / "matrix.json").read_text())
    # Name preserved despite collapse, per the design comment in runner.py.
    assert doc["baseline_cell"] == "none-local"
    # Every cell is itself did_not_run, so classify() short-circuits each
    # one to that tag -- the n/a tag is reserved for non-did_not_run cells
    # that can't ratio against a dead baseline. Here there are no such cells.
    for cell in doc["cells"]:
        assert cell["confound"] == "did_not_run"


def test_auto_baseline_warning_collapses_other_cells_to_na(
    tmp_path, patched_env, monkeypatch
):
    """When the only auto-baseline candidates are all did_not_run but
    OTHER (non-candidate) cells ran fine, the runner falls back to the
    soft warning + the surviving non-did_not_run cells render n/a (they
    can't ratio against the suppressed baseline).

    This recipe has two ``baseline-*`` cells (both did_not_run) and one
    cell with a mitigation that doesn't match the auto-resolution rules
    (so it can't BECOME the baseline) but did run successfully. The
    survivor stays in the matrix as a row but with no usable comparison.
    """
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe
    from aorta.triage.runner import MatrixIncompleteError

    def per_cell(request):
        # Only the two baseline-* cells (mitigations==("none",)) die in
        # setup; tf32-only and xnack-only both run cleanly. Without this
        # split the survivors get disqualified too and rule 4 (single
        # candidate) silently picks one -- never reaching the warn path.
        if request.mitigations == ("none",):
            return [_fake_trial_did_not_run(configured_iterations=50) for _ in range(2)]
        return [_fake_trial_completed(configured_iterations=50) for _ in range(2)]

    monkeypatch.setattr(runner, "run_trials", per_cell)

    # Two ``none``-mitigation cells (both did_not_run) plus two non-baseline
    # mitigations that ran fine. After skipping the dead candidates, no cell
    # matches auto-resolution rules 2-3, AND there's more than one candidate
    # left so rule 4 (single candidate) doesn't apply -> RecipeCellError ->
    # fallback path emits the warning. The two surviving cells render n/a
    # because they can't ratio against the dead baseline.
    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=2,
        steps=50,
        cells=(
            Cell(name="baseline-a", mitigations=("none",), environment="local"),
            Cell(name="baseline-b", mitigations=("none",), environment="local"),
            Cell(name="tf32-only", mitigations=("tf32_off",), environment="local"),
            Cell(name="xnack-only", mitigations=("xnack",), environment="local"),
        ),
        ticket="T-1",
        confound=ConfoundCfg(),
        inline_environments=(),
    )
    with pytest.raises(MatrixIncompleteError, match="No usable baseline") as ei:
        runner.run_recipe(r, output_dir=tmp_path)
    run_dir = ei.value.run_dir
    md = (run_dir / "matrix.md").read_text()
    assert "> [!WARNING]" in md
    assert "no other cell in the recipe survived" in md
    doc = json.loads((run_dir / "matrix.json").read_text())
    by_name = {c["name"]: c for c in doc["cells"]}
    assert by_name["baseline-a"]["confound"] == "did_not_run"
    assert by_name["baseline-b"]["confound"] == "did_not_run"
    assert by_name["tf32-only"]["confound"] == "n/a"
    assert by_name["xnack-only"]["confound"] == "n/a"


def test_explicit_baseline_pointing_to_did_not_run_raises_after_writing_matrix(
    tmp_path, patched_env, monkeypatch
):
    """The operator named the baseline deliberately and it ended up
    did_not_run. The runner emits the loud failure signal via
    ``MatrixIncompleteError`` AFTER writing matrix.md / matrix.json so
    the operator can still inspect what happened. The CLI catches the
    exception and exits non-zero (verified separately via the CLI test
    surface).

    Asymmetric on purpose vs. the auto-baseline-survives path: auto-
    resolution that finds a usable candidate proceeds silently;
    explicit naming that hits a dead cell raises the
    MatrixIncompleteError so CI scripts can detect it via exit code,
    while still leaving inspection artifacts behind.
    """
    from aorta.triage.recipe import Cell, ConfoundCfg, Recipe
    from aorta.triage.runner import MatrixIncompleteError

    def baseline_did_not_run(request):
        if request.mitigations == ("none",):
            return [_fake_trial_did_not_run(configured_iterations=50) for _ in range(2)]
        return [_fake_trial_completed(configured_iterations=50) for _ in range(2)]

    monkeypatch.setattr(runner, "run_trials", baseline_did_not_run)

    r = Recipe(
        schema_version=1,
        workload="fsdp",
        trials=2,
        steps=50,
        cells=(
            Cell(name="none-local", mitigations=("none",), environment="local"),
            Cell(name="tf32_off-local", mitigations=("tf32_off",), environment="local"),
        ),
        ticket="T-1",
        confound=ConfoundCfg(baseline_cell="none-local"),
        inline_environments=(),
    )
    with pytest.raises(MatrixIncompleteError, match="explicit baseline_cell 'none-local'") as ei:
        runner.run_recipe(r, output_dir=tmp_path)

    # Artifacts must be present for inspection.
    run_dir = ei.value.run_dir
    assert (run_dir / "matrix.md").exists()
    assert (run_dir / "matrix.json").exists()

    md = (run_dir / "matrix.md").read_text()
    assert "> [!WARNING]" in md
    assert "explicit baseline_cell 'none-local'" in md
    # Baseline cell carries the did_not_run tag; non-baseline cell collapses to n/a.
    none_row = next(line for line in md.splitlines() if "| none-local " in line)
    assert "did_not_run" in none_row
    tf32_row = next(line for line in md.splitlines() if "| tf32_off-local " in line)
    assert "n/a" in tf32_row

    doc = json.loads((run_dir / "matrix.json").read_text())
    assert doc["baseline_cell"] == "none-local"  # name preserved per design


def test_did_not_run_cell_summary_log_carries_warning_suffix(
    tmp_path, patched_env, monkeypatch, caplog
):
    """Per-cell terminal log line for an all-did_not_run cell must include
    the WARNING suffix so an operator watching stdout sees the same signal
    as the matrix.md reader. Pinned regardless of the post-run
    MatrixIncompleteError -- the per-cell logs are emitted before the
    final raise, so caplog catches them."""
    import logging

    from aorta.triage.runner import MatrixIncompleteError

    def all_did_not_run(request):
        return [_fake_trial_did_not_run(configured_iterations=50) for _ in range(2)]

    monkeypatch.setattr(runner, "run_trials", all_did_not_run)
    r = _simple_recipe(ticket="T-1")
    with caplog.at_level(logging.INFO, logger="aorta.triage.runner"):
        with pytest.raises(MatrixIncompleteError):
            runner.run_recipe(r, output_dir=tmp_path)
    cell_lines = [rec.getMessage() for rec in caplog.records if "done in" in rec.getMessage()]
    assert cell_lines, "expected per-cell summary lines in the runner log"
    for line in cell_lines:
        assert "outcome:" in line
        assert "iters: 0/50" in line
        assert "WARNING: workload never reached main work phase" in line


def test_completed_cell_summary_log_omits_warning_suffix(
    tmp_path, patched_env, monkeypatch, caplog
):
    """Healthy cells get the new bracket grammar but no WARNING tail."""
    import logging

    monkeypatch.setattr(
        runner,
        "run_trials",
        MagicMock(return_value=[_fake_trial_completed(configured_iterations=50) for _ in range(2)]),
    )
    r = _simple_recipe(ticket="T-1")
    with caplog.at_level(logging.INFO, logger="aorta.triage.runner"):
        runner.run_recipe(r, output_dir=tmp_path)
    cell_lines = [rec.getMessage() for rec in caplog.records if "done in" in rec.getMessage()]
    for line in cell_lines:
        assert "outcome: 2/2 completed" in line
        assert "iters: 50/50" in line
        assert "WARNING" not in line


def test_legacy_cell_summary_log_keeps_old_grammar(tmp_path, patched_env, patched_run_trials, caplog):
    """Legacy workloads (no main_work_started populated) keep their familiar
    ``[N/M trials passed]`` summary so existing log scrapers don't break."""
    import logging

    r = _simple_recipe(ticket="T-1")
    with caplog.at_level(logging.INFO, logger="aorta.triage.runner"):
        runner.run_recipe(r, output_dir=tmp_path)
    cell_lines = [rec.getMessage() for rec in caplog.records if "done in" in rec.getMessage()]
    for line in cell_lines:
        assert "trials passed" in line
        assert "outcome:" not in line


def test_partial_contract_cell_summary_uses_new_grammar(tmp_path, patched_env, monkeypatch, caplog):
    """Workload populates ``configured_iterations`` but NOT
    ``main_work_started`` -> the matrix.md ``Iters`` column shows up,
    so the terminal log must too. Pin the Copilot round-3 fix that the
    "new contract in use" predicate is ANY of the three new fields, not
    just main_work_started -- otherwise the same cell renders new-grammar
    in the table and legacy-grammar in the log, confusing the operator.
    """
    import logging

    def partial_contract(request):
        return [
            _FakeTrial(
                exit_status="ok",
                wall_clock_sec=1.0,
                result={
                    "passed": True,
                    "step_times_ms": [100.0],
                    "configured_iterations": 50,
                    "executed_iterations": 50,
                    # main_work_started intentionally absent
                },
            )
            for _ in range(2)
        ]

    monkeypatch.setattr(runner, "run_trials", partial_contract)
    r = _simple_recipe(ticket="T-1")
    with caplog.at_level(logging.INFO, logger="aorta.triage.runner"):
        runner.run_recipe(r, output_dir=tmp_path)
    cell_lines = [rec.getMessage() for rec in caplog.records if "done in" in rec.getMessage()]
    for line in cell_lines:
        assert "outcome:" in line, f"new bracket grammar expected; got: {line}"
        assert "iters: 50/50" in line
