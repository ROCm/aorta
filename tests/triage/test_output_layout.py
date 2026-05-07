"""Tests for output layout + writers (src/aorta/triage/output.py) via run_recipe()."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    # The baseline row's numeric cell: 0 failures of 2 trials.
    assert "0 / 2" in md
    # Legend line documents what "Failures" means.
    assert "`Failures` is `failed_count / trial_count`" in md


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
