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
    """Minimal non-partial EnvSnapshot for test isolation."""
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
    # Recipe with 4 cells across 2 unique environments (local, inline-docker).
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off",
        environment_axis="local,image:rocm/pytorch:nightly",
        trials=1,
        steps=10,
    )
    runner.run_recipe(r, output_dir=tmp_path)
    # 1 host probe + 2 per-env probes (local + _inline_*) = 3 total.
    assert patched_env.call_count == 3


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
    assert "none-local" in md
    assert "tf32_off-local" in md


def test_resolved_recipe_contains_expanded_mitigations(tmp_path, patched_env, patched_run_trials):
    r = _simple_recipe()
    run_dir = runner.run_recipe(r, output_dir=tmp_path)
    doc = yaml.safe_load((run_dir / "recipe.resolved.yaml").read_text())
    assert doc["workload"] == "fsdp"
    cell_names = {c["name"] for c in doc["cells"]}
    assert cell_names == {"none-local", "tf32_off-local"}
    tf32_cell = next(c for c in doc["cells"] if c["name"] == "tf32_off-local")
    # tf32_off mitigation bundle is DISABLE_TF32=1
    assert tf32_cell["resolved_mitigation_env"] == {"DISABLE_TF32": "1"}


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
