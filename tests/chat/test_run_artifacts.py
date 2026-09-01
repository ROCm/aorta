"""Run-artifact discovery, rendering, and the chat tools over them.

The property under test throughout is the one :mod:`aorta.artifacts` exists to
protect: **a field the run did not record is never rendered as a zero.** A
matrix cell whose ``failure_rate`` is absent must not read as "this cell
passed", because that turns a real reproduction into a clean bill of health --
silently, and in the reassuring direction.

The tools are also checked for the two things a tool must do that a library
need not: stay inside its sandbox, and answer every failure with a string
rather than an exception that would abort the whole graph run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.chat.config import configure, reset_settings
from aorta.chat.runs import (
    describe_run_dir,
    find_run_dirs,
    iter_artifacts,
    render_env,
    render_matrix,
)
from aorta.chat.tools import artifacts as artifacts_tool

FULL_CELL = {
    "name": "none-none",
    "mitigations": [],
    "trials": 4,
    "passed_count": 1,
    "failed_count": 3,
    "failure_rate": 0.75,
    "failure_hints": [["nan detected in loss", 3]],
    "exit_status_counts": {"workload_failed": 3, "ok": 1},
    "error": None,
}

MITIGATED_CELL = {
    "name": "tf32_off-none",
    "mitigations": ["tf32_off"],
    "trials": 4,
    "passed_count": 4,
    "failed_count": 0,
    "failure_rate": 0.0,
    "failure_hints": [],
    "exit_status_counts": {"ok": 4},
    "error": None,
}


def _matrix_doc(cells: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "workload": "hrx",
        "ticket": "AORTA-99",
        "run_timestamp": "2026-09-01T00:00:00Z",
        "steps_per_trial": 5,
        "trials_per_cell": 4,
        "cells": cells,
    }


def _env_doc(**overrides) -> dict:
    doc = {
        "schema_version": "1.17",
        "captured_at": "2026-09-01T00:00:00Z",
        "rocm": {"version": "7.0.1"},
        "partial": False,
        "partial_reasons": [],
    }
    doc.update(overrides)
    return doc


@pytest.fixture()
def run_root(tmp_path: Path):
    """A run root with one complete run, wired into settings."""
    root = tmp_path / "runs"
    run = root / "sweep_2026"
    run.mkdir(parents=True)
    (run / "matrix.json").write_text(json.dumps(_matrix_doc([FULL_CELL, MITIGATED_CELL])))
    (run / "env.json").write_text(json.dumps(_env_doc()))
    reset_settings()
    configure(runs_path=str(root))
    yield root
    reset_settings()


# ── discovery ─────────────────────────────────────────────────────────────


class TestDiscovery:
    def test_finds_both_artifact_kinds(self, run_root: Path):
        found = {path.name: kind for path, kind in iter_artifacts(run_root)}
        assert found == {"matrix.json": "matrix", "env.json": "env"}

    def test_host_env_is_recognised_as_an_env_snapshot(self, tmp_path: Path):
        """The probe writes the same shape under two names depending on invocation."""
        (tmp_path / "host_env.json").write_text(json.dumps(_env_doc()))
        assert [k for _, k in iter_artifacts(tmp_path)] == ["env"]

    def test_result_json_is_not_collected(self, tmp_path: Path):
        """Nothing consumes it, and one per trial would swamp the collection."""
        (tmp_path / "result.json").write_text("{}")
        assert list(iter_artifacts(tmp_path)) == []

    def test_noise_directories_are_pruned(self, tmp_path: Path):
        """Run trees are routinely created inside a source checkout."""
        for noise in (".git", "__pycache__", ".venv"):
            buried = tmp_path / noise / "nested"
            buried.mkdir(parents=True)
            (buried / "matrix.json").write_text("{}")
        assert list(iter_artifacts(tmp_path)) == []

    def test_walk_is_depth_capped(self, tmp_path: Path):
        """Pointing the setting at a home directory must not become a full scan."""
        deep = tmp_path.joinpath(*[f"d{i}" for i in range(8)])
        deep.mkdir(parents=True)
        (deep / "matrix.json").write_text("{}")
        assert list(iter_artifacts(tmp_path)) == []

    def test_symlinked_directories_are_not_followed(self, tmp_path: Path):
        """A run dir symlinked at / would otherwise crawl the filesystem."""
        real = tmp_path / "real"
        real.mkdir()
        (real / "matrix.json").write_text("{}")
        (tmp_path / "link").symlink_to(real, target_is_directory=True)
        sources = {p.parent.name for p, _ in iter_artifacts(tmp_path)}
        assert sources == {"real"}

    def test_find_run_dirs_deduplicates_a_directory_with_two_artifacts(self, run_root: Path):
        assert find_run_dirs(run_root) == [run_root / "sweep_2026"]

    def test_describe_run_dir_is_relative_to_the_root(self, run_root: Path):
        line = describe_run_dir(run_root / "sweep_2026", run_root)
        assert line == "sweep_2026: env.json, matrix.json"

    def test_missing_root_yields_nothing_rather_than_raising(self, tmp_path: Path):
        assert list(iter_artifacts(tmp_path / "absent")) == []


# ── rendering: the absence-is-not-zero rule ───────────────────────────────


class TestRenderingPreservesAbsence:
    def test_absent_failure_rate_renders_unknown_not_zero(self):
        """The exact hazard: ``failure_rate`` reaches matrix.json by one hand-written
        line, and a consumer reading it as 0 reports a reproduction as a clean run.
        """
        from aorta.artifacts import parse_matrix

        cell = dict(FULL_CELL)
        del cell["failure_rate"]
        text = render_matrix(parse_matrix(_matrix_doc([cell])))
        assert "failure_rate: unknown" in text
        assert "failure_rate: 0" not in text
        assert "NOT RECORDED in this cell: failure_rate" in text

    def test_a_recorded_zero_still_renders_as_zero(self):
        """The other half: a real 0.0 must not be dressed up as unknown."""
        from aorta.artifacts import parse_matrix

        text = render_matrix(parse_matrix(_matrix_doc([MITIGATED_CELL])))
        assert "failure_rate: 0.000" in text
        assert "unknown" not in text

    def test_absent_exit_status_counts_does_not_become_zero_failures(self):
        from aorta.artifacts import parse_matrix

        cell = dict(FULL_CELL)
        del cell["exit_status_counts"]
        text = render_matrix(parse_matrix(_matrix_doc([cell])))
        assert "workload_failed trials: unknown" in text

    def test_present_histogram_without_the_key_really_is_zero(self):
        """An absent status did not occur: the producer counts every trial."""
        from aorta.artifacts import parse_matrix

        cell = dict(FULL_CELL, exit_status_counts={"ok": 4})
        text = render_matrix(parse_matrix(_matrix_doc([cell])))
        assert "workload_failed trials: 0" in text

    def test_empty_and_absent_mitigations_are_distinguishable(self):
        from aorta.artifacts import parse_matrix

        absent = dict(FULL_CELL)
        del absent["mitigations"]
        assert "mitigations: unknown" in render_matrix(parse_matrix(_matrix_doc([absent])))
        assert "mitigations: none" in render_matrix(parse_matrix(_matrix_doc([FULL_CELL])))

    def test_unrecognised_schema_version_is_announced(self):
        from aorta.artifacts import parse_matrix

        doc = _matrix_doc([FULL_CELL])
        doc["schema_version"] = 99
        text = render_matrix(parse_matrix(doc))
        assert "NEWER" in text
        assert "note:" in text


class TestEnvRendering:
    def test_partial_probe_lists_its_reasons(self):
        from aorta.artifacts import parse_env

        doc = _env_doc(partial=True, partial_reasons=["triton: not importable"])
        text = render_env(parse_env(doc))
        assert "probe complete: NO" in text
        assert "triton: not importable" in text

    def test_absent_partial_flag_is_not_read_as_a_clean_probe(self):
        """``partial`` is the probe's honesty signal; its absence is not honesty."""
        from aorta.artifacts import parse_env

        doc = _env_doc()
        del doc["partial"]
        text = render_env(parse_env(doc))
        assert "probe complete: unknown" in text
        assert "probe complete: yes" not in text

    def test_null_rocm_version_is_reported_as_unknown(self):
        from aorta.artifacts import parse_env

        text = render_env(parse_env(_env_doc(rocm={"version": None})))
        assert "rocm.version: unknown" in text


# ── the tools ─────────────────────────────────────────────────────────────


class TestTools:
    def test_list_runs_names_each_run_and_its_artifacts(self, run_root: Path):
        from aorta.chat.tools.artifacts import list_runs

        out = list_runs.invoke({"path": "."})
        assert "sweep_2026: env.json, matrix.json" in out

    def test_list_runs_says_where_to_configure_when_empty(self, tmp_path: Path):
        from aorta.chat.tools.artifacts import list_runs

        empty = tmp_path / "empty"
        empty.mkdir()
        reset_settings()
        configure(runs_path=str(empty))
        out = list_runs.invoke({"path": "."})
        assert "No AORTA run artifacts found" in out
        assert "AORTA_CHAT_RUNS_PATH" in out
        reset_settings()

    def test_read_run_matrix_accepts_a_run_directory(self, run_root: Path):
        """The model should not have to guess the filename."""
        from aorta.chat.tools.artifacts import read_run_matrix

        out = read_run_matrix.invoke({"path": "sweep_2026"})
        assert "cell: none-none" in out
        assert "failure_rate: 0.750" in out

    def test_read_run_matrix_accepts_the_file_itself(self, run_root: Path):
        from aorta.chat.tools.artifacts import read_run_matrix

        out = read_run_matrix.invoke({"path": "sweep_2026/matrix.json"})
        assert "workload: hrx" in out

    def test_read_run_env_finds_either_filename(self, tmp_path: Path):
        from aorta.chat.tools.artifacts import read_run_env

        root = tmp_path / "runs"
        (root / "triage").mkdir(parents=True)
        (root / "triage" / "host_env.json").write_text(json.dumps(_env_doc()))
        reset_settings()
        configure(runs_path=str(root))
        assert "rocm.version: 7.0.1" in read_run_env.invoke({"path": "triage"})
        reset_settings()

    @pytest.mark.parametrize("escape", ["../outside", "../../etc", "/etc"])
    def test_paths_escaping_the_run_root_are_refused(self, run_root: Path, escape: str):
        from aorta.chat.tools.artifacts import read_run_matrix

        out = read_run_matrix.invoke({"path": escape})
        assert out.startswith("Error:")
        assert "escapes the run root" in out

    def test_a_sibling_directory_sharing_a_prefix_is_not_inside_the_root(self, tmp_path: Path):
        """``/runs-old`` starts with the characters of ``/runs`` without being in it."""
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs-old").mkdir()
        reset_settings()
        configure(runs_path=str(tmp_path / "runs"))
        from aorta.chat.tools.artifacts import read_run_matrix

        assert "escapes the run root" in read_run_matrix.invoke({"path": "../runs-old"})
        reset_settings()

    def test_a_missing_path_is_an_error_string_not_an_exception(self, run_root: Path):
        """A raised exception would abort the graph run; a string lets the model retry."""
        from aorta.chat.tools.artifacts import read_run_matrix

        out = read_run_matrix.invoke({"path": "no_such_run"})
        assert out == "Error: path 'no_such_run' does not exist."

    def test_a_directory_without_the_artifact_says_so(self, tmp_path: Path):
        from aorta.chat.tools.artifacts import read_run_matrix

        root = tmp_path / "runs"
        (root / "bare").mkdir(parents=True)
        reset_settings()
        configure(runs_path=str(root))
        out = read_run_matrix.invoke({"path": "bare"})
        assert "no matrix.json artifact" in out
        assert "list_runs" in out
        reset_settings()

    def test_a_truncated_artifact_reports_the_read_error(self, tmp_path: Path):
        """A half-written matrix.json usually means the job died -- that is the answer."""
        from aorta.chat.tools.artifacts import read_run_matrix

        root = tmp_path / "runs"
        (root / "killed").mkdir(parents=True)
        (root / "killed" / "matrix.json").write_text('{"cells": [')
        reset_settings()
        configure(runs_path=str(root))
        out = read_run_matrix.invoke({"path": "killed"})
        assert out.startswith("Error:")
        assert "not valid JSON" in out
        reset_settings()

    def test_long_renderings_are_truncated_with_the_total_named(self, tmp_path: Path):
        from aorta.chat.tools.artifacts import read_run_matrix

        root = tmp_path / "runs"
        (root / "wide").mkdir(parents=True)
        cells = [dict(FULL_CELL, name=f"cell-{i}") for i in range(400)]
        (root / "wide" / "matrix.json").write_text(json.dumps(_matrix_doc(cells)))
        reset_settings()
        configure(runs_path=str(root))
        out = read_run_matrix.invoke({"path": "wide"})
        assert "truncated" in out
        # The cap plus the one-line notice naming the true length, and nothing
        # like the full 400-cell rendering.
        assert len(out) < artifacts_tool._MAX_CHARS + 200
        reset_settings()


class TestToolRegistration:
    def test_the_graph_exposes_every_run_artifact_tool(self):
        from aorta.chat.graph.nodes import TOOL_REGISTRY

        for name in ("list_runs", "read_run_matrix", "read_run_env", "search_run_artifacts"):
            assert name in TOOL_REGISTRY

    def test_the_text_protocol_prompt_documents_them(self):
        """The ACTION: protocol only reaches tools the prompt lists."""
        from aorta.chat.graph.nodes import TOOL_DESCRIPTIONS, TOOL_REGISTRY

        for name in TOOL_REGISTRY:
            assert name in TOOL_DESCRIPTIONS

    def test_the_prompt_warns_against_reading_unknown_as_zero(self):
        from aorta.chat.graph.nodes import SYSTEM_PROMPT, TOOL_DESCRIPTIONS

        assert "NOT RECORDED" in TOOL_DESCRIPTIONS
        assert "NOT RECORDED" in SYSTEM_PROMPT
