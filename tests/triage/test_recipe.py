"""Tests for the recipe loader + flag-mode builder (src/aorta/triage/recipe.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.registry.errors import (
    UnknownEnvironmentError,
    UnknownMitigationError,
)
from aorta.triage.recipe import (
    SCHEMA_VERSION,
    Cell,
    ConfoundCfg,
    InlineEnv,
    Recipe,
    RecipeCellError,
    RecipeSchemaError,
    build_recipe_from_flags,
    inline_env_name,
    load_recipe,
)

# ---- Helpers --------------------------------------------------------------


def _write_yaml(tmp_path: Path, text: str, name: str = "r.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _write_json(tmp_path: Path, payload: dict, name: str = "r.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


_MINIMAL_YAML = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
"""


# ---- Happy-path loading ---------------------------------------------------


def test_load_minimal_yaml(tmp_path):
    path = _write_yaml(tmp_path, _MINIMAL_YAML)
    r = load_recipe(path)
    assert isinstance(r, Recipe)
    assert r.schema_version == SCHEMA_VERSION
    assert r.workload == "fsdp"
    assert r.trials == 2
    assert r.steps == 100
    assert r.ticket is None
    assert len(r.cells) == 1
    assert r.cells[0].name == "baseline-local"
    assert r.cells[0].mitigations == ("none",)
    assert r.cells[0].environment == "local"
    assert r.source_path == path
    assert isinstance(r.source_sha256, str) and len(r.source_sha256) == 64


def test_load_minimal_json(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "schema_version": 1,
            "workload": "fsdp",
            "trials": 2,
            "steps": 100,
            "cells": [{"name": "baseline-local", "mitigations": ["none"], "environment": "local"}],
        },
    )
    r = load_recipe(path)
    assert r.workload == "fsdp"
    assert len(r.cells) == 1


def test_load_with_ticket_and_confound(tmp_path):
    text = (
        _MINIMAL_YAML
        + """\
ticket: EXAMPLE-1
confound:
  threshold: 1.25
  baseline_cell: baseline-local
"""
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.ticket == "EXAMPLE-1"
    assert r.confound.threshold == 1.25
    assert r.confound.baseline_cell == "baseline-local"


def test_load_per_cell_overrides(tmp_path):
    text = (
        _MINIMAL_YAML
        + """\
  - name: heavier
    mitigations: [tf32_off]
    environment: local
    trials: 4
    steps: 500
"""
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.cells[1].trials == 4
    assert r.cells[1].steps == 500
    assert r.cells[1].effective_trials(r.trials) == 4
    assert r.cells[0].effective_trials(r.trials) == r.trials
    assert r.cells[0].effective_steps(r.steps) == r.steps


# ---- Schema validation ----------------------------------------------------


def test_unknown_schema_version_rejected(tmp_path):
    text = _MINIMAL_YAML.replace("schema_version: 1", "schema_version: 99")
    with pytest.raises(RecipeSchemaError, match="unsupported version 99"):
        load_recipe(_write_yaml(tmp_path, text))


def test_missing_required_top_level(tmp_path):
    text = _MINIMAL_YAML.replace("workload: fsdp\n", "")
    with pytest.raises(RecipeSchemaError, match="missing required key 'workload'"):
        load_recipe(_write_yaml(tmp_path, text))


def test_unknown_top_level_key_rejected(tmp_path):
    text = _MINIMAL_YAML + "garbage: 42\n"
    with pytest.raises(RecipeSchemaError, match="unknown top-level keys"):
        load_recipe(_write_yaml(tmp_path, text))


def test_empty_mitigations_rejected(tmp_path):
    text = _MINIMAL_YAML.replace("[none]", "[]")
    with pytest.raises(RecipeSchemaError, match="empty list not allowed"):
        load_recipe(_write_yaml(tmp_path, text))


def test_duplicate_cell_name_rejected(tmp_path):
    text = (
        _MINIMAL_YAML
        + """\
  - name: baseline-local
    mitigations: [tf32_off]
    environment: local
"""
    )
    with pytest.raises(RecipeCellError, match="duplicate cell name"):
        load_recipe(_write_yaml(tmp_path, text))


def test_trials_must_be_positive(tmp_path):
    text = _MINIMAL_YAML.replace("trials: 2", "trials: 0")
    with pytest.raises(RecipeSchemaError, match="trials"):
        load_recipe(_write_yaml(tmp_path, text))


def test_unknown_mitigation_raises_b3_error(tmp_path):
    text = _MINIMAL_YAML.replace("[none]", "[does_not_exist]")
    with pytest.raises(UnknownMitigationError):
        load_recipe(_write_yaml(tmp_path, text))


def test_unknown_environment_raises_b3_error(tmp_path):
    text = _MINIMAL_YAML.replace("environment: local", "environment: bogus_env")
    with pytest.raises(UnknownEnvironmentError):
        load_recipe(_write_yaml(tmp_path, text))


def test_unknown_confound_key_rejected(tmp_path):
    text = _MINIMAL_YAML + "confound:\n  bogus: 1\n"
    with pytest.raises(RecipeSchemaError, match="unknown keys"):
        load_recipe(_write_yaml(tmp_path, text))


def test_baseline_cell_must_exist(tmp_path):
    text = _MINIMAL_YAML + "confound:\n  baseline_cell: not_a_cell\n"
    with pytest.raises(RecipeCellError, match="does not match any cell name"):
        load_recipe(_write_yaml(tmp_path, text))


# ---- Cell-name path safety (issue #160 review, comment 5) -----------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape",
        "a/b",
        "a\\b",
        "..",
        ".",
        "matrix.md",  # would clobber sibling artifact
        "-leading-dash",  # leading character must be alphanum or _
        "with space",
        "tab\there",
    ],
)
def test_cell_name_path_unsafe_rejected(tmp_path, bad_name):
    text = _MINIMAL_YAML.replace("baseline-local", bad_name)
    with pytest.raises((RecipeCellError, RecipeSchemaError)):
        load_recipe(_write_yaml(tmp_path, text))


def test_cell_name_safe_chars_accepted(tmp_path):
    """Allowed: [A-Za-z0-9_] start, then [A-Za-z0-9_.-] -- covers tickets like 'PROJ-1.fix'."""
    text = _MINIMAL_YAML.replace("baseline-local", "PROJ_1.fix-cell")
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.cells[0].name == "PROJ_1.fix-cell"


# ---- Mitigation env-var collision detection (issue #160 review, comment 4)


def _stacked_mitigations_recipe_text(mitigations: list[str]) -> str:
    return f"""\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: stacked
    mitigations: {mitigations}
    environment: local
"""


def test_stacked_mitigations_with_disagreeing_keys_rejected(tmp_path, monkeypatch):
    """Two mitigations setting the same env-var to different values is an error."""
    bundles = {"mitA": {"FOO": "1"}, "mitB": {"FOO": "2"}}
    monkeypatch.setattr(
        "aorta.triage.recipe.get_mitigation",
        lambda name, extra_files=None: bundles[name],
    )
    text = _stacked_mitigations_recipe_text(["mitA", "mitB"])
    with pytest.raises(RecipeCellError, match="must agree on overlapping keys"):
        load_recipe(_write_yaml(tmp_path, text))


def test_stacked_mitigations_with_agreeing_keys_accepted(tmp_path, monkeypatch):
    """Two mitigations setting the same env-var to the SAME value is fine."""
    bundles = {"mitA": {"FOO": "1", "BAR": "1"}, "mitB": {"FOO": "1"}}
    monkeypatch.setattr(
        "aorta.triage.recipe.get_mitigation",
        lambda name, extra_files=None: bundles[name],
    )
    r = load_recipe(_write_yaml(tmp_path, _stacked_mitigations_recipe_text(["mitA", "mitB"])))
    assert r.cells[0].mitigations == ("mitA", "mitB")


def test_collision_error_names_the_cell_and_keys(tmp_path, monkeypatch):
    """The error message must point the user at WHICH cell and WHICH keys disagree."""
    bundles = {"mitA": {"FOO": "1"}, "mitB": {"FOO": "2"}}
    monkeypatch.setattr(
        "aorta.triage.recipe.get_mitigation",
        lambda name, extra_files=None: bundles[name],
    )
    text = _stacked_mitigations_recipe_text(["mitA", "mitB"])
    with pytest.raises(RecipeCellError) as exc_info:
        load_recipe(_write_yaml(tmp_path, text))
    msg = str(exc_info.value)
    assert "stacked" in msg  # cell name
    assert "FOO" in msg  # the conflicting key
    assert "mitA" in msg and "mitB" in msg  # both contributors


def test_extra_env_overrides_mitigation_silently(tmp_path):
    """extra_env is documented as the override knob; collision with a mitigation is intentional."""
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: override-tf32
    mitigations: [tf32_off]
    environment: local
    extra_env:
      DISABLE_TF32: "0"  # explicitly contradicts tf32_off's bundle
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.cells[0].extra_env == {"DISABLE_TF32": "0"}


# ---- Inline docker --------------------------------------------------------


def test_inline_env_name_deterministic():
    assert inline_env_name("rocm/pytorch:nightly") == inline_env_name("rocm/pytorch:nightly")
    assert inline_env_name("rocm/pytorch:nightly") != inline_env_name("rocm/pytorch:stable")
    name = inline_env_name("rocm/pytorch:nightly")
    assert name.startswith("_inline_")
    assert len(name) == len("_inline_") + 8


def test_inline_docker_roundtrip(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
  - name: try-nightly
    mitigations: [none]
    environment: { docker: "rocm/pytorch:nightly" }
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    assert len(r.inline_environments) == 1
    inline = r.inline_environments[0]
    assert isinstance(inline, InlineEnv)
    assert inline.docker == "rocm/pytorch:nightly"
    # Cell.environment stores the auto-name, not the mapping.
    assert r.cells[1].environment == inline.name


def test_inline_docker_extra_keys_rejected(tmp_path):
    text = _MINIMAL_YAML.replace(
        "environment: local",
        "environment: { docker: 'x', name: 'y' }",
    )
    with pytest.raises(RecipeSchemaError, match="only accepts"):
        load_recipe(_write_yaml(tmp_path, text))


def test_inline_docker_deterministic_across_cells(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: a
    mitigations: [none]
    environment: { docker: "x/y:1" }
  - name: b
    mitigations: [none]
    environment: { docker: "x/y:1" }
  - name: c
    mitigations: [none]
    environment: { docker: "x/y:2" }
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    # Two distinct refs => two inline envs.
    assert len({env.docker for env in r.inline_environments}) == 2
    assert r.cells[0].environment == r.cells[1].environment
    assert r.cells[0].environment != r.cells[2].environment


# ---- extra_env ------------------------------------------------------------


def test_extra_env_recorded(tmp_path):
    text = _MINIMAL_YAML.replace(
        "    environment: local\n",
        '    environment: local\n    extra_env:\n      MY_FLAG: "1"\n',
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.cells[0].extra_env == {"MY_FLAG": "1"}


def test_extra_env_must_be_strings(tmp_path):
    text = _MINIMAL_YAML.replace(
        "    environment: local\n",
        "    environment: local\n    extra_env:\n      MY_FLAG: 1\n",
    )
    with pytest.raises(RecipeSchemaError, match="extra_env"):
        load_recipe(_write_yaml(tmp_path, text))


# ---- Flag-mode builder ----------------------------------------------------


def test_build_recipe_from_flags_cartesian():
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off",
        environment_axis="local",
        trials=4,
        steps=100,
    )
    assert r.workload == "fsdp"
    assert r.trials == 4
    assert r.steps == 100
    assert r.ticket is None
    assert [c.name for c in r.cells] == ["none-local", "tf32_off-local"]
    assert r.cells[0].mitigations == ("none",)
    assert r.cells[1].mitigations == ("tf32_off",)


def test_build_recipe_from_flags_with_ticket_and_baseline():
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none,tf32_off",
        environment_axis="local",
        trials=4,
        steps=100,
        ticket="ABC-1",
        baseline_cell="none-local",
        confound_threshold=1.25,
    )
    assert r.ticket == "ABC-1"
    assert r.confound.baseline_cell == "none-local"
    assert r.confound.threshold == 1.25


def test_build_recipe_from_flags_baseline_unknown_raises():
    with pytest.raises(RecipeCellError, match="does not match any cell"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=2,
            steps=100,
            baseline_cell="not-a-cell",
        )


def test_build_recipe_from_flags_image_prefix_inline_docker():
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="local,image:rocm/pytorch:nightly",
        trials=2,
        steps=100,
    )
    assert len(r.cells) == 2
    assert r.cells[0].environment == "local"
    assert r.cells[1].environment.startswith("_inline_")
    assert r.cells[1].environment == r.inline_environments[0].name
    assert r.inline_environments[0].docker == "rocm/pytorch:nightly"


def test_build_recipe_from_flags_image_prefix_same_as_recipe_mode():
    """Option A (recipe inline docker) and Option B (CLI image:) must agree on auto-name."""
    from aorta.triage.recipe import inline_env_name

    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="image:rocm/pytorch:nightly",
        trials=1,
        steps=10,
    )
    expected_auto = inline_env_name("rocm/pytorch:nightly")
    assert r.cells[0].environment == expected_auto


def test_build_recipe_from_flags_empty_image_prefix_rejected():
    with pytest.raises(RecipeSchemaError, match="requires a ref"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="image:",
            trials=1,
            steps=10,
        )


def test_build_recipe_from_flags_unknown_mitigation():
    with pytest.raises(UnknownMitigationError):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="not_a_mitigation",
            environment_axis="local",
            trials=1,
            steps=10,
        )


def test_build_recipe_from_flags_requires_steps():
    with pytest.raises(RecipeSchemaError, match="--steps is required"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=1,
            steps=None,
        )


# ---- Flag-mode boundary validation (issue #160 review, comment 6) ---------


@pytest.mark.parametrize("trials", [0, -1])
def test_build_recipe_from_flags_rejects_non_positive_trials(trials):
    with pytest.raises(RecipeSchemaError, match="--trials"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=trials,
            steps=10,
        )


@pytest.mark.parametrize("steps", [0, -5])
def test_build_recipe_from_flags_rejects_non_positive_steps(steps):
    with pytest.raises(RecipeSchemaError, match="--steps"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=1,
            steps=steps,
        )


def test_build_recipe_from_flags_rejects_empty_workload():
    with pytest.raises(RecipeSchemaError, match="--workload"):
        build_recipe_from_flags(
            workload="",
            mitigation_axis="none",
            environment_axis="local",
            trials=1,
            steps=10,
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, -0.5])
def test_build_recipe_from_flags_rejects_non_positive_threshold(threshold):
    with pytest.raises(RecipeSchemaError, match="confound-threshold"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=1,
            steps=10,
            confound_threshold=threshold,
        )


@pytest.mark.parametrize("threshold", [0, 0.0, -0.5, -1])
def test_load_recipe_rejects_non_positive_confound_threshold(tmp_path, threshold):
    """Recipe-mode parity with build_recipe_from_flags: threshold must be > 0.

    Without this check the loader accepted thresholds like 0 or -1, which
    make `classify` flag every non-baseline cell as a speed confound.
    """
    text = (
        _MINIMAL_YAML
        + f"""\
confound:
  threshold: {threshold}
"""
    )
    with pytest.raises(RecipeSchemaError, match="confound.threshold"):
        load_recipe(_write_yaml(tmp_path, text))


def test_build_recipe_from_flags_rejects_empty_ticket():
    with pytest.raises(RecipeSchemaError, match="--ticket"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=1,
            steps=10,
            ticket="",
        )


def test_build_recipe_from_flags_empty_axis_rejected():
    with pytest.raises(RecipeSchemaError):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="",
            environment_axis="local",
            trials=1,
            steps=10,
        )


# ---- Dataclass defaults ---------------------------------------------------


def test_confound_cfg_defaults():
    c = ConfoundCfg()
    assert c.threshold == 1.15
    assert c.baseline_cell is None


def test_cell_effective_fallback():
    cell = Cell(name="x", mitigations=("none",), environment="local")
    assert cell.effective_trials(8) == 8
    assert cell.effective_steps(5000) == 5000


def test_cell_effective_override():
    cell = Cell(name="x", mitigations=("none",), environment="local", trials=3, steps=10)
    assert cell.effective_trials(8) == 3
    assert cell.effective_steps(5000) == 10


# ---- Recipe.sidecar_files (round-6 fix) -----------------------------------


def test_load_recipe_records_sidecar_files(tmp_path):
    """Round-6 fix: the loader's ``sidecar_files=`` argument is preserved on
    the returned ``Recipe`` so a programmatic ``run_recipe(recipe)`` call
    works without re-passing the same files.

    Pre-fix, ``Recipe`` discarded that list and any caller that didn't
    redundantly pass ``extra_sidecar_files=`` to the runner hit
    ``UnknownMitigationError`` at execute time.
    """
    sidecar = tmp_path / "ops.sidecar.json"
    sidecar.write_text(
        '{"version": 1, "mitigations": {"my_local_mit": {"FOO": "BAR"}}}',
        encoding="utf-8",
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(
        """\
schema_version: 1
workload: fsdp
trials: 1
steps: 5
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
  - name: sidecar-local
    mitigations: [my_local_mit]
    environment: local
""",
        encoding="utf-8",
    )
    r = load_recipe(path, sidecar_files=(sidecar,))
    assert r.sidecar_files == (sidecar,)


def test_build_recipe_from_flags_records_sidecar_files(tmp_path):
    """Same contract as ``load_recipe``: the flag-mode builder must surface
    ``sidecar_files`` on the resulting ``Recipe``.
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
        sidecar_files=(sidecar,),
    )
    assert r.sidecar_files == (sidecar,)


def test_load_recipe_no_sidecar_files_means_empty_tuple(tmp_path):
    """Default carries a stable empty tuple, not ``None`` -- callers can
    iterate without a None-guard.
    """
    path = tmp_path / "recipe.yaml"
    path.write_text(
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
    r = load_recipe(path)
    assert r.sidecar_files == ()
