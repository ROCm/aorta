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
    _parse_collect,
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


# ---- collect (collector recipes) ------------------------------------------


def test_collect_defaults_empty(tmp_path):
    r = load_recipe(_write_yaml(tmp_path, _MINIMAL_YAML))
    assert r.collect == ()


def test_collect_parsed_from_recipe(tmp_path):
    text = _MINIMAL_YAML + "collect: [layer_numerics]\n"
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect == ("layer_numerics",)


def test_collect_dedups_preserving_order(tmp_path):
    text = _MINIMAL_YAML + "collect: [layer_numerics, rocprof, layer_numerics]\n"
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect == ("layer_numerics", "rocprof")


def test_collect_unknown_recipe_rejected(tmp_path):
    text = _MINIMAL_YAML + "collect: [not_a_collector]\n"
    with pytest.raises(RecipeSchemaError, match="unknown collector recipe"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_wrong_type_rejected(tmp_path):
    text = _MINIMAL_YAML + "collect: layer_numerics\n"  # bare string, not list/mapping
    with pytest.raises(RecipeSchemaError, match="must be a list of collector names"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_recipe_error_uses_recipe_field_label():
    with pytest.raises(RecipeSchemaError, match=r"^recipe\.collect:"):
        _parse_collect("recipe", "layer_numerics")


# ---- collect mapping form (per-collector options) -------------------------


def test_collect_mapping_form_parses_names_and_options(tmp_path):
    text = _MINIMAL_YAML + (
        "collect:\n"
        "  layer_numerics:\n"
        "    NANLOG_SAMPLE_EVERY: \"1\"\n"
        "    NANLOG_PRE_CONTEXT: \"20\"\n"
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect == ("layer_numerics",)
    assert r.collect_options == {
        "layer_numerics": {"NANLOG_SAMPLE_EVERY": "1", "NANLOG_PRE_CONTEXT": "20"}
    }


def test_collect_mapping_form_null_options_enables_without_options(tmp_path):
    text = _MINIMAL_YAML + "collect:\n  layer_numerics:\n"  # enabled, no options
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect == ("layer_numerics",)
    assert r.collect_options == {}  # no entry for an option-less collector


def test_collect_list_form_has_empty_options(tmp_path):
    text = _MINIMAL_YAML + "collect: [layer_numerics]\n"
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect_options == {}


def test_collect_mapping_unknown_recipe_rejected(tmp_path):
    text = _MINIMAL_YAML + "collect:\n  not_a_collector:\n    K: \"v\"\n"
    with pytest.raises(RecipeSchemaError, match="unknown collector recipe"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_mapping_non_string_option_value_rejected(tmp_path):
    text = _MINIMAL_YAML + "collect:\n  layer_numerics:\n    NANLOG_SAMPLE_EVERY: 1\n"
    with pytest.raises(RecipeSchemaError, match="string->string"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_from_flags(tmp_path):
    r = build_recipe_from_flags(
        workload="fsdp",
        mitigation_axis="none",
        environment_axis="local",
        trials=1,
        steps=1,
        collect=("layer_numerics",),
    )
    assert r.collect == ("layer_numerics",)


def test_collect_flag_error_uses_cli_label():
    with pytest.raises(RecipeSchemaError, match=r"^--collect:"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=1,
            steps=1,
            collect=("not_a_collector",),
        )


def test_cell_collect_absent_inherits_recipe_collect(tmp_path):
    text = _MINIMAL_YAML + "collect: [layer_numerics]\n"
    r = load_recipe(_write_yaml(tmp_path, text))
    cell = r.cells[0]
    assert cell.collect is None
    assert cell.collect_options is None
    assert cell.effective_collect(r.collect) == ("layer_numerics",)
    assert cell.effective_collect_options(r.collect_options) == {}


def test_cell_collect_list_form_replaces_recipe_collect(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
collect: [rocprof]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect: [layer_numerics]
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    cell = r.cells[0]
    assert cell.collect == ("layer_numerics",)
    assert cell.collect_options == {}
    assert cell.effective_collect(r.collect) == ("layer_numerics",)
    assert cell.effective_collect_options(r.collect_options) == {}


def test_cell_collect_empty_list_disables_recipe_collect(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
collect: [layer_numerics]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect: []
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    cell = r.cells[0]
    assert cell.collect == ()
    assert cell.effective_collect(r.collect) == ()
    assert cell.effective_collect_options(r.collect_options) == {}


# ---- collect option schemas + cross-collector conflicts -------------------


def test_collect_rocprof_options_parsed(tmp_path):
    text = _MINIMAL_YAML + (
        "collect:\n"
        "  rocprof:\n"
        '    trace: "kernel,hip"\n'
        '    output_format: "csv"\n'
        '    summary_units: "msec"\n'
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect == ("rocprof",)
    assert r.collect_options == {
        "rocprof": {"trace": "kernel,hip", "output_format": "csv", "summary_units": "msec"}
    }


def test_collect_proton_options_parsed(tmp_path):
    text = _MINIMAL_YAML + (
        "collect:\n  proton:\n" '    mode: "cli"\n' '    backend: "roctracer"\n'
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect_options == {"proton": {"mode": "cli", "backend": "roctracer"}}


def test_collect_rocprof_unknown_option_rejected(tmp_path):
    """A typo must fail the recipe up front, not silently produce an
    unprofiled run."""
    text = _MINIMAL_YAML + 'collect:\n  rocprof:\n    traces: "kernel"\n'
    with pytest.raises(RecipeSchemaError, match="rocprof: unknown option"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_rocprof_out_of_domain_value_rejected(tmp_path):
    text = _MINIMAL_YAML + 'collect:\n  rocprof:\n    trace: "gpu"\n'
    with pytest.raises(RecipeSchemaError, match="unknown domain"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_proton_unknown_option_rejected(tmp_path):
    text = _MINIMAL_YAML + 'collect:\n  proton:\n    backends: "roctracer"\n'
    with pytest.raises(RecipeSchemaError, match="proton: unknown option"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_proton_out_of_domain_value_rejected(tmp_path):
    text = _MINIMAL_YAML + 'collect:\n  proton:\n    backend: "rocprof"\n'
    with pytest.raises(RecipeSchemaError, match="proton option 'backend'"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_option_error_is_labelled_recipe_collect(tmp_path):
    with pytest.raises(RecipeSchemaError, match=r"^recipe\.collect:"):
        _parse_collect("recipe", {"rocprof": {"traces": "kernel"}})


def test_collect_rocprof_plus_proton_rejected_at_recipe_scope(tmp_path):
    """Both install an HSA queue interceptor; the second to attach reports
    nothing, so the pairing fails at load rather than producing an empty
    artifact dir."""
    text = _MINIMAL_YAML + "collect: [rocprof, proton]\n"
    with pytest.raises(RecipeSchemaError, match="queue interceptor"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_rocprof_plus_proton_instrumentation_allowed(tmp_path):
    text = _MINIMAL_YAML + 'collect:\n  rocprof:\n  proton:\n    backend: "instrumentation"\n'
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.collect == ("rocprof", "proton")


def test_collect_conflict_rejected_at_cell_scope(tmp_path):
    """A cell-scope ``collect`` replaces the recipe list, so it needs the same
    conflict guard -- otherwise the pairing sneaks in one cell at a time."""
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
collect: [rocprof]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect: [rocprof, proton]
"""
    with pytest.raises(RecipeSchemaError, match=r"^cells\[0\]\.collect:.*queue interceptor"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_cell_scope_option_error_labelled_with_the_cell(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect:
      rocprof:
        summary_units: "minutes"
"""
    with pytest.raises(RecipeSchemaError, match=r"^cells\[0\]\.collect: rocprof option"):
        load_recipe(_write_yaml(tmp_path, text))


def test_collect_flag_mode_rejects_the_conflicting_pair():
    with pytest.raises(RecipeSchemaError, match="queue interceptor"):
        build_recipe_from_flags(
            workload="fsdp",
            mitigation_axis="none",
            environment_axis="local",
            trials=1,
            steps=1,
            collect=("rocprof", "proton"),
        )


def test_cell_collect_null_rejected(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
collect: [layer_numerics]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect:
"""
    with pytest.raises(RecipeSchemaError, match=r"^cells\[0\]\.collect: null"):
        load_recipe(_write_yaml(tmp_path, text))


def test_cell_collect_mapping_form_parses_names_and_options(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
collect: [rocprof]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect:
      layer_numerics:
        NANLOG_SAMPLE_EVERY: "10"
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    cell = r.cells[0]
    assert cell.collect == ("layer_numerics",)
    assert cell.collect_options == {"layer_numerics": {"NANLOG_SAMPLE_EVERY": "10"}}
    assert cell.effective_collect(r.collect) == ("layer_numerics",)
    assert cell.effective_collect_options(r.collect_options) == {
        "layer_numerics": {"NANLOG_SAMPLE_EVERY": "10"}
    }


def test_cell_collect_error_uses_cell_field_label(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
    collect: not_a_list_or_mapping
"""
    with pytest.raises(RecipeSchemaError, match=r"^cells\[0\]\.collect:"):
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


# ---- inline docker env field (#A6) ----------------------------------------
#
# An inline environment may include an ``env`` dict alongside ``docker``.
# The hash covers both the ref and the env content, so two cells with the
# same image but different ``env`` mappings are distinct environments.


def test_inline_docker_env_field_preserved(tmp_path):
    """Env vars declared in an inline environment reach the ``InlineEnv.env``
    field and are not silently dropped."""
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: baseline
    mitigations: [none]
    environment: local
  - name: with-env
    mitigations: [none]
    environment:
      docker: "rocm/pytorch:nightly"
      env:
        TORCH_LOGS: "+recompiles"
        TORCHDYNAMO_VERBOSE: "1"
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    assert len(r.inline_environments) == 1
    inline = r.inline_environments[0]
    assert inline.docker == "rocm/pytorch:nightly"
    assert inline.env == {"TORCH_LOGS": "+recompiles", "TORCHDYNAMO_VERBOSE": "1"}


def test_inline_docker_env_changes_auto_name(tmp_path):
    """Two cells with the same docker ref but different ``env`` dicts must
    produce different auto-names and register as separate environments.

    Without this, a cell that declares ``env: {A: 1}`` could silently collapse
    into another cell's environment that had ``env: {A: 99}``, applying the
    wrong baseline vars at runtime.
    """
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: no-env
    mitigations: [none]
    environment: { docker: "x/y:1" }
  - name: with-env
    mitigations: [none]
    environment:
      docker: "x/y:1"
      env: { EXTRA_FLAG: "1" }
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    assert len(r.inline_environments) == 2
    names = {e.name for e in r.inline_environments}
    assert len(names) == 2
    assert r.cells[0].environment != r.cells[1].environment


def test_inline_docker_same_env_deduplicates(tmp_path):
    """Two cells with identical docker ref AND identical env dict must share a
    single auto-name (dedup) so the environment probe is captured only once."""
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: a
    mitigations: [none]
    environment:
      docker: "x/y:1"
      env: { FOO: "bar" }
  - name: b
    mitigations: [tf32_off]
    environment:
      docker: "x/y:1"
      env: { FOO: "bar" }
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    assert len(r.inline_environments) == 1
    assert r.cells[0].environment == r.cells[1].environment


def test_inline_docker_env_requires_string_values(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: bad-env
    mitigations: [none]
    environment:
      docker: "x/y:1"
      env: { FOO: 1 }
"""
    with pytest.raises(RecipeSchemaError, match=r"environment\.env"):
        load_recipe(_write_yaml(tmp_path, text))


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


# ---- recipe-level extra_env -----------------------------------------------
#
# ``Recipe.extra_env`` is a top-level overlay applied to EVERY cell, sitting
# ABOVE mitigations but BELOW each cell's own ``extra_env``.  The runner
# merges ``{**recipe.extra_env, **cell.extra_env}`` into the single
# ``RunRequest.extra_env`` so a recipe can provide a global baseline without
# repeating it on every cell.


def test_recipe_level_extra_env_parsed(tmp_path):
    """A top-level ``extra_env`` key on the recipe is loaded into ``Recipe.extra_env``.

    This is distinct from the cell-level ``extra_env``: it lives at the recipe
    root and applies to all cells.  The runner merges them; the loader's job
    is to faithfully parse the key and expose it on the ``Recipe`` dataclass.
    """
    text = _MINIMAL_YAML + 'extra_env:\n  GLOBAL_FLAG: "1"\n  NANLOG_TRACE: "on"\n'
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.extra_env == {"GLOBAL_FLAG": "1", "NANLOG_TRACE": "on"}


def test_recipe_level_extra_env_defaults_empty(tmp_path):
    """A recipe with no top-level ``extra_env`` key has ``Recipe.extra_env == {}``.

    Back-compat: every existing recipe has no ``extra_env`` key at the root;
    the loader must not raise and must produce an empty dict, so the runner's
    ``{**recipe.extra_env, **cell.extra_env}`` merge is a no-op for these recipes.
    """
    r = load_recipe(_write_yaml(tmp_path, _MINIMAL_YAML))
    assert r.extra_env == {}


def test_trial_isolation_defaults_auto(tmp_path):
    r = load_recipe(_write_yaml(tmp_path, _MINIMAL_YAML))
    assert r.trial_isolation == "auto"


@pytest.mark.parametrize("mode", ["auto", "in_process", "process"])
def test_trial_isolation_parsed(tmp_path, mode):
    text = _MINIMAL_YAML + f"trial_isolation: {mode}\n"
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.trial_isolation == mode


@pytest.mark.parametrize("bad", ["thread", 1, True, []])
def test_trial_isolation_invalid_value_rejected(tmp_path, bad):
    import yaml

    data = yaml.safe_load(_MINIMAL_YAML)
    data["trial_isolation"] = bad
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(RecipeSchemaError, match="trial_isolation"):
        load_recipe(path)


def test_trial_isolation_is_top_level_only(tmp_path):
    text = _MINIMAL_YAML.replace(
        "    environment: local\n",
        "    environment: local\n    trial_isolation: process\n",
    )
    with pytest.raises(RecipeSchemaError, match="unknown keys.*trial_isolation"):
        load_recipe(_write_yaml(tmp_path, text))


def test_recipe_level_extra_env_non_string_value_rejected(tmp_path):
    """A non-string value in the top-level ``extra_env`` raises ``RecipeSchemaError``.

    Mirrors the cell-level validation: ``extra_env`` is a ``dict[str, str]``
    contract; an integer or list value must be caught at load time with a
    clear error rather than silently propagating a bad type into the runner.
    """
    text = _MINIMAL_YAML + "extra_env:\n  MY_FLAG: 1\n"
    with pytest.raises(RecipeSchemaError, match="extra_env"):
        load_recipe(_write_yaml(tmp_path, text))


def test_recipe_level_extra_env_non_string_key_is_redacted(tmp_path):
    """A secret-bearing YAML key reports only its type, not its decoded content."""
    text = _MINIMAL_YAML + """extra_env:
  !!binary c3VwZXItc2VjcmV0LWtleQ==: "value"
"""
    with pytest.raises(RecipeSchemaError, match="<bytes key>") as exc:
        load_recipe(_write_yaml(tmp_path, text))
    assert "super-secret-key" not in str(exc.value)


# ---- env-var NAME validation at load time ---------------------------------
#
# An invalid env-var NAME (e.g. ``"BAD KEY"``) must fail recipe loading, not
# slip through to run time. Before this, a malformed name passed loading and
# ``--dry-run``; the real run then set no vars per cell while the default CLI
# still exited zero and printed "Wrote matrix". The recipe parser now enforces
# the same ``_ENV_KEY_RE`` shape the dispatcher does, at all three env surfaces.


def test_recipe_level_extra_env_invalid_name_rejected(tmp_path):
    text = _MINIMAL_YAML + 'extra_env:\n  "BAD KEY": "1"\n'
    with pytest.raises(RecipeSchemaError, match="invalid environment-variable name"):
        load_recipe(_write_yaml(tmp_path, text))


def test_cell_level_extra_env_invalid_name_rejected(tmp_path):
    text = _MINIMAL_YAML.replace(
        "    environment: local\n",
        '    environment: local\n    extra_env:\n      "BAD KEY": "1"\n',
    )
    with pytest.raises(RecipeSchemaError, match="invalid environment-variable name"):
        load_recipe(_write_yaml(tmp_path, text))


def test_inline_env_invalid_name_rejected(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: bad-inline-env
    mitigations: [none]
    environment:
      docker: "x/y:1"
      env: { "BAD KEY": "1" }
"""
    with pytest.raises(RecipeSchemaError, match="invalid environment-variable name"):
        load_recipe(_write_yaml(tmp_path, text))


# ---- env-var VALUE (NUL) validation at load time --------------------------
#
# A NUL byte is a valid Python str character but cannot be stored in an OS
# environment variable. Like an invalid name, it must fail at load time (and
# thus ``--dry-run``) instead of only failing per-cell at run time while a
# non-strict sweep still writes a matrix and exits zero. The NUL is written via
# YAML's ``\0`` double-quoted escape.


def test_recipe_level_extra_env_nul_value_rejected(tmp_path):
    text = _MINIMAL_YAML + 'extra_env:\n  TOKEN: "before\\0after"\n'
    with pytest.raises(RecipeSchemaError, match="NUL") as exc:
        load_recipe(_write_yaml(tmp_path, text))
    # Value must not be echoed (it may be a secret).
    assert "before" not in str(exc.value) and "after" not in str(exc.value)


def test_cell_level_extra_env_nul_value_rejected(tmp_path):
    text = _MINIMAL_YAML.replace(
        "    environment: local\n",
        '    environment: local\n    extra_env:\n      TOKEN: "a\\0b"\n',
    )
    with pytest.raises(RecipeSchemaError, match="NUL"):
        load_recipe(_write_yaml(tmp_path, text))


def test_inline_env_nul_value_rejected(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 10
cells:
  - name: nul-inline-env
    mitigations: [none]
    environment:
      docker: "x/y:1"
      env: { TOKEN: "a\\0b" }
"""
    with pytest.raises(RecipeSchemaError, match="NUL"):
        load_recipe(_write_yaml(tmp_path, text))


# ---- Recipe.extra_env is keyword-only (public-API back-compat) ------------
#
# ``Recipe`` is public through ``aorta.triage``. ``extra_env`` was inserted
# before the pre-existing ``stop_after`` / ``probe_extras`` fields, so it MUST
# be keyword-only -- otherwise existing positional callers would silently
# rebind a ``StopAfter`` into ``extra_env``.


def test_recipe_extra_env_is_keyword_only():
    import dataclasses

    fld = next(f for f in dataclasses.fields(Recipe) if f.name == "extra_env")
    assert fld.kw_only is True


def test_recipe_trial_isolation_is_keyword_only():
    import dataclasses

    fld = next(f for f in dataclasses.fields(Recipe) if f.name == "trial_isolation")
    assert fld.kw_only is True


def test_recipe_positional_stop_after_still_binds_to_stop_after():
    """A positional call that previously set ``stop_after`` must still do so.

    Reproduces a pre-existing positional constructor shape: everything through
    ``collect_options`` positionally, then ``stop_after`` positionally. Because
    ``extra_env`` is keyword-only it is skipped by position, so the positional
    ``StopAfter`` binds to ``stop_after`` (not ``extra_env``) as before.
    """
    from aorta.triage.recipe import StopAfter

    sa = StopAfter(events=1, max_trials=3)
    r = Recipe(
        1,              # schema_version
        "fsdp",         # workload
        2,              # trials
        100,            # steps
        (),             # cells
        None,           # ticket
        ConfoundCfg(),  # confound
        (),             # inline_environments
        (),             # sidecar_files
        None,           # source_path
        "abc",          # source_sha256
        {},             # workload_config
        False,          # save_logs
        (),             # collect
        {},             # collect_options
        sa,             # stop_after  <-- positional, MUST NOT land in extra_env
    )
    assert r.stop_after is sa
    assert r.extra_env == {}


# ---- workload_config ------------------------------------------------------


def test_workload_config_defaults_empty(tmp_path):
    r = load_recipe(_write_yaml(tmp_path, _MINIMAL_YAML))
    assert r.workload_config == {}
    assert r.cells[0].workload_config == {}


def test_workload_config_recipe_scope(tmp_path):
    text = _MINIMAL_YAML + "workload_config:\n  shampoo_api: new\n  batch_size: 32\n"
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.workload_config == {"shampoo_api": "new", "batch_size": 32}


def test_workload_config_cell_scope(tmp_path):
    text = _MINIMAL_YAML.replace(
        "    environment: local\n",
        "    environment: local\n    workload_config:\n      shampoo_api: old\n",
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.cells[0].workload_config == {"shampoo_api": "old"}


def test_workload_config_must_be_mapping(tmp_path):
    text = _MINIMAL_YAML + "workload_config: not-a-dict\n"
    with pytest.raises(RecipeSchemaError, match="workload_config"):
        load_recipe(_write_yaml(tmp_path, text))


def test_workload_config_rejects_non_string_keys(tmp_path):
    # YAML int key triggers the str-key validator.
    text = _MINIMAL_YAML + "workload_config:\n  1: foo\n"
    with pytest.raises(RecipeSchemaError, match="keys must be strings"):
        load_recipe(_write_yaml(tmp_path, text))


def test_workload_config_rejects_reserved_steps_key(tmp_path):
    text = _MINIMAL_YAML + "workload_config:\n  steps: 9999\n"
    with pytest.raises(RecipeSchemaError, match="reserved"):
        load_recipe(_write_yaml(tmp_path, text))


def test_workload_config_rejects_aorta_prefix(tmp_path):
    text = _MINIMAL_YAML + "workload_config:\n  _aorta_environment: x\n"
    with pytest.raises(RecipeSchemaError, match="reserved"):
        load_recipe(_write_yaml(tmp_path, text))


def test_workload_config_cell_scope_validation_path(tmp_path):
    text = _MINIMAL_YAML.replace(
        "    environment: local\n",
        "    environment: local\n    workload_config:\n      steps: 1\n",
    )
    with pytest.raises(RecipeSchemaError, match=r"cells\[0\].workload_config"):
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
