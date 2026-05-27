"""Tests for the probe-mode recipe loader (issue #188 Phase 1).

The probe-mode loader lives in :mod:`aorta.probe.recipe_builder` and is
dispatched from :func:`aorta.triage.recipe._build_recipe` when
``data["mode"] == "probe"``. These tests pin the rubric's FR 1.6-1.8,
1.16, and 1.19 contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.probe.recipe_builder import (
    SUBPROCESS_WORKLOAD_NAME,
    ProbeExtras,
    build_probe_recipe_from_dict,
)
from aorta.registry.errors import UnknownMitigationError
from aorta.triage.recipe import (
    Recipe,
    RecipeCellError,
    RecipeSchemaError,
    load_recipe,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _write_yaml(tmp_path: Path, text: str, name: str = "r.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


_PROBE_MINIMAL = """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
"""

_PROBE_TWO_CELL = """\
schema_version: 1
mode: probe
trials: 2
mitigation_axis: [none, tf32_off]
diagnostic_axis: [none]
"""


# ---- FR 1.16 (back-compat) -----------------------------------------------


def test_mode_defaults_to_triage_and_existing_recipes_load(tmp_path):
    """Existing triage-mode recipes (no 'mode:' key) load byte-equivalently."""
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
"""
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.workload == "fsdp"
    assert r.probe_extras is None  # triage-mode never populates this
    assert len(r.cells) == 1


# ---- FR 1.6 (unknown / misplaced keys) -----------------------------------


def test_rejects_unknown_top_level(tmp_path):
    text = _PROBE_MINIMAL + "garbage: 42\n"
    with pytest.raises(RecipeSchemaError, match="unknown top-level keys"):
        load_recipe(_write_yaml(tmp_path, text))


def test_rejects_probe_keys_in_triage_mode(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 2
steps: 100
mitigation_axis: [none]
diagnostic_axis: [none]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
"""
    with pytest.raises(RecipeSchemaError, match="probe-mode only"):
        load_recipe(_write_yaml(tmp_path, text))


def test_rejects_triage_keys_in_probe_mode(tmp_path):
    text = (
        _PROBE_MINIMAL
        + """\
workload: fsdp
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
"""
    )
    with pytest.raises(RecipeSchemaError, match="triage-mode only"):
        load_recipe(_write_yaml(tmp_path, text))


def test_rejects_invalid_mode_value(tmp_path):
    text = """\
schema_version: 1
mode: bogus
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
"""
    with pytest.raises(RecipeSchemaError, match=r"mode.*'triage' or 'probe'"):
        load_recipe(_write_yaml(tmp_path, text))


# ---- FR 1.19 (Phase 2/3 rejection with pointer) --------------------------


def test_phase_2_3_keys_rejected_with_pointer(tmp_path):
    """Phase 2 keys raise with a 'deferred to Phase 2' pointer to the rubric."""
    text = (
        _PROBE_MINIMAL
        + """\
custom_patterns:
  - id: hip_oom
    match:
      regex: "hipErrorOutOfMemory"
"""
    )
    with pytest.raises(RecipeSchemaError) as exc:
        load_recipe(_write_yaml(tmp_path, text))
    msg = str(exc.value)
    assert "Phase 2" in msg
    assert "custom_patterns" in msg
    assert "aorta-probe-188-rubric.md" in msg


def test_phase_3_keys_rejected_with_pointer(tmp_path):
    text = (
        _PROBE_MINIMAL
        + """\
redaction:
  scrub_env_keys: ["AWS_*"]
"""
    )
    with pytest.raises(RecipeSchemaError) as exc:
        load_recipe(_write_yaml(tmp_path, text))
    msg = str(exc.value)
    assert "Phase 3" in msg
    assert "redaction" in msg


def test_condition_key_rejected_with_pointer(tmp_path):
    text = (
        _PROBE_MINIMAL
        + """\
condition:
  - "exit_code != 0"
"""
    )
    with pytest.raises(RecipeSchemaError, match="Phase 2"):
        load_recipe(_write_yaml(tmp_path, text))


# ---- FR 1.7 (cell synthesis and collision) -------------------------------


def test_cell_synthesis_and_collision(tmp_path):
    """Cells are synthesised as cartesian product; slug-collisions rejected."""
    r = load_recipe(_write_yaml(tmp_path, _PROBE_TWO_CELL))
    assert isinstance(r, Recipe)
    cell_names = sorted(c.name for c in r.cells)
    assert cell_names == ["none-none", "tf32_off-none"]
    # Mitigations carry BOTH axes -- the dispatcher unions the env bundles.
    for cell in r.cells:
        assert len(cell.mitigations) == 2
    # Workload pinned to the platform-internal name.
    assert r.workload == SUBPROCESS_WORKLOAD_NAME
    # probe_extras populated with the axes for the dry-run formatter.
    assert isinstance(r.probe_extras, ProbeExtras)
    assert r.probe_extras.mitigation_axis == ("none", "tf32_off")
    assert r.probe_extras.diagnostic_axis == ("none",)


def test_cell_name_collision_rejected(tmp_path):
    """Axis values that slug to the same cell name are rejected at load."""
    # ``foo-bar`` and ``foo`` paired with ``bar`` would both slug to
    # "foo-bar" if the cell-name builder didn't reject. Use synthetic
    # collisions via a mock registry to avoid registry coupling.
    from unittest.mock import patch

    with patch("aorta.probe.recipe_builder.get_mitigation", return_value={}):
        with pytest.raises(RecipeCellError, match="slug to"):
            build_probe_recipe_from_dict(
                {
                    "schema_version": 1,
                    "mode": "probe",
                    "trials": 1,
                    # Two pairs slug to the same name:
                    "mitigation_axis": ["a-b", "a"],
                    "diagnostic_axis": ["c", "b-c"],
                },
                sidecar_files=None,
            )


# ---- FR 1.8 (axis name resolution) ---------------------------------------


def test_axis_unknown_name_rejected(tmp_path):
    text = """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [no_such_mitigation]
diagnostic_axis: [none]
"""
    with pytest.raises(UnknownMitigationError):
        load_recipe(_write_yaml(tmp_path, text))


# ---- Validation -----------------------------------------------------------


def test_missing_required_probe_keys(tmp_path):
    text = """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [none]
"""
    with pytest.raises(RecipeSchemaError, match="missing required key 'diagnostic_axis'"):
        load_recipe(_write_yaml(tmp_path, text))


def test_step_time_regex_compile_validated(tmp_path):
    text = _PROBE_MINIMAL + "step_time_regex: '(unclosed group'\n"
    with pytest.raises(RecipeSchemaError, match="invalid regex"):
        load_recipe(_write_yaml(tmp_path, text))


def test_env_passthrough_mode_validated(tmp_path):
    text = _PROBE_MINIMAL + "env_passthrough_mode: bogus\n"
    with pytest.raises(RecipeSchemaError, match="inherit.*file"):
        load_recipe(_write_yaml(tmp_path, text))


def test_timeout_per_trial_must_be_positive(tmp_path):
    text = _PROBE_MINIMAL + "timeout_per_trial: 0\n"
    with pytest.raises(RecipeSchemaError, match="must be > 0"):
        load_recipe(_write_yaml(tmp_path, text))


# ---- PR #194 round-3 review: probe confound rejects unknown keys ---------


def test_probe_confound_unknown_key_rejected(tmp_path):
    """Probe-mode ``confound`` must reject typo'd keys like ``baseline_cel``.

    Regression for PR #194 review: the probe-mode parser previously
    used ``confound_raw.get("baseline_cell", default)`` which silently
    ignored typo'd keys, leaving the auto-derived baseline in place
    and weakening the loader's strict-schema contract that the
    triage-mode parser (:func:`aorta.triage.recipe._parse_confound`)
    enforces. Both parsers now share ``_VALID_CONFOUND_KEYS`` so a new
    confound key has to be opted into both code paths explicitly.
    """
    text = _PROBE_MINIMAL + "confound:\n  baseline_cel: none-none\n"
    with pytest.raises(RecipeSchemaError, match="unknown keys.*baseline_cel"):
        load_recipe(_write_yaml(tmp_path, text))


def test_probe_confound_known_keys_still_accepted(tmp_path):
    """Verify the strict-schema check doesn't reject the legitimate
    ``threshold`` / ``baseline_cell`` keys (upper bound on the fix).
    """
    text = _PROBE_MINIMAL + "confound:\n  threshold: 1.5\n  baseline_cell: none-none\n"
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.confound.threshold == 1.5
    assert r.confound.baseline_cell == "none-none"


# ---- Fixture loads -------------------------------------------------------


def test_minimal_fixture_loads():
    r = load_recipe(FIXTURES / "probe_minimal.yaml")
    assert r.workload == SUBPROCESS_WORKLOAD_NAME
    assert len(r.cells) == 1


def test_phase_2_fixture_rejected():
    with pytest.raises(RecipeSchemaError, match="Phase 2"):
        load_recipe(FIXTURES / "probe_with_phase_2_keys.yaml")
