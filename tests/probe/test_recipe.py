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


def test_custom_patterns_accepted_in_probe_mode(tmp_path):
    """Phase 2: ``custom_patterns`` loads in probe-mode (rubric §2.C)."""
    text = (
        _PROBE_MINIMAL
        + """\
custom_patterns:
  - id: hip_oom
    match:
      regex: "hipError_OutOfMemory"
"""
    )
    recipe = load_recipe(_write_yaml(tmp_path, text))
    assert recipe.probe_extras is not None
    assert len(recipe.probe_extras.custom_patterns) == 1
    assert recipe.probe_extras.custom_patterns[0].detector_id == "custom:hip_oom"


def test_custom_patterns_rejected_in_triage_mode(tmp_path):
    """``custom_patterns`` is probe-mode-only; triage-mode rejects it."""
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 1
cells:
  - name: c
    mitigations: [none]
    environment: local
custom_patterns:
  - id: hip_oom
    match:
      regex: "hipError_OutOfMemory"
"""
    with pytest.raises(RecipeSchemaError, match="probe-mode only"):
        load_recipe(_write_yaml(tmp_path, text))


def test_phase_3_keys_rejected_with_pointer(tmp_path):
    text = (
        _PROBE_MINIMAL
        + """\
condition:
  - "exit_code != 0"
"""
    )
    with pytest.raises(RecipeSchemaError) as exc:
        load_recipe(_write_yaml(tmp_path, text))
    msg = str(exc.value)
    assert "Phase 3" in msg
    assert "condition" in msg


def test_redaction_block_accepted_in_probe_mode(tmp_path):
    text = (
        _PROBE_MINIMAL
        + """\
redaction:
  scrub_env_keys: ["AWS_*"]
  scrub_paths: true
  scrub_ip_addresses: true
"""
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.probe_extras is not None
    assert r.probe_extras.redaction is not None
    assert r.probe_extras.redaction.scrub_paths is True


def test_redaction_block_rejected_in_triage_mode(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 1
steps: 1
cells:
  - name: baseline
    mitigations: [none]
    environment: local
redaction:
  scrub_paths: true
"""
    with pytest.raises(RecipeSchemaError, match="probe-mode only"):
        load_recipe(_write_yaml(tmp_path, text))


def test_redaction_block_unknown_key_rejected(tmp_path):
    text = (
        _PROBE_MINIMAL
        + """\
redaction:
  scrub_everything: true
"""
    )
    with pytest.raises(RecipeSchemaError, match="unknown keys"):
        load_recipe(_write_yaml(tmp_path, text))


def test_redaction_null_rejected(tmp_path):
    """``redaction: null`` is present-but-invalid, not "no redaction".

    Using ``data.get("redaction")`` conflated an explicit null with a
    missing key and silently disabled scrubbing. The loader now checks
    key presence so a null block fails validation (oyazdanb review).
    """
    text = _PROBE_MINIMAL + "redaction:\n"
    with pytest.raises(RecipeSchemaError, match="must be a mapping when present"):
        load_recipe(_write_yaml(tmp_path, text))


def test_top_level_condition_rejected_as_phase_3(tmp_path):
    """Top-level ``condition:`` (outside custom_patterns[*].match) is Phase 3."""
    text = (
        _PROBE_MINIMAL
        + """\
condition:
  - "exit_code != 0"
"""
    )
    with pytest.raises(RecipeSchemaError, match="Phase 3"):
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


# ---- issue #229: detector-disable knobs ----------------------------------


def test_disable_detector_knobs_parse(tmp_path):
    text = _PROBE_MINIMAL + (
        "disable_detectors: [tier2:hang, custom:my_id]\n"
        "disable_detector_tiers: [tier3]\n"
    )
    r = load_recipe(_write_yaml(tmp_path, text))
    assert r.probe_extras is not None
    assert r.probe_extras.disable_detectors == ("tier2:hang", "custom:my_id")
    assert r.probe_extras.disable_detector_tiers == ("tier3",)


def test_disable_detector_knobs_default_empty(tmp_path):
    r = load_recipe(_write_yaml(tmp_path, _PROBE_MINIMAL))
    assert r.probe_extras is not None
    assert r.probe_extras.disable_detectors == ()
    assert r.probe_extras.disable_detector_tiers == ()


def test_disable_detector_bad_id_rejected(tmp_path):
    text = _PROBE_MINIMAL + "disable_detectors: [hang]\n"
    with pytest.raises(RecipeSchemaError, match="malformed detector id"):
        load_recipe(_write_yaml(tmp_path, text))


def test_disable_detector_tier_unknown_rejected(tmp_path):
    text = _PROBE_MINIMAL + "disable_detector_tiers: [tier9]\n"
    with pytest.raises(RecipeSchemaError, match="unknown tier"):
        load_recipe(_write_yaml(tmp_path, text))


def test_disable_detector_keys_rejected_in_triage_mode(tmp_path):
    text = """\
schema_version: 1
workload: fsdp
trials: 1
disable_detectors: [tier2:hang]
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
"""
    with pytest.raises(RecipeSchemaError, match="probe-mode only"):
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


def test_phase_2_fixture_loads_in_probe_mode():
    """Phase 2: ``custom_patterns`` recipe loads cleanly (rubric §2.B FR 2.6)."""
    recipe = load_recipe(FIXTURES / "probe_with_phase_2_keys.yaml")
    assert recipe.probe_extras is not None
    assert len(recipe.probe_extras.custom_patterns) == 1


def test_hang_grace_period_zero_is_accepted(tmp_path):
    """Regression for PR #197 review: ``hang_grace_period_at_start: 0`` is
    a documented runtime value -- ``HangMonitor`` / ``evaluate_predicate``
    treat it as "no grace, fire as soon as the window elapses", useful
    for short-running repros where 60s of grace swallows the trial.
    The recipe validator used to reject it via the strict ``> 0`` check.
    ``hang_window_sec`` keeps the strict bound because a zero window
    would re-trip the predicate on every poll.
    """
    recipe_path = tmp_path / "probe_zero_grace.yaml"
    recipe_path.write_text(
        """\
schema_version: 1
mode: probe
ticket: ZERO-GRACE
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
hang_grace_period_at_start: 0
""",
        encoding="utf-8",
    )
    recipe = load_recipe(recipe_path)
    assert recipe.probe_extras is not None
    assert recipe.probe_extras.hang_grace_period_at_start == 0.0


def test_hang_window_sec_zero_still_rejected(tmp_path):
    """``hang_window_sec: 0`` stays rejected -- a zero window would
    re-trip the predicate on every poll. Keeps the strict ``> 0`` bound
    asymmetric with grace (which DOES allow 0).
    """
    recipe_path = tmp_path / "probe_zero_window.yaml"
    recipe_path.write_text(
        """\
schema_version: 1
mode: probe
ticket: ZERO-WINDOW
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
hang_window_sec: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(RecipeSchemaError, match="hang_window_sec.*> 0"):
        load_recipe(recipe_path)


def test_hang_grace_period_negative_is_rejected(tmp_path):
    """Even with ``allow_zero=True`` the validator still rejects
    negative values -- the predicate would silently ignore a negative
    grace period (clamped by ``elapsed > grace_period_sec``).
    """
    recipe_path = tmp_path / "probe_negative_grace.yaml"
    recipe_path.write_text(
        """\
schema_version: 1
mode: probe
ticket: NEG-GRACE
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
hang_grace_period_at_start: -1
""",
        encoding="utf-8",
    )
    with pytest.raises(RecipeSchemaError, match="hang_grace_period_at_start.*>= 0"):
        load_recipe(recipe_path)


# ---- tier3_vram_growth opt-out round-trip (PR #215 review) ----------------


def test_tier3_vram_growth_defaults_true(tmp_path):
    """Omitting ``tier3_vram_growth`` leaves the Tier-3 VRAM delta check
    enabled -- the knob defaults to ``True`` so existing recipes are
    unchanged. Pins the recipe -> ``ProbeExtras`` path that the original
    PR #215 left untested (Sonbol review).
    """
    recipe = load_recipe(_write_yaml(tmp_path, _PROBE_MINIMAL))
    assert recipe.probe_extras is not None
    assert recipe.probe_extras.tier3_vram_growth is True


def test_tier3_vram_growth_false_round_trips(tmp_path):
    """``tier3_vram_growth: false`` parses to a real bool and lands on
    ``ProbeExtras.tier3_vram_growth`` so the dispatcher can forward it into
    ``probe_extras`` and the workload can skip the whole-device VRAM check.
    """
    text = _PROBE_MINIMAL + "tier3_vram_growth: false\n"
    recipe = load_recipe(_write_yaml(tmp_path, text))
    assert recipe.probe_extras is not None
    assert recipe.probe_extras.tier3_vram_growth is False


def test_tier3_vram_growth_non_bool_rejected(tmp_path):
    """A non-bool ``tier3_vram_growth`` is rejected at load. Truthiness
    coercion (``bool("false") is True``) would silently re-enable the
    detector, so the builder fails closed with ``RecipeSchemaError``.
    """
    text = _PROBE_MINIMAL + 'tier3_vram_growth: "notabool"\n'
    with pytest.raises(RecipeSchemaError, match="tier3_vram_growth.*must be a boolean"):
        load_recipe(_write_yaml(tmp_path, text))
