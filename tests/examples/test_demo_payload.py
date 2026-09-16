"""The demo payload is a frozen contract, so its shape is a test.

A dashboard is being built against this schema while the emitter is still
moving, which makes "the keys and types are fixed" a claim someone else
depends on. These tests check the shape of a payload assembled from fixtures
written into ``tmp_path`` -- no GPU, no model, no network -- plus the shape of
the real one when the archived artifacts happen to be present.

They also pin the two things about the payload that are easiest to get
subtly wrong: that a ``fake`` episode is never labelled ``litellm``, and that
a value which does not exist is ``null`` rather than a plausible zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rl"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import emit_demo_payload as emit  # noqa: E402

PAYLOAD_KEYS = {
    "generated_at",
    "provenance",
    "scenarios",
    "action_space",
    "episode",
    "policies",
    "weights",
    "caveats",
}
PROVENANCE_KEYS = {
    "slurm_job",
    "node",
    "matrix_path",
    "model",
    "temperature",
    "episode_backend",
}
SCENARIO_KEYS = {
    "id",
    "symptom",
    "true_category",
    "true_resolvers",
    "cells_total",
    "trials_per_cell",
    "authored",
    "notes",
}
STEP_KEYS = {
    "n",
    "action",
    "mitigations",
    "diagnostics",
    "category",
    "hypothesis",
    "confidence",
    "cells_this_step",
    "cells_cumulative",
    "observed",
}
POLICY_NAMES = {
    "oracle",
    "qwen3-8b",
    "abstain_and_pick_first",
    "abstain_and_shotgun",
    "always_prose",
}


# ---------------------------------------------------------------------------
# Episode reconstruction, from a hand-written agent_log.jsonl
# ---------------------------------------------------------------------------


def _write_log(run_dir: Path, events: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "agent_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


def test_an_episode_reports_the_cells_the_loop_charged(tmp_path):
    """Read from the ``axis_growth`` event, not recomputed: the payload must
    show what the loop charged at the time."""
    _write_log(
        tmp_path,
        [
            {"type": "session_start"},
            {
                "type": "llm_step",
                "category": "illegal_mem",
                "hypothesis": "want the log first",
                "next_mitigations": ["tf32_off"],
                "next_diagnostics": ["amd_log_level_4"],
                "confidence": 0.6,
                "stop": False,
            },
            {
                "type": "axis_growth",
                "added_mitigations": ["tf32_off"],
                "added_diagnostics": ["amd_log_level_4"],
                "cells_added": 3,
                "cells_total": 4,
                "mitigation_axis": ["none", "tf32_off"],
                "diagnostic_axis": ["none", "amd_log_level_4"],
            },
            {"type": "converged", "winning_mitigation": "tf32_off"},
        ],
    )
    episode = emit.build_episode(tmp_path, "s")
    assert episode is not None
    assert [s["n"] for s in episode["steps"]] == [1]
    step = episode["steps"][0]
    assert step["action"] == "probe"
    assert step["mitigations"] == ["tf32_off"]
    assert step["diagnostics"] == ["amd_log_level_4"]
    assert step["cells_this_step"] == 3
    assert step["cells_cumulative"] == 4
    assert episode["outcome"] == "resolved"
    assert episode["winning_mitigation"] == "tf32_off"


def test_a_concluding_step_is_kept_even_though_it_grows_no_axis(tmp_path):
    _write_log(
        tmp_path,
        [
            {
                "type": "llm_step",
                "category": "unknown",
                "hypothesis": "done",
                "next_mitigations": [],
                "next_diagnostics": [],
                "confidence": 0.2,
                "stop": True,
            },
            {"type": "search_stopped", "outcome": "agent_stop"},
        ],
    )
    episode = emit.build_episode(tmp_path, "s")
    assert len(episode["steps"]) == 1
    assert episode["steps"][0]["action"] == "conclude"
    assert episode["steps"][0]["cells_this_step"] == 0
    assert episode["outcome"] == "gave_up"


def test_a_missing_log_is_no_episode_rather_than_an_empty_one(tmp_path):
    assert emit.build_episode(tmp_path / "nope", "s") is None


def test_every_step_carries_the_frozen_keys(tmp_path):
    _write_log(
        tmp_path,
        [
            {
                "type": "llm_step",
                "category": "unknown",
                "hypothesis": "h",
                "next_mitigations": ["tf32_off"],
                "confidence": 0.5,
                "stop": False,
            },
            {
                "type": "axis_growth",
                "cells_added": 1,
                "cells_total": 2,
                "mitigation_axis": ["none", "tf32_off"],
                "diagnostic_axis": ["none"],
            },
        ],
    )
    episode = emit.build_episode(tmp_path, "s")
    for step in episode["steps"]:
        assert set(step) == STEP_KEYS


def test_a_log_whose_llm_step_predates_next_diagnostics_reads_as_empty(tmp_path):
    """Backward compatibility, at the payload layer: every log written before
    the joint axis existed has no ``next_diagnostics`` key."""
    _write_log(
        tmp_path,
        [
            {
                "type": "llm_step",
                "category": "unknown",
                "hypothesis": "h",
                "next_mitigations": ["tf32_off"],
                "confidence": 0.5,
                "stop": False,
            },
            {"type": "axis_growth", "cells_added": 1, "cells_total": 2},
        ],
    )
    assert emit.build_episode(tmp_path, "s")["steps"][0]["diagnostics"] == []


def test_a_baseline_pass_log_is_not_offered_as_an_episode(tmp_path, monkeypatch):
    """A passing baseline short-circuits before the proposer is called, so its
    log has no decision steps. Offering it would show a sequence that never
    happened."""
    _write_log(tmp_path / "live" / "DEMO-JOINT-LIVE",
               [{"type": "session_start"}, {"type": "baseline_pass"}])
    monkeypatch.setattr(emit, "EPISODE_ROOT", tmp_path)
    episode, backend, _ = emit.find_episode()
    assert episode is None
    assert backend == "fake"


def test_a_fake_episode_is_never_labelled_as_a_real_one(tmp_path, monkeypatch):
    _write_log(
        tmp_path / "fake" / "DEMO-JOINT-FAKE",
        [
            {
                "type": "llm_step",
                "category": "unknown",
                "hypothesis": "h",
                "next_mitigations": ["tf32_off"],
                "confidence": 0.5,
                "stop": False,
            },
            {"type": "axis_growth", "cells_added": 1, "cells_total": 2},
        ],
    )
    monkeypatch.setattr(emit, "EPISODE_ROOT", tmp_path)
    episode, backend, _ = emit.find_episode()
    assert episode is not None
    assert backend == "fake"


def test_a_real_episode_is_preferred_over_a_fake_one(tmp_path, monkeypatch):
    steps = [
        {
            "type": "llm_step",
            "category": "unknown",
            "hypothesis": "h",
            "next_mitigations": ["tf32_off"],
            "confidence": 0.5,
            "stop": False,
        },
        {"type": "axis_growth", "cells_added": 1, "cells_total": 2},
    ]
    _write_log(tmp_path / "live" / "DEMO-JOINT-LIVE", steps)
    _write_log(tmp_path / "fake" / "DEMO-JOINT-FAKE", steps)
    monkeypatch.setattr(emit, "EPISODE_ROOT", tmp_path)
    _, backend, _ = emit.find_episode()
    assert backend == "litellm"


# ---------------------------------------------------------------------------
# The payload as a whole, against the real archived artifacts
# ---------------------------------------------------------------------------

_HAVE_ARTIFACTS = emit.ROLLOUTS.is_file() and emit.NAN_MATRIX.is_dir()
needs_artifacts = pytest.mark.skipif(
    not _HAVE_ARTIFACTS, reason="archived GPU artifacts not present on this host"
)


@pytest.fixture(scope="module")
def payload():
    return emit.build_payload()


@needs_artifacts
def test_the_payload_has_exactly_the_frozen_top_level_keys(payload):
    assert set(payload) == PAYLOAD_KEYS


@needs_artifacts
def test_provenance_has_the_frozen_keys_and_an_honest_backend(payload):
    assert set(payload["provenance"]) == PROVENANCE_KEYS
    assert payload["provenance"]["episode_backend"] in {"litellm", "fake"}
    assert payload["provenance"]["matrix_path"] == str(emit.NAN_MATRIX)


@needs_artifacts
def test_both_scenarios_are_present_and_one_of_them_is_unresolvable(payload):
    """The second scenario is load-bearing, not filler: it is the only one
    that exercises the empty-resolver-set rule."""
    scenarios = {s["id"]: s for s in payload["scenarios"]}
    assert set(scenarios) == {"uninit_workspace_nan", "fp16_overflow_nan"}
    for scenario in scenarios.values():
        assert set(scenario) == SCENARIO_KEYS
        assert scenario["authored"] is True
        assert scenario["notes"]
    assert scenarios["uninit_workspace_nan"]["true_resolvers"] == [
        "pytorch_no_cuda_memory_caching"
    ]
    assert scenarios["fp16_overflow_nan"]["true_resolvers"] == []


@needs_artifacts
def test_the_nan_scenario_matches_the_archive_it_was_measured_from(payload):
    scenario = next(s for s in payload["scenarios"] if s["id"] == "uninit_workspace_nan")
    assert scenario["cells_total"] == 8
    assert scenario["trials_per_cell"] == 4


@needs_artifacts
def test_true_category_is_null_rather_than_a_plausible_label(payload):
    """Refusing to label is the correct answer here: the symptom is a NaN and
    the cause is memory nobody wrote, and no autopsy category covers that."""
    assert all(s["true_category"] is None for s in payload["scenarios"])


@needs_artifacts
def test_the_action_space_has_both_axes_and_they_do_not_overlap(payload):
    space = payload["action_space"]
    assert set(space) == {"mitigations", "diagnostics"}
    assert space["diagnostics"] == ["amd_log_level_4", "hip_launch_blocking"]
    assert not set(space["mitigations"]) & set(space["diagnostics"])
    assert "none" not in space["mitigations"]


@needs_artifacts
def test_every_policy_carries_both_score_columns(payload):
    for policy in payload["policies"]:
        assert set(policy) == {
            "name", "reads_evidence", "cells_spent", "today", "proposed"
        }
        assert policy["name"] in POLICY_NAMES
        assert set(policy["today"]) == {"total", "form", "triage", "fix"}
        assert set(policy["proposed"]) == {
            "total", "gate_passed", "triage", "fix", "cost_penalty"
        }


@needs_artifacts
def test_the_prose_policy_fails_the_gate_and_the_others_pass_it(payload):
    by_name = {p["name"]: p for p in payload["policies"]}
    assert by_name["always_prose"]["proposed"]["gate_passed"] is False
    assert by_name["oracle"]["proposed"]["gate_passed"] is True


@needs_artifacts
def test_the_weights_in_the_payload_are_the_pre_registered_ones(payload):
    assert payload["weights"] == {"triage": 0.3, "fix": 0.5, "cost": 0.2}


@needs_artifacts
def test_the_caveats_name_the_authored_composition_and_the_sample_size(payload):
    joined = " ".join(payload["caveats"]).lower()
    assert "authored" in joined
    assert "effective sample size" in joined
    # And the part that flatters the proposal has to be in there too.
    assert "flattering" in joined


@needs_artifacts
def test_the_model_beats_every_constant_under_the_proposed_objective(payload):
    """The demo's claim, as an assertion rather than a slide."""
    by_name = {p["name"]: p for p in payload["policies"]}
    model = by_name["qwen3-8b"]["proposed"]["total"]
    constants = [
        p["proposed"]["total"] for p in payload["policies"]
        if not p["reads_evidence"]
    ]
    assert model > max(constants)


@needs_artifacts
def test_the_shotgun_spends_more_cells_than_the_budget_and_is_punished(payload):
    by_name = {p["name"]: p for p in payload["policies"]}
    shotgun = by_name["abstain_and_shotgun"]
    assert shotgun["cells_spent"] > payload["policies"][0].get("cells_spent", 0)
    assert shotgun["proposed"]["cost_penalty"] == pytest.approx(1.0)


@needs_artifacts
def test_the_payload_round_trips_through_json(payload):
    """It is written to disk and read by something else, so anything
    unserialisable is a bug in the contract, not a detail."""
    assert json.loads(json.dumps(payload)) == payload
