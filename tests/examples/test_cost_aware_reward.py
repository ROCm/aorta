"""The cost-aware objective: form as a gate, and cells that cost something.

CPU-only and engine-free throughout. The fix-half ground truth is either a
hand-built :class:`Resolution` or one recovered from a tiny archived matrix
written into ``tmp_path``; no GPU, no container, no model.

The claims under test, in order:

* the four rules -- gate, triage, fix, cost -- each do what they say;
* the special case: a scenario nothing resolves pays for saying so;
* the cost the reward charges is the cost the loop charges, because it is the
  same function;
* the pre-registered weights are the ones in the code;
* and the comparison that motivates the whole thing: the constant that wins
  under today's objective loses under this one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rl"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from cost_aware_reward import (  # noqa: E402
    DEFAULT_BUDGET_CELLS,
    W_COST,
    W_FIX,
    W_TRIAGE,
    cells_for_proposal,
    cost_penalty,
    form_gate,
    max_attainable,
    reported_unresolvable,
    score_cost_aware,
)
from fix_reward import Resolution, resolution_from_matrix  # noqa: E402
from triage_reward import Label  # noqa: E402

OFFERED = ["pytorch_no_cuda_memory_caching", "tf32_off", "hsa_no_sdma", "xnack"]
RESOLVER = "pytorch_no_cuda_memory_caching"


def _raw(**kwargs) -> str:
    body = {
        "category": "unknown",
        "hypothesis": "h",
        "next_mitigations": [],
        "confidence": 0.5,
        "stop": False,
    }
    body.update(kwargs)
    return json.dumps(body)


def _resolved(resolvers=(RESOLVER,)) -> Resolution:
    return Resolution(
        scenario_id="s",
        resolvers=frozenset(resolvers),
        baseline_failed=True,
        cells=8,
        source="test",
    )


def _unresolvable() -> Resolution:
    return Resolution(
        scenario_id="fp16",
        resolvers=frozenset(),
        baseline_failed=True,
        cells=4,
        source="test",
    )


def _score(raw: str, **kwargs):
    defaults = dict(
        offered=OFFERED,
        label=None,
        resolution=_resolved(),
        mitigation_axis=["none"],
        diagnostic_axis=["none"],
        cells_already_spent=1,
    )
    defaults.update(kwargs)
    return score_cost_aware(raw, **defaults)


# ---------------------------------------------------------------------------
# Pre-registration. If these move, the commit message is wrong.
# ---------------------------------------------------------------------------


def test_the_weights_are_the_pre_registered_ones():
    assert (W_TRIAGE, W_FIX, W_COST) == (0.3, 0.5, 0.2)


def test_the_ceiling_is_not_one_and_the_code_says_so():
    """Stated because the side-by-side columns are on different scales and a
    reader comparing values rather than ranks would be misled."""
    assert max_attainable() == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Rule 1: form is a gate, not a score
# ---------------------------------------------------------------------------


def test_a_well_formed_reply_earns_nothing_for_being_well_formed():
    """The change from today, in one assertion.

    ``abstain_and_pick_first`` -- valid JSON, abstains, names one available
    mitigation, reads nothing -- scores 0.9000 on the old form ladder. Here it
    passes the gate and then has to earn its score, and it has read no
    evidence, so it earns the fix term's 0.0 and pays for the cell.
    """
    score = _score(_raw(next_mitigations=["tf32_off"]))
    assert score.gate_passed
    assert score.fix == 0.0
    assert score.total == 0.0


def test_non_json_fails_the_gate_and_stops():
    score = _score("not JSON, just a sentence.")
    assert not score.gate_passed
    assert score.total == 0.0
    # Nothing downstream was *scored* -- the remaining terms would be reading
    # fields whose types were just rejected -- so this proposal added no cells.
    assert score.cells_added == 0


def test_a_gate_failure_still_reports_the_episodes_sunk_cells():
    """They were genuinely spent. Reporting 0 would make a policy that emits
    prose look like the cheapest one on a dashboard. Changes no score."""
    score = _score("not JSON", cells_already_spent=4)
    assert score.cells_spent == 4
    assert score.cost_penalty == pytest.approx(0.4)
    assert score.total == 0.0


def test_a_json_array_is_not_an_object():
    assert not form_gate("[1, 2]", OFFERED).passed


@pytest.mark.parametrize("missing", sorted(["category", "hypothesis", "next_mitigations",
                                            "confidence", "stop"]))
def test_every_required_key_is_required(missing):
    body = json.loads(_raw())
    del body[missing]
    gate = form_gate(json.dumps(body), OFFERED)
    assert not gate.passed
    assert missing in gate.reason


def test_a_bool_confidence_does_not_sneak_past_the_numeric_check():
    """``bool`` is a subclass of ``int``, so ``isinstance(True, (int, float))``
    is True and the check has to exclude it by hand."""
    gate = form_gate(_raw(confidence=True), OFFERED)
    assert not gate.passed
    assert "bool" in gate.reason


def test_a_category_outside_the_closed_set_fails_the_gate():
    gate = form_gate(_raw(category="numeric_instability"), OFFERED)
    assert not gate.passed
    assert "outside the closed set" in gate.reason


def test_an_unregistered_mitigation_fails_the_gate():
    gate = form_gate(_raw(next_mitigations=["PYTORCH_NO_CUDA_MEMORY_CACHING"]), OFFERED)
    assert not gate.passed


def test_a_registered_but_unoffered_name_fails_the_gate():
    """Not a deduction: a name the loop would silently drop makes the
    proposal's cost and effect unpredictable from its text."""
    gate = form_gate(_raw(next_mitigations=["fa_prefer_ck"]), OFFERED)
    assert not gate.passed
    assert "not offered" in gate.reason


def test_an_unoffered_diagnostic_fails_the_gate_too():
    gate = form_gate(_raw(next_diagnostics=["amd_log_level_4"]), OFFERED)
    assert not gate.passed
    assert "diagnostic" in gate.reason


def test_the_baseline_name_is_not_an_offence_on_either_axis():
    """``none`` is dropped by ``validate_step`` because it is already on every
    axis, so a reply naming it is harmless rather than out of bounds."""
    assert form_gate(_raw(next_mitigations=["none"], next_diagnostics=["none"]),
                     OFFERED).passed


def test_a_reply_with_no_diagnostics_field_passes_the_gate():
    """Backward compatibility: every recorded rollout predates the field."""
    assert form_gate(_raw(next_mitigations=["tf32_off"]), OFFERED).passed


# ---------------------------------------------------------------------------
# Rule 2: the triage term is the existing one, not a reimplementation
# ---------------------------------------------------------------------------


def test_the_triage_term_is_triage_rewards_own_arithmetic():
    """0.6 * verdict-correct + 0.4 * attribution-F1, reused verbatim."""
    label = Label(verdict="fail", failure_detectors=["tier4:nan_signature"])
    perfect = _score(
        _raw(verdict="fail", detectors=["tier4:nan_signature"]), label=label
    )
    assert perfect.triage == pytest.approx(1.0)

    verdict_only = _score(_raw(verdict="fail", detectors=[]), label=label)
    assert verdict_only.triage == pytest.approx(0.6)

    wrong = _score(_raw(verdict="pass", detectors=["tier1:exit_nonzero"]), label=label)
    assert wrong.triage == pytest.approx(0.0)


def test_the_triage_term_is_withheld_when_there_is_no_label():
    assert _score(_raw(), label=None).triage == 0.0


def test_omitting_the_verdict_scores_as_wrong_not_as_correct_by_default():
    """The constants emit no verdict at all. Defaulting an absent verdict to
    the label's would hand every policy that declines to read the evidence a
    free 0.6 -- the exact defect this objective exists to remove.
    """
    label = Label(verdict="fail", failure_detectors=["tier4:nan_signature"])
    assert _score(_raw(next_mitigations=["tf32_off"]), label=label).triage == 0.0


# ---------------------------------------------------------------------------
# Rule 3: the fix term is F1 over resolvers, reused from fix_reward
# ---------------------------------------------------------------------------


def test_naming_the_resolver_alone_earns_full_fix_credit():
    assert _score(_raw(next_mitigations=[RESOLVER])).fix == pytest.approx(1.0)


def test_padding_the_resolver_with_wrong_names_costs_fix_credit_and_cells():
    padded = _score(_raw(next_mitigations=[RESOLVER, "tf32_off", "xnack"]))
    assert padded.fix < 1.0
    assert padded.cells_added == 3


def test_the_fix_term_is_withheld_when_the_baseline_did_not_fail():
    resolution = Resolution(
        scenario_id="clean", resolvers=frozenset(), baseline_failed=False,
        cells=2, source="test",
    )
    score = _score(_raw(next_mitigations=[RESOLVER]), resolution=resolution)
    assert score.fix_withheld
    assert score.fix == 0.0


# ---------------------------------------------------------------------------
# The special case: a scenario nothing resolves
# ---------------------------------------------------------------------------


def test_reporting_that_nothing_resolves_it_earns_full_fix_credit():
    """Rule 4 of the brief. Without this the term is 0.0 for every policy on
    such a scenario, which teaches "always guess, guessing is free"."""
    score = _score(
        _raw(next_mitigations=[], stop=True), resolution=_unresolvable()
    )
    assert score.reported_unresolvable
    assert score.fix == pytest.approx(1.0)


def test_guessing_on_an_unresolvable_scenario_earns_nothing():
    score = _score(
        _raw(next_mitigations=[RESOLVER], stop=False), resolution=_unresolvable()
    )
    assert score.fix == 0.0


def test_an_empty_list_without_a_stop_is_not_the_claim():
    """"I have nothing to add" and "nothing in the registry fixes this" are
    different assertions, and only the second one is an answer."""
    assert not reported_unresolvable(
        _step(next_mitigations=[], stop=False)
    )
    score = _score(_raw(next_mitigations=[], stop=False), resolution=_unresolvable())
    assert score.fix == 0.0


def test_the_unresolvable_claim_earns_nothing_on_a_resolvable_scenario():
    """So a policy that always gives up cannot win both scenarios."""
    score = _score(_raw(next_mitigations=[], stop=True), resolution=_resolved())
    assert score.fix == 0.0


def _step(**kwargs):
    from aorta.agent.llm import AgentStep

    body = dict(category="unknown", hypothesis="", next_mitigations=[],
                confidence=0.0, stop=False)
    body.update(kwargs)
    return AgentStep(**body)


# ---------------------------------------------------------------------------
# Rule 4: cost, and that it is the loop's own number
# ---------------------------------------------------------------------------


def test_the_cost_is_computed_by_the_function_the_loop_charges():
    """Same call, same answer -- the reward cannot price an action differently
    from the code that runs it."""
    from aorta.agent.loop import plan_axis_growth

    direct = plan_axis_growth(["none", "a"], ["none"], ["b", "c"], ["x"]).cells_added
    assert cells_for_proposal(["b", "c"], ["x"], ["none", "a"], ["none"]) == direct


def test_a_diagnostic_costs_more_than_its_own_cell():
    """The cross product, priced. Three mitigations on the axis and one new
    diagnostic is three cells, so an objective that charged one per name would
    under-count by 3x here."""
    assert cells_for_proposal([], ["amd_log_level_4"], ["none", "a", "b"], ["none"]) == 3


def test_cost_penalty_is_the_cell_count_over_the_budget():
    assert cost_penalty(5, 10) == pytest.approx(0.5)
    assert cost_penalty(10, 10) == pytest.approx(1.0)


def test_cost_penalty_is_clipped_rather_than_unbounded():
    """Letting it grow would let cost dominate two bounded terms, which is a
    weight change by the back door."""
    assert cost_penalty(200, 10) == pytest.approx(1.0)


def test_a_zero_budget_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="budget_cells"):
        cost_penalty(1, 0)


def test_spending_more_cells_for_the_same_answer_scores_strictly_less():
    """The whole point of the objective, as a single comparison."""
    lean = _score(_raw(next_mitigations=[RESOLVER]))
    padded = _score(_raw(next_mitigations=[RESOLVER, "tf32_off", "xnack", "hsa_no_sdma"]))
    assert lean.total > padded.total
    assert lean.cells_spent < padded.cells_spent


def test_cells_already_spent_in_the_episode_are_charged():
    """Cost is per *episode*, not per step: a policy cannot reset the meter by
    proposing cheaply after an expensive start."""
    early = _score(_raw(next_mitigations=[RESOLVER]), cells_already_spent=1)
    late = _score(_raw(next_mitigations=[RESOLVER]), cells_already_spent=8)
    assert late.total < early.total


# ---------------------------------------------------------------------------
# The composite, and the floor
# ---------------------------------------------------------------------------


def test_the_total_is_the_weighted_sum_minus_the_cost():
    label = Label(verdict="fail", failure_detectors=["tier4:nan_signature"])
    score = _score(
        _raw(next_mitigations=[RESOLVER], verdict="fail",
             detectors=["tier4:nan_signature"]),
        label=label,
    )
    expected = 0.3 * 1.0 + 0.5 * 1.0 - 0.2 * (2 / DEFAULT_BUDGET_CELLS)
    assert score.total == pytest.approx(expected)


def test_the_total_is_floored_so_garbage_cannot_outscore_an_attempt():
    """Without the floor a gate failure (0.0) would beat a well-formed answer
    that earned little and spent a lot, which teaches the model to emit
    garbage rather than try."""
    expensive_and_wrong = _score(
        _raw(next_mitigations=["tf32_off", "xnack", "hsa_no_sdma"])
    )
    assert expensive_and_wrong.total == 0.0
    assert _score("garbage").total == 0.0


def test_the_floor_creates_a_known_flat_spot_and_this_pins_it():
    """A finding, not a desideratum. Several distinct policies that earn
    nothing all tie at 0.0, so GRPO sees no advantage between them. Recorded
    here rather than fixed by moving a weight; fixing it means changing what a
    gate failure is worth, which is a design decision, not a coefficient.
    """
    a = _score(_raw(next_mitigations=["tf32_off"]))
    b = _score(_raw(next_mitigations=["tf32_off", "xnack", "hsa_no_sdma"]))
    assert a.total == b.total == 0.0
    assert a.cells_spent != b.cells_spent  # they are NOT the same behaviour


# ---------------------------------------------------------------------------
# The comparison that motivates the change
# ---------------------------------------------------------------------------


def test_the_constant_that_wins_today_loses_under_the_cost_aware_objective():
    """Today: ``abstain_and_shotgun`` scores 0.7700 under containment and
    ``abstain_and_pick_first`` 0.9000 on the form ladder, both above
    Qwen3-8B. Here both are beaten by a policy that names the resolver.
    """
    label = Label(verdict="fail", failure_detectors=["tier4:nan_signature"])
    common = dict(label=label)
    shotgun = _score(_raw(next_mitigations=OFFERED), **common)
    pick_first = _score(_raw(next_mitigations=OFFERED[1:2]), **common)
    diagnostician = _score(
        _raw(next_mitigations=[RESOLVER], verdict="fail",
             detectors=["tier4:nan_signature"]),
        **common,
    )
    assert diagnostician.total > shotgun.total
    assert diagnostician.total > pick_first.total


# ---------------------------------------------------------------------------
# Against a real archived matrix shape, still without a GPU
# ---------------------------------------------------------------------------


def _write_cell(root: Path, cell: str, verdict: str, detectors: list[str]) -> None:
    trial = root / cell / "trial_0"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "cell_name": cell,
                "verdict": verdict,
                "failure_detectors_fired": detectors,
                "error_detectors_fired": [],
                "warn_detectors_fired": [],
            }
        ),
        encoding="utf-8",
    )


def test_the_resolver_set_is_recovered_from_cell_directories_and_scored(tmp_path):
    """End to end on the shape of the real archive: the fix half is an offline
    lookup, so ground truth costs a directory listing and no GPU."""
    matrix = tmp_path / "PROBE-NAN-WS-MATRIX"
    _write_cell(matrix, "none-none", "fail", ["tier4:nan_signature"])
    _write_cell(matrix, "tf32_off-none", "fail", ["tier4:nan_signature"])
    _write_cell(matrix, f"{RESOLVER}-none", "pass", [])

    resolution = resolution_from_matrix(matrix)
    assert resolution.resolvers == frozenset({RESOLVER})
    assert resolution.baseline_failed

    named = _score(_raw(next_mitigations=[RESOLVER]), resolution=resolution)
    missed = _score(_raw(next_mitigations=["tf32_off"]), resolution=resolution)
    assert named.total > missed.total
