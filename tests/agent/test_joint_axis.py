"""The joint-axis action: the agent grows the diagnostic axis, and pays for it.

Two things are under test and they are different claims. That the diagnostic
is now an *action* -- a ``next_diagnostics`` field the model can fill and a
loop that widens ``diagnostic_axis`` the way it already widened
``mitigation_axis``. And that the action is *priced correctly* -- a probe
recipe's cells are the cartesian product of the two axes, so a joint proposal
costs ``a*D + b*M + a*b`` cells rather than ``a + b``, and the number the code
charges has to be the number of cells it runs.

Everything here is CPU-only and engine-free: no GPU, no container, no model.
The loop tests drive ``run_agent_loop`` with a stub proposer and a probe
recipe whose workload is ``true``/``false``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aorta.agent.llm import AgentStep, FakeLLMProposer, _build_prompt, _step_from_content
from aorta.agent.loop import (
    AgentConfig,
    plan_axis_growth,
    probe_cell_names,
    run_agent_loop,
)
from aorta.agent.policy import AgentPolicy, PolicyViolation
from aorta.agent.state import wake

# ---------------------------------------------------------------------------
# AgentStep.from_dict -- the schema change, and that it is backward compatible
# ---------------------------------------------------------------------------


def test_a_reply_with_no_next_diagnostics_parses_to_an_empty_list():
    """The backward-compatibility claim, at the parse layer.

    Every recorded rollout in this project was generated against the
    single-axis schema. If the absent key parsed to anything but ``[]`` those
    replies would start proposing diagnostics they never named.
    """
    step = AgentStep.from_dict(
        {
            "category": "unknown",
            "hypothesis": "h",
            "next_mitigations": ["tf32_off"],
            "confidence": 0.5,
            "stop": False,
        }
    )
    assert step.next_diagnostics == []
    assert step.next_mitigations == ["tf32_off"]


def test_a_bare_string_diagnostic_is_not_exploded_into_characters():
    """The same trap ``next_mitigations`` is already defended against.

    ``list("xnack")`` is ``['x', 'n', 'a', 'c', 'k']`` -- five candidate names,
    none of them registered. Only a genuine JSON list is accepted.
    """
    step = AgentStep.from_dict(
        {
            "category": "unknown",
            "hypothesis": "h",
            "next_mitigations": [],
            "next_diagnostics": "xnack",
            "confidence": 0.5,
            "stop": False,
        }
    )
    assert step.next_diagnostics == []


@pytest.mark.parametrize("bad", [None, 3, {"a": 1}, True])
def test_a_non_list_diagnostic_field_falls_back_to_empty(bad):
    step = AgentStep.from_dict(
        {"category": "unknown", "hypothesis": "", "next_diagnostics": bad}
    )
    assert step.next_diagnostics == []


def test_diagnostic_names_are_coerced_to_str_like_mitigations_are():
    step = AgentStep.from_dict(
        {"category": "unknown", "hypothesis": "", "next_diagnostics": [1, "xnack"]}
    )
    assert step.next_diagnostics == ["1", "xnack"]


# ---------------------------------------------------------------------------
# _step_from_content -- the model cannot widen its own diagnostic allowlist
# ---------------------------------------------------------------------------


def _reply(**kwargs) -> str:
    body = {
        "category": "unknown",
        "hypothesis": "h",
        "next_mitigations": [],
        "confidence": 0.5,
        "stop": False,
    }
    body.update(kwargs)
    return json.dumps(body)


def test_an_unoffered_diagnostic_is_filtered_out():
    step = _step_from_content(
        _reply(next_diagnostics=["xnack", "amd_log_level_4"]),
        remaining=[],
        remaining_diagnostics=["amd_log_level_4"],
    )
    assert step.next_diagnostics == ["amd_log_level_4"]


def test_diagnostics_are_dropped_entirely_when_the_caller_offered_none():
    """Not leniency: a caller that named no diagnostic axis authorised none.

    The policy is unchanged; only how the caller says it has. This used to
    rely on ``remaining_diagnostics`` defaulting to None, which made "offering
    nothing" indistinguishable from omitting the argument, so the empty list
    is now passed explicitly and the drop is recorded rather than silent.
    """
    step = _step_from_content(
        _reply(next_diagnostics=["xnack"]), remaining=[], remaining_diagnostics=[]
    )
    assert step.next_diagnostics == []
    assert step.unresolved_diagnostics == ["xnack"]


# ---------------------------------------------------------------------------
# _build_prompt -- the second axis is described only when it is armed
# ---------------------------------------------------------------------------


def test_the_prompt_is_byte_identical_when_no_diagnostic_is_offered():
    """A single-axis run must not pay a prompt change for a feature it is not using."""
    without = _build_prompt("sym", [], ["tf32_off"], [])
    explicit_empty = _build_prompt("sym", [], ["tf32_off"], [], [], [])
    assert without == explicit_empty
    system, user = without
    assert "next_diagnostics" not in system
    assert "diagnostic_candidates" not in user


def test_the_prompt_names_the_second_axis_and_its_cost_when_armed():
    system, user = _build_prompt("sym", [], ["tf32_off"], [], ["amd_log_level_4"], [])
    assert "next_diagnostics" in system
    # The cross product is the thing a cost-aware policy has to know about.
    assert "cross product" in system
    assert json.loads(user)["diagnostic_candidates"] == ["amd_log_level_4"]


# ---------------------------------------------------------------------------
# FakeLLMProposer -- offline default unchanged unless a diagnostic is offered
# ---------------------------------------------------------------------------


def test_the_fake_proposer_proposes_no_diagnostic_when_none_is_offered():
    step = FakeLLMProposer().propose(
        symptom=None, cell_summaries=[], candidates=["tf32_off"], tried=[]
    )
    assert step.next_diagnostics == []


def test_the_fake_proposer_takes_one_diagnostic_at_a_time_when_offered():
    step = FakeLLMProposer().propose(
        symptom=None,
        cell_summaries=[],
        candidates=["tf32_off"],
        tried=[],
        diagnostic_candidates=["amd_log_level_4", "xnack"],
        tried_diagnostics=[],
    )
    assert step.next_diagnostics == ["amd_log_level_4"]
    step_two = FakeLLMProposer().propose(
        symptom=None,
        cell_summaries=[],
        candidates=["tf32_off"],
        tried=[],
        diagnostic_candidates=["amd_log_level_4", "xnack"],
        tried_diagnostics=["amd_log_level_4"],
    )
    assert step_two.next_diagnostics == ["xnack"]


# ---------------------------------------------------------------------------
# policy.validate_step -- both axes get the same guardrails
# ---------------------------------------------------------------------------


def test_validate_step_carries_diagnostics_through_and_dedupes_them():
    step = AgentPolicy().validate_step(
        AgentStep(
            category="unknown",
            hypothesis="h",
            next_mitigations=["tf32_off"],
            confidence=0.5,
            stop=False,
            next_diagnostics=["xnack", "xnack"],
        )
    )
    assert step.next_diagnostics == ["xnack"]


def test_the_baseline_diagnostic_is_dropped_rather_than_charged_for():
    """``none`` is on every diagnostic axis by construction, so admitting it
    would charge cells for cells that already exist."""
    step = AgentPolicy().validate_step(
        AgentStep(
            category="unknown",
            hypothesis="h",
            next_mitigations=[],
            confidence=0.5,
            stop=False,
            next_diagnostics=["none", "xnack"],
        )
    )
    assert step.next_diagnostics == ["xnack"]


def test_an_unregistered_diagnostic_is_a_policy_violation():
    with pytest.raises(PolicyViolation, match="not-a-real-mitigation"):
        AgentPolicy().validate_step(
            AgentStep(
                category="unknown",
                hypothesis="h",
                next_mitigations=[],
                confidence=0.5,
                stop=False,
                next_diagnostics=["not-a-real-mitigation"],
            )
        )


def test_a_shell_shaped_diagnostic_is_rejected_as_a_diagnostic():
    """The message must name the axis it came from, or a two-axis failure is
    impossible to attribute from the log."""
    with pytest.raises(PolicyViolation, match="diagnostic"):
        AgentPolicy().validate_step(
            AgentStep(
                category="unknown",
                hypothesis="h",
                next_mitigations=[],
                confidence=0.5,
                stop=False,
                next_diagnostics=["rm -rf /"],
            )
        )


# ---------------------------------------------------------------------------
# The cost model. This is the load-bearing arithmetic.
# ---------------------------------------------------------------------------


def test_the_cell_grid_is_the_cross_product_of_the_two_axes():
    assert probe_cell_names(["none", "tf32_off"], ["none", "xnack"]) == [
        "none-none",
        "none-xnack",
        "tf32_off-none",
        "tf32_off-xnack",
    ]


def test_one_mitigation_alone_costs_one_cell_per_existing_diagnostic():
    growth = plan_axis_growth(["none"], ["none"], ["tf32_off"], [])
    assert growth.cells_added == 1
    wider = plan_axis_growth(["none"], ["none", "xnack"], ["tf32_off"], [])
    assert wider.cells_added == 2


def test_one_diagnostic_alone_re_runs_every_mitigation_already_on_the_axis():
    """The asymmetry that makes the diagnostic action expensive.

    Four mitigations on the axis and one new diagnostic is four new cells, not
    one: the diagnostic has to be paired with each of them.
    """
    growth = plan_axis_growth(
        ["none", "a", "b", "c"], ["none"], [], ["amd_log_level_4"]
    )
    assert growth.cells_added == 4


def test_a_joint_proposal_costs_the_cross_term_too():
    """``a*D + b*M + a*b``, checked against the enumerated grid.

    M=4, D=1, a=1, b=1  ->  1*1 + 1*4 + 1*1 = 6 new cells for two names.
    """
    m_axis = ["none", "a", "b", "c"]
    growth = plan_axis_growth(m_axis, ["none"], ["d"], ["amd_log_level_4"])
    assert growth.cells_added == 6
    assert growth.cells_before == 4
    assert growth.cells_after == 10
    # And the number is the grid, not a formula that could drift from it.
    assert growth.cells_after == len(
        probe_cell_names(growth.mitigation_axis, growth.diagnostic_axis)
    )


@pytest.mark.parametrize(
    ("m", "d", "a", "b"),
    [(1, 1, 1, 1), (4, 1, 2, 1), (3, 2, 1, 2), (1, 1, 5, 0), (6, 1, 0, 3)],
)
def test_the_charged_cost_equals_the_closed_form_for_any_shape(m, d, a, b):
    m_axis = ["none", *[f"m{i}" for i in range(m - 1)]]
    d_axis = ["none", *[f"d{i}" for i in range(d - 1)]]
    growth = plan_axis_growth(
        m_axis,
        d_axis,
        [f"new_m{i}" for i in range(a)],
        [f"new_d{i}" for i in range(b)],
    )
    assert growth.cells_added == a * d + b * m + a * b


def test_a_name_already_on_an_axis_is_free_and_is_not_a_rejection():
    growth = plan_axis_growth(
        ["none", "tf32_off"], ["none", "xnack"], ["tf32_off"], ["xnack"]
    )
    assert growth.cells_added == 0
    assert growth.added_mitigations == []
    assert growth.rejected_mitigations == []
    assert not growth.grew


def test_the_cell_budget_trims_the_proposal_instead_of_overspending():
    """The defect this is built not to repeat.

    Three mitigations onto a 1x1 grid is three cells; a budget of two admits
    two names and rejects the third, and the charged cost never exceeds the
    budget.
    """
    growth = plan_axis_growth(
        ["none"], ["none"], ["a", "b", "c"], [], cells_affordable=2
    )
    assert growth.added_mitigations == ["a", "b"]
    assert growth.rejected_mitigations == ["c"]
    assert growth.cells_added == 2


def test_a_diagnostic_too_expensive_for_the_budget_is_rejected_not_truncated():
    """A diagnostic is all-or-nothing: you cannot buy half a cross product."""
    growth = plan_axis_growth(
        ["none", "a", "b", "c"], ["none"], [], ["amd_log_level_4"],
        cells_affordable=3,
    )
    assert growth.added_diagnostics == []
    assert growth.rejected_diagnostics == ["amd_log_level_4"]
    assert growth.cells_added == 0


def test_the_trim_follows_the_policys_stated_order_not_the_cheapest_first():
    """Deliberate: on a cross product the cheapest name is whichever axis is
    shorter, which is an artefact of history, not a judgement about the bug.

    M=1, D=1. The mitigation costs 1 and is admitted; the diagnostic then
    costs 2 (it must pair with both mitigations) and does not fit in the
    remaining 1, so it is rejected -- even though offering it first would have
    cost 1.
    """
    growth = plan_axis_growth(
        ["none"], ["none"], ["a"], ["amd_log_level_4"], cells_affordable=2
    )
    assert growth.added_mitigations == ["a"]
    assert growth.added_diagnostics == []
    assert growth.cells_added == 1


def test_an_unbounded_budget_admits_everything():
    growth = plan_axis_growth(["none"], ["none"], ["a", "b"], ["x"])
    assert growth.added_mitigations == ["a", "b"]
    assert growth.added_diagnostics == ["x"]
    assert growth.cells_added == 5


def test_the_baseline_stays_on_both_axes_through_any_growth():
    growth = plan_axis_growth(["none"], ["none"], ["a"], ["x"])
    assert growth.mitigation_axis[0] == "none"
    assert "none" in growth.diagnostic_axis
    assert "none-none" in probe_cell_names(
        growth.mitigation_axis, growth.diagnostic_axis
    )


# ---------------------------------------------------------------------------
# AgentPolicy cell budget
# ---------------------------------------------------------------------------


def test_the_cell_budget_is_off_by_default():
    assert AgentPolicy().max_probe_cells is None
    assert AgentPolicy().cells_affordable(10_000) is None
    AgentPolicy().check_cell_budget(10_000)  # must not raise


def test_the_cell_budget_raises_once_nothing_is_affordable():
    policy = AgentPolicy(max_probe_cells=4)
    assert policy.cells_affordable(1) == 3
    policy.check_cell_budget(3)
    with pytest.raises(PolicyViolation, match="probe cell budget exhausted"):
        policy.check_cell_budget(4)


def test_an_overspent_budget_reports_zero_capacity_not_a_negative():
    assert AgentPolicy(max_probe_cells=4).cells_affordable(9) == 0


def test_a_zero_cell_budget_is_user_error():
    with pytest.raises(PolicyViolation, match="max_probe_cells"):
        AgentPolicy(max_probe_cells=0)


# ---------------------------------------------------------------------------
# End to end through run_agent_loop, engine-free (workload is `false`)
# ---------------------------------------------------------------------------

_RECIPE = """\
schema_version: 1
mode: probe
ticket: JOINT-1
trials: 1
mitigation_axis: [none, tf32_off, xnack]
diagnostic_axis: [none]
"""


def _write_recipe(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(_RECIPE, encoding="utf-8")
    return path


class _JointProposer:
    """Proposes one mitigation and one diagnostic, once, then stops."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def propose(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > 1:
            return AgentStep(
                category="unknown",
                hypothesis="done",
                next_mitigations=[],
                confidence=0.4,
                stop=True,
                stop_reason="agent_requested",
            )
        return AgentStep(
            category="illegal_mem",
            hypothesis="uninitialised read; want the log to see it",
            next_mitigations=["tf32_off"],
            confidence=0.6,
            stop=False,
            next_diagnostics=["amd_log_level_4"],
        )


def _config(tmp_path: Path, **kwargs) -> AgentConfig:
    policy = kwargs.pop("policy", AgentPolicy(max_iterations=3))
    return AgentConfig(
        output_dir=tmp_path / "out",
        ticket="JOINT-1",
        subprocess_argv=("false",),
        symptom="loss=nan",
        policy=policy,
        recipe_path=_write_recipe(tmp_path),
        **kwargs,
    )


def test_the_loop_runs_the_cross_product_after_a_joint_proposal(tmp_path):
    proposer = _JointProposer()
    result = run_agent_loop(
        _config(tmp_path, diagnostics_allowlist=("amd_log_level_4",)),
        proposer=proposer,
    )
    cells = {p.name for p in result.run_dir.iterdir() if p.is_dir()}
    # none + tf32_off on the mitigation axis x none + amd_log_level_4 on the
    # diagnostic axis. The diagnostic paired itself with the baseline too,
    # which is exactly what makes it cost more than one cell.
    assert {
        "none-none",
        "none-amd_log_level_4",
        "tf32_off-none",
        "tf32_off-amd_log_level_4",
    } <= cells


def test_the_loop_offers_the_diagnostic_axis_to_the_proposer(tmp_path):
    proposer = _JointProposer()
    run_agent_loop(
        _config(tmp_path, diagnostics_allowlist=("amd_log_level_4",)),
        proposer=proposer,
    )
    assert proposer.calls[0]["diagnostic_candidates"] == ["amd_log_level_4"]
    # Second call sees the first one as spent, so the axis is not re-bought.
    assert proposer.calls[1]["tried_diagnostics"] == ["amd_log_level_4"]


def test_a_proposed_diagnostic_is_refused_when_the_axis_was_not_armed(tmp_path):
    """No ``--diagnostic`` means no diagnostic action, and the refusal is loud.

    This is the backward-compatibility guarantee at the loop layer: an
    existing run cannot acquire a second axis because a model mentioned one.
    """
    result = run_agent_loop(_config(tmp_path), proposer=_JointProposer())
    assert result.outcome == "policy_stop"
    assert "diagnostics outside the allowed candidate set" in result.recommended_action
    cells = {p.name for p in result.run_dir.iterdir() if p.is_dir()}
    assert not any("amd_log_level_4" in c for c in cells)


def test_the_log_records_the_cells_a_growth_cost_and_wake_replays_it(tmp_path):
    result = run_agent_loop(
        _config(tmp_path, diagnostics_allowlist=("amd_log_level_4",)),
        proposer=_JointProposer(),
    )
    events = [
        json.loads(line)
        for line in (result.run_dir / "agent_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    growth = [e for e in events if e["type"] == "axis_growth"]
    assert len(growth) == 1
    # M=1(none) x D=1(none) = 1 cell before. Adding tf32_off costs 1, then
    # amd_log_level_4 pairs with both mitigations for 2 more: 3 added, 4 total.
    assert growth[0]["cells_added"] == 3
    assert growth[0]["cells_total"] == 4
    assert growth[0]["added_diagnostics"] == ["amd_log_level_4"]
    assert [e for e in events if e["type"] == "diagnostic_tried"]

    replayed = wake(result.run_dir, ticket="JOINT-1")
    assert replayed.tried_diagnostics == ["amd_log_level_4"]
    assert replayed.probe_cells_spent == 4


def test_the_charged_cells_equal_the_cell_directories_actually_run(tmp_path):
    """The requirement, stated as one assertion.

    A cost model that is not this is the 160-cells-against-8 defect in a new
    place.
    """
    result = run_agent_loop(
        _config(tmp_path, diagnostics_allowlist=("amd_log_level_4", "xnack")),
        proposer=_JointProposer(),
    )
    on_disk = len([p for p in result.run_dir.iterdir() if p.is_dir() and "-" in p.name])
    assert result.state.probe_cells_spent == on_disk


def test_the_cell_budget_stops_the_loop_before_it_overspends(tmp_path):
    """A budget of 2 cannot afford the 3-cell joint proposal, so nothing on
    the diagnostic axis is bought and the run stops rather than overrunning."""

    class _Greedy:
        def propose(self, **kwargs):
            return AgentStep(
                category="unknown",
                hypothesis="everything at once",
                next_mitigations=["tf32_off", "xnack"],
                confidence=0.5,
                stop=False,
                next_diagnostics=["amd_log_level_4"],
            )

    result = run_agent_loop(
        _config(
            tmp_path,
            policy=AgentPolicy(max_iterations=8, max_probe_cells=2),
            diagnostics_allowlist=("amd_log_level_4",),
        ),
        proposer=_Greedy(),
    )
    on_disk = len([p for p in result.run_dir.iterdir() if p.is_dir() and "-" in p.name])
    assert on_disk <= 2
    assert result.outcome == "policy_stop"
    assert "budget" in result.recommended_action.lower()


def test_a_diagnostic_only_step_is_a_legal_action(tmp_path):
    """Buying evidence without testing a cause is the first move of a
    minimum-cost localisation, and it used to be unrepresentable."""

    class _InfoOnly:
        def __init__(self) -> None:
            self.n = 0

        def propose(self, **kwargs):
            self.n += 1
            if self.n > 1:
                return AgentStep(
                    category="unknown", hypothesis="done", next_mitigations=[],
                    confidence=0.1, stop=True, stop_reason="agent_requested",
                )
            return AgentStep(
                category="illegal_mem",
                hypothesis="need the log before guessing a fix",
                next_mitigations=[],
                confidence=0.3,
                stop=False,
                next_diagnostics=["amd_log_level_4"],
            )

    result = run_agent_loop(
        _config(tmp_path, diagnostics_allowlist=("amd_log_level_4",)),
        proposer=_InfoOnly(),
    )
    cells = {p.name for p in result.run_dir.iterdir() if p.is_dir()}
    assert "none-amd_log_level_4" in cells
    assert result.outcome == "agent_stop"


def test_a_resumed_run_does_not_re_charge_the_cells_it_already_paid_for(tmp_path):
    config = _config(tmp_path, diagnostics_allowlist=("amd_log_level_4",))
    first = run_agent_loop(config, proposer=_JointProposer())
    spent = first.state.probe_cells_spent

    class _Stop:
        def propose(self, **kwargs):
            return AgentStep(
                category="unknown", hypothesis="stop", next_mitigations=[],
                confidence=0.1, stop=True, stop_reason="agent_requested",
            )

    second = run_agent_loop(config, proposer=_Stop())
    assert second.state.probe_cells_spent == spent
    assert second.state.tried_diagnostics == ["amd_log_level_4"]
