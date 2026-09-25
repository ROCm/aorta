"""The discrete-event reward: what fires, what it is worth, and what it refuses.

CPU-only and engine-free. Ground truth is a tiny archived matrix written into
``tmp_path`` in the shape the probe harness writes, and episodes are
``agent_log.jsonl`` files in the shape ``run_agent_loop`` appends. Nothing here
asserts a result about a policy; it asserts that the formulation does what its
docstring says:

* the point table is pinned and nothing emits an unpriced event;
* admissibility keeps three distinguishable mistakes distinguishable, and
  agrees with ``validate_step`` about categories;
* the matrix's silence about a name is a third answer, not a wrong one;
* the resolver award fires once, and earliness decays to zero rather than below;
* terminals are read off records the loop writes, and an unresolvability claim
  pays only when it was earned and only when it was the policy's;
* and the invariants the point values were chosen to satisfy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rl"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from event_reward import (  # noqa: E402
    BASELINE,
    EARLINESS,
    POINTS,
    MatrixGrid,
    StepContext,
    admissible,
    attribution_points,
    classify_stop,
    episode_from_log,
    grid_from_matrix,
    score_completion,
    score_episode,
    step_events,
)
from triage_reward import Label, set_f1  # noqa: E402

from aorta.agent.llm import EVIDENCE_ONLY_CATEGORIES, PROBE_CATEGORIES  # noqa: E402

RESOLVER = "pytorch_no_cuda_memory_caching"
REFUTED = "tf32_off"
UNMEASURED = "hsa_no_scratch_reclaim"
OFFERED = frozenset({RESOLVER, REFUTED, UNMEASURED, "xnack"})
NAN = ["tier4:nan_signature"]


# ---------------------------------------------------------------------------
# fixtures, in the shape the real artifacts have
# ---------------------------------------------------------------------------


def _write_cell(
    root: Path,
    cell: str,
    verdict: str,
    detectors: list[str],
    capture: dict[str, str] | None = None,
    trials: int = 1,
) -> None:
    for index in range(trials):
        trial = root / cell / f"trial_{index}"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "cell_name": cell,
                    "verdict": verdict,
                    "exit_code": 0 if verdict == "pass" else 1,
                    "capture": capture or {},
                    "failure_detectors_fired": detectors,
                    "error_detectors_fired": [],
                    "warn_detectors_fired": [],
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture
def matrix(tmp_path):
    """Resolvable: baseline fails, one mitigation passes, one name has no cell."""
    root = tmp_path / "PROBE-RESOLVABLE"
    _write_cell(root, "none-none", "fail", NAN)
    _write_cell(root, f"{REFUTED}-none", "fail", NAN)
    _write_cell(root, "xnack-none", "fail", NAN)
    _write_cell(root, f"{RESOLVER}-none", "pass", [])
    return root


@pytest.fixture
def grid(matrix):
    return grid_from_matrix(matrix)


@pytest.fixture
def unresolvable_grid(tmp_path):
    """A real failure that nothing in the matrix resolves."""
    root = tmp_path / "PROBE-UNRESOLVABLE"
    _write_cell(root, "none-none", "fail", NAN)
    _write_cell(root, f"{REFUTED}-none", "fail", NAN)
    _write_cell(root, f"{RESOLVER}-none", "fail", NAN)
    return grid_from_matrix(root)


def _context(**kwargs) -> StepContext:
    base = {"offered_mitigations": OFFERED}
    base.update(kwargs)
    return StepContext(**base)


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


def _fire(raw, grid_, context=None, step=1, **kwargs) -> list[str]:
    events, _ = step_events(raw, context or _context(), grid_, step=step, **kwargs)
    return [event.name for event in events]


def _write_log(run_dir: Path, records: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "agent_log.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return run_dir


def _llm_step(mitigations=(), stop=False, unresolved=()):
    record = {
        "type": "llm_step",
        "category": "unknown",
        "hypothesis": "h",
        "next_mitigations": list(mitigations),
        "confidence": 0.5,
        "stop": stop,
        "stop_reason": "agent_requested" if stop else None,
    }
    if unresolved:
        record["unresolved_mitigations"] = list(unresolved)
    return record


def _tried(*names):
    return [{"type": "mitigation_tried", "mitigation": name} for name in names]


def _stopped(reason="agent_requested", unresolved=()):
    record = {"type": "search_stopped", "outcome": "agent_stop", "stop_reason": reason}
    if unresolved:
        record["unresolved_mitigations"] = list(unresolved)
    return record


# ---------------------------------------------------------------------------
# the point table
# ---------------------------------------------------------------------------


def test_the_point_table_is_pinned():
    """Every value, pinned: a later change to one has to move this line too and
    say so in the diff, rather than drifting to make a result look better."""
    assert POINTS == {
        "malformed_reply": -6.0,
        "resolver_named": 4.0,
        "resolver_named_on_step_1": 2.0,
        "resolver_named_on_step_2": 1.0,
        "name_refuted": -0.5,
        "name_unmeasured": 0.0,
        "name_not_offered": -1.0,
        "name_already_tried": -1.0,
        "verdict_correct": 1.0,
        "verdict_wrong": -1.0,
        "detector_attribution": 1.0,
        "terminal_converged": 2.0,
        "terminal_unresolvable_correct": 4.0,
        "terminal_unresolvable_unearned": 0.0,
        "terminal_gave_up": -3.0,
        "terminal_proposal_unresolved": -3.0,
        "cell_spent": -0.2,
    }


def test_the_manufactured_stop_is_priced_exactly_like_giving_up():
    """Equality is the statement: the event buys attribution, not a number,
    and a cheaper value would make stalling on a tried name the best exit."""
    assert POINTS["terminal_proposal_unresolved"] == POINTS["terminal_gave_up"]
    assert POINTS["terminal_unresolvable_correct"] == POINTS["resolver_named"]


def test_every_classifiable_terminal_has_a_price():
    """A terminal the classifiers can return but the table does not price
    would be silently withheld as 'unrecognised' on the rollout that first hit
    it. Enumerated from ``classify_stop`` itself rather than listed by hand."""
    seen = set()
    for reason in ("proposal_unresolved", "agent_requested"):
        for proposed_nothing in (True, False):
            for unresolvable in (True, False):
                for named in (True, False):
                    for tried in ([], [REFUTED], sorted(OFFERED)):
                        terminal, _ = classify_stop(
                            stop_reason=reason,
                            proposed_nothing=proposed_nothing,
                            unresolvable=unresolvable,
                            resolver_named=named,
                            offered=OFFERED,
                            tried=tried,
                        )
                        seen.add(terminal)
    seen |= {"converged"}
    priced = {t for t in seen if t != "other"}
    assert {f"terminal_{t}" for t in priced} <= set(POINTS), sorted(priced)
    assert seen == {
        "proposal_unresolved", "unresolvable_correct", "unresolvable_unearned",
        "gave_up", "other", "converged",
    }


def test_no_event_is_emitted_without_a_price(grid, unresolvable_grid, tmp_path):
    fired: set[str] = set()
    fired.update(_fire("prose", grid))
    fired.update(
        _fire(
            _raw(next_mitigations=[RESOLVER, REFUTED, UNMEASURED, "invented"], verdict="fail"),
            grid,
            _context(tried_mitigations=frozenset({"xnack"})),
            label=Label(verdict="fail"),
        )
    )
    for terminal_grid in (grid, unresolvable_grid):
        score = score_completion(_raw(stop=True), terminal_grid, _context(), cells_added=1)
        fired.update(event.name for event in score.events)
    run = _write_log(
        tmp_path / "CONVERGED",
        [_llm_step([RESOLVER]), *_tried(RESOLVER),
         {"type": "converged", "winning_mitigation": RESOLVER}],
    )
    fired.update(
        e.name for e in score_episode(episode_from_log(run, grid), grid, _context()).events
    )
    assert fired <= set(POINTS), sorted(fired - set(POINTS))


def test_the_earliness_table_only_ever_pays_and_never_charges():
    assert all(POINTS[key] > 0 for key in EARLINESS.values())
    assert sorted(EARLINESS) == [1, 2]


# ---------------------------------------------------------------------------
# admissibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ("not JSON, just a sentence.", "not JSON"),
        ("[1, 2, 3]", "not an object"),
        (json.dumps({"category": "unknown"}), "missing required key"),
        (_raw(confidence=True), "is a bool"),
        (_raw(stop="false"), "expected"),
        (_raw(category="not_a_category"), "not one the probe agent may claim"),
    ],
)
def test_an_inadmissible_reply_is_one_event_with_a_reason(raw, fragment, grid):
    parsed, reason = admissible(raw)
    assert parsed is None
    assert fragment in reason
    assert _fire(raw, grid) == ["malformed_reply"]


@pytest.mark.parametrize("category", sorted(EVIDENCE_ONLY_CATEGORIES))
def test_an_evidence_only_category_is_refused_as_the_loop_refuses_it(category, grid):
    """``validate_step`` refuses these from a probe step, so a reply carrying
    one does nothing in the loop and must not be scored as a proposal."""
    assert category not in PROBE_CATEGORIES
    assert _fire(_raw(category=category), grid) == ["malformed_reply"]


@pytest.mark.parametrize("category", sorted(PROBE_CATEGORIES))
def test_every_probe_category_is_admissible(category, grid):
    """Narrowness: the refusal above is the evidence-only set and nothing more."""
    assert admissible(_raw(category=category))[0] is not None


def test_an_unoffered_name_is_an_event_not_an_admissibility_failure(grid):
    """"You wrote prose" and "you named something unrunnable" stay two mistakes."""
    assert _fire(_raw(next_mitigations=["invented_mitigation"]), grid) == ["name_not_offered"]
    assert _fire("prose", grid) == ["malformed_reply"]
    assert POINTS["malformed_reply"] < POINTS["name_not_offered"]


# ---------------------------------------------------------------------------
# what the matrix knows, and what it does not
# ---------------------------------------------------------------------------


def test_the_matrix_is_read_for_what_it_measured_not_only_for_what_resolved(grid):
    assert grid.resolution.resolvers == frozenset({RESOLVER})
    assert grid.resolution.baseline_failed
    assert grid.measured_mitigations == frozenset({RESOLVER, REFUTED, "xnack"})


def test_a_refuted_name_and_an_unmeasured_name_are_different_events(grid):
    assert _fire(_raw(next_mitigations=[REFUTED]), grid) == ["name_refuted"]
    assert _fire(_raw(next_mitigations=[UNMEASURED]), grid) == ["name_unmeasured"]
    assert POINTS["name_unmeasured"] == 0.0 > POINTS["name_refuted"]


def test_a_pass_under_a_diagnostic_is_not_a_resolver(tmp_path):
    """``winning_mitigation`` credits only ``{m}-none``: a pass with a
    diagnostic on is not attributable to the mitigation alone."""
    root = tmp_path / "DIAG"
    _write_cell(root, "none-none", "fail", NAN)
    _write_cell(root, f"{REFUTED}-none", "fail", NAN)
    _write_cell(root, f"{REFUTED}-hip_launch_blocking", "pass", [])
    grid = grid_from_matrix(root)
    assert grid.resolution.resolvers == frozenset()
    assert grid.measured_mitigations == frozenset({REFUTED})


def test_a_matrix_with_no_baseline_cell_has_no_failure_to_resolve(tmp_path):
    """Fail-closed: a missing ``none-none`` must not read as a failing one."""
    root = tmp_path / "NOBASE"
    _write_cell(root, f"{RESOLVER}-none", "pass", [])
    grid = grid_from_matrix(root)
    assert grid.resolution.baseline_failed is False
    assert grid.resolution.resolvers == frozenset({RESOLVER})


def test_the_verdict_is_recomputed_not_read_off_the_artifact(tmp_path):
    """A cell that stored ``pass`` while firing a failure detector is a
    failure: the resolver decides, not the stored string."""
    root = tmp_path / "STALE"
    _write_cell(root, "none-none", "pass", NAN)
    assert grid_from_matrix(root).verdicts["none-none"] == "fail"


def test_a_name_already_on_the_axis_is_charged_once_and_as_a_repeat(grid):
    context = _context(
        offered_mitigations=OFFERED - {REFUTED}, tried_mitigations=frozenset({REFUTED})
    )
    assert _fire(_raw(next_mitigations=[REFUTED]), grid, context) == ["name_already_tried"]


def test_one_name_written_twice_is_one_proposal(grid):
    assert _fire(_raw(next_mitigations=[REFUTED, REFUTED]), grid) == ["name_refuted"]


def test_proposing_the_baseline_fires_nothing(grid):
    assert _fire(_raw(next_mitigations=[BASELINE]), grid) == []


# ---------------------------------------------------------------------------
# the resolver award
# ---------------------------------------------------------------------------


def test_naming_the_resolver_pays_the_award_and_the_first_step_bonus(grid):
    assert _fire(_raw(next_mitigations=[RESOLVER]), grid, step=1) == [
        "resolver_named",
        "resolver_named_on_step_1",
    ]


def test_the_earliness_bonus_decays_to_zero(grid):
    raw = _raw(next_mitigations=[RESOLVER])
    assert _fire(raw, grid, step=2) == ["resolver_named", "resolver_named_on_step_2"]
    assert _fire(raw, grid, step=3) == ["resolver_named"]


def test_the_award_is_withheld_once_it_has_been_paid(grid):
    assert _fire(_raw(next_mitigations=[RESOLVER]), grid, step=3,
                 resolver_already_named=True) == []


def test_a_stopping_reply_that_names_the_resolver_earns_nothing_for_it(grid):
    """The loop stops before it runs any name in a stopping reply, so the
    resolver was never tried: no award, no earliness bonus."""
    assert _fire(_raw(next_mitigations=[RESOLVER], stop=True), grid) == []


def test_a_stopping_reply_is_not_charged_for_its_names_either(grid):
    """Symmetric: a name that never runs is neither paid nor charged."""
    assert _fire(_raw(next_mitigations=[REFUTED, "invented"], stop=True), grid) == []


def test_the_same_reply_without_stop_still_earns_the_award(grid):
    """Narrowness: the rule is about stopping, not about naming the resolver."""
    assert _fire(_raw(next_mitigations=[RESOLVER], stop=False), grid) == [
        "resolver_named", "resolver_named_on_step_1",
    ]


def test_a_stop_naming_the_resolver_ends_the_episode_as_giving_up(tmp_path, grid):
    run = _write_log(
        tmp_path / "STOPNAMED",
        [_llm_step([RESOLVER], stop=True), _stopped()],
    )
    episode = episode_from_log(run, grid, OFFERED)
    assert episode.terminal == "gave_up"
    assert not score_episode(episode, grid, _context()).fired("resolver_named")


def test_a_resolver_that_was_not_offered_earns_nothing(grid):
    """It never becomes a cell, so it cannot have fixed anything."""
    context = _context(offered_mitigations=OFFERED - {RESOLVER})
    assert _fire(_raw(next_mitigations=[RESOLVER]), grid, context) == ["name_not_offered"]


def test_a_resolver_already_on_the_axis_earns_nothing_again(grid):
    """The filter drops a tried name, so re-proposing the resolver runs nothing.

    The menu is the full one here because that is how ``score_episode`` builds
    every step's context -- offered stays the whole menu and the axis grows --
    so "offered" alone does not exclude a tried name.
    """
    context = _context(tried_mitigations=frozenset({RESOLVER}))
    assert _fire(_raw(next_mitigations=[RESOLVER]), grid, context) == ["name_already_tried"]


# ---------------------------------------------------------------------------
# the verdict and attribution events
# ---------------------------------------------------------------------------


def test_the_verdict_is_scored_only_when_a_label_is_supplied(grid):
    assert _fire(_raw(verdict="fail"), grid) == []


def test_an_absent_verdict_is_wrong_rather_than_withheld(grid):
    label = Label(verdict="fail")
    assert _fire(_raw(), grid, label=label) == ["verdict_wrong", "detector_attribution"]
    assert _fire(_raw(verdict="fail"), grid, label=label) == [
        "verdict_correct", "detector_attribution"
    ]


def test_the_attribution_award_is_the_shared_set_f1_and_not_a_second_one():
    for cited, actual in ((["a"], ["a"]), ([], []), (["a"], ["b"]), (["a", "b"], ["a"]),
                          ([], ["a"]), (["a", "b", "c"], ["a", "b"])):
        assert attribution_points(cited, actual) == pytest.approx(
            POINTS["detector_attribution"] * (2 * set_f1(cited, actual) - 1)
        )


def test_the_attribution_award_is_symmetric_so_citing_nothing_is_not_free():
    assert attribution_points(["a"], ["a"]) == pytest.approx(1.0)
    assert attribution_points([], ["a"]) == pytest.approx(-1.0)
    assert attribution_points(["a", "b"], ["a", "c"]) == pytest.approx(0.0)


def test_an_absent_detectors_key_cites_nothing_rather_than_being_withheld(grid):
    label = Label(verdict="fail", failure_detectors=NAN)
    events, _ = step_events(_raw(verdict="fail"), _context(), grid, step=1, label=label)
    award = next(e for e in events if e.name == "detector_attribution")
    assert award.points == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# terminals, read off records
# ---------------------------------------------------------------------------


def test_a_converged_run_is_read_from_the_converged_record(tmp_path, grid):
    run = _write_log(
        tmp_path / "CONV",
        [{"type": "session_start"}, _llm_step([RESOLVER]), *_tried(RESOLVER),
         {"type": "iteration_complete", "iteration": 1},
         {"type": "converged", "winning_mitigation": RESOLVER}],
    )
    episode = episode_from_log(run, grid)
    assert episode.terminal == "converged"
    assert episode.cells == 2


@pytest.mark.parametrize("record", [
    {"type": "policy_stop", "reason": "iteration budget exhausted (8 max)"},
    {"type": "baseline_pass"},
    {"type": "approval_required", "mitigations": ["hip_launch_blocking"]},
    {"type": "registry_error", "reason": "x"},
    {"type": "error", "reason": "x"},
])
def test_an_ending_that_is_not_about_the_search_is_withheld(tmp_path, grid, record):
    run = _write_log(tmp_path / "OTHER", [_llm_step([REFUTED]), *_tried(REFUTED), record])
    episode = episode_from_log(run, grid, OFFERED)
    assert episode.terminal == "other"
    score = score_episode(episode, grid, _context())
    assert not any(e.name.startswith("terminal_") for e in score.events)
    assert any("terminal (other)" in note for note in score.withheld)


def test_a_log_with_no_terminal_is_unobserved_not_a_conclusion(tmp_path, grid):
    """A truncated log must not be read as any ending at all."""
    run = _write_log(tmp_path / "CUT", [_llm_step([REFUTED]), *_tried(REFUTED)])
    assert episode_from_log(run, grid).terminal == "unobserved"


def test_the_unresolvable_award_is_earned_by_exhausting_the_menu(tmp_path, unresolvable_grid):
    menu = [REFUTED, RESOLVER]
    run = _write_log(
        tmp_path / "EARNED",
        [_llm_step(menu), *_tried(*menu), _llm_step(stop=True), _stopped()],
    )
    episode = episode_from_log(run, unresolvable_grid, menu)
    assert episode.terminal == "unresolvable_correct"
    assert score_episode(episode, unresolvable_grid, _context()).fired(
        "terminal_unresolvable_correct"
    )


def test_one_candidate_short_of_the_menu_does_not_earn_the_award(tmp_path, unresolvable_grid):
    run = _write_log(
        tmp_path / "NEARLY",
        [_llm_step([REFUTED]), *_tried(REFUTED), _llm_step(stop=True), _stopped()],
    )
    episode = episode_from_log(run, unresolvable_grid, [REFUTED, RESOLVER])
    assert episode.terminal == "unresolvable_unearned"


def test_an_unrecorded_menu_does_not_buy_the_award(tmp_path, unresolvable_grid):
    """Silence must not be worth +4.0."""
    run = _write_log(
        tmp_path / "NOMENU",
        [_llm_step([REFUTED, RESOLVER]), *_tried(REFUTED, RESOLVER),
         _llm_step(stop=True), _stopped()],
    )
    assert episode_from_log(run, unresolvable_grid, []).terminal == "unresolvable_unearned"


def test_the_filter_emptying_a_list_is_not_the_unresolvability_claim(tmp_path, unresolvable_grid):
    """Every precondition of the claim holds -- unresolvable, menu exhausted,
    empty list at the stop -- and the stop is still the filter's."""
    menu = [REFUTED, RESOLVER]
    run = _write_log(
        tmp_path / "MANUFACTURED",
        [_llm_step(menu), *_tried(*menu),
         _llm_step([], unresolved=[REFUTED]),
         _stopped("proposal_unresolved", unresolved=[REFUTED])],
    )
    episode = episode_from_log(run, unresolvable_grid, menu)
    assert episode.terminal == "proposal_unresolved"
    assert REFUTED in episode.terminal_why


def test_the_same_stop_is_giving_up_when_a_resolver_exists(tmp_path, grid):
    run = _write_log(
        tmp_path / "GAVEUP",
        [_llm_step([REFUTED]), *_tried(REFUTED), _llm_step(stop=True), _stopped()],
    )
    assert episode_from_log(run, grid, OFFERED).terminal == "gave_up"


def test_stopping_after_naming_a_resolver_is_other(tmp_path, grid):
    """Unreachable in the loop (a named resolver converges first) and still
    classified, so it scores nothing rather than being mistaken for giving up."""
    run = _write_log(
        tmp_path / "AFTER",
        [_llm_step([RESOLVER]), _llm_step(stop=True), _stopped()],
    )
    assert episode_from_log(run, grid, OFFERED).terminal == "other"


# ---------------------------------------------------------------------------
# the dropped names, recovered from the log
# ---------------------------------------------------------------------------


def test_a_filter_dropped_name_is_scored_from_the_log(tmp_path, grid):
    """``unresolved_mitigations`` is folded back into the proposal, so a log
    scores what the model asked for rather than what survived the filter."""
    run = _write_log(
        tmp_path / "DROPPED",
        [_llm_step([REFUTED], unresolved=["invented"]), *_tried(REFUTED),
         {"type": "policy_stop", "reason": "budget"}],
    )
    episode = episode_from_log(run, grid, OFFERED)
    assert json.loads(episode.steps[0].raw)["next_mitigations"] == [REFUTED, "invented"]
    score = score_episode(episode, grid, _context())
    assert score.count("name_not_offered") == 1
    assert score.count("name_refuted") == 1


def test_a_dropped_name_already_on_the_axis_is_a_repeat_not_a_miss(tmp_path, grid):
    """The filter subtracts tried from offered in one step, so the log's
    unresolved list cannot tell the two apart -- the axis at the step can."""
    run = _write_log(
        tmp_path / "REPEAT",
        [_llm_step([REFUTED]), *_tried(REFUTED),
         _llm_step([], unresolved=[REFUTED]),
         _stopped("proposal_unresolved", unresolved=[REFUTED])],
    )
    score = score_episode(episode_from_log(run, grid, OFFERED), grid, _context())
    assert [e.step for e in score.events if e.name == "name_already_tried"] == [2]
    assert score.count("name_not_offered") == 0


def test_a_log_without_the_key_rebuilds_exactly_as_it_did_before(tmp_path, grid):
    run = _write_log(tmp_path / "OLD", [_llm_step([REFUTED])])
    raw = json.loads(episode_from_log(run, grid).steps[0].raw)
    assert raw["next_mitigations"] == [REFUTED]


# ---------------------------------------------------------------------------
# per-step credit
# ---------------------------------------------------------------------------


def test_every_step_event_carries_the_step_it_fired_on(tmp_path, grid):
    run = _write_log(
        tmp_path / "STEPS",
        [_llm_step([REFUTED]), *_tried(REFUTED), _llm_step([RESOLVER]), *_tried(RESOLVER),
         {"type": "converged", "winning_mitigation": RESOLVER}],
    )
    score = score_episode(episode_from_log(run, grid), grid, _context())
    by_step = {event.name: event.step for event in score.events}
    assert by_step["name_refuted"] == 1
    assert by_step["resolver_named"] == 2
    assert by_step["resolver_named_on_step_2"] == 2
    assert by_step["terminal_converged"] is None


def test_two_episodes_that_end_identically_are_told_apart_by_their_route(tmp_path, grid):
    """The credit-assignment claim: same terminal, same resolver, and the
    direct route scores higher than the wandering one."""
    direct = _write_log(
        tmp_path / "DIRECT",
        [_llm_step([RESOLVER]), *_tried(RESOLVER),
         {"type": "converged", "winning_mitigation": RESOLVER}],
    )
    wandering = _write_log(
        tmp_path / "WANDERING",
        [_llm_step([REFUTED]), *_tried(REFUTED), _llm_step(["xnack"]), *_tried("xnack"),
         _llm_step([RESOLVER]), *_tried(RESOLVER),
         {"type": "converged", "winning_mitigation": RESOLVER}],
    )
    fast = score_episode(episode_from_log(direct, grid), grid, _context())
    slow = score_episode(episode_from_log(wandering, grid), grid, _context())
    assert fast.terminal == slow.terminal == "converged"
    assert fast.total > slow.total


def test_the_cells_on_disk_outrank_a_log_that_undercounts(grid, matrix):
    episode = episode_from_log(matrix, grid)
    assert episode.steps == []
    assert episode.cells == 4


# ---------------------------------------------------------------------------
# the unresolvable claim on the single-reply path
# ---------------------------------------------------------------------------


def test_a_first_turn_unresolvability_claim_does_not_collect_the_award(unresolvable_grid):
    score = score_completion(_raw(stop=True), unresolvable_grid, _context(),
                             cells_already_spent=2)
    assert score.terminal == "unresolvable_unearned"
    assert score.total == pytest.approx(POINTS["cell_spent"] * 2)


def test_honest_abstention_is_not_made_unprofitable(unresolvable_grid):
    honest = score_completion(_raw(stop=True), unresolvable_grid, _context(),
                              cells_already_spent=2)
    guess = score_completion(_raw(next_mitigations=[REFUTED]), unresolvable_grid,
                             _context(), cells_already_spent=2, cells_added=1)
    assert honest.total > guess.total
    assert POINTS["terminal_unresolvable_unearned"] > POINTS["terminal_gave_up"]


def test_the_claim_is_giving_up_when_a_resolver_exists(grid):
    assert score_completion(_raw(stop=True), grid, _context()).terminal == "gave_up"


def test_a_completion_that_exhausted_the_menu_earns_the_award(unresolvable_grid):
    score = score_completion(_raw(stop=True), unresolvable_grid,
                             _context(tried_mitigations=OFFERED), cells_already_spent=2)
    assert score.terminal == "unresolvable_correct"


@pytest.mark.parametrize("names", [[RESOLVER], [REFUTED, "invented"], [RESOLVER, REFUTED]])
def test_a_stopping_reply_that_lists_names_gives_up_on_both_paths(tmp_path, grid, names):
    """The review's inconsistency: `stop: true` proposes nothing, and the
    episode path ends such a reply as giving up, so one reply must too."""
    single = score_completion(_raw(next_mitigations=names, stop=True), grid, _context())
    assert single.terminal == "gave_up" and single.fired("terminal_gave_up")
    episode = episode_from_log(
        _write_log(tmp_path / "STOP", [_llm_step(names, stop=True), _stopped()]), grid, OFFERED)
    assert episode.terminal == single.terminal


def test_a_stopping_reply_that_lists_names_is_not_the_unresolvable_claim(
    tmp_path, unresolvable_grid,
):
    """Only a stop that lists nothing claims "nothing resolves this"; one that
    lists names gives up, on the episode path and now on the single-reply one."""
    listed = score_completion(_raw(next_mitigations=[REFUTED], stop=True), unresolvable_grid,
                              _context(), cells_already_spent=2)
    empty = score_completion(_raw(stop=True), unresolvable_grid, _context(),
                             cells_already_spent=2)
    assert listed.terminal == "gave_up" and empty.terminal == "unresolvable_unearned"
    episode = episode_from_log(
        _write_log(tmp_path / "STOP", [_llm_step([REFUTED], stop=True), _stopped()]),
        unresolvable_grid, OFFERED)
    assert episode.terminal == listed.terminal


def test_an_empty_stop_and_a_non_stop_reply_are_unchanged(grid):
    """Narrowness: only a stopping reply with names moved."""
    assert score_completion(_raw(stop=True), grid, _context()).terminal == "gave_up"
    assert score_completion(_raw(next_mitigations=[RESOLVER]), grid,
                            _context()).terminal == "unobserved"


def test_a_single_completion_leaves_the_terminal_unobserved_and_says_so(grid):
    score = score_completion(_raw(next_mitigations=[REFUTED]), grid, _context())
    assert score.terminal == "unobserved"
    assert any("unobserved" in note for note in score.withheld)


# ---------------------------------------------------------------------------
# the invariants the point values exist to satisfy
# ---------------------------------------------------------------------------


def test_malformed_sits_below_every_wellformed_reply_that_spends_the_same_cells(
    grid, unresolvable_grid
):
    malformed = score_completion("prose", grid, _context())
    assert malformed.cells == 0
    for ground_truth in (grid, unresolvable_grid):
        for raw in (_raw(), _raw(stop=True)):
            wellformed = score_completion(raw, ground_truth, _context())
            assert wellformed.cells == 0
            assert wellformed.total > malformed.total, (raw, ground_truth)


def test_a_whole_budget_of_cells_cannot_outweigh_naming_the_resolver(grid):
    assert abs(POINTS["cell_spent"]) * 10 < POINTS["resolver_named"]
    found = score_completion(_raw(next_mitigations=[RESOLVER]), grid, _context(),
                             cells_already_spent=10)
    missed = score_completion(_raw(next_mitigations=[REFUTED]), grid, _context(),
                              cells_already_spent=1)
    assert found.total > missed.total


def test_there_is_no_floor_so_distinct_bad_policies_get_distinct_totals(grid):
    policies = {
        "prose": score_completion("prose", grid, _context()),
        "stop_and_say_nothing": score_completion(_raw(stop=True), grid, _context()),
        "one_refuted_name": score_completion(_raw(next_mitigations=[REFUTED]), grid,
                                             _context(), cells_added=1),
        "shotgun": score_completion(_raw(next_mitigations=sorted(OFFERED)), grid,
                                    _context(), cells_added=4),
    }
    totals = [round(score.total, 4) for score in policies.values()]
    assert len(set(totals)) == len(totals), dict(zip(policies, totals, strict=True))


def test_stalling_is_not_a_cheaper_exit_than_quitting(tmp_path, grid):
    """At equal terminal value, the stall also pays for the repeated name."""
    quit_now = _write_log(
        tmp_path / "QUIT",
        [_llm_step([REFUTED]), *_tried(REFUTED), _llm_step(stop=True), _stopped()],
    )
    stall = _write_log(
        tmp_path / "STALL",
        [_llm_step([REFUTED]), *_tried(REFUTED), _llm_step([], unresolved=[REFUTED]),
         _stopped("proposal_unresolved", unresolved=[REFUTED])],
    )
    total = {
        name: score_episode(episode_from_log(run, grid, OFFERED), grid, _context()).total
        for name, run in (("quit", quit_now), ("stall", stall))
    }
    assert total["stall"] < total["quit"], total


# ---------------------------------------------------------------------------
# withholding rather than zeroing
# ---------------------------------------------------------------------------


def test_without_a_matrix_every_ground_truth_event_is_withheld_and_flagged():
    score = score_completion(_raw(next_mitigations=[RESOLVER]), None, _context())
    assert score.total == 0.0
    assert any("no archived matrix" in note for note in score.withheld)


def test_a_grid_with_no_cells_knows_nothing_about_any_name(tmp_path):
    empty = MatrixGrid(verdicts={}, resolution=grid_from_matrix(tmp_path).resolution)
    assert empty.measured_mitigations == frozenset()
    assert _fire(_raw(next_mitigations=[REFUTED]), empty) == ["name_unmeasured"]


def test_the_cli_refuses_to_score_an_episode_without_a_menu(tmp_path, matrix, capsys):
    """Without ``--offered`` every name would score ``name_not_offered`` and
    the total would describe the missing flag, not the episode."""
    import event_reward

    run = _write_log(tmp_path / "RUN", [_llm_step([REFUTED])])
    assert event_reward.main(["--matrix", str(matrix), "--episode", str(run)]) == 2
    assert "--offered is required" in capsys.readouterr().err
    assert event_reward.main(
        ["--matrix", str(matrix), "--episode", str(run), "--offered", ",".join(OFFERED)]
    ) == 0


def test_an_episode_reads_its_labels_on_the_first_step_only(grid):
    """Labels for later steps are ignored by the scorer, not by its callers."""
    from event_reward import LoggedEpisode, LoggedStep

    def step(n, names, axis):
        return LoggedStep(
            n=n, raw=_raw(next_mitigations=names, verdict="fail", detectors=NAN),
            mitigation_axis_before=axis, stop=False,
        )

    episode = LoggedEpisode(
        run_dir=Path("."),
        steps=[
            step(1, [REFUTED], [BASELINE]),
            step(2, ["xnack"], [BASELINE, REFUTED]),
            step(3, [RESOLVER], [BASELINE, REFUTED, "xnack"]),
        ],
        terminal="converged", terminal_why="test", cells=4,
    )
    label = Label(verdict="fail", failure_detectors=list(NAN))
    score = score_episode(episode, grid, _context(), labels={1: label, 2: label, 3: label})
    read = [(e.name, e.step) for e in score.events
            if e.name in ("verdict_correct", "verdict_wrong", "detector_attribution")]
    assert read == [("verdict_correct", 1), ("detector_attribution", 1)]
