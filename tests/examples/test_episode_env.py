"""Multi-step episodes: does the offline driver decide what the real loop decides?

CPU-only and engine-free. Ground truth is a tiny archived probe matrix written
into ``tmp_path`` in the shape the real harness writes; there is no GPU, no
container, no model and no network. The tests that read the real corpus
archives skip unless ``AORTA_RL_CORPUS_ROOT`` points at them.

The claims under test, in order:

* **fidelity**, the reason this file exists: the driver does not call
  ``run_agent_loop``, it calls the same decision functions in the same order,
  and that is checked by running the real loop against the same archive with
  the same replies and requiring the two ``agent_log.jsonl`` files to match;
* an episode terminates, every way it can, and each terminal is the one
  ``event_reward.episode_from_log`` assigns from the log;
* the events a single reply cannot fire are reachable here;
* a stop the candidate filter manufactured is not credited as a conclusion;
* and the environment refuses an archive it cannot answer from.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import episode_env as env  # noqa: E402
from event_reward import (  # noqa: E402
    POINTS,
    StepContext,
    episode_from_log,
    score_episode,
)
from triage_reward import Label, label_trials  # noqa: E402

from aorta.agent.llm import AUTOPSY_CATEGORIES, PROBE_CATEGORIES  # noqa: E402
from aorta.agent.policy import AgentPolicy  # noqa: E402
from aorta.agent.state import read_trial_results  # noqa: E402

#: Four real registry names, because ``AgentPolicy.validate_step`` resolves
#: every proposed name through the registry and an invented one raises. Chosen
#: to exclude the two approval-gated entries so no test measures that gate by
#: accident; the one test that does adds one on purpose.
MENU = ["tf32_off", "xnack", "hsa_no_sdma", "roc_signal_pool_16k"]
ALPHA, BRAVO, CHARLIE, DELTA = MENU
GATED = "hip_launch_blocking"


# ---------------------------------------------------------------------------
# a tiny archive, in the shape the probe harness writes
# ---------------------------------------------------------------------------


def write_cell(root: Path, cell: str, verdicts: list[str], detectors: list[str]) -> None:
    for index, verdict in enumerate(verdicts):
        trial = root / cell / f"trial_{index}"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "cell_name": cell,
                    "verdict": verdict,
                    "exit_code": 0 if verdict == "pass" else 1,
                    "failure_detectors_fired": detectors if verdict != "pass" else [],
                    "warn_detectors_fired": [],
                    "capture": {},
                }
            ),
            encoding="utf-8",
        )


def build_archive(
    tmp_path: Path,
    resolver: str | None,
    name: str = "ARCHIVE",
    menu: list[str] = MENU,
) -> Path:
    """``none-none`` fails; every menu name fails except ``resolver``."""
    root = tmp_path / name
    write_cell(root, "none-none", ["fail"], ["tier1:exit_nonzero"])
    for mitigation in menu:
        verdict = "pass" if mitigation == resolver else "fail"
        write_cell(root, f"{mitigation}-none", [verdict], ["tier1:exit_nonzero"])
    return root


def make_scenario(root: Path, menu: list[str] = MENU) -> env.Scenario:
    return env.load_scenario("toy", root, "toy", offered=menu)


def reply(mitigations: list[str], *, stop: bool = False, category: str = "unknown") -> str:
    return json.dumps(
        {
            "verdict": "fail",
            "detectors": ["tier1:exit_nonzero"],
            "category": category,
            "hypothesis": "h",
            "next_mitigations": mitigations,
            "confidence": 0.5,
            "stop": stop,
        }
    )


def drive(scenario: env.Scenario, replies: list[str], **kwargs) -> env.Episode:
    """Run one episode to termination against a scripted reply list."""
    policy = kwargs.pop("policy", AgentPolicy(max_iterations=8))
    episode = env.Episode(scenario=scenario, index=0, policy=policy, **kwargs)
    for raw in replies:
        prompt = episode.observe()
        if prompt is None:
            return episode
        episode.pending_prompt = prompt
        episode.act(raw)
    episode.observe()
    return episode


def read_events(path: Path) -> list[dict]:
    return [
        {k: v for k, v in json.loads(line).items() if k != "ts"}
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_real_loop(tmp_path, monkeypatch, archive: Path, scripted: list[str], **policy_kw):
    """``run_agent_loop`` for real, with only the executor replaced."""
    from aorta.agent import loop as agent_loop
    from aorta.agent.llm import _remaining_candidates, _step_from_content

    def fake_execute(config, ticket, mitigation_axis, recipe_template):
        run_dir = config.output_dir / "TOY"
        run_dir.mkdir(parents=True, exist_ok=True)
        for mitigation in mitigation_axis:
            cell = f"{mitigation}-none"
            source = archive / cell / "trial_0" / "result.json"
            if source.is_file():
                target = run_dir / cell / "trial_0"
                target.mkdir(parents=True, exist_ok=True)
                (target / "result.json").write_text(source.read_text(encoding="utf-8"))
        return run_dir

    monkeypatch.setattr(agent_loop, "_execute_probe_matrix", fake_execute)

    class Scripted:
        """Parses the same wire text through the same parser, with no
        empty-menu shortcut -- the environment has none either."""

        def __init__(self) -> None:
            self.n = 0

        def propose(self, *, symptom, cell_summaries, candidates, tried):
            raw = scripted[min(self.n, len(scripted) - 1)]
            self.n += 1
            return _step_from_content(raw, _remaining_candidates(list(candidates), list(tried)))

    config = agent_loop.AgentConfig(
        output_dir=tmp_path / "loop-out",
        ticket="TOY",
        subprocess_argv=(),
        mitigations_allowlist=tuple(MENU),
        policy=AgentPolicy(max_iterations=policy_kw.pop("max_iterations", 8), **policy_kw),
    )
    return agent_loop.run_agent_loop(config, proposer=Scripted())


# ---------------------------------------------------------------------------
# 1. fidelity: the same decisions as the real loop, measured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, resolver, scripted, policy_kw",
    [
        ("converges", CHARLIE, [reply([ALPHA]), reply([BRAVO]), reply([CHARLIE])], {}),
        # A reply that re-proposes a tried name: the filter empties it, and the
        # loop stops on `proposal_unresolved` rather than crediting a decision.
        ("repeats_a_name", None, [reply([ALPHA]), reply([ALPHA])], {}),
        ("explicit_stop", DELTA, [reply([ALPHA]), reply([], stop=True)], {}),
        ("iteration_budget", None, [reply([ALPHA]), reply([BRAVO]), reply([CHARLIE])],
         {"max_iterations": 2}),
        ("unparseable_reply", DELTA, ["not JSON at all"], {}),
        ("refused_category", DELTA, [reply([ALPHA], category="gpu_race")], {}),
        ("partial_drop", DELTA, [reply([ALPHA, "invented"]), reply([DELTA])], {}),
    ],
)
def test_the_driver_and_run_agent_loop_write_the_same_log(
    tmp_path, monkeypatch, case, resolver, scripted, policy_kw
):
    """The fidelity claim, checked by execution rather than by reading.

    ``run_agent_loop`` is driven for real -- its own budget check, its own stop
    resolution, its own log writer -- with ``_execute_probe_matrix`` patched to
    materialise the archived cells for the current axis. Requiring the logs to
    match event for event pins the order of the records, the axis, and the stop
    classification; ``session_start`` differs only because one invocation ran
    a subprocess and the other did not.
    """
    archive = build_archive(tmp_path, resolver=resolver)
    scenario = make_scenario(archive)
    result = run_real_loop(tmp_path, monkeypatch, archive, scripted, **dict(policy_kw))
    loop_events = read_events(result.run_dir / "agent_log.jsonl")

    run_dir = tmp_path / "env-out"
    run_dir.mkdir()
    policy = AgentPolicy(max_iterations=policy_kw.get("max_iterations", 8))
    episode = drive(scenario, list(scripted), run_dir=run_dir, policy=policy)
    env_events = read_events(run_dir / "agent_log.jsonl")

    assert loop_events[0]["type"] == env_events[0]["type"] == "session_start"
    assert env_events[1:] == loop_events[1:], (
        f"[{case}] the offline driver diverged from run_agent_loop:\n"
        f"  loop: {json.dumps(loop_events[1:], indent=1)}\n"
        f"  env : {json.dumps(env_events[1:], indent=1)}"
    )
    assert result.outcome == episode.outcome, case


def test_a_repeated_name_stops_on_the_filter_and_says_which_name(tmp_path):
    archive = build_archive(tmp_path, resolver=None)
    episode = drive(make_scenario(archive), [reply([ALPHA]), reply([ALPHA])])
    assert episode.outcome == "proposal_unresolved"
    assert episode.steps[-1].unresolved_mitigations == [ALPHA]


# ---------------------------------------------------------------------------
# 2. episodes terminate, and the terminal is the one the scorer expects
# ---------------------------------------------------------------------------


def test_convergence_ends_the_episode_and_no_later_reply_is_consumed(tmp_path):
    episode = drive(make_scenario(build_archive(tmp_path, resolver=ALPHA)),
                    [reply([ALPHA]), reply([BRAVO])])
    assert episode.terminal == "converged"
    assert len(episode.steps) == 1


def test_the_iteration_budget_ends_an_episode_as_a_withheld_terminal(tmp_path):
    """A budget stop is a statement about the budget, not the search."""
    episode = drive(
        make_scenario(build_archive(tmp_path, resolver=None)),
        [reply([ALPHA]), reply([BRAVO]), reply([CHARLIE])],
        policy=AgentPolicy(max_iterations=2),
    )
    assert episode.outcome == "policy_stop"
    assert episode.terminal == "other"
    assert len(episode.steps) == 2


def test_a_stop_that_names_the_resolver_neither_converges_nor_earns_it(tmp_path):
    """The loop stops before running it, so the episode ends unresolved and
    the step-1 resolver rate does not count it."""
    scenario = make_scenario(build_archive(tmp_path, resolver=ALPHA))
    episode = drive(scenario, [reply([ALPHA], stop=True)])
    assert episode.terminal == "gave_up"
    assert not env.score(episode).fired("resolver_named")
    group, _samples, _wire = env.rollout_scenario(
        scenario, 2, AgentPolicy(), by_round([reply([ALPHA], stop=True)]),
        advantage_fn=grpo_advantages,
    )
    assert group["step1_resolver_rate"] == 0.0
    assert group["converged_rate"] == 0.0


def test_an_explicit_stop_on_a_solvable_scenario_is_giving_up(tmp_path):
    episode = drive(make_scenario(build_archive(tmp_path, resolver=ALPHA)),
                    [reply([], stop=True), reply([ALPHA])])
    assert episode.terminal == "gave_up"
    assert len(episode.steps) == 1


def test_a_malformed_reply_ends_the_episode_as_a_stop_not_an_exception(tmp_path):
    """An exception would abort a batched iteration holding the whole group."""
    episode = drive(make_scenario(build_archive(tmp_path, resolver=ALPHA)),
                    ["this is not JSON", reply([ALPHA])])
    assert episode.done
    assert len(episode.steps) == 1


def test_a_refused_category_is_scored_but_never_logged(tmp_path):
    """``validate_step`` runs before the log, so the artifact never sees the
    step; the scorer still prices the reply. The one place the two disagree,
    pinned so it stays deliberate."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    episode = drive(scenario, [reply([ALPHA], category="gpu_race")], run_dir=run_dir)
    assert episode.outcome == "policy_stop"
    assert [e["type"] for e in read_events(run_dir / "agent_log.jsonl")] == [
        "session_start", "policy_stop",
    ]
    assert env.score(episode).count("malformed_reply") == 1


def test_an_approval_gated_name_pauses_the_episode(tmp_path):
    menu = MENU + [GATED]
    scenario = make_scenario(build_archive(tmp_path, resolver=None, menu=menu), menu=menu)
    episode = drive(scenario, [reply([GATED])],
                    policy=AgentPolicy(max_iterations=8, require_approval=True))
    assert episode.outcome == "approval_required"
    assert episode.terminal == "other"


def test_an_exhausted_menu_still_asks_the_policy(tmp_path):
    """The shipped real-LLM proposers stop on an empty menu without calling
    the model. The environment asks anyway, because that shortcut's stop is
    the proposer's and crediting a claim to it would pay for nobody's
    conclusion."""
    scenario = make_scenario(build_archive(tmp_path, resolver=None))
    episode = env.Episode(scenario=scenario, index=0, policy=AgentPolicy(max_iterations=8))
    episode.pending_prompt = episode.observe()
    episode.act(reply(MENU))
    prompt = episode.observe()
    assert prompt is not None
    assert json.loads(prompt)["candidates"] == []


@pytest.mark.parametrize(
    "resolver, replies",
    [
        (CHARLIE, [reply([ALPHA]), reply([CHARLIE])]),
        (None, [reply([ALPHA]), reply([BRAVO]), reply([CHARLIE]), reply([DELTA])]),
        (None, [reply([ALPHA]), reply([], stop=True)]),
        (None, [reply(MENU), reply([], stop=True)]),
        (None, [reply(MENU), reply([ALPHA])]),
        (ALPHA, [reply([BRAVO]), reply([], stop=True)]),
        (DELTA, [reply([ALPHA, "invented"]), reply([ALPHA])]),
        (ALPHA, [reply([ALPHA], stop=True)]),
    ],
)
def test_the_driver_terminal_matches_episode_from_log(tmp_path, resolver, replies):
    """Two classifiers, one answer -- and the same score from either artifact.

    The driver decides the terminal in flight; ``episode_from_log`` decides it
    afterwards from the log the driver wrote. Agreeing on the terminal, the
    cells and the score is what lets a reader re-derive an episode's reward
    from the artifact without trusting the harness.
    """
    scenario = make_scenario(build_archive(tmp_path, resolver=resolver))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    episode = drive(scenario, replies, run_dir=run_dir)
    replayed = episode_from_log(run_dir, scenario.grid, scenario.offered)
    assert replayed.terminal == episode.terminal
    assert replayed.cells == episode.cells_spent
    assert len(replayed.steps) == len(episode.steps)
    context = StepContext(offered_mitigations=frozenset(scenario.offered))
    labels = {s.n: scenario.label for s in episode.steps}
    from_log = score_episode(replayed, scenario.grid, context, labels=labels)
    # The log records the parsed triage answer nowhere, so compare the events
    # the log can carry: everything but the two evidence-read events.
    def decided(score):
        return sorted(
            (e.name, e.points, e.step) for e in score.events
            if e.name not in ("verdict_correct", "verdict_wrong", "detector_attribution")
        )
    assert decided(from_log) == decided(env.score(episode))


# ---------------------------------------------------------------------------
# 3. the unresolvable terminal: earned, and only the policy's
# ---------------------------------------------------------------------------


def test_exhausting_the_menu_and_stopping_earns_the_unresolvable_terminal(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=None))
    episode = drive(scenario, [reply(MENU), reply([], stop=True)])
    assert episode.terminal == "unresolvable_correct"
    assert env.score(episode).fired("terminal_unresolvable_correct")


def test_stopping_early_on_an_unresolvable_scenario_is_unearned(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=None))
    episode = drive(scenario, [reply([ALPHA]), reply([], stop=True)])
    assert episode.terminal == "unresolvable_unearned"


def test_a_stop_the_filter_manufactured_is_not_the_unresolvability_claim(tmp_path):
    """Every precondition of the old rule holds -- unresolvable scenario, empty
    list at the stop, whole menu tried -- and the award is not paid, because
    the policy never said stop: the filter emptied its list."""
    scenario = make_scenario(build_archive(tmp_path, resolver=None))
    episode = drive(scenario, [reply([m]) for m in MENU] + [reply([ALPHA])])
    assert scenario.unresolvable
    assert set(episode.tried_mitigations) == set(MENU)
    assert not any(s.stop for s in episode.steps)
    assert episode.terminal == "proposal_unresolved"
    assert episode.steps[-1].unresolved_mitigations == [ALPHA]


def test_an_explicit_stop_carrying_a_tried_name_is_still_the_policys_claim(tmp_path):
    """Narrowness: the filter did drop a name from this reply, and the policy
    also said stop, so the stop is a decision and the claim is earned."""
    scenario = make_scenario(build_archive(tmp_path, resolver=None))
    episode = drive(scenario, [reply([m]) for m in MENU] + [reply([ALPHA], stop=True)])
    assert episode.steps[-1].unresolved_mitigations == [ALPHA]
    assert episode.terminal == "unresolvable_correct"


# ---------------------------------------------------------------------------
# 4. the step-1 prompt
# ---------------------------------------------------------------------------


def test_step_one_shows_the_baseline_cell_and_the_whole_menu(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    episode = env.Episode(scenario=scenario, index=0, policy=AgentPolicy())
    prompt = json.loads(episode.observe())
    assert [row["cell_name"] for row in prompt["cell_summaries"]] == ["none-none"]
    assert prompt["candidates"] == MENU


def test_the_prompt_offers_only_categories_the_loop_accepts():
    """A category ``validate_step`` refuses would end the episode, so the
    system prompt must not offer one."""
    offered = set(ast.literal_eval(
        env.SYSTEM.split("`category` must be one of ")[1].split(".\n")[0]
    ))
    assert offered == set(PROBE_CATEGORIES)
    assert offered < set(AUTOPSY_CATEGORIES)


def test_each_step_carries_its_own_prompt(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    episode = drive(scenario, [reply([ALPHA]), reply([BRAVO]), reply([DELTA])])
    prompts = [json.loads(s.prompt) for s in episode.steps]
    assert [len(p["cell_summaries"]) for p in prompts] == [1, 2, 3]
    assert [len(p["candidates"]) for p in prompts] == [4, 3, 2]


# ---------------------------------------------------------------------------
# 5. events a single reply cannot fire
# ---------------------------------------------------------------------------


def test_name_already_tried_is_reachable_in_an_episode(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    episode = drive(scenario, [reply([ALPHA]), reply([ALPHA, BRAVO])])
    assert env.score(episode).count("name_already_tried") == 1


def test_name_not_offered_is_reachable_and_the_log_agrees(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    episode = drive(scenario, [reply(["invented_name", ALPHA]), reply([DELTA])], run_dir=run_dir)
    assert env.score(episode).count("name_not_offered") == 1
    from_log = score_episode(
        episode_from_log(run_dir, scenario.grid, scenario.offered),
        scenario.grid,
        StepContext(offered_mitigations=frozenset(scenario.offered)),
    )
    assert from_log.count("name_not_offered") == 1
    assert episode.steps[0].unresolved_mitigations == ["invented_name"]


def test_malformed_reply_is_reachable_because_the_wire_text_is_scored(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=ALPHA))
    episode = drive(scenario, ["not json at all"])
    assert env.score(episode).count("malformed_reply") == 1


def test_stalling_out_is_not_a_cheaper_exit_than_quitting(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))

    def total(last: str) -> float:
        return env.score(drive(scenario, [reply([ALPHA]), last])).total

    assert total(reply([ALPHA])) < total(reply([], stop=True))


# ---------------------------------------------------------------------------
# 6. the refusals
# ---------------------------------------------------------------------------


def test_an_archive_that_cannot_answer_the_whole_menu_is_refused(tmp_path):
    archive = build_archive(tmp_path, resolver=None, menu=MENU[:2])
    with pytest.raises(ValueError, match="no cell for 2 offered"):
        make_scenario(archive)


def test_an_archive_whose_trials_disagree_is_refused(tmp_path):
    archive = build_archive(tmp_path, resolver=None)
    write_cell(archive, f"{ALPHA}-none", ["fail", "pass"], ["tier1:exit_nonzero"])
    with pytest.raises(ValueError, match="trials that disagree"):
        make_scenario(archive)


@pytest.mark.parametrize("damage, named", [
    ("{not json", "JSONDecodeError"),
    ("[1, 2]", "not a JSON object"),
    (None, "missing"),
    (b"\xff\xfe", "UnicodeDecodeError"),
])
def test_an_unreadable_trial_is_refused_and_named(tmp_path, damage, named):
    """The disagreeing trial may be the one that cannot be read, so a cell with
    one is not three trials that agree."""
    archive = build_archive(tmp_path, resolver=None)
    write_cell(archive, f"{ALPHA}-none", ["fail", "fail", "fail", "pass"], ["tier1:exit_nonzero"])
    bad = archive / f"{ALPHA}-none" / "trial_3" / "result.json"
    if damage is None:
        bad.unlink()
    elif isinstance(damage, bytes):
        bad.write_bytes(damage)
    else:
        bad.write_text(damage)
    with pytest.raises(ValueError, match=rf"cannot be read.*trial_3/result.json: {named}"):
        make_scenario(archive)


def test_an_unreadable_baseline_trial_is_refused(tmp_path):
    archive = build_archive(tmp_path, resolver=None)
    write_cell(archive, "none-none", ["fail", "fail"], ["tier1:exit_nonzero"])
    (archive / "none-none" / "trial_1" / "result.json").write_text("")
    with pytest.raises(ValueError, match="none-none.*cannot be read"):
        make_scenario(archive)


def test_strict_trial_results_orders_by_index_like_the_shared_reader(tmp_path):
    """Narrowness: on a clean cell it returns what the loop's reader returns."""
    archive = build_archive(tmp_path, resolver=None)
    verdicts = ["fail"] * 12
    write_cell(archive, f"{ALPHA}-none", verdicts, ["tier1:exit_nonzero"])
    cell = archive / f"{ALPHA}-none"
    for index in range(len(verdicts)):
        # Tell the trials apart, so trial_10 sorting before trial_2 would show.
        path = cell / f"trial_{index}" / "result.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "trial": index}))
    docs = env.strict_trial_results(cell)
    assert [d["trial"] for d in docs] == list(range(len(verdicts)))
    assert docs == read_trial_results(cell)


def test_agreeing_repeated_trials_are_accepted(tmp_path):
    """Narrowness: several trials are fine as long as they agree."""
    archive = build_archive(tmp_path, resolver=None)
    write_cell(archive, f"{ALPHA}-none", ["fail", "fail", "fail"], ["tier1:exit_nonzero"])
    assert make_scenario(archive).unresolvable


def test_an_archive_whose_baseline_passes_is_refused(tmp_path):
    archive = tmp_path / "CLEAN"
    write_cell(archive, "none-none", ["pass"], [])
    for mitigation in MENU:
        write_cell(archive, f"{mitigation}-none", ["pass"], [])
    with pytest.raises(ValueError, match="did not fail"):
        make_scenario(archive)


def test_an_archive_without_a_baseline_cell_is_refused(tmp_path):
    archive = build_archive(tmp_path, resolver=None)
    for trial in (archive / "none-none").iterdir():
        (trial / "result.json").unlink()
    with pytest.raises(ValueError, match="no none-none cell"):
        make_scenario(archive)


def test_the_archive_digest_is_its_content_not_its_location(tmp_path):
    archive = build_archive(tmp_path, resolver=DELTA)
    digest = env.archive_digest(archive)
    assert make_scenario(archive).digest == digest
    import shutil

    moved = shutil.copytree(archive, tmp_path / "elsewhere" / "ARCHIVE")
    assert env.archive_digest(moved) == digest, "the mount point is not the answer key"
    trial = moved / f"{ALPHA}-none" / "trial_0" / "result.json"
    trial.write_text(trial.read_text().replace('"fail"', '"pass"', 1))
    assert env.archive_digest(moved) != digest, "an edited verdict is a different answer key"


def test_an_archive_with_nothing_to_digest_is_refused(tmp_path):
    (tmp_path / "EMPTY").mkdir()
    with pytest.raises(ValueError, match="no trial"):
        env.archive_digest(tmp_path / "EMPTY")


def test_corpus_digests_refuses_an_unknown_id(tmp_path):
    with pytest.raises(SystemExit, match="not in the corpus"):
        env.corpus_digests(tmp_path, only=["typo"])


def test_no_corpus_root_is_a_refusal_naming_the_fix(monkeypatch):
    monkeypatch.delenv(env.CORPUS_ROOT_ENV, raising=False)
    with pytest.raises(SystemExit, match="--corpus-root"):
        env.resolve_corpus_root(None)


def test_an_unknown_scenario_id_is_an_error_not_a_smaller_corpus(tmp_path):
    with pytest.raises(SystemExit, match="not in the corpus"):
        env.load_corpus(tmp_path, only=["nan_uninit_workspace", "typo"])


# ---------------------------------------------------------------------------
# 7. accounting, and the rollout path the trainer calls
# ---------------------------------------------------------------------------


def grpo_advantages(values: list[float]) -> tuple[list[float], float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    sd = var**0.5
    return [(v - mean) / (sd + 1e-4) for v in values], mean, sd


def by_round(rounds: list[str]):
    """A generator keyed on the lockstep round, not on the episode: the point
    of the driver is that a whole group advances in one call."""
    state = {"depth": 0}

    def generate(users: list[str]) -> list[str]:
        state["depth"] += 1
        return [rounds[state["depth"] - 1]] * len(users)

    return generate


def test_cells_are_one_per_mitigation_on_the_axis(tmp_path):
    episode = drive(make_scenario(build_archive(tmp_path, resolver=None)),
                    [reply([ALPHA, BRAVO]), reply([CHARLIE]), reply([], stop=True)])
    assert episode.cells_spent == 4
    assert env.score(episode).count("cell_spent") == 1


def test_rollout_scenario_produces_one_sample_per_step(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    group, samples, wire = env.rollout_scenario(
        scenario, 4, AgentPolicy(max_iterations=8),
        by_round([reply([ALPHA]), reply([BRAVO]), reply([DELTA])]),
        advantage_fn=grpo_advantages,
    )
    assert group["n"] == 4
    assert group["steps_total"] == len(samples) == len(wire) == 12
    assert group["mean_steps"] == 3.0
    assert group["converged_rate"] == 1.0
    assert group["step1_resolver_rate"] == 0.0
    assert group["episode_resolver_rate"] == 1.0
    assert [s.step for s in samples[:3]] == [1, 2, 3]
    assert len({s.prompt for s in samples[:3]}) == 3
    assert group["events"]["resolver_named"] == 4


def test_the_episode_advantage_is_repeated_on_every_step_of_it(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    state = {"n": 0}

    def generate(users: list[str]) -> list[str]:
        state["n"] += 1
        if state["n"] == 1:
            return [reply([DELTA]), reply([ALPHA])]
        return [reply([DELTA])] * len(users)

    _group, samples, _wire = env.rollout_scenario(
        scenario, 2, AgentPolicy(max_iterations=8), generate, advantage_fn=grpo_advantages
    )
    by_episode: dict[int, set[float]] = {}
    for sample in samples:
        by_episode.setdefault(sample.episode, set()).add(sample.advantage)
    assert all(len(v) == 1 for v in by_episode.values()), by_episode
    assert len({next(iter(v)) for v in by_episode.values()}) == 2


def test_a_generator_returning_the_wrong_count_is_an_error(tmp_path):
    """``zip`` would silently leave the extra episodes waiting forever."""
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    with pytest.raises(ValueError, match="1 completion"):
        env.rollout_scenario(scenario, 3, AgentPolicy(), lambda users: [reply([ALPHA])],
                             advantage_fn=grpo_advantages)


def test_an_unresolvable_group_reports_no_resolver_rate(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=None))
    group, _samples, _wire = env.rollout_scenario(
        scenario, 3, AgentPolicy(max_iterations=8),
        by_round([reply(MENU), reply([], stop=True)]), advantage_fn=grpo_advantages,
    )
    assert group["events"].get("terminal_unresolvable_correct") == 3
    assert group["step1_resolver_rate"] is None
    assert group["episode_resolver_rate"] is None


def test_every_cell_on_screen_at_a_proposer_call_has_the_baseline_verdict(tmp_path):
    """Why the baseline label is attached to every step: a passing cell ends
    the episode before the policy is asked again."""
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    episode = env.Episode(scenario=scenario, index=0, policy=AgentPolicy())
    episode.pending_prompt = episode.observe()
    episode.act(reply([ALPHA, BRAVO]))
    prompt = json.loads(episode.observe())
    assert {row["verdict"] for row in prompt["cell_summaries"]} == {"fail"}
    assert isinstance(scenario.label, Label)


def test_the_single_reply_column_charges_the_episode_cell_accounting(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=None))
    score = env.score_completion_compat(reply([ALPHA, BRAVO]), scenario, cells_added=2)
    assert score.count("cell_spent") == 1
    cell_event = next(e for e in score.events if e.name == "cell_spent")
    assert cell_event.points == pytest.approx(3 * POINTS["cell_spent"])


def test_runnable_names_replays_the_filter():
    assert env.runnable_names(reply([ALPHA, "invented"]), MENU) == [ALPHA]
    assert env.runnable_names(reply([ALPHA], stop=True), MENU) == [], "a stop runs nothing"
    stop_as_text = json.dumps({"next_mitigations": [ALPHA], "stop": "true"})
    assert env.runnable_names(stop_as_text, MENU) == [ALPHA], "only a JSON true stops"
    assert env.runnable_names("prose", MENU) == []
    assert env.runnable_names(json.dumps({"next_mitigations": ALPHA}), MENU) == []


# ---------------------------------------------------------------------------
# 8. the real corpus, when it is available
# ---------------------------------------------------------------------------

REAL = os.environ.get(env.CORPUS_ROOT_ENV)
needs_corpus = pytest.mark.skipif(
    not REAL, reason=f"{env.CORPUS_ROOT_ENV} does not point at the archived corpus"
)


@needs_corpus
def test_every_corpus_entry_loads_and_answers_the_whole_menu():
    scenarios = env.load_corpus(REAL)
    assert [s.scenario_id for s in scenarios] == [sid for sid, _, _ in env.CORPUS]
    for scenario in scenarios:
        assert set(env.registered_mitigations()) <= set(scenario.grid.measured_mitigations)
    assert [s.scenario_id for s in scenarios if s.unresolvable], (
        "without an unresolvable scenario the earned claim has nothing to measure"
    )


@needs_corpus
def test_no_corpus_cell_fires_a_warn_detector():
    """So the failure/warn union the prompt performs is lossless here."""
    from aorta.agent.loop import _read_cell_summaries

    for _sid, archive, _family in env.CORPUS:
        for row in _read_cell_summaries(Path(REAL) / archive):
            assert not row.get("warn_detectors_fired"), (archive, row["cell_name"])


@needs_corpus
def test_the_baseline_label_is_the_cell_label_on_every_corpus_entry():
    for scenario in env.load_corpus(REAL):
        cell = scenario.root / "none-none"
        assert label_trials(read_trial_results(cell)).verdict == scenario.label.verdict == "fail"


# ---------------------------------------------------------------------------
# the evidence is read once per episode: a slower search must not pay
# ---------------------------------------------------------------------------
#
# The verdict and attribution events used to be paid on every step. A refuted
# step costs less than they pay and the earliness bonus stops at step 2, so a
# policy that copies the triage from its prompt gained on every step it
# delayed. `reply()` copies this archive's triage exactly, so every episode
# below is a reads-nothing policy with the triage matched.


def _padded(k: int) -> list[str]:
    """``k`` names that fix nothing, one per step, then the resolver."""
    return [reply([name]) for name in (ALPHA, BRAVO, CHARLIE)[:k]] + [reply([DELTA])]


def test_padding_a_reads_nothing_episode_never_raises_its_return(tmp_path):
    """The tripwire: the same answer reached later must score strictly less."""
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    totals = []
    for k in range(4):
        # Two identical episodes, because the normaliser needs a group of two.
        group, samples, _wire = env.rollout_scenario(
            scenario, 2, AgentPolicy(max_iterations=8), by_round(_padded(k)),
            advantage_fn=grpo_advantages,
        )
        assert len(samples) == 2 * (k + 1) and group["converged_rate"] == 1.0
        totals.append(group["rewards"][0])
    assert all(a > b for a, b in zip(totals, totals[1:], strict=False)), totals


def test_the_evidence_is_paid_once_per_episode_on_its_first_step(tmp_path):
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    group, _samples, _wire = env.rollout_scenario(
        scenario, 4, AgentPolicy(max_iterations=8), by_round(_padded(2)),
        advantage_fn=grpo_advantages,
    )
    assert group["steps_total"] == 12
    assert group["events"]["verdict_correct"] == 4
    assert group["events"]["detector_attribution"] == 4


def test_a_wrong_first_read_is_charged_and_not_repaired_by_a_later_copy(tmp_path):
    """Narrowness: the read is decided once, not deleted, and later proposals still score."""
    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    wrong = json.dumps(
        {"verdict": "pass", "detectors": [], "category": "unknown", "hypothesis": "h",
         "next_mitigations": [ALPHA], "confidence": 0.5, "stop": False}
    )
    episode = drive(scenario, [wrong, reply([BRAVO]), reply([DELTA])])
    score = score_episode(
        episode.as_logged(), scenario.grid,
        StepContext(offered_mitigations=frozenset(MENU)),
        labels={s.n: scenario.label for s in episode.steps},
    )
    fired = [(e.name, e.step) for e in score.events]
    assert ("verdict_wrong", 1) in fired and ("detector_attribution", 1) in fired
    assert not any(name == "verdict_correct" for name, _ in fired)
    assert ("name_refuted", 2) in fired and ("resolver_named", 3) in fired


def test_a_one_step_episode_reads_the_evidence_as_a_single_completion_does(tmp_path):
    from event_reward import score_completion

    scenario = make_scenario(build_archive(tmp_path, resolver=DELTA))
    raw = reply([DELTA])
    context = StepContext(offered_mitigations=frozenset(MENU))
    in_episode = score_episode(
        drive(scenario, [raw]).as_logged(), scenario.grid, context,
        labels={1: scenario.label},
    )
    single = score_completion(raw, scenario.grid, context, label=scenario.label)
    read = ("verdict_correct", "verdict_wrong", "detector_attribution")
    assert [(e.name, e.points) for e in in_episode.events if e.name in read] == [
        (e.name, e.points) for e in single.events if e.name in read
    ] == [("verdict_correct", 1.0), ("detector_attribution", 1.0)]
