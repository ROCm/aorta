"""The offline reports over an episode run: constants, the wire, the replay.

CPU-only, against a toy archive. The load-bearing one is the replay: it is how
a recorded run is shown to be reproducible from its own wire, so it has to
report a mismatch when there is one rather than agreeing on whatever it could
match.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "examples"))

import episode_env as env  # noqa: E402
import rescore_episodes as rescore  # noqa: E402
from test_episode_env import MENU, build_archive, make_scenario, reply  # noqa: E402

from aorta.agent.policy import AgentPolicy  # noqa: E402

ALPHA, BRAVO, CHARLIE, DELTA = MENU


@pytest.fixture
def scenario(tmp_path):
    return make_scenario(build_archive(tmp_path, resolver=CHARLIE))


def recorded(scenario, rounds: list[str], count: int = 2) -> list[dict]:
    state = {"depth": 0}

    def generate(users):
        state["depth"] += 1
        return [rounds[state["depth"] - 1]] * len(users)

    _group, _samples, wire = env.rollout_scenario(
        scenario, count, AgentPolicy(max_iterations=8), generate,
        advantage_fn=rescore.advantages,
    )
    return [{"iteration": 1, **row} for row in wire]


def test_the_normaliser_matches_the_one_the_rollout_path_was_tested_with():
    adv, mean, sd = rescore.advantages([1.0, 2.0, 3.0])
    assert mean == 2.0 and sd == pytest.approx(1.0)
    assert adv == pytest.approx([-1.0 / 1.0001, 0.0, 1.0 / 1.0001])
    assert rescore.advantages([5.0])[2] == 0.0


def test_a_recorded_run_replays_exactly(scenario):
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    result = rescore.replay(rows, [scenario], AgentPolicy(max_iterations=8))
    assert result["episodes"] == 2
    assert result["mismatched_steps"] == 0
    assert result["max_reward_diff"] == 0.0
    assert result["moves"] == {"converged -> converged": 2}


def test_a_tampered_reward_is_reported_not_absorbed(scenario):
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    for row in rows:
        row["reward"] += 1.0
    assert rescore.replay(rows, [scenario], AgentPolicy())["max_reward_diff"] == pytest.approx(1.0)


def test_a_replay_that_walks_a_different_path_is_flagged(scenario):
    """A record holding a reply after convergence cannot be replayed step for
    step: the environment ends the episode first, so the replay consumes
    fewer replies than the record holds and says so."""
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    extra = [dict(r, step=3, raw=reply([BRAVO])) for r in rows if r["step"] == 2]
    result = rescore.replay(rows + extra, [scenario], AgentPolicy())
    assert result["mismatched_steps"] == 2


def test_an_empty_wire_is_a_refusal_not_a_reproduction(scenario):
    """Zero episodes replayed is zero mismatches; that is not evidence."""
    with pytest.raises(ValueError, match="nothing to replay"):
        rescore.replay([], [scenario], AgentPolicy())


def test_a_scenario_the_corpus_does_not_have_is_a_refusal(scenario):
    rows = recorded(scenario, [reply([CHARLIE])])
    for row in rows:
        row["scenario_id"] = "elsewhere"
    with pytest.raises(ValueError, match="not in the corpus"):
        rescore.replay(rows, [scenario], AgentPolicy())


def test_the_cli_exits_non_zero_when_the_record_does_not_reproduce(
    tmp_path, scenario, monkeypatch
):
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    wire = tmp_path / "wire.jsonl"
    monkeypatch.setattr(rescore.env, "load_corpus", lambda *_a, **_k: [scenario])
    wire.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert rescore.main(["--wire", str(wire), "--replay"]) == 0
    rows[0]["reward"] += 0.5
    wire.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert rescore.main(["--wire", str(wire), "--replay"]) == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_recorded_reward_is_a_mismatch(scenario, bad):
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    rows[0]["reward"] = bad
    assert rescore.replay(rows, [scenario], AgentPolicy())["max_reward_diff"] == float("inf")


def test_a_moved_terminal_alone_is_a_non_zero_exit(tmp_path, scenario, monkeypatch):
    """Same steps, same reward, a different terminal: still not a reproduction."""
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    for row in rows:
        row["terminal"] = "other"
    result = rescore.replay(rows, [scenario], AgentPolicy())
    assert result["max_reward_diff"] == 0.0 and result["mismatched_steps"] == 0
    assert result["terminals_moved"] == 2
    wire = tmp_path / "wire.jsonl"
    wire.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(rescore.env, "load_corpus", lambda *_a, **_k: [scenario])
    assert rescore.main(["--wire", str(wire), "--replay"]) == 1


def _column(tmp_path, rows, corpus):
    column = tmp_path / "after.json"
    config = {} if corpus is None else {"corpus": corpus}
    column.write_text(json.dumps({"config": config, "groups": [], "wire": rows}))
    return column


def test_a_replay_against_changed_archives_is_refused_not_reported_as_a_rule_change(
    tmp_path, scenario, monkeypatch, capsys
):
    """Same scenario ID, different contents: the exit code must not say "the rule moved"."""
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    column = _column(tmp_path, rows, {scenario.scenario_id: "another-digest"})
    monkeypatch.setattr(rescore.env, "load_corpus", lambda *_a, **_k: [scenario])
    monkeypatch.setattr(rescore, "replay", lambda *a: pytest.fail("replayed a stale record"))
    assert rescore.main(["--wire", str(column), "--replay"]) == rescore.EXIT_CORPUS_CHANGED
    assert rescore.EXIT_CORPUS_CHANGED not in (0, 1, 2)
    assert f"for ['{scenario.scenario_id}']" in capsys.readouterr().err


def test_a_replay_against_the_recorded_archives_runs(tmp_path, scenario, monkeypatch):
    """Narrowness: matching digests replay as before, and say they were checked."""
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    column = _column(tmp_path, rows, {scenario.scenario_id: scenario.digest})
    monkeypatch.setattr(rescore.env, "load_corpus", lambda *_a, **_k: [scenario])
    out = tmp_path / "result.json"
    assert rescore.main(["--wire", str(column), "--replay", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["replay"]["corpus_verified"] is True


def test_a_trainer_wire_is_checked_against_its_train_log(tmp_path, scenario, monkeypatch):
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    wire = tmp_path / "wire.jsonl"
    wire.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(rescore.env, "load_corpus", lambda *_a, **_k: [scenario])
    log = tmp_path / "train-log.json"
    log.write_text(json.dumps({"corpus": {scenario.scenario_id: "another-digest"}}))
    assert rescore.main(["--wire", str(wire), "--replay"]) == rescore.EXIT_CORPUS_CHANGED
    log.write_text(json.dumps({"corpus": {scenario.scenario_id: scenario.digest}}))
    assert rescore.main(["--wire", str(wire), "--replay"]) == 0


def test_a_record_without_digests_replays_but_says_it_is_unverified(
    tmp_path, scenario, monkeypatch, capsys
):
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    column = _column(tmp_path, rows, None)
    monkeypatch.setattr(rescore.env, "load_corpus", lambda *_a, **_k: [scenario])
    out = tmp_path / "result.json"
    assert rescore.main(["--wire", str(column), "--replay", "--json", str(out)]) == 0
    assert "carries no corpus digests" in capsys.readouterr().err
    assert json.loads(out.read_text())["replay"]["corpus_verified"] is False


def test_only_the_scenarios_the_record_names_are_checked(scenario):
    rows = recorded(scenario, [reply([CHARLIE])])
    sid = scenario.scenario_id
    assert rescore.corpus_mismatch({sid: scenario.digest, "other": "x"}, rows, [scenario]) == []
    assert rescore.corpus_mismatch({"other": "x"}, rows, [scenario]) == [sid]


def test_the_trainer_records_what_a_replay_checks():
    """The other half: a train-log has to carry the digests or every replay of
    it is unverified."""
    source = (Path(__file__).resolve().parents[2] / "examples" / "rl" / "train_grpo_step.py")
    assert '"corpus": {s.scenario_id: s.digest for s in scenarios}' in source.read_text()


def test_an_eval_column_is_read_as_a_wire(tmp_path, scenario):
    rows = recorded(scenario, [reply([CHARLIE])])
    column = tmp_path / "after.json"
    column.write_text(json.dumps({"config": {}, "groups": [], "wire": rows}))
    assert rescore.read_wire(column)[0]["scenario_id"] == scenario.scenario_id
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"groups": []}))
    with pytest.raises(ValueError, match="'wire' list"):
        rescore.read_wire(bad)


def test_the_summary_separates_the_step_one_rate_from_the_episode_rate(scenario):
    rows = recorded(scenario, [reply([ALPHA]), reply([CHARLIE])])
    series = rescore.summarise(rows, [scenario])["scenarios"][scenario.scenario_id]
    assert series[0]["step1_resolver_rate"] == 0.0
    assert series[0]["episode_resolver_rate"] == 1.0
    assert series[0]["mean_steps"] == 2.0


def test_constants_are_scored_per_scenario_both_ways(tmp_path):
    scenarios = [
        env.load_scenario(f"s{i}", build_archive(tmp_path, resolver=r, name=f"S{i}"), "toy",
                          offered=MENU)
        for i, r in enumerate(sorted(MENU)[:2])
    ]
    rows = rescore.run_constants(scenarios, AgentPolicy(max_iterations=8))
    assert set(rows) >= {"always_the_cover", "say_nothing", "always_prose", "pick_the_first_name"}
    for row in rows.values():
        assert set(row["per_scenario"]) == set(row["per_scenario_single"]) == {"s0", "s1"}
    # `pick_the_first_name` names the alphabetically first name, which
    # resolves s0 on step 1 and is refuted on s1, where its second reply is
    # the same, now tried, name.
    first = rows["pick_the_first_name"]
    assert first["terminals"] == {"converged": 1, "proposal_unresolved": 1}
    assert rows["always_prose"]["mean"] < rows["say_nothing"]["mean"]


def test_the_hard_coded_cover_names_are_registered_mitigations():
    registered = set(env.registered_mitigations())
    assert set(rescore.COVER) <= registered
    assert set(rescore.LEARNED_REFLEX) <= registered
