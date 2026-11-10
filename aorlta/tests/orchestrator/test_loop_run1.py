"""The loop glue for the GPU stand-in runs, on the fakes and the virtual clock:

- D7's fraction counts only the candidates handed to the reward service; a chain an engine lost is a warning, so an
  engine error that C1 survives does not stop the run;
- an iteration where every group has zero advantage is a NOOP (warning, no step), not a stop; one where only some
  groups do still steps."""

from __future__ import annotations

import pytest

from aorlta.config import ConfigError
from aorlta.episodes.writer import complete_iterations, read_episodes, read_summary
from aorlta.orchestrator.fakes import build_fake_run
from aorlta.spans import read_jsonl

NO_ERRORS = {"error_rate": 0.0}


def run(spec):
    r = build_fake_run(spec)
    return r, r.orchestrator.run()


def test_d7_counts_only_the_candidates_handed_over_and_an_engines_lost_chains_are_a_warning(make_spec):
    # A fifth of requests fail (engine seed 4 keeps the start-up self-test clear of them): 8 of the 32 chains end
    # engine_error and never reach the reward service. Every candidate that does scores, so D7 over the candidates
    # trains; over all 32 chains (24 of 32 = 0.75) it would stop.
    r, res = run(make_spec(iterations=1, eval_every=None, engine={"error_rate": 0.2, "seed": 4}))
    assert res["status"] == "completed", res.get("stop")
    s = read_summary(r.cfg.run_dir, 1)
    eps = read_episodes(r.cfg.run_dir, 1)
    lost = [e for e in eps if e["end_reason"] == "engine_error"]
    handed = [e for e in eps if e["candidate_sha256"]]
    assert len(lost) >= 4 and len(handed) + len(lost) == 32 and len(handed) / 32 < 0.9
    first = s["policy"]["first"]
    assert (first["decision"], first["total"], first["scored"]) == ("train", len(handed), s["samples"]["scored"])
    assert s["policy"]["denominator"] == "candidates_handed_over" and s["policy"]["chains"] == 32
    assert s["policy"]["not_handed_over"]["engine_error"] == len(lost)
    assert "engine_error_chains" in s["warnings"]


def test_without_engine_errors_there_is_no_warning_and_the_fraction_is_over_every_chain(make_spec):
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS))
    assert res["status"] == "completed"
    s = read_summary(r.cfg.run_dir, 1)
    assert s["policy"]["first"]["total"] == 32 and "engine_error_chains" not in s["warnings"]


def test_an_iteration_where_every_group_has_zero_advantage_is_a_noop_not_a_stop(make_spec):
    r, res = run(make_spec(iterations=3, eval_every=None, engine=NO_ERRORS, reward={"constant": 0.5}))
    assert res["status"] == "completed", res.get("stop")
    assert complete_iterations(r.cfg.run_dir) == [1, 2, 3]
    assert r.trainer.steps == 0  # no step was ever taken
    for n in (1, 2, 3):
        s = read_summary(r.cfg.run_dir, n)
        assert "zero_advantage" in s["warnings"] and s["trainer"]["step_taken"] is False
        assert s["sync"]["published"] is False and s["groups"]["zero_advantage"] == s["groups"]["total"]
        assert s["spans"]["missing"] == []
        (mark,) = [x for x in read_jsonl(r.cfg.run_dir / "iterations" / f"iter-{n:06d}" / "spans.jsonl")
                   if x.name in ("landing", "no_step")]
        assert (mark.name, mark.t_start, mark.t_end) == ("no_step", s["timing"]["t_landed"], s["timing"]["t_landed"])
        assert mark.attrs == {"reason": "zero_advantage"}
        why = {e["excluded_reason"] for e in read_episodes(r.cfg.run_dir, n) if e["status"] == "scored"}
        assert "no_step" in why and why <= {"no_step", "logprob_error"}  # a logprob_error chain has no sample
    assert read_summary(r.cfg.run_dir, 3)["noop_streak"] == 3 and "noop_steps" in read_summary(r.cfg.run_dir, 2)["warnings"]


def test_an_iteration_with_no_trainable_sample_also_ends_at_its_no_step_marker(make_spec):
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS, trainer={"min_scored": 9}))  # K is 8
    assert res["status"] == "completed", res.get("stop")
    s = read_summary(r.cfg.run_dir, 1)
    assert "no_trainable_samples" in s["warnings"] and s["spans"]["missing"] == []
    spans = read_jsonl(r.cfg.run_dir / "iterations" / "iter-000001" / "spans.jsonl")
    (mark,) = [x for x in spans if x.name == "no_step"]
    assert mark.attrs == {"reason": "no_trainable_samples"} and mark.t_end == s["timing"]["t_landed"]


def test_an_iteration_where_only_some_groups_have_zero_advantage_still_steps(make_spec):
    base, _ = run(make_spec("probe", iterations=1, eval_every=None, engine=NO_ERRORS))
    prompts = sorted({e["prompt_id"] for e in read_episodes(base.cfg.run_dir, 1)})
    r, res = run(make_spec(iterations=1, eval_every=None, engine=NO_ERRORS,
                           reward={"constant": 0.5, "constant_prompts": prompts[1:]}))
    assert res["status"] == "completed"
    s = read_summary(r.cfg.run_dir, 1)
    assert r.trainer.steps == 1 and s["trainer"]["step_taken"] is True and "zero_advantage" not in s["warnings"]
    assert s["groups"]["zero_advantage"] == len(prompts) - 1


def test_a_run_refused_at_its_run_plan_stops_nothing(make_spec):
    """A second process under a run id that another holds started nothing, so it stops nothing: the rollout's
    shutdown removes the run's node directory, which holds the other process's weight slots."""
    spec = make_spec(iterations=1)
    assert run(spec)[1]["status"] == "completed"
    again = build_fake_run(spec).orchestrator
    stopped = []
    again.rollout.shutdown = lambda: stopped.append("rollout")
    again.reward.close = lambda: stopped.append("reward")
    with pytest.raises(ConfigError, match="already set"):
        again.run()
    assert stopped == []
