"""Score mode (`orchestrator.mode: score`): the config, the prompt plan, the report's rules, score runs over the fakes,
the entry point and the run job. Nothing trains and nothing is published; an unscored sample is never clean."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aorlta.config import ConfigError, ScoreOptions, config_from_dict, load_config
from aorlta.episodes.schema import EpisodeError, validate_episode
from aorlta.episodes.writer import complete_iterations, read_episodes
from aorlta.orchestrator import score as S
from aorlta.orchestrator.fakes import FakeRunSpec, build_fake_run, make_pool
from aorlta.orchestrator.ports import RunStopped
from aorlta.orchestrator.state import epoch_order

REPO = Path(__file__).resolve().parents[2]
NO_ERRORS = {"error_rate": 0.0}


def score_dict(tmp_path, **orchestrator) -> dict:
    return {"run": {"run_id": "s", "source": "standin", "root": str(tmp_path / "runs")},
            "model": {"base_dir": str(tmp_path / "base"), "source": "standin"},
            "data": {"prompt_pools": [{"path": str(tmp_path / "pool"), "source": "standin"}]},
            "rollout": {"engines": [0, 1], "K": 4, "P": 4, "compile_adapter": "fake.compile"},
            "reward": {"profile": "fake", "reward_fn": "standin.v11"},
            "orchestrator": {"mode": "score", "run_seed": 7, **orchestrator}}


def spec(tmp_path, name="a", **kw) -> FakeRunSpec:
    base = dict(mode="score", score={"prompts": 6, "passes": 2}, P=4, K=4, pool_size=8, eval_every=None)
    base.update(kw)
    return FakeRunSpec(root=str(tmp_path / name / "runs"), run_id=f"score-{name}", **base)


def scored(spec_):
    r = build_fake_run(spec_)
    return r, r.orchestrator.run()


def report(cfg) -> list[dict]:
    return [json.loads(x) for x in (cfg.run_dir / "score" / "report.jsonl").read_text().splitlines()]


def episodes(cfg) -> list[dict]:
    return [e for n in complete_iterations(cfg.run_dir) for e in read_episodes(cfg.run_dir, n)]


# --- the config ------------------------------------------------------------------------------------------------


def test_a_score_config_needs_no_iterations_and_scores_every_prompt_once_by_default(tmp_path):
    cfg = config_from_dict(score_dict(tmp_path))
    o = cfg.orchestrator
    assert (o.mode, o.iterations, o.score) == ("score", None, ScoreOptions(prompts=None, passes=1))
    o = config_from_dict(score_dict(tmp_path, score={"prompts": 16, "passes": 3})).orchestrator
    assert (o.score.prompts, o.score.passes) == (16, 3)
    adapter = dict(score_dict(tmp_path), model={"base_dir": "/m/base", "source": "standin",
                                                "adapter_dir": "/m/adapter"})
    assert config_from_dict(adapter).model.adapter_dir == "/m/adapter"


def test_a_site_file_may_name_the_adapter_a_score_run_merges(tmp_path):
    path, site = tmp_path / "score.yaml", tmp_path / "site.yaml"
    path.write_text(json.dumps(score_dict(tmp_path)))
    site.write_text("model: {base_dir: /site/base, adapter_dir: /site/adapter}\n")
    model = load_config(path, site=site).model
    assert (model.base_dir, model.adapter_dir) == ("/site/base", "/site/adapter")


@pytest.mark.parametrize("edit, message", [
    ({"orchestrator": {"iterations": 3}}, "iterations: a score run's batches follow from orchestrator.score"),
    ({"orchestrator": {"eval_every": 1}}, "orchestrator.eval_every: a score run trains nothing"),
    ({"sync": {"round_trip_at_start": False}}, "sync.round_trip_at_start: a score run confirms what the engines"),
    ({"orchestrator": {"score": {"prompts": 0}}}, "prompts: must be >= 1 or null"),
    ({"orchestrator": {"score": {"passes": 0}}}, "passes: must be >= 1"),
    ({"orchestrator": {"score": {"limit": 4}}}, "unknown key"),
    ({"model": {"adapter_dir": "relative/adapter"}}, "adapter_dir: must be an absolute path"),
])
def test_a_score_config_refuses_what_a_score_run_does_not_do(tmp_path, edit, message):
    data = score_dict(tmp_path)
    for section, values in edit.items():
        data[section] = {**data.get(section, {}), **values}
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_a_training_config_keeps_iterations_and_refuses_score_options_and_an_adapter(tmp_path):
    train = score_dict(tmp_path)
    train["orchestrator"] = {"run_seed": 7}
    with pytest.raises(ConfigError, match=r"iterations: a training run \(mode v1\) needs it"):
        config_from_dict(train)
    train["orchestrator"] = {"run_seed": 7, "iterations": 2, "score": {"prompts": 4}}
    with pytest.raises(ConfigError, match="score: is read only with mode: score"):
        config_from_dict(train)
    train["orchestrator"] = {"run_seed": 7, "iterations": 2}
    assert config_from_dict(train).orchestrator.mode == "v1"
    train["model"] = {"base_dir": "/m/base", "source": "standin", "adapter_dir": "/m/adapter"}
    with pytest.raises(ConfigError, match="model.adapter_dir: a training run starts from model.base_dir"):
        config_from_dict(train)


# --- the prompt plan -------------------------------------------------------------------------------------------


def test_the_prompts_are_the_first_n_in_the_run_seeds_order_p_at_a_time_once_per_pass():
    records = make_pool(10).records()
    chosen = S.select_score_prompts(records, 7, 6)
    assert chosen == epoch_order(records, 7, 0)[:6] and chosen != records[:6]  # the seeded order, not the pool's
    assert S.select_score_prompts(records, 7, None) == epoch_order(records, 7, 0)
    batches = S.plan_batches(chosen, 4, 2)
    assert [(b.iteration, b.pass_index, b.index, len(b.prompts)) for b in batches] == [
        (1, 1, 1, 4), (2, 1, 2, 2), (3, 2, 1, 4), (4, 2, 2, 2)]
    assert [p for b in batches[:2] for p in b.prompts] == chosen == [p for b in batches[2:] for p in b.prompts]
    with pytest.raises(RunStopped, match="score_prompts_exceed_pool"):
        S.select_score_prompts(records, 7, 11)


# --- the report's rules ----------------------------------------------------------------------------------------


def sample(pid="p", k=0, *, status="scored", reward=0.5, base=0.5, clean=False, end="compiled", turn=0,
           handed=True, legs=None, na=(), reason=None) -> S.Sample:
    return S.Sample(chain_id=f"r/1/{pid}/{k}", prompt_id=pid, iteration=1, end_reason=end, final_turn=turn,
                    handed_over=handed, status=status, reward=reward if status == "scored" else None,
                    base=base if status == "scored" else None, clean=clean,
                    unscored_reason=None if status == "scored" else (reason or "sanitizer_not_settled"),
                    legs=dict(legs or {}), not_applicable=tuple(na))


def episode(status="scored", reward=0.5, **kw) -> dict:
    return {"chain_id": "r/1/p/0", "prompt_id": "p", "iteration": 1, "end_reason": "compiled", "final_turn": 0,
            "candidate_sha256": "0" * 64, "status": status, "reward": reward if status == "scored" else None,
            "unscored_reason": None if status == "scored" else "clean_not_proven",
            "legs": {"gate.compile": "clean", "san.asan": "clean"}, **kw}


def test_an_unscored_sample_is_never_clean_whatever_its_reward_function_claims():
    from aorlta.orchestrator.loop import unscored_result

    # every leg clean and still unscored (the clean reward not proven): the claim does not make it clean
    s = S.sample_of(episode(status="unscored"), unscored_result("r/1/p/0", "clean_not_proven"), clean=True)
    assert (s.status, s.clean, s.reward) == ("unscored", False, None)
    assert S.sample_of(episode(), None, clean=True).clean is True
    claimed = sample(status="unscored", clean=True)  # a record that says clean on an unscored sample
    assert S.summarize([claimed])["clean"] == 0 and not S.is_clean(claimed)
    rows = S.build_report([claimed], [_prompt("p")], {})
    assert rows[1]["any_clean"] is False and rows[-1]["prompts_with_a_clean_sample"] == 0


def _prompt(pid: str):
    rec = make_pool(1).records()[0]
    return type(rec)(prompt_id=pid, turn0_ids=rec.turn0_ids, messages=None, prompt_ids_sha256=rec.prompt_ids_sha256,
                     task=rec.task, source="standin")


def test_the_scored_fraction_counts_every_sample_a_batch_started():
    samples = [sample(k=0), sample(k=1, clean=True, base=1.0, reward=0.97),
               sample(k=2, status="unscored", end="engine_error", turn=-1, handed=False, reason="engine_error"),
               sample(k=3, status="quarantined", end="quarantined", turn=1, handed=False, reason="quarantined")]
    t = S.summarize(samples)
    assert (t["samples"], t["scored"], t["unscored"], t["clean"]) == (4, 2, 2, 1)
    assert t["scored_fraction"] == 0.5 and t["clean_fraction"] == 0.25  # over the 4 started, not the 2 handed over
    assert t["handed_over"] == 2 and t["not_handed_over"] == {"engine_error": 1, "quarantined": 1}
    assert t["unscored_by_reason"] == {"engine_error": 1, "quarantined": 1}


def test_each_check_is_counted_pass_fail_unsettled_or_not_run_and_not_applicable_beside():
    samples = [sample(k=0, legs={"san.consan": "clean", "san.asan": "finding"}),
               sample(k=1, legs={"san.consan": "unsettled", "san.asan": "fail"}, na=("san.consan",)),
               sample(k=2, status="unscored", legs={"san.consan": "not_run", "san.asan": "skipped_by_plan"}),
               sample(k=3, legs={"san.consan": "unsettled", "san.asan": "clean"})]
    checks = S.summarize(samples)["checks"]
    assert checks["san.consan"] == {"pass": 1, "fail": 0, "unsettled": 2, "not_run": 1, "not_applicable": 1}
    assert checks["san.asan"] == {"pass": 1, "fail": 2, "unsettled": 0, "not_run": 1, "not_applicable": 0}
    assert [S.check_outcome(s) for s in ("clean", "finding", "fail", "unsettled", "not_run", "skipped_by_plan")] == [
        "pass", "fail", "fail", "unsettled", "not_run", "not_run"]


def test_a_sample_the_reward_service_never_saw_did_not_run_any_check():
    clean = sample("a", 0, legs={"san.asan": "clean"})
    died = sample("a", 1, status="unscored", end="engine_error", turn=-1, handed=False, reason="engine_error")
    assert S.summarize([clean, died])["checks"]["san.asan"] == {"pass": 1, "fail": 0, "unsettled": 0, "not_run": 1,
                                                                "not_applicable": 0}
    lost = sample("b", 0, status="quarantined", end="quarantined", turn=1, handed=False, reason="quarantined")
    rows = S.build_report([clean, died, lost], [_prompt("a"), _prompt("b")], {})
    assert rows[2]["checks"] == {"san.asan": {"pass": 0, "fail": 0, "unsettled": 0, "not_run": 1, "not_applicable": 0}}
    assert rows[-1]["checks"]["san.asan"]["not_run"] == 2


def test_the_reward_distribution_is_over_scored_samples_only():
    samples = [sample(k=0, reward=-0.95, base=-0.9), sample(k=1, reward=-0.92, base=-0.9),
               sample(k=2, reward=0.98, base=1.0), sample(k=3, status="unscored"), sample(k=4, reward=-1.0, base=-1.0)]
    r = S.summarize(samples)["reward"]
    assert r["n"] == 4 and r["min"] == -1.0 and r["max"] == 0.98
    assert r["mean"] == pytest.approx((-0.95 - 0.92 + 0.98 - 1.0) / 4) and r["median"] == pytest.approx(-0.935)
    assert r["by_base"] == {"-1": 1, "-0.9": 2, "1": 1}
    assert S.summarize([sample(status="unscored")])["reward"]["n"] == 0


def test_compile_success_and_repair_turns_are_counted_per_chain():
    samples = [sample(k=0, end="compiled", turn=0), sample(k=1, end="compiled", turn=2),
               sample(k=2, end="turns_exhausted", turn=2), sample(k=3, end="no_code_block", turn=0),
               sample(k=4, status="unscored", end="engine_error", turn=-1, handed=False)]
    t = S.summarize(samples)
    assert t["compile"] == {"compiled": 2, "fraction": 0.4, "at_turn": {"0": 1, "2": 1}}
    assert t["repair_turns"] == {"used": {"0": 2, "2": 2}, "mean": 1.0}  # the chain with no reply used none
    assert t["end_reasons"] == {"compiled": 2, "engine_error": 1, "no_code_block": 1, "turns_exhausted": 1}


def test_the_report_is_the_run_then_each_prompt_then_their_totals():
    samples = [sample("a", 0, clean=True, base=1.0, reward=0.98), sample("a", 1),
               sample("b", 0, status="unscored"), sample("b", 1, reward=-0.9, base=-0.9)]
    rows = S.build_report(samples, [_prompt("b"), _prompt("a")], {"run_id": "r"})
    assert [r["kind"] for r in rows] == ["run", "prompt", "prompt", "total"] and rows[0]["run_id"] == "r"
    assert [r["prompt_id"] for r in rows[1:3]] == ["b", "a"] and [r["any_clean"] for r in rows[1:3]] == [False, True]
    total = rows[-1]
    for key in ("samples", "scored", "unscored", "clean", "handed_over"):
        assert total[key] == rows[1][key] + rows[2][key]
    assert (total["prompts"], total["prompts_with_a_clean_sample"], total["scored_fraction"]) == (2, 1, 0.75)
    with pytest.raises(ValueError, match="was not selected"):
        S.build_report([sample("c")], [_prompt("a")], {})


def test_a_score_only_episode_samples_from_v0_and_a_trained_one_from_the_previous_version(tmp_path):
    r, res = scored(spec(tmp_path, score={"prompts": 4, "passes": 3}, engine=NO_ERRORS))
    assert res["status"] == "completed"
    e = read_episodes(r.cfg.run_dir, 3)[0]
    assert (e["iteration"], e["adapter_version"], e["excluded_reason"], e["trained"]) == (
        3, "score-a:v0", "score_only", False)
    for version, reason in (("score-a:v2", "score_only"), ("score-a:v0", "unscored")):
        with pytest.raises(EpisodeError, match="iteration n samples from v\\(n-1\\)"):
            validate_episode(dict(e, adapter_version=version, excluded_reason=reason))
    with pytest.raises(EpisodeError, match="a trained sample"):
        validate_episode(dict(e, trained=True))


# --- score runs over the fakes ---------------------------------------------------------------------------------


def test_a_score_run_scores_each_selected_prompt_k_times_per_pass_and_trains_nothing(tmp_path):
    r, res = scored(spec(tmp_path, engine=NO_ERRORS))
    cfg = r.cfg
    assert res["status"] == "completed" and res["mode"] == "score" and res["policy_version"] == "score-a:v0"
    assert (res["iterations_planned"], res["iterations_completed"]) == (4, 4) and complete_iterations(cfg.run_dir) == [
        1, 2, 3, 4]
    eps = episodes(cfg)
    chosen = S.select_score_prompts(r.pool.records(), cfg.orchestrator.run_seed, 6)
    assert len(eps) == 6 * 4 * 2 and len({e["chain_id"] for e in eps}) == 48
    assert {e["prompt_id"] for e in eps} == {p.prompt_id for p in chosen}
    assert all(not e["trained"] and e["excluded_reason"] == "score_only" and e["advantage"] is None
               and e["adapter_version"] == "score-a:v0" for e in eps)
    by_pass = {}
    for e in eps:  # the two passes draw their own seeds
        by_pass.setdefault((e["prompt_id"], e["k"]), []).append(e["turns"][0]["seed"])
    assert all(len(seeds) == 2 and seeds[0] != seeds[1] for seeds in by_pass.values())
    assert r.rollout.runs == [1, 2, 3, 4]
    assert not (cfg.run_dir / "ckpt").exists() and not (cfg.run_dir / "tb").exists()  # nothing trained or saved
    assert not any((cfg.run_dir / "iterations" / d / "summary.json").exists()
                   for d in os.listdir(cfg.run_dir / "iterations"))
    batch = json.loads((cfg.run_dir / "iterations" / "iter-000002" / "score.json").read_text())
    assert (batch["pass"], batch["batch"], batch["chains"], batch["prompts"]) == (
        1, 2, 8, [p.prompt_id for p in chosen[4:]])

    rows = report(cfg)
    assert [x["kind"] for x in rows] == ["run"] + ["prompt"] * 6 + ["total"]
    total, run = rows[-1], rows[0]
    assert total["samples"] == 48 and total["scored"] == sum(e["status"] == "scored" for e in eps)
    assert all(x["samples"] == 8 for x in rows[1:-1])
    assert total["scored_fraction"] == total["scored"] / 48 == res["score"]["scored_fraction"]
    assert run["sampling"] == {"K": 4, "temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 300,
                               "thinking": "off", "max_repair_turns": 2, "repair_turn_ids": "append",
                               "parse_adapter": "standin.parse", "feedback_template": "standin",
                               "compile_adapter": "fake.compile", "run_seed": 20261102,
                               "seed_rule": run["sampling"]["seed_rule"]}
    assert run["options"]["passes"] == 2 and run["options"]["prompts_selected"] == 6
    assert run["landing"]["verdict"] == "PROVEN" and run["model"]["kind"] == "merged_dir"
    summary = (cfg.run_dir / "score" / "summary.txt").read_text().splitlines()
    assert summary[1].startswith(f"Scored {total['scored']} of 48 samples (")
    plan = json.loads((cfg.run_dir / "run_plan.json").read_text())
    assert any(k.startswith("base/") for k in plan["extra_files"])


def test_the_unsettled_legs_are_rechecked_once_whatever_the_fraction(tmp_path):
    r0, _ = scored(spec(tmp_path, "base", score={"prompts": 4, "passes": 1}, engine=NO_ERRORS))
    ids = [e["chain_id"].replace("score-base", "score-retry") for e in episodes(r0.cfg) if e["status"] == "scored"]
    r, res = scored(spec(tmp_path, "retry", score={"prompts": 4, "passes": 1}, engine=NO_ERRORS,
                         reward={"unsettled": ids[:2], "faulty_attempts": [1]}))
    assert res["status"] == "completed" and sorted(r.reward.resubmits) == sorted(ids[:2])
    batch = json.loads((r.cfg.run_dir / "iterations" / "iter-000001" / "score.json").read_text())
    assert batch["first"]["retryable"] == 2 and batch["retry"]["candidates"] == 2  # 14 of 16 would have met D7
    eps = {e["chain_id"]: e for e in episodes(r.cfg)}
    assert all(eps[c]["status"] == "scored" for c in ids[:2])


def test_a_check_that_never_settles_leaves_its_sample_unscored_and_the_run_completes(tmp_path):
    r0, _ = scored(spec(tmp_path, "base", score={"prompts": 4, "passes": 1}, engine=NO_ERRORS))
    ids = [e["chain_id"].replace("score-base", "score-stall") for e in episodes(r0.cfg) if e["status"] == "scored"]
    r, res = scored(spec(tmp_path, "stall", score={"prompts": 4, "passes": 1}, engine=NO_ERRORS,
                         reward={"stall": ids[:6], "faulty_attempts": [1, 2]}))
    assert res["status"] == "completed", res  # 10 of 16 scored: D7 would stop a training run; a score run reports it
    total = report(r.cfg)[-1]
    assert total["scored"] == len(ids) - 6 and total["unscored_by_reason"] == {"sanitizer_not_settled": 6}
    assert total["checks"]["san.asan"]["unsettled"] == 6 and total["clean"] <= 10
    assert sorted(r.reward.resubmits) == sorted(ids[:6])


def test_every_check_counts_every_sample_when_engines_lose_chains(tmp_path):
    r, res = scored(spec(tmp_path, score={"prompts": 8, "passes": 1}, engine={"error_rate": 0.2}))
    rows = report(r.cfg)
    assert res["status"] == "completed" and rows[-1]["end_reasons"]["engine_error"] > 0
    for row in rows[1:]:
        assert row["checks"].keys() == rows[-1]["checks"].keys() != set()
        assert all(sum(c[o] for o in S.CHECK_OUTCOMES) == row["samples"] for c in row["checks"].values()), row


def test_not_applicable_legs_are_counted_beside_their_states(tmp_path):
    r, res = scored(spec(tmp_path, score={"prompts": 4, "passes": 1}, engine=NO_ERRORS,
                         reward={"not_applicable_legs": ["san.waitcheck"]}))
    checks, eps = report(r.cfg)[-1]["checks"], episodes(r.cfg)
    assert checks["san.waitcheck"]["not_applicable"] == sum("san.waitcheck" in e["legs"] for e in eps) > 0
    assert sum(checks["san.waitcheck"][o] for o in S.CHECK_OUTCOMES) == len(eps)
    assert checks["san.asan"]["not_applicable"] == 0


def test_engines_that_serve_another_model_stop_the_run_before_anything_is_scored(tmp_path):
    r, res = scored(spec(tmp_path, served="another-model"))
    assert res["status"] == "stopped" and res["stop"]["reason"] == "served_model_unconfirmed"
    assert res["stop"]["detail"]["landing"]["verdict"] == "NOT_PROVEN"
    assert r.rollout.runs == [] and complete_iterations(r.cfg.run_dir) == [] and r.reward.cands == {}
    model = json.loads((r.cfg.run_dir / "score" / "model.json").read_text())
    assert model["landing"]["verdict"] == "NOT_PROVEN" and not (r.cfg.run_dir / "score" / "report.jsonl").exists()


def test_a_score_run_is_not_resumed_and_needs_the_self_tests_prompts(tmp_path):
    r = build_fake_run(spec(tmp_path))
    res = r.orchestrator.run(resume=True)
    assert res["stop"]["reason"] == "score_runs_are_not_resumed" and not r.cfg.run_dir.exists()
    r, res = scored(spec(tmp_path, "few", score={"prompts": 3, "passes": 1}))
    assert res["stop"]["reason"] == "too_few_prompts" and r.rollout.runs == []


def _main(*args):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-m", "aorlta.orchestrator.run", *args], capture_output=True, text=True,
                          env=env, timeout=120)


def test_the_entry_point_runs_a_score_config_with_the_fakes_and_never_resumes_it(tmp_path):
    path = tmp_path / "score.yaml"
    path.write_text(json.dumps(score_dict(tmp_path, score={"prompts": 6, "passes": 1})))
    proc = _main("--config", str(path), "--components", "fake")
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert (out["status"], out["iterations_completed"], out["policy_version"]) == ("completed", 2, "s:v0")
    assert out["score"]["samples"] == 24 and 0 < out["score"]["scored"] <= 24
    run_dir = tmp_path / "runs" / "standin" / "s"
    assert (run_dir / "score" / "summary.txt").is_file() and (run_dir / "score" / "report.jsonl").is_file()
    again = _main("--config", str(path), "--components", "fake", "--resume")
    assert again.returncode == 2 and "a score run is not resumed" in again.stderr


def test_the_wiring_reads_no_trainer_setting_in_a_score_run_and_wants_the_self_tests_prompts(tmp_path):
    from aorlta.orchestrator import wiring

    data = score_dict(tmp_path)  # 2 engines and the default 4 colocated trainer ranks: a training run is refused
    data["reward"] = {"profile": "fake", "reward_fn": "standin.v11"}
    wiring.check_supported(config_from_dict(data))
    train = dict(data, orchestrator={"run_seed": 7, "iterations": 1})
    with pytest.raises(wiring.WiringError, match="trainer.ranks is 4"):
        wiring.check_supported(config_from_dict(train))
    few = score_dict(tmp_path, score={"prompts": 3})
    with pytest.raises(wiring.WiringError, match="the first batch of the score run has 3 prompts"):
        wiring.check_supported(config_from_dict(few))


def test_the_run_job_prints_a_score_runs_summary_and_says_it_is_never_resumed():
    text = (REPO / "jobs" / "aorlta-run.sbatch").read_text()
    post = text[text.index("Every committed iteration's summary"):]
    branch = post[post.index('if cfg.orchestrator.mode == "score":'):post.index("for n in complete_iterations")]
    assert "REPORT_DIR, SUMMARY_NAME" in branch and "sys.exit(0)" in branch
    header = text[:text.index("#SBATCH")]
    assert "orchestrator.mode: score" in header and "never resumed" in header


def test_the_stand_in_score_config_scores_16_prompts_4_times_on_the_training_runs_layout():
    from aorlta.config import load_config
    from aorlta.orchestrator import wiring

    cfg = load_config(REPO / "configs" / "score-standin.yaml")
    o, r = cfg.orchestrator, cfg.rollout
    assert (o.mode, o.iterations, o.score.prompts, o.score.passes, r.K, r.P) == ("score", None, 16, 1, 4, 16)
    assert r.engines == (0, 1, 2, 3) and cfg.model.adapter_dir is None and cfg.sync.round_trip_at_start
    assert cfg.reward.remote.adapters == {"standin.aorta_consan": (4, 6), "standin.aorta_asan": (5, 7)}
    assert (cfg.reward.profile, cfg.reward.reward_fn, cfg.reward.gpu_check_s) == (
        "standin-gpu", "standin.v11", {"san.consan": 30.0})
    wiring.check_supported(cfg)


def test_every_container_of_the_run_job_dumps_no_core():
    """A core dump lands in the container's working directory, on shared storage (150-550 MB files)."""
    text = (REPO / "jobs" / "aorlta-run.sbatch").read_text()
    runs = [text[i:text.index(' "$', text.index("--entrypoint", i))] for i in range(len(text))
            if text.startswith("docker run ", i)]
    assert len(runs) == 7 and "--ulimit core=0" in text[text.index("CHK_DOCKER=("):text.index(")", text.index(
        "CHK_DOCKER=("))]
    assert all("--ulimit core=0" in r or "${CHK_DOCKER[@]}" in r for r in runs), runs


def snapshot(cache: Path, names) -> Path:
    """A Hugging Face cache's snapshot directory: each file a link into the cache's blobs/."""
    snap, blobs = cache / "snapshots" / "0123abc", cache / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    for name in names:
        (blobs / f"blob-{name}").write_text(name)
        (snap / name).symlink_to(Path("..") / ".." / "blobs" / f"blob-{name}")
    return snap


def orchestrator_mounts(tmp_path, data: dict) -> list[str]:
    """The `-v` arguments of the orchestrator's container when jobs/aorlta-run.sbatch runs `data` (a score config),
    from a package root that is a git repository, with `docker` a stub on PATH that records its arguments and runs
    nothing."""
    pkg, bin_dir, venv = tmp_path / "pkg", tmp_path / "bin", tmp_path / "venv"
    for d in (pkg / "configs", bin_dir, venv / "bin", tmp_path / "tmpdir"):
        d.mkdir(parents=True)
    (pkg / "src").symlink_to(REPO / "src")
    (pkg / "configs" / "score.yaml").write_text(json.dumps(data))
    git = ["git", "-C", str(pkg), "-c", "user.name=aorlta", "-c", "user.email=aorlta@localhost", "-c",
           "commit.gpgsign=false"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "config"]):
        subprocess.run(git + args, check=True, capture_output=True)
    (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    log = tmp_path / "docker.log"
    (bin_dir / "docker").write_text(f"#!/bin/bash\nprintf '%s\\037' \"$@\" >> '{log}'\nprintf '\\036' >> '{log}'\n"
                                    "if [ \"$1 $2 $3\" = 'image inspect --format' ]; then echo stub; fi\n")
    for f in (venv / "bin" / "python", bin_dir / "docker"):
        f.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AORLTA_", "SLURM"))}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", SLURM_JOB_ID="4242", SLURM_SUBMIT_DIR=str(pkg),
               CONFIG="configs/score.yaml", COMPONENTS="fake", AORLTA_VENV=str(venv),
               AORLTA_TMPDIR=str(tmp_path / "tmpdir"), AORLTA_HF_HOME=str(tmp_path / "hf"))
    proc = subprocess.run(["bash", str(REPO / "jobs" / "aorlta-run.sbatch")], cwd=pkg, env=env, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    calls = [c.split("\x1f")[:-1] for c in log.read_text().split("\x1e")[:-1]]
    run = next(c for c in calls if c[:4] == ["run", "--rm", "--name", "aorlta-run-4242"])
    return [run[i + 1] for i, a in enumerate(run) if a == "-v"]


def visible(path: Path, mounts: list[str]) -> str | None:
    """The mode ("rw" or "ro") of the innermost mount that holds `path` in the container, each mount at its host path;
    None if no mount holds it."""
    held = [(len(dest), mode or "rw") for dest, _, mode in (m.partition(":")[2].partition(":") for m in mounts)
            if path == Path(dest) or Path(dest) in path.parents]
    return max(held)[1] if held else None


@pytest.mark.parametrize("adapter_layout", [None, "directory", "hf_snapshot"])
def test_the_run_job_mounts_the_model_the_adapter_read_only_and_the_pools(tmp_path, adapter_layout):
    base = snapshot(tmp_path / "hub" / "models--org--base", ["config.json", "model.safetensors"])
    names = ["adapter_config.json", "adapter_model.safetensors"]
    data = score_dict(tmp_path)
    data["model"] = {"base_dir": str(base), "source": "standin"}
    if adapter_layout == "hf_snapshot":
        adapter = snapshot(tmp_path / "lora-hub" / "models--org--lora", names)
    elif adapter_layout == "directory":
        adapter = tmp_path / "adapters" / "lora"
        adapter.mkdir(parents=True)
        for name in names:
            (adapter / name).write_text(name)
    if adapter_layout is not None:
        data["model"]["adapter_dir"] = str(adapter)
    (tmp_path / "pool" / "listings").mkdir(parents=True)
    (tmp_path / "pool" / "kernels.json").write_text("{}")
    data["data"] = {"prompt_pools": [{"path": str(tmp_path / "pool" / "listings"), "source": "standin"}]}
    mounts = orchestrator_mounts(tmp_path, data)
    assert visible(base, mounts) == "rw" and visible((base / "config.json").resolve(), mounts) == "rw", mounts
    assert visible(tmp_path / "pool", mounts) == "ro", mounts
    if adapter_layout is not None:
        assert visible(adapter, mounts) == "ro", mounts
        assert all(visible((adapter / n).resolve(), mounts) == "ro" for n in names), mounts  # a snapshot's links


def test_the_summary_leads_with_the_scored_fraction_and_lists_the_checks_that_ran():
    header = {"run_id": "r", "source": "standin", "written_at": "2026-11-06T00:00:00Z",
              "sampling": {"K": 3, "temperature": 0.7, "top_p": 0.95, "top_k": -1, "max_new_tokens": 64,
                           "thinking": "off", "max_repair_turns": 2},
              "model": {"kind": "merged_dir", "dir": "/m", "files_sha256": "a" * 64},
              "landing": {"verdict": "PROVEN", "engines": 4}, "options": {"passes": 1}}
    legs = {"san.asan": "clean", "ref.edge": "not_run"}
    samples = [sample("a", 0, clean=True, base=1.0, reward=0.98, legs=legs),
               sample("a", 1, status="unscored", legs={"san.asan": "unsettled", "ref.edge": "not_run"}),
               sample("a", 2, base=-0.9, reward=-0.92, legs=legs)]
    rows = S.build_report(samples, [_prompt("a")], header)
    text = S.summary_text(rows)
    assert text == S.summary_text([json.loads(x) for x in S.jsonl_bytes(rows).decode().splitlines()])  # as read back
    lines = text.splitlines()
    assert lines[1] == "Scored 2 of 3 samples (66.7%). An unscored sample is never counted clean."
    assert lines[2] == "Clean 1 of 3 samples (33.3%); 1 of 1 prompts have a clean sample."
    assert lines[3].endswith("by base: -0.9: 1, 1: 1")
    checks = next(x for x in lines if x.startswith("Checks that ran"))
    assert checks.endswith("san.asan 2/0/1/0 (0); 1 more never ran") and "ref.edge" not in checks
