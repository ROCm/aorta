"""The evaluator's refusal to pair or resume columns sampled under different settings.

A column records the system prompt and the sampler settings it was drawn under; pairing
two columns that differ in either would attribute a prompt or sampler change to training.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "examples"))

import eval_rocjitsu as ev  # noqa: E402
import rocjitsu_env as rj  # noqa: E402
from test_rocjitsu_env import build_corpus  # noqa: E402


@pytest.fixture(autouse=True)
def _default_reward_and_prompt(monkeypatch):
    monkeypatch.delenv(rj.REWARD_ENV, raising=False)
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)


def column(sha, scenarios=("s1",)):
    config = {"seed": 1, "episodes": 2, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 192,
              "max_episode_steps": 8, "corpus": {"s1": "d"}, "scenarios": list(scenarios)}
    if sha is not None:
        config["system_sha256"] = sha
    rows = [{"scenario_id": sid, "index": i, "total": 1.0, "solved": False, "outcome": "kernel_only"}
            for sid in scenarios for i in range(config["episodes"])]
    return {"config": config, "groups": [{"episodes": rows}], "metrics": {}}

def test_columns_under_different_or_unrecorded_prompts_do_not_pair():
    with pytest.raises(SystemExit, match="system prompts"):
        ev.compare(column("a"), column("b"))
    with pytest.raises(SystemExit, match="system prompts"):
        ev.compare(column(None), column("a"))
    with pytest.raises(SystemExit, match="system prompts"):
        ev.compare(column(None), column(None))
    assert ev.compare(column("a"), column("a"))["episodes_paired"] == 2

def test_columns_sampled_with_different_recorded_top_k_do_not_pair():
    def with_k(k):
        c = column("a")
        if k is not None:
            c["config"]["top_k"] = k
        return c

    with pytest.raises(SystemExit, match="top_k 20 and 50"):
        ev.compare(with_k(20), with_k(50))
    both = ev.compare(with_k(20), with_k(20))["top_k"]
    assert both == {"before": 20, "after": 20, "checked": True}
    legacy = ev.compare(with_k(None), with_k(20))["top_k"]
    assert legacy == {"before": None, "after": 20, "checked": False}, "an unrecorded side is reported, not assumed"

def test_a_column_records_the_top_k_it_was_sampled_with():
    args = ev.build_parser().parse_args(["--corpus-root", "x", "--out", "o", "--top-k", "7"])
    config = ev.column_config(args, "m", [], "s" * 64, "cublas", {})
    assert config["top_k"] == 7 and config["top_p"] == 0.95 and config["temperature"] == 0.7
    gen = ev.gen_args(args)
    assert (gen.top_k, gen.top_p, gen.temperature, gen.max_new_tokens) == (7, 0.95, 0.7, config["max_new_tokens"]), \
        "the sampler gets exactly what the column records"

def test_a_column_will_not_resume_under_another_top_k():
    assert ev.build_parser().parse_args(["--corpus-root", "x", "--out", "o"]).top_k == 20
    recorded = {"source": "m", "seed": 1, "system_sha256": "s" * 64, "top_k": 20}
    ev.check_resumable(Path("c.json"), recorded, "m", 1, "s" * 64, 20)
    with pytest.raises(SystemExit, match="top_k 20; refusing to resume it with top_k 50"):
        ev.check_resumable(Path("c.json"), recorded, "m", 1, "s" * 64, 50)
    legacy = dict(recorded)
    del legacy["top_k"]
    with pytest.raises(SystemExit, match="top_k None"):
        ev.check_resumable(Path("c.json"), legacy, "m", 1, "s" * 64, 20)
    with pytest.raises(SystemExit, match="seed"):
        ev.check_resumable(Path("c.json"), recorded, "m", 2, "s" * 64, 20)
    with pytest.raises(SystemExit, match="system prompt"):
        ev.check_resumable(Path("c.json"), recorded, "m", 1, "t" * 64, 20)


@pytest.mark.parametrize("setting", ["seed", "episodes", "temperature", "top_p", "max_new_tokens",
                                     "max_episode_steps", "corpus", "seeding", "param_dtype", "blas_backend",
                                     "runtime", "generation"])
def test_columns_sampled_under_any_other_setting_do_not_pair(setting):
    before, after = column("s" * 64), column("s" * 64)
    after["config"][setting] = {"s1": "other"} if setting == "corpus" else 3
    with pytest.raises(SystemExit, match=f"differ in {setting}"):
        ev.compare(before, after)


def test_columns_tokenized_from_different_trees_do_not_pair():
    before, after = column("s" * 64), column("s" * 64)
    before["config"]["checkpoint"] = {"weights": {"sha256": "w1"}, "tokenizer": {"sha256": "t"}}
    after["config"]["checkpoint"] = {"weights": {"sha256": "w2"}, "tokenizer": {"sha256": "t"}}
    assert ev.compare(before, after)["episodes_paired"] == 2, "different weights are what a pair measures"
    after["config"]["checkpoint"]["tokenizer"] = {"sha256": "u"}
    with pytest.raises(SystemExit, match="tokenized from different trees"):
        ev.compare(before, after)


def drop_row(c, index):
    c["groups"][0]["episodes"] = [r for r in c["groups"][0]["episodes"] if r["index"] != index]


def repeat_row(c, index):
    c["groups"][0]["episodes"].append(dict(c["groups"][0]["episodes"][index]))


def add_row(c, index):
    c["groups"][0]["episodes"].append(dict(c["groups"][0]["episodes"][0], index=index))


@pytest.mark.parametrize("damage, fragment", [
    (lambda c: drop_row(c, 1), "1 missing"),
    (lambda c: repeat_row(c, 0), "1 repeated"),
    (lambda c: add_row(c, 5), "1 not configured"),
    (lambda c: c.update(incomplete=True), "incomplete"),
    (lambda c: c.pop("metrics"), "incomplete"),
])
def test_a_column_missing_or_repeating_episodes_does_not_pair(damage, fragment):
    for side in ("before", "after"):
        before, after = column("s" * 64), column("s" * 64)
        damage(before if side == "before" else after)
        with pytest.raises(SystemExit, match=fragment):
            ev.compare(before, after)


def test_a_column_is_checked_against_the_selection_its_baselines_are_scored_on(tmp_path):
    corpus = build_corpus(tmp_path)
    picked = ev.pick_scenarios(rj.load_corpus(corpus, split="train"), 0, -1)
    system = rj.system_for(picked)
    good = {"before": {"config": {"scenarios": [s.scenario_id for s in picked],
                                  "corpus": {s.scenario_id: s.digest for s in picked},
                                  "system_sha256": hashlib.sha256(system.encode()).hexdigest()}}}
    ev.check_selection(good, picked, system)
    with pytest.raises(SystemExit, match="other scenarios"):
        ev.check_selection(good, picked[:1], system)
    stale = {"before": {"config": dict(good["before"]["config"],
                                       corpus={s.scenario_id: "other" for s in picked})}}
    with pytest.raises(SystemExit, match="other corpus"):
        ev.check_selection(stale, picked, system)
    with pytest.raises(SystemExit, match="other system_sha256"):
        ev.check_selection(good, picked, system + " ")


def test_a_resumed_column_with_scenarios_left_drops_its_old_metrics(tmp_path):
    corpus = build_corpus(tmp_path)
    picked = ev.pick_scenarios(rj.load_corpus(corpus, split="train"), 0, -1)
    done = {"groups": [{"scenario_id": picked[0].scenario_id}], "metrics": {"all": {}}}
    assert ev.remaining(done, picked) == picked[1:]
    assert "metrics" not in done, "the old metrics would mark an unfinished column as finished"
    whole = {"groups": [{"scenario_id": s.scenario_id} for s in picked], "metrics": {"all": {}}}
    assert ev.remaining(whole, picked) == [] and "metrics" in whole


def test_the_inherited_generation_settings_leave_out_what_generate_sets():
    settings = {"temperature": 0.6, "top_k": 20, "do_sample": True, "repetition_penalty": 1.1,
                "eos_token_id": [1, 2], "transformers_version": "4.57.0", "_from_model_config": False}
    assert ev.inherited_generation(settings) == {"eos_token_id": [1, 2], "repetition_penalty": 1.1}


def test_a_turn_budget_the_prompt_does_not_state_is_refused(tmp_path, capsys):
    corpus = build_corpus(tmp_path)
    assert ev.main(["--corpus-root", str(corpus), "--split", "train", "--out", str(tmp_path / "o"),
                    "--column", "none", "--max-episode-steps", "4"]) == ev.EXIT_REFUSED
    assert f"must be {rj.MAX_TURNS}" in capsys.readouterr().err


def test_columns_over_different_scenarios_do_not_pair():
    with pytest.raises(SystemExit, match="different scenarios"):
        ev.compare(column("s" * 64), column("s" * 64, scenarios=("s1", "s2")))


@pytest.mark.parametrize("text, expected", [(None, None), ("0/1", (0, 1)), ("3/4", (3, 4))])
def test_a_shard_is_k_of_n(text, expected):
    assert ev.parse_shard(text) == expected


@pytest.mark.parametrize("text", ["4/4", "-1/4", "0/0", "1", "a/b", "1/2/3"])
def test_a_shard_outside_its_range_is_refused(text):
    with pytest.raises(ValueError, match="--shard"):
        ev.parse_shard(text)


def test_a_column_will_not_resume_under_any_other_sampling_or_selection():
    args = ev.build_parser().parse_args(["--corpus-root", "x", "--out", "o"])
    config = ev.column_config(args, "m", [], "s" * 64, None, {})
    ev.check_resumable(Path("c.json"), config, "m", args.seed, "s" * 64, args.top_k, proposed=config)
    for key, value in [("episodes", 4), ("temperature", 1.0), ("max_episode_steps", 4), ("split", "val"),
                       ("scenarios", ["s9"]), ("corpus", {"s9": "d"}), ("param_dtype", "bfloat16"),
                       ("blas_backend", "other"), ("runtime", {"transformers": "x"}),
                       ("checkpoint", {"weights": {"sha256": "new"}}), ("generation", {"repetition_penalty": 1.1})]:
        with pytest.raises(SystemExit, match=f"other {key}; refusing"):
            ev.check_resumable(Path("c.json"), config, "m", args.seed, "s" * 64, args.top_k,
                               proposed=dict(config, **{key: value}))


def test_an_answer_with_no_kernel_is_counted_not_a_crash():
    rows = [{"scenario_id": "k_a+0x8.waitcnt_weaken", "sequence": "answer",
             "answer": {"class": "waitcnt", "kernel": None, "offset": None}}]
    assert ev.entrenchment(rows)["answer_top"] == "?@None"


def test_an_empty_shard_gets_the_splits_system_prompt(tmp_path, monkeypatch):
    corpus = build_corpus(tmp_path)
    seen = []
    monkeypatch.setattr(ev, "run_column", lambda checkpoint, args, scenarios, path, system=None:
                        seen.append((scenarios, system)))
    assert ev.main(["--corpus-root", str(corpus), "--split", "train", "--out", str(tmp_path / "o"),
                    "--column", "before", "--shard", "7/8"]) == 0
    assert seen == [([], rj.system_for(rj.load_corpus(corpus, split="train")))]


def test_a_split_that_selects_no_scenario_is_refused(tmp_path, capsys):
    corpus = build_corpus(tmp_path)
    capsys.readouterr()
    spec = json.loads((corpus / "split.json").read_text())
    (corpus / "split.json").write_text(json.dumps(dict(spec, train=[])))
    assert ev.main(["--corpus-root", str(corpus), "--split", "train", "--out", str(tmp_path / "o"),
                    "--column", "none"]) == ev.EXIT_REFUSED
    assert "selects no scenario" in capsys.readouterr().err
