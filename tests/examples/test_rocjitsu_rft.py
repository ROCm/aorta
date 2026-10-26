"""Reward-filtered self-training inputs: sharded rollout columns and harvesting them as demonstrations.

A shard spec that silently drops or repeats scenarios would bias the harvest; a
column generated under one system prompt and trained under another would teach
replies to a prompt the model is never shown; and "best first" has to mean the
highest-scoring solved episode, not merely a solved one.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "examples"))

import eval_rocjitsu as ev  # noqa: E402
import rocjitsu_demos as demos  # noqa: E402
import rocjitsu_env as rj  # noqa: E402
from test_rocjitsu_env_v2 import _toy_corpus  # noqa: E402

CORPUS_V2 = Path(os.environ.get("AORTA_RJ_TEST_CORPUS_V2") or Path(__file__).with_name("no-corpus-v2"))
ARCHIVE = CORPUS_V2 / "archive"
needs_archive = pytest.mark.skipif(not (ARCHIVE / "scenarios.jsonl").is_file(), reason="corpus-v2 archive absent")


def test_a_shard_spec_is_k_of_n_and_nothing_else():
    assert ev.parse_shard(None) is None
    assert ev.parse_shard("0/1") == (0, 1)
    assert ev.parse_shard("7/8") == (7, 8)
    for bad in ("8/8", "-1/4", "1/0", "a/b", "3", "1/2/3"):
        with pytest.raises(ValueError):
            ev.parse_shard(bad)


def test_main_refuses_a_bad_shard_before_touching_the_corpus(tmp_path, capsys):
    rc = ev.main(["--corpus-root", str(tmp_path / "absent"), "--out", str(tmp_path / "o"),
                  "--column", "none", "--shard", "4/4"])
    assert rc == ev.EXIT_REFUSED
    assert "--shard 4/4" in capsys.readouterr().err


@needs_archive
def test_main_runs_only_its_shard(tmp_path, capsys):
    picked = ev.pick_scenarios(rj.load_corpus(ARCHIVE, split="val"), 0, -1)
    assert ev.main(["--corpus-root", str(ARCHIVE), "--split", "val", "--out", str(tmp_path),
                    "--column", "none", "--shard", "1/3"]) == 0
    assert f"split val: {len(picked[1::3])} scenario(s)" in capsys.readouterr().out
    assert len(picked[1::3]) < len(picked)


def test_shards_cover_the_list_exactly_once():
    items = list(range(37))
    for n in range(1, 10):
        got = sorted(x for k in range(n) for x in items[k::n])
        assert got == items


def _scripted(scenario, replies, index, budget=rj.MAX_TURNS):
    def advancing(prompts):
        out = [replies[min(advancing.turn, len(replies) - 1)]] * len(prompts)
        advancing.turn += 1
        return out

    advancing.turn = 0

    def advantages(values):
        return [0.0] * len(values), sum(values) / len(values), 0.0

    _g, _s, wire = rj.rollout_scenario(scenario, 1, rj.make_policy(budget), advancing, advantage_fn=advantages)
    return [dict(w, episode=index) for w in wire]


def toy_corpus(root: Path) -> Path:
    """The v2 toy corpus with a clean control added, and a train split holding it and one fault."""
    _toy_corpus(root)
    clean = {"sid": "CLEAN@toy#0", "workload": "toy", "kid": None, "op": "none", "cls": "none", "offsets": [],
             "sim": {"rc": 0, "ops": {}, "races": [], "memwait": []}, "simq": {}, "native": [],
             "waitcheck_view": []}
    with open(root / "scenarios.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean) + "\n")
    (root / "splits" / "train.txt").write_text("CLEAN@toy#0\nran@toy\n")
    return root


@pytest.fixture(scope="module", params=["toy", "recorded"])
def corpus_root(request, tmp_path_factory):
    """Each harvest test runs on the toy corpus, and on the recorded archive where one is named."""
    if request.param == "toy":
        return toy_corpus(tmp_path_factory.mktemp("toy"))
    if not (ARCHIVE / "scenarios.jsonl").is_file():
        pytest.skip("corpus-v2 archive absent (set AORTA_RJ_TEST_CORPUS_V2)")
    return ARCHIVE


@pytest.fixture(scope="module")
def clean_scenario(corpus_root):
    scenarios = rj.load_corpus(corpus_root, split="train", reward="v2")
    return random.Random(11).choice([s for s in scenarios if s.clean])


def _column(tmp_path, scenario, rows, sha, budget=rj.MAX_TURNS, name="shard-0"):
    path = tmp_path / name / "before.json"
    path.parent.mkdir(parents=True)
    config = {"system_sha256": sha}
    if budget is not None:
        config["max_episode_steps"] = budget
    path.write_text(json.dumps({"config": config, "groups": [], "wire": rows}))
    return path


def _wire(run, rows, sha, budget=rj.MAX_TURNS):
    run.mkdir(parents=True)
    (run / "wire.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = {"max_episode_steps": budget}
    if sha is not None:
        log["system_prompt_sha256"] = sha
    (run / "train-log.json").write_text(json.dumps(log))
    return run / "wire.jsonl"


def _harvest(corpus_root, out, *flags):
    return demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--out", str(out),
                       "--max-wire-per-scenario", "5", *map(str, flags)])


def _sha(corpus_root, split="train"):
    scenarios = rj.load_corpus(corpus_root, split=split)
    return hashlib.sha256(rj.system_for(scenarios).encode()).hexdigest()


TOOLS = [json.dumps({"tool": "list_kernels"}), json.dumps({"tool": "waitcheck"})]
SIM = [json.dumps({"tool": "run_sim", "kernel": "all"})]
NONE = [json.dumps({"answer": {"class": "none"}})]


def test_a_column_is_harvested_only_under_its_own_system_prompt(tmp_path, corpus_root, clean_scenario, monkeypatch):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    rows = _scripted(clean_scenario, TOOLS + NONE, 0)
    good = _column(tmp_path / "a", clean_scenario, rows, _sha(corpus_root))
    out = tmp_path / "demos.jsonl"
    assert demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--column", str(good),
                       "--out", str(out)]) == 0
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert {k["scenario_id"] for k in kept} == {clean_scenario.scenario_id}
    assert [k["completion"] for k in kept] == TOOLS + NONE
    assert kept[0]["source"] == f"column:{good.resolve()}"
    stats = json.loads(out.with_suffix(".stats.json").read_text())
    assert stats["wire"] == {"episodes": 1, "solved": 1} and stats["wire_scenarios_kept"] == 1

    other = _column(tmp_path / "b", clean_scenario, rows, "0" * 64)
    with pytest.raises(SystemExit, match="generated under system prompt"):
        demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--column", str(other),
                    "--out", str(tmp_path / "x.jsonl")])


def _sha_under(profile):
    return hashlib.sha256(rj.system_prompt_v2("v2", profile=profile).encode()).hexdigest()


def test_a_column_from_a_teacher_prompt_is_harvested_only_when_declared(tmp_path, corpus_root, clean_scenario, monkeypatch):
    monkeypatch.setenv(rj.PROMPT_ENV, "procedure")
    monkeypatch.setenv(rj.REWARD_ENV, "v2")
    rows = _scripted(clean_scenario, TOOLS + NONE, 0)
    taught = _column(tmp_path / "a", clean_scenario, rows, _sha_under("procedure_quirks"))
    out = tmp_path / "d.jsonl"
    with pytest.raises(SystemExit, match="generated under system prompt"):
        demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--column", str(taught), "--out", str(out)])
    assert demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--column", str(taught), "--out", str(out),
                       "--generated-under", "procedure_quirks"]) == 0
    stats = json.loads(out.with_suffix(".stats.json").read_text())
    assert stats["system_sha256"] == _sha_under("procedure"), "the demos are trained under the deployment prompt"
    assert stats["generated_under"] == {"profile": "procedure_quirks", "system_sha256": _sha_under("procedure_quirks")}
    plain = _column(tmp_path / "b", clean_scenario, rows, _sha_under("plain"))
    with pytest.raises(SystemExit, match="generated under system prompt"):
        demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--column", str(plain),
                    "--out", str(tmp_path / "e.jsonl"), "--generated-under", "procedure_quirks"])


def test_an_unsolved_episode_is_never_harvested(tmp_path, corpus_root, clean_scenario, monkeypatch):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    stalls = _scripted(clean_scenario, TOOLS, 0)
    solves = _scripted(clean_scenario, TOOLS + NONE, 1)
    path = _column(tmp_path, clean_scenario, stalls + solves, _sha(corpus_root))
    out = tmp_path / "d.jsonl"
    assert demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--column", str(path),
                       "--max-wire-per-scenario", "5", "--out", str(out)]) == 0
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert {k["episode"] for k in kept} == {1}
    assert json.loads(out.with_suffix(".stats.json").read_text())["wire"] == {"episodes": 2, "solved": 1}


def test_best_first_keeps_the_cheaper_of_two_solved_episodes(tmp_path, corpus_root, clean_scenario, monkeypatch):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    dear = _scripted(clean_scenario, TOOLS + SIM + NONE, 0)
    cheap = _scripted(clean_scenario, TOOLS + NONE, 1)
    path = _column(tmp_path, clean_scenario, dear + cheap, _sha(corpus_root))
    picked = {}
    for flag in ([], ["--best-first"]):
        out = tmp_path / f"d{len(flag)}.jsonl"
        assert demos.main(["--corpus-root", str(corpus_root), "--split", "train", "--column", str(path),
                           "--max-wire-per-scenario", "1", "--out", str(out), *flag]) == 0
        picked[bool(flag)] = [json.loads(line)["completion"] for line in out.read_text().splitlines()]
    assert picked[False] == TOOLS + SIM + NONE
    assert picked[True] == TOOLS + NONE


@pytest.mark.parametrize("flags", [["--clip", "-1"], ["--clip", "nan"], ["--lr", "inf"], ["--lr", "0"],
                                   ["--epochs", "0"]])
def test_sft_refuses_settings_that_cannot_train(tmp_path, flags):
    pytest.importorskip("torch")
    import sft_rocjitsu

    rc = sft_rocjitsu.main(["--demos", str(tmp_path / "d.jsonl"), "--corpus-root", str(tmp_path / "absent"),
                            "--out", str(tmp_path / "o"), *flags])
    assert rc == sft_rocjitsu.tr.EXIT_REFUSED and not (tmp_path / "o").exists()


def test_sft_refuses_an_empty_harvest_before_loading_a_model(tmp_path, corpus_root, capsys):
    pytest.importorskip("torch")
    import sft_rocjitsu

    path = tmp_path / "d.jsonl"
    path.write_text("")
    path.with_suffix(".stats.json").write_text(json.dumps({"system_sha256": _sha(corpus_root)}))
    rc = sft_rocjitsu.main(["--demos", str(path), "--corpus-root", str(corpus_root), "--out", str(tmp_path / "o")])
    assert rc == sft_rocjitsu.tr.EXIT_REFUSED and "holds no demonstration" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_two_cycles_of_the_documented_layout_mix(tmp_path, corpus_root, clean_scenario, monkeypatch):
    """Every cycle harvests <cycle>/gen/before.json; the second cycle mixes with the first."""
    import mix_demos

    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    rows = _scripted(clean_scenario, TOOLS + NONE, 0)
    outs = []
    for cycle in ("cycle1", "cycle2"):
        col = _column(tmp_path / cycle, clean_scenario, rows, _sha(corpus_root), name="gen")
        outs.append(tmp_path / cycle / "new.jsonl")
        assert _harvest(corpus_root, outs[-1], "--column", col) == 0
    assert mix_demos.main(["--out", str(tmp_path / "mix.jsonl"), str(outs[1]), str(outs[0])]) == 0
    again = tmp_path / "again.jsonl"
    assert _harvest(corpus_root, again, "--column", tmp_path / "cycle1" / "gen" / "before.json") == 0
    assert mix_demos.main(["--out", str(tmp_path / "dup.jsonl"), str(outs[0]), str(again)]) == 3, \
        "the same input harvested twice is still one episode"


def test_two_iterations_of_one_wire_episode_stay_two_episodes(tmp_path, corpus_root, clean_scenario, monkeypatch):
    import mix_demos

    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    rows = ([dict(r, iteration=1) for r in _scripted(clean_scenario, TOOLS + NONE, 0)]
            + [dict(r, iteration=2) for r in _scripted(clean_scenario, TOOLS + SIM + NONE, 0)])
    wire = _wire(tmp_path / "run", rows, _sha(corpus_root))
    out = tmp_path / "d.jsonl"
    assert _harvest(corpus_root, out, "--wire", wire) == 0
    kept = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted({k["iteration"] for k in kept}) == [1, 2]
    assert {k["source"] for k in kept} == {f"wire:{wire.resolve()}"}
    assert mix_demos.main(["--out", str(tmp_path / "m.jsonl"), str(out)]) == 0
    mixed = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert [r["completion"] for r in mixed] == TOOLS + NONE + TOOLS + SIM + NONE


@pytest.mark.parametrize("sha, log, fragment", [
    ("other", True, "generated under system prompt"),
    (None, True, "generated under system prompt"),
    ("right", False, "no train-log.json"),
])
def test_a_wire_log_is_harvested_only_under_its_own_system_prompt(tmp_path, corpus_root, clean_scenario,
                                                                  monkeypatch, sha, log, fragment):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    rows = [dict(r, iteration=1) for r in _scripted(clean_scenario, TOOLS + NONE, 0)]
    wire = _wire(tmp_path / "run", rows, _sha(corpus_root) if sha == "right" else ("0" * 64 if sha else None))
    if not log:
        (tmp_path / "run" / "train-log.json").unlink()
    with pytest.raises(SystemExit, match=fragment):
        _harvest(corpus_root, tmp_path / "d.jsonl", "--wire", wire)


def test_an_input_is_replayed_under_the_turn_budget_it_recorded(tmp_path, corpus_root, clean_scenario, monkeypatch):
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)
    rows = _scripted(clean_scenario, TOOLS + NONE, 0, budget=4)
    col = _column(tmp_path, clean_scenario, rows, _sha(corpus_root), budget=4)
    out = tmp_path / "d.jsonl"
    assert _harvest(corpus_root, out, "--column", col) == 0
    prompts = [json.loads(line)["prompt"] for line in out.read_text().splitlines()]
    assert prompts and all("of 4." in p and "of 8." not in p for p in prompts)
    wire = _wire(tmp_path / "run", [dict(r, iteration=1) for r in rows], _sha(corpus_root), budget=4)
    assert _harvest(corpus_root, tmp_path / "w.jsonl", "--wire", wire) == 0
    assert [json.loads(line)["prompt"] for line in (tmp_path / "w.jsonl").read_text().splitlines()] == prompts
    unrecorded = _column(tmp_path / "u", clean_scenario, rows, _sha(corpus_root), budget=None)
    with pytest.raises(SystemExit, match="no usable max_episode_steps"):
        _harvest(corpus_root, tmp_path / "u.jsonl", "--column", unrecorded)
