"""Merging a sharded eval column must give back exactly the column an unsharded run writes.

Checked by cutting a column into shards the way ``--shard K/N`` would and merging
them: anything other than byte-for-byte equality means the merged column is not
the unsharded one, and every number computed from it would be about something
else. Each test runs on a synthetic column, built the way ``run_column`` builds
one but with scripted replies for the model, and again on a real recorded column
where ``$AORTA_RJ_TEST_COLUMN`` and ``$AORTA_RJ_TEST_CORPUS_V2`` name one.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "examples"))

import eval_rocjitsu as ev  # noqa: E402
import merge_columns as mc  # noqa: E402
import rocjitsu_env as rj  # noqa: E402
from test_rocjitsu_env import ANSWER, build_corpus  # noqa: E402

COLUMN = Path(os.environ.get("AORTA_RJ_TEST_COLUMN") or Path(__file__).with_name("no-column.json"))
CORPUS_V2 = Path(os.environ.get("AORTA_RJ_TEST_CORPUS_V2") or Path(__file__).with_name("no-corpus-v2"))
ARCHIVE = CORPUS_V2 / "archive"


@pytest.fixture(autouse=True)
def _default_reward_and_prompt(monkeypatch):
    monkeypatch.delenv(rj.REWARD_ENV, raising=False)
    monkeypatch.delenv(rj.PROMPT_ENV, raising=False)


def synthetic_column(root: Path) -> tuple[dict, Path, str]:
    """Four scenarios, three episodes each; odd episodes call a tool before answering."""
    corpus = build_corpus(root, clean_frac=0.5)
    scenarios = ev.pick_scenarios(rj.load_corpus(corpus, split="train"), 0, -1)
    doc: dict = {"groups": [], "wire": []}
    for s in scenarios:
        def reply(prompts, keys, s=s):
            done = {"answer": {"class": "none"}} if s.clean else ANSWER
            return [json.dumps({"tool": "waitcheck"} if step == 1 and episode % 2 else done)
                    for episode, step in keys]
        group, _, wire = rj.rollout_scenario(s, 3, rj.make_policy(rj.MAX_TURNS), reply, keyed=True,
                                             advantage_fn=lambda r: ([0.0] * len(r), sum(r) / len(r), 0.0))
        doc["groups"].append(group)
        doc["wire"].extend(wire)
    doc["config"] = {"source": "scripted", "seed": 0, "split": "train",
                     "scenarios": [s.scenario_id for s in scenarios],
                     "corpus": {s.scenario_id: s.digest for s in scenarios}}
    doc["metrics"] = ev.summarize(ev.episode_rows(doc))
    return json.loads(json.dumps(doc)), corpus, "train"


@pytest.fixture(params=["synthetic", "recorded"])
def case(request, tmp_path_factory):
    """A column, the corpus it was computed on, and its split."""
    if request.param == "synthetic":
        return synthetic_column(tmp_path_factory.mktemp("column"))
    if not COLUMN.is_file() or not (ARCHIVE / "scenarios.jsonl").is_file():
        pytest.skip("recorded column or corpus absent (set AORTA_RJ_TEST_COLUMN and AORTA_RJ_TEST_CORPUS_V2)")
    return json.loads(COLUMN.read_text()), ARCHIVE, "val"


def shard(column, k, n):
    picked = column["config"]["scenarios"][k::n]
    keep = set(picked)
    doc = {"groups": [g for g in column["groups"] if g["scenario_id"] in keep],
           "wire": [w for w in column["wire"] if w["scenario_id"] in keep],
           "config": dict(column["config"], scenarios=picked,
                          corpus={s: column["config"]["corpus"][s] for s in picked})}
    doc["metrics"] = ev.summarize(ev.episode_rows(doc))
    return doc


def same(a, b):
    """Byte-for-byte equality of two columns, as a bool, so a failure does not ask pytest to diff megabytes."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_synthetic_column_is_a_column_worth_merging(tmp_path):
    column, _, _ = synthetic_column(tmp_path)
    m = column["metrics"]
    assert len(column["groups"]) == 4 and m["all"]["n"] == 12
    assert m["clean"]["n"] == 6 and m["faulty"]["n"] == 6
    assert len({w["episode_steps"] for w in column["wire"]}) == 2, "both one- and two-turn episodes"


@pytest.mark.parametrize("n", [1, 3, 8])
def test_merging_the_shards_gives_back_the_column(case, n):
    column, _, _ = case
    shards = [shard(column, k, n) for k in range(n)]
    merged = mc.merge(shards[::-1], column["config"]["scenarios"])
    assert same(merged, column), "the merged column is not the recorded one"


def test_the_merge_refuses_what_is_not_one_column(case):
    column, _, _ = case
    picked = column["config"]["scenarios"]
    shards = [shard(column, k, 3) for k in range(3)]
    with pytest.raises(ValueError, match="holds|hold"):
        mc.merge(shards[:2], picked)
    with pytest.raises(ValueError, match="more than one shard"):
        mc.merge(shards + [shards[0]], picked)
    other = copy.deepcopy(shards)
    other[1]["config"]["seed"] += 1
    with pytest.raises(ValueError, match=r"differs from shard 0 in \['seed'\]"):
        mc.merge(other, picked)
    unfinished = copy.deepcopy(shards)
    del unfinished[2]["metrics"]
    with pytest.raises(ValueError, match="unfinished"):
        mc.merge(unfinished, picked)
    marked = copy.deepcopy(shards)
    marked[1]["incomplete"] = True
    with pytest.raises(ValueError, match="unfinished"):
        mc.merge(marked, picked)


def test_the_cli_checks_the_split_and_writes_the_column(case, tmp_path):
    column, corpus, split = case
    paths = []
    for k in range(4):
        p = tmp_path / f"shard-{k}" / "before.json"
        p.parent.mkdir()
        p.write_text(json.dumps(shard(column, k, 4)))
        paths.append(p)
    out = tmp_path / "merged" / "before.json"
    assert mc.main(["--corpus-root", str(corpus), "--split", split, "--out", str(out), *map(str, paths)]) == 0
    assert same(json.loads(out.read_text()), column), "the written column is not the recorded one"
    assert mc.main(["--corpus-root", str(corpus), "--split", "test", "--out", str(tmp_path / "x.json"),
                    *map(str, paths)]) == ev.EXIT_REFUSED
    assert mc.main(["--corpus-root", str(corpus), "--split", split, "--out", str(out),
                    *map(str, paths)]) == ev.EXIT_REFUSED
    relabelled = []
    for p in paths:
        doc = json.loads(p.read_text())
        doc["config"]["split"] = "test"
        q = p.with_name("relabelled.json")
        q.write_text(json.dumps(doc))
        relabelled.append(q)
    assert mc.main(["--corpus-root", str(corpus), "--split", split, "--out", str(tmp_path / "y.json"),
                    *map(str, relabelled)]) == ev.EXIT_REFUSED


def test_the_cli_refuses_shards_from_another_revision_of_the_corpus(case, tmp_path):
    column, corpus, split = case
    paths = []
    for k in range(2):
        doc = shard(column, k, 2)
        if k == 1:
            sid = doc["config"]["scenarios"][0]
            doc["config"]["corpus"][sid] = "0" * 64
        p = tmp_path / f"shard-{k}.json"
        p.write_text(json.dumps(doc))
        paths.append(p)
    assert mc.main(["--corpus-root", str(corpus), "--split", split, "--out", str(tmp_path / "m.json"),
                    *map(str, paths)]) == ev.EXIT_REFUSED
    assert not (tmp_path / "m.json").exists()
