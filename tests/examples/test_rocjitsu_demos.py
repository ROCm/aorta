"""Warm-start demonstrations: solved, built only from what the agent was shown, and faithful replays.

A demonstration that answers from information the agent never saw would teach
guessing; a replay that does not reproduce the run's reward would train on an
episode that never happened.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import rocjitsu_demos as demos  # noqa: E402
import rocjitsu_env as rj  # noqa: E402

CORPUS_V2 = Path(os.environ.get("AORTA_RJ_TEST_CORPUS_V2") or Path(__file__).with_name("no-corpus-v2"))
ARCHIVE, REFERENCE = CORPUS_V2 / "archive", CORPUS_V2 / "scripts" / "baselines2.py"
needs_archive = pytest.mark.skipif(not (ARCHIVE / "scenarios.jsonl").is_file(), reason="corpus-v2 archive absent")


@pytest.fixture(scope="module")
def train_sample():
    scenarios = rj.load_corpus(ARCHIVE, split="train", reward="v2")
    return random.Random(5).sample(scenarios, 120)


@needs_archive
def test_every_recipe_answer_was_shown_to_the_agent(train_sample):
    checked = 0
    for scenario in train_sample:
        episode = demos.play(scenario, demos.recipe)
        answer = episode.answer or {}
        if answer.get("class") in (None, "none"):
            continue
        shown = "\n".join(s.body for s in episode.steps if s.header.startswith('{"tool": "disasm"'))
        assert rj.hexoff(answer["offset"]) in shown, (scenario.scenario_id, answer)
        checked += 1
    assert checked > 50


@needs_archive
def test_the_recipe_names_runtime_kernels_only_from_list_kernels(train_sample):
    for scenario in train_sample[:30]:
        episode = demos.play(scenario, demos.recipe)
        assert episode.steps[0].action.tool == "list_kernels"


def test_the_visible_listing_stops_where_the_reply_stops():
    text = "\n".join(f"{i * 4:#06x}: s_nop 0" for i in range(rj.DISASM_MAX_LINES + 40))
    assert len(demos.visible_listing(text)) == rj.DISASM_MAX_LINES


@pytest.mark.skipif(not REFERENCE.is_file(), reason="baselines2.py absent")
def test_the_window_pick_is_baselines2s():
    import importlib.util

    spec = importlib.util.spec_from_file_location("baselines2", REFERENCE)
    sys.path.insert(0, str(REFERENCE.parent))
    b2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b2)
    rng = random.Random(0)
    texts = ("s_waitcnt vmcnt(0)", "s_barrier", "v_add_f32 v1, v2, v3", "ds_read_b32 v4, v5", "s_nop 0")
    for _ in range(300):
        lines = sorted({(rng.randrange(0, 0x400, 4), rng.choice(texts)) for _ in range(rng.randrange(1, 30))})
        lo = rng.randrange(0, 0x400, 4)
        hi = lo + rng.randrange(0, 0x100, 4)
        assert demos.pick_in_window(lines, lo, hi) == b2.pick_in_window(lines, lo, hi)


@needs_archive
def test_a_replayed_wire_reproduces_the_recorded_rewards_and_refuses_a_tampered_one(train_sample):
    scenario = train_sample[0]
    replies = [json.dumps(a) for a in ({"tool": "waitcheck"}, {"tool": "run_sim", "kernel": "all"},
                                       {"answer": {"class": "none"}})]

    def generate(prompts):
        return [replies[min(generate.turn, len(replies) - 1)]] * len(prompts)

    generate.turn = 0

    def advancing(prompts):
        out = generate(prompts)
        generate.turn += 1
        return out

    def advantages(values):
        mean = sum(values) / len(values)
        return [0.0] * len(values), mean, 0.0

    _group, _samples, wire = rj.rollout_scenario(scenario, 2, rj.TurnBudget(), advancing, advantage_fn=advantages)
    rows = [{"iteration": 1, **w} for w in wire]
    replayed = list(demos.replay_wire(rows, {scenario.scenario_id: scenario}))
    assert len(replayed) == 2
    for episode, row in replayed:
        assert round(episode.score().total, 4) == round(row["reward"], 4)
    tampered = [dict(r, reward=r["reward"] + 1.0) for r in rows]
    with pytest.raises(SystemExit, match="not the environment the run saw"):
        list(demos.replay_wire(tampered, {scenario.scenario_id: scenario}))


def test_an_sft_completion_ends_the_turn_as_the_policy_does():
    torch = pytest.importorskip("torch")  # noqa: F841
    import sft_rocjitsu

    class Tok:
        unk_token_id = 0
        eos_token_id = 2

        def convert_tokens_to_ids(self, token):
            return 7 if token == "<|im_end|>" else self.unk_token_id

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 50 + 10 for c in text]}

    row = {"scenario_id": "s", "prompt": "p", "completion": '{"tool": "waitcheck"}', "episode": 0, "step": 1}
    sample = sft_rocjitsu.demo_samples([row], Tok())[0]
    assert sample.completion.token_ids[-1] == 7
    assert sample.completion.token_ids[:-1] == tuple(ord(c) % 50 + 10 for c in row["completion"])
    assert sample.advantage == 1.0


def test_the_quirk_is_exactly_the_s_load_write_after_write_diagnostic():
    quirk = {"what": "missing s_waitcnt lgkmcnt(0) before def of s49", "producer": "s_load_dwordx8 s[44:51], s[0:1], 0x50",
             "consumer": "s_load_dword s49, s[0:1], 0x88"}
    assert demos.is_quirk(quirk)
    assert not demos.is_quirk(dict(quirk, what="missing s_waitcnt lgkmcnt(0) before use of s49"))
    assert not demos.is_quirk(dict(quirk, consumer="s_mov_b32 s49, 0"))
    assert not demos.is_quirk(dict(quirk, producer="buffer_load_dword v3, v1, s[4:7], 0 offen"))


@needs_archive
def test_quirk_demonstrations_are_verified_none_answers_on_quirk_only_clean_controls():
    scenarios = rj.load_corpus(ARCHIVE, split="train", reward="v2")
    kept, stats = demos.quirk_demos(scenarios)
    assert stats["kept"] == len(kept) > 30
    for e in kept:
        assert e.scenario.clean and e.outcome == "clean_correct" and e.answer["class"] == "none"
        shown = next(s for s in e.steps if s.action.tool == "waitcheck").body
        lines = [line for line in shown.splitlines() if "missing s_waitcnt" in line]
        assert lines and all("before def of s" in line for line in lines), "chosen from what the agent was shown"
    held_out = {s.scenario_id for split in ("val", "test", "test_reencode") for s in rj.load_corpus(ARCHIVE, split=split)}
    assert not {e.scenario.scenario_id for e in kept} & held_out


@needs_archive
def test_quirk_demonstrations_are_added_the_asked_number_of_times(tmp_path, monkeypatch):
    monkeypatch.setenv(rj.PROMPT_ENV, "procedure")
    out = tmp_path / "q.jsonl"
    assert demos.main(["--corpus-root", str(ARCHIVE), "--split", "train", "--quirk-demos", "2", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    stats = json.loads(out.with_suffix(".stats.json").read_text())
    starts = [r for r in rows if r["step"] == 1]
    assert {r["source"] for r in rows} == {"quirk:0", "quirk:1"}
    assert len(starts) == 2 * stats["quirk"]["kept"]
    assert all(json.loads(r["completion"]) == {"answer": {"class": "none"}} for r in rows
               if r["completion"].lstrip().startswith('{"answer"'))


def test_each_quirk_demo_guard_holds_on_its_own(monkeypatch):
    """The clean-only rule and the clean_correct rule each reject what the other would let through."""
    from types import SimpleNamespace

    def fake_play(scenario, policy, index=0):
        return SimpleNamespace(scenario=scenario, answer={"class": "none"}, outcome=scenario.outcome)

    monkeypatch.setattr(demos, "play", fake_play)
    faulty_but_scored_correct = SimpleNamespace(clean=False, scenario_id="f", outcome="clean_correct")
    clean_but_not_correct = SimpleNamespace(clean=True, scenario_id="c", outcome="false_alarm")
    good = SimpleNamespace(clean=True, scenario_id="g", outcome="clean_correct")
    kept, stats = demos.quirk_demos([faulty_but_scored_correct, clean_but_not_correct, good])
    assert [e.scenario.scenario_id for e in kept] == ["g", "g"]
    assert stats["not_clean_correct"] == 2 and stats["kept"] == 2
