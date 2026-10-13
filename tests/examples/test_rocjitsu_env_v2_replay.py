"""The adapter's v2 episodes against the reference environment, ``rjenv2.py``.

Both play the same replies with the same seed on the same archived scenarios;
the adapter must return the same structured tool result at every step, fire the
same events with the same points under reward v1, and the same events re-priced
by :func:`rocjitsu_env.reprice` under reward v2. Skipped unless
``$AORTA_RJ_TEST_CORPUS_V2`` names a corpus-v2 build directory holding the
archive and the reference environment (``archive/``, ``scripts/rjenv2.py``). The
same replay runs on a sample of ``archive-hard``'s new scenarios, which were
recorded without native execution, where it is skipped if that archive is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import rocjitsu_env as rj  # noqa: E402

#: A corpus-v2 build directory; unset, it names a path that does not exist.
CORPUS_V2 = Path(os.environ.get("AORTA_RJ_TEST_CORPUS_V2") or Path(__file__).with_name("no-corpus-v2"))
ARCHIVE, REFERENCE = CORPUS_V2 / "archive", CORPUS_V2 / "scripts" / "rjenv2.py"
ARCHIVE_HARD = CORPUS_V2 / "archive-hard"
pytestmark = pytest.mark.skipif(not (ARCHIVE / "scenarios.jsonl").is_file() or not REFERENCE.is_file(),
                                reason="corpus-v2 archive or rjenv2.py not present (set AORTA_RJ_TEST_CORPUS_V2)")
SAMPLE = 40
HARD_SAMPLE = 20


def reference_module():
    spec = importlib.util.spec_from_file_location("rjenv2", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ref():
    module = reference_module()
    return module, module.Archive(str(ARCHIVE))


@pytest.fixture(scope="module")
def sample():
    ids = rj.CorpusV2(ARCHIVE).split_ids("test")
    rng = random.Random(20261005)
    faulty = [s for s in ids if not s.startswith("CLEAN")]
    clean = [s for s in ids if s.startswith("CLEAN")]
    return sorted(rng.sample(faulty, SAMPLE - 8)) + sorted(rng.sample(clean, 8))


def scripts(scenario: rj.Scenario) -> list[list[dict]]:
    """Replies that exercise every tool, both run_sim modes, every terminal and a malformed call."""
    k = scenario.kernel or scenario.kernels[0]
    other = next((x for x in scenario.kernels if x != k), k)
    true = {"class": scenario.family, "kernel": k, "offset": scenario.offsets[0] if scenario.offsets else 0}
    near = dict(true, offset=true["offset"] + 4)
    return [
        [{"tool": "list_kernels"}, {"tool": "waitcheck"}, {"tool": "run_sim", "kernel": "all"},
         {"tool": "run_sim", "kernel": k, "seed": 2}, {"tool": "run_native", "kernel": "all"},
         {"tool": "disasm", "kernel": k, "start": max(0, true["offset"] - 16), "end": true["offset"] + 16},
         {"answer": true}],
        [{"tool": "waitcheck", "kernel": k}, {"tool": "run_native", "kernel": k}, {"tool": "consan", "kernel": k},
         {"answer": near}],
        [{"tool": "run_sim", "kernel": "no_such_kernel"}, {"answer": {"class": "none"}}],
        [{"answer": dict(true, kernel=other)}],
        [{"answer": {"class": "waitcnt" if scenario.family != "waitcnt" else "barrier", "kernel": k}}],
        [{"tool": "waitcheck"}] * 8,
    ]


def play_reference(module, archive, sid, replies, seed):
    episode = module.Episode(archive.scenario(sid), archive, seed=seed)
    results = []
    for reply in replies:
        if episode.done:
            break
        episode.observe()
        before = len(episode.steps)
        episode.act(json.dumps(reply))
        results.append(episode.last_obs if len(episode.steps) > before and episode.steps[-1][0] != "malformed"
                       else None)
    return episode, results


def play_adapter(scenario, replies, seed):
    episode = rj.Episode(scenario, seed=seed)
    results = []
    for reply in replies:
        if episode.done:
            break
        episode.observe()
        episode.last_result = None
        episode.act(json.dumps(reply))
        last = episode.steps[-1]
        results.append(episode.last_result if last.action.kind == "tool" else None)
    return episode, results


def replay(module, archive, scenarios, sample, reward):
    """Every script on every sampled scenario through both environments; the adapter's plays."""
    plays = []
    for sid in sample:
        scenario = scenarios[sid]
        for n, replies in enumerate(scripts(scenario)):
            seed = 1000 + n
            theirs, their_results = play_reference(module, archive, sid, replies, seed)
            ours, our_results = play_adapter(scenario, replies, seed)
            assert our_results == their_results, (sid, n)
            their_events = [(name, points) for _, name, points in theirs.events]
            if reward == "v2":
                their_events = rj.reprice(their_events, "v2")
            assert [(e.name, round(e.points, 4)) for e in ours.events] == \
                [(name, round(points, 4)) for name, points in their_events], (sid, n)
            assert round(ours.score().total, 4) == round(sum(p for _, p in their_events), 4), (sid, n)
            plays.append((replies, our_results, ours))
    return plays


@pytest.mark.parametrize("reward", ["v1", "v2"])
def test_v2_episodes_replay_the_reference_environment(ref, sample, reward):
    module, archive = ref
    scenarios = {s.scenario_id: s for s in rj.load_corpus(ARCHIVE, split="test", reward=reward, only=sample)}
    plays = replay(module, archive, scenarios, sample, reward)
    assert len(plays) == SAMPLE * len(scripts(scenarios[sample[0]]))


@pytest.mark.skipif(not (ARCHIVE_HARD / "splits" / "hard_all.txt").is_file(), reason="archive-hard not present")
@pytest.mark.parametrize("reward", ["v1", "v2"])
def test_hard_tier_episodes_replay_the_reference_and_run_native_is_not_run(reward):
    module = reference_module()
    archive = module.Archive(str(ARCHIVE_HARD))
    ids = rj.CorpusV2(ARCHIVE_HARD).split_ids("hard_all")
    sample = sorted(random.Random(20261005).sample(ids, HARD_SAMPLE))
    scenarios = {s.scenario_id: s for s in rj.load_corpus(ARCHIVE_HARD, split="hard_all", reward=reward, only=sample)}
    assert all(s.record.get("native_source") == "not_run" and s.record["native"] == [] for s in scenarios.values())
    plays = replay(module, archive, scenarios, sample, reward)
    assert len(plays) == HARD_SAMPLE * len(scripts(scenarios[sample[0]]))
    natives = 0
    for replies, results, episode in plays:
        assert all(e.name != "cost_run_native" for e in episode.events)
        for reply, result in zip(replies, results):
            if reply.get("tool") == "run_native":
                assert result and {r["status"] for r in result} == {"NOT_RUN"}
                natives += 1
    assert natives == 2 * HARD_SAMPLE


def test_v2_prompts_carry_no_answer_field(sample):
    scenarios = rj.load_corpus(ARCHIVE, split="test", only=sample)
    system = rj.system_for(scenarios)
    for scenario in scenarios:
        rec = scenario.record
        # operator_kind ("move", "weaken") is a category word the task description itself uses.
        secrets = {scenario.scenario_id, rec.get("op"), rec.get("note"), rec.get("tier"),
                   rec.get("provenance") if isinstance(rec.get("provenance"), str) else None}
        secrets = {s for s in secrets if isinstance(s, str) and len(s) > 3 and s != "none"}
        for replies in scripts(scenario)[:2]:
            episode = rj.Episode(scenario, seed=7)
            prompts = [system]
            for reply in replies:
                if episode.done:
                    break
                prompts.append(episode.observe())
                episode.act(json.dumps(reply))
            for prompt in prompts:
                for secret in secrets:
                    assert secret not in prompt, (scenario.scenario_id, secret)


def test_the_v2_system_prompt_states_the_active_reward():
    assert "paid only together with the correct class" in rj.system_prompt_v2("v2")
    assert "+0.5 correct kernel" in rj.system_prompt_v2("v2")
    assert "+2 correct kernel," in rj.system_prompt_v2("v1")
    assert "list_kernels" in rj.system_prompt_v2("v2") and "memorder" in rj.system_prompt_v2("v2")
