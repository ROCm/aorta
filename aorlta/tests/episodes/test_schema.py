"""C8.T1: every record validates against the schema (`aorlta.episode/1` and the iteration summary)."""

import copy
from collections import Counter

import pytest

from aorlta.episodes.schema import EPISODE_KEYS, EpisodeError, validate_episode, validate_summary
from aorlta.episodes.writer import complete_iterations, read_episodes, read_summary
from aorlta.orchestrator.fakes import FakeRunSpec, build_fake_run


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two fake runs that between them produce every end reason, status and exclusion the loop can write."""
    root = tmp_path_factory.mktemp("c8t1")
    out = []
    probe = build_fake_run(FakeRunSpec(root=str(root / "probe"), iterations=1, eval_every=None))
    assert probe.orchestrator.run()["status"] == "completed"
    eps = read_episodes(probe.cfg.run_dir, 1)
    first_pid = eps[0]["prompt_id"]
    drop = [e["chain_id"] for e in eps if e["prompt_id"] == first_pid][1:]  # 7 of one group: the group is dropped
    gated = [e["chain_id"] for e in eps if e["trained"] and e["prompt_id"] != first_pid][:1]
    for name, extra in (("limits", {"rollout": {"max_model_len": 1000}, "min_scored_fraction": 0.5,
                                    "reward": {"unscored": drop}, "trainer": {"gap_exclude": gated}}),
                        ("leak", {"leak_check": True, "rollout": {"quarantine_rate": 0.3}, "min_scored_fraction": 0.9})):
        spec = FakeRunSpec(root=str(root / name), iterations=2, eval_every=1, **extra)
        r = build_fake_run(spec)
        res = r.orchestrator.run()
        assert res["status"] == "completed", res["stop"]
        out.append(r)
    return out


def all_episodes(runs):
    return [e for r in runs for n in complete_iterations(r.cfg.run_dir) for e in read_episodes(r.cfg.run_dir, n)]


def test_c8_t1_every_chain_record_validates(runs):
    eps = all_episodes(runs)
    for e in eps:
        validate_episode(e)  # read_episodes has validated too; this is the explicit statement
    assert len(eps) == 2 * 2 * 32


def test_c8_t1_the_records_cover_every_end_reason_status_and_exclusion(runs):
    eps = all_episodes(runs)
    ends = Counter(e["end_reason"] for e in eps)
    assert set(ends) == {"compiled", "no_code_block", "turns_exhausted", "context_limit", "quarantined",
                         "engine_error", "logprob_error"}, ends
    assert {e["status"] for e in eps} == {"scored", "unscored", "quarantined"}
    excl = {e["excluded_reason"] for e in eps}
    assert {None, "quarantined", "engine_error", "unscored", "logprob_error", "group_dropped", "gap_gate"} <= excl
    assert {e["source"] for e in eps} == {"standin"}


def test_c8_t1_every_summary_validates(runs):
    for r in runs:
        for n in complete_iterations(r.cfg.run_dir):
            validate_summary(read_summary(r.cfg.run_dir, n))


@pytest.fixture
def record(runs):
    eps = all_episodes(runs)
    return copy.deepcopy(next(e for e in eps if e["trained"] and len(e["turns"]) >= 2))


def _set(path, value):
    def f(rec):
        obj = rec
        for k in path[:-1]:
            obj = obj[k]
        obj[path[-1]] = value
    return f


MUTATIONS = {
    "unknown key": lambda r: r.__setitem__("response_text", "x"),
    "missing key": lambda r: r.pop("legs"),
    "schema id": _set(["schema"], "aorlta.episode/2"),
    "source": _set(["source"], "partner"),
    "chain id not the record's": _set(["k"], 7),
    "adapter version not v(n-1)": _set(["adapter_version"], "fake-run:v5"),
    "adapter version of another run": _set(["adapter_version"], "other:v0"),
    "end reason": _set(["end_reason"], "timeout"),
    "final turn": _set(["final_turn"], 0),
    "turn number": _set(["turns", 1, "turn"], 2),
    "seed above 31 bits": _set(["turns", 0, "seed"], 2**31),
    "prompt hash": _set(["turns", 0, "prompt_ids_sha256"], "abc"),
    "negative token id": _set(["turns", 0, "response_ids", 0], -1),
    "log-probs misaligned": lambda r: r["turns"][0]["response_logprobs"].pop(),
    "log-prob above 0": _set(["turns", 0, "response_logprobs", 0], 0.5),
    "missing log-probs on a non-final turn": _set(["turns", 0, "response_logprobs"], []),
    "finish reason": _set(["turns", 0, "finish_reason"], {"type": "abort"}),
    "time backwards": _set(["turns", 0, "t_done"], 0.0),
    "compile text": _set(["turns", 0, "compile", "errors"], ["use of undeclared identifier"]),
    "compile key": lambda r: r["turns"][0]["compile"].__setitem__("stderr", "x"),
    "leak check": _set(["turns", 0, "leak_check"], "skipped"),
    "scored without reward": _set(["reward"], None),
    "reward out of range": _set(["reward"], 1.5),
    "scored with an unscored reason": _set(["unscored_reason"], "timeout"),
    "free-text reason": lambda r: (r.update(status="unscored", reward=None, advantage=None, trained=False,
                                            excluded_reason="unscored", unscored_reason="the kernel at line 3 is bad")),
    "status": _set(["status"], "done"),
    "candidate hash": _set(["candidate_sha256"], "x" * 64),
    "leg state": _set(["legs", "san.asan"], "passed"),
    "plan with text": _set(["plan"], {"selector_version": "s1", "skip": ["san.asan"], "context": "text"}),
    "trained but unscored": lambda r: r.update(status="unscored", reward=None, unscored_reason="timeout"),
    "trained without an advantage": _set(["advantage"], None),
    "trained with a reason": _set(["excluded_reason"], "gap_gate"),
    "untrained without a reason": lambda r: r.update(trained=False, excluded_reason=None),
    "unknown exclusion": lambda r: r.update(trained=False, excluded_reason="unlucky"),
    "advantage not finite": _set(["advantage"], float("nan")),
}


def test_the_valid_record_used_below_validates(record):
    validate_episode(record)
    assert set(record) == EPISODE_KEYS


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_c8_t1_a_record_that_breaks_a_rule_is_refused(record, name):
    MUTATIONS[name](record)
    with pytest.raises(EpisodeError):
        validate_episode(record)


def test_quarantined_turns_carry_no_ids(runs):
    q = next(e for e in all_episodes(runs) if e["end_reason"] == "quarantined")
    assert all(t["response_ids"] == [] and t["response_logprobs"] == [] for t in q["turns"])
    bad = copy.deepcopy(q)
    bad["turns"][0]["response_ids"] = [10_001]
    with pytest.raises(EpisodeError):
        validate_episode(bad)


def test_a_summary_that_breaks_a_rule_is_refused(runs):
    r = runs[0]
    s = read_summary(r.cfg.run_dir, 1)
    for mutate in (lambda s: s.pop("trainer"), lambda s: s.__setitem__("extra", 1),
                   lambda s: s.__setitem__("scored_fraction", 1.5),
                   lambda s: s["unscored_by_reason"].__setitem__("text with spaces", 1),
                   lambda s: s.__setitem__("warnings", ["a warning with text"]),
                   lambda s: s.__setitem__("policy_version_trained", "other:v1")):
        bad = copy.deepcopy(s)
        mutate(bad)
        with pytest.raises(EpisodeError):
            validate_summary(bad)
