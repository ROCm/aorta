"""The rocjitsu adapter on corpus v2, and reward v2.

The reward tests need nothing outside the repository, nor do the ``run_native``
tests, which build a three-record corpus in a temporary directory. The replay
tests run against the corpus-v2 archive and its reference environment
(``rjenv2.py``) and are skipped where they are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import rocjitsu_env as rj  # noqa: E402

#: Pre-registered in docs/rl-rocjitsu-reward-v2.md before any val/test score under v2.
REWARD_V2_SHA256 = "74db77d5286fd5707529dbb777b757cc8738f3b197af1875d897c28d20f3bfc5"
REWARD_V1_SHA256 = "4503f7f068887471412d7b191fa79b5a5b34cc419211475f955f7856719f8ea1"


def test_reward_v2_is_the_pre_registered_one():
    assert rj.reward_spec_sha256("v2") == REWARD_V2_SHA256
    assert rj.reward_spec_sha256("v1") == REWARD_V1_SHA256
    assert rj.POINTS_V2 == {**rj.POINTS, "kernel_correct": 0.5}
    assert rj.KERNEL_PAID_ONLY_WITH == {"v1": (), "v2": ("class_correct", "location_exact", "location_near")}


@pytest.mark.parametrize("with_", [(), ("class_correct",), ("location_exact",), ("location_near",),
                                   ("class_correct", "location_exact")])
def test_v2_pays_the_kernel_only_with_the_class_or_a_location(with_):
    events = [("kernel_correct", 2.0)] + [(n, rj.POINTS[n]) for n in with_] + [("cost_waitcheck", -0.05)]
    v2 = dict(rj.reprice(events, "v2"))
    assert v2["kernel_correct"] == (0.5 if with_ else 0.0)
    assert v2["cost_waitcheck"] == -0.05
    assert rj.reprice(events, "v1") == events


def test_reprice_keeps_every_other_terminal_and_every_cost():
    for name in rj.TERMINAL_EVENTS:
        if name != "kernel_correct":
            assert rj.reprice([(name, rj.POINTS[name])], "v2") == [(name, rj.POINTS[name])]
    assert rj.reprice([("cost_run_sim_all", -0.35)], "v2") == [("cost_run_sim_all", -0.35)]


def _toy_corpus(root: Path) -> Path:
    """One two-kernel workload and three faulty records that differ only in their native half."""
    (root / "listings").mkdir(parents=True)
    (root / "splits").mkdir()
    kernels = {k: {"kid": k, "symbol": f"sym_{k}", "demangled": f"kern_{k}", "family": "elementwise",
                   "library": "pytorch", "n_insns": 2, "n_exec": 2, "objects": [], "heldout": False}
               for k in ("k000", "k001")}
    (root / "kernels.json").write_text(json.dumps(kernels))
    workload = {"description": "toy workload", "kernels": list(kernels), "n_dispatches": 2, "total_wave_insts": 8,
                "trace": [{"kernel": k, "dispatches": 1, "ops": [f"aten::op{i}"]} for i, k in enumerate(kernels)],
                "ops": [{"i": i, "op": f"aten::op{i}", "kernels": [k]} for i, k in enumerate(kernels)]}
    (root / "workloads.json").write_text(json.dumps({"toy": workload}))
    for k in kernels:
        (root / "listings" / f"{k}.json").write_text(json.dumps([[0, "s_waitcnt vmcnt(0)"], [4, "s_endpgm"]]))
    base = {"workload": "toy", "kid": "k000", "op": "waitcnt_weaken_vm", "cls": "waitcnt", "offsets": [0],
            "sim": {"rc": 0, "ops": {}, "races": [], "memwait": []}, "simq": {}, "waitcheck_view": []}
    recs = [dict(base, sid="notrun@toy", native=[], native_source="not_run"),
            dict(base, sid="legacy@toy", native=[]),
            dict(base, sid="ran@toy", native=[{"rc": 0, "ops": {"0": [False, 1.0, 3], "1": [True, 0.0, 0]}, "ok": False}])]
    (root / "scenarios.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (root / "splits" / "test.txt").write_text("".join(r["sid"] + "\n" for r in recs))
    return root


def _run_native(scenario, kernel="all"):
    episode = rj.Episode(scenario, seed=0)
    episode.observe()
    episode.act(json.dumps({"tool": "run_native", "kernel": kernel}))
    return episode


def test_run_native_on_a_scenario_with_no_native_run_is_not_run_and_free(tmp_path):
    scenario = {s.scenario_id: s for s in rj.load_corpus(_toy_corpus(tmp_path), split="test")}["notrun@toy"]
    episode = _run_native(scenario)
    note = "no native run exists for this scenario"
    assert episode.last_result == [{"kernel": "k000", "status": "NOT_RUN", "note": note},
                                   {"kernel": "k001", "status": "NOT_RUN", "note": note}]
    assert episode.events == [] and episode.steps[-1].cost == 0 and episode.turns_used == 1
    assert "NOT_RUN" in episode.steps[-1].body and "no cost" in episode.steps[-1].body
    assert "CRASH" not in episode.steps[-1].body
    episode.act(json.dumps({"tool": "run_native", "kernel": "sym_k001"}))
    assert episode.last_result == [{"kernel": "k001", "status": "NOT_RUN", "note": note}]
    assert episode.events == [] and episode.turns_used == 2


def test_a_kernel_with_no_listing_file_is_refused_when_the_corpus_loads(tmp_path):
    root = _toy_corpus(tmp_path)
    (root / "listings" / "k001.json").unlink()
    with pytest.raises(SystemExit, match=r"k001 \(.*listings/k001\.json\)"):
        rj.load_corpus(root, split="test")


def test_a_complete_corpus_loads_its_listings_and_an_empty_listing_stays_legal(tmp_path):
    root = _toy_corpus(tmp_path)
    scenario = {s.scenario_id: s for s in rj.load_corpus(root, split="test")}["ran@toy"]
    assert scenario.insn_offsets == (0, 4)
    episode = rj.Episode(scenario, seed=0)
    episode.observe()
    episode.act(json.dumps({"tool": "disasm", "kernel": "k001"}))
    assert "0x0000: s_waitcnt vmcnt(0)" in episode.steps[-1].body and "0x0004: s_endpgm" in episode.steps[-1].body

    (root / "listings" / "k001.json").write_text("[]")
    scenario = {s.scenario_id: s for s in rj.load_corpus(root, split="test")}["ran@toy"]
    assert scenario.corpus.listing("k001") == []
    with pytest.raises(SystemExit, match="kernel k999 has no listing file"):
        scenario.corpus.listing("k999")


def test_run_native_is_unchanged_where_the_not_run_marker_is_absent(tmp_path):
    scenarios = {s.scenario_id: s for s in rj.load_corpus(_toy_corpus(tmp_path), split="test")}
    legacy = _run_native(scenarios["legacy@toy"])
    assert legacy.last_result == [{"kernel": "k000", "status": "CRASH", "rc": None},
                                  {"kernel": "k001", "status": "CRASH", "rc": None}]
    assert [(e.name, e.points) for e in legacy.events] == [("cost_run_native", -0.5)]
    ran = _run_native(scenarios["ran@toy"])
    assert [r["status"] for r in ran.last_result] == ["FAIL", "PASS"]
    assert [(e.name, e.points) for e in ran.events] == [("cost_run_native", -0.5)]


@pytest.mark.parametrize("offset, cls, total, outcome", [
    (0, "waitcnt", 5.5, "exact"),
    (4, "waitcnt", 3.5, "near"),  # the kernel's last instruction, one from the target
    (4, "barrier", 2.5, "near"),
    (8, "waitcnt", 1.5, "kernel_only"),  # one past the kernel's end: class and kernel only
    (8, "barrier", 0.0, "kernel_only"),  # ... and with the class wrong, v2 does not pay the kernel
    (-4, "waitcnt", 1.5, "kernel_only"),  # one before its start
    (2, "waitcnt", 1.5, "kernel_only"),  # inside an instruction
])
def test_v2_location_credit_needs_an_instruction_of_the_kernel(tmp_path, offset, cls, total, outcome):
    scenario = {s.scenario_id: s for s in rj.load_corpus(_toy_corpus(tmp_path), split="test", reward="v2")}["ran@toy"]
    episode = rj.Episode(scenario, seed=0)
    episode.observe()
    episode.act(json.dumps({"answer": {"class": cls, "kernel": "k000", "offset": offset}}))
    assert episode.outcome == outcome and episode.score().total == total


@pytest.mark.parametrize("offset, outcome", [
    (12, "near"),  # only the listing as built has it (the edit replaced it); two from the target
    (10, "near"),  # only the listing as served has it; placed at 8, one from the target
    (14, "near"),  # likewise; placed at 12
    (11, "kernel_only"),  # in neither listing
    (28, "kernel_only"),
])
def test_v2_location_credit_takes_an_instruction_of_either_listing(tmp_path, offset, outcome):
    root = _toy_corpus(tmp_path)
    (root / "listings" / "k000.json").write_text(json.dumps([[o, "s_nop 0"] for o in range(0, 28, 4)]))
    rec = {"workload": "toy", "kid": "k000", "op": "waitcnt_weaken_vm", "cls": "waitcnt", "offsets": [4],
           "native": [], "sim": {"rc": 0, "ops": {}, "races": [], "memwait": []}, "simq": {}, "waitcheck_view": [],
           "sid": "edit@toy", "mutation": {"range_off": 8, "old": "00" * 8, "new": "11" * 8,
                                           "window": [[8, "s_nop 1"], [10, "s_nop 2"], [14, "s_nop 3"]]}}
    (root / "scenarios.jsonl").write_text(json.dumps(rec) + "\n")
    (root / "splits" / "test.txt").write_text("edit@toy\n")
    [scenario] = rj.load_corpus(root, split="test", reward="v2")
    episode = rj.Episode(scenario, seed=0)
    assert [o for o, _ in episode.tools.listing("k000")] == [0, 4, 8, 10, 14, 16, 20, 24]
    episode.observe()
    episode.act(json.dumps({"answer": {"class": "waitcnt", "kernel": "k000", "offset": offset}}))
    assert episode.outcome == outcome


@pytest.mark.parametrize("seed", ["1e309", "-1e309", "NaN", "1.5", "true", "false", '"2"', '""', "4", "-1",
                                  "1" + "0" * 400, "[1]"])
def test_a_run_sim_seed_that_is_not_a_seed_is_a_malformed_reply_not_a_crash(tmp_path, seed):
    scenario = {s.scenario_id: s for s in rj.load_corpus(_toy_corpus(tmp_path), split="test", reward="v2")}["ran@toy"]
    episode = rj.Episode(scenario, seed=0)
    episode.observe()
    episode.act('{"tool": "run_sim", "kernel": "all", "seed": ' + seed + "}")
    assert [(e.name, e.points) for e in episode.events] == [("malformed", -1.0)]
    assert episode.steps[-1].header == "invalid reply" and "seed must be one of [0, 1, 2, 3]" in episode.steps[-1].body
    assert not episode.done and episode.turns_used == 1


@pytest.mark.parametrize("field, seed", [("", 0), (', "seed": null', 0), (', "seed": 0', 0), (', "seed": 3', 3),
                                         (', "seed": 2.0', 2)])
def test_a_run_sim_seed_that_is_a_seed_is_accepted_as_before(tmp_path, field, seed):
    action = rj.parse_action_v2('{"tool": "run_sim", "kernel": "all"' + field + "}", lambda k: k)
    assert action.kind == "tool" and action.seed == seed


def test_a_v2_corpus_still_takes_either_reward(tmp_path):
    root = _toy_corpus(tmp_path)
    v2 = rj.load_corpus(root, split="test", reward="v2")
    assert {s.reward for s in v2} == {"v2"} and rj.system_for(v2) == rj.system_prompt_v2("v2")
    v1 = rj.load_corpus(root, split="test", reward="v1")
    assert rj.with_reward(v1, "v2") == v2 and rj.with_reward(v2, "v1") == v1


def test_the_reward_version_is_chosen_by_argument_then_environment_then_v1(monkeypatch):
    monkeypatch.delenv(rj.REWARD_ENV, raising=False)
    assert rj.resolve_reward(None) == "v1"
    monkeypatch.setenv(rj.REWARD_ENV, "v2")
    assert rj.resolve_reward(None) == "v2"
    assert rj.resolve_reward("v1") == "v1"
    with pytest.raises(SystemExit):
        rj.resolve_reward("v3")
