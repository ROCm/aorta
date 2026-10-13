"""The rocjitsu episode adapter: parsing, rewards, turn limits, leakage, replay.

Most tests run on a small synthetic archive built here, so they need nothing
outside the repository. The replay tests at the end run against the
prototype's real campaign archive, named by ``$AORTA_RJ_TEST_PROTOTYPE``, and
are skipped where it is not given.
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

import prepare_rocjitsu_corpus as prep  # noqa: E402
import rocjitsu_baselines as rb  # noqa: E402
import rocjitsu_env as rj  # noqa: E402

#: The prototype's campaign directory; unset, it names a path that does not exist.
PROTOTYPE = Path(os.environ.get("AORTA_RJ_TEST_PROTOTYPE") or Path(__file__).with_name("no-prototype"))

# ---------------------------------------------------------------------------
# a synthetic archive in the campaign's layout
# ---------------------------------------------------------------------------

TEXT_VADDR = 0x1000
STARTS = {"k_a": 0x1000, "k_b": 0x1100}
LISTINGS = {
    "k_a": ["s_load_dwordx2 s[0:1], s[4:5], 0x0", "ds_read_b32 v1, v0", "s_waitcnt lgkmcnt(0)",
            "v_add_u32_e32 v2, v1, v2", "s_waitcnt vmcnt(0)", "v_mul_lo_u32 v3, v2, v2",
            "global_store_dword v[4:5], v3, off", "s_endpgm"],
    "k_b": ["ds_write_b32 v0, v1", "s_barrier", "ds_read_b32 v2, v0", "s_waitcnt lgkmcnt(0)",
            "v_add_u32_e32 v3, v2, v3", "s_endpgm"],
}
SECRET_NOTE = "lgkmcnt(0)->lgkmcnt(1)"


def _native(kernel: str, status: str = "PASS", bad: int = 0, digest: str = "good") -> list[dict]:
    return [{"rc": 0, "kernels": [{"kernel": kernel, "status": status, "bad": bad, "digest": digest}]}]


def _sim(kernel: str, races=(), memwait=()) -> list[dict]:
    return [{"kernels": [{"kernel": kernel, "status": "PASS", "digest": "good"}],
             "races": list(races), "memwait": list(memwait)}]


def _race(a: int, b: int, kind: str = "VGPR") -> dict:
    return {"type": kind, "access": "read", "wave": 0, "lane": 0,
            "marks": [{"kernel_offset": a, "insn": "ds_read_b32 v1, v0", "wave": 0},
                      {"kernel_offset": b, "insn": "v_add_u32_e32 v2, v1, v2", "wave": 1}]}


def _consan(conflicts=()) -> dict:
    return {"conflicts": list(conflicts), "verdicts": ["applicable=true analysis_complete=true"]}


def _baseline(kernel: str) -> dict:
    return {"site_id": f"BASELINE:{kernel}", "kernel": kernel, "op": "none", "native": _native(kernel),
            "sim": _sim(kernel), "waitcheck": {"diagnostics": [], "stdout": "rj_waitcheck: /work/suite.hsaco"},
            "consan": _consan()}


MUTANTS = {
    "k_a+0x8.waitcnt_weaken": {
        "kernel": "k_a", "kernel_offset": 8, "op": "waitcnt_weaken", "moved_to": None, "note": SECRET_NOTE,
        "native": _native("k_a", "FAIL", 12, "bad"),
        "sim": _sim("k_a", [_race(4, 12)], [{"kernel_offset": 12, "producer_offset": 4, "reg": "v1",
                                             "counter": "lgkmcnt", "threshold": 0}] * 3),
        "waitcheck": {"diagnostics": [{"text_offset": 0x0c, "what": "insufficient", "wait": "lgkmcnt(1)",
                                       "kind": "read", "reg": "v1", "producer_text_offset": 0x04,
                                       "producer_insn": "ds_read_b32 v1, v0", "consumer_text_offset": 0x0c,
                                       "consumer_insn": "v_add_u32_e32 v2, v1, v2"}],
                      "stdout": "rj_waitcheck: /work/mutants/k_a+0x8.waitcnt_weaken.hsaco"},
        "consan": _consan(),
    },
    "k_a+0x10.waitcnt_late": {
        "kernel": "k_a", "kernel_offset": 16, "op": "waitcnt_late", "moved_to": 20, "note": "after v_mul",
        "native": _native("k_a"),
        "sim": _sim("k_a", [], [{"kernel_offset": 20, "producer_offset": 0, "reg": "v4", "counter": "vmcnt",
                                 "threshold": 0}]),
        "waitcheck": {"diagnostics": []},
        "consan": _consan(),
    },
    "k_b+0x4.barrier_late": {
        "kernel": "k_b", "kernel_offset": 4, "op": "barrier_late", "moved_to": 8, "note": "after ds_read",
        "native": _native("k_b"),
        "sim": _sim("k_b", [_race(0, 8, "LDS")]),
        "waitcheck": {"diagnostics": []},
        "consan": _consan([{"first_instruction": "0x100", "second_instruction": "0x108",
                            "workgroup": [0, 0, 0]}]),
    },
}


def build_corpus(root: Path, test_kernels: str = "k_b", clean_frac: float = 0.25) -> Path:
    src = root / "src"
    src.mkdir(parents=True)
    offsets = {k: [4 * i for i in range(len(v))] for k, v in LISTINGS.items()}
    meta = {"text_vaddr": TEXT_VADDR, "kernel_starts": STARTS, "kernel_insn_offsets": offsets}
    (src / "sites.json").write_text(json.dumps({"meta": meta}))
    lines = []
    for k, insns in LISTINGS.items():
        lines.append(f"{STARTS[k]:016x} <{k}>:")
        lines += [f"\t{t:<40} // {STARTS[k] + 4 * i:012X}: {0xBF800000 + i:08X}" for i, t in enumerate(insns)]
    (src / "suite.dis").write_text("\n".join(lines) + "\n")
    recs = [_baseline(k) for k in LISTINGS]
    rows = []
    for sid, m in MUTANTS.items():
        recs.append({"site_id": sid, "path": f"/work/mutants/{sid}.hsaco", **m})
        rows.append({"site_id": sid, "kernel": m["kernel"], "op": m["op"], "kernel_offset": m["kernel_offset"],
                     "class": "manifest", "trace": False})
    rows.append({"site_id": "k_a+0x14.waitcnt_drop", "kernel": "k_a", "op": "waitcnt_drop",
                 "kernel_offset": 20, "class": "manifest", "trace": True})
    rows.append({"site_id": "k_a+0x0.waitcnt_weaken", "kernel": "k_a", "op": "waitcnt_weaken",
                 "kernel_offset": 0, "class": "silent", "trace": False})
    (src / "results.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (src / "analysis.json").write_text(json.dumps({"rows": rows}))
    out = root / "corpus"
    prep.main(["--source", str(src), "--out", str(out), "--test-kernels", test_kernels,
               "--clean-frac", str(clean_frac)])
    return out


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    return build_corpus(tmp_path)


def scenario(corpus: Path, sid: str) -> rj.Scenario:
    return next(s for s in rj.load_corpus(corpus, split=None) if s.scenario_id == sid)


def play(ep: rj.Episode, *actions) -> rj.Episode:
    for a in actions:
        ep.act(a if isinstance(a, str) else json.dumps(a))
    return ep


def events(ep: rj.Episode) -> list[str]:
    return [e.name for e in ep.events]


ANSWER = {"answer": {"class": "waitcnt", "kernel": "k_a", "offset": "0x0008"}}

# ---------------------------------------------------------------------------
# the points table and the prompt
# ---------------------------------------------------------------------------


def test_the_points_table_is_the_pre_registered_one_value_for_value():
    assert rj.POINTS == {
        "kernel_correct": 2.0, "class_correct": 1.0, "location_exact": 4.0, "location_near": 2.0,
        "kernel_wrong": -1.0, "clean_correct": 7.0, "false_alarm": -2.0, "missed_bug": -1.0,
        "no_answer": -1.0, "malformed": -1.0, "cost_run_native": -0.5, "cost_run_sim_kernel": -0.1,
        "cost_run_sim_all": -0.6, "cost_waitcheck": -0.05, "cost_consan": -1.0, "cost_disasm": -0.02,
    }
    assert rj.MAX_TURNS == 8 and rj.MAX_SCORE == 7.0 and rj.NEAR_INSTRUCTIONS == 2


@pytest.mark.skipif(not (PROTOTYPE / "scripts" / "rjenv.py").is_file(),
                    reason="prototype absent (set AORTA_RJ_TEST_PROTOTYPE)")
def test_the_points_table_matches_the_prototypes_file():
    sys.path.insert(0, str(PROTOTYPE / "scripts"))
    try:
        import rjenv
    finally:
        sys.path.remove(str(PROTOTYPE / "scripts"))
    assert rj.POINTS == rjenv.POINTS and rj.MAX_TURNS == rjenv.MAX_STEPS


def test_the_system_prompt_names_every_tool_its_cost_and_the_answer_format():
    text = rj.system_prompt(["k_a", "k_b"])
    for tool in rj.TOOLS:
        assert f'"tool": "{tool}"' in text
    for name in ("cost_waitcheck", "cost_run_sim_kernel", "cost_run_sim_all", "cost_run_native",
                 "cost_consan", "cost_disasm"):
        assert f"cost {-rj.POINTS[name]:g}" in text
    assert '"answer"' in text and '"class": "none"' in text and "k_a, k_b" in text
    assert f"after {rj.MAX_TURNS} turns" in text


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

KERNELS = ["k_a", "k_b"]


@pytest.mark.parametrize("raw, tool, kernel", [
    ('{"tool": "waitcheck"}', "waitcheck", ""),
    ('{"tool": "run_sim", "kernel": "all"}', "run_sim", "all"),
    ('{"tool": "run_sim", "kernel": "k_b"}', "run_sim", "k_b"),
    ('{"tool": "run_native"}', "run_native", "all"),
    ('{"tool": "consan", "kernel": "k_a"}', "consan", "k_a"),
])
def test_a_tool_call_parses(raw, tool, kernel):
    a = rj.parse_action(raw, KERNELS)
    assert (a.kind, a.tool, a.kernel, a.strict_json) == ("tool", tool, kernel, True)


def test_disasm_takes_a_hex_or_decimal_range_and_ignores_garbage():
    a = rj.parse_action('{"tool": "disasm", "kernel": "k_a", "start": "0x0010", "end": 40}', KERNELS)
    assert (a.start, a.end) == (16, 40)
    b = rj.parse_action('{"tool": "disasm", "kernel": "k_a", "start": "here"}', KERNELS)
    assert b.kind == "tool" and b.start is None


@pytest.mark.parametrize("offset, want", [("0x0bc", 188), ("0XBC", 188), (188, 188), ("188", 188),
                                          (188.0, 188), ("bc", None), (True, None), (None, None),
                                          ("0x", None), (-4, -4)])
def test_an_answer_offset_is_an_integer_or_hex_or_decimal_text(offset, want):
    a = rj.parse_action(json.dumps({"answer": {"class": "waitcnt", "kernel": "k_a", "offset": offset}}),
                        KERNELS)
    assert a.kind == "answer" and a.answer["offset"] == want


@pytest.mark.parametrize("raw, fragment", [
    ("", "empty"),
    ("I think the bug is in k_a.", "no JSON object"),
    ("[1, 2]", "no JSON object"),
    ('{"tool": "profile"}', "unknown tool"),
    ('{"kernel": "k_a"}', "unknown tool"),
    ('{"tool": "consan"}', "consan needs one kernel"),
    ('{"tool": "consan", "kernel": "all"}', "consan needs one kernel"),
    ('{"tool": "disasm", "kernel": "k_z"}', "disasm needs one kernel"),
    ('{"tool": "run_sim", "kernel": "k_z"}', "unknown kernel"),
    ('{"answer": "none"}', "must be an object"),
    ('{"answer": {"class": "race", "kernel": "k_a", "offset": 8}}', "answer class"),
    ('{"answer": {"kernel": "k_a", "offset": 8}}', "answer class"),
    ("<think>the answer is {\"answer\": {\"class\": \"none\"}}", "never closed"),
])
def test_a_malformed_reply_is_named_not_raised(raw, fragment):
    a = rj.parse_action(raw, KERNELS)
    assert a.kind == "malformed" and fragment in a.error


def test_the_shipped_unwrapping_runs_first_so_a_fence_or_reasoning_still_parses_strictly():
    for raw in ('```json\n{"tool": "waitcheck"}\n```', '<think>hmm</think>\n{"tool": "waitcheck"}'):
        a = rj.parse_action(raw, KERNELS)
        assert a.kind == "tool" and a.strict_json


def test_prose_around_the_object_parses_but_is_recorded_as_not_strict():
    a = rj.parse_action('Let me check first. {"tool": "waitcheck"} then I will answer.', KERNELS)
    assert a.kind == "tool" and a.parsed_json and not a.strict_json


def test_the_first_object_wins_and_an_answer_beats_a_tool_key():
    a = rj.parse_action('{"tool": "waitcheck"} {"answer": {"class": "none"}}', KERNELS)
    assert a.tool == "waitcheck"
    b = rj.parse_action('{"tool": "waitcheck", "answer": {"class": "none"}}', KERNELS)
    assert b.kind == "answer"


def test_a_class_is_matched_without_case_or_padding():
    a = rj.parse_action('{"answer": {"class": " Barrier ", "kernel": "k_b", "offset": 4}}', KERNELS)
    assert a.answer["class"] == "barrier"


# ---------------------------------------------------------------------------
# rewards
# ---------------------------------------------------------------------------


def test_the_oracle_answer_earns_kernel_class_and_exact(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), ANSWER)
    assert events(ep) == ["kernel_correct", "class_correct", "location_exact"]
    assert ep.score().total == 7.0 and ep.solved and ep.done and ep.outcome == "exact"


@pytest.mark.parametrize("offset", [16, 20])
def test_a_moved_instruction_is_exact_where_it_was_and_where_it_is(corpus, offset):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x10.waitcnt_late")),
              {"answer": {"class": "waitcnt", "kernel": "k_a", "offset": offset}})
    assert ep.score().total == 7.0


@pytest.mark.parametrize("offset, total, outcome", [
    (12, 5.0, "near"),  # one instruction away
    (0, 5.0, "near"),  # two away
    (24, 3.0, "kernel_only"),  # four away
    (9, 5.0, "near"),  # not an instruction boundary: the nearest instruction counts
])
def test_near_is_within_two_instructions_and_exclusive_with_exact(corpus, offset, total, outcome):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")),
              {"answer": {"class": "waitcnt", "kernel": "k_a", "offset": offset}})
    assert ep.score().total == total and ep.outcome == outcome
    assert not ep.solved


def test_the_wrong_class_keeps_kernel_and_location(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")),
              {"answer": {"class": "barrier", "kernel": "k_a", "offset": 8}})
    assert events(ep) == ["kernel_correct", "location_exact"] and ep.score().total == 6.0
    assert not ep.solved and ep.outcome == "exact"


def test_the_wrong_kernel_is_minus_one_and_nothing_else(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")),
              {"answer": {"class": "waitcnt", "kernel": "k_b", "offset": 8}})
    assert events(ep) == ["kernel_wrong"] and ep.score().total == -1.0


def test_an_answer_without_a_usable_offset_earns_kernel_and_class_only(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")),
              {"answer": {"class": "waitcnt", "kernel": "k_a", "offset": "somewhere"}})
    assert ep.score().total == 3.0 and ep.outcome == "kernel_only"


def test_none_is_a_missed_bug_on_a_fault_and_full_credit_on_a_clean_control(corpus):
    missed = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"answer": {"class": "none"}})
    assert events(missed) == ["missed_bug"] and missed.score().total == -1.0
    clean = play(rj.Episode(scenario(corpus, "CLEAN:train:0")), {"answer": {"class": "none"}})
    assert clean.score().total == 7.0 and clean.solved


def test_claiming_a_bug_on_a_clean_control_is_a_false_alarm(corpus):
    ep = play(rj.Episode(scenario(corpus, "CLEAN:train:0")), ANSWER)
    assert events(ep) == ["false_alarm"] and ep.score().total == -2.0 and not ep.solved


@pytest.mark.parametrize("action, cost", [
    ({"tool": "waitcheck"}, 0.05),
    ({"tool": "run_sim", "kernel": "k_a"}, 0.1),
    ({"tool": "run_sim", "kernel": "all"}, 0.6),
    ({"tool": "run_sim"}, 0.6),
    ({"tool": "run_native", "kernel": "k_a"}, 0.5),
    ({"tool": "run_native", "kernel": "all"}, 0.5),
    ({"tool": "consan", "kernel": "k_a"}, 1.0),
    ({"tool": "disasm", "kernel": "k_a"}, 0.02),
])
def test_each_tool_charges_its_price_once_per_call(corpus, action, cost):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), action, action, ANSWER)
    assert ep.score().total == pytest.approx(7.0 - 2 * cost)
    assert ep.tool_cost() == pytest.approx(2 * cost)


def test_a_malformed_reply_costs_one_and_uses_a_turn_but_the_episode_goes_on(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), "not json", '{"tool": "nope"}')
    assert events(ep) == ["malformed", "malformed"] and not ep.done and ep.turns_used == 2
    play(ep, ANSWER)
    assert ep.score().total == 5.0


def test_an_invalid_answer_is_malformed_not_an_answer(corpus):
    ep = play(rj.Episode(scenario(corpus, "CLEAN:train:0")), '{"answer": {"class": "race"}}')
    assert events(ep) == ["malformed"] and not ep.done and ep.answer is None


# ---------------------------------------------------------------------------
# the turn budget
# ---------------------------------------------------------------------------


def test_eight_turns_without_an_answer_end_the_episode_as_no_answer(corpus):
    ep = rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken"))
    for _ in range(rj.MAX_TURNS):
        assert ep.observe() is not None
        play(ep, {"tool": "waitcheck"})
    assert ep.done and ep.outcome == "no_answer" and ep.observe() is None
    assert events(ep).count("no_answer") == 1
    assert ep.score().total == pytest.approx(-1.0 - 8 * 0.05)


def test_an_answer_on_the_last_turn_counts_and_is_not_also_no_answer(corpus):
    ep = rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken"))
    play(ep, *[{"tool": "disasm", "kernel": "k_b"}] * (rj.MAX_TURNS - 1))
    assert "LAST turn" in ep.observe()
    play(ep, ANSWER)
    assert "no_answer" not in events(ep) and ep.score().total == pytest.approx(7.0 - 7 * 0.02)


def test_seven_malformed_replies_then_a_malformed_eighth_is_no_answer(corpus):
    ep = play(rj.Episode(scenario(corpus, "CLEAN:train:0")), *["x"] * rj.MAX_TURNS)
    assert ep.done and ep.score().total == pytest.approx(-9.0)


def test_nothing_after_the_end_is_scored(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), ANSWER, {"tool": "consan", "kernel": "k_a"},
              {"answer": {"class": "none"}})
    assert ep.score().total == 7.0 and ep.turns_used == 1


def test_a_shorter_budget_is_honoured_and_the_prompt_counts_down(corpus):
    ep = rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken"), budget=3)
    first = ep.observe()
    assert "Turn 1 of 3" in first and "2 turn(s) left" in first
    play(ep, {"tool": "waitcheck"}, {"tool": "waitcheck"})
    assert "Turn 3 of 3" in ep.observe() and "LAST turn" in ep.observe()
    play(ep, {"tool": "waitcheck"})
    assert ep.done and ep.outcome == "no_answer"


def test_an_episode_refuses_a_budget_of_zero(corpus):
    with pytest.raises(ValueError):
        rj.Episode(scenario(corpus, "CLEAN:train:0"), budget=0)


# ---------------------------------------------------------------------------
# what the policy is shown
# ---------------------------------------------------------------------------


def test_disasm_shows_the_faulty_listing_and_leaves_other_kernels_alone(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"tool": "disasm", "kernel": "k_a"})
    assert (8, "s_waitcnt lgkmcnt(1)") in ep.last_result["lines"]
    assert "lgkmcnt(0)" not in dict(ep.last_result["lines"])[8]
    late = play(rj.Episode(scenario(corpus, "k_a+0x10.waitcnt_late")), {"tool": "disasm", "kernel": "k_a"})
    lines = dict(late.last_result["lines"])
    assert lines[16] == "v_mul_lo_u32 v3, v2, v2" and lines[20] == "s_waitcnt vmcnt(0)"
    other = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"tool": "disasm", "kernel": "k_b"})
    assert [t for _, t in other.last_result["lines"]] == LISTINGS["k_b"]


def test_every_offset_is_shown_in_hex(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"tool": "run_sim", "kernel": "k_a"},
              {"tool": "waitcheck"})
    text = ep.observe()
    assert "0x0004 ds_read_b32" in text and "0x000c" in text and "producer 0x0004" in text
    assert ": 12 " not in text and "offset 12" not in text


@pytest.mark.parametrize("sid", list(MUTANTS) + ["CLEAN:train:0"])
def test_no_prompt_names_the_scenario_its_operator_or_its_mutant(corpus, sid):
    s = scenario(corpus, sid)
    ep = rj.Episode(s, budget=20)
    actions = [{"tool": "waitcheck"}, {"tool": "run_sim", "kernel": "all"}, {"tool": "run_native", "kernel": "all"}]
    actions += [{"tool": t, "kernel": k} for t in ("consan", "disasm") for k in KERNELS]
    prompts = [rj.system_prompt(KERNELS)]
    for a in actions:
        prompts.append(ep.observe())
        play(ep, a)
    prompts.append(ep.observe())
    text = "\n".join(prompts)
    for secret in (sid, "weaken", "_late", "barrier_late", "waitcnt_late", "mutant", ".hsaco", SECRET_NOTE,
                   "after ds_read", "after v_mul"):
        assert secret not in text, secret


def test_the_prompt_records_the_previous_turns_in_order(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"tool": "waitcheck"}, "oops",
              {"tool": "run_sim", "kernel": "k_a"})
    text = ep.observe()
    assert text.index("[turn 1]") < text.index("[turn 2] invalid reply") < text.index("[turn 3]")
    assert "your reply was: oops" in text and "Turn 4 of 8" in text
    assert "Tool cost so far: 0.15" in text


def test_a_listing_is_capped_per_call_and_says_so(corpus, monkeypatch):
    monkeypatch.setattr(rj, "DISASM_MAX_LINES", 3)
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"tool": "disasm", "kernel": "k_a"})
    assert len(ep.last_result["lines"]) == 3 and ep.last_result["omitted"] == 5
    assert "5 more instruction(s)" in ep.observe()
    window = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")),
                  {"tool": "disasm", "kernel": "k_a", "start": "0x0018", "end": "0x001c"})
    assert [o for o, _ in window.last_result["lines"]] == [24, 28]


def test_the_history_is_elided_oldest_listing_first_past_the_cap(corpus, monkeypatch):
    monkeypatch.setattr(rj, "MAX_HISTORY_CHARS", 700)
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"tool": "disasm", "kernel": "k_a"},
              {"tool": "run_sim", "kernel": "k_a"}, {"tool": "disasm", "kernel": "k_b"})
    text = ep.observe()
    first, second, third = text.split("[turn ")[1:4]
    assert rj.ELIDED in first, "the oldest listing goes first"
    assert "race VGPR" in second, "a non-listing result outlives a listing"
    assert "ds_write_b32" in third, "the newest listing is kept while it fits"


def test_native_and_consan_render_the_archived_verdicts(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_a+0x8.waitcnt_weaken")), {"tool": "run_native", "kernel": "k_a"})
    assert "k_a: FAIL (12 wrong output elements), output differs" in ep.observe()
    ep = play(rj.Episode(scenario(corpus, "k_b+0x4.barrier_late")), {"tool": "consan", "kernel": "k_b"})
    assert "0x0000 <-> 0x0008" in ep.observe()


def test_run_sim_all_folds_the_quiet_kernels_into_one_line(corpus):
    ep = play(rj.Episode(scenario(corpus, "k_b+0x4.barrier_late")), {"tool": "run_sim", "kernel": "all"})
    text = ep.observe()
    assert "no findings, simulated output PASS: k_a" in text and "race LDS read" in text


def test_native_runs_are_a_per_episode_stream_that_replays():
    assert rj.episode_seed("s", 3) == rj.episode_seed("s", 3) != rj.episode_seed("s", 4)


# ---------------------------------------------------------------------------
# the lockstep rollout
# ---------------------------------------------------------------------------


def _advantages(values):
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
    return [(v - mean) / (sd + 1e-4) for v in values], mean, sd


def test_one_advantage_per_episode_repeated_on_every_step(corpus):
    s = scenario(corpus, "k_a+0x8.waitcnt_weaken")
    seen = []

    def gen(prompts):
        seen.append(list(prompts))
        step = len(seen)
        # episode 0 answers at once; episode 1 checks first; episode 2 never answers.
        replies = {1: [json.dumps(ANSWER), '{"tool": "waitcheck"}', '{"tool": "waitcheck"}'],
                   2: [json.dumps(ANSWER), '{"tool": "waitcheck"}']}
        return replies.get(step, ['{"tool": "waitcheck"}'] * len(prompts))

    group, samples, wire = rj.rollout_scenario(s, 3, rj.TurnBudget(4), gen, advantage_fn=_advantages)
    rewards = [7.0, 7.0 - 0.05, -1.0 - 4 * 0.05]
    assert group["rewards"] == [round(r, 4) for r in rewards]
    adv, _, _ = _advantages(rewards)
    by_episode: dict[int, set[float]] = {}
    for x in samples:
        by_episode.setdefault(x.episode, set()).add(x.advantage)
    assert by_episode == {i: {adv[i]} for i in range(3)}
    assert [x.step for x in samples if x.episode == 2] == [1, 2, 3, 4]
    assert len(samples) == len(wire) == group["steps_total"] == 1 + 2 + 4
    assert [len(p) for p in seen] == [3, 2, 1, 1]
    assert group["solved_rate"] == pytest.approx(2 / 3, abs=1e-4) and group["max_depth"] == 4
    assert group["advantage_spread"] > 0 and group["terminals"] == {"exact": 2, "no_answer": 1}


def test_each_sample_carries_the_prompt_its_reply_answered(corpus):
    s = scenario(corpus, "k_a+0x8.waitcnt_weaken")
    asked = []

    def gen(prompts):
        asked.extend(prompts)
        return ['{"tool": "waitcheck"}' if "Turn 1" in p else json.dumps(ANSWER) for p in prompts]

    _, samples, _ = rj.rollout_scenario(s, 2, rj.TurnBudget(), gen, advantage_fn=_advantages)
    assert [x.prompt for x in samples] == [asked[0], asked[2], asked[1], asked[3]]


def test_a_generator_returning_the_wrong_count_is_an_error(corpus):
    with pytest.raises(ValueError, match="1 completion"):
        rj.rollout_scenario(scenario(corpus, "CLEAN:train:0"), 2, rj.TurnBudget(), lambda p: ["{}"],
                            advantage_fn=_advantages)


def test_a_keyed_generator_gets_episode_and_step(corpus):
    keys = []

    def gen(prompts, k):
        keys.append(list(k))
        return [json.dumps({"answer": {"class": "none"}})] * len(prompts)

    rj.rollout_scenario(scenario(corpus, "CLEAN:train:0"), 3, rj.TurnBudget(), gen, advantage_fn=_advantages,
                        keyed=True)
    assert keys == [[(0, 1), (1, 1), (2, 1)]]


def test_iteration_metrics_report_entrenchment(corpus):
    s = scenario(corpus, "k_a+0x8.waitcnt_weaken")
    g1, _, _ = rj.rollout_scenario(s, 4, rj.TurnBudget(), lambda p: [json.dumps(ANSWER)] * len(p),
                                   advantage_fn=_advantages)
    m = rj.iteration_metrics([g1])
    assert m["sequence_top_share"] == 1.0 and m["sequence_entropy_norm"] == 0.0 and m["groups_flat"] == 1
    assert m["solved_rate"] == 1.0 and m["first_tool"] == {"answer": 4}
    mixed = rj.sequence_stats({"a": 1, "b": 1, "c": 1, "d": 1})
    assert mixed["entropy_norm"] == pytest.approx(1.0) and mixed["top_share"] == 0.25


def test_episode_logs_are_written_for_the_first_episodes_only(corpus, tmp_path):
    rj.rollout_scenario(scenario(corpus, "CLEAN:train:0"), 3, rj.TurnBudget(),
                        lambda p: [json.dumps({"answer": {"class": "none"}})] * len(p),
                        advantage_fn=_advantages, log_root=tmp_path / "logs", log_first=2)
    assert sorted(p.name for p in (tmp_path / "logs" / "CLEAN:train:0").iterdir()) == ["ep000.jsonl", "ep001.jsonl"]


# ---------------------------------------------------------------------------
# the corpus and its split
# ---------------------------------------------------------------------------


def test_the_split_is_by_kernel_and_keeps_only_trace_free_usable_faults(corpus):
    spec = json.loads((corpus / "split.json").read_text())
    train = rj.load_corpus(corpus, split="train")
    test = rj.load_corpus(corpus, split="test")
    assert {s.kernel for s in train if not s.clean} == {"k_a"}
    assert {s.kernel for s in test if not s.clean} == {"k_b"}
    ids = {s.scenario_id for s in train + test}
    assert "k_a+0x14.waitcnt_drop" not in ids, "a drop has a listing trace"
    assert "k_a+0x0.waitcnt_weaken" not in ids, "a silent mutant has no evidence"
    assert spec["refused"] == {"op waitcnt_drop not in --ops": 1, "class silent": 1}
    assert sum(s.clean for s in train) == 1 and sum(s.clean for s in test) == 1


def test_an_unknown_scenario_or_split_is_refused(corpus):
    with pytest.raises(SystemExit):
        rj.load_corpus(corpus, split="train", only=["k_b+0x4.barrier_late"])
    with pytest.raises(SystemExit):
        rj.load_corpus(corpus, split="validation")


def test_a_split_file_from_the_source_is_honoured(tmp_path):
    root = build_corpus(tmp_path)
    split = tmp_path / "given.json"
    split.write_text(json.dumps({"train": ["k_b+0x4.barrier_late"], "test": ["k_a+0x8.waitcnt_weaken",
                                                                             "k_a+0x10.waitcnt_late"]}))
    out = tmp_path / "c2"
    prep.main(["--source", str(root.parent / "src"), "--out", str(out), "--split-file", str(split)])
    assert [s.scenario_id for s in rj.load_corpus(out, split="train") if not s.clean] == ["k_b+0x4.barrier_late"]


def test_a_split_that_shares_a_fault_kernel_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    split = tmp_path / "given.json"
    split.write_text(json.dumps({"train": ["k_a+0x8.waitcnt_weaken"], "test": ["k_a+0x10.waitcnt_late"]}))
    with pytest.raises(SystemExit, match="share fault kernel"):
        prep.main(["--source", str(root.parent / "src"), "--out", str(tmp_path / "c3"), "--split-file", str(split)])


def test_trace_free_selects_exactly_the_ops_with_no_trace(tmp_path):
    root = build_corpus(tmp_path)
    rows = {r["site_id"]: r for r in json.loads((root.parent / "src" / "analysis.json").read_text())["rows"]}
    assert prep.select_ops("trace-free", rows) == ("barrier_late", "waitcnt_late", "waitcnt_weaken")
    assert prep.select_ops("waitcnt_drop", rows) == ("waitcnt_drop",)


def test_an_op_the_answer_classes_cannot_score_is_refused():
    with pytest.raises(SystemExit, match="answer classes"):
        prep.select_ops("waitcnt_late,memorder_swap", {})
    with pytest.raises(SystemExit, match="selects no operator"):
        prep.select_ops(",", {})


def test_a_split_with_no_test_faults_is_refused_unless_asked_for(tmp_path):
    root = build_corpus(tmp_path)
    src = str(root.parent / "src")
    with pytest.raises(SystemExit, match="both must be non-empty"):
        prep.main(["--source", src, "--out", str(tmp_path / "c4"), "--test-kernels", ""])
    prep.main(["--source", src, "--out", str(tmp_path / "c5"), "--test-kernels", "", "--allow-empty-test"])
    assert not [s for s in rj.load_corpus(tmp_path / "c5", split="test") if not s.clean]


def test_train_and_test_id_lists_are_honoured_and_must_come_together(tmp_path):
    root = build_corpus(tmp_path)
    src = str(root.parent / "src")
    (tmp_path / "train.txt").write_text("# train\nk_b+0x4.barrier_late\n")
    (tmp_path / "test.jsonl").write_text(json.dumps({"scenario_id": "k_a+0x8.waitcnt_weaken"}) + "\n"
                                         + json.dumps({"site_id": "k_a+0x10.waitcnt_late"}) + "\n")
    out = tmp_path / "c6"
    prep.main(["--source", src, "--out", str(out), "--train-ids", str(tmp_path / "train.txt"),
               "--test-ids", str(tmp_path / "test.jsonl")])
    assert [s.scenario_id for s in rj.load_corpus(out, split="train") if not s.clean] == ["k_b+0x4.barrier_late"]
    assert sorted(s.scenario_id for s in rj.load_corpus(out, split="test") if not s.clean) == [
        "k_a+0x10.waitcnt_late", "k_a+0x8.waitcnt_weaken"]
    with pytest.raises(SystemExit, match="go together"):
        prep.main(["--source", src, "--out", str(tmp_path / "c7"), "--train-ids", str(tmp_path / "train.txt")])


def test_the_system_prompt_lists_the_corpus_kernels(corpus):
    text = rj.system_for(rj.load_corpus(corpus, split="train"))
    assert "2 HIP kernels: k_a, k_b" in text


# ---------------------------------------------------------------------------
# scripted policies on the synthetic archive
# ---------------------------------------------------------------------------


def test_the_scripted_policies_play_through_the_same_parser(corpus):
    scen = rj.load_corpus(corpus, split=None)
    oracle = rb.play(rb.SCRIPTED["oracle"], scen)
    assert all(r["total"] == 7.0 for r in oracle)
    window = {r["scenario_id"]: r for r in rb.play(rb.SCRIPTED["evidence_window"], scen)}
    # waitcheck names consumer 0x0c, producer 0x04: the sync in [4, 12] is the weakened wait at 8.
    assert window["k_a+0x8.waitcnt_weaken"]["outcome"] == "exact"
    assert window["CLEAN:train:0"]["outcome"] == "clean_correct"
    const, value = rb.best_constant(scen)
    assert const["class"] in rj.CLASSES and value <= 7.0


# ---------------------------------------------------------------------------
# replay against the prototype's archive (skipped where it is absent)
# ---------------------------------------------------------------------------

ARCHIVE = PROTOTYPE / "analysis" / "results_clean.jsonl"
needs_archive = pytest.mark.skipif(not ARCHIVE.is_file(),
                                   reason="prototype archive absent (set AORTA_RJ_TEST_PROTOTYPE)")


@pytest.fixture(scope="module")
def full_corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("rj") / "all"
    prep.main(["--source", str(PROTOTYPE), "--out", str(out), "--test-kernels", "", "--allow-empty-test",
               "--ops", "waitcnt_weaken,waitcnt_late,barrier_late,waitcnt_drop,barrier_drop,barrier_early"])
    return rj.load_corpus(out, split="train")


@needs_archive
def test_the_tools_replay_the_prototypes_archived_observations(full_corpus):
    sys.path.insert(0, str(PROTOTYPE / "scripts"))
    try:
        import rjenv
    finally:
        sys.path.remove(str(PROTOTYPE / "scripts"))
    arch = rjenv.Archive(str(PROTOTYPE / "suite/sites/sites.json"), str(ARCHIVE),
                         (PROTOTYPE / "analysis/suite.dis").read_text())
    rng = random.Random(7)
    picked = rng.sample([s for s in full_corpus if not s.clean], 60) + [s for s in full_corpus if s.clean][:2]
    compared = 0
    for s in picked:
        theirs = rjenv.Episode(arch.scenario(s.scenario_id if not s.clean else "CLEAN:0"), arch)
        mine = rj.Tools(s, random.Random(0))
        for k in [s.kernel or "k_scan", "k_bitonic"]:
            ours = [{key: v for key, v in r.items() if key != "n_memory_wait"} for r in mine.run_sim(k)]
            assert ours == theirs._sim(k), s.scenario_id
            assert mine.consan(k) == theirs._consan(k), s.scenario_id
            want = "\n".join(f"{o:#06x}: {t}" for o, t in mine.listing(k))
            assert want == theirs._disasm(k), s.scenario_id
            compared += 3
        assert mine.waitcheck() == theirs._waitcheck(), s.scenario_id
        compared += 1
    assert compared == 62 * 7


@needs_archive
def test_the_scripted_scores_reproduce_the_prototypes_recorded_totals(full_corpus):
    proto = json.loads((PROTOTYPE / "analysis" / "baselines.json").read_text())
    assert len(full_corpus) == 462 and sum(s.clean for s in full_corpus) == 46
    for name in ("oracle", "waitcheck_only", "race_top_hit", "evidence_window"):
        theirs = {p["sid"]: p["total"] for p in proto["policies"][name]["per_scenario"]}
        mine = {r["scenario_id"]: r["total"] for r in rb.play(rb.SCRIPTED[name], full_corpus)}
        faulty = [sid for sid in mine if not sid.startswith("CLEAN:")]
        assert len(faulty) == 416
        assert [sid for sid in faulty if mine[sid] != pytest.approx(theirs[sid], abs=1e-9)] == [], name
        clean_mine = {v for sid, v in mine.items() if sid.startswith("CLEAN:")}
        clean_theirs = {v for sid, v in theirs.items() if sid.startswith("CLEAN:")}
        assert clean_mine == clean_theirs, name


@needs_archive
def test_no_prompt_on_the_real_archive_leaks_its_scenario(full_corpus):
    rng = random.Random(3)
    for s in rng.sample([x for x in full_corpus if not x.clean], 40):
        ep = rj.Episode(s, budget=20)
        prompts = []
        for a in ({"tool": "waitcheck"}, {"tool": "run_sim", "kernel": "all"}, {"tool": "run_native"},
                  {"tool": "consan", "kernel": s.kernel}, {"tool": "disasm", "kernel": s.kernel}):
            prompts.append(ep.observe())
            play(ep, a)
        text = "\n".join(prompts + [ep.observe()])
        for secret in (s.scenario_id, s.op, s.record["path"], s.record.get("note") or "\0", "mutant"):
            assert secret not in text, (s.scenario_id, secret)
