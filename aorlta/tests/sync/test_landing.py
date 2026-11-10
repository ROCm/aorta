"""C5 step 5: the landing check's verdicts, tau, and the probe plumbing (design Section 6.5)."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from aorlta.sync import landing as L

PREV = [[-1.0, -2.0, -3.0], [-0.5, -0.25, -4.0]]


def shifted(rows, d):
    return [[v + d for v in r] for r in rows]


def test_landed_when_the_engine_moved_and_matches_the_trainer():
    now = shifted(PREV, 0.01)
    r = L.landing_verdict(5, now, PREV, shifted(now, 0.002), tau=0.01)
    assert r.verdict == "LANDED" and r.ok and r.engine_changed
    assert r.mean_gap == pytest.approx(0.002)


def test_noop_when_nothing_changed_anywhere():
    r = L.landing_verdict(0, PREV, [list(p) for p in PREV], shifted(PREV, 0.003), tau=0.01)
    assert r.verdict == "NOOP" and r.ok and not r.engine_changed


def test_not_landed_when_the_merge_changed_but_the_engine_did_not():
    r = L.landing_verdict(7, [list(p) for p in PREV], PREV, PREV, tau=0.01)
    assert r.verdict == "NOT_LANDED" and not r.ok


def test_not_landed_takes_precedence_over_a_large_gap():
    r = L.landing_verdict(7, [list(p) for p in PREV], PREV, shifted(PREV, 1.0), tau=0.01)
    assert r.verdict == "NOT_LANDED"


def test_mismatch_when_the_gap_exceeds_tau():
    now = shifted(PREV, 0.5)
    r = L.landing_verdict(3, now, PREV, PREV, tau=0.01)
    assert r.verdict == "MISMATCH" and not r.ok and r.mean_gap == pytest.approx(0.5)


def test_one_bitwise_difference_is_enough_to_count_as_moved():
    now = [list(p) for p in PREV]
    now[1][2] = math.nextafter(now[1][2], 0.0)
    r = L.landing_verdict(1, now, PREV, now, tau=0.01)
    assert r.verdict == "LANDED" and r.engine_changed


def test_engine_change_under_a_zero_update_stops_the_run():
    r = L.landing_verdict(0, shifted(PREV, 1e-6), PREV, PREV, tau=0.01)
    assert r.verdict == "MISMATCH" and not r.ok


def test_a_zero_update_with_the_gap_above_tau_is_a_mismatch():
    r = L.landing_verdict(0, [list(p) for p in PREV], PREV, shifted(PREV, 0.5), tau=0.01)
    assert r.verdict == "MISMATCH" and not r.ok and not r.engine_changed and r.mean_gap == pytest.approx(0.5)


def _served(tiny_base, tmp_path, tiny_probes, engine):
    """The base's targets, written as the slot the engine serves, and a SyncState judging against them."""
    from safetensors import safe_open

    from aorlta.sync import writer as W

    with safe_open(str(tiny_base / "model.safetensors"), framework="pt") as f:
        weights = {k: f.get_tensor(k) for k in W.target_keys(f.keys())}
    slot = W.write_slot(weights, base_dir=tiny_base, sync_dir=tmp_path, run_id="r", version=0).path
    prev = L.engine_logprobs(engine, tiny_probes)
    state = L.SyncState(run_id="r", sync_dir=str(tmp_path), base_dir=str(tiny_base), probes=tiny_probes,
                        tau={0: 0.01}, previous={0: prev}, in_flight=lambda: 0,
                        expected=W.expected_count(W.read_base_index(tiny_base), partial=True))
    return weights, slot, prev, state


def test_publish_stops_on_a_zero_update_the_trainer_does_not_match(tiny_base, tmp_path, tiny_probes,
                                                                   engine_factory):
    e = engine_factory(0)
    weights, served, prev, state = _served(tiny_base, tmp_path, tiny_probes, e)
    with pytest.raises(L.LandingFailed) as exc:
        L.publish(weights, 1, 0, engines=[e], state=state, trainer_side=lambda probes: shifted(prev, 0.5))
    pub = exc.value.publication
    assert pub.verdicts() == {0: "MISMATCH"} and not pub.results[0].engine_changed
    assert pub.pruned == [] and served.is_dir() and state.previous == {0: prev}


def test_publish_accepts_a_zero_update_within_tau(tiny_base, tmp_path, tiny_probes, engine_factory):
    e = engine_factory(0)
    weights, served, prev, state = _served(tiny_base, tmp_path, tiny_probes, e)
    pub = L.publish(weights, 1, 0, engines=[e], state=state, trainer_side=lambda probes: shifted(prev, 0.003))
    assert pub.verdicts() == {0: "NOOP"} and pub.pruned == [str(served)]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_log_probs_are_a_mismatch(bad):
    now = shifted(PREV, 0.01)
    now[0][1] = bad
    assert L.landing_verdict(2, now, PREV, now, tau=0.01).verdict == "MISMATCH"
    assert L.landing_verdict(0, PREV, PREV, now, tau=0.01).verdict == "MISMATCH"


def test_shapes_must_agree():
    with pytest.raises(L.LandingError):
        L.landing_verdict(1, PREV, PREV[:1], PREV, tau=0.01)
    with pytest.raises(L.LandingError):
        L.landing_verdict(1, PREV, PREV, [p[:2] for p in PREV], tau=0.01)
    with pytest.raises(L.LandingError):
        L.landing_verdict(-1, PREV, PREV, PREV, tau=0.01)


def test_tolerance_fallback_when_prefill_is_not_bitwise_repeatable():
    noisy = shifted(PREV, 1e-7)
    assert L.landing_verdict(4, noisy, PREV, noisy, tau=0.01, bitwise=False, repeat_eps=1e-6).verdict == "NOT_LANDED"
    moved = shifted(PREV, 1e-3)
    assert L.landing_verdict(4, moved, PREV, moved, tau=0.01, bitwise=False, repeat_eps=1e-6).verdict == "LANDED"


def test_tau_is_three_times_the_median_probe_mean_with_a_floor():
    engine = [[0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4]
    trainer = [[0.1, 0.1, 0.1, 0.1], [0.02] * 4, [0.04] * 4, [0.0, 0.0, 0.0, 4.0]]  # probe means .1 .02 .04 1.0
    c = L.calibrate_tau(engine, trainer, tau_floor=0.01)
    assert c.per_probe_gap == pytest.approx((0.1, 0.02, 0.04, 1.0))
    assert c.median_gap == pytest.approx(0.07) and c.tau == pytest.approx(0.21)
    assert L.calibrate_tau(engine, [[0.001] * 4] * 4, tau_floor=0.01).tau == 0.01


def test_v0_passes_its_own_calibration_when_probes_are_alike():
    import random

    rnd = random.Random(0)
    engine = [[rnd.uniform(-5, 0) for _ in range(500)] for _ in range(4)]
    # heavy tails: most tokens agree to 1e-3, one in fifty is off by several nats
    trainer = [[v + (rnd.choice([3.0, -2.5]) if rnd.random() < 0.02 else rnd.gauss(0, 1e-3)) for v in p] for p in engine]
    c = L.calibrate_tau(engine, trainer, tau_floor=0.01)
    assert c.pooled_median_token_gap < 0.01 < c.median_gap  # a token-median rule would fail v0 here
    r = L.landing_verdict(1, engine, shifted(engine, 0.5), trainer, tau=c.tau)
    assert r.verdict == "LANDED"


def _reply(ids, lps, cached=0):
    return {"output_ids": [1], "meta_info": {"cached_tokens": cached, "input_token_logprobs":
                                             [(None, ids[0], None)] + [(v, t, None) for v, t in zip(lps, ids[1:])]}}


def test_parse_input_logprobs_checks_alignment_and_freshness():
    ids = [5, 6, 7, 8]
    assert L.parse_input_logprobs(_reply(ids, [-1.0, -2.0, -3.0]), ids) == [-1.0, -2.0, -3.0]
    with pytest.raises(L.LandingError, match="fresh"):
        L.parse_input_logprobs(_reply(ids, [-1.0, -2.0, -3.0], cached=3), ids)
    with pytest.raises(L.LandingError):
        L.parse_input_logprobs(_reply(ids, [-1.0, -2.0]), ids)
    bad = _reply(ids, [-1.0, -2.0, -3.0])
    bad["meta_info"]["input_token_logprobs"][2] = (-2.0, 99, None)
    with pytest.raises(L.LandingError, match="token"):
        L.parse_input_logprobs(bad, ids)
    bad = _reply(ids, [-1.0, None, -3.0])
    with pytest.raises(L.LandingError):
        L.parse_input_logprobs(bad, ids)
    with pytest.raises(L.LandingError):
        L.parse_input_logprobs({"output_ids": []}, ids)


def test_score_request_is_a_fresh_prefill_with_prompt_logprobs():
    req = L.score_request([1, 2, 3])
    assert req["logprob_start_len"] == 0 and req["return_logprob"] is True
    assert req["sampling_params"]["max_new_tokens"] == 1 and req["sampling_params"]["temperature"] == 0.0


def test_repeatability():
    assert L.repeatability([PREV, PREV, [list(p) for p in PREV]])["bitwise"]
    r = L.repeatability([PREV, shifted(PREV, 1e-7)])
    assert not r["bitwise"] and r["max_abs_delta"] == pytest.approx(1e-7)


class CharTokenizer:
    """One token per character, and a chat template that frames each message."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs) + "<assistant>"


@pytest.mark.parametrize("text", ["x" * 200, "x" * 80, "x" * 100 + "\n" + "Q" * 20],
                         ids=["no_reply_left", "6_reply_tokens_left", "first_line_too_long"])
def test_build_probes_refuses_a_prompt_that_leaves_fewer_than_reply_min_reply_tokens(text):
    # the template adds 42 characters: a 200-character line leaves no reply token, an 80-character one 6
    with pytest.raises(L.LandingError, match="reply_min"):
        L.build_probes(CharTokenizer(), [text], system="s", tokens=128, reply="Z" * 40, reply_min=16)


def test_build_probes_cuts_a_long_prompt_at_a_line_boundary():
    (probe,) = L.build_probes(CharTokenizer(), ["x" * 60 + "\n" + "Q" * 60], system="s", tokens=128,
                              reply="Z" * 40, reply_min=16).ids
    assert len(probe) == 128 and probe.count(ord("Z")) == 26 and ord("Q") not in probe


HF_HOME = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
QWEN = HF_HOME / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


@pytest.mark.skipif(not (QWEN / "tokenizer.json").exists(), reason="Qwen3.5-4B tokenizer not in the HF cache (HF_HOME)")
def test_build_probes_gives_fixed_length_distinct_probes():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN))
    texts = [f"Fix kernel_{n}:\n" + "\n".join(f"0x{i:04x}: v_add_f32 v{i % 7}, v1, v2" for i in range(n))
             for n in (20, 400, 60, 5)]
    probes = L.build_probes(tok, texts, system="You are an expert AMD GPU kernel engineer.", tokens=1024)
    assert len(probes.ids) == 4 and all(len(p) == 1024 for p in probes.ids)
    assert len(set(probes.ids)) == 4
    for p in probes.ids:
        assert p.count(248045) >= 3  # <|im_start|> system, user, assistant
    again = L.build_probes(tok, texts, system="You are an expert AMD GPU kernel engineer.", tokens=1024)
    assert again.sha256 == probes.sha256
