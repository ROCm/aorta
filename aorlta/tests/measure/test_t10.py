"""T10's measurement (aorlta.measure.gap, aorlta.measure.t10) on CPU.

The unit tests pin what the GPU numbers rest on: the trainer-side scores are `raw_logprobs_at`'s arithmetic bitwise,
every turn's reply is scored at its own positions, fix 1 really reaches the kernel's inputs and is undone, and the
statistics mean what the report says. The end-to-end test runs `run_t10` over the integrated components with the
wiring test's stand-ins (a tiny Qwen3.5 with the real vocabulary, twin engines that reload slots as TokenSpeed does).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ORCH = Path(__file__).resolve().parents[1] / "orchestrator"
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

import test_wiring_e2e as E  # noqa: E402

from aorlta.config import config_from_dict  # noqa: E402
from aorlta.measure import gap as G  # noqa: E402
from aorlta.measure import t10 as T  # noqa: E402
from aorlta.orchestrator import wiring  # noqa: E402
from aorlta.rollout.compile_check import FakeCompiler  # noqa: E402
from aorlta.rollout.engine_proc import InProcessEngine  # noqa: E402
from aorlta.schemas import ChainRecord, TrainingSample, TurnRecord  # noqa: E402
from aorlta.trainer.batching import raw_logprobs_at  # noqa: E402
from aorlta.trainer.loss import TrainerInputError, sample_gap  # noqa: E402

tiny_base = E.tiny_base  # module-scoped fixtures of the wiring test, reused
dataset = E.dataset


@pytest.fixture(scope="module")
def tiny(tiny_base):
    return E.load_tiny(tiny_base)


def _ids(n, seed=0, lo=10000, hi=10255):
    g = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(lo, hi, (n,), generator=g)]


# --- scoring -------------------------------------------------------------------------------------------


def test_score_positions_is_raw_logprobs_at_bitwise(tiny):
    ids = _ids(40)
    pos = list(range(5, 40))
    got = G.score_positions(tiny, ids, pos, device="cpu", temperature=0.7, chunk_tokens=8)
    want = raw_logprobs_at(tiny, ids, pos, device="cpu", chunk_tokens=8)
    assert torch.equal(got["raw"], want.float())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        z = tiny(input_ids=torch.tensor([ids])).logits[0].float()
    ls_t = torch.log_softmax(z / 0.7, dim=-1)[torch.tensor(pos) - 1].gather(-1, torch.tensor(ids)[pos][:, None])[:, 0]
    assert torch.allclose(got["temp"], ls_t, atol=1e-4)  # the loss temperature, not the raw one
    assert not torch.allclose(got["temp"], got["raw"], atol=1e-3)
    assert (got["entropy"] >= 0).all() and got["entropy"].shape == got["raw"].shape


def test_score_positions_refuses_positions_outside_the_sequence(tiny):
    with pytest.raises(TrainerInputError):
        G.score_positions(tiny, _ids(10), [0, 3], device="cpu", temperature=0.7)
    with pytest.raises(TrainerInputError):
        G.score_positions(tiny, _ids(10), [10], device="cpu", temperature=0.7)


def test_fp32_linear_attention_feeds_the_kernel_fp32_and_is_undone(tiny):
    layers = G.linear_attention_layers(tiny)
    assert len(layers) == 3
    seen = []
    originals = [m.chunk_gated_delta_rule for m in layers]

    def spy(fn):
        def inner(query, key, value, g=None, beta=None, **kw):
            seen.append((query.dtype, key.dtype, value.dtype, g.dtype, beta.dtype))
            o, s = fn(query, key, value, g=g, beta=beta, **kw)
            seen.append(("out", o.dtype))
            return o, s
        return inner

    for m in layers:
        m.chunk_gated_delta_rule = spy(m.chunk_gated_delta_rule)
    spied = [m.chunk_gated_delta_rule for m in layers]
    try:
        ids = _ids(24, seed=3)
        G.score_positions(tiny, ids, list(range(1, 24)), device="cpu", temperature=0.7)
        bf, f32 = torch.bfloat16, torch.float32
        base_inputs = [s for s in seen if s[0] != "out"]
        assert len(base_inputs) == 3 and all(s == (bf, bf, bf, f32, bf) for s in base_inputs)  # g is fp32 already
        seen.clear()
        with G.fp32_linear_attention(tiny) as n:
            assert n == 3 and all(getattr(m.chunk_gated_delta_rule, "aorlta_fp32", False) for m in layers)
            G.score_positions(tiny, ids, list(range(1, 24)), device="cpu", temperature=0.7)
        inputs = [s for s in seen if s[0] != "out"]
        assert len(inputs) == 3 and all(s == (torch.float32,) * 5 for s in inputs)
        assert [m.chunk_gated_delta_rule for m in layers] == spied  # restored
    finally:
        for m, fn in zip(layers, originals):
            m.chunk_gated_delta_rule = fn


def test_fp32_linear_attention_refuses_a_model_without_those_layers():
    with pytest.raises(TrainerInputError):
        with G.fp32_linear_attention(torch.nn.Linear(2, 2)):
            pass


# --- items and arrays -----------------------------------------------------------------------------------


def _turn(t, input_len, reply, lps, cached=0):
    return TurnRecord(turn=t, seed=t, input_ids_sha256="x", input_len=input_len, response_ids=list(reply),
                      response_logprobs=list(lps), finish_reason={"type": "stop", "matched": 248046},
                      cached_tokens=cached, ideal_cached_tokens=0, t_submit=0.0, t_done=1.0, compile=None,
                      feedback_sha256=None, leak_check="not_run")


def _two_turn_chain():
    p0 = _ids(6, seed=1)
    r0 = _ids(4, seed=2) + [248046]
    fb = _ids(3, seed=4)
    p1 = p0 + r0 + fb
    r1 = _ids(5, seed=5) + [248046]
    cid = "t10test/1/standin:a/0"
    rec = ChainRecord(chain_id=cid, prompt_id="standin:a", k=0, engine_id=0, policy_version="t10test:v0",
                      turns=[_turn(0, len(p0), r0, [-0.1] * len(r0)), _turn(1, len(p1), r1, [-0.2] * len(r1), 9)],
                      final_turn=1, end_reason="compiled")
    sample = TrainingSample(chain_id=cid, iteration=1, prompt_id="standin:a", prompt_ids=p1, response_ids=r1,
                            response_logprobs=[-0.2] * len(r1))
    return rec, sample, p1 + r1


def test_chain_items_locate_every_turn_and_token_arrays_line_up():
    rec, sample, full = _two_turn_chain()
    (item,) = G.chain_items([rec], [sample])
    assert item.ids == tuple(full) and item.final_turn == 1
    assert [(s.turn, s.start, s.length) for s in item.spans] == [(0, 6, 5), (1, 14, 6)]
    assert item.positions == list(range(6, 11)) + list(range(14, 20))
    assert [m["cached_tokens"] for m in item.meta] == [0, 9]
    solo = [{"raw": torch.full((11,), -0.15), "temp": torch.full((11,), -0.3), "entropy": torch.ones(11)}]
    packed = {item.chain_id: torch.full((6,), -0.25)}
    a = T.token_arrays([item], {"solo": solo}, packed)
    assert a["turn"].tolist() == [0] * 5 + [1] * 6 and a["j"].tolist() == list(range(5)) + list(range(6))
    assert a["final"].tolist() == [False] * 5 + [True] * 6
    assert np.isnan(a["packed"][:5]).all() and np.allclose(a["packed"][5:], -0.25)
    assert set(a) >= {"solo", "solo_temp", "solo_entropy", "packed", "engine"}
    s = T.summarize(a, ["solo", "packed"])
    assert s["solo"]["turn0"]["tokens"]["n"] == 5 and s["solo"]["turn1_nofirst"]["tokens"]["n"] == 5
    assert math.isclose(s["solo"]["turn0"]["tokens"]["median"], 0.05, rel_tol=1e-5)
    assert s["packed"]["all"]["tokens"]["n"] == 6 and "turn0" not in s["packed"]
    assert T.reply_ids(a).tolist() == [0] * 5 + [1] * 6


def test_a_chain_without_a_sample_is_not_an_item():
    rec, sample, _ = _two_turn_chain()
    assert G.chain_items([rec], []) == []


def test_reply_stats_values():
    d = np.array([0.0, 0.1, -0.2, 2.0])
    s = G.reply_stats(d)
    assert s["tokens"] == 4 and s["max"] == 2.0 and math.isclose(s["mean"], 2.3 / 4)
    assert math.isclose(s["frac_gt_1"], 0.25) and math.isclose(s["frac_gt_0p5"], 0.25)
    assert math.isclose(s["mean_signed"], 1.9 / 4) and math.isclose(s["abs_mean_signed"], 1.9 / 4)
    assert math.isclose(s["k3_mean"], float(np.mean(np.expm1(d) - d)))
    assert G.exclusions([s, G.reply_stats(np.array([0.01]))], "max", [0.5, 5.0]) == {"0.5": 1, "5": 0}
    with pytest.raises(ValueError):
        G.reply_stats(np.array([]))


def test_t10_reward_follows_the_terminal_gates_and_the_length_term():
    rec, _, _ = _two_turn_chain()
    r = T.t10_reward(rec, 4096)
    assert r.status == "scored" and math.isclose(r.reward, 0.5 - 0.05 * 6 / 4096)
    for end, base in (("no_code_block", -1.0), ("turns_exhausted", -0.9), ("context_limit", -0.9)):
        rec.end_reason = end
        assert math.isclose(T.t10_reward(rec, 4096).reward, base - 0.05 * 6 / 4096)
    rec.end_reason = "logprob_error"
    assert T.t10_reward(rec, 4096).status == "unscored"
    for end in ("engine_error", "quarantined"):
        rec.end_reason = end
        assert T.t10_reward(rec, 4096) is None


# --- the whole run on CPU --------------------------------------------------------------------------------


def _cfg(tmp_path, base, ds, lr=1e-2):
    return config_from_dict({
        "run": {"run_id": "t10cpu", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(base), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(ds / "listings"), "source": "standin"}]},
        "rollout": {"engines": [0, 1], "K": 4, "P": 4, "max_new_tokens": 512, "request_timeout_s": 300,
                    "compile_adapter": "fake.compile"},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"iterations": 1, "run_seed": 7},
        "trainer": {"ranks": 1, "lr": lr, "micro_batch_tokens": 2048, "max_generation_gap": 1.0e9,
                    "max_off_policy_fraction": 1.0},
        "sync": {"dir": str(tmp_path / "shm")},
    })


@pytest.fixture(scope="module")
def t10_run(tmp_path_factory, tiny_base, dataset):
    """One `run_t10` over the wired stand-ins: (config, output directory, document)."""
    root = tmp_path_factory.mktemp("t10-run")
    cfg = _cfg(root, tiny_base, dataset)
    engines = [E.TwinEngine(tiny_base, E.FAKE, i) for i in range(2)]
    parts = wiring.Parts(tokenizer=E.ByteTokenizer(), handles=[InProcessEngine(i, e) for i, e in enumerate(engines)],
                         compiler=FakeCompiler(), load_model=E.load_trainer_model, trainer_device="cpu",
                         require_fast_path=False, probes=E.PROBES)
    out = root / "out"
    doc = T.run_t10(cfg, out, T.T10Plan(steps=2, extra_v0_batches=1, tis_grad_at=(0,)), parts=parts,
                    log=lambda s: None)
    return cfg, out, doc


@pytest.mark.timeout(1800)  # the fixture's CPU run alone has taken 15 minutes on a 4-vCPU CI runner
def test_run_t10_on_cpu_measures_steps_publishes_and_checks_the_weights(t10_run):
    cfg, out, doc = t10_run
    assert doc["status"] == "completed" and doc["round_trip"]["verdict"] == "PROVEN" and doc["v0_weff_equals_base"]
    assert [p["n"] for p in doc["points"]] == [0, 1, 2]
    assert [b["name"] for b in doc["points"][0]["batches"]] == ["main", "extra1"]
    assert [b["prompts"] for b in doc["points"][1]["batches"]] == [doc["points"][0]["batches"][0]["prompts"]]
    for n in (0, 1):
        p = doc["points"][n]
        assert set(p["publish"]["verdicts"].values()) <= {"LANDED", "NOOP"} and p["publish"]["c4t3"]["ok"]
        assert p["step"]["checks"] and p["step"]["excluded"] == 0
        assert "stale" in p["stale_control"]
    g = doc["points"][0]["gradient"]
    assert set(g["norm"]) == {"default", "tis", "fp32la", "fp32la_tis"} and all(v > 0 for v in g["norm"].values())
    assert -1.0 <= g["cos"]["default~tis"] <= 1.0
    assert g["norm"]["tis"] != g["norm"]["default"]  # the twin engine's log-probs differ a little: TIS reweights
    assert "base" in doc["points"][2]["base_control"]
    a = np.load(out / "point0-main.npz")
    meta = json.loads((out / "point0-main.json").read_text())
    assert {"solo", "solo_temp", "solo_entropy", "fp32la", "packed", "stale", "engine"} <= set(a.files)
    assert a["engine"].shape == a["solo"].shape == a["stale"].shape
    assert not np.array_equal(a["stale"], a["solo"])  # scored after the step, on W_eff(1): the weights moved
    assert len(meta["items"]) == int(a["chain"].max()) + 1
    assert not np.isnan(a["packed"][a["final"]]).any() and np.isnan(a["packed"][~a["final"]]).all()
    # the gate's view here is the trainer step's own: the step's gaps file has the same per-sample numbers
    rows = {json.loads(x)["chain_id"]: json.loads(x)
            for x in (cfg.run_dir / "trainer" / "gaps-000001.jsonl").read_text().splitlines()}
    assert rows
    rid = T.reply_ids(a)
    for i, item in enumerate(meta["items"]):
        if item["chain_id"] not in rows:
            continue
        m = (a["chain"] == i) & a["final"]
        mine = sample_gap(item["chain_id"], torch.from_numpy(a["packed"][m].astype(np.float64)),
                          a["engine"][m].astype(np.float64).tolist())
        assert math.isclose(mine.max_abs, rows[item["chain_id"]]["max_abs"], rel_tol=0, abs_tol=1e-6)
        assert math.isclose(mine.mean_abs, rows[item["chain_id"]]["mean_abs"], rel_tol=0, abs_tol=1e-7)
    assert rid.size == a["engine"].size
    assert not wiring.node_dir(cfg).exists()  # slots and scratch removed at shutdown


REPORT = Path(__file__).resolve().parents[2] / "jobs" / "aorlta-t10-report.py"


@pytest.mark.skipif(not REPORT.is_file(), reason="jobs/aorlta-t10-report.py is not in this tree")
def test_the_t10_report_reads_what_the_run_wrote(t10_run, tmp_path):
    import importlib.util

    _, out, _ = t10_run
    spec = importlib.util.spec_from_file_location("t10_report", REPORT)
    report = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ under jobs/
    try:
        spec.loader.exec_module(report)
    finally:
        sys.dont_write_bytecode = saved
    assert report.main([str(out), "--json", str(tmp_path / "report.json")]) == 0
    got = json.loads((tmp_path / "report.json").read_text())[str(out)]
    assert got["status"] == "completed" and got["v0_solo_by_turn"] and got["gate_candidates"]
    assert {"stale by one step (W_eff(n+1) on v(n)'s samples)", "normal (solo view, for reference)"} <= set(
        got["controls"])
    assert got["fix1"]["paired"] and got["tis"]["gradient"]["v0"]["norm"]["tis"] > 0
