"""The fake engine: reproducible under a seed, and each knob does what it says."""

import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from aorlta.rollout.fake_engine import (
    COMPILE_ERROR_MARKER,
    STOP_TOKEN_ID,
    ByteTokenizer,
    FakeEngine,
    FakeEngineConfig,
    FakeEngineError,
)

TRITON = {"sampling_backend": "triton", "enable_output_logprobs": True}


def sp(seed: int, max_new: int = 4096, temperature: float = 0.7) -> dict:
    return {"temperature": temperature, "top_p": 0.95, "top_k": -1, "max_new_tokens": max_new, "seed": seed,
            "stop_token_ids": [STOP_TOKEN_ID]}


def requests(n: int) -> list[tuple[list[int], dict]]:
    """n requests: n // 4 prompts x 4 seeds."""
    return [([248045, 100 + p, 200 + p % 7, 198], sp(1000 * p + s)) for p in range(n // 4) for s in range(4)]


def run(engine: FakeEngine, reqs, order=None, threads: int = 1) -> list[dict]:
    order = list(range(len(reqs))) if order is None else order
    outs: list = [None] * len(reqs)

    def one(i):
        ids, params = reqs[i]
        outs[i] = engine.generate(input_ids=ids, sampling_params=params, return_logprob=True)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, order))
    return outs


def test_reproducible_under_a_seed_whatever_the_order_or_threads():
    reqs = requests(64)
    a = run(FakeEngine(FakeEngineConfig(seed=7), **TRITON), reqs)
    order = list(range(len(reqs)))
    random.Random(1).shuffle(order)
    b = run(FakeEngine(FakeEngineConfig(seed=7), engine_id=3, **TRITON), reqs, order=order, threads=16)
    assert a == b


def test_the_seeds_matter():
    reqs = requests(64)
    a = run(FakeEngine(FakeEngineConfig(seed=7), **TRITON), reqs)
    other = run(FakeEngine(FakeEngineConfig(seed=8), **TRITON), reqs)
    assert all(x["output_ids"] != y["output_ids"] for x, y in zip(a, other))
    texts = [o["text"] for o in a]
    assert len(set(texts)) == len(texts)  # different request seeds give different samples


def test_greedy_engines_ignore_the_request_seed():
    reqs = requests(16)
    default = FakeEngine(FakeEngineConfig(seed=7), enable_output_logprobs=True)
    assert default.tokenizer_manager.server_args.sampling_backend == "greedy"  # TokenSpeed's default on AMD (A2)
    outs = run(default, reqs)
    for p in range(4):
        assert len({tuple(o["output_ids"]) for o in outs[4 * p: 4 * p + 4]}) == 1
    triton = FakeEngine(FakeEngineConfig(seed=7), **TRITON)
    zero_t = [triton.generate(input_ids=ids, sampling_params=dict(params, temperature=0.0)) for ids, params in reqs[:4]]
    assert len({tuple(o["output_ids"]) for o in zero_t}) == 1


def test_weights_change_the_responses():
    engine = FakeEngine(FakeEngineConfig(seed=7), **TRITON)
    ids, params = requests(4)[0]
    before = engine.generate(input_ids=ids, sampling_params=params)
    engine.set_weights("v1")
    assert engine.generate(input_ids=ids, sampling_params=params)["output_ids"] != before["output_ids"]
    engine.set_weights("v0")
    assert engine.generate(input_ids=ids, sampling_params=params) == before


def test_output_shape_matches_tokenspeed():
    engine = FakeEngine(FakeEngineConfig(seed=1), **TRITON)
    out = engine.generate(input_ids=[1, 2, 3], sampling_params=sp(5), return_logprob=True)
    assert set(out) == {"text", "output_ids", "meta_info"}
    meta = out["meta_info"]
    assert {"id", "finish_reason", "prompt_tokens", "completion_tokens", "cached_tokens", "output_token_logprobs",
            "weight_version", "e2e_latency"} <= set(meta)
    assert meta["prompt_tokens"] == 3 and meta["completion_tokens"] == len(out["output_ids"])
    assert out["text"] == engine.tokenizer.decode(out["output_ids"], skip_special_tokens=True)
    triples = meta["output_token_logprobs"]
    assert [t for _, t, _ in triples] == out["output_ids"] and all(x is None for _, _, x in triples)
    assert all(lp <= 0 for lp, _, _ in triples)


def test_length_knob_and_the_cap():
    engine = FakeEngine(FakeEngineConfig(seed=1, min_tokens=500, max_tokens=500), **TRITON)
    out = engine.generate(input_ids=[1], sampling_params=sp(1))
    assert len(out["output_ids"]) == 501 and out["output_ids"][-1] == STOP_TOKEN_ID
    assert out["meta_info"]["finish_reason"] == {"type": "stop", "matched": STOP_TOKEN_ID}
    capped = engine.generate(input_ids=[1], sampling_params=sp(1, max_new=300))
    assert len(capped["output_ids"]) == 300 and STOP_TOKEN_ID not in capped["output_ids"]
    assert capped["meta_info"]["finish_reason"] == {"type": "length", "length": 300}
    assert capped["output_ids"] == out["output_ids"][:300]  # the same answer, cut
    assert "```" not in capped["text"][-200:] or capped["text"].count("```") < 2
    lengths = {len(o["output_ids"]) for o in run(FakeEngine(FakeEngineConfig(seed=2, min_tokens=400, max_tokens=900),
                                                            **TRITON), requests(200))}
    assert min(lengths) >= 401 and max(lengths) <= 901 and len(lengths) > 50


def _fraction(outs, pred) -> float:
    return sum(map(pred, outs)) / len(outs)


@pytest.mark.parametrize("rate", [0.0, 0.3, 1.0])
def test_no_code_block_rate(rate):
    outs = run(FakeEngine(FakeEngineConfig(seed=3, no_code_block_rate=rate), **TRITON), requests(2000))
    got = _fraction(outs, lambda o: "```cpp\n" not in o["text"])
    assert got == rate if rate in (0.0, 1.0) else abs(got - rate) < 0.05


@pytest.mark.parametrize("rate", [0.0, 0.25, 1.0])
def test_compile_error_rate(rate):
    outs = run(FakeEngine(FakeEngineConfig(seed=4, compile_error_rate=rate), **TRITON), requests(2000))
    got = _fraction(outs, lambda o: COMPILE_ERROR_MARKER in o["text"])
    assert got == rate if rate in (0.0, 1.0) else abs(got - rate) < 0.05
    assert all("```cpp\n" in o["text"] for o in outs)


@pytest.mark.parametrize("rate", [0.0, 0.2, 1.0])
def test_error_rate_and_errors_are_deterministic(rate):
    engine = FakeEngine(FakeEngineConfig(seed=5, error_rate=rate), **TRITON)
    failed = []
    for ids, params in requests(2000):
        try:
            engine.generate(input_ids=ids, sampling_params=params)
            failed.append(False)
        except FakeEngineError:
            failed.append(True)
    got = sum(failed) / len(failed)
    assert got == rate if rate in (0.0, 1.0) else abs(got - rate) < 0.05
    again = FakeEngine(FakeEngineConfig(seed=5, error_rate=rate), **TRITON)
    for (ids, params), f in list(zip(requests(2000), failed))[:200]:
        if f:
            with pytest.raises(FakeEngineError):
                again.generate(input_ids=ids, sampling_params=params)
        else:
            again.generate(input_ids=ids, sampling_params=params)
    assert [r.outcome for r in engine.requests].count("error") == sum(failed)


def test_bad_logprobs_rate():
    outs = run(FakeEngine(FakeEngineConfig(seed=6, bad_logprobs_rate=0.3), **TRITON), requests(2000))

    def bad(o):
        triples = o["meta_info"]["output_token_logprobs"]
        return len(triples) != len(o["output_ids"]) or [t for _, t, _ in triples] != o["output_ids"]

    assert abs(_fraction(outs, bad) - 0.3) < 0.05
    kinds = {len(o["meta_info"]["output_token_logprobs"]) == len(o["output_ids"]) for o in outs if bad(o)}
    assert kinds == {True, False}  # both a missing entry and a wrong id occur


def test_logprobs_absent_when_not_enabled():
    engine = FakeEngine(FakeEngineConfig(seed=1), sampling_backend="triton")  # enable_output_logprobs defaults off
    out = engine.generate(input_ids=[1], sampling_params=sp(1), return_logprob=True)
    assert "output_token_logprobs" not in out["meta_info"]
    out = FakeEngine(FakeEngineConfig(seed=1), **TRITON).generate(input_ids=[1], sampling_params=sp(1))
    assert "output_token_logprobs" not in out["meta_info"]


def test_latency_and_time_scale():
    slept = []
    cfg = FakeEngineConfig(seed=1, latency_base_s=2.0, latency_per_token_s=0.01, latency_jitter_s=0.5, time_scale=0.1)
    engine = FakeEngine(cfg, sleep=slept.append, **TRITON)
    out = engine.generate(input_ids=[1], sampling_params=sp(1))
    latency = out["meta_info"]["e2e_latency"]
    floor = 2.0 + 0.01 * len(out["output_ids"])
    assert floor <= latency < floor + 0.5
    assert slept == [pytest.approx(latency * 0.1)]
    quiet = FakeEngine(FakeEngineConfig(seed=1, latency_base_s=2.0), sleep=slept.append, **TRITON)
    quiet.generate(input_ids=[1], sampling_params=sp(1))
    assert len(slept) == 1  # time_scale 0 never sleeps


def test_a_hang_takes_hang_s_then_fails():
    slept = []
    engine = FakeEngine(FakeEngineConfig(seed=1, hang_rate=1.0, hang_s=1e4, time_scale=0.001), sleep=slept.append, **TRITON)
    with pytest.raises(FakeEngineError, match="hang"):
        engine.generate(input_ids=[1], sampling_params=sp(1))
    assert slept == [pytest.approx(10.0)]
    assert engine.requests[-1].outcome == "hang"


def test_kill_and_restart():
    engine = FakeEngine(FakeEngineConfig(seed=1), **TRITON)
    engine.kill()
    with pytest.raises(FakeEngineError, match="dead"):
        engine.generate(input_ids=[1], sampling_params=sp(1))
    engine.restart()
    assert engine.generate(input_ids=[1], sampling_params=sp(1))["output_ids"]


def test_one_request_per_call_is_logged():
    engine = FakeEngine(FakeEngineConfig(seed=1), **TRITON)
    run(engine, requests(8), threads=4)
    assert len(engine.requests) == 8 and {r.outcome for r in engine.requests} == {"ok"}
    assert sorted(r.sampling_params["seed"] for r in engine.requests) == sorted(p["seed"] for _, p in requests(8))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"input_ids": [1], "sampling_params": dict(sp(1), n=2)}, "n > 1"),
        ({"input_ids": [1], "sampling_params": {"temperature": 0.7, "max_new_tokens": 8}}, "own int seed"),
        ({"input_ids": [1], "sampling_params": dict(sp(1), max_tokens=8)}, "unknown sampling params"),
        ({"input_ids": [1], "sampling_params": {"temperature": 0.7, "seed": 1}}, "max_new_tokens"),
        ({"input_ids": [], "sampling_params": sp(1)}, "non-empty"),
        ({"input_ids": [1, -2], "sampling_params": sp(1)}, "token id"),
        ({"prompt": "hi", "input_ids": [1], "sampling_params": sp(1)}, "never text"),
        ({"input_ids": [1], "sampling_params": dict(sp(1), top_k=2.5)}, "top_k"),
    ],
)
def test_requests_aorlta_never_sends_are_refused(kwargs, error):
    with pytest.raises(ValueError, match=error):
        FakeEngine(FakeEngineConfig(), **TRITON).generate(**kwargs)


def test_unmodelled_and_unknown_arguments():
    engine = FakeEngine(FakeEngineConfig(), **TRITON)
    with pytest.raises(NotImplementedError):
        engine.generate(input_ids=[1], sampling_params=sp(1), logprob_start_len=0)
    with pytest.raises(TypeError):
        engine.generate(input_ids=[1], sampling_params=sp(1), bogus=1)
    with pytest.raises(TypeError):
        FakeEngine(FakeEngineConfig(), sampling_backnd="triton")
    with pytest.raises(ValueError):
        FakeEngine(FakeEngineConfig(), sampling_backend="flashinfer")


@pytest.mark.parametrize(
    "kwargs",
    [{"min_tokens": 0}, {"min_tokens": 10, "max_tokens": 5}, {"error_rate": 1.5}, {"error_rate": 0.6, "hang_rate": 0.5},
     {"latency_base_s": -1.0}, {"time_scale": float("nan")}, {"seed": 1.0}],
)
def test_bad_knobs_are_refused(kwargs):
    with pytest.raises(ValueError):
        FakeEngineConfig(**kwargs)


def test_byte_tokenizer_round_trips():
    tok = ByteTokenizer()
    text = "```cpp\nint x = 1; // é\n```"
    assert tok.decode(tok.encode(text)) == text
    assert tok.decode(tok.encode("a") + [STOP_TOKEN_ID]) == "a<|im_end|>"
    assert tok.decode(tok.encode("a") + [STOP_TOKEN_ID], skip_special_tokens=True) == "a"
    with pytest.raises(ValueError):
        tok.decode([5])
