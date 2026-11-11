"""The wired components end to end on CPU (`aorlta.orchestrator.wiring.build`).

Everything is the integrated code: WP2's engine pool and rollout manager, WP5's reward service with WP6's log-only
hook, WP4's rank trainer over WP3's merged LoRA, WP3's slot writer, reload, landing check and round trip, WP7's loop,
and WP8's timeline over the spans it writes. Three things are stand-ins:
  - the model: a tiny Qwen3.5 with the real vocabulary, so the real token ids fit;
  - the engines: `TwinEngine`, FakeEngine's replies with their log-probs rescored by a tiny-model replica that
    reloads from the weight slots as TokenSpeed does, and answers the landing probes and round-trip draws itself;
  - the checks: the fake profile and FakeCompiler.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
import torch

C4 = Path(__file__).resolve().parents[1] / "trainer" / "c4"
if str(C4) not in sys.path:
    sys.path.insert(0, str(C4))

from c4_helpers import TINY_TEXT, TINY_VISION, honour_packing  # noqa: E402

from aorlta.config import config_from_dict  # noqa: E402
from aorlta.ids import standin_prompt_id  # noqa: E402
from aorlta.measure import timeline  # noqa: E402
from aorlta.orchestrator import wiring  # noqa: E402
from aorlta.rollout.compile_check import FakeCompiler  # noqa: E402
from aorlta.rollout.engine_proc import InProcessEngine  # noqa: E402
from aorlta.rollout.fake_engine import BYTE_TOKEN_BASE, FakeEngine, FakeEngineConfig  # noqa: E402
from aorlta.rollout.templates import THINK_PREFIX  # noqa: E402
from aorlta.sync.landing import ProbeSet  # noqa: E402
from aorlta.trainer.ranks import rank_agreement_cosine  # noqa: E402

import re  # noqa: E402

pytestmark = pytest.mark.heavy

VOCAB = 248320  # Qwen3.5's: the byte ids (10000+) and the control ids (248044-248069) are real rows
ENGINE_KW = {"dtype": "bfloat16", "sampling_backend": "triton", "enable_output_logprobs": True,
             "enable_prefix_caching": True}
SPECIAL = {"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046, "<think>": 248068,
           "</think>": 248069}
_SPECIAL_RE = re.compile("|".join(re.escape(s) for s in SPECIAL))
_SPECIAL_TEXT = {v: k for k, v in SPECIAL.items()}


class ByteTokenizer:
    """The fake engine's byte tokenizer plus the control tokens, with Qwen3.5's system/user template part."""

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        for m in _SPECIAL_RE.finditer(text):
            out += [BYTE_TOKEN_BASE + b for b in text[i:m.start()].encode("utf-8")]
            out.append(SPECIAL[m.group(0)])
            i = m.end()
        return out + [BYTE_TOKEN_BASE + b for b in text[i:].encode("utf-8")]

    def decode(self, ids) -> str:
        parts, buf = [], bytearray()
        for t in ids:
            if t in _SPECIAL_TEXT:
                parts += [buf.decode("utf-8", errors="replace"), _SPECIAL_TEXT[t]]
                buf.clear()
            else:
                buf.append(t - BYTE_TOKEN_BASE)
        return "".join(parts + [buf.decode("utf-8", errors="replace")])

    def render(self, messages, thinking, add_generation_prompt=True) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" + THINK_PREFIX[thinking] if add_generation_prompt else "")


def load_tiny(base_dir) -> torch.nn.Module:
    from transformers import Qwen3_5ForConditionalGeneration

    m = Qwen3_5ForConditionalGeneration.from_pretrained(str(base_dir), dtype=torch.bfloat16)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_trainer_model(base_dir, device):
    m = load_tiny(base_dir)
    honour_packing(m)  # on CPU the linear-attention torch path ignores packing; this runs it per sequence
    return m, None


class TwinEngine:
    """FakeEngine's replies, with each output token's log-prob from a tiny-model replica (raw, temperature 1, as
    TokenSpeed reports them). The replica reloads a slot like TokenSpeed: every tensor it knows is copied, the reply
    counts what was read. Requests with prompt log-probs (the landing probes, the round trip's greedy draws) are the
    replica's own: a fresh prefill and greedy tokens."""

    CHUNK = 32

    def __init__(self, base_dir: Path, fake: FakeEngineConfig, engine_id: int):
        self.fake = FakeEngine(fake, engine_id=engine_id, **ENGINE_KW)
        self.tokenizer_manager = self.fake.tokenizer_manager
        self.model = load_tiny(base_dir)
        self.params = dict(self.model.named_parameters())
        self.reloads: list[str] = []
        self._lock = threading.Lock()

    def restart(self):
        self.fake.restart()

    def kill(self):
        self.fake.kill()

    def update_weights_from_disk(self, model_path: str, flush_cache: bool = True):
        from safetensors import safe_open

        path = Path(model_path)
        wm = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        read = 0
        with self._lock, torch.no_grad():
            for fn in sorted(set(wm.values())):
                with safe_open(str(path / fn), framework="pt") as f:
                    for name in f.keys():
                        read += 1
                        if name in self.params:
                            self.params[name].copy_(f.get_tensor(name))
            self.reloads.append(str(path))
            self.fake.set_weights(f"{path.name}#{len(self.reloads)}")
        if flush_cache:
            self.fake.flush_prefix_cache()
        return (True, f"Succeeded to update model weights from {model_path} ({read} checkpoint tensors read)", 0)

    def _logprobs_of(self, ids: list[int], out: list[int]) -> list[float]:
        with self._lock, torch.no_grad():
            logits = self.model(input_ids=torch.tensor([ids + out]), logits_to_keep=len(out) + 1).logits[0, :-1]
            vals = []
            for i in range(0, len(out), self.CHUNK):
                lp = torch.log_softmax(logits[i:i + self.CHUNK].float(), dim=-1)
                vals += lp.gather(-1, torch.tensor(out[i:i + self.CHUNK])[:, None])[:, 0].tolist()
        return vals

    def _prefill_and_greedy(self, ids: list[int], max_new: int) -> dict:
        with self._lock, torch.no_grad():
            lp = torch.log_softmax(self.model(input_ids=torch.tensor([ids])).logits[0].float(), dim=-1)
            vals = lp[:-1].gather(-1, torch.tensor(ids[1:])[:, None])[:, 0].tolist()
            out, last, cur = [], lp[-1], list(ids)
            for _ in range(max_new):
                nxt = int(torch.argmax(last))
                out.append((float(last[nxt]), nxt, None))
                cur.append(nxt)
                last = torch.log_softmax(self.model(input_ids=torch.tensor([cur]), logits_to_keep=1).logits[0, -1]
                                         .float(), dim=-1)
        meta = {"id": "twin", "finish_reason": {"type": "length", "length": max_new}, "prompt_tokens": len(ids),
                "completion_tokens": max_new, "cached_tokens": 0, "weight_version": "default", "e2e_latency": 0.0,
                "input_token_logprobs": [(None, ids[0], None)] + [(v, t, None) for v, t in zip(vals, ids[1:])],
                "output_token_logprobs": out}
        return {"text": "", "output_ids": [t for _, t, _ in out], "meta_info": meta}

    def generate(self, input_ids=None, sampling_params=None, return_logprob=False, logprob_start_len=None, **kw):
        if logprob_start_len is not None:
            if logprob_start_len != 0 or (sampling_params or {}).get("temperature", 1.0) != 0.0:
                raise NotImplementedError("the twin scores prompts only from position 0, greedily")
            return self._prefill_and_greedy(list(input_ids), int(sampling_params["max_new_tokens"]))
        out = self.fake.generate(input_ids=input_ids, sampling_params=sampling_params, return_logprob=return_logprob,
                                 **kw)
        meta = out["meta_info"]
        if meta.get("output_token_logprobs") is not None:
            vals = self._logprobs_of(list(input_ids), list(out["output_ids"]))
            meta["output_token_logprobs"] = [(v, t, None) for v, t in zip(vals, out["output_ids"])]
        return out


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> Path:
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    d = tmp_path_factory.mktemp("tiny-base-real-vocab")
    torch.manual_seed(1234)
    cfg = Qwen3_5Config(text_config=dict(TINY_TEXT, vocab_size=VOCAB, dtype="bfloat16"), vision_config=TINY_VISION,
                        tie_word_embeddings=True)
    Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(str(d), safe_serialization=True)
    if not (d / "model.safetensors.index.json").exists():
        wm = {}
        for p in sorted(d.glob("*.safetensors")):
            with safe_open(str(p), framework="pt") as f:
                wm.update({k: p.name for k in f.keys()})
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return d


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """Seven stand-in kernels: five to train on, one held out, one too long."""
    root = tmp_path_factory.mktemp("standin-dataset")
    kernels = {"k1": (80, False), "k2": (120, False), "k3": (200, True), "k4": (300, False), "k5": (900, False),
               "k6": (150, False), "k7": (390, False)}
    (root / "listings").mkdir()
    full = {}
    for kid, (n, held) in kernels.items():
        (root / "listings" / f"{kid}.json").write_text(json.dumps([[0, f"s_load_dword s0, s[0:1], 0x{n:x}"],
                                                                   [8, "s_waitcnt lgkmcnt(0)"], [12, "s_endpgm"]]))
        full[kid] = {"kid": kid, "n_insns": n, "heldout": held, "demangled": f"kernel_{kid}", "family": "gemm",
                     "library": "rocblas"}
    (root / "kernels.json").write_text(json.dumps(full))
    return root


PROBES = ProbeSet(tuple(tuple(int(t) for t in torch.randint(BYTE_TOKEN_BASE, BYTE_TOKEN_BASE + 256, (40,),
                                                                generator=torch.Generator().manual_seed(s)))
                        for s in range(3)))
FAKE = FakeEngineConfig(seed=11, min_tokens=1, max_tokens=1, compile_error_rate=0.5, no_code_block_rate=0.1)


def run_config(tmp_path, base, ds, *, run_id="wired", iterations=2, lr=1e-2, stochastic=False, gap_gate="enforce",
               warmup=False, engines=2, ranks=1):
    return config_from_dict({
        "run": {"run_id": run_id, "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(base), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(ds / "listings"), "source": "standin"}]},
        "rollout": {"engines": list(range(engines)), "K": 4, "P": 4, "max_new_tokens": 512, "request_timeout_s": 300,
                    "compile_adapter": "fake.compile"},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"iterations": iterations, "run_seed": 7},
        "trainer": {"ranks": ranks, "lr": lr, "micro_batch_tokens": 2048, "gap_gate": gap_gate,
                    "startup_warmup": warmup},
        "sync": {"dir": str(tmp_path / "shm"), "stochastic_rounding": stochastic},
    })


def build(cfg, base):
    engines = [TwinEngine(base, FAKE, i) for i in range(len(cfg.rollout.engines))]
    parts = wiring.Parts(tokenizer=ByteTokenizer(), handles=[InProcessEngine(i, e) for i, e in enumerate(engines)],
                         compiler=FakeCompiler(), load_model=load_trainer_model, trainer_device="cpu",
                         require_fast_path=False, probes=PROBES)
    return wiring.build(cfg, parts=parts), engines


def iteration_dir(cfg, n: int) -> Path:
    return cfg.run_dir / "iterations" / f"iter-{n:06d}"


def summary(cfg, n: int) -> dict:
    return json.loads((iteration_dir(cfg, n) / "summary.json").read_text())


def episodes(cfg, n: int) -> list[dict]:
    return [json.loads(line) for line in (iteration_dir(cfg, n) / "episodes.jsonl").read_text().splitlines()]


def serves(engine: TwinEngine, weights) -> bool:
    return all(torch.equal(engine.params[k], w) for k, w in weights.items())


def test_two_wired_iterations_train_publish_and_land(tmp_path, tiny_base, dataset):
    cfg = run_config(tmp_path, tiny_base, dataset)
    orch, engines = build(cfg, tiny_base)
    kids = ("k1", "k2", "k4", "k6", "k7")  # sorted, without the held-out k3 and the too-long k5
    assert [r.prompt_id for r in orch.pool.records()] == [standin_prompt_id(k) for k in kids]
    result = orch.run()
    assert result["status"] == "completed", result
    assert result["iterations_completed"] == 2 and result["policy_version"] == "wired:v2"

    trained_multi_turn = 0
    for n in (1, 2):
        s = summary(cfg, n)
        assert s["samples"]["total"] == 16 and s["scored_fraction"] >= 0.9, s["samples"]
        assert s["trainer"]["step_taken"] and s["samples"]["trained"] > 0
        assert s["trainer"]["gap"]["statistic"] == "max_abs" and s["trainer"]["gap_p99"] < 0.5
        assert s["trainer"]["gap_gate"]["mode"] == "enforce" and s["trainer"]["gap_gate"]["compared"] > 0
        assert s["sync"]["landing"] == "LANDED" and s["sync"]["rounding"] == "nearest" and s["sync"]["changed"] > 0
        assert s["sync"]["slot"].endswith(f"/w{n % 2}")
        assert s["spans"]["missing"] == [], s["spans"]
        gaps = [json.loads(x) for x in (cfg.run_dir / "trainer" / f"gaps-{n:06d}.jsonl").read_text().splitlines()]
        assert len(gaps) == s["samples"]["trained"]
        eps = episodes(cfg, n)
        assert all(e["prompt_id"] != standin_prompt_id("k3") for e in eps)  # held out, never sampled
        trained_multi_turn += sum(1 for e in eps if e["trained"] and len(e["turns"]) > 1)
        out = tmp_path / f"timeline-{n}.json"
        assert timeline.main([str(iteration_dir(cfg, n) / "spans.jsonl"), "--json", str(out)]) == 0
    assert trained_multi_turn > 0  # a repaired chain went through C1's turns into C4's turn spans

    final = orch.trainer.merged.weights
    assert all(serves(e, final) for e in engines)  # both engines hold exactly the trainer's W_eff(2)
    base = orch.trainer.merged.base
    assert not any(torch.equal(final[k], base[k]) for k in final)  # and it moved off the base everywhere
    # round trip (w1 perturbed, w0 restored), then one publication per iteration
    assert [Path(p).name for p in engines[0].reloads] == ["w1", "w0", "w1", "w0"]
    assert not wiring.node_dir(cfg).exists()  # slots and scratch are removed at shutdown
    assert "setup.warmup" not in [s["name"] for s in setup_spans(cfg)]  # the warm-up is off by default


@pytest.mark.cpu_run  # two wired runs, one of them with four rank processes: on a runner of its own in CI
def test_four_wired_ranks_train_the_model_one_rank_trains(tmp_path, tiny_base, dataset):
    """`trainer.ranks: 4`, one rank beside each of 4 engines (spawned processes over gloo on CPU), against one rank,
    from the same seeds and run id: the same samples train with the same totals, every group is one `train.stream`
    span on the rank the ledger names, every publication lands and the engines serve rank 0's W_eff, and iteration 1's
    step (same data, same weights) is the one rank's within C4.T1's limits. Iteration 2 starts from weights that
    differ by summation order, which Adam amplifies where a gradient element is near zero, so it is compared loosely.
    The warm-up runs on every rank. Rank 0 runs in this process, so it gets the spawned ranks' two threads: a bf16
    matmul's result depends on the thread count."""
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        _four_against_one(tmp_path, tiny_base, dataset)
    finally:
        torch.set_num_threads(threads)


def _four_against_one(tmp_path, tiny_base, dataset):
    out = {}
    for ranks in (1, 4):
        cfg = run_config(tmp_path / f"r{ranks}", tiny_base, dataset, run_id="ranked", engines=4, ranks=ranks,
                         warmup=True)
        orch, engines = build(cfg, tiny_base)
        grads = {}
        orch._point = lambda name, n, o=orch: grads.setdefault(n, o.trainer.trainer.last_flat_grad.clone()) \
            if name == "train.step" else None
        result = orch.run()
        assert result["status"] == "completed", result
        weights = orch.trainer.merged.weights
        assert all(serves(e, weights) for e in engines)
        steps = []
        for n in (1, 2):
            s = summary(cfg, n)
            assert s["trainer"]["step_taken"] and s["sync"]["landing"] == "LANDED" and s["spans"]["missing"] == []
            checks = s["trainer"]["checks"]
            assert checks["groups_trained_once"]["passed"] and checks["replicas_identical"]["passed"]
            spans = [json.loads(x) for x in (iteration_dir(cfg, n) / "spans.jsonl").read_text().splitlines()]
            streams = [sp for sp in spans if sp["name"] == "train.stream"]
            assert len(streams) == s["groups"]["trained"] and {sp["attrs"]["rank"] for sp in streams} <= set(range(4))
            assert timeline.main([str(iteration_dir(cfg, n) / "spans.jsonl"), "--json", str(tmp_path / "t.json")]) == 0
            steps.append((s["samples"]["trained"], s["trainer"]["counts"]["samples"], s["trainer"]["loss"],
                          checks["groups_trained_once"]["per_rank"]))
        warm = next(x for x in setup_spans(cfg) if x["name"] == "setup.warmup")["attrs"]
        assert warm["state"] == "unchanged" and "error" not in warm
        out[ranks] = (steps, grads)
    one, four = out[1], out[4]
    assert [s[:2] for s in one[0]] == [s[:2] for s in four[0]]  # the same samples trained in both iterations
    assert all(len(s[3]) == 4 and sum(s[3]) > 0 for s in four[0]) and all(len(s[3]) == 1 for s in one[0])
    assert rel(four[0][0][2], one[0][0][2]) <= 1e-6  # C4.T1's loss limit, iteration 1
    g4, g1 = four[1][1].double(), one[1][1].double()
    assert cosine(g4, g1) >= rank_agreement_cosine(torch.bfloat16) and float((g4 - g1).norm() / g1.norm()) < 1e-5
    assert rel(four[0][1][2], one[0][1][2]) <= 1e-4 and cosine(four[1][2].double(), one[1][2].double()) >= 0.999


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def cosine(a, b) -> float:
    return float(a @ b) / (float(a.norm()) * float(b.norm()) + 1e-300)


def setup_spans(cfg, attempt: int = 1) -> list[dict]:
    path = cfg.run_dir / "attempts" / f"attempt-{attempt:03d}-setup-spans.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_a_startup_warmup_runs_in_setup_and_one_that_moves_the_state_stops_the_run(tmp_path, tiny_base, dataset,
                                                                                    monkeypatch):
    """`trainer.startup_warmup`: after the round trip, one forward and backward on a synthetic group from the
    landing probes' tokens, as a setup span; the run then trains and lands as usual. A warm-up that leaves the
    latent moved stops the run before iteration 1."""
    from aorlta.trainer.ranks import RankTrainer

    cfg = run_config(tmp_path, tiny_base, dataset, run_id="warm", iterations=1, warmup=True)
    orch, engines = build(cfg, tiny_base)
    result = orch.run()
    assert result["status"] == "completed", result
    names = [s["name"] for s in setup_spans(cfg)]
    assert names.count("setup.warmup") == 1 and names.index("setup.round_trip") < names.index("setup.warmup")
    w = setup_spans(cfg)[names.index("setup.warmup")]["attrs"]
    assert (w["microbatches"], w["state"]) == (2, "unchanged") and "error" not in w
    assert w["forward_tokens"] > w["loss_tokens"] > 0 and w["backward_s"] > 0
    s = summary(cfg, 1)
    assert s["trainer"]["step_taken"] and s["sync"]["landing"] == "LANDED"
    assert all(serves(e, orch.trainer.merged.weights) for e in engines)

    real = RankTrainer.warmup

    def moves_the_latent(self, group):
        out = real(self, group)
        with torch.no_grad():
            self.params[0].add_(1.0)
        return out

    monkeypatch.setattr(RankTrainer, "warmup", moves_the_latent)
    bad = run_config(tmp_path, tiny_base, dataset, run_id="warm-moved", iterations=1, warmup=True)
    orch, _ = build(bad, tiny_base)
    result = orch.run()
    assert result["status"] == "stopped" and result["stop"]["reason"] == "warmup_failed", result
    assert result["iterations_completed"] == 0 and not iteration_dir(bad, 1).exists()
    assert result["stop"]["detail"]["error"] == "WiringError: the start-up warm-up changed the trainer's state"


class Crash(BaseException):
    pass


def test_a_wired_run_killed_after_its_first_commit_resumes_onto_the_same_weights(tmp_path, tiny_base, dataset):
    cfg = run_config(tmp_path, tiny_base, dataset, run_id="resumed")
    first, _ = build(cfg, tiny_base)
    held = {}

    def die_after_commit(name, n):
        if name == "record.committed" and n == 1:
            held.update({k: w.clone() for k, w in first.trainer.merged.weights.items()})
            raise Crash()

    first._point = die_after_commit
    with pytest.raises(Crash):
        first.run()

    again, engines = build(cfg, tiny_base)
    seen = {}

    def check_restored(name, n):
        if name == "select" and n == 2:
            seen["restored"] = all(serves(e, held) for e in engines)

    again._point = check_restored
    result = again.run(resume=True)
    assert result["status"] == "completed", result
    assert result["iterations_completed"] == 2 and result["attempts"][-1]["resumed_from"] == 1
    assert seen == {"restored": True}  # the restored latent re-merged to the very W_eff(1) the first process served
    assert summary(cfg, 2)["sync"]["landing"] == "LANDED"
    setup = [json.loads(x) for x in (cfg.run_dir / "attempts" / "attempt-002-setup-spans.jsonl").read_text()
             .splitlines()]
    restore = [s for s in setup if s["name"] in ("merge", "write", "reload", "landing")]
    assert [s["name"] for s in restore] == ["merge", "write", "reload", "landing"]  # recorded as setup, iteration 0
    assert restore[-1]["attrs"] == {"version": 1, "verdict": ["LANDED", "LANDED"]}


def test_stochastic_rounding_is_what_the_engines_serve_and_keeps_a_small_step(tmp_path, tiny_base, dataset):
    got = {}
    for mode in ("nearest", "stochastic"):
        cfg = run_config(tmp_path, tiny_base, dataset, run_id=f"sr-{mode}", iterations=1, lr=1e-4,
                         stochastic=mode == "stochastic", gap_gate="report_only")
        orch, engines = build(cfg, tiny_base)
        assert orch.run()["status"] == "completed"
        gate = summary(cfg, 1)["trainer"]["gap_gate"]  # report-only reaches the trainer: logged, nothing excluded
        assert gate["mode"] == "report_only" and summary(cfg, 1)["trainer"]["excluded"] == 0
        s = summary(cfg, 1)["sync"]
        assert s["rounding"] == mode and s["landing"] in ("LANDED", "NOOP")
        assert all(serves(e, orch.trainer.merged.weights) for e in engines)
        got[mode] = s
    near, sr = got["nearest"], got["stochastic"]
    assert sr["landing"] == "LANDED" and sr["changed"] > near["changed"]
    assert abs(sr["aligned"] - 1.0) < 0.05 and sr["aligned"] > (near["aligned"] or 0.0)
