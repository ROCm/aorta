"""A deterministic stand-in for one TokenSpeed engine (design Section 8, WP0).

`FakeEngine.generate` takes the arguments AORLTA passes to TokenSpeed's `Engine.generate` and
returns the same shape: `{"text", "output_ids", "meta_info"}`, with `meta_info` carrying
`finish_reason` ({"type": "stop", "matched": 248046} or {"type": "length", "length": N}, as
measured on the engine), token counts, `cached_tokens` and, when asked for and enabled,
`output_token_logprobs` as (logprob, token_id, None) triples, which is what TokenSpeed returns at
3b684074 (`engine/logprobs.py:255-259`).

A response is a pure function of (fake seed, input ids, sampling settings, weights tag): it does not
depend on call order, thread, engine id or wall time. It is greedy, so independent of the request
seed, when the engine's sampler is "greedy" (TokenSpeed's default on AMD, A2) or the temperature is
0 (A4), so the start-up self-test can be shown to catch a greedy engine.

Responses are written with the fake's own byte-level tokenizer (`ByteTokenizer`, one token per
byte); decode them with `engine.tokenizer` or read `text`. Input ids are only hashed, so any
tokenizer's ids can be sent.
"""

from __future__ import annotations

import math
import socket
import threading
import time
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..ids import INT32_MAX, HashRng, sha256_ids
from .ports import CLUSTER_OFFSETS

STOP_TOKEN_ID = 248046  # <|im_end|>, the token the engine stops on (F2, Appendix C)
BYTE_TOKEN_BASE = 10_000  # ByteTokenizer: token id = BYTE_TOKEN_BASE + byte value
FAKE_ENGINE_SCHEMA = "aorlta.fake_engine/1"
GREEDY_EPS = 1e-6  # TokenSpeed treats T below this as greedy with seed 0 (A4)

# Keys AORLTA may send (Section 6.1, Sampling). `n` is accepted only as 1: AORLTA never uses n > 1.
SAMPLING_KEYS = frozenset(
    {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_new_tokens", "seed", "stop_token_ids", "n"}
)
# Engine.generate arguments the fake does not model; passing one is an error rather than a silent no-op.
UNMODELLED_GENERATE_ARGS = frozenset(
    {
        "logprob_start_len",
        "top_logprobs_num",
        "token_ids_logprob",
        "return_text_in_logprobs",
        "logprob_format",
        "custom_logit_processor",
        "return_hidden_states",
        "stream",
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
        "data_parallel_rank",
    }
)

_CODE_ANSWER = (
    "```cpp\n"
    "// aorlta fake answer {tag}\n"
    "#include <hip/hip_runtime.h>\n"
    "__global__ void aorlta_fake_{tag}(float* x, int n) {{\n"
    "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (i < n) x[i] = x[i] * {factor};\n"
    "}}\n"
    "```\n"
)
COMPILE_ERROR_MARKER = "aorlta_fake_undeclared"  # an undeclared identifier: real hipcc rejects it too
_NO_CODE_ANSWER = "I could not produce a corrected kernel for this input ({tag}).\n"
_PROSE = ("the", "kernel", "wave", "lane", "buffer", "barrier", "load", "store", "sync", "thread", "block",
          "index", "offset", "shared", "memory", "reads", "writes", "before", "after", "each")


class FakeEngineError(RuntimeError):
    """The request failed: an injected engine error, a hung request, or a killed engine."""


class ByteTokenizer:
    """Reversible tokenizer for the fake's own responses: one token per UTF-8 byte, plus the stop token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [BYTE_TOKEN_BASE + b for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        data = bytearray()
        parts: list[str] = []
        for t in ids:
            if t == STOP_TOKEN_ID:
                if not skip_special_tokens:
                    parts.append(data.decode("utf-8", errors="replace"))
                    parts.append("<|im_end|>")
                    data.clear()
            elif BYTE_TOKEN_BASE <= t < BYTE_TOKEN_BASE + 256:
                data.append(t - BYTE_TOKEN_BASE)
            else:
                raise ValueError(f"token id {t} is not a ByteTokenizer id")
        parts.append(data.decode("utf-8", errors="replace"))
        return "".join(parts)


def _rate(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _nonneg(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")


@dataclass(frozen=True)
class FakeEngineConfig:
    """The fake's knobs. Rates are per request; every draw is deterministic in the request."""

    seed: int = 0
    min_tokens: int = 256  # an answer's natural length is uniform in [min_tokens, max_tokens] tokens,
    max_tokens: int = 2048  # then capped at the request's max_new_tokens (finish "length")
    latency_base_s: float = 0.0  # simulated time = base + per_token * output tokens + U[0, jitter)
    latency_per_token_s: float = 0.0
    latency_jitter_s: float = 0.0
    time_scale: float = 0.0  # real seconds slept per simulated second; 0 never sleeps
    no_code_block_rate: float = 0.0  # answers with no ```cpp block
    compile_error_rate: float = 0.0  # code answers whose code holds an undeclared identifier
    error_rate: float = 0.0  # requests that raise FakeEngineError (engine_error)
    hang_rate: float = 0.0  # requests that take hang_s simulated seconds, then raise FakeEngineError
    hang_s: float = 1.0e4
    bad_logprobs_rate: float = 0.0  # responses whose log-probs are missing an entry or name a wrong id
    # WP2: a minimal prefix-cache model. Off: cached_tokens is always 0. On: cached_tokens is the longest
    # prefix shared with the inputs, and inputs + outputs minus the last sampled token, of earlier requests
    # since the last flush(), at most len(input) - 1, as measured on TokenSpeed (F15). It depends on
    # request order, so it is off by default and the responses themselves never depend on it.
    prefix_cache: bool = False
    # Bind the ports a one-node TokenSpeed engine binds for its `port` (rollout/ports.py) on 127.0.0.1 until
    # shutdown(), so a taken port fails the start with "Address already in use", as on the engine. Off: the ports
    # are only reported (`port_args`).
    bind_ports: bool = False

    def __post_init__(self) -> None:
        if not (isinstance(self.seed, int) and not isinstance(self.seed, bool)):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not (isinstance(self.min_tokens, int) and isinstance(self.max_tokens, int) and 1 <= self.min_tokens <= self.max_tokens):
            raise ValueError(f"need 1 <= min_tokens <= max_tokens, got {self.min_tokens}, {self.max_tokens}")
        for name in ("latency_base_s", "latency_per_token_s", "latency_jitter_s", "time_scale", "hang_s"):
            _nonneg(name, getattr(self, name))
        for name in ("no_code_block_rate", "compile_error_rate", "error_rate", "hang_rate", "bad_logprobs_rate"):
            _rate(name, getattr(self, name))
        if self.error_rate + self.hang_rate > 1.0:
            raise ValueError("error_rate + hang_rate must be <= 1")


@dataclass
class FakeServerArgs:
    """The engine kwargs the fake accepts, with TokenSpeed's defaults at 3b684074 (server_args.py),
    resolved as on non-NVIDIA hardware: no sampling backend given means "greedy" (A2)."""

    model: str = "fake"
    tokenizer: str | None = None
    dtype: str = "auto"
    sampling_backend: str | None = None
    enable_output_logprobs: bool = False
    enable_prefix_caching: bool = True
    disable_kvstore: bool = False
    disable_autotune: bool = False
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    port: int = 8000
    seed: int | None = None
    log_level: str = "info"

    def __post_init__(self) -> None:
        if self.sampling_backend is None:
            self.sampling_backend = "greedy"
        if self.sampling_backend not in ("greedy", "triton"):
            raise ValueError(f"the fake models the greedy and triton samplers only, got {self.sampling_backend!r}")


@dataclass(frozen=True)
class FakeRequest:
    """One generate call, as the fake saw it (for tests that count requests per chain and turn)."""

    request_id: str
    engine_id: int
    input_len: int
    input_ids_sha256: str
    sampling_params: dict
    outcome: str  # "ok" | "error" | "hang" | "dead"
    completion_tokens: int


def _check_ids(input_ids: Any) -> list[int]:
    if not isinstance(input_ids, (list, tuple)) or not input_ids:
        raise ValueError("input_ids must be a non-empty list of token ids")
    for t in input_ids:
        if not (isinstance(t, int) and not isinstance(t, bool) and 0 <= t <= INT32_MAX):
            raise ValueError(f"token id {t!r} is not an int in [0, 2**31)")
    return list(input_ids)


class FakeEngine:
    def __init__(
        self,
        fake: FakeEngineConfig | None = None,
        *,
        engine_id: int = 0,
        sleep: Callable[[float], None] = time.sleep,
        **engine_kwargs: Any,
    ):
        self.fake = fake if fake is not None else FakeEngineConfig()
        self.engine_id = engine_id
        self.server_args = FakeServerArgs(**engine_kwargs)
        self.tokenizer_manager = types.SimpleNamespace(server_args=self.server_args)
        d = self.server_args.port + CLUSTER_OFFSETS[0]
        self.port_args = types.SimpleNamespace(  # as TokenSpeed's PortArgs when none of the ports is taken
            dist_init_addr=f"127.0.0.1:{d}", tokenizer_ipc_name=f"tcp://127.0.0.1:{d + 1}",
            rpc_ipc_name=f"tcp://127.0.0.1:{d + 3}", metrics_ipc_name=f"tcp://127.0.0.1:{d + 4}",
            scheduler_input_ipc_name=f"tcp://127.0.0.1:{d + 5}", nccl_port=self.server_args.port + 100)
        self._sockets: list[socket.socket] = []
        if self.fake.bind_ports:
            try:
                for p in (d, d + 1, d + 3, d + 4, d + 5):
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._sockets.append(s)
                    s.bind(("127.0.0.1", p))
                    s.listen(1)
            except OSError:
                self._close_sockets()
                raise
        self.tokenizer = ByteTokenizer()
        self.requests: list[FakeRequest] = []
        self._sleep = sleep
        self._lock = threading.Lock()
        self._alive = True
        self._weights = "v0"
        self._seen: list[tuple[int, ...]] = []  # prefix_cache: what the cache holds since the last flush

    # --- control ----------------------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        """Every request from now on fails, as after an engine process death, until `restart()`."""
        self._alive = False

    def restart(self) -> None:
        self._alive = True

    def set_weights(self, tag: str) -> None:
        """Change the weights the fake serves; every response depends on the tag (v0 at start)."""
        if not isinstance(tag, str) or not tag:
            raise ValueError("weights tag must be a non-empty str")
        self._weights = tag

    def shutdown(self) -> None:
        self._alive = False
        self._close_sockets()

    def _close_sockets(self) -> None:
        for s in self._sockets:
            s.close()
        self._sockets = []

    def flush_prefix_cache(self) -> None:
        with self._lock:
            self._seen.clear()

    def _cached_tokens(self, ids: list[int], output_ids: list[int]) -> int:
        if not (self.fake.prefix_cache and self.server_args.enable_prefix_caching):
            return 0
        key = tuple(ids)
        with self._lock:
            best = 0
            for seen in self._seen:
                n = min(len(seen), len(key))
                i = 0
                while i < n and seen[i] == key[i]:
                    i += 1
                best = max(best, i)
            self._seen.append(key)
            self._seen.append(key + tuple(output_ids[:-1]))
        return min(best, len(ids) - 1)

    # --- generation -------------------------------------------------------------------------------

    def generate(
        self,
        prompt: Any = None,
        sampling_params: dict | None = None,
        input_ids: list[int] | None = None,
        return_logprob: bool = False,
        **unmodelled: Any,
    ) -> dict:
        bad = sorted(set(unmodelled) - UNMODELLED_GENERATE_ARGS)
        if bad:
            raise TypeError(f"generate() got unexpected keyword argument(s) {bad}")
        if unmodelled:
            raise NotImplementedError(f"the fake engine does not model {sorted(unmodelled)}")
        if prompt is not None:
            raise ValueError("AORLTA sends token ids, never text (Section 6.1)")
        ids = _check_ids(input_ids)
        sp = self._check_sampling(sampling_params)
        greedy = self.server_args.sampling_backend == "greedy" or sp["temperature"] < GREEDY_EPS
        if greedy:
            sampling: list[Any] = ["greedy"]
        else:
            sampling = ["triton", sp["seed"], float(sp["temperature"]), float(sp.get("top_p", 1.0)),
                        int(sp.get("top_k", -1)), float(sp.get("min_p", 0.0)), float(sp.get("repetition_penalty", 1.0))]
        ids_sha = sha256_ids(ids)
        rng = HashRng([FAKE_ENGINE_SCHEMA, self.fake.seed, self._weights, ids_sha, sampling])
        request_id = rng.hex("request_id", 32)

        def log(outcome: str, completion_tokens: int = 0) -> None:
            with self._lock:
                self.requests.append(
                    FakeRequest(request_id, self.engine_id, len(ids), ids_sha, dict(sp), outcome, completion_tokens)
                )

        if not self._alive:
            log("dead")
            raise FakeEngineError(f"engine {self.engine_id} is dead")
        u = rng.uniform("failure")
        if u < self.fake.error_rate:
            log("error")
            self._wait(self.fake.latency_base_s)
            raise FakeEngineError(f"injected engine error (request {request_id})")
        if u < self.fake.error_rate + self.fake.hang_rate:
            log("hang")
            self._wait(self.fake.hang_s)
            raise FakeEngineError(f"injected hang of {self.fake.hang_s} s (request {request_id})")

        content = self.tokenizer.encode(self._answer_text(rng))
        max_new = sp["max_new_tokens"]
        if len(content) + 1 <= max_new:
            output_ids = content + [STOP_TOKEN_ID]
            finish = {"type": "stop", "matched": STOP_TOKEN_ID}
        else:
            output_ids = content[:max_new]
            finish = {"type": "length", "length": max_new}
        latency = (self.fake.latency_base_s + self.fake.latency_per_token_s * len(output_ids)
                   + self.fake.latency_jitter_s * rng.uniform("jitter"))
        meta: dict[str, Any] = {
            "id": request_id,
            "finish_reason": finish,
            "prompt_tokens": len(ids),
            "completion_tokens": len(output_ids),
            "cached_tokens": self._cached_tokens(ids, output_ids),  # 0 unless the prefix_cache knob is on
            "weight_version": "default",  # TokenSpeed never updates it (A9)
            "e2e_latency": latency,
        }
        if return_logprob and self.server_args.enable_output_logprobs:
            meta["output_token_logprobs"] = self._logprobs(rng, output_ids)
        log("ok", len(output_ids))
        self._wait(latency)
        return {"text": self.tokenizer.decode(output_ids, skip_special_tokens=True), "output_ids": output_ids,
                "meta_info": meta}

    def _check_sampling(self, sampling_params: Any) -> dict:
        if not isinstance(sampling_params, dict):
            raise ValueError("sampling_params must be a dict")
        unknown = sorted(set(sampling_params) - SAMPLING_KEYS)
        if unknown:
            raise ValueError(f"unknown sampling params {unknown}")
        sp = dict(sampling_params)
        if sp.get("n", 1) != 1:
            raise ValueError("n > 1 is never used: one request per chain per turn (Section 6.1)")
        max_new = sp.get("max_new_tokens")
        if not (isinstance(max_new, int) and not isinstance(max_new, bool) and max_new >= 1):
            raise ValueError(f"max_new_tokens must be an int >= 1, got {max_new!r}")
        temperature = sp.setdefault("temperature", 1.0)
        if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and temperature >= 0):
            raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")
        for name in ("top_p", "min_p", "repetition_penalty"):
            v = sp.get(name)
            if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
        top_k = sp.get("top_k")
        if top_k is not None and not (isinstance(top_k, int) and not isinstance(top_k, bool)):
            raise ValueError(f"top_k must be an int, got {top_k!r}")
        stop = sp.get("stop_token_ids")
        if stop is not None and not (isinstance(stop, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in stop)):
            raise ValueError(f"stop_token_ids must be a list of ints, got {stop!r}")
        if temperature >= GREEDY_EPS and self.server_args.sampling_backend != "greedy":
            s = sp.get("seed")
            if not (isinstance(s, int) and not isinstance(s, bool)):
                raise ValueError("every sampled request carries its own int seed (Section 6.1)")
        return sp

    def _answer_text(self, rng: HashRng) -> str:
        tag = rng.hex("tag", 8)
        if rng.uniform("code_block") < self.fake.no_code_block_rate:
            tail = _NO_CODE_ANSWER.format(tag=tag)
        else:
            broken = rng.uniform("compile_error") < self.fake.compile_error_rate
            tail = _CODE_ANSWER.format(tag=tag, factor=COMPILE_ERROR_MARKER if broken else "2.0f")
        target = rng.randint("length", self.fake.min_tokens, self.fake.max_tokens)
        need = target - len(tail.encode("utf-8"))
        if need <= 0:
            return tail
        words = rng.stream("prose")
        prose: list[str] = []
        size = 0
        while size < need:
            word = _PROSE[int(words.random() * len(_PROSE))]
            prose.append(word)
            size += len(word) + 1
        return (" ".join(prose))[: need - 1] + "\n" + tail

    def _logprobs(self, rng: HashRng, output_ids: list[int]) -> list[tuple[float, int, None]]:
        draws = rng.stream("logprobs")
        triples = [(-4.0 * draws.random() ** 2, t, None) for t in output_ids]
        if rng.uniform("bad_logprobs") < self.fake.bad_logprobs_rate:
            if rng.uniform("bad_logprobs_kind") < 0.5 or len(triples) < 2:
                triples = triples[:-1]
            else:
                i = rng.randint("bad_logprobs_at", 0, len(triples) - 1)
                lp, t, _ = triples[i]
                triples[i] = (lp, t + 1, None)
        return triples

    def _wait(self, simulated_s: float) -> None:
        if self.fake.time_scale > 0 and simulated_s > 0:
            self._sleep(simulated_s * self.fake.time_scale)
