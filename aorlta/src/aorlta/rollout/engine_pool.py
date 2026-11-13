"""The rollout engines (design Section 6.1: Engine start, Group formation, Failure modes).

- One engine per GPU, each in its own process (engine_proc). Engine kwargs are the Section 6.1 table.
- After start the resolved ServerArgs are read back, and the run refuses to start unless
  `sampling_backend == "triton"` and log-probs, prefix caching and dtype are as set. Then each engine
  runs a self-test: 4 prompts x 128 tokens at T 0.7, two seeds each, one request at a time. Different
  seeds must differ on at least 3 of 4, and log-probs must align with ids on all 8.
- Groups go to engines by longest processing time first, on estimated tokens
  (prompt length + K x running mean response length), at most ceil(P/E) groups per engine, so each
  engine's chains fit its `max_num_seqs = ceil(P*K/E)` and all are in flight at once.
- `ideal_cached_tokens` is the longest prefix of a request this engine has already seen since its
  last flush: earlier inputs (from their submit) and earlier input + output (from their reply).
- A dead or timed-out engine fails its in-flight requests (`engine_error`), is restarted with the same
  kwargs, passes start-up and self-test again, and serves again only after the `on_restart` hook
  (the reload and landing check of Section 6.5, owned by WP3/WP7) returns True. A second death in one
  iteration stops the run.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..ids import seed
from .templates import STOP_TOKEN_IDS

SELF_TEST_PROMPTS = 4
SELF_TEST_TOKENS = 128
SELF_TEST_TEMPERATURE = 0.7
SELF_TEST_TOP_P = 0.95
SELF_TEST_MIN_DIFFER = 3
PORT_BASE, PORT_STRIDE = 20000, 2000


class RolloutStop(RuntimeError):
    """A condition under which Section 6.1 stops the run."""


# --- prefix index ------------------------------------------------------------------------------------


class PrefixIndex:
    """A radix tree of token sequences, for "the longest prefix this engine has already seen"."""

    def __init__(self) -> None:
        self._root: dict[int, list] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._root = {}

    @staticmethod
    def _common(edge: tuple, ids: tuple, i: int) -> int:
        n = min(len(edge), len(ids) - i)
        if edge[:n] == ids[i:i + n]:
            return n
        lo, hi = 0, n  # edge[:lo] matches, edge[:hi] does not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if edge[:mid] == ids[i:i + mid]:
                lo = mid
            else:
                hi = mid
        return lo

    def longest_prefix(self, ids: Sequence[int]) -> int:
        seq = tuple(ids)
        with self._lock:
            node, i = self._root, 0
            while i < len(seq):
                entry = node.get(seq[i])
                if entry is None:
                    return i
                edge, child = entry
                m = self._common(edge, seq, i)
                i += m
                if m < len(edge):
                    return i
                node = child
            return i

    def insert(self, ids: Sequence[int]) -> None:
        seq = tuple(ids)
        with self._lock:
            node, i = self._root, 0
            while i < len(seq):
                entry = node.get(seq[i])
                if entry is None:
                    node[seq[i]] = [seq[i:], {}]
                    return
                edge, child = entry
                m = self._common(edge, seq, i)
                if m < len(edge):
                    # split the edge at m
                    mid: dict[int, list] = {edge[m]: [edge[m:], child]}
                    entry[0], entry[1] = edge[:m], mid
                    child = mid
                i += m
                node = child


# --- start-up checks -------------------------------------------------------------------------------


def expected_server_args(dtype: str = "bfloat16") -> dict:
    return {"sampling_backend": "triton", "enable_output_logprobs": True, "enable_prefix_caching": True,
            "dtype": dtype}


def server_arg_problems(info: dict, expected: dict) -> list[str]:
    """Each read-back setting that differs from what AORLTA set (a missing one counts)."""
    return [f"{k} is {info.get(k, '<missing>')!r}, expected {v!r}" for k, v in expected.items() if info.get(k) != v]


MAX_LOGPROB = 1e-3  # an engine's log-probability of 1 can round slightly above 0; the trainer takes the same bound


def logprobs_aligned(res: dict) -> bool:
    """Log-probs present, one per output id, naming the same ids, all finite and at most MAX_LOGPROB."""
    ids, vals, lp_ids = res.get("output_ids"), res.get("lp_vals"), res.get("lp_ids")
    if not ids or vals is None or lp_ids is None:
        return False
    if len(vals) != len(ids) or list(lp_ids) != list(ids):
        return False
    return all(isinstance(v, float) and math.isfinite(v) and v <= MAX_LOGPROB for v in vals)


def self_test_seeds(run_seed: int, i: int) -> tuple[int, int]:
    return (seed(run_seed, 1, f"selftest:{i}", 0, 0), seed(run_seed, 1, f"selftest:{i}", 1, 0))


@dataclass
class SelfTest:
    engine: int
    passed: bool
    differ: int
    aligned: int
    requests: int
    seconds: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"engine": self.engine, "passed": self.passed, "differ": self.differ, "aligned": self.aligned,
                "requests": self.requests, "seconds": round(self.seconds, 3), "errors": self.errors}


def run_self_test(engine: Any, prompts: Sequence[Sequence[int]], run_seed: int, timeout_s: float,
                  stop_token_ids: Sequence[int] = STOP_TOKEN_IDS) -> SelfTest:
    if len(prompts) < SELF_TEST_PROMPTS:
        raise ValueError(f"the self-test needs {SELF_TEST_PROMPTS} prompts, got {len(prompts)}")
    t0 = time.monotonic()
    differ = aligned = requests = 0
    errors: list[str] = []
    for i, ids in enumerate(prompts[:SELF_TEST_PROMPTS]):
        outs = []
        for s in self_test_seeds(run_seed, i):  # one at a time
            sp = {"temperature": SELF_TEST_TEMPERATURE, "top_p": SELF_TEST_TOP_P, "top_k": -1,
                  "max_new_tokens": SELF_TEST_TOKENS, "seed": s, "stop_token_ids": list(stop_token_ids)}
            res = engine.generate(list(ids), sp, timeout_s=timeout_s)
            requests += 1
            if "error" in res:
                errors.append(f"prompt {i} seed {s}: {res['error'][:300]}")
                outs.append(None)
                continue
            aligned += logprobs_aligned(res)
            outs.append(res["output_ids"])
        if outs[0] is not None and outs[1] is not None and outs[0] != outs[1]:
            differ += 1
    n = 2 * SELF_TEST_PROMPTS
    passed = not errors and differ >= SELF_TEST_MIN_DIFFER and aligned == n
    return SelfTest(engine.index, passed, differ, aligned, requests, time.monotonic() - t0, errors)


# --- sizing and placement --------------------------------------------------------------------------


def max_model_len(max_prompt: int, max_new: int, max_feedback_tokens: int, repairs: int = 2) -> int:
    """`max_prompt + 3*max_new + 2*max_feedback_tokens` for 2 repairs (Section 6.1 table)."""
    return max_prompt + (repairs + 1) * max_new + repairs * max_feedback_tokens


def engine_kwargs(
    *, model_dir: str, dtype: str, P: int, K: int, n_engines: int, max_len: int, gpu_memory_utilization: float,
    run_seed: int, log_level: str = "warning",
) -> dict:
    """The Section 6.1 engine kwargs, without `port` (set per engine)."""
    return {
        "model": model_dir, "tokenizer": model_dir, "dtype": dtype, "sampling_backend": "triton",
        "enable_output_logprobs": True, "enable_prefix_caching": True, "disable_kvstore": True,
        "disable_autotune": True, "max_num_seqs": math.ceil(P * K / n_engines), "max_model_len": max_len,
        "gpu_memory_utilization": gpu_memory_utilization, "seed": run_seed, "log_level": log_level,
    }


def engine_port(index: int) -> int:
    return PORT_BASE + PORT_STRIDE * index


def lpt_assign(estimates: Sequence[float], n_engines: int) -> list[int]:
    """Engine index per group: largest estimate first, each to the least-loaded engine that still has
    room (at most ceil(G/E) groups each). Ties go to the lower engine index and then to the earlier
    group, so the assignment is deterministic."""
    if n_engines < 1:
        raise ValueError("n_engines must be >= 1")
    cap = math.ceil(len(estimates) / n_engines)
    load = [0.0] * n_engines
    count = [0] * n_engines
    out = [-1] * len(estimates)
    for g in sorted(range(len(estimates)), key=lambda g: (-estimates[g], g)):
        e = min((e for e in range(n_engines) if count[e] < cap), key=lambda e: (load[e], e))
        out[g] = e
        load[e] += estimates[g]
        count[e] += 1
    return out


# --- the pool --------------------------------------------------------------------------------------


@dataclass
class GenOutcome:
    result: dict
    ideal_cached_tokens: int
    t_submit: float
    t_done: float

    @property
    def ok(self) -> bool:
        return "error" not in self.result


class EnginePool:
    """The engines one rollout manager drives. `handles` are EngineProc or InProcessEngine objects."""

    def __init__(
        self,
        handles: Sequence[Any],
        *,
        run_seed: int,
        expected: dict | None = None,
        request_timeout_s: float = 1800.0,
        self_test_timeout_s: float = 300.0,
        on_restart: Callable[[int], bool] | None = None,
        make_handle: Callable[[int], Any] | None = None,
        clock: Callable[[], float] = time.time,
        stop_token_ids: Sequence[int] = STOP_TOKEN_IDS,  # the model's (templates.chat_format); Qwen3.5's by default
    ):
        if not handles:
            raise ValueError("an engine pool needs at least one engine")
        self.handles = list(handles)
        self.run_seed = run_seed
        self.expected = expected if expected is not None else expected_server_args()
        self.request_timeout_s = request_timeout_s
        self.self_test_timeout_s = self_test_timeout_s
        self.on_restart = on_restart
        self.make_handle = make_handle
        self.clock = clock
        self.stop_token_ids = tuple(stop_token_ids)
        self.prefix = [PrefixIndex() for _ in self.handles]
        self.available = [False] * len(self.handles)
        self.self_tests: list[dict] = []
        self.ready: list[dict | None] = [None] * len(self.handles)
        self.deaths: list[dict] = []
        self.stop_reason: str | None = None
        self._iteration = 0
        self._lock = threading.Lock()
        self._restarting: set[int] = set()
        self._restart_threads: list[threading.Thread] = []

    def __len__(self) -> int:
        return len(self.handles)

    # start-up

    def _bring_up(self, i: int, selftest_prompts: Sequence[Sequence[int]]) -> None:
        handle = self.handles[i]
        ready = handle.start()
        self.ready[i] = ready
        problems = server_arg_problems(ready.get("info", {}), self.expected)
        if problems:
            handle.stop()
            raise RolloutStop(f"engine {i} refused: " + "; ".join(problems))
        test = run_self_test(handle, selftest_prompts, self.run_seed, self.self_test_timeout_s, self.stop_token_ids)
        with self._lock:
            self.self_tests.append(test.to_dict())
        if not test.passed:
            handle.stop()
            raise RolloutStop(f"engine {i} failed its self-test: {test.to_dict()}")
        self.prefix[i].clear()  # the self-test is not part of any iteration's cache accounting

    def start(self, selftest_prompts: Sequence[Sequence[int]]) -> None:
        """Start every engine in parallel; any refusal or failed self-test stops the run."""
        errors: list[BaseException | None] = [None] * len(self.handles)

        def one(i: int) -> None:
            try:
                self._bring_up(i, selftest_prompts)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors[i] = exc

        threads = [threading.Thread(target=one, args=(i,)) for i in range(len(self.handles))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        failed = [(i, e) for i, e in enumerate(errors) if e is not None]
        if failed:
            self.shutdown()
            raise RolloutStop("engine start-up failed: " + " | ".join(f"engine {i}: {e}" for i, e in failed))
        self.available = [True] * len(self.handles)
        self._selftest_prompts = [list(p) for p in selftest_prompts]

    # iteration accounting

    def begin_iteration(self, iteration: int) -> None:
        with self._lock:
            self._iteration = iteration

    def deaths_in(self, iteration: int) -> int:
        with self._lock:
            return sum(1 for d in self.deaths if d["iteration"] == iteration)

    def note_flush(self, i: int) -> None:
        """A reload with flush_cache=True on engine i (WP3): nothing is cached any more."""
        self.prefix[i].clear()

    # generation

    def generate(self, i: int, ids: list[int], sp: dict, *, extra: dict | None = None) -> GenOutcome:
        ids = list(ids)
        if not self.available[i]:
            t = self.clock()
            return GenOutcome({"error": f"engine {i} is not in service", "kind": "dead"}, 0, t, t)
        ideal = self.prefix[i].longest_prefix(ids)
        self.prefix[i].insert(ids)
        t_submit = self.clock()
        res = self.handles[i].generate(ids, sp, timeout_s=self.request_timeout_s, extra=extra)
        t_done = self.clock()
        if "error" in res:
            if res.get("kind") in ("dead", "timeout"):
                self._on_death(i, res)
        elif res.get("output_ids"):
            self.prefix[i].insert(ids + list(res["output_ids"]))
        return GenOutcome(res, ideal, t_submit, t_done)

    def _on_death(self, i: int, res: dict) -> None:
        with self._lock:
            if not self.available[i] or i in self._restarting:
                return  # this incident is already being handled
            self.available[i] = False
            self.deaths.append({"engine": i, "iteration": self._iteration, "kind": res.get("kind"),
                                "error": str(res.get("error"))[:300], "t": self.clock()})
            second = sum(1 for d in self.deaths if d["iteration"] == self._iteration) >= 2
            if second:
                self.stop_reason = self.stop_reason or f"second engine death in iteration {self._iteration}"
                return
            self._restarting.add(i)
        self.handles[i].kill(f"{res.get('kind')}: {str(res.get('error'))[:200]}")
        t = threading.Thread(target=self._restart, args=(i,), name=f"engine-{i}-restart", daemon=True)
        self._restart_threads.append(t)
        t.start()

    def _restart(self, i: int) -> None:
        try:
            if self.make_handle is not None:
                self.handles[i] = self.make_handle(i)
            self._bring_up(i, self._selftest_prompts)
            ok = bool(self.on_restart(i)) if self.on_restart is not None else False
            with self._lock:
                self.available[i] = ok
                if not ok:
                    self.deaths[-1]["restart"] = "awaiting landing check"
                else:
                    self.deaths[-1]["restart"] = "back in service"
        except BaseException as exc:  # noqa: BLE001
            with self._lock:
                self.stop_reason = self.stop_reason or f"engine {i} could not be restarted: {exc}"[:500]
        finally:
            with self._lock:
                self._restarting.discard(i)

    def wait_restarts(self, timeout_s: float | None = None) -> None:
        for t in list(self._restart_threads):
            t.join(timeout_s)

    def shutdown(self) -> None:
        for h in self.handles:
            try:
                h.stop()
            except Exception:  # noqa: BLE001
                pass
