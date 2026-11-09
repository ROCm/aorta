"""C5 step 5: the landing check after every reload (design Section 6.5).

Engine side: 4 fixed probe sequences (a stand-in prompt plus a fixed reply, `probe_tokens` long) are scored one at
a time with prompt log-probs (`return_logprob=True, logprob_start_len=0, max_new_tokens=1`). TokenSpeed never
serves those positions from the prefix cache (the admission probe is capped at `logprob_start_len`,
`request_handler.py` at 3b684074), so each is a fresh prefill whatever the cache holds; the reply's
`cached_tokens` must be 0. Trainer side: the same tokens' raw log-probs under W_eff(n).

Verdicts, with changed(n) = elements where W_eff(n) != W_eff(n-1):
- LANDED: changed(n) > 0, the engine's probe log-probs differ bitwise from its previous ones, and the mean
  |engine - trainer| per token is <= tau.
- NOOP: changed(n) == 0 and the engine's log-probs are bitwise unchanged.
- NOT_LANDED: changed(n) > 0 but the engine's log-probs are bitwise unchanged. Stop.
- MISMATCH: the mean gap exceeds tau, or the engine changed under a zero update, or a log-prob is not finite. Stop.

tau = max(tau_floor, 3 x the median over the v0 probes of each probe's mean per-token gap). The calibration and
the verdict use the same statistic, a per-token mean, so v0 itself passes. If a solo prefill is not bitwise
repeatable (C5.T3), "unchanged" falls back to a tolerance: the largest per-token change is <= repeat_eps.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from aorlta.ids import sha256_ids, sha256_json
from aorlta.sync.writer import SyncError

VERDICTS = ("LANDED", "NOOP", "NOT_LANDED", "MISMATCH")
PASSING = frozenset({"LANDED", "NOOP"})

# The fixed reply of every probe: a plain, correct HIP source, the shape of answer the model is trained to give.
PROBE_REPLY = """Here is a corrected, complete HIP source file for this kernel.

```cpp
#include <hip/hip_runtime.h>
#include <stdint.h>

// One block reduces a tile of the input into shared memory, then writes one partial sum per block.
template <int BLOCK>
__global__ void block_reduce_sum(const float* __restrict__ in, float* __restrict__ partial, int n)
{
    __shared__ float tile[BLOCK];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * BLOCK * 2 + tid;
    float v = 0.0f;
    if (base < n) v += in[base];
    if (base + BLOCK < n) v += in[base + BLOCK];
    tile[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) tile[tid] += tile[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = tile[0];
}

// y = a * x + y over n elements, one element per thread, grid-stride loop.
__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

// Tiled matrix transpose through shared memory; the +1 column avoids bank conflicts.
constexpr int TILE = 32;
constexpr int ROWS = 8;
__global__ void transpose(float* __restrict__ out, const float* __restrict__ in, int width, int height)
{
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        if (x < width && (y + j) < height) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS) {
        if (x < height && (y + j) < width) out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// Exclusive prefix sum of one block of 256 values (Hillis-Steele scan in shared memory).
__global__ void block_scan(const int* __restrict__ in, int* __restrict__ out, int n)
{
    __shared__ int buf[2][256];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * 256 + tid;
    int src = 0;
    buf[src][tid] = (gid < n && tid > 0) ? in[gid - 1] : 0;
    __syncthreads();
    for (int offset = 1; offset < 256; offset <<= 1) {
        const int dst = 1 - src;
        int v = buf[src][tid];
        if (tid >= offset) v += buf[src][tid - offset];
        buf[dst][tid] = v;
        __syncthreads();
        src = dst;
    }
    if (gid < n) out[gid] = buf[src][tid];
}

// Histogram of byte values with per-block shared counters, merged with one atomic add per bin.
__global__ void histogram256(const uint8_t* __restrict__ data, unsigned int* __restrict__ bins, int n)
{
    __shared__ unsigned int local[256];
    for (int b = threadIdx.x; b < 256; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&local[data[i]], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += blockDim.x) atomicAdd(&bins[b], local[b]);
}
```

Each kernel synchronizes every shared-memory phase with `__syncthreads()` before another thread reads it, and
every global access is bounds-checked against `n`, `width` or `height`.
"""


class LandingError(SyncError):
    """A probe result is malformed or was not a fresh prefill: the check cannot be judged, so the run stops."""


@dataclass(frozen=True)
class ProbeSet:
    ids: tuple[tuple[int, ...], ...]

    @property
    def sha256(self) -> str:
        return sha256_json([sha256_ids(p) for p in self.ids])


def build_probes(tokenizer, user_texts: Sequence[str], *, system: str, tokens: int = 1024,
                 reply: str = PROBE_REPLY, reply_min: int = 256) -> ProbeSet:
    """One probe per user text: the chat-rendered prompt (thinking off) plus the fixed reply, exactly `tokens`
    long. Long prompts are cut at a line boundary so that at least `reply_min` reply tokens fit."""
    reply_ids = tokenizer(reply, add_special_tokens=False)["input_ids"]
    if len(reply_ids) < reply_min:
        raise LandingError(f"the probe reply has {len(reply_ids)} tokens, fewer than reply_min {reply_min}")
    out = []
    for text in user_texts:
        lines = text.split("\n")
        while True:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]
            rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
            prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if len(prompt_ids) + reply_min <= tokens or len(lines) <= 1:
                break
            lines = lines[:-1]
        ids = (prompt_ids + reply_ids)[:tokens]
        if len(ids) != tokens:
            raise LandingError(f"a probe has {len(ids)} tokens, not {tokens}")
        out.append(tuple(int(t) for t in ids))
    if len(set(out)) != len(out):
        raise LandingError("two probes are identical")
    return ProbeSet(tuple(out))


# --- scoring ----------------------------------------------------------------------------------------------


def score_request(ids: Sequence[int]) -> dict[str, Any]:
    """The engine request for one probe: a fresh prefill with prompt log-probs, one greedy token."""
    return dict(input_ids=list(ids), sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                return_logprob=True, logprob_start_len=0)


def parse_input_logprobs(out: Any, ids: Sequence[int]) -> list[float]:
    """The prompt log-probs of positions 1..L-1 from a TokenSpeed reply, checked against `ids`.

    `meta_info.input_token_logprobs` is `[(None, ids[0], None), (lp, ids[1], None), ...]`. Anything missing,
    misaligned, non-numeric, or from a request that read the prefix cache, raises LandingError."""
    meta = out.get("meta_info") if isinstance(out, dict) else None
    if not isinstance(meta, dict):
        raise LandingError(f"reply has no meta_info: {str(out)[:200]}")
    cached = meta.get("cached_tokens")
    if cached != 0:
        raise LandingError(f"probe reply has cached_tokens={cached!r}: not a fresh prefill")
    rows = meta.get("input_token_logprobs")
    if not isinstance(rows, (list, tuple)) or len(rows) != len(ids):
        raise LandingError(f"input_token_logprobs has {len(rows) if isinstance(rows, (list, tuple)) else rows!r} "
                           f"entries for {len(ids)} tokens")
    vals: list[float] = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 2 or row[1] != ids[i]:
            raise LandingError(f"input_token_logprobs[{i}] = {row!r} does not name token {ids[i]}")
        if i == 0:
            if row[0] is not None:
                raise LandingError(f"input_token_logprobs[0] should carry no log-prob, got {row[0]!r}")
            continue
        lp = row[0]
        if isinstance(lp, bool) or not isinstance(lp, (int, float)):
            raise LandingError(f"input_token_logprobs[{i}] log-prob is {lp!r}")
        vals.append(float(lp))
    return vals


def engine_logprobs(engine, probes: ProbeSet) -> list[list[float]]:
    """Score each probe alone, one after another."""
    return [parse_input_logprobs(engine.generate(**score_request(ids)), ids) for ids in probes.ids]


def trainer_logprobs(model, probes: ProbeSet, *, device=None, autocast_dtype=None) -> list[list[float]]:
    """Raw (temperature-1) log-probs of each probe's tokens under the trainer's current weights: fp32
    log-softmax of the logits, as TokenSpeed returns them (A3)."""
    import contextlib

    import torch

    device = device if device is not None else next(model.parameters()).device
    device = torch.device(device)
    out = []
    ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype is not None
           else contextlib.nullcontext())
    with torch.no_grad():
        for ids in probes.ids:
            x = torch.tensor([list(ids)], device=device)
            with ctx:
                logits = model(input_ids=x).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1)
            tgt = x[0, 1:]
            out.append(lp[:-1].gather(-1, tgt[:, None])[:, 0].double().cpu().tolist())
            del logits, lp
    return out


# --- calibration and verdict ---------------------------------------------------------------------------


def _check_shapes(name: str, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> None:
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise LandingError(f"{name}: probe counts or lengths differ ({[len(x) for x in a]} vs {[len(y) for y in b]})")
    if not a or any(len(x) == 0 for x in a):
        raise LandingError(f"{name}: no probe tokens")


def per_probe_gaps(engine: Sequence[Sequence[float]], trainer: Sequence[Sequence[float]]) -> list[float]:
    """Each probe's mean per-token |engine - trainer| (not finite if any value is not)."""
    _check_shapes("gap", engine, trainer)
    return [sum(abs(e - t) for e, t in zip(es, ts)) / len(es) for es, ts in zip(engine, trainer)]


@dataclass(frozen=True)
class Calibration:
    tau: float
    tau_floor: float
    per_probe_gap: tuple[float, ...]
    median_gap: float
    pooled_median_token_gap: float  # reported only: the median over every token, not used for tau


def calibrate_tau(engine_v0: Sequence[Sequence[float]], trainer_v0: Sequence[Sequence[float]],
                  tau_floor: float) -> Calibration:
    gaps = per_probe_gaps(engine_v0, trainer_v0)
    if not all(math.isfinite(g) for g in gaps):
        raise LandingError(f"v0 probe gaps are not finite: {gaps}")
    if not (tau_floor > 0):
        raise LandingError(f"tau_floor must be > 0, got {tau_floor}")
    med = statistics.median(gaps)
    tokens = [abs(e - t) for es, ts in zip(engine_v0, trainer_v0) for e, t in zip(es, ts)]
    return Calibration(tau=max(tau_floor, 3.0 * med), tau_floor=tau_floor, per_probe_gap=tuple(gaps),
                       median_gap=med, pooled_median_token_gap=statistics.median(tokens))


@dataclass(frozen=True)
class LandingResult:
    verdict: str
    changed: int
    mean_gap: float
    per_probe_gap: tuple[float, ...]
    tau: float
    engine_changed: bool
    engine_max_delta: float
    reason: str
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict in PASSING

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "ok": self.ok, "changed": self.changed, "mean_gap": self.mean_gap,
                "per_probe_gap": list(self.per_probe_gap), "tau": self.tau, "engine_changed": self.engine_changed,
                "engine_max_delta": self.engine_max_delta, "reason": self.reason, **self.extra}


def landing_verdict(changed: int, engine_now: Sequence[Sequence[float]], engine_prev: Sequence[Sequence[float]],
                    trainer_now: Sequence[Sequence[float]], tau: float, *, bitwise: bool = True,
                    repeat_eps: float = 0.0) -> LandingResult:
    if not isinstance(changed, int) or isinstance(changed, bool) or changed < 0:
        raise LandingError(f"changed(n) must be an int >= 0, got {changed!r}")
    _check_shapes("engine now vs previous", engine_now, engine_prev)
    _check_shapes("engine vs trainer", engine_now, trainer_now)
    flat_now = [v for p in engine_now for v in p]
    flat_prev = [v for p in engine_prev for v in p]
    flat_tr = [v for p in trainer_now for v in p]
    finite = all(math.isfinite(v) for v in flat_now + flat_prev + flat_tr)
    deltas = [abs(a - b) for a, b in zip(flat_now, flat_prev)] if finite else [math.inf]
    max_delta = max(deltas)
    engine_changed = any(a != b for a, b in zip(flat_now, flat_prev)) if bitwise else max_delta > repeat_eps
    gaps = tuple(per_probe_gaps(engine_now, trainer_now)) if finite else tuple(math.nan for _ in engine_now)
    mean_gap = sum(abs(a - b) for a, b in zip(flat_now, flat_tr)) / len(flat_now) if finite else math.nan

    def result(verdict: str, reason: str) -> LandingResult:
        return LandingResult(verdict, changed, mean_gap, gaps, tau, engine_changed, max_delta, reason)

    if not finite:
        return result("MISMATCH", "a probe log-prob is not finite")
    if changed == 0:
        if engine_changed:
            return result("MISMATCH", "the merge changed nothing, yet the engine's log-probs changed")
        return result("NOOP", "the update fell below bf16 resolution and the engine is unchanged")
    if not engine_changed:
        return result("NOT_LANDED", f"{changed} elements changed but the engine's log-probs did not")
    if mean_gap > tau:
        return result("MISMATCH", f"mean gap {mean_gap:.6g} > tau {tau:.6g}")
    return result("LANDED", f"mean gap {mean_gap:.6g} <= tau {tau:.6g}")


# --- C5.T3 -----------------------------------------------------------------------------------------------


def repeatability(runs: Sequence[Sequence[Sequence[float]]]) -> dict:
    """Bitwise agreement of repeated probe scorings under fixed weights, and the largest disagreement."""
    if len(runs) < 2:
        raise LandingError("repeatability needs at least 2 runs")
    first = runs[0]
    worst = 0.0
    identical = 0
    for r in runs[1:]:
        _check_shapes("repeat", first, r)
        flat_a = [v for p in first for v in p]
        flat_b = [v for p in r for v in p]
        if flat_a == flat_b:
            identical += 1
        worst = max(worst, max(abs(a - b) for a, b in zip(flat_a, flat_b)))
    return {"runs": len(runs), "identical_to_first": identical, "bitwise": identical == len(runs) - 1,
            "max_abs_delta": worst}


def landing_check(engines: Sequence[Any], probes: ProbeSet, changed: int, previous: dict[int, list[list[float]]],
                  trainer_now: Sequence[Sequence[float]], tau: dict[int, float] | float, *, bitwise: bool = True,
                  repeat_eps: float = 0.0, score: Callable = engine_logprobs) -> dict[int, tuple[LandingResult, list]]:
    """Score the probes on every engine and judge each against its own previous probes. Returns, per engine id,
    the result and the new engine log-probs (the caller keeps them as the next step's `previous`)."""
    out = {}
    for e in engines:
        now = score(e, probes)
        t = tau[e.engine_id] if isinstance(tau, dict) else tau
        out[e.engine_id] = (landing_verdict(changed, now, previous[e.engine_id], trainer_now, t, bitwise=bitwise,
                                            repeat_eps=repeat_eps), now)
    return out


# --- one publication: write, reload, landing check, cleanup (steps 3, 4, 5 and 7) -------------------------


class LandingFailed(SyncError):
    """NOT_LANDED or MISMATCH on some engine: stop, keeping both slots and the probe log-probs."""

    def __init__(self, message: str, publication: "Publication"):
        super().__init__(message)
        self.publication = publication


@dataclass
class SyncState:
    """What the driver carries from one publication to the next."""

    run_id: str
    sync_dir: str
    base_dir: str
    probes: ProbeSet
    tau: dict[int, float]
    previous: dict[int, list[list[float]]]  # each engine's probe log-probs after its last passed check
    in_flight: Callable[[], int]
    expected: int
    slots: int = 2
    partial: bool = True
    others: Any = None  # the base's other tensors, for full checkpoints
    bitwise: bool = True
    repeat_eps: float = 0.0
    book: Any = None  # reload.VersionBook


@dataclass(frozen=True)
class Publication:
    version: int
    slot: str
    changed: int
    results: dict[int, LandingResult]
    engine_logprobs: dict[int, list[list[float]]]
    trainer_logprobs: list[list[float]]
    seconds: dict[str, float]
    pruned: list[str]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results.values())

    def verdicts(self) -> dict[int, str]:
        return {e: r.verdict for e, r in self.results.items()}


def publish(weights, version: int, changed: int, *, engines: Sequence[Any], state: SyncState,
            trainer_side: Callable[[ProbeSet], list[list[float]]]) -> Publication:
    """Write W_eff(version) to its slot, reload every engine, run the landing check, prune the older slot.

    `changed` is the merge's changed(n) for these weights; `trainer_side(probes)` returns the trainer's raw probe
    log-probs under the same W_eff. Raises LandingFailed (after keeping every slot) unless every engine passes."""
    import time

    from aorlta.sync.reload import reload_all
    from aorlta.sync.writer import prune_slots, write_slot

    secs: dict[str, float] = {}
    t = time.monotonic()
    info = write_slot(weights, base_dir=state.base_dir, sync_dir=state.sync_dir, run_id=state.run_id,
                      version=version, slots=state.slots, partial=state.partial, others=state.others)
    secs["write"] = time.monotonic() - t
    t = time.monotonic()
    reload_all(engines, info.path, state.expected, in_flight=state.in_flight)
    secs["reload"] = time.monotonic() - t
    if state.book is not None:
        for e in engines:
            state.book.record(e.engine_id, version, info.path)
    t = time.monotonic()
    trainer_now = trainer_side(state.probes)
    checked = landing_check(engines, state.probes, changed, state.previous, trainer_now, state.tau,
                            bitwise=state.bitwise, repeat_eps=state.repeat_eps)
    secs["landing"] = time.monotonic() - t
    results = {e: r for e, (r, _) in checked.items()}
    now = {e: lps for e, (_, lps) in checked.items()}
    pub = Publication(version, str(info.path), changed, results, now, trainer_now, secs, [])
    if not pub.ok:
        raise LandingFailed(f"landing check failed at version {version}: {pub.verdicts()}", pub)
    state.previous = now
    pruned = [str(p) for p in prune_slots(state.sync_dir, state.run_id, keep=info.path)]
    return Publication(version, str(info.path), changed, results, now, trainer_now, secs, pruned)
