"""C4 over R ranks, one per GPU beside its engine (design Section 4 and decision D10; Section 6.4, Ranks
and Streaming gradient accumulation; Section 6.6, step 4).

Rank 0 runs in the orchestrator's process: C5 writes its W_eff and the landing check runs its forward. Ranks 1..R-1
are spawned processes (`worker_main`), one per GPU. Every rank is built by the same three calls (`load_policy`,
`join_world`, `make_trainer`), so every rank makes the same collective calls in the same order: torch.distributed
(RCCL on GPU, gloo on CPU) inside `RankTrainer` for the start check, the gradient all-reduce and the step's gathers,
and a pair of queues per worker for everything else.

One iteration (exact; Section 6.4's streaming accumulation):
  - `begin(n)` opens a streamed step on every rank.
  - `assign(plan)`: as each group is complete, `OnlineAssigner` picks the rank with the fewest tokens still queued
    (then the fewest assigned, then the lowest rank). The assignment (seq, group key, rank, samples, tokens) is
    broadcast to every rank, and only that rank gets the group's samples. Each rank accumulates its groups'
    unnormalised gradients while generation and checks go on (rank 0 on its own thread).
  - `finish()` waits until every rank has trained every group it was given (a rank that failed or died stops the
    step here, before any collective). Then every rank enters `RankTrainer.stream_finish` with its own copy of the
    ledger: the all-reduce, the global normaliser, `groups_trained_once`, one optimizer step on identical replicas,
    the W_eff refresh, and `replicas_identical` over the latent and every rank's W_eff. Any rank's failed check fails
    the step, so nothing is published.
  - `discard()` drops the open step on every rank.

Each group's forward and backward is a `train.stream` span naming the rank that ran it. A worker's `time.monotonic()`
is the orchestrator's (one host), so its times go onto the span recorder's timeline as they are.

A collective that a failed rank never joins blocks the others until `spec.timeout_s`; so the step's collectives
start only after every rank has reported its last group (the drain), and a rank that dies is noticed by its queue.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import pickle
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import torch

from aorlta.atomic import atomic_write_bytes
from aorlta.trainer import checks as C
from aorlta.trainer.loss import GroupPlan, LossSettings, TrainerInputError
from aorlta.trainer.ranks import (Assignment, DistWorld, LocalWorld, OnlineAssigner, OptimSettings, RankTrainer,
                                  broadcast_lora, group_tokens, weights_checksum)

log = logging.getLogger(__name__)

TRAINER_STATE = "trainer.pt"
TRAINER_STATE_SCHEMA = "aorlta.trainer_state/1"
START_TIMEOUT_S = 1800.0  # one rank's model load and set-up
CALL_TIMEOUT_S = 1800.0  # one command's reply; a drain waits as long
COLLECTIVE_TIMEOUT_S = 600.0  # torch.distributed's: every collective starts after the drain, so seconds suffice
POLL_S = 0.5


class RankError(RuntimeError):
    """A rank failed, died, or did not answer in time. The open step is not taken."""


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


# --- what a rank is built from ----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RankSpec:
    """Everything a rank needs to build its trainer. It is pickled to the spawned workers, so `load_model` must be
    importable by name (a module-level function), returning (model, fallback-log watcher or None)."""

    base_dir: str
    devices: tuple[str, ...]  # rank r's torch device: "cuda:<gpu>" or "cpu"
    memory_fractions: tuple[float | None, ...]  # rank r's cap on its GPU's memory (None: no cap)
    loss: LossSettings
    optim: OptimSettings
    micro_batch_tokens: int
    load_model: Callable
    merged_kwargs: dict = field(default_factory=dict)  # MergedLoRA's rounding arguments
    merged_factory: Callable | None = None  # (model, base keys, **merged_kwargs) -> merged forward; None: MergedLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: str = ""
    seed: int = 0  # torch's seed before PEFT draws A (rank 0's A is then broadcast anyway)
    amp: bool = True
    gap_gate: bool = True
    require_fast_path: bool = True
    grad_checkpointing: bool = False
    backend: str = "gloo"  # "nccl" (RCCL) when every rank is on a GPU
    init_method: str = ""
    timeout_s: float = COLLECTIVE_TIMEOUT_S  # torch.distributed's collective timeout
    worker_threads: int | None = None  # torch's CPU threads in each spawned rank (None: torch's default)

    def __post_init__(self) -> None:
        if not self.devices:
            raise TrainerInputError("a rank pool needs at least one device")
        if len(self.memory_fractions) != len(self.devices):
            raise TrainerInputError("one memory fraction per rank")
        if len(self.devices) > 1 and not self.init_method:
            raise TrainerInputError("more than one rank needs torch.distributed's init_method")

    @property
    def size(self) -> int:
        return len(self.devices)


def file_init_method(directory) -> str:
    """torch.distributed's rendezvous for one pool: a new file store in `directory` (one host). A file cannot be taken
    by another process between choosing it and binding it, as a "free" TCP port can."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    return f"file://{d / f'trainer-pg-{uuid.uuid4().hex}'}"


def load_policy_model(base_dir: str, device: str):
    """The bf16 base as WP4 loads it, with the log watcher that `prepare_model` reads. The transformers class is the
    checkpoint's family's (`model_family`): AutoModelForImageTextToText for Qwen3.5, AutoModelForCausalLM for Qwen3."""
    import transformers

    from aorlta.model_family import read_family
    from aorlta.trainer.kernels import FallbackLogWatcher

    auto = getattr(transformers, read_family(base_dir).auto_class)
    with FallbackLogWatcher() as watcher:
        model = auto.from_pretrained(base_dir, dtype=torch.bfloat16, local_files_only=True)
        model.to(device)
    return model, watcher


def prepare_device(spec: RankSpec, rank: int) -> None:
    dev = spec.devices[rank]
    if dev.startswith("cuda"):
        torch.cuda.set_device(dev)
        frac = spec.memory_fractions[rank]
        if frac is not None:  # the engine on this GPU holds the rest (gpu_memory_utilization)
            torch.cuda.set_per_process_memory_fraction(frac, torch.device(dev))


def load_policy(spec: RankSpec, rank: int):
    """Rank `rank`'s policy before any collective: the frozen base, prepared for training, with a fresh LoRA."""
    from peft import LoraConfig, get_peft_model

    from aorlta.trainer.batching import prepare_model

    model, watcher = spec.load_model(spec.base_dir, spec.devices[rank])
    for p in model.parameters():
        p.requires_grad_(False)
    prepare_model(model, require_fast_path=spec.require_fast_path, watcher=watcher,
                  grad_checkpointing=spec.grad_checkpointing)
    torch.manual_seed(spec.seed)
    return get_peft_model(model, LoraConfig(r=spec.lora_r, lora_alpha=spec.lora_alpha, lora_dropout=0.0,
                                            target_modules=spec.target_modules, bias="none"))


def join_world(spec: RankSpec, rank: int):
    if spec.size == 1:
        return LocalWorld()
    import torch.distributed as dist

    kwargs = {"device_id": torch.device(spec.devices[rank])} if spec.backend == "nccl" else {}
    dist.init_process_group(spec.backend, init_method=spec.init_method, rank=rank, world_size=spec.size,
                            timeout=timedelta(seconds=spec.timeout_s), **kwargs)
    return DistWorld()


def make_trainer(spec: RankSpec, rank: int, model, world, merged_factory: Callable | None = None):
    """The collective part of a rank's set-up, the same on every rank: rank 0's latent to all, the merged forward on
    that latent, then the trainer, whose constructor refuses ranks that start apart (latent, base, W_eff)."""
    from aorlta.sync.writer import read_base_index
    from aorlta.trainer.merged_lora import MergedLoRA

    broadcast_lora(model, world)
    factory = merged_factory or spec.merged_factory or MergedLoRA
    merged = factory(model, read_base_index(spec.base_dir), **spec.merged_kwargs).install()
    trainer = RankTrainer(model, merged, spec.loss, spec.optim, world=world, micro_batch_tokens=spec.micro_batch_tokens,
                          device=spec.devices[rank], amp=spec.amp, gap_gate=spec.gap_gate)
    return merged, trainer


def without_turns(plan: GroupPlan) -> GroupPlan:
    """The group as a rank trains it: each sample's turn spans (the gap by turn, C4.T10) stay with the orchestrator."""
    return dataclasses.replace(plan, samples=tuple(dataclasses.replace(s, turns=()) for s in plan.samples))


def memory_peak(device: str, *, reset: bool = True) -> dict | None:
    """This process's peak memory on `device` since the last reset (then reset), and the whole GPU's memory in use
    now, by every process on it (the engine beside the rank included); None off a GPU."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return None
    dev = torch.device(device)
    free, total = torch.cuda.mem_get_info(dev)
    out = {"device": device, "max_allocated_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
           "max_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30, "gpu_used_gib": (total - free) / 2**30,
           "gpu_total_gib": total / 2**30}
    if reset:
        torch.cuda.reset_peak_memory_stats(dev)
    return out


# --- a rank's trained state ----------------------------------------------------------------------------------


def _tensor_bytes(t) -> bytes:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_digest(t: RankTrainer) -> str:
    """sha256 over the trained latent and the optimizer: what a resume has to restore bit for bit."""
    named = t.lora.named()
    h = hashlib.sha256(json.dumps({"schema": TRAINER_STATE_SCHEMA, "steps_done": t.steps_done,
                                   "names": sorted(named)}, sort_keys=True).encode())
    for name in sorted(named):
        p = named[name]
        h.update(f"{name}|{p.dtype}|{tuple(p.shape)}".encode())
        h.update(_tensor_bytes(p))
    sd = t.optimizer.state_dict()
    h.update(json.dumps(sd["param_groups"], sort_keys=True, default=str).encode())
    for idx in sorted(sd["state"]):
        for key in sorted(sd["state"][idx]):
            v = sd["state"][idx][key]
            h.update(f"{idx}|{key}".encode())
            h.update(_tensor_bytes(v) if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


def save_state(t: RankTrainer, directory) -> None:
    state = {"schema": TRAINER_STATE_SCHEMA, "steps_done": t.steps_done,
             "lora": {name: p.detach().cpu().clone() for name, p in t.lora.named().items()},
             "optimizer": t.optimizer.state_dict(),
             "rng": {"torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    buf = io.BytesIO()
    torch.save(state, buf)
    atomic_write_bytes(Path(directory) / TRAINER_STATE, buf.getvalue())


def load_state(t: RankTrainer, directory, device: str, *, rng: bool = True) -> None:
    """The latent, the optimizer and the step count from `save_state`'s file; the RNG states too with `rng` (a
    worker leaves its own: nothing in a step draws from them)."""
    state = torch.load(Path(directory) / TRAINER_STATE, map_location=device, weights_only=True)
    if state.get("schema") != TRAINER_STATE_SCHEMA:
        raise TrainerInputError(f"{directory}: trainer state schema {state.get('schema')!r}")
    named = t.lora.named()
    if sorted(state["lora"]) != sorted(named):
        raise TrainerInputError(f"{directory}: the saved LoRA tensors are not this model's")
    with torch.no_grad():
        for name, p in named.items():
            p.copy_(state["lora"][name])
    t.optimizer.load_state_dict(state["optimizer"])
    t.steps_done = int(state["steps_done"])
    if rng:
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if state["rng"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])


def remerge(merged, version: int) -> None:
    """Re-merge the current latent as policy version `version` (after a restore: the version it was trained to)."""
    merged.version = version - 1
    merged.refresh()


# --- a worker rank, in its own process -------------------------------------------------------------------------


def worker_main(rank: int, spec_bytes: bytes, cmd_q, reply_q) -> None:
    """Rank `rank` (>= 1) in its own process: build the trainer, then serve commands in order until 'quit'.
    The spec arrives pickled so that nothing heavy is imported before this process's own set-up."""
    put = reply_q.put
    try:
        spec = pickle.loads(spec_bytes)
        if spec.worker_threads:
            torch.set_num_threads(spec.worker_threads)
        prepare_device(spec, rank)
        model = load_policy(spec, rank)
        put(("loaded", rank))
        world = join_world(spec, rank)
        merged, trainer = make_trainer(spec, rank, model, world)
        put(("ready", rank, {"device": spec.devices[rank], "pid": os.getpid()}))
    except BaseException as exc:  # noqa: BLE001 - reported to rank 0, which refuses to start
        put(("fatal", rank, describe(exc)))
        return
    try:
        _Worker(spec, rank, trainer, merged, put).serve(cmd_q)
    finally:
        if spec.size > 1:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:  # noqa: BLE001 - the process is ending
                pass


class _Worker:
    """A worker rank's commands; each reply names the command it answers."""

    def __init__(self, spec: RankSpec, rank: int, trainer: RankTrainer, merged, put):
        self.spec, self.rank, self.trainer, self.merged, self.put = spec, rank, trainer, merged, put
        self.device = spec.devices[rank]
        self.ledger: list[Assignment] = []

    def serve(self, cmd_q) -> None:
        while True:
            msg = cmd_q.get()
            kind, args = msg[0], msg[1:]
            if kind == "quit":
                self.put(("bye", self.rank))
                return
            try:
                getattr(self, f"cmd_{kind}")(*args)
            except BaseException as exc:  # noqa: BLE001 - the coordinator decides
                self.put(("error", kind, describe(exc)))

    def cmd_begin(self, n: int) -> None:
        self.merged.version = n - 1  # the step's refresh makes version n
        self.trainer.stream_begin()
        self.ledger = []

    def cmd_assign(self, entry: tuple, plan: GroupPlan | None) -> None:
        a = Assignment(*entry)
        if a.seq != len(self.ledger):
            raise RankError(f"assignment {a.seq} arrived after {len(self.ledger)} others")
        self.ledger.append(a)
        if plan is None:  # another rank's group: only its place in the ledger
            return
        before = dict(self.trainer.seconds)
        t0, err = time.monotonic(), None
        try:
            self.trainer.stream_add(plan)
        except BaseException as exc:  # noqa: BLE001 - this group's error stops the step at the drain
            err = describe(exc)
        t1 = time.monotonic()
        self.put(("done", a.seq, t0, t1, {k: v - before.get(k, 0.0) for k, v in self.trainer.seconds.items()}, err))

    def cmd_finish(self) -> None:
        try:
            report = self.trainer.stream_finish(self.ledger)
            out = {"ok": True, "checks": {k: v.get("passed") for k, v in report["checks"].items()}}
        except C.TrainerCheckFailed as exc:
            out = {"ok": False, "kind": type(exc).__name__, "failed": list(exc.failed)}
        out["memory"] = memory_peak(self.device)
        self.put(("finish", self.rank, out))

    def cmd_report(self) -> None:
        self.trainer.stream_report()
        self.put(("report", self.rank, {"memory": memory_peak(self.device)}))

    def cmd_discard(self) -> None:
        self.trainer.stream_discard()
        self.ledger = []
        self.put(("discard", self.rank))

    def cmd_warmup(self, group: GroupPlan) -> None:
        before, version = state_digest(self.trainer), self.merged.version
        t0 = time.monotonic()
        with torch.random.fork_rng(devices=[torch.device(self.device)] if self.device.startswith("cuda") else []):
            report = self.trainer.warmup(group)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
        report["wall_s"] = time.monotonic() - t0
        if (state_digest(self.trainer) != before or self.merged.version != version
                or any(p.grad is not None for p in self.trainer.params)):
            raise RankError(f"the start-up warm-up changed rank {self.rank}'s state")
        report["state"] = "unchanged"
        report["memory"] = memory_peak(self.device)
        self.put(("warmup", self.rank, report))

    def cmd_load(self, directory: str) -> None:
        load_state(self.trainer, directory, self.device, rng=False)
        self.put(("load", self.rank, state_digest(self.trainer)))

    def cmd_remerge(self, version: int) -> None:
        remerge(self.merged, version)
        self.put(("remerge", self.rank, weights_checksum(self.merged)))

    def cmd_digest(self) -> None:
        self.put(("digest", self.rank, state_digest(self.trainer)))

    def cmd_weights(self) -> None:
        self.put(("weights", self.rank, weights_checksum(self.merged)))

    def cmd_tamper(self, what: str) -> None:
        """Tests only: plant a divergence on this rank, to check that the checks see it. "latent" moves one latent
        element now; "w_eff" makes this rank's merges differ from the others' by one bit from the next refresh on
        (a merge that is not the same function on every rank)."""
        if what == "latent":
            with torch.no_grad():
                self.trainer.params[0].view(-1)[0] += 1e-3
        elif what == "w_eff":
            merged, refresh = self.merged, self.merged.refresh

            def flipped():
                out = refresh()
                with torch.no_grad():
                    merged._weights[sorted(merged._weights)[0]].view(torch.int16).view(-1)[0] ^= 1
                return out

            merged.refresh = flipped
        else:
            raise RankError(f"unknown tamper {what!r}")
        self.put(("tamper", self.rank))


# --- the coordinator's view of each rank ------------------------------------------------------------------------


class _LocalRank:
    """Rank 0's groups, trained on their own thread in the order they were assigned. One thread per process trains at
    a time (`TRAIN_LOCK`): two pools in one process trained at once failed on GPU with "'NoneType' object is not a
    mapping" inside a group's forward and backward, most likely Triton's autotuner, which keeps per-call
    state on the kernel object (FLA's kernels are autotuned). The orchestrator has one such thread, so it never waits."""

    TRAIN_LOCK = threading.Lock()

    def __init__(self, trainer: RankTrainer, device: str, on_done: Callable):
        self.trainer, self.device, self._on_done = trainer, device, on_done
        self._q: queue.Queue = queue.Queue()
        self._skip = False
        self._idle = threading.Condition()
        self._busy = 0
        self._thread = threading.Thread(target=self._run, name="trainer-rank0", daemon=True)
        self._thread.start()

    def put(self, seq: int, plan: GroupPlan) -> None:
        with self._idle:
            self._busy += 1
        self._q.put((seq, plan))

    def _run(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        while True:
            item = self._q.get()
            if item is None:
                return
            seq, plan = item
            if self._skip:
                self._on_done(0, seq, None, None, {}, None)
            else:
                with self.TRAIN_LOCK:
                    before = dict(self.trainer.seconds)
                    t0, err = time.monotonic(), None
                    try:
                        self.trainer.stream_add(plan)
                    except BaseException as exc:  # noqa: BLE001 - reported at the drain
                        err = describe(exc)
                    t1 = time.monotonic()
                self._on_done(0, seq, t0, t1, {k: v - before.get(k, 0.0) for k, v in self.trainer.seconds.items()},
                              err)
            with self._idle:
                self._busy -= 1
                self._idle.notify_all()

    def flush(self, *, skip: bool) -> None:
        """Wait until every queued group has been handled: trained, or with `skip` dropped untrained."""
        self._skip = skip
        with self._idle:
            self._idle.wait_for(lambda: self._busy == 0)
        self._skip = False

    def stop(self, timeout_s: float = 600.0) -> None:
        """End the thread, dropping the groups still queued (the pool is going away): at most the group in hand is
        finished. A thread left training past the interpreter's exit can keep the process from ending."""
        self._skip = True
        self._q.put(None)
        self._thread.join(timeout=timeout_s)


class _RemoteRank:
    """A worker process, its command and reply queues, and a thread that reads its replies as they come. A command
    sent without waiting for its reply (`begin`, `assign`) that fails, and the process dying, go to `on_failure` at
    once, so a drain stops on them instead of waiting out its timeout."""

    ASYNC = ("begin", "assign")

    def __init__(self, ctx, spec_bytes: bytes, rank: int, on_done: Callable, on_failure: Callable):
        self.rank = rank
        self.cmd_q, self.reply_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=worker_main, args=(rank, spec_bytes, self.cmd_q, self.reply_q),
                                name=f"aorlta-trainer-rank{rank}", daemon=True)
        self.proc.start()
        self.replies: queue.Queue = queue.Queue()
        self.dead: str | None = None
        self._on_done, self._on_failure = on_done, on_failure
        self._rx = threading.Thread(target=self._receive, name=f"trainer-rank{rank}-replies", daemon=True)
        self._rx.start()

    def _receive(self) -> None:
        while True:
            try:
                msg = self.reply_q.get(timeout=POLL_S)
            except queue.Empty:
                if not self.proc.is_alive():
                    self._died(f"rank {self.rank}'s process exited (code {self.proc.exitcode})")
                    return
                continue
            except (EOFError, OSError) as exc:
                self._died(f"rank {self.rank}'s reply queue closed: {describe(exc)}")
                return
            if msg[0] == "done":
                self._on_done(self.rank, *msg[1:])
                continue
            if msg[0] == "error" and msg[1] in self.ASYNC:
                self._on_failure(self.rank, f"rank {self.rank}, {msg[1]}: {msg[2]}")
                continue
            self.replies.put(msg)
            if msg[0] in ("bye", "fatal"):
                return

    def _died(self, why: str) -> None:
        self.dead = why
        self.replies.put(("died", why))
        self._on_failure(self.rank, why)

    def send(self, *msg) -> None:
        if self.dead:
            raise RankError(self.dead)
        self.cmd_q.put(msg)

    def wait(self, expect: str, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise RankError(f"rank {self.rank}: no {expect!r} reply within {timeout_s:.0f} s")
            try:
                msg = self.replies.get(timeout=min(POLL_S, left))
            except queue.Empty:
                continue
            if msg[0] == expect:
                return msg
            if msg[0] == "died":
                raise RankError(msg[1])
            if msg[0] == "fatal":
                raise RankError(f"rank {self.rank} could not start: {msg[2]}")
            if msg[0] == "error":
                raise RankError(f"rank {self.rank}, {msg[1]}: {msg[2]}")
            raise RankError(f"rank {self.rank}: expected a {expect!r} reply, got {msg[0]!r}")


# --- the coordinator -----------------------------------------------------------------------------------------


class RankPool:
    """R ranks, rank 0 in this process (see the module docstring). `spans` receives each group's `train.stream`
    span; `merged_factory` builds rank 0's merged forward (the wiring's records its merges as spans)."""

    def __init__(self, spec: RankSpec, *, spans=None, merged_factory: Callable | None = None,
                 start_timeout_s: float = START_TIMEOUT_S, call_timeout_s: float = CALL_TIMEOUT_S):
        self.spec, self.spans, self.merged_factory = spec, spans, merged_factory
        self.start_timeout_s, self.call_timeout_s = start_timeout_s, call_timeout_s
        self.size = spec.size
        self.model = self.merged = self.trainer = None
        self.local: _LocalRank | None = None
        self.remote: list[_RemoteRank] = []
        self.iteration = 0
        self._cond = threading.Condition()
        self._reset_ledger()

    def _reset_ledger(self) -> None:
        self.ledger: list[Assignment] = []
        self.assigner = OnlineAssigner(self.size)
        self._groups: dict[int, tuple[GroupPlan, dict]] = {}
        self._sent: list[set[int]] = [set() for _ in range(self.size)]
        self._acked: list[set[int]] = [set() for _ in range(self.size)]
        self._errors: list[str] = []

    # --- start and stop ---

    def start(self) -> "RankPool":
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        spec_bytes = pickle.dumps(self.spec)
        try:
            self.remote = [_RemoteRank(ctx, spec_bytes, r, self._on_done, self._on_failure)
                           for r in range(1, self.size)]
            prepare_device(self.spec, 0)
            model = load_policy(self.spec, 0)
            for w in self.remote:  # every rank loaded before any collective, so a failed load cannot hang one
                w.wait("loaded", self.start_timeout_s)
            world = join_world(self.spec, 0)
            self.merged, self.trainer = make_trainer(self.spec, 0, model, world, self.merged_factory)
            for w in self.remote:
                w.wait("ready", self.start_timeout_s)
        except BaseException:
            self.shutdown()
            raise
        self.model = model
        self.local = _LocalRank(self.trainer, self.spec.devices[0], self._on_done)
        return self

    def shutdown(self, timeout_s: float = 60.0) -> None:
        if self.local is not None:
            self.local.stop()
            self.local = None
        for w in self.remote:
            try:
                w.send("quit")
            except RankError:
                pass
        for w in self.remote:
            w.proc.join(timeout_s)
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(10)
            if w.proc.is_alive():
                w.proc.kill()
                w.proc.join(10)
        self.remote = []
        if self.size > 1:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:  # noqa: BLE001 - shutting down
                pass

    # --- events from the ranks ---

    def _on_done(self, rank: int, seq: int, t0: float | None, t1: float | None, seconds: dict,
                 err: str | None) -> None:
        with self._cond:
            plan, meta = self._groups.get(seq, (None, {}))
            if plan is not None:
                self.assigner.done(rank, group_tokens(plan))
            self._acked[rank].add(seq)
            if err:
                self._errors.append(f"rank {rank}, group {seq}: {err}")
            self._cond.notify_all()
        if self.spans is None or plan is None or t0 is None:
            return
        wall, mono = self.spans.now(), time.monotonic()  # one host, one monotonic clock: its times are ours
        attrs = dict(meta, rank=rank, seq=seq, samples=len(plan.samples), tokens=sum(s.length for s in plan.samples),
                     forward_s=float(seconds.get("forward", 0.0)), backward_s=float(seconds.get("backward", 0.0)))
        if err:
            attrs["error"] = err.split(":", 1)[0]
        self.spans.record("train.stream", self.iteration, wall - (mono - t0), wall - (mono - t1), **attrs)

    def _on_failure(self, rank: int, why: str) -> None:
        with self._cond:
            self._errors.append(why)
            self._cond.notify_all()

    def _on_rank0_device(self):
        """Rank 0's GPU as the calling thread's current device: RCCL's object gathers run on the current device,
        and any thread may call the pool (or another pool in this process may have moved it)."""
        import contextlib

        dev = self.spec.devices[0]
        return torch.cuda.device(dev) if dev.startswith("cuda") else contextlib.nullcontext()

    def _usable(self) -> None:
        if self.trainer is None:
            raise RankError("the rank pool was not started")
        dead = [w.dead for w in self.remote if w.dead]
        if dead:
            raise RankError(dead[0])

    # --- one iteration ---

    def begin(self, n: int) -> None:
        self._usable()
        with self._cond:
            self._reset_ledger()
            self.iteration = n
        self.merged.version = n - 1  # the step's refresh makes version n
        self.trainer.stream_begin()
        for w in self.remote:
            w.send("begin", n)

    def assign(self, plan: GroupPlan, meta: dict | None = None) -> int:
        """Give one complete group to a rank (returned). The assignment goes to every rank, the samples to one."""
        if not plan.samples:
            raise TrainerInputError(f"group {plan.key} has no samples to train")
        self._usable()
        light = without_turns(plan)
        with self._cond:  # held while sending, so every rank receives the assignments in seq order
            r = self.assigner.assign(plan)
            a = Assignment.of(len(self.ledger), plan, r)
            self.ledger.append(a)
            self._groups[a.seq] = (plan, dict(meta or {}))
            self._sent[r].add(a.seq)
            for w in self.remote:
                w.send("assign", tuple(a), light if w.rank == r else None)
        if r == 0:
            self.local.put(a.seq, plan)
        return r

    def drain(self, timeout_s: float | None = None) -> float:
        """Wait until every rank has trained every group it was given; seconds waited. Raises if a rank failed or
        died: before any collective, so no rank is left waiting in one."""
        t0 = time.monotonic()
        deadline = t0 + (self.call_timeout_s if timeout_s is None else timeout_s)
        with self._cond:
            while True:
                if self._errors:
                    raise RankError("; ".join(self._errors[:3]))
                if all(self._sent[r] <= self._acked[r] for r in range(self.size)):
                    return time.monotonic() - t0
                left = deadline - time.monotonic()
                if left <= 0:
                    pending = {r: sorted(self._sent[r] - self._acked[r]) for r in range(self.size)}
                    raise RankError(f"groups still training after {timeout_s or self.call_timeout_s:.0f} s: {pending}")
                self._cond.wait(min(POLL_S, left))

    def finish(self) -> dict:
        """The step on every rank (see the module docstring); rank 0's report, with `drain_sec` and `by_rank`.
        Raises `TrainerCheckFailed` (or `StepRefused`) if any rank's check failed."""
        drain_s = self.drain()
        for w in self.remote:
            w.send("finish")
        local_failure, report = None, None
        try:
            with self._on_rank0_device():
                report = self.trainer.stream_finish(self.ledger)
        except C.TrainerCheckFailed as exc:
            local_failure, report = exc, exc.report
        replies = [w.wait("finish", self.call_timeout_s)[2] for w in self.remote]
        report["drain_sec"] = drain_s
        report["by_rank"] = [{"rank": 0, "groups": sum(1 for a in self.ledger if a.rank == 0),
                              "samples": sum(a.samples for a in self.ledger if a.rank == 0),
                              "memory": memory_peak(self.spec.devices[0])}]
        for w, out in zip(self.remote, replies):
            report["by_rank"].append({"rank": w.rank, "groups": sum(1 for a in self.ledger if a.rank == w.rank),
                                      "samples": sum(a.samples for a in self.ledger if a.rank == w.rank),
                                      "memory": out.get("memory"), "ok": out["ok"]})
        if local_failure is not None:
            raise local_failure
        failed = sorted({name for out in replies if not out["ok"] for name in out["failed"]})
        if failed:
            refused = all(out.get("kind") == "StepRefused" for out in replies if not out["ok"])
            raise (C.StepRefused if refused else C.TrainerCheckFailed)(failed, report)
        return report

    def report(self) -> dict:
        """What the open step has measured, without stepping (rank 0's `stream_report`, gathered from all)."""
        self.drain()
        for w in self.remote:
            w.send("report")
        with self._on_rank0_device():
            rep = self.trainer.stream_report()
        replies = [w.wait("report", self.call_timeout_s)[2] for w in self.remote]
        rep["by_rank"] = [{"rank": 0, "memory": memory_peak(self.spec.devices[0])}] + [
            {"rank": w.rank, "memory": out.get("memory")} for w, out in zip(self.remote, replies)]
        return rep

    def discard(self) -> None:
        """Drop the open step on every rank. Rank 0 drops the groups still queued untrained; a worker trains its
        queue first (it reads its commands in order)."""
        if self.local is not None:
            self.local.flush(skip=True)
        if self.trainer is not None:
            self.trainer.stream_discard()
        problems = []
        for w in self.remote:
            try:
                w.send("discard")
                w.wait("discard", self.call_timeout_s)
            except RankError as exc:
                problems.append(str(exc))
        with self._cond:
            self._reset_ledger()
        if problems:
            raise RankError("; ".join(problems[:3]))

    # --- set-up, checkpoints, the merged weights ---

    def warmup(self, group: GroupPlan, local: Callable[[], dict]) -> list[dict]:
        """`trainer.startup_warmup` on every rank at once: `local` runs rank 0's (the wiring's, with its state
        check); each worker checks its own state and refuses if it moved."""
        light = without_turns(group)
        for w in self.remote:
            w.send("warmup", light)
        try:
            first = local()
        except BaseException:
            for w in self.remote:  # their replies, so that none is read as the answer to a later command
                try:
                    w.wait("warmup", self.call_timeout_s)
                except RankError:
                    pass
            raise
        return [first] + [w.wait("warmup", self.call_timeout_s)[2] for w in self.remote]

    def load_state(self, directory) -> str:
        """Every rank loads the same checkpoint; refuses unless every rank's trained state is then rank 0's."""
        self._usable()
        for w in self.remote:
            w.send("load", str(directory))
        load_state(self.trainer, directory, self.spec.devices[0])
        mine = state_digest(self.trainer)
        theirs = [w.wait("load", self.call_timeout_s)[2] for w in self.remote]
        if any(d != mine for d in theirs):
            raise RankError("the ranks loaded different trained states")
        return mine

    def remerge(self, version: int) -> None:
        """Re-merge every rank's latent as `version`; refuses unless every rank's W_eff is rank 0's, bit for bit."""
        self._usable()
        for w in self.remote:
            w.send("remerge", version)
        with self._on_rank0_device():
            remerge(self.merged, version)
        mine = weights_checksum(self.merged) if self.remote else None
        theirs = [w.wait("remerge", self.call_timeout_s)[2] for w in self.remote]
        if any(s != mine for s in theirs):
            raise RankError(f"the ranks' merged weights differ after re-merging version {version}")

    def state_digest(self) -> str:
        """Rank 0's trained-state digest; refuses unless every rank's is the same."""
        self._usable()
        for w in self.remote:
            w.send("digest")
        mine = state_digest(self.trainer)
        theirs = [w.wait("digest", self.call_timeout_s)[2] for w in self.remote]
        if any(d != mine for d in theirs):
            raise RankError("the ranks' trained states differ (latent or optimizer)")
        return mine

    def weights_checksums(self) -> list:
        """Every rank's W_eff checksum, rank 0 first (for checks and tests)."""
        self._usable()
        for w in self.remote:
            w.send("weights")
        mine = weights_checksum(self.merged)
        return [mine] + [w.wait("weights", self.call_timeout_s)[2] for w in self.remote]

    def tamper(self, rank: int, what: str) -> None:
        """Tests only: plant a divergence on worker `rank`."""
        w = self.remote[rank - 1]
        w.send("tamper", what)
        w.wait("tamper", self.call_timeout_s)
