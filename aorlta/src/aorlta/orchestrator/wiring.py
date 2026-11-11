"""The real components behind the orchestrator's ports (Section 8, step 2 of the integration).

C1 is WP2's engine pool and rollout manager (`RolloutAdapter`), C2 WP5's reward service with WP6's log-only
selector hook inside it (`RewardAdapter`), C4 WP4's rank trainer over WP3's merged-LoRA latent
(`TrainerAdapter`), and C5 WP3's publication, landing check and round trip (`SyncAdapter`). C9's CLIs read the
run directory these write. The fakes in `fakes.py` stay what the orchestrator's own tests use.

The trainer runs `trainer.ranks` ranks (D10): rank r beside engine r on its GPU, or on `trainer.devices` for a
separate layout. Rank 0 is in this process, the others are spawned processes (`aorlta.trainer.rank_pool`); each
group goes to one rank as it completes, and the step is one all-reduce and one optimizer step on every rank.

What this wiring does not do, each refused by `check_supported` or stated where it bites:
  - streaming accumulation only, one step per iteration;
  - stand-in prompts only, with the held-out kernels left out of training;
  - no eval exporter (`orchestrator.eval_every` must be null);
  - the selector off or log-only with no planner endpoint;
  - the generation deadline stops chains from starting another turn, but does not abort a request already running
    (it ends at `rollout.request_timeout_s`): TokenSpeed's offline engine has no abort this wiring can drive safely;
  - an engine that dies is restarted out of service (WP2 needs a landing check before it serves again), so the
    next iteration stops.

Each chain is handed to the orchestrator as it ends, so groups stream to the trainer during generation; D7's retry
re-checks a candidate's unsettled legs through the reward service's `resubmit`. With `reward.remote`, the named
adapters run in their own checker processes (`aorlta.reward.remote`), which `jobs/aorlta-run.sbatch` starts.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..atomic import atomic_write_bytes
from ..config import RunConfig
from ..rollout.engine_pool import SELF_TEST_PROMPTS
from .ports import BarrierViolation, ChainEnd, GroupInput, GroupOutcome, RunStopped, describe_error

log = logging.getLogger(__name__)

LORA_R, LORA_ALPHA = 16, 32  # Section 6.4: a fresh r16/alpha32 LoRA
PROBE_KERNELS = 40  # WP3's probes are the last `sync.probes` of these stand-in kernels
WORKER_THREADS = 8  # CPU threads per spawned trainer rank: its CPU work is small, and the compile pool shares the node
ENGINE_CALL_TIMEOUT_S = 600.0  # a probe, a greedy draw or a reload, as EngineProc's own reload default
FAKE_FAIL_RATE = 0.05  # per leg, in the fake profile: enough failures that rewards differ within a group
STANDIN_REWARD_FNS = ("standin.v11",)
MERGE_TOOL = Path(__file__).resolve().parents[3] / "tools" / "merge_base_adapter.py"
MERGE_MANIFEST = "merge-manifest.json"  # what the merge tool writes beside its output
MERGE_TIMEOUT_S = 3600.0


class WiringError(RuntimeError):
    """The configuration asks for something this wiring cannot run, or a component could not be set up."""


class EngineCallError(RuntimeError):
    """An engine call returned an error instead of a reply."""


def check_supported(cfg: RunConfig) -> None:
    """Refuse, before anything starts, every configuration this wiring would otherwise run wrongly."""
    problems = []
    if cfg.run.source != "standin":
        problems.append(f"run.source is {cfg.run.source!r}: only stand-in runs are wired")
    t = cfg.trainer
    if cfg.orchestrator.mode == "score":
        problems += score_problems(cfg)  # no trainer is built: its settings and layout are not read
    else:
        problems += layout_problems(cfg)
        if not t.streaming_accumulation:
            problems.append("trainer.streaming_accumulation is false: only the streamed step is wired")
        if t.steps_per_iteration != 1:
            problems.append(f"trainer.steps_per_iteration is {t.steps_per_iteration}: streaming takes one step")
    if cfg.orchestrator.eval_every is not None:
        problems.append("orchestrator.eval_every is set: there is no eval exporter yet")
    pools = cfg.data.prompt_pools
    if len(pools) != 1 or pools[0].source != "standin":
        problems.append("data.prompt_pools must be exactly one stand-in pool")
    if cfg.reward.reward_fn not in STANDIN_REWARD_FNS:
        problems.append(f"reward.reward_fn is {cfg.reward.reward_fn!r}: only standin.v11 is wired")
    if cfg.selector.endpoint is not None:
        problems.append("selector.endpoint is set: no planner is wired (log-only runs without one)")
    if cfg.rollout.P < SELF_TEST_PROMPTS:
        problems.append(f"rollout.P is {cfg.rollout.P}: the engines' start-up self-test runs on the first "
                        f"iteration's prompts and needs {SELF_TEST_PROMPTS}")
    if cfg.reward.remote is not None:
        from ..reward.profiles import load_profile

        served = {p.adapter for p in load_profile(cfg.reward.profile).pools.values()}
        stray = sorted(set(cfg.reward.remote.adapters) - served)
        if stray:
            problems.append(f"reward.remote names {stray}, which profile {cfg.reward.profile!r} does not use")
    if problems:
        raise WiringError("; ".join(problems))


def score_problems(cfg: RunConfig) -> list[str]:
    """What a score run needs besides what every run does: the first batch holds the engines' self-test prompts, and
    an adapter needs the merge tool beside the package."""
    out = []
    limit = cfg.orchestrator.score.prompts
    first = cfg.rollout.P if limit is None else min(cfg.rollout.P, limit)
    if first < SELF_TEST_PROMPTS:
        out.append(f"the first batch of the score run has {first} prompts (rollout.P {cfg.rollout.P}, "
                   f"orchestrator.score.prompts {limit}): the engines' start-up self-test runs on them and needs "
                   f"{SELF_TEST_PROMPTS}")
    if cfg.model.adapter_dir is not None and not MERGE_TOOL.is_file():
        out.append(f"model.adapter_dir is set and the merge tool is not at {MERGE_TOOL}")
    return out


def served_model_dir(cfg: RunConfig) -> Path:
    """The directory the engines load: `model.base_dir`, or, with `model.adapter_dir` (a score run), the merge of the
    two that tools/merge_base_adapter.py writes into the run's node-local directory."""
    return Path(cfg.model.base_dir) if cfg.model.adapter_dir is None else node_dir(cfg) / "model"


def trainer_gpus(cfg: RunConfig) -> tuple[int, ...]:
    """Rank r's GPU: `trainer.devices[r]`, or engine r's GPU when that is unset (colocated, D10)."""
    t = cfg.trainer
    return tuple(t.devices) if t.devices is not None else tuple(cfg.rollout.engines[:t.ranks])


def trainer_memory_fraction(cfg: RunConfig, gpu: int) -> float | None:
    """A rank beside an engine may use what the engine leaves (1 - gpu_memory_utilization); alone, no cap."""
    if gpu in cfg.rollout.engines:
        return max(0.05, 1.0 - cfg.rollout.gpu_memory_utilization)
    return None


def layout_problems(cfg: RunConfig) -> list[str]:
    from ..reward.remote import servers

    t, engines = cfg.trainer, cfg.rollout.engines
    out = []
    if t.devices is None and t.ranks > len(engines):
        out.append(f"trainer.ranks is {t.ranks}: colocated, each rank sits beside one engine and there are "
                   f"{len(engines)} (set trainer.devices for a separate layout)")
    checkers = {g for _, g, _ in servers(cfg.reward.remote)} if cfg.reward.remote is not None else set()
    clash = sorted(set(trainer_gpus(cfg)) & checkers)
    if clash:
        out.append(f"trainer GPUs {clash} serve the remote checkers (reward.remote)")
    return out


def merged_lora_kwargs(cfg: RunConfig) -> dict:
    """`sync.stochastic_rounding` as MergedLoRA's arguments. The rounding bits come
    from the run seed, so a resumed run merges every version to the same weights."""
    return {"rounding": "stochastic" if cfg.sync.stochastic_rounding else "nearest",
            "run_seed": cfg.orchestrator.run_seed}


def standin_dataset_dir(path: str | Path) -> Path:
    """The stand-in dataset for a configured pool path: the path or its parent (the configs name
    `<dataset>/listings`), whichever holds both `kernels.json` and `listings/`."""
    p = Path(path)
    for d in (p, p.parent):
        if (d / "kernels.json").is_file() and (d / "listings").is_dir():
            return d
    raise WiringError(f"{path}: neither it nor its parent holds kernels.json and listings/")


def node_dir(cfg: RunConfig) -> Path:
    """This run's node-local directory, which also holds its weight slots: `<sync.dir>/<run_id>`."""
    return Path(cfg.sync.dir) / cfg.run.run_id


def scratch_dir(cfg: RunConfig) -> Path:
    return node_dir(cfg) / "scratch"


def remove_node_dir(cfg: RunConfig) -> None:
    shutil.rmtree(node_dir(cfg), ignore_errors=True)


# --- C1 ------------------------------------------------------------------------------------------


class StandinPromptPool:
    """The PromptPool port over the stand-in dataset: every kernel with 60-400 instructions that is not held out."""

    source = "standin"

    def __init__(self, tokenizer, thinking: str, dataset: str | Path):
        self.tokenizer, self.thinking, self.dataset = tokenizer, thinking, Path(dataset)
        self._records: list | None = None

    def records(self) -> list:
        if self._records is None:
            from ..rollout.templates import standin_training_prompts

            self._records = standin_training_prompts(self.tokenizer, self.thinking, self.dataset)
        return list(self._records)


class CountedHandle:
    """An engine handle that counts its `generate` calls in flight; everything else passes through. The
    generation barrier and the reload barrier read the sum over the pool."""

    def __init__(self, handle):
        self.handle = handle
        self._busy = 0
        self._lock = threading.Lock()

    @property
    def busy(self) -> int:
        with self._lock:
            return self._busy

    def generate(self, *args, **kwargs):
        with self._lock:
            self._busy += 1
        try:
            return self.handle.generate(*args, **kwargs)
        finally:
            with self._lock:
                self._busy -= 1

    def __getattr__(self, name: str):
        return getattr(self.handle, name)


def rollout_max_len(cfg: RunConfig, tokenizer, records: Sequence) -> int:
    """The engines' context: the longest turn 0 of the pool, plus every repair turn's feedback and generation."""
    from ..rollout.engine_pool import max_model_len
    from ..rollout.feedback import feedback_token_bound
    from ..rollout.templates import turn0_ids

    r = cfg.rollout
    longest = max(len(turn0_ids(p, tokenizer, r.thinking)) for p in records)
    return max_model_len(longest, r.max_new_tokens, feedback_token_bound(tokenizer, r.max_feedback_chars, r.thinking),
                         r.max_repair_turns)


def engine_handles(cfg: RunConfig, max_len: int) -> list:
    """One TokenSpeed engine process per configured GPU, as WP2's harness starts them, on `served_model_dir`."""
    from ..rollout.engine_pool import engine_kwargs, engine_port
    from ..rollout.engine_proc import EngineProc

    r = cfg.rollout
    kw = engine_kwargs(model_dir=str(served_model_dir(cfg)), dtype="bfloat16", P=r.P, K=r.K, n_engines=len(r.engines),
                       max_len=max_len, gpu_memory_utilization=r.gpu_memory_utilization,
                       run_seed=cfg.orchestrator.run_seed)
    return [EngineProc(i, str(g), dict(kw, port=engine_port(i))) for i, g in enumerate(r.engines)]


def rollout_compiler(cfg: RunConfig):
    from ..rollout.compile_check import CompileUnavailable, EvidenceStore, make_compiler

    try:
        return make_compiler(cfg.rollout.compile_adapter, scratch_dir(cfg) / "compile",
                             EvidenceStore(cfg.run_dir / "evidence" / "rollout"))
    except CompileUnavailable as exc:
        raise WiringError(f"rollout.compile_adapter {cfg.rollout.compile_adapter!r}: {exc}") from None


class RolloutAdapter:
    """C1 behind the RolloutPort."""

    def __init__(self, cfg: RunConfig, tokenizer, handles: Sequence[Any], *, max_len: int, spans, compiler,
                 on_shutdown: Sequence[Callable[[], None]] = (), clock=None):
        from ..rollout.engine_pool import EnginePool

        self.cfg, self.tok, self.spans, self.max_len, self.compiler = cfg, tokenizer, spans, max_len, compiler
        self.clock = clock  # the orchestrator's clock: its deadlines are read on it
        self.handles = [CountedHandle(h) for h in handles]
        self.pool = EnginePool(self.handles, run_seed=cfg.orchestrator.run_seed,
                               request_timeout_s=cfg.rollout.request_timeout_s, clock=spans.now)
        self.manager = None
        self._on_shutdown = list(on_shutdown)

    def start(self, selftest_prompts: Sequence) -> dict:
        from ..rollout.chain import RolloutManager, RolloutSettings, make_leak_check
        from ..rollout.engine_pool import RolloutStop
        from ..rollout.templates import turn0_ids

        r = self.cfg.rollout
        try:
            self.pool.start([turn0_ids(p, self.tok, r.thinking) for p in selftest_prompts])
        except RolloutStop as exc:
            raise RunStopped("engine_start_failed", {"error": describe_error(exc, self.cfg.run.source)}) from None
        settings = RolloutSettings.from_config(self.cfg, self.max_len)
        if self.cfg.orchestrator.mode == "score":  # a pass's last batch may hold fewer than P prompts
            settings = dataclasses.replace(settings, P=None)
        self.manager = RolloutManager(settings, self.pool, self.tok,
                                      self.compiler, leak_check=make_leak_check(r.leak_check_adapter),
                                      spans=self.spans, compile_workers=min(96, os.cpu_count() or 8))
        return {"engines": len(self.handles), "max_model_len": self.max_len, "self_tests": self.pool.self_tests}

    def run(self, prompts, iteration: int, policy_version: str, *, on_candidate, on_chain=None, deadline=None):
        """Each chain goes to `on_chain` as it ends (its candidate with it), so groups train during generation.
        `deadline` is on the orchestrator's clock; the rollout reads time on the span recorder's."""
        if self.manager is None:
            raise RuntimeError("the rollout was not started")
        cutoff = None
        if deadline is not None:
            if self.clock is None:
                raise RuntimeError("a generation deadline needs the orchestrator's clock")
            cutoff = self.spans.now() + (deadline - self.clock.now())
        on_end = None
        if on_chain is not None:
            def on_end(o):
                on_chain(ChainEnd(o.record, o.candidate, o.sample))
        return self.manager.run(prompts, iteration, policy_version,
                                on_candidate=on_candidate if on_chain is None else None,
                                on_chain_end=on_end, deadline=cutoff)

    def in_flight(self) -> int:
        return sum(h.busy for h in self.handles)

    def shutdown(self) -> None:
        try:
            self.pool.shutdown()
        finally:
            for fn in self._on_shutdown:
                fn()


# --- C5's view of an engine ----------------------------------------------------------------------


class SyncEngine:
    """A pool engine as WP3's sync code expects one: `engine_id`, `update_weights_from_disk`, and a
    TokenSpeed-shaped `generate` for the landing check's probes and the round trip's greedy draws."""

    def __init__(self, pool, index: int, *, timeout_s: float = ENGINE_CALL_TIMEOUT_S):
        self.pool, self.engine_id, self.timeout_s = pool, index, timeout_s

    @property
    def handle(self):
        return self.pool.handles[self.engine_id]

    def update_weights_from_disk(self, model_path, flush_cache: bool = True):
        res = self.handle.reload(str(model_path), flush_cache=bool(flush_cache), timeout_s=self.timeout_s)
        if "error" in res:
            raise EngineCallError(f"engine {self.engine_id} reload: {str(res['error'])[:300]}")
        if res.get("ok") and flush_cache:
            self.pool.note_flush(self.engine_id)
        value = res.get("value")
        return tuple(value) if value is not None else res.get("reply")

    def generate(self, input_ids, sampling_params, return_logprob: bool = True, logprob_start_len=None):
        extra = {} if logprob_start_len is None else {"logprob_start_len": logprob_start_len}
        res = self.handle.generate(list(input_ids), dict(sampling_params), timeout_s=self.timeout_s, extra=extra)
        if "error" in res:
            raise EngineCallError(f"engine {self.engine_id} generate: {str(res['error'])[:300]}")
        meta = {"cached_tokens": res.get("cached_tokens"), "finish_reason": res.get("finish_reason")}
        vals, ids = res.get("in_lp_vals"), res.get("in_lp_ids")
        if vals is not None and ids is not None:
            meta["input_token_logprobs"] = [(v, t, None) for v, t in zip(vals, ids)]
        return {"output_ids": list(res.get("output_ids") or []), "meta_info": meta}


# --- C2 + C3 -------------------------------------------------------------------------------------


def fake_reward_factories(cfg: RunConfig, clock) -> dict:
    """The fake profile's checkers, for CPU runs of this wiring, as WP5's fake report builds them."""
    from ..reward.adapters.base import SequentialBatch
    from ..reward.adapters.fake import FakeAdapterConfig, FakeCheckAdapter, FakeLegProfile

    base = FakeAdapterConfig(seed=cfg.orchestrator.run_seed, default=FakeLegProfile(fail_rate=FAKE_FAIL_RATE))
    asan = dataclasses.replace(base, legs=("san.asan",),
                               per_leg={k: v for k, v in base.per_leg.items() if k == "san.asan"},
                               overrides={k: v for k, v in base.overrides.items() if k[1] == "san.asan"})
    return {"fake": lambda w: FakeCheckAdapter(base, clock=clock.wall),
            "fake-asan": lambda w: SequentialBatch(FakeCheckAdapter(asan, clock=clock.wall))}


def remote_factories(cfg: RunConfig, clock) -> dict:
    """`reward.remote`: a RemoteAdapter factory for each named adapter, in place of its in-process one. An adapter on
    several GPUs gets a RemoteFactory, which spreads its workers over the servers."""
    from ..reward.remote import RemoteAdapter, RemoteFactory, servers, socket_path

    remote = cfg.reward.remote
    if remote is None:
        return {}
    out = {}
    for name in remote.adapters:
        paths = [path for adapter, _, path in servers(remote) if adapter == name]
        if len(paths) > 1:
            out[name] = RemoteFactory(paths, name, connect_timeout_s=remote.connect_timeout_s, clock=clock.wall)
        else:
            out[name] = (lambda wid, name=name: RemoteAdapter(socket_path(remote.socket_dir, name), name, wid,
                                                              connect_timeout_s=remote.connect_timeout_s,
                                                              clock=clock.wall))
    return out


class RewardAdapter:
    """C2 behind the RewardPort, with C3's log-only hook inside it. The service starts its workers (and the
    pools' controls) when it is made, so it is made on the first `reopen`, after the run plan is written."""

    def __init__(self, cfg: RunConfig, *, clock, spans, factories: dict | None = None):
        from ..reward.profiles import load_profile
        from ..reward.rewardfn.for_profile import UnsupportedRewardFn, standin_rewardfn

        self.cfg, self.clock, self.spans = cfg, clock, spans
        self.profile = load_profile(cfg.reward.profile)
        try:
            self.rewardfn = standin_rewardfn(self.profile)
        except UnsupportedRewardFn as exc:
            raise WiringError(str(exc)) from None
        if self.profile.reward_fn != self.rewardfn.name:
            raise WiringError(f"profile {self.profile.name!r} is scored by {self.profile.reward_fn}, "
                              f"not {self.rewardfn.name}")
        if factories is None:
            if self.profile.name == "fake":
                factories = fake_reward_factories(cfg, clock)
            else:
                from ..reward.factories import standin_factories

                factories = standin_factories(self.profile, clock, timeouts=dataclasses.asdict(cfg.reward.timeouts),
                                              check_s=cfg.reward.gpu_check_s)
            factories.update(remote_factories(cfg, clock))
        missing = sorted({p.adapter for p in self.profile.pools.values()} - set(factories))
        if missing:
            raise WiringError(f"profile {self.profile.name!r} needs adapters with no factory: {missing}")
        self.factories = factories
        self.work_dir = scratch_dir(cfg) / "reward"
        self.plan_log = cfg.run_dir / "selector" / "plan_log.jsonl"
        self._svc = None
        self._lock = threading.Lock()

    def service(self):
        with self._lock:
            if self._svc is None:
                from ..reward.service import RewardService
                from ..selector.log_only import selector_hook

                hook = selector_hook(self.cfg.selector, None, audit_seed=self.cfg.orchestrator.run_seed)
                self._svc = RewardService(self.cfg.reward, self.profile, self.rewardfn, self.factories,
                                          cache_dir=self.cfg.run_dir / "reward" / "cache", work_dir=self.work_dir,
                                          clock=self.clock, spans=self.spans, selector=hook)
            return self._svc

    def submit(self, candidate):
        return self.service().submit(candidate)

    def resubmit(self, candidate):
        return self.service().resubmit(candidate)

    def drain(self, deadline_monotonic: float) -> Any:
        return self.service().drain(deadline_monotonic)

    def reopen(self) -> None:
        self.service().reopen()

    def stats(self) -> dict:
        svc = self.service()
        self._write_plan_log()
        return svc.stats()

    def claims_clean(self, result) -> bool:
        """A score run's clean: a scored result whose reward function claims the clean reward."""
        return result.status == "scored" and bool(self.rewardfn.claims_clean(result.reward, result.parts))

    def not_applicable(self, result, candidate) -> tuple[str, ...]:
        """The legs the reward function did not need for this record (for example ConSan on a launch with no
        LDS), for the score report's counts."""
        rule = getattr(self.rewardfn, "not_applicable_legs", None)
        if rule is None or candidate is None:
            return ()
        try:
            return tuple(sorted(rule(result.legs, candidate)))
        except Exception:  # noqa: BLE001 - nothing is counted not applicable then; the score stands
            return ()

    def _write_plan_log(self) -> None:
        hook = self._svc.selector if self._svc is not None else None
        if hook is not None:
            self.plan_log.parent.mkdir(parents=True, exist_ok=True)
            hook.write_plan_log(self.plan_log)

    def close(self) -> None:
        if self._svc is None:
            return
        try:
            self._svc.close()
            self._write_plan_log()
        finally:
            self.keep_evidence()
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def keep_evidence(self) -> Path | None:
        """The GPU legs' logs, harness sources and parsed sources, into the run's evidence store, before the scratch
        that holds them is removed (`<run_dir>/evidence/reward/gpu-legs-<k>.tgz`, one per attempt). A failure to
        archive is logged; it never stops the shutdown."""
        from ..reward.evidence import archive_legs, next_archive

        try:
            dest = next_archive(self.cfg.run_dir / "evidence" / "reward")
            n = archive_legs(self.work_dir / "pkg", dest)
            if n:
                log.info("kept the GPU legs' evidence: %d files in %s", n, dest)
                return dest
        except Exception as exc:  # noqa: BLE001
            log.warning("keeping the GPU legs' evidence failed: %s", type(exc).__name__)
        return None


# --- C4 ------------------------------------------------------------------------------------------


def load_policy_model(base_dir: str, device: str):
    """The bf16 base as WP4 loads it, with the log watcher that `prepare_model` reads (`rank_pool`'s loader, which
    the spawned ranks import by name)."""
    from ..trainer.rank_pool import load_policy_model as load

    return load(base_dir, device)


def _recorded_merged_lora():
    from ..trainer.merged_lora import MergedLoRA

    class RecordedMergedLoRA(MergedLoRA):
        """MergedLoRA whose every refresh is a `merge` span, tagged with the version it makes."""

        def __init__(self, *args, spans, **kwargs):
            self._spans = spans
            self.span_iteration = None
            super().__init__(*args, **kwargs)

        def refresh(self):
            it = self.version + 1 if self.span_iteration is None else self.span_iteration
            with self._spans.span("merge", it, rounding=self.rounding):
                return super().refresh()

    return RecordedMergedLoRA


def trainer_devices(cfg: RunConfig, device: str | None = None) -> tuple[str, ...]:
    """Each rank's torch device: `cuda:<gpu>` from `trainer_gpus`, or every rank on the CPU (`device="cpu"`, tests).
    Any other `device` names rank 0's device and is allowed for one rank only."""
    ranks = cfg.trainer.ranks
    if device == "cpu":
        return ("cpu",) * ranks
    if device is not None:
        if ranks != 1:
            raise WiringError(f"a trainer device override ({device}) is for one rank; trainer.ranks is {ranks}")
        return (device,)
    return tuple(f"cuda:{g}" for g in trainer_gpus(cfg))


class TrainerAdapter:
    """C4 behind the TrainerPort: `trainer.ranks` ranks, streaming (Section 6.4; D10 for the layout). Rank 0 is in
    this process; the others are spawned when the model is first needed (`rank_pool.RankPool`). Each group goes to
    one rank as it completes, and that rank records the group's `train.stream` span."""

    records_train_spans = True  # the loop's trainer thread only hands groups over; the ranks time them

    def __init__(self, cfg: RunConfig, *, spans, load_model: Callable | None = None, device: str | None = None,
                 require_fast_path: bool | None = None, amp: bool = True):
        self.cfg, self.spans = cfg, spans
        self.load_model = load_model or load_policy_model
        self.devices = trainer_devices(cfg, device)
        self.device = self.devices[0]
        self.require_fast_path = cfg.trainer.require_fast_path if require_fast_path is None else require_fast_path
        self.amp = amp
        self.pool = None
        self.trainer = None  # rank 0's RankTrainer
        self.merged = None  # rank 0's merged forward: what C5 writes and the landing check runs
        self.model = None
        self.iteration: int | None = None
        self.warmup_ids: Callable[[], Sequence[Sequence[int]]] | None = None  # token ids for `warmup`; `build` sets it

    def rank_spec(self):
        """What every rank is built from (`rank_pool.RankSpec`)."""
        from ..sync.merge import TARGET_MODULE_PATTERN
        from ..trainer.loss import LossSettings
        from ..trainer.rank_pool import RankSpec, file_init_method
        from ..trainer.ranks import OptimSettings

        cfg, t = self.cfg, self.cfg.trainer
        on_cpu = all(d == "cpu" for d in self.devices)
        fractions = tuple(trainer_memory_fraction(cfg, int(d.split(":", 1)[1])) if d.startswith("cuda:") else None
                          for d in self.devices)
        return RankSpec(
            base_dir=cfg.model.base_dir, devices=self.devices, memory_fractions=fractions,
            loss=LossSettings.from_trainer(t, min_scored_per_group=cfg.orchestrator.min_scored_per_group),
            optim=OptimSettings.from_trainer(t), micro_batch_tokens=t.micro_batch_tokens, load_model=self.load_model,
            merged_kwargs=merged_lora_kwargs(cfg), lora_r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULE_PATTERN, seed=cfg.orchestrator.run_seed, amp=self.amp,
            gap_gate=t.gap_gate == "enforce", require_fast_path=self.require_fast_path,
            grad_checkpointing=t.grad_checkpointing, backend="gloo" if on_cpu else "nccl",
            init_method=file_init_method(node_dir(cfg)) if t.ranks > 1 else "",  # removed with the node directory
            worker_threads=2 if on_cpu else WORKER_THREADS)

    def ensure(self) -> "TrainerAdapter":
        if self.pool is not None:
            return self
        import functools

        from ..trainer.rank_pool import RankPool

        pool = RankPool(self.rank_spec(), spans=self.spans,
                        merged_factory=functools.partial(_recorded_merged_lora(), spans=self.spans)).start()
        self.pool, self.trainer, self.merged, self.model = pool, pool.trainer, pool.merged, pool.model
        return self

    def shutdown(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    # the iteration

    def begin(self, iteration: int) -> None:
        self.ensure()
        self.iteration = iteration
        self.pool.begin(iteration)  # every rank: the step's refresh makes version `iteration`

    def add_group(self, group: GroupInput) -> GroupOutcome:
        from ..trainer.loss import build_groups

        members = group.members
        plans = build_groups([m.sample for m in members if m.sample is not None],
                             [m.result for m in members if m.result is not None],
                             min_scored=self.cfg.orchestrator.min_scored_per_group, chains=group.chains,
                             rerendered=self.cfg.rollout.repair_turn_ids == "rerender")
        excluded = {}
        for m in members:
            if m.result is None:
                excluded[m.chain_id] = m.record.end_reason
            elif m.result.status != "scored":
                excluded[m.chain_id] = "unscored"
        scored = sum(1 for m in members if m.result is not None and m.result.status == "scored")
        if not plans:
            return GroupOutcome(group.prompt_id, scored, None, "no_group", {}, (), excluded)
        if len(plans) != 1:
            raise RuntimeError(f"group {group.prompt_id} split into {len(plans)} trainer groups")
        plan = plans[0]
        if plan.dropped is not None:
            return GroupOutcome(group.prompt_id, scored, None, plan.dropped, {}, (), excluded)
        advantages = {m.chain_id: m.result.reward - plan.mean_reward for m in members
                      if m.result is not None and m.result.status == "scored"}
        trained = tuple(s.chain_id for s in plan.samples)
        for m in members:
            if m.chain_id in advantages and m.chain_id not in trained:
                excluded[m.chain_id] = "no_sample"
        if plan.samples:  # to one rank, which trains it while generation and checks go on
            self.pool.assign(plan, {"prompt_id": group.prompt_id, "members": len(members)})
        return GroupOutcome(group.prompt_id, scored, plan.mean_reward, None, advantages, trained, excluded)

    def finish(self) -> dict:
        from ..trainer.checks import TrainerCheckFailed

        try:
            report = self.pool.finish()
        except TrainerCheckFailed as exc:
            self._write_gaps(exc.report)
            raise
        report["memory"] = self._memory(report.pop("by_rank", None))
        self._write_gaps(report)
        return report

    def _memory(self, by_rank: list | None) -> dict | None:
        """Every rank's peak GPU memory since the previous step (each shares its GPU with an engine when
        colocated; each rank resets its own), with the largest of them under the keys the summary reads."""
        mems = [dict(r.get("memory") or {}, rank=r["rank"], groups=r.get("groups"), samples=r.get("samples"))
                for r in by_rank or [] if r.get("memory")]
        if not mems:
            return None
        return {"max_allocated_gib": max(m["max_allocated_gib"] for m in mems),
                "max_reserved_gib": max(m["max_reserved_gib"] for m in mems),
                "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "by_rank": mems}

    def _write_gaps(self, report: dict) -> None:
        """Per-sample gaps beside the run (the T10 measurement reads them), and the gate's statistic, the
        per-sample `trainer.gap_statistic`, under the keys the iteration summary reads."""
        rows = [dataclasses.asdict(g) if dataclasses.is_dataclass(g) else dict(g) for g in report.get("gaps") or []]
        report["gaps"] = rows
        gap = report.get("gap")
        stat = self.cfg.trainer.gap_statistic
        if isinstance(gap, dict) and isinstance(gap.get(stat), dict):
            gap.setdefault("statistic", stat)
            gap.setdefault("median", gap[stat].get("median"))
            gap.setdefault("p99", gap[stat].get("p99"))
        path = self.cfg.run_dir / "trainer" / f"gaps-{self.iteration:06d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows))

    def noop_report(self) -> dict | None:
        """For an iteration that takes no step: what the open step measured (the gap and the gap gate's verdict),
        with its gaps written as a step's are. None if no step is open."""
        if self.trainer is None or not self.trainer._streaming:
            return None
        report = self.pool.report()
        report["memory"] = self._memory(report.pop("by_rank", None))
        self._write_gaps(report)
        return report

    def discard(self) -> None:
        """Drop the open step on every rank. It runs on the way out of an iteration that is not trained, so a rank
        that cannot be reached is logged rather than raised: the next `begin` or checkpoint refuses that pool."""
        from ..trainer.rank_pool import RankError

        if self.pool is None:
            return
        try:
            self.pool.discard()
        except RankError as exc:
            log.warning("discarding the open step: %s", describe_error(exc, self.cfg.run.source))

    def warmup(self) -> dict:
        """`trainer.startup_warmup`: one forward and backward on a synthetic group shaped like a step's micro-batches
        (`ranks.warmup_group`, from `warmup_ids`), gradient dropped, before iteration 1. Refuses if the trained state
        (latent, Adam moments, step count), the merged version or the gradients are not exactly as before; the RNG
        states are restored. Every rank warms up at once and checks its own state the same way. Returns rank 0's
        counts, the wall time, and every rank's peak memory up to its end (then reset, so iteration 1's figure is
        its own)."""
        import time

        import torch

        from ..trainer.rank_pool import memory_peak, state_digest
        from ..trainer.ranks import warmup_group

        if self.warmup_ids is None:
            raise WiringError("the warm-up has no token ids (warmup_ids is not set)")
        self.ensure()
        group = warmup_group(self.warmup_ids(), self.cfg.trainer.micro_batch_tokens, self.cfg.rollout.max_new_tokens)

        def rank0() -> dict:
            before, version = state_digest(self.trainer), self.merged.version
            cuda = self.device.startswith("cuda")
            t0 = time.monotonic()
            with torch.random.fork_rng(devices=[torch.device(self.device)] if cuda else []):
                report = self.trainer.warmup(group)
                if cuda:
                    torch.cuda.synchronize(torch.device(self.device))
            report["wall_s"] = time.monotonic() - t0
            if (state_digest(self.trainer) != before or self.merged.version != version
                    or any(p.grad is not None for p in self.trainer.params)):
                raise WiringError("the start-up warm-up changed the trainer's state")
            report["memory"] = memory_peak(self.device)
            report["state"] = "unchanged"
            return report

        t0 = time.monotonic()
        reports = self.pool.warmup(group, rank0)
        out = dict(reports[0], wall_s=time.monotonic() - t0)
        out["memory"] = self._memory([{"rank": r, "memory": rep.get("memory")} for r, rep in enumerate(reports)])
        out["wall_s_by_rank"] = [rep.get("wall_s") for rep in reports]
        return out

    # C5's trainer side

    def probe_logprobs(self, probes, *, base: bool = False) -> list[list[float]]:
        """Raw next-token log-probs of each probe, positions 1..L-1, from the merged forward (or the base)."""
        import contextlib

        from ..trainer.batching import raw_logprobs_at

        self.ensure()
        out = []
        ctx = self.model.disable_adapter() if base else contextlib.nullcontext()
        with ctx:
            for ids in probes.ids:
                lp = raw_logprobs_at(self.model, list(ids), range(1, len(ids)), device=self.device, amp=self.amp,
                                     lm_head_fp32=self.cfg.trainer.lm_head_fp32)
                out.append([float(x) for x in lp.tolist()])
        return out

    # checkpoint

    def save_state(self, directory) -> None:
        """Rank 0's latent, Adam moments and RNG states: every rank holds the same (`state_digest` checks it)."""
        from ..trainer.rank_pool import save_state

        save_state(self.ensure().trainer, directory)

    def load_state(self, directory) -> None:
        """Every rank loads the checkpoint, and must then hold the same trained state as rank 0."""
        from ..trainer.loss import TrainerInputError

        try:
            self.ensure().pool.load_state(directory)
        except TrainerInputError as exc:
            raise WiringError(str(exc)) from None

    def state_digest(self) -> str:
        """sha256 over the trained latent and the optimizer: what a resume has to restore bit for bit. Over R ranks
        it refuses unless every rank's is the same."""
        return self.ensure().pool.state_digest()

    def remerge(self, version: int) -> None:
        """After a restore: every rank re-merges its (just loaded) latent as `version`, and must then hold the same
        W_eff as rank 0, which is the one published."""
        self.ensure().pool.remerge(version)


# --- C5 ------------------------------------------------------------------------------------------


def standin_probes(cfg: RunConfig, tokenizer, dataset):
    """The landing probes: the last `sync.probes` of the stand-in kernels picked from the dataset, each with the fixed
    reply."""
    from ..rollout.templates import STANDIN_SYSTEM, pick_standin_kernels, standin_messages
    from ..sync.landing import build_probes

    kernels = pick_standin_kernels(PROBE_KERNELS, dataset)[-cfg.sync.probes:]
    texts = [standin_messages(k)[1]["content"] for k in kernels]
    return build_probes(tokenizer.tok, texts, system=STANDIN_SYSTEM, tokens=cfg.sync.probe_tokens)


class SyncAdapter:
    """C5 behind the SyncPort: publish the merged weights, reload every engine, check that they landed."""

    def __init__(self, cfg: RunConfig, tokenizer, trainer: TrainerAdapter, rollout: RolloutAdapter, *, spans,
                 dataset: str | Path | None, probes=None):
        from ..sync.writer import expected_count, load_base_tensors, read_base_index, target_keys

        if dataset is None and probes is None:
            raise WiringError("the landing probes need the stand-in dataset or a given probe set")
        self.cfg, self.tok, self.trainer, self.rollout, self.spans = cfg, tokenizer, trainer, rollout, spans
        self.dataset = None if dataset is None else Path(dataset)
        keys = list(read_base_index(cfg.model.base_dir))
        self.expected = expected_count(keys, partial=cfg.sync.partial)
        targets = target_keys(keys)
        self.others = (None if cfg.sync.partial
                       else load_base_tensors(cfg.model.base_dir, [k for k in keys if k not in targets]))
        self._probes = probes
        self.state = None
        self.calibration = None
        self._held: str | None = None

    def engines(self) -> list[SyncEngine]:
        return [SyncEngine(self.rollout.pool, i) for i in range(len(self.rollout.handles))]

    def probes(self):
        if self._probes is None:
            self._probes = standin_probes(self.cfg, self.tok, self.dataset)
        return self._probes

    def _write(self, weights, version: int):
        from ..sync.writer import write_slot

        s = self.cfg.sync
        return write_slot(weights, base_dir=self.cfg.model.base_dir, sync_dir=s.dir, run_id=self.cfg.run.run_id,
                          version=version, slots=s.slots, partial=s.partial, others=self.others)

    def round_trip(self) -> dict:
        """C5.T1 on the real engines: a synthetic perturbation must land and the base must come back. The base
        goes to slot w0 (version 0) and the perturbation to w1, which is deleted afterwards."""
        from ..sync.roundtrip import round_trip, synthetic_perturbation
        from ..sync.writer import delete_slot

        base = self.trainer.ensure().merged.base
        restore = self._write(base, 0)
        perturbed = self._write(synthetic_perturbation(base), 1)
        try:
            with self.spans.span("sync.round_trip", 0):
                res = round_trip(self.engines(), self.probes(), perturbed_slot=perturbed.path,
                                 restore_slot=restore.path, expected=self.expected,
                                 in_flight=self.rollout.in_flight)
        finally:
            delete_slot(perturbed.path)
        out = res.as_dict()
        if res.verdict == "PROVEN":
            self._calibrate()
            self._held = str(restore.path)
            out["calibration"] = {str(e): vars(c) for e, c in self.calibration.items()}
        return out

    def _calibrate(self) -> None:
        """tau per engine from the v0 probe gap (Section 6.3), with every engine serving the base."""
        from ..sync.landing import SyncState, calibrate_tau, engine_logprobs
        from ..sync.reload import VersionBook

        probes = self.probes()
        trainer_v0 = self.trainer.probe_logprobs(probes, base=True)
        self.calibration = {}
        previous = {}
        for e in self.engines():
            v0 = engine_logprobs(e, probes)
            self.calibration[e.engine_id] = calibrate_tau(v0, trainer_v0, self.cfg.sync.tau_floor)
            previous[e.engine_id] = v0
        self.state = SyncState(run_id=self.cfg.run.run_id, sync_dir=self.cfg.sync.dir,
                               base_dir=self.cfg.model.base_dir, probes=probes,
                               tau={e: c.tau for e, c in self.calibration.items()}, previous=previous,
                               in_flight=self.rollout.in_flight, expected=self.expected, slots=self.cfg.sync.slots,
                               partial=self.cfg.sync.partial, others=self.others, book=VersionBook())

    def _ensure_state(self) -> None:
        if self.state is None:
            self._calibrate()

    def publish(self, version: int, *, in_flight) -> dict:
        return self._publish(version, in_flight, span_iteration=version)

    def _publish(self, version: int, in_flight, *, span_iteration: int) -> dict:
        from ..sync.landing import LandingFailed
        from ..sync.landing import publish as publish_weights
        from ..sync.reload import BarrierViolation as ReloadBarrier

        merged = self.trainer.ensure().merged
        if merged.version != version:
            raise RuntimeError(f"publish v{version}, but the merged weights are v{merged.version}")
        self._ensure_state()
        self.state.in_flight = in_flight
        stats = merged.last_stats
        try:
            pub = publish_weights(merged.weights, version, stats.changed, engines=self.engines(), state=self.state,
                                  trainer_side=self.trainer.probe_logprobs)
        except ReloadBarrier as exc:
            raise BarrierViolation(str(exc)) from None
        except LandingFailed as exc:
            self._record_spans(version, exc.publication, span_iteration)
            raise
        self._record_spans(version, pub, span_iteration)
        self._held = str(pub.slot)
        return {"verdicts": pub.verdicts(), "changed": pub.changed, "changed_vs_base": stats.changed_vs_base,
                "coverage": stats.coverage, "aligned": stats.aligned, "changed_frac": stats.changed_frac,
                "rounding": stats.rounding, "slot": str(pub.slot), "seconds": dict(pub.seconds),
                "pruned": list(pub.pruned), "landing": {str(e): r.as_dict() for e, r in pub.results.items()}}

    def _record_spans(self, version: int, pub, iteration: int) -> None:
        """write, reload and landing as consecutive spans ending now (publish reports their durations), under
        `iteration`: the published version's, or 0 for the restore, which is part of setup."""
        sec = pub.seconds
        verdicts = pub.verdicts()
        t1 = self.spans.now()
        t0 = t1 - sum(sec.get(k, 0.0) for k in ("write", "reload", "landing"))
        for name in ("write", "reload", "landing"):
            d = float(sec.get(name, 0.0))
            attrs = {"version": version}
            if name == "landing":
                attrs["verdict"] = [verdicts[e] for e in sorted(verdicts)]  # one per engine, as C9 reads it
            self.spans.record(name, iteration, t0, t0 + d, **attrs)
            t0 += d

    def restore(self, version: int) -> dict:
        """After a resume: re-merge the restored latent to `version` on every rank (they must agree) and publish it
        before anything is served."""
        merged = self.trainer.ensure().merged
        self._ensure_state()
        merged.span_iteration = 0
        try:
            self.trainer.remerge(version)
        finally:
            merged.span_iteration = None
        return self._publish(version, self.rollout.in_flight, span_iteration=0)

    def held_slot(self) -> str | None:
        return self._held


# --- a score run's model -------------------------------------------------------------------------


def top_files(directory: str | Path) -> dict[str, str]:
    """sha256 of each top-level, not hidden file of a model directory, by name, without the merge manifest: the
    files an engine loads."""
    from ..ids import sha256_file

    return {p.name: sha256_file(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and not p.name.startswith(".") and p.name != MERGE_MANIFEST}


def _top_level(recorded: dict[str, str], prefix: str) -> dict[str, str]:
    """The run plan's hashes of one directory's top-level files (`<prefix>/<name>`), as `top_files` names them."""
    out = {}
    for key, sha in recorded.items():
        head, _, name = key.partition("/")
        if head == prefix and "/" not in name and name != MERGE_MANIFEST:
            out[name] = sha
    return out


class ServedModel:
    """What a score run serves (`orchestrator.mode: score`), and the start-up landing check on it.

    `prepare` takes `model.base_dir` as it is, or merges `model.adapter_dir` into it with tools/merge_base_adapter.py,
    the trainer's and the sync's merge formula, into the run's node-local directory. The tool checks what it wrote and
    records the sha256 of every file in its merge manifest; a merge whose base or adapter is not the one the run plan
    hashed is refused. So is a base whose index names a shard below its top level: TokenSpeed loads the top-level
    shards the index names, as a Hugging Face checkpoint lays them out, and only top-level files are confirmed.

    `confirm` runs once the engines have started on that directory:
      1. its files are hashed again and must be the ones prepared (the run plan's, or the merge manifest's);
      2. every one of its tensors is written to weight slot w0, whatever `sync.partial` says: a partial slot would
         leave the engines' own copies of the other tensors, from whatever they loaded at start, unchecked. The slot's
         xxh64 manifest is checked against the slot's own files (`sync.writer.verify_slot`);
      3. the start-up round trip (`sync.roundtrip`): each engine's draws on what it loaded at start, then on a
         synthetic perturbation (every draw must change, so a reload lands), then on slot w0 (every draw must be the
         first one again). PROVEN means the engines serve exactly those tensors, and served the same at start.
    The engines then hold slot w0 for the whole run; nothing is reloaded after it. The tensors are read, perturbed and
    written on the CPU: a score run has no trainer on a GPU."""

    def __init__(self, cfg: RunConfig, tokenizer, rollout: RolloutAdapter, *, spans, dataset: str | Path | None,
                 probes=None, merge_tool: Path = MERGE_TOOL, python: str | None = None):
        import sys

        if dataset is None and probes is None:
            raise WiringError("the landing probes need the stand-in dataset or a given probe set")
        self.cfg, self.tok, self.rollout, self.spans = cfg, tokenizer, rollout, spans
        self.dataset = None if dataset is None else Path(dataset)
        self.dir = served_model_dir(cfg)
        self.merge_tool, self.python = Path(merge_tool), python or sys.executable
        self._probes = probes
        self.identity: dict | None = None

    def probes(self):
        if self._probes is None:
            self._probes = standin_probes(self.cfg, self.tok, self.dataset)
        return self._probes

    def _safe(self, text: str) -> str:
        from ..ids import sha256_text

        return str(text)[:500] if self.cfg.run.source == "standin" else f"sha256:{sha256_text(str(text))[:16]}"

    def prepare(self, recorded: dict[str, str]) -> dict:
        """`recorded`: the run plan's sha256 of the model's files, by `base/<name>` and `adapter/<name>`."""
        from ..sync.writer import INDEX_NAME, ShardNameError, read_base_index
        from .score import files_digest

        m = self.cfg.model
        if (Path(m.base_dir) / INDEX_NAME).is_file():
            try:
                read_base_index(m.base_dir)
            except ShardNameError as exc:
                detail = {"base_dir": m.base_dir, "shards": exc.names[:10]}
                raise RunStopped("model_shards_not_top_level", detail) from None
        base = _top_level(recorded, "base")
        if m.adapter_dir is None:
            self.identity = {"kind": "merged_dir", "dir": str(self.dir), "files": base}
        else:
            self.identity = self._merge(base, _top_level(recorded, "adapter"))
        self.identity["files_sha256"] = files_digest(self.identity["files"])
        return self.identity

    def _merge(self, base: dict[str, str], adapter: dict[str, str]) -> dict:
        import subprocess
        import time

        from ..atomic import read_text
        from ..ids import sha256_file

        out, m = self.dir, self.cfg.model
        if out.exists() or out.with_name(out.name + ".partial").exists():
            raise RunStopped("merge_output_exists", {"dir": str(out)})
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.python, str(self.merge_tool), "merge", "--base", m.base_dir, "--adapter", m.adapter_dir,
               "--out", str(out)]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=MERGE_TIMEOUT_S,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        except subprocess.TimeoutExpired:
            raise RunStopped("model_merge_failed", {"rc": None, "why": f"no result in {MERGE_TIMEOUT_S:g} s"}) from None
        if proc.returncode != 0:
            raise RunStopped("model_merge_failed", {"rc": proc.returncode,
                                                    "stderr": self._safe(proc.stderr.strip()[-500:])})
        manifest = json.loads(read_text(out / MERGE_MANIFEST))
        problems = []
        if manifest.get("pass") is not True:
            problems.append("the merge manifest does not record a pass")
        made_from = {n: f["sha256"] for n, f in manifest["base"]["files"].items()
                     if not n.startswith(".") and n != MERGE_MANIFEST}
        if made_from != base:
            problems.append("the merged base is not the base the run plan hashed")
        ad = manifest["adapter"]
        if (adapter.get("adapter_model.safetensors"), adapter.get("adapter_config.json")) != (
                ad["weights_sha256"], ad["config_sha256"]):
            problems.append("the merged adapter is not the adapter the run plan hashed")
        if problems:
            raise RunStopped("model_merge_unverified", {"problems": problems})
        checks = manifest["checks"]["bytes"]
        return {"kind": "base_plus_adapter", "dir": str(out), "base_dir": m.base_dir, "adapter_dir": m.adapter_dir,
                "files": {n: f["sha256"] for n, f in manifest["output"]["files"].items()
                          if not n.startswith(".") and n != MERGE_MANIFEST},
                "merge": {"tool": "tools/merge_base_adapter.py", "manifest_sha256": sha256_file(out / MERGE_MANIFEST),
                          "rank": ad["rank"], "lora_alpha": ad["lora_alpha"], "modules": ad["modules"],
                          "targets": checks["targets"], "targets_changed": checks["targets_changed"],
                          "tensor_digest": manifest["output"]["tensor_digest"],
                          "target_digest": manifest["output"]["target_digest"], "seconds": time.monotonic() - t0}}

    def confirm(self) -> dict:
        import time

        from ..sync.roundtrip import round_trip, synthetic_perturbation
        from ..sync.writer import (delete_slot, expected_count, load_base_tensors, read_base_index, target_keys,
                                   verify_slot, write_slot)
        from .score import files_digest

        if self.identity is None:
            raise RuntimeError("confirm before prepare")
        cfg, s = self.cfg, self.cfg.sync
        secs: dict[str, float] = {}
        t = time.monotonic()
        now = top_files(self.dir)
        secs["hash"] = time.monotonic() - t
        if now != self.identity["files"]:
            changed = sorted(n for n in set(now) | set(self.identity["files"])
                             if now.get(n) != self.identity["files"].get(n))
            raise RunStopped("served_model_changed", {"dir": str(self.dir), "files": changed[:10]})
        t = time.monotonic()
        keys = list(read_base_index(self.dir))
        targets = sorted(target_keys(keys))
        weights = load_base_tensors(self.dir, targets)
        others = load_base_tensors(self.dir, [k for k in keys if k not in set(targets)])
        expected = expected_count(keys, partial=False)
        secs["read"] = time.monotonic() - t
        t = time.monotonic()
        common = dict(base_dir=self.dir, sync_dir=s.dir, run_id=cfg.run.run_id, slots=s.slots, partial=False,
                      others=others)
        restore = write_slot(weights, version=0, **common)
        held = verify_slot(restore.path)
        if not held["ok"]:
            raise RunStopped("served_slot_mismatch", {"slot": str(restore.path), "problems": held["problems"]})
        perturbed = write_slot(synthetic_perturbation(weights), version=1, **common)
        del weights, others, common
        secs["slots"] = time.monotonic() - t
        engines = [SyncEngine(self.rollout.pool, i) for i in range(len(self.rollout.handles))]
        try:
            with self.spans.span("sync.round_trip", 0):
                res = round_trip(engines, self.probes(), perturbed_slot=perturbed.path, restore_slot=restore.path,
                                 expected=expected, in_flight=self.rollout.in_flight)
        finally:
            delete_slot(perturbed.path)
        secs["round_trip"] = res.seconds
        return {"verdict": res.verdict, "engines": len(engines), "files_checked": len(now),
                "files_sha256": files_digest(now), "slot": str(restore.path), "slot_tensors": restore.tensors,
                "slot_manifest_sha256": restore.manifest_sha256, "partial": False, "tensors_read": expected,
                "round_trip": res.as_dict(), "seconds": secs}


# --- the whole -----------------------------------------------------------------------------------


@dataclass
class Parts:
    """What `build` otherwise makes itself. Tests pass in-process stand-ins; a run passes nothing."""

    tokenizer: Any = None
    pool: Any = None  # a PromptPool
    handles: Sequence[Any] | None = None
    compiler: Any = None
    reward_factories: dict | None = None
    load_model: Callable | None = None
    trainer_device: str | None = None
    require_fast_path: bool | None = None  # None: trainer.require_fast_path
    amp: bool = True
    probes: Any = None


def build(cfg: RunConfig, *, config_path=None, commit: str | None = None, parts: Parts | None = None):
    """The orchestrator over the real components. Nothing is started here: engines start in the orchestrator's
    setup, after its run plan is written, and the model loads on first use."""
    from ..reward.clock import Clock
    from ..rollout.templates import HFTokenizer, TemplateError
    from ..spans import SpanRecorder
    from .loop import Orchestrator

    parts = parts or Parts()
    check_supported(cfg)
    clock = Clock()
    spans = SpanRecorder(monotonic=clock.now, wall=clock.wall)
    dataset = None
    if parts.pool is None or parts.probes is None:  # the prompts and the landing probes come from the dataset
        dataset = standin_dataset_dir(cfg.data.prompt_pools[0].path)
    tok = parts.tokenizer or HFTokenizer.load(cfg.model.base_dir)
    pool = parts.pool if parts.pool is not None else StandinPromptPool(tok, cfg.rollout.thinking, dataset)
    try:
        records = pool.records()
        max_len = rollout_max_len(cfg, tok, records)
    except TemplateError as exc:
        raise WiringError(f"stand-in prompts: {exc}") from None
    compiler = parts.compiler if parts.compiler is not None else rollout_compiler(cfg)
    handles = parts.handles if parts.handles is not None else engine_handles(cfg, max_len)
    if cfg.orchestrator.mode == "score":  # no trainer, no optimizer, no weight sync
        from .score import Scorer

        rollout = RolloutAdapter(cfg, tok, handles, max_len=max_len, spans=spans, compiler=compiler,
                                 on_shutdown=[lambda: remove_node_dir(cfg)], clock=clock)
        model = ServedModel(cfg, tok, rollout, spans=spans, dataset=dataset, probes=parts.probes)
        reward = RewardAdapter(cfg, clock=clock, spans=spans, factories=parts.reward_factories)
        return Scorer(cfg, pool=pool, rollout=rollout, reward=reward, model=model, clock=clock, spans=spans,
                      commit=commit, config_path=config_path)
    trainer = TrainerAdapter(cfg, spans=spans, load_model=parts.load_model, device=parts.trainer_device,
                             require_fast_path=parts.require_fast_path, amp=parts.amp)
    rollout = RolloutAdapter(cfg, tok, handles, max_len=max_len, spans=spans, compiler=compiler,
                             on_shutdown=[trainer.shutdown, lambda: remove_node_dir(cfg)], clock=clock)
    sync = SyncAdapter(cfg, tok, trainer, rollout, spans=spans, dataset=dataset, probes=parts.probes)
    trainer.warmup_ids = lambda: sync.probes().ids  # the landing probes: real stand-in prompt tokens
    reward = RewardAdapter(cfg, clock=clock, spans=spans, factories=parts.reward_factories)
    return Orchestrator(cfg, pool=pool, rollout=rollout, reward=reward, trainer=trainer, sync=sync, clock=clock,
                        spans=spans, commit=commit, config_path=config_path)
