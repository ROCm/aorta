"""T10 (C4.T10, WP4): the on-policy trainer-engine gap at step 0 and after a few real training steps.

The run uses the integrated components unchanged, as the stand-in run does (`orchestrator.wiring`): WP2's engine
pool and rollout manager on TokenSpeed, WP4's streaming rank trainer over WP3's merged LoRA, and WP3's publication
(merge, write, reload, landing check) and start-up round trip. What it adds is measurement:

  for n in 0..steps:                                       (point n: the engines serve W_eff(n))
    rollout   P x K chains (turn 0 + repair turns) at W_eff(n); at n = 0 also `extra_v0_batches` more prompt blocks
    score     every turn's reply, trainer vs engine, raw log-probs, on the same samples:
                solo     one sequence alone (batching.raw_logprobs_at's arithmetic), + log-prob at T_loss (or at the
                         plan's control temperature), + entropy
                packed   the gap gate's own view (each group packed as the streaming step packs it)
                fp32la   fix 1: the linear-attention core on fp32 inputs (state in fp32), trainer side only
    gradient  (at `tis_grad_at`) the step's gradient with TIS off and on (fix 2), with and without fix 1; no step
    step      the real streaming step (the loss rules as configured; the gap gate wide, report only)
    stale     the same samples re-scored under W_eff(n + 1): the September bug's shape, as a control
    publish   W_eff(n + 1) to every engine, landing check, and C4.T3 (the slot's xxh64 = the trainer's tensors)

The training steps' reward is `t10_reward`: standin.v11's terminal gates and length term, with the checks this
measurement does not run counted clean. It only has to move the weights as a step would; it is not a result.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aorlta.atomic import atomic_write_bytes, atomic_write_json
from aorlta.config import plain
from aorlta.ids import policy_version
from aorlta.measure import gap as G
from aorlta.orchestrator import wiring as W
from aorlta.orchestrator.ports import GroupInput, GroupMember
from aorlta.orchestrator.state import select_prompts
from aorlta.schemas import RewardResult
from aorlta.trainer.loss import build_groups

GATE_BASE = {"compiled": 0.5, "no_code_block": -1.0, "turns_exhausted": -0.9, "context_limit": -0.9}
REWARD_FN = {"name": "t10.compile_gates", "version": "1", "code_sha256": ""}
EXTRA_ITERATION_BASE = 1000  # chain ids and seeds of the extra v0 batches: iterations 1001, 1002, ...


@dataclass(frozen=True)
class T10Plan:
    steps: int = 4
    extra_v0_batches: int = 1
    fp32la_every_point: bool = True
    tis_grad_at: tuple[int, ...] = (0,)
    tis_cap: float = 2.0  # Miles' --tis-clip default
    stale_control: bool = True
    base_control_at_last: bool = True
    trainer_memory_fraction: float | None = None  # None: the adapter's own cap (1 - gpu_memory_utilization)
    # The temperature of the solo scoring's `temp` array, the loss-temperature mix-up control. None: the loss
    # temperature. With the loss at T 1, `temp` would equal the raw log-probs, so a run at T 1 names the rollout's 0.7.
    control_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.steps < 0 or self.extra_v0_batches < 0:
            raise ValueError("steps and extra_v0_batches must be >= 0")
        if not self.tis_cap > 0:
            raise ValueError("tis_cap must be > 0")
        if self.trainer_memory_fraction is not None and not 0 < self.trainer_memory_fraction <= 1:
            raise ValueError("trainer_memory_fraction must be in (0, 1]")
        if self.control_temperature is not None and not self.control_temperature > 0:
            raise ValueError("control_temperature must be > 0")


def t10_reward(record, max_new_tokens: int) -> RewardResult | None:
    """standin.v11's terminal gates (parse -1.0, compile -0.9) and length term (-0.05 tokens / max_new_tokens),
    with the legs this measurement does not run counted clean (+0.5, a self-written reference). None for a chain
    that handed over no candidate; unscored for a logprob_error chain, which never trains."""
    end = record.end_reason
    if end == "logprob_error":
        return RewardResult(candidate_id=record.chain_id, reward=None, status="unscored", reason="logprob_error",
                            parts=None, reward_fn=dict(REWARD_FN), legs={}, full_check=False, plan=None, audit=False,
                            cache_key="", timings={})
    if end not in GATE_BASE or not record.turns:
        return None
    base = GATE_BASE[end]
    length = -0.05 * len(record.turns[-1].response_ids) / max_new_tokens
    return RewardResult(candidate_id=record.chain_id, reward=base + length, status="scored", reason=None,
                        parts={"base": base, "broke_healthy": 0.0, "length": length}, reward_fn=dict(REWARD_FN),
                        legs={}, full_check=True, plan=None, audit=False, cache_key="", timings={})


def group_inputs(res, iteration: int, prompts: Sequence, max_new_tokens: int) -> list[GroupInput]:
    """One GroupInput per prompt, members in k order, as the orchestrator's ledger hands them to the trainer."""
    samples = {s.chain_id: s for s in res.samples}
    by_prompt: dict[str, list] = {}
    for rec in res.chains:
        by_prompt.setdefault(rec.prompt_id, []).append(rec)
    out = []
    for p in prompts:
        recs = sorted(by_prompt.get(p.prompt_id, []), key=lambda r: r.k)
        members = tuple(GroupMember(r.chain_id, r.k, r, samples.get(r.chain_id), t10_reward(r, max_new_tokens))
                        for r in recs)
        out.append(GroupInput(iteration, p.prompt_id, members))
    return out


def plans_of(groups: Sequence[GroupInput], min_scored: int) -> list:
    """The trainer's GroupPlans, built exactly as `TrainerAdapter.add_group` builds them."""
    out = []
    for gi in groups:
        out.extend(build_groups([m.sample for m in gi.members if m.sample is not None],
                                [m.result for m in gi.members if m.result is not None],
                                min_scored=min_scored, chains=gi.chains))
    return out


def token_arrays(items: Sequence[G.ChainItem], modes: dict[str, list[dict]], packed: dict | None) -> dict:
    """Per-token arrays over every item's turns, in item then turn order: chain index, turn, position in the reply,
    the engine's raw log-prob, and each mode's trainer raw log-prob (`packed` covers the final turn only; NaN
    elsewhere)."""
    chain, turn, j, engine, final = [], [], [], [], []
    for i, it in enumerate(items):
        for s in it.spans:
            chain += [i] * s.length
            turn += [s.turn] * s.length
            j += list(range(s.length))
            engine += list(s.engine_logprobs)
            final += [s.turn == it.final_turn] * s.length
    out = {"chain": np.asarray(chain, np.int32), "turn": np.asarray(turn, np.int8), "j": np.asarray(j, np.int32),
           "engine": np.asarray(engine, np.float32), "final": np.asarray(final, bool)}
    n = len(engine)
    for name, scored in modes.items():
        for key in ("raw", "temp", "entropy"):
            if scored and key in scored[0]:
                arr = torch.cat([s[key] for s in scored]).numpy().astype(np.float32)
                if arr.shape[0] != n:
                    raise RuntimeError(f"mode {name}: {arr.shape[0]} values for {n} tokens")
                out[name if key == "raw" else f"{name}_{key}"] = arr
    if packed is not None:
        arr = np.full(n, np.nan, np.float32)
        offset = 0
        for it in items:
            for s in it.spans:
                if s.turn == it.final_turn and it.chain_id in packed:
                    v = packed[it.chain_id].numpy()
                    if v.shape[0] != s.length:
                        raise RuntimeError(f"{it.chain_id}: packed {v.shape[0]} values for a {s.length}-token reply")
                    arr[offset:offset + s.length] = v
                offset += s.length
        out["packed"] = arr
    return out


def reply_ids(arrays: dict) -> np.ndarray:
    """Each token's reply index: replies are contiguous runs of one (chain, turn) in `token_arrays`' order."""
    key = arrays["chain"].astype(np.int64) * 8 + arrays["turn"].astype(np.int64)
    change = np.ones(key.size, bool)
    change[1:] = key[1:] != key[:-1]
    return np.cumsum(change) - 1


def summarize(arrays: dict, modes: Sequence[str]) -> dict:
    """A compact summary for the log and the JSON: per mode, per turn, all tokens and without each reply's first
    token, the per-token |delta| quantiles and the per-reply mean |delta| quantiles."""
    out = {}
    rid = reply_ids(arrays)
    n_replies = int(rid[-1]) + 1 if rid.size else 0
    turns = sorted(set(int(t) for t in arrays["turn"]))
    for mode in modes:
        if mode not in arrays:
            continue
        d_all = arrays[mode].astype(np.float64) - arrays["engine"].astype(np.float64)
        rows = {}
        for label, sel in [("all", np.ones(rid.size, bool))] + [(f"turn{t}", arrays["turn"] == t) for t in turns]:
            for first in (True, False):
                m = sel & ~np.isnan(d_all)
                if not first:
                    m &= arrays["j"] > 0
                if not m.any():
                    continue
                s = np.bincount(rid[m], weights=np.abs(d_all[m]), minlength=n_replies)
                c = np.bincount(rid[m], minlength=n_replies)
                rows[label + ("" if first else "_nofirst")] = {"tokens": G.token_quantiles(d_all[m]),
                                                               "reply_mean_abs": G.quantiles(s[c > 0] / c[c > 0])}
        out[mode] = rows
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a @ b) / (a.norm() * b.norm()))


def gradient_effect(adapter, plans: Sequence, *, tis_cap: float, fp32la: bool) -> dict:
    """The step's (unnormalised) LoRA gradient on these plans, without stepping, under TIS off/on and, with
    `fp32la`, also under fix 1. Uses the rank trainer's own accumulation (`_add`, streaming packing), so the
    gradient is the one the step would take, up to the global normaliser and the clip (both common factors)."""
    rt = adapter.trainer
    base = rt.loss
    variants = [("default", None, False), ("tis", tis_cap, False)]
    if fp32la:
        variants += [("fp32la", None, True), ("fp32la_tis", tis_cap, True)]
    grads, secs, errors = {}, {}, {}
    live = [p for p in plans if p.samples]
    try:
        for name, tis, f32 in variants:
            rt.loss = dataclasses.replace(base, tis=tis)
            rt._reset()
            ctx = G.fp32_linear_attention(adapter.model) if f32 else contextlib.nullcontext()
            t0 = time.monotonic()
            try:
                with ctx:
                    rt._add(live, None, across_groups=False)
            except torch.OutOfMemoryError as exc:  # recorded for this variant; the others still run
                errors[name] = f"out of memory: {str(exc)[:300]}"
                rt._reset()
                torch.cuda.empty_cache()
                continue
            grads[name] = rt._flat_grads().detach().clone()
            secs[name] = time.monotonic() - t0
            rt._reset()
    finally:
        rt.loss = base
        rt._reset()
    names = list(grads)
    out = {"norm": {k: float(grads[k].double().norm()) for k in names}, "seconds": secs, "cos": {},
           "errors": errors}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["cos"][f"{a}~{b}"] = _cos(grads[a], grads[b])
    return out


def _weights_match_slot(weights: dict, slot: str) -> dict:
    from aorlta.sync.writer import read_manifest, tensor_xxh64

    entries = read_manifest(slot)["tensors"]
    bad = [k for k, w in weights.items() if k not in entries or tensor_xxh64(w.detach().cpu()) != entries[k]["xxh64"]]
    return {"tensors": len(weights), "slot_tensors": len(entries), "mismatched": bad[:10], "ok": not bad
            and set(entries) == set(weights)}


class Recorder:
    def __init__(self, out_dir: Path, log: Callable[[str], None]):
        self.out, self.log = out_dir, log
        self.out.mkdir(parents=True, exist_ok=True)
        self.doc: dict = {"points": [], "events": []}
        self.t0 = time.monotonic()

    def event(self, what: str, **kw) -> None:
        kw = {"t": round(time.monotonic() - self.t0, 1), "what": what, **kw}
        self.doc["events"].append(kw)
        self.log(json.dumps(kw, default=str)[:2000])
        self.flush()

    def flush(self) -> None:
        atomic_write_json(self.out / "t10.json", self.doc)

    def arrays(self, name: str, arrays: dict, items: Sequence[G.ChainItem], extra: dict) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        atomic_write_bytes(self.out / f"{name}.npz", buf.getvalue())
        meta = {"items": [{"chain_id": it.chain_id, "prompt_id": it.prompt_id, "k": it.k, "end_reason": it.end_reason,
                           "final_turn": it.final_turn, "length": len(it.ids), "turns": list(it.meta)}
                          for it in items], **extra}
        atomic_write_json(self.out / f"{name}.json", meta)


def run_t10(cfg, out_dir: str | Path, plan: T10Plan, *, parts: W.Parts | None = None, commit: str | None = None,
            trainer_env: dict[str, str] | None = None, log: Callable[[str], None] = print) -> dict:
    """The whole measurement; returns t10.json's document. Engines and slots are shut down and removed on exit.

    `trainer_env` is set after the engine processes have been spawned and before the trainer's first CUDA call, so
    it reaches the trainer's allocator only (PYTORCH_CUDA_ALLOC_CONF); it is recorded,
    with whether CUDA was still uninitialised when it was set."""
    from aorlta.reward.clock import Clock
    from aorlta.rollout.templates import HFTokenizer
    from aorlta.spans import SpanRecorder

    parts = parts or W.Parts()
    W.check_supported(cfg)
    rec = Recorder(Path(out_dir), log)
    rec.doc.update({"schema": "aorlta.t10/1", "commit": commit, "host": platform.node(), "plan": dataclasses.asdict(plan),
                    "config": plain(cfg)})
    clock = Clock()
    spans = SpanRecorder(monotonic=clock.now, wall=clock.wall)
    dataset = None
    if parts.pool is None or parts.probes is None:
        dataset = W.standin_dataset_dir(cfg.data.prompt_pools[0].path)
    tok = parts.tokenizer or HFTokenizer.load(cfg.model.base_dir)
    pool = parts.pool if parts.pool is not None else W.StandinPromptPool(tok, cfg.rollout.thinking, dataset)
    records = pool.records()
    max_len = W.rollout_max_len(cfg, tok, records)
    compiler = parts.compiler if parts.compiler is not None else W.rollout_compiler(cfg)
    handles = parts.handles if parts.handles is not None else W.engine_handles(cfg, max_len)
    rollout = W.RolloutAdapter(cfg, tok, handles, max_len=max_len, spans=spans, compiler=compiler,
                               on_shutdown=[lambda: W.remove_node_dir(cfg)])
    adapter = W.TrainerAdapter(cfg, spans=spans, load_model=parts.load_model, device=parts.trainer_device,
                               require_fast_path=parts.require_fast_path, amp=parts.amp)
    sync = W.SyncAdapter(cfg, tok, adapter, rollout, spans=spans, dataset=dataset, probes=parts.probes)
    r, o = cfg.rollout, cfg.orchestrator
    T = cfg.trainer.loss_temperature
    T_control = plan.control_temperature if plan.control_temperature is not None else T
    try:
        main_prompts, _ = select_prompts(records, 1, o.run_seed, r.P)
        rec.event("start", prompts=[p.prompt_id for p in main_prompts], pool=len(records), max_model_len=max_len)
        started = rollout.start(main_prompts[:4])
        rec.doc["engines"] = {"start": started, "ready": rollout.pool.ready}
        rec.event("engines_started", self_tests=started.get("self_tests"))
        if trainer_env:
            import os

            rec.doc["trainer_env"] = {"set": dict(trainer_env), "cuda_initialised_before": torch.cuda.is_initialized()}
            os.environ.update(trainer_env)
        adapter.ensure()
        model, device = adapter.model, adapter.device
        if plan.trainer_memory_fraction is not None and str(device).startswith("cuda"):
            # the stand-in run's 1 - gpu_memory_utilization cap shares a GPU with an engine; T10's trainer has its
            # own GPU, so the cap can be raised. Memory placement only: no kernel or shape changes.
            torch.cuda.set_per_process_memory_fraction(plan.trainer_memory_fraction, torch.device(device))
            rec.doc["trainer_memory_fraction"] = plan.trainer_memory_fraction
        rec.doc["trainer"] = {"device": str(device), "linear_attention_layers": len(G.linear_attention_layers(model)),
                              "lr": cfg.trainer.lr, "loss": dataclasses.asdict(adapter.trainer.loss)}
        rt = sync.round_trip()
        rec.doc["round_trip"] = rt
        rec.event("round_trip", verdict=rt.get("verdict"))
        if rt.get("verdict") != "PROVEN":
            raise RuntimeError(f"round trip {rt.get('verdict')}: the engines cannot be trusted to serve W_eff")
        merged = adapter.merged
        v0_is_base = all(torch.equal(merged.weights[k], merged.base[k]) for k in merged.weights)
        rec.doc["v0_weff_equals_base"] = v0_is_base
        if not v0_is_base:
            raise RuntimeError("W_eff(0) is not the base: the trainer does not start on the weights the engines serve")
        for n in range(plan.steps + 1):
            point = {"n": n, "version": merged.version, "batches": []}
            rec.doc["points"].append(point)
            batches = [("main", n + 1, main_prompts)]
            if n == 0:
                for b in range(plan.extra_v0_batches):
                    batches.append((f"extra{b + 1}", EXTRA_ITERATION_BASE + b + 1,
                                    select_prompts(records, 2 + b, o.run_seed, r.P)[0]))
            main = None
            for name, it, prompts in batches:
                t0 = time.monotonic()
                res = rollout.manager.run(prompts, it, policy_version(cfg.run.run_id, n))
                t_gen = time.monotonic() - t0
                items = G.chain_items(res.chains, res.samples)
                groups = group_inputs(res, it, prompts, r.max_new_tokens)
                plans = plans_of(groups, o.min_scored_per_group)
                t0 = time.monotonic()
                modes = {"solo": G.score_items(model, items, device=device, temperature=T_control, amp=adapter.amp)}
                secs = {"generate": t_gen, "solo": time.monotonic() - t0}
                if plan.fp32la_every_point:
                    t0 = time.monotonic()
                    with G.fp32_linear_attention(model):
                        modes["fp32la"] = G.score_items(model, items, device=device, temperature=T, amp=adapter.amp,
                                                        extras=False)
                    secs["fp32la"] = time.monotonic() - t0
                t0 = time.monotonic()
                packed = G.score_packed(model, plans, micro_batch_tokens=cfg.trainer.micro_batch_tokens,
                                        device=device, temperature=T, amp=adapter.amp)
                secs["packed"] = time.monotonic() - t0
                arrays = token_arrays(items, modes, packed)
                rewards = {m.chain_id: (None if m.result is None else m.result.reward)
                           for gi in groups for m in gi.members}
                batch = {"name": name, "iteration": it, "prompts": [p.prompt_id for p in prompts],
                         "chains": len(res.chains), "items": len(items), "tokens": int(arrays["engine"].size),
                         "metrics": {k: res.metrics.get(k) for k in ("end_reasons", "per_turn", "generation_wall_s",
                                                                       "repair_cache_ok_frac", "distinct_turn0_mean",
                                                                       "logprobs_aligned_frac", "deaths")},
                         "stop_reasons": list(res.stop_reasons), "seconds": secs,
                         "summary": summarize(arrays, list(modes) + ["packed"])}
                point["batches"].append(batch)
                rec.arrays(f"point{n}-{name}", arrays, items, {"rewards": rewards})
                rec.event("measured", n=n, batch=name, items=len(items), tokens=batch["tokens"], seconds=secs,
                          solo_all=batch["summary"]["solo"]["all"]["tokens"],
                          solo_reply_mean=batch["summary"]["solo"]["all"]["reply_mean_abs"])
                if res.stop_reasons:
                    rec.event("rollout_stop_reasons", reasons=list(res.stop_reasons))
                if name == "main":
                    main = (res, items, groups, plans, arrays, prompts, it)
            res, items, groups, plans, arrays, prompts, it = main
            if n == plan.steps and plan.base_control_at_last and n > 0:
                with model.disable_adapter():
                    base_scored = G.score_items(model, items, device=device, temperature=T, amp=adapter.amp,
                                                extras=False)
                arrays["base"] = token_arrays(items, {"base": base_scored}, None)["base"]
                rec.arrays(f"point{n}-main", arrays, items, {"rewards": {}})
                point["base_control"] = summarize(arrays, ["base"])
                rec.event("base_control", n=n)
            if n in plan.tis_grad_at:
                eff = gradient_effect(adapter, plans, tis_cap=plan.tis_cap, fp32la=True)
                point["gradient"] = eff
                rec.event("gradient", n=n, **eff)
            if n == plan.steps:
                break
            # the real step at W_eff(n), on this batch
            t0 = time.monotonic()
            adapter.begin(it)
            outcomes = [adapter.add_group(gi) for gi in groups]
            report = adapter.finish()
            step = {k: report.get(k) for k in ("step", "loss", "grad_norm", "totals", "normaliser", "counts",
                                                "clip_coefficient", "merge", "gap", "step_sec")}
            step["excluded"] = len(report.get("excluded") or [])
            step["checks"] = {k: (v.get("passed") if isinstance(v, dict) else v)
                              for k, v in (report.get("checks") or {}).items()}
            step["groups_dropped"] = sum(1 for g in outcomes if g.dropped)
            step["seconds"] = time.monotonic() - t0
            point["step"] = step
            rec.event("step", n=n, loss=step["loss"], grad_norm=step["grad_norm"], merge=step["merge"],
                      seconds=step["seconds"])
            if plan.stale_control:
                t0 = time.monotonic()
                stale = G.score_items(model, items, device=device, temperature=T, amp=adapter.amp, extras=False)
                arrays["stale"] = token_arrays(items, {"stale": stale}, None)["stale"]
                rec.arrays(f"point{n}-main", arrays, items, {"rewards": {m.chain_id: (None if m.result is None
                                                                                       else m.result.reward)
                                                                          for gi in groups for m in gi.members}})
                point["stale_control"] = summarize(arrays, ["stale"])
                rec.event("stale_control", n=n, seconds=time.monotonic() - t0,
                          stale_all=point["stale_control"]["stale"]["all"]["tokens"])
            t0 = time.monotonic()
            pub = sync.publish(it, in_flight=rollout.in_flight)
            verdicts = pub.get("verdicts") or {}
            c4t3 = _weights_match_slot(merged.weights, pub["slot"])
            point["publish"] = {"verdicts": {str(k): v for k, v in verdicts.items()}, "changed": pub.get("changed"),
                                "changed_vs_base": pub.get("changed_vs_base"), "coverage": pub.get("coverage"),
                                "aligned": pub.get("aligned"), "changed_frac": pub.get("changed_frac"),
                                "landing": pub.get("landing"), "c4t3": c4t3, "seconds": time.monotonic() - t0}
            rec.event("published", n=n, version=merged.version, verdicts=point["publish"]["verdicts"], c4t3=c4t3["ok"])
            if not c4t3["ok"] or any(v not in ("LANDED", "NOOP") for v in verdicts.values()):
                raise RuntimeError(f"publication of v{merged.version}: verdicts {verdicts}, C4.T3 {c4t3}")
        rec.doc["status"] = "completed"
    except Exception as exc:  # recorded, then raised: the arrays written so far stay usable
        rec.doc["status"] = "failed"
        rec.doc["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        rec.flush()
        raise
    finally:
        try:
            rollout.shutdown()
        finally:
            rec.doc["seconds"] = round(time.monotonic() - rec.t0, 1)
            rec.flush()
    return rec.doc
