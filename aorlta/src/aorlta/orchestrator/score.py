"""Score mode (`orchestrator.mode: score`): generate and score over the prompt pool, with no trainer step, no optimizer
and no weight sync.

    setup     the run plan, with the model's files hashed (`model.base_dir`, and `model.adapter_dir` if set);
              the model to serve (`ModelPort.prepare`: the directory as it is, or a PEFT adapter merged into its
              base by tools/merge_base_adapter.py); the engines start on it; the start-up landing check confirms
              that they serve exactly that model, by its hashes, before anything is scored (`ModelPort.confirm`)
    batches   the selected prompts (the first `orchestrator.score.prompts` of the pool in the run seed's order, or
              all of them), `rollout.P` at a time, once per pass. A batch is one rollout of K chains per prompt and
              one scoring phase, whose unsettled legs are re-checked once, and is committed as `iterations/iter-<n>/`
              (episodes, spans and `score.json`); iteration n numbers the batch, and every batch samples from v0
    report    `score/report.jsonl`: a run row (the model, the landing check, the sampling settings), one row per
              prompt and the totals; `score/summary.txt`, the same in a few lines

D7 does not apply: nothing is trained, so no scored fraction stops the run; it is the report's first number. An
unscored sample is never clean: a sample is clean only when it is scored and its reward function claims the clean
reward. Every sample a batch starts is counted, so a chain an engine lost or the leak check quarantined is unscored.
A score run is not resumed: a stopped one is run again under a new run id.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .. import config as config_mod
from ..atomic import atomic_write_bytes, atomic_write_json, read_text
from ..config import RunConfig
from ..episodes.schema import SCORE_ONLY, build_episode
from ..episodes.writer import IterationOutputs, jsonl_bytes
from ..ids import policy_version, sha256_json, sha256_text
from ..rollout.engine_pool import SELF_TEST_PROMPTS
from ..schemas import RewardResult
from ..spans import SpanRecorder
from . import deadlines as dl
from .loop import RESULT_WAIT_S, RUN_RESULT_SCHEMA, GenerationBarrier, IterationLedger
from .ports import BarrierViolation, MonotonicClock, RunStopped, describe_error
from .state import epoch_order

log = logging.getLogger("aorlta.orchestrator.score")

SCORE_REPORT_SCHEMA = "aorlta.score_report/1"
SCORE_BATCH_SCHEMA = "aorlta.score_batch/1"
REPORT_DIR, REPORT_NAME, SUMMARY_NAME, MODEL_NAME = "score", "report.jsonl", "summary.txt", "model.json"
CHECK_OUTCOMES = ("pass", "fail", "unsettled", "not_run")


def files_digest(files: dict[str, str]) -> str:
    """The model's identity: sha256 over its files' names and sha256s (top-level, not hidden, no merge manifest)."""
    return sha256_json(dict(sorted(files.items())))


class ModelPort(Protocol):
    def prepare(self, recorded: dict[str, str]) -> dict:
        """The model the engines will serve, from the run plan's file hashes (`base/<name>`, `adapter/<name>`): its
        directory, its kind and the sha256 of each of its files (`files`). Merges an adapter if one is configured."""
        ...

    def confirm(self) -> dict:
        """The start-up landing check, after the engines started: `verdict` is PROVEN only when the engines serve
        exactly the prepared model."""
        ...


# --- which prompts, in which batches -------------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    iteration: int  # numbers the batch: chain ids and seeds are the iteration's
    pass_index: int  # from 1
    index: int  # the batch within its pass, from 1
    prompts: tuple


def select_score_prompts(records: Sequence, run_seed: int, limit: int | None) -> list:
    """The pool in the run seed's order (the first epoch's permutation, as a training run sees it), cut to `limit`."""
    order = epoch_order(records, run_seed, 0)
    if limit is not None and limit > len(order):
        raise RunStopped("score_prompts_exceed_pool", {"prompts": limit, "pool": len(order)})
    return order if limit is None else order[:limit]


def plan_batches(prompts: Sequence, P: int, passes: int) -> list[Batch]:
    """Every pass scores every prompt, `P` at a time; the last batch of a pass may be short."""
    out: list[Batch] = []
    for p in range(1, passes + 1):
        for b, i in enumerate(range(0, len(prompts), P), start=1):
            out.append(Batch(len(out) + 1, p, b, tuple(prompts[i:i + P])))
    return out


# --- the report ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One chain as the report reads it."""

    chain_id: str
    prompt_id: str
    iteration: int
    end_reason: str
    final_turn: int  # -1: the chain ended before its first reply
    handed_over: bool  # its candidate reached the reward service
    status: str  # scored | unscored | quarantined
    reward: float | None
    base: float | None  # the reward's base part: its rung on the ladder
    clean: bool
    unscored_reason: str | None
    legs: dict = field(default_factory=dict)  # leg -> state
    not_applicable: tuple = ()  # legs its reward function did not need (for example ConSan on a launch with no LDS)


def sample_of(episode: dict, result: RewardResult | None, *, clean: bool, not_applicable: Iterable[str] = ()) -> Sample:
    """`clean` is the reward function's claim; it counts only for a scored sample."""
    scored = episode["status"] == "scored"
    parts = result.parts if (scored and result is not None and isinstance(result.parts, dict)) else {}
    return Sample(chain_id=episode["chain_id"], prompt_id=episode["prompt_id"], iteration=episode["iteration"],
                  end_reason=episode["end_reason"], final_turn=episode["final_turn"],
                  handed_over=episode["candidate_sha256"] is not None, status=episode["status"],
                  reward=episode["reward"] if scored else None, base=parts.get("base"), clean=scored and bool(clean),
                  unscored_reason=episode["unscored_reason"], legs=dict(episode["legs"]),
                  not_applicable=tuple(sorted(not_applicable)))


def is_clean(s: Sample) -> bool:
    """Unscored is never clean, whatever its reward function claims."""
    return s.status == "scored" and s.clean


def check_outcome(state: str) -> str:
    """A leg's state as the report counts it: clean passes, a finding or a failure fails, unsettled is neither, and a
    leg that did not run (or a plan skipped) is not run."""
    if state == "clean":
        return "pass"
    if state in ("finding", "fail"):
        return "fail"
    if state == "unsettled":
        return "unsettled"
    return "not_run"


def _fraction(n: int, d: int) -> float:
    return n / d if d else 0.0


def reward_distribution(rewards: Sequence[float], bases: Sequence[float | None]) -> dict:
    """Over scored samples only: an unscored sample has no reward, and none is made up for it."""
    xs = sorted(float(r) for r in rewards)
    rungs = Counter(f"{b:g}" for b in bases if b is not None)
    if not xs:
        return {"n": 0, "mean": None, "min": None, "p25": None, "median": None, "p75": None, "max": None,
                "by_base": {}}
    q = statistics.quantiles(xs, n=4, method="inclusive") if len(xs) > 1 else [xs[0]] * 3
    return {"n": len(xs), "mean": statistics.fmean(xs), "min": xs[0], "p25": q[0], "median": q[1], "p75": q[2],
            "max": xs[-1], "by_base": dict(sorted(rungs.items(), key=lambda kv: float(kv[0])))}


def summarize(samples: Sequence[Sample], checks: Iterable[str] = ()) -> dict:
    """The counts of one prompt's samples, or of all of them. Each check (`checks`, and every leg a sample has) is
    counted over every sample: a sample without that leg, such as a chain the reward service never saw, did not run
    it."""
    n = len(samples)
    scored = [s for s in samples if s.status == "scored"]
    clean = sum(1 for s in samples if is_clean(s))
    counts = {leg: dict.fromkeys(CHECK_OUTCOMES + ("not_applicable",), 0)
              for leg in sorted(set(checks).union(*(s.legs for s in samples)))}
    for s in samples:
        for leg, row in counts.items():
            row[check_outcome(s.legs.get(leg, "not_run"))] += 1
            row["not_applicable"] += leg in s.not_applicable
    replied = [s for s in samples if s.final_turn >= 0]
    compiled = [s for s in samples if s.end_reason == "compiled"]
    return {
        "samples": n, "scored": len(scored), "unscored": n - len(scored), "clean": clean,
        "scored_fraction": _fraction(len(scored), n), "clean_fraction": _fraction(clean, n),
        "handed_over": sum(1 for s in samples if s.handed_over),
        "not_handed_over": dict(sorted(Counter(s.end_reason for s in samples if not s.handed_over).items())),
        "unscored_by_reason": dict(sorted(Counter(s.unscored_reason or "no_reason" for s in samples
                                                  if s.status != "scored").items())),
        "reward": reward_distribution([s.reward for s in scored], [s.base for s in scored]),
        "checks": counts,
        "compile": {"compiled": len(compiled), "fraction": _fraction(len(compiled), n),
                    "at_turn": {str(t): c for t, c in sorted(Counter(s.final_turn for s in compiled).items())}},
        "repair_turns": {"used": {str(t): c for t, c in sorted(Counter(s.final_turn for s in replied).items())},
                         "mean": statistics.fmean(s.final_turn for s in replied) if replied else None},
        "end_reasons": dict(sorted(Counter(s.end_reason for s in samples).items())),
    }


def build_report(samples: Sequence[Sample], prompts: Sequence, header: dict) -> list[dict]:
    """The report's rows: the run, one per prompt (in the order scored), then the totals. Every row counts the same
    checks: the legs of any sample."""
    by_prompt: dict[str, list[Sample]] = {p.prompt_id: [] for p in prompts}
    for s in samples:
        if s.prompt_id not in by_prompt:
            raise ValueError(f"{s.chain_id}: prompt {s.prompt_id!r} was not selected")
        by_prompt[s.prompt_id].append(s)
    checks = set().union(*(s.legs for s in samples))
    rows = [dict(header, kind="run", schema=SCORE_REPORT_SCHEMA)]
    for p in prompts:
        mine = by_prompt[p.prompt_id]
        rows.append({"kind": "prompt", "prompt_id": p.prompt_id, "task_sha256": p.task.task_sha256,
                     "any_clean": any(is_clean(s) for s in mine), **summarize(mine, checks)})
    total = {"kind": "total", "prompts": len(prompts),
             "prompts_with_a_clean_sample": sum(1 for r in rows[1:] if r["any_clean"]), **summarize(samples, checks)}
    return rows + [total]


def _num(x: float | None, fmt: str = ".3f") -> str:
    return "-" if x is None else format(x, fmt)


def summary_text(rows: Sequence[dict]) -> str:
    """A few lines a person reads first; the scored fraction leads."""
    run, total = rows[0], rows[-1]
    s, m, lc = run["sampling"], run["model"], run["landing"]
    rw, comp = total["reward"], total["compile"]
    ran = {leg: c for leg, c in total["checks"].items() if c["pass"] + c["fail"] + c["unsettled"] > 0}
    never = len(total["checks"]) - len(ran)  # legs the reward function reads that the profile never ran
    lines = [
        f"AORLTA score run {run['run_id']} ({run['source']}), {run['written_at']}",
        f"Scored {total['scored']} of {total['samples']} samples ({100 * total['scored_fraction']:.1f}%). "
        f"An unscored sample is never counted clean.",
        f"Clean {total['clean']} of {total['samples']} samples ({100 * total['clean_fraction']:.1f}%); "
        f"{total['prompts_with_a_clean_sample']} of {total['prompts']} prompts have a clean sample.",
        f"Reward over the {rw['n']} scored samples: mean {_num(rw['mean'])}, median {_num(rw['median'])}, "
        f"min {_num(rw['min'])}, max {_num(rw['max'])}; by base: "
        + (", ".join(f"{k}: {v}" for k, v in sorted(rw["by_base"].items(), key=lambda kv: float(kv[0]))) or "-"),
        f"Compiled {comp['compiled']} of {total['samples']} ({100 * comp['fraction']:.1f}%); repair turns used: "
        + (", ".join(f"{k}: {v}" for k, v in total["repair_turns"]["used"].items()) or "-"),
        "Checks that ran (pass / fail / unsettled / not run; of them not applicable): "
        + ("; ".join(f"{leg} {c['pass']}/{c['fail']}/{c['unsettled']}/{c['not_run']} ({c['not_applicable']})"
                     for leg, c in ran.items()) or "-")
        + (f"; {never} more never ran" if never else ""),
        "Unscored by reason: " + (", ".join(f"{k}: {v}" for k, v in total["unscored_by_reason"].items()) or "-"),
        f"Model: {m['kind']} {m['dir']} (files sha256 {m['files_sha256'][:16]}); landing check {lc['verdict']} on "
        f"{lc.get('engines')} engines",
        f"Sampling: K {s['K']}, {run['options']['passes']} pass(es) over {total['prompts']} prompts, temperature "
        f"{s['temperature']}, top_p {s['top_p']}, top_k {s['top_k']}, max_new_tokens {s['max_new_tokens']}, thinking "
        f"{s['thinking']}, {s['max_repair_turns']} repair turns",
    ]
    return "\n".join(lines) + "\n"


# --- the run -------------------------------------------------------------------------------------------------


def model_files(cfg: RunConfig) -> dict[str, Path]:
    """Every file of the model directory (`base/<name>`) and of the adapter (`adapter/<name>`), for the run plan."""
    out: dict[str, Path] = {}
    dirs = [("base", cfg.model.base_dir)] + ([("adapter", cfg.model.adapter_dir)] if cfg.model.adapter_dir else [])
    for prefix, d in dirs:
        root = Path(d)
        if not root.is_dir():
            raise RunStopped(f"{prefix}_dir_missing", {f"{prefix}_dir": str(root)})
        mine = {f"{prefix}/{p.relative_to(root).as_posix()}": p for p in sorted(root.rglob("*"))
                if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)}
        if not mine:
            raise RunStopped(f"{prefix}_dir_empty", {f"{prefix}_dir": str(root)})
        out.update(mine)
    return out


class Scorer:
    """The score run over the same ports as the training loop, without the trainer and the sync: the rollout, the
    reward service (which must also say whether a result is clean, and which legs its reward function did not need),
    and the model to serve."""

    def __init__(self, cfg: RunConfig, *, pool: Any, rollout: Any, reward: Any, model: ModelPort, clock: Any = None,
                 spans: SpanRecorder | None = None, point: Callable[[str, int], None] | None = None,
                 commit: str | None = None, package_dir: str | Path | None = None,
                 config_path: str | Path | None = None):
        if cfg.orchestrator.mode != "score":
            raise ValueError(f"the scorer runs orchestrator.mode score, not {cfg.orchestrator.mode!r}")
        self.cfg, self.run_dir = cfg, cfg.run_dir
        self.source, self.run_id = cfg.run.source, cfg.run.run_id
        self.pool, self.rollout, self.reward, self.model = pool, rollout, reward, model
        self.clock = clock or MonotonicClock()
        self.spans = spans or SpanRecorder(monotonic=self.clock.now)
        self._point = point or (lambda name, iteration: None)
        self.commit_id, self.package_dir, self.config_path = commit, package_dir, config_path
        self.barrier = GenerationBarrier(rollout.in_flight)
        self.history: dict[str, list[float]] = {"generation_s": [], "scoring_s": []}
        self.batches: list[Batch] = []
        self.prompts: list = []
        self.samples: list[Sample] = []
        self.identity: dict | None = None
        self.landing: dict | None = None
        self.engines: dict | None = None
        self.totals: dict | None = None
        self.committed = 0
        self._owns_run = False

    def point(self, name: str, iteration: int) -> None:
        self._point(name, iteration)

    def run(self, *, resume: bool = False) -> dict:
        try:
            if resume:
                raise RunStopped("score_runs_are_not_resumed", {"run_dir": str(self.run_dir)})
            self._setup()
            for batch in self.batches:
                self._batch(batch)
            self._report()
            return self._write_result("completed")
        except RunStopped as stop:
            log.error("score run stopped after %d of %d batches: %s", self.committed, len(self.batches), stop.reason)
            return self._write_result("stopped", stop)
        except Exception as exc:
            self._write_result("stopped", RunStopped("unexpected_error", {"error": describe_error(exc, self.source)}))
            raise
        finally:
            self._close()

    # --- setup -------------------------------------------------------------------------------------------------

    def _setup(self) -> None:
        cfg, o = self.cfg, self.cfg.orchestrator
        for name in ("claims_clean", "not_applicable"):
            if not callable(getattr(self.reward, name, None)):
                raise RunStopped("reward_port_cannot_score", {"missing": name})
        with self.spans.span("setup.plan", 0):
            config_mod.write_run_plan(cfg, config_path=self.config_path, commit=self.commit_id,
                                      package_dir=self.package_dir, extra_files=model_files(cfg))
            self._owns_run = True
            self._write_result("running")
        records = list(self.pool.records())
        for r in records:
            if r.source != self.source:
                raise RunStopped("mixed_sources", {"prompt_id": r.prompt_id, "prompt_source": r.source})
        self.prompts = select_score_prompts(records, o.run_seed, o.score.prompts)
        self.batches = plan_batches(self.prompts, cfg.rollout.P, o.score.passes)
        if not self.batches or len(self.batches[0].prompts) < SELF_TEST_PROMPTS:
            raise RunStopped("too_few_prompts", {"selected": len(self.prompts), "needed": SELF_TEST_PROMPTS,
                                                 "why": "the engines' start-up self-test runs on the first batch"})
        plan = json.loads(read_text(self.run_dir / config_mod.RUN_PLAN_NAME))
        recorded = {k: v["sha256"] for k, v in plan.get("extra_files", {}).items()}
        with self.spans.span("setup.model", 0, adapter=cfg.model.adapter_dir is not None):
            try:
                self.identity = self.model.prepare(recorded)
            except RunStopped:
                raise
            except Exception as exc:
                raise RunStopped("model_prepare_failed", {"error": describe_error(exc, self.source)}) from None
        with self.spans.span("setup.engines", 0):
            self.engines = self.rollout.start(list(self.batches[0].prompts[:SELF_TEST_PROMPTS]))
        with self.spans.span("setup.landing", 0):
            try:
                self.landing = self.model.confirm()
            except RunStopped:
                raise
            except Exception as exc:
                raise RunStopped("served_model_unconfirmed", {"error": describe_error(exc, self.source)}) from None
        out = self.run_dir / REPORT_DIR
        out.mkdir(exist_ok=True)
        atomic_write_json(out / MODEL_NAME, {"model": self.identity, "landing": self.landing, "engines": self.engines})
        if self.landing.get("verdict") != "PROVEN":
            raise RunStopped("served_model_unconfirmed", {"landing": self.landing})
        attempt_dir = self.run_dir / "attempts"
        attempt_dir.mkdir(exist_ok=True)
        atomic_write_bytes(attempt_dir / "attempt-001-setup-spans.jsonl",
                           jsonl_bytes(s.to_json() for s in self.spans.spans(0)))

    # --- one batch ---------------------------------------------------------------------------------------------

    def _batch(self, batch: Batch) -> None:
        n, o, K = batch.iteration, self.cfg.orchestrator, self.cfg.rollout.K
        self.point("select", n)
        warnings: list[str] = []
        sampled = policy_version(self.run_id, 0)
        self.reward.reopen()
        ledger = IterationLedger(self.run_id, n, batch.prompts, K, self.reward, lambda group: None, self.point)
        gen_s = dl.generation_deadline_s(o.deadlines.generation_s, self.history["generation_s"])
        t_gen = self.clock.now()
        try:
            outcome = self.rollout.run(list(batch.prompts), n, sampled, on_candidate=ledger.on_candidate,
                                       on_chain=ledger.on_chain, deadline=t_gen + gen_s)
        except RunStopped:
            raise
        except Exception as exc:
            raise RunStopped("rollout_failed", {"error": describe_error(exc, self.source)}) from None
        ledger.generation_done(outcome)
        self.point("gen.done", n)
        try:
            self.barrier.assert_idle("generation barrier")
        except BarrierViolation as exc:
            raise RunStopped("barrier_violation", {"iteration": n, "error": str(exc)[:300]}) from None
        t_barrier = self.clock.now()
        stops = [self._safe(s) for s in getattr(outcome, "stop_reasons", [])]
        if stops or ledger.problems:
            raise RunStopped("rollout_stop", {"stop_reasons": stops,
                                              "problems": [self._safe(p) for p in ledger.problems]})
        if any(rec.end_reason == "engine_error" for rec in ledger.records.values()):
            warnings.append("engine_error_chains")
        score_s = dl.scoring_deadline_s(o.deadlines.scoring_s, self.history["scoring_s"])
        with self.spans.span("score.wait", n, deadline_s=score_s):
            drain = self.reward.drain(t_barrier + score_s)
            late = ledger.wait_results(RESULT_WAIT_S)
        score_dur = self.clock.now() - t_barrier  # the first phase only, as the training loop takes it for the history
        if late:
            warnings.append("reward_unresolved")
        outstanding = list(getattr(drain, "outstanding_at_deadline", []) or [])
        first = {"scored": ledger.scored(), "candidates": ledger.handed_over(), "retryable": len(ledger.retryable())}
        retry = None
        cands = ledger.retryable()
        if cands:  # the unsettled legs, once more, with 1.5 x the deadline: as D7's retry, whatever the fraction
            retry_s = dl.RETRY_DEADLINE_FACTOR * score_s
            with self.spans.span("score.retry", n, candidates=len(cands), deadline_s=retry_s):
                self.reward.reopen()
                ledger.resubmit(cands)
                drain2 = self.reward.drain(self.clock.now() + retry_s)
                ledger.wait_results(RESULT_WAIT_S)
            outstanding += list(getattr(drain2, "outstanding_at_deadline", []) or [])
            retry = {"candidates": len(cands), "deadline_s": retry_s, "scored": ledger.scored()}
        self.point("score.drained", n)
        episodes, samples = self._episodes(n, ledger)
        outs = IterationOutputs(self.run_dir, n)
        outs.write_episodes(episodes)
        outs.write_json("score.json", {
            "schema": SCORE_BATCH_SCHEMA, "iteration": n, "pass": batch.pass_index, "batch": batch.index,
            "prompts": [p.prompt_id for p in batch.prompts], "policy_version_sampled": sampled,
            "chains": ledger.total, "first": first, "retry": retry,
            "scored": sum(1 for s in samples if s.status == "scored"), "clean": sum(1 for s in samples if is_clean(s)),
            "deadlines": {"generation_s": gen_s, "generation_dur_s": t_barrier - t_gen, "scoring_s": score_s,
                          "scoring_dur_s": score_dur, "outstanding_at_deadline": outstanding,
                          "generation_outstanding": list(getattr(outcome, "outstanding", []) or [])},
            "warnings": warnings, "rollout": self._safe_metrics(getattr(outcome, "metrics", {}) or {}),
        })
        outs.write_spans(self.spans.spans(n))
        outs.commit()
        self.samples.extend(samples)
        self.committed = n
        self.history = {"generation_s": dl.push_history(self.history["generation_s"], t_barrier - t_gen),
                        "scoring_s": dl.push_history(self.history["scoring_s"], score_dur)}
        self.point("record.committed", n)
        log.info("batch %d of %d (pass %d) committed: scored %d of %d", n, len(self.batches), batch.pass_index,
                 sum(1 for s in samples if s.status == "scored"), len(samples))

    def _episodes(self, n: int, ledger: IterationLedger) -> tuple[list[dict], list[Sample]]:
        episodes, samples = [], []
        for cid in sorted(ledger.expected):
            rec, cand = ledger.records[cid], ledger.candidates.get(cid)
            result = ledger.results.get(cid) if cand is not None else None
            ep = build_episode(source=self.source, run_id=self.run_id, iteration=n, record=rec, candidate=cand,
                               result=result, advantage=None, trained=False, excluded_reason=SCORE_ONLY)
            clean = result is not None and bool(self.reward.claims_clean(result))
            na = self.reward.not_applicable(result, cand) if result is not None else ()
            episodes.append(ep)
            samples.append(sample_of(ep, result, clean=clean, not_applicable=na))
        return episodes, samples

    # --- the report and the run's records -----------------------------------------------------------------------

    def header(self) -> dict:
        cfg, r, o = self.cfg, self.cfg.rollout, self.cfg.orchestrator
        model = {k: v for k, v in (self.identity or {}).items() if k != "files"}
        return {
            "run_id": self.run_id, "source": self.source,
            "written_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "policy_version": policy_version(self.run_id, 0), "model": model,
            "landing": {k: v for k, v in (self.landing or {}).items() if k != "round_trip"},
            "options": {"prompts": o.score.prompts, "passes": o.score.passes, "prompts_selected": len(self.prompts),
                        "prompts_per_batch": r.P, "batches": len(self.batches), "order_seed": o.run_seed},
            "sampling": {"K": r.K, "temperature": r.temperature, "top_p": r.top_p, "top_k": r.top_k,
                         "max_new_tokens": r.max_new_tokens, "thinking": r.thinking,
                         "max_repair_turns": r.max_repair_turns, "repair_turn_ids": r.repair_turn_ids,
                         "parse_adapter": r.parse_adapter, "feedback_template": r.feedback_template,
                         "compile_adapter": r.compile_adapter, "run_seed": o.run_seed,
                         "seed_rule": "ids.seed(run_seed, iteration, prompt_id, k, turn), one request per turn"},
            "reward": {"profile": cfg.reward.profile, "reward_fn": cfg.reward.reward_fn},
            "engines": {k: v for k, v in (self.engines or {}).items() if k != "self_tests"},
        }

    def _report(self) -> None:
        rows = build_report(self.samples, self.prompts, self.header())
        out = self.run_dir / REPORT_DIR
        atomic_write_bytes(out / REPORT_NAME, jsonl_bytes(rows))
        text = summary_text(rows)
        atomic_write_bytes(out / SUMMARY_NAME, text.encode("utf-8"))
        self.totals = {k: v for k, v in rows[-1].items() if k in (
            "prompts", "samples", "scored", "unscored", "clean", "scored_fraction", "clean_fraction",
            "prompts_with_a_clean_sample")}
        log.info("score report: %s", text.splitlines()[1])

    def _write_result(self, status: str, stop: RunStopped | None = None) -> dict:
        result = {
            "schema": RUN_RESULT_SCHEMA, "mode": "score", "run_id": self.run_id, "source": self.source,
            "status": status, "iterations_planned": len(self.batches), "iterations_completed": self.committed,
            "policy_version": policy_version(self.run_id, 0),
            "stop": None if stop is None else {"reason": stop.reason, "detail": stop.detail},
            "attempts": [{"attempt": 1, "status": status, "resumed_from": None, "iterations_completed": self.committed,
                          "stop": None if stop is None else {"reason": stop.reason, "detail": stop.detail}}],
            "eval": None, "score": self.totals,
        }
        if self._owns_run:  # a process refused at the run plan never touches the run's records
            atomic_write_json(self.run_dir / "run_result.json", result)
        return result

    def _close(self) -> None:
        if not self._owns_run:  # nothing was started, and the rollout's shutdown removes the run's node directory
            return
        for name, fn in (("reward", getattr(self.reward, "close", None)),
                         ("rollout", getattr(self.rollout, "shutdown", None))):
            if fn is not None:
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    log.warning("closing %s: %s", name, describe_error(exc, self.source))

    def _safe(self, text: str) -> str:
        """A component's message for the run records: kept for stand-in runs, hashed for any other source (6.10)."""
        return str(text)[:500] if self.source == "standin" else f"sha256:{sha256_text(str(text))[:16]}"

    def _safe_metrics(self, metrics: dict) -> dict:
        m = dict(metrics)
        if "deaths" in m:
            m["deaths"] = [{k: (self._safe(v) if k == "error" else v) for k, v in d.items()} for d in m["deaths"]]
        return m
