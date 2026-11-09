"""The iteration summary (design Section 6.8, "Iteration summary"; Section 6.9 for the timing).

A pure function of the iteration's records: reward mean, std and histogram; status counts; unscored by reason;
gate failure counts; compile success per turn and the repair rescue rate; truncation; spans for every phase; cache
hit rate per leg; the reward service's worker and ASan figures; trainer stats; sync stats; the scored fraction.
Everything is ids, counts and timings (Section 6.10 item 3).
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import REWARD_MAX, REWARD_MIN
from ..schemas import RewardResult
from ..spans import Span
from .schema import SUMMARY_SCHEMA

HIST_WIDTH = 0.15  # 17 bins over [-1.55, 1.0]
SPAN_GAP_FLAG = 0.05  # a gap above 5% of the iteration's wall time is flagged (Section 6.6, failure modes)
ALWAYS_SPANS = ("select", "gen.turn0", "score.wait", "record", "ckpt")
STEP_SPANS = ("train.final", "merge", "write", "reload", "landing")


def plain_stats(stats: Any) -> dict:
    """A component's stats object as plain JSON (a dataclass or a mapping); anything else is not reported."""
    if dataclasses.is_dataclass(stats) and not isinstance(stats, type):
        stats = dataclasses.asdict(stats)
    return _jsonable(stats) if isinstance(stats, Mapping) else {}


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(type(x).__name__)


def histogram(values: Sequence[float]) -> dict:
    n_bins = round((REWARD_MAX - REWARD_MIN) / HIST_WIDTH)
    edges = [round(REWARD_MIN + i * HIST_WIDTH, 10) for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in values:
        i = min(n_bins - 1, max(0, int((v - REWARD_MIN) / HIST_WIDTH)))
        counts[i] += 1
    return {"edges": edges, "counts": counts}


def union_length(intervals: Sequence[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def span_check(spans: Sequence[Span], t0: float, t1: float, *, step_taken: bool, any_repair: bool,
               any_compile: bool, any_trained_group: bool) -> dict:
    names = Counter(s.name for s in spans)
    required = set(ALWAYS_SPANS)
    if step_taken:
        required |= set(STEP_SPANS)
    if any_repair:
        required.add("gen.repair")
    if any_compile:
        required.add("compile")
    if any_trained_group:
        required.add("train.stream")
    wall = max(t1 - t0, 1e-9)
    covered = union_length([(max(s.t_start, t0), min(s.t_end, t1)) for s in spans if s.t_end > t0 and s.t_start < t1])
    gap = max(0.0, 1.0 - covered / wall)
    return {"names": dict(sorted(names.items())), "missing": sorted(required - set(names)), "gap_fraction": gap,
            "gap_flag": gap > SPAN_GAP_FLAG}


def phases(spans: Sequence[Span]) -> dict:
    by: dict[str, list[Span]] = {}
    for s in spans:
        by.setdefault(s.name, []).append(s)
    return {name: {"count": len(ss), "total_s": math.fsum(s.duration for s in ss),
                   "union_s": union_length([(s.t_start, s.t_end) for s in ss]),
                   "first_start": min(s.t_start for s in ss), "last_end": max(s.t_end for s in ss)}
            for name, ss in sorted(by.items())}


def _pct(xs: Sequence[float], q: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def trainer_stats(step: Mapping | None, noop: Mapping | None = None) -> dict:
    """The step's figures; for an iteration that took no step, what the open step had measured (`noop`), if any."""
    if step is None:
        out: dict[str, Any] = {"step_taken": False}
        if noop is not None:
            out["noop"] = {"counts": _jsonable(noop.get("counts") or {}), "gap": _jsonable(noop.get("gap") or {}),
                           "gap_gate": _jsonable(noop.get("gap_gate")), "memory": _jsonable(noop.get("memory"))}
        return out
    counts = step.get("counts") or {}
    gap = step.get("gap") or {}
    loss_tokens = counts.get("loss_tokens") or 0
    return {
        "step_taken": True, "step": step.get("step"), "loss": step.get("loss"), "grad_norm": step.get("grad_norm"),
        "excluded": len(step.get("excluded") or []), "excluded_ids": list(step.get("excluded") or []),
        "gap_median": gap.get("median"), "gap_p99": gap.get("p99"), "gap": _jsonable(gap),
        "gap_gate": _jsonable(step.get("gap_gate")), "memory": _jsonable(step.get("memory")),
        "seconds": _jsonable({"forward_backward_by_rank": step.get("seconds_by_rank"), "step": step.get("step_sec"),
                              "reduce": step.get("reduce_sec")}),
        "clip_fraction": (counts.get("clipped_tokens", 0) / loss_tokens) if loss_tokens else None,
        "kl": step.get("kl"), "counts": _jsonable(counts), "checks": _jsonable(step.get("checks") or {}),
    }


def sync_stats(report: Mapping | None) -> dict:
    if report is None:
        return {"published": False, "landing": None}
    verdicts = {str(k): v for k, v in (report.get("verdicts") or {}).items()}
    values = set(verdicts.values())
    landing = values.pop() if len(values) == 1 else ("MIXED" if values else None)
    seconds = report.get("seconds") or {}
    return {"published": True, "verdicts": verdicts, "landing": landing, "changed": report.get("changed"),
            "coverage": report.get("coverage"), "changed_frac": report.get("changed_frac"),
            "aligned": report.get("aligned"), "changed_vs_base": report.get("changed_vs_base"),
            "rounding": report.get("rounding"),
            "reload_s": seconds.get("reload"), "seconds": _jsonable(seconds), "slot": report.get("slot")}


def asan_delta(now: Mapping, before: Mapping) -> dict:
    """The iteration's ASan requests from two cumulative snapshots of the reward service's stats."""
    a, b = now.get("asan_requests") or {}, before.get("asan_requests") or {}
    sizes_a, sizes_b = a.get("sizes") or {}, b.get("sizes") or {}
    sizes = {k: int(v) - int(sizes_b.get(k, 0)) for k, v in sizes_a.items() if int(v) - int(sizes_b.get(k, 0)) > 0}
    n = sum(sizes.values())
    return {"requests": (a.get("count") or 0) - (b.get("count") or 0), "sizes": dict(sorted(sizes.items())),
            "batch_size_mean": (sum(int(k) * v for k, v in sizes.items()) / n) if n else None,
            "max_wait_s": a.get("max_wait_s"), "max_wait_is_cumulative": True}


def build_summary(*, run_id: str, source: str, iteration: int, sampled: str, trained: str, episodes: Sequence[dict],
                  results: Mapping[str, RewardResult], total: int, reward_stats: Any, groups: Mapping[str, Any],
                  step: Mapping | None, sync: Mapping | None, policy: Mapping, deadlines: Mapping,
                  spans: Sequence[Span], t_start: float, t_landed: float, noop_streak: int, warnings: Sequence[str],
                  eval_events: Sequence[dict], rollout_metrics: Mapping | None, reward_stats_before: Any = None,
                  trainer_noop: Mapping | None = None) -> dict:
    status = Counter(e["status"] for e in episodes)
    ends = Counter(e["end_reason"] for e in episodes)
    unscored = Counter(e["unscored_reason"] for e in episodes if e["status"] != "scored")
    scored_rewards = [e["reward"] for e in episodes if e["status"] == "scored"]
    samples = {"total": total, "chains": len(episodes), "candidates": sum(1 for e in episodes if e["candidate_sha256"]),
               "scored": status.get("scored", 0), "unscored": status.get("unscored", 0),
               "quarantined": status.get("quarantined", 0), "trained": sum(1 for e in episodes if e["trained"])}
    gates: Counter = Counter()
    cache: dict[str, dict] = {}
    for r in results.values():
        for leg, lr in r.legs.items():
            if leg.startswith("gate.") and lr.state == "fail":
                gates[leg] += 1
            if lr.state in ("not_run", "skipped_by_plan"):
                continue
            row = cache.setdefault(leg, {"runs": 0, "hits": 0})
            row["runs"] += 1
            row["hits"] += bool(lr.cache_hit)
    for row in cache.values():
        row["hit_rate"] = row["hits"] / row["runs"] if row["runs"] else None
    with_turns = [e for e in episodes if e["turns"]]
    failed0 = [e for e in with_turns if e["turns"][0]["compile"] is not None and not e["turns"][0]["compile"]["ok"]]
    per_turn: dict[str, dict] = {}
    max_t = max((len(e["turns"]) for e in with_turns), default=0)
    for t in range(max_t):
        rows = [e["turns"][t] for e in with_turns if len(e["turns"]) > t]
        comp = [tr for tr in rows if tr["compile"] is not None]
        ok = sum(1 for tr in comp if tr["compile"]["ok"])
        rescued = sum(1 for e in failed0 if len(e["turns"]) > t and t > 0 and e["turns"][t]["compile"] is not None
                      and e["turns"][t]["compile"]["ok"])
        per_turn[str(t)] = {
            "turns": len(rows), "compiled": len(comp), "ok": ok, "ok_frac": ok / len(comp) if comp else None,
            "length_capped": sum(1 for tr in rows if tr["finish_reason"].get("type") == "length"),
            "rescued": rescued if t > 0 else None,
            "rescue_frac": (rescued / len(failed0) if failed0 else None) if t > 0 else None,
        }
    rescued_total = sum(1 for e in failed0 if e["end_reason"] == "compiled")
    final_capped = sum(1 for e in with_turns if e["turns"][-1]["finish_reason"].get("type") == "length")
    final_lens = [len(e["turns"][-1]["response_ids"]) for e in with_turns if e["end_reason"] != "quarantined"]
    groups_trained = sum(1 for g in groups.values() if getattr(g, "trained", ()))
    dropped = Counter(g.dropped for g in groups.values() if getattr(g, "dropped", None))
    zero_adv = sum(1 for g in groups.values() if not g.dropped and g.advantages
                   and all(a == 0.0 for a in g.advantages.values()))
    rstats = plain_stats(reward_stats)
    # The service's counters are cumulative over its life (which a resume restarts): report them as such, and the
    # iteration's own ASan requests as the difference from the previous snapshot.
    reward_service = {"cumulative": rstats, "asan": asan_delta(rstats, plain_stats(reward_stats_before))}
    wall = t_landed - t_start
    scored = samples["scored"]
    any_compile = any(tr["compile"] is not None for e in with_turns for tr in e["turns"])
    span_info = span_check(spans, t_start, t_landed, step_taken=step is not None,
                           any_repair=any(len(e["turns"]) > 1 for e in with_turns), any_compile=any_compile,
                           any_trained_group=groups_trained > 0)
    warnings = list(warnings) + (["span_gap"] if span_info["gap_flag"] else [])
    return {
        "schema": SUMMARY_SCHEMA, "source": source, "run_id": run_id, "iteration": iteration,
        "policy_version_sampled": sampled, "policy_version_trained": trained, "samples": samples,
        "scored_fraction": scored / total if total else 0.0, "status_counts": dict(sorted(status.items())),
        "unscored_by_reason": dict(sorted(unscored.items())), "end_reasons": dict(sorted(ends.items())),
        "reward": {"count": len(scored_rewards),
                   "mean": statistics.fmean(scored_rewards) if scored_rewards else None,
                   "std": statistics.pstdev(scored_rewards) if len(scored_rewards) > 1 else None,
                   "min": min(scored_rewards, default=None), "max": max(scored_rewards, default=None),
                   "histogram": histogram(scored_rewards)},
        "gates": dict(sorted(gates.items())),
        "compile": {"per_turn": per_turn, "repair_rescue": {"failed_turn0": len(failed0), "rescued": rescued_total,
                                                            "rate": rescued_total / len(failed0) if failed0 else None}},
        "truncation": {"final_length_capped": final_capped,
                       "rate": final_capped / len(with_turns) if with_turns else None},
        "length": {"response_mean": statistics.fmean(final_lens) if final_lens else None,
                   "response_p50": _pct(final_lens, 0.5), "response_max": max(final_lens, default=None)},
        "cache": dict(sorted(cache.items())), "reward_service": reward_service,
        "groups": {"total": len(groups), "trained": groups_trained, "dropped": dict(sorted(dropped.items())),
                   "zero_advantage": zero_adv},
        "trainer": trainer_stats(step, trainer_noop), "sync": sync_stats(sync), "policy": _jsonable(policy),
        "deadlines": _jsonable(deadlines),
        "timing": {"t_start": t_start, "t_landed": t_landed, "wall_s": wall,
                   "per_256_scored_s": 256.0 * wall / scored if scored else None, "phases": phases(spans)},
        "spans": span_info, "noop_streak": noop_streak, "warnings": warnings, "eval": _jsonable(list(eval_events)),
        "rollout": _jsonable(dict(rollout_metrics or {})),
    }
