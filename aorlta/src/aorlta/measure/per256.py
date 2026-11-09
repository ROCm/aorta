"""C9 per-256 metric: wall time per 256 scored samples (design Section 6.9, Metrics).

    per256(n) = 256 x wall(n) / scored(n)

Reported over iterations 2..N as the median and IQR, and as the mean with and without stalls
(a baseline may exclude its own stalls). Iteration 1 is left out of all four: it is the first contact of
cold caches and fresh workers. Unscored samples (reward None) never count, so a run cannot look
faster by scoring less, and each iteration is divided by its own scored count.

A stall is whatever the input flags as one (for AORLTA, an iteration that hit a deadline, Section 6.6;
for a baseline, the iterations its owner excluded). Nothing here infers stalls from durations, since a threshold
chosen after seeing the data would decide the result.

Quartiles are the "inclusive" method of `statistics.quantiles` (linear interpolation between order
statistics; numpy's default).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic import read_text
from ..spans import Span
from . import MeasureError
from .timeline import run_bounds

SCORED_UNIT = 256
FIRST_REPORTED = 2
EPISODE_SCHEMA = "aorlta.episode/1"


class Per256Error(MeasureError):
    pass


@dataclass(frozen=True)
class IterationStats:
    iteration: int
    wall_s: float
    samples: int
    scored: int
    stalled: bool
    stall_reason: str | None = None

    def __post_init__(self) -> None:
        n = self.iteration
        if type(n) is not int or n < 1:
            raise Per256Error(f"iteration must be an int >= 1, got {n!r}")
        if type(self.wall_s) not in (int, float) or not math.isfinite(self.wall_s) or self.wall_s <= 0:
            raise Per256Error(f"iteration {n}: wall_s must be a finite number > 0, got {self.wall_s!r}")
        for key in ("samples", "scored"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise Per256Error(f"iteration {n}: {key} must be an int >= 0, got {getattr(self, key)!r}")
        if self.scored > self.samples:
            raise Per256Error(f"iteration {n}: scored {self.scored} > samples {self.samples}")
        if type(self.stalled) is not bool:
            raise Per256Error(f"iteration {n}: stalled must be true or false, got {self.stalled!r}")
        if self.stalled and not (isinstance(self.stall_reason, str) and self.stall_reason):
            raise Per256Error(f"iteration {n}: a stall needs its reason")


def per256(wall_s: float, scored: int) -> float:
    if type(scored) is not int or scored <= 0:
        raise Per256Error(f"no scored samples ({scored!r}): wall time per 256 scored is undefined")
    return SCORED_UNIT * wall_s / scored


@dataclass(frozen=True)
class Per256Summary:
    iterations: tuple[int, ...]  # the reported iterations, 2..N
    per256_s: dict[int, float]  # every iteration given, 1..N
    median_s: float
    q1_s: float | None  # None with a single reported iteration
    q3_s: float | None
    mean_with_stalls_s: float
    mean_without_stalls_s: float | None  # None if every reported iteration stalled
    stalled: tuple[int, ...]  # reported iterations flagged as stalls
    scored: int  # over the reported iterations
    samples: int

    @property
    def iqr_s(self) -> float | None:
        return None if self.q1_s is None else self.q3_s - self.q1_s

    @property
    def scored_fraction(self) -> float:
        return self.scored / self.samples

    def statistic(self, name: str) -> float | None:
        return {"median": self.median_s, "mean_with_stalls": self.mean_with_stalls_s,
                "mean_without_stalls": self.mean_without_stalls_s}[name]

    def to_json(self) -> dict[str, Any]:
        return {"iterations": list(self.iterations), "per256_s": {str(k): v for k, v in self.per256_s.items()},
                "median_s": self.median_s, "q1_s": self.q1_s, "q3_s": self.q3_s, "iqr_s": self.iqr_s,
                "mean_with_stalls_s": self.mean_with_stalls_s, "mean_without_stalls_s": self.mean_without_stalls_s,
                "stalled": list(self.stalled), "scored": self.scored, "samples": self.samples,
                "scored_fraction": self.scored_fraction}


def summarize(stats: Sequence[IterationStats], *, last: int | None = None) -> Per256Summary:
    """Iterations 1..N must all be present (N = `last`, or the largest given); 2..N are reported."""
    by_it: dict[int, IterationStats] = {}
    for s in stats:
        if s.iteration in by_it:
            raise Per256Error(f"iteration {s.iteration} appears twice")
        by_it[s.iteration] = s
    if not by_it:
        raise Per256Error("no iterations")
    n_last = max(by_it) if last is None else last
    missing = [n for n in range(1, n_last + 1) if n not in by_it]
    if missing:
        raise Per256Error(f"iterations missing from 1..{n_last}: {missing}; a hole would bias the summary")
    reported = [by_it[n] for n in range(FIRST_REPORTED, n_last + 1)]
    if not reported:
        raise Per256Error(f"no iterations from {FIRST_REPORTED} on: the metric leaves out iteration 1")
    values = {n: per256(by_it[n].wall_s, by_it[n].scored) for n in range(1, n_last + 1)}
    xs = [values[s.iteration] for s in reported]
    calm = [values[s.iteration] for s in reported if not s.stalled]
    q1 = q3 = None
    if len(xs) >= 2:
        q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return Per256Summary(
        iterations=tuple(s.iteration for s in reported), per256_s=values, median_s=statistics.median(xs),
        q1_s=q1, q3_s=q3, mean_with_stalls_s=statistics.fmean(xs),
        mean_without_stalls_s=statistics.fmean(calm) if calm else None,
        stalled=tuple(s.iteration for s in reported if s.stalled),
        scored=sum(s.scored for s in reported), samples=sum(s.samples for s in reported))


def resources(summary: Per256Summary, *, gpus_held: int, cpu_nodes_held: int) -> dict[str, float | None]:
    """GPU-hours and CPU-node hours per 256 scored samples, reported separately (Section 6.9). Every
    GPU held counts, including partner ASan GPUs when they are dedicated to the run."""
    for key, v in (("gpus_held", gpus_held), ("cpu_nodes_held", cpu_nodes_held)):
        if type(v) is not int or v < 0:
            raise Per256Error(f"{key} must be an int >= 0, got {v!r}")
    out: dict[str, float | None] = {}
    for unit, held in (("gpu_hours", gpus_held), ("cpu_node_hours", cpu_nodes_held)):
        for stat in ("median", "mean_without_stalls"):
            v = summary.statistic(stat)
            out[f"{unit}_per256_{stat}"] = None if v is None else held * v / 3600.0
    return out


# --- from an AORLTA run ----------------------------------------------------------------------------


def walls_from_spans(spans: Sequence[Span], mode: str = "v1",
                     verdicts: Mapping[int, Sequence[str]] | None = None) -> dict[int, float]:
    """wall(n) for every training iteration, from the `select` and `landing` markers (v1) or
    consecutive publications (v2). The span envelope is refused: it starts after prompt selection,
    so it would understate AORLTA's time. `verdicts` (from the iteration summaries) stand in for a
    landing span that carries no verdict of its own."""
    if mode not in ("v1", "v2"):
        raise Per256Error(f"mode must be v1 or v2 for the per-256 metric, got {mode!r}")
    return {n: b.wall_s for n, b in run_bounds(spans, mode, verdicts).items()}


@dataclass(frozen=True)
class EpisodeCounts:
    iteration: int
    samples: int
    scored: int
    unscored_by_reason: dict[str, int] = field(default_factory=dict)
    truncated: int = 0  # final answers that ended on the length cap
    reward_sum: float = 0.0

    @property
    def mean_train_reward(self) -> float | None:
        """Mean reward over scored samples; unscored ones are excluded from every mean (F4)."""
        return self.reward_sum / self.scored if self.scored else None


def _final_finish(rec: Mapping[str, Any], where: str) -> str:
    turns, final = rec.get("turns"), rec.get("final_turn")
    if not isinstance(turns, list) or type(final) is not int or not 0 <= final < len(turns):
        raise Per256Error(f"{where}: turns/final_turn do not name a final turn")
    finish = turns[final].get("finish_reason") if isinstance(turns[final], Mapping) else None
    if not isinstance(finish, Mapping) or not isinstance(finish.get("type"), str):
        raise Per256Error(f"{where}: the final turn has no finish_reason type")
    return finish["type"]


def counts_from_episodes(records: Iterable[Mapping[str, Any]]) -> dict[int, EpisodeCounts]:
    """Per-iteration counts from `aorlta.episode/1` chain records (Section 6.8). Fails closed on a
    record that does not say plainly whether it was scored."""
    acc: dict[int, dict[str, Any]] = defaultdict(lambda: {"samples": 0, "scored": 0, "truncated": 0,
                                                          "reward_sum": 0.0, "unscored": defaultdict(int)})
    seen: set[str] = set()
    for i, rec in enumerate(records):
        where = f"episode record {i}"
        if not isinstance(rec, Mapping) or rec.get("schema") != EPISODE_SCHEMA:
            raise Per256Error(f"{where}: schema must be {EPISODE_SCHEMA!r}")
        cid, n = rec.get("chain_id"), rec.get("iteration")
        if not isinstance(cid, str) or type(n) is not int or n < 1:
            raise Per256Error(f"{where}: needs a chain_id and an iteration >= 1")
        if cid in seen:
            raise Per256Error(f"{where}: chain {cid} appears twice")
        seen.add(cid)
        a = acc[n]
        a["samples"] += 1
        status, reward = rec.get("status"), rec.get("reward")
        if status == "scored":
            if type(reward) not in (int, float) or not math.isfinite(reward):
                raise Per256Error(f"{where}: scored, but its reward is {reward!r}")
            a["scored"] += 1
            a["reward_sum"] += float(reward)
        elif status == "unscored":
            reason = rec.get("unscored_reason")
            if reward is not None or not (isinstance(reason, str) and reason):
                raise Per256Error(f"{where}: unscored needs reward null and an unscored_reason")
            a["unscored"][reason] += 1
        else:
            raise Per256Error(f"{where}: status must be 'scored' or 'unscored', got {status!r}")
        if _final_finish(rec, where) == "length":
            a["truncated"] += 1
    return {n: EpisodeCounts(n, a["samples"], a["scored"], dict(a["unscored"]), a["truncated"], a["reward_sum"])
            for n, a in sorted(acc.items())}


def read_episodes(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        for k, line in enumerate(read_text(path).splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise Per256Error(f"{path}:{k}: {exc}") from None
    return out


def iteration_stats(walls: Mapping[int, float], counts: Mapping[int, EpisodeCounts],
                    stalls: Mapping[int, str]) -> list[IterationStats]:
    """Join walls (from spans), counts (from episodes) and the stalls the orchestrator recorded
    (iteration -> reason; an iteration not listed did not stall). The iteration sets must agree."""
    if set(walls) != set(counts):
        raise Per256Error(f"iterations with spans {sorted(walls)} and with episodes {sorted(counts)} differ")
    unknown = sorted(set(stalls) - set(walls))
    if unknown:
        raise Per256Error(f"stalls listed for iterations with no data: {unknown}")
    return [IterationStats(n, walls[n], counts[n].samples, counts[n].scored, n in stalls, stalls.get(n))
            for n in sorted(walls)]
