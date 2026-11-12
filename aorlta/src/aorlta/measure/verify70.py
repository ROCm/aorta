"""verify70: is an iteration waiting on checks for at least 70% of its wall time?
(design Section 6.9, "Verifying the 70% from a baseline's iteration summaries"; fact F6.)

With phase timestamps, the check share is

    |union of the check-phase intervals, minus every generation or training interval| / wall

so overlapping phases count once, as wall-clock coverage, never as a sum. The check phases are gate,
waiting for scoring, ASan settle, and the compile checks inside compile-repair. A phase that holds
both check and generation time, such as a compile-repair envelope without its compile-check
sub-intervals, is "mixed": the share is then bracketed by counting it as generation (lower bound)
and as check (upper bound).

With durations only, the bounds are those of the doc:
    lower = the longest check phase / wall
    upper = min(1, (the check phases + the mixed phases, summed) / wall)
The lower bound holds only if those phases did not overlap generation, which durations cannot show.

An iteration is confirmed if its lower bound reaches the threshold (0.7) and, with durations only,
the baseline's owner has also confirmed that its phases run sequentially. It is refuted if its upper bound is below
the threshold, and undetermined otherwise (timestamps would settle it). Every phase name must be mapped
to a category; an unmapped one stops the run rather than being guessed.

The script runs on the baseline's files where they are. Its output holds numbers and fixed status codes only:
no phase names, and nothing else from those records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..atomic import read_bytes, read_text
from ..config import ConfigError, parse_yaml, strict_decode
from ..ids import sha256_hex
from ..spans import Span
from . import MeasureError
from .timeline import DEFAULT_CATEGORIES, UNCLASSIFIED, category_of, clip, measure, run_bounds, subtract, union

THRESHOLD = 0.7
CATEGORIES = ("check", "mixed", "generation", "training", "other")
MAPPING_SCHEMA = "aorlta.verify70.mapping/1"
WALL_TOLERANCE_S, WALL_TOLERANCE_REL = 1.0, 1e-3  # a stated wall may differ from t_end - t_start by rounding

# Fixed reason codes: the output never carries text from the input.
EXACT = "exact"  # timestamps, no mixed phase: lower == upper
MIXED = "mixed_phase_bounds"  # timestamps with a mixed phase
SEQUENTIAL = "durations_sequential_confirmed"  # the baseline's owner confirmed its phases are sequential
UPPER_BELOW = "durations_upper_bound_below_threshold"
UNSEQUENCED = "durations_need_sequential_confirmation"  # lower bound passes, but overlap is unknown
STRADDLE = "durations_bounds_straddle_threshold"  # needs timestamps


class Verify70Error(MeasureError):
    pass


@dataclass(frozen=True)
class IterationPhases:
    """One iteration of a summary, in neutral form. With `phases` (timestamps) it needs t_begin and
    t_end; with `durations` only, a wall (wall_s, or t_begin and t_end)."""

    iteration: int
    t_begin: float | None = None
    t_end: float | None = None
    wall_s: float | None = None
    phases: tuple[tuple[str, float, float], ...] | None = None
    durations: dict[str, float] | None = None

    @property
    def mode(self) -> str:
        return "timestamps" if self.phases is not None else "durations"


@dataclass(frozen=True)
class Row:
    iteration: int
    mode: str  # "timestamps" | "durations"
    wall_s: float
    lower: float  # check share, lower bound
    upper: float
    status: str  # "confirmed" | "refuted" | "undetermined"
    reason: str  # a fixed code
    seconds: dict[str, float]  # check_only_low, check_only_high, and per category (wall-clock coverage)

    def to_json(self) -> dict[str, Any]:
        return {"iteration": self.iteration, "mode": self.mode, "wall_s": self.wall_s, "lower": self.lower,
                "upper": self.upper, "status": self.status, "reason": self.reason, "seconds": self.seconds}


def _wall(it: IterationPhases) -> float:
    span = None
    if it.t_begin is not None or it.t_end is not None:
        if it.t_begin is None or it.t_end is None:
            raise Verify70Error(f"iteration {it.iteration}: has only one of t_start and t_end")
        span = it.t_end - it.t_begin
    if span is not None and it.wall_s is not None:
        if abs(span - it.wall_s) > max(WALL_TOLERANCE_S, WALL_TOLERANCE_REL * span):
            raise Verify70Error(f"iteration {it.iteration}: wall_s {it.wall_s} contradicts t_end - t_start = {span}")
    wall = span if span is not None else it.wall_s
    if wall is None:
        raise Verify70Error(f"iteration {it.iteration}: no wall time (give wall_s, or t_start and t_end)")
    if not (math.isfinite(wall) and wall > 0):
        raise Verify70Error(f"iteration {it.iteration}: wall time {wall} is not positive")
    return wall


def _status(lower: float, upper: float, threshold: float, *, confirmable: bool) -> str:
    if confirmable and lower >= threshold:
        return "confirmed"
    if upper < threshold:
        return "refuted"
    return "undetermined"


def _category(classes: Mapping[str, str], name: str, iteration: int) -> str:
    cat = classes.get(name)
    if cat is None:
        raise Verify70Error(f"iteration {iteration}: phase {name!r} has no category in the mapping")
    if cat not in CATEGORIES:
        raise Verify70Error(f"phase {name!r}: unknown category {cat!r}")
    return cat


def timestamps_row(it: IterationPhases, classes: Mapping[str, str], threshold: float = THRESHOLD) -> Row:
    if it.phases is None or it.t_begin is None or it.t_end is None:
        raise Verify70Error(f"iteration {it.iteration}: timestamps need phases plus t_start and t_end")
    wall = _wall(it)
    by_cat: dict[str, list[tuple[float, float]]] = {c: [] for c in CATEGORIES}
    unknown = sorted({name for name, _, _ in it.phases if name not in classes})
    if unknown:
        raise Verify70Error(f"iteration {it.iteration}: {len(unknown)} phase name(s) have no category in the "
                            f"mapping: {unknown}")
    outside = 0.0
    for name, start, end in it.phases:
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise Verify70Error(f"iteration {it.iteration}: phase {name!r} ends before it starts")
        inside = clip([(start, end)], it.t_begin, it.t_end)
        outside += (end - start) - sum(b - a for a, b in inside)
        by_cat[_category(classes, name, it.iteration)].extend(inside)
    check, mixed = union(by_cat["check"]), union(by_cat["mixed"])
    busy = union(by_cat["generation"] + by_cat["training"])
    low = measure(subtract(check, busy + mixed))
    high = measure(subtract(check + mixed, busy))
    lower, upper = low / wall, high / wall
    seconds = {"check_only_low": low, "check_only_high": high, "outside_iteration": outside}
    seconds.update({c: measure(v) for c, v in by_cat.items()})
    return Row(it.iteration, "timestamps", wall, lower, upper, _status(lower, upper, threshold, confirmable=True),
               EXACT if not mixed else MIXED, seconds)


def durations_row(it: IterationPhases, classes: Mapping[str, str], *, sequential_confirmed: bool,
                  threshold: float = THRESHOLD, absent_means_zero: Iterable[str] = ()) -> Row:
    if it.durations is None:
        raise Verify70Error(f"iteration {it.iteration}: no durations")
    wall = _wall(it)
    cats: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for name, d in it.durations.items():
        cat = _category(classes, name, it.iteration)
        if not (math.isfinite(d) and d >= 0):
            raise Verify70Error(f"iteration {it.iteration}: phase {name!r} has duration {d}")
        if d > wall + max(WALL_TOLERANCE_S, WALL_TOLERANCE_REL * wall):
            raise Verify70Error(f"iteration {it.iteration}: phase {name!r} ({d} s) is longer than the wall ({wall} s)")
        cats[cat].append(d)
    zero_ok = set(absent_means_zero)
    missing = sorted(n for n, c in classes.items() if c in ("check", "mixed") and n not in it.durations
                     and n not in zero_ok)
    if missing:
        raise Verify70Error(f"iteration {it.iteration}: no duration for check phase(s) {missing}; a missing phase "
                            "is unknown, not zero (list it under absent_means_zero if it is)")
    lower = max(cats["check"], default=0.0) / wall
    upper = min(1.0, (sum(cats["check"]) + sum(cats["mixed"])) / wall)
    status = _status(lower, upper, threshold, confirmable=sequential_confirmed)
    if status == "confirmed":
        reason = SEQUENTIAL
    elif status == "refuted":
        reason = UPPER_BELOW
    else:
        reason = UNSEQUENCED if lower >= threshold else STRADDLE
    seconds = {"check_longest": max(cats["check"], default=0.0), "check_sum": sum(cats["check"]),
               "mixed_sum": sum(cats["mixed"])}
    return Row(it.iteration, "durations", wall, lower, upper, status, reason, seconds)


@dataclass(frozen=True)
class Verify70Report:
    rows: tuple[Row, ...]
    threshold: float

    @property
    def counts(self) -> dict[str, int]:
        out = {"confirmed": 0, "refuted": 0, "undetermined": 0}
        for r in self.rows:
            out[r.status] += 1
        return out

    @property
    def fraction_confirmed(self) -> float:
        return self.counts["confirmed"] / len(self.rows)

    def to_json(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "iterations": len(self.rows), "counts": self.counts,
                "fraction_confirmed": self.fraction_confirmed, "rows": [r.to_json() for r in self.rows]}

    def markdown(self) -> str:
        lines = ["| iteration | data | wall s | check share | status |", "|---|---|---|---|---|"]
        for r in self.rows:
            share = f"{r.lower:.3f}" if r.lower == r.upper else f"{r.lower:.3f} to {r.upper:.3f}"
            lines.append(f"| {r.iteration} | {r.mode} | {r.wall_s:.1f} | {share} | {r.status} |")
        c = self.counts
        lines.append("")
        lines.append(f"Confirmed (>= {self.threshold:g}) in {c['confirmed']} of {len(self.rows)} iterations "
                     f"({self.fraction_confirmed:.0%}); refuted in {c['refuted']}; undetermined in {c['undetermined']}.")
        return "\n".join(lines) + "\n"


def verify(iterations: Sequence[IterationPhases], classes: Mapping[str, str], *, sequential_confirmed: bool = False,
           threshold: float = THRESHOLD, absent_means_zero: Iterable[str] = ()) -> Verify70Report:
    if not iterations:
        raise Verify70Error("no iterations")
    if not 0 < threshold <= 1:
        raise Verify70Error(f"threshold must be in (0, 1], got {threshold}")
    seen = set()
    rows = []
    for it in sorted(iterations, key=lambda x: x.iteration):
        if it.iteration in seen:
            raise Verify70Error(f"iteration {it.iteration} appears twice")
        seen.add(it.iteration)
        if it.mode == "timestamps":
            rows.append(timestamps_row(it, classes, threshold))
        else:
            rows.append(durations_row(it, classes, sequential_confirmed=sequential_confirmed, threshold=threshold,
                                      absent_means_zero=absent_means_zero))
    return Verify70Report(tuple(rows), threshold)


# --- AORLTA's own spans ----------------------------------------------------------------------------

SPAN_CLASS = {"generation": "generation", "training": "training", "check": "check", "sync": "other", "other": "other"}


def phases_from_spans(spans: Sequence[Span], mode: str = "v1", *,
                      verdicts: Mapping[int, Sequence[str]] | None = None,
                      categories: Sequence[tuple[str, str]] = DEFAULT_CATEGORIES
                      ) -> tuple[list[IterationPhases], dict[str, str]]:
    """AORLTA's spans as verify70 input: each span of a training iteration is a phase, classed through
    the timeline's categories. A span name with no category stops it (it could be generation)."""
    classes: dict[str, str] = {}
    for s in spans:
        if s.iteration < 1:
            continue  # run set-up belongs to no iteration's wall
        cat = category_of(s.name, categories) or UNCLASSIFIED
        if cat == UNCLASSIFIED:
            raise Verify70Error(f"span {s.name!r} has no category")
        classes[s.name] = SPAN_CLASS[cat]
    out = []
    for n, b in run_bounds(spans, mode, verdicts).items():
        phases = tuple((s.name, s.t_start, s.t_end) for s in spans if s.iteration == n)
        out.append(IterationPhases(n, t_begin=b.t_begin, t_end=b.t_end, phases=phases))
    return out, classes


# --- the baseline's files ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldMap:
    """Dotted paths to each field inside one iteration record; null for a field the file lacks."""

    iteration: str = "iteration"
    t_start: str | None = "t_start"
    t_end: str | None = "t_end"
    wall_s: str | None = "wall_s"
    phases: str | None = "phases"
    durations: str | None = "durations"


@dataclass(frozen=True)
class PhaseEntryMap:
    """Keys inside one phase entry, when phases are a list of objects."""

    name: str = "name"
    t_start: str = "t_start"
    t_end: str = "t_end"


@dataclass(frozen=True)
class Classes:
    check: tuple[str, ...] = ()
    mixed: tuple[str, ...] = ()
    generation: tuple[str, ...] = ()
    training: tuple[str, ...] = ()
    other: tuple[str, ...] = ()

    def by_name(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for cat in CATEGORIES:
            for name in getattr(self, cat):
                if name in out:
                    raise Verify70Error(f"phase {name!r} is listed under both {out[name]} and {cat}")
                out[name] = cat
        return out


@dataclass(frozen=True)
class SummaryMapping:
    schema: Literal["aorlta.verify70.mapping/1"]
    classes: Classes
    records: str | None = None  # key of the iteration list when the file is a JSON object
    fields: FieldMap = field(default_factory=FieldMap)
    phase_entry: PhaseEntryMap = field(default_factory=PhaseEntryMap)
    time_unit: Literal["s", "ms", "us"] = "s"  # numeric timestamps
    duration_unit: Literal["s", "ms"] = "s"
    absent_means_zero: tuple[str, ...] = ()
    sequential_confirmed: bool = False  # the baseline's owner confirmed its phases never overlap (durations only)
    threshold: float = THRESHOLD

    def __post_init__(self) -> None:
        names = self.classes.by_name()
        if not names:
            raise Verify70Error("the mapping classes no phase")
        stray = sorted(n for n in self.absent_means_zero if names.get(n) not in ("check", "mixed"))
        if stray:
            raise Verify70Error(f"absent_means_zero lists phases that are not check or mixed: {stray}")
        if not 0 < self.threshold <= 1:
            raise Verify70Error(f"threshold must be in (0, 1], got {self.threshold}")


def load_mapping(path: str | Path) -> SummaryMapping:
    try:
        return strict_decode(SummaryMapping, parse_yaml(read_text(path), str(path)), "mapping")
    except ConfigError as exc:
        raise Verify70Error(str(exc)) from None


_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


def _get(record: Mapping[str, Any], path: str | None) -> Any:
    if path is None:
        return None
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


class _Clock:
    """Turns one record's timestamps into seconds; refuses to mix zone-aware and naive times."""

    def __init__(self, unit: str, where: str):
        self.scale, self.where, self.kinds = _UNITS[unit], where, set()

    def __call__(self, value: Any, what: str) -> float:
        if type(value) in (int, float) and math.isfinite(value):
            self.kinds.add("number")
            out = float(value) * self.scale
        elif isinstance(value, str):
            try:
                t = dt.datetime.fromisoformat(value)
            except ValueError:
                raise Verify70Error(f"{self.where}: {what} is not a number or an ISO-8601 time") from None
            self.kinds.add("naive" if t.tzinfo is None else "aware")
            out = (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).timestamp()
        else:
            raise Verify70Error(f"{self.where}: {what} is not a number or an ISO-8601 time")
        if len(self.kinds) > 1:
            raise Verify70Error(f"{self.where}: mixes {' and '.join(sorted(self.kinds))} timestamps")
        return out


def _phases(raw: Any, m: SummaryMapping, clock: _Clock, where: str) -> tuple[tuple[str, float, float], ...]:
    out = []
    if isinstance(raw, list):
        for i, e in enumerate(raw):
            if not isinstance(e, Mapping) or not isinstance(e.get(m.phase_entry.name), str):
                raise Verify70Error(f"{where}: phase entry {i} has no name")
            name = e[m.phase_entry.name]
            out.append((name, clock(e.get(m.phase_entry.t_start), f"phase {i} start"),
                        clock(e.get(m.phase_entry.t_end), f"phase {i} end")))
    elif isinstance(raw, Mapping):
        for name, v in raw.items():
            pairs = v if isinstance(v, list) and v and all(isinstance(x, list) for x in v) else [v]
            for pair in pairs:
                if not (isinstance(pair, list) and len(pair) == 2):
                    raise Verify70Error(f"{where}: a phase must be [start, end] or a list of them")
                out.append((name, clock(pair[0], "phase start"), clock(pair[1], "phase end")))
    else:
        raise Verify70Error(f"{where}: phases must be a list of entries or an object of [start, end]")
    return tuple(out)


def iteration_from_record(record: Mapping[str, Any], m: SummaryMapping, index: int) -> IterationPhases:
    where = f"record {index}"
    if not isinstance(record, Mapping):
        raise Verify70Error(f"{where}: not an object")
    n = _get(record, m.fields.iteration)
    if type(n) is not int:
        raise Verify70Error(f"{where}: no integer iteration at {m.fields.iteration!r}")
    where = f"iteration {n}"
    clock = _Clock(m.time_unit, where)
    t0, t1 = _get(record, m.fields.t_start), _get(record, m.fields.t_end)
    t_begin = None if t0 is None else clock(t0, "t_start")
    t_end = None if t1 is None else clock(t1, "t_end")
    wall = _get(record, m.fields.wall_s)
    if wall is not None and not (type(wall) in (int, float) and math.isfinite(wall)):
        raise Verify70Error(f"{where}: wall_s is not a number")
    raw_phases, raw_durations = _get(record, m.fields.phases), _get(record, m.fields.durations)
    phases = durations = None
    if raw_phases is not None:
        phases = _phases(raw_phases, m, clock, where)
    elif raw_durations is not None:
        if not isinstance(raw_durations, Mapping):
            raise Verify70Error(f"{where}: durations must be an object of seconds")
        scale = _UNITS[m.duration_unit]
        durations = {}
        for k, v in raw_durations.items():
            if type(v) not in (int, float) or not math.isfinite(v):
                raise Verify70Error(f"{where}: duration of {k!r} is not a number")
            durations[k] = float(v) * scale
    else:
        raise Verify70Error(f"{where}: neither phase timestamps nor durations")
    return IterationPhases(n, t_begin, t_end, None if wall is None else float(wall), phases, durations)


def read_summaries(path: str | Path, m: SummaryMapping) -> list[IterationPhases]:
    path = Path(path)
    if path.suffix == ".jsonl":
        records = []
        for k, line in enumerate(read_text(path).splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    raise Verify70Error(f"{path.name}:{k}: not JSON") from None
    else:
        try:
            data = json.loads(read_text(path))
        except json.JSONDecodeError:
            raise Verify70Error(f"{path.name}: not JSON") from None
        if isinstance(data, Mapping):
            if m.records is None or not isinstance(data.get(m.records), list):
                raise Verify70Error(f"{path.name}: a JSON object needs `records` naming its list of iterations")
            data = data[m.records]
        if not isinstance(data, list):
            raise Verify70Error(f"{path.name}: expected a list of iteration records")
        records = data
    return [iteration_from_record(r, m, i) for i, r in enumerate(records)]


def run(summaries: str | Path, mapping: str | Path) -> dict[str, Any]:
    m = load_mapping(mapping)
    report = verify(read_summaries(summaries, m), m.classes.by_name(), sequential_confirmed=m.sequential_confirmed,
                    threshold=m.threshold, absent_means_zero=m.absent_means_zero)
    out = report.to_json()
    out["mapping_sha256"] = sha256_hex(read_bytes(mapping))
    out["summaries_sha256"] = sha256_hex(read_bytes(summaries))
    out["sequential_confirmed"] = m.sequential_confirmed
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aorlta.measure.verify70", description=__doc__.split("\n\n")[0])
    ap.add_argument("summaries", type=Path, help="per-iteration summaries (.json or .jsonl)")
    ap.add_argument("--mapping", type=Path, required=True, help="verify70 mapping YAML (field paths and phase classes)")
    ap.add_argument("--json", type=Path, help="write the numbers here")
    args = ap.parse_args(argv)
    try:
        out = run(args.summaries, args.mapping)
    except MeasureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rows = tuple(Row(**{**r}) for r in out["rows"])
    sys.stdout.write(Verify70Report(rows, out["threshold"]).markdown())
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
