"""C9 timeline: an iteration's wall-clock bounds, its critical path, and phase coverage
(design Section 6.9, "Timeline"; spans as in Section 5 item 5).

Phases overlap: checks stream while chains generate, and gradients accumulate while checks run. So
nothing here sums span durations to explain wall time. Each phase gets three numbers, and only the
first adds up to the wall:

- critical seconds. The critical path is found by walking back from the iteration's end. At each
  point, the span that finished last at or before that point is the one the rest of the iteration
  waited for; the walk moves to that span's start and repeats. The segments tile the wall exactly.
  Time where no span ends at the current point is an explicit gap segment.
- coverage seconds: the wall-clock length of the union of the phase's spans.
- busy seconds: the sum of the phase's span durations. This double-counts overlap and is shown only
  for contrast.

Wall time, v1: from prompt selection (the start of the iteration's `select` span) to the end of its
last `landing` span, whose `verdict` must pass (LANDED or NOOP, Section 6.5). v2: from the previous
iteration's passed landing to this iteration's (consecutive weight publications). An iteration that
took no optimizer step publishes nothing: it ends at its zero-length `no_step` marker instead, in
both modes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from ..ids import canonical_json
from ..spans import Span, read_jsonl
from . import MeasureError

Interval = tuple[float, float]

GAP = "(gap)"
IDLE = "idle"
UNCLASSIFIED = "unclassified"
GAP_LIMIT = 0.05  # Section 6.6: a span gap of more than 5% of wall is flagged
BEGIN_SPAN = "select"
END_SPAN = "landing"
NO_STEP_SPAN = "no_step"
PASSING_VERDICTS = frozenset({"LANDED", "NOOP"})
CAUSAL_KEYS = ("chain_id", "candidate_id")  # candidate_id == chain_id (Section 5 item 1)

# Span name pattern -> category, first match wins. The names are Section 6.9's list, plus the ones
# WP2, WP5 and WP7 already emit (`engine.start`, `control`, `score.retry`, `setup.*`) and the `select`
# and `no_step` markers this module reads.
DEFAULT_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("gen.turn0", "generation"),
    ("gen.repair", "generation"),
    ("compile", "check"),
    ("gate.*", "check"),
    ("ref.*", "check"),
    ("san.*", "check"),
    ("asan.queue", "check"),
    ("asan.run", "check"),
    ("score.wait", "check"),
    ("score.retry", "check"),  # D7: the unsettled legs re-submitted once
    ("control", "check"),
    ("train.stream", "training"),
    ("train.final", "training"),
    ("merge", "sync"),
    ("write", "sync"),
    ("reload", "sync"),
    ("landing", "sync"),
    ("no_step", "other"),
    ("select", "other"),
    ("record", "other"),
    ("ckpt", "other"),
    ("eval.request", "other"),
    ("engine.start", "other"),
    ("setup.*", "other"),
)
# For exclusive attribution: an instant belongs to the first active category in this order. With
# generation and training ahead of check, "check" is time when only checks were running, which is
# the 70% reading of Section 6.9.
PRIORITY = ("generation", "training", "check", "sync", "other", UNCLASSIFIED)


class TimelineError(MeasureError):
    pass


# --- intervals -------------------------------------------------------------------------------------


def _checked(intervals: Iterable[Interval]) -> list[Interval]:
    out = []
    for a, b in intervals:
        a, b = float(a), float(b)
        if not (math.isfinite(a) and math.isfinite(b)):
            raise TimelineError(f"interval with a non-finite end: ({a}, {b})")
        if b < a:
            raise TimelineError(f"interval ends before it starts: ({a}, {b})")
        if b > a:
            out.append((a, b))
    return out


def union(intervals: Iterable[Interval]) -> list[Interval]:
    """Sorted, disjoint intervals covering exactly the points of the inputs (touching ones merge)."""
    merged: list[list[float]] = []
    for a, b in sorted(_checked(intervals)):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def measure(intervals: Iterable[Interval]) -> float:
    """Wall-clock length of the union: overlapping time counts once."""
    return sum(b - a for a, b in union(intervals))


def clip(intervals: Iterable[Interval], lo: float, hi: float) -> list[Interval]:
    return [(max(a, lo), min(b, hi)) for a, b in _checked(intervals) if min(b, hi) > max(a, lo)]


def subtract(minuend: Iterable[Interval], subtrahend: Iterable[Interval]) -> list[Interval]:
    """The points of `minuend` that are not in `subtrahend`, as a union."""
    keep, cut = union(minuend), union(subtrahend)
    out: list[Interval] = []
    j = 0
    for a, b in keep:
        lo = a
        while j < len(cut) and cut[j][1] <= lo:
            j += 1
        k = j
        while k < len(cut) and cut[k][0] < b:
            c, d = cut[k]
            if c > lo:
                out.append((lo, c))
            lo = max(lo, d)
            if lo >= b:
                break
            k += 1
        if lo < b:
            out.append((lo, b))
    return out


# --- categories ------------------------------------------------------------------------------------


def category_of(name: str, table: Sequence[tuple[str, str]] = DEFAULT_CATEGORIES) -> str | None:
    for pattern, category in table:
        if fnmatchcase(name, pattern):
            return category
    return None


# --- iteration bounds ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bounds:
    iteration: int
    t_begin: float
    t_end: float
    begin_from: str  # "select", "landing(n-1)", "no_step(n-1)" or "envelope"
    end_from: str  # "landing:<verdicts>", "no_step" or "envelope"

    @property
    def wall_s(self) -> float:
        return self.t_end - self.t_begin

    @property
    def from_markers(self) -> bool:
        return self.begin_from != "envelope" and self.end_from != "envelope"


def landing_verdicts(span: Span) -> tuple[str, ...] | None:
    """A landing span's verdict: one string, or one per engine. None if it carries none."""
    v = span.attrs.get("verdict")
    if isinstance(v, str):
        return (v,)
    if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        return tuple(v)
    return None


Verdicts = Mapping[int, Sequence[str]]  # iteration -> the landing verdict per engine, from its summary


def _passed_landing_end(spans: Sequence[Span], iteration: int, given: Verdicts | None = None) -> tuple[float, str]:
    """End of the iteration's last landing span, which must have passed. The verdict is the span's own
    `verdict` attr or, when the span has none, the one in `given` (the iteration summary's); if both
    exist they must agree. An iteration with no landing span ends at its one `no_step` marker."""
    landings = [s for s in spans if s.iteration == iteration and s.name == END_SPAN]
    no_step = [s for s in spans if s.iteration == iteration and s.name == NO_STEP_SPAN]
    if no_step:
        if landings or len(no_step) != 1:
            raise TimelineError(f"iteration {iteration}: {len(no_step)} '{NO_STEP_SPAN}' markers and {len(landings)} "
                                f"'{END_SPAN}' spans; an iteration that took no step has one marker and no landing")
        if given is not None and given.get(iteration):
            raise TimelineError(f"iteration {iteration}: it took no step, but its summary gives landing verdicts")
        return no_step[0].t_end, NO_STEP_SPAN
    if not landings:
        raise TimelineError(f"iteration {iteration}: no '{END_SPAN}' span and no '{NO_STEP_SPAN}' marker, so nothing "
                            "says where it ended")
    last = max(landings, key=lambda s: (s.t_end, s.t_start))
    own = landing_verdicts(last)
    other = None if given is None or iteration not in given else tuple(given[iteration])
    if own is not None and other is not None and set(own) != set(other):
        raise TimelineError(f"iteration {iteration}: the landing span says {sorted(set(own))}, its summary says "
                            f"{sorted(set(other))}")
    verdicts = own if own is not None else other
    if not verdicts or not all(isinstance(v, str) for v in verdicts):
        raise TimelineError(f"iteration {iteration}: its last '{END_SPAN}' span has no 'verdict' attr and no "
                            "verdict was given; a landing counts as passed only when something says so")
    failed = sorted(set(v for v in verdicts if v not in PASSING_VERDICTS))
    if failed:
        raise TimelineError(f"iteration {iteration}: its last landing did not pass ({', '.join(failed)})")
    return last.t_end, "landing:" + "/".join(sorted(set(verdicts)))


def bounds_v1(spans: Sequence[Span], iteration: int, verdicts: Verdicts | None = None) -> Bounds:
    """Prompt selection to the passed landing check (Section 6.9, v1)."""
    selects = [s for s in spans if s.iteration == iteration and s.name == BEGIN_SPAN]
    if len(selects) != 1:
        raise TimelineError(f"iteration {iteration}: {len(selects)} '{BEGIN_SPAN}' spans; wall(n) starts at "
                            "prompt selection, so exactly one is needed")
    t_end, end_from = _passed_landing_end(spans, iteration, verdicts)
    t_begin = selects[0].t_start
    if not t_end > t_begin:
        raise TimelineError(f"iteration {iteration}: the landing ends ({t_end}) before selection starts ({t_begin})")
    return Bounds(iteration, t_begin, t_end, BEGIN_SPAN, end_from)


def bounds_v2(spans: Sequence[Span], iteration: int, verdicts: Verdicts | None = None) -> Bounds:
    """Between consecutive weight publications (Section 6.9, v2). Iteration 1 needs a passed landing
    in iteration 0 (the start-of-run publication)."""
    t_begin, begin_from = _passed_landing_end(spans, iteration - 1, verdicts)
    t_end, end_from = _passed_landing_end(spans, iteration, verdicts)
    if not t_end > t_begin:
        raise TimelineError(f"iteration {iteration}: its publication ({t_end}) is not after the previous one ({t_begin})")
    previous = NO_STEP_SPAN if begin_from == NO_STEP_SPAN else END_SPAN
    return Bounds(iteration, t_begin, t_end, f"{previous}({iteration - 1})", end_from)


def bounds_envelope(spans: Sequence[Span], iteration: int, verdicts: Verdicts | None = None) -> Bounds:
    """First span start to last span end. For looking at runs without markers only: it starts after
    prompt selection, so it understates wall time and is never used for the per-256 metric."""
    mine = [s for s in spans if s.iteration == iteration]
    if not mine:
        raise TimelineError(f"iteration {iteration}: no spans")
    t_begin, t_end = min(s.t_start for s in mine), max(s.t_end for s in mine)
    if not t_end > t_begin:
        raise TimelineError(f"iteration {iteration}: its spans have no extent")
    return Bounds(iteration, t_begin, t_end, "envelope", "envelope")


BOUNDS = {"v1": bounds_v1, "v2": bounds_v2, "envelope": bounds_envelope}


def run_bounds(spans: Sequence[Span], mode: str = "v1", verdicts: Verdicts | None = None) -> dict[int, Bounds]:
    """Bounds of every training iteration (1 and up) that has spans."""
    if mode not in BOUNDS:
        raise TimelineError(f"unknown mode {mode!r}; one of {', '.join(BOUNDS)}")
    return {n: BOUNDS[mode](spans, n, verdicts) for n in sorted({s.iteration for s in spans if s.iteration >= 1})}


# --- critical path ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    t_start: float
    t_end: float
    name: str  # the span's name, or GAP
    span: Span | None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"t_start": self.t_start, "t_end": self.t_end, "name": self.name}
        if self.span is not None:
            key = causal_key(self.span)
            if key is not None:
                out["id"] = key
        return out


def causal_key(span: Span) -> str | None:
    for key in CAUSAL_KEYS:
        value = span.attrs.get(key)
        if isinstance(value, str):
            return value
    return None


def critical_path(spans: Sequence[Span], t_begin: float, t_end: float) -> list[Segment]:
    """Segments tiling [t_begin, t_end], found by walking back from t_end.

    At each point the chosen span is the one that ended last at or before it. Ties go to the span
    with the same chain or candidate id as the segment after it, then to the later start, then by
    name and attrs, so the result does not depend on the order of `spans`. Zero-length spans explain
    no time and are skipped; spans ending after t_end (for example `record` and `ckpt` after the
    landing) or at or before t_begin are not candidates.
    """
    if not (math.isfinite(t_begin) and math.isfinite(t_end) and t_end > t_begin):
        raise TimelineError(f"empty or invalid window [{t_begin}, {t_end}]")
    pool = sorted((s for s in spans if s.t_end > s.t_start and t_begin < s.t_end <= t_end), key=lambda s: s.t_end)
    ends = [s.t_end for s in pool]
    segments: list[Segment] = []
    cursor, after = t_end, None
    while cursor > t_begin:
        i = bisect.bisect_right(ends, cursor)
        if i == 0:
            segments.append(Segment(t_begin, cursor, GAP, None))
            break
        latest = ends[i - 1]
        tied = pool[bisect.bisect_left(ends, latest, 0, i):i]
        best = min(tied, key=lambda s: (after is None or causal_key(s) != after, -s.t_start, s.name,
                                        canonical_json(s.to_json())))
        if latest < cursor:
            segments.append(Segment(latest, cursor, GAP, None))
        start = max(best.t_start, t_begin)
        segments.append(Segment(start, latest, best.name, best))
        cursor, after = start, causal_key(best)
    segments.reverse()
    return segments


# --- per-iteration summary -------------------------------------------------------------------------


@dataclass(frozen=True)
class IterationTimeline:
    bounds: Bounds
    path: tuple[Segment, ...]
    critical_s: dict[str, float]  # by span name, and GAP
    coverage_s: dict[str, float]  # by span name: union of the name's spans, clipped to the wall
    busy_s: dict[str, float]  # by span name: sum of clipped durations
    count: dict[str, int]  # spans that overlap the wall, by name
    category_critical_s: dict[str, float]
    category_coverage_s: dict[str, float]
    category_exclusive_s: dict[str, float]  # priority attribution, plus IDLE; sums to the wall
    uncovered_s: float  # wall time inside no span
    outside: dict[str, int]  # the iteration's spans entirely outside the wall, by name
    unclassified: tuple[str, ...]

    @property
    def iteration(self) -> int:
        return self.bounds.iteration

    @property
    def wall_s(self) -> float:
        return self.bounds.wall_s

    @property
    def gap_flagged(self) -> bool:
        return self.uncovered_s > GAP_LIMIT * self.wall_s

    def to_json(self) -> dict[str, Any]:
        b = self.bounds
        return {"iteration": b.iteration, "t_begin": b.t_begin, "t_end": b.t_end, "wall_s": b.wall_s,
                "begin_from": b.begin_from, "end_from": b.end_from,
                "critical_path": [s.to_json() for s in self.path], "critical_s": self.critical_s,
                "coverage_s": self.coverage_s, "busy_s": self.busy_s, "count": self.count,
                "category_critical_s": self.category_critical_s, "category_coverage_s": self.category_coverage_s,
                "category_exclusive_s": self.category_exclusive_s, "uncovered_s": self.uncovered_s,
                "gap_flagged": self.gap_flagged, "outside": self.outside, "unclassified": list(self.unclassified)}


def _overlaps(span: Span, lo: float, hi: float) -> bool:
    if span.t_start == span.t_end:
        return lo <= span.t_start <= hi
    return span.t_start < hi and span.t_end > lo


def summarize_iteration(spans: Sequence[Span], bounds: Bounds, *,
                        categories: Sequence[tuple[str, str]] = DEFAULT_CATEGORIES) -> IterationTimeline:
    lo, hi = bounds.t_begin, bounds.t_end
    mine = [s for s in spans if s.iteration == bounds.iteration]
    inside = [s for s in mine if _overlaps(s, lo, hi)]
    outside: dict[str, int] = defaultdict(int)
    for s in mine:
        if not _overlaps(s, lo, hi):
            outside[s.name] += 1

    by_name: dict[str, list[Interval]] = defaultdict(list)
    count: dict[str, int] = defaultdict(int)
    for s in inside:
        count[s.name] += 1
        by_name[s.name].extend(clip([(s.t_start, s.t_end)], lo, hi))
    category = {name: category_of(name, categories) or UNCLASSIFIED for name in count}

    path = critical_path(inside, lo, hi)
    critical: dict[str, float] = defaultdict(float)
    category_critical: dict[str, float] = defaultdict(float)
    for seg in path:
        critical[seg.name] += seg.duration
        category_critical[GAP if seg.span is None else category[seg.name]] += seg.duration

    by_category: dict[str, list[Interval]] = defaultdict(list)
    for name, ivs in by_name.items():
        by_category[category[name]].extend(ivs)
    exclusive: dict[str, float] = {}
    covered: list[Interval] = []
    for cat in PRIORITY:
        ivs = union(by_category.get(cat, []))
        exclusive[cat] = measure(subtract(ivs, covered))
        covered = union(covered + ivs)
    exclusive[IDLE] = bounds.wall_s - measure(covered)

    return IterationTimeline(
        bounds=bounds, path=tuple(path), critical_s=dict(critical),
        coverage_s={n: measure(v) for n, v in by_name.items()},
        busy_s={n: sum(b - a for a, b in v) for n, v in by_name.items()}, count=dict(count),
        category_critical_s=dict(category_critical),
        category_coverage_s={c: measure(v) for c, v in by_category.items()},
        category_exclusive_s=exclusive, uncovered_s=bounds.wall_s - measure(covered), outside=dict(outside),
        unclassified=tuple(sorted(n for n, c in category.items() if c == UNCLASSIFIED)))


def summarize_run(spans: Sequence[Span], mode: str = "v1", *, verdicts: Verdicts | None = None,
                  categories: Sequence[tuple[str, str]] = DEFAULT_CATEGORIES) -> list[IterationTimeline]:
    return [summarize_iteration(spans, b, categories=categories) for b in run_bounds(spans, mode, verdicts).values()]


# --- rendering and CLI -----------------------------------------------------------------------------


def _pct(x: float, wall: float) -> str:
    return f"{100.0 * x / wall:.1f}%"


def markdown(timelines: Sequence[IterationTimeline]) -> str:
    cats = [c for c in PRIORITY if any(t.category_exclusive_s.get(c) for t in timelines)] + [IDLE]
    head = ["iteration", "wall s"] + [f"{c} (excl.)" for c in cats] + ["uncovered", "critical path: largest three"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in timelines:
        top = sorted(t.critical_s.items(), key=lambda kv: -kv[1])[:3]
        cells = [str(t.iteration), f"{t.wall_s:.1f}"] + [_pct(t.category_exclusive_s.get(c, 0.0), t.wall_s) for c in cats]
        cells.append(_pct(t.uncovered_s, t.wall_s) + (" FLAG" if t.gap_flagged else ""))
        cells.append(", ".join(f"{n} {s:.1f}s" for n, s in top))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aorlta.measure.timeline", description=__doc__.split("\n\n")[0])
    ap.add_argument("spans", type=Path, help="spans.jsonl (Section 5 item 5)")
    ap.add_argument("--mode", default="v1", choices=sorted(BOUNDS),
                    help="v1: select to landing; v2: between publications; envelope: spans only (no per-256)")
    ap.add_argument("--path", type=int, metavar="N", help="also print iteration N's critical path")
    ap.add_argument("--json", type=Path, help="write every iteration's summary here")
    args = ap.parse_args(argv)
    try:
        timelines = summarize_run(read_jsonl(args.spans), args.mode)
    except MeasureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(markdown(timelines))
    if args.path is not None:
        chosen = {t.iteration: t for t in timelines}.get(args.path)
        if chosen is None:
            print(f"error: no iteration {args.path}", file=sys.stderr)
            return 1
        t = chosen
        for seg in t.path:
            print(f"{seg.t_start - t.bounds.t_begin:10.3f} {seg.t_end - t.bounds.t_begin:10.3f} {seg.duration:9.3f}  "
                  f"{seg.name}  {seg.to_json().get('id', '')}")
    if args.json:
        args.json.write_text(json.dumps([t.to_json() for t in timelines], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
