"""C9 compute curves: the ScaleRL sigmoid fit and the descriptive plots (design Section 6.9,
"The paper's compute-curve method"; arXiv 2510.13786 eq. 1 and appendix A.5).

    R_C - R_0 = (A - R_0) / (1 + (C_mid / C)^B)

The fit follows A.5: a grid over A (step 0.005) and over C_mid (100 linearly spaced values); for each
(A, C_mid), the B with the least sum of squared residuals; the cell with the least sum wins. R_0, the
validation reward before training, is given rather than fitted, as in eq. 1. Only points at or after
the burn-in count (the paper fits after about one epoch).

For AORLTA the curves are descriptive only (decision D1). The fitter refuses to report A, B or C_mid
with fewer than 8 post-burn-in points, or from a validation set that is not much larger than the
33 x 4 held-out set, because a three-parameter fit on a handful of noisy points would be noise with
error bars. "Much larger" is taken as at least 10 x 132 = 1,320 samples per point.

SciPy is not in the pinned environment, so B is fitted directly: a scan of 81 log-spaced values in
[0.01, 100], then golden-section search between the best scan value's neighbours. (The paper used
curve_fit from its default start, B = 1.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

from . import MeasureError

MIN_POINTS = 8
MIN_VALIDATION_SAMPLES = 10 * 33 * 4
A_STEP = 0.005
CMID_POINTS = 100
PAPER_A_RANGE = (0.450, 0.800)  # A.5's grid, for pass rates; other metrics need their own range
PAPER_CMID_RANGE = (100.0, 40000.0)  # GPU-hours
B_RANGE = (0.01, 100.0)
B_SCAN = 81
GOLDEN_STEPS = 48
_GOLD = (math.sqrt(5.0) - 1.0) / 2.0


class CurveError(MeasureError):
    pass


@dataclass(frozen=True)
class Point:
    compute: float  # GPU-hours (or wall-hours) of training so far
    reward: float  # validation reward or pass rate at that compute


def sigmoid(c: float, *, a: float, b: float, c_mid: float, r0: float) -> float:
    return r0 + (a - r0) * _tail(b * math.log(c_mid / c))


def _tail(z: float) -> float:
    """1 / (1 + e^z) without overflow."""
    if z > 0:
        e = math.exp(-z)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(z))


def a_grid(lo: float, hi: float, step: float = A_STEP) -> list[float]:
    n = round((hi - lo) / step)
    if n < 0 or abs(lo + n * step - hi) > 1e-9:
        raise CurveError(f"the A range [{lo}, {hi}] is not a whole number of {step} steps")
    return [round(lo + i * step, 10) for i in range(n + 1)]


def cmid_grid(lo: float, hi: float, n: int = CMID_POINTS) -> list[float]:
    if not (0 < lo < hi) or n < 2:
        raise CurveError(f"bad C_mid range [{lo}, {hi}] with {n} points")
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


@dataclass(frozen=True)
class SigmoidFit:
    status: str  # "fitted" | "refused"
    reason: str | None  # why it was refused
    n_points: int
    n_post_burn_in: int
    a: float | None = None
    b: float | None = None
    c_mid: float | None = None
    ssr: float | None = None
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {"status": self.status, "reason": self.reason, "n_points": self.n_points,
                "n_post_burn_in": self.n_post_burn_in, "a": self.a, "b": self.b, "c_mid": self.c_mid,
                "ssr": self.ssr, "warnings": list(self.warnings)}


def _ssr(ys: Sequence[float], us: Sequence[float], g: float, b: float) -> float:
    return math.fsum((y - g * _tail(b * u)) ** 2 for y, u in zip(ys, us))


def _fit_b(ys, us, g, k, log_bs) -> tuple[float, float]:
    """Golden-section search for B between the scan values next to scan index k."""
    lo, hi = log_bs[max(k - 1, 0)], log_bs[min(k + 1, len(log_bs) - 1)]
    x1, x2 = hi - _GOLD * (hi - lo), lo + _GOLD * (hi - lo)
    f1, f2 = _ssr(ys, us, g, math.exp(x1)), _ssr(ys, us, g, math.exp(x2))
    for _ in range(GOLDEN_STEPS):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - _GOLD * (hi - lo)
            f1 = _ssr(ys, us, g, math.exp(x1))
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + _GOLD * (hi - lo)
            f2 = _ssr(ys, us, g, math.exp(x2))
    best = min((f1, x1), (f2, x2), (_ssr(ys, us, g, math.exp(log_bs[k])), log_bs[k]))
    return math.exp(best[1]), best[0]


def fit_sigmoid(points: Sequence[Point], *, r0: float, burn_in: float, validation_samples: int,
                a_range: tuple[float, float], cmid_range: tuple[float, float], a_step: float = A_STEP,
                cmid_points: int = CMID_POINTS, b_range: tuple[float, float] = B_RANGE) -> SigmoidFit:
    for p in points:
        if not (math.isfinite(p.compute) and p.compute > 0 and math.isfinite(p.reward)):
            raise CurveError(f"bad point {p}: compute must be > 0 and both finite")
    if not math.isfinite(r0):
        raise CurveError("r0 must be finite")
    used = sorted((p for p in points if p.compute >= burn_in), key=lambda p: p.compute)
    n_all, n_used = len(points), len(used)

    def refuse(reason: str) -> SigmoidFit:
        return SigmoidFit("refused", reason, n_all, n_used)

    if type(validation_samples) is not int or validation_samples < MIN_VALIDATION_SAMPLES:
        return refuse(f"validation set of {validation_samples} samples per point; at least "
                      f"{MIN_VALIDATION_SAMPLES} (10 x the 33 x 4 held-out set) are needed to fit")
    if n_used < MIN_POINTS:
        return refuse(f"{n_used} points at or after the burn-in; at least {MIN_POINTS} are needed to fit")
    a_values, cmids = a_grid(*a_range, a_step), cmid_grid(*cmid_range, cmid_points)
    if a_values[0] <= r0:
        raise CurveError(f"the A grid starts at {a_values[0]}, not above R_0 = {r0}: a rising curve needs A > R_0")
    ys = [p.reward - r0 for p in used]
    log_c = [math.log(p.compute) for p in used]
    lb_lo, lb_hi = math.log(b_range[0]), math.log(b_range[1])
    log_bs = [lb_lo + (lb_hi - lb_lo) * k / (B_SCAN - 1) for k in range(B_SCAN)]
    syy = math.fsum(y * y for y in ys)
    best: tuple[float, float, float, float] | None = None  # (ssr, a, c_mid, b)
    for cm in cmids:
        us = [math.log(cm) - x for x in log_c]
        scan = []
        for lb in log_bs:
            s = [_tail(math.exp(lb) * u) for u in us]
            scan.append((math.fsum(y * si for y, si in zip(ys, s)), math.fsum(si * si for si in s)))
        for a in a_values:
            g = a - r0
            k = min(range(B_SCAN), key=lambda i: syy - 2 * g * scan[i][0] + g * g * scan[i][1])
            b, ssr = _fit_b(ys, us, g, k, log_bs)
            if best is None or ssr < best[0]:
                best = (ssr, a, cm, b)
    ssr, a, cm, b = best
    warnings = []
    if a in (a_values[0], a_values[-1]):
        warnings.append("A is at the edge of its grid: the asymptote may lie outside it")
    if cm in (cmids[0], cmids[-1]):
        warnings.append("C_mid is at the edge of its grid")
    if not (math.exp(log_bs[1]) < b < math.exp(log_bs[-2])):
        warnings.append("B is at the edge of its search range")
    return SigmoidFit("fitted", None, n_all, n_used, a=a, b=b, c_mid=cm, ssr=ssr, warnings=tuple(warnings))


# --- descriptive plots -----------------------------------------------------------------------------

PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step + 1e-9) + 1)]


def svg_plot(series: Mapping[str, Sequence[tuple[float, float]]], *, title: str, x_label: str, y_label: str,
             width: int = 640, height: int = 400) -> str:
    """A line plot with markers, one line per series, as a standalone SVG document."""
    pts = [p for ps in series.values() for p in ps]
    if not pts:
        raise CurveError("nothing to plot")
    for x, y in pts:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CurveError(f"non-finite point ({x}, {y})")
    x0, x1 = min(0.0, min(x for x, _ in pts)), max(x for x, _ in pts)
    y0, y1 = min(y for _, y in pts), max(y for _, y in pts)
    pad = 0.08 * (y1 - y0 or 1.0)
    y0, y1 = y0 - pad, y1 + pad
    if x1 <= x0:
        x1 = x0 + 1.0
    left, right, top, bottom = 70, 20, 40, 55
    pw, ph = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + pw * (x - x0) / (x1 - x0)

    def sy(y: float) -> float:
        return top + ph * (1 - (y - y0) / (y1 - y0))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width / 2:.1f}" y="22" text-anchor="middle" font-size="14">{escape(title)}</text>',
           f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#444"/>']
    for t in _ticks(x0, x1):
        out.append(f'<line x1="{sx(t):.1f}" y1="{top + ph}" x2="{sx(t):.1f}" y2="{top + ph + 5}" stroke="#444"/>')
        out.append(f'<text x="{sx(t):.1f}" y="{top + ph + 18}" text-anchor="middle">{t:g}</text>')
    for t in _ticks(y0, y1):
        out.append(f'<line x1="{left - 5}" y1="{sy(t):.1f}" x2="{left}" y2="{sy(t):.1f}" stroke="#444"/>')
        out.append(f'<text x="{left - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    out.append(f'<text x="{left + pw / 2:.1f}" y="{height - 12}" text-anchor="middle">{escape(x_label)}</text>')
    out.append(f'<text x="16" y="{top + ph / 2:.1f}" text-anchor="middle" '
               f'transform="rotate(-90 16 {top + ph / 2:.1f})">{escape(y_label)}</text>')
    for i, (label, ps) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        ordered = sorted(ps)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ordered)
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        out += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>' for x, y in ordered]
        ly = top + 16 + 18 * i
        out.append(f'<line x1="{left + 12}" y1="{ly}" x2="{left + 36}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{left + 42}" y="{ly + 4}">{escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
