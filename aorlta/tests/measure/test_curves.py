"""Compute curves (C9.T4): the A.5 sigmoid fitter and the descriptive plots."""

import math
import xml.etree.ElementTree as ET

import pytest

from aorlta.measure import curves as c
from aorlta.measure.curves import CurveError, Point

R0 = 0.30
COMPUTE = [1500, 2000, 3000, 4500, 6000, 8000, 10000, 12000, 16000, 20000]  # GPU-hours, all past the burn-in
CMIDS = c.cmid_grid(*c.PAPER_CMID_RANGE)
FIT = dict(r0=R0, burn_in=1500, validation_samples=16000, a_range=c.PAPER_A_RANGE, cmid_range=c.PAPER_CMID_RANGE)


def curve(a, b, c_mid, xs=COMPUTE):
    return [Point(x, c.sigmoid(x, a=a, b=b, c_mid=c_mid, r0=R0)) for x in xs]


# Table 1 of the paper (ScaleRL, -8gen, -32k), with C_mid moved to the nearest A.5 grid value.
TRUTHS = [(0.610, 1.92, 6), (0.585, 2.44, 7), (0.645, 1.89, 27)]


# --- C9.T4 -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("a, b, k", TRUTHS)
def test_c9_t4_noiseless_points_give_back_a_b_and_c_mid(a, b, k):
    fit = c.fit_sigmoid(curve(a, b, CMIDS[k]), **FIT)
    assert fit.status == "fitted" and fit.reason is None and fit.warnings == ()
    assert fit.a == a and fit.c_mid == CMIDS[k]  # the planted grid cell, exactly
    assert fit.b == pytest.approx(b, rel=1e-8)
    assert fit.ssr < 1e-20
    assert (fit.n_points, fit.n_post_burn_in) == (10, 10)


def test_c9_t4_points_before_the_burn_in_do_not_count():
    early = [Point(200, 0.31), Point(600, 0.38), Point(1000, 0.47)]  # the near-linear first epoch, off the curve
    a, b, k = TRUTHS[0]
    fit = c.fit_sigmoid(early + curve(a, b, CMIDS[k]), **FIT)
    assert (fit.n_points, fit.n_post_burn_in) == (13, 10)
    assert (fit.a, fit.c_mid) == (a, CMIDS[k]) and fit.b == pytest.approx(b, rel=1e-8)


def test_c9_t4_refuses_below_eight_post_burn_in_points():
    a, b, k = TRUTHS[0]
    seven = c.fit_sigmoid(curve(a, b, CMIDS[k], COMPUTE[:7]), **FIT)
    assert (seven.status, seven.a, seven.b, seven.c_mid, seven.ssr) == ("refused", None, None, None, None)
    assert "7 points at or after the burn-in; at least 8" in seven.reason
    eight = c.fit_sigmoid(curve(a, b, CMIDS[k], COMPUTE[:8]), **FIT)
    assert eight.status == "fitted" and eight.a == a and eight.b == pytest.approx(b, rel=1e-6)


def test_c9_t4_the_floor_counts_post_burn_in_points_only():
    a, b, k = TRUTHS[0]
    pts = curve(a, b, CMIDS[k], [300, 600, 900, 1200, 1499] + COMPUTE[:7])  # 12 points, 7 from the burn-in on
    fit = c.fit_sigmoid(pts, **FIT)
    assert fit.status == "refused" and (fit.n_points, fit.n_post_burn_in) == (12, 7)


def test_c9_t4_refuses_a_small_validation_set():
    a, b, k = TRUTHS[0]
    pts = curve(a, b, CMIDS[k])
    for n, status in [(132, "refused"), (1319, "refused"), (1320, "fitted")]:
        fit = c.fit_sigmoid(pts, **{**FIT, "validation_samples": n})
        assert fit.status == status, n
        if status == "refused":
            assert "1320" in fit.reason and fit.a is None and fit.b is None


def test_an_off_grid_truth_comes_back_within_the_grid_resolution():
    # C_mid = 2700 lies between grid values 2518 and 2921; with C_mid on 2518, a lower A compensates
    # (they trade off), so A lands 1.5 steps low, measured: A 0.605, C_mid 2518.2, B 1.888, max |residual|
    # 0.0066. That is the resolution of A.5's grid, well inside the paper's +-0.02 margin on A.
    pts = curve(0.6123, 1.8, 2700.0)
    fit = c.fit_sigmoid(pts, **FIT)
    assert abs(fit.a - 0.6123) <= 2 * c.A_STEP
    assert abs(fit.c_mid - 2700.0) <= CMIDS[1] - CMIDS[0]
    assert fit.b == pytest.approx(1.8, rel=0.1)
    worst = max(abs(p.reward - c.sigmoid(p.compute, a=fit.a, b=fit.b, c_mid=fit.c_mid, r0=R0)) for p in pts)
    assert worst < 0.01


def test_an_asymptote_beyond_the_grid_is_flagged():
    fit = c.fit_sigmoid(curve(0.86, 1.5, CMIDS[10]), **FIT)
    assert fit.a == 0.8 and any("A is at the edge" in w for w in fit.warnings)


# --- the formula and the grids -----------------------------------------------------------------------


def test_the_sigmoid_is_equation_1():
    kw = dict(a=0.61, b=1.92, c_mid=2542.0, r0=R0)
    assert c.sigmoid(2542.0, **kw) == pytest.approx(R0 + (0.61 - R0) / 2)
    assert c.sigmoid(1e12, **kw) == pytest.approx(0.61)
    assert c.sigmoid(1e-9, **kw) == pytest.approx(R0)
    x = 5000.0
    assert c.sigmoid(x, **kw) == pytest.approx(R0 + (0.61 - R0) / (1 + (2542.0 / x) ** 1.92), rel=1e-14)
    assert c.sigmoid(1e-300, a=0.61, b=100.0, c_mid=40000.0, r0=R0) == pytest.approx(R0)  # no overflow


def test_the_grids_are_those_of_a5():
    a = c.a_grid(*c.PAPER_A_RANGE)
    assert (len(a), a[0], a[-1], a[32]) == (71, 0.45, 0.8, 0.61)
    assert all(round(y - x, 10) == 0.005 for x, y in zip(a, a[1:]))
    assert (len(CMIDS), CMIDS[0], CMIDS[-1]) == (100, 100.0, 40000.0)
    assert CMIDS[1] - CMIDS[0] == pytest.approx(39900 / 99)
    with pytest.raises(CurveError, match="whole number"):
        c.a_grid(0.45, 0.8012)


@pytest.mark.parametrize("points, kwargs, message", [
    ([Point(0, 0.5)], {}, "compute must be > 0"),
    ([Point(10, math.nan)], {}, "finite"),
    (curve(0.61, 1.92, CMIDS[6]), {"a_range": (0.30, 0.80)}, "not above R_0"),
    (curve(0.61, 1.92, CMIDS[6]), {"r0": math.inf}, "r0 must be finite"),
])
def test_bad_input_is_refused(points, kwargs, message):
    with pytest.raises(CurveError, match=message):
        c.fit_sigmoid(points, **{**FIT, **kwargs})


# --- plots ---------------------------------------------------------------------------------------


def test_svg_plot_draws_each_series_and_escapes_text():
    svg = c.svg_plot({"AORLTA <v1>": [(1, 0.3), (3, 0.4), (2, 0.35)], "Miles & co": [(2, 0.32), (6, 0.41)]},
                     title="validation reward vs GPU-hours", x_label="GPU-hours", y_label="reward")
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    lines = root.findall(f"{ns}polyline")
    assert len(lines) == 2 and len(root.findall(f"{ns}circle")) == 5
    xs = [float(p.split(",")[0]) for p in lines[0].get("points").split()]
    assert xs == sorted(xs)  # each line is drawn in x order
    texts = [t.text for t in root.iter(f"{ns}text")]
    assert "AORLTA <v1>" in texts and "Miles & co" in texts and "validation reward vs GPU-hours" in texts


def test_svg_plot_refuses_nothing_or_non_finite():
    with pytest.raises(CurveError, match="nothing"):
        c.svg_plot({}, title="t", x_label="x", y_label="y")
    with pytest.raises(CurveError, match="non-finite"):
        c.svg_plot({"a": [(1, math.inf)]}, title="t", x_label="x", y_label="y")
