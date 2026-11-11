"""The comparison report (D1): statistics, validity, verdicts, the bundle builder and the sample."""

import json
import math
import random
import re
import statistics
import subprocess
from dataclasses import replace

import pytest
from measure_helpers import T0, v1_iteration

from aorlta.config import plain
from aorlta.episodes.writer import SPANS_NAME, complete_iterations, iteration_dir_name, read_episodes, read_summary
from aorlta.measure import compare as cmp
from aorlta.measure import per256
from aorlta.measure.compare import (Allocation, Bundle, CompareError, Heldout, IterationRecord, Protocol, Validation,
                                    ValidationPoint)
from aorlta.orchestrator.fakes import FakeRunSpec, build_fake_run
from aorlta.spans import Span, read_jsonl

# --- statistics --------------------------------------------------------------------------------------

# Student t quantiles from printed tables (two-sided 95%, one-sided 95%, two-sided 99%).
T_TABLE = [(0.975, 1, 12.7062), (0.975, 2, 4.3027), (0.975, 4, 2.7764), (0.975, 5, 2.5706), (0.975, 9, 2.2622),
           (0.975, 30, 2.0423), (0.95, 10, 1.8125), (0.995, 4, 4.6041), (0.975, 1000, 1.9623)]


@pytest.mark.parametrize("p, df, expected", T_TABLE)
def test_t_quantile_matches_the_table(p, df, expected):
    assert cmp.t_quantile(p, df) == pytest.approx(expected, abs=6e-5)
    assert cmp.t_quantile(1 - p, df) == pytest.approx(-expected, abs=6e-5)


@pytest.mark.parametrize("df", [1, 3, 9, 40])
def test_t_cdf_agrees_with_integrating_the_density(df):
    def pdf(x):
        return math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi) \
            * (1 + x * x / df) ** (-(df + 1) / 2)

    for t in (0.3, 1.0, 2.2, 4.0):
        n = 4000  # Simpson's rule on [0, t]
        h = t / n
        area = h / 3 * (pdf(0) + pdf(t) + sum((4 if k % 2 else 2) * pdf(k * h) for k in range(1, n)))
        assert cmp.t_cdf(t, df) == pytest.approx(0.5 + area, abs=1e-10)
        assert cmp.t_cdf(-t, df) == pytest.approx(0.5 - area, abs=1e-10)


def test_paired_t_interval_is_the_hand_calculation():
    # mean 0.008; deviations 0.002 -0.028 0.022 -0.008 0.012; sum of squares 1.48e-3; sd sqrt(3.7e-4)
    # = 0.0192353841; se = sd / sqrt(5) = 0.0086023253; t(0.975, 4) = 2.7764451052; half = 0.0238838
    ci, sd, df = cmp.paired_t_interval([0.01, -0.02, 0.03, 0.00, 0.02], 0.95)
    assert (ci.estimate, sd, df) == (pytest.approx(0.008), pytest.approx(0.0192353841), 4)
    assert (ci.lo, ci.hi) == (pytest.approx(0.008 - 0.0238838, abs=1e-7), pytest.approx(0.008 + 0.0238838, abs=1e-7))
    with pytest.raises(CompareError):
        cmp.paired_t_interval([0.1], 0.95)


def _reference_item_bootstrap(d, resamples, seed, level):
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))

    def q(p):
        h = (len(means) - 1) * p
        k = math.floor(h)
        return means[k] + (h - k) * (means[k + 1] - means[k])

    return q((1 - level) / 2), q(1 - (1 - level) / 2)


def test_the_cluster_bootstrap_resamples_whole_items():
    rng = random.Random(3)
    d = [rng.choice([-0.75, -0.25, 0.0, 0.25, 0.5]) for _ in range(33)]
    ci = cmp.cluster_bootstrap(d, resamples=4000, seed=11, level=0.95)
    lo, hi = _reference_item_bootstrap(d, 4000, 11, 0.95)
    assert (ci.lo, ci.hi) == (pytest.approx(lo, abs=1e-12), pytest.approx(hi, abs=1e-12))
    assert ci.estimate == pytest.approx(statistics.fmean(d))
    assert cmp.cluster_bootstrap(d, resamples=4000, seed=11, level=0.95) == ci  # same seed, same interval
    with pytest.raises(CompareError):
        cmp.cluster_bootstrap([0.1], resamples=1000, seed=1, level=0.95)


# --- bundles and protocols ----------------------------------------------------------------------------

HELD = {"merged_base_sha256": "a" * 64, "P": 32, "K": 8, "reward_version": "11", "lr": 1e-5}
N = 6


def _protocol(**over):
    raw = {"schema": "aorlta.measure.protocol/1", "title": "test", "iterations": N, "held_equal": list(HELD),
           "speed": {"statistic": "median", "min_speedup": 1.0}, "parity": {"tolerance": 0.05, "ci_level": 0.95},
           "heldout": {"modes": ["full"], "margin": 0.10, "ci_level": 0.95, "resamples": 2000, "seed": 5}}
    for key, value in over.items():
        raw[key] = value
    return cmp._decode(Protocol, raw, "protocol")


def _heldout(rates, n_items=40, per_item=8, seed=0):
    rng = random.Random(seed)
    return Heldout(N, "e1", {"full": {f"i{k}": tuple(int(rng.random() < rates[k % len(rates)]) for _ in range(per_item))
                                      for k in range(n_items)}})


def _bundle(name, walls, rewards, *, scored=250, held=HELD, prompts=None, heldout=None, stalls=()):
    its = tuple(IterationRecord(iteration=n, wall_s=w, samples=256, scored=scored, stalled=n in stalls,
                                stall_reason="deadline" if n in stalls else None, mean_train_reward=r,
                                prompts_sha256=(prompts or {}).get(n, f"p{n}"), truncated=10, noop=False)
                for n, (w, r) in enumerate(zip(walls, rewards), 1))
    return Bundle("aorlta.measure.bundle/1", name, False, dict(held), Allocation(8, 1), its, heldout)


BASE_REWARDS = [-0.40, -0.35, -0.31, -0.26, -0.22, -0.20]


def _pair(a_walls=(600,) * N, b_walls=(2100,) * N, a_shift=(0.01, -0.01, 0.0, 0.01, -0.01, 0.0), **kw):
    a = _bundle("AORLTA", a_walls, [r + s for r, s in zip(BASE_REWARDS, a_shift)], heldout=_heldout([0.3, 0.6]), **kw)
    b = _bundle("baseline", b_walls, BASE_REWARDS, heldout=_heldout([0.3, 0.6]))
    return a, b


def test_a_valid_comparison_that_passes_everything():
    a, b = _pair()
    c = cmp.compare(_protocol(), a, b)
    assert c.valid and c.overall == "PASS"
    assert [(x.key, x.verdict) for x in c.criteria] == [("speed", "PASS"), ("parity", "PASS"), ("heldout.full", "PASS")]
    speed = c.criteria[0].numbers
    assert speed["speedup"] == pytest.approx(3.5) and speed["aorlta_s"] == pytest.approx(256 * 600 / 250)


@pytest.mark.parametrize("change, problem", [
    (lambda a: replace(a, held_equal={**HELD, "lr": 2e-5}), "'lr' differs"),
    (lambda a: replace(a, held_equal={k: v for k, v in HELD.items() if k != "K"}), "'K' is not recorded by AORLTA"),
    (lambda a: replace(a, iterations=a.iterations[:-1]), r"AORLTA lacks iterations \[6\]"),
    (lambda a: replace(a, iterations=tuple(replace(r, prompts_sha256="other") if r.iteration == 3 else r
                                           for r in a.iterations)), r"prompt sets differ at iterations \[3\]"),
    (lambda a: replace(a, heldout=replace(a.heldout, iteration=N - 1)), "held-out eval is at iteration 5"),
])
def test_an_invalid_comparison_evaluates_no_criterion(change, problem):
    a, b = _pair()
    c = cmp.compare(_protocol(), change(a), b)
    assert not c.valid and c.overall == "INVALID" and c.criteria == ()
    assert any(re.search(problem, x) for x in c.problems), c.problems


def test_speed_needs_more_than_the_pre_registered_factor():
    a, b = _pair(a_walls=(1000,) * N, b_walls=(2000,) * N)
    assert cmp.compare(_protocol(speed={"statistic": "median", "min_speedup": 1.9}), a, b).criteria[0].verdict == "PASS"
    assert cmp.compare(_protocol(speed={"statistic": "median", "min_speedup": 2.0}), a, b).criteria[0].verdict == "FAIL"
    slower = _pair(a_walls=(2100,) * N, b_walls=(2000,) * N)
    assert cmp.compare(_protocol(), *slower).criteria[0].verdict == "FAIL"


def test_the_speed_statistic_is_the_one_pre_registered():
    walls = (600, 600, 600, 600, 600, 6000)  # AORLTA stalls in iteration 6
    a, b = _pair(a_walls=walls, b_walls=(1300,) * N, stalls=(6,))
    verdicts = {s: cmp.compare(_protocol(speed={"statistic": s, "min_speedup": 1.0}), a, b).criteria[0].verdict
                for s in ("median", "mean_without_stalls", "mean_with_stalls")}
    assert verdicts == {"median": "PASS", "mean_without_stalls": "PASS", "mean_with_stalls": "FAIL"}


def test_parity_needs_the_whole_interval_inside_the_tolerance():
    inside = (0.03, 0.035, 0.032, 0.03, 0.033, 0.031)  # mean 0.0318, tight: [0.0298, 0.0339]
    assert cmp.compare(_protocol(), *_pair(a_shift=inside)).criteria[1].verdict == "PASS"
    noisy = (0.09, -0.06, 0.08, -0.05, 0.07, -0.04)  # mean 0.015 inside +-0.05, interval [-0.064, 0.094] is not
    c = cmp.compare(_protocol(), *_pair(a_shift=noisy))
    parity = c.criteria[1]
    assert parity.verdict == "FAIL" and abs(parity.numbers["mean"]) < 0.05 and parity.numbers["lo"] < -0.05
    assert c.overall == "FAIL" and "a speed result means nothing while learning per iteration differs" in c.overall_why
    off = (0.08, 0.09, 0.085, 0.08, 0.09, 0.085)
    assert cmp.compare(_protocol(), *_pair(a_shift=off)).criteria[1].verdict == "FAIL"


def test_heldout_non_inferiority_uses_the_lower_bound():
    a, b = _pair()
    a = replace(a, heldout=_heldout([0.3, 0.6], seed=1))
    loose = cmp.compare(_protocol(heldout={"modes": ["full"], "margin": 0.2, "ci_level": 0.95, "resamples": 2000,
                                           "seed": 5}), a, b).criteria[2]
    tight = cmp.compare(_protocol(heldout={"modes": ["full"], "margin": 0.0, "ci_level": 0.95, "resamples": 2000,
                                           "seed": 5}), a, b).criteria[2]
    assert loose.verdict == "PASS" and loose.numbers["lo"] > -0.2
    assert tight.verdict == "FAIL" and tight.numbers["lo"] < 0 < tight.numbers["hi"]
    worse = replace(a, heldout=_heldout([0.1, 0.3], seed=1))
    assert cmp.compare(_protocol(), worse, b).criteria[2].verdict == "FAIL"


def test_heldout_pairs_items_and_reports_rates():
    a, b = _pair()
    a = replace(a, heldout=_heldout([0.3, 0.6], seed=1))  # the two sides must differ, or every pairing agrees
    hd = cmp.compare(_protocol(), a, b).criteria[2].numbers
    assert hd["items"] == 40 and hd["samples"] == {"aorlta": 320, "baseline": 320}
    d = [statistics.fmean(a.heldout.modes["full"][i]) - statistics.fmean(b.heldout.modes["full"][i])
         for i in sorted(a.heldout.modes["full"])]
    assert len(set(d)) > 3
    lo, hi = _reference_item_bootstrap(d, 2000, 5, 0.95)
    assert (hd["lo"], hd["hi"]) == (pytest.approx(lo, abs=1e-12), pytest.approx(hi, abs=1e-12))
    assert hd["lo"] < hd["mean_diff"] < hd["hi"]


def test_the_bootstrap_unit_is_the_item_not_the_sample():
    """Samples within an item agree (all 1 or all 0), so the 8 samples of an item carry one item's worth
    of evidence. Resampling items gives the wide interval; resampling 8 x 40 samples would be ~sqrt(8)
    times narrower."""
    a, b = _pair()
    per_item = {f"i{k}": (1,) * 8 if k % 2 else (0,) * 8 for k in range(40)}
    flipped = {f"i{k}": (0,) * 8 if k % 2 else (1,) * 8 for k in range(40)}
    a = replace(a, heldout=Heldout(N, "e1", {"full": per_item}))
    b = replace(b, heldout=Heldout(N, "e1", {"full": flipped}))
    hd = cmp.compare(_protocol(), a, b).criteria[2].numbers
    assert hd["mean_diff"] == 0.0
    # 40 item differences of +-1: sd 1, so the 95% interval is about +-1.96/sqrt(40) = +-0.31
    assert hd["hi"] - hd["lo"] == pytest.approx(2 * 1.96 / math.sqrt(40), rel=0.12)


def test_missing_inputs_leave_criteria_not_evaluated():
    a, b = _pair()
    a = replace(a, iterations=tuple(replace(r, mean_train_reward=None) if r.iteration == 2 else r for r in a.iterations),
                heldout=None)
    c = cmp.compare(_protocol(), a, b)
    assert c.valid and c.overall == "INCOMPLETE"
    assert [(x.key, x.verdict) for x in c.criteria] == [("speed", "PASS"), ("parity", "NOT EVALUATED"),
                                                        ("heldout.full", "NOT EVALUATED")]
    assert "iterations [2]" in c.criteria[1].why
    a2, b2 = _pair()
    a2 = replace(a2, iterations=tuple(replace(r, prompts_sha256=None) for r in a2.iterations))
    assert "pairing is unverified" in cmp.compare(_protocol(), a2, b2).criteria[1].why
    b3 = replace(b2, heldout=replace(b2.heldout, modes={"full": dict(list(b2.heldout.modes["full"].items())[:-1])}))
    assert "item sets differ" in cmp.compare(_protocol(), *_pair()[:1], b3).criteria[2].why


@pytest.mark.parametrize("over, message", [
    ({"speed": {"statistic": "median", "min_speedup": 0.9}}, "min_speedup"),
    ({"speed": {"statistic": "fastest", "min_speedup": 1.0}}, "expected one of"),
    ({"parity": {"tolerance": 0.0, "ci_level": 0.95}}, "tolerance > 0"),
    ({"heldout": {"modes": ["full"], "margin": 0.1, "ci_level": 0.95, "resamples": 100, "seed": 1}}, "resamples"),
    ({"iterations": 1}, "at least 2"),
    ({"held_equal": []}, "names nothing"),
    ({"parity": {"tolerance": 0.05}}, "missing required key"),
    ({"extra": 1}, "unknown key"),
])
def test_the_protocol_has_no_defaults_and_no_unknown_keys(over, message):
    with pytest.raises(CompareError, match=message):
        _protocol(**over)


def test_bundles_are_checked_on_load(tmp_path):
    a, _ = _pair()
    good = tmp_path / "a.json"
    good.write_text(json.dumps(plain(a)))
    assert cmp.load_bundle(good) == a
    raw = plain(a)
    raw["heldout"]["modes"]["full"]["i0"] = [1, 2]
    good.write_text(json.dumps(raw))
    with pytest.raises(CompareError, match="0 and 1"):
        cmp.load_bundle(good)
    raw = plain(a)
    raw["iterations"][0]["unscored_by_reason"] = {"asan_timeout": 1}
    good.write_text(json.dumps(raw))
    with pytest.raises(CompareError, match="unscored by reason sums to 1, not 6"):
        cmp.load_bundle(good)
    raw = plain(a)
    raw["iterations"][1]["scored"] = 0
    good.write_text(json.dumps(raw))
    cmp.load_bundle(good)  # loads; the metric then refuses it
    with pytest.raises(per256.Per256Error, match="no scored samples"):
        cmp.compare(_protocol(), cmp.load_bundle(good), _pair()[1])
    raw = plain(a)
    raw["iterations"][0]["scored"] = 0  # iteration 1 is not reported, so it may score nothing
    good.write_text(json.dumps(raw))
    cmp.compare(_protocol(), cmp.load_bundle(good), _pair()[1])


# --- AORLTA's bundle from spans and episodes -----------------------------------------------------------


def _run(n_iter=3):
    spans, episodes = [], []
    for n in range(1, n_iter + 1):
        shift = (n - 1) * 300.0
        spans += [Span(s.name, n, s.t_start + shift, s.t_end + shift, s.attrs) for s in v1_iteration(n)]
        for j in range(20):
            unscored = j < 4
            episodes.append({"schema": "aorlta.episode/1", "iteration": n, "chain_id": f"r/{n}/standin:p{j // 4}/{j % 4}",
                             "prompt_id": f"standin:p{j // 4}", "status": "unscored" if unscored else "scored",
                             "reward": None if unscored else 0.1 * n, "unscored_reason": "asan_timeout" if unscored else None,
                             "turns": [{"finish_reason": {"type": "length" if j == 7 else "stop"}}], "final_turn": 0})
    return spans, episodes


def test_bundle_from_run_reads_walls_phases_and_counts():
    spans, episodes = _run()
    b = cmp.bundle_from_run(spans, episodes, system="AORLTA", held_equal=HELD, allocation=Allocation(8, 1),
                            stalls={2: "scoring deadline"})
    assert [r.wall_s for r in b.iterations] == [176.0, 176.0, 176.0]
    r = b.iterations[1]
    assert (r.samples, r.scored, r.unscored_by_reason, r.truncated, r.stalled) == (20, 16, {"asan_timeout": 4}, 1, True)
    assert r.mean_train_reward == pytest.approx(0.2) and r.noop is False
    assert r.check_share == pytest.approx((37.5 / 176, 37.5 / 176)) and r.check_status == "refuted"
    assert r.category_coverage_s["generation"] == pytest.approx(79)
    assert sum(r.category_critical_s.values()) == pytest.approx(176)
    assert r.prompts_sha256 == cmp.prompt_set_sha256([f"standin:p{k}" for k in range(5)])
    assert len({x.prompts_sha256 for x in b.iterations}) == 1  # the same five prompts each iteration


def test_bundle_from_run_needs_prompt_ids_and_markers():
    spans, episodes = _run()
    del episodes[3]["prompt_id"]
    with pytest.raises(CompareError, match="no prompt_id"):
        cmp.bundle_from_run(spans, episodes, system="A", held_equal=HELD, allocation=Allocation(8, 1), stalls={})
    spans, episodes = _run()
    with pytest.raises(CompareError, match="v1 or v2"):
        cmp.bundle_from_run(spans, episodes, system="A", held_equal=HELD, allocation=Allocation(8, 1), stalls={},
                            mode="envelope")


@pytest.mark.parametrize("spec, planted", [
    (dict(engine={"error_rate": 0.2, "seed": 4}), lambda eps, sums: any(not e["turns"] for e in eps)),
    (dict(leak_check=True, engine={"error_rate": 0.0}, rollout={"quarantine_rate": 0.5}),
     lambda eps, sums: any(e["status"] == "quarantined" for e in eps)),
    (dict(engine={"error_rate": 0.0}, reward={"constant": 0.5}),
     lambda eps, sums: not any(s["trainer"]["step_taken"] for s in sums.values())),
], ids=["lost-before-a-reply", "quarantined", "no-step"])
def test_bundle_from_run_measures_every_iteration_a_fake_run_committed(tmp_path, spec, planted):
    r = build_fake_run(FakeRunSpec(root=str(tmp_path / "runs"), iterations=2, eval_every=None, **spec))
    assert r.orchestrator.run()["status"] == "completed"
    its = complete_iterations(r.cfg.run_dir)
    sums = {n: read_summary(r.cfg.run_dir, n) for n in its}
    eps = [e for n in its for e in read_episodes(r.cfg.run_dir, n)]
    assert planted(eps, sums)
    spans = [s for n in its for s in read_jsonl(r.cfg.run_dir / "iterations" / iteration_dir_name(n) / SPANS_NAME)]
    verdicts = {n: list(s["sync"]["verdicts"].values()) for n, s in sums.items() if s["sync"]["published"]}
    b = cmp.bundle_from_run(spans, eps, system="AORLTA", held_equal=HELD, allocation=Allocation(8, 1), stalls={},
                            landing_verdicts=verdicts)
    assert [x.iteration for x in b.iterations] == its == [1, 2]
    for x in b.iterations:
        s = sums[x.iteration]
        assert (x.samples, x.scored, x.unscored_by_reason) == (s["samples"]["total"], s["samples"]["scored"],
                                                                s["unscored_by_reason"])
        assert x.wall_s == pytest.approx(s["timing"]["wall_s"], rel=1e-12)
        assert x.noop == (not s["trainer"]["step_taken"])


def test_the_prompt_set_hash_ignores_order_and_repeats():
    assert cmp.prompt_set_sha256(["b", "a", "a"]) == cmp.prompt_set_sha256(["a", "b"]) != cmp.prompt_set_sha256(["a"])


# --- pre-registration -----------------------------------------------------------------------------------


def _git(repo, *args):
    return subprocess.run(["/usr/bin/git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           *args], capture_output=True, text=True, check=True).stdout.strip()


def test_the_protocol_must_match_its_pre_registration_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    proto = repo / "protocol.yaml"
    proto.write_text("schema: aorlta.measure.protocol/1\n")
    _git(repo, "add", "protocol.yaml")
    _git(repo, "commit", "-q", "-m", "pre-register")
    commit = _git(repo, "rev-parse", "HEAD")
    assert cmp.verify_preregistration(proto, commit, repo) == f"identical to protocol.yaml at commit {commit[:12]}"
    proto.write_text("schema: aorlta.measure.protocol/1\n# tolerance loosened after seeing the data\n")
    with pytest.raises(CompareError, match="differs from the copy pre-registered"):
        cmp.verify_preregistration(proto, commit, repo)
    with pytest.raises(CompareError, match="has no"):
        cmp.verify_preregistration(repo / "other.yaml", commit, repo)


# --- the report and the sample ----------------------------------------------------------------------------


def test_the_report_states_every_verdict_and_the_problems(tmp_path):
    a, b = _pair()
    text = cmp.markdown(cmp.compare(_protocol(), a, b))
    assert "**Overall: PASS.**" in text and "| VALID |" in text and "SYNTHETIC" not in text
    bad = cmp.markdown(cmp.compare(_protocol(), replace(a, held_equal={**HELD, "lr": 3e-5}), b))
    assert "**Overall: INVALID.**" in bad and "- held-equal field 'lr' differs" in bad and "| lr | `3e-05` | `1e-05` | **no** |" in bad
    assert "## 1." not in bad


def test_the_sample_report_is_generated_from_synthetic_data(tmp_path):
    c = cmp.make_sample(tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"report.md", "report.json", "protocol.yaml", "aorlta.bundle.json", "baseline.bundle.json",
            "baseline-summaries.json", "baseline-mapping.yaml", "validation-gpu-hours.svg",
            "validation-wall-hours.svg"} <= names
    text = (tmp_path / "report.md").read_text()
    assert text.count("SYNTHETIC") >= 3 and "NOT VERIFIED" in text
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["synthetic"] is True and report["valid"] is True
    assert [x["key"] for x in report["criteria"]] == ["speed", "parity", "heldout.full", "heldout.evidence_only"]
    # the report's numbers follow from its own inputs, recomputed here without compare.py
    a, b = cmp.load_bundle(tmp_path / "aorlta.bundle.json"), cmp.load_bundle(tmp_path / "baseline.bundle.json")
    sa = per256.summarize([r.stats() for r in a.iterations])
    sb = per256.summarize([r.stats() for r in b.iterations])
    speed = report["criteria"][0]["numbers"]
    assert (speed["aorlta_s"], speed["baseline_s"]) == (pytest.approx(sa.median_s), pytest.approx(sb.median_s))
    diffs = [x.mean_train_reward - y.mean_train_reward for x, y in zip(a.iterations, b.iterations)]
    assert report["criteria"][1]["numbers"]["mean"] == pytest.approx(statistics.fmean(diffs))
    assert {6} == set(sa.stalled) and {4} == set(sb.stalled)  # the planted stalls
    fits = report["descriptive"]["aorlta"]["curves"]["fits"]
    assert all(f["status"] == "refused" and "at least 8" in f["reason"] for f in fits.values())  # 5 points, D1
    assert c.overall in ("PASS", "FAIL", "INCOMPLETE")
    assert cmp.make_sample(tmp_path / "again").to_json()["criteria"] == c.to_json()["criteria"]  # seeded


def test_the_cli(tmp_path, capsys):
    assert cmp.main(["sample", "--out", str(tmp_path / "s")]) == 0
    assert "report:" in capsys.readouterr().out
    s = tmp_path / "s"
    assert cmp.main(["run", "--protocol", str(s / "protocol.yaml"), "--aorlta", str(s / "aorlta.bundle.json"),
                     "--baseline", str(s / "baseline.bundle.json"), "--out", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "report.md").read_text() == (s / "report.md").read_text()
    assert cmp.main(["run", "--protocol", str(s / "protocol.yaml"), "--aorlta", str(s / "aorlta.bundle.json"),
                     "--baseline", str(s / "baseline.bundle.json"), "--out", str(tmp_path / "r2"),
                     "--protocol-commit", "HEAD"]) == 1  # the sample directory is no git repository
    assert "error:" in capsys.readouterr().err


def test_validation_points_need_no_fit_to_be_plotted(tmp_path):
    a, b = _pair()
    pts = tuple(ValidationPoint(n, 1.3 * n, 0.17 * n, 0.27 + 0.01 * n) for n in (2, 4, 6))
    a = replace(a, validation=Validation(0.27, 1600, pts))
    out = cmp.write_report(cmp.compare(_protocol(), a, b), tmp_path)
    assert "validation reward against GPU-hours" in out and out["validation reward against GPU-hours"].exists()
    text = out["report"].read_text()
    assert "not fitted (no curve rule in the protocol)" in text and "baseline: no validation points." in text
