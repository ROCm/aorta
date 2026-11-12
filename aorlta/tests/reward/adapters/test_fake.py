"""The fake check adapter: reproducible under a seed, and each injection does what it says."""

import dataclasses
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from aorlta.ids import sha256_text
from aorlta.reward.adapters.fake import (
    ALL_LEGS,
    FakeAdapterConfig,
    FakeCheckAdapter,
    FakeLegProfile,
    FakeOutcome,
)
from aorlta.schemas import LegJob, LegResult

TIME_FIELDS = ("t_queued", "t_start", "t_end")


def job(i: int, leg: str = "san.asan", attempt: int = 1, source: str | None = None, cand: str | None = None) -> LegJob:
    return LegJob(
        job_id=f"job-{i}-{leg}-{attempt}",
        leg=leg,
        candidate_id=cand or f"run/1/standin:k{i:04d}/0",
        control_id=None,
        source_sha256=sha256_text(source if source is not None else f"source-{i}"),
        task_sha256=sha256_text(f"task-{i % 5}"),
        attempt=attempt,
        worker_id=f"w{i % 3}",
    )


def no_times(result: LegResult) -> dict:
    d = dataclasses.asdict(result)
    for name in TIME_FIELDS:
        d.pop(name)
    return d


MIXED = FakeAdapterConfig(
    seed=11,
    default=FakeLegProfile(fail_rate=0.2, error_rate=0.1, stall_rate=0.05),
    per_leg={"san.asan": FakeLegProfile(finding_rate=0.3, fail_rate=0.1, error_rate=0.1, stall_rate=0.05)},
)


def jobs(n: int) -> list[LegJob]:
    return [job(i, leg) for i in range(n) for leg in ("gate.compile", "ref.native", "san.asan")]


def test_reproducible_under_a_seed_whatever_the_order_or_threads():
    js = jobs(200)
    a = [no_times(FakeCheckAdapter(MIXED).run(j)) for j in js]
    shuffled = list(js)
    random.Random(2).shuffle(shuffled)
    adapter = FakeCheckAdapter(MIXED)
    with ThreadPoolExecutor(16) as pool:
        by_id = dict(zip((j.job_id for j in shuffled), pool.map(adapter.run, shuffled)))
    assert [no_times(by_id[j.job_id]) for j in js] == a
    assert [FakeCheckAdapter(MIXED).plan(j) for j in js] == [FakeCheckAdapter(MIXED).plan(j) for j in js]


def test_the_seed_matters():
    js = jobs(200)
    a = [no_times(FakeCheckAdapter(MIXED).run(j)) for j in js]
    b = [no_times(FakeCheckAdapter(dataclasses.replace(MIXED, seed=12)).run(j)) for j in js]
    assert sum(x != y for x, y in zip(a, b)) > len(js) // 4


def test_the_verdict_follows_the_content_not_the_candidate_or_attempt():
    adapter = FakeCheckAdapter(FakeAdapterConfig(seed=3, per_leg={"san.asan": FakeLegProfile(finding_rate=0.5)}))
    verdicts = {}
    for i in range(200):
        results = [adapter.run(job(i, source="same-" + str(i % 20), cand=f"r/1/p{i}/{c}", attempt=a))
                   for c in range(2) for a in (1, 2)]
        assert len({(r.state, r.reason, str(r.findings)) for r in results}) == 1
        verdicts.setdefault(i % 20, set()).add(results[0].state)
    assert all(len(v) == 1 for v in verdicts.values())
    assert {next(iter(v)) for v in verdicts.values()} == {"clean", "finding"}


def test_infrastructure_events_are_per_attempt_so_a_retry_can_succeed():
    adapter = FakeCheckAdapter(FakeAdapterConfig(seed=5, default=FakeLegProfile(error_rate=0.5)))
    first = [adapter.run(job(i, "gate.compile")) for i in range(400)]
    retry = [adapter.run(job(i, "gate.compile", attempt=2)) for i in range(400)]
    recovered = sum(f.state == "unsettled" and r.state == "clean" for f, r in zip(first, retry))
    assert 50 < recovered < 150  # about a quarter: unsettled at attempt 1 (1/2) and clean at attempt 2 (1/2)
    assert all(r.reason == "adapter_error" for r in first + retry if r.state == "unsettled")


@pytest.mark.parametrize("leg, rate", [("san.asan", 0.3), ("san.consan", 0.0), ("san.waitcheck", 1.0)])
def test_finding_rate(leg, rate):
    adapter = FakeCheckAdapter(FakeAdapterConfig(seed=7, per_leg={leg: FakeLegProfile(finding_rate=rate)}))
    results = [adapter.run(job(i, leg)) for i in range(2000)]
    got = sum(r.state == "finding" for r in results) / len(results)
    assert got == rate if rate in (0.0, 1.0) else abs(got - rate) < 0.05
    for r in results:
        if r.state == "finding":
            assert r.findings and r.findings[0]["kind"] == "fake_finding" and r.repeats == 1
        else:
            assert r.state == "clean" and r.findings == []


def test_fail_reasons_default_by_leg_and_can_be_set():
    adapter = FakeCheckAdapter(FakeAdapterConfig(
        seed=8,
        default=FakeLegProfile(fail_rate=1.0),
        per_leg={"san.consan": FakeLegProfile(fail_rate=1.0, fail_reasons=("crash",))},
    ))
    assert {adapter.run(job(i, "gate.compile")).reason for i in range(50)} == {"compile_error"}
    assert {adapter.run(job(i, "gate.parse")).reason for i in range(50)} == {"parse"}
    assert {adapter.run(job(i, "ref.native")).reason for i in range(200)} == {"wrong_output", "crash", "timeout"}
    assert {adapter.run(job(i, "ref.edge")).reason for i in range(200)} == {"edge_mismatch", "fault", "timeout"}
    assert {adapter.run(job(i, "san.consan")).reason for i in range(50)} == {"crash"}
    assert all(adapter.run(job(i, "ref.native")).state == "fail" for i in range(20))


def test_only_sanitizer_legs_report_findings():
    with pytest.raises(ValueError, match="only sanitizer legs"):
        FakeAdapterConfig(default=FakeLegProfile(finding_rate=0.1))
    with pytest.raises(ValueError, match="only sanitizer legs"):
        FakeAdapterConfig(per_leg={"ref.edge": FakeLegProfile(finding_rate=0.1)})
    with pytest.raises(ValueError, match="only sanitizer legs"):
        FakeAdapterConfig(overrides={("c", "gate.parse"): FakeOutcome("finding")})
    FakeAdapterConfig(legs=("san.asan",), default=FakeLegProfile(finding_rate=0.1))


def test_a_stall_blocks_for_stall_s_and_is_never_clean():
    slept = []
    adapter = FakeCheckAdapter(FakeAdapterConfig(seed=1, default=FakeLegProfile(stall_rate=1.0), stall_s=1e4,
                                                 time_scale=0.001), sleep=slept.append)
    result = adapter.run(job(1, "ref.native"))
    assert (result.state, result.reason, result.repeats) == ("unsettled", "worker_stalled", 0)
    assert slept == [pytest.approx(10.0)]
    assert adapter.plan(job(1, "ref.native")).stalled


def test_close_interrupts_a_stall_with_the_default_sleep():
    adapter = FakeCheckAdapter(FakeAdapterConfig(default=FakeLegProfile(stall_rate=1.0), stall_s=1e4, time_scale=1.0))
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("r", adapter.run(job(1, "ref.native"))))
    t0 = time.monotonic()
    worker.start()
    deadline = time.monotonic() + 5
    while adapter.health()["stalls_in_progress"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert adapter.health()["stalls_in_progress"] == 1
    adapter.close()
    worker.join(5)
    assert not worker.is_alive() and time.monotonic() - t0 < 5
    assert out["r"].state == "unsettled" and adapter.health()["stalls_in_progress"] == 0
    assert adapter.run(job(2, "ref.native")).state == "unsettled"  # a closed adapter checks nothing


def test_latency_is_drawn_and_slept_on_the_time_scale():
    slept, ticks = [], iter(range(100, 1000))
    adapter = FakeCheckAdapter(
        FakeAdapterConfig(seed=2, default=FakeLegProfile(latency_s=0.31, jitter_s=10.7), time_scale=0.5),
        sleep=slept.append, clock=lambda: float(next(ticks)),
    )
    plans = [adapter.plan(job(i, "ref.native")) for i in range(100)]
    assert all(0.31 <= p.sleep_s < 11.01 for p in plans) and len({p.sleep_s for p in plans}) == 100
    result = adapter.run(job(0, "ref.native"))
    assert slept == [pytest.approx(plans[0].sleep_s * 0.5)]
    assert (result.t_start, result.t_end, result.t_queued) == (100.0, 101.0, 100.0)


def test_overrides_force_outcomes_and_attempt_specific_ones_win():
    cand = "run/1/standin:k0001/0"
    adapter = FakeCheckAdapter(FakeAdapterConfig(seed=1, overrides={
        (cand, "san.asan"): FakeOutcome("finding"),
        (cand, "san.asan", 1): FakeOutcome("stall"),
        (cand, "ref.edge"): FakeOutcome("fail", "edge_mismatch"),
        (cand, "gate.compile"): FakeOutcome("error", "worker_died"),
    }))
    assert adapter.run(job(1, "san.asan", cand=cand)).reason == "worker_stalled"
    assert adapter.run(job(1, "san.asan", cand=cand, attempt=2)).state == "finding"
    edge = adapter.run(job(1, "ref.edge", cand=cand))
    assert (edge.state, edge.reason) == ("fail", "edge_mismatch")
    compile_ = adapter.run(job(1, "gate.compile", cand=cand))
    assert (compile_.state, compile_.reason) == ("unsettled", "worker_died")
    assert adapter.run(job(2, "gate.compile")).state == "clean"  # other candidates are untouched


@pytest.mark.parametrize(
    "bad",
    [
        "not a job",
        None,
        LegJob("j", "san.asan", "c", "ctl", "a" * 64, "b" * 64),  # both ids
        LegJob("j", "san.asan", None, None, "a" * 64, "b" * 64),  # neither
        LegJob("j", "san.unknown", "c", None, "a" * 64, "b" * 64),
        LegJob("j", "san.asan", "c", None, "short", "b" * 64),
        LegJob("j", "san.asan", "c", None, "a" * 64, "b" * 64, attempt=0),
        LegJob("j", "san.asan", "c", None, "a" * 64, "b" * 64, leg_settings={"x": float("nan")}),
    ],
)
def test_run_never_raises_and_never_returns_clean_for_a_bad_job(bad):
    result = FakeCheckAdapter(FakeAdapterConfig()).run(bad)
    assert isinstance(result, LegResult)
    assert (result.state, result.reason, result.repeats) == ("unsettled", "adapter_error", 0)


def test_controls_are_clean_unless_dirty_is_injected():
    clean = FakeCheckAdapter(FakeAdapterConfig(seed=1))
    specs = clean.controls()
    assert [s.leg for s in specs] == list(ALL_LEGS) and len({s.control_id for s in specs}) == len(specs)

    def control_job(spec, n):
        return LegJob(f"{spec.control_id}#{n}", spec.leg, None, spec.control_id, spec.source_sha256, spec.task_sha256)

    assert {clean.run(control_job(s, n)).state for s in specs for n in range(5)} == {"clean"}
    dirty = FakeCheckAdapter(FakeAdapterConfig(seed=1, control_dirty_rate=1.0))
    for s in specs:
        r = dirty.run(control_job(s, 0))
        assert r.reason == "control_dirty" and r.state == ("finding" if s.leg.startswith("san.") else "fail")
    half = FakeCheckAdapter(FakeAdapterConfig(seed=1, control_dirty_rate=0.5))
    states = [half.run(control_job(specs[0], n)).state for n in range(400)]
    assert 150 < states.count("fail") < 250  # each control run is its own draw


def test_result_fields_and_health():
    adapter = FakeCheckAdapter(FakeAdapterConfig(name="fake.sim", legs=("ref.native", "ref.edge"), resource="sim_cpu",
                                                 repeats=3))
    r = adapter.run(job(4, "ref.edge", attempt=2))
    assert (r.leg, r.repeats, r.attempts, r.worker_id, r.cache_hit, r.bracket) == ("ref.edge", 3, 2, "w1", False, None)
    assert adapter.health() == {"name": "fake.sim", "version": "1", "resource": "sim_cpu", "legs": ["ref.native", "ref.edge"],
                                "closed": False, "runs": 1, "stalls_in_progress": 0}
    assert (adapter.name, adapter.version, adapter.legs, adapter.resource) == ("fake.sim", "1", ("ref.native", "ref.edge"), "sim_cpu")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "sim"},
        {"legs": ()},
        {"legs": ("gate.parse", "gate.parse")},
        {"legs": ("other.x",)},
        {"resource": "cloud"},
        {"per_leg": {"san.nope": FakeLegProfile()}},
        {"repeats": 0},
        {"control_dirty_rate": 2.0},
        {"overrides": {("c",): FakeOutcome("clean")}},
        {"overrides": {("c", "san.asan", 0): FakeOutcome("clean")}},
    ],
)
def test_bad_adapter_configs_are_refused(kwargs):
    with pytest.raises(ValueError):
        FakeAdapterConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"fail_rate": 0.7, "finding_rate": 0.4}, {"error_rate": 0.6, "stall_rate": 0.6},
                                    {"latency_s": -1.0}, {"fail_reasons": ("",)}, {"error_rate": -0.1}])
def test_bad_leg_profiles_are_refused(kwargs):
    with pytest.raises(ValueError):
        FakeLegProfile(**kwargs)
