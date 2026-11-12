"""The fake-checker report: the service under injected latencies, timeouts, stalls,
dirty controls and ASan batching, with the doc's default thresholds, on a scaled clock.

    python -m aorlta.reward.fake_report OUT_DIR [--scale 0.01]

Writes OUT_DIR/fake-checker-report.json and .md. Every number is measured from one run of the real
service with fake adapters; nothing touches a GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from ..config import RewardSection
from ..ids import candidate_id, sha256_text
from ..schemas import Candidate, TaskRef
from ..spans import SpanRecorder
from .adapters.base import SequentialBatch
from .adapters.fake import FakeAdapterConfig, FakeCheckAdapter, FakeLegProfile, FakeOutcome
from .clock import Clock
from .profiles import load_profile
from .rewardfn.standin_v11 import StandinV11
from .service import RewardService

SIM = ("ref.native", "ref.edge", "san.consan", "san.waitcheck")
GATES0 = ("gate.parse", "gate.contract", "gate.static", "gate.reference", "gate.no_artifact", "gate.spec_invalid",
          "gate.static_reject", "gate.no_change_on_failing_parent")


def _cand(i: int, run: str = "fake-report", text: str | None = None) -> Candidate:
    pid = f"standin:r{i}"
    return Candidate(candidate_id(run, 1, pid, 0), run, 1, pid, 0, f"{run}:v0", text or f"```cpp\n// {i}\n```", 1024,
                     4096, TaskRef("c", None, None, "frozen", sha256_text("task")), "fake", None)


def _service(scale: float, cfg: FakeAdapterConfig, reward: RewardSection, work: Path | None = None):
    clock = Clock(scale)
    spans = SpanRecorder(monotonic=clock.now, wall=clock.wall)
    cfg = dataclasses.replace(cfg, time_scale=scale)
    asan_cfg = dataclasses.replace(cfg, legs=("san.asan",),
                                   per_leg={k: v for k, v in cfg.per_leg.items() if k == "san.asan"},
                                   overrides={k: v for k, v in cfg.overrides.items() if k[1] == "san.asan"})
    svc = RewardService(reward, load_profile("fake"), StandinV11(),
                        {"fake": lambda w: FakeCheckAdapter(cfg, clock=clock.wall),
                         "fake-asan": lambda w: SequentialBatch(FakeCheckAdapter(asan_cfg, clock=clock.wall))},
                        cache_dir=None if work is None else work / "cache", work_dir=work, clock=clock, spans=spans)
    return svc, clock, spans


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def simulator_like(scale: float, n: int = 256, control_dirty_rate: float = 0.0, error_rate: float = 0.0) -> dict:
    lat = {g: FakeLegProfile(latency_s=0.2, jitter_s=0.5) for g in GATES0}
    lat["gate.compile"] = FakeLegProfile(latency_s=0.5, jitter_s=0.2)
    for name in SIM:
        lat[name] = FakeLegProfile(latency_s=4.0, jitter_s=14.0, error_rate=error_rate)
    lat["san.asan"] = FakeLegProfile(latency_s=7.0, jitter_s=3.0)
    cfg = FakeAdapterConfig(seed=11, per_leg=lat, control_dirty_rate=control_dirty_rate)
    reward = RewardSection(profile="fake", reward_fn="standin.v11", sim_workers=32)
    svc, clock, spans = _service(scale, cfg, reward)
    try:
        t0 = clock.now()
        futures = []
        for i in range(n):  # chains end spread over 300 s of generation, then the generation barrier
            clock.sleep(t0 + 300.0 * i / n - clock.now())
            futures.append(svc.submit(_cand(i)))
        report = svc.drain(clock.now() + 4 * 3600)
        results = [f.result() for f in futures]
        wall = clock.now() - t0
        by_name: dict[str, list[float]] = {}
        for s in spans.spans():
            by_name.setdefault(s.name, []).append(s.duration)
        reqs = [e for e in svc.events_log if e["event"] == "asan_request"]
        waits = [w for e in reqs for w in e["waits"]]
        stats = svc.stats()
        # a second pass over the same content: every leg is a cache hit
        runs_before = sum(sum(v.values()) for v in stats.leg_runs.values())
        svc.reopen()  # the next iteration
        again = [svc.submit(_cand(i + n, text=f"```cpp\n// {i}\n```")) for i in range(n)]
        again_results = [f.result(timeout=600) for f in again]
        runs_after = sum(sum(v.values()) for v in svc.stats().leg_runs.values())
        return {
            "candidates": n, "simulated_wall_s": round(wall, 1), "resolved": report.resolved, "scored": report.scored,
            "unscored_by_reason": report.unscored_by_reason,
            "span_seconds": {k: {"n": len(v), "median": round(statistics.median(v), 2), "p90": round(_q(v, 0.9), 2)}
                             for k, v in sorted(by_name.items()) if k in ("gate.parse", "gate.compile", "ref.native",
                                                                          "san.consan", "asan.queue", "asan.run", "control")},
            "asan": {"requests": len(reqs), "request_sizes": dict(sorted(Counter(s for e in reqs for s in e["request_sizes"]).items())),
                     "max_wait_s": round(max(waits), 2) if waits else None, "median_wait_s": round(statistics.median(waits), 2) if waits else None},
            "controls": stats.controls, "retries": stats.retries, "restarts": stats.worker_restarts,
            "second_pass": {"resolved": len(again_results), "extra_leg_runs": runs_after - runs_before,
                            "rewards_equal": [r.reward for r in again_results] == [r.reward for r in results],
                            "reward_cache_hits": sum(bool(r.timings.get("reward_cache_hit")) for r in again_results)},
        }
    finally:
        svc.close()


def batching_while_streaming(scale: float, n: int = 120) -> dict:
    """S2 keeps up with generation here, so ASan sees packages while chains are still ending."""
    lat = {name: FakeLegProfile(latency_s=1.0, jitter_s=4.0) for name in SIM}
    lat["san.asan"] = FakeLegProfile(latency_s=7.0, jitter_s=3.0)
    cfg = FakeAdapterConfig(seed=5, per_leg=lat)
    svc, clock, spans = _service(scale, cfg, RewardSection(profile="fake", reward_fn="standin.v11", sim_workers=32))
    try:
        t0 = clock.now()
        futures = []
        for i in range(n):  # one chain ends every 2.5 s for 300 s
            clock.sleep(t0 + 300.0 * i / n - clock.now())
            futures.append(svc.submit(_cand(i)))
        gen_end = clock.now()
        svc.drain(clock.now() + 3600)
        [f.result() for f in futures]
        reqs = [e for e in svc.events_log if e["event"] == "asan_request"]
        before = [e for e in reqs if not e["generation_ended"]]
        waits = [w for e in before for w in e["waits"]]
        trig = Counter("full" if len(e["candidates"]) == 14 else "max_wait" for e in before)
        return {"candidates": n, "requests_during_generation": len(before), "requests_after": len(reqs) - len(before),
                "sizes": dict(sorted(Counter(s for e in reqs for s in e["request_sizes"]).items())),
                "triggered_by": dict(trig), "max_wait_s": round(max(waits), 2) if waits else None,
                "max_size": max(s for e in reqs for s in e["request_sizes"]),
                "last_request_after_generation_end_s": round(reqs[-1]["t"] - gen_end, 2) if reqs else None}
    finally:
        svc.close()


def timeouts(scale: float) -> list[dict]:
    out = []
    reward = RewardSection(profile="fake", reward_fn="standin.v11", sim_workers=2)
    for leg in load_profile("fake").legs:
        c = _cand(0)
        cfg = FakeAdapterConfig(overrides={(c.candidate_id, leg): FakeOutcome("clean", latency_s=1e5)})
        svc, clock, spans = _service(scale, cfg, reward)
        try:
            r = svc.submit(c).result(timeout=120)
            restart = next(e for e in svc.events_log if e["event"] == "restart")
            retries = [e for e in svc.events_log if e["event"] == "retry"]
            out.append({"leg": leg, "state": r.legs[leg].state, "reason": r.legs[leg].reason, "reward": r.reward,
                        "unscored_reason": r.reason, "detected_after_s": round(restart["t"] - restart["item_started"], 1),
                        "restart": restart["reason"], "retried": any(e["leg"] == leg for e in retries),
                        "siblings_retried": sorted({f"{e['leg']} ({e['reason']})" for e in retries if e["leg"] != leg})})
        finally:
            svc.close()
    return out


def stalls(scale: float) -> list[dict]:
    out = []
    reward = RewardSection(profile="fake", reward_fn="standin.v11", sim_workers=2)
    for leg in ("gate.static", "gate.compile", "ref.native", "san.consan", "san.asan"):
        c = _cand(0)
        cfg = FakeAdapterConfig(stall_s=1e4, overrides={(c.candidate_id, leg, 1): FakeOutcome("stall")})
        svc, clock, spans = _service(scale, cfg, reward)
        try:
            r = svc.submit(c).result(timeout=120)
            restart = next(e for e in svc.events_log if e["event"] == "restart")
            retry = [e for e in svc.events_log if e["event"] == "retry"]
            out.append({"leg": leg, "restart": restart["reason"],
                        "restarted_after_s": round(restart["t"] - restart["item_started"], 1),
                        "retried": [(e["reason"], e["attempt"]) for e in retry], "final_attempts": r.legs[leg].attempts,
                        "final_state": r.legs[leg].state, "reward": r.reward})
        finally:
            svc.close()
    return out


def dirty_control(scale: float) -> dict:
    reward = RewardSection(profile="fake", reward_fn="standin.v11", sim_workers=2)
    reward = dataclasses.replace(reward, controls=dataclasses.replace(reward.controls, every_jobs=8))
    cfg = FakeAdapterConfig(overrides={("control:sim/1:3:ref.native", "ref.native"): FakeOutcome("fail", "control_dirty")})
    svc, clock, spans = _service(scale, cfg, reward)
    try:
        results = [f.result(timeout=300) for f in [svc.submit(_cand(i)) for i in range(32)]]
        dirty = [e for e in svc.events_log if e["event"] == "control_dirty"]
        retries = [e for e in svc.events_log if e["event"] == "retry"]
        ledger = [r for r in svc.ledger_controls.snapshot() if r.worker_id == "sim/1"][:5]
        return {"dirty_control": dirty[0]["control"] if dirty else None,
                "legs_in_its_bracket": len(dirty[0]["closes"]) if dirty else 0,
                "retried": len(retries), "retried_on": sorted({x.worker_id for r in results for n, x in r.legs.items()
                                                              if x.attempts == 2}),
                "sim1_controls": [(r.control_run_id, r.clean, r.brackets) for r in ledger],
                "scored": sum(r.reward is not None for r in results), "candidates": len(results)}
    finally:
        svc.close()


def markdown(rep: dict) -> str:
    L = ["# Fake-checker report", "",
         f"Generated by `python -m aorlta.reward.fake_report` at clock scale {rep['scale']} (real seconds per simulated "
         "second). Thresholds are the doc's defaults: timeouts gate 60 s, compile 120 s, simulator run 120 s, ASan request "
         "900 s; `stall_s` 300 s; ASan `max_packages` 16, `max_wait_s` 20 s; controls every 32 jobs or 300 s.", ""]
    for key, title in (("simulator_like", "Simulator-backed checker latencies, 256 candidates"),
                       ("simulator_like_dirty", "The same with 5% of control jobs dirty (18.5% of 4-leg simulator control "
                                            "runs, 9.8% of ASan request brackets) and 2% adapter errors on simulator legs")):
        x = rep[key]
        L += [f"## {title}", "",
              f"- simulated wall {x['simulated_wall_s']} s; resolved {x['resolved']}, scored {x['scored']}; "
              f"unscored by reason {x['unscored_by_reason'] or '{}'}",
              f"- ASan: {x['asan']['requests']} requests, sizes {x['asan']['request_sizes']}, wait median "
              f"{x['asan']['median_wait_s']} s, max {x['asan']['max_wait_s']} s",
              f"- controls: {x['controls']}; retries {x['retries'] or '{}'}; restarts {x['restarts'] or '{}'}",
              f"- second pass over the same content (the next iteration, after `reopen()`): {x['second_pass']['resolved']} "
              f"resolved, {x['second_pass']['reward_cache_hits']} from the reward cache, "
              f"{x['second_pass']['extra_leg_runs']} extra leg runs; same rewards: {x['second_pass']['rewards_equal']}", "",
              "| span | n | median s | p90 s |", "|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['median']} | {v['p90']} |" for k, v in x["span_seconds"].items()]
        L.append("")
    b = rep["batching"]
    L += ["## ASan batching while chains stream (120 chains over 300 s, S2 keeping up; in-band controls, so 14 "
          "candidates + 2 controls fill a request)", "",
          f"- {b['requests_during_generation']} requests during generation, triggered by {b['triggered_by']}; "
          f"{b['requests_after']} after the generation barrier (the final partial flush)",
          f"- request sizes (packages, controls included) {b['sizes']}; largest {b['max_size']} (limit 16)",
          f"- longest wait of a package queued during generation {b['max_wait_s']} s (limit max_wait_s 20 s + 1 s)",
          f"- last request sent {b['last_request_after_generation_end_s']} s after generation ended", ""]
    L += ["## Infrastructure timeouts, one leg at a time (the leg's job runs 10^5 s on a live worker)", "",
          "Siblings are legs that had finished on the restarted worker and were still waiting for their "
          "after-control: their bracket is lost, so they are retried once elsewhere.", "",
          "| leg | leg state | reason | reward | unscored reason | detected after s | restart | leg retried | siblings retried |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {t['leg']} | {t['state']} | {t['reason']} | {t['reward']} | {t['unscored_reason']} | {t['detected_after_s']} | "
          f"{t['restart']} | {t['retried']} | {', '.join(t['siblings_retried']) or '-'} |" for t in rep["timeouts"]]
    L += ["", "## Stalls (the worker goes silent for 10^4 s on attempt 1)", "",
          "| leg | restart | restarted after s | retries | final attempts | final state | reward |", "|---|---|---|---|---|---|---|"]
    L += [f"| {s['leg']} | {s['restart']} | {s['restarted_after_s']} | {s['retried']} | {s['final_attempts']} | "
          f"{s['final_state']} | {s['reward']} |" for s in rep["stalls"]]
    d = rep["dirty_control"]
    L += ["", "## One dirty after-control (32 candidates, 2 simulator workers, controls every 8 jobs)", "",
          f"- dirty control: `{d['dirty_control']}`, closing {d['legs_in_its_bracket']} legs; {d['retried']} retries, "
          f"all on {d['retried_on']}; scored {d['scored']} of {d['candidates']}",
          f"- sim/1's first controls (run, clean, legs it closed): {d['sim1_controls']}", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--scale", type=float, default=0.01)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    rep = {"scale": args.scale,
           "simulator_like": simulator_like(args.scale),
           "simulator_like_dirty": simulator_like(args.scale, control_dirty_rate=0.05, error_rate=0.02),
           "batching": batching_while_streaming(args.scale * 5),
           "timeouts": timeouts(args.scale / 5),
           "stalls": stalls(args.scale),
           "dirty_control": dirty_control(args.scale),
           "real_seconds": None}
    rep["real_seconds"] = round(time.time() - t, 1)
    (args.out / "fake-checker-report.json").write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    (args.out / "fake-checker-report.md").write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
