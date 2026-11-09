"""C2.T11: the stand-in end to end on a GPU, through the reward service (design Section 6.2).

aorta's four sanitizer fixtures go in as candidates' final answers, through parse -> hipcc ->
rj_waitcheck -> ConSan (RocJITsu DBI hook) -> ASan (batched, with in-band controls), and are scored by
standin.v11. Pass: the clean fixtures' sanitizer legs are clean and the racy and heap-overflow
fixtures' are findings.

GPU only, inside the Slurm job `jobs/aorlta-reward-standin.sbatch` (which sets AORLTA_STANDIN_GPU=1).
Anywhere else it is skipped, so nothing here touches a GPU outside that job. The adapters take the site paths
from the same environment variables as the CPU stand-in tests (tests/reward/adapters/test_standin_cpu.py).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import pytest

from aorlta.config import RewardSection
from aorlta.reward.clock import Clock
from aorlta.reward.factories import standin_factories
from aorlta.reward.profiles import load_profile
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService
from aorlta.schemas import to_dict
from aorlta.spans import SpanRecorder
from reward_helpers import cand

pytestmark = pytest.mark.skipif(
    os.environ.get("AORLTA_STANDIN_GPU") != "1" or not os.environ.get("SLURM_JOB_ID"),
    reason="GPU stand-in test: runs only inside jobs/aorlta-reward-standin.sbatch")

AORTA_ROOT = Path(os.environ.get("AORLTA_AORTA_ROOT", "/opt/aorta"))
RJ = os.environ.get("AORLTA_RJ_WAITCHECK", "/opt/rocjitsu/bin/rj_waitcheck")
SHIM = os.environ.get("AORLTA_LIBSTDCXX_SHIM", "/opt/libstdcxx-shim")
FIXTURES = AORTA_ROOT / "recipes" / "sanitizers" / "fixtures" / "repro"


def site_profile():
    """standin-gpu with its adapters on this site's paths: aorta's src and control fixtures under AORLTA_AORTA_ROOT,
    rj_waitcheck and its libstdc++, and the ConSan hook of the same rocjitsu install (lib/ beside its bin/)."""
    profile = load_profile("standin-gpu")
    aorta = str(AORTA_ROOT / "src")
    site = {"standin.rj_waitcheck": {"binary": RJ, "ld_library_path": SHIM, "aorta_src": aorta},
            "standin.aorta_consan": {"hook": str(Path(RJ).parent.parent / "lib" / "librocjitsu_dbi_hooks.so"),
                                     "aorta_src": aorta, "control_source": str(FIXTURES / "consan_lds_race.hip")},
            "standin.aorta_asan": {"aorta_src": aorta, "control_source": str(FIXTURES / "asan_heap_overflow.hip")}}
    return dataclasses.replace(profile, adapters={name: {**cfg, **site.get(name, {})}
                                                  for name, cfg in profile.adapters.items()})


# fixture -> (source file, prefix that selects the fixture's own documented variant, leg, expected state)
CASES = {
    "consan_clean": ("consan_lds_race.hip", "", "san.consan", "clean"),
    "consan_racy": ("consan_lds_race_2wave.hip", "", "san.consan", "finding"),
    "asan_clean": ("asan_heap_overflow.hip", "", "san.asan", "clean"),
    "asan_heap_overflow": ("asan_heap_overflow.hip", "#define ASAN_REPRO_OVERFLOW 1\n", "san.asan", "finding"),
}


def test_the_four_aorta_fixtures_score_as_expected():
    out = Path(os.environ.get("AORLTA_T11_OUT", str(Path.home() / "aorlta-t11")))
    out.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    spans = SpanRecorder(monotonic=clock.now, wall=clock.wall)
    profile = site_profile()
    reward = RewardSection(profile="fake", reward_fn="standin.v11")  # timeouts and controls at the defaults
    timeouts = {"gate": reward.timeouts.gate, "compile": reward.timeouts.compile, "sim_run": reward.timeouts.sim_run,
                "asan_request": reward.timeouts.asan_request}
    svc = RewardService(reward, profile, StandinV11(), standin_factories(profile, clock, timeouts=timeouts),
                        cache_dir=out / "cache", work_dir=out / "work", clock=clock, spans=spans)
    try:
        futures = {}
        for i, (name, (src, prefix, _, _)) in enumerate(CASES.items()):
            text = (FIXTURES / src).read_text(encoding="utf-8")
            response = f"The corrected kernel:\n```cpp\n{prefix}{text}```\n"
            futures[name] = svc.submit(cand(i, text=response, tokens=900, max_new=4096, profile="standin-gpu"))
        t0 = time.monotonic()
        report = svc.drain(clock.now() + 1800)
        results = {name: f.result(timeout=10) for name, f in futures.items()}
        verdicts = {}
        for name, r in results.items():
            verdicts[name] = {
                "reward": r.reward, "status": r.status, "reason": r.reason, "parts": r.parts,
                "legs": {leg: {"state": x.state, "reason": x.reason, "findings": [f.get("kind") for f in x.findings],
                               "attempts": x.attempts, "worker": x.worker_id, "bracket": x.bracket}
                         for leg, x in r.legs.items() if x.state != "not_run" or leg.startswith("san.")},
            }
        evidence = {
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "node": os.uname().nodename,
            "seconds": time.monotonic() - t0, "drain": to_dict(report) if hasattr(report, "__dataclass_fields__") else str(report),
            "verdicts": verdicts, "results": svc.results_json(),
            "controls": [vars(rec) for rec in svc.ledger_controls.snapshot()],
            "events": svc.events_log, "spans": [s.to_json() for s in spans.spans()],
            "stats": vars(svc.stats()),
        }
        (out / "t11-evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str))
        for name, (_, _, leg, expected) in CASES.items():
            assert results[name].legs[leg].state == expected, (name, verdicts[name])
        # the clean ConSan fixture is clean under every sanitizer, so it earns the clean reward
        assert results["consan_clean"].reward == 1.0 + -0.05 * (900 / 4096)
        # the racy and overflow fixtures are penalised as findings (Appendix B rule 6)
        for name in ("consan_racy", "asan_heap_overflow"):
            assert results[name].parts["base"] < 0 and results[name].status == "scored"
    finally:
        svc.close()
