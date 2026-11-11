"""The remote checkers' GPU smoke test (`SMOKE=checkers sbatch jobs/aorlta-run.sbatch`): drive each checker server (every
adapter on every GPU it names) the way the reward service does, through `RemoteAdapter`, on aorta's own fixtures:

  - each adapter's trusted controls, which must come back clean;
  - the kernel-only fixtures `tiny_vecadd.hip` (no LDS) and `lds_reduce.hip` (LDS and barriers), which get the stand-in
    launch harness, as a model's candidate does;
  - the ASan fixture with its own documented overflow switch (`#define ASAN_REPRO_OVERFLOW 1`), which ASan must catch.

No new kernel or bug is written. The fixtures are read from aorta's checkout, `$AORLTA_AORTA_ROOT` (default
/opt/aorta). Prints one line per job and writes `<out>/smoke-results.json`; exits 0 only if every verdict is the
expected one. A job gets the longest of the reward service's pool timeouts, as its watchdog would give it: a checker
that has not answered by then fails the smoke test, and its server's other jobs are not run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import time
from pathlib import Path

from aorlta.config import load_config
from aorlta.ids import sha256_text
from aorlta.reward.remote import RemoteAdapter, servers
from aorlta.schemas import LegJob

FIXTURES = Path(os.environ.get("AORLTA_AORTA_ROOT", "/opt/aorta")) / "recipes" / "sanitizers" / "fixtures"
# fixture -> {leg: (expected state, the start of the expected reason or None)}. ConSan on a kernel with no LDS,
# barrier, atomic or fence site is NOT_CHECKED as not applicable, which stays unsettled; another unsettled reason
# (an adapter error, a harness that did not build, a timeout) fails the smoke test.
NOT_APPLICABLE = "checker_not_settled:not_checked/not_checked: consan_not_applicable"
EXPECTED = {
    "tiny_vecadd": {"san.asan": ("clean", None), "san.consan": ("unsettled", NOT_APPLICABLE)},
    "lds_reduce": {"san.asan": ("clean", None), "san.consan": ("clean", None)},
    "asan_overflow": {"san.asan": ("finding", None)},
}


def sources(out: Path) -> dict[str, Path]:
    d = out / "smoke-sources"
    d.mkdir(parents=True, exist_ok=True)
    srcs = {}
    for name in ("tiny_vecadd", "lds_reduce"):
        p = d / f"{name}.hip"
        p.write_text((FIXTURES / "kernels" / f"{name}.hip").read_text())
        srcs[name] = p
    p = d / "asan_overflow.hip"  # the fixture's own build switch, as its -DASAN_REPRO_OVERFLOW=1 line documents
    p.write_text("#define ASAN_REPRO_OVERFLOW 1\n" + (FIXTURES / "repro" / "asan_heap_overflow.hip").read_text())
    srcs["asan_overflow"] = p
    return srcs


def job_limit(cfg) -> float:
    """The longest any job may run under the reward service's watchdog: its longest pool timeout."""
    return max(dataclasses.asdict(cfg.reward.timeouts).values())


def run_bounded(adapter, job: LegJob, limit_s: float):
    """`adapter.run(job)`, or None if it has not returned in `limit_s`. RemoteAdapter's socket has no timeout, since
    the reward service's watchdog bounds its jobs, and there is no service here. On expiry the adapter is closed, which
    ends the job on the checker."""
    box: dict = {}

    def work() -> None:
        try:
            box["result"] = adapter.run(job)
        except Exception as exc:  # noqa: BLE001 - reported as the job's failure
            box["error"] = exc

    t = threading.Thread(target=work, name=f"smoke-{job.job_id}"[:60], daemon=True)
    t.start()
    t.join(limit_s)
    if t.is_alive():
        adapter.close()
        return None
    if "error" in box:
        raise box["error"]
    return box["result"]


def main(config: str, out_dir: str) -> int:
    cfg = load_config(config)
    out = Path(out_dir)
    remote = cfg.reward.remote
    limit = job_limit(cfg)
    srcs = sources(out)
    rows, ok = [], True
    for name, gpu, sock in servers(remote):
        a = RemoteAdapter(sock, name, f"smoke/{name}")
        leg = a.legs[0]
        where = f"{name}@gpu{gpu}"
        jobs = []
        for c in a.controls():
            jobs.append(("control:" + c.control_id, None, LegJob(
                job_id=f"smoke:{c.control_id}", leg=c.leg, candidate_id=None, control_id=c.control_id,
                source_sha256=c.source_sha256, task_sha256=c.task_sha256, worker_id=f"smoke/{name}", inputs=dict(c.inputs),
                output_dir=str(out / "smoke" / where / "control"))))
        for fixture, src in srcs.items():
            want = EXPECTED[fixture].get(leg)
            if want is None:
                continue
            cid = f"smoke/1/standin:{fixture}/0"
            jobs.append((fixture, want, LegJob(
                job_id=f"{cid}#{leg}#a1", leg=leg, candidate_id=cid, control_id=None,
                source_sha256=sha256_text(src.read_text()), task_sha256=sha256_text(fixture), worker_id=f"smoke/{name}",
                inputs={"source": str(src)}, output_dir=str(out / "smoke" / where / fixture))))
        hung = False
        for label, want, job in jobs:
            Path(job.output_dir).mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            r = run_bounded(a, job, limit)
            dt = time.monotonic() - t0
            want, why = ("clean", None) if want is None else want  # a control must be clean
            if r is None:
                ok, hung = False, True
                rows.append({"adapter": name, "gpu": gpu, "job": label, "leg": leg, "expected": want, "state": None,
                             "reason": f"no answer in {limit:g} s", "findings": [], "seconds": round(dt, 2)})
                print(f"BAD  {where:27s} {label:40s} {'-':9s} want {want:9s} no answer in {limit:g} s", flush=True)
                break  # the adapter is closed: its server's other jobs cannot run
            good = r.state == want and (why is None or (r.reason or "").startswith(why))
            ok &= good
            rows.append({"adapter": name, "gpu": gpu, "job": label, "leg": leg, "expected": want, "state": r.state,
                         "reason": r.reason, "findings": [f.get("kind") for f in r.findings], "seconds": round(dt, 2)})
            print(f"{'OK  ' if good else 'BAD '} {where:27s} {label:40s} {r.state:9s} want {want:9s} "
                  f"{(r.reason or '')[:90]:90s} {dt:6.1f} s", flush=True)
        if not hung:
            print(json.dumps({"adapter": name, "gpu": gpu, "health": a.health()}), flush=True)
            a.close()
    (out / "smoke-results.json").write_text(json.dumps({"ok": ok, "rows": rows}, indent=1))
    print("SMOKE", "PASSED" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
