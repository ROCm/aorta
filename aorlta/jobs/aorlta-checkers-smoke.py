"""The remote checkers' GPU smoke test (`SMOKE=checkers sbatch jobs/aorlta-run.sbatch`): drive each checker server (every
adapter on every GPU it names) the way the reward service does, through `RemoteAdapter`, on aorta's own fixtures:

  - each adapter's trusted controls, which must come back clean;
  - the kernel-only fixtures `tiny_vecadd.hip` (no LDS) and `lds_reduce.hip` (LDS and barriers), which get the stand-in
    launch harness, as a model's candidate does;
  - the ASan fixture with its own documented overflow switch (`#define ASAN_REPRO_OVERFLOW 1`), which ASan must catch.

No new kernel or bug is written. The fixtures are read from aorta's checkout, `$AORLTA_AORTA_ROOT` (default
/opt/aorta). Prints one line per job and writes `<out>/smoke-results.json`; exits 0 only if every verdict is the
expected one.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from aorlta.config import load_config
from aorlta.ids import sha256_text
from aorlta.reward.remote import RemoteAdapter, servers
from aorlta.schemas import LegJob

FIXTURES = Path(os.environ.get("AORLTA_AORTA_ROOT", "/opt/aorta")) / "recipes" / "sanitizers" / "fixtures"
# fixture -> {leg: expected state}; ConSan on a kernel with no LDS, barrier, atomic or fence site is NOT_CHECKED,
# which stays unsettled
EXPECTED = {
    "tiny_vecadd": {"san.asan": "clean", "san.consan": "unsettled"},
    "lds_reduce": {"san.asan": "clean", "san.consan": "clean"},
    "asan_overflow": {"san.asan": "finding"},
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


def main(config: str, out_dir: str) -> int:
    cfg = load_config(config)
    out = Path(out_dir)
    remote = cfg.reward.remote
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
        for label, want, job in jobs:
            Path(job.output_dir).mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            r = a.run(job)
            dt = time.monotonic() - t0
            want = "clean" if want is None else want  # a control must be clean
            good = r.state == want
            ok &= good
            rows.append({"adapter": name, "gpu": gpu, "job": label, "leg": leg, "expected": want, "state": r.state,
                         "reason": r.reason, "findings": [f.get("kind") for f in r.findings], "seconds": round(dt, 2)})
            print(f"{'OK  ' if good else 'BAD '} {where:27s} {label:40s} {r.state:9s} want {want:9s} "
                  f"{(r.reason or '')[:90]:90s} {dt:6.1f} s", flush=True)
        print(json.dumps({"adapter": name, "gpu": gpu, "health": a.health()}), flush=True)
        a.close()
    (out / "smoke-results.json").write_text(json.dumps({"ok": ok, "rows": rows}, indent=1))
    print("SMOKE", "PASSED" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
