"""`standin.aorta_consan` and `standin.aorta_asan`: ConSan through the RocJITsu DBI hook and device
ASan, on a real GPU, by aorta's own backends (design Section 6.2). These run the candidate
as a program. aorta's sanitizer fixtures have their own `main`; a kernel-only stand-in candidate gets
the stand-in launch harness (`standin_harness`). They are stand-ins: a production checker may run ConSan
elsewhere, in a CPU simulator for example.

Each job builds the candidate's source (or its harness) into an executable with hipcc (an explicit
offload arch, plus `:xnack+ -fsanitize=address -shared-libsan` for ASan, as the fixtures' own build
lines do), then calls `run_consan(strict)` / `run_asan` on it with one selected kernel, and maps the verdict:
PASS -> clean; FAIL with findings -> finding; ERROR, NOT_CHECKED, TIMED_OUT -> unsettled. An ASan
PASS on a node whose health canary is NOT_VERIFIED is already NOT_CHECKED in aorta, so unsettled.
For ConSan, an unhealthy waitcheck preflight (ERROR or NOT_CHECKED) also leaves the leg unsettled,
since aorta's report would not be clean either. A harness that does not build is unsettled.

Everything a job builds, runs and leaves as evidence is in `<run>`, a directory of its own under the leg's output
directory (`base.run_dir`): that directory is shared by every job with identical content, and by a retry that can
overlap the retired worker's run of the same job.

A kernel-only source is checked one kernel at a time (`standin_harness.plan`: each kernel instance gets its own
harness program, in `<run>/k<i>/`, and its own aorta run, since aorta's ConSan and ASan each take one selected
kernel). The leg is a finding if any kernel's run found something, clean only if every kernel's run is clean, and
unsettled otherwise (a kernel the harness cannot launch included); `<run>/kernels.json` lists each kernel's
verdict. A source with its own `main` (aorta's fixtures, the trusted controls) runs as it is, on its one kernel.

Trusted controls are aorta's clean fixtures (the single-wave LDS repro for ConSan, the in-bounds
store for ASan): a control run is clean only if the fixture comes back PASS.

Each aorta run has a per-check wall-clock limit (`reward.gpu_check_s`, inside the adapter's own share of the pool
timeout): aorta stops the program at the limit, and if aorta has not returned `CHECK_GRACE_S` later everything it
launched for the run is killed here (the program, or ASan's node-health canary and its build if the program has not
started), and what it launches for the run after that is killed every `REAP_S` until aorta returns. A check that
reached the limit is unsettled (`check_timeout`) whatever came back: a timeout is never clean. A kernel that hangs then holds its worker for the limit, not for the whole pool timeout
(one such kernel held the only ConSan worker for 110 s).

aorta launches the checked program (and ASan's node-health canary, and the canary's build) with its own subprocess
calls, which this adapter's runner never sees. So `close()` (the watchdog retiring the worker, a client going away)
kills every process whose command line names the directory of a run in progress, and goes on doing so every `REAP_S`
until those runs have ended; no check starts after it, and a check it interrupted is unsettled (`worker_died`).

These adapters run only inside a Slurm job (`jobs/aorlta-reward-standin.sbatch`), never on the
interactive node.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from pathlib import Path

from ...ids import sha256_text
from ...schemas import ControlSpec, LegJob, LegResult
from . import aorta_import
from .base import ProcessRunner, aorta_findings, make_result, map_aorta_check, run_dir, unsettled
from .standin_harness import (HARNESS_VERSION, GLOBAL_FN, HarnessUnsupported, has_main, plan, strip_comments,
                              write_harness)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def single_kernel(source_text: str) -> str | None:
    """The source's one __global__ function, or None if it has none or several. Comments are not code."""
    names = sorted(set(GLOBAL_FN.findall(strip_comments(source_text))))
    return names[0] if len(names) == 1 else None


CHECK_GRACE_S = 10.0  # past the per-check limit, before the program is killed here rather than by aorta
REAP_S = 0.05  # after close(), how often what the runs still start is killed, until they have ended


def kill_in(dirs: list[Path]) -> int:
    """SIGKILL every process whose command line names a path inside one of `dirs`, a run's own directories: the program
    aorta launched, the ASan node-health canary and its build. Returns how many were signalled."""
    inside, n = [str(d).rstrip("/") + "/" for d in dirs], 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if any(p in a for a in argv for p in inside):
            try:
                os.kill(int(d.name), signal.SIGKILL)
                n += 1
            except OSError:
                pass
    return n


def kill_program(exe: str | Path) -> int:
    """SIGKILL every process whose argv[0] is `exe` (the harness program aorta launched for this check; its path is in
    the run's own directory). Returns how many were signalled."""
    target, n = str(exe), 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv0 = (d / "cmdline").read_bytes().split(b"\0", 1)[0].decode("utf-8", "replace")
        except OSError:
            continue
        if argv0 == target:
            try:
                os.kill(int(d.name), signal.SIGKILL)
                n += 1
            except OSError:
                pass
    return n


def _worklist(models, kernel: str, target: str):
    ident = models.KernelIdentity(name=kernel, target=target)
    return models.KernelWorklist(requirement=models.SelectionRequirement.TOP_DISPATCH_COUNT, top_n=1,
                                 kernels=(models.KernelObservation(identity=ident, dispatch_count=1,
                                                                   sources=("aorlta",)),))


class _AortaGpu:
    leg: str = ""
    resource = "own_gpu"

    def __init__(self, *, aorta_src: str, hipcc: str = "/opt/rocm/bin/hipcc", target: str = "gfx950",
                 timeout_s: float = 600.0, check_timeout_s: float | None = None, control_source: str | None = None,
                 clock=time.time):
        self.aorta_src = aorta_src
        self.hipcc = hipcc
        self.target = target
        self.timeout_s = timeout_s
        # the per-check limit: reward.gpu_check_s, never more than the adapter's own share of the pool timeout
        self.check_s = timeout_s if check_timeout_s is None else min(check_timeout_s, timeout_s)
        self.control_source = control_source
        self._clock = clock
        self.runner = ProcessRunner()
        self.legs = (self.leg,)
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._runs: set[Path] = set()  # the directories of the runs in progress
        self._lingering: set[Path] = set()  # those of checks aorta has not returned from past their limit

    def _build(self, src: str, out: Path, flags: list[str]) -> tuple[Path | None, str | None]:
        exe = out / "harness"
        out.mkdir(parents=True, exist_ok=True)
        proc = self.runner.run([self.hipcc, *flags, "-g", "-O1", src, "-o", str(exe)], timeout_s=300)
        if proc.killed:
            return None, "worker_died"
        if proc.returncode != 0 or not exe.is_file():
            try:  # stand-in sources only: the build's stderr stays beside the leg, for whoever reads the package
                (out / "harness_build.log").write_text(proc.stderr[-20000:], encoding="utf-8")
            except OSError:
                pass
            return None, "harness_build_failed"
        return exe, None

    def _execute(self, argv, timeout_s: float):
        proc = self.runner.run(argv, timeout_s=timeout_s)
        return (None if proc.killed or proc.timed_out else proc.returncode), proc.stdout

    def run(self, job: LegJob) -> LegResult:
        t0 = self._clock()
        try:
            src = job.inputs.get("source")
            if job.leg != self.leg or not src or not job.output_dir:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
            if self._closed.is_set():
                return unsettled(job, "worker_died", t_start=t0, t_end=self._clock())
            text = Path(src).read_text(encoding="utf-8")
            models, why = aorta_import.load("models", self.aorta_src)
            if models is None:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
            out = run_dir(job)
            with self._lock:
                self._runs.add(out)
            try:
                state, reason, findings = self._check_source(models, Path(src), text, out, job.leg_settings)
            finally:
                with self._lock:
                    self._runs.discard(out)
            return make_result(job, state, reason, findings=findings, t_start=t0, t_end=self._clock())
        except Exception:
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())

    def _check_source(self, models, src: Path, text: str, out: Path, settings: dict):
        if not has_main(text):  # a kernel-only candidate: the stand-in harness launches each kernel
            return self._each_kernel(models, src, out)
        kernel = settings.get("kernel") or single_kernel(text)
        if not kernel:
            return "unsettled", "harness_unsupported: a program with no one kernel to select", []
        exe, why = self._build(str(src), out, self.build_flags())
        if exe is None:
            return "unsettled", why, []
        return self.bounded_check(models, exe, kernel, out)

    def _each_kernel(self, models, src: Path, out: Path):
        """Every kernel of a kernel-only source, one harness program and one aorta run each, combined."""
        out.mkdir(parents=True, exist_ok=True)
        try:
            planned = plan(src, hipcc=self.hipcc, target=self.target, workdir=out, run=self._execute)
        except HarnessUnsupported as exc:
            (out / "kernels.json").write_text(json.dumps({"error": str(exc)[:500]}), encoding="utf-8")
            return "unsettled", f"harness_unsupported: {exc}"[:300], []
        results, rows = [], []
        for i, p in enumerate(planned):
            kdir = out / f"k{i}"
            if p.instance is None:
                r = ("unsettled", f"harness_unsupported: {p.why}"[:300], [])
            else:
                exe, why = self._build(str(write_harness(src, [p.instance], kdir)), kdir, self.build_flags())
                r = ("unsettled", why, []) if exe is None else self.bounded_check(models, exe, p.kernel.name, kdir)
            results.append(r)
            rows.append({"kernel": p.instance.label if p.instance else p.kernel.qualified, "dir": kdir.name,
                         "state": r[0], "reason": r[1], "findings": len(r[2])})
        (out / "kernels.json").write_text(json.dumps({"kernels": rows}, indent=1), encoding="utf-8")
        return combine(results)

    def bounded_check(self, models, exe: Path, kernel: str, out: Path):
        """`check` under the per-check wall-clock limit. aorta stops the program at `check_s` itself; a check still
        running `CHECK_GRACE_S` later has what it launched for `out` killed here, and goes on having it killed until it
        returns (`_linger`). Reaching the limit is unsettled, never clean."""
        limit, box = self.check_s, {}
        if self._closed.is_set():
            return "unsettled", "worker_died", []

        def work() -> None:
            try:
                box["result"] = self.check(models, exe, kernel, out)
            except Exception:  # noqa: BLE001 - nothing that failed is clean
                box["result"] = ("unsettled", "adapter_error", [])

        t0 = time.monotonic()
        thread = threading.Thread(target=work, name=f"aorta-check-{kernel}"[:60], daemon=True)
        thread.start()
        thread.join(limit + CHECK_GRACE_S)
        if thread.is_alive():
            kill_program(exe)
            kill_in([out])  # the ASan canary and its build, if the program has not started
            thread.join(CHECK_GRACE_S)
        if thread.is_alive():
            self._linger(thread, exe, out)
        elapsed = time.monotonic() - t0
        if self._closed.is_set():  # its program was killed by close(): whatever came back is no verdict
            return "unsettled", "worker_died", []
        if thread.is_alive() or elapsed >= limit:
            return "unsettled", f"check_timeout: {elapsed:.1f} s, limit {limit:g} s", []
        return box["result"]

    def _linger(self, thread: threading.Thread, exe: Path, out: Path) -> None:
        """A check aorta has not returned from: what it launches for `out` is killed every `REAP_S` until it returns,
        and `close()` covers `out` until then."""
        with self._lock:
            self._lingering.add(out)

        def reap() -> None:
            while thread.is_alive():
                kill_program(exe)
                kill_in([out])
                thread.join(REAP_S)
            with self._lock:
                self._lingering.discard(out)

        threading.Thread(target=reap, name=f"{self.name}-linger", daemon=True).start()

    def controls(self) -> list[ControlSpec]:
        if not self.control_source:
            return []
        return [ControlSpec(control_id=f"{self.name}/control", leg=self.leg,
                            source_sha256=_sha256(self.control_source),
                            task_sha256=sha256_text(f"{self.name}/control-task"),
                            inputs={"source": self.control_source})]

    def health(self) -> dict:
        return {"name": self.name, "responsive": True}

    def close(self) -> None:
        """Stop the job in progress: the runner's subprocess, and every process aorta launched for a run in progress or
        launches for it until the run ends (`kill_in`, every REAP_S); no check starts after this."""
        self._closed.set()
        self.runner.close()
        with self._lock:
            live = list(self._runs | self._lingering)
        if live:
            kill_in(live)
            threading.Thread(target=self._reap, name=f"{self.name}-reap", daemon=True).start()

    def _reap(self) -> None:
        while True:
            with self._lock:
                live = list(self._runs)
            if not live:
                return
            time.sleep(REAP_S)
            kill_in(live)


def combine(results: list[tuple[str, str | None, list]]) -> tuple[str, str | None, list]:
    """One leg from its kernels' runs: any finding is the leg's finding; clean only if every run is clean;
    otherwise unsettled, with the first unsettled run's reason."""
    found = [f for state, _, fs in results if state == "finding" for f in fs]
    if found:
        return "finding", None, found
    if results and all(state == "clean" for state, _, _ in results):
        return "clean", None, []
    first = next(((s, r) for s, r, _ in results if s != "clean"), ("unsettled", "harness_unsupported: no kernel"))
    return "unsettled", first[1] or "unsettled", []


class AortaConsan(_AortaGpu):
    name = "standin.aorta_consan"
    leg = "san.consan"

    def __init__(self, *, hook: str, **kw):
        super().__init__(**kw)
        self.hook = hook
        try:
            ident = _sha256(hook)[:16]
        except OSError:
            ident = "missing"
        self.version = (f"2+target-{self.target}+harness-{HARNESS_VERSION}+hook-{ident}"
                        f"+aorta-{aorta_import.commit(self.aorta_src)[:12]}")

    def build_flags(self) -> list[str]:
        return [f"--offload-arch={self.target}"]

    def check(self, models, exe: Path, kernel: str, out: Path):
        consan, why = aorta_import.load("consan", self.aorta_src)
        if consan is None:
            return "unsettled", "adapter_error", []
        r = consan.run_consan(_worklist(models, kernel, self.target), command=exe, hook_lib=Path(self.hook),
                              output_dir=out / "consan", strict=True, timeout_seconds=self.check_s)
        state, reason = map_aorta_check(r.consan)
        pf = r.waitcheck_preflight
        pf_state = getattr(getattr(pf, "verdict", None), "value", None)
        if state != "unsettled" and pf_state not in ("pass", "warn"):
            state, reason = "unsettled", f"consan_preflight_not_settled:{pf_state}"
        return state, reason, (aorta_findings(r.consan) if state == "finding" else [])


class AortaAsan(_AortaGpu):
    name = "standin.aorta_asan"
    leg = "san.asan"
    max_packages = 16
    inband_controls = True

    def __init__(self, **kw):
        super().__init__(**kw)
        lib = os.environ.get("AORTA_ASAN_ROCM_LIB", "")
        self.version = (f"2+target-{self.target}+harness-{HARNESS_VERSION}+asanlib-{sha256_text(lib)[:12]}"
                        f"+aorta-{aorta_import.commit(self.aorta_src)[:12]}")

    def build_flags(self) -> list[str]:
        return [f"--offload-arch={self.target}:xnack+", "-fsanitize=address", "-shared-libsan"]

    def check(self, models, exe: Path, kernel: str, out: Path):
        asan, why = aorta_import.load("asan", self.aorta_src)
        if asan is None:
            return "unsettled", "adapter_error", []
        runtime = asan.resolve_asan_runtime()
        check = asan.run_asan(_worklist(models, kernel, self.target), command=exe, output_dir=out / "asan",
                              runtime=runtime, timeout_seconds=self.check_s)
        state, reason = map_aorta_check(check)
        return state, reason, (aorta_findings(check) if state == "finding" else [])

    def run_batch(self, jobs):
        return [self.run(j) for j in jobs]
