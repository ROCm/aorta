"""`standin.rj_waitcheck`: static waitcheck on the compiled object (design Section 6.2),
by aorta's `run_waitcheck` with the installed `rj_waitcheck` (`/opt/rocjitsu/bin/rj_waitcheck`; it needs
a libstdc++ with GLIBCXX_3.4.31, so `ld_library_path` names one on el9).

The object is scanned whole (no entry offset): aorta's whole-code-object identity. Its verdicts map
as Section 6.2 says: PASS -> clean, WARN with findings -> finding, ERROR / NOT_CHECKED / TIMED_OUT ->
unsettled. A capped scan (`diagnostics=>=N`) still has findings, so it is a finding.

The object is the compile gate's, unless the source defines a template kernel and has no `main`: a template has no
code until something instantiates it, so then the scan is of the instantiation the GPU legs launch, the stand-in
harness (`standin_harness.plan`) with every kernel's instance, compiled `-c -O1` for the leg's target
(`instantiation.o`). A kernel with no instance leaves the leg unsettled, since nothing scanned it, and so does a
source whose kernels cannot be read (clang's dump fails, times out or does not start, or a kernel is a shape the
harness does not know): a template among them would have no code in the gate's object. Only a source clang reads as
defining no kernel is scanned as the gate's object alone.

The harness, its instantiation and aorta's logs are in a directory of the run's own under the leg's output directory
(`base.run_dir`), which every job with identical content shares.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from ...schemas import ControlSpec, LegJob, LegResult
from . import aorta_import
from .base import ProcessRunner, aorta_findings, make_result, map_aorta_check, run_dir, unsettled
from .standin_harness import (HARNESS_VERSION, HarnessUnsupported, NoKernel, find_kernels, has_main, plan,
                              write_harness)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class StandinRjWaitcheck:
    name = "standin.rj_waitcheck"
    legs = ("san.waitcheck",)
    resource = "local_cpu"

    def __init__(self, binary: str, *, aorta_src: str | None = None, ld_library_path: str | None = None,
                 timeout_s: float = 110.0, clock=time.time, hipcc: str = "/opt/rocm/bin/hipcc"):
        self.binary = binary
        self.hipcc = hipcc
        self.aorta_src = aorta_src
        self.ld_library_path = ld_library_path
        self.timeout_s = timeout_s
        self._clock = clock
        self.runner = ProcessRunner()
        try:
            ident = _sha256(binary)[:16]
        except OSError:
            ident = "missing"
        self.version = f"2+harness-{HARNESS_VERSION}+rj_waitcheck-{ident}+aorta-{aorta_import.commit(aorta_src)[:12]}"

    def _execute(self, argv, *, timeout_seconds, env=None):
        execution, why = aorta_import.load("execution", self.aorta_src)
        environ = dict(os.environ if env is None else env)
        if self.ld_library_path:
            environ["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [self.ld_library_path,
                                                                       environ.get("LD_LIBRARY_PATH", "")]))
        proc = self.runner.run(argv, timeout_s=min(timeout_seconds, self.timeout_s), env=environ)
        return execution.ProcessResult(argv=tuple(argv), returncode=proc.returncode, stdout=proc.stdout,
                                       stderr=proc.stderr, timed_out=proc.timed_out,
                                       launch_error=proc.launch_error if not proc.killed else "killed by the watchdog")

    def run(self, job: LegJob) -> LegResult:
        t0 = self._clock()
        try:
            waitcheck, why = aorta_import.load("waitcheck", self.aorta_src)
            models, why2 = aorta_import.load("models", self.aorta_src)
            obj = job.inputs.get("object")
            target = job.leg_settings.get("target", "gfx950")
            if waitcheck is None or models is None or job.leg != "san.waitcheck" or not obj or not job.output_dir:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
            out = run_dir(job)
            obj, why = self._object(job, obj, target, out)
            if obj is None:
                return unsettled(job, why, t_start=t0, t_end=self._clock())
            digest = _sha256(obj)
            ident = models.KernelIdentity(name=f"object:{digest[:12]}", target=target, code_object=obj,
                                          code_object_sha256=digest, code_object_index=0)
            worklist = models.KernelWorklist(
                requirement=models.SelectionRequirement.TOP_DISPATCH_COUNT, top_n=1,
                kernels=(models.KernelObservation(identity=ident, dispatch_count=1, sources=("aorlta",)),))
            check = waitcheck.run_waitcheck(worklist, output_dir=out, binary=Path(self.binary),
                                            timeout_seconds=self.timeout_s, execute=self._execute,
                                            max_diagnostics=job.leg_settings.get("max_diagnostics"))
            state, reason = map_aorta_check(check)
            findings = aorta_findings(check) if state == "finding" else []
            if state == "finding" and getattr(check, "diagnostics_truncated", False):
                findings.append({"kind": "diagnostics_truncated"})
            return make_result(job, state, reason, findings=findings, t_start=t0, t_end=self._clock())
        except Exception:
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())

    def _run_tool(self, argv, timeout_s: float):
        proc = self.runner.run(argv, timeout_s=timeout_s)
        return (None if proc.killed or proc.timed_out else proc.returncode), proc.stdout

    def _object(self, job: LegJob, obj: str, target: str, out: Path) -> tuple[str | None, str | None]:
        """The object to scan: the compile gate's, or, for a kernel-only source with a template kernel, the harness's
        instantiation of every kernel, built in `out` (None and why, when a kernel has no instance or the build
        fails)."""
        src = job.inputs.get("source")
        if not src:
            return obj, None
        text = Path(src).read_text(encoding="utf-8", errors="replace")
        if has_main(text):
            return obj, None
        try:
            if not any(k.template is not None for k in find_kernels(src, hipcc=self.hipcc, target=target,
                                                                   run=self._run_tool)):
                return obj, None
            planned = plan(src, hipcc=self.hipcc, target=target, workdir=out, run=self._run_tool)
        except NoKernel:
            return obj, None  # nothing for the harness to instantiate: the gate's object, as it is
        except HarnessUnsupported as exc:
            return None, f"harness_unsupported: {exc}"[:300]
        missing = [p for p in planned if p.instance is None]
        if missing:
            return None, f"harness_unsupported: {missing[0].kernel.qualified}: {missing[0].why}"[:300]
        harness = write_harness(src, [p.instance for p in planned], out)
        inst = out / "instantiation.o"
        proc = self.runner.run([self.hipcc, f"--offload-arch={target}", "-c", "-O1", str(harness), "-o", str(inst)],
                               timeout_s=self.timeout_s)
        if proc.killed:
            return None, "worker_died"
        if proc.returncode != 0 or not inst.is_file():
            try:
                (out / "harness_build.log").write_text(proc.stderr[-20000:], encoding="utf-8")
            except OSError:
                pass
            return None, "harness_build_failed"
        return str(inst), None

    def controls(self) -> list[ControlSpec]:
        return []

    def health(self) -> dict:
        return {"name": self.name, "responsive": True}

    def close(self) -> None:
        self.runner.close()
