"""`standin.hipcc`: the stand-in compile gate (design Section 6.2):
`hipcc --offload-arch=gfx950 -c -O1 source.hip -o object.o`, on CPU [M 0.5-0.7 s per file].

The offload arch is always explicit, so hipcc never probes a GPU for it (the interactive node has
GPUs, and nothing GPU-capable may run there). A compiler that ran and printed `error:` is the
candidate's compile error (`fail`, reason `compile_error`, with the error count and the errors'
classes, never their text). Anything else that does not produce an object (a launch failure, a
crash, the adapter's own timeout, a kill by the watchdog) is `unsettled`: it says nothing about the
candidate.

A clean compile also carries a build fact (`base.TRUSTED_BUILD`) of the launch the stand-in ConSan and ASan legs
make, when it can be read. A reward function may use it, for example to treat ConSan as not applicable on a launch
with no LDS; standin.v11 does not read it:

  - the kernels are the stand-in harness's own instances (`standin_harness.plan`, as the GPU legs plan them), built
    into one device code object from a harness that launches all of them (`--cuda-device-only`, the gate's flags);
  - every kernel descriptor in that code object gives its static LDS (`descriptors.read_group_segments`);
  - the dynamic LDS is the harness's (`standin_harness.dynamic_lds_bytes`);
  - `launched_lds` is the largest static LDS of any kernel in the code object (a kernel the harness does not launch
    counts too), the dynamic LDS, and the kernel's symbol when there is one; `kernels` lists each descriptor.

There is no fact when the source has its own `main` (its own launch sets the
dynamic LDS), the harness cannot launch one of its kernels, the harness does not build, a descriptor or the metadata
cannot be read, or the time runs out. The compile stays clean either way. `<out>/launched_lds.json` says which.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from pathlib import Path

from ...atomic import atomic_write_json
from ...schemas import ControlSpec, LegJob, LegResult
from . import descriptors
from .base import TRUSTED_BUILD, ProcessRunner, make_result, unsettled
from .standin_harness import HARNESS_VERSION, HarnessUnsupported, dynamic_lds_bytes, has_main, plan, write_harness

# "fatal error" too: a missing header (`#include <cuda_runtime.h>`) is the candidate's compile error, not the checker's.
ERROR_LINE = re.compile(r"^[^\n]*?:\d+:\d+: (?:fatal )?error: (?P<msg>[^\n]*)$", re.MULTILINE)
QUOTED = re.compile(r"'[^']*'")
FACT_SOURCE = "standin.hipcc"  # the build fact's `source`
FACT_FILE = "launched_lds.json"
MIN_STEP_S = 1.0  # time left below which the fact is not attempted


def error_classes(stderr: str, limit: int = 20) -> list[str]:
    """The clang error messages with every quoted name replaced by <name> (Section 6.1's `cls`)."""
    return [QUOTED.sub("<name>", m.group("msg")).strip()[:160] for m in ERROR_LINE.finditer(stderr)][:limit]


class _Killed(Exception):
    """The worker was closed while the fact was being read."""


class StandinHipcc:
    name = "standin.hipcc"
    legs = ("gate.compile",)
    resource = "local_cpu"

    def __init__(self, hipcc: str = "/opt/rocm/bin/hipcc", *, readelf: str = "/opt/rocm/llvm/bin/llvm-readelf",
                 timeout_s: float = 110.0, clock=time.time, runner: ProcessRunner | None = None):
        self.hipcc = hipcc
        self.readelf = readelf
        self.timeout_s = timeout_s
        self._clock = clock
        self.runner = runner or ProcessRunner()
        self.version = (f"2+{self._identity(self.hipcc)}+readelf-{self._identity(self.readelf)}"
                        f"+harness-{HARNESS_VERSION}")

    @staticmethod
    def _identity(path: str) -> str:
        try:
            real = os.path.realpath(path)
            with open(real, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except OSError:
            return "missing"

    def run(self, job: LegJob) -> LegResult:
        t0 = self._clock()
        deadline = time.monotonic() + self.timeout_s
        try:
            arch = job.leg_settings.get("offload_arch")
            flags = list(job.leg_settings.get("flags", ["-c", "-O1"]))
            src = job.inputs.get("source")
            if job.leg != "gate.compile" or not arch or not src or not job.output_dir or "-c" not in flags:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
            out = Path(job.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            tmp = out / f"object.o.partial-{os.getpid()}-{job.attempt}"
            argv = [self.hipcc, f"--offload-arch={arch}", *flags, src, "-o", str(tmp)]
            proc = self.runner.run(argv, timeout_s=self.timeout_s)
            if proc.killed:
                return unsettled(job, "worker_died", t_start=t0, t_end=self._clock())
            if proc.launch_error is not None:
                return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
            if proc.timed_out:
                return unsettled(job, "timeout", t_start=t0, t_end=self._clock())
            if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                os.replace(tmp, out / "object.o")
                fact, why = self.build_fact(src, arch, flags, out / f"lds.partial-{os.getpid()}-{job.attempt}",
                                            deadline)
                atomic_write_json(out / FACT_FILE, {"fact": fact, "why": why}, canonical=True)
                return make_result(job, "clean", findings=[fact] if fact else [], t_start=t0, t_end=self._clock())
            tmp.unlink(missing_ok=True)
            classes = error_classes(proc.stderr)
            if proc.returncode not in (0, None) and classes:
                return make_result(job, "fail", "compile_error",
                                   findings=[{"kind": "compile_error", "errors": len(classes), "classes": classes}],
                                   t_start=t0, t_end=self._clock())
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())
        except _Killed:
            return unsettled(job, "worker_died", t_start=t0, t_end=self._clock())
        except Exception:
            return unsettled(job, "adapter_error", t_start=t0, t_end=self._clock())

    # --- the trusted build fact ---------------------------------------------------------------------------------

    def _bounded(self, deadline: float):
        """A `run(argv, timeout_s)` for the harness planner and readelf: the worker's killable runner, inside what is
        left of the gate's own timeout. A closed worker raises _Killed; anything that did not finish is (None, "")."""
        def run(argv, timeout_s: float):
            left = deadline - time.monotonic()
            if left < MIN_STEP_S:
                return None, ""
            proc = self.runner.run(argv, timeout_s=min(timeout_s, left))
            if proc.killed:
                raise _Killed
            if proc.launch_error is not None or proc.timed_out:
                return None, ""
            return proc.returncode, proc.stdout
        return run

    def build_fact(self, src: str, arch: str, flags: list[str], workdir: Path, deadline: float
                   ) -> tuple[dict | None, str | None]:
        """(the trusted build fact, None), or (None, why) when the launch's LDS cannot be read (module docstring)."""
        try:
            return self._fact(src, arch, flags, workdir, self._bounded(deadline), deadline)
        except _Killed:
            raise
        except Exception as exc:  # noqa: BLE001 - a fact that could not be read is no fact
            return None, f"descriptor_error: {type(exc).__name__}"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _fact(self, src: str, arch: str, flags: list[str], workdir: Path, run, deadline: float
              ) -> tuple[dict | None, str | None]:
        text = Path(src).read_text(encoding="utf-8", errors="replace")
        if has_main(text):
            return None, "own_main: the program's own launch sets its dynamic LDS"
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            planned = plan(src, hipcc=self.hipcc, target=arch, workdir=workdir, run=run)
        except HarnessUnsupported as exc:
            return None, f"harness_unsupported: {exc}"[:300]
        missing = [p.kernel.qualified for p in planned if p.instance is None]
        if missing or not planned:
            return None, f"no_instance: {', '.join(missing)}"[:300]
        instances = [p.instance for p in planned]
        harness = write_harness(src, instances, workdir, name="aorlta_lds.hip")
        code_object = workdir / "launched.co"
        rc, _ = run([self.hipcc, f"--offload-arch={arch}", "--cuda-device-only", "--no-gpu-bundle-output", *flags,
                     str(harness), "-o", str(code_object)], self.timeout_s)
        if rc != 0 or not code_object.is_file():
            return None, f"harness_build_failed: exit {rc}"
        sizes, why = descriptors.read_group_segments(code_object, readelf=self.readelf, run=run,
                                                     timeout_s=max(MIN_STEP_S, deadline - time.monotonic()))
        if sizes is None:
            return None, why
        if len(sizes) < len(instances):
            return None, f"descriptor_unreadable: {len(sizes)} descriptors for {len(instances)} launched kernels"
        kernels = [{"symbol": s, "static_bytes": sizes[s]} for s in sorted(sizes)]
        fact = {"kind": TRUSTED_BUILD, "source": FACT_SOURCE, "language": "hip",
                "launched_lds": {"static_bytes": max(sizes.values()), "dynamic_bytes": dynamic_lds_bytes(text),
                                 "symbol": kernels[0]["symbol"] if len(kernels) == 1 else None},
                "kernels": kernels, "launches": [i.label for i in instances]}
        return fact, None

    def controls(self) -> list[ControlSpec]:
        return []

    def health(self) -> dict:
        return {"name": self.name, "responsive": True}

    def close(self) -> None:
        self.runner.close()
