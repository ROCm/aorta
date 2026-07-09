"""HRX performance benchmark workload.

Companion to the correctness-focused :mod:`aorta.workloads.hrx` probe workload.
Where ``hrx`` answers "does this HIP launch path produce the right bytes under
HRX?", ``hrx_perf`` answers "how *fast* is it under HRX vs stock ROCm HIP?".

It builds one of a small set of deliberately *big* HIP benchmarks with
``hipcc`` at :meth:`setup` and execs it as a child process:

    * ``gemm``  -- compute-bound tiled SGEMM (C = A*B, N x N floats).
    * ``triad`` -- bandwidth-bound STREAM triad (a = b + s*c over N floats).

Each bench runs ``warmup`` untimed then ``iters`` timed iterations, timing each
iteration host-side (launch + ``hipDeviceSynchronize``) so the per-step number
includes runtime/launch overhead -- the axis on which an alternate HIP runtime
can differ. Those per-iteration times are reported as
:attr:`WorkloadResult.step_times_ms`, so ``aorta sweep run``'s matrix renders a
per-cell **Mean step (ms)** and a confound ratio between the ``hrx_on`` and
``hrx_off`` cells -- i.e. the HRX-vs-stock speed comparison. Achieved throughput
(GFLOP/s or GB/s) is reported in :attr:`WorkloadResult.metrics`.

Runtime routing (HRX vs stock HIP) is an environment concern, identical to the
``hrx`` workload: the ``hrx_on`` cell injects an ``LD_PRELOAD`` +
``LD_LIBRARY_PATH`` + ``HRX_GPU_DRIVER`` bundle that takes effect at the
probe's ``exec()``. The same guards apply -- the bundle is stripped from the
``hipcc`` build env, a nonexistent ``LD_PRELOAD`` fails setup, and an ignored
preload fails the run -- so a misrouted cell can't silently benchmark the wrong
runtime.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from aorta.workloads._base import Workload, WorkloadResult
from aorta.workloads.hrx import (
    _GPU_KFD_NODE,
    _LDSO_PRELOAD_IGNORED_RE,
    _build_env,
    _gpu_available,
    _missing_preload_objects,
    _resolve_hipcc,
)

log = logging.getLogger(__name__)

_KERNELS_DIR = Path(__file__).parent / "hrx_kernels"

_PASS_RESULT = "PERF_OK"
_RESULT_RE = re.compile(r"^RESULT=([A-Z_]+)", re.MULTILINE)
_STEP_RE = re.compile(r"^step_ms=([-\d.eE+]+)", re.MULTILINE)


@dataclass(frozen=True)
class _BenchSpec:
    """Build + parse recipe for one benchmark.

    Attributes:
        source: vendored ``.cpp`` compiled to the benchmark executable.
        binary: output executable name.
        throughput_token: stdout token carrying the achieved throughput
            (``GFLOPS`` or ``GBPS``).
        throughput_metric: key under which the parsed throughput lands in
            ``WorkloadResult.metrics``.
        default_size: problem size when ``workload_config.size`` is unset
            (matrix dimension N for gemm; element count for triad).
    """

    source: str
    binary: str
    throughput_token: str
    throughput_metric: str
    default_size: int


_BENCHES: dict[str, _BenchSpec] = {
    "gemm": _BenchSpec("hrx_perf_gemm.cpp", "hrx_perf_gemm", "GFLOPS", "gflops", 4096),
    "triad": _BenchSpec("hrx_perf_triad.cpp", "hrx_perf_triad", "GBPS", "gbps", 64_000_000),
}

_DEFAULT_BENCH = "gemm"
_DEFAULT_ARCH = "gfx942"
_DEFAULT_ITERS = 50
_DEFAULT_WARMUP = 10
_DEFAULT_TIMEOUT_SEC = 600

_RESERVED_KEYS = {"steps"}
_KNOWN_KEYS = {
    "bench",
    "gpu_arch",
    "size",
    "iters",
    "warmup",
    "hipcc",
    "build_dir",
    "timeout_sec",
    "keep_build",
}


class HrxPerfWorkload(Workload):
    """Build and time a big HIP benchmark; report per-step times + throughput.

    ``workload_config`` keys:
        bench: which benchmark to run (default ``"gemm"``); ``gemm`` or ``triad``.
        gpu_arch: ``--offload-arch`` target (default ``"gfx942"``).
        size: problem size (gemm: matrix dim N; triad: element count).
            Defaults per bench (gemm 4096, triad 64e6).
        iters: timed iterations (default ``50``).
        warmup: untimed warmup iterations (default ``10``).
        hipcc: explicit hipcc path (default: ``$HIPCC`` /
            ``/opt/rocm/bin/hipcc`` / ``hipcc`` on PATH).
        build_dir: directory for built binaries (default: a temp dir removed
            on cleanup unless ``keep_build`` is set).
        timeout_sec: per-run subprocess timeout (default ``600``).
        keep_build: keep ``build_dir`` after cleanup (default ``False``).
    """

    name: ClassVar[str] = "hrx_perf"

    def _validated_config(self) -> tuple[str, int, int, int]:
        for key in self.config:
            if key in _KNOWN_KEYS or key in _RESERVED_KEYS or key.startswith("_aorta_"):
                continue
            log.warning("hrx_perf: ignoring unknown workload_config key %r", key)
        bench = self.config.get("bench", _DEFAULT_BENCH)
        if bench not in _BENCHES:
            raise ValueError(
                f"hrx_perf: unknown bench {bench!r}; choose one of {sorted(_BENCHES)}"
            )
        spec = _BENCHES[bench]
        size = int(self.config.get("size", spec.default_size))
        iters = int(self.config.get("iters", _DEFAULT_ITERS))
        warmup = int(self.config.get("warmup", _DEFAULT_WARMUP))
        if size <= 0 or iters <= 0 or warmup < 0:
            raise ValueError(
                f"hrx_perf: size ({size}) and iters ({iters}) must be > 0 and "
                f"warmup ({warmup}) must be >= 0"
            )
        return bench, size, iters, warmup

    def setup(self) -> None:
        # Same fail-fast guard as the hrx workload: a nonexistent LD_PRELOAD is
        # only a loader warning, so an hrx_on cell would otherwise benchmark the
        # DEFAULT HIP runtime and report a meaningless comparison.
        missing = _missing_preload_objects()
        if missing:
            raise RuntimeError(
                "hrx_perf: LD_PRELOAD names object(s) that do not exist: "
                + ", ".join(repr(m) for m in missing)
                + ". The dynamic loader would ignore these with only a stderr "
                "warning and run the benchmark against the DEFAULT HIP runtime, "
                "so the cell would silently measure stock HIP. Fix the path(s) "
                "in the cell's extra_env (absolute paths)."
            )
        self._bench, self._size, self._iters, self._warmup = self._validated_config()
        self._spec = _BENCHES[self._bench]
        self._arch = str(self.config.get("gpu_arch", _DEFAULT_ARCH))
        # Validate here: subprocess.run(..., timeout=<=0) would raise at run()
        # time and be misclassified as an infrastructure failure. Fail fast in
        # setup() with a clear config error instead.
        self._timeout = int(self.config.get("timeout_sec", _DEFAULT_TIMEOUT_SEC))
        if self._timeout <= 0:
            raise ValueError(
                f"hrx_perf: timeout_sec ({self._timeout}) must be > 0"
            )
        keep_build = self.config.get("keep_build", False)
        if not isinstance(keep_build, bool):
            raise ValueError(
                f"hrx_perf: keep_build must be a bool, got {type(keep_build).__name__}"
            )
        self._keep_build = keep_build

        hipcc = _resolve_hipcc(self.config.get("hipcc"))
        if hipcc is None:
            raise RuntimeError(
                "hrx_perf: hipcc not found (looked at $HIPCC, /opt/rocm/bin/hipcc, "
                "and PATH). This workload builds HIP benchmarks from source and "
                "requires a ROCm/hipcc toolchain. Run it in a ROCm environment, "
                "or set workload_config.hipcc."
            )
        self._hipcc = hipcc

        # Fail in setup() (not run()) when no GPU is reachable: a benchmark with
        # no device produces no step times, which the matrix would treat as a
        # did-not-run anyway, but raising here makes the classification explicit
        # and matches the hrx workload's contract.
        if not _gpu_available():
            raise RuntimeError(
                f"hrx_perf: no accessible ROCm GPU ({_GPU_KFD_NODE} is not "
                "readable+writable). This workload times a real HIP benchmark "
                "and needs a GPU; run it on a ROCm host (or a container started "
                "with --device=/dev/kfd --device=/dev/dri and the render group)."
            )

        build_dir = self.config.get("build_dir")
        if build_dir:
            # Absolutize: _build runs hipcc with cwd=_KERNELS_DIR and run() execs
            # with cwd=build_dir, so a relative path would split -o from the exec.
            self._build_dir = Path(build_dir).resolve()
            self._build_dir.mkdir(parents=True, exist_ok=True)
            self._owns_build_dir = False
        else:
            self._build_dir = Path(tempfile.mkdtemp(prefix="aorta-hrx-perf-"))
            self._owns_build_dir = True

        self._binary = self._build()

    def _build(self) -> Path:
        # Artifacts live in an arch subdir so a shared build_dir reused across
        # differing --offload-arch builds can't reuse a benchmark compiled for a
        # different GPU arch (the binary name encodes the bench but not the
        # arch). size/iters/warmup are runtime argv, not compile-time, so arch
        # is the only compile-time discriminator that needs isolating.
        build_root = self._build_dir / self._arch
        build_root.mkdir(parents=True, exist_ok=True)
        binary = build_root / self._spec.binary
        # Reuse an already-built benchmark. setup() runs per trial, so a run
        # pinning a shared build_dir would otherwise recompile the same binary
        # each trial. The default build_dir is a fresh temp dir, so this only
        # fires for an operator-supplied build_dir.
        # Require the execute bit before reusing: a leftover artifact with wrong
        # permissions or a partial write would otherwise be reused and fail
        # run() with PermissionError -- a cell error rather than a recoverable
        # rebuild.
        if binary.is_file() and os.access(binary, os.X_OK):
            log.debug("hrx_perf: reusing existing %s in %s", self._spec.binary, build_root)
            return binary
        cmd = [
            self._hipcc,
            "-O3",
            f"--offload-arch={self._arch}",
            str(_KERNELS_DIR / self._spec.source),
            "-o",
            str(binary),
        ]
        log.debug("hrx_perf: building %s: %s", self._spec.binary, " ".join(cmd))
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(_KERNELS_DIR), env=_build_env()
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"hrx_perf: hipcc failed to build {self._spec.binary} "
                f"(exit {proc.returncode}).\ncmd: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr.strip()}"
            )
        return binary

    def run(self) -> WorkloadResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [str(self._binary), str(self._size), str(self._iters), str(self._warmup)],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._build_dir),
            )
            timed_out = False
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            exit_code = None
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
        elapsed = time.monotonic() - start

        step_times = [float(m) for m in _STEP_RE.findall(stdout)]
        result_match = _RESULT_RE.search(stdout)
        result_token = result_match.group(1) if result_match else None
        throughput = self._parse_throughput(stdout)

        # Measurable work == at least one timed iteration. A bare RESULT line
        # with no step_ms (only the C bad-args path, unreachable given the
        # Python-side validation) is NOT work, so it must not mint a
        # did-run/iteration signal for the matrix (issue #274, KB #7).
        main_work_started = bool(step_times)

        preload_requested = bool(os.environ.get("LD_PRELOAD", "").strip())
        preload_ignored = preload_requested and bool(
            _LDSO_PRELOAD_IGNORED_RE.search(stderr)
        )

        passed = (
            (not timed_out)
            and exit_code == 0
            and result_token == _PASS_RESULT
            and not preload_ignored
        )

        mean_step_ms = sum(step_times) / len(step_times) if step_times else None
        metrics: dict[str, Any] = {
            "bench": self._bench,
            "gpu_arch": self._arch,
            "size": self._size,
            "iters": self._iters,
            "warmup": self._warmup,
            "result": result_token,
            "mean_step_ms": mean_step_ms,
            self._spec.throughput_metric: throughput,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "preload_ignored": preload_ignored,
        }

        failure_details: list[dict[str, Any]] = []
        if not passed:
            if preload_ignored:
                hint = (
                    f"bench {self._bench!r}: LD_PRELOAD was set but the loader "
                    "ignored it (see stderr) -- the benchmark ran against the "
                    "DEFAULT HIP runtime, so this timing does NOT reflect the "
                    "intended runtime. Check the LD_PRELOAD path/arch/deps."
                )
            elif timed_out:
                hint = f"bench {self._bench!r} timed out (>{self._timeout}s)"
            elif result_token is None:
                hint = (
                    f"bench {self._bench!r} produced no RESULT line "
                    f"(exit {exit_code}); see stdout/stderr"
                )
            else:
                hint = f"bench {self._bench!r} result {result_token} (expected PERF_OK)"
            failure_details.append(
                {
                    "bench": self._bench,
                    "result": result_token,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "preload_ignored": preload_ignored,
                    "stdout_tail": stdout.strip()[-2000:],
                    "stderr_tail": stderr.strip()[-2000:],
                    "hint": hint,
                }
            )

        executed = len(step_times)
        # Contract (WorkloadResult): None only when passing. When we failed but
        # timed iterations did run (e.g. a PERF_FAIL checksum after the timed
        # loop), report a best-effort index of 0 -- the perf failure isn't tied
        # to a single iteration, but 0 keeps us within the documented
        # 0..total_iterations-1 range and consistent with the hrx workload. A
        # setup-only failure/timeout (no step_times) stays None.
        first_failure_iteration = 0 if (not passed and main_work_started) else None
        return WorkloadResult(
            passed=passed,
            failure_count=0 if passed else 1,
            first_failure_iteration=first_failure_iteration,
            failure_details=failure_details,
            total_iterations=executed,
            step_times_ms=step_times,
            elapsed_sec=elapsed,
            metrics=metrics,
            main_work_started=main_work_started,
            executed_iterations=executed,
            configured_iterations=self._iters,
        )

    def _parse_throughput(self, stdout: str) -> float | None:
        m = re.search(rf"^{self._spec.throughput_token}=([-\d.eE+]+)", stdout, re.MULTILINE)
        return float(m.group(1)) if m else None

    def cleanup(self) -> None:
        if getattr(self, "_owns_build_dir", False) and not getattr(
            self, "_keep_build", False
        ):
            shutil.rmtree(self._build_dir, ignore_errors=True)
