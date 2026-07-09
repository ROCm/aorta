"""HRX HIP-launch probe workload.

Exposes the HRX HIP-compatibility-layer launch probes (from the
``ROCm/hrx-system`` #156 / #158 / #160 investigation) as an ``aorta``
workload so they can be run and A/B-compared (HRX-on vs stock ROCm HIP)
through the normal ``aorta run`` / ``aorta sweep run`` flows (the in-process
workload flow; not the deprecated ``aorta probe`` subprocess alias).

Design:
    * **What to run** is this workload: one of a fixed set of single-purpose
      HIP kernel-launch probes, selected by ``workload_config.probe``. Each
      computes ``out[i] = in[i] + 100`` (in=7, out pre-zeroed) and prints a
      ``VERDICT=`` line + an ``out[0]=`` sentinel. ``107``/``FULLY_WORKS`` is
      the only passing outcome; ``0``/``OUTPUT_NOT_WRITTEN`` is the #156 bug.
    * **Which runtime** (HRX vs stock HIP) is NOT decided here. It is an
      environment/mitigation concern: the HRX-on cell applies an
      ``LD_PRELOAD`` + ``LD_LIBRARY_PATH`` + ``HRX_GPU_DRIVER`` env bundle.
      Because the probe is exec'd as a *child* process, that ``LD_PRELOAD``
      takes effect at ``exec()`` (an in-process torch workload could not do
      this — see the HRX handover notes). A bad routing bundle is caught, not
      run silently: :meth:`setup` fails if ``LD_PRELOAD`` names a path that
      doesn't exist, and :meth:`run` marks the result failed if ``ld.so`` reports
      it ignored the preload — either case would otherwise fall back to the
      default HIP runtime and mint a misleading ``FULLY_WORKS``.
    * That routing bundle is stripped from the ``hipcc`` build env (see
      :data:`_RUNTIME_ROUTING_VARS`): the dispatcher merges the cell env into
      ``os.environ`` before ``setup()`` runs, so without stripping, a preloaded
      HRX ``libamdhip64.so`` would run inside the *compiler/link* step. Building
      against the stock toolchain makes the binary identical across cells; only
      :meth:`run` inherits the bundle, so routing applies to the probe alone.

The probe binary is built from vendored source with ``hipcc`` at
:meth:`setup`. On a host without ``hipcc`` (or without a GPU) ``setup``
raises, which the runner classifies as a setup failure / ``did_not_run`` —
never a false ``OUTPUT_NOT_WRITTEN``.
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

log = logging.getLogger(__name__)

_KERNELS_DIR = Path(__file__).parent / "hrx_kernels"

# The only passing verdict. All probes print "VERDICT=<TOKEN> ..." where TOKEN
# is one of these; the trailing parenthetical (e.g. "(H2D/in-arg broken)") is
# descriptive and not part of the token.
_PASS_VERDICT = "FULLY_WORKS"
_VERDICT_RE = re.compile(r"^VERDICT=([A-Z_]+)", re.MULTILINE)
_OUT0_RE = re.compile(r"^out\[0\]=([-\d.eE+]+)", re.MULTILINE)

# ld.so emits this (to stderr) when it cannot load an LD_PRELOAD object and
# falls back to running the process WITHOUT it, e.g.
#   ERROR: ld.so: object '/x/libamdhip64.so' from LD_PRELOAD cannot be
#   preloaded (cannot open shared object file): ignored.
# For this workload that silent fallback means the probe ran against the
# default HIP runtime, not the one the cell intended -- a false green.
_LDSO_PRELOAD_IGNORED_RE = re.compile(r"from LD_PRELOAD cannot be preloaded")

# ld.so dynamic string tokens are expanded by the loader at exec time; we can't
# resolve them here, so LD_PRELOAD entries containing them are not existence-checked.
_LD_DYNAMIC_TOKEN_RE = re.compile(r"\$(?:ORIGIN|LIB|PLATFORM)\b|\$\{")


@dataclass(frozen=True)
class _ProbeSpec:
    """Build recipe for one probe.

    Attributes:
        source: main ``.cpp`` compiled to the probe executable.
        binary: output executable name.
        kernel_source: optional device ``.cpp`` compiled to a standalone
            code object with ``--genco`` and loaded at runtime (module path).
        kernel_object: output ``.code`` name for ``kernel_source``.
    """

    source: str
    binary: str
    kernel_source: str | None = None
    kernel_object: str | None = None


# Registry of supported probes. Keys are the user-facing ``probe`` config values.
_PROBES: dict[str, _ProbeSpec] = {
    "static": _ProbeSpec("hrx_probe.cpp", "hrx_probe"),
    "module": _ProbeSpec(
        "hrx_probe_module.cpp",
        "hrx_probe_module",
        kernel_source="hrx_probe_module_kernel.cpp",
        kernel_object="hrx_probe_module_kernel.code",
    ),
    "graph_add": _ProbeSpec("hrx_probe_graph_add.cpp", "hrx_probe_graph_add"),
    "graph_setparams": _ProbeSpec(
        "hrx_probe_graph_setparams.cpp", "hrx_probe_graph_setparams"
    ),
    "graph_execsetparams": _ProbeSpec(
        "hrx_probe_graph_execsetparams.cpp", "hrx_probe_graph_execsetparams"
    ),
}

_DEFAULT_PROBE = "module"
_DEFAULT_ARCH = "gfx942"
_DEFAULT_TIMEOUT_SEC = 120

# Env vars the HRX-on cell injects to route the *probe exec* through an
# alternate HIP runtime. The dispatcher merges the cell env into os.environ
# before setup() (the build) runs, so these are stripped from the hipcc build
# env -- otherwise a preloaded HRX libamdhip64.so (or its LD_LIBRARY_PATH)
# would run inside the compiler/link step instead of the probe. run() still
# inherits os.environ, so routing takes effect at the probe's exec().
_RUNTIME_ROUTING_VARS = ("LD_PRELOAD", "LD_LIBRARY_PATH", "HRX_GPU_DRIVER")


def _build_env() -> dict[str, str]:
    """os.environ minus the HRX runtime-routing vars (for the hipcc build)."""
    return {k: v for k, v in os.environ.items() if k not in _RUNTIME_ROUTING_VARS}


def _missing_preload_objects() -> list[str]:
    """Path-like ``LD_PRELOAD`` entries that don't resolve to an existing file.

    A missing ``LD_PRELOAD`` object is NOT fatal to the dynamic loader: ``ld.so``
    prints a warning to stderr and runs the process anyway -- against the
    *default* ``libamdhip64``. For this workload that means an ``hrx_on`` cell
    whose ``LD_PRELOAD`` points at a placeholder or mistyped path would silently
    exercise stock HIP and report a misleading ``FULLY_WORKS``. Surface it
    up-front instead.

    Only *path-like* entries (absolute, or containing a ``/``) are checked;
    bare sonames are resolved via ``LD_LIBRARY_PATH`` / the ldconfig cache, which
    we can't reliably enumerate here (the run()-time stderr backstop catches
    those). Entries with unresolved dynamic tokens ($ORIGIN/$LIB/$PLATFORM) are
    skipped for the same reason.
    """
    raw = os.environ.get("LD_PRELOAD", "").strip()
    if not raw:
        return []
    missing: list[str] = []
    # The loader accepts whitespace and ``:`` as LD_PRELOAD separators.
    for entry in (p for p in re.split(r"[:\s]+", raw) if p):
        if _LD_DYNAMIC_TOKEN_RE.search(entry):
            continue
        if (os.path.isabs(entry) or "/" in entry) and not os.path.isfile(entry):
            missing.append(entry)
    return missing

# Platform-injected config keys that are not HrxWorkload knobs (the dispatcher
# writes `steps` into every workload config; `_aorta_*` carry probe/env
# metadata). Kept silent so the unknown-key guard doesn't spam warnings.
_RESERVED_KEYS = {"steps"}
_KNOWN_KEYS = {
    "probe",
    "gpu_arch",
    "hipcc",
    "build_dir",
    "timeout_sec",
    "keep_build",
}


def _resolve_hipcc(configured: str | None) -> str | None:
    """Return an invocable hipcc path, or None if none is found."""
    candidates = [
        configured,
        os.environ.get("HIPCC"),
        "/opt/rocm/bin/hipcc",
        "hipcc",
    ]
    for cand in candidates:
        if not cand:
            continue
        found = shutil.which(cand) or (cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None)
        if found:
            return found
    return None


# The ROCm kernel-fusion driver node. HIP cannot initialise a GPU context
# without opening it read/write, so its absence/inaccessibility is the
# canonical "no usable ROCm GPU here" signal (a CPU-only host or a container
# started without ``--device=/dev/kfd``).
_GPU_KFD_NODE = "/dev/kfd"


def _gpu_available() -> bool:
    """Best-effort check that a ROCm GPU is reachable by the HIP runtime.

    Returns ``True`` when the KFD compute node (:data:`_GPU_KFD_NODE`) is
    readable+writable by this process. This is a cheap, dependency-free proxy
    for "HIP can init a device" -- it does not spawn ``rocminfo`` and does not
    prove a *specific* arch is present, only that the compute driver node is
    accessible. It is deliberately conservative (checks only the definitive
    node) so a real ROCm host never false-negatives; the point is to catch the
    plainly-no-GPU case in :meth:`HrxWorkload.setup` so the cell is classified
    ``did_not_run`` instead of minting a false ``OUTPUT_NOT_WRITTEN``.

    Split out as a module-level function so GPU-free unit tests can monkeypatch
    it.
    """
    return os.access(_GPU_KFD_NODE, os.R_OK | os.W_OK)


class HrxWorkload(Workload):
    """Run one HRX HIP-launch probe and report its verdict.

    ``workload_config`` keys:
        probe: which launch path to exercise (default ``"module"``); one of
            ``static`` / ``module`` / ``graph_add`` / ``graph_setparams`` /
            ``graph_execsetparams``.
        gpu_arch: ``--offload-arch`` target (default ``"gfx942"``).
        hipcc: explicit hipcc path (default: ``$HIPCC`` /
            ``/opt/rocm/bin/hipcc`` / ``hipcc`` on PATH).
        build_dir: directory for built binaries (default: a temp dir removed
            on cleanup unless ``keep_build`` is set).
        timeout_sec: per-run subprocess timeout (default ``120``).
        keep_build: keep ``build_dir`` after cleanup (default ``False``).
    """

    name: ClassVar[str] = "hrx"

    def _validated_probe(self) -> str:
        for key in self.config:
            if key in _KNOWN_KEYS or key in _RESERVED_KEYS or key.startswith("_aorta_"):
                continue
            log.warning("hrx: ignoring unknown workload_config key %r", key)
        probe = self.config.get("probe", _DEFAULT_PROBE)
        if probe not in _PROBES:
            raise ValueError(
                f"hrx: unknown probe {probe!r}; choose one of {sorted(_PROBES)}"
            )
        return probe

    def setup(self) -> None:
        # Fail fast on a bad routing bundle: a nonexistent LD_PRELOAD object is
        # only a stderr warning to the loader, so an hrx_on cell would otherwise
        # silently run against stock HIP and report a false FULLY_WORKS.
        missing = _missing_preload_objects()
        if missing:
            raise RuntimeError(
                "hrx: LD_PRELOAD names object(s) that do not exist: "
                + ", ".join(repr(m) for m in missing)
                + ". The dynamic loader would ignore these with only a stderr "
                "warning and run the probe against the DEFAULT HIP runtime, so "
                "the cell would silently test stock HIP and report a misleading "
                "FULLY_WORKS. Fix the path(s) in the cell's extra_env -- replace "
                "the /path/to/hrx-root placeholder with your installed HRX "
                "prefix, and make sure LD_PRELOAD/LD_LIBRARY_PATH are absolute."
            )
        self._probe = self._validated_probe()
        self._spec = _PROBES[self._probe]
        self._arch = str(self.config.get("gpu_arch", _DEFAULT_ARCH))
        # Validate here: subprocess.run(..., timeout=<=0) would raise at run()
        # time and be misclassified as an infrastructure failure. Fail fast in
        # setup() with a clear config error instead.
        self._timeout = int(self.config.get("timeout_sec", _DEFAULT_TIMEOUT_SEC))
        if self._timeout <= 0:
            raise ValueError(
                f"hrx: timeout_sec ({self._timeout}) must be > 0"
            )
        # Validate explicitly rather than bool(...): a stray "false"/"0" string
        # from a recipe would coerce to True and silently keep build dirs.
        keep_build = self.config.get("keep_build", False)
        if not isinstance(keep_build, bool):
            raise ValueError(
                f"hrx: keep_build must be a bool, got {type(keep_build).__name__}"
            )
        self._keep_build = keep_build

        hipcc = _resolve_hipcc(self.config.get("hipcc"))
        if hipcc is None:
            raise RuntimeError(
                "hrx: hipcc not found (looked at $HIPCC, /opt/rocm/bin/hipcc, "
                "and PATH). This workload builds HIP probes from source and "
                "requires a ROCm/hipcc toolchain. Run it in a ROCm environment, "
                "or set workload_config.hipcc."
            )
        self._hipcc = hipcc

        # Fail in setup() (not run()) when no GPU is reachable, so a host with
        # hipcc but no accessible device is classified did_not_run (a setup
        # failure, excluded from the failure rate) rather than producing a
        # passed=False result the matrix would miscount as a reproduced bug.
        if not _gpu_available():
            raise RuntimeError(
                f"hrx: no accessible ROCm GPU ({_GPU_KFD_NODE} is not "
                "readable+writable). This workload dispatches a real HIP kernel "
                "and needs a GPU; run it on a ROCm host (or a container started "
                "with --device=/dev/kfd --device=/dev/dri and the render group). "
                "Raising here keeps the cell classified as did_not_run rather "
                "than a false OUTPUT_NOT_WRITTEN."
            )

        build_dir = self.config.get("build_dir")
        if build_dir:
            # Resolve to absolute: _run_hipcc runs with cwd=_KERNELS_DIR and
            # run() with cwd=self._build_dir, so a relative build_dir would send
            # the hipcc -o output under _KERNELS_DIR and make the probe exec look
            # for <build_dir>/<build_dir>/<binary>.
            self._build_dir = Path(build_dir).resolve()
            self._build_dir.mkdir(parents=True, exist_ok=True)
            self._owns_build_dir = False
        else:
            # mkdtemp already returns an absolute path.
            self._build_dir = Path(tempfile.mkdtemp(prefix="aorta-hrx-"))
            self._owns_build_dir = True

        self._binary = self._build()

    def _run_hipcc(self, args: list[str], what: str) -> None:
        cmd = [self._hipcc, *args]
        log.debug("hrx: building %s: %s", what, " ".join(cmd))
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(_KERNELS_DIR), env=_build_env()
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"hrx: hipcc failed to build {what} (exit {proc.returncode}).\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr.strip()}"
            )

    def _build(self) -> Path:
        spec = self._spec
        binary = self._build_dir / spec.binary
        kernel_obj = (
            self._build_dir / spec.kernel_object if spec.kernel_object else None
        )
        # Reuse an already-built probe. setup() runs once per trial, so a run
        # that pins a shared build_dir would otherwise recompile identical
        # artifacts every trial. The default build_dir is a fresh temp dir
        # (unique per setup), so this fast-path only fires for an operator-
        # supplied build_dir. Assumes that dir is not shared across differing
        # probe/arch configs -- the binary name encodes neither -- which is the
        # same assumption a fixed build_dir already implies.
        if binary.is_file() and (kernel_obj is None or kernel_obj.is_file()):
            log.debug("hrx: reusing existing %s in %s", spec.binary, self._build_dir)
            return binary
        # Standalone code object first (module path only), placed next to the
        # binary because the probe loads it from its own exe directory.
        if spec.kernel_source and spec.kernel_object:
            self._run_hipcc(
                [
                    "--genco",
                    f"--offload-arch={self._arch}",
                    str(_KERNELS_DIR / spec.kernel_source),
                    "-o",
                    str(self._build_dir / spec.kernel_object),
                ],
                spec.kernel_object,
            )
        self._run_hipcc(
            [
                "-O2",
                f"--offload-arch={self._arch}",
                str(_KERNELS_DIR / spec.source),
                "-o",
                str(binary),
            ],
            spec.binary,
        )
        return binary

    def run(self) -> WorkloadResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [str(self._binary)],
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

        verdict_match = _VERDICT_RE.search(stdout)
        out0_match = _OUT0_RE.search(stdout)
        verdict = verdict_match.group(1) if verdict_match else None
        out0 = float(out0_match.group(1)) if out0_match else None

        # main_work_started: the probe reached the point of printing a result
        # (it dispatched / read back), as opposed to dying during HIP init.
        main_work_started = out0_match is not None or verdict_match is not None

        # Even a FULLY_WORKS verdict is a false green if the cell requested an
        # LD_PRELOAD that the loader ended up ignoring: the probe then ran
        # against the default HIP runtime, not the intended one. The setup-time
        # check catches nonexistent paths; this catches exists-but-unloadable
        # (wrong arch, missing transitive deps) via ld.so's stderr warning.
        preload_requested = bool(os.environ.get("LD_PRELOAD", "").strip())
        preload_ignored = preload_requested and bool(
            _LDSO_PRELOAD_IGNORED_RE.search(stderr)
        )

        passed = (
            (not timed_out)
            and exit_code == 0
            and verdict == _PASS_VERDICT
            and not preload_ignored
        )

        metrics: dict[str, Any] = {
            "probe": self._probe,
            "gpu_arch": self._arch,
            "verdict": verdict,
            "out0": out0,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "preload_ignored": preload_ignored,
        }

        failure_details: list[dict[str, Any]] = []
        if not passed:
            if preload_ignored:
                hint = (
                    f"probe {self._probe!r}: LD_PRELOAD was set but the loader "
                    "ignored it (see stderr) -- the probe ran against the "
                    "DEFAULT HIP runtime, so this result does NOT reflect the "
                    "intended runtime. Check the LD_PRELOAD object's path, "
                    "architecture, and shared-library dependencies."
                )
            elif timed_out:
                hint = f"probe {self._probe!r} hung (>{self._timeout}s) at launch/sync"
            elif verdict is None:
                hint = (
                    f"probe {self._probe!r} produced no VERDICT line "
                    f"(exit {exit_code}); see stdout/stderr"
                )
            else:
                hint = f"probe {self._probe!r} verdict {verdict} (out[0]={out0}, expected 107)"
            failure_details.append(
                {
                    "probe": self._probe,
                    "verdict": verdict,
                    "out0": out0,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "preload_ignored": preload_ignored,
                    # The probes printf HIP errors / diagnostics to stdout, so
                    # capture both tails -- for a crash before the VERDICT line,
                    # the useful signal is on stdout, not stderr.
                    "stdout_tail": stdout.strip()[-2000:],
                    "stderr_tail": stderr.strip()[-2000:],
                    "hint": hint,
                }
            )

        # No iteration ran unless the probe dispatched and read back. Report 0
        # iterations (and no failing-iteration index) when main work never
        # started, so the matrix elapsed_per_iter fallback -- which divides
        # elapsed_sec by total_iterations *before* the did-not-run suppression
        # (aorta.triage.matrix._extract_step_times) -- can't mint a misleading
        # step time for a setup-only crash/timeout.
        executed = 1 if main_work_started else 0
        return WorkloadResult(
            passed=passed,
            failure_count=0 if passed else 1,
            first_failure_iteration=0 if (not passed and main_work_started) else None,
            failure_details=failure_details,
            total_iterations=executed,
            elapsed_sec=elapsed,
            metrics=metrics,
            main_work_started=main_work_started,
            executed_iterations=executed,
            configured_iterations=1,
        )

    def cleanup(self) -> None:
        if getattr(self, "_owns_build_dir", False) and not getattr(
            self, "_keep_build", False
        ):
            shutil.rmtree(self._build_dir, ignore_errors=True)
