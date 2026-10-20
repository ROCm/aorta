"""``aorta run --sanitizer asan``: run any workload under AddressSanitizer.

ASan has to be in a process from its first instruction -- the host runtime is
preloaded and the instrumented HIP runtime bound at load time -- so a workload
cannot switch it on for itself. The run mode therefore splits ``aorta run`` in
two. The parent stays uninstrumented: it finds the ASan runtime, makes an
evidence directory and runs the same command again with
:func:`asan_run_environment`. The child runs the workload exactly as it would
otherwise, and every process it starts inherits that environment and writes
into the evidence directory: ASan's own reports (``log_path``), the HIP
runtime's dispatch log, and glibc's record of the files each process loaded
(``LD_DEBUG=files``).

The parent then decides what ASan actually checked, kernel by kernel. A
dispatched kernel counts as checked only when the device code that defines it
-- found in a file the same process loaded, a file the workload declared with
:func:`declare_code_objects`, or Triton's kernel cache -- is ASan-instrumented
and xnack+ everywhere aorta can see it defined. A kernel defined nowhere aorta
can read counts as unchecked, so the rule can only under-state coverage.
PyTorch's own kernels are the common case: its device code is built without
ASan and shipped compressed, and the report says so instead of passing.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .asan import (
    _RUNTIME_KERNEL_PREFIX,
    _RUNTIME_KERNELS,
    ASAN_OPTIONS_PINNED,
    DISPATCH_LOG_NAME,
    AsanRuntime,
    KernelCodeObject,
    KernelInventory,
    _function_name,
    asan_environment,
    dispatched_kernels,
    kernel_inventory,
    parse_asan_output,
)
from .models import Finding, Verdict

ENV_ASAN_RUN = "AORTA_ASAN_RUN"
REPORT_NAME = "asan_report.json"
REPORT_SCHEMA = "aorta.asan_run/1"
# ``aorta run`` exits with this when the workload ran clean but ASan could not
# vouch for every kernel it dispatched; 1 stays "something failed".
EXIT_NOT_CHECKED = 3

# File stems inside the evidence directory. ASan and glibc append ``.<pid>``,
# the HIP runtime ``_<pid>``.
_ASAN_LOG = "asan"
_LD_LOG = "ld"
_DECLARED = "code_objects.jsonl"
_TRITON_CACHE = "triton-cache"

# ASAN_OPTIONS splits on these, so log_path cannot contain them.
_UNSAFE_IN_OPTIONS = re.compile(r"[\s:,'\"]")

# Preloaded by bare name after the host runtime. PyTorch reaches its bundled
# copies through DT_RPATH=$ORIGIN, which wins over LD_LIBRARY_PATH; an object
# preloaded under the name its DT_NEEDED asks for is reused instead, so the
# instrumented build is the one every library binds to.
_PRELOAD_BY_NAME = ("libamdhip64.so", "libhsa-runtime64.so")

# Settings the run mode owns. A cell's environment may repeat one, but changing
# it would silently blind the check (an HRX cell's LD_PRELOAD replacing the
# instrumented HIP runtime, an ASAN_OPTIONS without log_path).
OWNED_SETTINGS = frozenset(
    {
        "AMDGCN_USE_BUFFER_OPS",
        "AMD_LOG_LEVEL",
        "AMD_LOG_LEVEL_FILE",
        "AMD_LOG_MASK",
        "ASAN_OPTIONS",
        "ASAN_SYMBOLIZER_PATH",
        "HSA_XNACK",
        "LD_DEBUG",
        "LD_DEBUG_OUTPUT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
        "PYTORCH_NO_HIP_MEMORY_CACHING",
        "TRITON_CACHE_DIR",
        "TRITON_ENABLE_ASAN",
        ENV_ASAN_RUN,
    }
)


# --------------------------------------------------------------------------- #
# The child environment
# --------------------------------------------------------------------------- #


def torch_library_dir() -> Path | None:
    """PyTorch's bundled library directory, found without importing torch."""

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "lib"
        if candidate.is_dir():
            return candidate
    return None


def _preload_entries(value: str) -> list[str]:
    return [entry for entry in re.split(r"[:\s]+", value) if entry]


def asan_run_environment(
    base: Mapping[str, str],
    runtime: AsanRuntime,
    evidence_dir: Path,
    *,
    torch_lib: Path | None = None,
) -> dict[str, str]:
    """The environment every process of an ASan run starts with.

    On top of :func:`~.asan.asan_environment`: ASan reports go to per-process
    files in ``evidence_dir`` (a workload is free to discard its children's
    stderr, and aorta's trial workers do not capture theirs), the HIP and HSA
    runtimes are preloaded by name, and the loader records every file each
    process loads. ``torch_lib`` goes on the library path because ASan's
    ``dlopen`` interceptor breaks ``$ORIGIN`` for the libraries PyTorch opens
    itself. PyTorch's caching allocator is turned off (a block it hands out
    again is never poisoned, so overflows inside it go unseen), Triton compiles
    with ASan and without buffer instructions (which ASan does not check), and
    Triton's cache is moved into the evidence so its kernels can be inspected.
    """

    # Absolute, because a workload may run its children from another directory
    # (hrx execs its probe from its build directory).
    evidence_dir = evidence_dir.absolute()
    if _UNSAFE_IN_OPTIONS.search(str(evidence_dir)):
        raise ValueError(
            f"evidence directory {str(evidence_dir)!r} contains a character "
            "ASAN_OPTIONS cannot carry (whitespace, ':', ',' or a quote); choose a "
            "results directory without one"
        )
    env = asan_environment(base, runtime, dispatch_log=evidence_dir / DISPATCH_LOG_NAME)
    env["ASAN_OPTIONS"] = f"{ASAN_OPTIONS_PINNED}:log_path={evidence_dir / _ASAN_LOG}"

    by_name = [
        name
        for name in _PRELOAD_BY_NAME
        if any((directory / name).exists() for directory in runtime.rocm_lib_dirs)
    ]
    inherited_preload = _preload_entries(base.get("LD_PRELOAD", ""))
    env["LD_PRELOAD"] = ":".join([str(runtime.host_runtime), *by_name, *inherited_preload])

    library_path = [str(directory) for directory in runtime.rocm_lib_dirs]
    library_path.append(str(runtime.host_runtime.parent))
    if torch_lib is not None:
        library_path.append(str(torch_lib))
    if inherited := base.get("LD_LIBRARY_PATH", ""):
        library_path.append(inherited)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_path)

    env.update(
        {
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
            "PYTORCH_NO_HIP_MEMORY_CACHING": "1",
            "TRITON_ENABLE_ASAN": "1",
            "AMDGCN_USE_BUFFER_OPS": "0",
            "TRITON_CACHE_DIR": str(evidence_dir / _TRITON_CACHE),
            "LD_DEBUG": "files",
            "LD_DEBUG_OUTPUT": str(evidence_dir / _LD_LOG),
            ENV_ASAN_RUN: str(evidence_dir),
        }
    )
    return env


def _asan_runtime_loaded() -> bool:
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "libclang_rt.asan" in maps


def active_asan_run(environ: Mapping[str, str] | None = None) -> Path | None:
    """The evidence directory when this process is the child of an ASan run.

    Needs both the marker and the ASan runtime actually mapped into this
    process: the marker alone would let a run claim ASan without the runtime
    that does the checking.
    """

    env = os.environ if environ is None else environ
    evidence = env.get(ENV_ASAN_RUN, "").strip()
    if not evidence or not _asan_runtime_loaded():
        return None
    path = Path(evidence)
    return path if path.is_dir() else None


def owned_setting_conflicts(
    overlay: Mapping[str, str], environ: Mapping[str, str] | None = None
) -> list[str]:
    """Settings in ``overlay`` that would change what the run mode set."""

    env = os.environ if environ is None else environ
    return sorted(
        key for key, value in overlay.items() if key in OWNED_SETTINGS and env.get(key) != value
    )


def _inventory_record(inventory: KernelInventory) -> dict[str, object]:
    return {
        "path": inventory.path,
        "compressed_bundles": inventory.compressed_bundles,
        "unreadable": inventory.unreadable,
        "code_objects": [
            {
                "triple": code_object.triple,
                "processor": code_object.processor,
                "features": list(code_object.features),
                "asan_instrumented": code_object.asan_instrumented,
                "kernels": list(code_object.kernels),
            }
            for code_object in inventory.code_objects
        ],
    }


def _inventory_from_record(record: Mapping[str, object]) -> KernelInventory | None:
    try:
        return KernelInventory(
            path=str(record["path"]),
            compressed_bundles=int(record["compressed_bundles"]),  # type: ignore[call-overload]
            unreadable=record["unreadable"] if isinstance(record["unreadable"], str) else None,
            code_objects=tuple(
                KernelCodeObject(
                    triple=str(item["triple"]),
                    processor=str(item["processor"]),
                    features=tuple(str(feature) for feature in item["features"]),
                    asan_instrumented=item["asan_instrumented"] is True,
                    kernels=tuple(str(kernel) for kernel in item["kernels"]),
                )
                for item in record["code_objects"]  # type: ignore[union-attr]
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def declare_code_objects(paths: Iterable[Path | str]) -> None:
    """Record device code a workload built, for the run's coverage check.

    Two reasons a workload has to say so. ``hipModuleLoad`` reads a code object
    without going through the dynamic loader, so the loader's record never
    lists it. And a workload may delete its build before the run ends -- hrx
    removes its temporary build directory in ``cleanup()`` -- so what each file
    defines is read here, while it exists, rather than after the run. A no-op
    outside an ASan run.
    """

    evidence = os.environ.get(ENV_ASAN_RUN, "").strip()
    if not evidence or not Path(evidence).is_dir():
        return
    lines = [
        json.dumps(_inventory_record(kernel_inventory(Path(path).resolve())), sort_keys=True) + "\n"
        for path in paths
    ]
    if not lines:
        return
    descriptor = os.open(
        Path(evidence) / _DECLARED,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
        0o644,
    )
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.writelines(lines)


# --------------------------------------------------------------------------- #
# Reading the evidence
# --------------------------------------------------------------------------- #


def _per_process(evidence_dir: Path, stem: str) -> dict[int, str]:
    """The files ``<stem>.<pid>`` / ``<stem>_<pid>`` in ``evidence_dir``, by pid."""

    pattern = re.compile(re.escape(stem) + r"[._](\d+)")
    found: dict[int, str] = {}
    try:
        entries = sorted(evidence_dir.iterdir())
    except OSError:
        return found
    for entry in entries:
        match = pattern.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_file():
            continue
        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pid = int(match.group(1))
        found[pid] = f"{found[pid]}\n{text}" if pid in found else text
    return found


_LD_ENTRY = re.compile(r"^\s*\d+:\s*(?P<body>.*?)\s*$")


@dataclass(frozen=True)
class LoadedFiles:
    """What the dynamic loader recorded one process loading."""

    programs: tuple[str, ...]
    objects: tuple[str, ...]


def loaded_files(ld_log: str) -> LoadedFiles:
    """Parse one process's ``LD_DEBUG=files`` output.

    ``calling init:`` names every loaded object that has initialisers, with its
    resolved path -- which includes every HIP fat binary, since that is how one
    registers its device code -- and ``initialize program:`` names the
    executable. An ``exec`` without ``fork`` keeps the pid, so one log can name
    several programs.
    """

    programs: list[str] = []
    objects: list[str] = []
    for line in ld_log.splitlines():
        match = _LD_ENTRY.match(line)
        if match is None:
            continue
        body = match.group("body")
        if body.startswith("initialize program: "):
            programs.append(body.removeprefix("initialize program: "))
        elif body.startswith("calling init: "):
            objects.append(body.removeprefix("calling init: "))
        elif body.startswith("file=/") and (end := body.find(" [")) > 0:
            objects.append(body[len("file=") : end])
    return LoadedFiles(tuple(dict.fromkeys(programs)), tuple(dict.fromkeys(objects)))


def _declared_inventories(evidence_dir: Path) -> dict[str, list[KernelInventory]]:
    """What each declared file defined when it was declared, by path."""

    manifest = evidence_dir / _DECLARED
    if manifest.is_symlink() or not manifest.is_file():
        return {}
    try:
        lines = manifest.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    declared: dict[str, list[KernelInventory]] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        inventory = _inventory_from_record(record)
        if inventory is not None and inventory.path.startswith("/"):
            declared.setdefault(inventory.path, []).append(inventory)
    return declared


def _jit_objects(evidence_dir: Path) -> list[str]:
    cache = evidence_dir / _TRITON_CACHE
    if cache.is_symlink() or not cache.is_dir():
        return []
    try:
        return sorted(str(path) for path in cache.rglob("*.hsaco") if path.is_file())
    except OSError:
        return []


def _normalized(signature: str) -> str:
    return re.sub(r"\s+", "", signature)


def _is_runtime_kernel(signature: str) -> bool:
    name = _function_name(signature)
    return name in _RUNTIME_KERNELS or name.startswith(_RUNTIME_KERNEL_PREFIX)


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelCoverage:
    """Whether ASan checked one dispatched kernel, and why."""

    signature: str
    checked: bool
    reason: str
    defined_in: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "checked": self.checked,
            "reason": self.reason,
            "defined_in": list(self.defined_in),
        }


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    programs: tuple[str, ...]
    kernels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"pid": self.pid, "programs": list(self.programs), "kernels": list(self.kernels)}


@dataclass(frozen=True)
class AsanRunReport:
    """What one ASan run found and what it could vouch for."""

    verdict: Verdict
    reason: str
    returncode: int | None
    findings: tuple[Finding, ...]
    kernels: tuple[KernelCoverage, ...]
    processes: tuple[ProcessRecord, ...]
    opaque_device_code: tuple[str, ...]
    evidence_dir: str
    command: tuple[str, ...] = ()
    runtime: Mapping[str, object] | None = None

    @property
    def unchecked(self) -> tuple[KernelCoverage, ...]:
        return tuple(kernel for kernel in self.kernels if not kernel.checked)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REPORT_SCHEMA,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "returncode": self.returncode,
            "command": list(self.command),
            "findings": [finding.to_dict() for finding in self.findings],
            "kernels": {
                "dispatched": len(self.kernels),
                "checked": sum(1 for kernel in self.kernels if kernel.checked),
                "detail": [kernel.to_dict() for kernel in self.kernels],
            },
            "processes": [process.to_dict() for process in self.processes],
            "opaque_device_code": list(self.opaque_device_code),
            "runtime": dict(self.runtime) if self.runtime is not None else None,
            "evidence_dir": self.evidence_dir,
        }

    def summary(self) -> list[str]:
        label = {
            Verdict.PASS: "PASS",
            Verdict.FAIL: "FAIL",
            Verdict.ERROR: "ERROR",
            Verdict.NOT_CHECKED: "NOT CHECKED",
        }.get(self.verdict, self.verdict.value.upper())
        lines = [f"ASan: {label} -- {self.reason}"]
        for finding in self.findings[:3]:
            metadata = dict(finding.metadata)
            where = metadata.get("location") or finding.kernel_name or "unknown location"
            lines.append(f"  {finding.code} ({metadata.get('side', 'host')}) at {where}")
        if self.verdict is not Verdict.FAIL:
            unchecked = self.unchecked
            for kernel in unchecked[:3]:
                lines.append(f"  unchecked: {kernel.signature} -- {kernel.reason}")
            if len(unchecked) > 3:
                lines.append(f"  ... and {len(unchecked) - 3} more unchecked kernel(s)")
        lines.append(f"Evidence: {self.evidence_dir}")
        return lines


def _finding_summary(finding: Finding) -> str:
    where = dict(finding.metadata).get("location") or finding.kernel_name
    return f"{finding.code} at {where}" if where else str(finding.code)


def evaluate_asan_run(
    evidence_dir: Path,
    *,
    returncode: int | None,
    command: Sequence[str] = (),
    runtime: AsanRuntime | None = None,
) -> AsanRunReport:
    """Read an ASan run's evidence directory and decide what it shows.

    A memory error anywhere fails the run. Otherwise anything that means a
    detection could have been lost -- the stock runtime's hostcall abort, the
    host runtime not first, two runtimes, ASan failing to start -- is an error.
    Only then does coverage decide between pass and not checked.
    """

    reports = _per_process(evidence_dir, _ASAN_LOG)
    hip_logs = _per_process(evidence_dir, DISPATCH_LOG_NAME)
    ld_logs = _per_process(evidence_dir, _LD_LOG)

    findings: list[Finding] = []
    lost = not_first = incompatible = init_failed = False
    for _pid, text in sorted(reports.items()):
        parsed = parse_asan_output(text)
        findings.extend(parsed.findings)
        lost |= parsed.device_report_lost
        not_first |= parsed.runtime_not_preloaded
        incompatible |= parsed.incompatible_runtimes
        init_failed |= parsed.init_failed
    for text in hip_logs.values():
        # The runtime's own messages, the stock runtime's hostcall abort among
        # them, go to this log once AMD_LOG_LEVEL_FILE is set.
        lost |= parse_asan_output(text).device_report_lost

    declared = _declared_inventories(evidence_dir)
    shared = [*declared, *_jit_objects(evidence_dir)]
    inventories: dict[str, list[KernelInventory]] = dict(declared)

    def inventory(path: str) -> list[KernelInventory]:
        if path not in inventories:
            inventories[path] = [kernel_inventory(Path(path))]
        return inventories[path]

    coverage: dict[str, KernelCoverage] = {}
    processes: list[ProcessRecord] = []
    opaque: dict[str, None] = {}
    for pid, hip_log in sorted(hip_logs.items()):
        signatures = tuple(
            dict.fromkeys(
                signature
                for signature in dispatched_kernels(hip_log)
                if not _is_runtime_kernel(signature)
            )
        )
        loaded = loaded_files(ld_logs.get(pid, ""))
        processes.append(ProcessRecord(pid, loaded.programs, signatures))
        if not signatures:
            continue
        definitions: dict[str, list[tuple[str, bool, str]]] = {}
        process_opaque: dict[str, None] = {}
        for path in dict.fromkeys((*loaded.programs, *loaded.objects, *shared)):
            # A relative program path names the executable from the process's
            # own working directory, which the loader does not record.
            if not path.startswith("/"):
                continue
            for found in inventory(path):
                if found.compressed_bundles:
                    process_opaque.setdefault(path, None)
                for code_object in found.code_objects:
                    for kernel in code_object.kernels:
                        definitions.setdefault(_normalized(kernel), []).append(
                            (path, code_object.checked, code_object.triple)
                        )
        for path in process_opaque:
            opaque.setdefault(path, None)
        for signature in signatures:
            defined = definitions.get(_normalized(signature), [])
            if not defined:
                reason = "defined in no device code aorta could read from this process's files"
                if process_opaque:
                    reason += (
                        f" (it also loaded device code aorta cannot open: "
                        f"{len(process_opaque)} file(s) with compressed bundles)"
                    )
                decision = KernelCoverage(signature, False, reason)
            elif all(checked for _, checked, _ in defined):
                decision = KernelCoverage(
                    signature,
                    True,
                    "ASan-instrumented, xnack+ device code",
                    tuple(dict.fromkeys(path for path, _, _ in defined)),
                )
            else:
                plain = next((path, triple) for path, checked, triple in defined if not checked)
                decision = KernelCoverage(
                    signature,
                    False,
                    f"defined in device code built without ASan or xnack+: {plain[0]} [{plain[1]}]",
                    tuple(dict.fromkeys(path for path, _, _ in defined)),
                )
            previous = coverage.get(signature)
            if previous is None or (previous.checked and not decision.checked):
                coverage[signature] = decision

    kernels = tuple(coverage.values())
    unchecked = [kernel for kernel in kernels if not kernel.checked]
    if findings:
        verdict = Verdict.FAIL
        reason = f"{len(findings)} memory error(s); first: {_finding_summary(findings[0])}"
    elif lost:
        verdict = Verdict.ERROR
        reason = (
            "a device-side detection was lost: a process aborted with 'Hostcall: no "
            "handler found for service ID 4', so it ran the stock HIP runtime rather "
            "than the ASan build"
        )
    elif not_first:
        verdict = Verdict.ERROR
        reason = "the ASan runtime was not the first library loaded in a process"
    elif incompatible:
        verdict = Verdict.ERROR
        reason = "a process loaded two incompatible ASan runtimes"
    elif init_failed:
        verdict = Verdict.ERROR
        reason = "ASan failed to start in a process"
    elif not hip_logs:
        verdict = Verdict.NOT_CHECKED
        reason = "no process wrote a HIP runtime log, so no GPU kernel ran under the check"
    elif not kernels:
        verdict = Verdict.NOT_CHECKED
        reason = "no workload kernel was dispatched"
    elif unchecked:
        verdict = Verdict.NOT_CHECKED
        reason = (
            f"no memory error observed, but {len(unchecked)} of {len(kernels)} dispatched "
            "kernel(s) ran from device code ASan does not check"
        )
    else:
        verdict = Verdict.PASS
        reason = (
            f"no memory error in {len(kernels)} dispatched kernel(s), all from "
            "ASan-instrumented device code"
        )

    return AsanRunReport(
        verdict=verdict,
        reason=reason,
        returncode=returncode,
        findings=tuple(findings),
        kernels=kernels,
        processes=tuple(processes),
        opaque_device_code=tuple(opaque),
        evidence_dir=str(evidence_dir),
        command=tuple(command),
        runtime=_runtime_record(runtime) if runtime is not None else None,
    )


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _runtime_record(runtime: AsanRuntime) -> dict[str, object]:
    return {
        "rocm_asan_lib_dirs": [str(directory) for directory in runtime.rocm_lib_dirs],
        "hip_runtime": str(runtime.hip_runtime),
        "hip_runtime_sha256": _sha256(runtime.hip_runtime),
        "host_runtime": str(runtime.host_runtime),
        "symbolizer": str(runtime.symbolizer) if runtime.symbolizer is not None else None,
        "asan_options": ASAN_OPTIONS_PINNED,
    }


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def new_evidence_dir(root: Path) -> Path:
    """A fresh, empty directory under ``root`` for one run's evidence."""

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    evidence = root / f"{stamp}-{os.getpid()}"
    evidence.mkdir()
    return evidence


def write_report(report: AsanRunReport, evidence_dir: Path) -> Path:
    """Write ``report`` beside the evidence, refusing to follow or reuse a name."""

    path = evidence_dir / REPORT_NAME
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


@dataclass(frozen=True)
class AsanRunOutcome:
    returncode: int
    report: AsanRunReport
    report_path: Path


def run_under_asan(
    command: Sequence[str],
    *,
    evidence_dir: Path,
    runtime: AsanRuntime,
    environ: Mapping[str, str] | None = None,
) -> AsanRunOutcome:
    """Run ``command`` with the ASan run environment and judge what it left."""

    base = os.environ if environ is None else environ
    env = asan_run_environment(base, runtime, evidence_dir, torch_lib=torch_library_dir())
    completed = subprocess.run(list(command), env=env, check=False)
    report = evaluate_asan_run(
        evidence_dir, returncode=completed.returncode, command=command, runtime=runtime
    )
    return AsanRunOutcome(completed.returncode, report, write_report(report, evidence_dir))
