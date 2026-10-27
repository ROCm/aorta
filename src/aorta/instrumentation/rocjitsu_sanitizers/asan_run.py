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
and xnack+ everywhere aorta can see it defined, and the process loaded nothing
aorta cannot read. A kernel defined nowhere aorta can read counts as unchecked,
so the rule can only under-state coverage.
PyTorch's own kernels are the common case: its device code is built without
ASan and shipped compressed, and the report says so instead of passing.

Before any of that, the parent checks the node (:func:`check_node_health`): on
a node whose GPU page migration corrupts memory, ASan's shadow memory is
corrupted with it, and nothing the workload's run reports could be trusted.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aorta.run import _fsafe

from ..rocm_paths import resolve_rocm_roots
from .asan import (
    _LOADER_STEM,
    _REPORT_STEM,
    _RUNTIME_KERNEL_PREFIX,
    _RUNTIME_KERNELS,
    _UNSAFE_IN_OPTIONS,
    ASAN_OPTIONS_PINNED,
    DISPATCH_LOG_NAME,
    AsanRuntime,
    KernelCodeObject,
    KernelInventory,
    _function_name,
    asan_environment,
    dispatched_kernels,
    kernel_inventory,
    normalized_signature,
    parse_asan_output,
    uninterpreted_reports,
)
from .execution import ProcessResult, run_argv
from .models import Finding, Verdict

ENV_ASAN_RUN = "AORTA_ASAN_RUN"
REPORT_NAME = "asan_report.json"
REPORT_SCHEMA = "aorta.asan_run/1"
# ``aorta run`` exits with this when the workload ran clean but ASan could not
# vouch for every kernel it dispatched; 1 stays "something failed".
EXIT_NOT_CHECKED = 3

# File stems inside the evidence directory. ASan and glibc append ``.<pid>``,
# the HIP runtime ``_<pid>``.
_ASAN_LOG = _REPORT_STEM
_LD_LOG = _LOADER_STEM
_DECLARED = "code_objects.jsonl"
_TRITON_CACHE = "triton-cache"

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

# Which GPUs a process sees. The node-health canary checks every GPU the parent sees,
# so a cell may not point its workload at others.
GPU_SELECTION = frozenset(
    {"CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}
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
    """Settings in ``overlay`` that would change what the run mode set or checked."""

    env = os.environ if environ is None else environ
    return sorted(
        key
        for key, value in overlay.items()
        if (key in OWNED_SETTINGS or key in GPU_SELECTION) and env.get(key) != value
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


def declare_code_objects(paths: Iterable[Path | str], *, loaded_by: Path | str) -> None:
    """Record device code a workload built, for the run's coverage check.

    Two reasons a workload has to say so. ``hipModuleLoad`` reads a code object
    without going through the dynamic loader, so the loader's record never
    lists it. And a workload may delete its build before the run ends -- hrx
    removes its temporary build directory in ``cleanup()`` -- so what each file
    defines is read here, while it exists, rather than after the run.
    ``loaded_by`` is the program that loads the files: the declaration is
    evidence for a process running that program and for no other. A no-op
    outside an ASan run.
    """

    evidence = os.environ.get(ENV_ASAN_RUN, "").strip()
    if not evidence or not Path(evidence).is_dir():
        return
    program = str(Path(loaded_by).absolute())
    lines = [
        json.dumps(
            {**_inventory_record(kernel_inventory(Path(path).resolve())), "loaded_by": program},
            sort_keys=True,
        )
        + "\n"
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


def _per_process(evidence_dir: Path, stem: str, unread: dict[str, str]) -> dict[int, str]:
    """The files ``<stem>.<pid>`` / ``<stem>_<pid>`` in ``evidence_dir``, by pid.

    A file that is there but cannot be read goes into ``unread`` with the reason: it
    could hold the evidence that decides the run, so it is not the same as no file.
    """

    pattern = re.compile(re.escape(stem) + r"[._](\d+)")
    found: dict[int, str] = {}
    try:
        entries = sorted(evidence_dir.iterdir())
    except OSError as error:
        unread.setdefault(str(evidence_dir), error.strerror or type(error).__name__)
        return found
    for entry in entries:
        match = pattern.fullmatch(entry.name)
        if match is None:
            continue
        if entry.is_symlink() or not entry.is_file():
            unread[entry.name] = "a link" if entry.is_symlink() else "not a regular file"
            continue
        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            unread[entry.name] = error.strerror or type(error).__name__
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


def _declared_inventories(
    evidence_dir: Path, unread: dict[str, str]
) -> dict[str, dict[str, list[KernelInventory]]]:
    """What each declared file defined when it was declared, by the program that loads it.

    Keyed both as the program was named and as it resolves, since the loader records a
    program the way it was invoked. A manifest or a line aorta cannot read goes into
    ``unread``: the declaration it holds could be the one saying a file is opaque.
    """

    manifest = evidence_dir / _DECLARED
    if not manifest.is_symlink() and not manifest.exists():
        return {}
    if manifest.is_symlink() or not manifest.is_file():
        unread[_DECLARED] = "a link" if manifest.is_symlink() else "not a regular file"
        return {}
    try:
        lines = manifest.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        unread[_DECLARED] = error.strerror or type(error).__name__
        return {}
    declared_by: dict[str, dict[str, list[KernelInventory]]] = {}
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        program = record.get("loaded_by") if isinstance(record, dict) else None
        inventory = _inventory_from_record(record) if isinstance(program, str) else None
        # declare_code_objects writes both paths absolute.
        if inventory is None or not (inventory.path.startswith("/") and program.startswith("/")):
            unread.setdefault(_DECLARED, f"line {number} does not parse")
            continue
        for key in dict.fromkeys((program, os.path.realpath(program))):
            declared_by.setdefault(key, {}).setdefault(inventory.path, []).append(inventory)
    return declared_by


def _opaque_summary(opaque: Mapping[str, str]) -> str:
    counts = Counter(opaque.values())
    return f"{len(opaque)} file(s): " + ", ".join(f"{n} {why}" for why, n in counts.items())


def _jit_objects(evidence_dir: Path) -> list[str]:
    cache = evidence_dir / _TRITON_CACHE
    if cache.is_symlink() or not cache.is_dir():
        return []
    try:
        return sorted(str(path) for path in cache.rglob("*.hsaco") if path.is_file())
    except OSError:
        return []


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


def _finding_line(finding: Finding) -> str:
    metadata = dict(finding.metadata)
    where = metadata.get("location") or finding.kernel_name or "unknown location"
    return f"{finding.code} ({metadata.get('side', 'host')}) at {where}"


class NodeHealthStatus(str, Enum):
    """What the node-health canary established before the workload ran."""

    HEALTHY = "healthy"
    FAILED = "failed"
    NOT_VERIFIED = "not_verified"


@dataclass(frozen=True)
class NodeHealth:
    """Whether this node's ASan results can be trusted, from :func:`check_node_health`.

    ``failed`` is any canary that did not come back clean -- a memory error from a
    program whose every access is in bounds, data that GPU page migration corrupted, a
    crash, a hang, a kernel ASan could not be shown to check -- because a workload run
    on that node cannot be told apart from the node's own fault. ``not_verified`` means
    no canary could run, so nothing either way.
    """

    status: NodeHealthStatus
    reason: str
    evidence_dir: str | None = None
    seconds: float | None = None
    gpus: int | None = None
    canary: AsanRunReport | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "evidence_dir": self.evidence_dir,
            "seconds": round(self.seconds, 2) if self.seconds is not None else None,
            "gpus": self.gpus,
            "canary": self.canary.to_dict() if self.canary is not None else None,
        }

    def summary(self) -> list[str]:
        if self.status is NodeHealthStatus.HEALTHY:
            took = f" in {self.seconds:.1f} s" if self.seconds is not None else ""
            return [f"Node health: verified{took} -- {self.reason}"]
        if self.status is NodeHealthStatus.NOT_VERIFIED:
            return [f"Node health: NOT VERIFIED -- {self.reason}"]
        lines = [f"Node health: FAILED -- the canary's evidence is in {self.evidence_dir}"]
        if self.canary is not None:
            lines += [f"  canary: {_finding_line(finding)}" for finding in self.canary.findings[:2]]
        return lines


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
    node_health: NodeHealth | None = None

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
            "node_health": self.node_health.to_dict() if self.node_health is not None else None,
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
        if self.node_health is not None:
            lines += self.node_health.summary()
        lines += [f"  {_finding_line(finding)}" for finding in self.findings[:3]]
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
    node_health: NodeHealth | None = None,
) -> AsanRunReport:
    """Read an ASan run's evidence directory and decide what it shows.

    A memory error anywhere fails the run. Otherwise anything that means a
    detection could have been lost -- the stock runtime's hostcall abort, the
    host runtime not first, two runtimes, ASan failing to start -- is an error.
    Only then does coverage decide between pass and not checked.

    ``node_health`` comes before all of it: on a node that failed its health
    check the run is an error whatever its evidence says, and on one nothing
    could verify, a run that would pass is not checked instead.
    """

    unread: dict[str, str] = {}
    reports = _per_process(evidence_dir, _ASAN_LOG, unread)
    hip_logs = _per_process(evidence_dir, DISPATCH_LOG_NAME, unread)
    ld_logs = _per_process(evidence_dir, _LD_LOG, unread)

    findings: list[Finding] = []
    lost = not_first = incompatible = init_failed = False
    for text in reports.values():
        parsed = parse_asan_output(text)
        findings.extend(parsed.findings)
        lost |= parsed.device_report_lost
        not_first |= parsed.runtime_not_preloaded
        incompatible |= parsed.incompatible_runtimes
        init_failed |= parsed.init_failed
    uninterpreted = uninterpreted_reports(reports, _ASAN_LOG)
    for text in hip_logs.values():
        # The runtime's own messages, the stock runtime's hostcall abort among
        # them, go to this log once AMD_LOG_LEVEL_FILE is set.
        lost |= parse_asan_output(text).device_report_lost

    declared_by = _declared_inventories(evidence_dir, unread)
    jit = _jit_objects(evidence_dir)
    inventories: dict[str, KernelInventory] = {}

    def inventory(path: str) -> KernelInventory:
        if path not in inventories:
            inventories[path] = kernel_inventory(Path(path))
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
        process_opaque: dict[str, str] = {}
        # exec keeps the pid, and the loader log does not say which program loaded
        # what, so one program's declarations could describe another's files.
        if len(loaded.programs) > 1:
            for program in loaded.programs:
                process_opaque[program] = "run by one process in turn (exec)"
        # A declaration is evidence only for the program that declared it, and Triton's
        # cache only for a process that loaded Triton, so a kernel another process
        # declared or compiled is never taken as proof of what this one ran.
        own: dict[str, list[KernelInventory]] = {}
        for program in loaded.programs:
            for key in dict.fromkeys((program, os.path.realpath(program))):
                for path, found in declared_by.get(key, {}).items():
                    snapshots = own.setdefault(path, [])
                    snapshots.extend(item for item in found if item not in snapshots)
        if any(Path(path).name.startswith("libtriton") for path in loaded.objects):
            for path in jit:
                own.setdefault(path, [inventory(path)])
        for path in dict.fromkeys((*loaded.programs, *loaded.objects, *own)):
            # A relative path names a file from the process's own working directory,
            # which the loader does not record, so aorta cannot tell what it defined.
            if not path.startswith("/"):
                process_opaque.setdefault(path, "not an absolute path")
                continue
            for found in own.get(path) or [inventory(path)]:
                if found.unreadable is not None:
                    process_opaque.setdefault(path, "unreadable")
                if found.compressed_bundles:
                    process_opaque.setdefault(path, "with compressed bundles")
                for code_object in found.code_objects:
                    for kernel in code_object.kernels:
                        definitions.setdefault(normalized_signature(kernel), []).append(
                            (path, code_object.checked, code_object.triple)
                        )
        for path in process_opaque:
            opaque.setdefault(path, None)
        for signature in signatures:
            defined = definitions.get(normalized_signature(signature), [])
            if not defined:
                reason = "defined in no device code aorta could read from this process's files"
                if process_opaque:
                    reason += (
                        " (it also loaded files aorta cannot open: "
                        f"{_opaque_summary(process_opaque)})"
                    )
                decision = KernelCoverage(signature, False, reason)
            elif process_opaque and all(checked for _, checked, _ in defined):
                # A file aorta cannot open could hold the copy that actually ran.
                decision = KernelCoverage(
                    signature,
                    False,
                    "defined in ASan-instrumented code, but the process also loaded files "
                    f"aorta cannot open ({_opaque_summary(process_opaque)}), which could "
                    "define the copy that ran",
                    tuple(dict.fromkeys(path for path, _, _ in defined)),
                )
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
    elif uninterpreted:
        verdict = Verdict.ERROR
        reason = "ASan wrote a report aorta could not interpret: " + ", ".join(
            f"{name} ({first})" for name, first in uninterpreted.items()
        )
    elif unread:
        verdict = Verdict.ERROR
        reason = "aorta could not read the run's evidence, which could hold a report: " + ", ".join(
            f"{name} ({why})" for name, why in unread.items()
        )
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
    elif returncode != 0:
        # As for a recipe: a run that did not finish cleanly and reported nothing cannot
        # vouch for what it never reached, or for a report it died before delivering.
        verdict = Verdict.ERROR
        status = "reported no exit status" if returncode is None else f"exited {returncode}"
        reason = f"the run {status} without an ASan report, so ASan cannot vouch for it"
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

    if node_health is not None and node_health.status is NodeHealthStatus.FAILED:
        verdict = Verdict.ERROR
        reason = (
            "node health check failed, so ASan results on this node cannot be trusted: "
            f"{node_health.reason}"
        )
    elif (
        node_health is not None
        and node_health.status is NodeHealthStatus.NOT_VERIFIED
        and verdict is Verdict.PASS
    ):
        verdict = Verdict.NOT_CHECKED
        reason = (
            f"{reason}, but nothing verified that ASan works on this node: {node_health.reason}"
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
        node_health=node_health,
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
# Node health
# --------------------------------------------------------------------------- #

CANARY_SOURCE = Path(__file__).with_name("asan_canary.hip")
NODE_HEALTH_DIR = "node-health"
KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")
_CANARY_BINARY = "asan_canary"
_CANARY_BUILD_FLAGS = ("-O1", "-g", "-fsanitize=address", "-shared-libsan")
_BUILD_TIMEOUT = 300.0
# About ten seconds on a sound 8-GPU node.
_CANARY_TIMEOUT = 120.0
_GFX_TARGET_VERSION = re.compile(r"^gfx_target_version (?P<version>\d+)\s*$", re.MULTILINE)
_CANARY_FAILED_CHECK = re.compile(
    r"^AORTA_ASAN_CANARY gpu=(?P<gpu>\d+) check=(?P<check>\S+)"
    r"(?: bad_words=(?P<bad>\d+) of=(?P<of>\d+))? FAIL\s*$",
    re.MULTILINE,
)
_CANARY_PASSED = re.compile(r"^AORTA_ASAN_CANARY verdict=PASS gpus=(?P<gpus>\d+)\s*$", re.MULTILINE)
_CANARY_HIP_ERROR = re.compile(r"^AORTA_ASAN_CANARY hip_error (?P<what>.+?)\s*$", re.MULTILINE)
_SHADOW_SUSPECT = "GPU page migration on this node is likely corrupting ASan's shadow memory"


def _gfx_target(version: int) -> str:
    """The target a KFD ``gfx_target_version`` names: 90500 is gfx950, 90010 gfx90a."""

    return f"gfx{version // 10000}{version // 100 % 100:x}{version % 100:x}"


def gpu_targets(topology: Path = KFD_TOPOLOGY) -> tuple[str, ...]:
    """The GPU targets the kernel driver lists, read from sysfs without touching a GPU."""

    try:
        nodes = sorted(topology.iterdir())
    except OSError:
        return ()
    targets: dict[str, None] = {}
    for node in nodes:
        try:
            properties = (node / "properties").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _GFX_TARGET_VERSION.search(properties)
        # CPU nodes report version 0.
        if match is not None and int(match.group("version")):
            targets.setdefault(_gfx_target(int(match.group("version"))), None)
    return tuple(targets)


def _hipcc(environ: Mapping[str, str]) -> str | None:
    candidates = (
        environ.get("HIPCC", ""),
        str(resolve_rocm_roots(dict(environ)).bin_dir / "hipcc"),
        shutil.which("hipcc", path=environ.get("PATH")) or "",
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _write_new(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def _build_canary(
    hipcc: str, targets: Sequence[str], directory: Path, environ: Mapping[str, str]
) -> tuple[Path | None, str]:
    """Build the canary into ``directory``: the binary, or ``None`` and why not."""

    binary = directory / _CANARY_BINARY
    command = (
        hipcc,
        *_CANARY_BUILD_FLAGS,
        *(f"--offload-arch={target}:xnack+" for target in targets),
        str(CANARY_SOURCE),
        "-o",
        str(binary),
    )
    env = dict(environ)
    # hipcc shells out to clang-offload-bundler, which is on PATH in neither install layout.
    env["PATH"] = os.pathsep.join(
        part for part in (str(resolve_rocm_roots(env).llvm_bin_dir), env.get("PATH", "")) if part
    )
    built = run_argv(command, timeout_seconds=_BUILD_TIMEOUT, env=env)
    _write_new(directory / "build.log", f"$ {shlex.join(command)}\n{built.stdout}{built.stderr}")
    if built.returncode == 0 and binary.is_file():
        return binary, ""
    if built.timed_out:
        return None, f"hipcc did not finish in {_BUILD_TIMEOUT:.0f} s"
    if built.launch_error is not None:
        return None, f"hipcc could not start: {built.launch_error}"
    lines = [
        line.strip() for line in f"{built.stdout}\n{built.stderr}".splitlines() if line.strip()
    ]
    error = next((line for line in lines if "error" in line.lower()), lines[-1] if lines else "")
    return None, f"hipcc exited {built.returncode}" + (f": {error}" if error else "")


def _judge_canary(report: AsanRunReport, process: ProcessResult) -> tuple[NodeHealthStatus, str]:
    output = f"{process.stdout}\n{process.stderr}"
    passed = _CANARY_PASSED.search(output)
    if report.findings:
        first = report.findings[0]
        metadata = dict(first.metadata)
        where = metadata.get("location") or first.kernel_name
        found = f"{first.code} ({metadata.get('side', 'host')})" + (f" at {where}" if where else "")
        return (
            NodeHealthStatus.FAILED,
            f"the canary, whose every access is in bounds, reported {found}; {_SHADOW_SUSPECT}",
        )
    if failed := list(_CANARY_FAILED_CHECK.finditer(output)):
        details = "; ".join(
            f"GPU {check['gpu']} {check['check']}"
            + (f": {check['bad']} of {check['of']} words wrong" if check["bad"] is not None else "")
            for check in failed[:3]
        ) + (f"; and {len(failed) - 3} more" if len(failed) > 3 else "")
        if all(check["bad"] is None for check in failed):
            return (
                NodeHealthStatus.FAILED,
                f"a GPU cannot reach pageable host memory, which device ASan needs to read its "
                f"shadow memory ({details})",
            )
        return (
            NodeHealthStatus.FAILED,
            f"data the canary sent through GPU page migration came back wrong ({details}), "
            "and ASan's shadow memory takes the same path",
        )
    if process.timed_out:
        return NodeHealthStatus.FAILED, f"the canary did not finish in {_CANARY_TIMEOUT:.0f} s"
    if process.launch_error is not None:
        return NodeHealthStatus.FAILED, f"the canary could not start: {process.launch_error}"
    if hip_error := _CANARY_HIP_ERROR.search(output):
        return NodeHealthStatus.FAILED, f"the canary hit a HIP error: {hip_error['what']}"
    if report.verdict is Verdict.PASS and passed is not None:
        return (
            NodeHealthStatus.HEALTHY,
            f"an in-bounds canary ran clean under ASan on {passed['gpus']} GPU(s), all "
            f"{len(report.kernels)} of its kernels checked and its data intact through GPU "
            "page migration",
        )
    if report.verdict is Verdict.PASS:
        return (
            NodeHealthStatus.FAILED,
            "the canary's ASan run passed, but the canary never reported its own checks passing",
        )
    return (
        NodeHealthStatus.FAILED,
        f"the canary's own ASan run was {report.verdict.value}: {report.reason}",
    )


def check_node_health(
    evidence_dir: Path,
    runtime: AsanRuntime,
    *,
    environ: Mapping[str, str] | None = None,
    torch_lib: Path | None = None,
) -> NodeHealth:
    """Run the node-health canary under the environment a workload gets, and judge it.

    Device ASan reads its shadow memory through the GPU's XNACK page faults, and on some
    nodes the page migration behind them corrupts data. There every check misfires:
    in-bounds code reports memory errors, and a real error can be missed. Nothing in a
    workload's own evidence tells that apart from a real result. So :data:`CANARY_SOURCE`,
    built here with device ASan for every GPU target the driver lists, sends known data
    down that path on every visible GPU with every access in bounds, and is judged as a
    workload is: it has to PASS, every kernel checked, with its own data checks intact.
    The build takes about a second and the run about ten on an 8-GPU node. Its evidence,
    and its ``asan_report.json``, go to ``evidence_dir/node-health``.
    """

    started = time.monotonic()
    env = os.environ if environ is None else environ
    targets = gpu_targets()
    if not targets:
        return NodeHealth(
            NodeHealthStatus.NOT_VERIFIED,
            f"the kernel driver lists no GPU ({KFD_TOPOLOGY}), so no canary could run",
        )
    hipcc = _hipcc(env)
    if hipcc is None:
        return NodeHealth(
            NodeHealthStatus.NOT_VERIFIED,
            "hipcc was not found ($HIPCC, the ROCm bin directory, PATH), so the canary "
            "could not be built",
        )
    if not CANARY_SOURCE.is_file():
        return NodeHealth(
            NodeHealthStatus.NOT_VERIFIED,
            f"the canary source {CANARY_SOURCE} is missing from this installation",
        )
    directory = evidence_dir.absolute() / NODE_HEALTH_DIR
    os.mkdir(directory)
    binary, why = _build_canary(hipcc, targets, directory, env)
    if binary is None:
        return NodeHealth(
            NodeHealthStatus.NOT_VERIFIED,
            f"the canary did not build: {why}",
            evidence_dir=str(directory),
            seconds=time.monotonic() - started,
        )
    process = run_argv(
        (str(binary),),
        timeout_seconds=_CANARY_TIMEOUT,
        env=asan_run_environment(env, runtime, directory, torch_lib=torch_lib),
    )
    _write_new(directory / "canary.log", f"{process.stdout}{process.stderr}")
    report = evaluate_asan_run(
        directory, returncode=process.returncode, command=process.argv, runtime=runtime
    )
    write_report(report, directory)
    status, reason = _judge_canary(report, process)
    passed = _CANARY_PASSED.search(f"{process.stdout}\n{process.stderr}")
    return NodeHealth(
        status,
        reason,
        evidence_dir=str(directory),
        seconds=time.monotonic() - started,
        gpus=int(passed["gpus"]) if passed is not None else None,
        canary=report,
    )


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def new_evidence_dir(results_dir: Path, components: Sequence[str]) -> Path:
    """A fresh, empty directory at ``results_dir``/*components*/``<stamp>-<pid>``.

    Everything below ``results_dir`` is created by descending from it without
    following a symlink, as the dispatcher creates its workload directory: a stale
    or planted ``<results>/<workload>`` or ``asan`` link would otherwise send the
    evidence, and every runtime log written into it, outside the tree the operator
    named. ``results_dir`` itself is the operator's boundary and may be a link.
    """

    results_dir = results_dir.absolute()
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    try:
        with _fsafe.open_dir_nofollow(results_dir, list(components), create_missing=True) as fd:
            os.mkdir(stamp, dir_fd=fd)
    except (_fsafe.UnsafePathError, FileExistsError) as error:
        raise ValueError(
            f"refusing to create ASan evidence under {results_dir.joinpath(*components)}: "
            f"{error}. Remove the link, or point --results-dir at the intended location."
        ) from error
    return results_dir.joinpath(*components, stamp)


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
    # None when the node failed its health check and the command was not run.
    returncode: int | None
    report: AsanRunReport
    report_path: Path


def run_under_asan(
    command: Sequence[str],
    *,
    evidence_dir: Path,
    runtime: AsanRuntime,
    environ: Mapping[str, str] | None = None,
) -> AsanRunOutcome:
    """Check the node, run ``command`` with the ASan run environment, and judge what it left.

    A node that fails :func:`check_node_health` is not given the command at all. Whatever
    ASan reported there could be the node's fault rather than the command's, and a clean
    run could have missed what the node hid, so the result would be an error either way;
    the canary has already shown that in seconds, where the workload could take minutes.
    """

    base = os.environ if environ is None else environ
    torch_lib = torch_library_dir()
    health = check_node_health(evidence_dir, runtime, environ=base, torch_lib=torch_lib)
    returncode = None
    if health.status is not NodeHealthStatus.FAILED:
        env = asan_run_environment(base, runtime, evidence_dir, torch_lib=torch_lib)
        returncode = subprocess.run(list(command), env=env, check=False).returncode
    report = evaluate_asan_run(
        evidence_dir,
        returncode=returncode,
        command=command,
        runtime=runtime,
        node_health=health,
    )
    return AsanRunOutcome(returncode, report, write_report(report, evidence_dir))
