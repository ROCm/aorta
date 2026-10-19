"""Fail-closed GPU AddressSanitizer (ASan) backend.

Unlike Waitcheck and ConSan, ASan is compile-time instrumentation: the program
under test has to be BUILT with it (``hipcc -fsanitize=address -shared-libsan
--offload-arch=gfx950:xnack+``), and aorta's part is to run that build in an
environment where the instrumentation can actually report, then turn the report
into findings. Every rule below exists because the naive version of it was
measured to pass a run that checked nothing (gfx950, ROCm 7.0.2 and ROCm 10):

* Device ASan only exists for an ``xnack+`` code object. ``-fsanitize=address``
  with a plain ``gfx950`` target builds and links cleanly, with only a compiler
  warning, and produces a binary whose GPU code carries no instrumentation at
  all -- so an out-of-bounds store runs to exit 0. :func:`inspect_device_instrumentation`
  therefore reads the device code object out of the binary instead of trusting
  how it was built.
* The device half reports by calling back to the host, and only the
  ASan-instrumented ROCm runtime (``lib/asan``) handles that call. With the stock
  runtime a real detection aborts the process with ``Hostcall: no handler found
  for service ID 4`` and no report, so :func:`resolve_asan_runtime` refuses a
  runtime directory whose ``libamdhip64`` carries no ASan instrumentation.
* An xnack+ code object only loads with XNACK on, so ``HSA_XNACK=1`` is pinned.
* LeakSanitizer reports the stock runtime's own allocations on a correct program
  and rewrites the exit code when it does, so leak checking is off.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..rocm_paths import resolve_rocm_roots, safe_is_dir
from .consan import _launch_diagnostic
from .execution import ProcessResult, run_argv
from .models import (
    CheckResult,
    ExecutionState,
    Finding,
    FindingSeverity,
    KernelCheckResult,
    KernelIdentity,
    KernelWorklist,
    Verdict,
)

ASAN = "asan"

# Replaces any inherited ASAN_OPTIONS outright rather than extending it: an
# inherited ``exitcode=0`` or ``halt_on_error=0`` would weaken the check, and the
# point of pinning is that the parent environment cannot. halt_on_error and
# exitcode restate the defaults for that reason.
ASAN_OPTIONS_PINNED = "detect_leaks=0:halt_on_error=1:exitcode=1:symbolize=1"

ENV_ROCM_ASAN_LIB = "AORTA_ASAN_ROCM_LIB"
ENV_ASAN_RUNTIME = "AORTA_ASAN_RUNTIME"
# TheRock's documented name for the root of an ASan-instrumented install.
ENV_ROCM_ASAN_PATH = "ROCM_ASAN_PATH"

_HOST_RUNTIME_NAMES = ("libclang_rt.asan-x86_64.so", "libclang_rt.asan.so")
_HOST_RUNTIME_ARCH_DIRS = ("linux", "x86_64-unknown-linux-gnu")
_ASAN_MARKER = b"__asan_"

_HEADER = re.compile(r"^==(?P<pid>\d+)==ERROR: AddressSanitizer: (?P<rest>.*?)\s*$")
_DEVICE_HEADER = re.compile(
    r"^(?P<kind>[A-Za-z0-9_-]+) on amdgpu device (?P<device>\d+) at pc (?P<pc>0x[0-9a-fA-F]+)"
)
_ACCESS = re.compile(r"^(?P<access>READ|WRITE) of size (?P<size>\d+)\s+(?P<where>.*?)\s*$")
_WORKGROUP = re.compile(r"in workgroup id \((?P<wg>[^)]*)\)")
_FRAME0 = re.compile(r"^\s*#0 0x[0-9a-fA-F]+ in (?P<body>.+?)\s*$")
_FRAME = re.compile(r"^\s*#\d+ 0x[0-9a-fA-F]+ in (?P<body>.+?)\s*$")
_LOCATED = re.compile(r"^0x[0-9a-fA-F]+ is (?P<what>located .+?)\s*$")
# Device-side LDS is lowered into a device-malloc'd buffer when ASan is on, so an
# LDS overflow is located relative to that buffer by this line; the "located"
# line that follows it describes the 2 MiB device heap chunk, which says nothing.
_DEVICE_MALLOC_NEIGHBOUR = re.compile(
    r"^0x[0-9a-fA-F]+ is (?P<what>\d+ bytes (?:above|below) an address from a device malloc.*?)"
    r"(?: from)?\s*$"
)
# Source paths of the sanitizer and ROCm runtimes. An allocation stack's first
# located frames are the ASan interceptor and HIP's header-template wrappers
# (hipMalloc<T>), so the frame worth reporting is the first one outside them.
_RUNTIME_SOURCE_MARKERS = ("/compiler-rt/", "/include/hip/", "/clr/", "/rocr-runtime/")
# Headers whose first word is a verb rather than the error class.
_LEADING_VERB_KINDS = {
    "requested": "allocation-size-too-big",
    "allocator": "out-of-memory",
    "calloc": "calloc-overflow",
}
_STACK_HEADER = re.compile(
    r"^(?P<which>allocated|freed|previously allocated) by thread \S+ here:\s*$"
)
_SOURCE_LOCATION = re.compile(r"(?P<loc>\S+:\d+(?::\d+)?)$")
_LANE = re.compile(r"\b\d+ : 0x[0-9a-fA-F]+")
_SUMMARY_LOC = re.compile(r"^SUMMARY: AddressSanitizer: \S+ (?P<loc>\S+:\d+(?::\d+)?) in ")
_LEAK_REPORT = re.compile(r"^==\d+==ERROR: LeakSanitizer:", re.MULTILINE)
_HOSTCALL_LOST = re.compile(r"Hostcall: no handler found for service ID 4\b")
_RUNTIME_NOT_FIRST = "ASan runtime does not come first in initial library list"
_INCOMPATIBLE_RUNTIMES = "linked against incompatible ASan runtimes"
_INIT_FAILURE = re.compile(
    r"Shadow memory range interleaves with an existing memory mapping"
    r"|ERROR: AddressSanitizer failed to allocate"
    r"|AddressSanitizer: CHECK failed"
)
_INVALID_DEVICE_FUNCTION = re.compile(r"hipErrorInvalidDeviceFunction|invalid device function")

_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
_COMPRESSED_BUNDLE_MAGIC = b"CCOB"
_ELF_MAGIC = b"\x7fELF"
# Bounds on what a well-formed bundle header can declare, so a stray copy of the
# magic inside unrelated bytes is skipped instead of being read as a bundle.
_MAX_BUNDLE_ENTRIES = 4096
_MAX_TRIPLE_BYTES = 4096


# --------------------------------------------------------------------------- #
# Report parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParsedAsanOutput:
    """Everything the runtime said, before any verdict is decided."""

    findings: tuple[Finding, ...]
    leak_reports: int
    device_report_lost: bool
    runtime_not_preloaded: bool
    incompatible_runtimes: bool
    init_failed: bool


def _split_frame(body: str) -> tuple[str, str | None]:
    """Split a symbolized frame into ``(function, file:line[:col])``.

    Device frames spell it ``<function> at <file>:<line>:<col>``; host frames
    put the location after a space with no ``at``; an unsymbolized frame names
    only its module in parentheses and has no location.
    """

    function, sep, location = body.rpartition(" at ")
    if sep and _SOURCE_LOCATION.fullmatch(location):
        return function.strip(), location
    unparenthesised = re.sub(r"\s*\((?:BuildId:|/)[^)]*\)", "", body).strip()
    match = _SOURCE_LOCATION.search(unparenthesised)
    if match and " " in unparenthesised:
        return unparenthesised[: match.start()].strip(), match.group("loc")
    return unparenthesised, None


def _function_name(signature: str) -> str:
    """The bare name of a demangled signature: no parameters, no return type.

    ``write_n_plus_one(int*, int)`` is ``write_n_plus_one``;
    ``hipError_t hipMalloc<int>(int**, unsigned long)`` is ``hipMalloc<int>``.
    """

    depth = 0
    end = len(signature)
    for index, char in enumerate(signature):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "(" and depth == 0:
            end = index
            break
    head = signature[:end].strip()
    depth = 0
    for index in range(len(head) - 1, -1, -1):
        char = head[index]
        if char == ">":
            depth += 1
        elif char == "<":
            depth -= 1
        elif char == " " and depth == 0:
            return head[index + 1 :]
    return head


def _report_blocks(lines: list[str]) -> list[tuple[re.Match[str], list[str]]]:
    starts = [(index, match) for index, line in enumerate(lines) if (match := _HEADER.match(line))]
    blocks = []
    for position, (index, match) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        blocks.append((match, lines[index + 1 : end]))
    return blocks


def _finding_from_block(header: re.Match[str], body: list[str]) -> Finding:
    rest = header.group("rest")
    device = _DEVICE_HEADER.match(rest)
    if device:
        kind = device.group("kind")
    else:
        kind = rest.split(" ", 1)[0] or "unknown"
        # "attempting double-free on 0x..." / "attempting free on address which
        # was not malloc()-ed" lead with a verb, not with the error class.
        if kind == "attempting":
            kind = "double-free" if "double-free" in rest else "bad-free"
        kind = _LEADING_VERB_KINDS.get(kind, kind)

    metadata: dict[str, str] = {
        "pid": header.group("pid"),
        "side": "device" if device else "host",
        "header": rest,
    }
    if device:
        metadata["device"] = device.group("device")
        metadata["pc"] = device.group("pc")

    access = None
    function = None
    location = None
    lanes = 0
    in_lanes = False
    stack: str | None = None
    for line in body:
        if access is None and (match := _ACCESS.match(line)):
            access = match
            continue
        if function is None and (match := _FRAME0.match(line)):
            function, location = _split_frame(match.group("body"))
            continue
        if line.startswith("Thread ids and accessed addresses:"):
            in_lanes = True
            continue
        if in_lanes:
            if _LANE.search(line):
                lanes += len(_LANE.findall(line))
                continue
            in_lanes = False
        if "region" not in metadata and (match := _LOCATED.match(line)):
            metadata["region"] = match.group("what")
            continue
        if "device_malloc_neighbour" not in metadata and (
            match := _DEVICE_MALLOC_NEIGHBOUR.match(line)
        ):
            metadata["device_malloc_neighbour"] = match.group("what")
            stack = None
            continue
        if match := _STACK_HEADER.match(line):
            stack = match.group("which")
            continue
        if stack is not None and (match := _FRAME.match(line)):
            _, frame_location = _split_frame(match.group("body"))
            key = "freed_at" if stack == "freed" else "allocated_at"
            if (
                frame_location is not None
                and key not in metadata
                and not any(marker in frame_location for marker in _RUNTIME_SOURCE_MARKERS)
            ):
                metadata[key] = frame_location
            continue
        if location is None and (match := _SUMMARY_LOC.match(line)):
            location = match.group("loc")

    if access is not None:
        metadata["access"] = access.group("access")
        metadata["size"] = access.group("size")
        if workgroup := _WORKGROUP.search(access.group("where")):
            # Only the x component is trustworthy: y and z read as uninitialised
            # values (4294967295, 524288) on gfx950, so the raw text is kept and
            # nothing downstream should parse it.
            metadata["workgroup"] = workgroup.group("wg")
    if lanes:
        metadata["lanes"] = str(lanes)
    if function is not None:
        metadata["function"] = function
    if location is not None:
        metadata["location"] = location

    kernel_name = _function_name(function) if device and function else None
    parts = [f"{kind} on {'amdgpu device ' + device.group('device') if device else 'host'}"]
    if access is not None:
        parts.append(f"{access.group('access')} of size {access.group('size')}")
    if function is not None:
        parts.append(f"in {function}" + (f" at {location}" if location else ""))
    if "device_malloc_neighbour" in metadata:
        parts.append(metadata["device_malloc_neighbour"])
    elif "region" in metadata:
        parts.append(metadata["region"])
    return Finding(
        sanitizer=ASAN,
        severity=FindingSeverity.MEMORY_ERROR,
        code=kind,
        message="; ".join(parts),
        kernel_name=kernel_name,
        metadata=tuple(sorted(metadata.items())),
    )


def parse_asan_output(text: str) -> ParsedAsanOutput:
    """Parse ASan runtime output into findings and runtime-health signals.

    Device reports carry no ``SUMMARY:`` line (host reports do), so a report is
    recognised by its ``==PID==ERROR: AddressSanitizer:`` header alone. Every
    signal here can only make a run fail or error, never pass, so reading program
    output that imitates one costs a false alarm at worst.
    """

    lines = text.splitlines()
    findings = tuple(_finding_from_block(header, body) for header, body in _report_blocks(lines))
    return ParsedAsanOutput(
        findings=findings,
        leak_reports=len(_LEAK_REPORT.findall(text)),
        device_report_lost=bool(_HOSTCALL_LOST.search(text)),
        runtime_not_preloaded=_RUNTIME_NOT_FIRST in text,
        incompatible_runtimes=_INCOMPATIBLE_RUNTIMES in text,
        init_failed=bool(_INIT_FAILURE.search(text)),
    )


# --------------------------------------------------------------------------- #
# Device instrumentation preflight
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeviceCodeObject:
    """One GPU code object found in an offload bundle."""

    triple: str
    processor: str
    features: tuple[str, ...]
    asan_instrumented: bool

    @property
    def xnack_on(self) -> bool:
        return "xnack+" in self.features


@dataclass(frozen=True)
class InstrumentationEvidence:
    """What the device code inside one binary says about ASan."""

    path: str
    code_objects: tuple[DeviceCodeObject, ...]
    compressed_bundles: int

    def instrumented_for(self, target: str) -> bool:
        return any(
            item.processor == target and item.xnack_on and item.asan_instrumented
            for item in self.code_objects
        )

    def describe(self) -> str:
        if not self.code_objects:
            return f"{self.path}: no readable device code object" + (
                f" ({self.compressed_bundles} compressed bundle(s))"
                if self.compressed_bundles
                else ""
            )
        listed = ", ".join(
            f"{item.triple}[asan={'yes' if item.asan_instrumented else 'no'}]"
            for item in self.code_objects
        )
        return f"{self.path}: {listed}"


def _parse_bundle(data: bytes, base: int) -> list[DeviceCodeObject]:
    """Read the AMDGPU entries of one uncompressed clang offload bundle.

    Layout: the 24-byte magic, a little-endian u64 entry count, then per entry
    u64 offset, u64 size, u64 triple length and the triple, with offsets taken
    from the start of the bundle. Anything implausible is treated as not a
    bundle rather than as evidence.
    """

    try:
        offset = base + len(_BUNDLE_MAGIC)
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        if count == 0 or count > _MAX_BUNDLE_ENTRIES:
            return []
        entries = []
        for _ in range(count):
            entry_offset, entry_size, triple_size = struct.unpack_from("<QQQ", data, offset)
            offset += 24
            if triple_size == 0 or triple_size > _MAX_TRIPLE_BYTES:
                return []
            triple = data[offset : offset + triple_size].decode("ascii", errors="replace")
            offset += triple_size
            entries.append((entry_offset, entry_size, triple))
    except struct.error:
        return []

    objects = []
    for entry_offset, entry_size, triple in entries:
        if "amdgcn" not in triple:
            continue
        start = base + entry_offset
        end = start + entry_size
        if entry_size == 0 or end > len(data):
            continue
        processor_and_features = triple.rpartition("--")[2]
        processor, *features = processor_and_features.split(":")
        objects.append(
            DeviceCodeObject(
                triple=triple,
                processor=processor,
                features=tuple(features),
                asan_instrumented=_ASAN_MARKER in data[start:end],
            )
        )
    return objects


def _count_compressed_bundles(data: bytes) -> int:
    count = 0
    start = 0
    while (index := data.find(_COMPRESSED_BUNDLE_MAGIC, start)) != -1:
        start = index + 1
        try:
            version, method = struct.unpack_from("<HH", data, index + 4)
        except struct.error:
            break
        if version in (1, 2, 3) and method in (1, 2):
            count += 1
    return count


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == _ELF_MAGIC
    except OSError:
        return False


def inspect_device_instrumentation(path: Path) -> InstrumentationEvidence:
    """Read every offload bundle in ``path`` and report its device code objects.

    Instrumentation is decided per code object: the ``__asan_`` marker has to be
    inside the device ELF itself, because the host half of an ASan build carries
    ``__asan_report_*`` references even when the device half was dropped.
    Compressed (``CCOB``) bundles are counted, not decompressed; a binary that
    holds only those is reported as unreadable rather than as uninstrumented.
    """

    data = path.read_bytes()
    objects: list[DeviceCodeObject] = []
    start = 0
    while (index := data.find(_BUNDLE_MAGIC, start)) != -1:
        objects.extend(_parse_bundle(data, index))
        start = index + len(_BUNDLE_MAGIC)
    return InstrumentationEvidence(
        path=str(path),
        code_objects=tuple(objects),
        compressed_bundles=_count_compressed_bundles(data),
    )


# --------------------------------------------------------------------------- #
# Runtime resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AsanRuntime:
    """The pieces a device ASan run needs beside the instrumented program."""

    rocm_lib_dirs: tuple[Path, ...]
    hip_runtime: Path
    host_runtime: Path
    symbolizer: Path | None


def _instrumented_hip_runtime(lib_dir: Path) -> Path | None:
    if not safe_is_dir(lib_dir):
        return None
    try:
        candidates = sorted(lib_dir.glob("libamdhip64.so*"))
    except OSError:
        return None
    for candidate in candidates:
        try:
            if candidate.is_file() and _ASAN_MARKER in candidate.read_bytes():
                return candidate
        except OSError:
            continue
    return None


def _host_runtime(core: Path) -> Path | None:
    clang_root = core / "lib" / "llvm" / "lib" / "clang"
    if not safe_is_dir(clang_root):
        return None
    try:
        versions = sorted(
            (entry for entry in clang_root.iterdir() if entry.is_dir()),
            key=lambda entry: [
                int(part) if part.isdigit() else 0 for part in entry.name.split(".")
            ],
            reverse=True,
        )
    except OSError:
        return None
    for version in versions:
        for arch_dir in _HOST_RUNTIME_ARCH_DIRS:
            for name in _HOST_RUNTIME_NAMES:
                candidate = version / "lib" / arch_dir / name
                if candidate.is_file():
                    return candidate
    return None


def _lib_dir_candidates(environ: Mapping[str, str], core: Path) -> list[Path]:
    explicit = environ.get(ENV_ROCM_ASAN_LIB, "").strip()
    if explicit:
        return [Path(explicit)]
    candidates = []
    asan_root = environ.get(ENV_ROCM_ASAN_PATH, "").strip()
    if asan_root:
        # Classic packages install beside the stock libraries in lib/asan;
        # TheRock's ASan tarball is a whole instrumented tree with them in lib.
        candidates += [Path(asan_root) / "lib" / "asan", Path(asan_root) / "lib"]
    candidates.append(core / "lib" / "asan")
    return candidates


def resolve_asan_runtime(environ: Mapping[str, str] | None = None) -> AsanRuntime | None:
    """Locate an ASan-instrumented ROCm runtime and the matching host runtime.

    ``AORTA_ASAN_ROCM_LIB`` names the instrumented library directory outright;
    otherwise ``ROCM_ASAN_PATH`` (TheRock's name for an ASan install root) and
    then ``lib/asan`` under the resolved ROCm root are tried. A directory only
    counts if its ``libamdhip64`` is itself instrumented: the stock runtime
    loads, runs, and then aborts on the first detection without a report.
    ``AORTA_ASAN_RUNTIME`` overrides the host ``libclang_rt.asan`` lookup.
    Returns ``None`` when either half is missing.
    """

    env = os.environ if environ is None else environ
    core = resolve_rocm_roots(dict(env)).core
    hip_runtime = None
    lib_dir = None
    for candidate in _lib_dir_candidates(env, core):
        hip_runtime = _instrumented_hip_runtime(candidate)
        if hip_runtime is not None:
            lib_dir = candidate
            break
    if hip_runtime is None or lib_dir is None:
        return None

    explicit_runtime = env.get(ENV_ASAN_RUNTIME, "").strip()
    host_runtime = Path(explicit_runtime) if explicit_runtime else _host_runtime(core)
    if host_runtime is None or not host_runtime.is_file():
        return None

    # TheRock's ASan tree keeps LLVM and system dependencies beside lib/, and its
    # documented LD_LIBRARY_PATH lists both.
    extra = tuple(
        directory
        for directory in (lib_dir / "llvm" / "lib", lib_dir / "rocm_sysdeps" / "lib")
        if safe_is_dir(directory)
    )
    symbolizer = core / "lib" / "llvm" / "bin" / "llvm-symbolizer"
    return AsanRuntime(
        rocm_lib_dirs=(lib_dir, *extra),
        hip_runtime=hip_runtime,
        host_runtime=host_runtime,
        symbolizer=symbolizer if symbolizer.is_file() else None,
    )


def asan_environment(base: Mapping[str, str], runtime: AsanRuntime) -> dict[str, str]:
    """The child environment for an ASan run: the parent's, plus the contract.

    The instrumented libraries are prepended to ``LD_LIBRARY_PATH`` and the host
    runtime to ``LD_PRELOAD``, ahead of anything inherited. That is a deliberate
    exception to leaving the process under test's library path alone: the
    instrumented runtime IS the sanitizer here, as ``HSA_TOOLS_LIB`` is for
    ConSan, and a stock library ahead of it on the path silently turns every
    device detection into an abort. The preload is what lets an uninstrumented
    launcher (Python, a shell script) load instrumented libraries at all.
    """

    env = dict(base)
    env["HSA_XNACK"] = "1"
    env["ASAN_OPTIONS"] = ASAN_OPTIONS_PINNED
    library_path = os.pathsep.join(
        str(directory) for directory in (*runtime.rocm_lib_dirs, runtime.host_runtime.parent)
    )
    inherited = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{library_path}{os.pathsep}{inherited}" if inherited else library_path
    preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = (
        f"{runtime.host_runtime}:{preload}" if preload else str(runtime.host_runtime)
    )
    if runtime.symbolizer is not None:
        env["ASAN_SYMBOLIZER_PATH"] = str(runtime.symbolizer)
    return env


# --------------------------------------------------------------------------- #
# Evaluation and execution
# --------------------------------------------------------------------------- #


def _check(
    *,
    state: ExecutionState,
    verdict: Verdict,
    reason: str | None = None,
    returncode: int | None = None,
    findings: tuple[Finding, ...] = (),
) -> CheckResult:
    return CheckResult(
        sanitizer=ASAN,
        state=state,
        verdict=verdict,
        reason=reason,
        returncode=returncode,
        findings=findings,
    )


def _not_checked(reason: str) -> CheckResult:
    return _check(state=ExecutionState.NOT_CHECKED, verdict=Verdict.NOT_CHECKED, reason=reason)


def _error(reason: str, *, returncode: int | None = None) -> CheckResult:
    return _check(
        state=ExecutionState.ERROR, verdict=Verdict.ERROR, reason=reason, returncode=returncode
    )


def evaluate_asan_output(process: ProcessResult) -> CheckResult:
    """Decide one ASan run. Only a clean exit with no report is a PASS.

    A report is a result whatever the exit code says: ASan aborts the process on
    its first error by design, so a nonzero exit beside a report is the report,
    not a backend failure.
    """

    parsed = parse_asan_output(f"{process.stderr}\n{process.stdout}")
    if process.timed_out:
        return _check(
            state=ExecutionState.TIMED_OUT,
            verdict=Verdict.FAIL if parsed.findings else Verdict.ERROR,
            reason="asan_timeout",
            findings=parsed.findings,
        )
    if process.launch_error is not None:
        return _error(f"asan_launch_error: {process.launch_error}")
    if parsed.findings:
        return _check(
            state=ExecutionState.RAN,
            verdict=Verdict.FAIL,
            returncode=process.returncode,
            findings=parsed.findings,
        )
    if parsed.device_report_lost:
        return _error(
            "asan_device_report_lost: device ASan detected a memory error, but the "
            "loaded HIP runtime has no handler for the sanitizer hostcall, so the "
            "process aborted without a report. The stock runtime was loaded ahead "
            "of the ASan-instrumented one (lib/asan)",
            returncode=process.returncode,
        )
    if parsed.runtime_not_preloaded:
        return _error(
            "asan_runtime_not_preloaded: an uninstrumented executable loaded "
            "instrumented libraries without the ASan runtime in LD_PRELOAD",
            returncode=process.returncode,
        )
    if parsed.incompatible_runtimes:
        return _error(
            "asan_incompatible_runtimes: the program links a static ASan runtime; "
            "build it with -shared-libsan so it shares one runtime with the "
            "instrumented ROCm libraries",
            returncode=process.returncode,
        )
    if parsed.init_failed:
        return _error(
            "asan_init_failed: the ASan runtime could not initialise",
            returncode=process.returncode,
        )
    if process.returncode != 0:
        reason = f"asan_command_exit_{process.returncode}{_launch_diagnostic(process)}"
        if _INVALID_DEVICE_FUNCTION.search(f"{process.stderr}\n{process.stdout}"):
            reason += (
                ": the kernel would not load -- an xnack+ code object needs "
                "HSA_XNACK=1 on an XNACK-capable GPU"
            )
        if parsed.leak_reports:
            reason += ": LeakSanitizer ran although aorta pins detect_leaks=0"
        return _error(reason, returncode=process.returncode)
    return _check(state=ExecutionState.RAN, verdict=Verdict.PASS, returncode=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(objects: Sequence[Path], *, target: str) -> tuple[CheckResult | None, list[str]]:
    """Fail closed unless every named binary carries device ASan for ``target``."""

    descriptions = []
    for path in objects:
        if not path.is_file():
            return _error(f"asan_object_not_found: {path}"), descriptions
        evidence = inspect_device_instrumentation(path)
        descriptions.append(evidence.describe())
        if not evidence.code_objects and evidence.compressed_bundles:
            return (
                _error(
                    f"asan_instrumentation_unverified: {path} holds only compressed "
                    "offload bundles, which aorta does not decompress; rebuild "
                    "without --offload-compress"
                ),
                descriptions,
            )
        if not evidence.instrumented_for(target):
            return (
                _error(
                    f"asan_device_not_instrumented: {evidence.describe()}. No "
                    f"{target}:xnack+ code object carries ASan instrumentation, so "
                    "device memory errors would go unseen; -fsanitize=address is "
                    f"silently dropped for a target without :xnack+ (build with "
                    f"--offload-arch={target}:xnack+)"
                ),
                descriptions,
            )
    return None, descriptions


def run_asan(
    worklist: KernelWorklist,
    *,
    command: Path,
    output_dir: Path,
    runtime: AsanRuntime | None = None,
    objects: Sequence[Path] = (),
    timeout_seconds: float = 900.0,
    target: KernelIdentity | None = None,
) -> CheckResult:
    """Run one ASan-instrumented repro and report what the sanitizer saw.

    Scoped like ConSan: one explicitly selected kernel, so a PASS is attributed
    to a real selection. ``objects`` names the binaries whose device code is
    checked for instrumentation before anything runs; it defaults to ``command``
    when that is an ELF executable, and must be given when it is not (a script
    wrapping the real binary), because an unverifiable build fails closed.
    """

    if len(worklist.kernels) != 1:
        return _not_checked("asan_requires_one_targeted_repro")
    selected = worklist.kernels[0].identity
    if target is not None and target.stable_key != selected.stable_key:
        return _not_checked("asan_target_does_not_match_worklist")
    if not command.is_file():
        return _not_checked("asan_command_not_found")

    # The build is checked before the runtime is looked for: it needs no GPU and
    # no ASan runtime, so a mis-built repro is reported on any host.
    checked = tuple(objects) or ((command,) if is_elf(command) else ())
    if not checked:
        return _error(
            f"asan_instrumentation_unverifiable: {command} is not an ELF executable, "
            "so its device code cannot be inspected; name the instrumented binaries "
            "in sanitizer_plan.source.asan_objects"
        )
    failure, evidence = _preflight(checked, target=selected.target)
    backend = {
        "command": str(command),
        "command_sha256": _sha256_file(command),
        "device_instrumentation": " | ".join(evidence),
        "selected_kernel": selected.name,
        "selected_identity_sha256": hashlib.sha256(selected.stable_key.encode("utf-8")).hexdigest(),
    }

    check = failure
    if check is None:
        resolved = runtime if runtime is not None else resolve_asan_runtime()
        if resolved is None:
            check = _not_checked(
                "asan_runtime_not_found: device ASan reports need the ASan-instrumented "
                f"ROCm runtime; set {ENV_ROCM_ASAN_LIB} to its lib/asan directory or "
                f"{ENV_ROCM_ASAN_PATH} to the install root"
            )
        else:
            backend.update(
                {
                    "asan_options": ASAN_OPTIONS_PINNED,
                    "hsa_xnack": "1",
                    "rocm_asan_lib_dirs": os.pathsep.join(
                        str(item) for item in resolved.rocm_lib_dirs
                    ),
                    "hip_runtime": str(resolved.hip_runtime),
                    "hip_runtime_sha256": _sha256_file(resolved.hip_runtime),
                    "host_runtime": str(resolved.host_runtime),
                    "symbolizer": str(resolved.symbolizer) if resolved.symbolizer else "not_found",
                }
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            process = run_argv(
                (str(command),),
                timeout_seconds=timeout_seconds,
                env=asan_environment(os.environ, resolved),
            )
            (output_dir / "asan.log").write_text(
                f"{process.stdout}\n{process.stderr}", encoding="utf-8"
            )
            check = evaluate_asan_output(process)

    kernel_result = KernelCheckResult(
        identity=selected,
        state=check.state,
        verdict=check.verdict,
        findings=check.findings,
        reason=check.reason,
        returncode=check.returncode,
    )
    return CheckResult(
        sanitizer=ASAN,
        state=check.state,
        verdict=check.verdict,
        reason=check.reason,
        returncode=check.returncode,
        findings=check.findings,
        kernel_results=(kernel_result,),
        backend=tuple(sorted(backend.items())),
    )


def scoped_asan_not_checked(worklist: KernelWorklist) -> CheckResult:
    """Fail closed when no ASan-instrumented repro command is provisioned."""

    return _not_checked(
        "asan_command_not_provisioned: an ASan-instrumented repro is required; "
        f"aorta cannot instrument {len(worklist.kernels)} selected kernel(s) after "
        "the fact, because ASan is applied when the program is compiled"
    )
