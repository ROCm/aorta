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
  how it was built. That proves the object was built with device ASan, not that
  every access in it is checked: buffer-addressed stores get no shadow check.
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

import ctypes
import ctypes.util
import functools
import hashlib
import os
import re
import struct
from collections.abc import Callable, Mapping, Sequence
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
ASAN_OPTIONS_PINNED = "detect_leaks=0:halt_on_error=1:exitcode=1:symbolize=1:color=never"

ENV_ROCM_ASAN_LIB = "AORTA_ASAN_ROCM_LIB"
ENV_ASAN_RUNTIME = "AORTA_ASAN_RUNTIME"
# TheRock's documented name for the root of an ASan-instrumented install.
ENV_ROCM_ASAN_PATH = "ROCM_ASAN_PATH"

_HOST_RUNTIME_NAMES = ("libclang_rt.asan-x86_64.so", "libclang_rt.asan.so")
_HOST_RUNTIME_ARCH_DIRS = ("linux", "x86_64-unknown-linux-gnu")
_ASAN_MARKER = b"__asan_"

# Searched, not anchored: a header can follow unterminated program output on
# the same line, and log_exe_name puts the executable's name before the pid.
_HEADER = re.compile(r"==(?:[^=\s]+==)?(?P<pid>\d+)==ERROR: AddressSanitizer: (?P<rest>.*?)\s*$")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
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
_LEAK_REPORT = re.compile(r"==(?:[^=\s]+==)?\d+==ERROR: LeakSanitizer:")
_HOSTCALL_LOST = re.compile(r"Hostcall: no handler found for service ID 4\b")
_RUNTIME_NOT_FIRST = "ASan runtime does not come first in initial library list"
_INCOMPATIBLE_RUNTIMES = "linked against incompatible ASan runtimes"
_INIT_FAILURE = re.compile(
    r"Shadow memory range interleaves with an existing memory mapping"
    r"|ERROR: AddressSanitizer failed to allocate"
    r"|AddressSanitizer: CHECK failed"
)
# HIP's own runtime log records every kernel dispatch by name at level 3 under
# LOG_KERN, one file per process when AMD_LOG_LEVEL_FILE is set (the runtime
# appends _<pid>). AMD_LOG_MASK is read as decimal: 128 is LOG_KERN, and a
# 0x-prefixed value selects nothing at all.
DISPATCH_LOG_NAME = "hip_dispatch.log"
_DISPATCH_LOG_ENV = {"AMD_LOG_LEVEL": "3", "AMD_LOG_MASK": "128"}
_DISPATCH_LINE = re.compile(r"ShaderName : (?P<signature>.+?)\s*$", re.MULTILINE)
# Dispatched by the runtime itself: ASan's module constructor and destructor, and
# the HIP runtime's own __amd_rocclr_* blit and heap kernels.
_RUNTIME_KERNELS = frozenset({"amdgcn.device.init", "amdgcn.device.fini"})
_RUNTIME_KERNEL_PREFIX = "__amd_rocclr_"

# The kernel never ran, so a clean exit afterwards says nothing about it.
_KERNEL_NOT_LOADED = re.compile(
    r"hipErrorInvalidDeviceFunction|invalid device function|hipErrorNoBinaryForGpu"
    r"|no kernel image is available|Unable to find code object"
)

_EM_AMDGPU = 224
# The EF_AMDGPU_MACH field of a code object's e_flags (low byte). Unlisted values
# name no target, so such an object can never satisfy a recipe's target.
_AMDGPU_MACH = {
    0x030: "gfx908",
    0x03F: "gfx90a",
    0x040: "gfx940",
    0x04B: "gfx941",
    0x04C: "gfx942",
    0x04F: "gfx950",
}
_XNACK_FEATURE = {0x300: "xnack+", 0x200: "xnack-"}
_SRAMECC_FEATURE = {0xC00: "sramecc+", 0x800: "sramecc-"}

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
    starts = [(index, match) for index, line in enumerate(lines) if (match := _HEADER.search(line))]
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

    text = _ANSI_ESCAPE.sub("", text)
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
    # Which of the kernel names asked about occur in this code object.
    kernels_found: tuple[str, ...] = ()

    @property
    def xnack_on(self) -> bool:
        return "xnack+" in self.features

    @property
    def checked(self) -> bool:
        return self.xnack_on and self.asan_instrumented


@dataclass(frozen=True)
class InstrumentationEvidence:
    """What the device code inside one binary says about ASan."""

    path: str
    code_objects: tuple[DeviceCodeObject, ...]
    compressed_bundles: int

    @property
    def has_device_code(self) -> bool:
        return bool(self.code_objects) or self.compressed_bundles > 0

    def problems(self, target: str) -> list[str]:
        """Why this binary's device code cannot vouch for an ASan run on ``target``.

        Every code object has to be instrumented, not just one. A binary links one
        bundle per translation unit, the runtime loads each, and an object built
        without ``:xnack+`` still runs with XNACK on -- so one unchecked object
        beside a checked one runs its kernels unchecked. Objects for other targets
        are held to the same rule, because the GPU that runs the binary is not
        necessarily the one the recipe names.
        """

        found = []
        if self.compressed_bundles:
            found.append(
                f"{self.compressed_bundles} compressed offload bundle(s), which aorta "
                "cannot inspect (rebuild without --offload-compress)"
            )
        unchecked = [item.triple for item in self.code_objects if not item.checked]
        if unchecked:
            found.append("device code without ASan: " + ", ".join(unchecked))
        if not any(item.processor == target for item in self.code_objects):
            found.append(f"no {target} code object")
        return found

    def instrumented_for(self, target: str) -> bool:
        return not self.problems(target)

    def holds_kernel(self, target: str, name: str) -> bool:
        return any(
            item.processor == target and name in item.kernels_found for item in self.code_objects
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


def _bundle_entries(data: bytes, base: int) -> list[tuple[str, int, int]]:
    """The AMDGPU entries of one uncompressed clang offload bundle.

    Layout: the 24-byte magic, a little-endian u64 entry count, then per entry
    u64 offset, u64 size, u64 triple length and the triple, with offsets taken
    from the start of the bundle. Returns ``(triple, start, end)`` with offsets
    into ``data``. Anything implausible is treated as not a bundle rather than
    as evidence.
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

    found = []
    for entry_offset, entry_size, triple in entries:
        if "amdgcn" not in triple:
            continue
        start = base + entry_offset
        end = start + entry_size
        if entry_size == 0 or end > len(data):
            continue
        found.append((triple, start, end))
    return found


def _split_target(triple: str) -> tuple[str, tuple[str, ...]]:
    processor, *features = triple.rpartition("--")[2].split(":")
    return processor, tuple(features)


def _parse_bundle(data: bytes, base: int, kernels: Sequence[str]) -> list[DeviceCodeObject]:
    """Read the AMDGPU entries of one uncompressed clang offload bundle."""

    objects = []
    for triple, start, end in _bundle_entries(data, base):
        processor, features = _split_target(triple)
        blob = data[start:end]
        objects.append(
            DeviceCodeObject(
                triple=triple,
                processor=processor,
                features=features,
                asan_instrumented=_ASAN_MARKER in blob,
                kernels_found=tuple(name for name in kernels if name.encode() in blob),
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


def _amdgpu_target(data: bytes) -> tuple[str, tuple[str, ...]] | None:
    """Processor and features of ``data`` when it is an AMDGPU code object."""

    if len(data) < 52 or data[:4] != _ELF_MAGIC or data[4] != 2 or data[5] != 1:
        return None
    (machine,) = struct.unpack_from("<H", data, 18)
    if machine != _EM_AMDGPU:
        return None
    (flags,) = struct.unpack_from("<I", data, 48)
    processor = _AMDGPU_MACH.get(flags & 0xFF, f"amdgpu-mach-0x{flags & 0xFF:02x}")
    features = tuple(
        feature
        for feature in (_SRAMECC_FEATURE.get(flags & 0xC00), _XNACK_FEATURE.get(flags & 0x300))
        if feature
    )
    return processor, features


def _bare_code_object(data: bytes, kernels: Sequence[str]) -> DeviceCodeObject | None:
    """Describe ``data`` when it is itself an AMDGPU code object (a ``.hsaco``)."""

    if (target := _amdgpu_target(data)) is None:
        return None
    processor, features = target
    return DeviceCodeObject(
        triple=f"amdgcn-amd-amdhsa--{':'.join((processor, *features))}",
        processor=processor,
        features=features,
        asan_instrumented=_ASAN_MARKER in data,
        kernels_found=tuple(name for name in kernels if name.encode() in data),
    )


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == _ELF_MAGIC
    except OSError:
        return False


def inspect_device_instrumentation(
    path: Path, *, kernels: Sequence[str] = ()
) -> InstrumentationEvidence:
    """Read every offload bundle in ``path`` and report its device code objects.

    Instrumentation is decided per code object: the ``__asan_`` marker has to be
    inside the device ELF itself, because the host half of an ASan build carries
    ``__asan_report_*`` references even when the device half was dropped.
    Compressed (``CCOB``) bundles are counted, not decompressed. ``kernels`` are
    looked up by name in each code object, which matches a plain name inside its
    mangled symbol.
    """

    data = path.read_bytes()
    if (bare := _bare_code_object(data, kernels)) is not None:
        return InstrumentationEvidence(path=str(path), code_objects=(bare,), compressed_bundles=0)
    objects: list[DeviceCodeObject] = []
    start = 0
    while (index := data.find(_BUNDLE_MAGIC, start)) != -1:
        objects.extend(_parse_bundle(data, index, kernels))
        start = index + len(_BUNDLE_MAGIC)
    return InstrumentationEvidence(
        path=str(path),
        code_objects=tuple(objects),
        compressed_bundles=_count_compressed_bundles(data),
    )


# --------------------------------------------------------------------------- #
# Kernel inventory: which kernels a file's device code defines
# --------------------------------------------------------------------------- #

_SHT_SYMTAB = 2
_SHT_DYNSYM = 11
_SYMBOL_SIZE = 24
_FATBIN_SECTION = ".hip_fatbin"
# Every kernel has a descriptor symbol, ``<kernel>.kd``; it is what the runtime
# looks a kernel up by, so it is the authoritative list of what an object defines.
_KERNEL_DESCRIPTOR = ".kd"


@dataclass(frozen=True)
class _Section:
    name: str
    kind: int
    offset: int
    size: int
    link: int
    entsize: int


def _elf_sections(read: Callable[[int, int], bytes], size: int) -> list[_Section] | None:
    """The section headers of a little-endian ELF64 image, or ``None``."""

    header = read(0, 64)
    if len(header) < 64 or header[:4] != _ELF_MAGIC or header[4] != 2 or header[5] != 1:
        return None
    (table_offset,) = struct.unpack_from("<Q", header, 0x28)
    entry_size, count, names_index = struct.unpack_from("<HHH", header, 0x3A)
    if (
        table_offset == 0
        or entry_size < 64
        or count == 0
        or names_index >= count
        or table_offset + entry_size * count > size
    ):
        return None
    table = read(table_offset, entry_size * count)
    raw = [struct.unpack_from("<IIQQQQIIQQ", table, index * entry_size) for index in range(count)]
    names_offset, names_size = raw[names_index][4], raw[names_index][5]
    names = read(names_offset, names_size) if names_offset + names_size <= size else b""
    sections = []
    for name_offset, kind, _flags, _address, offset, length, link, _info, _align, entsize in raw:
        end = names.find(b"\0", name_offset) if name_offset < len(names) else -1
        name = names[name_offset:end].decode("ascii", errors="replace") if end != -1 else ""
        sections.append(_Section(name, kind, offset, length, link, entsize))
    return sections


def _kernel_symbols(blob: bytes) -> tuple[str, ...]:
    """Mangled names of the kernels an AMDGPU code object defines."""

    sections = _elf_sections(lambda offset, length: blob[offset : offset + length], len(blob))
    if not sections:
        return ()
    names: set[str] = set()
    for section in sections:
        if section.kind not in (_SHT_SYMTAB, _SHT_DYNSYM) or section.entsize < _SYMBOL_SIZE:
            continue
        if not 0 <= section.link < len(sections):
            continue
        strings_section = sections[section.link]
        strings = blob[strings_section.offset : strings_section.offset + strings_section.size]
        end = min(section.offset + section.size, len(blob))
        for entry in range(section.offset, end - _SYMBOL_SIZE + 1, section.entsize):
            (name_offset,) = struct.unpack_from("<I", blob, entry)
            terminator = strings.find(b"\0", name_offset)
            if terminator <= name_offset:
                continue
            name = strings[name_offset:terminator].decode("utf-8", errors="replace")
            if name.endswith(_KERNEL_DESCRIPTOR) and len(name) > len(_KERNEL_DESCRIPTOR):
                names.add(name[: -len(_KERNEL_DESCRIPTOR)])
    return tuple(sorted(names))


@functools.cache
def _cxa_demangle() -> Callable[[str], str | None] | None:
    try:
        library = ctypes.CDLL(ctypes.util.find_library("stdc++") or "libstdc++.so.6")
        libc = ctypes.CDLL(None)
    except OSError:
        return None
    demangle_fn = library["__cxa_demangle"]
    demangle_fn.restype = ctypes.c_void_p
    demangle_fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    free = libc.free
    free.argtypes = [ctypes.c_void_p]
    free.restype = None

    def call(symbol: str) -> str | None:
        status = ctypes.c_int()
        pointer = demangle_fn(symbol.encode(), None, None, ctypes.byref(status))
        if not pointer:
            return None
        try:
            if status.value != 0:
                return None
            return ctypes.string_at(pointer).decode("utf-8", errors="replace")
        finally:
            free(pointer)

    return call


def demangle(symbol: str) -> str:
    """``symbol`` as the HIP runtime prints it in a dispatch log.

    The runtime demangles with the C++ ABI's ``__cxa_demangle``, so this calls
    the same function rather than an approximation of it. A name that is not
    mangled (an ``extern "C"`` kernel, a Triton kernel) is returned unchanged.
    """

    if not symbol.startswith("_Z"):
        return symbol
    call = _cxa_demangle()
    demangled = call(symbol) if call is not None else None
    return demangled or symbol


@dataclass(frozen=True)
class KernelCodeObject:
    """One GPU code object and the kernels it defines, as the runtime names them."""

    triple: str
    processor: str
    features: tuple[str, ...]
    asan_instrumented: bool
    kernels: tuple[str, ...]

    @property
    def checked(self) -> bool:
        return "xnack+" in self.features and self.asan_instrumented


@dataclass(frozen=True)
class KernelInventory:
    """The device code one file carries, kernel by kernel."""

    path: str
    code_objects: tuple[KernelCodeObject, ...] = ()
    compressed_bundles: int = 0
    unreadable: str | None = None

    @property
    def has_device_code(self) -> bool:
        return bool(self.code_objects) or self.compressed_bundles > 0


def _kernel_code_object(
    triple: str, processor: str, features: tuple[str, ...], blob: bytes
) -> KernelCodeObject:
    return KernelCodeObject(
        triple=triple,
        processor=processor,
        features=features,
        asan_instrumented=_ASAN_MARKER in blob,
        kernels=tuple(demangle(name) for name in _kernel_symbols(blob)),
    )


def kernel_inventory(path: Path) -> KernelInventory:
    """Which kernels ``path`` defines, per device code object.

    A host binary is read through its ``.hip_fatbin`` section only: the
    libraries a PyTorch process loads run to gigabytes and their device code to
    a fraction of that. A standalone offload bundle or code object -- what
    ``hipModuleLoad`` and Triton load -- is read whole. Compressed bundles are
    counted and not opened, so a file can carry device code this cannot name.
    """

    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size

            def read(offset: int, length: int) -> bytes:
                stream.seek(offset)
                return stream.read(length)

            head = read(0, 64)
            if head[:4] == _ELF_MAGIC:
                if len(head) >= 20 and struct.unpack_from("<H", head, 18)[0] == _EM_AMDGPU:
                    data = read(0, size)
                    if (target := _amdgpu_target(data)) is None:
                        return KernelInventory(path=str(path))
                    processor, features = target
                    triple = f"amdgcn-amd-amdhsa--{':'.join((processor, *features))}"
                    return KernelInventory(
                        path=str(path),
                        code_objects=(_kernel_code_object(triple, processor, features, data),),
                    )
                sections = _elf_sections(read, size) or []
                region = b"".join(
                    read(section.offset, section.size)
                    for section in sections
                    if section.name == _FATBIN_SECTION and section.offset + section.size <= size
                )
            elif head.startswith((_BUNDLE_MAGIC, _COMPRESSED_BUNDLE_MAGIC)):
                region = read(0, size)
            else:
                return KernelInventory(path=str(path))
    except OSError as error:
        return KernelInventory(path=str(path), unreadable=f"{type(error).__name__}: {error}")

    objects = []
    start = 0
    while (index := region.find(_BUNDLE_MAGIC, start)) != -1:
        for triple, begin, end in _bundle_entries(region, index):
            processor, features = _split_target(triple)
            objects.append(_kernel_code_object(triple, processor, features, region[begin:end]))
        start = index + len(_BUNDLE_MAGIC)
    return KernelInventory(
        path=str(path),
        code_objects=tuple(objects),
        compressed_bundles=_count_compressed_bundles(region),
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


def _host_runtime(llvm_root: Path) -> Path | None:
    clang_root = llvm_root / "lib" / "clang"
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

    # An instrumented tree that ships its own compiler runtime (TheRock's ASan
    # tarball does, under lib/llvm) is preferred to the stock ROCm root's, so the
    # host runtime matches the libraries it is loaded with.
    llvm_roots = list(
        dict.fromkeys((lib_dir / "llvm", lib_dir.parent / "llvm", core / "lib" / "llvm"))
    )
    explicit_runtime = env.get(ENV_ASAN_RUNTIME, "").strip()
    if explicit_runtime:
        host_runtime: Path | None = Path(explicit_runtime)
    else:
        host_runtime = next(
            (found for root in llvm_roots if (found := _host_runtime(root)) is not None), None
        )
    if host_runtime is None or not host_runtime.is_file():
        return None

    # TheRock's ASan tree keeps LLVM and system dependencies beside lib/, and its
    # documented LD_LIBRARY_PATH lists both.
    extra = tuple(
        directory
        for directory in (lib_dir / "llvm" / "lib", lib_dir / "rocm_sysdeps" / "lib")
        if safe_is_dir(directory)
    )
    symbolizer = next(
        (
            candidate
            for root in llvm_roots
            if (candidate := root / "bin" / "llvm-symbolizer").is_file()
        ),
        None,
    )
    return AsanRuntime(
        rocm_lib_dirs=(lib_dir, *extra),
        hip_runtime=hip_runtime,
        host_runtime=host_runtime,
        symbolizer=symbolizer,
    )


def asan_environment(
    base: Mapping[str, str], runtime: AsanRuntime, *, dispatch_log: Path | None = None
) -> dict[str, str]:
    """The child environment for an ASan run: the parent's, plus the contract.

    The instrumented libraries are prepended to ``LD_LIBRARY_PATH`` and the host
    runtime to ``LD_PRELOAD``, ahead of anything inherited. That is a deliberate
    exception to leaving the process under test's library path alone: the
    instrumented runtime IS the sanitizer here, as ``HSA_TOOLS_LIB`` is for
    ConSan, and a stock library ahead of it on the path silently turns every
    device detection into an abort. The preload is what lets an uninstrumented
    launcher (Python, a shell script) load instrumented libraries at all.
    ``dispatch_log`` pins HIP's kernel-dispatch logging to that file, replacing
    any inherited ``AMD_LOG_*`` setting.
    """

    env = dict(base)
    if dispatch_log is not None:
        env.update(_DISPATCH_LOG_ENV)
        env["AMD_LOG_LEVEL_FILE"] = str(dispatch_log)
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


def read_hip_log(output_dir: Path) -> str | None:
    """Every process's HIP runtime log from an ASan run, concatenated.

    ``None`` when no process wrote one, which is not the same as a log that
    names no kernel. The runtime's own messages land here rather than on stderr
    once AMD_LOG_LEVEL_FILE is set -- the unhandled-hostcall abort among them --
    so it is evidence for every runtime signal, not only for dispatches.
    """

    try:
        logs = sorted(output_dir.glob(f"{DISPATCH_LOG_NAME}*"))
    except OSError:
        return None
    if not logs:
        return None
    texts = []
    for log in logs:
        try:
            texts.append(log.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(texts)


def dispatched_kernels(hip_log: str) -> tuple[str, ...]:
    """Kernel signatures a HIP runtime log records as dispatched, in order."""

    return tuple(match.group("signature") for match in _DISPATCH_LINE.finditer(hip_log))


def _was_dispatched(kernel: str, signatures: Sequence[str]) -> bool:
    for signature in signatures:
        name = _function_name(signature)
        if kernel in (name, signature) or (kernel.startswith("_Z") and name in kernel):
            return True
    return False


def _workload_kernels(signatures: Sequence[str]) -> list[str]:
    return [
        name
        for name in dict.fromkeys(_function_name(signature) for signature in signatures)
        if name not in _RUNTIME_KERNELS and not name.startswith(_RUNTIME_KERNEL_PREFIX)
    ]


def evaluate_asan_output(
    process: ProcessResult,
    *,
    kernel: str | None = None,
    hip_log: str | None = None,
) -> CheckResult:
    """Decide one ASan run. Only a clean exit with no report is a PASS.

    A report is a result whatever the exit code says: ASan aborts the process on
    its first error by design, so a nonzero exit beside a report is the report,
    not a backend failure. ``hip_log`` (:func:`read_hip_log`) is read for the
    same signals as the process's own streams. When ``kernel`` is given, a clean
    exit also has to show that kernel dispatched in it: ASan leaves no trace of a
    kernel that ran clean, so without it a repro that never launched its kernel
    would pass.
    """

    streams = f"{process.stderr}\n{process.stdout}\n{hip_log or ''}"
    parsed = parse_asan_output(streams)
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
    # Whatever the exit code: a repro that reports its launch failure and still
    # exits 0 checked nothing.
    if _KERNEL_NOT_LOADED.search(streams):
        return _error(
            "asan_kernel_not_loaded: the kernel would not load, so nothing was "
            "checked -- an xnack+ code object needs HSA_XNACK=1 on an XNACK-capable GPU",
            returncode=process.returncode,
        )
    if process.returncode != 0:
        reason = f"asan_command_exit_{process.returncode}{_launch_diagnostic(process)}"
        if parsed.leak_reports:
            reason += ": LeakSanitizer ran although aorta pins detect_leaks=0"
        return _error(reason, returncode=process.returncode)
    if kernel is not None:
        if hip_log is None:
            return _error(
                "asan_dispatch_unverified: the HIP runtime wrote no dispatch log "
                "(AMD_LOG_LEVEL_FILE), so nothing shows the selected kernel ran",
                returncode=0,
            )
        dispatched = dispatched_kernels(hip_log)
        if not _was_dispatched(kernel, dispatched):
            ran = _workload_kernels(dispatched)
            return _error(
                f"asan_kernel_not_dispatched: {kernel} was never dispatched, so a clean "
                "exit says nothing about it; the run dispatched "
                + (", ".join(ran[:10]) if ran else "no workload kernel"),
                returncode=0,
            )
    return _check(state=ExecutionState.RAN, verdict=Verdict.PASS, returncode=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(
    objects: Sequence[Path], *, target: str, kernel: str
) -> tuple[CheckResult | None, list[str]]:
    """Fail closed unless the device code that can run is all ASan-instrumented.

    Every binary in ``objects`` that carries device code is held to
    :meth:`InstrumentationEvidence.problems`, and the selected kernel has to be in
    one of them, so a PASS cannot come from checking some other build.
    """

    descriptions: list[str] = []
    holders = []
    for path in objects:
        if not path.is_file():
            return _error(f"asan_object_not_found: {path}"), descriptions
        evidence = inspect_device_instrumentation(path, kernels=(kernel,))
        descriptions.append(evidence.describe())
        if not evidence.has_device_code:
            continue
        if problems := evidence.problems(target):
            advice = (
                ". Device code without ASan runs unchecked. -fsanitize=address is "
                "silently dropped for a target without :xnack+, and -fno-gpu-sanitize "
                f"drops it too; build only for --offload-arch={target}:xnack+"
                if any(problem.startswith("device code without ASan") for problem in problems)
                else ""
            )
            return (
                _error(f"asan_device_not_instrumented: {path}: {'; '.join(problems)}{advice}"),
                descriptions,
            )
        holders.append(evidence)
    if not holders:
        return (
            _error(
                "asan_device_not_instrumented: none of "
                f"{', '.join(str(path) for path in objects)} carries device code"
            ),
            descriptions,
        )
    if not any(evidence.holds_kernel(target, kernel) for evidence in holders):
        return (
            _error(
                f"asan_kernel_not_in_checked_code: {kernel} is in no {target} code object "
                "of the checked binaries, so their instrumentation says nothing about it"
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
    # no ASan runtime, so a mis-built repro is reported on any host. An ELF command
    # is always checked, named objects or not, so a recipe cannot verify one build
    # and run another; one without device code (a launcher) is skipped there.
    checked = list(dict.fromkeys(objects))
    if is_elf(command) and command not in checked:
        checked.insert(0, command)
    if selected.code_object is not None and Path(selected.code_object).is_file():
        checked.append(Path(selected.code_object))
    checked = list(dict.fromkeys(checked))
    if not checked:
        return _error(
            f"asan_instrumentation_unverifiable: {command} is not an ELF executable, "
            "so its device code cannot be inspected; name the instrumented binaries "
            "in sanitizer_plan.source.asan_objects"
        )
    failure, evidence = _preflight(checked, target=selected.target, kernel=selected.name)
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
            # A log left by an earlier run in this directory would vouch for a
            # dispatch this run never made.
            for stale in output_dir.glob(f"{DISPATCH_LOG_NAME}*"):
                stale.unlink()
            process = run_argv(
                (str(command),),
                timeout_seconds=timeout_seconds,
                env=asan_environment(
                    os.environ, resolved, dispatch_log=output_dir / DISPATCH_LOG_NAME
                ),
            )
            (output_dir / "asan.log").write_text(
                f"{process.stdout}\n{process.stderr}", encoding="utf-8"
            )
            hip_log = read_hip_log(output_dir)
            backend["dispatched_kernels"] = (
                ", ".join(_workload_kernels(dispatched_kernels(hip_log))) or "none"
                if hip_log is not None
                else "no dispatch log"
            )
            check = evaluate_asan_output(process, kernel=selected.name, hip_log=hip_log)

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
