"""GPU AddressSanitizer backend: parsing, preflight, runtime, and verdicts.

The parser fixtures are verbatim gfx950 output (see fixtures/asan/README.md);
every other input is synthetic and built here, so this file needs no GPU, no
ROCm and no compiler.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import stat
import struct
import zlib
from pathlib import Path

import pytest

from aorta.cia.autopsy.adapters.base import BundleContext
from aorta.cia.autopsy.adapters.sanitizer_report import (
    SIGNAL_MEMORY_ERROR,
    SIGNAL_RACE,
    SanitizerReportAdapter,
    classify_sanitizer,
)
from aorta.instrumentation.rocjitsu_sanitizers import (
    ExecutionState,
    FindingSeverity,
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
    Verdict,
    run_sanitizers,
    support,
)
from aorta.instrumentation.rocjitsu_sanitizers.asan import (
    ASAN_OPTIONS_PINNED,
    AsanRuntime,
    asan_environment,
    evaluate_asan_output,
    inspect_device_instrumentation,
    parse_asan_output,
    resolve_asan_runtime,
    run_asan,
)
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe
from aorta.triage.recipe import RecipeSchemaError

FIXTURES = Path(__file__).parent / "fixtures" / "asan"
REPO = Path(__file__).resolve().parents[3]


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _process(stderr: str = "", *, returncode: int | None = 0, stdout: str = "", **kw):
    return ProcessResult(argv=("repro",), returncode=returncode, stdout=stdout, stderr=stderr, **kw)


def _worklist(*names: str, target: str = "gfx950") -> KernelWorklist:
    names = names or ("store_index",)
    return KernelWorklist(
        requirement=SelectionRequirement.TOP_DISPATCH_COUNT,
        top_n=len(names),
        kernels=tuple(
            KernelObservation(
                identity=KernelIdentity(name=name, target=target),
                dispatch_count=1,
                sources=("test",),
            )
            for name in names
        ),
    )


def _meta(finding) -> dict[str, str]:
    return dict(finding.metadata)


# --------------------------------------------------------------------------- #
# Parsing real reports
# --------------------------------------------------------------------------- #


_RUNTIME_ONLY_STACKS = """\
==4242==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010 at pc 0x55 bp 0x7f sp 0x7f
READ of size 4 at 0x602000000010 thread T1
    #0 0x555555550001 in worker /src/app.cpp:12:5

0x602000000010 is located 0 bytes inside of 4-byte region [0x602000000010,0x602000000014)
freed by thread T0 here:
    #0 0x7f0000000001 in free /build/llvm-project/compiler-rt/lib/asan/asan_malloc_linux.cpp:52:3
    #1 0x7f0000000002  (/opt/rocm/lib/asan/libamdhip64.so.7+0xa3b226)

previously allocated by thread T0 here:
    #0 0x7f0000000003 in malloc /build/llvm-project/compiler-rt/lib/asan/asan_malloc_linux.cpp:68:3

Thread T1 created by T0 here:
    #0 0x7f0000000004 in pthread_create /build/llvm-project/compiler-rt/lib/asan/asan_interceptors.cpp:245:3
    #1 0x555555550002 in main /src/app.cpp:30:3

SUMMARY: AddressSanitizer: heap-use-after-free /src/app.cpp:12:5 in worker
"""


class TestParseRealReports:
    def test_a_stack_ends_where_its_frames_do(self):
        """Every frame of both stacks is runtime code, so neither site is known; the
        thread-creation line that follows them is not where anything was allocated."""

        (finding,) = parse_asan_output(_RUNTIME_ONLY_STACKS).findings
        metadata = _meta(finding)
        assert "allocated_at" not in metadata and "freed_at" not in metadata
        assert metadata["location"] == "/src/app.cpp:12:5"

    def test_a_frame_without_a_symbol_does_not_end_its_stack(self):
        report = _RUNTIME_ONLY_STACKS.replace(
            "    #1 0x7f0000000002  (/opt/rocm/lib/asan/libamdhip64.so.7+0xa3b226)\n",
            "    #1 0x7f0000000002  (/opt/rocm/lib/asan/libamdhip64.so.7+0xa3b226)\n"
            "    #2 0x555555550003 in release /src/app.cpp:20:7\n",
        )
        (finding,) = parse_asan_output(report).findings
        assert _meta(finding)["freed_at"] == "/src/app.cpp:20:7"

    def test_device_write_overflow_names_the_kernel_line_and_allocation(self):
        parsed = parse_asan_output(_fixture("device_heap_buffer_overflow_write.txt"))
        (finding,) = parsed.findings
        meta = _meta(finding)
        assert finding.sanitizer == "asan"
        assert finding.severity is FindingSeverity.MEMORY_ERROR
        assert finding.code == "heap-buffer-overflow"
        assert finding.kernel_name == "write_n_plus_one"
        assert meta["side"] == "device"
        assert meta["device"] == "0"
        assert meta["access"] == "WRITE"
        assert meta["size"] == "4"
        assert meta["location"] == "/work/src/asan_probe.hip:51:20"
        assert meta["region"].startswith("located 0 bytes after 4000-byte region")
        # The caller's hipMalloc line, past the ASan interceptor and HIP's
        # header-template wrapper that precede it in the allocation stack.
        assert meta["allocated_at"] == "/work/src/asan_probe.hip:93:3"
        assert meta["lanes"] == "1"

    def test_device_reports_carry_no_summary_line(self):
        """Why the header, not SUMMARY:, is the anchor: device reports omit it."""

        text = _fixture("device_heap_buffer_overflow_write.txt")
        assert "SUMMARY: AddressSanitizer" not in text
        assert len(parse_asan_output(text).findings) == 1

    def test_device_read_overflow_is_a_read(self):
        (finding,) = parse_asan_output(_fixture("device_heap_buffer_overflow_read.txt")).findings
        assert finding.code == "heap-buffer-overflow"
        assert finding.kernel_name == "read_n_plus_one"
        assert _meta(finding)["access"] == "READ"

    def test_use_after_free_reports_the_whole_wavefront(self):
        (finding,) = parse_asan_output(_fixture("device_heap_use_after_free.txt")).findings
        meta = _meta(finding)
        assert finding.code == "heap-use-after-free"
        assert finding.kernel_name == "write_n"
        assert meta["lanes"] == "64"
        # The free stack stops inside the runtime: no user frame, so none recorded.
        assert "freed_at" not in meta

    def test_lds_overflow_is_located_by_its_device_malloc_buffer(self):
        (finding,) = parse_asan_output(_fixture("device_lds_overflow.txt")).findings
        meta = _meta(finding)
        assert finding.kernel_name == "lds_oob"
        assert meta["device_malloc_neighbour"] == (
            "288 bytes above an address from a device malloc (or free) call of size 352"
        )
        assert "2097152-byte region" not in finding.message
        assert "288 bytes above" in finding.message

    def test_host_overflow_is_host_side_with_no_kernel(self):
        (finding,) = parse_asan_output(_fixture("host_heap_buffer_overflow.txt")).findings
        meta = _meta(finding)
        assert finding.kernel_name is None
        assert meta["side"] == "host"
        assert meta["function"] == "host_oob()"
        assert meta["location"] == "/work/src/asan_probe.hip:78:9"
        assert meta["allocated_at"] == "/work/src/asan_probe.hip:76:30"

    def test_stock_runtime_abort_is_a_lost_report_not_a_finding(self):
        parsed = parse_asan_output(_fixture("stock_runtime_hostcall_abort.txt"))
        assert parsed.findings == ()
        assert parsed.device_report_lost

    def test_leak_report_is_counted_and_is_not_a_memory_error(self):
        parsed = parse_asan_output(_fixture("stock_runtime_leak_report.txt"))
        assert parsed.findings == ()
        assert parsed.leak_reports == 1

    def test_uninstrumented_launcher_signal(self):
        parsed = parse_asan_output(_fixture("uninstrumented_launcher_not_preloaded.txt"))
        assert parsed.findings == ()
        assert parsed.runtime_not_preloaded

    def test_every_report_in_a_multi_process_log_is_kept(self):
        text = _fixture("device_heap_buffer_overflow_write.txt") + _fixture(
            "host_heap_buffer_overflow.txt"
        )
        sides = [_meta(finding)["side"] for finding in parse_asan_output(text).findings]
        assert sides == ["device", "host"]

    @pytest.mark.parametrize(
        "signature,name",
        [
            ("write_n_plus_one(int*, int)", "write_n_plus_one"),
            # Itanium demangling prints a template function's return type.
            ("void store_index<float>(float*, int)", "store_index<float>"),
            (
                "void rocprim::detail::k<std::pair<int, int> >(std::pair<int, int>*)",
                "rocprim::detail::k<std::pair<int, int> >",
            ),
            ("_triton_store_kernel", "_triton_store_kernel"),
        ],
    )
    def test_kernel_name_drops_parameters_and_return_type(self, signature, name):
        from aorta.instrumentation.rocjitsu_sanitizers.asan import _function_name

        assert _function_name(signature) == name

    @pytest.mark.parametrize(
        "header,kind",
        [
            ("attempting double-free on 0x602000000010 in thread T0:", "double-free"),
            (
                "attempting free on address which was not malloc()-ed: 0x7ffd in thread T0",
                "bad-free",
            ),
            (
                "requested allocation size 0xffffffffff (0x10000001000 after adjustments) "
                "exceeds maximum supported size of 0x10000000000",
                "allocation-size-too-big",
            ),
            ("SEGV on unknown address 0x000000000000 (pc 0x55 bp 0x7f sp 0x7f T0)", "SEGV"),
        ],
    )
    def test_headers_that_lead_with_a_verb_keep_their_error_class(self, header, kind):
        (finding,) = parse_asan_output(f"==7==ERROR: AddressSanitizer: {header}\n").findings
        assert finding.code == kind

    @pytest.mark.parametrize(
        "text",
        [
            "this binary was built with AddressSanitizer\n",
            "ERROR: AddressSanitizer: heap-buffer-overflow (no pid prefix)\n",
            "==12==ERROR: LeakSanitizer: detected memory leaks\n",
        ],
    )
    def test_text_that_only_mentions_asan_is_not_a_finding(self, text):
        assert parse_asan_output(text).findings == ()

    @pytest.mark.parametrize(
        "prefix",
        [
            "  ",
            # Program output with no trailing newline before the runtime writes.
            "step 3/10 loss=0.42",
            # ASan colours its output when stderr is a terminal.
            "\x1b[1m\x1b[31m",
            # log_exe_name=1 puts the executable before the pid.
            "==repro",
        ],
        ids=["indented", "after-unterminated-output", "ansi-colour", "exe-name"],
    )
    def test_a_header_that_does_not_start_its_line_is_found(self, prefix):
        before, after = _fixture("device_heap_buffer_overflow_write.txt").split("==203==", 1)
        (finding,) = parse_asan_output(f"{before}{prefix}==203=={after}").findings
        assert finding.kernel_name == "write_n_plus_one"

    def test_colour_codes_on_the_body_lines_are_stripped(self):
        """ASan's decorator colours the access, location and allocation lines too,
        each starting with an escape sequence, when it colours at all."""

        coloured = []
        for line in _fixture("device_heap_buffer_overflow_write.txt").splitlines():
            if line.startswith(("WRITE of size", "0x", "allocated by thread")):
                line = f"\x1b[1m\x1b[34m{line}\x1b[1m\x1b[0m"
            coloured.append(line)
        (finding,) = parse_asan_output("\n".join(coloured)).findings
        meta = _meta(finding)
        assert meta["access"] == "WRITE"
        assert meta["region"].startswith("located 0 bytes after 4000-byte region")
        assert meta["allocated_at"] == "/work/src/asan_probe.hip:93:3"


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


class TestEvaluate:
    def test_a_report_is_a_fail_whatever_the_exit_code(self):
        check = evaluate_asan_output(
            _process(_fixture("device_heap_buffer_overflow_write.txt"), returncode=1)
        )
        assert check.state is ExecutionState.RAN
        assert check.verdict is Verdict.FAIL
        assert check.returncode == 1
        assert check.findings[0].code == "heap-buffer-overflow"

    def test_a_report_on_stdout_is_still_read(self):
        """A wrapper that redirects 2>&1 must not turn a detection into an error."""

        check = evaluate_asan_output(
            _process(stdout=_fixture("device_heap_buffer_overflow_write.txt"), returncode=1)
        )
        assert check.verdict is Verdict.FAIL

    def test_a_clean_exit_with_no_report_is_the_only_pass(self):
        check = evaluate_asan_output(_process("asan repro: 1000 elements verified\n"))
        assert (check.state, check.verdict) == (ExecutionState.RAN, Verdict.PASS)

    def test_a_lost_device_report_is_an_error_not_a_pass(self):
        check = evaluate_asan_output(
            _process(_fixture("stock_runtime_hostcall_abort.txt"), returncode=134)
        )
        assert check.verdict is Verdict.ERROR
        assert check.reason.startswith("asan_device_report_lost:")

    def test_an_uninstrumented_launcher_is_an_error(self):
        check = evaluate_asan_output(
            _process(_fixture("uninstrumented_launcher_not_preloaded.txt"), returncode=1)
        )
        assert check.reason.startswith("asan_runtime_not_preloaded:")

    def test_xnack_off_is_explained(self):
        check = evaluate_asan_output(
            _process(_fixture("xnack_off_invalid_device_function.txt"), returncode=2)
        )
        assert check.verdict is Verdict.ERROR
        assert check.reason.startswith("asan_kernel_not_loaded:")
        assert "HSA_XNACK=1" in check.reason

    @pytest.mark.parametrize(
        "stderr",
        [
            "PROBE_HIP_ERROR hipGetLastError() -> "
            "hipErrorInvalidDeviceFunction (invalid device function)\n",
            "hipErrorNoBinaryForGpu: Unable to find code object for all current devices!\n",
            "Triton Error [HIP]: Code: 209, Messsage: no kernel image is available\n",
        ],
    )
    def test_a_kernel_that_never_loaded_is_an_error_even_at_exit_0(self, stderr):
        """A repro that reports its launch failure and still exits 0 checked nothing."""

        check = evaluate_asan_output(_process(stderr, returncode=0))
        assert check.verdict is Verdict.ERROR
        assert check.reason.startswith("asan_kernel_not_loaded:")

    def test_init_failure_is_an_error(self):
        check = evaluate_asan_output(
            _process(
                "==5==Shadow memory range interleaves with an existing memory mapping. "
                "ASan cannot proceed correctly. ABORTING.\n",
                returncode=1,
            )
        )
        assert check.reason.startswith("asan_init_failed:")

    def test_a_leak_report_with_no_memory_error_is_an_error(self):
        check = evaluate_asan_output(
            _process(_fixture("stock_runtime_leak_report.txt"), returncode=1)
        )
        assert check.verdict is Verdict.ERROR
        assert "LeakSanitizer" in check.reason

    def test_static_runtime_conflict_is_an_error(self):
        check = evaluate_asan_output(
            _process(
                "==9==Your application is linked against incompatible ASan runtimes.\n",
                returncode=1,
            )
        )
        assert check.reason.startswith("asan_incompatible_runtimes:")

    def test_any_other_nonzero_exit_is_an_error(self):
        check = evaluate_asan_output(_process("segfault elsewhere\n", returncode=139))
        assert (check.verdict, check.reason) == (Verdict.ERROR, "asan_command_exit_139")

    def test_timeout_keeps_any_finding(self):
        hung = evaluate_asan_output(_process("", returncode=None, timed_out=True))
        found = evaluate_asan_output(
            _process(
                _fixture("device_heap_buffer_overflow_write.txt"),
                returncode=None,
                timed_out=True,
            )
        )
        assert (hung.state, hung.verdict) == (ExecutionState.TIMED_OUT, Verdict.ERROR)
        assert (found.state, found.verdict) == (ExecutionState.TIMED_OUT, Verdict.FAIL)

    @pytest.mark.parametrize(
        "kernel,dispatched,ok",
        [
            ("store_index", ["store_index(int*, int)"], True),
            ("store_index(int*, int)", ["store_index(int*, int)"], True),
            ("_Z11store_indexPii", ["store_index(int*, int)"], True),
            ("store_index", ["amdgcn.device.init", "__amd_rocclr_initHeap"], False),
            ("store_index", ["store_index_v2(int*, int)"], False),
            # A mangled selection names one kernel: a namesake or another overload is not it.
            ("_Z14store_index_v2Pii", ["store_index(int*, int)"], False),
            ("_Z11store_indexPii", ["store_index(float*, int)"], False),
            ("store_index(int*, int)", ["store_index(float*, int)"], False),
            ("_Znot_a_mangled_name", ["store_index(int*, int)"], False),
            ("vec_add", ["void vec_add<float>(float*, float const*, float const*, int)"], True),
            ("foo", ["ns::foo(int)"], True),
        ],
    )
    def test_the_selected_kernel_must_be_among_the_dispatched(self, kernel, dispatched, ok):
        hip_log = "\n".join(f":3: 1 us:  ShaderName : {signature}" for signature in dispatched)
        check = evaluate_asan_output(_process(""), kernel=kernel, hip_log=hip_log)
        assert (check.verdict is Verdict.PASS) is ok

    def test_the_dispatch_log_is_read_from_every_process(self, tmp_path):
        from aorta.instrumentation.rocjitsu_sanitizers.asan import (
            dispatched_kernels,
            read_hip_log,
        )

        assert read_hip_log(tmp_path) is None
        (tmp_path / "hip_dispatch.log_10").write_text(
            ":3: 1 us:  ShaderName : amdgcn.device.init\n:3: 2 us:  ShaderName : warmup(float*)\n"
        )
        (tmp_path / "hip_dispatch.log_11").write_text(
            ":3: 3 us:  ShaderName : store_index(int*, int)\n"
        )
        assert dispatched_kernels(read_hip_log(tmp_path)) == (
            "amdgcn.device.init",
            "warmup(float*)",
            "store_index(int*, int)",
        )

    def test_a_lost_report_recorded_only_in_the_runtime_log_is_recognised(self):
        """Measured on gfx950: with AMD_LOG_LEVEL_FILE set, the HIP runtime writes its
        unhandled-hostcall abort to that file, not to stderr."""

        check = evaluate_asan_output(
            _process("", returncode=-6),
            kernel="store_index",
            hip_log=_fixture("stock_runtime_hostcall_abort.txt"),
        )
        assert check.reason.startswith("asan_device_report_lost:")

    def test_a_load_failure_recorded_only_in_the_runtime_log_is_recognised(self):
        check = evaluate_asan_output(
            _process("", returncode=0),
            kernel="store_index",
            hip_log=":3: 1 us:  hipLaunchKernel: Returned hipErrorInvalidDeviceFunction\n",
        )
        assert check.reason.startswith("asan_kernel_not_loaded:")

    def test_launch_error(self):
        check = evaluate_asan_output(
            _process(returncode=None, launch_error="PermissionError: denied")
        )
        assert check.reason == "asan_launch_error: PermissionError: denied"


# --------------------------------------------------------------------------- #
# Device instrumentation preflight
# --------------------------------------------------------------------------- #


def _bundle(*entries: tuple[str, bytes]) -> bytes:
    """An uncompressed clang offload bundle with the given (triple, payload) entries."""

    header_size = 24 + 8 + sum(24 + len(triple) for triple, _ in entries)
    table = b""
    payloads = b""
    offset = header_size
    for triple, payload in entries:
        table += struct.pack("<QQQ", offset, len(payload), len(triple)) + triple.encode()
        payloads += payload
        offset += len(payload)
    return b"__CLANG_OFFLOAD_BUNDLE__" + struct.pack("<Q", len(entries)) + table + payloads


def _amdgpu_elf(symbols, *, instrumented: bool, e_flags: int = 0x34F) -> bytes:
    """A real AMDGPU ELF defining ``symbols`` as kernels, each with its ``.kd`` descriptor."""

    strings = b"\0"
    table = b"\0" * 24
    for name in (*symbols, *(f"{symbol}.kd" for symbol in symbols)):
        table += struct.pack("<IBBHQQ", len(strings), 0x12, 0, 1, 0, 0)
        strings += name.encode() + b"\0"
    text = b"\0call __asan_report_store4\0__asan_init\0" if instrumented else b"\0plain code\0"
    sections = [
        (".text", 1, text, 0, 0),
        (".symtab", 2, table, 3, 24),
        (".strtab", 3, strings, 0, 0),
    ]
    names = b"\0"
    name_offsets = []
    for name, *_ in sections:
        name_offsets.append(len(names))
        names += name.encode() + b"\0"
    shstrtab = len(names)
    names += b".shstrtab\0"
    body = b""
    offsets = []
    for _, _, data, _, _ in sections:
        offsets.append(64 + len(body))
        body += data
    names_offset = 64 + len(body)
    body += names
    count = len(sections) + 2
    header = (
        b"\x7fELF\x02\x01\x01"
        + b"\0" * 9
        + struct.pack(
            "<HHIQQQIHHHHHH",
            1,
            224,
            1,
            0,
            0,
            64 + len(body),
            e_flags,
            64,
            0,
            0,
            64,
            count,
            count - 1,
        )
    )
    shdrs = b"\0" * 64
    for (_, kind, data, link, entsize), offset, name_offset in zip(
        sections, offsets, name_offsets, strict=True
    ):
        shdrs += struct.pack(
            "<IIQQQQIIQQ", name_offset, kind, 0, 0, offset, len(data), link, 0, 1, entsize
        )
    shdrs += struct.pack("<IIQQQQIIQQ", shstrtab, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
    return header + body + shdrs


def _device_elf(*, instrumented: bool, kernel: str = "store_index") -> bytes:
    """A code object defining ``kernel(int*, int)``."""

    return _amdgpu_elf([f"_Z{len(kernel)}{kernel}Pii"], instrumented=instrumented)


def _bare_code_object(tmp_path: Path, *, e_flags: int, instrumented: bool) -> Path:
    """A standalone AMDGPU code object, i.e. a ``.hsaco``."""

    path = tmp_path / "kernel.hsaco"
    path.write_bytes(
        _amdgpu_elf(["_Z11store_indexPii"], instrumented=instrumented, e_flags=e_flags)
    )
    return path


_HOST = "host-x86_64-unknown-linux-gnu-"


def _binary(tmp_path: Path, *bundles: bytes, host: bytes = b"host code", name="repro") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x7fELF" + b"\0" * 60 + host + b"".join(bundles) + b"\0trailer")
    return path


def _libzstd():
    try:
        return ctypes.CDLL(ctypes.util.find_library("zstd") or "libzstd.so.1")
    except OSError:
        return None


def _zstd_compress(data: bytes) -> bytes:
    library = _libzstd()
    library.ZSTD_compressBound.restype = ctypes.c_size_t
    library.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    library.ZSTD_compress.restype = ctypes.c_size_t
    library.ZSTD_compress.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    bound = library.ZSTD_compressBound(len(data))
    out = ctypes.create_string_buffer(bound)
    written = library.ZSTD_compress(out, bound, data, len(data), 3)
    return out.raw[:written]


_ZLIB, _ZSTD, _NOT_A_METHOD = 0, 1, 2


def _ccob(
    bundle: bytes, *, method: int = _ZLIB, version: int = 2, payload=None, size=None
) -> bytes:
    """A compressed offload bundle, laid out as clang-offload-bundler writes one."""

    if payload is None:
        payload = zlib.compress(bundle) if method == _ZLIB else _zstd_compress(bundle)
    size = len(bundle) if size is None else size
    if version == 2:
        header = struct.pack("<HHIIQ", version, method, 24 + len(payload), size, 0)
    else:
        header = struct.pack("<HHQQQ", version, method, 32 + len(payload), size, 0)
    return b"CCOB" + header + payload


# A zstd header whose payload is not there: well formed, and impossible to open.
_UNOPENABLE = b"CCOB" + struct.pack("<HH", 2, _ZSTD) + b"\0" * 32


class TestPreflight:
    def test_an_xnack_plus_instrumented_code_object_counts(self, tmp_path):
        binary = _binary(
            tmp_path,
            _bundle(
                ("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True)),
                (_HOST, b""),
            ),
        )
        evidence = inspect_device_instrumentation(binary)
        assert evidence.instrumented_for("gfx950")
        (code_object,) = evidence.code_objects
        assert (code_object.processor, code_object.features) == ("gfx950", ("xnack+",))

    def test_host_side_asan_strings_do_not_vouch_for_the_device(self, tmp_path):
        """Measured on gfx950: ``-fsanitize=address -fno-gpu-sanitize`` with a
        ``gfx950:xnack+`` target keeps the xnack+ code object and 20 host-side
        ``__asan_report_*`` strings, and the GPU code carries no instrumentation.
        Only reading the marker inside the device ELF tells the two apart."""

        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=False))),
            host=b"\0__asan_report_store4\0__asan_report_load4\0",
        )
        evidence = inspect_device_instrumentation(binary)
        assert b"__asan_report" in binary.read_bytes()
        assert not evidence.instrumented_for("gfx950")

    def test_asan_symbols_without_xnack_plus_do_not_count(self, tmp_path):
        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950", _device_elf(instrumented=True))),
        )
        assert not inspect_device_instrumentation(binary).instrumented_for("gfx950")

    def test_xnack_plus_without_asan_does_not_count(self, tmp_path):
        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=False))),
        )
        assert not inspect_device_instrumentation(binary).instrumented_for("gfx950")

    def test_another_targets_instrumentation_does_not_count(self, tmp_path):
        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx942:xnack+", _device_elf(instrumented=True))),
        )
        evidence = inspect_device_instrumentation(binary)
        assert evidence.instrumented_for("gfx942")
        assert evidence.problems("gfx950") == ["no gfx950 code object"]

    def test_an_unchecked_object_for_another_target_still_fails(self, tmp_path):
        """The GPU that runs the binary is not necessarily the one the recipe names:
        a gfx942 box behind a gfx950 label would run the unchecked object."""

        binary = _binary(
            tmp_path,
            _bundle(
                ("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True)),
                ("hipv4-amdgcn-amd-amdhsa--gfx942", _device_elf(instrumented=False)),
            ),
        )
        problems = inspect_device_instrumentation(binary).problems("gfx950")
        assert problems == ["device code without ASan: hipv4-amdgcn-amd-amdhsa--gfx942"]

    def test_other_target_features_beside_xnack(self, tmp_path):
        binary = _binary(
            tmp_path,
            _bundle(
                (
                    "hipv4-amdgcn-amd-amdhsa--gfx950:sramecc+:xnack+",
                    _device_elf(instrumented=True),
                )
            ),
        )
        assert inspect_device_instrumentation(binary).instrumented_for("gfx950")

    def test_one_unchecked_translation_unit_fails_the_binary(self, tmp_path):
        """One bundle per translation unit, and the runtime loads each: a TU built
        without :xnack+ still runs with XNACK on, unchecked, beside a checked one."""

        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950", _device_elf(instrumented=False))),
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True))),
        )
        evidence = inspect_device_instrumentation(binary)
        assert len(evidence.code_objects) == 2
        assert not evidence.instrumented_for("gfx950")
        assert "device code without ASan: hipv4-amdgcn-amd-amdhsa--gfx950" in (
            evidence.problems("gfx950")
        )

    def test_every_translation_unit_instrumented_is_fine(self, tmp_path):
        triple = "hipv4-amdgcn-amd-amdhsa--gfx950:xnack+"
        binary = _binary(
            tmp_path,
            _bundle((triple, _device_elf(instrumented=True))),
            _bundle((triple, _device_elf(instrumented=True, kernel="other_kernel"))),
        )
        evidence = inspect_device_instrumentation(binary, kernels=("store_index", "other_kernel"))
        assert evidence.instrumented_for("gfx950")
        assert evidence.holds_kernel("gfx950", "store_index")
        assert evidence.holds_kernel("gfx950", "other_kernel")
        assert not evidence.holds_kernel("gfx950", "absent_kernel")

    @pytest.mark.parametrize(
        "selection,found",
        [
            ("store_index", True),
            ("store_index(int*, int)", True),
            ("_Z11store_indexPii", True),
            ("store_index(float*, int)", False),
            ("_Z11store_indexPfi", False),
            ("store_index_v2", False),
            # A byte search would find these inside the mangled symbol.
            ("index", False),
            ("Pii", False),
        ],
    )
    def test_a_selection_is_matched_against_the_kernel_symbols(self, tmp_path, selection, found):
        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True))),
        )
        evidence = inspect_device_instrumentation(binary, kernels=(selection,))
        assert evidence.holds_kernel("gfx950", selection) is found

    @pytest.mark.parametrize(
        "selection",
        [
            "foo",
            "ns::foo",
            "ns::foo<float>",
            "ns::foo<float>(float*)",
            "void ns::foo<float>(float*)",
        ],
    )
    def test_template_and_namespaced_kernels_match_by_name(self, tmp_path, selection):
        elf = _amdgpu_elf(["_ZN2ns3fooIfEEvPT_"], instrumented=True)
        binary = _binary(tmp_path, _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", elf)))
        assert inspect_device_instrumentation(binary, kernels=(selection,)).holds_kernel(
            "gfx950", selection
        )

    def test_an_unopenable_compressed_bundle_beside_a_readable_one_fails(self, tmp_path):
        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True))),
            _UNOPENABLE,
        )
        (problem,) = inspect_device_instrumentation(binary).problems("gfx950")
        assert problem.startswith("1 compressed offload bundle(s) that aorta could not open")

    @pytest.mark.parametrize("version", [2, 3])
    def test_a_compressed_bundle_is_inflated_and_counts(self, tmp_path, version):
        """PyTorch's ROCm build compresses by default, so this is the common case."""

        triple = "hipv4-amdgcn-amd-amdhsa--gfx950:xnack+"
        compressed = _ccob(_bundle((triple, _device_elf(instrumented=True))), version=version)
        evidence = inspect_device_instrumentation(
            _binary(tmp_path, compressed), kernels=("store_index",)
        )
        assert evidence.compressed_bundles == 0
        assert evidence.instrumented_for("gfx950")
        assert evidence.holds_kernel("gfx950", "store_index")

    def test_inflated_code_is_held_to_the_every_object_rule(self, tmp_path):
        compressed = _ccob(
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950", _device_elf(instrumented=False)))
        )
        binary = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True))),
            compressed,
        )
        assert inspect_device_instrumentation(binary).problems("gfx950") == [
            "device code without ASan: hipv4-amdgcn-amd-amdhsa--gfx950"
        ]

    @pytest.mark.skipif(_libzstd() is None, reason="no libzstd to compress with")
    def test_a_zstd_bundle_is_inflated(self, tmp_path):
        """zstd is what hipcc --offload-compress writes (measured: method 1)."""

        triple = "hipv4-amdgcn-amd-amdhsa--gfx950:xnack+"
        compressed = _ccob(_bundle((triple, _device_elf(instrumented=True))), method=_ZSTD)
        evidence = inspect_device_instrumentation(_binary(tmp_path, compressed))
        assert evidence.compressed_bundles == 0
        assert evidence.instrumented_for("gfx950")

    def test_a_header_with_an_unknown_method_is_not_a_bundle(self, tmp_path):
        stray = b"CCOB" + struct.pack("<HH", 2, _NOT_A_METHOD) + b"\0" * 32
        assert inspect_device_instrumentation(_binary(tmp_path, stray)).compressed_bundles == 0

    def test_a_corrupt_payload_is_counted_as_unopened(self, tmp_path):
        corrupt = _ccob(b"", payload=b"not zlib at all", size=4096)
        evidence = inspect_device_instrumentation(_binary(tmp_path, corrupt))
        assert (evidence.code_objects, evidence.compressed_bundles) == ((), 1)

    def test_an_absurd_size_is_not_inflated(self, tmp_path):
        """The header names the buffer to allocate; a corrupt one must not get it."""

        bundle = _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True)))
        bomb = _ccob(bundle, version=3, size=1 << 40)
        evidence = inspect_device_instrumentation(_binary(tmp_path, bomb))
        assert (evidence.code_objects, evidence.compressed_bundles) == ((), 1)

    @pytest.mark.skipif(_libzstd() is None, reason="no libzstd to compress with")
    def test_an_absurd_size_is_not_allocated_for_zstd(self, tmp_path):
        """zstd decompresses into a buffer of the header's size, so the bound is all that
        stands between a corrupt header and a terabyte allocation."""

        bundle = _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True)))
        bomb = _ccob(bundle, method=_ZSTD, version=3, size=1 << 40)
        evidence = inspect_device_instrumentation(_binary(tmp_path, bomb))
        assert (evidence.code_objects, evidence.compressed_bundles) == ((), 1)
        # Linux overcommit would let the allocation itself succeed lazily, so pin the
        # guard: a header that claims this much is refused before anything is allocated.
        import aorta.instrumentation.rocjitsu_sanitizers.asan as asan_module

        assert asan_module._ccob_extent(bomb, 0) is None

    def test_a_size_that_does_not_match_the_payload_is_not_trusted(self, tmp_path):
        bundle = _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950:xnack+", _device_elf(instrumented=True)))
        evidence = inspect_device_instrumentation(
            _binary(tmp_path, _ccob(bundle, size=len(bundle) - 1))
        )
        assert (evidence.code_objects, evidence.compressed_bundles) == ((), 1)

    def test_an_inflated_payload_that_is_not_a_bundle_is_unopened(self, tmp_path):
        evidence = inspect_device_instrumentation(_binary(tmp_path, _ccob(b"just some bytes")))
        assert (evidence.code_objects, evidence.compressed_bundles) == ((), 1)

    @pytest.mark.parametrize(
        "e_flags,instrumented,expected",
        [
            (0x74F, True, True),  # gfx950, sramecc any, xnack+ -- measured
            (0x54F, True, False),  # xnack any
            (0x74F, False, False),  # -fno-gpu-sanitize shape: xnack+ without ASan
            (0x74C, True, False),  # gfx942
        ],
    )
    def test_a_bare_code_object_is_read_from_its_elf_header(
        self, tmp_path, e_flags, instrumented, expected
    ):
        path = _bare_code_object(tmp_path, e_flags=e_flags, instrumented=instrumented)
        evidence = inspect_device_instrumentation(path, kernels=("store_index",))
        assert evidence.instrumented_for("gfx950") is expected
        (code_object,) = evidence.code_objects
        assert code_object.kernels_found == ("store_index",)

    def test_a_truncated_or_bogus_bundle_is_skipped(self, tmp_path):
        bogus = b"__CLANG_OFFLOAD_BUNDLE__" + struct.pack("<Q", 3) + b"\x01\x02"
        binary = _binary(tmp_path, bogus)
        evidence = inspect_device_instrumentation(binary)
        assert evidence.code_objects == ()
        assert not evidence.instrumented_for("gfx950")

    def test_an_unopenable_compressed_bundle_is_counted_not_guessed(self, tmp_path):
        evidence = inspect_device_instrumentation(_binary(tmp_path, _UNOPENABLE))
        assert evidence.code_objects == ()
        assert evidence.compressed_bundles == 1

    def test_stray_ccob_bytes_are_not_a_compressed_bundle(self, tmp_path):
        """Four bytes recur by chance in a large binary; the header fields must fit."""

        stray = b"CCOB" + struct.pack("<HH", 0x4142, 0x4344)
        assert inspect_device_instrumentation(_binary(tmp_path, stray)).compressed_bundles == 0


# --------------------------------------------------------------------------- #
# Runtime resolution and environment
# --------------------------------------------------------------------------- #


def _rocm_tree(root: Path, *, instrumented: bool = True, host_runtime: bool = True) -> Path:
    """A fake classic ROCm root with an lib/asan runtime and a clang ASan runtime."""

    asan = root / "lib" / "asan"
    asan.mkdir(parents=True)
    marker = b"\0__asan_init\0__asan_report_load4\0" if instrumented else b"\0stock\0"
    (asan / "libamdhip64.so.7").write_bytes(b"\x7fELF" + marker)
    clang = root / "lib" / "llvm" / "lib" / "clang" / "20" / "lib" / "linux"
    clang.mkdir(parents=True)
    if host_runtime:
        (clang / "libclang_rt.asan-x86_64.so").write_bytes(b"\x7fELF")
    symbolizer = root / "lib" / "llvm" / "bin"
    symbolizer.mkdir(parents=True)
    (symbolizer / "llvm-symbolizer").write_bytes(b"#!/bin/sh\n")
    return root


class TestRuntime:
    def test_resolves_lib_asan_beside_the_rocm_root(self, tmp_path):
        root = _rocm_tree(tmp_path / "rocm")
        runtime = resolve_asan_runtime({"ROCM_PATH": str(root)})
        assert runtime is not None
        assert runtime.rocm_lib_dirs == (root / "lib" / "asan",)
        assert runtime.hip_runtime == root / "lib" / "asan" / "libamdhip64.so.7"
        assert runtime.host_runtime.name == "libclang_rt.asan-x86_64.so"
        assert runtime.symbolizer == root / "lib" / "llvm" / "bin" / "llvm-symbolizer"

    def test_a_stock_runtime_directory_is_refused(self, tmp_path):
        """The stock libamdhip64 loads and then aborts on the first detection."""

        root = _rocm_tree(tmp_path / "rocm", instrumented=False)
        assert resolve_asan_runtime({"ROCM_PATH": str(root)}) is None

    def test_no_host_runtime_is_no_runtime(self, tmp_path):
        root = _rocm_tree(tmp_path / "rocm", host_runtime=False)
        assert resolve_asan_runtime({"ROCM_PATH": str(root)}) is None

    def test_explicit_overrides_win(self, tmp_path):
        root = _rocm_tree(tmp_path / "rocm")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "libamdhip64.so.7.0").write_bytes(b"\0__asan_init\0")
        preload = tmp_path / "libclang_rt.asan-x86_64.so"
        preload.write_bytes(b"\x7fELF")
        runtime = resolve_asan_runtime(
            {
                "ROCM_PATH": str(root),
                "AORTA_ASAN_ROCM_LIB": str(elsewhere),
                "AORTA_ASAN_RUNTIME": str(preload),
            }
        )
        assert runtime is not None
        assert runtime.rocm_lib_dirs == (elsewhere,)
        assert runtime.host_runtime == preload

    def test_therock_asan_tree_brings_its_llvm_and_sysdeps_dirs(self, tmp_path):
        root = _rocm_tree(tmp_path / "rocm")
        therock = tmp_path / "therock-asan"
        (therock / "lib" / "llvm" / "lib").mkdir(parents=True)
        (therock / "lib" / "rocm_sysdeps" / "lib").mkdir(parents=True)
        (therock / "lib" / "libamdhip64.so.7").write_bytes(b"\0__asan_init\0")
        runtime = resolve_asan_runtime({"ROCM_PATH": str(root), "ROCM_ASAN_PATH": str(therock)})
        assert runtime is not None
        assert runtime.rocm_lib_dirs == (
            therock / "lib",
            therock / "lib" / "llvm" / "lib",
            therock / "lib" / "rocm_sysdeps" / "lib",
        )
        # No compiler runtime of its own here, so the ROCm root's is used.
        assert runtime.host_runtime.is_relative_to(root)

    def test_an_asan_tree_with_its_own_compiler_runtime_uses_it(self, tmp_path):
        root = _rocm_tree(tmp_path / "rocm")
        therock = tmp_path / "therock-asan"
        (therock / "lib").mkdir(parents=True)
        (therock / "lib" / "libamdhip64.so.7").write_bytes(b"\0__asan_init\0")
        own = therock / "lib" / "llvm" / "lib" / "clang" / "23" / "lib" / "linux"
        own.mkdir(parents=True)
        (own / "libclang_rt.asan-x86_64.so").write_bytes(b"\x7fELF")
        runtime = resolve_asan_runtime({"ROCM_PATH": str(root), "ROCM_ASAN_PATH": str(therock)})
        assert runtime is not None
        assert runtime.host_runtime == own / "libclang_rt.asan-x86_64.so"

    def test_environment_pins_the_contract_over_hostile_inheritance(self, tmp_path):
        runtime = resolve_asan_runtime({"ROCM_PATH": str(_rocm_tree(tmp_path / "rocm"))})
        env = asan_environment(
            {
                "ASAN_OPTIONS": "exitcode=0:halt_on_error=0",
                "HSA_XNACK": "0",
                "LD_LIBRARY_PATH": "/operator/lib",
                "LD_PRELOAD": "/operator/libhook.so",
                "UNRELATED": "kept",
            },
            runtime,
        )
        assert env["ASAN_OPTIONS"] == ASAN_OPTIONS_PINNED
        options = dict(item.split("=", 1) for item in env["ASAN_OPTIONS"].split(":"))
        assert options == {
            "detect_leaks": "0",
            "halt_on_error": "1",
            "exitcode": "1",
            "symbolize": "1",
            "color": "never",
        }
        assert env["HSA_XNACK"] == "1"
        library_path = env["LD_LIBRARY_PATH"].split(os.pathsep)
        assert library_path[0] == str(runtime.rocm_lib_dirs[0])
        assert library_path[-1] == "/operator/lib"
        assert env["LD_PRELOAD"] == f"{runtime.host_runtime}:/operator/libhook.so"
        assert env["ASAN_SYMBOLIZER_PATH"] == str(runtime.symbolizer)
        assert env["UNRELATED"] == "kept"
        assert "AMD_LOG_LEVEL_FILE" not in env

    def test_dispatch_logging_is_pinned_over_an_inherited_log_level(self, tmp_path):
        runtime = resolve_asan_runtime({"ROCM_PATH": str(_rocm_tree(tmp_path / "rocm"))})
        env = asan_environment(
            {"AMD_LOG_LEVEL": "4", "AMD_LOG_MASK": "0x7fffffff"},
            runtime,
            dispatch_log=tmp_path / "hip_dispatch.log",
        )
        # Decimal: the runtime reads a 0x-prefixed mask as selecting nothing.
        assert (env["AMD_LOG_LEVEL"], env["AMD_LOG_MASK"]) == ("3", "128")
        assert env["AMD_LOG_LEVEL_FILE"] == str(tmp_path / "hip_dispatch.log")


# --------------------------------------------------------------------------- #
# run_asan end to end, with a real subprocess and synthetic binaries
# --------------------------------------------------------------------------- #


def _script(
    tmp_path: Path,
    *,
    stderr_fixture: str | None,
    code: int,
    dispatch: str | None = "store_index(int*, int)",
) -> Path:
    """A stand-in repro. ``dispatch`` is the kernel it logs as dispatched, the way
    the HIP runtime does under AMD_LOG_LEVEL_FILE (one file per pid)."""

    marker = tmp_path / "ran"
    lines = [
        "#!/bin/sh",
        f'echo ran > "{marker}"',
        'echo "ASAN_OPTIONS=$ASAN_OPTIONS HSA_XNACK=$HSA_XNACK"',
    ]
    if dispatch is not None:
        lines.append(f'echo ":3: 1 us:  ShaderName : {dispatch}" >> "${{AMD_LOG_LEVEL_FILE}}_$$"')
    if stderr_fixture is not None:
        lines.append(f'cat "{FIXTURES / stderr_fixture}" >&2')
    lines.append(f"exit {code}")
    script = tmp_path / "repro.sh"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _instrumented(tmp_path: Path, *, instrumented: bool = True) -> Path:
    triple = "hipv4-amdgcn-amd-amdhsa--gfx950:xnack+"
    return _binary(
        tmp_path,
        _bundle((triple, _device_elf(instrumented=instrumented))),
        name="repro.bin",
    )


@pytest.fixture
def runtime(tmp_path) -> AsanRuntime:
    resolved = resolve_asan_runtime({"ROCM_PATH": str(_rocm_tree(tmp_path / "rocm"))})
    assert resolved is not None
    return resolved


class TestRunAsan:
    def test_a_detection_fails_and_is_attributed_to_the_selected_kernel(self, tmp_path, runtime):
        script = _script(tmp_path, stderr_fixture="device_heap_buffer_overflow_write.txt", code=1)
        check = run_asan(
            _worklist(),
            command=script,
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert (check.state, check.verdict) == (ExecutionState.RAN, Verdict.FAIL)
        (kernel,) = check.kernel_results
        assert kernel.identity.name == "store_index"
        assert kernel.findings == check.findings
        backend = dict(check.backend)
        assert backend["asan_options"] == ASAN_OPTIONS_PINNED
        assert "gfx950:xnack+[asan=yes]" in backend["device_instrumentation"]
        assert len(backend["hip_runtime_sha256"]) == 64
        log = (tmp_path / "out" / "asan.log").read_text()
        assert f"ASAN_OPTIONS={ASAN_OPTIONS_PINNED} HSA_XNACK=1" in log

    def test_a_clean_run_passes(self, tmp_path, runtime):
        check = run_asan(
            _worklist(),
            command=_script(tmp_path, stderr_fixture=None, code=0),
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert (check.state, check.verdict) == (ExecutionState.RAN, Verdict.PASS)

    def test_a_clean_run_that_logged_no_dispatch_is_unverified(self, tmp_path, runtime):
        check = run_asan(
            _worklist(),
            command=_script(tmp_path, stderr_fixture=None, code=0, dispatch=None),
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert check.verdict is Verdict.ERROR
        assert check.reason.startswith("asan_dispatch_unverified:")

    def test_a_clean_run_that_dispatched_another_kernel_is_an_error(self, tmp_path, runtime):
        """The ASan-adjacent failure the review found: a repro that never launched
        the selected kernel exits 0 with no report."""

        check = run_asan(
            _worklist(),
            command=_script(tmp_path, stderr_fixture=None, code=0, dispatch="warmup(float*)"),
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert check.reason.startswith("asan_kernel_not_dispatched: store_index")
        assert "warmup" in check.reason
        assert dict(check.backend)["dispatched_kernels"] == "warmup"

    def test_a_stale_dispatch_log_does_not_count(self, tmp_path, runtime):
        out = tmp_path / "out"
        out.mkdir()
        (out / "hip_dispatch.log_1").write_text("ShaderName : store_index(int*, int)\n")
        check = run_asan(
            _worklist(),
            command=_script(tmp_path, stderr_fixture=None, code=0, dispatch=None),
            output_dir=out,
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert check.reason.startswith("asan_dispatch_unverified:")

    def test_a_detection_needs_no_dispatch_evidence(self, tmp_path, runtime):
        check = run_asan(
            _worklist(),
            command=_script(
                tmp_path,
                stderr_fixture="device_heap_buffer_overflow_write.txt",
                code=1,
                dispatch=None,
            ),
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert check.verdict is Verdict.FAIL

    def test_an_uninstrumented_build_errors_before_anything_runs(self, tmp_path, runtime):
        script = _script(tmp_path, stderr_fixture=None, code=0)
        check = run_asan(
            _worklist(),
            command=script,
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path, instrumented=False),),
        )
        assert check.verdict is Verdict.ERROR
        assert check.reason.startswith("asan_device_not_instrumented:")
        assert "--offload-arch=gfx950:xnack+" in check.reason
        assert not (tmp_path / "ran").exists()

    def test_a_script_with_no_named_objects_is_unverifiable(self, tmp_path, runtime):
        check = run_asan(
            _worklist(),
            command=_script(tmp_path, stderr_fixture=None, code=0),
            output_dir=tmp_path / "out",
            runtime=runtime,
        )
        assert check.reason.startswith("asan_instrumentation_unverifiable:")
        assert not (tmp_path / "ran").exists()

    def test_an_elf_command_is_its_own_object(self, tmp_path, runtime):
        binary = _instrumented(tmp_path, instrumented=False)
        check = run_asan(_worklist(), command=binary, output_dir=tmp_path / "out", runtime=runtime)
        assert check.reason.startswith("asan_device_not_instrumented:")

    def test_an_elf_command_is_checked_even_beside_named_objects(self, tmp_path, runtime):
        """Naming objects must not let a recipe verify one build and run another."""

        unchecked = _binary(
            tmp_path,
            _bundle(("hipv4-amdgcn-amd-amdhsa--gfx950", _device_elf(instrumented=False))),
            name="build-plain-app",
        )
        check = run_asan(
            _worklist(),
            command=unchecked,
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert check.reason.startswith("asan_device_not_instrumented:")
        assert "build-plain-app" in check.reason

    def test_an_elf_launcher_with_no_device_code_is_not_held_to_it(self, tmp_path, runtime):
        launcher = tmp_path / "launcher"
        launcher.write_bytes(Path("/bin/true").read_bytes())
        launcher.chmod(0o755)
        check = run_asan(
            _worklist(),
            command=launcher,
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        # Past the preflight and run; /bin/true dispatches nothing, which is the
        # next rule's business, not the preflight's.
        assert check.reason.startswith("asan_dispatch_unverified:")

    def test_a_kernel_absent_from_the_checked_code_is_refused(self, tmp_path, runtime):
        check = run_asan(
            _worklist("some_other_kernel"),
            command=_script(tmp_path, stderr_fixture=None, code=0),
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
        )
        assert check.reason.startswith("asan_kernel_not_in_checked_code:")
        assert not (tmp_path / "ran").exists()

    def test_no_runtime_is_not_checked_and_runs_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROCM_PATH", str(tmp_path / "missing"))
        monkeypatch.delenv("AORTA_ASAN_ROCM_LIB", raising=False)
        monkeypatch.delenv("ROCM_ASAN_PATH", raising=False)
        import aorta.instrumentation.rocjitsu_sanitizers.asan as asan_module

        monkeypatch.setattr(asan_module, "resolve_asan_runtime", lambda environ=None: None)
        check = run_asan(
            _worklist(),
            command=_script(tmp_path, stderr_fixture=None, code=0),
            output_dir=tmp_path / "out",
            objects=(_instrumented(tmp_path),),
        )
        assert check.state is ExecutionState.NOT_CHECKED
        assert check.reason.startswith("asan_runtime_not_found:")
        assert not (tmp_path / "ran").exists()

    def test_a_mis_built_repro_is_reported_even_without_a_runtime(self, tmp_path, monkeypatch):
        import aorta.instrumentation.rocjitsu_sanitizers.asan as asan_module

        monkeypatch.setattr(asan_module, "resolve_asan_runtime", lambda environ=None: None)
        check = run_asan(
            _worklist(),
            command=_script(tmp_path, stderr_fixture=None, code=0),
            output_dir=tmp_path / "out",
            objects=(_instrumented(tmp_path, instrumented=False),),
        )
        assert check.reason.startswith("asan_device_not_instrumented:")

    @pytest.mark.parametrize(
        "worklist,target,reason",
        [
            (_worklist("a", "b"), None, "asan_requires_one_targeted_repro"),
            (
                _worklist(),
                KernelIdentity(name="other", target="gfx950"),
                "asan_target_does_not_match_worklist",
            ),
        ],
    )
    def test_scoping_fails_closed(self, tmp_path, runtime, worklist, target, reason):
        check = run_asan(
            worklist,
            command=_script(tmp_path, stderr_fixture=None, code=0),
            output_dir=tmp_path / "out",
            runtime=runtime,
            objects=(_instrumented(tmp_path),),
            target=target,
        )
        assert (check.state, check.reason) == (ExecutionState.NOT_CHECKED, reason)

    def test_a_missing_command_is_not_checked(self, tmp_path, runtime):
        check = run_asan(
            _worklist(), command=tmp_path / "absent", output_dir=tmp_path, runtime=runtime
        )
        assert check.reason == "asan_command_not_found"


# --------------------------------------------------------------------------- #
# Recipe, pipeline, support matrix
# --------------------------------------------------------------------------- #

_RECIPE = """\
schema_version: 1
mode: sanitizer
sanitizer_plan:
  target: gfx950
  source:
    kind: kernel
    kernel: {{name: store_index}}
{source_extra}
  scope: {{kind: kernel}}
  selection: {{requirement: top_dispatch_count, top_n: 1}}
  sanitizers: [{sanitizers}]
  policy: {{consan_policy: strict, on_missing_backend: fail}}
  output: {{report: sanitizer_report.json}}
"""


def _recipe(tmp_path: Path, *, source_extra: str = "", sanitizers: str = "asan") -> Path:
    path = tmp_path / "recipe.yaml"
    path.write_text(_RECIPE.format(source_extra=source_extra, sanitizers=sanitizers))
    return path


class TestRecipe:
    def test_asan_command_and_objects_resolve_against_the_recipe(self, tmp_path):
        recipe = load_sanitizer_recipe(
            _recipe(
                tmp_path,
                source_extra="    asan_command: bin/run.sh\n    asan_objects: [bin/repro, lib/k.so]",
            )
        )
        assert recipe.sanitizers == ("asan",)
        assert recipe.asan_command == (tmp_path / "bin" / "run.sh").resolve()
        assert recipe.asan_objects == (
            (tmp_path / "bin" / "repro").resolve(),
            (tmp_path / "lib" / "k.so").resolve(),
        )

    def test_asan_command_without_asan_requested_is_rejected(self, tmp_path):
        with pytest.raises(RecipeSchemaError, match="asan is not in"):
            load_sanitizer_recipe(
                _recipe(tmp_path, source_extra="    asan_command: bin/run", sanitizers="consan")
            )

    def test_asan_objects_without_a_command_is_rejected(self, tmp_path):
        with pytest.raises(RecipeSchemaError, match="names no asan_command"):
            load_sanitizer_recipe(_recipe(tmp_path, source_extra="    asan_objects: [bin/repro]"))

    @pytest.mark.parametrize("objects", ["[]", "bin/repro", "[1]", "['']"])
    def test_malformed_asan_objects_are_rejected(self, tmp_path, objects):
        with pytest.raises(RecipeSchemaError, match="non-empty list of paths"):
            load_sanitizer_recipe(
                _recipe(
                    tmp_path,
                    source_extra=f"    asan_command: bin/run\n    asan_objects: {objects}",
                )
            )

    @pytest.mark.parametrize("name", ["asan-heap-overflow.yaml", "asan-clean.yaml"])
    def test_shipped_recipes_load(self, name):
        recipe = load_sanitizer_recipe(REPO / "recipes" / "sanitizers" / name)
        assert recipe.sanitizers == ("asan",)
        assert recipe.asan_command is not None
        assert recipe.asan_command.parent.name == "bin"


class TestPipeline:
    def test_asan_without_a_command_is_not_checked(self, tmp_path):
        report = run_sanitizers(
            _worklist(), target="gfx950", sanitizers=("asan",), output_dir=tmp_path
        )
        (check,) = report.checks
        assert check.sanitizer == "asan"
        assert check.state is ExecutionState.NOT_CHECKED
        assert check.reason.startswith("asan_command_not_provisioned:")

    def test_asan_runs_through_the_pipeline(self, tmp_path, runtime):
        report = run_sanitizers(
            _worklist(),
            target="gfx950",
            sanitizers=("asan",),
            output_dir=tmp_path / "out",
            asan_command=_script(
                tmp_path, stderr_fixture="device_heap_buffer_overflow_write.txt", code=1
            ),
            asan_objects=(_instrumented(tmp_path),),
            asan_runtime=runtime,
        )
        assert report.overall_verdict is Verdict.FAIL
        assert (tmp_path / "out" / "asan" / "asan.log").is_file()


def test_support_matrix():
    assert (support("asan", "gfx950")["level"], support("asan", "gfx942")["level"]) == (
        "full",
        "full",
    )
    # The compiler builds xnack+ objects for these; no prebuilt runtime exists.
    assert support("asan", "gfx90a")["level"] == "partial"
    assert support("asan", "gfx908")["runnable"] is True
    assert support("asan", "gfx1100")["runnable"] is False


# --------------------------------------------------------------------------- #
# CIA classification
# --------------------------------------------------------------------------- #


def _report(*checks: dict) -> dict:
    return {
        "schema": "aorta.sanitizer_report/0.1",
        "target": "gfx950",
        "overall_verdict": "fail",
        "checks": list(checks),
    }


_ASAN_FINDING = {
    "sanitizer": "asan",
    "severity": "memory_error",
    "code": "heap-buffer-overflow",
    "message": "heap-buffer-overflow on amdgpu device 0; WRITE of size 4",
    "kernel_name": "store_index",
    "metadata": {"side": "device"},
}
_CONSAN_FINDING = {"sanitizer": "consan", "severity": "race", "code": "race", "message": "race"}


class TestClassification:
    def test_an_asan_finding_is_an_illegal_access_not_a_race(self):
        verdict = classify_sanitizer(
            _report({"sanitizer": "asan", "state": "ran", "findings": [_ASAN_FINDING]})
        )
        assert verdict.category == "illegal_mem"
        assert verdict.signals == [SIGNAL_MEMORY_ERROR]
        assert "store_index" in verdict.rationale

    def test_the_memory_error_outranks_a_race_in_the_same_report(self):
        verdict = classify_sanitizer(
            _report(
                {"sanitizer": "consan", "state": "ran", "findings": [_CONSAN_FINDING]},
                {"sanitizer": "asan", "state": "ran", "findings": [_ASAN_FINDING]},
            )
        )
        assert verdict.category == "illegal_mem"

    @pytest.mark.parametrize(
        "finding,phrase",
        [(_ASAN_FINDING, "reported 1 invalid memory access"), (_CONSAN_FINDING, "1 race finding")],
    )
    def test_a_finding_repeated_per_kernel_is_counted_once(self, finding, phrase):
        """Both dynamic backends repeat a check's findings in kernel_results."""

        sanitizer = finding["sanitizer"]
        verdict = classify_sanitizer(
            _report(
                {
                    "sanitizer": sanitizer,
                    "state": "ran",
                    "findings": [finding],
                    "kernel_results": [{"findings": [finding]}],
                }
            )
        )
        assert phrase in verdict.rationale

    def test_a_finding_from_a_timed_out_check_is_kept(self):
        """ASan is the first backend to return findings from a check that did not
        finish; the report already says fail, and CIA must not say nothing ran."""

        verdict = classify_sanitizer(
            _report(
                {
                    "sanitizer": "asan",
                    "state": "timed_out",
                    "verdict": "fail",
                    "reason": "asan_timeout",
                    "findings": [_ASAN_FINDING],
                }
            )
        )
        assert (verdict.category, verdict.confidence) == ("illegal_mem", 0.95)

    def test_a_lost_device_report_is_a_detection_not_a_gap(self):
        verdict = classify_sanitizer(
            _report(
                {
                    "sanitizer": "asan",
                    "state": "error",
                    "verdict": "error",
                    "reason": "asan_device_report_lost: device ASan detected ...",
                    "findings": [],
                }
            )
        )
        assert (verdict.category, verdict.signals) == ("illegal_mem", [SIGNAL_MEMORY_ERROR])
        assert "No sanitizer actually executed" not in verdict.rationale

    def test_a_consan_race_alone_is_unchanged(self):
        verdict = classify_sanitizer(
            _report({"sanitizer": "consan", "state": "ran", "findings": [_CONSAN_FINDING]})
        )
        assert (verdict.category, verdict.signals) == ("gpu_race", [SIGNAL_RACE])

    def test_an_asan_check_that_did_not_run_names_what_asan_needs(self, tmp_path):
        root = tmp_path / "bundle"
        (root / "aorta").mkdir(parents=True)
        (root / "aorta" / "sanitizer_report.json").write_text(
            json.dumps(
                _report(
                    {
                        "sanitizer": "asan",
                        "state": "not_checked",
                        "verdict": "not_checked",
                        "reason": "asan_runtime_not_found: device ASan reports need ...",
                        "findings": [],
                    }
                )
            )
        )
        artifact = SanitizerReportAdapter().collect(
            BundleContext(
                root=root,
                manifest={"paths": {"sanitizer_report": "aorta/sanitizer_report.json"}},
                job_id="asan-test",
            )
        )
        (gap,) = artifact.tooling_gaps
        assert "lib/asan" in gap["description"]
        assert "RocJITsu" not in gap["description"]
        assert gap["missing_signal"] == SIGNAL_MEMORY_ERROR

    def test_the_adapter_carries_a_lost_report_as_evidence_too(self, tmp_path):
        """The autopsy router reads the adapter, not classify_sanitizer: without this the
        detection reached it only as a tooling gap."""

        root = tmp_path / "bundle"
        (root / "aorta").mkdir(parents=True)
        (root / "aorta" / "sanitizer_report.json").write_text(
            json.dumps(
                _report(
                    {
                        "sanitizer": "asan",
                        "state": "error",
                        "verdict": "error",
                        "reason": "asan_device_report_lost: device ASan detected ...",
                        "findings": [],
                    }
                )
            )
        )
        artifact = SanitizerReportAdapter().collect(
            BundleContext(
                root=root,
                manifest={"paths": {"sanitizer_report": "aorta/sanitizer_report.json"}},
                job_id="asan-test",
            )
        )
        assert SIGNAL_MEMORY_ERROR in artifact.signals
        (evidence,) = [item for item in artifact.evidence if item["signal"] == SIGNAL_MEMORY_ERROR]
        assert "aborted before reporting it" in evidence["excerpt"]
        (gap,) = artifact.tooling_gaps
        assert gap["missing_signal"] == SIGNAL_MEMORY_ERROR


def test_the_asan_detector_id_routes_to_illegal_mem():
    """The tier-4 id carries the word the illegal_mem leg keys on, and no word an
    earlier leg wants, so the agent's offline category inference needs no change."""

    from aorta.agent.llm import _infer_category_from_detectors

    assert _infer_category_from_detectors(["tier4:asan_memory_error"]) == "illegal_mem"
    assert (
        _infer_category_from_detectors(["tier1:exit_nonzero", "tier4:asan_memory_error"])
        == "illegal_mem"
    )
