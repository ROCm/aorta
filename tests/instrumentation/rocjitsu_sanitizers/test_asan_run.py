"""``aorta run --sanitizer asan``: the run environment, the evidence and the verdict.

CPU-only. Device code is synthesised as real ELF images (an x86-64 host binary
with a ``.hip_fatbin`` section, clang offload bundles, AMDGPU code objects with
kernel-descriptor symbols), so the inventory is exercised on the layout it
reads in production rather than on stubs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import struct
import subprocess
import zlib
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from aorta.cli import run as run_cli
from aorta.instrumentation.rocjitsu_sanitizers import asan_run
from aorta.instrumentation.rocjitsu_sanitizers.asan import (
    ASAN_OPTIONS_PINNED,
    AsanRuntime,
    _kernel_symbols,
    demangle,
    kernel_inventory,
    uninterpreted_reports,
)
from aorta.instrumentation.rocjitsu_sanitizers.asan_run import (
    ENV_ASAN_RUN,
    EXIT_NOT_CHECKED,
    REPORT_NAME,
    AsanRunOutcome,
    NodeHealth,
    NodeHealthStatus,
    active_asan_run,
    asan_run_environment,
    declare_code_objects,
    evaluate_asan_run,
    loaded_files,
    owned_setting_conflicts,
    write_report,
)
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocjitsu_sanitizers.models import Verdict
from aorta.instrumentation.rocm_paths import RocmRoots, resolve_rocm_roots
from aorta.run import _trial_worker
from aorta.run import dispatcher as dispatcher_module
from aorta.run.dispatcher import RunRequest, run_trials
from aorta.workloads import Workload, WorkloadResult

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

FIXTURES = Path(__file__).parent / "fixtures" / "asan"

_EM_X86_64 = 62
_EM_AMDGPU = 224
_GFX950 = 0x4F
_XNACK_ON = 0x300
_GFX950_XNACK = "hipv4-amdgcn-amd-amdhsa--gfx950:xnack+"
_GFX950_PLAIN = "hipv4-amdgcn-amd-amdhsa--gfx950"
_STORE_INDEX = "_Z11store_indexPii"


# --------------------------------------------------------------------------- #
# Synthetic device code
# --------------------------------------------------------------------------- #


def _elf(machine: int, flags: int, sections: Sequence[tuple[str, int, bytes, int, int]]) -> bytes:
    """A little-endian ELF64 image; sections are (name, type, data, link, entsize)."""

    names = b"\0"
    name_offsets = []
    for name, *_ in sections:
        name_offsets.append(len(names))
        names += name.encode() + b"\0"
    shstrtab_name = len(names)
    names += b".shstrtab\0"
    body = b""
    offsets = []
    for _, _, data, _, _ in sections:
        offsets.append(64 + len(body))
        body += data
    names_offset = 64 + len(body)
    body += names
    table_offset = 64 + len(body)
    count = len(sections) + 2
    header = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    header += struct.pack(
        "<HHIQQQIHHHHHH", 1, machine, 1, 0, 0, table_offset, flags, 64, 0, 0, 64, count, count - 1
    )
    table = b"\0" * 64
    for (_, kind, data, link, entsize), offset, name_offset in zip(
        sections, offsets, name_offsets, strict=True
    ):
        table += struct.pack(
            "<IIQQQQIIQQ", name_offset, kind, 0, 0, offset, len(data), link, 0, 1, entsize
        )
    table += struct.pack(
        "<IIQQQQIIQQ", shstrtab_name, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0
    )
    return header + body + table


def _code_object(kernels: Sequence[str], *, asan: bool, xnack: bool = True) -> bytes:
    """An AMDGPU code object defining ``kernels`` (descriptor symbols and all)."""

    strings = b"\0"
    symbols = b"\0" * 24
    for name in (*kernels, *(f"{kernel}.kd" for kernel in kernels), "helper"):
        offset = len(strings)
        strings += name.encode() + b"\0"
        symbols += struct.pack("<IBBHQQ", offset, 0x12, 0, 1, 0, 0)
    text = b"\0call __asan_report_store4\0" if asan else b"\0plain code\0"
    return _elf(
        _EM_AMDGPU,
        _GFX950 | (_XNACK_ON if xnack else 0),
        [(".text", 1, text, 0, 0), (".symtab", 2, symbols, 3, 24), (".strtab", 3, strings, 0, 0)],
    )


def _bundle(entries: Sequence[tuple[str, bytes]]) -> bytes:
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    offset = len(magic) + 8 + sum(24 + len(triple) for triple, _ in entries)
    header = magic + struct.pack("<Q", len(entries))
    payload = b""
    for triple, blob in entries:
        header += struct.pack("<QQQ", offset, len(blob), len(triple)) + triple.encode()
        payload += blob
        offset += len(blob)
    return header + payload


def _host_binary(fatbin: bytes, *, elsewhere: bytes = b"\0") -> bytes:
    """An x86-64 binary carrying ``fatbin`` in ``.hip_fatbin`` and ``elsewhere`` in ``.text``."""

    return _elf(_EM_X86_64, 0, [(".text", 1, elsewhere, 0, 0), (".hip_fatbin", 1, fatbin, 0, 0)])


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _instrumented_binary(path: Path, kernels: Sequence[str] = (_STORE_INDEX,)) -> Path:
    return _write(
        path,
        _host_binary(_bundle([(_GFX950_XNACK, _code_object(kernels, asan=True))])),
    )


def _plain_binary(path: Path, kernels: Sequence[str] = (_STORE_INDEX,)) -> Path:
    return _write(
        path,
        _host_binary(_bundle([(_GFX950_PLAIN, _code_object(kernels, asan=False, xnack=False))])),
    )


def _compressed_binary(path: Path) -> Path:
    # A zstd CCOB header (method 1) with no payload behind it, so it cannot be opened.
    return _write(path, _host_binary(b"CCOB" + struct.pack("<HH", 2, 1) + b"\0" * 32))


def _inflatable_binary(path: Path, kernels: Sequence[str], *, asan: bool) -> Path:
    """A host binary whose device code sits in a zlib-compressed bundle."""

    triple = _GFX950_XNACK if asan else _GFX950_PLAIN
    bundle = _bundle([(triple, _code_object(kernels, asan=asan, xnack=asan))])
    payload = zlib.compress(bundle)
    header = b"CCOB" + struct.pack("<HHIIQ", 2, 0, 24 + len(payload), len(bundle), 0)
    return _write(path, _host_binary(header + payload))


def _python(tmp_path: Path) -> Path:
    """A launcher with no device code of its own."""

    return _write(tmp_path / "python3", _elf(_EM_X86_64, 0, [(".text", 1, b"\0", 0, 0)]))


# --------------------------------------------------------------------------- #
# Kernel inventory
# --------------------------------------------------------------------------- #


def test_kernel_symbols_are_the_descriptor_names():
    assert _kernel_symbols(_code_object([_STORE_INDEX, "add100"], asan=True)) == (
        _STORE_INDEX,
        "add100",
    )


def test_demangle_spells_a_kernel_as_the_dispatch_log_does():
    assert demangle(_STORE_INDEX) == "store_index(int*, int)"
    assert demangle("_Z6add100PfPKfm") == "add100(float*, float const*, unsigned long)"
    assert demangle("add100") == "add100"
    assert demangle("_Znot_a_name") == "_Znot_a_name"


def test_inventory_reads_the_fatbin_section_and_nothing_else(tmp_path):
    decoy = _bundle([(_GFX950_PLAIN, _code_object(["_Z5decoyv"], asan=False, xnack=False))])
    binary = _write(
        tmp_path / "app",
        _host_binary(
            _bundle([(_GFX950_XNACK, _code_object([_STORE_INDEX], asan=True))]), elsewhere=decoy
        ),
    )

    inventory = kernel_inventory(binary)

    assert [obj.triple for obj in inventory.code_objects] == [_GFX950_XNACK]
    (code_object,) = inventory.code_objects
    assert code_object.checked
    assert code_object.kernels == ("store_index(int*, int)",)


def test_inventory_reads_every_bundle_in_the_section(tmp_path):
    fatbin = _bundle([(_GFX950_XNACK, _code_object([_STORE_INDEX], asan=True))]) + _bundle(
        [(_GFX950_PLAIN, _code_object(["_Z5otherv"], asan=False, xnack=False))]
    )
    inventory = kernel_inventory(_write(tmp_path / "app", _host_binary(fatbin)))

    assert [(obj.checked, obj.kernels) for obj in inventory.code_objects] == [
        (True, ("store_index(int*, int)",)),
        (False, ("other()",)),
    ]


def test_inventory_of_a_standalone_bundle_and_a_bare_code_object(tmp_path):
    bundle = _write(
        tmp_path / "k.code", _bundle([(_GFX950_XNACK, _code_object(["add100"], asan=True))])
    )
    bare = _write(tmp_path / "k.hsaco", _code_object(["add100"], asan=True))

    for path in (bundle, bare):
        (code_object,) = kernel_inventory(path).code_objects
        assert code_object.checked and code_object.kernels == ("add100",)


def test_an_xnack_off_object_is_not_checked_even_when_instrumented(tmp_path):
    binary = _write(
        tmp_path / "app",
        _host_binary(
            _bundle([(_GFX950_PLAIN, _code_object([_STORE_INDEX], asan=True, xnack=False))])
        ),
    )
    (code_object,) = kernel_inventory(binary).code_objects
    assert code_object.asan_instrumented and not code_object.checked


def test_compressed_device_code_is_counted_and_names_no_kernel(tmp_path):
    inventory = kernel_inventory(_compressed_binary(tmp_path / "libtorch_hip.so"))
    assert inventory.compressed_bundles == 1
    assert inventory.code_objects == ()
    assert inventory.has_device_code


def test_files_without_device_code_and_unreadable_files(tmp_path):
    host_only = _write(tmp_path / "python", _elf(_EM_X86_64, 0, [(".text", 1, b"\0", 0, 0)]))
    script = _write(tmp_path / "run.sh", b"#!/bin/sh\n")

    assert not kernel_inventory(host_only).has_device_code
    assert not kernel_inventory(script).has_device_code
    missing = kernel_inventory(tmp_path / "gone.so")
    assert missing.unreadable and not missing.has_device_code


# --------------------------------------------------------------------------- #
# The run environment
# --------------------------------------------------------------------------- #


def _runtime(tmp_path: Path, *, with_hsa: bool = True) -> AsanRuntime:
    lib = tmp_path / "rocm-asan" / "lib" / "asan"
    lib.mkdir(parents=True)
    (lib / "libamdhip64.so").write_bytes(b"__asan_")
    if with_hsa:
        (lib / "libhsa-runtime64.so").write_bytes(b"")
    rt = tmp_path / "llvm" / "lib" / "clang" / "20" / "lib" / "linux"
    rt.mkdir(parents=True)
    host = rt / "libclang_rt.asan-x86_64.so"
    host.write_bytes(b"")
    symbolizer = tmp_path / "llvm" / "bin" / "llvm-symbolizer"
    symbolizer.parent.mkdir(parents=True)
    symbolizer.write_bytes(b"")
    return AsanRuntime(
        rocm_lib_dirs=(lib,),
        hip_runtime=lib / "libamdhip64.so",
        host_runtime=host,
        symbolizer=symbolizer,
    )


def test_reports_and_logs_go_to_per_process_files_in_the_evidence(tmp_path):
    evidence = tmp_path / "evidence"
    env = asan_run_environment({"ASAN_OPTIONS": "exitcode=0"}, _runtime(tmp_path), evidence)

    assert env["ASAN_OPTIONS"] == f"{ASAN_OPTIONS_PINNED}:log_path={evidence}/asan"
    assert env["AMD_LOG_LEVEL_FILE"] == f"{evidence}/hip_dispatch.log"
    assert (env["AMD_LOG_LEVEL"], env["AMD_LOG_MASK"]) == ("3", "128")
    assert (env["LD_DEBUG"], env["LD_DEBUG_OUTPUT"]) == ("files", f"{evidence}/ld")
    assert env["TRITON_CACHE_DIR"] == f"{evidence}/triton-cache"
    assert env[ENV_ASAN_RUN] == str(evidence)
    assert env["HSA_XNACK"] == "1"


def test_a_relative_evidence_dir_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = asan_run_environment({}, _runtime(tmp_path), Path("results/hrx/asan/run"))
    assert env[ENV_ASAN_RUN] == str(tmp_path / "results/hrx/asan/run")
    assert f"log_path={tmp_path}/results/hrx/asan/run/asan" in env["ASAN_OPTIONS"]


def test_an_evidence_dir_asan_options_cannot_carry_is_refused(tmp_path):
    with pytest.raises(ValueError, match="ASAN_OPTIONS cannot carry"):
        asan_run_environment({}, _runtime(tmp_path), tmp_path / "a:b")


def test_hip_and_hsa_are_preloaded_by_name_behind_the_host_runtime(tmp_path):
    runtime = _runtime(tmp_path)
    env = asan_run_environment({"LD_PRELOAD": "/opt/mine.so"}, runtime, tmp_path / "e")
    assert env["LD_PRELOAD"].split(":") == [
        str(runtime.host_runtime),
        "libamdhip64.so",
        "libhsa-runtime64.so",
        "/opt/mine.so",
    ]


def test_only_libraries_the_instrumented_tree_has_are_preloaded_by_name(tmp_path):
    runtime = _runtime(tmp_path, with_hsa=False)
    env = asan_run_environment({}, runtime, tmp_path / "e")
    assert env["LD_PRELOAD"].split(":") == [str(runtime.host_runtime), "libamdhip64.so"]


def test_torch_lib_follows_the_instrumented_libraries_on_the_path(tmp_path):
    runtime = _runtime(tmp_path)
    torch_lib = tmp_path / "torch" / "lib"
    env = asan_run_environment(
        {"LD_LIBRARY_PATH": "/opt/rocm/lib"}, runtime, tmp_path / "e", torch_lib=torch_lib
    )
    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str(runtime.rocm_lib_dirs[0]),
        str(runtime.host_runtime.parent),
        str(torch_lib),
        "/opt/rocm/lib",
    ]


def test_pytorch_and_triton_are_set_up_for_asan(tmp_path):
    env = asan_run_environment(
        {"PYTORCH_NO_CUDA_MEMORY_CACHING": "0"}, _runtime(tmp_path), tmp_path / "e"
    )
    assert env["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
    assert env["PYTORCH_NO_HIP_MEMORY_CACHING"] == "1"
    assert env["TRITON_ENABLE_ASAN"] == "1"
    assert env["AMDGCN_USE_BUFFER_OPS"] == "0"


def test_every_setting_the_run_mode_makes_is_owned(tmp_path):
    base = {"PATH": "/usr/bin"}
    env = asan_run_environment(base, _runtime(tmp_path), tmp_path / "e")
    added = {key for key, value in env.items() if base.get(key) != value}
    assert added == set(asan_run.OWNED_SETTINGS)


def test_a_cell_may_repeat_an_owned_setting_but_not_change_it():
    environ = {"HSA_XNACK": "1", "LD_PRELOAD": "/asan/rt.so:libamdhip64.so"}
    overlay = {
        "HSA_XNACK": "1",
        "LD_PRELOAD": "/opt/hrx/libamdhip64.so",
        "ASAN_OPTIONS": "detect_leaks=1",
        "HIP_LAUNCH_BLOCKING": "1",
    }
    assert owned_setting_conflicts(overlay, environ) == ["ASAN_OPTIONS", "LD_PRELOAD"]


@pytest.mark.parametrize(
    "name",
    ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"],
)
def test_a_cell_may_not_move_the_workload_off_the_gpus_the_canary_checked(name):
    environ = {name: "0,1"}
    assert owned_setting_conflicts({name: "0,1"}, environ) == []
    assert owned_setting_conflicts({name: "1"}, environ) == [name]
    assert owned_setting_conflicts({name: "3"}, {}) == [name]


def test_an_asan_run_needs_the_marker_and_the_runtime_loaded(tmp_path, monkeypatch):
    marker = {ENV_ASAN_RUN: str(tmp_path)}
    monkeypatch.setattr(asan_run, "_asan_runtime_loaded", lambda: True)
    assert active_asan_run(marker) == tmp_path
    assert active_asan_run({}) is None
    assert active_asan_run({ENV_ASAN_RUN: str(tmp_path / "missing")}) is None

    monkeypatch.setattr(asan_run, "_asan_runtime_loaded", lambda: False)
    assert active_asan_run(marker) is None


def test_this_process_does_not_have_the_runtime_loaded():
    assert not asan_run._asan_runtime_loaded()


def _declarations(evidence: Path) -> list[dict]:
    return [json.loads(line) for line in (evidence / "code_objects.jsonl").read_text().splitlines()]


def test_a_declaration_records_what_each_file_defines_then(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    build = tmp_path / "build"
    build.mkdir()
    _instrumented_binary(build / "probe")
    _write(build / "kernel.code", _bundle([(_GFX950_XNACK, _code_object(["add100"], asan=True))]))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))

    declare_code_objects([Path("build/probe"), build / "kernel.code"], loaded_by=build / "probe")
    declare_code_objects(["build/missing"], loaded_by=build / "probe")

    records = _declarations(evidence)
    assert [record["path"] for record in records] == [
        f"{build}/probe",
        f"{build}/kernel.code",
        f"{build}/missing",
    ]
    assert [obj["kernels"] for obj in records[0]["code_objects"]] == [["store_index(int*, int)"]]
    assert records[1]["code_objects"][0]["asan_instrumented"] is True
    assert records[2]["code_objects"] == [] and records[2]["unreadable"]


def test_a_declaration_survives_the_workload_deleting_its_build(tmp_path, monkeypatch):
    """hrx removes its temporary build directory in cleanup(), before the parent looks."""

    evidence = _evidence(tmp_path)
    build = tmp_path / "aorta-hrx-tmp" / "gfx950-asan"
    build.mkdir(parents=True)
    probe = _instrumented_binary(build / "hrx_probe")
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([probe], loaded_by=probe)
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])
    probe.unlink()

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.PASS, report.reason


def test_without_a_declaration_a_deleted_build_is_not_checked(tmp_path):
    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "hrx_probe")
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])
    probe.unlink()

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.NOT_CHECKED


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[1, 2]",
        json.dumps(
            {
                "path": "relative/probe",
                "compressed_bundles": 0,
                "unreadable": None,
                "code_objects": [],
                "loaded_by": "/bin/probe",
            }
        ),
        json.dumps({"path": "/x", "code_objects": "wrong", "loaded_by": "/bin/probe"}),
        json.dumps({"path": "/x", "compressed_bundles": 1, "unreadable": None, "code_objects": []}),
    ],
)
def test_a_declaration_the_parent_cannot_parse_is_an_error(tmp_path, line):
    """It could be the one saying a loaded file was compressed or uninstrumented."""

    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "hrx_probe")
    (evidence / "code_objects.jsonl").write_text(line + "\n")
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert "code_objects.jsonl (line 1 does not parse)" in report.reason


def test_a_declaration_manifest_aorta_cannot_read_is_an_error(tmp_path, monkeypatch):
    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "hrx_probe")
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([probe], loaded_by=probe)
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])
    _unreadable(monkeypatch, "code_objects.jsonl")

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert "code_objects.jsonl (Permission denied)" in report.reason


def test_a_linked_declaration_manifest_is_an_error(tmp_path):
    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "hrx_probe")
    (tmp_path / "elsewhere.jsonl").write_text("")
    (evidence / "code_objects.jsonl").symlink_to(tmp_path / "elsewhere.jsonl")
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert "code_objects.jsonl (a link)" in report.reason


def test_no_declaration_manifest_means_no_declarations(tmp_path):
    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "hrx_probe")
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.PASS


@pytest.mark.parametrize(
    "fixture",
    [
        "device_heap_buffer_overflow_write.txt",
        "stock_runtime_hostcall_abort.txt",
        "uninstrumented_launcher_not_preloaded.txt",
    ],
)
def test_a_report_aorta_recognises_is_not_uninterpreted(fixture):
    text = (FIXTURES / fixture).read_text()
    assert uninterpreted_reports({8: text}, "asan") == {}


def test_a_report_file_aorta_cannot_interpret_is_an_error(tmp_path):
    """A report ASan wrote that matches nothing aorta knows could still be an error."""

    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )
    (evidence / "asan.8").write_text(
        '==8==AddressSanitizer CHECK failed: asan_allocator.cpp:190 "((m->chunk_state)) == ((1))"\n'
    )

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert "asan.8" in report.reason and "CHECK failed" in report.reason


@pytest.mark.parametrize("text", ["", "\n  \n"])
def test_an_empty_report_file_is_not_a_report(tmp_path, text):
    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )
    (evidence / "asan.8").write_text(text)

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.PASS


def test_declaring_outside_an_asan_run_does_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_ASAN_RUN, raising=False)
    monkeypatch.chdir(tmp_path)
    declare_code_objects([tmp_path / "probe"], loaded_by=tmp_path / "probe")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Reading the evidence
# --------------------------------------------------------------------------- #

_LD_LOG = """\
   4242:	file=libamdhip64.so [0];  needed by /work/build/probe [0]
   4242:	file=libamdhip64.so [0];  generating link map
   4242:	file=/usr/lib64/libpython3.so [0];  dynamically loaded by /usr/bin/python3 [0]
   4242:	calling init: /lib64/libc.so.6
   4242:	calling init: /opt/rocm/lib/asan/libamdhip64.so.7
   4242:	initialize program: /work/build/probe
   4242:	transferring control: /work/build/probe
   4242:	calling init: /work/lib/libkernels.so
"""


def test_loaded_files_reads_programs_and_every_loaded_object():
    loaded = loaded_files(_LD_LOG)
    assert loaded.programs == ("/work/build/probe",)
    assert loaded.objects == (
        "/usr/lib64/libpython3.so",
        "/lib64/libc.so.6",
        "/opt/rocm/lib/asan/libamdhip64.so.7",
        "/work/lib/libkernels.so",
    )


def _evidence(tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    return evidence


def _process(
    evidence: Path, pid: int, *, program: Path | str | None, kernels: Sequence[str], objects=()
) -> None:
    lines = [f"   {pid}:\tcalling init: {path}" for path in objects]
    if program is not None:
        lines.append(f"   {pid}:\tinitialize program: {program}")
    (evidence / f"ld.{pid}").write_text("\n".join(lines) + "\n")
    shaders = ["__amd_rocclr_initHeap", "amdgcn.device.init", *kernels, "amdgcn.device.fini"]
    (evidence / f"hip_dispatch.log_{pid}").write_text(
        "".join(
            f":3:rocvirtual.cpp :3153: 0us: [pid:{pid}] ShaderName : {name}\n" for name in shaders
        )
    )


def test_a_clean_run_of_instrumented_kernels_passes(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.PASS, report.reason
    (kernel,) = report.kernels
    assert kernel.checked and kernel.defined_in == (str(tmp_path / "probe"),)


def test_runtime_kernels_are_not_counted(tmp_path):
    evidence = _evidence(tmp_path)
    _process(evidence, 7, program=_instrumented_binary(tmp_path / "probe"), kernels=[])

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.kernels == ()
    assert report.verdict is Verdict.NOT_CHECKED
    assert report.reason == "no workload kernel was dispatched"


def test_a_kernel_from_plain_device_code_is_not_checked(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence, 7, program=_plain_binary(tmp_path / "probe"), kernels=["store_index(int*, int)"]
    )

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    (kernel,) = report.kernels
    assert not kernel.checked
    assert "built without ASan" in kernel.reason and _GFX950_PLAIN in kernel.reason


def test_a_kernel_defined_in_compressed_code_is_unattributed_and_not_checked(tmp_path):
    """The PyTorch case: its kernels live in compressed bundles aorta cannot open."""

    evidence = _evidence(tmp_path)
    torch_lib = _compressed_binary(tmp_path / "libtorch_hip.so")
    signature = "void at::native::vectorized_elementwise_kernel<4, float>(int, float)"
    _process(evidence, 7, program=_python(tmp_path), kernels=[signature], objects=[torch_lib])

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    assert report.reason.startswith("no memory error observed, but 1 of 1 dispatched kernel(s)")
    (kernel,) = report.kernels
    assert "defined in no device code" in kernel.reason and "compressed" in kernel.reason
    assert report.opaque_device_code == (str(torch_lib),)


def test_a_kernel_in_a_compressed_asan_library_is_checked(tmp_path):
    """An ASan PyTorch built the default way: its device code is compressed."""

    evidence = _evidence(tmp_path)
    library = _inflatable_binary(tmp_path / "libtorch_hip.so", [_STORE_INDEX], asan=True)
    _process(
        evidence,
        7,
        program=_python(tmp_path),
        kernels=["store_index(int*, int)"],
        objects=[library],
    )

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.PASS, report.reason
    assert report.opaque_device_code == ()


def test_a_kernel_in_compressed_plain_code_is_named_as_built_without_asan(tmp_path):
    """Today's pip PyTorch: inflated, its kernels are found, and found uninstrumented."""

    evidence = _evidence(tmp_path)
    library = _inflatable_binary(tmp_path / "libtorch_hip.so", [_STORE_INDEX], asan=False)
    _process(
        evidence,
        7,
        program=_python(tmp_path),
        kernels=["store_index(int*, int)"],
        objects=[library],
    )

    (kernel,) = evaluate_asan_run(evidence, returncode=0).kernels

    assert not kernel.checked
    assert "built without ASan" in kernel.reason and kernel.defined_in == (str(library),)


def test_a_kernel_also_defined_in_plain_code_is_not_checked(tmp_path):
    """Two definitions of one signature: aorta cannot tell which one ran."""

    evidence = _evidence(tmp_path)
    plain = _plain_binary(tmp_path / "libplain.so")
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
        objects=[plain],
    )

    (kernel,) = evaluate_asan_run(evidence, returncode=0).kernels
    assert not kernel.checked
    assert kernel.defined_in == (str(tmp_path / "probe"), str(plain))


@pytest.mark.parametrize("plain_pid", [7, 8], ids=["plain-first", "plain-last"])
def test_unchecked_in_one_process_is_unchecked_for_the_run(tmp_path, plain_pid):
    evidence = _evidence(tmp_path)
    instrumented_pid = 15 - plain_pid
    _process(
        evidence,
        instrumented_pid,
        program=_instrumented_binary(tmp_path / "a"),
        kernels=["store_index(int*, int)"],
    )
    _process(
        evidence,
        plain_pid,
        program=_plain_binary(tmp_path / "b"),
        kernels=["store_index(int*, int)"],
    )

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    (kernel,) = report.kernels
    assert not kernel.checked


def test_a_relative_program_path_is_not_resolved(tmp_path, monkeypatch):
    _instrumented_binary(tmp_path / "probe")
    monkeypatch.chdir(tmp_path)
    evidence = _evidence(tmp_path)
    _process(evidence, 7, program="./probe", kernels=["store_index(int*, int)"])

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.NOT_CHECKED


@pytest.mark.parametrize("relative", ["program", "object"])
def test_a_relative_path_could_hold_the_copy_that_ran(tmp_path, monkeypatch, relative):
    """A file aorta cannot locate is not a file that defined nothing."""

    evidence = _evidence(tmp_path)
    _plain_binary(tmp_path / "plain")
    instrumented = _instrumented_binary(tmp_path / "instrumented")
    monkeypatch.chdir(tmp_path)
    if relative == "program":
        program, objects = "./plain", [instrumented]
    else:
        program, objects = instrumented, ["./plain"]
    _process(evidence, 7, program=program, kernels=["store_index(int*, int)"], objects=objects)

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    (kernel,) = report.kernels
    assert not kernel.checked and "could define the copy that ran" in kernel.reason
    assert report.opaque_device_code == ("./plain",)
    assert "1 not an absolute path" in kernel.reason


def test_an_unreadable_loaded_file_could_hold_the_copy_that_ran(tmp_path):
    """The process loaded a plain library that is gone by the time the parent looks."""

    evidence = _evidence(tmp_path)
    plain = _plain_binary(tmp_path / "libplain.so")
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
        objects=[plain],
    )
    plain.unlink()

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    (kernel,) = report.kernels
    assert not kernel.checked and "could define the copy that ran" in kernel.reason
    assert report.opaque_device_code == (str(plain),)
    assert "1 unreadable" in kernel.reason


def test_a_declaration_supplies_a_file_its_program_loaded_and_deleted(tmp_path, monkeypatch):
    """The declaration was taken while the file existed, so it stands in for the file."""

    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "probe")
    library = _instrumented_binary(tmp_path / "libkernels.so")
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([library], loaded_by=probe)
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"], objects=[library])
    library.unlink()

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.PASS, report.reason
    assert report.opaque_device_code == ()


def test_another_programs_declaration_does_not_describe_a_file_this_process_loaded(
    tmp_path, monkeypatch
):
    """One program declares an instrumented library; it is rebuilt plain; another loads it."""

    evidence = _evidence(tmp_path)
    library = _instrumented_binary(tmp_path / "libkernels.so")
    declaring = _plain_binary(tmp_path / "probe", kernels=())
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([library], loaded_by=declaring)
    _plain_binary(library)
    _process(
        evidence,
        8,
        program=_plain_binary(tmp_path / "other", kernels=()),
        kernels=["store_index(int*, int)"],
        objects=[library],
    )

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    (kernel,) = report.kernels
    assert not kernel.checked and "built without ASan" in kernel.reason


def test_a_declaration_made_through_a_link_holds_for_the_program_it_names(tmp_path, monkeypatch):
    """The workload names its program through a link; the loader records where it leads."""

    evidence = _evidence(tmp_path)
    module = _write(
        tmp_path / "add100.code", _bundle([(_GFX950_XNACK, _code_object(["add100"], asan=True))])
    )
    probe = _plain_binary(tmp_path / "probe", kernels=())
    link = tmp_path / "probe-link"
    link.symlink_to(probe)
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([module], loaded_by=link)
    _process(evidence, 7, program=probe, kernels=["add100"])

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.PASS


def test_the_declaring_program_reads_its_own_declaration_of_a_rebuilt_file(tmp_path, monkeypatch):
    """The snapshot, not the file as it is now, is what the declaring program loaded."""

    evidence = _evidence(tmp_path)
    library = _instrumented_binary(tmp_path / "libkernels.so")
    declaring = _plain_binary(tmp_path / "probe", kernels=())
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([library], loaded_by=declaring)
    _plain_binary(library)
    _process(evidence, 7, program=declaring, kernels=["store_index(int*, int)"], objects=[library])

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.PASS, report.reason


def test_a_declared_code_object_attributes_a_module_kernel(tmp_path, monkeypatch):
    """hrx's module probe: the kernel arrives through hipModuleLoad, not the loader."""

    evidence = _evidence(tmp_path)
    module = _write(
        tmp_path / "add100.code", _bundle([(_GFX950_XNACK, _code_object(["add100"], asan=True))])
    )
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([module], loaded_by=tmp_path / "probe")
    _process(evidence, 7, program=_plain_binary(tmp_path / "probe", kernels=()), kernels=["add100"])

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.PASS, report.reason


def test_without_the_declaration_a_module_kernel_is_not_checked(tmp_path):
    evidence = _evidence(tmp_path)
    _write(
        tmp_path / "add100.code", _bundle([(_GFX950_XNACK, _code_object(["add100"], asan=True))])
    )
    _process(evidence, 7, program=_plain_binary(tmp_path / "probe", kernels=()), kernels=["add100"])

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.NOT_CHECKED


def _triton_cache(evidence: Path) -> None:
    cache = evidence / "triton-cache" / "abc123"
    cache.mkdir(parents=True)
    _write(cache / "add_kernel.hsaco", _code_object(["add_kernel"], asan=True))


def test_triton_cache_kernels_are_attributed_to_a_process_that_loaded_triton(tmp_path):
    evidence = _evidence(tmp_path)
    _triton_cache(evidence)
    (tmp_path / "triton" / "_C").mkdir(parents=True)
    libtriton = _write(tmp_path / "triton" / "_C" / "libtriton.so", b"")
    _process(
        evidence, 7, program=_python(tmp_path), kernels=["add_kernel"], objects=[libtriton]
    )

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.PASS


def test_the_triton_cache_is_no_evidence_for_a_process_that_never_loaded_triton(tmp_path):
    evidence = _evidence(tmp_path)
    _triton_cache(evidence)
    _process(evidence, 7, program=_python(tmp_path), kernels=["add_kernel"])

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.NOT_CHECKED


def test_a_declaration_is_evidence_only_for_the_program_that_made_it(tmp_path, monkeypatch):
    """One process declares an instrumented add100; another runs an add100 nobody declared."""

    evidence = _evidence(tmp_path)
    module = _write(
        tmp_path / "add100.code", _bundle([(_GFX950_XNACK, _code_object(["add100"], asan=True))])
    )
    declaring = _plain_binary(tmp_path / "probe", kernels=())
    other = _plain_binary(tmp_path / "other", kernels=())
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([module], loaded_by=declaring)
    _process(evidence, 7, program=declaring, kernels=["add100"])
    _process(evidence, 8, program=other, kernels=["add100"])

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    (kernel,) = report.kernels
    assert not kernel.checked and "defined in no device code" in kernel.reason


def test_unopenable_device_code_can_hold_the_copy_that_ran(tmp_path):
    """A readable instrumented definition proves nothing beside code aorta cannot open."""

    evidence = _evidence(tmp_path)
    opaque = _compressed_binary(tmp_path / "libopaque.so")
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
        objects=[opaque],
    )

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.NOT_CHECKED
    (kernel,) = report.kernels
    assert not kernel.checked and "could define the copy that ran" in kernel.reason


@pytest.mark.parametrize("returncode", [1, -9, None])
def test_a_run_that_did_not_finish_cleanly_is_an_error(tmp_path, returncode):
    """Every kernel checked, no report -- and the run failed, so it vouches for nothing."""

    evidence = _evidence(tmp_path)
    _process(
        evidence, 7, program=_instrumented_binary(tmp_path / "probe"), kernels=["store_index(int*, int)"]
    )

    report = evaluate_asan_run(evidence, returncode=returncode)

    assert report.verdict is Verdict.ERROR
    assert "without an ASan report, so ASan cannot vouch for it" in report.reason


def test_a_memory_error_fails_the_run(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence, 219, program=_plain_binary(tmp_path / "probe"), kernels=["store_index(int*, int)"]
    )
    (evidence / "asan.219").write_text(
        (FIXTURES / "device_heap_buffer_overflow_read.txt").read_text()
    )

    report = evaluate_asan_run(evidence, returncode=1)

    assert report.verdict is Verdict.FAIL
    (finding,) = report.findings
    assert finding.code == "heap-buffer-overflow"
    assert (
        report.reason
        == "1 memory error(s); first: heap-buffer-overflow at /work/src/asan_probe.hip:61:28"
    )


def _unreadable(monkeypatch, *names: str) -> None:
    read_text = Path.read_text

    def refuse(self, *args, **kwargs):
        if self.name in names:
            raise PermissionError(13, "Permission denied", str(self))
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)


def test_an_asan_report_aorta_cannot_read_is_an_error(tmp_path, monkeypatch):
    """The workload survived its child's failure, so only the report could show it."""

    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )
    (evidence / "asan.8").write_text((FIXTURES / "device_heap_buffer_overflow_read.txt").read_text())
    _unreadable(monkeypatch, "asan.8")

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert "could not read the run's evidence" in report.reason and "asan.8" in report.reason


def test_a_hip_log_aorta_cannot_read_is_an_error(tmp_path, monkeypatch):
    """Its process's kernels would otherwise be missing from the count, not unchecked."""

    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "probe")
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])
    _process(evidence, 8, program=_plain_binary(tmp_path / "plain"), kernels=["other(int*)"])
    _unreadable(monkeypatch, "hip_dispatch.log_8")

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert "hip_dispatch.log_8" in report.reason


def test_a_loader_log_aorta_cannot_read_is_an_error(tmp_path, monkeypatch):
    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )
    _unreadable(monkeypatch, "ld.7")

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert "ld.7 (Permission denied)" in report.reason


def test_an_evidence_directory_aorta_cannot_list_is_an_error(tmp_path, monkeypatch):
    evidence = _evidence(tmp_path)

    def refuse(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "iterdir", refuse)

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.verdict is Verdict.ERROR
    assert f"{evidence} (Permission denied)" in report.reason


def test_a_memory_error_still_fails_beside_an_unreadable_report(tmp_path, monkeypatch):
    evidence = _evidence(tmp_path)
    _process(
        evidence, 7, program=_plain_binary(tmp_path / "probe"), kernels=["store_index(int*, int)"]
    )
    (evidence / "asan.7").write_text((FIXTURES / "device_heap_buffer_overflow_read.txt").read_text())
    (evidence / "asan.8").write_text("")
    _unreadable(monkeypatch, "asan.8")

    report = evaluate_asan_run(evidence, returncode=1)

    assert report.verdict is Verdict.FAIL
    assert [finding.code for finding in report.findings] == ["heap-buffer-overflow"]


def test_a_lost_device_report_is_an_error(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )
    with (evidence / "hip_dispatch.log_7").open("a") as log:
        log.write((FIXTURES / "stock_runtime_hostcall_abort.txt").read_text())

    report = evaluate_asan_run(evidence, returncode=134)

    assert report.verdict is Verdict.ERROR
    assert "stock HIP runtime" in report.reason


def test_a_run_that_never_started_the_hip_runtime_is_not_checked(tmp_path):
    report = evaluate_asan_run(_evidence(tmp_path), returncode=0)
    assert report.verdict is Verdict.NOT_CHECKED
    assert "no process wrote a HIP runtime log" in report.reason


def test_evidence_is_created_below_the_results_dir(tmp_path):
    evidence = asan_run.new_evidence_dir(tmp_path / "results", ("hrx", "asan"))
    assert evidence.parent == tmp_path / "results" / "hrx" / "asan"
    assert evidence.is_dir() and not evidence.is_symlink()


@pytest.mark.parametrize("linked", [("hrx",), ("hrx", "asan")])
def test_a_planted_link_below_the_results_dir_is_refused(tmp_path, linked):
    results = tmp_path / "results"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = results.joinpath(*linked)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)

    with pytest.raises(ValueError, match="refusing to create ASan evidence"):
        asan_run.new_evidence_dir(results, ("hrx", "asan"))
    assert list(outside.iterdir()) == []


def test_a_symlinked_evidence_file_is_not_read(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )
    elsewhere = _write(
        tmp_path / "planted", (FIXTURES / "device_heap_buffer_overflow_read.txt").read_bytes()
    )
    (evidence / "asan.7").symlink_to(elsewhere)

    report = evaluate_asan_run(evidence, returncode=0)

    assert report.findings == ()
    assert report.verdict is Verdict.ERROR
    assert "asan.7 (a link)" in report.reason


def test_the_report_is_written_once_and_round_trips(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "probe"),
        kernels=["store_index(int*, int)"],
    )
    report = evaluate_asan_run(evidence, returncode=0, command=["python", "-m", "aorta.cli", "run"])

    path = write_report(report, evidence)
    data = json.loads(path.read_text())

    assert path.name == REPORT_NAME
    assert data["verdict"] == "pass"
    assert data["kernels"] == {
        "dispatched": 1,
        "checked": 1,
        "detail": [kernel.to_dict() for kernel in report.kernels],
    }
    assert data["command"] == ["python", "-m", "aorta.cli", "run"]
    with pytest.raises(FileExistsError):
        write_report(report, evidence)


def test_the_summary_names_unchecked_kernels(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence, 7, program=_plain_binary(tmp_path / "probe"), kernels=["store_index(int*, int)"]
    )

    summary = evaluate_asan_run(evidence, returncode=0).summary()

    assert summary[0].startswith("ASan: NOT CHECKED -- no memory error observed")
    assert summary[1].startswith(
        "  unchecked: store_index(int*, int) -- defined in device code built"
    )
    assert summary[-1] == f"Evidence: {evidence}"


# --------------------------------------------------------------------------- #
# Node health
# --------------------------------------------------------------------------- #

# The start of what the canary reported on a node whose GPU page migration corrupts
# memory: a host read of a page the GPU had touched. Paths shortened.
_SHADOW_CORRUPTION_REPORT = """\
=================================================================
==3299796==ERROR: AddressSanitizer: unknown-crash on address 0x7b8be5000000 at pc 0x0000002254ae bp 0x7ffdff2a61b0 sp 0x7ffdff2a61a8
READ of size 8 at 0x7b8be5000000 thread T0
    #0 0x0000002254ad in main /src/asan_canary.hip:124:50
    #1 0x7fcc1262a60f in __libc_start_call_main (/lib64/libc.so.6+0x2a60f)

Address 0x7b8be5000000 is a wild pointer inside of access range of size 0x000000000008.
SUMMARY: AddressSanitizer: unknown-crash /src/asan_canary.hip:124:50 in main
==3299796==ABORTING
"""
# The same, tripped inside the HSA runtime, whose frames carry no symbols.
_SHADOW_CORRUPTION_IN_THE_RUNTIME = """\
=================================================================
==1068==ERROR: AddressSanitizer: unknown-crash on address 0x7b06eddfcb88 at pc 0x7f06fc507bae bp 0x7fffd98261f0 sp 0x7fffd98261e8
READ of size 8 at 0x7b06eddfcb88 thread T0
    #0 0x7f06fc507bad  (/opt/rocm/lib/asan/libhsa-runtime64.so+0x19ebad) (BuildId: 41079eb4d84220e2ac5fcf56df7c118ae5330f64)
    #1 0x7f06fd3d9361  (/opt/rocm/lib/asan/libamdhip64.so+0xa48361) (BuildId: 6629337bed05411cf93d51729ebc8ae6606495b7)

==1068==ABORTING
"""
_CANARY_OK = (
    "AORTA_ASAN_CANARY gpu=0 check=untouched_gpu_read bad_words=0 of=8388608 ok\n"
    "AORTA_ASAN_CANARY verdict=PASS gpus=8\n"
)
_CANARY_KERNEL = "store_index(int*, int)"


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def _topology(root: Path, *versions: int) -> Path:
    """A KFD topology with one node per ``gfx_target_version``; 0 is a CPU."""

    for index, version in enumerate(versions):
        node = root / str(index)
        node.mkdir(parents=True)
        (node / "properties").write_text(
            f"cpu_cores_count 0\ngfx_target_version {version}\nmax_waves_per_simd 16\n"
        )
    return root


def test_gpu_targets_come_from_the_driver_topology(tmp_path):
    topology = _topology(tmp_path / "nodes", 0, 90500, 90500, 90402)
    assert asan_run.gpu_targets(topology) == ("gfx950", "gfx942")


@pytest.mark.parametrize(
    ("version", "target"),
    [
        (90500, "gfx950"),
        (90402, "gfx942"),
        (90010, "gfx90a"),
        (90008, "gfx908"),
        (110001, "gfx1101"),
    ],
)
def test_a_target_version_names_its_target(version, target):
    assert asan_run._gfx_target(version) == target


def test_no_topology_and_cpu_nodes_list_no_gpu(tmp_path):
    assert asan_run.gpu_targets(tmp_path / "missing") == ()
    assert asan_run.gpu_targets(_topology(tmp_path / "cpus", 0, 0)) == ()


def test_hipcc_is_looked_for_as_hrx_looks_for_it(tmp_path, monkeypatch):
    rocm = tmp_path / "rocm"
    monkeypatch.setattr(
        asan_run,
        "resolve_rocm_roots",
        lambda environ: RocmRoots(rocm, rocm, rocm, layout="classic", source="ROCM_PATH"),
    )
    path = {"PATH": str(tmp_path / "path")}
    assert asan_run._hipcc(path) is None
    on_path = _executable(tmp_path / "path" / "hipcc")
    assert asan_run._hipcc(path) == str(on_path)
    in_rocm = _executable(rocm / "bin" / "hipcc")
    assert asan_run._hipcc(path) == str(in_rocm)
    named = _executable(tmp_path / "mine" / "hipcc")
    assert asan_run._hipcc({**path, "HIPCC": str(named)}) == str(named)


def _canary_ok(directory: Path, binary: Path, argv: tuple[str, ...]) -> ProcessResult:
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    return ProcessResult(argv, 0, _CANARY_OK, "")


def _canary_shadow_corrupted(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    (directory / "asan.41").write_text(_SHADOW_CORRUPTION_REPORT)
    return ProcessResult(argv, 1, "", "")


def _canary_shadow_corrupted_in_the_runtime(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    (directory / "asan.41").write_text(_SHADOW_CORRUPTION_IN_THE_RUNTIME)
    return ProcessResult(argv, 1, "", "")


def _canary_data_corrupted(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    return ProcessResult(
        argv,
        3,
        "AORTA_ASAN_CANARY gpu=0 check=cpu_pattern_gpu bad_words=8388608 of=8388608 FAIL\n"
        "AORTA_ASAN_CANARY gpu=0 check=gpu_pattern_cpu bad_words=12 of=8388608 FAIL\n"
        "AORTA_ASAN_CANARY verdict=FAIL gpus=8\n",
        "",
    )


def _canary_crashed(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    return ProcessResult(argv, -11, "", "")


def _canary_hung(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    return ProcessResult(argv, None, "", "", timed_out=True)


def _canary_unlaunchable(directory: Path, binary: Path, argv: tuple[str, ...]):
    return ProcessResult(argv, None, "", "", launch_error="PermissionError: [Errno 13]")


def _canary_without_a_gpu(directory: Path, binary: Path, argv: tuple[str, ...]):
    return ProcessResult(
        argv,
        2,
        "AORTA_ASAN_CANARY hip_error hipGetDeviceCount(&gpus): no ROCm-capable device is detected\n",
        "",
    )


def _canary_without_pageable_access(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    return ProcessResult(
        argv,
        3,
        "AORTA_ASAN_CANARY gpu=3 check=pageable_memory_access FAIL\n"
        "AORTA_ASAN_CANARY verdict=FAIL gpus=8\n",
        "",
    )


def _canary_silent(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    return ProcessResult(argv, 0, "", "")


def _canary_says_it_failed(directory: Path, binary: Path, argv: tuple[str, ...]):
    _process(directory, 41, program=binary, kernels=[_CANARY_KERNEL])
    return ProcessResult(argv, 0, "AORTA_ASAN_CANARY verdict=FAIL gpus=8\n", "")


class _Node:
    """``run_argv`` on a node with a gfx950 GPU: hipcc builds a synthetic canary, which runs as told."""

    def __init__(self, hipcc: Path) -> None:
        self.hipcc = hipcc
        self.canary = _canary_ok
        self.build_returncode = 0
        self.instrumented = True
        self.builds: list[tuple[tuple[str, ...], dict[str, str]]] = []
        self.runs: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(self, argv, *, timeout_seconds, env):
        argv = tuple(argv)
        if argv[0] == str(self.hipcc):
            self.builds.append((argv, dict(env)))
            if self.build_returncode:
                error = "clang: error: invalid target ID 'gfx1100:xnack+'\n"
                return ProcessResult(argv, self.build_returncode, "", error)
            output = Path(argv[argv.index("-o") + 1])
            (_instrumented_binary if self.instrumented else _plain_binary)(output)
            return ProcessResult(argv, 0, "", "")
        self.runs.append((argv, dict(env)))
        return self.canary(Path(env[ENV_ASAN_RUN]), Path(argv[0]), argv)


@pytest.fixture
def node(tmp_path, monkeypatch):
    hipcc = _executable(tmp_path / "rocm" / "bin" / "hipcc")
    fake = _Node(hipcc)
    monkeypatch.setattr(asan_run, "gpu_targets", lambda: ("gfx950",))
    monkeypatch.setattr(asan_run, "_hipcc", lambda environ: str(hipcc))
    monkeypatch.setattr(asan_run, "run_argv", fake)
    return fake


def _check(tmp_path: Path, **kwargs) -> NodeHealth:
    return asan_run.check_node_health(
        _evidence(tmp_path), _runtime(tmp_path), environ={"PATH": "/usr/bin"}, **kwargs
    )


def test_a_clean_canary_vouches_for_the_node(tmp_path, node):
    health = _check(tmp_path)

    assert health.status is NodeHealthStatus.HEALTHY, health.reason
    assert health.gpus == 8
    assert health.reason == (
        "an in-bounds canary ran clean under ASan on 8 GPU(s), all 1 of its kernels checked "
        "and its data intact through GPU page migration"
    )
    directory = tmp_path / "evidence" / "node-health"
    assert health.evidence_dir == str(directory)
    assert health.canary is not None and health.canary.verdict is Verdict.PASS
    assert json.loads((directory / REPORT_NAME).read_text())["verdict"] == "pass"
    assert (directory / "canary.log").read_text() == _CANARY_OK


def test_the_canary_is_built_with_device_asan_for_every_gpu_target(tmp_path, node, monkeypatch):
    monkeypatch.setattr(asan_run, "gpu_targets", lambda: ("gfx950", "gfx942"))
    _check(tmp_path)

    ((argv, env),) = node.builds
    directory = tmp_path / "evidence" / "node-health"
    assert argv == (
        str(node.hipcc),
        "-O1",
        "-g",
        "-fsanitize=address",
        "-shared-libsan",
        "--offload-arch=gfx950:xnack+",
        "--offload-arch=gfx942:xnack+",
        str(asan_run.CANARY_SOURCE),
        "-o",
        str(directory / "asan_canary"),
    )
    llvm_bin = str(resolve_rocm_roots({"PATH": "/usr/bin"}).llvm_bin_dir)
    assert env["PATH"].split(os.pathsep) == [llvm_bin, "/usr/bin"]
    assert (directory / "build.log").read_text().startswith(f"$ {node.hipcc} -O1 -g")


def test_the_canary_runs_in_the_environment_the_workload_gets(tmp_path, node):
    runtime = _runtime(tmp_path)
    evidence = _evidence(tmp_path)
    base = {"PATH": "/usr/bin", "HIP_VISIBLE_DEVICES": "0,1", "LD_PRELOAD": "/opt/mine.so"}
    torch_lib = tmp_path / "torch" / "lib"

    asan_run.check_node_health(evidence, runtime, environ=base, torch_lib=torch_lib)

    ((argv, env),) = node.runs
    directory = evidence / "node-health"
    workload = asan_run_environment(base, runtime, evidence, torch_lib=torch_lib)
    assert argv == (str(directory / "asan_canary"),)
    assert env == {
        key: value.replace(str(evidence), str(directory)) for key, value in workload.items()
    }


@pytest.mark.parametrize(
    ("canary", "reason"),
    [
        (
            _canary_shadow_corrupted,
            "the canary, whose every access is in bounds, reported unknown-crash (host) at "
            "/src/asan_canary.hip:124:50; GPU page migration on this node is likely corrupting "
            "ASan's shadow memory",
        ),
        (
            _canary_shadow_corrupted_in_the_runtime,
            "the canary, whose every access is in bounds, reported unknown-crash (host); GPU "
            "page migration on this node is likely corrupting ASan's shadow memory",
        ),
        (
            _canary_data_corrupted,
            "data the canary sent through GPU page migration came back wrong (GPU 0 "
            "cpu_pattern_gpu: 8388608 of 8388608 words wrong; GPU 0 gpu_pattern_cpu: 12 of "
            "8388608 words wrong), and ASan's shadow memory takes the same path",
        ),
        (
            _canary_crashed,
            "the canary's own ASan run was error: the run exited -11 without an ASan report, "
            "so ASan cannot vouch for it",
        ),
        (_canary_hung, "the canary did not finish in 120 s"),
        (_canary_unlaunchable, "the canary could not start: PermissionError: [Errno 13]"),
        (
            _canary_without_a_gpu,
            "the canary hit a HIP error: hipGetDeviceCount(&gpus): no ROCm-capable device is "
            "detected",
        ),
        (
            _canary_without_pageable_access,
            "a GPU cannot reach pageable host memory, which device ASan needs to read its "
            "shadow memory (GPU 3 pageable_memory_access)",
        ),
        (
            _canary_silent,
            "the canary's ASan run passed, but the canary never reported its own checks passing",
        ),
        (
            _canary_says_it_failed,
            "the canary's ASan run passed, but the canary never reported its own checks passing",
        ),
    ],
    ids=[
        "memory-error",
        "memory-error-in-the-runtime",
        "corrupted-data",
        "crash",
        "hang",
        "unlaunchable",
        "hip-error",
        "no-pageable-access",
        "silent",
        "says-it-failed",
    ],
)
def test_a_canary_that_is_not_clean_fails_the_node(tmp_path, node, canary, reason):
    node.canary = canary

    health = _check(tmp_path)

    assert health.status is NodeHealthStatus.FAILED
    assert health.reason == reason
    assert health.canary is not None
    assert (Path(health.evidence_dir) / REPORT_NAME).is_file()


def test_a_canary_kernel_asan_does_not_check_fails_the_node(tmp_path, node):
    """The canary checking 0 kernels: here a build that dropped device ASan."""

    node.instrumented = False

    health = _check(tmp_path)

    assert health.status is NodeHealthStatus.FAILED
    assert health.reason.startswith("the canary's own ASan run was not_checked: no memory error")
    assert health.canary is not None and health.canary.unchecked


def test_more_than_three_failed_checks_are_counted(tmp_path, node):
    lines = "".join(
        f"AORTA_ASAN_CANARY gpu={gpu} check=cpu_pattern_gpu bad_words=9 of=9 FAIL\n"
        for gpu in range(5)
    )
    node.canary = lambda directory, binary, argv: ProcessResult(argv, 3, lines, "")

    reason = _check(tmp_path).reason

    assert "GPU 2 cpu_pattern_gpu: 9 of 9 words wrong; and 2 more)" in reason
    assert "GPU 3" not in reason


def test_without_a_gpu_the_node_is_not_verified(tmp_path, node, monkeypatch):
    monkeypatch.setattr(asan_run, "gpu_targets", lambda: ())

    health = _check(tmp_path)

    assert health.status is NodeHealthStatus.NOT_VERIFIED
    assert health.reason.startswith("the kernel driver lists no GPU")
    assert node.builds == node.runs == []
    assert not (tmp_path / "evidence" / "node-health").exists()


def test_without_hipcc_the_node_is_not_verified(tmp_path, node, monkeypatch):
    monkeypatch.setattr(asan_run, "_hipcc", lambda environ: None)

    health = _check(tmp_path)

    assert health.status is NodeHealthStatus.NOT_VERIFIED
    assert health.reason.startswith("hipcc was not found")
    assert node.builds == node.runs == []


def test_without_the_canary_source_the_node_is_not_verified(tmp_path, node, monkeypatch):
    monkeypatch.setattr(asan_run, "CANARY_SOURCE", tmp_path / "gone.hip")

    health = _check(tmp_path)

    assert health.status is NodeHealthStatus.NOT_VERIFIED
    assert (
        health.reason
        == f"the canary source {tmp_path / 'gone.hip'} is missing from this installation"
    )
    assert node.builds == node.runs == []


def test_a_canary_that_does_not_build_leaves_the_node_not_verified(tmp_path, node):
    node.build_returncode = 1

    health = _check(tmp_path)

    assert health.status is NodeHealthStatus.NOT_VERIFIED
    assert health.reason == (
        "the canary did not build: hipcc exited 1: clang: error: invalid target ID 'gfx1100:xnack+'"
    )
    assert node.runs == []
    assert "invalid target ID" in (tmp_path / "evidence" / "node-health" / "build.log").read_text()


def _clean_workload_run(evidence: Path, tmp_path: Path) -> None:
    _process(
        evidence, 7, program=_instrumented_binary(tmp_path / "probe"), kernels=[_CANARY_KERNEL]
    )


def _memory_error_run(evidence: Path, tmp_path: Path) -> None:
    _process(evidence, 7, program=_plain_binary(tmp_path / "probe"), kernels=[_CANARY_KERNEL])
    (evidence / "asan.7").write_text(
        (FIXTURES / "device_heap_buffer_overflow_read.txt").read_text()
    )


@pytest.mark.parametrize(
    "run", [_clean_workload_run, _memory_error_run], ids=["clean", "memory-error"]
)
def test_a_node_that_failed_its_health_check_makes_any_run_an_error(tmp_path, run):
    evidence = _evidence(tmp_path)
    run(evidence, tmp_path)
    health = NodeHealth(NodeHealthStatus.FAILED, "the canary reported unknown-crash")

    report = evaluate_asan_run(evidence, returncode=0, node_health=health)

    assert report.verdict is Verdict.ERROR
    assert report.reason == (
        "node health check failed, so ASan results on this node cannot be trusted: "
        "the canary reported unknown-crash"
    )
    assert report.node_health is health


def test_an_unverified_node_cannot_pass(tmp_path):
    evidence = _evidence(tmp_path)
    _clean_workload_run(evidence, tmp_path)
    health = NodeHealth(NodeHealthStatus.NOT_VERIFIED, "hipcc was not found")

    report = evaluate_asan_run(evidence, returncode=0, node_health=health)

    assert report.verdict is Verdict.NOT_CHECKED
    assert report.reason == (
        "no memory error in 1 dispatched kernel(s), all from ASan-instrumented device code, "
        "but nothing verified that ASan works on this node: hipcc was not found"
    )


@pytest.mark.parametrize(
    ("run", "verdict"),
    [(_memory_error_run, Verdict.FAIL), (None, Verdict.NOT_CHECKED)],
    ids=["memory-error", "not-checked"],
)
def test_an_unverified_node_changes_no_other_verdict(tmp_path, run, verdict):
    evidence = _evidence(tmp_path)
    if run is not None:
        run(evidence, tmp_path)
    health = NodeHealth(NodeHealthStatus.NOT_VERIFIED, "hipcc was not found")

    unverified = evaluate_asan_run(evidence, returncode=1 if run else 0, node_health=health)
    plain = evaluate_asan_run(evidence, returncode=1 if run else 0)

    assert (unverified.verdict, unverified.reason) == (verdict, plain.reason)


@pytest.mark.parametrize(
    "run", [_clean_workload_run, _memory_error_run], ids=["clean", "memory-error"]
)
def test_a_healthy_node_changes_no_verdict(tmp_path, run):
    evidence = _evidence(tmp_path)
    run(evidence, tmp_path)
    health = NodeHealth(NodeHealthStatus.HEALTHY, "an in-bounds canary ran clean")

    healthy = evaluate_asan_run(evidence, returncode=0, node_health=health)
    plain = evaluate_asan_run(evidence, returncode=0)

    assert (healthy.verdict, healthy.reason) == (plain.verdict, plain.reason)


class _Workload:
    """``subprocess.run`` for the workload: leaves ``evidence`` in the run's evidence directory."""

    def __init__(self, tmp_path: Path, evidence) -> None:
        self.tmp_path = tmp_path
        self.evidence = evidence
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def run(self, command, *, env, check):
        self.calls.append((list(command), dict(env)))
        self.evidence(Path(env[ENV_ASAN_RUN]), self.tmp_path)
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def workload(tmp_path, monkeypatch):
    fake = _Workload(tmp_path, _clean_workload_run)
    monkeypatch.setattr(asan_run, "subprocess", MagicMock(run=fake.run))
    monkeypatch.setattr(asan_run, "torch_library_dir", lambda: None)
    return fake


def _run_under_asan(tmp_path: Path) -> AsanRunOutcome:
    return asan_run.run_under_asan(
        ["aorta", "run"],
        evidence_dir=_evidence(tmp_path),
        runtime=_runtime(tmp_path),
        environ={"PATH": "/usr/bin"},
    )


def test_a_node_that_fails_its_health_check_is_not_given_the_workload(tmp_path, node, workload):
    node.canary = _canary_shadow_corrupted

    outcome = _run_under_asan(tmp_path)

    assert workload.calls == []
    assert outcome.returncode is None
    report = outcome.report
    assert report.verdict is Verdict.ERROR
    assert report.reason.startswith(
        "node health check failed, so ASan results on this node cannot be trusted: the canary, "
        "whose every access is in bounds, reported unknown-crash (host)"
    )
    assert report.findings == () and report.kernels == ()
    data = json.loads(outcome.report_path.read_text())
    assert data["node_health"]["status"] == "failed"
    assert [finding["code"] for finding in data["node_health"]["canary"]["findings"]] == [
        "unknown-crash"
    ]


def test_a_healthy_node_runs_the_workload_and_judges_it_alone(tmp_path, node, workload):
    """The canary's kernel, its evidence and its report stay out of the workload's."""

    workload.evidence = lambda evidence, root: _process(
        evidence, 7, program=_instrumented_binary(root / "probe", ["add100"]), kernels=["add100"]
    )

    outcome = _run_under_asan(tmp_path)

    ((command, env),) = workload.calls
    assert command == ["aorta", "run"]
    assert env[ENV_ASAN_RUN] == str(tmp_path / "evidence")
    report = outcome.report
    assert outcome.returncode == 0
    assert report.verdict is Verdict.PASS, report.reason
    assert [kernel.signature for kernel in report.kernels] == ["add100"]
    assert report.node_health is not None
    assert report.node_health.status is NodeHealthStatus.HEALTHY
    data = json.loads(outcome.report_path.read_text())
    assert data["node_health"]["canary"]["kernels"]["detail"][0]["signature"] == _CANARY_KERNEL


def test_the_canary_and_the_workload_get_the_same_environment(
    tmp_path, node, workload, monkeypatch
):
    torch_lib = tmp_path / "torch" / "lib"
    monkeypatch.setattr(asan_run, "torch_library_dir", lambda: torch_lib)
    evidence = _evidence(tmp_path)

    asan_run.run_under_asan(
        ["aorta", "run"],
        evidence_dir=evidence,
        runtime=_runtime(tmp_path),
        environ={"PATH": "/usr/bin", "HIP_VISIBLE_DEVICES": "3"},
    )

    ((_, canary),) = node.runs
    ((_, ran),) = workload.calls
    directory = str(evidence / "node-health")
    assert canary == {key: value.replace(str(evidence), directory) for key, value in ran.items()}
    assert canary["HIP_VISIBLE_DEVICES"] == "3"
    assert str(torch_lib) in canary["LD_LIBRARY_PATH"].split(os.pathsep)


def test_the_canary_prints_the_lines_its_judge_reads():
    source = asan_run.CANARY_SOURCE.read_text(encoding="utf-8")
    formats = re.findall(r'printf\("(AORTA_ASAN_CANARY [^"]*)\\n"', source)

    def printed(prefix: str, *args: object) -> str:
        (fmt,) = [fmt for fmt in formats if fmt.startswith(prefix)]
        return re.sub(r"%(?:ll|z)([du])", r"%\1", fmt) % args

    failed_word = re.search(r'bad \? "(\w+)" : "ok"', source)
    passed_word = re.search(r'failures \? "\w+" : "(\w+)"', source)
    assert failed_word is not None and passed_word is not None
    check = asan_run._CANARY_FAILED_CHECK.fullmatch(
        printed("AORTA_ASAN_CANARY gpu=%d check=%s ", 1, "cpu_pattern_gpu", 7, 9, failed_word[1])
    )
    assert check is not None
    assert (check["gpu"], check["check"], check["bad"], check["of"]) == (
        "1",
        "cpu_pattern_gpu",
        "7",
        "9",
    )
    pageable = asan_run._CANARY_FAILED_CHECK.fullmatch(
        printed("AORTA_ASAN_CANARY gpu=%d check=pageable", 3)
    )
    assert pageable is not None and pageable["bad"] is None
    passed = asan_run._CANARY_PASSED.fullmatch(
        printed("AORTA_ASAN_CANARY verdict=", passed_word[1], 8)
    )
    assert passed is not None and passed["gpus"] == "8"
    error = asan_run._CANARY_HIP_ERROR.fullmatch(
        printed("AORTA_ASAN_CANARY hip_error %s", "hipMalloc(&bad)", "out of memory")
    )
    assert error is not None and error["what"] == "hipMalloc(&bad): out of memory"


def test_an_unverified_node_runs_the_workload_but_it_cannot_pass(
    tmp_path, node, workload, monkeypatch
):
    monkeypatch.setattr(asan_run, "gpu_targets", lambda: ())

    outcome = _run_under_asan(tmp_path)

    assert len(workload.calls) == 1
    assert outcome.report.verdict is Verdict.NOT_CHECKED
    assert outcome.report.reason.endswith(
        "but nothing verified that ASan works on this node: the kernel driver lists no GPU "
        f"({asan_run.KFD_TOPOLOGY}), so no canary could run"
    )


def test_the_summary_reports_node_health(tmp_path, node, workload):
    healthy = _run_under_asan(tmp_path).report.summary()
    assert healthy[0].startswith("ASan: PASS -- ")
    assert healthy[1].startswith("Node health: verified in ")
    assert healthy[1].endswith(
        " s -- an in-bounds canary ran clean under ASan on 8 GPU(s), "
        "all 1 of its kernels checked and its data intact through GPU page migration"
    )


def test_the_summary_names_what_a_failed_canary_reported(tmp_path, node, workload):
    node.canary = _canary_shadow_corrupted

    summary = _run_under_asan(tmp_path).report.summary()

    assert summary[0].startswith("ASan: ERROR -- node health check failed, so ASan results")
    assert summary[1] == (
        f"Node health: FAILED -- the canary's evidence is in {tmp_path / 'evidence' / 'node-health'}"
    )
    assert summary[2] == "  canary: unknown-crash (host) at /src/asan_canary.hip:124:50"


def test_the_summary_says_when_the_node_was_not_verified(tmp_path):
    health = NodeHealth(NodeHealthStatus.NOT_VERIFIED, "hipcc was not found")
    report = evaluate_asan_run(_evidence(tmp_path), returncode=0, node_health=health)
    assert report.summary()[1] == "Node health: NOT VERIFIED -- hipcc was not found"


def test_the_canary_source_ships_in_the_wheel():
    root = Path(__file__).resolve().parents[3]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    entry = "instrumentation/rocjitsu_sanitizers/asan_canary.hip"
    assert entry in data["tool"]["setuptools"]["package-data"]["aorta"]
    assert root / "src" / "aorta" / entry == asan_run.CANARY_SOURCE
    assert asan_run.CANARY_SOURCE.is_file()


# --------------------------------------------------------------------------- #
# The dispatcher
# --------------------------------------------------------------------------- #


class _ConfigRecordingWorkload(Workload):
    seen: dict = {}

    def setup(self) -> None:
        _ConfigRecordingWorkload.seen = dict(self.config)

    def run(self) -> WorkloadResult:
        return WorkloadResult(passed=True, total_iterations=1)

    def cleanup(self) -> None:
        pass


def _registered(workload: type[Workload], policy: str | None = None):
    """Register ``workload`` as "recording"; ``policy`` names an isolation policy target."""

    workload_entry = MagicMock()
    workload_entry.name = "recording"
    workload_entry.load.return_value = workload
    groups = {"aorta.workloads": [workload_entry]}
    if policy is not None:
        policy_entry = MagicMock()
        policy_entry.name = "recording"
        policy_entry.group = "aorta.workload_policies"
        policy_entry.value = policy
        groups["aorta.workload_policies"] = [policy_entry]
    entries = MagicMock()
    entries.select.side_effect = lambda group=None, **_: groups.get(group, [])
    return patch("importlib.metadata.entry_points", return_value=entries)


def test_a_sanitizer_request_is_refused_outside_an_asan_run(tmp_path):
    request = RunRequest(workload="recording", trials=1, results_dir=tmp_path, sanitizer="asan")
    with pytest.raises(ValueError, match="aorta run --sanitizer asan"):
        run_trials(request)


def test_only_asan_is_a_sanitizer(tmp_path):
    request = RunRequest(workload="recording", trials=1, results_dir=tmp_path, sanitizer="tsan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sanitizer must be 'asan'"):
        run_trials(request)


@pytest.fixture
def inside_an_asan_run(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    monkeypatch.setattr(asan_run, "_asan_runtime_loaded", lambda: True)
    return evidence


def test_an_asan_run_tells_the_workload(tmp_path, inside_an_asan_run):
    with _registered(_ConfigRecordingWorkload):
        (result,) = run_trials(
            RunRequest(workload="recording", trials=1, results_dir=tmp_path, sanitizer="asan")
        )
    assert result.exit_status == "ok"
    assert _ConfigRecordingWorkload.seen["_aorta_sanitizer"] == "asan"
    assert result.config["_aorta_sanitizer"] == "asan"


def test_a_run_without_a_sanitizer_carries_no_key(tmp_path):
    with _registered(_ConfigRecordingWorkload):
        (result,) = run_trials(RunRequest(workload="recording", trials=1, results_dir=tmp_path))
    assert "_aorta_sanitizer" not in result.config


def test_an_asan_run_refuses_a_cell_that_changes_what_it_owns(
    tmp_path, inside_an_asan_run, monkeypatch
):
    monkeypatch.setenv("LD_PRELOAD", "/asan/libclang_rt.asan-x86_64.so")
    request = RunRequest(
        workload="recording",
        trials=1,
        results_dir=tmp_path,
        extra_env={"LD_PRELOAD": "/opt/hrx/lib/libamdhip64.so"},
        sanitizer="asan",
    )
    with (
        _registered(_ConfigRecordingWorkload),
        pytest.raises(ValueError, match="owns LD_PRELOAD, "),
    ):
        run_trials(request)


def test_an_asan_run_refuses_extra_env_that_selects_other_gpus(
    tmp_path, inside_an_asan_run, monkeypatch
):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    request = RunRequest(
        workload="recording",
        trials=1,
        results_dir=tmp_path,
        extra_env={"HIP_VISIBLE_DEVICES": "3"},
        sanitizer="asan",
    )
    with (
        _registered(_ConfigRecordingWorkload),
        pytest.raises(ValueError, match="owns HIP_VISIBLE_DEVICES, .*node-health canary"),
    ):
        run_trials(request)


class _ProcessWorkload(_ConfigRecordingWorkload):
    trial_isolation_default = "process"
    trial_isolation_required = True
    trial_isolation_supported = frozenset({"process"})


def test_a_process_isolated_trial_hands_the_sanitizer_to_its_worker(
    tmp_path, inside_an_asan_run, monkeypatch
):
    payloads = []

    class _StopError(Exception):
        pass

    def capture(payload, **kwargs):
        payloads.append(payload)
        raise _StopError

    monkeypatch.setattr(dispatcher_module, "launch_trial_worker", capture)
    with (
        _registered(_ProcessWorkload, "aorta.run.validation:PROCESS_REQUIRED_POLICY"),
        contextlib.suppress(_StopError),
    ):
        run_trials(
            RunRequest(workload="recording", trials=1, results_dir=tmp_path, sanitizer="asan")
        )

    (payload,) = payloads
    assert payload["run_request"]["sanitizer"] == "asan"


def test_the_trial_worker_keeps_the_sanitizer(tmp_path):
    request = _trial_worker._run_request_from_dict(
        {"workload": "recording", "results_dir": str(tmp_path), "sanitizer": "asan"}
    )
    assert request.sanitizer == "asan"


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def test_the_child_command_carries_every_option_given(tmp_path):
    sidecar = tmp_path / "m.json"
    sidecar.write_text("{}")
    args = [
        "--workload",
        "hrx",
        "--trials",
        "2",
        "--mitigations",
        "xnack",
        "--mitigations-file",
        str(sidecar),
        "--extra-env",
        "A=-1,B=2",
        "--sanitizer",
        "asan",
        "-vv",
    ]
    ctx = run_cli.run.make_context("run", list(args))

    assert run_cli._forwarded_args(ctx) == [
        "--workload=hrx",
        "--trials=2",
        "--mitigations=xnack",
        f"--mitigations-file={sidecar}",
        "--extra-env=A=-1,B=2",
        "--sanitizer=asan",
        "--verbose",
        "--verbose",
    ]
    reparsed = run_cli.run.make_context("run", run_cli._forwarded_args(ctx))
    assert reparsed.params == ctx.params


def test_without_an_asan_runtime_the_parent_refuses(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_ASAN_RUN, raising=False)
    monkeypatch.setattr(
        "aorta.instrumentation.rocjitsu_sanitizers.asan.resolve_asan_runtime", lambda: None
    )
    result = CliRunner().invoke(
        run_cli.run, ["--workload", "hrx", "--sanitizer", "asan", "--results-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "needs the ASan-instrumented ROCm runtime" in result.output
    assert list(tmp_path.iterdir()) == []


def _outcome(verdict: Verdict, returncode: int, tmp_path: Path) -> AsanRunOutcome:
    report = asan_run.AsanRunReport(
        verdict=verdict,
        reason="because",
        returncode=returncode,
        findings=(),
        kernels=(),
        processes=(),
        opaque_device_code=(),
        evidence_dir=str(tmp_path),
    )
    return AsanRunOutcome(returncode, report, tmp_path / REPORT_NAME)


@pytest.mark.parametrize(
    ("verdict", "returncode", "exit_code"),
    [
        (Verdict.PASS, 0, 0),
        (Verdict.NOT_CHECKED, 0, EXIT_NOT_CHECKED),
        (Verdict.FAIL, 1, 1),
        (Verdict.ERROR, 134, 1),
        (Verdict.NOT_CHECKED, 1, 1),
        (Verdict.PASS, 2, 2),
    ],
)
def test_the_parent_exit_status_follows_the_verdict(
    monkeypatch, tmp_path, verdict, returncode, exit_code
):
    monkeypatch.delenv(ENV_ASAN_RUN, raising=False)
    monkeypatch.setattr(
        "aorta.instrumentation.rocjitsu_sanitizers.asan.resolve_asan_runtime",
        lambda: _runtime(tmp_path),
    )
    commands: list[list[str]] = []

    def fake_run(command, *, evidence_dir, runtime):
        commands.append(list(command))
        assert evidence_dir.parent == (tmp_path / "results" / "hrx" / "asan").absolute()
        return _outcome(verdict, returncode, evidence_dir)

    monkeypatch.setattr(asan_run, "run_under_asan", fake_run)
    result = CliRunner().invoke(
        run_cli.run,
        ["--workload", "hrx", "--sanitizer", "asan", "--results-dir", str(tmp_path / "results")],
    )

    assert result.exit_code == exit_code, result.output
    assert f"ASan: {verdict.value.replace('_', ' ').upper()} -- because" in result.output
    (command,) = commands
    assert command == run_cli._child_command(
        ["--workload=hrx", f"--results-dir={tmp_path / 'results'}", "--sanitizer=asan"]
    )


def test_a_node_that_fails_its_health_check_exits_1_and_says_so(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_ASAN_RUN, raising=False)
    monkeypatch.setattr(
        "aorta.instrumentation.rocjitsu_sanitizers.asan.resolve_asan_runtime",
        lambda: _runtime(tmp_path),
    )

    def fake_run(command, *, evidence_dir, runtime):
        report = asan_run.AsanRunReport(
            verdict=Verdict.ERROR,
            reason="node health check failed, so ASan results on this node cannot be trusted",
            returncode=None,
            findings=(),
            kernels=(),
            processes=(),
            opaque_device_code=(),
            evidence_dir=str(evidence_dir),
            node_health=NodeHealth(NodeHealthStatus.FAILED, "the canary reported unknown-crash"),
        )
        return AsanRunOutcome(None, report, evidence_dir / REPORT_NAME)

    monkeypatch.setattr(asan_run, "run_under_asan", fake_run)
    result = CliRunner().invoke(
        run_cli.run,
        ["--workload", "hrx", "--sanitizer", "asan", "--results-dir", str(tmp_path / "results")],
    )

    assert result.exit_code == 1, result.output
    assert "Node health: FAILED" in result.output
    assert (
        "Error: this node failed ASan's health check, so the workload was not run; run it on "
        "another node" in result.output
    )


def test_the_child_command_runs_aorta_run():
    """Executed, not compared: a command that only looks right is the failure mode."""

    completed = subprocess.run(
        run_cli._child_command(["--help"]), capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("Usage: aorta run ")
    assert "--sanitizer" in completed.stdout


def test_an_unsafe_workload_name_never_becomes_a_path(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_ASAN_RUN, raising=False)
    monkeypatch.setattr(
        "aorta.instrumentation.rocjitsu_sanitizers.asan.resolve_asan_runtime",
        lambda: _runtime(tmp_path),
    )
    results = tmp_path / "results"
    result = CliRunner().invoke(
        run_cli.run, ["--workload", "../../x", "--sanitizer", "asan", "--results-dir", str(results)]
    )
    assert result.exit_code == 1
    assert "is not a workload name" in result.output
    assert not results.exists()


def test_the_child_runs_the_workload_with_the_sanitizer(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ASAN_RUN, str(tmp_path))
    requests: list[RunRequest] = []

    def fake_run_trials(request):
        requests.append(request)
        return []

    monkeypatch.setattr(run_cli, "run_trials", fake_run_trials)
    monkeypatch.setattr(run_cli, "summarize_results", lambda results: MagicMock(failed=0, total=0))
    result = CliRunner().invoke(run_cli.run, ["--workload", "hrx", "--sanitizer", "asan"])

    assert result.exit_code == 0, result.output
    (request,) = requests
    assert request.sanitizer == "asan"


def test_the_sanitizer_option_is_documented():
    option = next(param for param in run_cli.run.params if param.name == "sanitizer")
    assert isinstance(option.type, click.Choice) and list(option.type.choices) == ["asan"]
    assert "3 when the run was clean" in (option.help or "")
