"""``aorta run --sanitizer asan``: the run environment, the evidence and the verdict.

CPU-only. Device code is synthesised as real ELF images (an x86-64 host binary
with a ``.hip_fatbin`` section, clang offload bundles, AMDGPU code objects with
kernel-descriptor symbols), so the inventory is exercised on the layout it
reads in production rather than on stubs.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
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
)
from aorta.instrumentation.rocjitsu_sanitizers.asan_run import (
    ENV_ASAN_RUN,
    EXIT_NOT_CHECKED,
    REPORT_NAME,
    AsanRunOutcome,
    active_asan_run,
    asan_run_environment,
    declare_code_objects,
    evaluate_asan_run,
    loaded_files,
    owned_setting_conflicts,
    write_report,
)
from aorta.instrumentation.rocjitsu_sanitizers.models import Verdict
from aorta.run import _trial_worker
from aorta.run.dispatcher import RunRequest, run_trials
from aorta.workloads import Workload, WorkloadResult

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
    # A CCOB v2 header (zstd); the payload is never opened.
    return _write(path, _host_binary(b"CCOB" + struct.pack("<HH", 2, 2) + b"\0" * 32))


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

    declare_code_objects([Path("build/probe"), build / "kernel.code"])
    declare_code_objects(["build/missing"])

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
    declare_code_objects([probe])
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


def test_a_declaration_the_parent_cannot_parse_is_ignored(tmp_path):
    evidence = _evidence(tmp_path)
    probe = _instrumented_binary(tmp_path / "hrx_probe")
    (evidence / "code_objects.jsonl").write_text(
        "not json\n"
        "[1, 2]\n"
        + json.dumps(
            {
                "path": "relative/probe",
                "compressed_bundles": 0,
                "unreadable": None,
                "code_objects": [],
            }
        )
        + "\n"
        + json.dumps({"path": "/x", "code_objects": "wrong"})
        + "\n"
    )
    _process(evidence, 7, program=probe, kernels=["store_index(int*, int)"])

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.PASS


def test_declaring_outside_an_asan_run_does_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_ASAN_RUN, raising=False)
    monkeypatch.chdir(tmp_path)
    declare_code_objects([tmp_path / "probe"])
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


def test_unchecked_in_one_process_is_unchecked_for_the_run(tmp_path):
    evidence = _evidence(tmp_path)
    _process(
        evidence,
        7,
        program=_instrumented_binary(tmp_path / "a"),
        kernels=["store_index(int*, int)"],
    )
    _process(evidence, 8, program=_plain_binary(tmp_path / "b"), kernels=["store_index(int*, int)"])

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


def test_a_declared_code_object_attributes_a_module_kernel(tmp_path, monkeypatch):
    """hrx's module probe: the kernel arrives through hipModuleLoad, not the loader."""

    evidence = _evidence(tmp_path)
    module = _write(
        tmp_path / "add100.code", _bundle([(_GFX950_XNACK, _code_object(["add100"], asan=True))])
    )
    monkeypatch.setenv(ENV_ASAN_RUN, str(evidence))
    declare_code_objects([module])
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


def test_triton_cache_kernels_are_attributed(tmp_path):
    evidence = _evidence(tmp_path)
    cache = evidence / "triton-cache" / "abc123"
    cache.mkdir(parents=True)
    _write(cache / "add_kernel.hsaco", _code_object(["add_kernel"], asan=True))
    _process(evidence, 7, program=_python(tmp_path), kernels=["add_kernel"])

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.PASS


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
    report = evaluate_asan_run(_evidence(tmp_path), returncode=1)
    assert report.verdict is Verdict.NOT_CHECKED
    assert "no process wrote a HIP runtime log" in report.reason


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

    assert evaluate_asan_run(evidence, returncode=0).verdict is Verdict.PASS


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


def _registered(workload: type[Workload]):
    entry = MagicMock()
    entry.name = "recording"
    entry.load.return_value = workload
    entries = MagicMock()
    entries.select.return_value = [entry]
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
        pytest.raises(ValueError, match="sets LD_PRELOAD itself"),
    ):
        run_trials(request)


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
