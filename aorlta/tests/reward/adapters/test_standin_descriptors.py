"""The compile gate's build fact: standin.hipcc reads the launched kernels' static LDS from the kernel descriptors of
the stand-in harness's device code object and adds the harness's dynamic LDS. hipcc runs on CPU with an explicit
offload arch (nothing here opens a GPU); fake tool scripts stand in for a readelf or a compiler that fails. The tests
that compile skip without hipcc and llvm-readelf."""

from __future__ import annotations

import json
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from aorlta.ids import sha256_text
from aorlta.reward.adapters import descriptors as D
from aorlta.reward.adapters.base import TRUSTED_BUILD, checked_result
from aorlta.reward.adapters.standin_hipcc import FACT_FILE, FACT_SOURCE, StandinHipcc
from aorlta.schemas import LegJob

HIPCC = "/opt/rocm/bin/hipcc"
READELF = "/opt/rocm/llvm/bin/llvm-readelf"
H = sha256_text("x")
COMPILE = {"offload_arch": "gfx950", "flags": ["-c", "-O1"]}
needs_hipcc = pytest.mark.skipif(not (Path(HIPCC).exists() and Path(READELF).exists()), reason="no hipcc or readelf")

I = "#include <hip/hip_runtime.h>\n"
SOURCES = {
    "no_lds": I + "__global__ void scale_kernel(float* data, int n) {\n"
                  "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < n) data[i] *= 2.0f;\n}\n",
    "static_lds": I + "__global__ void reverse_kernel(float* data) {\n  __shared__ float tile[64];\n"
                      "  tile[threadIdx.x % 64] = data[threadIdx.x];\n  __syncthreads();\n"
                      "  data[threadIdx.x] = tile[63 - threadIdx.x % 64];\n}\n",
    "dynamic_lds": I + "__global__ void stage_kernel(float* data) {\n  extern __shared__ float staged[];\n"
                       "  staged[threadIdx.x] = data[threadIdx.x];\n  __syncthreads();\n"
                       "  data[threadIdx.x] = staged[0];\n}\n",
    "template_lds": I + "template <typename T>\n__global__ void copy_kernel(T* data) {\n  __shared__ T tile[32];\n"
                        "  tile[threadIdx.x % 32] = data[threadIdx.x];\n  __syncthreads();\n"
                        "  data[threadIdx.x] = tile[0];\n}\n",
    "template_no_lds": I + "template <typename T>\n"
                           "__global__ void bump_kernel(T* data) { data[threadIdx.x] += T(1); }\n",
    "two_kernels": I + "__global__ void fill_kernel(float* data) { data[threadIdx.x] = 1.0f; }\n"
                       "__global__ void gather_kernel(float* data) {\n  __shared__ float part[8];\n"
                       "  part[threadIdx.x % 8] = data[threadIdx.x];\n  __syncthreads();\n  data[0] = part[1];\n}\n",
    "own_main": I + "__global__ void scale_kernel(float* data) { data[0] = 1.0f; }\nint main() { return 0; }\n",
    "no_instance": I + "__global__ void fill_kernel(float* data) { data[threadIdx.x] = 1.0f; }\n"
                       "template <template <typename> class Policy>\n"
                       "__global__ void policy_kernel(float* data) { data[0] = 2.0f; }\n",
}


def job(leg_name: str, out: Path, source: Path, **settings) -> LegJob:
    return LegJob(f"j#{leg_name}", leg_name, "c", None, H, H, 1, "w", {**COMPILE, **settings}, 0.0, None,
                  {"source": str(source)}, str(out))


def script(path: Path, body: str) -> str:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def gate_fact(result) -> dict | None:
    facts = [f for f in result.findings if f.get("kind") == TRUSTED_BUILD]
    assert len(facts) <= 1
    return facts[0] if facts else None


class _Gate(dict):
    """name -> (LegResult, launched_lds.json) of SOURCES[name] through the real compile gate, compiled on first use."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def __missing__(self, name: str):
        src = self.root / name / "source.hip"
        src.parent.mkdir()
        src.write_text(SOURCES[name])
        result = StandinHipcc(HIPCC, readelf=READELF).run(job("gate.compile", self.root / name / "out", src))
        self[name] = (result, json.loads((self.root / name / "out" / FACT_FILE).read_text()))
        return self[name]


@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    if not (Path(HIPCC).exists() and Path(READELF).exists()):
        pytest.skip("no hipcc or readelf")
    return _Gate(tmp_path_factory.mktemp("gate"))


@pytest.fixture(scope="module")
def code_object(tmp_path_factory) -> Path:
    """A device code object with one kernel without LDS and one with 256 bytes."""
    if not (Path(HIPCC).exists() and Path(READELF).exists()):
        pytest.skip("no hipcc or readelf")
    root = tmp_path_factory.mktemp("co")
    src = root / "k.hip"
    src.write_text(SOURCES["no_lds"] + SOURCES["static_lds"].replace(I, ""))
    obj = root / "k.co"
    subprocess.run([HIPCC, "--offload-arch=gfx950", "--cuda-device-only", "--no-gpu-bundle-output", "-c", "-O1",
                    str(src), "-o", str(obj)], check=True, capture_output=True, timeout=300)
    return obj


# --- the descriptor reader -------------------------------------------------------------------------------------------


def run_tool(argv, timeout_s):
    proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s)
    return proc.returncode, proc.stdout


@needs_hipcc
def test_the_descriptors_and_the_metadata_note_agree_on_a_real_code_object(code_object):
    kd = D.kd_group_segments(code_object.read_bytes())
    assert kd == {"_Z12scale_kernelPfi": 0, "_Z14reverse_kernelPf": 256}
    assert D.note_group_segments(run_tool([READELF, "--notes", str(code_object)], 60)[1]) == kd
    assert D.read_group_segments(code_object, readelf=READELF, run=run_tool) == (kd, None)


def _kd_offset(path: Path, kernel: str) -> int:
    """Where `<kernel>.kd` starts in the file, found independently of the reader under test (llvm-readelf)."""
    syms = run_tool([READELF, "-sW", str(path)], 60)[1]
    value = next(int(line.split()[1], 16) for line in syms.splitlines() if line.endswith(f" {kernel}.kd"))
    for line in run_tool([READELF, "-SW", str(path)], 60)[1].splitlines():
        parts = line.replace("[ ", "[").split()
        if len(parts) > 5 and parts[1] == ".rodata":
            addr, off = int(parts[3], 16), int(parts[4], 16)
            return off + value - addr
    raise AssertionError("no .rodata")


@needs_hipcc
def test_a_descriptor_that_disagrees_with_its_metadata_gives_no_sizes(code_object, tmp_path):
    data = bytearray(code_object.read_bytes())
    at = _kd_offset(code_object, "_Z12scale_kernelPfi")
    assert int.from_bytes(data[at:at + 4], "little") == 0
    data[at:at + 4] = (512).to_bytes(4, "little")
    doctored = tmp_path / "doctored.co"
    doctored.write_bytes(bytes(data))
    assert D.kd_group_segments(bytes(data))["_Z12scale_kernelPfi"] == 512
    sizes, why = D.read_group_segments(doctored, readelf=READELF, run=run_tool)
    assert sizes is None and why.startswith("descriptor_mismatch")


@needs_hipcc
@pytest.mark.parametrize("case", ["not_elf", "truncated", "readelf_fails", "no_metadata", "metadata_without_size",
                                  "no_kernel"])
def test_an_unreadable_descriptor_or_metadata_gives_no_sizes(case, code_object, tmp_path):
    path, readelf = code_object, READELF
    if case == "not_elf":
        path = tmp_path / "x.co"
        path.write_bytes(b"not an elf object at all" * 8)
    elif case == "truncated":
        path = tmp_path / "x.co"
        path.write_bytes(code_object.read_bytes()[:200])
    elif case == "readelf_fails":
        readelf = script(tmp_path / "readelf", "exit 1\n")
    elif case == "no_metadata":
        readelf = script(tmp_path / "readelf", "echo 'Displaying notes found in: .note'\n")
    elif case == "metadata_without_size":
        notes = run_tool([READELF, "--notes", str(code_object)], 60)[1]
        stripped = "\n".join(line for line in notes.splitlines() if ".group_segment_fixed_size" not in line)
        (tmp_path / "notes.txt").write_text(stripped)
        readelf = script(tmp_path / "readelf", f"cat {tmp_path / 'notes.txt'}\n")
    elif case == "no_kernel":
        src = tmp_path / "f.hip"
        src.write_text(I + "__device__ float twice(float x) { return 2.0f * x; }\n")
        path = tmp_path / "f.co"
        subprocess.run([HIPCC, "--offload-arch=gfx950", "--cuda-device-only", "--no-gpu-bundle-output", "-c", "-O1",
                        str(src), "-o", str(path)], check=True, capture_output=True, timeout=300)
    sizes, why = D.read_group_segments(path, readelf=readelf, run=run_tool)
    assert sizes is None and why


# --- the compile gate's fact -----------------------------------------------------------------------------------------


@needs_hipcc
def test_a_kernel_without_lds_gets_a_trusted_zero_lds_fact(gate):
    result, side = gate["no_lds"]
    fact = gate_fact(result)
    assert result.state == "clean" and side == {"fact": fact, "why": None}
    assert fact["source"] == FACT_SOURCE and fact["language"] == "hip"
    assert fact["launched_lds"] == {"static_bytes": 0, "dynamic_bytes": 0, "symbol": "_Z12scale_kernelPfi"}
    assert fact["kernels"] == [{"symbol": "_Z12scale_kernelPfi", "static_bytes": 0}]
    assert fact["launches"] == ["scale_kernel"]


@needs_hipcc
@pytest.mark.parametrize("name, static, dynamic, kernels", [
    ("static_lds", 256, 0, [("_Z14reverse_kernelPf", 256)]),
    ("dynamic_lds", 0, 16 << 10, [("_Z12stage_kernelPf", 0)]),  # the harness launches it with 16 KiB
    ("template_lds", 128, 0, [("_Z11copy_kernelIfEvPT_", 128)]),  # read from the harness's own instance
    ("template_no_lds", 0, 0, [("_Z11bump_kernelIfEvPT_", 0)]),
    ("two_kernels", 32, 0, [("_Z11fill_kernelPf", 0), ("_Z13gather_kernelPf", 32)]),  # the largest decides
])
def test_each_launched_kernels_descriptor_lds_and_the_harnesss_dynamic_lds_are_recorded(gate, name, static, dynamic,
                                                                                         kernels):
    result, _ = gate[name]
    fact = gate_fact(result)
    assert result.state == "clean"
    assert fact["kernels"] == [{"symbol": s, "static_bytes": b} for s, b in kernels]
    assert (fact["launched_lds"]["static_bytes"], fact["launched_lds"]["dynamic_bytes"]) == (static, dynamic)
    assert fact["launched_lds"]["symbol"] == (kernels[0][0] if len(kernels) == 1 else None)


@needs_hipcc
@pytest.mark.parametrize("name, why", [
    ("own_main", "own_main"),        # its own launch sets the dynamic LDS: not the harness's to know
    ("no_instance", "no_instance"),  # one kernel the harness cannot launch, so its LDS is unknown
])
def test_no_fact_when_a_launched_kernels_lds_is_unknown_and_the_compile_stays_clean(gate, name, why):
    result, side = gate[name]
    assert result.state == "clean" and gate_fact(result) is None
    assert side["fact"] is None and side["why"].startswith(why)


@needs_hipcc
def test_no_fact_when_the_descriptor_cannot_be_read(tmp_path):
    src = tmp_path / "source.hip"
    src.write_text(SOURCES["no_lds"])
    for name, body in (("fails", "exit 1\n"), ("silent", "exit 0\n")):
        readelf = script(tmp_path / f"readelf-{name}", body)
        result = StandinHipcc(HIPCC, readelf=readelf).run(job("gate.compile", tmp_path / name, src))
        side = json.loads((tmp_path / name / FACT_FILE).read_text())
        assert result.state == "clean" and gate_fact(result) is None, name
        assert side["why"].startswith("metadata_unreadable"), name
    missing = StandinHipcc(HIPCC, readelf=str(tmp_path / "no-readelf"))
    assert "+readelf-missing+" in missing.version
    assert gate_fact(missing.run(job("gate.compile", tmp_path / "missing", src))) is None


@needs_hipcc
def test_a_metadata_note_that_hides_the_lds_is_caught_by_the_descriptor(tmp_path):
    """A readelf whose metadata says 0 for a kernel whose descriptor says 256: the two readings disagree, so no fact."""
    src = tmp_path / "source.hip"
    src.write_text(SOURCES["static_lds"])
    probe = StandinHipcc(HIPCC, readelf=READELF)
    real = probe.run(job("gate.compile", tmp_path / "real", src))
    assert gate_fact(real)["launched_lds"]["static_bytes"] == 256
    lying = script(tmp_path / "readelf", f"{READELF} \"$@\" | sed 's/group_segment_fixed_size: 256/"
                                         "group_segment_fixed_size: 0/'\n")
    result = StandinHipcc(HIPCC, readelf=lying).run(job("gate.compile", tmp_path / "lying", src))
    side = json.loads((tmp_path / "lying" / FACT_FILE).read_text())
    assert result.state == "clean" and gate_fact(result) is None and side["why"].startswith("descriptor_mismatch")


@needs_hipcc
def test_a_compile_error_carries_no_fact_and_a_clean_fact_passes_the_adapter_check(gate, tmp_path):
    src = tmp_path / "bad.hip"
    src.write_text(I + "__global__ void k(int* p) { p[0] = nope; }\n")
    bad = StandinHipcc(HIPCC, readelf=READELF).run(job("gate.compile", tmp_path / "bad", src))
    assert (bad.state, [f["kind"] for f in bad.findings]) == ("fail", ["compile_error"])
    result, _ = gate["no_lds"]
    j = job("gate.compile", tmp_path / "x", src)
    assert checked_result(j, result, 0.0, 1.0) is result  # a trusted build fact is not a finding


def compiles_then(after_ast: str) -> str:
    """A fake hipcc: the compile writes its object at once; a clang AST dump (the harness planner) runs `after_ast`."""
    return ('for a in "$@"; do case "$a" in -ast-dump*) ' + after_ast + '; exit 0;; esac; done\n'
            'prev=; for a in "$@"; do [ "$prev" = -o ] && echo object > "$a"; prev=$a; done\nexit 0\n')


def test_the_fact_stays_inside_the_gates_own_timeout(tmp_path):
    hipcc = script(tmp_path / "hipcc", compiles_then("sleep 30"))
    src = tmp_path / "source.hip"
    src.write_text(SOURCES["no_lds"])
    t0 = time.monotonic()
    result = StandinHipcc(hipcc, readelf=READELF, timeout_s=3.0).run(job("gate.compile", tmp_path / "out", src))
    assert time.monotonic() - t0 < 15.0
    assert result.state == "clean" and gate_fact(result) is None


def test_closing_the_worker_while_the_fact_is_read_leaves_the_leg_unsettled(tmp_path):
    hipcc = script(tmp_path / "hipcc", compiles_then("sleep 60"))
    src = tmp_path / "source.hip"
    src.write_text(SOURCES["no_lds"])
    gate_adapter, out = StandinHipcc(hipcc, readelf=READELF, timeout_s=50.0), {}
    t = threading.Thread(target=lambda: out.setdefault("r", gate_adapter.run(job("gate.compile", tmp_path / "o", src))))
    t.start()
    time.sleep(1.0)
    gate_adapter.close()
    t.join(timeout=15)
    assert not t.is_alive() and (out["r"].state, out["r"].reason) == ("unsettled", "worker_died")
