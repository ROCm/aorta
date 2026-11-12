"""The stand-in launch harness (`standin_harness`): a kernel-only candidate (the stand-in prompt asks for no `main`) gets
a `main` that launches each of its kernels, one program per kernel, so aorta's ConSan and ASan can run it, and its
template kernels are instantiated, so waitcheck has code to scan. A kernel the harness cannot launch leaves the leg
unsettled, never clean.

Nothing here opens a GPU or runs a device program: hipcc runs on CPU with an explicit offload arch and only builds;
aorta's own ELF reader names the kernels a build holds; and the harness's argument rules run on the host, compiled
with a host compiler against a stub HIP header (`fakehip/`) whose launch only records what it was given."""

from __future__ import annotations

import json
import subprocess
import os
from pathlib import Path

import pytest

from aorlta.ids import sha256_text
from aorlta.reward.adapters import standin_harness as H
from aorlta.reward.adapters.standin_aorta import AortaAsan, AortaConsan, combine, single_kernel
from aorlta.reward.adapters.standin_harness import (BLOCK, DYNAMIC_LDS_BYTES, FUNCTOR, HARNESS_NAME, INT_ARG,
                                                     NTTP_INT, WAVEFRONT, HarnessUnsupported, Instance, Kernel,
                                                     TemplateParam, declared_names, find_kernels, harness_source,
                                                     has_main, launch_block, plan, template_attempts, write_harness)
from aorlta.reward.adapters.standin_rj_waitcheck import StandinRjWaitcheck
from aorlta.schemas import LegJob

HIPCC = "/opt/rocm/bin/hipcc"
CLANGXX = "/opt/rocm/llvm/bin/clang++"
# Site paths, each optional (see test_standin_cpu.py).
AORTA = str(Path(os.environ.get("AORLTA_AORTA_ROOT", "/opt/aorta")) / "src")
RJ = os.environ.get("AORLTA_RJ_WAITCHECK", "/opt/rocjitsu/bin/rj_waitcheck")
RJ_LIB = os.environ.get("AORLTA_LIBSTDCXX_SHIM", "/opt/libstdcxx-shim")
FAKEHIP = Path(__file__).parent / "fakehip"
needs_hipcc = pytest.mark.skipif(not Path(HIPCC).exists(), reason="no hipcc")
needs_aorta = pytest.mark.skipif(not Path(AORTA).exists(), reason="no aorta checkout (AORLTA_AORTA_ROOT)")
needs_clangxx = pytest.mark.skipif(not Path(CLANGXX).exists(), reason="no host clang++")
needs_rj = pytest.mark.skipif(not (Path(RJ).exists() and Path(AORTA).exists()),
                              reason="no rj_waitcheck or aorta (AORLTA_RJ_WAITCHECK, AORLTA_AORTA_ROOT)")

VECADD = """#include <hip/hip_runtime.h>
// a comment naming another kernel: __global__ void old_kernel(float* x) and int main() { }
extern "C" __global__ void vecadd(const float* __restrict__ a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
"""
HALF_BOUNDS = """#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
__global__ __launch_bounds__(128) void scale(const __half* x, __half* y, __half s, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __hmul(x[i], s);
}
"""
WITH_MAIN = """#include <hip/hip_runtime.h>
__global__ void k(float* a) { a[threadIdx.x] = 1.0f; }
int main() { float* a; (void)hipMalloc(&a, 1024); hipLaunchKernelGGL(k, 1, 64, 0, 0, a); return hipDeviceSynchronize(); }
"""
# Every shape an earlier harness could not launch: templates (type and value parameters, a trailing
# default), a namespace (named and anonymous), overloads, an explicit specialisation, several kernels, struct and
# reference parameters. Plus what clang must not count: a commented-out kernel, one compiled out, a prototype.
SHAPES = """#include <hip/hip_runtime.h>
#include <array>
// __global__ void in_a_comment(int* x) {}
#if 0
__global__ void compiled_out(int* x) { x[0] = 1; }
#endif
__global__ void prototype_only(int* x);
namespace outer { namespace {
struct Meta { void* addresses[2][20]; int numel[20]; };
template <typename T, int NUM_THREADS, bool LOG, typename acc_t = T>
__global__ void softmax(T* out, const T* in, int n, const Meta& meta) {
  __shared__ acc_t s[NUM_THREADS];
  if (threadIdx.x < NUM_THREADS) s[threadIdx.x] = in[threadIdx.x % n];
  __syncthreads();
  if (threadIdx.x < n) out[threadIdx.x] = LOG ? -s[0] : s[0];
}
} }
__global__ void over(float* x) { x[0] = 1; }
__global__ void over(int* x, int n) { x[0] = n; }
template <typename T> __global__ void spec(T* x) { x[0] = T(1); }
template <> __global__ void spec<double>(double* x) { x[0] = 2.0; }
__global__ void elementwise(int n, std::array<char*, 2> data) {
  if (threadIdx.x < n) ((float*)data[0])[threadIdx.x] = ((const float*)data[1])[threadIdx.x];
}
"""
INDEXED = """#include <hip/hip_runtime.h>
template <typename T> __global__ void mod(T* x, int n) { if (threadIdx.x < n) x[threadIdx.x] = x[threadIdx.x] % 7; }
"""
CALLED = """#include <hip/hip_runtime.h>
template <typename Functor, int VEC> __global__ void apply(int n, Functor f, float* x) {
  if (threadIdx.x < n) x[threadIdx.x] = f(x[threadIdx.x]) * VEC;
}
"""
BROKEN_TEMPLATE = """#include <hip/hip_runtime.h>
template <typename T> __global__ void broken(const T* out) { out[threadIdx.x] = T(0); }
"""
NO_KERNEL = """#include <hip/hip_runtime.h>
// only a comment: __global__ void k(int* x) {}
#if defined(NOT_DEFINED_ANYWHERE)
__global__ void k(int* x) { x[0] = 1; }
#endif
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name / "source.hip"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _inventory(exe: Path) -> list[str]:
    import sys

    if AORTA not in sys.path:
        sys.path.append(AORTA)
    from aorta.instrumentation.rocjitsu_sanitizers.asan import kernel_inventory

    return sorted(k for co in kernel_inventory(exe).code_objects for k in co.kernels)


# --- what the source says, as text -----------------------------------------------------------------------------


def test_main_and_declared_names_are_read_from_code_not_comments():
    assert not has_main(VECADD) and has_main(WITH_MAIN)
    assert declared_names(VECADD) == ["vecadd"]
    assert declared_names(HALF_BOUNDS) == ["scale"]  # __launch_bounds__ between __global__ and void
    assert declared_names("__global__ void __attribute__((noinline)) a(int* x) {}") == ["a"]
    assert single_kernel(VECADD) == "vecadd" and single_kernel(VECADD + "\n__global__ void two(int* x) {}") is None
    assert "in_a_comment" not in declared_names(SHAPES)


@pytest.mark.parametrize("text, defines_main", [
    ('constexpr const char* banner = "int main(";\n', False),  # a string literal is not code
    ('const char* s = "a \\" int main() {";\n', False),  # an escaped quote does not end it
    ('auto r = R"x(int main() { )x";\n', False),  # nor does a raw string
    ("int main();\n", False),  # a declaration defines nothing
    ('const char* u = "http://x"; char q = \'"\';\nint main(int argc, char** argv)\n{\n  return 0;\n}\n', True),
])
def test_main_is_a_definition_in_code_not_text_in_a_string_literal(text, defines_main):
    assert has_main(text) is defines_main


def test_the_launch_respects_launch_bounds_and_dynamic_shared_memory():
    assert launch_block(VECADD) == BLOCK and launch_block(HALF_BOUNDS) == 128
    k = Kernel("vecadd", (), None, "void (const float *, const float *, float *, int)")
    src = harness_source("/x/vecadd.hip", VECADD, [Instance(k, None, launch_block(VECADD))])
    assert '#include "/x/vecadd.hip"' in src and "run(&vecadd, 256u)" in src and "dim3(4), dim3(block), argv, 0," in src
    dyn = "__global__ void d(float* o) { extern __shared__ float s[]; s[threadIdx.x] = 1; o[0] = s[0]; }"
    dk = Kernel("d", (), None, "void (float *)")
    assert f"argv, {DYNAMIC_LDS_BYTES}," in harness_source("/x/d.hip", dyn, [Instance(dk, None, BLOCK)])


@pytest.mark.parametrize("bad", [Kernel('d); system("x', (), None, "void ()"), Kernel("d", ("a b",), None, "void ()")])
def test_a_malformed_instance_is_refused(bad):
    with pytest.raises(ValueError):
        harness_source("/x/d.hip", "", [Instance(bad, None, BLOCK)])
    with pytest.raises(ValueError):
        harness_source("/x/d.hip", "", [Instance(Kernel("d", (), (), "void ()"), ("int; x",), BLOCK)])
    with pytest.raises(ValueError):
        harness_source("/x/d.hip", "", [Instance(Kernel("d", (), None, "void ()"), None, 0)])


# --- the kernels, from clang -----------------------------------------------------------------------------------


@needs_hipcc
def test_clang_finds_every_kernel_with_its_namespace_template_overload_and_specialisation(tmp_path):
    ks = {(k.qualified, k.overloaded, k.specialization) for k in find_kernels(_write(tmp_path, "s", SHAPES))}
    assert ks == {("outer::(anonymous namespace)::softmax", False, None), ("over", True, None),
                  ("over", True, None), ("spec", True, None), ("spec", True, ("double",)), ("elementwise", False, None)}
    soft = next(k for k in find_kernels(_write(tmp_path, "s2", SHAPES)) if k.name == "softmax")
    assert soft.scope == ("outer", "") and [(p.kind, p.name, p.has_default) for p in soft.template] == [
        ("type", "T", False), ("value", "NUM_THREADS", False), ("value", "LOG", False), ("type", "acc_t", True)]


@needs_hipcc
@pytest.mark.parametrize("text", [NO_KERNEL, "#include <hip/hip_runtime.h>\n// nothing here\n"],
                         ids=["compiled-out", "comments-only"])
def test_a_source_with_no_kernel_definition_is_unsupported(tmp_path, text):
    with pytest.raises(HarnessUnsupported):
        find_kernels(_write(tmp_path, "n", text))


# --- template arguments ------------------------------------------------------------------------------------------


def _tk(*params):
    return Kernel("k", (), tuple(TemplateParam(*p) for p in params), "void (T *)")


def test_representative_template_arguments():
    k = _tk(("type", "scalar_t", None, False, False), ("type", "index_t", None, False, False),
            ("type", "Functor", None, False, False), ("value", "BLOCK_SIZE", "int", False, False),
            ("value", "kWarpSize", "unsigned int", False, False), ("value", "ILP", "const int", False, False),
            ("value", "IsLog", "bool", False, False), ("type", "acc_t", None, True, False))
    assert template_attempts(k, 128) == [["float", "int", FUNCTOR, "128", str(WAVEFRONT), str(NTTP_INT), "false"],
                                         ["int", "int", FUNCTOR, "128", str(WAVEFRONT), str(NTTP_INT), "false"]]
    assert template_attempts(_tk(("type", "T", None, False, False), ("type", "Ts", None, False, True)), 256) == [
        ["float"], ["int"]]  # a trailing pack is left empty
    assert template_attempts(_tk(("type", "index_t", None, False, False)), 256) == [["int"]]


@pytest.mark.parametrize("params", [
    [("template", "C", None, False, False)],  # a template template parameter
    [("value", "M", "Mode", False, False)],  # an enum, or any type but an integer and bool
    [("value", "P", "float *", False, False)],
    [("type", "Ts", None, False, True), ("type", "T", None, False, False)],  # a pack that is not last
])
def test_a_template_the_harness_cannot_instantiate_is_unsupported(params):
    with pytest.raises(HarnessUnsupported):
        template_attempts(_tk(*params), 256)


@needs_hipcc
def test_plan_takes_the_first_instantiation_that_compiles(tmp_path):
    (p,) = plan(_write(tmp_path, "i", INDEXED), workdir=tmp_path / "i")  # `%` has no float overload: int it is
    assert p.instance.template_args == ("int",) and p.instance.label == "mod<int>"
    (p,) = plan(_write(tmp_path, "c", CALLED), workdir=tmp_path / "c")
    assert p.instance.template_args == (FUNCTOR, str(NTTP_INT))
    (p,) = plan(_write(tmp_path, "b", BROKEN_TEMPLATE), workdir=tmp_path / "b")  # writes through a const pointer
    assert p.instance is None and p.why == "no instantiation compiles"


@needs_hipcc
@needs_aorta
@pytest.mark.parametrize("cls", [AortaConsan, AortaAsan])
def test_each_kernel_instance_builds_into_its_own_program_holding_it(tmp_path, cls):
    adapter = cls(hook="/nonexistent", aorta_src=AORTA) if cls is AortaConsan else cls(aorta_src=AORTA)
    src = _write(tmp_path, "s", SHAPES)
    planned = plan(src, workdir=tmp_path / "s")
    held = {}
    for i, p in enumerate(planned):
        exe, why = adapter._build(str(write_harness(src, [p.instance], tmp_path / f"k{i}")), tmp_path / f"k{i}",
                                  adapter.build_flags())
        assert exe is not None, (p.instance.label, why, (tmp_path / f"k{i}" / "harness_build.log").read_text()[-800:])
        held[p.instance.label] = _inventory(exe)
    assert sorted(held) == ["elementwise", "outer::(anonymous namespace)::softmax<float, 256, false>",
                            "over [void (float *)]", "over [void (int *, int)]", "spec<double>", "spec<float>"]
    soft = held["outer::(anonymous namespace)::softmax<float, 256, false>"]
    assert any("outer::(anonymous namespace)::softmax<float, 256, false, float>(float*, float const*, int" in k
               for k in soft)  # instantiated with representative arguments, the trailing default kept
    assert "void spec<float>(float*)" in held["spec<float>"] and "void spec<double>(double*)" in held["spec<double>"]
    assert all("softmax" not in k for k in held["elementwise"])  # each program instantiates only its own kernel


# --- the argument rules, run on the host --------------------------------------------------------------------------

ARGS = """#include <hip/hip_runtime.h>
#include <array>
struct Meta { void* addresses[2][20]; int numel[20]; int start; };
struct Pair { float* p; int n; };
struct Fill { long v; Fill(long x) : v(x) {} };
enum class Mode { A, B };
__global__ void k(std::array<char*, 2> data, Pair pair, const Meta& meta, Meta* pmeta, Fill fill, Mode mode,
                  void (*fn)(float), float** pp, const int& r, int n, bool b, double d) {}
"""
PROBE = r"""#define main harness_main
#include "HARNESS"
#undef main
#include <cstdio>
template <typename T> T at(void** argv, int i) { return *static_cast<T*>(argv[i]); }
static void report(const FakeLaunch& l) {
  auto data = at<std::array<char*, 2>>(l.argv, 0);
  auto pair = at<Pair>(l.argv, 1);
  const Meta* meta = at<const Meta*>(l.argv, 2);
  Meta* pmeta = at<Meta*>(l.argv, 3);
  int filled = 0, zero_after = 1;
  for (int i = 0; i < 20; ++i) filled += meta->addresses[0][i] != nullptr;
  for (int i = 0; i < 20; ++i) zero_after &= meta->addresses[1][i] == nullptr && meta->numel[i] == 0;
  std::printf("{\"data\": [%d, %d], \"data_distinct\": %d, \"data_zero\": %d, \"pair_p\": %d, \"pair_n\": %d, "
              "\"meta_filled\": %d, \"meta_rest_zero\": %d, \"pmeta0\": %d, \"pmeta1\": %d, \"fill\": %ld, "
              "\"mode\": %d, \"fn\": %d, \"pp\": %d, \"pp0\": %d, \"r\": %d, \"n\": %d, \"b\": %d, \"d\": %g, "
              "\"grid\": %u, \"block\": %u, \"lds\": %zu}\n",
              data[0] != nullptr, data[1] != nullptr, data[0] != data[1], data[0][0] == 0 && data[1][100] == 0,
              pair.p != nullptr, pair.n, filled, zero_after, pmeta->addresses[0][0] != nullptr,
              pmeta[1].addresses[0][0] != nullptr, at<Fill>(l.argv, 4).v, (int)at<Mode>(l.argv, 5),
              at<void (*)(float)>(l.argv, 6) != nullptr, at<float**>(l.argv, 7) != nullptr,
              at<float**>(l.argv, 7)[0] != nullptr, *at<const int*>(l.argv, 8), at<int>(l.argv, 9),
              (int)at<bool>(l.argv, 10), at<double>(l.argv, 11), l.grid, l.block, l.lds);
}
int main() {
  fake_on_launch() = report;
  return harness_main();
}
"""


def run_on_host(tmp_path: Path, text: str, kernel: Kernel, block: int = BLOCK) -> dict:
    src = _write(tmp_path, "h", text)
    h = write_harness(src, [Instance(kernel, None, block)], tmp_path / "h")
    probe = tmp_path / "h" / "probe.cpp"
    probe.write_text(PROBE.replace("HARNESS", str(h)))
    exe = tmp_path / "h" / "probe"
    subprocess.run([CLANGXX, "-std=c++17", "-I", str(FAKEHIP), "-x", "c++", str(probe), "-o", str(exe)], check=True,
                   capture_output=True, text=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


@needs_clangxx
def test_struct_reference_and_pointer_arguments_follow_one_rule(tmp_path):
    k = Kernel("k", (), None, "void (...)")
    got = run_on_host(tmp_path, ARGS, k, block=128)
    assert got["data"] == [1, 1] and got["data_distinct"] == 1 and got["data_zero"] == 1  # std::array<char*, 2>
    assert got["pair_p"] == 1 and got["pair_n"] == INT_ARG  # a struct's members by the parameters' rule
    assert got["meta_filled"] == H.FILL_MAX and got["meta_rest_zero"] == 1  # up to FILL_MAX members, the rest zero
    assert got["pmeta0"] == 1 and got["pmeta1"] == 0  # a pointer to an aggregate: one value, in its first element
    assert got["fill"] == 1 and got["mode"] == 0  # a non-aggregate struct from 1.0f; an enum zero
    assert got["fn"] == 0  # a function pointer is null: there is no device function to give it
    assert got["pp"] == 1 and got["pp0"] == 0  # a pointer to a pointer: a zero-filled buffer
    assert got["r"] == INT_ARG  # a reference refers to device storage holding its value
    assert (got["n"], got["b"], got["d"]) == (INT_ARG, 1, 1.0)
    assert (got["grid"], got["block"], got["lds"]) == (H.GRID, 128, 0)


# --- the legs ------------------------------------------------------------------------------------------------------


def _job(leg: str, src: Path, out: Path, **inputs) -> LegJob:
    text = src.read_text()
    return LegJob(job_id=f"c/1/p/0#{leg}#a1", leg=leg, candidate_id="c/1/p/0", control_id=None,
                  source_sha256=sha256_text(text), task_sha256=sha256_text("t"), attempt=1, worker_id="w/1",
                  inputs={"source": str(src), **inputs}, output_dir=str(out),
                  leg_settings={"target": "gfx950", "max_diagnostics": 32} if leg == "san.waitcheck" else {})


class _Check:
    """Stands in for aorta's GPU run: records each program it is given; answers what the test says per kernel."""

    def __init__(self, answers: dict[str, tuple]):
        self.answers, self.seen = answers, []

    def __call__(self, models, exe, kernel, out):
        assert Path(exe).is_file()
        self.seen.append((kernel, out.name))
        return self.answers.get(kernel, ("clean", None, []))


@needs_hipcc
@needs_aorta
def test_each_kernel_is_checked_and_the_leg_is_clean_only_if_every_kernel_is(tmp_path, monkeypatch):
    adapter = AortaConsan(hook="/nonexistent", aorta_src=AORTA)
    src = _write(tmp_path, "s", SHAPES)
    for answers, expect in (({}, ("clean", None)),
                            ({"over": ("unsettled", "checker_not_settled:x", [])}, ("unsettled", "checker_not_settled:x")),
                            ({"spec": ("finding", None, [{"kind": "race"}]), "over": ("unsettled", "u", [])},
                             ("finding", None))):
        check = _Check(answers)
        monkeypatch.setattr(adapter, "check", check)
        out = tmp_path / f"out-{len(answers)}"
        r = adapter.run(_job("san.consan", src, out))
        assert (r.state, r.reason) == expect
        assert sorted(k for k, _ in check.seen) == ["elementwise", "over", "over", "softmax", "spec", "spec"]
        assert sorted(d for _, d in check.seen) == [f"k{i}" for i in range(6)]  # each kernel, its own program
        (run,) = out.iterdir()  # the run's own directory
        rows = json.loads((run / "kernels.json").read_text())["kernels"]
        assert [row["dir"] for row in rows] == [f"k{i}" for i in range(6)]
    assert src.read_text() == SHAPES  # the candidate itself is never edited


@needs_hipcc
@needs_aorta
@pytest.mark.parametrize("text,reason", [(BROKEN_TEMPLATE, "harness_unsupported: no instantiation compiles"),
                                         (NO_KERNEL, "harness_unsupported: no kernel definition in the source")],
                         ids=["broken-template", "no-kernel"])
def test_a_candidate_the_harness_cannot_launch_is_unsettled(tmp_path, monkeypatch, text, reason):
    adapter = AortaAsan(aorta_src=AORTA)
    check = _Check({})
    monkeypatch.setattr(adapter, "check", check)
    r = adapter.run(_job("san.asan", _write(tmp_path, "b", text), tmp_path / "out"))
    assert (r.state, r.reason) == ("unsettled", reason) and check.seen == []


def test_combine_is_a_finding_if_any_kernel_finds_and_clean_only_if_all_are():
    fd = [{"kind": "race"}]
    assert combine([("clean", None, []), ("clean", None, [])]) == ("clean", None, [])
    assert combine([("clean", None, []), ("unsettled", "why", [])]) == ("unsettled", "why", [])
    assert combine([("unsettled", "why", []), ("finding", None, fd)]) == ("finding", None, fd)
    assert combine([]) == ("unsettled", "harness_unsupported: no kernel", [])


@needs_hipcc
@needs_rj
def test_waitcheck_scans_the_instantiation_of_a_template_source(tmp_path):
    wc = StandinRjWaitcheck(RJ, aorta_src=AORTA, ld_library_path=RJ_LIB)
    for name, text, expect, instantiated in (("t", INDEXED, "clean", True), ("v", VECADD, "clean", False),
                                             ("b", BROKEN_TEMPLATE, "unsettled", False)):
        src = _write(tmp_path, name, text)
        obj = src.parent / "object.o"
        subprocess.run([HIPCC, "--offload-arch=gfx950", "-c", "-O1", str(src), "-o", str(obj)], check=True,
                       capture_output=True)
        out = src.parent / "wc"
        r = wc.run(_job("san.waitcheck", src, out, object=str(obj)))
        assert r.state == expect, (name, r.reason)
        (run,) = out.iterdir()  # the run's own directory
        assert (run / "instantiation.o").is_file() == instantiated
    gate_only = StandinRjWaitcheck(RJ, aorta_src=AORTA, ld_library_path=RJ_LIB)  # the gate's object alone: no code
    src = tmp_path / "t" / "source.hip"
    r = gate_only.run(LegJob(job_id="c#san.waitcheck#a1", leg="san.waitcheck", candidate_id="c", control_id=None,
                             source_sha256="0" * 64, task_sha256="0" * 64, attempt=1, worker_id="w",
                             inputs={"object": str(src.parent / "object.o")}, output_dir=str(tmp_path / "g"),
                             leg_settings={"target": "gfx950"}))
    assert r.state == "unsettled"


@pytest.mark.parametrize("spelling", ["#define GPU __global__\n", "#define KERNEL(name) __global__ void name\n",
                                      "__attribute__((global)) void hidden2(float* p) {}\n"])
def test_a_kernel_the_source_text_cannot_list_makes_discovery_unsupported(tmp_path, spelling):
    """A template spelled through a macro would be missing from the list, and the gate's object, which does not
    instantiate it, would be checked in its place."""
    src = tmp_path / "k.hip"
    src.write_text("#include <hip/hip_runtime.h>\n" + spelling + "__global__ void plain(float* p) { p[0] = 1.0f; }\n"
                   "template <typename T> GPU void hidden(T* p) { p[0] = T(1); }\n")

    def clang(argv, timeout_s):
        raise AssertionError("refused before clang is asked")

    with pytest.raises(HarnessUnsupported, match="does not list its kernels"):
        H.find_kernels(src, run=clang)
