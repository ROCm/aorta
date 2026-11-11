"""The CPU stand-in adapters (design Section 6.2): standin.parse, standin.hipcc and
standin.rj_waitcheck, and aorta's verdict mapping. hipcc and rj_waitcheck are the real tools, run on CPU
with an explicit offload arch (nothing here opens a GPU); fake tool scripts pin the verdicts a real
object cannot produce on demand. And the standin-static profile end to end."""

from __future__ import annotations

import dataclasses
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aorlta.config import RewardSection
from aorlta.ids import sha256_text
from aorlta.reward.adapters import aorta_import, standin_harness, standin_hipcc, toolchain
from aorlta.reward.adapters.base import ProcessRunner, map_aorta_check
from aorlta.reward.adapters.standin_aorta import AortaAsan, AortaConsan
from aorlta.reward.adapters.standin_hipcc import StandinHipcc, error_classes
from aorlta.reward.adapters.standin_parse import StandinParse, last_cpp_block
from aorlta.reward.adapters.standin_rj_waitcheck import StandinRjWaitcheck
from aorlta.reward.clock import Clock
from aorlta.reward.factories import standin_factories
from aorlta.reward.profiles import load_profile
from aorlta.reward.rewardfn.standin_v11 import StandinV11
from aorlta.reward.service import RewardService
from aorlta.schemas import LegJob
from reward_helpers import cand, wait_all

HIPCC = "/opt/rocm/bin/hipcc"
# Site paths, each optional (a test that needs one skips without it): aorta at d22f6116, rocjitsu's rj_waitcheck and
# the libstdc++ it needs.
AORTA_ROOT = Path(os.environ.get("AORLTA_AORTA_ROOT", "/opt/aorta"))
RJ = os.environ.get("AORLTA_RJ_WAITCHECK", "/opt/rocjitsu/bin/rj_waitcheck")
SHIM = os.environ.get("AORLTA_LIBSTDCXX_SHIM", "/opt/libstdcxx-shim")
AORTA = str(AORTA_ROOT / "src")
FIXTURES = AORTA_ROOT / "recipes" / "sanitizers" / "fixtures"
H = sha256_text("x")
needs_hipcc = pytest.mark.skipif(not Path(HIPCC).exists(), reason="no hipcc")
needs_fixtures = pytest.mark.skipif(not FIXTURES.is_dir(), reason="no aorta checkout (AORLTA_AORTA_ROOT)")
needs_rj = pytest.mark.skipif(not (Path(RJ).exists() and Path(AORTA).exists()),
                              reason="no rj_waitcheck or aorta (AORLTA_RJ_WAITCHECK, AORLTA_AORTA_ROOT)")


def job(leg, tmp_path, **kw):
    out = tmp_path / leg
    return LegJob(f"j#{leg}", leg, "c", None, H, H, 1, "w", kw.pop("settings", {}), 0.0, None, kw.pop("inputs", {}),
                  str(out))


def script(path: Path, body: str) -> str:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


# --- standin.parse ----------------------------------------------------------------------------------


def test_parse_takes_the_last_cpp_block(tmp_path):
    text = "a\n```cpp\nfirst\n```\nthen\n```cpp\nsecond\n```\n```python\nno\n```"
    assert last_cpp_block(text) == "second\n"
    (tmp_path / "response.txt").write_text(text)
    r = StandinParse().run(job("gate.parse", tmp_path, inputs={"response": str(tmp_path / "response.txt")}))
    assert r.state == "clean" and (tmp_path / "gate.parse" / "source.hip").read_text() == "second\n"


@pytest.mark.parametrize("text", ["no code at all", "```cpp\nnever closed", "```c\nint x;\n```", "```\nbare\n```"])
def test_no_cpp_block_fails_the_parse_gate(tmp_path, text):
    (tmp_path / "response.txt").write_text(text)
    r = StandinParse().run(job("gate.parse", tmp_path, inputs={"response": str(tmp_path / "response.txt")}))
    assert (r.state, r.reason) == ("fail", "parse")


def test_parse_without_its_input_is_unsettled_not_a_failure(tmp_path):
    r = StandinParse().run(job("gate.parse", tmp_path, inputs={"response": str(tmp_path / "missing.txt")}))
    assert (r.state, r.reason) == ("unsettled", "adapter_error")


def test_identical_answers_parsed_at_once_both_publish_the_source(tmp_path, monkeypatch):
    """Candidates with identical answers share the content-addressed parse directory, so two parse workers can write
    source.hip at once; here both have written before either renames."""
    (tmp_path / "response.txt").write_text("```cpp\n__global__ void k(float* p) { p[0] = 1.0f; }\n```")
    replace, both = os.replace, threading.Barrier(2, timeout=10)

    def replace_together(src, dst):
        both.wait()
        replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_together)
    results = []
    jobs = [job("gate.parse", tmp_path, inputs={"response": str(tmp_path / "response.txt")}) for _ in range(2)]
    threads = [threading.Thread(target=lambda j=j: results.append(StandinParse().run(j))) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert [(r.state, r.reason) for r in results] == [("clean", None)] * 2
    assert sorted(p.name for p in (tmp_path / "gate.parse").iterdir()) == ["source.hip"]


# --- standin.hipcc ----------------------------------------------------------------------------------

COMPILE = {"offload_arch": "gfx950", "flags": ["-c", "-O1"]}


@needs_hipcc
@needs_fixtures
def test_hipcc_compiles_a_kernel_on_cpu(tmp_path):
    src = FIXTURES / "kernels" / "tiny_vecadd.hip"
    r = StandinHipcc(HIPCC).run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
    assert r.state == "clean" and (tmp_path / "gate.compile" / "object.o").stat().st_size > 0


@needs_hipcc
def test_a_compile_error_is_the_candidates_with_classes_not_text(tmp_path):
    src = tmp_path / "bad.hip"
    src.write_text("__global__ void k(int* p) { p[0] = secret_name_xyz; }\n")
    r = StandinHipcc(HIPCC).run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
    assert (r.state, r.reason) == ("fail", "compile_error")
    assert r.findings[0]["classes"] == ["use of undeclared identifier <name>"]
    assert "secret_name_xyz" not in repr(r.findings)
    assert not (tmp_path / "gate.compile" / "object.o").exists()


def test_error_classes_drop_quoted_names():
    stderr = "x.hip:3:7: error: use of undeclared identifier 'tid'\nx.hip:4:1: warning: w\n"
    assert error_classes(stderr) == ["use of undeclared identifier <name>"]
    stderr = "x.hip:2:3: error: static_cast from 'size_t' (aka 'unsigned long') to 'float *' is not allowed\n"
    assert error_classes(stderr) == ["static_cast from <name> to <name> is not allowed"]


CANARY_STDERR = ("x.hip:2:2: error: private_source_canary\n"
                 "x.hip:3:15: error: static assertion failed due to requirement 'sizeof(int) == 3': secret_assert_text\n"
                 "x.hip:4:38: error: use of undeclared identifier 'secret_identifier_xyz'\n")


def test_an_error_message_that_is_not_a_known_class_is_reported_by_its_hash_never_its_text():
    classes = error_classes(CANARY_STDERR)
    assert classes[2] == "use of undeclared identifier <name>"
    assert all(c.startswith("sha256:") and len(c) == 19 for c in classes[:2])
    assert not any(s in repr(classes) for s in ("private_source_canary", "secret_assert_text", "secret_identifier"))
    assert error_classes("y.hip:9:1: error: private_source_canary\n") == classes[:1]  # the same message, the same code


@needs_hipcc
def test_an_error_directive_in_the_candidate_never_reaches_the_findings(tmp_path):
    src = tmp_path / "canary.hip"
    src.write_text('#include <hip/hip_runtime.h>\n#error private_source_canary\n'
                   'static_assert(sizeof(int) == 3, "secret_assert_text");\n')
    r = StandinHipcc(HIPCC).run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
    assert (r.state, r.reason) == ("fail", "compile_error") and r.findings[0]["errors"] == 2
    assert not any(s in repr(r.findings) for s in ("private_source_canary", "secret_assert_text"))


def test_hipcc_never_runs_without_an_explicit_offload_arch(tmp_path):
    marker = tmp_path / "ran"
    fake = script(tmp_path / "hipcc", f"touch {marker}\nexit 0\n")
    r = StandinHipcc(fake).run(job("gate.compile", tmp_path, settings={"flags": ["-c"]}, inputs={"source": "x.hip"}))
    assert r.state == "unsettled" and not marker.exists()


@needs_hipcc
def test_a_missing_header_is_the_candidates_compile_error_not_the_checkers(tmp_path):
    # In one stand-in run 39 of 256 candidates included a CUDA header, and hipcc's "fatal error" went unsettled.
    src = tmp_path / "cuda.hip"
    src.write_text("#include <cuda_runtime.h>\n__global__ void k(float* p) { p[0] = 1.0f; }\n")
    r = StandinHipcc(HIPCC).run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
    assert (r.state, r.reason) == ("fail", "compile_error")
    assert r.findings[0]["classes"] == ["<name> file not found"]


@pytest.mark.parametrize("body, reason", [
    ("sleep 30\n", "timeout"),                      # its own subprocess deadline
    ("exit 139\n", "adapter_error"),                # crashed, printed no error
    ("echo 'x.hip:1:1: warning: w' >&2; exit 1\n", "adapter_error"),
    ("echo \"x.hip:1:1: error: use of undeclared identifier 'y'\" >&2; kill -KILL $$\n", "adapter_error"),  # a signal
])
def test_a_compiler_that_does_not_settle_is_unsettled(tmp_path, body, reason):
    fake = script(tmp_path / "hipcc", body)
    src = tmp_path / "k.hip"
    src.write_text("// k\n")
    r = StandinHipcc(fake, timeout_s=1.0).run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
    assert (r.state, r.reason) == ("unsettled", reason)


def test_hipcc_close_kills_a_running_compile(tmp_path):
    fake = script(tmp_path / "hipcc", "sleep 60\n")
    a = StandinHipcc(fake, timeout_s=50.0)
    out = {}
    src = tmp_path / "k.hip"
    src.write_text("// k\n")
    t = threading.Thread(target=lambda: out.setdefault("r", a.run(job("gate.compile", tmp_path, settings=COMPILE,
                                                                       inputs={"source": str(src)}))))
    t.start()
    time.sleep(0.5)
    a.close()
    t.join(timeout=10)
    assert not t.is_alive() and out["r"].state == "unsettled" and out["r"].reason == "worker_died"


def test_a_close_before_the_subprocess_is_published_still_kills_it(monkeypatch):
    runner, popen = ProcessRunner(), subprocess.Popen

    def popen_then_close(*args, **kw):  # close() lands after run() checked for it, before it published the process
        proc = popen(*args, **kw)
        runner.close()
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen_then_close)
    r = runner.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=8.0)
    assert (r.killed, r.timed_out, r.returncode) == (True, False, -9)  # killed at once, not at its timeout


def _run_together(runs):
    results = [None] * len(runs)
    threads = [threading.Thread(target=lambda i=i, f=f: results.__setitem__(i, f())) for i, f in enumerate(runs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return results


def test_identical_compiles_at_once_each_rename_their_own_object(tmp_path):
    """Candidates with identical responses and settings share the compile's output directory, and their first attempts
    in one process have one PID and attempt number. Here both compilers finish writing before either object is
    renamed."""
    arrived = tmp_path / "arrived"
    arrived.mkdir()
    fake = script(tmp_path / "hipcc", f"""while [ $# -gt 1 ]; do [ "$1" = -o ] && out=$2; shift; done
echo obj > "$out"
touch {arrived}/$$
for _ in $(seq 200); do n=({arrived}/*); [ ${{#n[@]}} -ge 2 ] && exit 0; sleep 0.05; done
""")
    src = tmp_path / "k.hip"
    src.write_text("// k\n")
    a = StandinHipcc(fake)
    results = _run_together([lambda: a.run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
                             for _ in range(2)])
    assert [(r.state, r.reason) for r in results] == [("clean", None)] * 2
    assert (tmp_path / "gate.compile" / "object.o").read_text() == "obj\n"


def test_identical_build_facts_at_once_each_read_their_own_work_directory(tmp_path, monkeypatch):
    """Two identical compiles one after the other, whose build facts are then read at once: one fact's cleanup must
    not remove the other's work directory."""
    fake = script(tmp_path / "hipcc", 'while [ $# -gt 1 ]; do [ "$1" = -o ] && out=$2; shift; done\necho obj > "$out"\n')
    src = tmp_path / "k.hip"
    src.write_text("__global__ void k(float* p) { p[0] = 1.0f; }\n")
    first_in, both, seen = threading.Event(), threading.Barrier(2, timeout=10), []

    def plan_side_by_side(source, *, workdir, **kw):
        (Path(workdir) / str(threading.get_ident())).write_text("x")
        first_in.set()
        both.wait()
        seen.append(sorted(p.name for p in Path(workdir).iterdir()))
        both.wait()
        return []

    monkeypatch.setattr(standin_hipcc, "plan", plan_side_by_side)
    a = StandinHipcc(fake)

    def run():
        return a.run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))

    def second():
        first_in.wait(timeout=30)
        return run()

    results = _run_together([run, second])
    assert [len(names) for names in seen] == [1, 1]  # each saw only its own file
    assert [(r.state, r.reason) for r in results] == [("clean", None)] * 2
    assert sorted(p.name for p in (tmp_path / "gate.compile").iterdir()) == ["launched_lds.json", "object.o"]


# --- standin.rj_waitcheck ---------------------------------------------------------------------------


@needs_hipcc
@needs_rj
def test_waitcheck_scans_a_real_object(tmp_path):
    src = FIXTURES / "kernels" / "tiny_vecadd.hip"
    StandinHipcc(HIPCC).run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
    obj = tmp_path / "gate.compile" / "object.o"
    a = StandinRjWaitcheck(RJ, aorta_src=AORTA, ld_library_path=SHIM)
    r = a.run(job("san.waitcheck", tmp_path, settings={"target": "gfx950"}, inputs={"object": str(obj)}))
    assert r.state == "clean", r


@needs_rj
@pytest.mark.parametrize("output, rc, state", [
    ("rj_waitcheck: o:gfx950[0]: instructions=9 memory-events=2 diagnostics=1\nmissing s_waitcnt vmcnt(0) before use", 4, "finding"),
    ("rj_waitcheck: o:gfx950[0]: instructions=9 memory-events=2 diagnostics=0", 0, "clean"),
    ("o: failed to parse input executable or code object", 2, "unsettled"),
    ("rj_waitcheck: o:gfx950[0]: instructions=9 memory-events=2 diagnostics=1", 4, "unsettled"),  # hazard exit, no diagnostic
    ("garbage", 0, "unsettled"),                                                                  # no analysis summary
])
def test_waitcheck_verdicts_map_as_the_design_says(tmp_path, output, rc, state):
    fake = script(tmp_path / "rj_waitcheck", f"cat <<'OUT'\n{output}\nOUT\nexit {rc}\n")
    obj = tmp_path / "o"
    obj.write_bytes(b"\x7fELF fake object")
    r = StandinRjWaitcheck(fake, aorta_src=AORTA).run(
        job("san.waitcheck", tmp_path, settings={"target": "gfx950"}, inputs={"object": str(obj)}))
    assert r.state == state, r
    if state == "finding":
        assert r.findings and all("message" not in f for f in r.findings)


def _fake_aorta_waitcheck(monkeypatch, seen: list) -> None:
    """aorta's models and waitcheck backend, answering PASS for whatever object they are given (recorded in `seen`
    with the output directory)."""
    models = SimpleNamespace(KernelIdentity=lambda **k: k, KernelWorklist=lambda **k: k, KernelObservation=lambda **k: k,
                             SelectionRequirement=SimpleNamespace(TOP_DISPATCH_COUNT=1))
    passed = SimpleNamespace(state=SimpleNamespace(value="ran"), verdict=SimpleNamespace(value="pass"), findings=(),
                             reason=None)

    def run_waitcheck(worklist, *, output_dir, **kw):
        seen.append((worklist["kernels"][0]["identity"]["code_object"], output_dir))
        return passed

    mods = {"models": models, "waitcheck": SimpleNamespace(run_waitcheck=run_waitcheck)}
    monkeypatch.setattr(aorta_import, "load", lambda name, src: (mods.get(name), None))


def test_two_waitcheck_runs_of_one_leg_directory_each_have_their_own(tmp_path, monkeypatch):
    seen: list = []
    _fake_aorta_waitcheck(monkeypatch, seen)
    obj = tmp_path / "object.o"
    obj.write_bytes(b"\x7fELF")
    a = StandinRjWaitcheck("/nonexistent/rj_waitcheck")
    results = [a.run(job("san.waitcheck", tmp_path, settings={"target": "gfx950"}, inputs={"object": str(obj)}))
               for _ in range(2)]
    assert [r.state for r in results] == ["clean", "clean"]
    dirs = [Path(d) for _, d in seen]
    assert len(set(dirs)) == 2 and all(d.parent == tmp_path / "san.waitcheck" for d in dirs)


def _waitcheck_of(tmp_path, text: str, hipcc: str):
    src = tmp_path / "source.hip"
    src.write_text(text)
    obj = tmp_path / "object.o"
    obj.write_bytes(b"\x7fELF")
    return obj, StandinRjWaitcheck("/nonexistent/rj_waitcheck", hipcc=hipcc).run(
        job("san.waitcheck", tmp_path, settings={"target": "gfx950"}, inputs={"object": str(obj), "source": str(src)}))


MIXED = "template <typename T> __global__ void scale(T* p) { p[0] *= 2; }\n__global__ void zero(float* p) { p[0] = 0; }\n"


@pytest.mark.parametrize("hipcc_body", [None, "sleep 30\n"], ids=["cannot_start", "times_out"])
def test_a_waitcheck_whose_kernels_cannot_be_read_is_unsettled_not_the_gate_objects_verdict(tmp_path, monkeypatch,
                                                                                            hipcc_body):
    """The gate's object has no code for an uninstantiated template, so when kernel discovery fails the scan of that
    object alone says nothing about the source's templates."""
    seen: list = []
    _fake_aorta_waitcheck(monkeypatch, seen)
    monkeypatch.setattr(standin_harness, "AST_TIMEOUT_S", 0.5)
    hipcc = "/nonexistent/hipcc" if hipcc_body is None else script(tmp_path / "hipcc", hipcc_body)
    _, r = _waitcheck_of(tmp_path, MIXED, hipcc)
    assert (r.state, r.reason) == ("unsettled", "harness_unsupported: clang could not read the source (exit None)")
    assert seen == []  # nothing was scanned


def test_a_waitcheck_of_a_source_that_defines_no_kernel_scans_the_gates_object(tmp_path, monkeypatch):
    seen: list = []
    _fake_aorta_waitcheck(monkeypatch, seen)
    obj, r = _waitcheck_of(tmp_path, "__device__ float twice(float x) { return 2 * x; }\n", "/nonexistent/hipcc")
    assert r.state == "clean" and [o for o, _ in seen] == [str(obj)]


@pytest.mark.parametrize("make", [
    lambda hipcc: StandinRjWaitcheck("/nonexistent/rj_waitcheck", hipcc=hipcc),  # a template kernel's instantiation
    lambda hipcc: AortaConsan(hook="/nonexistent", aorta_src="/nonexistent", hipcc=hipcc),
    lambda hipcc: AortaAsan(aorta_src="/nonexistent", hipcc=hipcc),
], ids=["rj_waitcheck", "aorta_consan", "aorta_asan"])
def test_an_adapter_that_builds_with_its_own_compiler_names_it_in_its_identity(tmp_path, make):
    a, b = script(tmp_path / "hipcc-a", "exit 0\n"), script(tmp_path / "hipcc-b", "exit 1\n")
    first, again, other = (make(h).version for h in (a, a, b))
    assert first == again != other


@needs_rj
def test_a_waitcheck_timeout_is_unsettled(tmp_path):
    fake = script(tmp_path / "rj_waitcheck", "sleep 30\n")
    obj = tmp_path / "o"
    obj.write_bytes(b"\x7fELF")
    r = StandinRjWaitcheck(fake, aorta_src=AORTA, timeout_s=1.0).run(
        job("san.waitcheck", tmp_path, settings={"target": "gfx950"}, inputs={"object": str(obj)}))
    assert r.state == "unsettled"
@pytest.mark.parametrize("make", [
    lambda hipcc: StandinHipcc(hipcc),
    lambda hipcc: StandinRjWaitcheck("/nonexistent/rj_waitcheck", hipcc=hipcc),
    lambda hipcc: AortaConsan(hook="/nonexistent", aorta_src="/nonexistent", hipcc=hipcc),
    lambda hipcc: AortaAsan(aorta_src="/nonexistent", hipcc=hipcc),
], ids=["hipcc", "rj_waitcheck", "aorta_consan", "aorta_asan"])
def test_the_identity_names_the_compiler_hipcc_runs_not_only_the_driver(tmp_path, monkeypatch, make):
    """The same hipcc runs another clang under HIP_CLANG_PATH, or adds flags from HIPCC_COMPILE_FLAGS_APPEND: either
    is another compiler, and a cache entry of the first must not be reused for it."""
    for k in toolchain.HIPCC_ENV:
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "rocm" / "bin").mkdir(parents=True)
    hipcc = script(tmp_path / "rocm" / "bin" / "hipcc", "exit 0\n")
    llvm, other = tmp_path / "rocm" / "lib" / "llvm" / "bin", tmp_path / "other-llvm"
    for d, body in ((llvm, b"clang one"), (other, b"clang two")):
        d.mkdir(parents=True)
        (d / "clang").write_bytes(body)
    (llvm / "clang++.cfg").write_text("")
    first = make(hipcc).version
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")  # a worker's GPU is not the compiler
    assert make(hipcc).version == first
    monkeypatch.setenv("HIP_CLANG_PATH", str(other))
    assert make(hipcc).version != first
    monkeypatch.delenv("HIP_CLANG_PATH")
    monkeypatch.setenv("HIPCC_COMPILE_FLAGS_APPEND", "-O3")
    assert make(hipcc).version != first
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND")
    (llvm / "clang").write_bytes(b"clang one, rebuilt")
    assert make(hipcc).version != first
    (llvm / "clang").write_bytes(b"clang one")
    (llvm / "clang++.cfg").write_text("-fno-gpu-rdc\n")
    assert make(hipcc).version != first




# --- aorta's verdicts -------------------------------------------------------------------------------


@pytest.mark.skipif(not Path(AORTA).exists(), reason="no aorta checkout (AORLTA_AORTA_ROOT)")
@pytest.mark.parametrize("state, verdict, with_finding, expected", [
    ("ran", "pass", False, "clean"),
    ("ran", "fail", True, "finding"),
    ("ran", "warn", True, "finding"),
    ("error", "error", False, "unsettled"),
    ("error", "fail", True, "unsettled"),   # a finding from a run that did not finish cleanly is not trusted
    ("not_checked", "not_checked", False, "unsettled"),
    ("timed_out", "error", False, "unsettled"),
])
def test_aorta_verdicts_map_to_leg_states(state, verdict, with_finding, expected):
    models, _ = aorta_import.load("models", AORTA)
    findings = (models.Finding("asan", models.FindingSeverity.MEMORY_ERROR, "heap-buffer-overflow", "m"),) if with_finding else ()
    check = models.CheckResult("asan", models.ExecutionState(state), models.Verdict(verdict),
                               reason=None if state != "not_checked" else "asan_node_health_not_verified: x",
                               findings=findings)
    assert map_aorta_check(check)[0] == expected


CANARY_REASON = "consan_output_parse_error: ConSan field reader must be an integer, got 'program_output_canary'"


def test_an_unsettled_reason_keeps_aortas_code_and_only_a_hash_of_its_text():
    def unsettled(reason):
        return map_aorta_check(SimpleNamespace(state=SimpleNamespace(value="error"),
                                               verdict=SimpleNamespace(value="error"), findings=(), reason=reason))
    state, reason = unsettled(CANARY_REASON)
    assert state == "unsettled" and "program_output_canary" not in reason
    assert reason.startswith("checker_not_settled:error/error: consan_output_parse_error: sha256:")
    assert reason == unsettled(CANARY_REASON)[1] != unsettled(CANARY_REASON + "2")[1]
    assert unsettled("combined_hook_timeout")[1] == "checker_not_settled:error/error: combined_hook_timeout"


@pytest.mark.skipif(not Path(AORTA).exists(), reason="no aorta checkout (AORLTA_AORTA_ROOT)")
def test_consans_parse_error_on_what_the_program_printed_never_reaches_the_leg_reason():
    consan, _ = aorta_import.load("consan", AORTA)
    execution, _ = aorta_import.load("execution", AORTA)
    printed = "[rocjitsu-dbi-hooks] ConSan auto report reader=program_output_canary\n"
    _, check = consan.evaluate_consan_output(execution.ProcessResult(
        argv=("harness",), returncode=0, stdout=printed, stderr="", timed_out=False, launch_error=None), strict=True)
    assert check.reason == CANARY_REASON
    state, reason = map_aorta_check(check)
    assert state == "unsettled" and "program_output_canary" not in reason


# --- standin-static end to end ----------------------------------------------------------------------


@needs_hipcc
@needs_rj
def test_standin_static_end_to_end_on_cpu(tmp_path):
    clock = Clock()
    profile = load_profile("standin-static")
    wc = {**profile.adapters["standin.rj_waitcheck"], "binary": RJ, "ld_library_path": SHIM, "aorta_src": AORTA}
    profile = dataclasses.replace(profile, adapters={**profile.adapters, "standin.rj_waitcheck": wc})
    svc = RewardService(RewardSection(profile="standin-static", reward_fn="standin.v11"), profile, StandinV11(),
                        standin_factories(profile, clock), cache_dir=tmp_path / "cache", work_dir=tmp_path / "work",
                        clock=clock)
    try:
        good = (FIXTURES / "kernels" / "tiny_vecadd.hip").read_text()
        cands = {
            "clean": cand(0, text=f"x\n```cpp\n{good}```", profile="standin-static"),
            "compile_error": cand(1, text="```cpp\n__global__ void k(int* p) { p[0] = nope; }\n```", profile="standin-static"),
            "no_block": cand(2, text="I could not fix it.", profile="standin-static"),
        }
        res = dict(zip(cands, wait_all([svc.submit(c) for c in cands.values()])))
        c = cands["compile_error"]
        assert res["compile_error"].reward == -0.9 + -0.05 * (c.response_tokens / c.max_new_tokens)
        assert res["no_block"].reward == -1.0 + -0.05 * (1024 / 4096)
        # ConSan and ASan never ran here, and a leg that never ran cannot count: unscored, never clean
        assert res["clean"].reward is None and res["clean"].reason == "sanitizer_not_settled"
        assert res["clean"].legs["san.waitcheck"].state == "clean"
        assert res["clean"].legs["san.consan"].state == "not_run"
    finally:
        svc.close()
