"""The CPU stand-in adapters (design Section 6.2): standin.parse, standin.hipcc and
standin.rj_waitcheck, and aorta's verdict mapping. hipcc and rj_waitcheck are the real tools, run on CPU
with an explicit offload arch (nothing here opens a GPU); fake tool scripts pin the verdicts a real
object cannot produce on demand. And the standin-static profile end to end."""

from __future__ import annotations

import dataclasses
import os
import shutil
import stat
from pathlib import Path

import pytest

from aorlta.config import RewardSection
from aorlta.ids import sha256_text
from aorlta.reward.adapters import aorta_import
from aorlta.reward.adapters.base import map_aorta_check
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
])
def test_a_compiler_that_does_not_settle_is_unsettled(tmp_path, body, reason):
    fake = script(tmp_path / "hipcc", body)
    src = tmp_path / "k.hip"
    src.write_text("// k\n")
    r = StandinHipcc(fake, timeout_s=1.0).run(job("gate.compile", tmp_path, settings=COMPILE, inputs={"source": str(src)}))
    assert (r.state, r.reason) == ("unsettled", reason)


def test_hipcc_close_kills_a_running_compile(tmp_path):
    import threading
    import time
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


@needs_rj
def test_a_waitcheck_timeout_is_unsettled(tmp_path):
    fake = script(tmp_path / "rj_waitcheck", "sleep 30\n")
    obj = tmp_path / "o"
    obj.write_bytes(b"\x7fELF")
    r = StandinRjWaitcheck(fake, aorta_src=AORTA, timeout_s=1.0).run(
        job("san.waitcheck", tmp_path, settings={"target": "gfx950"}, inputs={"object": str(obj)}))
    assert r.state == "unsettled"


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
