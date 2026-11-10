"""The rollout's parse and compile checks (compile_check), on CPU: a stand-in `hipcc` script replays a
canned clang output, so the subprocess path, the scratch layout and the evidence store are exercised
without a compiler."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aorlta.ids import sha256_text
from aorlta.rollout import compile_check
from aorlta.rollout.compile_check import CompileUnavailable, EvidenceStore, FakeCompiler, HipccCompiler, parse_code
from aorlta.rollout.fake_engine import COMPILE_ERROR_MARKER

DATA = Path(__file__).resolve().parent / "data" / "clang"


def test_parse_takes_the_last_cpp_block():
    text = "intro\n```cpp\nint a;\n```\nthen\n```cpp \nint b;\n```\n```python\nx = 1\n```\n"
    assert parse_code(text, "off") == "int b;\n"
    assert parse_code("```c++\nint a;\n```", "off") is None  # the stand-in reads ```cpp only
    assert parse_code("```cpp\nint a;\n", "off") is None  # unclosed (a length-capped reply)
    assert parse_code("no code here", "off") is None


def test_with_thinking_on_only_the_text_after_think_is_parsed():
    reasoning = "<think>\nmaybe:\n```cpp\nint draft;\n```\n</think>\n\n"
    assert parse_code(reasoning + "```cpp\nint final;\n```", "on") == "int final;\n"
    assert parse_code(reasoning + "no code", "on") is None
    assert parse_code("<think>\n```cpp\nint a;\n```\n", "on") is None  # the think block never closed
    with pytest.raises(ValueError):
        parse_code("x", "maybe")


def test_the_fake_compiler_fails_exactly_the_marked_code():
    fc = FakeCompiler()
    ok = fc.check("__global__ void k() {}\n", "c0-0-t0")
    assert ok.ok and ok.rc == 0 and ok.errors == [] and ok.raw_stderr_sha256 == sha256_text("")
    code = "#include <hip/hip_runtime.h>\n  x[i] = x[i] * " + COMPILE_ERROR_MARKER + ";\n"
    bad = fc.check(code, "c0-0-t1")
    assert not bad.ok and bad.rc == 1
    assert bad.errors == [{"cls": "use of undeclared identifier <name>", "line": 2, "col": 17,
                           "token": COMPILE_ERROR_MARKER}]
    assert fc.calls == ["c0-0-t0", "c0-0-t1"]


def _fake_hipcc(tmp_path: Path, case: str, rc: int) -> Path:
    stderr = DATA / case / "stderr.txt"
    script = tmp_path / "hipcc"
    script.write_text(f"#!/bin/bash\n"
                      f"[ \"$*\" = \"--offload-arch=gfx950 -c -O1 -o kernel.o kernel.hip\" ] || exit 99\n"
                      f"[ -f kernel.hip ] || exit 98\n"
                      f"pwd > {tmp_path}/cwd.txt\n"
                      f"cat {stderr} >&2\nexit {rc}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_the_hipcc_check_runs_in_its_own_directory_and_seals_the_raw_stderr(tmp_path):
    hipcc = _fake_hipcc(tmp_path, "19-mixed-errors", 1)
    evidence = EvidenceStore(tmp_path / "evidence")
    cc = HipccCompiler(tmp_path / "work", hipcc=str(hipcc), evidence=evidence)
    source = (DATA / "19-mixed-errors" / "kernel.hip").read_text()
    res = cc.check(source, "c3-1-t0")
    raw = (DATA / "19-mixed-errors" / "stderr.txt").read_text()
    assert not res.ok and res.rc == 1 and len(res.errors) == 5
    assert res.errors[0] == {"cls": "use of undeclared identifier <name>", "line": 5, "col": 20, "token": "undefined_a"}
    assert res.raw_stderr_sha256 == sha256_text(raw)
    stored = tmp_path / "evidence" / res.raw_stderr_sha256[:2] / f"{res.raw_stderr_sha256}.txt"
    assert stored.read_text() == raw  # raw text only in the sealed store
    assert (tmp_path / "cwd.txt").read_text().strip() == str(tmp_path / "work" / "c3-1-t0")
    assert not (tmp_path / "work" / "c3-1-t0").exists()  # scratch removed


def test_a_clean_compile_is_ok(tmp_path):
    script = tmp_path / "hipcc"
    script.write_text("#!/bin/bash\nexit 0\n")
    script.chmod(0o755)
    res = HipccCompiler(tmp_path / "w", hipcc=str(script)).check("int x;\n", "t")
    assert res.ok and res.errors == []


def test_concurrent_compiles_with_the_same_stderr_all_seal_it(tmp_path, monkeypatch):
    """Every clean compile has the same (empty) stderr, so the compile pool's checks put one digest together."""
    script = tmp_path / "hipcc"
    script.write_text("#!/bin/bash\nexit 0\n")
    script.chmod(0o755)
    evidence = EvidenceStore(tmp_path / "evidence")
    cc = HipccCompiler(tmp_path / "w", hipcc=str(script), evidence=evidence)
    n = 4
    written = threading.Barrier(n, timeout=10)
    replace = os.replace

    def replace_once_all_have_written(src, dst):
        if Path(dst).is_relative_to(evidence.root):
            try:
                written.wait()
            except threading.BrokenBarrierError:  # a store that lets one writer through at a time never fills it
                pass
        replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_once_all_have_written)
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda i: cc.check("int x;\n", f"c0-{i}-t0"), range(n)))
    digest = sha256_text("")
    assert all(r.ok and r.raw_stderr_sha256 == digest for r in results)
    stored = [p.relative_to(evidence.root) for p in evidence.root.rglob("*") if p.is_file()]
    assert stored == [Path(digest[:2]) / f"{digest}.txt"]  # one sealed copy, no temporary left behind


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie has stopped; reaping it is up to its new parent
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


@pytest.mark.parametrize("on_term, graces", [('echo TERM > "${0%/*}/hipcc.term"; exit 143', 0), ("", 1)],
                         ids=["stops-on-sigterm", "ignores-sigterm"])
def test_a_compile_timeout_is_a_failed_compile_with_a_diagnostic(tmp_path, on_term, graces):
    """hipcc's own children (clang) are stopped with it: SIGTERM first, SIGKILL for what ignores it."""
    child_pid = tmp_path / "child.pid"
    script = tmp_path / "hipcc"
    script.write_text(f"#!/bin/bash\ntrap '{on_term}' TERM\nsleep 30 &\necho $! > {child_pid}\nwait\n")
    script.chmod(0o755)
    grace = 2.0
    for timeout_s in (0.5, 5.0):  # the longer one only for a loaded runner that had not started the child in time
        cc = HipccCompiler(tmp_path / "w", hipcc=str(script), timeout_s=timeout_s, kill_grace_s=grace)
        res = cc.check("int x;\n", "t")
        if child_pid.exists() and child_pid.read_text().strip():
            break
    else:
        pytest.fail("the stand-in hipcc never started its child")
    assert not res.ok and res.rc == -9
    assert res.errors == [{"cls": f"compile timed out after {timeout_s:g} s", "line": None, "col": None,
                           "token": None}]
    assert not _running(int(child_pid.read_text()))
    assert not (tmp_path / "w" / "t").exists()
    assert (tmp_path / "hipcc.term").exists() == bool(on_term)  # SIGTERM came first, and hipcc could handle it
    assert res.seconds < timeout_s + (graces + 1) * grace  # a grace period is waited out only for what ignores SIGTERM


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="a child subreaper is Linux-only")
def test_a_compile_timeout_reaps_what_hipcc_leaves_to_this_process(tmp_path):
    """As PID 1 of a container run without an init, the compiling process inherits the compilers of a hipcc that
    exits on SIGTERM; the stop reaps them too, rather than leaving zombies that hold it for a second grace period."""
    child_pid = tmp_path / "child.pid"
    script = tmp_path / "hipcc"
    script.write_text("#!/bin/bash\ntrap 'exit 143' TERM\n"
                      f"(trap '' TERM; echo $BASHPID > {child_pid}; exec sleep 30) &\nwait\n")
    script.chmod(0o755)
    grace = 2.0
    driver = """
import ctypes, os, sys
from aorlta.rollout.compile_check import HipccCompiler
if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    sys.exit(print("no subreaper"))
work, hipcc, child_pid, grace = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
for timeout_s in (0.5, 5.0):
    res = HipccCompiler(work, hipcc=hipcc, timeout_s=timeout_s, kill_grace_s=grace).check("int x;\\n", "t")
    if os.path.exists(child_pid):
        break
try:
    left = os.waitpid(-1, os.WNOHANG)  # (0, 0): a child still running; (pid, status): one left unreaped
except ChildProcessError:
    left = None
print(res.rc, res.seconds, timeout_s, left is None)
"""
    src = str(Path(compile_check.__file__).resolve().parents[2])
    out = subprocess.run([sys.executable, "-c", driver, str(tmp_path / "w"), str(script), str(child_pid), str(grace)],
                         capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": src})
    if out.stdout.strip() == "no subreaper":
        pytest.skip("PR_SET_CHILD_SUBREAPER refused")
    assert out.returncode == 0, out.stderr
    rc, seconds, timeout_s, nothing_left = out.stdout.split()
    assert child_pid.exists() and rc == "-9" and nothing_left == "True"
    assert float(seconds) < float(timeout_s) + 2 * grace  # one grace period for the child that ignores SIGTERM


def test_a_missing_compiler_is_infrastructure_not_a_compile_error(tmp_path):
    with pytest.raises(CompileUnavailable):
        HipccCompiler(tmp_path, hipcc=str(tmp_path / "no-such-hipcc"))
    with pytest.raises(ValueError):
        FakeCompiler().check("x", "../escape")
