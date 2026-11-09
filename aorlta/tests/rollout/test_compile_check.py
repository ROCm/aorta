"""The rollout's parse and compile checks (compile_check), on CPU: a stand-in `hipcc` script replays a
canned clang output, so the subprocess path, the scratch layout and the evidence store are exercised
without a compiler."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from aorlta.ids import sha256_text
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


def test_a_compile_timeout_is_a_failed_compile_with_a_diagnostic(tmp_path):
    script = tmp_path / "hipcc"
    script.write_text("#!/bin/bash\nsleep 5\n")
    script.chmod(0o755)
    res = HipccCompiler(tmp_path / "w", hipcc=str(script), timeout_s=0.3).check("int x;\n", "t")
    assert not res.ok and res.rc == -9
    assert res.errors == [{"cls": "compile timed out after 0.3 s", "line": None, "col": None, "token": None}]


def test_a_missing_compiler_is_infrastructure_not_a_compile_error(tmp_path):
    with pytest.raises(CompileUnavailable):
        HipccCompiler(tmp_path, hipcc=str(tmp_path / "no-such-hipcc"))
    with pytest.raises(ValueError):
        FakeCompiler().check("x", "../escape")
