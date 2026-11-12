"""The rollout's parse and compile checks (design Section 6.1, chain state machine).

- Parse (the stand-in parse gate): the last ```cpp block of the reply. With thinking on, only
  the text after the last `</think>` is searched; a reply that never closes its think block has no
  code block.
- Compile (`standin.hipcc`): `hipcc --offload-arch=gfx950 -c -O1` on the parsed source, each in its own
  directory as `kernel.hip`, so clang's diagnostics name only `kernel.hip` and never a scratch path.
  The raw stderr is kept only in the run's sealed evidence store, by sha256; the result carries the
  structured errors and that hash. The errors are structured by the adapter's `structure` hook:
  the stand-in's (feedback.compile_errors) by default; another feedback format can bring its own.
- `fake.compile`: a deterministic compiler for CPU tests. Code holding the fake engine's undeclared
  identifier fails with a clang-format diagnostic at each occurrence.

The rollout compile only decides whether a chain gets a repair turn. The reward service compiles
again with its own settings (`reward.reuse_rollout_compile` is false), so nothing here can make a
candidate clean.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..atomic import atomic_write_bytes
from ..ids import sha256_text
from ..schemas import CompileResult
from .fake_engine import COMPILE_ERROR_MARKER
from .feedback import compile_errors

SOURCE_NAME = "kernel.hip"
_CPP_BLOCK_RE = re.compile(r"```cpp[ \t]*\r?\n(.*?)```", re.S)
_CPP_OPEN_RE = re.compile(r"```cpp[ \t]*(?:\r?\n|\Z)")
THINK_CLOSE = "</think>"
_TAG_RE = re.compile(r"[A-Za-z0-9_.-]+")


Structure = Callable[[str, str, str], list]  # (raw stderr, compiled source, its file name) -> structured errors


class CompileUnavailable(RuntimeError):
    """The compiler cannot be run at all (missing binary, unwritable directory): an infrastructure
    failure that stops the run, never a candidate's compile error."""


@dataclass(frozen=True)
class ParseOutcome:
    """What a parse adapter makes of one reply.

    - "edits": `code` is the candidate to compile;
    - "no_change": the reply leaves the kernel as shown (an explicit no-change answer, or edits that change nothing); the
      chain ends without a compile;
    - "rejected": nothing to compile. `rejection` is (grading message, reason) when the parse's format gives a
      rejected answer a repair turn, None when it ends the chain (the stand-in's)."""

    kind: Literal["edits", "no_change", "rejected"]
    code: str | None = None
    rejection: tuple[str, str] | None = None


class Compiler(Protocol):
    name: str

    def check(self, code: str, tag: str) -> CompileResult:
        ...


def parse_code(text: str, thinking: str) -> str | None:
    """Stand-in parse: the last ```cpp block, searched after the last `</think>` when thinking is on. A last block
    that never closes (a length-capped reply) is no code, also after a complete one."""
    if thinking not in ("off", "on"):
        raise ValueError(f"thinking must be 'off' or 'on', got {thinking!r}")
    if thinking == "on":
        cut = text.rfind(THINK_CLOSE)
        if cut < 0:
            return None
        text = text[cut + len(THINK_CLOSE):]
    blocks = list(_CPP_BLOCK_RE.finditer(text))
    if _CPP_OPEN_RE.search(text, blocks[-1].end() if blocks else 0):
        return None
    return blocks[-1].group(1) if blocks else None


def _check_tag(tag: str) -> str:
    if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag) or tag in (".", ".."):
        raise ValueError(f"compile tag must match {_TAG_RE.pattern}, got {tag!r}")
    return tag


class EvidenceStore:
    """Content-addressed raw text, inside the run's sealed area (Section 6.10 item 3)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._sealed = False  # made owner-only at the first write, which comes after the run plan

    def put(self, text: str) -> str:
        if not self._sealed:
            self.root.mkdir(parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)  # an existing root included: what it holds quotes the sources
            self._sealed = True
        digest = sha256_text(text)
        path = self.root / digest[:2] / f"{digest}.txt"
        if not path.exists():
            path.parent.mkdir(mode=0o700, exist_ok=True)
            atomic_write_bytes(path, text.encode("utf-8"))
            os.chmod(path, 0o600)  # the root already keeps others out; a copy of the tree keeps the files owner-only
        return digest


def _run_in_own_group(argv: list[str], cwd: Path, timeout_s: float, grace_s: float) -> tuple[int, str]:
    """(returncode, stderr) of `argv`, run in a session of its own. When it outlives `timeout_s` (or the wait is
    interrupted), its whole process group, the compilers hipcc started included, is stopped before the exception
    leaves this function."""
    with subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                          start_new_session=True) as proc:
        try:
            stderr = proc.communicate(timeout=timeout_s)[1]
        except BaseException:
            _stop_group(proc, grace_s)
            raise
    return proc.returncode, stderr


def _stop_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the process group `proc` leads, SIGKILL it if a member outlasts `grace_s`, then give the members up to
    `grace_s` more to be gone; `proc`, and any member left to this process, is reaped."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        deadline = time.monotonic() + grace_s
        while (left := _group_left(proc)) and time.monotonic() < deadline:
            time.sleep(0.02)
        if not left:
            break
    proc.wait()


def _group_left(proc: subprocess.Popen) -> bool:
    if proc.poll() is not None:  # hipcc is reaped: its zombie is not a member, and what is left is what it started
        try:  # left to this process to reap when it is the reaper (PID 1 of a container run without an init)
            while os.waitpid(-proc.pid, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
    try:
        os.killpg(proc.pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class HipccCompiler:
    name = "standin.hipcc"

    def __init__(
        self,
        workdir: str | Path,
        *,
        hipcc: str | None = None,
        offload_arch: str = "gfx950",
        opt: str = "-O1",
        timeout_s: float = 120.0,
        kill_grace_s: float = 5.0,
        evidence: EvidenceStore | None = None,
        structure: Structure = compile_errors,
    ):
        self.workdir = Path(workdir)
        if hipcc is None:
            hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
        elif os.sep not in hipcc:
            hipcc = shutil.which(hipcc) or hipcc
        self.hipcc = os.path.abspath(hipcc)  # check() runs the compiler from each check's own directory
        self.offload_arch = offload_arch
        self.opt = opt
        self.timeout_s = timeout_s
        self.kill_grace_s = kill_grace_s
        self.evidence = evidence
        self.structure = structure
        if not os.access(self.hipcc, os.X_OK):
            raise CompileUnavailable(f"hipcc is not executable: {self.hipcc}")

    def argv(self) -> list[str]:
        return [self.hipcc, f"--offload-arch={self.offload_arch}", "-c", self.opt, "-o", "kernel.o", SOURCE_NAME]

    def check(self, code: str, tag: str) -> CompileResult:
        d = self.workdir / _check_tag(tag)
        try:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            (d / SOURCE_NAME).write_text(code, encoding="utf-8")
        except OSError as exc:
            raise CompileUnavailable(f"cannot prepare {d}: {exc}") from exc
        t0 = time.monotonic()
        try:
            rc, stderr = _run_in_own_group(self.argv(), d, self.timeout_s, self.kill_grace_s)
        except subprocess.TimeoutExpired:
            rc, stderr = -9, f"hipcc: error: compile timed out after {self.timeout_s:g} s\n"
        except OSError as exc:
            raise CompileUnavailable(f"cannot run {self.hipcc}: {exc}") from exc
        finally:
            seconds = time.monotonic() - t0
        shutil.rmtree(d, ignore_errors=True)
        digest = self.evidence.put(stderr) if self.evidence is not None else sha256_text(stderr)
        return CompileResult(
            ok=rc == 0, rc=rc, seconds=round(seconds, 4), errors=self.structure(stderr, code, SOURCE_NAME),
            raw_stderr_sha256=digest,
        )


def make_compiler(name: str, workdir: str | Path, evidence: EvidenceStore | None = None,
                  structure: Structure = compile_errors) -> Compiler:
    """The configured rollout compile check (`rollout.compile_adapter`)."""
    if name == "standin.hipcc":
        return HipccCompiler(workdir, evidence=evidence, structure=structure)
    if name == "fake.compile":
        return FakeCompiler(evidence=evidence, structure=structure)
    raise CompileUnavailable(f"compile adapter {name!r} is not available; built: standin.hipcc, fake.compile")


class FakeCompiler:
    """Fails exactly the code that holds `marker`, with one clang-format error per occurrence."""

    name = "fake.compile"

    def __init__(
        self,
        marker: str = COMPILE_ERROR_MARKER,
        *,
        latency_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        evidence: EvidenceStore | None = None,
        structure: Structure = compile_errors,
    ):
        self.marker = marker
        self.latency_s = latency_s
        self._sleep = sleep
        self.evidence = evidence
        self.structure = structure
        self.calls: list[str] = []

    def stderr_for(self, code: str) -> str:
        lines = []
        for n, text in enumerate(code.split("\n"), 1):
            start = 0
            while (i := text.find(self.marker, start)) >= 0:
                col = len(text[:i].encode("utf-8")) + 1
                lines.append(f"{SOURCE_NAME}:{n}:{col}: error: use of undeclared identifier '{self.marker}'")
                lines.append(f"  {n:4d} | {text}")
                start = i + len(self.marker)
        if lines:
            count = len(lines) // 2
            lines.append(f"{count} error{'s' if count > 1 else ''} generated when compiling for gfx950.")
        return "\n".join(lines) + ("\n" if lines else "")

    def check(self, code: str, tag: str) -> CompileResult:
        _check_tag(tag)
        self.calls.append(tag)
        if self.latency_s:
            self._sleep(self.latency_s)
        stderr = self.stderr_for(code)
        digest = self.evidence.put(stderr) if self.evidence is not None else sha256_text(stderr)
        rc = 1 if stderr else 0
        return CompileResult(
            ok=rc == 0, rc=rc, seconds=float(self.latency_s), errors=self.structure(stderr, code, SOURCE_NAME),
            raw_stderr_sha256=digest,
        )
