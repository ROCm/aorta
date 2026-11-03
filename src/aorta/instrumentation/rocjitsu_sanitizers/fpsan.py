"""Triton FpSan backend: run a reference/candidate command and compare digests.

Triton FpSan (``TRITON_INSTRUMENTATION_MODE=fpsan``) rewrites floating-point ops
into a deterministic payload algebra, so two kernels that compute the same thing
produce bit-identical sanitized outputs. The recipe's ``source.fpsan_command``
runs a reference kernel and a candidate kernel that way and prints one
``AORTA_FPSAN_RESULT`` line (see :mod:`.fpsan_result`); aorta neither compiles
Triton nor sets up the two kernels.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .backends import support
from .execution import ProcessResult, run_argv
from .fpsan_result import RESULT_MARKER, evaluate_fpsan_output
from .models import CheckResult, ExecutionState, KernelWorklist, Verdict

FPSAN = "fpsan"
TRITON_INSTRUMENTATION_MODE = "fpsan"


class ProcessExecutor(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _not_checked(reason: str) -> CheckResult:
    return CheckResult(
        sanitizer=FPSAN,
        state=ExecutionState.NOT_CHECKED,
        verdict=Verdict.NOT_CHECKED,
        reason=reason,
    )


def fpsan_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The command's environment: the caller's, with FpSan instrumentation on.

    The command still owns cache separation between its two kernels; this only
    makes a Triton compile it forgets to configure instrumented rather than not.
    """

    env = dict(environ)
    env["TRITON_INSTRUMENTATION_MODE"] = TRITON_INSTRUMENTATION_MODE
    return env


def run_fpsan(
    worklist: KernelWorklist,
    *,
    command: Path,
    target: str,
    output_dir: Path,
    timeout_seconds: float = 900.0,
    execute: ProcessExecutor = run_argv,
) -> CheckResult:
    """Run one recipe FpSan command and judge its ``AORTA_FPSAN_RESULT`` line.

    The worklist must name exactly one candidate, as ASan and ConSan require.
    The command still owns the reference kernel and the comparison; the selected
    name is recorded in ``backend`` so a verdict names what it was run for.
    An empty selection, or more than one, is not checked rather than passed:
    the result line carries one comparison and cannot cover a second candidate.
    """

    policy = support(FPSAN, target)
    if policy["level"] == "none":
        return _not_checked(f"fpsan_target_unsupported: {policy['note']}")
    if not worklist.kernels:
        return _not_checked("empty_worklist")
    if len(worklist.kernels) != 1:
        return _not_checked("fpsan_requires_one_candidate_kernel")
    if not command.is_file():
        return _not_checked("fpsan_command_not_found")

    backend = {
        "command": str(command),
        "command_sha256": _sha256_file(command),
        "result_marker": RESULT_MARKER,
        "selected_kernels": ", ".join(item.identity.name for item in worklist.kernels),
        "triton_instrumentation_mode": TRITON_INSTRUMENTATION_MODE,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    env = fpsan_environment(os.environ)
    process = execute((str(command),), timeout_seconds=timeout_seconds, env=env)
    (output_dir / "fpsan.log").write_text(f"{process.stdout}\n{process.stderr}", encoding="utf-8")
    check = evaluate_fpsan_output(process, sanitizer=FPSAN, target=target)
    backend.update(check.backend)
    return replace(check, backend=tuple(sorted(backend.items())))


def scoped_fpsan_not_checked(worklist: KernelWorklist) -> CheckResult:
    """Fail closed when no reference/candidate FpSan command is provisioned."""

    return _not_checked(
        "fpsan_command_not_provisioned: a command that runs a reference and a candidate "
        f"kernel under FpSan is required; aorta cannot compare {len(worklist.kernels)} "
        "selected kernel(s) without one, because FpSan checks that two kernels agree"
    )
