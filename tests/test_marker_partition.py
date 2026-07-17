"""Guard that CPU and GPU CI marker selections partition the pytest suite.

The Phase 1 CPU gate runs ``pytest -m "not gpu and not rocm"``; the Phase 2
GPU gate runs ``pytest -m "gpu or rocm"``. Together they must cover every test
exactly once so new tests cannot silently fall outside both gates.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

CPU_EXPR = "not gpu and not rocm"
GPU_EXPR = "gpu or rocm"

_NODEID_RE = re.compile(r"^(tests/.+::.+)$")


def _collect_nodeids(markexpr: str | None = None) -> set[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TESTS_DIR),
        "--collect-only",
        "-q",
        "--disable-warnings",
    ]
    if markexpr is not None:
        cmd.extend(["-m", markexpr])

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    nodeids: set[str] = set()
    for line in proc.stdout.splitlines():
        match = _NODEID_RE.match(line.strip())
        if match:
            nodeids.add(match.group(1))
    return nodeids


def test_cpu_and_gpu_gates_partition_the_suite() -> None:
    all_ids = _collect_nodeids()
    cpu_ids = _collect_nodeids(CPU_EXPR)
    gpu_ids = _collect_nodeids(GPU_EXPR)

    missing = sorted(all_ids - cpu_ids - gpu_ids)
    assert not missing, f"tests missing from both CPU and GPU gates: {missing}"

    overlap = sorted(cpu_ids & gpu_ids)
    assert not overlap, f"tests selected by both CPU and GPU gates: {overlap}"
