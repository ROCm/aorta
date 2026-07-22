"""Guard that CPU and GPU CI marker selections partition the pytest suite.

The Phase 1 CPU gate runs ``pytest -m "not gpu and not rocm"``; the Phase 2
GPU gate runs ``pytest -m "gpu or rocm"``. Together they must cover every test
exactly once so new tests cannot silently fall outside both gates.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CPU_EXPR = "not gpu and not rocm"
GPU_EXPR = "gpu or rocm"

_NODEID_RE = re.compile(r"^(tests/.+::.+)$")


def _collect_nodeids(markexpr: str | None = None) -> set[str]:
    # Override the repo-level ``addopts`` (``-o addopts=``) so this nested
    # collection is not sensitive to future pytest.ini changes (e.g. ``-v`` /
    # colored output) that would perturb the ``-q`` node-id formatting parsed
    # below. Re-add ``--strict-markers`` explicitly (clearing addopts also drops
    # the repo's strict-marker policy) since this guard is about marker hygiene:
    # an unknown/typo'd marker should fail here with actionable output. Use a
    # relative ``tests`` path (with cwd=REPO_ROOT) so node ids are stable, and
    # force color off.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
        "--strict-markers",
        "-p",
        "no:cacheprovider",
        "--color=no",
    ]
    if markexpr is not None:
        cmd.extend(["-m", markexpr])

    env = {**os.environ, "PY_COLORS": "0", "NO_COLOR": "1"}
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            "pytest --collect-only failed"
            f" (mark={markexpr!r}, exit={exc.returncode}).\n"
            f"--- stdout ---\n{exc.stdout}\n--- stderr ---\n{exc.stderr}"
        ) from exc

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
