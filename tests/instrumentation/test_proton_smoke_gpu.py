"""Real Proton capture of the shipped Triton examples.

The counterpart of ``test_rocprof_smoke_gpu.py``: it runs an example payload
through the collector seam and asserts the parsed metrics describe the kernel
the payload actually launched.

Proton is harder to self-host than rocprofv3, so the skip conditions carry more
weight here. It runs inside the *workload's own* interpreter, which therefore
needs Triton plus a ROCm PyTorch build -- an aorta host venv with a CPU-only
torch cannot do it, and the reference environment is a ROCm PyTorch container
with aorta installed inside it. Every unmet precondition self-skips with the
reason spelled out, so this never fails for being on the wrong host and never
passes green for having silently measured nothing.

    pytest tests/instrumentation/test_proton_smoke_gpu.py -m "gpu and rocm"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.instrumentation.proton import OUTPUT_SUBDIR, PROFILE_BASENAME
from aorta.run.collectors import (
    CONFIG_KEY_COLLECT,
    CONFIG_KEY_COLLECT_DIR,
    CONFIG_KEY_COLLECT_OPTIONS,
    summarize_collectors,
    wrap_argv_for_collectors,
)

pytestmark = [pytest.mark.gpu, pytest.mark.rocm]

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTON_EXAMPLES = REPO_ROOT / "examples" / "profiling" / "proton"

#: ``(example dir name, payload file, payload args, the Triton kernel's name in
#: the hatchet tree)``. The kernel name is what makes the assertion mean
#: something: a tree containing only torch's own kernels would otherwise pass.
_EXAMPLES = [
    ("triton-vecadd", "vecadd.py", ["--size", "262144", "--iters", "10"], "add_kernel"),
    (
        "triton-softmax",
        "softmax.py",
        ["--rows", "512", "--cols", "1024", "--iters", "10"],
        "softmax_kernel",
    ),
]

#: Wall-clock budget for one payload subprocess, sized from measurement rather
#: than guessed: the captures timed on MI350 land at ~2.5 s each with a cold
#: Triton cache, so this is ~50x headroom. These are ten iterations of one
#: small Triton kernel; nothing here has a legitimate reason to take minutes.
#:
#: It sits deliberately below the GPU workflow's ``--timeout``, so a wedged
#: payload surfaces as ``TimeoutExpired`` from here -- naming the payload and
#: carrying its partial output -- rather than as pytest-timeout killing the
#: test from outside with no such detail. The former 3600 s could do neither:
#: it matched the CI job's own 60-minute cap exactly, so the job always died
#: first and the budget could never actually fire.
_CHILD_TIMEOUT_S = 120

#: Proton's dlopen failure on an image whose ROCm comes from Python wheels:
#: those ship only ``libroctracer64.so.4`` while Proton dlopens the unversioned
#: name. An environment defect with a known fix, not a collector bug, so it
#: skips with that fix named rather than failing.
_DLOPEN_MARKER = "Could not load `libroctracer64.so`"


def _skip_reason() -> str | None:
    if not Path("/dev/kfd").exists():
        return "no /dev/kfd; not a ROCm GPU host"
    if not PROTON_EXAMPLES.is_dir():
        return f"proton examples missing: {PROTON_EXAMPLES}"
    try:
        import triton  # noqa: F401
    except Exception as exc:  # a broken native install raises more than ImportError
        return f"triton not importable ({exc.__class__.__name__}); Proton ships inside Triton"
    try:
        import torch
    except Exception as exc:
        return f"torch not importable ({exc.__class__.__name__}); the payloads need it"
    if getattr(torch.version, "hip", None) is None:
        return (
            "torch is a CPU-only build (torch.version.hip is None); the Triton "
            "payloads cannot dispatch GPU work. Run this inside a ROCm PyTorch "
            "container with aorta installed there."
        )
    if not torch.cuda.is_available():
        return "torch reports no available GPU device"
    return None


_SKIP_REASON = _skip_reason()
skip_no_proton = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def _capture(example: str, payload: str, args: list[str], out_root: Path, options: dict):
    """Run one example under the collector seam; return (proc, metrics)."""
    config = {
        CONFIG_KEY_COLLECT: ["proton"],
        CONFIG_KEY_COLLECT_DIR: str(out_root),
        CONFIG_KEY_COLLECT_OPTIONS: {"proton": options},
    }
    script = PROTON_EXAMPLES / example / payload
    argv = wrap_argv_for_collectors(config, [sys.executable, str(script), *args])
    proc = subprocess.run(
        argv,
        cwd=out_root,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
        env={**os.environ, "TRITON_CACHE_DIR": str(out_root / "triton-cache")},
    )
    if _DLOPEN_MARKER in proc.stderr:
        pytest.skip(
            f"{_DLOPEN_MARKER}: this environment's ROCm ships only versioned "
            "sonames (a wheel-provided ROCm), while Proton dlopens the "
            "unversioned name. Add the unversioned symlinks to "
            "$LD_LIBRARY_PATH, or run on a host with a system ROCm install."
        )
    return proc, summarize_collectors(config)


@skip_no_proton
@pytest.mark.parametrize(
    ("example", "payload", "args", "kernel"),
    _EXAMPLES,
    ids=[entry[0] for entry in _EXAMPLES],
)
def test_example_capture_has_real_kernel_data(example, payload, args, kernel, tmp_path):
    proc, metrics = _capture(
        example, payload, args, tmp_path, {"backend": "auto", "context": "shadow", "data": "tree"}
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout, "the profiler must not perturb the payload's own result"

    profile = tmp_path / OUTPUT_SUBDIR / f"{PROFILE_BASENAME}.hatchet"
    assert profile.is_file(), sorted(str(p) for p in (tmp_path / OUTPUT_SUBDIR).rglob("*"))
    assert metrics["proton_kernel_count"] > 0
    assert 0.0 < metrics["proton_gpu_time_ms"] < 60_000.0
    assert 0.0 < metrics["proton_top_kernel_ms"] <= metrics["proton_gpu_time_ms"]
    assert kernel in metrics["proton_top_kernels"], metrics["proton_top_kernels"]
    assert metrics["proton_artifact_dir"] == str(tmp_path / OUTPUT_SUBDIR)


@skip_no_proton
def test_default_backend_is_accepted_by_the_installed_proton(tmp_path):
    """``backend: auto`` omits Proton's ``-b`` so Proton picks for itself.

    This is the whole reason the default is ``auto``: Triton 3.7.x's CLI lists
    only cupti/roctracer/instrumentation and exits at argparse on
    ``-b rocprofiler``, before the payload runs, while newer Triton prefers
    ``rocprofiler``. Passing no backend is the only spelling that survives both,
    and this asserts it against whatever Triton is actually installed.
    """
    example, payload, args, kernel = _EXAMPLES[0]
    proc, metrics = _capture(example, payload, args, tmp_path, {})
    assert "invalid choice" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert kernel in metrics["proton_top_kernels"]


@skip_no_proton
def test_python_context_attributes_to_call_paths(tmp_path):
    """``context: python`` keys the tree by Python call path instead of launch
    site -- the configuration the softmax example ships."""
    example, payload, args, kernel = _EXAMPLES[1]
    proc, metrics = _capture(example, payload, args, tmp_path, {"context": "python"})
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert metrics["proton_kernel_count"] > 0
    assert metrics["proton_gpu_time_ms"] > 0.0
    assert kernel in metrics["proton_top_kernels"], metrics["proton_top_kernels"]


@skip_no_proton
def test_hip_visible_devices_is_translated_not_refused(tmp_path, monkeypatch):
    """Proton raises outright when ``HIP_VISIBLE_DEVICES`` is set on AMD
    (``Proton does not work when the environment variable HIP_VISIBLE_DEVICES
    is set on AMD GPUs``). The wrap moves it to ``ROCR_VISIBLE_DEVICES`` so a
    device-pinned cell profiles rather than crashing, and the profile it gets
    back is a real one rather than an empty tree."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    example, payload, args, kernel = _EXAMPLES[0]
    proc, metrics = _capture(example, payload, args, tmp_path, {})
    assert "HIP_VISIBLE_DEVICES is set" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert kernel in metrics["proton_top_kernels"]
