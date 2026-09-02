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

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.instrumentation.proton import (
    ENV_PREFIX,
    OUTPUT_SUBDIR,
    PROFILE_BASENAME,
    build_argv_prefix,
    parse_summary,
)
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
#: those ship only versioned sonames (``libroctracer64.so.4``) while Proton
#: dlopens the unversioned name. An environment defect with a known fix, not a
#: collector bug, so it skips with that fix named rather than failing.
#:
#: Matched by shape rather than by one filename, because which library is missing
#: depends on the backend: ``libroctracer64.so`` for ``roctracer`` and
#: ``librocprofiler-sdk.so`` for the ``rocprofiler`` backend that Triton 3.8.0
#: made reachable -- and on 3.8.0 ``backend: auto`` resolves to the latter, so a
#: single-name match would start failing where it used to skip.
_DLOPEN_RE = re.compile(r"Could not load `lib[^`]+`")


def _dlopen_failure(stderr: str) -> str | None:
    """Return Proton's dlopen complaint from ``stderr``, or ``None``."""
    found = _DLOPEN_RE.search(stderr)
    return found.group(0) if found else None


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


def _triton_minor() -> tuple[int, int] | None:
    """``(major, minor)`` of the installed Triton, or ``None`` if unreadable."""
    try:
        import triton
    except Exception:
        return None
    parts = str(getattr(triton, "__version__", "")).split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


_TRITON = _triton_minor()

#: Whether the installed Triton is one where a ``mode: cli`` pin of
#: ``roctracer`` actually captures. On 3.7.x it comes back as a 160-byte bare
#: ``ROOT`` -- the empty capture the ``wrap_argv`` guard exists to refuse -- and
#: from 3.8.0 on the same pin records normally (measured on gfx950 / ROCm 10:
#: 3090 bytes, byte-for-byte the size ``backend: auto`` produces). The test
#: below asserts whichever of the two is true here, rather than skipping on the
#: newer stack, so the day the version floor moves the guard's premise is still
#: under test instead of merely unasserted.
_CLI_PIN_CAPTURES = _TRITON is not None and _TRITON >= (3, 8)


def _raw_proton_env(**extra: str) -> dict[str, str]:
    """Child env for a test that drives Proton directly, bypassing the collector.

    Proton refuses ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` on AMD
    before it does anything else, and the collector's ``_device_env_prefix``
    normally translates them. A test that builds its own argv skips that
    translation, so on a runner that pins devices the refusal would preempt
    whatever the test is actually about -- the empty-tree capture, or the
    unsupported-mode path -- and fail for a reason unrelated to the assertion.

    Translated rather than merely dropped, and with the collector's own
    precedence: the HIP spelling wins when both are set, and an explicit
    ``ROCR_VISIBLE_DEVICES`` already in the environment is left alone. Dropping
    the selection instead would quietly widen the test to every GPU on the host.
    """
    env = {**os.environ, **extra}
    present = [
        (name, env[name])
        for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        if env.get(name) is not None
    ]
    for name, _ in present:
        env.pop(name, None)
    if present and env.get("ROCR_VISIBLE_DEVICES") is None:
        env["ROCR_VISIBLE_DEVICES"] = present[0][1]
    return env


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
    missing = _dlopen_failure(proc.stderr)
    if missing is not None:
        pytest.skip(
            f"{missing}: this environment's ROCm ships only versioned "
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
def test_hook_flag_is_accepted_by_the_installed_proton(tmp_path):
    """``hook: triton`` renders Proton's ``-k triton``, which older Tritons could
    plausibly not have. Being *in the schema* says nothing about being accepted
    by the Proton that runs, so this asserts the wrap still captures with it on
    rather than dying at argparse the way an unknown ``-b`` does."""
    example, payload, args, kernel = _EXAMPLES[0]
    proc, metrics = _capture(example, payload, args, tmp_path, {"hook": "triton"})
    assert "invalid choice" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert metrics.get("proton_kernel_count", 0) > 0, metrics
    assert kernel in metrics.get("proton_top_kernels", []), metrics


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


# ---- Pinning an explicit AMD backend ------------------------------------
#
# Proton's CLI front-end calls ``_select_backend()`` only when ``-b`` is
# absent, and that call is what initialises the HIP driver. ``roctracer`` pinned
# through ``mode: cli`` therefore attaches ahead of the runtime it is meant to
# trace and records nothing, exiting 0 with a hatchet holding a bare ROOT frame.
# Only ``roctracer``: ``rocprofiler`` is configured from a ``libproton.so``
# constructor and wants to land before the runtime, so its CLI pin is allowed. ``wrap_argv`` refuses that combination and names
# ``mode: env``, where the payload starts Proton itself; these two tests are the
# regression that would have caught the empty capture, and the check that the
# upstream ordering it works around is still there.


@skip_no_proton
def test_env_mode_pins_roctracer_and_gets_a_non_empty_tree(tmp_path):
    """The route the ``mode: cli`` rejection names, end to end.

    Asserts the launch count rather than merely that metrics exist: the capture
    spans exactly the payload's profiled loop, which dispatches its three Triton
    kernels once per iteration and nothing else, so the total is predictable and
    an over- or under-count is a real defect rather than noise.
    """
    iterations = 5
    proc, metrics = _capture(
        "amd-roctracer",
        "pipeline.py",
        ["--rows", "512", "--cols", "1024", "--iters", str(iterations)],
        tmp_path,
        {"mode": "env", "backend": "roctracer", "context": "shadow", "data": "tree"},
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout
    # Presence first, and with the whole mapping in the message: an empty tree
    # is the failure this test exists for, and it shows up as *absent* metrics
    # rather than wrong ones.
    assert {"proton_kernel_count", "proton_gpu_time_ms"} <= set(metrics), metrics
    kernels = {"scale_kernel", "bias_gelu_kernel", "row_sum_kernel"}
    assert kernels <= set(metrics.get("proton_top_kernels", [])), metrics
    assert metrics.get("proton_kernel_count") == len(kernels) * iterations, metrics
    assert metrics.get("proton_gpu_time_ms", 0.0) > 0.0, metrics


@skip_no_proton
def test_cli_mode_pin_captures_nothing_only_on_the_versions_the_guard_targets(tmp_path):
    """Pins the premise of the ``wrap_argv`` guard against the installed Triton.

    Builds the wrap the guard refuses -- ``build_argv_prefix`` renders the flags
    without the attach-mode check -- and asserts what that Triton actually does
    with it: an empty tree on 3.7.x, a populated one from 3.8.0.

    Both halves are load-bearing. The 3.7.x half is the regression that would
    have caught the silent empty capture in the first place. The 3.8.0 half is
    what says the guard has become a no-op there, so removing it stops being a
    guess -- and it fails, loudly, if a later Triton reintroduces the ordering
    bug while the guard is gone.
    """
    out_dir = tmp_path / OUTPUT_SUBDIR
    out_dir.mkdir()
    script = PROTON_EXAMPLES / "triton-vecadd" / "vecadd.py"
    argv = [
        *build_argv_prefix(out_dir, {"backend": "roctracer"}, python=sys.executable),
        str(script),
        "--size",
        "262144",
        "--iters",
        "10",
    ]
    proc = subprocess.run(
        argv,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=3600,
        env=_raw_proton_env(TRITON_CACHE_DIR=str(tmp_path / "triton-cache")),
    )
    missing = _dlopen_failure(proc.stderr)
    if missing is not None:
        pytest.skip(missing)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout, "the payload itself must still have run"

    metrics = parse_summary(out_dir)
    if _CLI_PIN_CAPTURES:
        # The guard's premise has expired on this Triton. Assert a real capture
        # rather than merely "not empty": an ordering bug that recorded one
        # stray kernel would satisfy the weaker check.
        assert metrics.get("proton_kernel_count", 0) > 0, metrics
        assert metrics.get("proton_gpu_time_ms", 0.0) > 0.0, metrics
        assert "add_kernel" in metrics.get("proton_top_kernels", []), metrics
    else:
        # The silent part of the failure: a clean exit, an artifact directory,
        # and no measurement anywhere in it.
        assert metrics == {"proton_artifact_dir": str(out_dir)}


@skip_no_proton
def test_a_backend_that_refuses_the_mode_exits_two_not_a_traceback(tmp_path):
    """A backend can be present and still refuse the mode, and that has to read
    as "this environment cannot take the measurement" rather than as a crash.

    Triton 3.8.0 is the case that motivates it: it registers ``rocprofiler``, so
    the payload's availability probe passes, but its ``RocprofSDKProfiler``
    accepts only ``periodic_flushing`` and raises ``ValueError: [PROTON]
    RocprofSDKProfiler: unsupported mode: pcsampling``. That version is not
    installable here, so the same path is driven through a mode no backend
    accepts -- what is under test is the payload's handling, not Proton's
    taxonomy of modes.
    """
    out_dir = tmp_path / "proton"
    out_dir.mkdir()
    script = PROTON_EXAMPLES / "amd-rocprofiler" / "gelu.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--size", "262144", "--iters", "5"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=3600,
        env=_raw_proton_env(
            **{
                "TRITON_CACHE_DIR": str(tmp_path / "triton-cache"),
                f"{ENV_PREFIX}NAME": str(out_dir / "proton"),
                f"{ENV_PREFIX}BACKEND": "roctracer",
                f"{ENV_PREFIX}MODE": "definitely_not_a_mode",
                f"{ENV_PREFIX}CONTEXT": "shadow",
                f"{ENV_PREFIX}DATA": "tree",
            }
        ),
    )
    missing = _dlopen_failure(proc.stderr)
    if missing is not None:
        pytest.skip(missing)
    assert proc.returncode == 2, (
        "an unsupported mode must exit 2, the same code an unobtainable backend "
        f"gets, not crash or traceback:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "Traceback" not in proc.stderr, f"should be handled, not raised:\n{proc.stderr}"
    # Proton's own words, then the payload's reading of them.
    assert "unsupported mode" in proc.stderr, proc.stderr
    assert "does not accept mode=" in proc.stderr, proc.stderr
    # The classification is the point, not the exit code: the same handler sees
    # dlopen and device failures, and blaming the mode for those is what this
    # asserts against. Measured on aorta's ROCm 10 base, where every mode fails
    # at `Could not load \`librocprofiler-sdk.so\`` -- a mode diagnosis there
    # would be confidently wrong.
    assert "could not dlopen" not in proc.stderr, proc.stderr
    assert "ROCR_VISIBLE_DEVICES instead" not in proc.stderr, proc.stderr


def _pcsampling_skip_reason(tmp_path) -> str | None:
    """Report why ``rocprofiler`` + ``pcsampling`` cannot run here, or ``None``.

    Asked by attempting it, because there is no way to ask directly: the backend
    being listed in ``get_available_profilers()`` does not mean the mode is
    implemented (Triton 3.8.0 lists the backend and rejects the mode), and
    neither answers whether the backend's library will load.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import triton.profiler as proton;"
            "proton.start('probe', backend='rocprofiler', mode='pcsampling');"
            "proton.finalize()",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        env=_raw_proton_env(TRITON_CACHE_DIR=str(tmp_path / "triton-cache")),
    )
    if probe.returncode == 0:
        return None
    tail = [line for line in probe.stderr.strip().splitlines() if line.strip()]
    return tail[-1] if tail else f"probe exited {probe.returncode} with no stderr"


@skip_no_proton
def test_pcsampling_captures_once_the_environment_can_do_it(tmp_path):
    """The success path `amd-rocprofiler` advertises, gated on being possible.

    Every other test around this example asserts a *handled failure*, which
    leaves the advertised capture untested -- so the example could stay broken
    after a capable Triton finally reaches CI, and nothing would say so. This
    runs the recipe's real option pair and skips, with the reason, wherever that
    pair cannot run: no Triton obtainable today can (3.8.0 rejects the mode, and
    aorta's ROCm 10 base cannot even load `librocprofiler-sdk.so`), so it is a
    skip everywhere for now and starts enforcing by itself when that changes.

    The assertion is deliberately shape-agnostic. What PC-sampling leaves in a
    hatchet tree has not been observed here, so requiring particular metric keys
    would be guessing; requiring the tree to carry something under ROOT is the
    property that distinguishes a real capture from the empty one.
    """
    reason = _pcsampling_skip_reason(tmp_path)
    if reason is not None:
        pytest.skip(f"this Proton cannot do rocprofiler+pcsampling: {reason}")

    proc, metrics = _capture(
        "amd-rocprofiler",
        "gelu.py",
        ["--size", "262144", "--iters", "10"],
        tmp_path,
        {
            "mode": "env",
            "backend": "rocprofiler",
            "backend_mode": "pcsampling",
            "context": "shadow",
            "data": "tree",
        },
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout, proc.stdout
    profile = tmp_path / OUTPUT_SUBDIR / f"{PROFILE_BASENAME}.hatchet"
    assert profile.is_file(), sorted(str(q) for q in (tmp_path / OUTPUT_SUBDIR).rglob("*"))
    tree = json.loads(profile.read_text(encoding="utf-8"))
    root = tree[0] if isinstance(tree, list) else tree
    assert root.get("children"), f"PC-sampling capture is an empty ROOT tree: {tree}"
    assert metrics["proton_artifact_dir"] == str(tmp_path / OUTPUT_SUBDIR)


@skip_no_proton
def test_instrumentation_captures_scopes_inside_one_kernel(tmp_path):
    """The intra-kernel backend attributes cycles to regions within a kernel.

    Its leaves carry ``cycles`` / ``normalized_cycles`` and neither ``count``
    nor ``time (<unit>)``, so the collector's summary -- which keys on wall-clock
    time -- publishes only the artifact directory. That is asserted here too, so
    the documented metric gap cannot drift out of step with the parser.
    """
    proc, metrics = _capture(
        "amd-instrumentation",
        "hotspot.py",
        ["--size", "65536", "--steps", "8", "--iters", "3"],
        tmp_path,
        {
            "mode": "env",
            "backend": "instrumentation",
            "instrumentation_mode": "default",
            "context": "shadow",
            "data": "tree",
        },
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout

    profile = tmp_path / OUTPUT_SUBDIR / f"{PROFILE_BASENAME}.hatchet"
    assert profile.is_file(), sorted(str(p) for p in (tmp_path / OUTPUT_SUBDIR).rglob("*"))
    cycles = _scope_cycles(profile)
    assert {"cheap", "expensive"} <= set(cycles), cycles
    assert all(value > 0 for value in cycles.values()), cycles
    assert metrics == {"proton_artifact_dir": str(tmp_path / OUTPUT_SUBDIR)}


def _scope_cycles(profile: Path) -> dict[str, float]:
    """Map each leaf frame name to its ``cycles`` metric in a hatchet profile."""
    found: dict[str, float] = {}

    def walk(node):
        if not isinstance(node, dict):
            return
        children = node.get("children") or []
        for child in children:
            walk(child)
        if children:
            return
        name = node.get("frame", {}).get("name")
        value = node.get("metrics", {}).get("cycles")
        if isinstance(name, str) and isinstance(value, (int, float)):
            found[name] = float(value)

    for root in json.loads(profile.read_text(encoding="utf-8")):
        walk(root)
    return found


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
