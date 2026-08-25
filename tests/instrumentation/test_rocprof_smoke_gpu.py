"""Real ``rocprofv3`` capture of the shipped HIP GEMM example.

Everything in ``test_rocprof.py`` is fixture-driven; this file is the part
those tests cannot be: it compiles the example payload with ``hipcc``, runs it
under the collector seam on real hardware, and asserts the parsed metrics are
sane. It also pins the on-disk layout of an end-to-end ``aorta sweep run``,
including the fact that collector metrics reach ``perf.md`` and
``matrix.json::cells[*].metrics_summary``.

Self-skips on a host without a GPU / ROCm toolchain, so the CPU gate
(``pytest -m "not gpu and not rocm"``) never sees it and a laptop run does not
fail.

    pytest tests/instrumentation/test_rocprof_smoke_gpu.py -m "gpu and rocm"
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.instrumentation.rocprof import OUTPUT_SUBDIR, SUMMARY_FILENAME
from aorta.run.collectors import (
    CONFIG_KEY_COLLECT,
    CONFIG_KEY_COLLECT_DIR,
    CONFIG_KEY_COLLECT_OPTIONS,
    summarize_collectors,
    wrap_argv_for_collectors,
)

pytestmark = [pytest.mark.gpu, pytest.mark.rocm]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "profiling" / "rocprof" / "hip-gemm"
PAYLOAD = EXAMPLE_DIR / "gemm.hip"
RECIPE = EXAMPLE_DIR / "recipe.yaml"

# Small enough to finish in well under a second, big enough that the kernel
# time is comfortably above timer noise.
_GEMM_N = "512"
_GEMM_ITERS = "20"


def _skip_reason() -> str | None:
    if shutil.which("rocprofv3") is None:
        return "rocprofv3 not on PATH (ships with ROCm)"
    if shutil.which("hipcc") is None:
        return "hipcc not on PATH (needed to build the example payload)"
    if not Path("/dev/kfd").exists():
        return "no /dev/kfd; not a ROCm GPU host"
    if not PAYLOAD.is_file():
        return f"example payload missing: {PAYLOAD}"
    return None


_SKIP_REASON = _skip_reason()
skip_no_gpu = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


@pytest.fixture(scope="module")
def gemm_binary(tmp_path_factory) -> Path:
    """Compile the example payload once for the whole module."""
    out = tmp_path_factory.mktemp("hip-gemm") / "gemm"
    proc = subprocess.run(
        ["hipcc", "-O3", str(PAYLOAD), "-o", str(out)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(f"hipcc failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return out


def _run(argv, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=900)


# ---- The payload itself -------------------------------------------------


@skip_no_gpu
def test_payload_runs_and_self_checks(gemm_binary, tmp_path):
    """The example is self-checking, so a bad result is a failed trial rather
    than a suspiciously fast one."""
    proc = _run([str(gemm_binary), _GEMM_N, _GEMM_ITERS], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout


# ---- The collector seam on real hardware -------------------------------


@skip_no_gpu
def test_collector_seam_captures_real_kernel_data(gemm_binary, tmp_path):
    """The generic argv-wrap seam, end to end: wrap, run, parse."""
    config = {
        CONFIG_KEY_COLLECT: ["rocprof"],
        CONFIG_KEY_COLLECT_DIR: str(tmp_path),
        CONFIG_KEY_COLLECT_OPTIONS: {
            "rocprof": {"trace": "kernel", "stats": "1", "summary_units": "msec"}
        },
    }
    argv = wrap_argv_for_collectors(config, [str(gemm_binary), _GEMM_N, _GEMM_ITERS])
    proc = _run(argv, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout, "the profiler must not perturb the payload's own result"

    out_dir = tmp_path / OUTPUT_SUBDIR
    csvs = sorted(path.name for path in out_dir.rglob("*.csv"))
    assert csvs, f"rocprofv3 wrote no CSVs into {out_dir}"
    assert any(name.endswith("_kernel_stats.csv") for name in csvs), csvs
    assert any(name.endswith("_kernel_trace.csv") for name in csvs), csvs

    metrics = summarize_collectors(config)
    # The payload launches exactly `iters` kernels plus warmups, so the count
    # is bounded rather than exact -- but it is certainly not zero.
    assert metrics["rocprof_kernel_count"] >= int(_GEMM_ITERS)
    assert 0.0 < metrics["rocprof_gpu_time_ms"] < 60_000.0
    assert 0.0 < metrics["rocprof_top_kernel_ms"] <= metrics["rocprof_gpu_time_ms"]
    assert any("sgemm" in name for name in metrics["rocprof_top_kernels"]), metrics[
        "rocprof_top_kernels"
    ]
    assert metrics["rocprof_artifact_dir"] == str(out_dir)


@skip_no_gpu
def test_summary_lands_in_a_file_not_on_stderr(gemm_binary, tmp_path):
    """``rocprofv3 -S`` prints to stderr by default, where the probe
    classifier's stderr detectors would read it as workload output.

    ``--summary-output-file`` takes a filename STEM relative to ``-d``, not a
    path -- handing it an absolute path splices that path into the middle of a
    filename and scatters directories under ``-d``. This asserts the real file
    appears where the package says it does, and nowhere else.
    """
    config = {
        CONFIG_KEY_COLLECT: ["rocprof"],
        CONFIG_KEY_COLLECT_DIR: str(tmp_path),
    }
    argv = wrap_argv_for_collectors(config, [str(gemm_binary), _GEMM_N, "5"])
    proc = _run(argv, tmp_path)
    assert proc.returncode == 0, proc.stderr

    out_dir = tmp_path / OUTPUT_SUBDIR
    summary = out_dir / SUMMARY_FILENAME
    assert summary.is_file(), sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*"))
    assert "ROCPROFV3 SUMMARY" in summary.read_text(encoding="utf-8")
    assert "ROCPROFV3 SUMMARY" not in proc.stderr
    # Everything rocprofv3 wrote is a plain file directly under the artifact
    # dir; a spliced path would have created nested directories instead.
    assert not [path for path in out_dir.iterdir() if path.is_dir()]


@skip_no_gpu
def test_no_gpu_work_produces_no_artifacts_and_no_metrics(tmp_path):
    """Verified behaviour of rocprofv3: a command that dispatches no kernels
    makes it write nothing at all. That is a legitimate outcome (a probe of
    ``/bin/echo``), so it must not raise or invent numbers."""
    config = {CONFIG_KEY_COLLECT: ["rocprof"], CONFIG_KEY_COLLECT_DIR: str(tmp_path)}
    argv = wrap_argv_for_collectors(config, ["/bin/echo", "hi"])
    proc = _run(argv, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "hi" in proc.stdout

    out_dir = tmp_path / OUTPUT_SUBDIR
    assert out_dir.is_dir(), "the seam pre-creates the dir so the trial tree keeps its shape"
    assert not sorted(out_dir.rglob("*.csv"))
    assert summarize_collectors(config) == {"rocprof_artifact_dir": str(out_dir)}


# ---- End-to-end sweep: artifact layout + report plumbing ---------------


@skip_no_gpu
def test_sweep_run_layout_and_reports(gemm_binary, tmp_path):
    """A real multi-trial ``aorta sweep run`` of the shipped example recipe.

    Pins three things at once, all of which have to be true together for the
    feature to be worth anything:

    1. The collector artifact directory is
       ``<cell>/<workload>/trial_d<d>_m<m>_t<t>/rocprof/`` -- a SIBLING of the
       hand-written ``<cell>/trial_<n>/`` tree, not inside it.
    2. Collector metrics ride ``WorkloadResult.metrics`` into the dispatcher's
       trial JSON at ``.result.metrics``, and NOT into
       ``<cell>/trial_<n>/result.json``, which ``SubprocessWorkload`` writes
       before the summary is parsed.
    3. The numeric metrics reach ``perf.md``'s metrics table and
       ``matrix.json::cells[*].metrics_summary`` with one entry per trial.
    """
    trials = 3
    recipe_text = RECIPE.read_text(encoding="utf-8").replace("trials: 1", f"trials: {trials}")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(recipe_text, encoding="utf-8")
    output = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from aorta.cli import main; main()",
            "sweep",
            "run",
            "--recipe",
            str(recipe),
            "--output",
            str(output),
            "--",
            str(gemm_binary),
            _GEMM_N,
            _GEMM_ITERS,
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

    run_dirs = [path for path in output.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    # (1) + (2): per-trial layout.
    trial_jsons = sorted(run_dir.glob("*/*/trial_d*_m*_t*.json"))
    assert len(trial_jsons) == trials, [str(p) for p in trial_jsons]
    for trial_json in trial_jsons:
        doc = json.loads(trial_json.read_text(encoding="utf-8"))
        metrics = doc["result"]["metrics"]
        assert metrics["rocprof_kernel_count"] >= int(_GEMM_ITERS)
        assert metrics["rocprof_gpu_time_ms"] > 0.0

        artifact_dir = trial_json.with_suffix("") / OUTPUT_SUBDIR
        assert artifact_dir.is_dir(), f"no collector artifacts beside {trial_json.name}"
        assert sorted(artifact_dir.glob("*_kernel_stats.csv"))
        assert metrics["rocprof_artifact_dir"] == str(artifact_dir)

        # The hand-written result.json is a sibling of the workload subdir and
        # predates the summary parse, so it carries no collector metrics.
        hand_written = Path(metrics["result_json_path"])
        assert hand_written.is_file()
        assert artifact_dir not in hand_written.parents
        assert hand_written.parent not in artifact_dir.parents
        plain = json.loads(hand_written.read_text(encoding="utf-8"))
        assert not [key for key in plain if key.startswith("rocprof_")]

    # (3): the reports.
    perf = (run_dir / "perf.md").read_text(encoding="utf-8")
    for key in ("rocprof_gpu_time_ms", "rocprof_kernel_count", "rocprof_top_kernel_ms"):
        assert key in perf, f"{key} missing from perf.md"
    # The non-numeric channels are deliberately absent from the metrics table.
    assert "rocprof_top_kernels" not in perf
    assert "rocprof_artifact_dir" not in perf

    matrix = json.loads((run_dir / "matrix.json").read_text(encoding="utf-8"))
    summaries = [cell["metrics_summary"] for cell in matrix["cells"]]
    assert summaries, matrix["cells"]
    for summary in summaries:
        entry = summary["rocprof_gpu_time_ms"]
        assert entry["n"] == trials
        assert entry["min"] > 0.0
        assert entry["min"] <= entry["mean"] <= entry["max"]
