"""GPU + docker smoke tests for the layer_numerics logger.

These mirror the hand-run shell scripts in the aorta-workspace
(``run_layer_numerics_all_scenarios.sh`` and
``run_layer_numerics_collect_smoke.sh``) as native pytest tests, so ``git pull``
keeps them in sync with the code. They exercise the REAL paths the unit tests
cannot: a torchrec ``DistributedModelParallel`` model under ROCm inside the
customer repro docker image.

They self-skip unless everything needed is present, so a normal CPU/CI run stays
green:

  - ``docker`` on PATH,
  - the customer repro dir + entry script (env ``AORTA_LN_REPRO_DIR``,
    default the recom_repro source path),
  - the docker image pulled (env ``AORTA_LN_IMAGE``),
  - opt-in via ``AORTA_LN_GPU_SMOKE=1`` (so the heavy docker run never fires by
    accident on a shared box).

Run on a GPU box:

    AORTA_LN_GPU_SMOKE=1 \
    AORTA_LN_REPRO_DIR=/apps/.../recom_repro/source \
    pytest tests/instrumentation/test_layer_numerics_smoke_gpu.py -v -m gpu

The collect-path test additionally needs the ``aorta`` CLI + an aorta-internal
checkout (the recom_repro workload); it skips if ``aorta`` is not on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aorta.instrumentation.layer_numerics import OUTPUT_SUBDIR, SCRIPT_PATH

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# --- gating ------------------------------------------------------------------
_OPT_IN = os.environ.get("AORTA_LN_GPU_SMOKE") == "1"
_IMAGE = os.environ.get("AORTA_LN_IMAGE", "rocm/pytorch-private:nan-repro-hipblaslt-may-drop")
_REPRO_DIR = Path(os.environ.get(
    "AORTA_LN_REPRO_DIR",
    "/apps/oyazdanb/aorta-internal/aorta_internal/workloads/recom_repro/source",
))
_SCRIPT_NAME = os.environ.get("AORTA_LN_SCRIPT_NAME", "repro3_precision_test_shampoo_old.py")
_NUM_STEPS = os.environ.get("AORTA_LN_NUM_STEPS", "20")
_LOGGER_DIR = SCRIPT_PATH.parent


def _reason() -> str | None:
    """Return why the smoke should skip, or None if it can run."""
    if not _OPT_IN:
        return "set AORTA_LN_GPU_SMOKE=1 to run the GPU+docker smoke"
    if shutil.which("docker") is None:
        return "docker not on PATH"
    if not (_REPRO_DIR / _SCRIPT_NAME).is_file():
        return f"repro entry not found: {_REPRO_DIR / _SCRIPT_NAME} (set AORTA_LN_REPRO_DIR)"
    return None


skip_no_env = pytest.mark.skipif(_reason() is not None, reason=_reason() or "")


# --- scenarios (mirror run_layer_numerics_all_scenarios.sh) -------------------
# label -> NANLOG_SPEC json. Kept in lockstep with the doc's scenario table.
_SCENARIOS = {
    "s1_watch_block":
        '{"watch":[{"scope":{"types":["AttentionBlock","MLP"]},'
        '"tensors":["input","output","weight","bias","grad"]}],"sample_every":1}',
    "s2_watch_names":
        '{"watch":[{"scope":{"names":["emb_proj"]},"tensors":["input","output"]}],"sample_every":1}',
    "s3_follow_stage":
        '{"follow":[{"tensor":"embedding_features","at":"stage","bounds":[0,60]}],"sample_every":1}',
    "s4_follow_stride":
        '{"follow":[{"tensor":"embedding_features","at":"stride:8",'
        '"scope":{"names":["emb_proj"]}}],"sample_every":1}',
    "s5_follow_blockedge":
        '{"follow":[{"tensor":"embedding_features","at":"stride:1",'
        '"scope":{"types":["AttentionBlock","MLP"]}}],"sample_every":1}',
    "s6_follow_named":
        '{"follow":[{"tensor":"embedding_features","at":"stride:1",'
        '"scope":{"names":["emb_proj.projections.0","towers.0.layers.1.attn"]}}],"sample_every":1}',
    "s7_grad_umbrella":
        '{"watch":[{"scope":{"types":["Linear"]},"tensors":["output","grad"]}],"sample_every":1}',
    "s8_full_capture":
        '{"watch":[{"scope":{"types":["Linear","LinearProjection","AttentionBlock",'
        '"InteractionLayer","MLP","EmbeddingGate"]},'
        '"tensors":["input","output","weight","bias","grad"]}],'
        '"follow":[{"tensor":"embedding_features","at":"stage","bounds":[0,60]}],'
        '"sample_every":1,"pre_context":5}',
}
# scenarios whose `grad` umbrella must also produce param grads (wgrad/bgrad)
_GRAD_SCENARIOS = {"s7_grad_umbrella", "s8_full_capture"}


def _run_logger(spec: str, out_dir: Path) -> None:
    """Run the repro through the logger in docker with the given NANLOG_SPEC."""
    nanlog_out = out_dir / OUTPUT_SUBDIR
    nanlog_out.mkdir(parents=True, exist_ok=True)
    argv = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--group-add", "video",
        "--security-opt", "seccomp=unconfined", "--shm-size", "8g",
        "-v", f"{_REPRO_DIR}:/repro",
        "-v", f"{_LOGGER_DIR}:/nanlog",
        "-v", f"{nanlog_out.resolve()}:/output_logs",
        "-e", "HIP_VISIBLE_DEVICES=0",
        "-e", "RANK=0", "-e", "WORLD_SIZE=1", "-e", "LOCAL_RANK=0",
        "-e", f"NUM_STEPS={_NUM_STEPS}",
        "-e", "NANLOG_DIR=/output_logs",
        "-e", f"NANLOG_SPEC={spec}",
        _IMAGE,
        "python", "/nanlog/instrument_nan_logger.py", f"/repro/{_SCRIPT_NAME}",
    ]
    log = out_dir / "run.log"
    with open(log, "w") as fh:
        proc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
    assert proc.returncode == 0, f"container exited {proc.returncode}; see {log}"


def _summary(out_dir: Path) -> dict:
    return json.loads((out_dir / OUTPUT_SUBDIR / "summary_rank0.json").read_text())


def _record_count(out_dir: Path) -> int:
    jsonl = out_dir / OUTPUT_SUBDIR / "layers_rank0.jsonl"
    if not jsonl.exists():
        return 0
    return sum(1 for line in jsonl.read_text().splitlines() if line.strip())


@skip_no_env
@pytest.mark.parametrize("label", list(_SCENARIOS))
def test_standalone_scenario(label, tmp_path):
    """Each NANLOG_SPEC scenario runs, applies the spec, and captures records.

    N==0 layer hooks is EXPECTED for a follow-only stage spec (the pipeline follow
    does the capture), so the real gate is: spec_applied and records > 0.
    """
    _run_logger(_SCENARIOS[label], tmp_path)
    smy = _summary(tmp_path)
    assert smy["spec_applied"] is True, f"{label}: spec not applied ({smy.get('spec_error')})"
    assert _record_count(tmp_path) > 0, f"{label}: no records written"
    if label in _GRAD_SCENARIOS:
        # the `grad` umbrella must produce PARAM grads too, not just igrad
        assert smy.get("grad_records_stashed", 0) > 0, \
            f"{label}: grad umbrella captured no param grads (grad_records_stashed=0)"


@skip_no_env
def test_malformed_spec_falls_back_cleanly(tmp_path):
    """A malformed spec must not crash the run; it falls back to flat vars with
    spec_applied=false (the sidecar-never-takes-the-job-down contract)."""
    _run_logger('{"sample_every":0}', tmp_path)   # 0 is invalid -> rejected
    smy = _summary(tmp_path)
    assert smy["spec_present"] is True
    assert smy["spec_applied"] is False
    assert smy["spec_error"]


# --- collect path (mirror run_layer_numerics_collect_smoke.sh) ----------------
_HAVE_AORTA = shutil.which("aorta") is not None
_COLLECT_RECIPE = os.environ.get("AORTA_LN_COLLECT_RECIPE")  # NANLOG_SPEC collect recipe
_COLLECT_SIDECAR = os.environ.get("AORTA_LN_COLLECT_SIDECAR")


@skip_no_env
@pytest.mark.skipif(not _HAVE_AORTA, reason="aorta CLI not on PATH (run from an aorta-internal checkout)")
@pytest.mark.skipif(
    not (_COLLECT_RECIPE and _COLLECT_SIDECAR),
    reason="set AORTA_LN_COLLECT_RECIPE and AORTA_LN_COLLECT_SIDECAR to run the collect-path smoke",
)
def test_collect_path_produces_artifacts(tmp_path):
    """The recipe `collect:` -> aorta sweep -> recom_repro opt-in path produces a
    layer_numerics artifact set per cell/trial, with spec_applied=true."""
    log = tmp_path / "sweep.log"
    argv = ["aorta", "sweep", "run", "--recipe", _COLLECT_RECIPE,
            "--mitigations-file", _COLLECT_SIDECAR, "--output", str(tmp_path), "-v"]
    with open(log, "w") as fh:
        proc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
    assert proc.returncode == 0, f"aorta sweep exited {proc.returncode}; see {log}"

    summaries = list(tmp_path.glob(f"**/{OUTPUT_SUBDIR}/summary_rank0.json"))
    assert summaries, (
        "no layer_numerics artifacts produced -- the workload may not have opted "
        "into the collector (check recom_repro _collect.py)"
    )
    for sum_path in summaries:
        smy = json.loads(sum_path.read_text())
        jsonl = sum_path.parent / "layers_rank0.jsonl"
        recs = sum(1 for line in jsonl.read_text().splitlines() if line.strip()) if jsonl.exists() else 0
        assert recs > 0, f"{sum_path.parent}: no records written"
        assert smy["spec_applied"] is True, \
            f"{sum_path.parent}: spec_applied={smy.get('spec_applied')} ({smy.get('spec_error')})"
