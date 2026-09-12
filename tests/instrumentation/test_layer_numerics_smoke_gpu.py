"""GPU + docker smoke tests for the layer_numerics logger.

Native pytest that exercises the REAL path the unit tests cannot: a torchrec
``DistributedModelParallel`` model under ROCm, run inside a docker image, with the
logger driven by ``NANLOG_SPEC``. This validates that a spec actually attaches,
applies, and captures records end to end.

This is a GENERIC harness -- it names no specific model, image, or workload. You
supply the model/image via environment variables; nothing runs (and nothing is
documented) unless you do. A normal CPU/CI run self-skips, so this never breaks
public CI and carries no private assumptions.

Required environment (all must be set, or the tests skip):

  - ``AORTA_LN_GPU_SMOKE=1``    -- explicit opt-in (the docker run is heavy).
  - ``AORTA_LN_IMAGE``          -- docker image to run in. MUST be digest-pinned
                                   (``name@sha256:...``) so a gate is reproducible.
  - ``AORTA_LN_REPRO_DIR``      -- host dir containing your entry script; bind-mounted.
  - ``AORTA_LN_SCRIPT_NAME``    -- entry script filename inside that dir.

Optional:

  - ``AORTA_LN_NUM_STEPS``      -- steps to run (default "20").
  - ``AORTA_LN_SCOPE_NAMES``    -- comma path-substrings for the name-scoped
                                   scenarios; if unset, those scenarios skip.
  - ``AORTA_LN_SCOPE_TYPES``    -- comma module class names for the type-scoped
                                   scenarios; if unset, those scenarios skip.
  - ``AORTA_LN_FOLLOW_TENSOR``  -- batch attribute to follow (default
                                   "embedding_features", the torchrec convention).
  - ``AORTA_LN_COLLECT_RECIPE`` / ``AORTA_LN_COLLECT_SIDECAR`` -- enable the
                                   collect-path test (needs the ``aorta`` CLI).

Docker requires ``/dev/kfd`` (ROCm), so this runs only on a Linux GPU host.

Run example (values are yours to provide -- none are baked in here):

    AORTA_LN_GPU_SMOKE=1 \
    AORTA_LN_IMAGE=<registry/image@sha256:...> \
    AORTA_LN_REPRO_DIR=</path/to/entry/dir> \
    AORTA_LN_SCRIPT_NAME=<entry.py> \
    AORTA_LN_SCOPE_TYPES=Linear \
      pytest tests/instrumentation/test_layer_numerics_smoke_gpu.py -v -m gpu
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

# --- config (all from env; NO defaults that name any model/image/workload) ---
_OPT_IN = os.environ.get("AORTA_LN_GPU_SMOKE") == "1"
_IMAGE = os.environ.get("AORTA_LN_IMAGE", "")
_REPRO_DIR = Path(os.environ["AORTA_LN_REPRO_DIR"]) if os.environ.get("AORTA_LN_REPRO_DIR") else None
_SCRIPT_NAME = os.environ.get("AORTA_LN_SCRIPT_NAME", "")
_NUM_STEPS = os.environ.get("AORTA_LN_NUM_STEPS", "20")
_FOLLOW_TENSOR = os.environ.get("AORTA_LN_FOLLOW_TENSOR", "embedding_features")
_SCOPE_NAMES = [s for s in os.environ.get("AORTA_LN_SCOPE_NAMES", "").split(",") if s.strip()]
_SCOPE_TYPES = [s for s in os.environ.get("AORTA_LN_SCOPE_TYPES", "").split(",") if s.strip()]
_LOGGER_DIR = SCRIPT_PATH.parent


def _skip_reason() -> str | None:
    """Why to SKIP (not opted in / no docker). Once the operator opts in with docker
    present, we do NOT skip on bad config -- that FAILS (see _require_valid_config),
    so a misconfigured gate cannot pass as a green skip."""
    if not _OPT_IN:
        return "set AORTA_LN_GPU_SMOKE=1 to run the GPU+docker smoke"
    if shutil.which("docker") is None:
        return "docker not on PATH"
    return None


def _require_valid_config() -> None:
    """After opt-in, invalid/missing required config is a FAILURE, not a skip -- so a
    regression gate can't be misconfigured (e.g. an unpinned image) and pass."""
    if not _IMAGE:
        pytest.fail("AORTA_LN_IMAGE is required (a digest-pinned docker image)")
    if "@sha256:" not in _IMAGE:
        pytest.fail(f"AORTA_LN_IMAGE must be digest-pinned (name@sha256:...), got {_IMAGE!r}")
    if _REPRO_DIR is None or not _SCRIPT_NAME:
        pytest.fail("AORTA_LN_REPRO_DIR and AORTA_LN_SCRIPT_NAME are required")
    if not (_REPRO_DIR / _SCRIPT_NAME).is_file():
        pytest.fail(f"entry not found: {_REPRO_DIR / _SCRIPT_NAME}")


# Evaluate the skip gate once so the condition and the message can't disagree.
_SKIP_REASON = _skip_reason()
skip_no_env = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# --- scenarios ----------------------------------------------------------------
# Each entry: (needs, spec_builder). `needs` names which optional env a scenario
# requires ("names"/"types"/None); a scenario whose scope env is unset self-skips,
# so nothing hardcodes a model's module names.
def _watch(tensors, scope):
    return json.dumps({"watch": [{"scope": scope, "tensors": tensors}], "sample_every": 1})


def _follow(*, stages=False, scope=None, stride=None, bounds=None, stage_reads=False):
    entry = {"tensor": _FOLLOW_TENSOR}
    if stages:
        entry["stages"] = True
    if scope:
        entry["scope"] = scope
    if stride is not None:
        entry["stride"] = stride
    if bounds:
        entry["bounds"] = bounds
    if stage_reads:
        entry["stage_reads"] = True
    return json.dumps({"follow": [entry], "sample_every": 1})


def _scenarios() -> dict:
    names_scope = {"names": _SCOPE_NAMES}
    types_scope = {"types": _SCOPE_TYPES}
    out = {}
    # watch by type (needs AORTA_LN_SCOPE_TYPES)
    if _SCOPE_TYPES:
        out["watch_types_all_tensors"] = (
            _watch(["input", "output", "weight", "bias", "grad"], types_scope), True)
        out["watch_types_grad_umbrella"] = (_watch(["output", "grad"], types_scope), True)
        out["follow_stride_types"] = (
            _follow(scope=types_scope, stride=1), False)
    # watch by name (needs AORTA_LN_SCOPE_NAMES)
    if _SCOPE_NAMES:
        out["watch_names_io"] = (_watch(["input", "output"], names_scope), False)
        out["follow_stride_names"] = (
            _follow(scope=names_scope, stride=8), False)
    # follow at pipeline stages (no scope needed)
    out["follow_stage"] = (
        _follow(stages=True, bounds=[0, 60]), False)
    out["follow_stage_reads"] = (
        _follow(stages=True, stage_reads=True, bounds=[0, 60]), False)
    return out


_SCENARIOS = _scenarios()


def _run_logger(spec: str, out_dir: Path) -> None:
    """Run the entry through the logger in docker with the given NANLOG_SPEC."""
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
@pytest.mark.parametrize("label", list(_SCENARIOS) or ["_none"])
def test_standalone_scenario(label, tmp_path):
    """Each NANLOG_SPEC scenario runs, applies the spec, and captures records.

    N==0 layer hooks is EXPECTED for a follow-only stage spec (the pipeline follow
    does the capture), so the real gate is: spec_applied and records > 0.
    """
    if label == "_none":
        pytest.skip("no scenario scope provided (set AORTA_LN_SCOPE_TYPES/NAMES)")
    _require_valid_config()
    spec, is_grad = _SCENARIOS[label]
    _run_logger(spec, tmp_path)
    smy = _summary(tmp_path)
    assert smy["spec_applied"] is True, f"{label}: spec not applied ({smy.get('spec_error')})"
    assert _record_count(tmp_path) > 0, f"{label}: no records written"
    if label == "follow_stage_reads":
        assert smy["stage_reads"] is True
        assert smy["stage_reads_active"] is True
        assert smy["stage_evidence_valid"] is True, smy.get("stage_skip_reasons")
        counts = smy.get("stage_phase_counts", {})
        for phase in ("copy", "sparse_start", "sparse_wait", "forward"):
            assert counts.get(phase, {}).get("trusted", 0) > 0, \
                f"{label}: no trusted observation for {phase}: {counts}"
    if is_grad:
        # the `grad` umbrella must produce PARAM grads too, not just igrad
        assert smy.get("grad_records_stashed", 0) > 0, \
            f"{label}: grad umbrella captured no param grads (grad_records_stashed=0)"


@skip_no_env
def test_malformed_spec_falls_back_cleanly(tmp_path):
    """A malformed spec must not crash the run; it falls back to flat vars with
    spec_applied=false (the sidecar-never-takes-the-job-down contract)."""
    _require_valid_config()
    _run_logger('{"sample_every":0}', tmp_path)   # 0 is invalid -> rejected
    smy = _summary(tmp_path)
    assert smy["spec_present"] is True
    assert smy["spec_applied"] is False
    assert smy["spec_error"]


# --- collect path -------------------------------------------------------------
_HAVE_AORTA = shutil.which("aorta") is not None
_COLLECT_RECIPE = os.environ.get("AORTA_LN_COLLECT_RECIPE")
_COLLECT_SIDECAR = os.environ.get("AORTA_LN_COLLECT_SIDECAR")


def _recipe_uses_spec(recipe_path: str) -> bool:
    """True if the recipe's layer_numerics collector options set NANLOG_SPEC, checked
    at both recipe-level and per-cell `collect:` blocks. Parses the YAML so a comment
    mentioning NANLOG_SPEC doesn't count. If PyYAML is unavailable it FAILS the test
    (rather than substring-guessing) -- set AORTA_LN_COLLECT_EXPECTS_SPEC=1/0 to state
    the expectation explicitly and skip the parse."""
    try:
        import yaml
    except Exception:
        pytest.fail("PyYAML not available to parse the collect recipe; install it or "
                    "set AORTA_LN_COLLECT_EXPECTS_SPEC=1/0 to state the expectation")

    def _collect_has_spec(collect) -> bool:
        opts = collect.get("layer_numerics") if isinstance(collect, dict) else None
        return isinstance(opts, dict) and "NANLOG_SPEC" in opts

    doc = yaml.safe_load(Path(recipe_path).read_text(encoding="utf-8")) or {}
    if _collect_has_spec(doc.get("collect", {})):
        return True
    for cell in doc.get("cells", []) or []:
        if isinstance(cell, dict) and _collect_has_spec(cell.get("collect", {})):
            return True
    return False


@skip_no_env
@pytest.mark.skipif(not _HAVE_AORTA, reason="aorta CLI not on PATH")
@pytest.mark.skipif(
    not (_COLLECT_RECIPE and _COLLECT_SIDECAR),
    reason="set AORTA_LN_COLLECT_RECIPE and AORTA_LN_COLLECT_SIDECAR to run the collect-path smoke",
)
def test_collect_path_produces_artifacts(tmp_path):
    """The recipe `collect:` -> aorta sweep -> workload opt-in path produces a
    layer_numerics artifact set for every trial that opted in.

    spec_applied is required ONLY when the recipe actually uses NANLOG_SPEC; a
    flat-var collect recipe correctly has spec_applied=false with real records.
    """
    _require_valid_config()
    # A flat-var recipe legitimately produces spec_applied=false; only require
    # spec_applied=true when the recipe under test actually sets NANLOG_SPEC. Detect
    # that by parsing the recipe's collect.layer_numerics block (top-level + per-cell),
    # NOT a substring match -- a comment mentioning NANLOG_SPEC must not flip it. An
    # explicit AORTA_LN_COLLECT_EXPECTS_SPEC=1/0 overrides the auto-detection.
    override = os.environ.get("AORTA_LN_COLLECT_EXPECTS_SPEC")
    if override is not None:
        expects_spec = override == "1"
    else:
        expects_spec = _recipe_uses_spec(_COLLECT_RECIPE)

    log = tmp_path / "sweep.log"
    argv = ["aorta", "sweep", "run", "--recipe", _COLLECT_RECIPE,
            "--mitigations-file", _COLLECT_SIDECAR, "--output", str(tmp_path), "-v"]
    with open(log, "w") as fh:
        proc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, timeout=1800)
    assert proc.returncode == 0, f"aorta sweep exited {proc.returncode}; see {log}"

    # Cross-check per trial: a trial whose config requested the collector MUST have
    # its OWN artifacts. The collector writes to _aorta_collect_dir, which is
    # "<results>/<trial_stem>", so the artifacts live at
    # "<trial_json_dir>/<trial_stem>/layer_numerics/". Scoping to that exact dir (not
    # a recursive glob from the parent) is what stops trial 0's output from making
    # trial 1 look covered.
    trial_jsons = list(tmp_path.glob("**/trial_*.json"))
    checked = 0
    for tj in trial_jsons:
        try:
            cfg = json.loads(tj.read_text())
        except Exception:
            continue
        cfg = cfg.get("config", cfg)
        if not cfg.get("_aorta_collect"):
            continue
        checked += 1
        artifact_dir = tj.parent / tj.stem / OUTPUT_SUBDIR
        summary = artifact_dir / "summary_rank0.json"
        assert summary.is_file(), \
            f"{tj.name}: opted into the collector but produced no artifacts at {artifact_dir}"
        smy = json.loads(summary.read_text())
        jsonl = artifact_dir / "layers_rank0.jsonl"
        recs = sum(1 for line in jsonl.read_text().splitlines() if line.strip()) if jsonl.exists() else 0
        assert recs > 0, f"{tj.name}: no records written"
        if expects_spec:
            assert smy["spec_present"] is True and smy["spec_applied"] is True, \
                f"{tj.name}: NANLOG_SPEC recipe but spec_applied={smy.get('spec_applied')} " \
                f"({smy.get('spec_error')})"

    assert checked > 0, (
        "no trial opted into the collector -- did the workload thread _aorta_collect, "
        "and does the recipe set collect: layer_numerics?"
    )
