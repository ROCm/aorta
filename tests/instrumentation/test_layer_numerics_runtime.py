"""CPU runtime tests for the ``instrument_nan_logger`` script itself.

Unlike ``test_layer_numerics.py`` (which covers only the ``build_env`` collector
plumbing), this exercises the logger's actual hook/drain behavior on CPU tensors —
no GPU and no torchrec required. It drives the same entry points the auto-hook would
call (``_attach`` / ``_root_pre_hook`` / ``_checkpoint``) on tiny eager models.

The logger reads all config from ``NANLOG_*`` env vars AT IMPORT, so each test loads
a FRESH copy of the module by path with its own env + throwaway output dir.

Skipped cleanly when torch is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

from aorta.instrumentation.layer_numerics import SCRIPT_PATH

torch = pytest.importorskip("torch")


def _load_logger(env: dict, monkeypatch, tmp_path_factory) -> object:
    """Import a fresh instance of the logger with the given NANLOG_* env applied.

    A unique module name per load avoids sys.modules caching so env changes take
    effect; a per-load output dir keeps the JSONL/summary isolated.
    """
    out = tmp_path_factory.mktemp("nanlog")
    # Clean NANLOG_* slate: the spec front-end derives flat vars into os.environ,
    # which would otherwise leak across loads in this one process (a real run is a
    # fresh process). Mirrors the isolation in test_layer_numerics_spec.py.
    for key in list(os.environ):
        if key.startswith("NANLOG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NANLOG_DIR", str(out))
    monkeypatch.setenv("RANK", "0")
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    # Load by path under a unique name so repeated loads don't collide in sys.modules.
    name = f"_nanlog_{out.name}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._OUT_DIR = out  # attach for the test to read back
    return mod


def _records(mod) -> list:
    jsonl = mod._OUT_DIR / "layers_rank0.jsonl"
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]


def _summary(mod) -> dict:
    return json.loads((mod._OUT_DIR / "summary_rank0.json").read_text())


# ---------------------------------------------------------------------------
# Channels: independently selectable, off unless named
# ---------------------------------------------------------------------------
def test_channels_are_independently_selectable(monkeypatch, tmp_path_factory):
    """NANLOG_CHANNELS=act,input must emit act+input records and NOT igrad/weight."""
    nl = _load_logger(
        {"NANLOG_CHANNELS": "act,input", "NANLOG_WATCH_TYPES": "Linear",
         "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    assert nl._attach(model) > 0
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(2):
        nl._root_pre_hook(None, None)
        opt.zero_grad()
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
    nl._write_summary()
    roles = {r["role"] for r in _records(nl)}
    assert "act" in roles and "input" in roles
    assert "igrad" not in roles and "weight" not in roles and "wgrad" not in roles


def test_default_channels_do_not_include_new_ones(monkeypatch, tmp_path_factory):
    """Bare default channels are act,igrad only — new channels off by default."""
    nl = _load_logger(
        {"NANLOG_WATCH_TYPES": "Linear", "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset({"act", "igrad"})


# ---------------------------------------------------------------------------
# Default-off paths: heavy features stay off with a bare config
# ---------------------------------------------------------------------------
def test_heavy_features_default_off(monkeypatch, tmp_path_factory):
    nl = _load_logger({"NANLOG_WATCH_TYPES": "Linear"}, monkeypatch, tmp_path_factory)
    assert nl._CAPTURE_ADDR is False        # NANLOG_ADDR now defaults off
    assert nl._PIPELINE is False
    assert nl._TRACK_EVERY_LAYER is False
    assert nl._SPARSE is False and nl._SPARSE_HEAVY is False
    assert nl._LOCATE is False and nl._BAD_VALUES is False
    assert nl._BOUNDS_ACTIVE is False


def test_device_stats_handles_non_float_tensors(monkeypatch, tmp_path_factory):
    """Integer/bool tensors (e.g. KJT indices) must not crash _device_stats — it
    cannot fill +/-inf into an int dtype. Reduces directly for non-float."""
    nl = _load_logger({"NANLOG_WATCH_TYPES": "Linear"}, monkeypatch, tmp_path_factory)
    for t in (torch.tensor([1, 2, 3], dtype=torch.int64),
              torch.tensor([True, False, True]),
              torch.tensor([0.0, 1.0, float("nan")])):
        stats = nl._device_stats(t)
        assert int(stats["numel"]) == 3          # no exception, sane result


def test_addr_off_omits_address_fields(monkeypatch, tmp_path_factory):
    nl = _load_logger(
        {"NANLOG_CHANNELS": "act", "NANLOG_WATCH_TYPES": "Linear",
         "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    model = torch.nn.Linear(8, 8)
    nl._attach(model)
    nl._root_pre_hook(None, None)
    model(torch.randn(4, 8))
    nl._write_summary()
    recs = _records(nl)
    assert recs and all("data_ptr" not in r for r in recs)


# ---------------------------------------------------------------------------
# Bounds / OOB detection + summary
# ---------------------------------------------------------------------------
def test_bounds_flag_out_of_range_and_summary(monkeypatch, tmp_path_factory):
    """A value outside [0,60] on a bounded tensor is bad with kind=oob, and the
    summary surfaces first_oob / oob_records / peak range."""
    nl = _load_logger(
        {"NANLOG_CHANNELS": "input", "NANLOG_WATCH_TYPES": "Linear",
         "NANLOG_SAMPLE_EVERY": "1", "NANLOG_BOUNDS": "proj:0:60",
         "NANLOG_BAD_VALUES": "1"},
        monkeypatch, tmp_path_factory)

    class M(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(4, 4)

        def forward(s, x):
            return s.proj(x)

    m = M()
    nl._attach(m)
    for _ in range(2):
        nl._root_pre_hook(None, None)
        m(torch.rand(3, 4) * 50.0)          # in-range
    nl._root_pre_hook(None, None)            # advances _step; the model runs at this step
    bad_step = nl._step
    x = torch.rand(3, 4) * 50.0
    x[0, 0] = -9.0                           # small NEGATIVE: huge threshold can't catch it
    m(x)
    nl._root_pre_hook(None, None)            # drain the bad step
    nl._write_summary()

    smy = _summary(nl)
    assert smy["oob_records"] >= 1
    assert smy["first_oob"] is not None and smy["first_oob"]["step"] == bad_step
    assert smy["first_bad"] and smy["first_bad"]["kind"] == "oob"
    assert smy["peak_finite_min"] == pytest.approx(-9.0)


def test_bounds_are_scoped_per_pattern(monkeypatch, tmp_path_factory):
    """A bound on 'proj' must not flag an unrelated 'other' layer."""
    nl = _load_logger(
        {"NANLOG_CHANNELS": "input", "NANLOG_WATCH_TYPES": "Linear",
         "NANLOG_SAMPLE_EVERY": "1", "NANLOG_BOUNDS": "proj:0:60"},
        monkeypatch, tmp_path_factory)
    assert nl._bound_for("emb.proj.layers.0") == (0.0, 60.0)
    assert nl._bound_for("other.dense") is None


# ---------------------------------------------------------------------------
# Pipeline: forward-stage checkpoint + batch_id, no watched layer needed
# ---------------------------------------------------------------------------
def test_pipeline_forward_stage_and_batch_id(monkeypatch, tmp_path_factory):
    """With NANLOG_PIPELINE=1 (no WATCH_NAMES, no TRACK_EVERY_LAYER), the tracked
    tensors are scanned at copy/sparse_start/sparse_wait AND forward, and one batch's
    records share a batch_id."""
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_TRACK_ATTR": "embedding_features",
         "NANLOG_BOUNDS": "embedding_features:0:60", "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    # This test drives the stage checkpoints (copy/sparse_*) by hand instead of
    # installing the real TorchRec wrappers, so mark them installed -- batch-id
    # minting keys off _pipeline_installed (what actually ran), not the requested
    # _PIPELINE, so `copy` is the id-minting first sighting here.
    nl._pipeline_installed = True

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.rand(4, 8) * 60.0 for _ in range(3)]

    for _ in range(2):
        b = Batch()
        nl._checkpoint(b, "copy")
        nl._checkpoint(b, "sparse_start")
        nl._checkpoint(b, "sparse_wait")
        nl._root_pre_hook(None, (b,))        # forward-stage checkpoint
    nl._root_pre_hook(None, (Batch(),))      # drain the last step
    nl._write_summary()

    recs = _records(nl)
    track = [r for r in recs if r["role"] in ("track", "sparse")]
    phases = {r["phase"] for r in track}
    assert {"copy", "sparse_start", "sparse_wait", "forward"} <= phases
    # forward records must be STAGE records (no per-layer checkpoint field)
    fwd = [r for r in track if r["phase"] == "forward"]
    assert fwd and all("checkpoint" not in r for r in fwd)
    # all 3 tracked tensors present by name
    assert len({r["layer_name"] for r in fwd}) == 3
    # a single copy checkpoint's records share one non-null batch_id
    copy_recs = [r for r in track if r["phase"] == "copy"]
    assert copy_recs and all(r["batch_id"] is not None for r in copy_recs)


def test_pipeline_requested_but_wrappers_not_installed_mints_at_forward(
    monkeypatch, tmp_path_factory
):
    """Degraded case (Copilot PR #296 review): NANLOG_PIPELINE=1 was requested but the
    stage wrappers never installed (torchrec absent / API changed), so `copy` never
    fires. Batch-id minting and the checkpoint counters must key off what ACTUALLY ran
    (_pipeline_installed), not the request (_PIPELINE): the forward-entry checkpoint
    must still mint a non-null batch_id, and it must count as a forward checkpoint, not
    a phantom stage (pipeline) checkpoint."""
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_TRACK_ATTR": "embedding_features",
         "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True             # requested
    assert nl._pipeline_installed is False  # but wrappers never installed (no torchrec)

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.rand(4, 8) for _ in range(3)]

    # Only the forward-entry path runs (the stage wrappers that would call
    # _checkpoint(b, "copy") were never installed).
    for _ in range(2):
        nl._root_pre_hook(None, (Batch(),))
    nl._root_pre_hook(None, (Batch(),))     # drain the last step
    nl._write_summary()

    recs = _records(nl)
    fwd = [r for r in recs if r["role"] == "track" and r["phase"] == "forward"]
    assert fwd, "forward-entry follow wrote no track records in the degraded run"
    # batch_id is minted at forward (not left null just because `copy` never came).
    assert all(r["batch_id"] is not None for r in fwd)

    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    # No phantom stage checkpoints; the forward-entry capture is counted honestly.
    assert smy["pipeline_installed"] is False
    assert smy["pipeline_checkpoints"] == 0
    assert smy["forward_checkpoints"] > 0


def test_pipeline_off_produces_no_track_records(monkeypatch, tmp_path_factory):
    nl = _load_logger(
        {"NANLOG_CHANNELS": "act", "NANLOG_WATCH_TYPES": "Linear",
         "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is False
    m = torch.nn.Linear(4, 4)
    nl._attach(m)
    nl._root_pre_hook(None, (torch.rand(3, 4),))   # must be a no-op for the pipeline path
    m(torch.rand(3, 4))
    nl._write_summary()
    assert all(r["role"] != "track" for r in _records(nl))
