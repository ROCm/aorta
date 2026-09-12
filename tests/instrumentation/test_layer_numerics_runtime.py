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

import gc
import importlib.util
import json
import os
import sys
import types

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


def _publish_fake_torchrec(monkeypatch, pipeline_cls) -> None:
    root = types.ModuleType("torchrec")
    distributed = types.ModuleType("torchrec.distributed")
    train_pipeline = types.ModuleType("torchrec.distributed.train_pipeline")
    train_pipeline.TrainPipelineSparseDist = pipeline_cls
    distributed.train_pipeline = train_pipeline
    root.distributed = distributed
    monkeypatch.setitem(sys.modules, "torchrec", root)
    monkeypatch.setitem(sys.modules, "torchrec.distributed", distributed)
    monkeypatch.setitem(
        sys.modules, "torchrec.distributed.train_pipeline", train_pipeline)


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
    # installing the real TorchRec wrappers, so simulate that install: mark them
    # installed and set the mint phase to the earliest stage this test fires (`copy`).
    # Batch-id minting keys off _pipeline_installed + _pipeline_mint_phase (what
    # actually ran), not the requested _PIPELINE.
    nl._pipeline_installed = True
    nl._pipeline_mint_phase = "copy"

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


def test_v6_pipeline_wrappers_resolve_context_and_keep_batch_external(
    monkeypatch, tmp_path_factory
):
    """Exercise the real wrappers with modern `(batch, context)` / wait(context)."""
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self, value):
            self.embedding_features = [torch.full((2, 3), value)]

    class FakePipeline:
        def __init__(self):
            # None explicitly means these stages use the current/default stream.
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter), Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            batch, context = self.copy_batch_to_gpu(data_iter)
            self.start_sparse_data_dist(batch=batch, context=context)
            self.wait_sparse_data_dist(context=context)
            nl._root_pre_hook(None, (batch,))
            return batch

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    batch = Batch(5.0)
    pipeline = FakePipeline()
    result = pipeline.progress(iter([batch]))
    assert result is batch
    nl._write_summary()

    records = [
        r for r in _records(nl)
        if r["role"] == "track" and r["layer_name"] == "embedding_features[0]"
    ]
    by_phase = {r["phase"]: r for r in records}
    assert {"copy", "sparse_start", "sparse_wait", "forward"} <= set(by_phase)
    batch_ids = {by_phase[p]["batch_id"] for p in (
        "copy", "sparse_start", "sparse_wait", "forward")}
    assert len(batch_ids) == 1 and None not in batch_ids
    assert all(by_phase[p]["pipeline_tick"] == 1 for p in by_phase)
    assert not hasattr(batch, "_nanlog_batch_id")
    observer = nl._get_pipeline_observer(pipeline)
    assert (
        not observer.tokens
        and not observer.by_batch
        and not observer.by_context
        and not observer.by_tensor_signature
    )
    smy = _summary(nl)
    assert smy["stage_phase_counts"]["sparse_wait"]["inline_cpu"] == 1


def test_v6_post_step_without_compute_token_fails_closed(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_POST_STEP": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter), Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            # This is a prefetch-only tick: the copied batch remains pre-forward.
            batch, context = self.copy_batch_to_gpu(data_iter)
            self.start_sparse_data_dist(batch, context)
            self.wait_sparse_data_dist(context)
            return batch

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    FakePipeline().progress(iter([Batch()]))
    nl._write_summary()

    post = [r for r in _records(nl) if r["phase"] == "post_step"]
    assert not post
    smy = _summary(nl)
    assert smy["post_step"] is True
    assert smy["post_step_active"] is False
    assert smy["post_step_executions"] == 1
    assert smy["post_step_valid"] is False
    assert smy["stage_evidence_valid"] is False
    assert smy["stage_skip_reasons"]["post_step:compute_batch_unresolved"] == 1


def test_v6_post_step_links_prefetch_buffer_to_compute_batch(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_POST_STEP": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self, value):
            self.embedding_features = [torch.full((2, 3), value)]

    class FakePipeline:
        def __init__(self, current, future):
            self._memcpy_stream = None
            self._data_dist_stream = None
            self.current = current
            self.future = future

        def copy_batch_to_gpu(self, _data_iter):
            return self.future, Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            self.copy_batch_to_gpu(data_iter)
            nl._root_pre_hook(None, (self.current,))
            return self.current

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    current, future = Batch(1.0), Batch(2.0)
    pipeline = FakePipeline(current, future)
    observer = nl._get_pipeline_observer(pipeline)
    current_token = observer.token_for_batch(current, create=True)
    pipeline.progress(iter(()))
    nl._write_summary()

    post = [r for r in _records(nl) if r["phase"] == "post_step"]
    assert post
    assert all(r["compute_batch_id"] == current_token.batch_id for r in post)
    assert all(r["batch_id"] != current_token.batch_id for r in post)


def test_v6_post_step_requires_an_emitted_prefetch_observation(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_POST_STEP": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    class FakePipeline:
        def __init__(self, current):
            self._memcpy_stream = None
            self._data_dist_stream = None
            self.current = current

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter)

        def start_sparse_data_dist(self, batch):
            return batch

        def wait_sparse_data_dist(self, batch):
            return batch

        def progress(self, _data_iter):
            nl._root_pre_hook(None, (self.current,))
            return self.current

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    current = Batch()
    pipeline = FakePipeline(current)
    nl._get_pipeline_observer(pipeline).token_for_batch(current, create=True)
    pipeline.progress(iter(()))
    nl._write_summary()
    smy = _summary(nl)
    assert smy["post_step_executions"] == 1
    assert smy["post_step_observations"] == 0
    assert smy["post_step_active"] is False
    assert smy["post_step_valid"] is False
    assert smy["stage_evidence_valid"] is False


def test_v6_unmet_flat_post_step_prerequisites_fail_closed(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {"NANLOG_POST_STEP": "1"},
        monkeypatch,
        tmp_path_factory,
    )
    nl._write_summary()
    smy = _summary(nl)
    assert smy["post_step_requested"] is True
    assert smy["post_step"] is False
    assert smy["post_step_prerequisites_valid"] is False
    assert smy["post_step_valid"] is False
    assert smy["stage_evidence_valid"] is False


def test_v6_patches_concrete_subclass_overrides_and_autodiscovers_batch(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "missing_named_attr",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self):
            self.payload = [torch.ones(2, 3)]

    class BasePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter), Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            raise AssertionError("subclass override should run")

    class CustomPipeline(BasePipeline):
        def wait_sparse_data_dist(self):
            return self._context

        def progress(self, data_iter):
            batch, context = self.copy_batch_to_gpu(data_iter)
            self._context = context
            self.start_sparse_data_dist(batch, context)
            self.wait_sparse_data_dist()
            nl._root_pre_hook(None, (batch,))
            return batch

    _publish_fake_torchrec(monkeypatch, BasePipeline)
    nl._install_pipeline_hook()
    CustomPipeline().progress(iter([Batch()]))
    nl._write_summary()
    phases = {r["phase"] for r in _records(nl) if r["role"] == "track"}
    assert {"copy", "sparse_start", "sparse_wait", "forward"} <= phases


def test_v6_nested_progress_override_counts_one_tick(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_TRACK_ATTR": "embedding_features"},
        monkeypatch,
        tmp_path_factory,
    )

    class BasePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter)

        def start_sparse_data_dist(self, batch):
            return batch

        def wait_sparse_data_dist(self, batch):
            return batch

        def progress(self, _data_iter):
            return "base"

    class CustomPipeline(BasePipeline):
        def progress(self, data_iter):
            return super().progress(data_iter)

    _publish_fake_torchrec(monkeypatch, BasePipeline)
    nl._install_pipeline_hook()
    pipeline = CustomPipeline()
    assert pipeline.progress(iter(())) == "base"
    assert nl._get_pipeline_observer(pipeline).tick == 1


def test_v6_outermost_stage_override_defines_observation_boundary(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.zeros(4)]

    class BasePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter), Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return self.copy_batch_to_gpu(data_iter)

    class CustomPipeline(BasePipeline):
        def copy_batch_to_gpu(self, data_iter):
            batch, context = super().copy_batch_to_gpu(data_iter)
            batch.embedding_features[0].fill_(float("nan"))
            return batch, context

    _publish_fake_torchrec(monkeypatch, BasePipeline)
    nl._install_pipeline_hook()
    CustomPipeline().copy_batch_to_gpu(iter([Batch()]))
    nl._write_summary()
    copy = next(r for r in _records(nl) if r["phase"] == "copy")
    assert copy["nan_count"] == 4


def test_v6_copy_exhaustion_does_not_reuse_previous_queue_tail(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None
            self.batches = [Batch()]
            self.contexts = [Context()]

        def copy_batch_to_gpu(self, _data_iter):
            return None, Context()  # explicit exhaustion, not an in-place copy

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return self.copy_batch_to_gpu(data_iter)

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    pipeline = FakePipeline()
    pipeline.copy_batch_to_gpu(iter(()))
    nl._write_summary()
    assert not _records(nl)
    smy = _summary(nl)
    assert smy["stage_skip_reasons"] == {}
    assert smy["stage_phase_counts"]["copy"]["exhausted"] == 1


def test_v6_inplace_copy_success_and_exhaustion_are_distinguished(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None
            self.batches = []
            self.contexts = []

        def inplace_copy_batch_to_gpu(self, batch, context):
            if batch is None:
                return None, context
            self.batches.append(batch)
            self.contexts.append(context)
            return None

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return None

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    pipeline = FakePipeline()
    context = Context()
    pipeline.inplace_copy_batch_to_gpu(Batch(), context)
    # Current TorchRec uses this tuple to signal finite-pipeline exhaustion.
    pipeline.inplace_copy_batch_to_gpu(None, Context())
    nl._write_summary()
    copy = [r for r in _records(nl) if r["phase"] == "copy"]
    assert copy
    assert _summary(nl)["stage_skip_reasons"] == {}
    assert _summary(nl)["stage_phase_counts"]["copy"]["exhausted"] == 1


def test_v6_copy_pair_unpacks_container_batch_before_context(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "missing_named_attr",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, _data_iter):
            batch = (torch.ones(2, 3), torch.zeros(2, 3))
            return batch, Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return self.copy_batch_to_gpu(data_iter)

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    FakePipeline().copy_batch_to_gpu(iter(()))
    nl._write_summary()
    copy = [r for r in _records(nl) if r["phase"] == "copy"]
    assert len(copy) == 2
    assert {r["layer_name"] for r in copy} == {"batch[0]", "batch[1]"}
    assert len({r["batch_id"] for r in copy}) == 1


def test_v6_threaded_copy_future_is_invalid_not_stale_queue_reuse(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Future:
        def then(self, callback):
            return callback

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None
            self.batches = [Batch()]
            self.contexts = [Context()]

        def copy_batch_to_gpu(self, _data_iter):
            return Future(), Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return self.copy_batch_to_gpu(data_iter)

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    pipeline = FakePipeline()
    _future, context = pipeline.copy_batch_to_gpu(iter(()))
    materialized = Batch()
    pipeline.start_sparse_data_dist(materialized, context)
    pipeline.wait_sparse_data_dist(context)
    nl._root_pre_hook(None, (materialized,))
    nl._write_summary()
    records = _records(nl)
    assert not [r for r in records if r["phase"] == "copy"]
    recovered = [
        r for r in records if r["phase"] in {"sparse_start", "sparse_wait", "forward"}
    ]
    assert recovered
    assert len({r["batch_id"] for r in recovered}) == 1
    smy = _summary(nl)
    assert smy["stage_evidence_valid"] is False
    assert smy["stage_skip_reasons"][
        "copy:threaded_copy_future_unresolved"
    ] == 1


def test_v6_threaded_inplace_future_recovers_at_sparse_start(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Future:
        def then(self, callback):
            return callback

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def inplace_copy_batch_to_gpu(self, _batch, context):
            return Future(), context

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return None

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    pipeline = FakePipeline()
    context = Context()
    pipeline.inplace_copy_batch_to_gpu(None, context)
    materialized = Batch()
    pipeline.start_sparse_data_dist(materialized, context)
    pipeline.wait_sparse_data_dist(context)
    nl._root_pre_hook(None, (materialized,))
    nl._write_summary()
    recovered = [
        r for r in _records(nl) if r["phase"] in {"sparse_start", "sparse_wait", "forward"}
    ]
    assert recovered and len({r["batch_id"] for r in recovered}) == 1
    assert _summary(nl)["stage_evidence_valid"] is False  # copy was unobserved


def test_v6_rotates_token_when_batch_object_is_reused(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger({}, monkeypatch, tmp_path_factory)

    class Pipeline:
        pass

    class Batch:
        pass

    observer = nl._get_pipeline_observer(Pipeline())
    batch = Batch()
    first = observer.token_for_batch(batch, create=True)
    first.forward_seen = True
    second = observer.token_for_batch(batch, create=True)
    assert second.batch_id != first.batch_id
    assert observer.token_for_batch(batch) is second


def test_v6_copy_wrapper_rotates_forwarded_reused_batch(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter), Context()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return self.copy_batch_to_gpu(data_iter)

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    pipeline = FakePipeline()
    observer = nl._get_pipeline_observer(pipeline)
    batch = Batch()
    previous = observer.token_for_batch(batch, create=True)
    previous.forward_seen = True
    pipeline.copy_batch_to_gpu(iter([batch]))
    current = observer.token_for_exact_batch(batch)
    assert current is not None and current.batch_id != previous.batch_id


def test_v6_resolves_rewrapped_batch_by_tracked_tensor_identity(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {"NANLOG_TRACK_ATTR": "embedding_features"},
        monkeypatch,
        tmp_path_factory,
    )

    class Pipeline:
        pass

    class Batch:
        def __init__(self, tensor):
            self.embedding_features = [tensor]

    observer = nl._get_pipeline_observer(Pipeline())
    tensor = torch.ones(2, 3)
    original = Batch(tensor)
    rewrapped = Batch(tensor)
    token = observer.token_for_batch(original, create=True)
    assert observer.token_for_batch(rewrapped, create=False) is token


def test_v6_pipeline_reset_retires_unforwarded_tokens(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = None
            self._data_dist_stream = None

        def copy_batch_to_gpu(self, data_iter):
            return next(data_iter), object()

        def start_sparse_data_dist(self, batch, context):
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            return self.copy_batch_to_gpu(data_iter)

        def reset(self):
            return None

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(2, 3)]

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    pipeline = FakePipeline()
    # object() is not recognized as a context, so create the token directly to isolate
    # reset lifecycle behavior.
    observer = nl._get_pipeline_observer(pipeline)
    token = observer.token_for_batch(Batch(), create=True)

    class IncompleteEvent:
        def query(self):
            return False

    record = {"observation_status": "scheduled"}
    observation = {
        "phase": "copy",
        "done_event": IncompleteEvent(),
        "records": [record],
        "closed": False,
    }
    token.open_observations.append(observation)
    assert observer.tokens
    pipeline.reset()
    assert not observer.tokens and not observer.by_batch
    assert record["observation_status"] == "overlapped_next_stage"
    assert record["closed_by_phase"] == "reset"


def test_v6_summary_keeps_tick_counters_after_pipeline_gc(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger({}, monkeypatch, tmp_path_factory)

    class Pipeline:
        pass

    pipeline = Pipeline()
    key = id(pipeline)
    observer = nl._get_pipeline_observer(pipeline)
    for _ in range(20):
        observer.begin_tick()
    assert nl._pipeline_observers_created == 1
    assert nl._pipeline_tick_high_watermark == 20

    del pipeline
    gc.collect()
    assert key not in nl._pipeline_observers

    nl._write_summary()
    smy = _summary(nl)
    assert smy["pipeline_observer_count"] == 1
    assert smy["pipeline_ticks"] == 20


def test_stage_reads_still_capture_all_stages(monkeypatch, tmp_path_factory):
    """With NANLOG_STAGE_READS=1 the copy/sparse/forward stage reads still produce the
    same records (the side stream only changes WHERE the reduction runs, not WHAT is
    captured). The tracked tensors are put on the SAME device the side stream needs
    (cuda when available, else cpu), so the summary assertions below actually reflect
    which path ran -- the side stream only engages for a CUDA tensor (t.is_cuda), so a
    CPU tensor on a CUDA box would use the explicitly labelled CPU-only inline path, not
    exercise CUDA timing evidence. On a CPU box capture must remain functional."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_STAGE_READS": "1",
         "NANLOG_TRACK_ATTR": "embedding_features",
         "NANLOG_BOUNDS": "embedding_features:0:60", "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    assert nl._STAGE_READS is True
    nl._pipeline_installed = True
    nl._pipeline_mint_phase = "copy"

    class Batch:
        def __init__(self):
            self.embedding_features = [
                (torch.rand(4, 8, device=dev) * 60.0) for _ in range(3)]

    for _ in range(2):
        b = Batch()
        source = torch.cuda.current_stream() if dev == "cuda" else None
        nl._checkpoint(
            b, "copy", source_stream=source, source_kind="test_producer")
        nl._checkpoint(
            b, "sparse_start", source_stream=source, source_kind="test_producer")
        nl._checkpoint(
            b, "sparse_wait", source_stream=source, source_kind="test_producer")
        nl._root_pre_hook(None, (b,))
    nl._root_pre_hook(None, (Batch(),))
    nl._write_summary()

    track = [r for r in _records(nl) if r["role"] in ("track", "sparse")]
    phases = {r["phase"] for r in track}
    assert {"copy", "sparse_start", "sparse_wait", "forward"} <= phases
    # stats came through regardless of which stream the reduction ran on
    assert all("nan_count" in r for r in track)

    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["stage_reads"] is True
    # With CUDA tensors on a CUDA box the side stream engages; on CPU it falls back to
    # inline. Both are honestly reported.
    if torch.cuda.is_available():
        assert smy["stage_reads_active"] is True
        assert smy["follow_mode"] == "stage_wrappers_side_read"
        assert smy["stage_read_count"] > 0
    else:
        assert smy["stage_reads_active"] is False
        assert smy["stage_evidence_valid"] is False
        assert smy["follow_mode"] == "stage_wrappers_side_read_invalid"


def test_v6_forward_entry_stats_precede_model_body_mutation(
    monkeypatch, tmp_path_factory
):
    """Forward entry is intentionally compute-stream ordered like Pass A."""
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )
    tensor = torch.zeros(8)

    class Batch:
        def __init__(self):
            self.embedding_features = [tensor]

    nl._checkpoint(
        Batch(),
        "forward",
        source_stream="current",
        source_kind="compute_current",
        relative_slot="forward_entry",
    )
    tensor.fill_(float("nan"))  # represents the model body after the pre-hook
    nl._write_summary()
    forward = next(r for r in _records(nl) if r["phase"] == "forward")
    assert forward["nan_count"] == 0
    assert forward["observation_status"] == "trusted"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for side stream")
def test_stage_reads_drain_reads_side_stream_values_correctly(
    monkeypatch, tmp_path_factory
):
    """BLOCKER regression: the drain must not read the side-stream reduction outputs
    before they are computed. Feed a tensor with a KNOWN nan/oob content, run the stage
    read on the side stream, drain, and assert the emitted counts match the tensor
    exactly -- i.e. the drain's device-side wait_stream ordered the casts after the
    side-stream reductions (a missing/late wait would give wrong or stale counts)."""
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_STAGE_READS": "1",
         "NANLOG_TRACK_ATTR": "embedding_features",
         "NANLOG_BOUNDS": "embedding_features:0:60", "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    assert nl._STAGE_READS is True
    nl._pipeline_installed = True
    nl._pipeline_mint_phase = "copy"

    # Known content: 2 NaNs and 1 out-of-range (100 > 60) among finite values.
    ef = torch.tensor([[float("nan"), float("nan"), 100.0, 5.0]], device="cuda")

    class Batch:
        def __init__(self):
            self.embedding_features = ef

    nl._checkpoint(
        Batch(),
        "copy",
        source_stream=torch.cuda.current_stream(),
        source_kind="test_producer",
    )  # reduction runs on the side stream
    nl._root_pre_hook(None, (Batch(),))  # drain the step (device-side wait then read-back)
    nl._write_summary()

    recs = [r for r in _records(nl) if r["role"] == "track" and r["phase"] == "copy"]
    assert recs, "no copy-stage track record"
    r = recs[0]
    assert r["nan_count"] == 2       # exact -- proves the drain waited for the side stream
    assert r["oob_count"] == 1
    assert nl._stage_reads_active is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >=2 GPUs")
def test_stage_reads_cross_device_ambient_differs_from_stream(
    monkeypatch, tmp_path_factory
):
    """Copilot #297 review (#1/#2): the side stream is created on the tracked tensor's
    device, but the wait_stream calls (context manager + drain) previously used an
    UNPINNED current_stream(). On a multi-GPU rank where the ambient current device
    differs from the stream's device, that orders against the wrong device's stream or
    raises cross-device. This is the real ROCm case (8x MI350X ranks on non-zero
    devices).

    Put the tracked tensor on cuda:1 while the AMBIENT device is cuda:0, run a stage
    read + drain, and assert it neither raises nor mis-reports -- i.e. the device-pinned
    waits ordered correctly across the device mismatch."""
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_STAGE_READS": "1",
         "NANLOG_TRACK_ATTR": "embedding_features",
         "NANLOG_BOUNDS": "embedding_features:0:60", "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    nl._pipeline_installed = True
    nl._pipeline_mint_phase = "copy"

    # Tensor on device 1; known content (2 NaN, 1 oob).
    ef = torch.tensor([[float("nan"), float("nan"), 100.0, 5.0]], device="cuda:1")

    class Batch:
        def __init__(self):
            self.embedding_features = ef

    # Ambient device deliberately 0 (!= the tensor's device 1) for the whole flow.
    with torch.cuda.device(0):
        with torch.cuda.device(1):
            producer = torch.cuda.current_stream()
        nl._checkpoint(
            Batch(),
            "copy",
            source_stream=producer,
            source_kind="test_producer",
        )
        nl._root_pre_hook(None, (Batch(),))
    nl._write_summary()

    recs = [r for r in _records(nl) if r["role"] == "track" and r["phase"] == "copy"]
    assert recs, "no copy-stage track record"
    r = recs[0]
    assert r["nan_count"] == 2       # correct across the device mismatch -> pin worked
    assert r["oob_count"] == 1
    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["stage_reads_active"] is True
    # the side stream lives on the tensor's device, not the ambient device
    assert nl._stage_stream_device == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"),
    reason="CUDA _sleep required for deterministic producer delay",
)
def test_v6_stage_read_waits_for_delayed_producer_write(
    monkeypatch, tmp_path_factory
):
    """The explicit producer event must catch a write current_stream() could miss."""
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_BOUNDS": "embedding_features:0:60",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )
    nl._pipeline_installed = True
    nl._pipeline_mint_phase = "copy"

    tensor = torch.zeros(128, device="cuda")
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(100_000_000)
        tensor.fill_(float("nan"))

    class Batch:
        def __init__(self):
            self.embedding_features = [tensor]

    nl._checkpoint(
        Batch(),
        "copy",
        source_stream=producer,
        source_kind="_memcpy_stream",
    )
    nl._write_summary()
    copy_records = [r for r in _records(nl) if r["phase"] == "copy"]
    assert copy_records and copy_records[0]["nan_count"] == tensor.numel()
    assert copy_records[0]["observation_status"] == "trusted"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"),
    reason="CUDA _sleep required for deterministic producer delay",
)
def test_v6_wrappers_use_real_memcpy_and_data_dist_streams(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_BOUNDS": "embedding_features:0:60",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )

    class Context:
        pass

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.zeros(64, device="cuda")]

    class FakePipeline:
        def __init__(self):
            self._memcpy_stream = torch.cuda.Stream()
            self._data_dist_stream = torch.cuda.Stream()

        def copy_batch_to_gpu(self, data_iter):
            batch = next(data_iter)
            with torch.cuda.stream(self._memcpy_stream):
                torch.cuda._sleep(100_000_000)
                batch.embedding_features[0].fill_(float("nan"))
            return batch, Context()

        def start_sparse_data_dist(self, batch, context):
            self._data_dist_stream.wait_stream(self._memcpy_stream)
            return context

        def wait_sparse_data_dist(self, context):
            return context

        def progress(self, data_iter):
            batch, context = self.copy_batch_to_gpu(data_iter)
            self.start_sparse_data_dist(batch, context)
            self.wait_sparse_data_dist(context)
            nl._root_pre_hook(None, (batch,))
            return batch

    _publish_fake_torchrec(monkeypatch, FakePipeline)
    nl._install_pipeline_hook()
    FakePipeline().progress(iter([Batch()]))
    nl._write_summary()

    copy_record = next(r for r in _records(nl) if r["phase"] == "copy")
    assert copy_record["nan_count"] == 64
    assert copy_record["source_stream_kind"] == "_memcpy_stream"
    sparse = [r for r in _records(nl) if r["phase"].startswith("sparse")]
    assert sparse and all(r["source_stream_kind"] == "_data_dist_stream" for r in sparse)


def test_v6_root_pre_hook_does_not_drain_current_tick_producer(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger({}, monkeypatch, tmp_path_factory)

    class IncompleteEvent:
        def query(self):
            return False

    record = {"step": 0}
    observation = {
        "phase": "copy",
        "done_event": IncompleteEvent(),
        "records": [record],
        "closed": False,
    }
    record["_stage_observation"] = observation
    nl._pending.append((record, {}, None))

    nl._root_pre_hook(None, None)
    assert nl._step == 1
    assert len(nl._pending) == 1  # incomplete producer evidence was deferred
    nl._step = nl._MAX_STAGE_DEFERRAL_STEPS + 1
    nl._root_pre_hook(None, None)
    assert not nl._pending
    assert nl._stage_skip_reasons["copy:completion_timeout"] == 1


def test_v6_root_pre_hook_drains_all_ready_backlog_steps(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {"NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch,
        tmp_path_factory,
    )

    class Event:
        def __init__(self, ready):
            self.ready = ready

        def query(self):
            return self.ready

    for step, ready in ((0, True), (1, True), (2, False)):
        nl._step = step
        record = nl._stash(
            f"tracked[{step}]", "pipeline", torch.ones(1), role="track")
        observation = {
            "phase": "copy",
            "done_event": Event(ready),
            "records": [record],
            "closed": False,
            "remaining_records": 1,
            "token": None,
        }
        record["_stage_observation"] = observation
        nl._stage_observations.append(observation)

    nl._root_pre_hook(None, None)
    assert [item[0]["step"] for item in nl._pending] == [2]
    assert len(_records(nl)) == 2


def test_v6_marks_read_overlapping_next_stage_without_blocking_producer(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger({}, monkeypatch, tmp_path_factory)

    class Pipeline:
        pass

    class Batch:
        pass

    class IncompleteEvent:
        def query(self):
            return False

    observer = nl._get_pipeline_observer(Pipeline())
    observer.begin_tick()
    batch = Batch()
    token = observer.token_for_batch(batch, create=True)
    record = {"observation_status": "scheduled"}
    observation = {
        "phase": "copy",
        "done_event": IncompleteEvent(),
        "records": [record],
        "closed": False,
    }
    token.open_observations.append(observation)
    nl._close_token_observations(token, "sparse_start")
    assert record["observation_status"] == "overlapped_next_stage"
    assert record["closed_by_phase"] == "sparse_start"


def test_v6_event_query_error_invalidates_evidence(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger({}, monkeypatch, tmp_path_factory)

    class Pipeline:
        pass

    class Batch:
        pass

    class BrokenEvent:
        def query(self):
            raise RuntimeError("event query failed")

    observer = nl._get_pipeline_observer(Pipeline())
    token = observer.token_for_batch(Batch(), create=True)
    record = {"observation_status": "scheduled"}
    token.open_observations.append({
        "phase": "copy",
        "done_event": BrokenEvent(),
        "records": [record],
        "closed": False,
    })

    nl._close_token_observations(token, "sparse_start")
    assert record["observation_status"] == "poll_error"
    assert record["closed_by_phase"] == "sparse_start"
    assert nl._stage_evidence_valid is False
    assert nl._stage_skip_reasons[
        "copy:completion_event_query_error:RuntimeError"
    ] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_v6_unresolved_producer_skips_instead_of_inline_read(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {
            "NANLOG_PIPELINE": "1",
            "NANLOG_STAGE_READS": "1",
            "NANLOG_TRACK_ATTR": "embedding_features",
            "NANLOG_SAMPLE_EVERY": "1",
        },
        monkeypatch,
        tmp_path_factory,
    )
    nl._pipeline_installed = True
    nl._pipeline_mint_phase = "copy"

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.ones(4, device="cuda")]

    assert nl._checkpoint(
        Batch(), "copy", source_stream=None, source_kind="_memcpy_stream:missing"
    ) == 0
    nl._write_summary()
    assert not _records(nl)
    smy = _summary(nl)
    assert smy["stage_evidence_valid"] is False
    assert smy["stage_skip_reasons"]["copy:producer_stream_unresolved"] == 1


def test_v6_summary_requires_complete_trusted_phase_coverage(
    monkeypatch, tmp_path_factory
):
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_STAGE_READS": "1"},
        monkeypatch,
        tmp_path_factory,
    )
    nl._pipeline_installed = True
    for phase in ("copy", "forward"):
        nl._stage_phase_counts[phase]["scheduled"] = 1
        nl._stage_phase_counts[phase]["trusted"] = 1
    nl._write_summary()
    smy = _summary(nl)
    assert smy["stage_evidence_valid"] is False
    assert smy["stage_missing_phases"] == ["sparse_start", "sparse_wait"]


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


def test_pipeline_mints_at_earliest_installed_stage_when_copy_absent(
    monkeypatch, tmp_path_factory
):
    """Copilot PR #296 follow-up: _pipeline_installed can be true even when
    copy_batch_to_gpu was NOT patched (a torchrec API change that keeps only sparse_*).
    Then `copy` never fires, so minting must happen at the EARLIEST stage that WAS
    installed (here sparse_start), not a hardcoded `copy` -- otherwise every stage
    record's batch_id stays null."""
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1", "NANLOG_TRACK_ATTR": "embedding_features",
         "NANLOG_SAMPLE_EVERY": "1"},
        monkeypatch, tmp_path_factory)
    # Simulate an install where copy_batch_to_gpu was absent: only sparse_* patched,
    # so the earliest firing stage -- and thus the mint phase -- is sparse_start.
    nl._pipeline_installed = True
    nl._pipeline_mint_phase = "sparse_start"

    class Batch:
        def __init__(self):
            self.embedding_features = [torch.rand(4, 8) for _ in range(3)]

    for _ in range(2):
        b = Batch()
        # No `copy` checkpoint -- that wrapper was never installed.
        nl._checkpoint(b, "sparse_start")
        nl._checkpoint(b, "sparse_wait")
        nl._root_pre_hook(None, (b,))       # forward-entry checkpoint
    nl._root_pre_hook(None, (Batch(),))     # drain the last step
    nl._write_summary()

    track = [r for r in _records(nl) if r["role"] in ("track", "sparse")]
    # sparse_start (the earliest installed stage) mints a non-null batch_id.
    start_recs = [r for r in track if r["phase"] == "sparse_start"]
    assert start_recs and all(r["batch_id"] is not None for r in start_recs)
    # and the later same-batch stages share it (they read the minted id back).
    wait_recs = [r for r in track if r["phase"] == "sparse_wait"]
    assert wait_recs and all(r["batch_id"] is not None for r in wait_recs)


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
