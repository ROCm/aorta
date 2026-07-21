"""Tests for the ``NANLOG_SPEC`` structured-config front-end.

The spec is translated into the flat ``NANLOG_*`` vars the engine already reads
(see ``instrument_nan_logger.py`` -> ``_apply_spec``). These tests load a fresh
copy of the logger per case (config is read at import) and assert the derived
engine state + runtime behavior, plus that the spec WINS over flat vars and that
a malformed spec falls back to the flat path without crashing.

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
    out = tmp_path_factory.mktemp("nanlog")
    # Start from a clean NANLOG_* slate: the spec front-end derives flat vars into
    # os.environ, which would otherwise leak across loads in this one process (a
    # real standalone run is a fresh process, so this mirrors that isolation).
    for key in list(os.environ):
        if key.startswith("NANLOG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NANLOG_DIR", str(out))
    monkeypatch.setenv("RANK", "0")
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    name = f"_nanlog_{out.name}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._OUT_DIR = out
    return mod


def _records(mod) -> list:
    jsonl = mod._OUT_DIR / "layers_rank0.jsonl"
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# watch: scope + tensors -> WATCH_* + CHANNELS
# ---------------------------------------------------------------------------
def test_watch_spec_maps_tensors_to_channels(monkeypatch, tmp_path_factory):
    """`tensors:[input,output,weight]` -> channels input,act,weight; `types` -> WATCH_TYPES."""
    spec = {"watch": [{"scope": {"types": ["Linear"]},
                       "tensors": ["input", "output", "weight"]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._SPEC_PRESENT is True and nl._SPEC_APPLIED is True
    assert nl._WATCH_TYPES == ("Linear",)
    # output->act, input->input, weight->weight
    assert nl._CHANNELS == frozenset({"input", "act", "weight"})


def test_igrad_token_is_activation_input_grad_only(monkeypatch, tmp_path_factory):
    """`igrad` selects ONLY the activation-input grad; `grad` is the wider umbrella."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(
            {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output", "igrad"]}]})},
        monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset({"act", "igrad"})   # no wgrad/bgrad
    assert nl._GRAD_CHANNELS == frozenset()              # no param-grad channels


def test_watch_stride(monkeypatch, tmp_path_factory):
    """`watch[].stride: N` -> NANLOG_WATCH_STRIDE=N; _attach hooks every Nth match."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(
            {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"], "stride": 3}]})},
        monkeypatch, tmp_path_factory)
    assert nl._WATCH_STRIDE == 3
    model = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(7)])
    assert nl._attach(model) == 3   # ceil(7/3): modules 0,3,6
    # the resolved stride is recorded in the summary so a thinned run is auditable
    nl._write_summary()
    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["watch_stride"] == 3


def test_watch_stride_ignored_when_scoped_follow_needs_all_hooks(monkeypatch, tmp_path_factory):
    """A scoped follow re-scans inside the forward hook, so watch[].stride must NOT
    thin the hooks (that would drop follow evidence). The watch stride is ignored
    (falls back to 1) with a warning; the follow re-scan is preserved."""
    spec = {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"], "stride": 4}],
            "follow": [{"tensor": "ef", "scope": {"types": ["MLP"]}, "stride": 1}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is True
    assert nl._WATCH_STRIDE == 1          # watch stride dropped, not honored
    assert nl._TRACK_EVERY_LAYER is True  # follow re-scan still armed


def test_watch_stride_applied_with_stages_only_follow(monkeypatch, tmp_path_factory):
    """A stages-only follow uses no forward hooks, so watch[].stride is safe to apply."""
    spec = {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"], "stride": 4}],
            "follow": [{"tensor": "ef", "stages": True}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._WATCH_STRIDE == 4          # applied: no module-hook follow to protect
    assert nl._TRACK_EVERY_LAYER is False


def test_diagnostics_block(monkeypatch, tmp_path_factory):
    """`diagnostics: [...]` enables the matching flat toggles; unlisted ones stay off."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(
            {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"]}],
             "diagnostics": ["addr", "bad_values", "alloc_snapshot"]})},
        monkeypatch, tmp_path_factory)
    assert nl._CAPTURE_ADDR is True
    assert nl._BAD_VALUES is True
    assert nl._ALLOC_SNAPSHOT is True
    assert nl._LOCATE is False          # not listed
    assert nl._DUMP_TENSOR is False     # not listed


@pytest.mark.parametrize("spec", [
    {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"], "stride": 0}]},   # <1
    {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"], "stride": "3"}]}, # non-int
    {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"], "bogus": 1}]},    # unknown key
    {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"]}], "diagnostics": ["bogus"]},
    {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output"]}], "diagnostics": "addr"},  # not a list
])
def test_new_watch_diag_invalid_rolls_back(spec, monkeypatch, tmp_path_factory):
    """Malformed watch stride / unknown watch key / bad diagnostics roll the spec back."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(spec), "NANLOG_CHANNELS": "act,igrad"},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is False
    assert nl._CHANNELS == frozenset({"act", "igrad"})


def test_watch_spec_names_scope(monkeypatch, tmp_path_factory):
    spec = {"watch": [{"scope": {"names": ["emb_proj"]}, "tensors": ["input", "output"]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._WATCH_NAMES == ("emb_proj",)
    assert nl._CHANNELS == frozenset({"input", "act"})


def test_grad_tensor_maps_to_all_gradient_channels(monkeypatch, tmp_path_factory):
    """`tensors:[grad]` is the umbrella for ALL gradients: activation-input grad
    (igrad) AND parameter grads (wgrad/bgrad) -- so a one-shot NaN hunt gets them."""
    spec = {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["output", "grad"]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset({"act", "igrad", "wgrad", "bgrad"})
    # param-grad channels are recognized as such (drives the optimizer step-hook)
    assert nl._GRAD_CHANNELS == frozenset({"wgrad", "bgrad"})


def test_watch_spec_runtime_capture(monkeypatch, tmp_path_factory):
    """End-to-end: a watch spec actually hooks and writes input/act records."""
    spec = {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["input", "output"]}],
            "sample_every": 1}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    assert nl._attach(model) > 0
    for _ in range(2):
        nl._root_pre_hook(None, None)
        model(torch.randn(4, 8))
    nl._write_summary()
    roles = {r["role"] for r in _records(nl)}
    assert "input" in roles and "act" in roles


# ---------------------------------------------------------------------------
# follow: at:stage -> PIPELINE, bounds -> BOUNDS
# ---------------------------------------------------------------------------
def test_follow_stage_spec(monkeypatch, tmp_path_factory):
    spec = {"follow": [{"tensor": "embedding_features", "stages": True, "bounds": [0, 60]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_ATTR == ("embedding_features",)
    assert nl._BOUNDS_ACTIVE is True
    assert nl._bound_for("embedding_features") == (0.0, 60.0)
    assert nl._TRACK_EVERY_LAYER is False


def test_follow_stride_spec(monkeypatch, tmp_path_factory):
    """follow `scope` + `stride:8` -> TRACK_EVERY_LAYER + stride 8; scope -> WATCH_*."""
    spec = {"follow": [{"tensor": "embedding_features", "scope": {"names": ["emb_proj"]}, "stride": 8}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_EVERY_LAYER is True
    assert nl._TRACK_LAYER_STRIDE == 8
    assert nl._WATCH_NAMES == ("emb_proj",)


# ---------------------------------------------------------------------------
# follow cadence: explicit `stages` + `scope`/`stride` (replaces the `at` overload)
# ---------------------------------------------------------------------------
def test_follow_stages_only(monkeypatch, tmp_path_factory):
    """`stages: true` alone -> pipeline stage scan, no per-module re-scan."""
    spec = {"follow": [{"tensor": "ef", "stages": True, "bounds": [0, 60]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_EVERY_LAYER is False
    assert nl._BOUNDS_ACTIVE is True


def test_follow_default_is_stages_only(monkeypatch, tmp_path_factory):
    """A follow entry with neither `stages` nor `scope` defaults to stages-only."""
    spec = {"follow": [{"tensor": "ef"}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_EVERY_LAYER is False


def test_follow_stages_and_scope(monkeypatch, tmp_path_factory):
    """`stages:true` + `scope` -> BOTH the pipeline stages AND a per-module re-scan
    at every module in scope (stride defaults to 1)."""
    spec = {"follow": [{"tensor": "ef", "stages": True,
                        "scope": {"types": ["MLP"]}, "bounds": [0, 60]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_EVERY_LAYER is True
    assert nl._TRACK_LAYER_STRIDE == 1
    assert nl._WATCH_TYPES == ("MLP",)


def test_follow_scope_with_stride(monkeypatch, tmp_path_factory):
    """`scope` + `stride:N` -> re-scan every Nth module in scope."""
    spec = {"follow": [{"tensor": "ef", "scope": {"names": ["emb"]}, "stride": 8}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._TRACK_EVERY_LAYER is True
    assert nl._TRACK_LAYER_STRIDE == 8
    assert nl._WATCH_NAMES == ("emb",)


# ---------------------------------------------------------------------------
# follow pipeline:false -> forward/block follow WITHOUT the stage wrappers
# (timing-safe mode for a race the copy/sparse wrappers would suppress)
# ---------------------------------------------------------------------------
def test_follow_pipeline_false_scope_no_stage_wrappers(monkeypatch, tmp_path_factory):
    """`pipeline:false` + scope -> per-block re-scan armed, stage wrappers OFF."""
    spec = {"follow": [{"tensor": "embedding_features", "pipeline": False,
                        "scope": {"names": ["emb_proj"]}, "bounds": [0, 60]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is True
    assert nl._PIPELINE is False              # no copy/sparse stage wrappers
    assert nl._PIPELINE_OFF_FOLLOW is True    # explicit spec-derived opt-in
    assert nl._TRACK_EVERY_LAYER is True      # per-block re-scan still armed
    assert nl._FOLLOW_FWD is True             # forward-entry checkpoint still runs
    assert nl._WATCH_NAMES == ("emb_proj",)
    assert nl._bound_for("embedding_features") == (0.0, 60.0)


def test_flat_track_every_layer_without_pipeline_is_noop(monkeypatch, tmp_path_factory):
    """REGRESSION GUARD: a legacy flat-var run with NANLOG_TRACK_EVERY_LAYER=1 and
    NANLOG_PIPELINE=0 (no NANLOG_SPEC) must NOT silently activate the pipeline-off
    forward/block follow. That combination historically warned and no-op'd; keeping
    it a no-op is what prevents a legacy run from being switched into a different,
    timing-sensitive capture profile without an explicit opt-in. Only a validated
    `pipeline: false` follow spec (which sets NANLOG_PIPELINE_OFF_FOLLOW) turns the
    forward capture on."""
    nl = _load_logger(
        {"NANLOG_TRACK_EVERY_LAYER": "1", "NANLOG_PIPELINE": "0"},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_PRESENT is False           # pure flat-var run
    assert nl._TRACK_EVERY_LAYER is True        # the flat var is read...
    assert nl._PIPELINE is False
    assert nl._PIPELINE_OFF_FOLLOW is False     # ...but the new mode is NOT inferred
    assert nl._FOLLOW_FWD is False              # so forward capture stays OFF (no-op)


def test_flat_pipeline_off_follow_var_alone_is_not_honored(monkeypatch, tmp_path_factory):
    """NANLOG_PIPELINE_OFF_FOLLOW is a spec-INTERNAL derived var, not a public flat
    knob. Setting it directly in a no-spec run is reset to its baseline by the
    spec-owned-vars machinery only when a spec applies; with no spec it is read as-is,
    so this test documents that it is not part of the supported flat surface. (The
    supported entry point is a `pipeline: false` follow spec.)"""
    # No NANLOG_SPEC -> the flat var is read directly; we assert the plumbing exists
    # (the attribute is defined) rather than encouraging this as a public path.
    nl = _load_logger({}, monkeypatch, tmp_path_factory)
    assert hasattr(nl, "_PIPELINE_OFF_FOLLOW")
    assert nl._PIPELINE_OFF_FOLLOW is False     # default off when unset


def test_follow_pipeline_false_default_stride_is_one(monkeypatch, tmp_path_factory):
    """A scope with no stride means every matched block, wrappers still off."""
    spec = {"follow": [{"tensor": "ef", "pipeline": False, "scope": {"types": ["MLP"]}}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is False
    assert nl._TRACK_EVERY_LAYER is True
    assert nl._TRACK_LAYER_STRIDE == 1
    assert nl._WATCH_TYPES == ("MLP",)


def test_follow_pipeline_true_is_default_and_unchanged(monkeypatch, tmp_path_factory):
    """Omitting `pipeline` keeps the historical behavior: stage wrappers ON."""
    spec = {"follow": [{"tensor": "ef", "scope": {"types": ["MLP"]}, "stride": 5}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_EVERY_LAYER is True


@pytest.mark.parametrize("follow_entry", [
    {"tensor": "ef", "pipeline": False, "stages": True},   # stages needs wrappers
    {"tensor": "ef", "pipeline": False},                   # no scope -> captures nothing
    {"tensor": "ef", "pipeline": "no", "scope": {"types": ["MLP"]}},  # non-bool
])
def test_follow_pipeline_false_invalid_rolls_back(follow_entry, monkeypatch, tmp_path_factory):
    """pipeline:false with stages:true (contradiction), with no scope (captures
    nothing), or a non-bool value rejects the whole spec (flat fallback)."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps({"follow": [follow_entry]}),
         "NANLOG_CHANNELS": "act,igrad"},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is False
    assert nl._CHANNELS == frozenset({"act", "igrad"})
    assert nl._PIPELINE is False


def test_follow_pipeline_false_summary_mode(monkeypatch, tmp_path_factory):
    """The summary records follow_mode=forward_blocks so a pipeline-off run is
    auditable (distinct from stage_wrappers and off)."""
    spec = {"follow": [{"tensor": "ef", "pipeline": False, "scope": {"types": ["MLP"]}}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    nl._write_summary()
    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["pipeline"] is False
    assert smy["follow_fwd"] is True
    assert smy["follow_mode"] == "forward_blocks"


def test_follow_stage_summary_mode(monkeypatch, tmp_path_factory):
    """A stage-wrapper follow records follow_mode=stage_wrappers."""
    spec = {"follow": [{"tensor": "ef", "stages": True}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    nl._write_summary()
    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["follow_mode"] == "stage_wrappers"


def test_follow_pipeline_false_runtime_capture(monkeypatch, tmp_path_factory):
    """End-to-end: a pipeline-off follow re-scans the tracked tensor at each scoped
    block via the forward hook, with NO pipeline hook installed. The track records
    carry checkpoint=<block name> and a resolved (non-null) batch_id."""
    class Batch:
        def __init__(self, ef):
            self.embedding_features = ef

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(3)])

        def forward(self, batch):
            x = batch.embedding_features
            for b in self.blocks:
                x = b(x)
            return x

    spec = {"follow": [{"tensor": "embedding_features", "pipeline": False,
                        "scope": {"types": ["Linear"]}}],
            "sample_every": 1}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    model = Net()
    assert nl._attach(model) == 3     # forward hooks installed despite no channels
    for _ in range(2):
        batch = Batch(torch.randn(4, 4))
        nl._root_pre_hook(None, (batch,))   # forward-entry checkpoint mints batch_id
        model(batch)
    nl._write_summary()
    recs = _records(nl)
    track = [r for r in recs if r.get("role") == "track"]
    assert track, "pipeline-off follow wrote no track records"
    # re-scanned at the scoped blocks (checkpoint = block name), not a stage phase
    assert any("blocks" in (r.get("checkpoint") or "") for r in track)
    # batch_id is resolved at forward entry (not null) even with the wrappers off
    assert any(r.get("batch_id") is not None for r in track)
    # A pipeline-off run must NOT report stage-wrapper checkpoints -- that would
    # imply stage instrumentation ran on a timing-safe capture. The forward-entry
    # checkpoints are counted separately.
    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["follow_mode"] == "forward_blocks"
    assert smy["pipeline"] is False
    assert smy["pipeline_installed"] is False
    assert smy["pipeline_checkpoints"] == 0
    assert smy["forward_checkpoints"] > 0


def test_follow_at_key_is_rejected(monkeypatch, tmp_path_factory):
    """The old `at` key was removed: it is now an unknown follow key and rolls the
    whole spec back (no silent back-compat)."""
    spec = {"follow": [{"tensor": "ef", "at": "stage"}]}
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(spec), "NANLOG_CHANNELS": "act,igrad"},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is False
    assert nl._CHANNELS == frozenset({"act", "igrad"})


@pytest.mark.parametrize("follow_entry", [
    {"tensor": "ef", "at": "stage"},                        # removed `at` key -> unknown
    {"tensor": "ef", "stride": 4},                          # stride without scope
    {"tensor": "ef", "stages": False},                      # captures nothing
    {"tensor": "ef", "stages": "yes"},                      # non-bool stages
    {"tensor": "ef", "scope": {"types": ["MLP"]}, "stride": 0},   # stride < 1
])
def test_follow_cadence_invalid_rolls_back(follow_entry, monkeypatch, tmp_path_factory):
    """Malformed / contradictory follow cadence rejects the whole spec (flat fallback)."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps({"follow": [follow_entry]}),
         "NANLOG_CHANNELS": "act,igrad"},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is False
    assert nl._CHANNELS == frozenset({"act", "igrad"})


@pytest.mark.parametrize("watch_scope", [
    {"names": ["blk"]},   # names-only watch scope
    {"types": ["MLP"]},   # types-only watch scope (regression: warning missed this)
])
def test_watch_plus_follow_stride_warns_on_merged_scope(watch_scope, monkeypatch, tmp_path_factory, capsys):
    """A watch scope AND a follow-stride scope share the engine's single module
    filter, so the merge warning must fire for BOTH names-only and types-only watch
    scopes (PR #292 review: types-only was silently missed)."""
    spec = {"watch": [{"scope": watch_scope, "tensors": ["input"]}],
            "follow": [{"tensor": "ef", "scope": {"names": ["emb"]}, "stride": 4}]}
    _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert "both watch scope and follow scope set" in capsys.readouterr().err


def test_follow_entry_without_tensor_does_not_capture_default(monkeypatch, tmp_path_factory):
    """A follow entry with no `tensor` is malformed; it must NOT silently fall through
    to the engine default (embedding_features). With no valid follow entry and no flat
    vars asking for pipeline, the run rolls back to flat defaults (pipeline off)."""
    spec = {"follow": [{"stages": True}]}   # no tensor
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is False           # rolled back; not a real capture
    assert nl._TRACK_ATTR == ("embedding_features",)  # the flat DEFAULT, not spec-driven


def test_follow_mixed_valid_and_tensorless_rolls_back(monkeypatch, tmp_path_factory):
    """A tensorless entry makes the WHOLE spec invalid (atomic validation): the run
    rolls back to flat vars rather than silently applying only the valid entry."""
    spec = {"follow": [{"stages": True}, {"tensor": "ef", "stages": True}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is False           # rolled back; not a partial apply
    assert nl._TRACK_ATTR == ("embedding_features",)  # flat default, not "ef"


def test_follow_per_entry_bounds(monkeypatch, tmp_path_factory):
    """Each followed tensor gets its OWN bounds, not the first entry's applied to all."""
    spec = {"follow": [{"tensor": "a", "bounds": [0, 1]},
                       {"tensor": "b", "bounds": [10, 20]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._bound_for("a") == (0.0, 1.0)
    assert nl._bound_for("b") == (10.0, 20.0)


def test_follow_stage_no_watch_warning(monkeypatch, tmp_path_factory, capsys):
    """A follow-stage spec (empty watch scope by design) must NOT print the
    'no modules will be watched' warning; a stride follow with no scope still does."""
    stage = {"follow": [{"tensor": "ef", "stages": True}]}
    _load_logger({"NANLOG_SPEC": json.dumps(stage)}, monkeypatch, tmp_path_factory)
    assert "no modules will be watched" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# SPEC wins: a follow-only spec must NOT inherit flat layer-channel defaults
# ---------------------------------------------------------------------------
def test_follow_only_clears_inherited_layer_defaults(monkeypatch, tmp_path_factory):
    """With the collector's 7-channel + Linear defaults already in env, a follow-only
    stage spec must clear the layer channels AND the Linear watch default, so it does
    not silently hook Linear layers the user never asked to watch."""
    spec = {"follow": [{"tensor": "embedding_features", "stages": True}]}
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(spec),
         "NANLOG_CHANNELS": "act,input,igrad,weight,bias,wgrad,bgrad"},
        monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset()
    assert nl._WATCH_TYPES == ()
    assert nl._PIPELINE is True   # the follow itself is still active


def test_follow_stride_clears_channels_but_keeps_scope(monkeypatch, tmp_path_factory):
    """A stride follow clears inherited layer channels but keeps its own scope, and
    the engine still installs forward hooks (the re-scan runs inside the fwd hook)."""
    spec = {"follow": [{"tensor": "ef", "scope": {"types": ["Linear"]}, "stride": 2}]}
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(spec), "NANLOG_CHANNELS": "act,igrad"},
        monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset()
    assert nl._WATCH_TYPES == ("Linear",)
    assert nl._TRACK_EVERY_LAYER is True
    # want_fwd must be True from _TRACK_EVERY_LAYER even with no layer channels,
    # else _attach installs no forward hook and the re-scan never fires.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    assert nl._attach(model) == 2


def test_watch_plus_follow_does_not_clear_watch_channels(monkeypatch, tmp_path_factory):
    """The clear only applies to follow-ONLY specs: a watch group present means its
    channels are honored, not wiped."""
    spec = {"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["input"]}],
            "follow": [{"tensor": "ef", "stages": True}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset({"input"})
    assert nl._WATCH_TYPES == ("MLP",)


# ---------------------------------------------------------------------------
# precedence + fallback
# ---------------------------------------------------------------------------
def test_spec_wins_over_flat_vars(monkeypatch, tmp_path_factory):
    """When both are set, NANLOG_SPEC overwrites the flat channel var."""
    spec = {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["weight"]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec),
                       "NANLOG_CHANNELS": "act,igrad"}, monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset({"weight"})


def test_malformed_spec_falls_back_to_flat(monkeypatch, tmp_path_factory):
    """A broken spec must not crash import; the flat vars still apply."""
    nl = _load_logger({"NANLOG_SPEC": "{not valid json",
                       "NANLOG_CHANNELS": "input", "NANLOG_WATCH_TYPES": "Linear"},
                      monkeypatch, tmp_path_factory)
    # spec was active (set) but ignored; flat vars win
    assert nl._CHANNELS == frozenset({"input"})
    assert nl._WATCH_TYPES == ("Linear",)


def test_no_spec_uses_flat_vars(monkeypatch, tmp_path_factory):
    nl = _load_logger({"NANLOG_CHANNELS": "act", "NANLOG_WATCH_TYPES": "Linear"},
                      monkeypatch, tmp_path_factory)
    assert nl._SPEC_PRESENT is False and nl._SPEC_APPLIED is False
    assert nl._CHANNELS == frozenset({"act"})


# ---------------------------------------------------------------------------
# malformed spec must never crash import (sidecar contract)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stride", [0, -3, "8", 1.5, True])
def test_malformed_stride_rolls_back(stride, monkeypatch, tmp_path_factory):
    """A bad `stride` value (non-int, <1, bool) rolls back to flat vars rather than
    crashing the int() at config time or arming a bogus re-scan."""
    spec = {"follow": [{"tensor": "ef", "scope": {"names": ["x"]}, "stride": stride}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._TRACK_LAYER_STRIDE == 1
    assert nl._TRACK_EVERY_LAYER is False
    assert nl._PIPELINE is False   # whole spec rejected atomically


@pytest.mark.parametrize("spec", [
    {"watch": ["MLP"]},                       # group is a string, not a mapping
    {"watch": [{"scope": {"types": ["Linear"], "tensors": ["input"]}}, "oops"]},  # mixed
    {"follow": ["embedding_features"]},       # follow entry is a string
    {"watch": "MLP"},                         # watch is a scalar, not a list
    {"follow": {"tensor": "ef"}},             # follow is a mapping, not a list
])
def test_malformed_entry_shapes_do_not_crash(spec, monkeypatch, tmp_path_factory):
    """Malformed watch/follow shapes roll back to flat vars, never crash import."""
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec),
                       "NANLOG_CHANNELS": "act,igrad", "NANLOG_WATCH_TYPES": "Linear"},
                      monkeypatch, tmp_path_factory)
    # present (set) but NOT applied (rejected + rolled back)
    assert nl._SPEC_PRESENT is True and nl._SPEC_APPLIED is False
    assert nl._SPEC_ERROR is not None                    # reason recorded for the summary
    assert nl._CHANNELS == frozenset({"act", "igrad"})   # flat vars intact
    assert nl._PIPELINE is False


# ---------------------------------------------------------------------------
# schema validation: bad shapes/values roll back atomically (never half-apply)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", [
    {"watch": [{"scope": {"names": "emb_proj"}, "tensors": ["input"]}]},  # names scalar string
    {"watch": [{"scope": {"types": "MLP"}, "tensors": ["input"]}]},       # types scalar string
    {"watch": [{"scope": {"types": ["MLP"]}}]},                           # no tensors
    {"watch": [{"scope": {"types": ["MLP"]}, "tensors": []}]},            # empty tensors
    {"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["bogus"]}]},     # unknown tensor
    {"sample_every": "abc"},                                              # non-int
    {"sample_every": 0},                                                  # < 1 (div-by-zero risk)
    {"pre_context": -1},                                                  # < 0
    {"follow": [{"tensor": "ef", "stages": True, "bounds": {"lo": 0, "hi": 1}}]},  # bounds dict
    {"follow": [{"tensor": "ef", "stages": True, "bounds": [0]}]},         # bounds wrong length
    {"follow": [{"tensor": "ef", "stages": True, "bounds": ["x", "y"]}]},  # bounds non-numeric
    # a 2nd watch group with a non-list `tensors` must give the clean tensors error,
    # not a TypeError from the multi-group merge check (regression, PR #292 review).
    {"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["input"]},
               {"scope": {"types": ["Linear"]}, "tensors": None}]},
    # explicit falsy scope must NOT collapse to "absent" and inherit the Linear
    # default -- it is malformed and must roll back (PR #292 review).
    {"watch": [{"scope": [], "tensors": ["input"]}]},           # falsy list scope
    {"watch": [{"scope": "", "tensors": ["input"]}]},           # falsy str scope
    {"follow": [{"tensor": "ef", "scope": "", "stride": 1}]},   # falsy follow scope
    {"watch": None},                                            # explicit null
    {"follow": None},
    # non-finite bounds would make the OOB check meaningless while looking applied.
    {"follow": [{"tensor": "ef", "stages": True, "bounds": [float("nan"), 60]}]},
    {"follow": [{"tensor": "ef", "stages": True, "bounds": [0, float("inf")]}]},
    {"follow": [{"tensor": "ef", "stages": True, "bounds": [float("-inf"), 60]}]},
    # a malformed field on a NON-FIRST follow entry must still reject the whole spec
    # (atomic validation), even though only the first entry's cadence is honored
    # (PR #292 review).
    {"follow": [{"tensor": "a", "stages": True}, {"tensor": "b", "at": "bogus"}]},  # leftover `at` key
    {"follow": [{"tensor": "a", "stages": True}, {"tensor": "b", "scope": {"names": ["x"]}, "stride": 0}]},
    {"follow": [{"tensor": "a", "stages": True}, {"tensor": "b", "stages": True, "scope": []}]},
    {"follow": [{"tensor": "a", "stages": True}, {"tensor": "b", "stages": True, "bounds": [1, 2, 3]}]},
])
def test_invalid_spec_rolls_back_to_flat_vars(spec, monkeypatch, tmp_path_factory):
    """Every malformed shape/value rejects the WHOLE spec and falls back to the flat
    vars — no broadened capture, no disabled check, no deferred crash."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(spec),
         "NANLOG_CHANNELS": "act,igrad", "NANLOG_WATCH_TYPES": "Linear",
         "NANLOG_SAMPLE_EVERY": "50"},
        monkeypatch, tmp_path_factory)
    # flat vars intact; nothing from the bad spec leaked through
    assert nl._CHANNELS == frozenset({"act", "igrad"})
    assert nl._WATCH_TYPES == ("Linear",)
    assert nl._WATCH_NAMES == ()
    assert nl._PIPELINE is False
    assert nl._BOUNDS_ACTIVE is False
    assert nl._SAMPLE_EVERY == 50


def test_multigroup_bad_tensors_gives_clean_error(monkeypatch, tmp_path_factory):
    """A 2nd watch group with a non-list `tensors` must fail with the clear
    'watch[].tensors must be ...' message, not a TypeError from the merge check."""
    spec = {"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["input"]},
                      {"scope": {"types": ["Linear"]}, "tensors": None}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is False
    assert "watch[].tensors" in (nl._SPEC_ERROR or "")
    assert "NoneType" not in (nl._SPEC_ERROR or "")


def test_mid_translation_crash_rolls_back_to_flat_vars(monkeypatch, tmp_path_factory):
    """A spec that derives some flat vars and THEN crashes (scope is a string, not a
    mapping) must roll back fully to the original flat vars — no half-applied
    NANLOG_PIPELINE / TRACK_ATTR / stride survives."""
    spec = {"follow": [{"tensor": "ef", "scope": "bad", "stride": 8}]}
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps(spec),
         "NANLOG_CHANNELS": "act,igrad", "NANLOG_PIPELINE": "0"},
        monkeypatch, tmp_path_factory)
    # original flat vars intact; none of the partially-derived follow state leaked
    assert nl._PIPELINE is False
    assert nl._TRACK_EVERY_LAYER is False
    assert nl._TRACK_LAYER_STRIDE == 1
    assert nl._TRACK_ATTR == ("embedding_features",)   # the flat default, not "ef"
    assert nl._CHANNELS == frozenset({"act", "igrad"})


# ---------------------------------------------------------------------------
# summary must report whether the spec was actually APPLIED, not just present
# ---------------------------------------------------------------------------
def test_summary_reports_spec_applied_false_on_rejection(monkeypatch, tmp_path_factory):
    """A rejected spec must show spec_present=true, spec_applied=false, and an error in
    the summary — never imply the structured config ran when flat fallback was used."""
    nl = _load_logger({"NANLOG_SPEC": json.dumps({"sample_every": "abc"})},
                      monkeypatch, tmp_path_factory)
    nl._write_summary()
    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["spec_present"] is True
    assert smy["spec_applied"] is False
    assert smy["spec_error"] is not None


def test_summary_reports_spec_applied_true_on_success(monkeypatch, tmp_path_factory):
    spec = {"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["input"]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    nl._write_summary()
    smy = json.loads((nl._OUT_DIR / "summary_rank0.json").read_text())
    assert smy["spec_present"] is True
    assert smy["spec_applied"] is True
    assert smy["spec_error"] is None


# ---------------------------------------------------------------------------
# SPEC WINS: a stale flat follow/bounds/scope var must NOT leak into a spec run
# ---------------------------------------------------------------------------
def test_spec_clears_stale_flat_pipeline(monkeypatch, tmp_path_factory):
    """A watch-only spec must not inherit a lingering flat NANLOG_PIPELINE=1."""
    nl = _load_logger(
        {"NANLOG_PIPELINE": "1",
         "NANLOG_SPEC": json.dumps({"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["input"]}]})},
        monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is False


def test_spec_clears_stale_flat_pipeline_off_follow(monkeypatch, tmp_path_factory):
    """A watch-only spec must not inherit a lingering flat NANLOG_PIPELINE_OFF_FOLLOW=1
    (the spec-owned reset covers the new var too), so a stale value can't switch a
    watch run into forward-capture mode."""
    nl = _load_logger(
        {"NANLOG_PIPELINE_OFF_FOLLOW": "1",
         "NANLOG_SPEC": json.dumps({"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["input"]}]})},
        monkeypatch, tmp_path_factory)
    assert nl._PIPELINE_OFF_FOLLOW is False
    assert nl._FOLLOW_FWD is False


def test_spec_stage_clears_stale_flat_track_every_layer(monkeypatch, tmp_path_factory):
    """A stage follow must not keep a lingering flat NANLOG_TRACK_EVERY_LAYER=1."""
    nl = _load_logger(
        {"NANLOG_TRACK_EVERY_LAYER": "1",
         "NANLOG_SPEC": json.dumps({"follow": [{"tensor": "ef", "stages": True}]})},
        monkeypatch, tmp_path_factory)
    assert nl._TRACK_EVERY_LAYER is False


def test_spec_without_bounds_clears_stale_flat_bounds(monkeypatch, tmp_path_factory):
    """A follow spec with no bounds must not keep a lingering flat NANLOG_BOUNDS."""
    nl = _load_logger(
        {"NANLOG_BOUNDS": "ef:0:60",
         "NANLOG_SPEC": json.dumps({"follow": [{"tensor": "ef", "stages": True}]})},
        monkeypatch, tmp_path_factory)
    assert nl._BOUNDS_ACTIVE is False


def test_follow_only_spec_clears_stale_flat_watch_names(monkeypatch, tmp_path_factory):
    """A follow-only spec must not keep a lingering flat NANLOG_WATCH_NAMES."""
    nl = _load_logger(
        {"NANLOG_WATCH_NAMES": "emb_proj",
         "NANLOG_SPEC": json.dumps({"follow": [{"tensor": "ef", "stages": True}]})},
        monkeypatch, tmp_path_factory)
    assert nl._WATCH_NAMES == ()
    assert nl._WATCH_TYPES == ()   # and the Linear default does not re-arm


# ---------------------------------------------------------------------------
# spec source: file (NANLOG_SPEC_FILE) + precedence
# ---------------------------------------------------------------------------
def test_spec_file_env_applies(monkeypatch, tmp_path_factory, tmp_path):
    """NANLOG_SPEC_FILE points at a JSON file whose contents are used as the spec."""
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(
        {"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["input", "output"]}]}))
    nl = _load_logger({"NANLOG_SPEC_FILE": str(spec_file)}, monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is True
    assert nl._SPEC_SOURCE.startswith("NANLOG_SPEC_FILE=")
    assert nl._CHANNELS == frozenset({"act", "input"})


def test_spec_file_beats_inline(monkeypatch, tmp_path_factory, tmp_path):
    """When both NANLOG_SPEC_FILE and inline NANLOG_SPEC are set, the file wins."""
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps({"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["weight"]}]}))
    nl = _load_logger(
        {"NANLOG_SPEC_FILE": str(spec_file),
         "NANLOG_SPEC": json.dumps({"watch": [{"scope": {"types": ["Linear"]}, "tensors": ["input"]}]})},
        monkeypatch, tmp_path_factory)
    assert nl._CHANNELS == frozenset({"weight"})   # from the file, not inline


def test_missing_spec_file_falls_back_but_records_source(monkeypatch, tmp_path_factory):
    """A NANLOG_SPEC_FILE that can't be read falls back to flat vars, but the artifact
    still records that a spec WAS requested (present=true) + why it fell back, so the
    summary never hides the fallback."""
    nl = _load_logger(
        {"NANLOG_SPEC_FILE": "/no/such/spec.json",
         "NANLOG_CHANNELS": "act", "NANLOG_WATCH_TYPES": "Linear"},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_PRESENT is True        # requested, even though unreadable
    assert nl._SPEC_APPLIED is False
    assert nl._SPEC_SOURCE.startswith("NANLOG_SPEC_FILE=")
    assert nl._SPEC_ERROR and "cannot read" in nl._SPEC_ERROR
    assert nl._CHANNELS == frozenset({"act"})   # flat fallback intact


def test_spec_dir_key_is_rejected(monkeypatch, tmp_path_factory):
    """`dir` is not a valid spec key -- output location is the separate NANLOG_DIR env
    var. A spec `dir` must roll back (not silently redirect artifacts)."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps({"follow": [{"tensor": "ef", "stages": True}], "dir": "/tmp/x"})},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is False
    assert "dir" in (nl._SPEC_ERROR or "")
    assert nl._PIPELINE is False   # whole spec rolled back


def test_inline_spec_still_works(monkeypatch, tmp_path_factory):
    """The original inline NANLOG_SPEC path is unchanged (lowest precedence)."""
    nl = _load_logger(
        {"NANLOG_SPEC": json.dumps({"follow": [{"tensor": "ef", "stages": True}]})},
        monkeypatch, tmp_path_factory)
    assert nl._SPEC_APPLIED is True
    assert nl._SPEC_SOURCE == "NANLOG_SPEC"
    assert nl._PIPELINE is True


@pytest.mark.parametrize("argv,expect_cfg,expect_out", [
    (["prog", "--config", "s.json", "t.py", "--m", "x"], "s.json", ["prog", "t.py", "--m", "x"]),
    (["prog", "--config=s.json", "t.py", "a"], "s.json", ["prog", "t.py", "a"]),
    (["prog", "t.py", "a"], None, ["prog", "t.py", "a"]),
    # the target script's OWN --config (after the target path) must pass through
    # untouched -- the logger only consumes a LEADING --config.
    (["prog", "t.py", "--config", "train.yaml"], None, ["prog", "t.py", "--config", "train.yaml"]),
    (["prog", "--config", "s.json", "t.py", "--config", "train.yaml"], "s.json",
     ["prog", "t.py", "--config", "train.yaml"]),
    # whitespace around the path is trimmed.
    (["prog", "--config", "  s.json  ", "t.py"], "s.json", ["prog", "t.py"]),
    (["prog", "--config=  s.json  ", "t.py"], "s.json", ["prog", "t.py"]),
])
def test_extract_config_arg(argv, expect_cfg, expect_out, monkeypatch, tmp_path_factory):
    """`--config <file>` / `--config=<file>` is pulled out of argv (setting the config
    path) and the target + its args are left intact for the script."""
    nl = _load_logger({}, monkeypatch, tmp_path_factory)
    nl._CONFIG_FILE_ARG = None
    out = nl._extract_config_arg(argv)
    assert nl._CONFIG_FILE_ARG == expect_cfg
    assert out == expect_out


@pytest.mark.parametrize("argv", [
    ["prog", "--config=", "t.py"],       # empty =value
    ["prog", "--config", "", "t.py"],    # empty space-value
    ["prog", "--config", "   ", "t.py"], # whitespace-only
    ["prog", "--config"],                # missing value entirely
])
def test_extract_config_arg_empty_is_usage_error(argv, monkeypatch, tmp_path_factory):
    """An empty/whitespace `--config` value is a usage error (exit 2), not a silent
    fall-through that looks like 'no spec requested'."""
    nl = _load_logger({}, monkeypatch, tmp_path_factory)
    nl._CONFIG_FILE_ARG = None
    with pytest.raises(SystemExit) as e:
        nl._extract_config_arg(argv)
    assert e.value.code == 2


def test_config_source_precedence_and_file_read(monkeypatch, tmp_path_factory, tmp_path):
    """--config (via _CONFIG_FILE_ARG) beats NANLOG_SPEC_FILE beats inline NANLOG_SPEC,
    and its file contents are what gets read."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["output"]}]}))
    nl = _load_logger({"NANLOG_SPEC": json.dumps({"sample_every": 7})},
                      monkeypatch, tmp_path_factory)
    # simulate the --config extraction having run, then re-resolve + apply
    nl._CONFIG_FILE_ARG = str(cfg)
    text, source, error = nl._resolve_spec_source()
    assert source.startswith("--config=")
    assert error is None
    applied, err = nl._apply_spec(text)
    assert applied is True and err is None
    assert nl._SPEC_TENSOR_TO_CHANNEL["output"] == ("act",)   # sanity on the mapping
