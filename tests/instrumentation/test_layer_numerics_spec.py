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
    spec = {"follow": [{"tensor": "embedding_features", "at": "stage", "bounds": [0, 60]}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_ATTR == ("embedding_features",)
    assert nl._BOUNDS_ACTIVE is True
    assert nl._bound_for("embedding_features") == (0.0, 60.0)
    assert nl._TRACK_EVERY_LAYER is False


def test_follow_stride_spec(monkeypatch, tmp_path_factory):
    """`at:stride:8` -> TRACK_EVERY_LAYER + stride 8; follow scope merges into WATCH_*."""
    spec = {"follow": [{"tensor": "embedding_features", "at": "stride:8",
                        "scope": {"names": ["emb_proj"]}}]}
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is True
    assert nl._TRACK_EVERY_LAYER is True
    assert nl._TRACK_LAYER_STRIDE == 8
    assert nl._WATCH_NAMES == ("emb_proj",)


def test_follow_entry_without_tensor_does_not_capture_default(monkeypatch, tmp_path_factory):
    """A follow entry with no `tensor` is malformed; it must NOT silently fall through
    to the engine default (embedding_features). With no valid follow entry and no flat
    vars asking for pipeline, the run rolls back to flat defaults (pipeline off)."""
    spec = {"follow": [{"at": "stage"}]}   # no tensor
    nl = _load_logger({"NANLOG_SPEC": json.dumps(spec)}, monkeypatch, tmp_path_factory)
    assert nl._PIPELINE is False           # rolled back; not a real capture
    assert nl._TRACK_ATTR == ("embedding_features",)  # the flat DEFAULT, not spec-driven


def test_follow_mixed_valid_and_tensorless_rolls_back(monkeypatch, tmp_path_factory):
    """A tensorless entry makes the WHOLE spec invalid (atomic validation): the run
    rolls back to flat vars rather than silently applying only the valid entry."""
    spec = {"follow": [{"at": "stage"}, {"tensor": "ef", "at": "stage"}]}
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
    stage = {"follow": [{"tensor": "ef", "at": "stage"}]}
    _load_logger({"NANLOG_SPEC": json.dumps(stage)}, monkeypatch, tmp_path_factory)
    assert "no modules will be watched" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# SPEC wins: a follow-only spec must NOT inherit flat layer-channel defaults
# ---------------------------------------------------------------------------
def test_follow_only_clears_inherited_layer_defaults(monkeypatch, tmp_path_factory):
    """With the collector's 7-channel + Linear defaults already in env, a follow-only
    stage spec must clear the layer channels AND the Linear watch default, so it does
    not silently hook Linear layers the user never asked to watch."""
    spec = {"follow": [{"tensor": "embedding_features", "at": "stage"}]}
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
    spec = {"follow": [{"tensor": "ef", "at": "stride:2", "scope": {"types": ["Linear"]}}]}
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
            "follow": [{"tensor": "ef", "at": "stage"}]}
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
@pytest.mark.parametrize("at", ["stride:", "stride:abc", "stride:0", "stride:-3"])
def test_malformed_stride_rolls_back(at, monkeypatch, tmp_path_factory):
    """A bad stride value must not crash the int() at config time; it rolls back to
    flat vars (stride 1, no per-layer re-scan armed)."""
    spec = {"follow": [{"tensor": "ef", "at": at}]}
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
    {"follow": [{"tensor": "ef", "at": "stage", "bounds": {"lo": 0, "hi": 1}}]},  # bounds dict
    {"follow": [{"tensor": "ef", "at": "stage", "bounds": [0]}]},         # bounds wrong length
    {"follow": [{"tensor": "ef", "at": "stage", "bounds": ["x", "y"]}]},  # bounds non-numeric
    # a 2nd watch group with a non-list `tensors` must give the clean tensors error,
    # not a TypeError from the multi-group merge check (regression, PR #292 review).
    {"watch": [{"scope": {"types": ["MLP"]}, "tensors": ["input"]},
               {"scope": {"types": ["Linear"]}, "tensors": None}]},
    # explicit falsy scope must NOT collapse to "absent" and inherit the Linear
    # default -- it is malformed and must roll back (PR #292 review).
    {"watch": [{"scope": [], "tensors": ["input"]}]},           # falsy list scope
    {"watch": [{"scope": "", "tensors": ["input"]}]},           # falsy str scope
    {"follow": [{"tensor": "ef", "at": "stride:1", "scope": ""}]},   # falsy follow scope
    {"watch": None},                                            # explicit null
    {"follow": None},
    # non-finite bounds would make the OOB check meaningless while looking applied.
    {"follow": [{"tensor": "ef", "at": "stage", "bounds": [float("nan"), 60]}]},
    {"follow": [{"tensor": "ef", "at": "stage", "bounds": [0, float("inf")]}]},
    {"follow": [{"tensor": "ef", "at": "stage", "bounds": [float("-inf"), 60]}]},
    # a malformed field on a NON-FIRST follow entry must still reject the whole spec
    # (atomic validation), even though only the first entry's cadence is honored
    # (PR #292 review).
    {"follow": [{"tensor": "a", "at": "stage"}, {"tensor": "b", "at": "bogus"}]},
    {"follow": [{"tensor": "a", "at": "stage"}, {"tensor": "b", "at": "stride:0"}]},
    {"follow": [{"tensor": "a", "at": "stage"}, {"tensor": "b", "at": "stage", "scope": []}]},
    {"follow": [{"tensor": "a", "at": "stage"}, {"tensor": "b", "at": "stage", "bounds": [1, 2, 3]}]},
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
    spec = {"follow": [{"tensor": "ef", "at": "stride:8", "scope": "bad"}]}
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
