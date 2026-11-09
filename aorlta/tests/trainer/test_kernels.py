"""Fast-path detection (WP1). CPU only: the fakes mirror transformers 5.12's modeling_qwen3_5 and never import
torch; the one test against the real module is skipped where transformers is not installed."""

import importlib.util
import inspect
import logging
import types

import pytest

from aorlta.trainer import kernels


def fake_modeling(conv=True, fla=True, flag="computed", drop=()):
    """A stand-in for modeling_qwen3_5: module-level kernels exactly as its import block resolves them."""
    mod = types.ModuleType("fake_modeling_qwen3_5")

    def torch_causal_conv1d_update(*a, **k): ...
    def torch_chunk_gated_delta_rule(*a, **k): ...
    def torch_recurrent_gated_delta_rule(*a, **k): ...
    def causal_conv1d_fn(*a, **k): ...
    def causal_conv1d_update(*a, **k): ...
    def chunk_gated_delta_rule(*a, **k): ...
    def fused_recurrent_gated_delta_rule(*a, **k): ...

    class Qwen3_5RMSNormGated: ...

    class FusedRMSNormGated: ...

    mod.torch_causal_conv1d_update = torch_causal_conv1d_update
    mod.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    mod.torch_recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
    mod.Qwen3_5RMSNormGated = Qwen3_5RMSNormGated
    mod.causal_conv1d_fn = causal_conv1d_fn if conv else None
    mod.causal_conv1d_update = causal_conv1d_update if conv else None
    mod.chunk_gated_delta_rule = chunk_gated_delta_rule if fla else None
    mod.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule if fla else None
    mod.FusedRMSNormGated = FusedRMSNormGated if fla else None
    if flag == "computed":
        flag = all((mod.causal_conv1d_fn, mod.causal_conv1d_update, mod.chunk_gated_delta_rule,
                    mod.fused_recurrent_gated_delta_rule))
    mod.is_fast_path_available = flag

    class Qwen3_5GatedDeltaNet:
        def __init__(self):
            self.norm = Qwen3_5RMSNormGated() if mod.FusedRMSNormGated is None else mod.FusedRMSNormGated()
            self.causal_conv1d_fn = mod.causal_conv1d_fn
            self.causal_conv1d_update = mod.causal_conv1d_update or mod.torch_causal_conv1d_update
            self.chunk_gated_delta_rule = mod.chunk_gated_delta_rule or mod.torch_chunk_gated_delta_rule
            self.recurrent_gated_delta_rule = (mod.fused_recurrent_gated_delta_rule
                                               or mod.torch_recurrent_gated_delta_rule)

    mod.Qwen3_5GatedDeltaNet = Qwen3_5GatedDeltaNet
    for name in drop:
        delattr(mod, name)
    return mod


class FakeModel:
    def __init__(self, layers, layer_types="matching", nested=True):
        self.layers = list(layers)
        if layer_types == "matching":
            layer_types = ["linear_attention"] * len(self.layers) + ["full_attention"]
        text = types.SimpleNamespace(layer_types=layer_types) if layer_types is not None else types.SimpleNamespace()
        self.config = types.SimpleNamespace(text_config=text) if nested else text

    def named_modules(self):
        yield "", self
        for i, layer in enumerate(self.layers):
            yield f"model.language_model.layers.{i}.linear_attn", layer
            yield f"model.language_model.layers.{i}.mlp", object()


def fast_model(mod, n=3, **kw):
    return FakeModel([mod.Qwen3_5GatedDeltaNet() for _ in range(n)], **kw)


# --- environment ----------------------------------------------------------------------------------------

def test_environment_available_when_every_kernel_imported():
    rep = kernels.check_environment(fake_modeling())
    assert rep.available and rep.reasons == ()
    assert all(v != "missing" for v in rep.components.values())
    assert set(rep.components) == set(kernels.FAST_NAMES)


@pytest.mark.parametrize("conv,fla,missing", [
    (False, True, {"causal_conv1d_fn", "causal_conv1d_update"}),
    (True, False, {"chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"}),
    (False, False, set(kernels.FAST_NAMES)),
])
def test_environment_names_each_missing_kernel(conv, fla, missing):
    rep = kernels.check_environment(fake_modeling(conv=conv, fla=fla))
    assert not rep.available
    assert {n for n, v in rep.components.items() if v == "missing"} == missing
    for name in missing:
        assert any(r.startswith(f"{name} is missing") for r in rep.reasons)


def test_transformers_flag_false_wins_over_present_kernels():
    rep = kernels.check_environment(fake_modeling(flag=False))
    assert not rep.available
    assert any("is_fast_path_available=False" in r for r in rep.reasons)


@pytest.mark.parametrize("flag", [1, "True", "yes", object()])
def test_flag_must_be_exactly_true(flag):
    assert not kernels.check_environment(fake_modeling(flag=flag)).available


@pytest.mark.parametrize("name", ["chunk_gated_delta_rule", "causal_conv1d_fn", "FusedRMSNormGated"])
def test_missing_kernel_refused_even_if_flag_claims_true(name):
    mod = fake_modeling(flag=True)
    setattr(mod, name, None)
    rep = kernels.check_environment(mod)
    assert not rep.available
    assert any(r.startswith(f"{name} is missing") for r in rep.reasons)


def test_unknown_layout_without_flag_is_not_available():
    rep = kernels.check_environment(fake_modeling(drop=("is_fast_path_available",)))
    assert not rep.available
    assert any("unknown transformers layout" in r for r in rep.reasons)


def test_import_failure_is_not_available(monkeypatch):
    monkeypatch.setattr(kernels, "MODELING", "aorlta_no_such_package.modeling_qwen3_5")
    rep = kernels.check_environment()
    assert not rep.available
    assert rep.reasons and rep.reasons[0].startswith("cannot import")


class Exploding(types.ModuleType):
    @property
    def is_fast_path_available(self):
        raise RuntimeError("probe blew up")


def test_probe_that_raises_is_not_available():
    mod = Exploding("exploding")
    for name in kernels.FAST_NAMES:
        setattr(mod, name, lambda *a: None)
    rep = kernels.check_environment(mod)
    assert not rep.available
    assert any("probe failed" in r and "probe blew up" in r for r in rep.reasons)


def test_versions_report_absent_distributions(monkeypatch):
    monkeypatch.setattr(kernels, "VERSION_DISTS", ("aorlta-no-such-dist",))
    assert kernels.installed_versions() == {"aorlta-no-such-dist": "absent"}
    assert kernels.check_environment(fake_modeling()).versions == {"aorlta-no-such-dist": "absent"}


# --- model ------------------------------------------------------------------------------------------------

def test_model_available_when_every_layer_holds_fast_kernels():
    mod = fake_modeling()
    rep = kernels.check_model(fast_model(mod, n=24), mod)
    assert rep.available and rep.reasons == ()
    assert rep.linear_attention_layers == 24


def test_model_built_without_kernels_is_not_available():
    mod = fake_modeling(conv=False, fla=False)
    rep = kernels.check_model(fast_model(mod), mod)
    assert not rep.available
    assert rep.linear_attention_layers == 3
    layer_reasons = [r for r in rep.reasons if "uses the torch path" in r]
    assert len(layer_reasons) == 3


@pytest.mark.parametrize("attr", ["causal_conv1d_update", "chunk_gated_delta_rule", "recurrent_gated_delta_rule"])
def test_one_layer_on_the_torch_function_is_caught(attr):
    mod = fake_modeling()
    model = fast_model(mod)
    torch_fn = getattr(mod, kernels.TORCH_FALLBACKS[attr])
    setattr(model.layers[1], attr, torch_fn)
    rep = kernels.check_model(model, mod)
    assert not rep.available
    assert rep.reasons == (f"model.language_model.layers.1.linear_attn uses the torch path for {attr}",)


def test_layer_without_causal_conv1d_fn_is_caught():
    mod = fake_modeling()
    model = fast_model(mod)
    model.layers[2].causal_conv1d_fn = None
    rep = kernels.check_model(model, mod)
    assert rep.reasons == ("model.language_model.layers.2.linear_attn uses the torch path for causal_conv1d_fn",)


def test_layer_with_the_torch_norm_is_caught():
    mod = fake_modeling()
    model = fast_model(mod)
    model.layers[0].norm = mod.Qwen3_5RMSNormGated()
    rep = kernels.check_model(model, mod)
    assert rep.reasons == ("model.language_model.layers.0.linear_attn uses the torch path for norm",)


def test_layer_with_a_missing_kernel_attribute_is_caught():
    mod = fake_modeling()
    model = fast_model(mod)
    model.layers[0].chunk_gated_delta_rule = None
    assert not kernels.check_model(model, mod).available


def test_model_without_linear_attention_layers_is_not_available():
    mod = fake_modeling()
    rep = kernels.check_model(FakeModel([], layer_types=["full_attention"] * 4), mod)
    assert not rep.available
    assert rep.linear_attention_layers == 0
    assert any("no Qwen3.5 linear-attention layers" in r for r in rep.reasons)


def test_layer_count_must_match_the_config():
    mod = fake_modeling()
    model = fast_model(mod, n=2, layer_types=["linear_attention"] * 3 + ["full_attention"])
    rep = kernels.check_model(model, mod)
    assert not rep.available
    assert rep.reasons == ("found 2 linear-attention layers, the config declares 3",)


def test_layer_count_read_from_a_flat_config():
    mod = fake_modeling()
    assert kernels.check_model(fast_model(mod, n=3, nested=False), mod).available
    assert not kernels.check_model(fast_model(mod, n=3, nested=False, layer_types=["linear_attention"] * 4),
                                   mod).available


def test_config_without_layer_types_still_needs_fast_layers():
    mod = fake_modeling()
    assert kernels.check_model(fast_model(mod, n=2, layer_types=None), mod).available
    assert not kernels.check_model(FakeModel([], layer_types=None), mod).available


def test_unknown_layer_class_is_not_available():
    mod = fake_modeling()
    model = fast_model(mod)
    del mod.Qwen3_5GatedDeltaNet
    rep = kernels.check_model(model, mod)
    assert not rep.available
    assert any("Qwen3_5GatedDeltaNet" in r for r in rep.reasons)


def test_model_check_includes_the_environment_check():
    mod = fake_modeling()
    model = fast_model(mod)
    mod.is_fast_path_available = False
    assert not kernels.check_model(model, mod).available


# --- the gate -----------------------------------------------------------------------------------------------

def test_require_fast_path_refuses_when_missing():
    mod = fake_modeling(fla=False)
    with pytest.raises(kernels.FastPathUnavailable) as err:
        kernels.ensure_fast_path(True, modeling=mod)
    assert not err.value.report.available
    assert "chunk_gated_delta_rule is missing" in str(err.value)


def test_require_fast_path_refuses_a_model_on_the_torch_path():
    mod = fake_modeling()
    model = fast_model(mod)
    model.layers[0].chunk_gated_delta_rule = mod.torch_chunk_gated_delta_rule
    with pytest.raises(kernels.FastPathUnavailable):
        kernels.ensure_fast_path(True, model, modeling=mod)


def test_not_required_returns_the_report_and_warns(caplog):
    mod = fake_modeling(conv=False)
    with caplog.at_level(logging.WARNING, logger=kernels.log.name):
        rep = kernels.ensure_fast_path(False, modeling=mod)
    assert not rep.available
    assert any("torch fallback" in r.getMessage() for r in caplog.records)


def test_required_and_available_passes():
    mod = fake_modeling()
    rep = kernels.ensure_fast_path(True, fast_model(mod), modeling=mod)
    assert rep.available and rep.linear_attention_layers == 3


def test_logged_fallback_refuses_an_otherwise_fast_report():
    mod = fake_modeling()
    with kernels.FallbackLogWatcher() as w:
        logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").warning(
            "The fast path is not available because one of the required library is not installed. Falling back "
            "to torch implementation.")
    assert w.fallback_logged
    with pytest.raises(kernels.FastPathUnavailable) as err:
        kernels.ensure_fast_path(True, fast_model(mod), modeling=mod, watcher=w)
    assert "logged the torch fallback" in str(err.value)
    rep = kernels.ensure_fast_path(False, fast_model(mod), modeling=mod, watcher=w)
    assert not rep.available and rep.linear_attention_layers == 3


def test_watcher_ignores_other_warnings():
    mod = fake_modeling()
    with kernels.FallbackLogWatcher() as w:
        logging.getLogger("transformers.modeling_utils").warning("some weights were not initialised")
    assert not w.fallback_logged
    assert kernels.ensure_fast_path(True, fast_model(mod), modeling=mod, watcher=w).available


def test_watcher_sees_the_warning_when_transformers_logs_errors_only():
    lib = logging.getLogger("transformers")
    saved = lib.level
    lib.setLevel(logging.ERROR)
    try:
        with kernels.FallbackLogWatcher() as w:
            logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").warning(
                "The fast path is not available because ...")
        assert w.fallback_logged
        assert lib.level == logging.ERROR
    finally:
        lib.setLevel(saved)


def test_watcher_detaches_on_exit():
    with kernels.FallbackLogWatcher() as w:
        pass
    logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5").warning("The fast path is not available")
    assert not w.fallback_logged
    assert w not in logging.getLogger("transformers").handlers


# --- the real module ---------------------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers is not installed here")
def test_real_transformers_has_the_names_detection_reads():
    mod = importlib.import_module(kernels.MODELING)
    for name in ("is_fast_path_available", "Qwen3_5GatedDeltaNet", "Qwen3_5RMSNormGated", *kernels.FAST_NAMES,
                 *kernels.TORCH_FALLBACKS.values()):
        assert hasattr(mod, name), name
    init = inspect.getsource(mod.Qwen3_5GatedDeltaNet.__init__)
    assert kernels.FALLBACK_MARKER in init
    for attr in ("causal_conv1d_fn", "norm", *kernels.TORCH_FALLBACKS):
        assert f"self.{attr} =" in init, attr
