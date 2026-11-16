"""Fast-path detection for Qwen3.5's linear-attention layers.

transformers 5.12 runs Qwen3.5's gated-DeltaNet layers on its fast path only when
flash-linear-attention (`fla`) and `causal_conv1d` import. Otherwise it logs "The fast path is not
available ... Falling back to torch implementation" once and runs a Python-loop torch version that took
18.0 s for one 9,021-token step (F16). The trainer calls `ensure_fast_path` before its first step and
refuses to train when `require_fast_path` is set and the fast path is missing.

Detection is fail-closed: an import error, an unknown transformers layout, a model with no (or too few)
linear-attention layers, or a probe that raises all count as "not available".

The check applies only to a model that has linear attention. A model whose config declares its layer types and
none of them `linear_attention`, and which holds no linear-attention module (dense Qwen3, say), runs full attention
only: there is no fast path to miss, and the report says the check does not apply (`applies` False). A model whose
config does not declare its layer types is checked as before.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
FALLBACK_MARKER = "fast path is not available"

# The module-level names transformers' `is_fast_path_available` is computed from, plus the fused norm
# that comes from the same `fla` import.
FAST_NAMES = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "FusedRMSNormGated",
)
# Layer attribute -> the torch implementation transformers substitutes when the fast kernel is missing.
TORCH_FALLBACKS = {
    "causal_conv1d_update": "torch_causal_conv1d_update",
    "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
    "recurrent_gated_delta_rule": "torch_recurrent_gated_delta_rule",
}
# The layer attributes a forward without a KV cache runs: the only ones that see a packed row's sequence boundaries
# (`seq_idx`, `cu_seq_lens_q`). On the torch path neither boundary is read.
PACKING_KERNELS = ("causal_conv1d_fn", "chunk_gated_delta_rule")
VERSION_DISTS = (
    "torch",
    "transformers",
    "fla-core",
    "flash-linear-attention",
    "causal-conv1d",
    "triton",
    "triton-rocm",
    "peft",
    "accelerate",
    "einops",
)


@dataclass(frozen=True)
class FastPathReport:
    available: bool
    reasons: tuple[str, ...]
    components: dict = field(default_factory=dict)
    linear_attention_layers: int | None = None
    versions: dict = field(default_factory=dict)
    applies: bool = True  # False: the model has no linear attention, so there is no fast path to check

    def to_dict(self) -> dict:
        out = {
            "available": self.available,
            "reasons": list(self.reasons),
            "components": dict(self.components),
            "linear_attention_layers": self.linear_attention_layers,
            "versions": dict(self.versions),
        }
        if not self.applies:
            out["applies"] = False
        return out


class FastPathUnavailable(RuntimeError):
    def __init__(self, report: FastPathReport):
        super().__init__("refusing to train: the Qwen3.5 linear-attention fast path is missing: "
                         + "; ".join(report.reasons))
        self.report = report


def installed_versions() -> dict:
    out = {}
    for dist in VERSION_DISTS:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "absent"
    return out


def _describe(obj) -> str:
    if obj is None:
        return "missing"
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', type(obj).__name__)}"


def _environment_reasons(mod) -> tuple[list[str], dict]:
    reasons: list[str] = []
    components: dict = {}
    if not hasattr(mod, "is_fast_path_available"):
        reasons.append(f"{MODELING} has no is_fast_path_available (unknown transformers layout)")
    elif mod.is_fast_path_available is not True:
        reasons.append(f"transformers reports is_fast_path_available={mod.is_fast_path_available!r}")
    for name in FAST_NAMES:
        obj = getattr(mod, name, None)
        components[name] = _describe(obj)
        if obj is None:
            reasons.append(f"{name} is missing")
    return reasons, components


def _expected_linear_layers(model) -> int | None:
    cfg = getattr(model, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        types = getattr(c, "layer_types", None)
        if isinstance(types, (list, tuple)):
            return sum(1 for t in types if t == "linear_attention")
    return None


def without_linear_attention(model) -> bool:
    """True only when the model positively has no linear attention: its config (or text config) lists its layer
    types and none is `linear_attention`, and no module is a linear-attention layer (named `linear_attn`, or of a
    class whose name holds `GatedDeltaNet`). Anything less certain is False, so the fast-path check runs."""
    if _expected_linear_layers(model) != 0:
        return False
    for name, m in model.named_modules():
        if name.rsplit(".", 1)[-1] == "linear_attn" or "GatedDeltaNet" in type(m).__name__:
            return False
    return True


def _torch_path(layer, mod) -> list[str]:
    """The kernels `layer` runs on transformers' torch path (a missing one included)."""
    bad = []
    if getattr(layer, "causal_conv1d_fn", None) is None:
        bad.append("causal_conv1d_fn")
    for attr, torch_name in TORCH_FALLBACKS.items():
        fn = getattr(layer, attr, None)
        if fn is None or fn is getattr(mod, torch_name, None):
            bad.append(attr)
    return bad


def _model_reasons(model, mod) -> tuple[list[str], int]:
    reasons: list[str] = []
    layer_cls = getattr(mod, "Qwen3_5GatedDeltaNet", None)
    if layer_cls is None:
        return [f"{MODELING} has no Qwen3_5GatedDeltaNet (unknown transformers layout)"], 0
    fused_norm = getattr(mod, "FusedRMSNormGated", None)
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, layer_cls)]
    if not layers:
        reasons.append("the model has no Qwen3.5 linear-attention layers to check")
    expected = _expected_linear_layers(model)
    if expected is not None and len(layers) != expected:
        reasons.append(f"found {len(layers)} linear-attention layers, the config declares {expected}")
    for name, layer in layers:
        bad = _torch_path(layer, mod)
        if fused_norm is None or not isinstance(getattr(layer, "norm", None), fused_norm):
            bad.append("norm")
        if bad:
            reasons.append(f"{name or '<root>'} uses the torch path for {', '.join(bad)}")
    return reasons, len(layers)


def _check(model, modeling) -> FastPathReport:
    versions = installed_versions()
    if model is not None:
        try:
            dense = without_linear_attention(model)
        except Exception as exc:
            return FastPathReport(False, (f"fast-path probe failed: {exc!r}",), versions=versions)
        if dense:
            return FastPathReport(True, (), {}, 0, versions, applies=False)
    try:
        mod = modeling if modeling is not None else importlib.import_module(MODELING)
    except Exception as exc:
        return FastPathReport(False, (f"cannot import {MODELING}: {exc!r}",), versions=versions)
    try:
        reasons, components = _environment_reasons(mod)
        n_layers = None
        if model is not None:
            model_reasons, n_layers = _model_reasons(model, mod)
            reasons += model_reasons
    except Exception as exc:
        return FastPathReport(False, (f"fast-path probe failed: {exc!r}",), versions=versions)
    return FastPathReport(not reasons, tuple(reasons), components, n_layers, versions)


def check_environment(modeling=None) -> FastPathReport:
    """Whether transformers' Qwen3.5 module resolved every fast-path kernel when it was imported."""
    return _check(None, modeling)


def check_model(model, modeling=None) -> FastPathReport:
    """`check_environment`, plus every linear-attention layer of `model` actually holds the fast kernels."""
    return _check(model, modeling)


def honours_packing(model, modeling=None) -> bool:
    """Whether a packed row's sequences stay apart in `model`'s linear-attention layers: no layer runs one of
    `PACKING_KERNELS` on the torch path, which runs the row as one sequence (each sequence after the first starts
    from the state the earlier ones left). A model without linear attention (`without_linear_attention`) does.
    Fail-closed: an import error or an unknown layout counts as not."""
    try:
        if without_linear_attention(model):
            return True
        mod = modeling if modeling is not None else importlib.import_module(MODELING)
        if not (hasattr(mod, "Qwen3_5GatedDeltaNet") and hasattr(mod, TORCH_FALLBACKS["chunk_gated_delta_rule"])):
            return False
        layers = [m for _, m in model.named_modules() if isinstance(m, mod.Qwen3_5GatedDeltaNet)]
        return not any(set(PACKING_KERNELS) & set(_torch_path(m, mod)) for m in layers)
    except Exception:
        return False


class FallbackLogWatcher(logging.Handler):
    """Records transformers' fallback warning while a model is built.

    transformers emits it with `warning_once`, so a model built earlier in the process hides it from a
    later watcher. A logged warning therefore marks the fast path missing; silence proves nothing.
    """

    def __init__(self, logger_name: str = "transformers"):
        super().__init__(level=logging.WARNING)
        self.logger_name = logger_name
        self.fallback_logged = False
        self._saved_level = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        if FALLBACK_MARKER in msg:
            self.fallback_logged = True

    def __enter__(self) -> FallbackLogWatcher:
        logger = logging.getLogger(self.logger_name)
        self._saved_level = logger.level
        if logger.getEffectiveLevel() > logging.WARNING:
            logger.setLevel(logging.WARNING)
        logger.addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        logger = logging.getLogger(self.logger_name)
        logger.removeHandler(self)
        logger.setLevel(self._saved_level)


def ensure_fast_path(
    require_fast_path: bool,
    model=None,
    *,
    modeling=None,
    watcher: FallbackLogWatcher | None = None,
) -> FastPathReport:
    """Check the fast path; raise `FastPathUnavailable` if it is missing and `require_fast_path` is set."""
    report = check_model(model, modeling) if model is not None else check_environment(modeling)
    if not report.applies:
        log.info("the model has no linear-attention layers: the fast-path check does not apply")
        return report
    if watcher is not None and watcher.fallback_logged:
        report = FastPathReport(
            False,
            report.reasons + ("transformers logged the torch fallback while the model was built",),
            report.components,
            report.linear_attention_layers,
            report.versions,
        )
    if not report.available:
        if require_fast_path:
            raise FastPathUnavailable(report)
        log.warning("training on the torch fallback: %s", "; ".join(report.reasons))
    return report
