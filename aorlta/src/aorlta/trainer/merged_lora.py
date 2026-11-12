"""The merged-weight forward (design Section 6.4, decision D3).

TokenSpeed has no LoRA adapters, so it serves W_eff = bf16(W + (alpha/r) B A) for the LoRA target modules. Small
updates round away in that cast. The trainer therefore runs its forward on the same W_eff, and passes the LoRA
gradient straight through the rounding:

    forward:  y  = x @ W_eff.T                        (exactly F.linear(x, W_eff))
    backward: dx = G @ W_eff
              dB = (alpha/r) * G.T @ (x @ A.T)
              dA = (alpha/r) * (G @ B).T @ x

`MergedLoRA` keeps W_eff for every PEFT LoRA layer, refreshes it with `aorlta.sync.merge.merge_all` after every
optimizer step (the function whose output C5 writes), and replaces each layer's forward. PEFT still selects the
modules, holds A and B, and writes the adapter format.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.nn.functional as F

from aorlta.sync.merge import MergeResult, merge_all
from aorlta.sync.writer import name_map


class MergedLoRAError(RuntimeError):
    """The model cannot run the merged-weight forward as configured."""


class MergedLoRALinear(torch.autograd.Function):
    """y = F.linear(x, w_eff, bias); gradients for x, lora_a and lora_b with the rounding treated as identity.

    The forward runs under the caller's autocast state, so it is bitwise the plain F.linear call. Backward runs
    with autocast off, so its dtypes are explicit: dx in the dtype the forward's matmul used, and the low-rank
    products in the autocast dtype if autocast was on (as an ordinary PEFT LoRA backward does), else in the
    promoted dtype of x and A. Gradients come back in each input's own dtype.
    """

    @staticmethod
    def forward(ctx, x, w_eff, lora_a, lora_b, scale, bias=None):  # noqa: D401
        y = F.linear(x, w_eff, bias)
        device_type = x.device.type
        ctx.lowp = torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else None
        ctx.lin_dtype = y.dtype
        ctx.scale = float(scale)
        ctx.save_for_backward(x, w_eff, lora_a, lora_b)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w_eff, a, b = ctx.saved_tensors
        s = ctx.scale
        g = grad_out.reshape(-1, grad_out.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        dx = da = db = None
        if ctx.needs_input_grad[0]:
            dx = (g.to(ctx.lin_dtype) @ w_eff.to(ctx.lin_dtype)).to(x.dtype).reshape(x.shape)
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            cd = ctx.lowp or torch.promote_types(x.dtype, a.dtype)
            gc, xc = g.to(cd), x2.to(cd)
            if ctx.needs_input_grad[3]:
                xa = xc @ a.to(cd).t()  # [N, r]
                db = (gc.t() @ xa).to(b.dtype) * s
            if ctx.needs_input_grad[2]:
                gb = gc @ b.to(cd)  # [N, r]
                da = (gb.t() @ xc).to(a.dtype) * s
        return dx, None, da, db, None, None


def merged_lora_linear(x, w_eff, lora_a, lora_b, scale, bias=None):
    return MergedLoRALinear.apply(x, w_eff, lora_a, lora_b, scale, bias)


def lora_layers(model: torch.nn.Module, adapter: str = "default") -> dict[str, torch.nn.Module]:
    """Every PEFT LoRA layer that holds `adapter`, by module name."""
    out = {}
    for name, mod in model.named_modules():
        lora_a = getattr(mod, "lora_A", None)
        if isinstance(lora_a, torch.nn.ModuleDict) and adapter in lora_a and hasattr(mod, "base_layer"):
            out[name] = mod
    return out


def _check_layer(name: str, layer: torch.nn.Module, adapter: str) -> None:
    dropout = layer.lora_dropout[adapter]
    if not isinstance(dropout, torch.nn.Identity):
        raise MergedLoRAError(
            f"{name}: LoRA dropout is {dropout!r}; the merged-weight forward cannot apply dropout "
            "(Section 6.4: ask whether the run plan relies on it)"
        )
    if getattr(layer, "use_dora", {}).get(adapter, False):
        raise MergedLoRAError(f"{name}: DoRA is not a plain LoRA update and cannot be merged this way")
    if getattr(layer, "merged_adapters", None):
        raise MergedLoRAError(f"{name}: PEFT has merged the adapter into the base weight; unmerge it first")
    if list(getattr(layer, "active_adapters", [adapter])) != [adapter]:
        raise MergedLoRAError(f"{name}: active adapters {layer.active_adapters!r}, expected only {adapter!r}")
    if not isinstance(layer.base_layer, torch.nn.Linear):
        raise MergedLoRAError(f"{name}: base layer is {type(layer.base_layer).__name__}, not nn.Linear")
    if layer.base_layer.weight.requires_grad:
        raise MergedLoRAError(f"{name}: the base weight must be frozen")


class MergedLoRA:
    """The LoRA target layers of a PEFT model, their checkpoint keys, and the cached W_eff they run on.

    `base_index_keys` are the base checkpoint's tensor names; every LoRA layer must map to exactly one of the
    target keys among them, and the targets must all be covered (the name map of Section 6.5, step 2).
    Version 0 is the merge of the latent as constructed (the base itself while B = 0). `rounding` and `run_seed`
    go to every `merge_all` call (`sync.stochastic_rounding` selects "stochastic"); the tensors a refresh returns
    are the ones the forward then uses, so the writer and the trainer share the same random bits.
    """

    def __init__(self, model: torch.nn.Module, base_index_keys, *, adapter: str = "default", version: int = 0,
                 rounding: str = "nearest", run_seed: int | None = None):
        layers = lora_layers(model, adapter)
        if not layers:
            raise MergedLoRAError("the model has no LoRA layers for adapter " + repr(adapter))
        for name, layer in layers.items():
            _check_layer(name, layer, adapter)
        self.adapter = adapter
        self.module_keys = name_map(list(layers), list(base_index_keys))
        self.layers = {self.module_keys[name]: layer for name, layer in layers.items()}
        self.base = {key: layer.base_layer.weight for key, layer in self.layers.items()}
        self._installed: dict[str, torch.nn.Module] = {}
        self._capture: dict[str, torch.Tensor] | None = None
        self.rounding, self.run_seed = rounding, run_seed
        first = merge_all(self.base, self.factors(), self.base, version, rounding=rounding, run_seed=run_seed)
        self._weights = first.weights
        self.version = version
        self.last_stats = first.stats

    @property
    def keys(self) -> list[str]:
        return sorted(self.layers)

    @property
    def weights(self) -> Mapping[str, torch.Tensor]:
        """The W_eff the trainer's forward uses now (read-only view)."""
        return dict(self._weights)

    def factors(self) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
        out = {}
        for key, layer in self.layers.items():
            out[key] = (
                layer.lora_A[self.adapter].weight.detach(),
                layer.lora_B[self.adapter].weight.detach(),
                float(layer.scaling[self.adapter]),
            )
        return out

    def refresh(self) -> MergeResult:
        """Merge the current latent, make it the trainer's W_eff, and return it for C5 to write.

        `changed(n)` is counted against the W_eff the trainer used until now, W_eff(n-1)."""
        result = merge_all(self.base, self.factors(), self._weights, self.version + 1, rounding=self.rounding,
                           run_seed=self.run_seed)
        self._weights = result.weights
        self.version = result.stats.version
        self.last_stats = result.stats
        return result

    def step(self, optimizer: torch.optim.Optimizer) -> MergeResult:
        """One optimizer step, then the refresh: W_eff is never stale after a step."""
        optimizer.step()
        return self.refresh()

    def install(self) -> "MergedLoRA":
        for key, layer in self.layers.items():
            if key in self._installed:
                continue
            layer.forward = self._make_forward(key, layer)
            self._installed[key] = layer
        return self

    def uninstall(self) -> None:
        for layer in self._installed.values():
            del layer.forward
        self._installed.clear()

    @contextlib.contextmanager
    def capture(self) -> Iterator[dict[str, torch.Tensor]]:
        """Record the W_eff tensor each layer's forward actually used, keyed by checkpoint key."""
        seen: dict[str, torch.Tensor] = {}
        self._capture = seen
        try:
            yield seen
        finally:
            self._capture = None

    def _make_forward(self, key: str, layer: torch.nn.Module):
        adapter = self.adapter

        def forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if args or kwargs:
                raise MergedLoRAError(f"{key}: unexpected forward arguments {args!r} {sorted(kwargs)!r}")
            if layer.disable_adapters:  # PEFT's disable_adapter(): the reference model, W_eff = W
                return layer.base_layer(x)
            w_eff = self._weights[key]
            if self._capture is not None:
                self._capture[key] = w_eff
            return MergedLoRALinear.apply(
                x,
                w_eff,
                layer.lora_A[adapter].weight,
                layer.lora_B[adapter].weight,
                float(layer.scaling[adapter]),
                layer.base_layer.bias,
            )

        return forward
