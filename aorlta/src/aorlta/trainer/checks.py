"""C4's checks every step (design Section 6.4), the old trainer's adapted to LoRA.

From `train_grpo_step.py` (rocjitsu-env 54cc4ce1): `fingerprint`, `bit_identical` and the shape of
`update_checks`, and the gap gate's constants `MAX_GENERATION_GAP` (2.0) and `MAX_OFF_POLICY_FRACTION` (0.02),
which arrive here as `trainer.max_generation_gap` and `trainer.max_off_policy_fraction`.

Before the optimizer step (a failure refuses the step; nothing changes):
- `gradient_is_finite_and_nonzero`;
- `loss_is_finite`;
- `off_policy_fraction`: more than `max_off_policy_fraction` of the compared samples were excluded by the gap gate;
- `loss_temperature_applied`: the log-probs the loss used are log_softmax(z / T_loss), recomputed independently;
- `groups_trained_once`: every rank holds the same assignment ledger, and every ledger group was trained exactly
  once, on its own rank (`ranks.assignment_check`).

After it (a failure stops the run before the weights are published, so the engines never serve them):
- `trained_tensors_are_finite`;
- `every_trained_tensor_moved`: every B moved, and every A moved unless its B was exactly zero before the
  step. Then dL/dA is exactly zero (dA = s (G B)^T x), so A must be bitwise unchanged; that is what happens at
  step 1 with PEFT's B = 0 init. With weight decay > 0 such an A is not checked, since decay moves it.
- `frozen_control_unchanged`: `embed_tokens` and 8 sampled target base weights are bit-identical to load time;
- `w_eff_refreshed`: the merged weights were rebuilt after the step (their version advanced);
- `replicas_identical`: every rank holds the same LoRA latent after the step, and over R > 1 ranks the same W_eff
  (an exact checksum of every target tensor, `ranks.weights_checksum`).

Advisory, recorded but not gating, as in the old trainer: `step_descends_the_gradient`, cos(delta, -g) on an
audit subset. Adam's per-coordinate scaling does not guarantee a positive cosine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from aorlta.sync.writer import tensor_xxh64

GATING_PRE = ("gradient_is_finite_and_nonzero", "loss_is_finite", "off_policy_fraction", "loss_temperature_applied",
              "groups_trained_once")
GATING_POST = ("trained_tensors_are_finite", "every_trained_tensor_moved", "frozen_control_unchanged",
               "w_eff_refreshed", "replicas_identical")
ADVISORY = ("step_descends_the_gradient",)
N_SAMPLED_TARGETS = 8


class TrainerCheckFailed(RuntimeError):
    """A gating check failed: the run stops before this step's weights are published."""

    def __init__(self, failed: list[str], report: dict):
        super().__init__(f"trainer checks failed: {failed}")
        self.failed = failed
        self.report = report


class StepRefused(TrainerCheckFailed):
    """A check before the optimizer step failed, so the step was not taken."""


def fingerprint(t: torch.Tensor) -> tuple[float, float, float]:
    """(sum, sum of squares, sum of absolutes) in float64, for the finiteness check. A permutation of the elements
    keeps all three, so they cannot tell whether a tensor moved or two tensors are the same (`digest` can)."""
    f = t.detach().double()
    return float(f.sum()), float(f.pow(2).sum()), float(f.abs().sum())


def digest(t: torch.Tensor) -> tuple[str, tuple[int, ...], str]:
    """(dtype, shape, xxh64 of the bytes): equal for two tensors with the same bits in the same places."""
    return str(t.dtype).removeprefix("torch."), tuple(t.shape), tensor_xxh64(t.detach().cpu())


def bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """The same bits (torch.equal treats -0.0 as 0.0 and NaN as unequal to itself; raw bytes do not)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8)))


# --- what is trained, and the negative control -------------------------------------------------------


@dataclass(frozen=True)
class LoraTensors:
    """The trained latent: for each PEFT LoRA module (by name), its A and B weights."""

    modules: dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]]

    @classmethod
    def of(cls, model, adapter: str = "default") -> "LoraTensors":
        mods = {}
        for name, m in model.named_modules():
            a = getattr(m, "lora_A", None)
            if isinstance(a, torch.nn.ModuleDict) and adapter in a and hasattr(m, "base_layer"):
                mods[name] = (a[adapter].weight, m.lora_B[adapter].weight)
        if not mods:
            raise ValueError(f"the model has no LoRA modules for adapter {adapter!r}")
        return cls(dict(sorted(mods.items())))

    def params(self) -> list[torch.nn.Parameter]:
        return [p for a, b in self.modules.values() for p in (a, b)]

    def named(self) -> dict[str, torch.nn.Parameter]:
        out = {}
        for name, (a, b) in self.modules.items():
            out[f"{name}.lora_A"] = a
            out[f"{name}.lora_B"] = b
        return out


def structural_problems(model, lora: LoraTensors, optimizer: torch.optim.Optimizer | None = None) -> list[str]:
    """Only the LoRA latent is trainable, and the optimizer holds exactly it."""
    trained = {id(p) for p in lora.params()}
    problems = [f"{n} requires grad but is not a LoRA weight" for n, p in model.named_parameters()
                if p.requires_grad and id(p) not in trained]
    problems += [f"{n} is a LoRA weight that does not require grad" for n, p in lora.named().items()
                 if not p.requires_grad]
    if optimizer is not None:
        held = [id(p) for g in optimizer.param_groups for p in g["params"]]
        if sorted(held) != sorted(trained) or len(held) != len(set(held)):
            problems.append("the optimizer does not hold exactly the LoRA latent")
    return problems


def _embed_weight(model) -> tuple[str, torch.Tensor]:
    for name, p in model.named_parameters():
        if name.endswith("embed_tokens.weight"):
            return name, p
    raise ValueError("the model has no embed_tokens weight for the negative control")


@dataclass
class BaseControl:
    """Exact copies, taken at load, of `embed_tokens` and 8 target base weights spanning depth."""

    tensors: dict[str, torch.Tensor]
    copies: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model, lora: LoraTensors, n_targets: int = N_SAMPLED_TARGETS) -> "BaseControl":
        name, emb = _embed_weight(model)
        tensors = {name: emb}
        mods = dict(model.named_modules())
        names = list(lora.modules)
        step = max(1, len(names) // n_targets)
        for mname in names[::step][:n_targets]:
            tensors[f"{mname}.base_layer.weight"] = mods[mname].base_layer.weight
        return cls(tensors, {k: t.detach().clone() for k, t in tensors.items()})

    def moved(self) -> list[str]:
        return sorted(k for k, t in self.tensors.items() if not bit_identical(t, self.copies[k]))


# --- the step ------------------------------------------------------------------------------------------


@dataclass
class PreStep:
    """What the post-step checks compare against: taken after the final gradient, right before the step."""

    digests: dict[str, tuple[str, tuple[int, ...], str]]
    b_zero: dict[str, bool]  # module -> its B was exactly zero and its A had no first moment: A cannot move
    audit: dict[str, torch.Tensor]
    audit_grad: dict[str, torch.Tensor]
    w_eff_version: Any

    @classmethod
    def take(cls, lora: LoraTensors, merged, n_audit: int = 8, *, optimizer=None) -> "PreStep":
        named = lora.named()
        digests = {n: digest(p) for n, p in named.items()}
        state = getattr(optimizer, "state", {})

        def still(a) -> bool:  # AdamW moves A at a zero gradient while its first moment is not zero
            m = state.get(a, {}).get("exp_avg") if hasattr(state, "get") else None
            return m is None or not bool(m.ne(0).any())

        b_zero = {m: not bool(b.detach().ne(0).any()) and still(a) for m, (a, b) in lora.modules.items()}
        names = sorted(named)
        pick = names[:: max(1, len(names) // n_audit)][:n_audit]
        audit = {n: named[n].detach().clone() for n in pick}
        grads = {n: (named[n].grad.detach().clone() if named[n].grad is not None else torch.zeros_like(named[n]))
                 for n in pick}
        return cls(digests, b_zero, audit, grads, getattr(merged, "version", None))


def cosine_delta_vs_negative_grad(pre: PreStep, lora: LoraTensors) -> float:
    named = lora.named()
    dot = nd = ng = 0.0
    for n, before in pre.audit.items():
        d = (named[n].detach() - before).double()
        g = pre.audit_grad[n].double()
        dot += float((d * -g).sum())
        nd += float(d.pow(2).sum())
        ng += float(g.pow(2).sum())
    return dot / (math.sqrt(nd) * math.sqrt(ng) + 1e-30)


def post_step_checks(pre: PreStep, lora: LoraTensors, control: BaseControl, merged, *, weight_decay: float,
                     replicas_identical: bool) -> dict[str, dict]:
    named = lora.named()
    non_finite = sorted(n for n, p in named.items() if not all(math.isfinite(x) for x in fingerprint(p)))
    post = {n: digest(p) for n, p in named.items()}
    unmoved, a_moved_with_b_zero, a_unchecked = [], [], []
    for mname in lora.modules:
        a_key, b_key = f"{mname}.lora_A", f"{mname}.lora_B"
        if post[b_key] == pre.digests[b_key]:
            unmoved.append(b_key)
        a_same = post[a_key] == pre.digests[a_key]
        if not pre.b_zero[mname]:
            if a_same:
                unmoved.append(a_key)
        elif weight_decay == 0:
            if not a_same:
                a_moved_with_b_zero.append(a_key)
        else:
            a_unchecked.append(a_key)
    moved_control = control.moved()
    version_after = getattr(merged, "version", None)
    refreshed = pre.w_eff_version is not None and version_after is not None and version_after == pre.w_eff_version + 1
    cosine = cosine_delta_vs_negative_grad(pre, lora)
    return {
        "trained_tensors_are_finite": {"non_finite": len(non_finite), "examples": non_finite[:5],
                                       "passed": bool(post) and not non_finite},
        "every_trained_tensor_moved": {
            "tensors": len(post), "unmoved": len(unmoved), "unmoved_examples": unmoved[:5],
            "a_expected_unchanged": sum(pre.b_zero.values()) if weight_decay == 0 else 0,
            "a_moved_while_b_was_zero": a_moved_with_b_zero[:5], "a_not_checked": len(a_unchecked),
            "passed": bool(post) and not unmoved and not a_moved_with_b_zero,
        },
        "frozen_control_unchanged": {"tensors": sorted(control.tensors), "moved": moved_control,
                                     "passed": bool(control.tensors) and not moved_control},
        "w_eff_refreshed": {"version_before": pre.w_eff_version, "version_after": version_after, "passed": refreshed},
        "replicas_identical": {"passed": bool(replicas_identical)},
        "step_descends_the_gradient": {"cosine_delta_vs_negative_grad": cosine, "passed": cosine > 0.0,
                                       "advisory": True},
    }


def pre_step_checks(*, grad_norm: float, loss: float, compared: int, excluded: int, max_fraction: float,
                    temperature_check: Mapping[str, Any]) -> dict[str, dict]:
    fraction = excluded / compared if compared else (1.0 if excluded else 0.0)
    return {
        "gradient_is_finite_and_nonzero": {"grad_norm": grad_norm,
                                           "passed": math.isfinite(grad_norm) and grad_norm > 0.0},
        "loss_is_finite": {"loss": loss, "passed": math.isfinite(loss)},
        "off_policy_fraction": {"compared": compared, "excluded": excluded, "fraction": fraction,
                                "bound": max_fraction, "passed": compared > 0 and fraction <= max_fraction},
        "loss_temperature_applied": dict(temperature_check),
    }


def failed(checks: Mapping[str, Mapping[str, Any]], names) -> list[str]:
    return [n for n in names if not checks[n]["passed"]]


@dataclass
class TemperatureAudit:
    """Accumulates the loss-temperature check over micro-batches: the loss's log-probs against
    log_softmax(z / T_loss) recomputed from the same hidden states, and their distance from the raw ones."""

    temperature: float
    compared: int = 0
    mismatched: int = 0
    max_abs_vs_recomputed: float = 0.0
    max_abs_vs_raw: float = 0.0
    notes: list[str] = field(default_factory=list)

    def add(self, used: torch.Tensor, recomputed: torch.Tensor, raw: torch.Tensor) -> None:
        used, recomputed, raw = used.detach().double(), recomputed.detach().double(), raw.detach().double()
        self.compared += int(used.numel())
        self.mismatched += int((used != recomputed).sum())
        if used.numel():
            self.max_abs_vs_recomputed = max(self.max_abs_vs_recomputed, float((used - recomputed).abs().max()))
            self.max_abs_vs_raw = max(self.max_abs_vs_raw, float((used - raw).abs().max()))

    def merge(self, other: "TemperatureAudit") -> None:
        self.compared += other.compared
        self.mismatched += other.mismatched
        self.max_abs_vs_recomputed = max(self.max_abs_vs_recomputed, other.max_abs_vs_recomputed)
        self.max_abs_vs_raw = max(self.max_abs_vs_raw, other.max_abs_vs_raw)

    def result(self) -> dict:
        # At T != 1 the loss's log-probs must differ from the raw ones somewhere: equality means T was ignored.
        distinct = self.temperature == 1.0 or self.max_abs_vs_raw > 0.0
        return {"temperature": self.temperature, "compared": self.compared, "mismatched": self.mismatched,
                "max_abs_vs_recomputed": self.max_abs_vs_recomputed, "max_abs_vs_raw": self.max_abs_vs_raw,
                "passed": self.compared > 0 and self.mismatched == 0 and distinct}


def gap_summary(gaps: list) -> dict:
    """Distribution of the per-sample gap statistics (for the report and C4.T10)."""
    if not gaps:
        return {"samples": 0}
    out: dict[str, Any] = {"samples": len(gaps)}
    for key in ("max_abs", "mean_abs", "median_abs", "p99_abs", "frac_gt_1", "mean_signed"):
        v = torch.tensor([getattr(g, key) for g in gaps], dtype=torch.float64)
        q = torch.quantile(v, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)).tolist()
        out[key] = {"median": q[0], "p90": q[1], "p99": q[2], "max": float(v.max()), "mean": float(v.mean())}
    return out
