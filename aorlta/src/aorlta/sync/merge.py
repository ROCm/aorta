"""C5 step 1, the merge (design Section 6.5), and the trainer's W_eff cache (Section 6.4).

The engine serves W_eff = bf16(W.float() + (alpha/r) * B.float() @ A.float()) for the LoRA target modules. The
trainer's forward uses exactly those tensors, so both sides call `merge_all`, and C5 writes the tensors it returns.

`merge_all` also reports what the merge did:
- `changed(n)`: elements where W_eff(n) differs bitwise from W_eff(n-1), the previous step's merge. The landing
  check needs it against the previous step, not the base: from step 2 on the two differ.
- `coverage(n) = ||W_eff(n) - W||_F / ||(alpha/r) B A||_F`: how much of the LoRA update survives bf16 rounding.
- the changed fractions, against the previous step and against the base.
- `aligned(n) = <W_eff(n) - W, (alpha/r) B A> / ||(alpha/r) B A||_F^2` and `residual(n) = ||(W_eff(n) - W) -
  (alpha/r) B A||_F / ||(alpha/r) B A||_F`: the share of the update the served change carries in the update's own
  direction, and what rounding added or removed. Coverage alone cannot tell the two apart, because an element that
  does change jumps by a whole bf16 step, larger than its update.

The fp32 -> bf16 cast rounds to nearest by default. `rounding="stochastic"` (the `sync.stochastic_rounding`
switch) rounds each element up with probability equal to how far it sits towards the next bf16
value, so the served weights equal W + (alpha/r) B A in expectation. The random bits come from a generator seeded
by `rounding_seed(run_seed, version, key)`: one merge call produces the tensors both the trainer's cache and the
writer use, and a re-merge of the same latent at the same version on the same device gives the same bits.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from aorlta.ids import canonical_json

ROUNDINGS = ("nearest", "stochastic")
ROUNDING_SEED_TAG = "aorlta.stochastic_rounding/1"

# PEFT target modules (the prototype's TARGETS) and the checkpoint keys they map to (Section 6.4, Model): the
# attention and MLP projections of every text decoder layer, and in Qwen3.5 the linear-attention projections too (a
# Qwen3 layer has none, so those names match nothing there). The decoder's layers are `model.language_model.layers.N`
# in a Qwen3.5 checkpoint and `model.layers.N` in a Qwen3 one (`model_family`); Qwen3.5's vision tower and its MTP
# head (`mtp.layers.0`) are never targets.
TARGET_MODULE_PATTERN = (
    r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))"
)
TARGET_KEY_RE = re.compile(
    r"model\.(?:language_model\.)?layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj)|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))\.weight"
)
GROUP_RE = re.compile(r"\.(self_attn|mlp|linear_attn)\.([a-z_]+)\.weight$")


class MergeError(ValueError):
    """The merge's inputs do not fit together."""


def is_target_key(key: str) -> bool:
    return TARGET_KEY_RE.fullmatch(key) is not None


def group_of(key: str) -> str:
    """`self_attn.q_proj`, `mlp.down_proj`, `linear_attn.in_proj_qkv`, ...; used only to break stats down."""
    m = GROUP_RE.search(key)
    return f"{m.group(1)}.{m.group(2)}" if m else "other"


def _check(key: str, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    if w.dtype != torch.bfloat16:
        raise MergeError(f"{key}: the base weight must be bf16, got {w.dtype}")
    if w.dim() != 2 or a.dim() != 2 or b.dim() != 2:
        raise MergeError(f"{key}: W, A and B must be 2-D, got {tuple(w.shape)}, {tuple(a.shape)}, {tuple(b.shape)}")
    if b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != tuple(w.shape):
        raise MergeError(f"{key}: B {tuple(b.shape)} @ A {tuple(a.shape)} does not give W's shape {tuple(w.shape)}")


def rounding_seed(run_seed: int, version: int, key: str) -> int:
    """The stochastic-rounding seed of one tensor at one version: the first 8 bytes of sha256 over the canonical
    JSON of [tag, run_seed, version, key], little-endian, masked to 63 bits."""
    if not isinstance(run_seed, int) or isinstance(run_seed, bool):
        raise MergeError(f"run_seed must be an int, got {run_seed!r}")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise MergeError(f"version must be an int >= 0, got {version!r}")
    if not isinstance(key, str) or not key:
        raise MergeError(f"key must be a non-empty str, got {key!r}")
    digest = hashlib.sha256(canonical_json([ROUNDING_SEED_TAG, run_seed, version, key])).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def stochastic_bf16(x: torch.Tensor, seed: int) -> torch.Tensor:
    """fp32 -> bf16, rounding each finite element to one of its two bf16 neighbours with probability proportional
    to its distance from the other: the low 16 bits of the fp32 pattern plus a uniform draw from [0, 2**16), then
    truncated. Exactly unbiased; a value already in bf16 is unchanged; non-finite elements round to nearest. The
    draws come from one generator on x's device seeded with `seed`."""
    if x.dtype != torch.float32:
        raise MergeError(f"stochastic rounding takes fp32, got {x.dtype}")
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    noise = torch.randint(0, 1 << 16, x.shape, generator=gen, device=x.device, dtype=torch.int32)
    rounded = ((x.contiguous().view(torch.int32) + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    return torch.where(torch.isfinite(x), rounded, x.to(torch.bfloat16))


def _merge_one(
    w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(W_eff in bf16, the fp32 update (alpha/r) B A). The formula is Section 6.5's, unfused on purpose. A seed
    selects stochastic rounding for the cast."""
    lora = scale * (b.float() @ a.float())
    exact = w.float() + lora
    if seed is None:
        return exact.to(torch.bfloat16), lora
    return stochastic_bf16(exact, seed), lora


def _check_rounding(rounding: str, run_seed: int | None) -> bool:
    """True for stochastic rounding, False for nearest; anything else, or stochastic without a run seed, raises."""
    if rounding not in ROUNDINGS:
        raise MergeError(f"rounding must be one of {ROUNDINGS}, got {rounding!r}")
    if rounding == "stochastic" and run_seed is None:
        raise MergeError("stochastic rounding needs the run seed")
    return rounding == "stochastic"


def merge_weight(
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    key: str = "weight",
    *,
    rounding: str = "nearest",
    run_seed: int | None = None,
    version: int = 0,
) -> torch.Tensor:
    """W_eff = bf16(W.float() + scale * B.float() @ A.float()) for one module, rounded as `merge_all` would round
    `key` at `version`."""
    _check(key, w, a, b)
    stochastic = _check_rounding(rounding, run_seed)
    with torch.no_grad():
        return _merge_one(w, a, b, scale, rounding_seed(run_seed, version, key) if stochastic else None)[0]


def bits(t: torch.Tensor) -> torch.Tensor:
    """A bf16 tensor's bit patterns, so that 'changed' means changed bitwise (NaN and -0.0 included)."""
    if t.dtype != torch.bfloat16:
        raise MergeError(f"expected bf16, got {t.dtype}")
    return t.view(torch.int16)


@dataclass
class GroupStats:
    changed: int = 0
    changed_vs_base: int = 0
    total: int = 0
    delta_sq: float = 0.0
    lora_sq: float = 0.0
    dot: float = 0.0  # <W_eff - W, (alpha/r) B A>

    def add(self, other: "GroupStats") -> None:
        self.changed += other.changed
        self.changed_vs_base += other.changed_vs_base
        self.total += other.total
        self.delta_sq += other.delta_sq
        self.lora_sq += other.lora_sq
        self.dot += other.dot

    @property
    def coverage(self) -> float | None:
        return math.sqrt(self.delta_sq / self.lora_sq) if self.lora_sq > 0 else None

    @property
    def aligned(self) -> float | None:
        return self.dot / self.lora_sq if self.lora_sq > 0 else None

    @property
    def residual(self) -> float | None:
        if self.lora_sq <= 0:
            return None
        return math.sqrt(max(0.0, self.delta_sq - 2 * self.dot + self.lora_sq) / self.lora_sq)

    def as_dict(self) -> dict:
        return {
            "changed": self.changed,
            "changed_vs_base": self.changed_vs_base,
            "total": self.total,
            "changed_frac": self.changed / self.total if self.total else None,
            "changed_vs_base_frac": self.changed_vs_base / self.total if self.total else None,
            "delta_fro": math.sqrt(self.delta_sq),
            "lora_fro": math.sqrt(self.lora_sq),
            "coverage": self.coverage,
            "aligned": self.aligned,
            "residual": self.residual,
        }


@dataclass(frozen=True)
class MergeStats:
    version: int
    changed: int  # elements where W_eff(n) != W_eff(n-1), bitwise
    changed_vs_base: int  # elements where W_eff(n) != W, bitwise
    total: int
    delta_fro: float  # ||W_eff(n) - W||_F
    lora_fro: float  # ||(alpha/r) B A||_F
    seconds: float
    groups: dict[str, dict] = field(default_factory=dict)
    delta_dot: float = 0.0  # <W_eff(n) - W, (alpha/r) B A>
    rounding: str = "nearest"

    @property
    def coverage(self) -> float | None:
        """None when the update is exactly zero (B = 0), where the ratio is undefined."""
        return self.delta_fro / self.lora_fro if self.lora_fro > 0 else None

    @property
    def aligned(self) -> float | None:
        return self.delta_dot / self.lora_fro**2 if self.lora_fro > 0 else None

    @property
    def residual(self) -> float | None:
        if self.lora_fro <= 0:
            return None
        sq = self.delta_fro**2 - 2 * self.delta_dot + self.lora_fro**2
        return math.sqrt(max(0.0, sq)) / self.lora_fro

    @property
    def changed_frac(self) -> float:
        return self.changed / self.total

    @property
    def changed_vs_base_frac(self) -> float:
        return self.changed_vs_base / self.total

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "changed": self.changed,
            "changed_vs_base": self.changed_vs_base,
            "total": self.total,
            "changed_frac": self.changed_frac,
            "changed_vs_base_frac": self.changed_vs_base_frac,
            "delta_fro": self.delta_fro,
            "lora_fro": self.lora_fro,
            "coverage": self.coverage,
            "aligned": self.aligned,
            "residual": self.residual,
            "seconds": self.seconds,
            "groups": self.groups,
            "rounding": self.rounding,
        }


@dataclass(frozen=True)
class MergeResult:
    weights: dict[str, torch.Tensor]  # checkpoint key -> W_eff (bf16), on the base weights' device
    stats: MergeStats


def merge_all(
    base: Mapping[str, torch.Tensor],
    factors: Mapping[str, tuple[torch.Tensor, torch.Tensor, float]],
    prev: Mapping[str, torch.Tensor],
    version: int,
    *,
    rounding: str = "nearest",
    run_seed: int | None = None,
) -> MergeResult:
    """Merge every module. `base` and `factors` (A, B, scale) are keyed by checkpoint key; `prev` is W_eff(n-1),
    the tensors the previous merge returned (for n = 1, the v0 merge, which equals the base when B = 0).
    `rounding="stochastic"` seeds each module's cast with `rounding_seed(run_seed, version, key)`."""
    stochastic = _check_rounding(rounding, run_seed)
    keys = sorted(base)
    if set(factors) != set(keys) or set(prev) != set(keys):
        raise MergeError(
            f"base, factors and prev must name the same modules: {len(keys)} / {len(factors)} / {len(prev)}; "
            f"differences: {sorted(set(keys) ^ set(factors))[:3]} {sorted(set(keys) ^ set(prev))[:3]}"
        )
    t0 = time.monotonic()
    out: dict[str, torch.Tensor] = {}
    groups: dict[str, GroupStats] = {}
    with torch.no_grad():
        for key in keys:
            w = base[key]
            a, b, scale = factors[key]
            _check(key, w, a, b)
            old = prev[key]
            if old.shape != w.shape or old.dtype != torch.bfloat16:
                raise MergeError(f"{key}: the previous W_eff is {old.dtype} {tuple(old.shape)}, not bf16 {tuple(w.shape)}")
            merged, lora = _merge_one(w, a, b, scale, rounding_seed(run_seed, version, key) if stochastic else None)
            delta = merged.float() - w.float()
            g = GroupStats(
                changed=int((bits(merged) != bits(old)).sum()),
                changed_vs_base=int((bits(merged) != bits(w)).sum()),
                total=merged.numel(),
                delta_sq=float(delta.double().pow(2).sum()),
                lora_sq=float(lora.double().pow(2).sum()),
                dot=float((delta.double() * lora.double()).sum()),
            )
            del delta
            groups.setdefault(group_of(key), GroupStats()).add(g)
            out[key] = merged
    if out and next(iter(out.values())).is_cuda:
        torch.cuda.synchronize()
    total = GroupStats()
    for g in groups.values():
        total.add(g)
    stats = MergeStats(
        version=version,
        changed=total.changed,
        changed_vs_base=total.changed_vs_base,
        total=total.total,
        delta_fro=math.sqrt(total.delta_sq),
        lora_fro=math.sqrt(total.lora_sq),
        seconds=time.monotonic() - t0,
        groups={name: g.as_dict() for name, g in sorted(groups.items())},
        delta_dot=total.dot,
        rounding=rounding,
    )
    return MergeResult(weights=out, stats=stats)
