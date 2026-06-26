"""``training`` workload: real PyTorch training loops over DDP / FSDP.

Unlike :mod:`aorta.workloads.race` (a runtime corruption / overlap stress
reproducer) this workload runs *normal* model training: forward → loss →
backward → optimizer step, with numeric checks (NaN/inf on loss, gradients,
and model outputs), per-step timings, and a :class:`WorkloadResult` summary.
DDP and FSDP are two variants of the same experiment, selected by the recipe:

.. code-block:: yaml

    workload: training
    workload_config:
      parallelism: fsdp   # ddp|fsdp

Launch modes
------------
* **One-rank smoke** — ``aorta run --workload training --steps 2`` runs a
  singleton distributed group (``WORLD_SIZE`` absent or ``1``). This exercises
  the full lifecycle + JSON schema; it is NOT a distributed-performance signal.
* **Single-node multi-rank** —
  ``torchrun --standalone --nproc_per_node=2 $(which aorta) run
  --workload training --steps 2``.

Distributed ownership
---------------------
The workload owns ``torch.distributed.init_process_group``; the platform does
not initialize distributed. We init inline (CPU→gloo / GPU→nccl) rather than
reusing ``aorta.training.fsdp_trainer.init_distributed`` because that helper
unconditionally calls ``torch.cuda.set_device`` and wires per-rank file
logging, which breaks the CPU / singleton smoke path. The AdamW optimizer and
LR scheduler ARE reused from ``fsdp_trainer`` (``configure_optimizer`` /
``configure_scheduler``), and the FSDP/DDP wrapping blocks are copied from
``build_fsdp_model`` / ``build_ddp_model`` (those functions hardcode
``RankingTransformerModel`` so they are not imported directly).

Reuse: model topologies come from :mod:`aorta.models.repeated_block`
(``RepeatedBlockModel`` — dense for ``transformer``, top-1 MoE for
``moe_transformer``); a small local MLP covers the smallest lifecycle smoke.
Synthetic ``input_ids`` / ``targets`` are generated inline with a seeded
generator (the in-tree ``synthetic_dataset`` is a ranking dataset and does not
produce token-id / class-target tensors).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from aorta.workloads._base import Workload, WorkloadResult

# torch (and the torch-dependent aorta helpers) are imported lazily so this
# module can be IMPORTED for workload discovery / registration in an
# environment without torch (e.g. a CLI venv that only drives docker-based
# runs). setup() raises a clear error if torch is unavailable at run time.
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception as exc:  # pragma: no cover - exercised only in torch-less envs
    _DTYPES: dict[str, "torch.dtype"] = {}
    _IMPORT_ERROR: Exception | None = exc
else:
    from aorta.instrumentation.determinism import enable_deterministic
    from aorta.models import (
        BlockConfig,
        RepeatedBlockModel,
        RepeatedTransformerBlock,
    )

    # Accept both the verbose recipe spellings (issue #238 recipe shape) and
    # the short forms used by llm_determinism, mapping to one torch dtype.
    _DTYPES = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    _IMPORT_ERROR = None

log = logging.getLogger(__name__)

_VALID_DTYPE_NAMES = ("bfloat16", "bf16", "float16", "fp16", "float32", "fp32")
_VALID_PARALLELISM = ("ddp", "fsdp")
_VALID_DEVICES = ("auto", "cuda", "cpu")
_VALID_MODEL_KINDS = ("mlp", "transformer", "moe_transformer")


# --------------------------------------------------------------------------- #
# Typed config
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """Model topology. ``kind`` selects which in-tree model is built."""

    kind: Literal["mlp", "transformer", "moe_transformer"] = "transformer"
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    ffn_size: int = 1024
    vocab_size: int = 32_000
    # MoE knobs. The repeated-block MoE is top-1 only (no top_k); experts are
    # driven purely by ``num_experts``.
    num_experts: int = 4

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        d = dict(d or {})
        moe = dict(d.pop("moe", {}) or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            log.warning(
                "ModelSpec: unknown keys in model config (possible typos): %s",
                sorted(unknown),
            )
        # ``enabled`` is accepted (silently ignored) — the issue/#238 recipe
        # shape includes ``model.moe.enabled`` and the docstring says this module
        # accepts verbose recipe spellings; topology is driven by ``kind`` only.
        unknown_moe = set(moe) - {"num_experts", "enabled"}
        if unknown_moe:
            log.warning(
                "ModelSpec: unknown keys in model.moe config (possible typos): %s",
                sorted(unknown_moe),
            )
        spec = cls(**{k: v for k, v in d.items() if k in known})
        if "num_experts" in moe:
            spec.num_experts = int(moe["num_experts"])
        if spec.kind not in _VALID_MODEL_KINDS:
            raise ValueError(f"model.kind must be one of {list(_VALID_MODEL_KINDS)}, got {spec.kind!r}")
        if spec.num_layers < 1:
            raise ValueError(f"model.num_layers must be >= 1, got {spec.num_layers}")
        if spec.num_experts < 1:
            raise ValueError(f"model.num_experts must be >= 1, got {spec.num_experts}")
        if spec.kind == "moe_transformer" and spec.num_experts < 2:
            raise ValueError(
                f"model.num_experts must be >= 2 for moe_transformer, got {spec.num_experts}"
            )
        if spec.kind in ("transformer", "moe_transformer"):
            if spec.num_heads < 1:
                raise ValueError(
                    f"model.num_heads must be >= 1, got {spec.num_heads}"
                )
            if spec.hidden_size % spec.num_heads != 0:
                raise ValueError(
                    f"model.hidden_size ({spec.hidden_size}) must be divisible by "
                    f"num_heads ({spec.num_heads})"
                )
        return spec

    @property
    def effective_experts(self) -> int:
        """Expert count used for model construction: >=2 for MoE, 1 for dense."""
        if self.kind == "moe_transformer":
            return self.num_experts
        return 1


@dataclass
class OptimizerSpec:
    kind: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizerSpec":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            log.warning(
                "OptimizerSpec: unknown keys in optimizer config (possible typos): %s",
                sorted(unknown),
            )
        spec = cls(**{k: v for k, v in d.items() if k in known})
        if str(spec.kind).lower() != "adamw":
            raise ValueError(f"optimizer.kind must be 'adamw', got {spec.kind!r}")
        if isinstance(spec.betas, list):
            spec.betas = tuple(spec.betas)  # type: ignore[assignment]
        if len(spec.betas) != 2:
            raise ValueError(f"optimizer.betas must be a pair (beta1, beta2), got length {len(spec.betas)}")
        return spec


@dataclass
class ChecksSpec:
    fail_on_nan_loss: bool = True
    fail_on_nan_grad: bool = True
    fail_on_nonfinite_output: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChecksSpec":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            log.warning(
                "ChecksSpec: unknown keys in checks config (possible typos): %s",
                sorted(unknown),
            )
        return cls(**{k: bool(v) for k, v in d.items() if k in known})


@dataclass
class TrainingConfig:
    """Top-level recipe knobs for :class:`TrainingWorkload`."""

    parallelism: Literal["ddp", "fsdp"] = "fsdp"
    seed: int = 1234
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: str = "bfloat16"
    batch_size: int = 2
    seq_len: int = 128
    warmup_steps: int = 1
    steps: int = 4
    model: ModelSpec = field(default_factory=ModelSpec)
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    checks: ChecksSpec = field(default_factory=ChecksSpec)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        d = dict(d or {})
        # ``steps`` arrives from the dispatcher (``--steps`` / recipe field) as
        # config["steps"]; honour it when present.
        model = ModelSpec.from_dict(d.get("model", {}))
        optimizer = OptimizerSpec.from_dict(d.get("optimizer", {}))
        checks = ChecksSpec.from_dict(d.get("checks", {}))
        scalar_keys = {
            "parallelism",
            "seed",
            "device",
            "dtype",
            "batch_size",
            "seq_len",
            "warmup_steps",
            "steps",
        }
        all_known = scalar_keys | {"model", "optimizer", "checks"}
        unknown = {k for k in set(d) - all_known if not k.startswith("_aorta_")}
        if unknown:
            log.warning(
                "TrainingConfig: unknown keys in workload_config (possible typos): %s",
                sorted(unknown),
            )
        cfg = cls(
            model=model,
            optimizer=optimizer,
            checks=checks,
            **{k: d[k] for k in scalar_keys if k in d and d[k] is not None},
        )
        if cfg.parallelism not in _VALID_PARALLELISM:
            raise ValueError(f"parallelism must be one of {list(_VALID_PARALLELISM)}, got {cfg.parallelism!r}")
        if cfg.device not in _VALID_DEVICES:
            raise ValueError(f"device must be one of {list(_VALID_DEVICES)}, got {cfg.device!r}")
        if cfg.dtype not in _VALID_DTYPE_NAMES:
            raise ValueError(f"dtype must be one of {list(_VALID_DTYPE_NAMES)}, got {cfg.dtype!r}")
        if cfg.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
        if cfg.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {cfg.seq_len}")
        if cfg.steps < 1:
            raise ValueError(f"steps must be >= 1, got {cfg.steps}")
        if cfg.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {cfg.warmup_steps}")
        return cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dependency at the base layer)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _build_mlp(spec: ModelSpec) -> "nn.Module":
    """Smallest lifecycle-smoke topology: embedding → MLP → vocab head.

    Returns ``[B, T, vocab]`` logits so the inline train step (cross-entropy
    over the vocab) is identical across all three model kinds.
    """

    class _MlpModel(nn.Module):
        def __init__(self, vocab: int, hidden: int, num_layers: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            layers: list[nn.Module] = []
            for _ in range(num_layers):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.GELU())
            self.mlp = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.mlp(self.embed(input_ids)))

    return _MlpModel(spec.vocab_size, spec.hidden_size, spec.num_layers)


def _build_model(spec: ModelSpec, seq_len: int) -> "nn.Module":
    """Build an unwrapped model for ``spec.kind`` from in-tree definitions."""
    if spec.kind == "mlp":
        return _build_mlp(spec)
    # transformer (dense, num_experts=1) and moe_transformer (top-1 MoE).
    block = BlockConfig(
        vocab_size=spec.vocab_size,
        hidden_size=spec.hidden_size,
        ffn_size=spec.ffn_size,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        seq_len=seq_len,
        num_experts=spec.effective_experts,
    )
    return RepeatedBlockModel(block)


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #
class TrainingWorkload(Workload):
    """Real PyTorch training loop with selectable DDP / FSDP parallelism.

    ``launch_mode = "distributed"`` with ``min_world_size = 1``: a bare
    ``aorta run`` (``WORLD_SIZE`` 1) is a valid singleton smoke, and a
    ``torchrun`` launch exercises the real collective path.
    """

    name: ClassVar[str] = "training"
    launch_mode = "distributed"
    min_world_size = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._cfg: TrainingConfig | None = None
        self._rank = 0
        self._world_size = 1
        self._owns_process_group = False
        self._device = None
        self._dtype = None
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._parameter_count = 0
        self._input_gen = None

    # -- lifecycle ---------------------------------------------------------- #
    def setup(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "TrainingWorkload requires PyTorch, which failed to import: "
                f"{_IMPORT_ERROR!r}"
            )

        cfg = TrainingConfig.from_dict(self.config)
        # ``--steps`` from the dispatcher overrides the recipe/default count.
        if self.config.get("steps") is not None:
            cfg.steps = int(self.config["steps"])
            if cfg.steps < 1:
                raise ValueError(f"steps must be >= 1, got {cfg.steps}")
        self._cfg = cfg

        self._device, backend = self._resolve_device_and_backend(cfg.device)
        self._dtype = _DTYPES[cfg.dtype]

        self._init_distributed(backend)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self._device.type == "cuda":
            torch.cuda.set_device(local_rank)
            self._device = torch.device(f"cuda:{local_rank}")

        # Fixed seed → identical initial weights across ranks; CPU/Python RNGs
        # here, the CUDA device is seeded after set_device (per enable_deterministic).
        enable_deterministic(cfg.seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed(cfg.seed)
        # Per-rank input generator: distinct rank streams, deterministic per step.
        self._input_gen = torch.Generator(device="cpu").manual_seed(
            cfg.seed + 7919 * (self._rank + 1)
        )

        model = _build_model(cfg.model, cfg.seq_len)
        self._parameter_count = sum(p.numel() for p in model.parameters())
        self._model = self._wrap_model(model)

        self._optimizer, self._scheduler = self._build_optim(self._model, cfg)

        if self._rank == 0:
            log.info(
                "TrainingWorkload setup: parallelism=%s model=%s world_size=%d "
                "device=%s dtype=%s params=%d steps=%d",
                cfg.parallelism,
                cfg.model.kind,
                self._world_size,
                self._device,
                cfg.dtype,
                self._parameter_count,
                cfg.steps,
            )

    def run(self) -> WorkloadResult:
        import torch.distributed as dist

        assert self._cfg is not None
        cfg = self._cfg
        vocab = cfg.model.vocab_size
        model = self._model
        optimizer = self._optimizer
        scheduler = self._scheduler

        model.train()
        t0 = time.perf_counter()
        step_times: list[float] = []
        failures: list[dict[str, Any]] = []
        first_failure: int | None = None
        executed = 0
        main_work_started = False
        final_loss = float("nan")

        for step in range(cfg.steps):
            main_work_started = True
            input_ids, targets = self._make_batch(cfg, vocab)

            step_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.reshape(-1, vocab).float(), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if self._device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append((time.perf_counter() - step_t0) * 1000.0)
            executed += 1

            loss_val = float(loss.detach().float().item())
            final_loss = loss_val
            problems = self._numeric_checks(cfg.checks, loss_val, outputs, model)
            if problems:
                if first_failure is None:
                    first_failure = step
                failures.append({"step": step, "rank": self._rank, "problems": problems})
                log.error("[rank %d] numeric check failed at step %d: %s", self._rank, step, problems)

        # Global verdict: any rank seeing a failure fails the cell.
        local_fail = torch.tensor([len(failures)], dtype=torch.long, device=self._device)
        # Reduce earliest failure step across ranks (-1 means no failure locally).
        local_first = torch.tensor(
            [first_failure if first_failure is not None else cfg.steps],
            dtype=torch.long,
            device=self._device,
        )
        if dist.is_initialized():
            dist.all_reduce(local_fail, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_first, op=dist.ReduceOp.MIN)
            dist.barrier()
        global_failures = int(local_fail.item())
        global_first_failure: int | None = int(local_first.item())
        if global_first_failure >= cfg.steps:
            global_first_failure = None

        # When the global verdict is failed but this rank has no local failures,
        # inject a synthetic record so the rank-0 JSON is debuggable.
        if global_failures > 0 and not failures and global_first_failure is not None:
            failures.append({
                "step": global_first_failure,
                "rank": "remote",
                "problems": ["numeric_failure_on_remote_rank"],
            })

        elapsed = time.perf_counter() - t0
        passed = global_failures == 0
        # Percentiles over post-warmup steps when any remain, else all steps.
        timed = step_times[cfg.warmup_steps:] or step_times

        if self._rank == 0:
            log.info(
                "TrainingWorkload %s: %d step(s), final_loss=%.5f, failures=%d",
                "PASSED" if passed else "FAILED",
                cfg.steps,
                final_loss,
                global_failures,
            )

        return WorkloadResult(
            passed=passed,
            failure_count=global_failures,
            first_failure_iteration=global_first_failure,
            failure_details=failures,
            total_iterations=cfg.steps,
            step_times_ms=step_times,
            elapsed_sec=elapsed,
            main_work_started=main_work_started,
            executed_iterations=executed,
            configured_iterations=cfg.steps,
            metrics={
                "parallelism": cfg.parallelism,
                "rank": self._rank,
                "world_size": self._world_size,
                "device": str(self._device),
                "dtype": cfg.dtype,
                "model_kind": cfg.model.kind,
                "parameter_count": self._parameter_count,
                "final_loss": final_loss,
                "step_time_p50": _percentile(timed, 50.0),
                "step_time_p99": _percentile(timed, 99.0),
            },
        )

    def cleanup(self) -> None:
        import torch.distributed as dist

        self._model = None
        self._optimizer = None
        self._scheduler = None
        if dist.is_initialized():
            dist.barrier()

    # -- internals ---------------------------------------------------------- #
    def _resolve_device_and_backend(self, device_pref: str) -> tuple["torch.device", str]:
        if device_pref == "cpu":
            return torch.device("cpu"), "gloo"
        if device_pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but no CUDA/ROCm device is available")
            return torch.device("cuda"), "nccl"
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda"), "nccl"
        return torch.device("cpu"), "gloo"

    def _init_distributed(self, backend: str) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            return
        # Singleton smoke: no launcher set the rendezvous env. Provide local
        # defaults so a bare ``aorta run`` can still form a 1-rank group.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend=backend)
        self._owns_process_group = True
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

    def _wrap_model(self, model: "nn.Module") -> "nn.Module":
        """Apply DDP or FSDP wrapping (blocks copied from fsdp_trainer)."""
        assert self._cfg is not None
        device = self._device
        if self._cfg.parallelism == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP

            # ~15-line DDP wrapping block (build_ddp_model), minus the hardcoded
            # RankingTransformerModel and torch.compile branch.
            device_ids = None
            if device.type == "cuda":
                device_ids = [device.index if device.index is not None else torch.cuda.current_device()]
            # Top-1 MoE leaves unselected experts grad-less, which DDP's bucket
            # reduction rejects unless find_unused_parameters is enabled.
            find_unused = self._cfg.model.effective_experts > 1
            return DDP(
                model.to(device=device, dtype=self._dtype),
                device_ids=device_ids,
                gradient_as_bucket_view=True,
                static_graph=False,
                bucket_cap_mb=25,
                find_unused_parameters=find_unused,
            )

        # FSDP
        from functools import partial

        from torch.distributed.fsdp import (
            BackwardPrefetch,
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # ~15-line FSDP wrapping block (build_fsdp_model). The auto-wrap policy
        # targets RepeatedTransformerBlock (build_fsdp_model used
        # nn.TransformerEncoderLayer for the ranking model); harmless for the
        # MLP smoke which has no such submodule.
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={RepeatedTransformerBlock},
        )
        return FSDP(
            model.to(device=device, dtype=self._dtype),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            forward_prefetch=True,
            device_id=torch.cuda.current_device() if device.type == "cuda" else None,
            sync_module_states=self._world_size > 1,
        )

    def _build_optim(self, model: "nn.Module", cfg: TrainingConfig):
        # AdamW + LR scheduler reused directly from fsdp_trainer (issue #238).
        from aorta.training.fsdp_trainer import (
            OptimizerConfig,
            SchedulerConfig,
            configure_optimizer,
            configure_scheduler,
        )

        opt_cfg = OptimizerConfig(
            name="adamw",
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
        )
        optimizer = configure_optimizer(model, opt_cfg, cfg.parallelism)
        sched_cfg = SchedulerConfig(warmup_steps=cfg.warmup_steps, total_steps=cfg.steps)
        scheduler = configure_scheduler(optimizer, sched_cfg, cfg.steps)
        return optimizer, scheduler

    def _make_batch(self, cfg: TrainingConfig, vocab: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        shape = (cfg.batch_size, cfg.seq_len)
        input_ids = torch.randint(0, vocab, shape, generator=self._input_gen)
        targets = torch.randint(0, vocab, shape, generator=self._input_gen)
        return input_ids.to(self._device), targets.to(self._device)

    def _numeric_checks(
        self,
        checks: ChecksSpec,
        loss_val: float,
        outputs: "torch.Tensor",
        model: "nn.Module",
    ) -> list[str]:
        import math

        problems: list[str] = []
        if checks.fail_on_nan_loss and not math.isfinite(loss_val):
            problems.append("non_finite_loss")
        if checks.fail_on_nonfinite_output and not torch.isfinite(outputs).all():
            problems.append("non_finite_output")
        if checks.fail_on_nan_grad:
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    problems.append("non_finite_grad")
                    break
        return problems


__all__ = [
    "TrainingWorkload",
    "TrainingConfig",
    "ModelSpec",
    "OptimizerSpec",
    "ChecksSpec",
]
