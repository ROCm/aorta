"""Per-block deterministic-replay probe for a repeated transformer block.

War-room execution model (from the diagram): a training step is a chain
of repeating ``comm_kernel_i -> compute_kernel_i`` stages. We model one
such stage as one repetition of a single transformer block (see
:mod:`aorta.models.repeated_block`). Under FSDP2 ``fully_shard`` each
block boundary is also a collective boundary — the activation entering /
leaving each block is the tensor crossing GPU boundaries.

What this workload does
-----------------------
1. Build the repeated-block model and wrap each block with FSDP2.
2. Register forward pre/post hooks on every block to compute a bit-exact
   checksum (see :mod:`aorta.instrumentation.checksum`) of the boundary
   activation.
3. One-step replay: snapshot params + RNG, run fwd+bwd, capture per-block
   checksums + loss + per-param grad/param checksums. Restore. Run the
   same fwd+bwd. Compare.
4. RANK-LOCAL pass/fail (rule #4 of the spec): each rank compares its
   own run-1 vs run-2 lists; the trial fails if any rank diverges.
5. Capture mode: when ``capture_dir`` is set, every step on every rank
   writes a ``rank<N>.jsonl`` line with all per-block checksums for
   offline inspection on the 40-GPU capture run.

What this workload does NOT do
------------------------------
* No ``torch.compile``, no CUDA/HIP graphs.
* No optimizer step between the two replays.
* No cross-rank checksum equality assertion by default (rule #4). The
  optional ``checksum_mode: "global"`` adds an all-reduced fingerprint
  for the replay compare only; capture mode keeps per-rank data raw.
* No kernel-level interception. Module hooks are a practical proxy for
  the comm boundary — documented limitation, per the spec's "if direct
  kernel interception is not practical" clause.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from aorta.workloads._base import Workload, WorkloadResult

# torch (and the torch-dependent aorta helpers) are imported lazily so this
# module can be IMPORTED for workload discovery / registration in a lightweight
# environment that has no torch (e.g. a CLI venv that only drives docker-based
# runs). The class methods and module helpers reference these as module
# globals; they are bound for real whenever torch is installed. setup() raises
# a clear error if torch is unavailable when the workload actually runs.
#
# Only the torch imports are guarded, and they catch ``Exception`` (not just
# ``ImportError``) because a broken install -- mismatched CUDA/ROCm runtime,
# unloadable shared library -- surfaces as OSError/RuntimeError, and discovery
# must stay clean for all of those. The torch-dependent ``aorta`` helpers live
# in ``else`` so a genuine bug in those modules raises loudly during discovery
# instead of being misattributed to "torch not importable".
try:
    import torch
    import torch.distributed as dist
    from torch import nn  # noqa: F401  (referenced only in string annotations)
except Exception as exc:
    _DTYPES: dict[str, torch.dtype] = {}
    _IMPORT_ERROR: Exception | None = exc
else:
    from aorta.instrumentation.checksum import (
        ChecksumSet,
        compare,
        global_checksum,
        tensor_checksum,
    )
    from aorta.instrumentation.determinism import enable_deterministic
    from aorta.models import BlockConfig, RepeatedBlockModel

    _DTYPES = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    _IMPORT_ERROR = None

log = logging.getLogger(__name__)

# dtype names valid in a recipe, kept independent of torch so config validation
# works even when torch is not importable.
_VALID_DTYPE_NAMES = ("bf16", "fp16", "fp32")


@dataclass
class LlmDeterminismConfig:
    """Knobs exposed to the recipe / CLI.

    Defaults are sensible for local-rank smoke; the 40-GPU capture run
    typically only overrides ``capture_dir``, ``num_layers``, and the
    launcher's world size.
    """

    # Model shape — see BlockConfig. Size is a knob for how much data
    # crosses GPU boundaries, not a parameter-count target.
    num_layers: int = 24
    hidden_size: int = 2048
    ffn_size: int = 5632
    num_heads: int = 16
    vocab_size: int = 32_000
    seq_len: int = 512
    batch_size: int = 1
    num_experts: int = 1
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    # Replay / capture knobs.
    seed: int = 1234
    steps: int = 1
    checksum_mode: Literal["per_rank", "global"] = "per_rank"
    capture_dir: str | None = None  # when set, dump per-step per-block JSONL.
    model_label: str = "generic-repeated-block"  # advisory metric only.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LlmDeterminismConfig:
        known = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        if cfg.dtype not in _VALID_DTYPE_NAMES:
            raise ValueError(f"dtype must be one of {list(_VALID_DTYPE_NAMES)}, got {cfg.dtype!r}")
        if cfg.checksum_mode not in ("per_rank", "global"):
            raise ValueError(f"checksum_mode must be per_rank|global, got {cfg.checksum_mode!r}")
        if cfg.steps < 1:
            raise ValueError(f"steps must be >= 1, got {cfg.steps}")
        return cfg


@dataclass
class _BlockChecksums:
    """One forward pass worth of per-block boundary checksums."""

    pre: list[int] = field(default_factory=list)   # activation entering each block.
    post: list[int] = field(default_factory=list)  # activation leaving each block.


class _BlockHookManager:
    """Attaches forward pre/post hooks to a list of blocks and collects checksums.

    The hooks intentionally checksum only the first positional input/output
    tensor. The repeated block's contract is ``(b, t, h) -> (b, t, h)``;
    that's also the FSDP2 comm-boundary tensor.
    """

    def __init__(self, blocks: list[nn.Module]) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current: _BlockChecksums | None = None
        for idx, block in enumerate(blocks):
            self._handles.append(block.register_forward_pre_hook(self._make_pre(idx)))
            self._handles.append(block.register_forward_hook(self._make_post(idx)))
        self._num_blocks = len(blocks)

    def _make_pre(self, idx: int):
        def _hook(_module: nn.Module, args: tuple) -> None:
            if self._current is None or not args or not isinstance(args[0], torch.Tensor):
                return
            # Resize to num_blocks on first record so out-of-order calls would surface.
            while len(self._current.pre) <= idx:
                self._current.pre.append(0)
            self._current.pre[idx] = tensor_checksum(_local(args[0].detach()))
        return _hook

    def _make_post(self, idx: int):
        def _hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> None:
            if self._current is None or not isinstance(output, torch.Tensor):
                return
            while len(self._current.post) <= idx:
                self._current.post.append(0)
            self._current.post[idx] = tensor_checksum(_local(output.detach()))
        return _hook

    def start_capture(self) -> _BlockChecksums:
        self._current = _BlockChecksums()
        return self._current

    def stop_capture(self) -> None:
        self._current = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class LlmDeterminismWorkload(Workload):
    """Per-block deterministic replay on a single repeated transformer block."""

    name: ClassVar[str] = "llm_determinism"
    launch_mode: ClassVar[Literal["single_process", "distributed"]] = "distributed"
    min_world_size: ClassVar[int] = 1  # also runs single-rank for local dev.

    def setup(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "llm_determinism requires PyTorch, which is not importable in "
                f"this environment: {_IMPORT_ERROR}. Install torch, or run this "
                "workload inside the container/venv that provides it."
            ) from _IMPORT_ERROR
        self._cfg = LlmDeterminismConfig.from_dict(self.config)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        self._rank = dist.get_rank()
        self._world = dist.get_world_size()
        # set_device BEFORE any CUDA work (incl. determinism setup) so we
        # don't create an incidental cuda:0 context on every rank.
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", self._rank % max(1, torch.cuda.device_count()))))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = _DTYPES[self._cfg.dtype]
        # CPU + Python RNGs seeded here; per-rank CUDA seed goes through
        # manual_seed (current device only) to avoid touching non-local GPUs.
        enable_deterministic(self._cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._cfg.seed)

        model_cfg = BlockConfig(
            vocab_size=self._cfg.vocab_size,
            hidden_size=self._cfg.hidden_size,
            ffn_size=self._cfg.ffn_size,
            num_heads=self._cfg.num_heads,
            num_layers=self._cfg.num_layers,
            seq_len=self._cfg.seq_len,
            num_experts=self._cfg.num_experts,
        )
        model = RepeatedBlockModel(model_cfg).to(self._device).to(self._dtype)
        self._model = _maybe_fsdp_shard(model)
        self._hooks = _BlockHookManager(list(self._raw_blocks()))

        gen = torch.Generator(device="cpu").manual_seed(self._cfg.seed + self._rank)
        ids = torch.randint(0, self._cfg.vocab_size, (self._cfg.batch_size, self._cfg.seq_len), generator=gen)
        self._input_ids = ids.to(self._device)
        # Next-token labels (shifted by one, last position wraps). Identity
        # labels (`= self._input_ids.clone()`) collapse to loss=0 with the
        # tied-embedding LM head: residual path lets the input token dominate
        # its own logit, so the model trivially "predicts" itself. Shifting
        # breaks that and gives a non-trivial loss signal to checksum.
        self._labels = torch.roll(self._input_ids, shifts=-1, dims=-1)

        self._capture_path: Path | None = None
        if self._cfg.capture_dir:
            cdir = Path(self._cfg.capture_dir)
            cdir.mkdir(parents=True, exist_ok=True)
            self._capture_path = cdir / f"rank{self._rank:03d}.jsonl"
            # Truncate so a replay from the same dir doesn't append to stale data.
            self._capture_path.write_text("")

    def run(self) -> WorkloadResult:
        t0 = time.perf_counter()
        all_reasons: list[str] = []
        per_step_metrics: list[dict[str, Any]] = []
        first_failure_step: int | None = None

        for step in range(self._cfg.steps):
            snapshot = _snapshot(self._model)
            r1, blocks1 = self._run_once()
            _restore(self._model, snapshot)
            r2, blocks2 = self._run_once()

            reasons = compare(r1, r2)
            reasons += _compare_block_lists(blocks1, blocks2)
            if reasons and first_failure_step is None:
                first_failure_step = step
            all_reasons.extend(f"step{step}: {r}" for r in reasons)

            if self._capture_path is not None:
                self._dump_capture(step, r1, blocks1, run="r1")
                self._dump_capture(step, r2, blocks2, run="r2")

            per_step_metrics.append({
                "step": step,
                "loss_bits_r1": r1.loss_bits,
                "loss_bits_r2": r2.loss_bits,
                "num_blocks_checked": len(blocks1.post),
            })

        local_fail = 1 if all_reasons else 0
        global_fail = local_fail
        if dist.is_initialized() and self._world > 1:
            t = torch.tensor([local_fail], dtype=torch.int64, device=self._device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_fail = int(t.item())

        elapsed = time.perf_counter() - t0
        passed = global_fail == 0
        metrics: dict[str, Any] = {
            "rank": self._rank,
            "world_size": self._world,
            "num_layers": self._cfg.num_layers,
            "num_experts": self._cfg.num_experts,
            "dtype": self._cfg.dtype,
            "checksum_mode": self._cfg.checksum_mode,
            "capture_dir": self._cfg.capture_dir,
            "model_label": self._cfg.model_label,
            "ranks_with_divergence": global_fail,
            "steps": per_step_metrics,
        }
        if all_reasons:
            metrics["divergence_reasons"] = all_reasons[:32]
            log.error("[rank %d] llm_determinism divergence:\n  %s", self._rank, "\n  ".join(all_reasons[:32]))

        # One iteration = one replay PAIR (r1+r2), matching the recipe's
        # `steps:` semantics. The earlier `steps * 2` accounting made the
        # matrix.md "Iters" column disagree with the recipe and halved the
        # elapsed-per-iter fallback timing.
        return WorkloadResult(
            passed=passed,
            failure_count=len(all_reasons),
            first_failure_iteration=first_failure_step,
            failure_details=[{"rank": self._rank, "reason": r} for r in all_reasons],
            total_iterations=self._cfg.steps,
            elapsed_sec=elapsed,
            metrics=metrics,
            main_work_started=True,
            executed_iterations=self._cfg.steps,
            configured_iterations=self._cfg.steps,
        )

    def cleanup(self) -> None:
        # setup() assigns _hooks partway through; if it raised before reaching
        # that line (e.g. port conflict in dist.init_process_group), _hooks is
        # never set. cleanup() runs unconditionally in the dispatcher's finally
        # block, so guard against a partially-initialized instance.
        hooks = getattr(self, "_hooks", None)
        if hooks is not None:
            hooks.remove()
        if dist.is_initialized():
            dist.barrier()
        # Process group teardown left to the launcher.

    def _run_once(self) -> tuple[ChecksumSet, _BlockChecksums]:
        # Re-seed CPU + CUDA-current-device + Python `random` so both
        # replays enter forward with identical RNG. `manual_seed` (not
        # `_all`) keeps the reseed scoped to LOCAL_RANK's device.
        random.seed(self._cfg.seed + 1)
        torch.manual_seed(self._cfg.seed + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._cfg.seed + 1)

        self._model.zero_grad(set_to_none=True)
        block_cs = self._hooks.start_capture()
        logits = self._model(self._input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            self._labels.view(-1),
        )
        loss.backward()
        self._hooks.stop_capture()

        loss_bits = tensor_checksum(loss.detach())
        output_bits = tensor_checksum(logits.detach())
        # Under FSDP2 `fully_shard`, params/grads are DTensors; `tensor_checksum`
        # needs a plain tensor (bit-reinterpret + local sum, not a distributed
        # reduction). `_local` returns the rank-local shard so the per-rank
        # compare actually compares this rank's bytes.
        grads = {n: tensor_checksum(_local(p.grad)) for n, p in self._model.named_parameters() if p.grad is not None}
        params = {n: tensor_checksum(_local(p.detach())) for n, p in self._model.named_parameters()}
        gbits = global_checksum(loss_bits ^ output_bits) if self._cfg.checksum_mode == "global" else None
        return (
            ChecksumSet(loss_bits=loss_bits, output_bits=output_bits, grads=grads, params=params, global_bits=gbits),
            block_cs,
        )

    def _raw_blocks(self):
        # Underlying ``RepeatedBlockModel.blocks`` is intact even after
        # FSDP2 ``fully_shard`` (it's a composable wrapper, not a module-swap).
        return self._model.blocks

    def _dump_capture(self, step: int, cs: ChecksumSet, blocks: _BlockChecksums, *, run: str) -> None:
        assert self._capture_path is not None  # narrowing for mypy/readers.
        record = {
            "rank": self._rank,
            "step": step,
            "run": run,
            "loss_bits": cs.loss_bits,
            "output_bits": cs.output_bits,
            "block_pre": blocks.pre,
            "block_post": blocks.post,
        }
        with self._capture_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _local(t: torch.Tensor) -> torch.Tensor:
    """Return the rank-local shard for DTensor params; passthrough otherwise.

    FSDP2 `fully_shard` turns parameters into DTensors. Checksumming a DTensor
    directly would trigger a distributed reduction (and `.view(int_dtype)`
    isn't supported on DTensor). We want each rank to checksum *its* shard
    so the per-rank replay compare actually catches a kernel that produced
    different bits on this rank between the two runs.
    """
    to_local = getattr(t, "to_local", None)
    return to_local() if callable(to_local) else t


def _compare_block_lists(a: _BlockChecksums, b: _BlockChecksums) -> list[str]:
    """Per-block-index divergence reasons. Empty == every block matched."""
    reasons: list[str] = []
    if len(a.pre) != len(b.pre) or len(a.post) != len(b.post):
        return [f"block-list shape mismatch: pre {len(a.pre)} vs {len(b.pre)}, post {len(a.post)} vs {len(b.post)}"]
    for i, (x, y) in enumerate(zip(a.pre, b.pre, strict=True)):
        if x != y:
            reasons.append(f"block[{i}].pre {x} != {y}")
    for i, (x, y) in enumerate(zip(a.post, b.post, strict=True)):
        if x != y:
            reasons.append(f"block[{i}].post {x} != {y}")
    return reasons


def _maybe_fsdp_shard(model: nn.Module) -> nn.Module:
    """Wrap each repeated block + the top-level model with FSDP2 ``fully_shard``.

    Falls back to the unwrapped model when FSDP2 isn't importable or when
    CUDA isn't available — keeps the CPU dev path runnable.
    """
    try:
        from torch.distributed._composable.fsdp import fully_shard
    except ImportError:
        log.warning("FSDP2 fully_shard not available; running unsharded")
        return model
    if not torch.cuda.is_available():
        return model
    for block in model.blocks:
        fully_shard(block)
    fully_shard(model)
    return model


def _snapshot(model: nn.Module) -> dict[str, Any]:
    # `get_rng_state()` defaults to the current CUDA device — does NOT
    # touch every visible GPU like `get_rng_state_all()` would. Each
    # torchrun rank owns one LOCAL_RANK device; that's all we need.
    return {
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "py_rng": random.getstate(),
    }


def _restore(model: nn.Module, snap: dict[str, Any]) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(snap["params"][n])
    torch.set_rng_state(snap["cpu_rng"])
    if snap["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(snap["cuda_rng"])
    random.setstate(snap["py_rng"])


__all__ = ["LlmDeterminismConfig", "LlmDeterminismWorkload"]
