"""Adapter exposing the in-tree RCCL race reproducer as an `aorta` workload.

`aorta run` resolves ``workload: race`` to this class. It is a thin adapter
over :mod:`aorta.race` — it filters the recipe's ``workload_config`` to known
:class:`~aorta.race.config.ReproducerConfig` fields, builds the config, and
delegates to :func:`aorta.race.modes.create_reproducer`. The reproducer mode
(``default`` / ``ddp`` / ``fsdp``) is selected by the ``mode`` config key, not
by separate workload registrations.

Unknown ``workload_config`` keys are DROPPED (we never do
``ReproducerConfig(**config)``, which would ``TypeError`` on an unknown key).
Because a silently dropped stress lever is a false green for the war room,
every dropped key is logged at WARNING so a misconfigured recipe is visible
rather than passing as a no-op.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Literal

from aorta.workloads._base import Workload, WorkloadResult

# torch (and the ``aorta.race`` package, whose ``__init__`` eagerly pulls a
# torch-dependent reproducer base) is imported lazily so this module can be
# IMPORTED for workload discovery / registration in a lightweight environment
# that has no torch (e.g. a CLI venv that only drives docker-based runs). The
# class methods reference these names as module globals; they are bound for
# real whenever torch is installed. setup() raises a clear error if torch is
# unavailable when the workload actually runs.
#
# Only ``import torch`` is guarded, and it catches ``Exception`` (not just
# ``ImportError``) because a broken install -- mismatched CUDA/ROCm runtime,
# unloadable shared library -- surfaces as OSError/RuntimeError, and discovery
# must stay clean for all of those. The ``aorta.race`` imports live in ``else``
# so a genuine bug in that package raises loudly during discovery instead of
# being misattributed to "torch not importable".
try:
    import torch
    import torch.distributed as dist
except Exception as exc:
    _IMPORT_ERROR: Exception | None = exc
else:
    from aorta.race.config import ReproducerConfig
    from aorta.race.modes import create_reproducer

    _IMPORT_ERROR = None

log = logging.getLogger(__name__)

_VALID_MODES = {"default", "ddp", "fsdp"}
_VALID_DTYPES = {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}
_VALID_COMPUTE_TYPES = {"gemm", "transformer"}

# Platform-injected config keys that are NOT ReproducerConfig fields but are
# always present (the dispatcher writes `steps` into every workload config;
# `_aorta_*` keys carry environment / probe metadata). These are expected, not
# user typos, so the unknown-key guard must stay silent on them -- otherwise
# every run logs a spurious "ignoring unknown ... key 'steps'" that dilutes the
# real false-green warning. `steps` is consumed by the launcher, not by the
# reproducer (race uses warmup_iterations + verify_iterations).
_RESERVED_KEYS = {"steps"}


class RaceWorkload(Workload):
    """Thin adapter over the `aorta.race` reproducer (modes: default|ddp|fsdp)."""

    name: ClassVar[str] = "race"
    launch_mode: ClassVar[Literal["single_process", "distributed"]] = "distributed"
    min_world_size: ClassVar[int] = 2

    def _race_config_from_dict(self, d: dict[str, Any]) -> ReproducerConfig:
        known = set(ReproducerConfig.__dataclass_fields__)
        for key in d:
            if key in known or key in _RESERVED_KEYS or key.startswith("_aorta_"):
                continue
            log.warning("race: ignoring unknown workload_config key %r", key)
        cfg = ReproducerConfig(**{k: v for k, v in d.items() if k in known})
        if cfg.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {cfg.mode!r}")
        if cfg.dtype not in _VALID_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_VALID_DTYPES)}, got {cfg.dtype!r}")
        if cfg.compute_type not in _VALID_COMPUTE_TYPES:
            # Reject typos (e.g. "transfomer") that would silently fall back to
            # the GEMM path and produce a false green.
            raise ValueError(
                f"compute_type must be one of {sorted(_VALID_COMPUTE_TYPES)}, got {cfg.compute_type!r}"
            )
        if cfg.shared_layer_weights and cfg.compute_type != "transformer":
            log.warning(
                "race: shared_layer_weights=True has no effect with compute_type=%r "
                "(only applies to compute_type='transformer')",
                cfg.compute_type,
            )
        return cfg

    def setup(self) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "race requires PyTorch, which is not importable in this "
                f"environment: {_IMPORT_ERROR}. Install torch, or run this "
                "workload inside the container/venv that provides it."
            ) from _IMPORT_ERROR
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        self._rank = dist.get_rank()
        self._world = dist.get_world_size()
        # set_device before any CUDA work so we don't create an incidental
        # cuda:0 context on every rank (mirrors llm_determinism).
        if torch.cuda.is_available():
            torch.cuda.set_device(
                int(os.environ.get("LOCAL_RANK", self._rank % max(1, torch.cuda.device_count())))
            )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cfg = self._race_config_from_dict(self.config)

    def run(self) -> WorkloadResult:
        rep = create_reproducer(self._cfg, self._rank, self._world)
        res = rep.run()
        return WorkloadResult(
            passed=res.passed,
            failure_count=res.corruption_count,
            first_failure_iteration=res.first_corruption_iter,
            failure_details=res.corruption_details,
            total_iterations=res.total_iterations,
            elapsed_sec=res.elapsed_time_sec,
            metrics={
                "avg_step_time_ms": res.avg_step_time_ms,
                "mode": self._cfg.mode,
                "compute_type": self._cfg.compute_type,
                "layers_verified": res.layers_verified,
                "layer_checksum_mismatches": res.layer_checksum_mismatches,
                "eff_num_heads": res.eff_num_heads,
                "eff_ffn_size": res.eff_ffn_size,
                "eff_seq_len": res.eff_seq_len,
                "eff_batch_size": res.eff_batch_size,
                "rank": self._rank,
                "world_size": self._world,
            },
            main_work_started=True,
            executed_iterations=res.total_iterations,
            configured_iterations=self._cfg.warmup_iterations + self._cfg.verify_iterations,
        )

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.barrier()
        # Process group teardown left to the launcher.
