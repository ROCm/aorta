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

import torch
import torch.distributed as dist

from aorta.race.config import ReproducerConfig
from aorta.race.modes import create_reproducer
from aorta.workloads._base import Workload, WorkloadResult

log = logging.getLogger(__name__)

_VALID_MODES = {"default", "ddp", "fsdp"}
_VALID_DTYPES = {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}

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
        return cfg

    def setup(self) -> None:
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
