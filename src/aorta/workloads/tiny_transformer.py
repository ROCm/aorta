"""``tiny_transformer`` workload: the runs/04 transformer through training.

The model and the training loop live in :class:`aorta.workloads.training.TrainingWorkload`.
This workload only fixes that recipe's shape and chooses how the ranks start:

* ``inprocess`` (default) runs one rank here. World size 1 uses the gloo
  defaults in ``TrainingWorkload``. CI does not need Monarch, Spur, or a GPU.
* ``spur`` and ``monarch`` spawn workers from the controller process. A
  process with ``AORTA_INNER=1``, or with ``RANK`` and ``WORLD_SIZE`` already
  set, is a worker and runs the loop instead of spawning again.

``launcher`` stays inside ``workload_config``. The recipe loader rejects
unknown top-level keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from aorta.workloads._base import Workload, WorkloadResult
from aorta.workloads.distributed_launch import (
    is_launched_rank,
    launch_monarch,
    launch_spur,
    run_training_rank,
    workload_result_from_mapping,
)
from aorta.workloads.training import TrainingConfig

log = logging.getLogger(__name__)

_LAUNCHERS = ("inprocess", "monarch", "spur")
_LAUNCH_KEYS = ("launcher", "nodes", "gpus_per_node", "spur_bin")

# runs/04 shape. Recipes can override individual fields; omitted fields keep
# these values so a bare ``aorta run --workload tiny_transformer`` matches.
_MODEL_DEFAULTS: dict[str, Any] = {
    "kind": "transformer",
    "hidden_size": 256,
    "num_layers": 4,
    "num_heads": 4,
    "ffn_size": 512,
    "vocab_size": 256,
}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


@dataclass
class TinyTransformerConfig:
    """Launcher knobs. Training knobs stay on :class:`TrainingWorkload`."""

    launcher: str = "inprocess"
    nodes: int = 1
    gpus_per_node: int = 1
    spur_bin: str = "spur"
    expected_world_size: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TinyTransformerConfig:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError(f"workload config must be a dict, got {type(raw).__name__}")
        launcher = raw.get("launcher", "inprocess")
        if not isinstance(launcher, str):
            raise TypeError(f"launcher must be a str, got {type(launcher).__name__}")
        if launcher not in _LAUNCHERS:
            raise ValueError(f"launcher must be one of {list(_LAUNCHERS)}, got {launcher!r}")
        nodes = _positive_int(raw.get("nodes", 1), "nodes")
        gpus_per_node = _positive_int(raw.get("gpus_per_node", 1), "gpus_per_node")
        spur_bin = raw.get("spur_bin", "spur")
        if not isinstance(spur_bin, str) or not spur_bin.strip():
            raise ValueError("spur_bin must be a non-empty command or path")
        expected = raw.get("expected_world_size", None)
        if expected is not None:
            expected = _positive_int(expected, "expected_world_size")
        mesh = nodes * gpus_per_node
        if launcher in ("monarch", "spur") and expected is not None and expected != mesh:
            raise ValueError(
                f"expected_world_size ({expected}) must equal "
                f"nodes * gpus_per_node ({mesh})"
            )
        return cls(
            launcher=launcher,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            spur_bin=spur_bin,
            expected_world_size=expected,
        )


def training_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Training config with the runs/04 model filled in for omitted fields.

    Launcher keys are removed so :class:`TrainingConfig` does not warn about
    them. ``expected_world_size`` stays; the training loop enforces it.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"workload config must be a dict, got {type(raw).__name__}")
    payload = dict(raw)
    for key in _LAUNCH_KEYS:
        payload.pop(key, None)
    model = dict(_MODEL_DEFAULTS)
    user_model = payload.get("model")
    if user_model is None:
        payload["model"] = model
    elif isinstance(user_model, dict):
        model.update(user_model)
        payload["model"] = model
    else:
        raise TypeError(f"model must be a mapping, got {type(user_model).__name__}")
    payload.setdefault("parallelism", "ddp")
    payload.setdefault("steps", 20)
    payload.setdefault("dtype", "float32")
    checks = payload.get("checks")
    if checks is None:
        payload["checks"] = {}
    elif not isinstance(checks, dict):
        raise TypeError(f"checks must be a mapping, got {type(checks).__name__}")
    return payload


class TinyTransformerWorkload(Workload):
    """Runs/04 tiny transformer, launched in-process, by Spur, or by Monarch.

    ``launch_mode = "distributed"`` and ``min_world_size = 1`` match
    :class:`TrainingWorkload`: a bare ``aorta run`` is a one-rank smoke.
    ``expected_world_size`` is what rejects a short process group on the
    multi-node recipes.
    """

    name: ClassVar[str] = "tiny_transformer"
    launch_mode = "distributed"
    min_world_size = 1

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._spec: TinyTransformerConfig | None = None
        self._payload: dict[str, Any] | None = None
        self._train_here = False

    def setup(self) -> None:
        self._spec = TinyTransformerConfig.from_dict(self.config)
        self._payload = training_payload(self.config)
        # Reject a bad training config before a controller spawns anyone.
        # Torch import failures surface from ``run_training_rank`` on the
        # rank that actually trains.
        TrainingConfig.from_dict(self._payload)
        self._train_here = self._spec.launcher == "inprocess" or is_launched_rank()
        log.info(
            "tiny_transformer launcher=%s nodes=%d gpus_per_node=%d train_here=%s",
            self._spec.launcher,
            self._spec.nodes,
            self._spec.gpus_per_node,
            self._train_here,
        )

    def run(self) -> WorkloadResult:
        assert self._spec is not None
        assert self._payload is not None
        spec = self._spec
        if self._train_here:
            # Same rank function the Monarch actor calls. Spur workers get
            # here by re-execing ``aorta`` with ``AORTA_INNER=1``.
            return workload_result_from_mapping(
                run_training_rank(self._payload),
                launcher=spec.launcher,
            )
        if spec.launcher == "spur":
            return launch_spur(
                self._payload,
                nodes=spec.nodes,
                gpus_per_node=spec.gpus_per_node,
                spur_bin=spec.spur_bin,
            )
        if spec.launcher == "monarch":
            return launch_monarch(
                self._payload,
                nodes=spec.nodes,
                gpus_per_node=spec.gpus_per_node,
            )
        raise RuntimeError(f"unsupported launcher {spec.launcher!r}")

    def cleanup(self) -> None:
        # ``run_training_rank`` barriers before it returns. The controller
        # does not own a process group.
        return


__all__ = [
    "TinyTransformerConfig",
    "TinyTransformerWorkload",
    "training_payload",
]
