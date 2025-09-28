"""Device discovery helpers."""

from __future__ import annotations

import logging
import os
from typing import List, Literal

import torch


Accelerator = Literal["nvidia", "amd", "cpu"]

log = logging.getLogger(__name__)


def detect_accelerator() -> Accelerator:
    """Detect the active accelerator type."""

    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "amd"
    return "nvidia"


def _visible_device_indices() -> List[int]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    if env:
        indices = [token.strip() for token in env.split(",") if token.strip()]
        try:
            return [int(token) for token in indices]
        except ValueError:
            log.warning("Non-integer device identifiers in *_VISIBLE_DEVICES=%s", env)
            return list(range(len(indices)))

    try:
        count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unable to query device count via torch: %s", exc)
        count = 0
    return list(range(count))


def get_device(local_rank: int) -> torch.device:
    """Return the torch.device for the current process."""

    accelerator = detect_accelerator()
    if accelerator == "cpu":
        return torch.device("cpu")

    visible = _visible_device_indices()
    if not visible:
        raise RuntimeError("CUDA/HIP backend reports zero visible devices")

    mapped_index = visible[local_rank % len(visible)]
    if local_rank >= len(visible):
        log.warning(
            "local_rank %s exceeds visible device count %s; remapping to device %s",
            local_rank,
            len(visible),
            mapped_index,
        )

    torch.cuda.set_device(mapped_index)
    return torch.device("cuda", mapped_index)


def get_distributed_backend() -> str:
    """Select the distributed backend based on accelerator."""

    accelerator = detect_accelerator()
    if accelerator == "cpu":
        return "gloo"
    return "nccl"


__all__ = ["detect_accelerator", "get_device", "get_distributed_backend"]
