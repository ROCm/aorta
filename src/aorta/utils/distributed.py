"""
Distributed initialization and process group utilities.

Provides helpers for ``torch.distributed`` setup so that workloads and the
CLI do not need to duplicate boilerplate.  All functions are safe to call
in non-distributed (single-process) contexts -- they return sensible
defaults (rank 0, world size 1) when ``torch.distributed`` is not
initialised.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialisation / teardown
# ---------------------------------------------------------------------------

def init_distributed(backend: str = "nccl") -> None:
    """Initialise ``torch.distributed`` from environment variables.

    Reads ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, and
    ``MASTER_PORT`` (all set automatically by ``torchrun``).  If the
    process group is already initialised this is a no-op.

    When CUDA is available and the local rank is within the device count,
    ``torch.cuda.set_device`` is called so that each process owns its own
    GPU.

    Args:
        backend: Communication backend (``"nccl"`` or ``"gloo"``).
    """
    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    logger.info(
        "Distributed init: rank=%d  world_size=%d  local_rank=%d  backend=%s",
        dist.get_rank(),
        dist.get_world_size(),
        local_rank,
        backend,
    )


def cleanup_distributed() -> None:
    """Destroy the default process group if it exists."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Return ``True`` when ``torch.distributed`` is initialised."""
    return dist.is_initialized()


# ---------------------------------------------------------------------------
# Rank / world-size helpers (safe in non-distributed mode)
# ---------------------------------------------------------------------------

def get_rank() -> int:
    """Return the global rank, or ``0`` when not distributed."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return the world size, or ``1`` when not distributed."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_local_rank() -> int:
    """Return the local rank from the ``LOCAL_RANK`` env var, or ``0``."""
    return int(os.environ.get("LOCAL_RANK", "0"))


# ---------------------------------------------------------------------------
# Process-group helpers
# ---------------------------------------------------------------------------

def parse_process_groups(spec: str) -> Dict[int, List[int]]:
    """Parse a process-group specification string.

    The format mirrors the one used by *multistream_bench*::

        "[0,1,2,3],[4,5,6,7]"

    Returns a dict mapping a zero-based group id to the list of ranks in
    that group.  For example the string above produces
    ``{0: [0,1,2,3], 1: [4,5,6,7]}``.

    Args:
        spec: Comma-separated bracket groups, e.g. ``"[0,1],[2,3]"``.

    Returns:
        Mapping from group id to list of ranks.

    Raises:
        ValueError: If the spec is empty, contains empty groups, or has
            non-integer tokens.
    """
    cleaned = spec.replace(" ", "")
    if not cleaned:
        raise ValueError("Process-group spec must be a non-empty string.")

    stripped = cleaned.strip()
    if stripped in ("[]", "[", "]"):
        raise ValueError(
            f"Process-group spec {spec!r} does not contain any ranks."
        )

    groups: Dict[int, List[int]] = {}
    inner = stripped.strip("[]")
    raw_groups = inner.split("],[")

    for idx, raw in enumerate(raw_groups):
        if not raw:
            raise ValueError(
                f"Empty process group at index {idx} in spec {spec!r}."
            )
        try:
            ranks = [int(r) for r in raw.split(",") if r]
        except ValueError as exc:
            raise ValueError(
                f"Non-integer rank in process-group spec {spec!r} "
                f"(group {idx}: {raw!r})."
            ) from exc
        if not ranks:
            raise ValueError(
                f"Empty process group at index {idx} in spec {spec!r}."
            )
        negative = [r for r in ranks if r < 0]
        if negative:
            raise ValueError(
                f"Negative rank(s) {negative} in process-group spec {spec!r} "
                f"(group {idx})."
            )
        groups[idx] = ranks
    return groups


def create_process_groups(
    group_ranks: Dict[int, List[int]],
    backend: Optional[str] = None,
) -> Dict[int, dist.ProcessGroup]:
    """Create ``torch.distributed`` process groups from a rank mapping.

    Every rank in the world must call this function collectively (even if
    it does not belong to a particular group) because ``dist.new_group``
    is a collective operation.

    Args:
        group_ranks: Mapping from group id to list of ranks (as returned
            by :func:`parse_process_groups`).
        backend: Optional backend override for the new groups.

    Returns:
        Mapping from group id to the created ``ProcessGroup``.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialised.
        ValueError: If any group contains duplicate or out-of-range ranks.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialised before creating process groups"
        )

    world_size = dist.get_world_size()
    all_ranks = set(range(world_size))
    groups: Dict[int, dist.ProcessGroup] = {}

    for pg_id, ranks in group_ranks.items():
        rank_set = set(ranks)
        if len(rank_set) != len(ranks):
            raise ValueError(
                f"Duplicate ranks in process group {pg_id}: {ranks}"
            )
        out_of_range = [r for r in rank_set if r < 0 or r >= world_size]
        if out_of_range:
            raise ValueError(
                f"Out-of-range rank(s) in process group {pg_id}: "
                f"{sorted(out_of_range)} (world size is {world_size})"
            )

        if rank_set == all_ranks:
            groups[pg_id] = dist.group.WORLD
        else:
            groups[pg_id] = dist.new_group(ranks=ranks, backend=backend)

    return groups
