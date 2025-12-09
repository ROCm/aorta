"""CUDA/HIP stream management utilities.

This module provides utilities for:
- Stream creation and management
- Stream synchronization
- GPU warmup operations
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, List, Optional

import torch


def create_streams(
    count: int, device: str = "cuda:0", priorities: Optional[List[int]] = None
) -> List[torch.cuda.Stream]:
    """Create a list of CUDA/HIP streams.

    Args:
        count: Number of streams to create
        device: Target device
        priorities: Optional list of priorities for each stream.
                   Lower values = higher priority. If None, default priority is used.

    Returns:
        List of torch.cuda.Stream objects
    """
    device_obj = torch.device(device)

    if priorities is not None and len(priorities) != count:
        raise ValueError(f"priorities length ({len(priorities)}) must match count ({count})")

    streams = []
    for i in range(count):
        priority = priorities[i] if priorities is not None else 0
        # Note: PyTorch stream priority support is limited
        # On ROCm, this maps to HIP stream priorities
        stream = torch.cuda.Stream(device=device_obj, priority=priority)
        streams.append(stream)

    return streams


def sync_all_streams(streams: List[torch.cuda.Stream]) -> None:
    """Synchronize all streams.

    Args:
        streams: List of streams to synchronize
    """
    for stream in streams:
        stream.synchronize()


def sync_stream(stream: torch.cuda.Stream) -> None:
    """Synchronize a single stream.

    Args:
        stream: Stream to synchronize
    """
    stream.synchronize()


@contextmanager
def cuda_stream_context(stream: torch.cuda.Stream) -> Generator[torch.cuda.Stream, None, None]:
    """Context manager that sets the current CUDA stream.

    Args:
        stream: Stream to use as current

    Yields:
        The stream
    """
    with torch.cuda.stream(stream):
        yield stream


def get_stream_id(stream: torch.cuda.Stream) -> int:
    """Get the underlying stream ID (pointer/handle).

    This can be useful for debugging and correlating with profiler output.

    Args:
        stream: PyTorch CUDA stream

    Returns:
        Integer representation of the stream handle
    """
    return stream.cuda_stream


def warmup_gpu(device: str = "cuda:0", iterations: int = 10) -> None:
    """Warm up the GPU with simple operations.

    This helps ensure consistent timing by:
    - Triggering GPU frequency scaling
    - Warming up the memory subsystem
    - Initializing any lazy GPU state

    Args:
        device: Device to warm up
        iterations: Number of warmup iterations
    """
    device_obj = torch.device(device)

    # Create some tensors and do operations
    for _ in range(iterations):
        a = torch.randn(1024, 1024, device=device_obj)
        b = torch.randn(1024, 1024, device=device_obj)
        c = torch.mm(a, b)
        torch.cuda.synchronize(device_obj)

    # Clean up
    del a, b, c
    torch.cuda.empty_cache()


__all__ = [
    "create_streams",
    "sync_all_streams",
    "sync_stream",
    "cuda_stream_context",
    "get_stream_id",
    "warmup_gpu",
]
