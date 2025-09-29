"""Prototype for measuring SDMA overlap on ROCm."""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch


HIP_MEMCPY_DEVICE_TO_HOST = 2  # hipMemcpyDeviceToHost


class HipRuntimeError(RuntimeError):
    """Raised when HIP runtime operations fail."""


def _load_hip() -> Tuple[ctypes.CDLL, ctypes.CFUNCTYPE]:
    lib_name = os.environ.get("HIP_RUNTIME_LIBRARY", "libamdhip64.so")
    try:
        hip = ctypes.CDLL(lib_name)
    except OSError as exc:  # pragma: no cover - depends on runtime environment
        raise HipRuntimeError(f"Failed to load HIP runtime '{lib_name}': {exc}") from exc

    hipMemcpyAsync = hip.hipMemcpyAsync
    hipMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    hipMemcpyAsync.restype = ctypes.c_int

    hipGetErrorString = hip.hipGetErrorString
    hipGetErrorString.argtypes = [ctypes.c_int]
    hipGetErrorString.restype = ctypes.c_char_p

    return hipMemcpyAsync, hipGetErrorString


HIP_MEMCPY_ASYNC, HIP_ERROR_STRING = _load_hip()


def _hip_check(ret: int, name: str) -> None:
    if ret != 0:
        err = HIP_ERROR_STRING(ret)
        message = err.decode("utf-8") if err else f"error code {ret}"
        raise HipRuntimeError(f"{name} failed: {message}")


@dataclass
class BenchmarkConfig:
    device: int = 0
    matrix_size: int = 4096
    copy_megabytes: int = 64
    iterations: int = 20
    cold_iters: int = 3


def _hip_memcpy_async(dst_ptr: int, src_ptr: int, num_bytes: int, stream_ptr: int) -> None:
    ret = HIP_MEMCPY_ASYNC(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(num_bytes),
        ctypes.c_int(HIP_MEMCPY_DEVICE_TO_HOST),
        ctypes.c_void_p(stream_ptr),
    )
    _hip_check(ret, "hipMemcpyAsync")


def _run_variant(cfg: BenchmarkConfig, overlap: bool) -> float:
    torch.cuda.set_device(cfg.device)
    compute_stream = torch.cuda.Stream(priority=0)
    copy_stream = torch.cuda.Stream(priority=-1)

    size = cfg.matrix_size
    mat_a = torch.randn((size, size), dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn((size, size), dtype=torch.bfloat16, device="cuda")

    copy_elems = cfg.copy_megabytes * 1024 * 1024 // mat_a.element_size()
    copy_elems = max(copy_elems, size * size)
    copy_src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
    copy_dst = torch.empty(copy_elems, dtype=torch.float32, device="cpu", pin_memory=True)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for iteration in range(cfg.cold_iters + cfg.iterations):
        with torch.cuda.stream(compute_stream):
            result = torch.matmul(mat_a, mat_b)
            result = result.relu()
        if overlap:
            if iteration >= cfg.cold_iters:
                _hip_memcpy_async(
                    copy_dst.data_ptr(),
                    copy_src.data_ptr(),
                    copy_elems * copy_src.element_size(),
                    copy_stream.cuda_stream,
                )
        else:
            compute_stream.synchronize()
            if iteration >= cfg.cold_iters:
                _hip_memcpy_async(
                    copy_dst.data_ptr(),
                    copy_src.data_ptr(),
                    copy_elems * copy_src.element_size(),
                    copy_stream.cuda_stream,
                )
                copy_stream.synchronize()

    compute_stream.synchronize()
    copy_stream.synchronize()

    end_event.record()
    torch.cuda.synchronize()
    total_ms = start_event.elapsed_time(end_event)
    return total_ms / max(cfg.iterations, 1)


def run_sdma_benchmark(cfg: BenchmarkConfig) -> dict:
    sequential = _run_variant(cfg, overlap=False)
    overlapped = _run_variant(cfg, overlap=True)
    return {
        "sequential_ms": sequential,
        "overlapped_ms": overlapped,
        "savings_percent": max(0.0, (sequential - overlapped) / sequential * 100.0),
    }


__all__ = ["BenchmarkConfig", "run_sdma_benchmark"]
