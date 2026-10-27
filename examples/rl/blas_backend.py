"""Opt-in: set PyTorch's preferred BLAS library from ``AORTA_PREFERRED_BLAS`` before the first GEMM.

On ROCm ``cublas`` (rocBLAS) also needs ``ROCBLAS_USE_HIPBLASLT=0``; unset, nothing changes.
"""
from __future__ import annotations

import os

ENV = "AORTA_PREFERRED_BLAS"


def apply() -> str | None:
    want = os.environ.get(ENV, "")
    if not want:
        return None
    import torch

    hip = getattr(torch.version, "hip", None)
    if want == "cublas" and hip and os.environ.get("ROCBLAS_USE_HIPBLASLT") != "0":
        raise SystemExit(
            f"{ENV}=cublas without ROCBLAS_USE_HIPBLASLT=0 still routes bf16 "
            "GEMMs to hipBLASLt; set both or neither"
        )
    torch.backends.cuda.preferred_blas_library(want)
    got = str(torch.backends.cuda.preferred_blas_library())
    if got.rsplit(".", 1)[-1].lower() != want.lower():
        raise SystemExit(f"{ENV}={want} requested but torch reports {got}")
    print(
        f"[blas] preferred -> {got} "
        f"ROCBLAS_USE_HIPBLASLT={os.environ.get('ROCBLAS_USE_HIPBLASLT')}",
        flush=True,
    )
    return got
