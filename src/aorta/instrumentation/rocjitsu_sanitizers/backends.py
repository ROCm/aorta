"""Static support matrix for RocJITsu sanitizer backends."""

from __future__ import annotations

from typing import Literal

SupportLevel = Literal["full", "partial", "none"]
SupportRequires = Literal["none", "hardware-or-simulator"]

_WAITCHECK_TARGETS = frozenset(
    {
        "gfx942",
        "gfx950",
        "gfx1100",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
        "gfx1250",
    }
)
_CONSAN_FULL = frozenset({"gfx942", "gfx950", "gfx1100", "gfx1201", "gfx1250"})
_CONSAN_PARTIAL: frozenset[str] = frozenset()
# Device ASan needs an xnack+ code object, so only XNACK-capable targets. These
# are the two TheRock publishes ASan-instrumented ROCm for.
_ASAN_DEVICE = frozenset({"gfx942", "gfx950"})


def support(sanitizer: str, target: str) -> dict[str, object]:
    """Return the static support policy for one sanitizer on one target."""

    if sanitizer == "waitcheck":
        if target in _WAITCHECK_TARGETS:
            return {
                "level": "full",
                "runnable": True,
                "requires": "none",
                "note": (
                    "waitcheck supported (native code objects; supported-form "
                    "coverage, not every ISA memory op)"
                ),
            }
        return {
            "level": "none",
            "runnable": False,
            "requires": "none",
            "note": f"waitcheck not supported on {target}",
        }
    if sanitizer == "consan":
        if target in _CONSAN_FULL:
            return {
                "level": "full",
                "runnable": True,
                "requires": "hardware-or-simulator",
                "note": (
                    f"{target} ConSan is full: native instrumentation on real "
                    "hardware (no gfx950 simulator)"
                ),
            }
        if target in _CONSAN_PARTIAL:
            return {
                "level": "partial",
                "runnable": True,
                "requires": "hardware-or-simulator",
                "note": f"{target} ConSan is partial",
            }
        return {
            "level": "none",
            "runnable": False,
            "requires": "hardware-or-simulator",
            "note": f"ConSan not supported on {target}",
        }
    if sanitizer == "asan":
        if target in _ASAN_DEVICE:
            return {
                "level": "full",
                "runnable": True,
                "requires": "hardware-or-simulator",
                "note": (
                    f"{target} device ASan: needs an {target}:xnack+ build, "
                    "HSA_XNACK=1 and the ASan-instrumented ROCm runtime (lib/asan); "
                    "measured on gfx950"
                ),
            }
        return {
            "level": "none",
            "runnable": False,
            "requires": "hardware-or-simulator",
            "note": f"device ASan not supported on {target}: no xnack+ code object target",
        }
    raise ValueError(f"unknown sanitizer {sanitizer!r}")
