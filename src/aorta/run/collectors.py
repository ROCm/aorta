"""Reserved collector recipe names.

Collectors are profiling/instrumentation tools that can be attached to
workload runs. MVP implementation validates names but does not attach
actual collectors (deferred to P1).

Supported recipes:
    rocprof: AMD ROCm profiler integration
    numerics: Numeric health monitoring (NaN/Inf detection)
    layer_numerics: Per-layer NaN/magnitude logger (aorta.instrumentation.layer_numerics)
    amd_log: AMD internal logging collector
"""

KNOWN_RECIPES: frozenset[str] = frozenset(
    {
        "rocprof",
        "numerics",
        "layer_numerics",
        "amd_log",
    }
)


__all__ = ["KNOWN_RECIPES"]
