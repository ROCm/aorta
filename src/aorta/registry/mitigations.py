"""Mitigations registry: built-ins + entry-point discovery + collision detection.

`load_mitigations()` returns the merged registry of built-ins and plugin
contributions, keyed by name. Each entry carries its `source_package` so
collision errors can name the conflicting parties.

Plugin authors register one entry-point per mitigation in their `pyproject.toml`
under the `aorta.mitigations` group. The entry-point name IS the mitigation
name; the loaded object is the env-var bundle (`dict[str, str]`). This mirrors
the existing `aorta.workloads` extension-point pattern.
"""

from importlib.metadata import entry_points
from pathlib import Path

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownMitigationError,
)
from aorta.registry.sidecar import check_sidecar_basenames, load_sidecar_mitigations
from aorta.registry.types import Mitigation

_GROUP = "aorta.mitigations"

# Only runtime-level flags belong here — env vars read by a runtime or library
# (ROCm, hipBLASLt, PyTorch, NCCL, OpenMP, the kernel, etc.), transparent to
# the workload. Workload-internal env vars (e.g. AMP_DTYPE,
# SHAMPOO_PRECONDITIONER_DTYPE) only "work" if the workload's training script
# literally reads os.environ for them; those belong with the workload's own
# package, registered via the `aorta.mitigations` entry-point group.
# See src/aorta/registry/README.md for the full criterion.
BUILTIN_MITIGATIONS: dict[str, dict[str, str]] = {
    # --- core ---
    "none":     {},
    "tf32_off": {"DISABLE_TF32": "1"},  # consumed by hipBLASLt itself
    "xnack":    {"HSA_XNACK": "1"},     # consumed by ROCm runtime
    # --- hardware queue & ROCm runtime ---
    "gpu_max_hw_queues_2": {
        "GPU_MAX_HW_QUEUES": "2",
    },
    "roc_aql_queue_size_1024": {
        "ROC_AQL_QUEUE_SIZE": "1024",
    },
    "hsa_no_sdma": {
        "HSA_ENABLE_SDMA": "0",
    },
    "hsa_no_scratch_reclaim": {
        "HSA_NO_SCRATCH_RECLAIM": "1",
    },
    "roc_signal_pool_16k": {
        "ROC_SIGNAL_POOL_SIZE": "16384",
    },
    "gpu_force_blit_copy_128": {
        "GPU_FORCE_BLIT_COPY_SIZE": "128",
    },
    "debug_clr_no_batch_cpu_sync": {
        "DEBUG_CLR_BATCH_CPU_SYNC_SIZE": "0",
    },
    # --- RCCL / NCCL ---
    "nccl_launch_order_implicit": {
        "NCCL_LAUNCH_ORDER_IMPLICIT": "1",
    },
    "rccl_gfx942_cheap_fence_off": {
        "RCCL_GFX942_CHEAP_FENCE_OFF": "1",
    },
    # --- PyTorch CCA ---
    "pytorch_no_cuda_memory_caching": {
        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
    },
    "pytorch_alloc_expandable_segments": {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    # --- HIP / AMD diagnostics ---
    "hip_launch_blocking": {
        "HIP_LAUNCH_BLOCKING": "1",
    },
    "amd_log_level_4": {
        "AMD_LOG_LEVEL": "4",
    },
    # --- SDPA backend pin ---
    "fa_prefer_ck": {
        "TORCH_ROCM_FA_PREFER_CK": "1",
    },
    "fa_prefer_aotriton": {
        "TORCH_ROCM_FA_PREFER_CK": "0",
    },
}


def load_mitigations(
    extra_files: list[Path] | None = None,
) -> dict[str, Mitigation]:
    """Discover and merge all mitigations: built-ins, then entry-point plugins, then sidecars.

    Sidecar files (`extra_files`) are merged in the order given. The same
    collision rule applies across all three sources — there is no winner; a
    duplicate name raises `RegistryCollisionError` naming both sides.

    No caching — re-reads entry-points each call. Cheap for MVP; revisit if
    profiling shows it matters.

    Raises:
        RegistryCollisionError: two contributors registered the same mitigation name.
        RegistryError: a plugin's entry-point payload was not a `dict[str, str]`,
            or a sidecar file failed schema validation.
    """
    registry: dict[str, Mitigation] = {
        name: Mitigation(name=name, env=dict(env), source_package="aorta")
        for name, env in BUILTIN_MITIGATIONS.items()
    }

    for ep in entry_points(group=_GROUP):
        env = ep.load()
        plugin_name = ep.dist.name
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise RegistryError(
                f"plugin '{plugin_name}' mitigation '{ep.name}' must resolve to "
                f"dict[str, str]; got {type(env).__name__}"
                + (f" with non-string entries {dict(env)!r}" if isinstance(env, dict) else "")
            )
        if ep.name in registry:
            existing = registry[ep.name].source_package
            raise RegistryCollisionError(
                f"mitigation '{ep.name}' registered by both '{existing}' "
                f"and '{plugin_name}' — rename one or remove the duplicate"
            )
        registry[ep.name] = Mitigation(
            name=ep.name, env=dict(env), source_package=plugin_name
        )

    check_sidecar_basenames(extra_files)
    sidecar_paths: dict[str, Path] = {}
    for path in extra_files or ():
        for name, mit in load_sidecar_mitigations(path).items():
            if name in registry:
                existing = registry[name].source_package
                existing_path_hint = (
                    f" (path: {sidecar_paths[name]})"
                    if name in sidecar_paths
                    else ""
                )
                raise RegistryCollisionError(
                    f"mitigation '{name}' registered by both "
                    f"'{existing}'{existing_path_hint} and "
                    f"'{mit.source_package}' (path: {path}) "
                    f"— rename one or remove the duplicate"
                )
            registry[name] = mit
            sidecar_paths[name] = path

    return registry


def get_mitigation(
    name: str, extra_files: list[Path] | None = None
) -> dict[str, str]:
    """Return the env-var bundle for a mitigation name. Empty dict for 'none'.

    Returns a defensive copy — mutating the result does not affect the registry.
    """
    registry = load_mitigations(extra_files=extra_files)
    if name not in registry:
        raise UnknownMitigationError(
            f"unknown mitigation '{name}'; available: {sorted(registry)}; "
            f"if you expected a plugin-contributed entry, ensure the plugin is installed"
        )
    return dict(registry[name].env)
