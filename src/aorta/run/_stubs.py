"""Temporary stubs for B3 registry and A1 environment probe.

Remove this file when B3 lands: src/aorta/registry/ with get_mitigation()
and get_environment().

When A1 lands (PR #152): Replace collect_env and EnvSnapshot imports with:
    from aorta.instrumentation.environment import collect_env, EnvSnapshot

Note: These stubs provide minimal implementations that allow B1 to function
while dependencies are being developed. They should be replaced with real
implementations as soon as those dependencies are available.
"""

from dataclasses import dataclass, field
from typing import Any
import os


# =============================================================================
# A1 Stubs (Environment Probe) - Replace when A1 lands in PR #152
# =============================================================================


@dataclass
class EnvSnapshot:
    """Stub for A1's environment snapshot.

    When A1 lands, replace with:
        from aorta.instrumentation.environment import EnvSnapshot
    """

    hostname: str
    python_version: str
    pytorch_version: str | None
    rocm_version: str | None
    env_vars: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "hostname": self.hostname,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "rocm_version": self.rocm_version,
            "env_vars": self.env_vars,
        }


def collect_env() -> EnvSnapshot:
    """Stub for A1's environment collection.

    When A1 lands, replace with:
        from aorta.instrumentation.environment import collect_env
    """
    import platform
    import sys

    # Try to get PyTorch version
    pytorch_version = None
    try:
        import torch

        pytorch_version = torch.__version__
    except ImportError:
        pass

    # Try to get ROCm version from environment
    rocm_version = os.environ.get("ROCM_VERSION") or os.environ.get("ROCM_PATH")

    # Capture relevant environment variables
    relevant_env_vars = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("ROCM", "HIP", "HSA", "CUDA", "NCCL", "TORCH", "DISABLE_TF32"))
        or k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    }

    return EnvSnapshot(
        hostname=platform.node(),
        python_version=sys.version,
        pytorch_version=pytorch_version,
        rocm_version=rocm_version,
        env_vars=relevant_env_vars,
    )


# =============================================================================
# B3 Stubs (Registry) - Replace when B3 lands
# =============================================================================


@dataclass
class Environment:
    """Stub for B3's environment descriptor."""

    name: str
    kind: str = "local"
    docker: str | None = None
    venv: str | None = None
    rocm: str | None = None
    source_package: str = "aorta"


@dataclass
class Mitigation:
    """Stub for B3's mitigation descriptor."""

    name: str
    env_vars: dict[str, str] = field(default_factory=dict)


# Known environments (stub registry)
_ENVIRONMENTS: dict[str, Environment] = {
    "local": Environment(name="local", kind="local"),
}

# Known mitigations (stub registry)
_MITIGATIONS: dict[str, Mitigation] = {
    "none": Mitigation(name="none", env_vars={}),
    "tf32_off": Mitigation(name="tf32_off", env_vars={"DISABLE_TF32": "1"}),
}


def get_environment(name: str) -> Environment:
    """Stub for B3's environment resolver.

    When B3 lands, replace with:
        from aorta.registry import get_environment
    """
    if name in _ENVIRONMENTS:
        return _ENVIRONMENTS[name]
    available = sorted(_ENVIRONMENTS.keys())
    raise ValueError(f"Unknown environment: '{name}'. Available: {available}")


def get_mitigation(name: str) -> Mitigation:
    """Stub for B3's mitigation resolver.

    When B3 lands, replace with:
        from aorta.registry import get_mitigation
    """
    if name in _MITIGATIONS:
        return _MITIGATIONS[name]
    available = sorted(_MITIGATIONS.keys())
    raise ValueError(f"Unknown mitigation: '{name}'. Available: {available}")


__all__ = [
    "EnvSnapshot",
    "collect_env",
    "Environment",
    "Mitigation",
    "get_environment",
    "get_mitigation",
]
