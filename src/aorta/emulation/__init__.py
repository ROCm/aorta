"""GPU-emulation launch backends for AORTA.

Bridges AORTA's environment axis to the mirage control plane + rocjitsu
software GPU emulator so workloads can run with no physical GPU. See
:mod:`aorta.emulation.mirage_launch` for the launch helpers and
``docs/plans/mirage-aorta-integration.md`` for the design.
"""

from aorta.emulation.mirage_launch import (
    EmulationError,
    MirageLaunchSpec,
    is_emulated_environment,
    resolve_emulation,
    wrap_argv_for_environment,
)

__all__ = [
    "EmulationError",
    "MirageLaunchSpec",
    "is_emulated_environment",
    "resolve_emulation",
    "wrap_argv_for_environment",
]
