"""Public API for the mitigations + environments + agents registry.

Import everything you need from here. Sub-modules (`mitigations.py`,
`environments.py`, `agents.py`, `types.py`, `errors.py`) are internal plumbing
and may be reorganized; the names in `__all__` are the supported public surface.
"""

from aorta.registry.agents import get_agent, load_agents
from aorta.registry.environments import get_environment, load_environments
from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownAgentError,
    UnknownEnvironmentError,
    UnknownMitigationError,
)
from aorta.registry.mitigations import get_mitigation, load_mitigations
from aorta.registry.sidecar import (
    load_sidecar_environments,
    load_sidecar_mitigations,
)
from aorta.registry.types import Agent, Environment, Mitigation

__all__ = [
    "load_mitigations",
    "get_mitigation",
    "load_environments",
    "get_environment",
    "load_agents",
    "get_agent",
    "load_sidecar_mitigations",
    "load_sidecar_environments",
    "Mitigation",
    "Environment",
    "Agent",
    "RegistryError",
    "RegistryCollisionError",
    "UnknownMitigationError",
    "UnknownEnvironmentError",
    "UnknownAgentError",
]
