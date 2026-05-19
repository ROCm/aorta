"""Environments registry: built-ins + entry-point discovery + collision detection.

Mirrors the mitigations registry. Plugin payloads are validated against
`_VALID_ENV_KEYS` — `docker`, `venv`, and `buck_target` are accepted. ROCm
version is intentionally not a valid key (see `Environment` docstring).

Plugin authors register one entry-point per environment in their `pyproject.toml`
under the `aorta.environments` group. The entry-point name IS the environment
name; the loaded object is the recipe (`dict[str, str | None]` with any of the
keys `docker`, `venv`, `buck_target`). Mirrors the `aorta.workloads`
extension-point pattern.
"""

from importlib.metadata import entry_points
from pathlib import Path

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownEnvironmentError,
)
from aorta.registry.sidecar import check_sidecar_basenames, load_sidecar_environments
from aorta.registry.types import Environment

_GROUP = "aorta.environments"
# `buck_target` joins `docker` / `venv` as a peer baseline-recipe key (#182).
# Order in the frozenset is irrelevant; spelled this way for grep-ability.
_VALID_ENV_KEYS = frozenset({"docker", "venv", "buck_target"})

# Built-in environments. `local` and `default` are both "current process, no
# overrides" — `default` is reserved as a site-configurable alias. Customer
# docker recipes (nan-repro, hipblaslt-develop) ship from aorta-internal via
# the `aorta.environments` entry-point group, NOT here.
BUILTIN_ENVIRONMENTS: dict[str, dict[str, str | None]] = {
    "local":   {},
    "default": {},
}


def load_environments(
    extra_files: list[Path] | None = None,
) -> dict[str, Environment]:
    """Discover and merge all environments: built-ins, then entry-point plugins, then sidecars.

    Sidecar files (`extra_files`) are merged in the order given. The same
    collision rule applies across all three sources — a duplicate name raises
    `RegistryCollisionError` naming both sides.

    No caching — re-reads entry-points each call.

    Raises:
        RegistryCollisionError: two contributors registered the same environment name.
        RegistryError: a plugin's payload was not a dict, contained keys other
            than `docker` / `venv`, or had non-`str | None` values; or a sidecar
            file failed schema validation.
    """
    registry: dict[str, Environment] = {
        name: Environment(
            name=name,
            docker=spec.get("docker"),
            venv=spec.get("venv"),
            buck_target=spec.get("buck_target"),
            source_package="aorta",
        )
        for name, spec in BUILTIN_ENVIRONMENTS.items()
    }

    for ep in entry_points(group=_GROUP):
        spec = ep.load()
        plugin_name = ep.dist.name
        if not isinstance(spec, dict):
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' must resolve to "
                f"dict[str, str | None]; got {type(spec).__name__}"
            )
        non_string_keys = [k for k in spec if not isinstance(k, str)]
        if non_string_keys:
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' has non-string "
                f"keys {[repr(k) for k in non_string_keys]}; allowed keys: "
                f"{sorted(_VALID_ENV_KEYS)}"
            )
        invalid = set(spec) - _VALID_ENV_KEYS
        if invalid:
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' has invalid "
                f"keys {sorted(invalid)}; allowed keys: {sorted(_VALID_ENV_KEYS)}"
            )
        bad_values = {k: v for k, v in spec.items() if v is not None and not isinstance(v, str)}
        if bad_values:
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' has non-string values "
                f"{ {k: type(v).__name__ for k, v in bad_values.items()} }; "
                f"each value must be `str | None`"
            )
        if ep.name in registry:
            existing = registry[ep.name].source_package
            raise RegistryCollisionError(
                f"environment '{ep.name}' registered by both '{existing}' "
                f"and '{plugin_name}' — rename one or remove the duplicate"
            )
        registry[ep.name] = Environment(
            name=ep.name,
            docker=spec.get("docker"),
            venv=spec.get("venv"),
            buck_target=spec.get("buck_target"),
            source_package=plugin_name,
        )

    check_sidecar_basenames(extra_files)
    sidecar_paths: dict[str, Path] = {}
    for path in extra_files or ():
        for name, env in load_sidecar_environments(path).items():
            if name in registry:
                existing = registry[name].source_package
                existing_path_hint = (
                    f" (path: {sidecar_paths[name]})"
                    if name in sidecar_paths
                    else ""
                )
                raise RegistryCollisionError(
                    f"environment '{name}' registered by both "
                    f"'{existing}'{existing_path_hint} and "
                    f"'{env.source_package}' (path: {path}) "
                    f"— rename one or remove the duplicate"
                )
            registry[name] = env
            sidecar_paths[name] = path

    return registry


def get_environment(
    name: str, extra_files: list[Path] | None = None
) -> Environment:
    """Return the Environment dataclass for a given name.

    Unlike `get_mitigation` (which returns a dict), environments are richer than
    a flat env-var bundle, so the dataclass IS the public surface.
    """
    registry = load_environments(extra_files=extra_files)
    if name not in registry:
        raise UnknownEnvironmentError(
            f"unknown environment '{name}'; available: {sorted(registry)}; "
            f"if you expected a plugin-contributed entry, ensure the plugin is installed"
        )
    return registry[name]
