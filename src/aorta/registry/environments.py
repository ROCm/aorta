"""Environments registry: built-ins + entry-point discovery + collision detection.

Mirrors the mitigations registry. Plugin payloads are validated against
`_ALLOWED_ENV_TOP_LEVEL` — `docker`, `venv`, `buck_target`, `emulator`,
`mirage_profile` (each `str | None`) and `env` (a nested `dict[str, str]` of
baseline environment variables) are accepted. ROCm version is intentionally not
a valid key (see `Environment` docstring).

Plugin authors register one entry-point per environment in their `pyproject.toml`
under the `aorta.environments` group. The entry-point name IS the environment
name; the loaded object is the recipe (`dict` with any of the keys `docker`,
`venv`, `buck_target`, `emulator`, `mirage_profile` mapping to `str | None`,
plus an optional `env` mapping to `dict[str, str]`). Mirrors the
`aorta.workloads` extension-point pattern.
"""

from importlib.metadata import entry_points
from pathlib import Path
from typing import TypedDict

from aorta._env_rules import is_valid_env_name, value_has_nul
from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownEnvironmentError,
)
from aorta.registry.sidecar import check_sidecar_basenames, load_sidecar_environments
from aorta.registry.types import Environment

_GROUP = "aorta.environments"
# `buck_target` joins `docker` / `venv` as a peer baseline-recipe key (#182).
# `emulator` / `mirage_profile` add a GPU-emulated baseline axis (mirage +
# rocjitsu) so workloads can run with no physical GPU. Order in the frozenset
# is irrelevant; spelled this way for grep-ability.
#
# NOTE: `env` (baseline env-var mapping) is a valid top-level key too, but it
# is validated separately (`_validate_env_mapping`) because its value is a
# nested `dict[str, str]`, not the `str | None` shape the recipe keys share.
# `_VALID_ENV_KEYS` therefore lists only the `str | None` recipe keys; the
# allowed-key check unions in `env` explicitly.
_VALID_ENV_KEYS = frozenset({"docker", "venv", "buck_target", "emulator", "mirage_profile"})
_ALLOWED_ENV_TOP_LEVEL = _VALID_ENV_KEYS | {"env"}


class _EnvironmentPayload(TypedDict, total=False):
    docker: str | None
    venv: str | None
    buck_target: str | None
    emulator: str | None
    mirage_profile: str | None
    env: dict[str, str]


def _validate_env_mapping(source_hint: str, name: str, raw: object) -> dict[str, str]:
    """Validate an environment payload's ``env`` mapping into ``dict[str, str]``.

    ``env`` is the baseline env-var overlay for an environment (lowest layer of
    the platform env contract). Keys AND values must be strings -- numbers /
    booleans are rejected rather than silently ``str()``-coerced, mirroring the
    mitigation-sidecar and recipe ``extra_env`` rules so the same value in a
    YAML/JSON number position fails everywhere consistently. Env-var NAME shape
    and NUL-in-VALUE are enforced here too (via the shared ``aorta._env_rules``
    predicates): a NAMED environment bypasses the recipe parser entirely, so
    without this a malformed name or NUL value would pass loading / ``--dry-run``
    and only fail per-cell at run time. Returns a fresh ``dict`` (the
    ``Environment`` constructor deep-copies again defensively).
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegistryError(
            f"{source_hint} environment '{name}': 'env' must be a mapping of "
            f"str -> str, got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise RegistryError(
                f"{source_hint} environment '{name}': 'env' keys and values must "
                f"be strings, got {type(k).__name__} -> {type(v).__name__}"
            )
        if not is_valid_env_name(k):
            raise RegistryError(
                f"{source_hint} environment '{name}': 'env' has invalid "
                f"environment-variable name {k!r}; expected "
                "[A-Za-z_][A-Za-z0-9_]* (POSIX env-var name shape)."
            )
        if value_has_nul(v):
            # Value NOT echoed -- it may be a secret.
            raise RegistryError(
                f"{source_hint} environment '{name}': 'env' value for key {k!r} "
                "contains a NUL byte and cannot be stored in an environment "
                "variable."
            )
        out[k] = v
    return out

# Built-in environments. `local` and `default` are both "current process, no
# overrides" — `default` is reserved as a site-configurable alias. Customer
# docker recipes ship from downstream private packages via the
# `aorta.environments` entry-point group, NOT here.
BUILTIN_ENVIRONMENTS: dict[str, _EnvironmentPayload] = {
    "local":   {},
    "default": {},
    # GPU-emulated baseline: run the workload under the mirage control plane +
    # rocjitsu software emulator (no physical GPU). `mirage_profile` names the
    # mirage profile `mi350x` (MI350X = gfx950, single-GPU CDNA4 node). Scripts
    # under `scripts/emulation/` create this profile on first use if missing.
    # `emulator` is a convenience hint. Consumed by aorta.emulation.mirage_launch.
    "emulated-rocjitsu": {
        "emulator": "rocjitsu",
        "mirage_profile": "mi350x",
    },
    # Same as emulated-rocjitsu but uses rocjitsu-dbt (dynamic binary translation
    # onto a physical GPU). Requires supported host hardware.
    "emulated-rocjitsu-dbt": {
        "emulator": "rocjitsu-dbt",
        "mirage_profile": "dbt-mi350x",
    },
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
        RegistryError: a plugin's payload was not a dict, contained keys outside
            ``_VALID_ENV_KEYS`` (``docker``, ``venv``, ``buck_target``,
            ``emulator``, ``mirage_profile``, ``env``), had non-``str | None``
            launch-hint values, had a malformed ``dict[str, str]`` ``env``
            mapping, or a sidecar file failed schema validation.
    """
    registry: dict[str, Environment] = {
        name: Environment(
            name=name,
            docker=spec.get("docker"),
            venv=spec.get("venv"),
            buck_target=spec.get("buck_target"),
            emulator=spec.get("emulator"),
            mirage_profile=spec.get("mirage_profile"),
            source_package="aorta",
            env=_validate_env_mapping("built-in", name, spec.get("env")),
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
                f"{sorted(_ALLOWED_ENV_TOP_LEVEL)}"
            )
        invalid = set(spec) - _ALLOWED_ENV_TOP_LEVEL
        if invalid:
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' has invalid "
                f"keys {sorted(invalid)}; allowed keys: {sorted(_ALLOWED_ENV_TOP_LEVEL)}"
            )
        # ``env`` is validated separately (nested mapping); exclude it from the
        # ``str | None`` recipe-key value check so a valid ``env`` dict doesn't
        # trip the "non-string values" guard.
        bad_values = {
            k: v
            for k, v in spec.items()
            if k != "env" and v is not None and not isinstance(v, str)
        }
        if bad_values:
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' has non-string values "
                f"{ {k: type(v).__name__ for k, v in bad_values.items()} }; "
                f"each value must be `str | None`"
            )
        env_mapping = _validate_env_mapping(f"plugin '{plugin_name}'", ep.name, spec.get("env"))
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
            emulator=spec.get("emulator"),
            mirage_profile=spec.get("mirage_profile"),
            source_package=plugin_name,
            env=env_mapping,
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
