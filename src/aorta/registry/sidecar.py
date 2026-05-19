"""JSON sidecar loader for ad-hoc mitigations / environments.

A sidecar is a JSON file declaring named mitigations and/or environments with
the same payload shape as entry-point-registered entries. It is the third path
into the registry, between the one-off CLI `--env KEY=VALUE` and a full
pip-installable plugin package — see `README.md` "Path 3" for the rationale.

Both top-level keys (`mitigations`, `environments`) are optional; `version: 1`
is required. Schema errors point at the JSON file path and the offending key
so the operator can fix the file without spelunking through a stack trace.
"""

from __future__ import annotations

import json
from pathlib import Path

from aorta.registry.errors import RegistryError
from aorta.registry.types import Environment, Mitigation

SCHEMA_VERSION = 1
_VALID_TOP_LEVEL = frozenset({"version", "mitigations", "environments"})
# Keep in sync with `aorta.registry.environments._VALID_ENV_KEYS`; the two
# allow-lists are intentionally identical so sidecar payloads and entry-point
# payloads accept the same schema. `buck_target` peers `docker` / `venv` per #182.
_VALID_ENV_KEYS = frozenset({"docker", "venv", "buck_target"})


def _source_tag(path: Path) -> str:
    """Source-package tag for sidecar entries: `sidecar:<basename>`.

    Basename only -- keeps list output terse. The same sidecar passed via
    relative vs. absolute paths therefore produces the same tag, and two
    different files with the same basename would collapse to one tag (and
    look indistinguishable in `list` output / collision errors). To prevent
    that ambiguity, `load_mitigations` / `load_environments` call
    `check_sidecar_basenames` upfront and reject the load if two sidecar
    paths share a basename. Resolver collision messages additionally
    include the full path so the operator knows which file to edit.
    """
    return f"sidecar:{path.name}"


def check_sidecar_basenames(extra_files: list[Path] | None) -> None:
    """Reject sidecar lists where two paths share a basename.

    The source tag (`_source_tag`) is the basename only, so two sidecars
    with the same filename would render identically in `list` output and
    in collision errors -- the operator could not tell which file to fix.
    Catch it upfront with a clear message instead of producing ambiguous
    downstream output. Same-file-passed-twice falls out of the same check
    with the same message, which is also what the operator needs to see.
    """
    seen: dict[str, Path] = {}
    for p in extra_files or ():
        if p.name in seen:
            raise RegistryError(
                f"two sidecars share basename '{p.name}': '{seen[p.name]}' "
                f"and '{p}'; rename one so list output and collision errors "
                f"can distinguish them"
            )
        seen[p.name] = p


def _load_json(path: Path) -> dict:
    """Read+parse the sidecar; wrap IO/JSON errors with the file path."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RegistryError(f"sidecar {path}: cannot read file ({e})") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RegistryError(
            f"sidecar {path}: invalid JSON at line {e.lineno} col {e.colno}: {e.msg}"
        ) from e
    if not isinstance(data, dict):
        raise RegistryError(
            f"sidecar {path}: top-level must be a JSON object, got {type(data).__name__}"
        )
    return data


def _validate_top_level(path: Path, data: dict) -> None:
    if "version" not in data:
        raise RegistryError(f"sidecar {path}: missing required key 'version'")
    if data["version"] != SCHEMA_VERSION:
        raise RegistryError(
            f"sidecar {path}: unsupported version {data['version']!r}; "
            f"this build understands version {SCHEMA_VERSION}"
        )
    unknown = set(data) - _VALID_TOP_LEVEL
    if unknown:
        raise RegistryError(
            f"sidecar {path}: unknown top-level keys {sorted(unknown)}; "
            f"allowed: {sorted(_VALID_TOP_LEVEL)}"
        )


def _validate_mitigation_payload(path: Path, name: str, payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise RegistryError(
            f"sidecar {path}: mitigations.{name}: must be an object of "
            f"env vars, got {type(payload).__name__}"
        )
    for k, v in payload.items():
        if not isinstance(k, str):
            raise RegistryError(
                f"sidecar {path}: mitigations.{name}: env var name must be "
                f"string, got {type(k).__name__} ({k!r})"
            )
        if not isinstance(v, str):
            raise RegistryError(
                f"sidecar {path}: mitigations.{name}.{k}: env var value must "
                f"be string, got {type(v).__name__} ({v!r})"
            )
    return dict(payload)


def _validate_environment_payload(path: Path, name: str, payload: object) -> dict[str, str | None]:
    if not isinstance(payload, dict):
        raise RegistryError(
            f"sidecar {path}: environments.{name}: must be an object, "
            f"got {type(payload).__name__}"
        )
    non_string_keys = [k for k in payload if not isinstance(k, str)]
    if non_string_keys:
        raise RegistryError(
            f"sidecar {path}: environments.{name}: non-string keys "
            f"{[repr(k) for k in non_string_keys]}; allowed: {sorted(_VALID_ENV_KEYS)}"
        )
    invalid = set(payload) - _VALID_ENV_KEYS
    if invalid:
        raise RegistryError(
            f"sidecar {path}: environments.{name}: invalid keys "
            f"{sorted(invalid)}; allowed: {sorted(_VALID_ENV_KEYS)}"
        )
    for k, v in payload.items():
        if v is not None and not isinstance(v, str):
            raise RegistryError(
                f"sidecar {path}: environments.{name}.{k}: value must be "
                f"string or null, got {type(v).__name__} ({v!r})"
            )
    return dict(payload)


def load_sidecar_mitigations(path: Path) -> dict[str, Mitigation]:
    """Parse + validate a sidecar; return its mitigations as `{name: Mitigation}`.

    Returns an empty dict if the file declares only environments. Each
    `Mitigation.source_package` is set to `sidecar:<filename>`.
    """
    data = _load_json(path)
    _validate_top_level(path, data)
    raw = data.get("mitigations")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegistryError(
            f"sidecar {path}: 'mitigations' must be an object, got {type(raw).__name__}"
        )
    src = _source_tag(path)
    out: dict[str, Mitigation] = {}
    for name, payload in raw.items():
        if not isinstance(name, str):
            raise RegistryError(
                f"sidecar {path}: mitigation name must be string, got "
                f"{type(name).__name__} ({name!r})"
            )
        env = _validate_mitigation_payload(path, name, payload)
        out[name] = Mitigation(name=name, env=env, source_package=src)
    return out


def load_sidecar_environments(path: Path) -> dict[str, Environment]:
    """Parse + validate a sidecar; return its environments as `{name: Environment}`.

    Returns an empty dict if the file declares only mitigations.
    """
    data = _load_json(path)
    _validate_top_level(path, data)
    raw = data.get("environments")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegistryError(
            f"sidecar {path}: 'environments' must be an object, got {type(raw).__name__}"
        )
    src = _source_tag(path)
    out: dict[str, Environment] = {}
    for name, payload in raw.items():
        if not isinstance(name, str):
            raise RegistryError(
                f"sidecar {path}: environment name must be string, got "
                f"{type(name).__name__} ({name!r})"
            )
        spec = _validate_environment_payload(path, name, payload)
        out[name] = Environment(
            name=name,
            docker=spec.get("docker"),
            venv=spec.get("venv"),
            buck_target=spec.get("buck_target"),
            source_package=src,
        )
    return out
