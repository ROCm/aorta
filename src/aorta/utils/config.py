"""Configuration loading helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable


try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML missing
    yaml = None  # type: ignore


class ConfigError(RuntimeError):
    """Configuration-related error."""


def load_config(path: Path) -> Dict[str, Any]:
    """Load a configuration file in YAML or JSON format."""

    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    try:
        with path.open("r", encoding="utf-8") as handle:
            if suffix in {".yaml", ".yml"}:
                if yaml is None:
                    raise ConfigError(
                        "PyYAML is required to read YAML configs. Install with `pip install pyyaml`."
                    )
                return yaml.safe_load(handle) or {}
            if suffix == ".json":
                return json.load(handle)
            raise ConfigError(f"Unsupported config extension: {suffix}")
    except ConfigError:
        raise
    except Exception as exc:  # pragma: no cover
        raise ConfigError(f"Failed to parse config {path}: {exc}") from exc


def merge_cli_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply dotted-key overrides specified on the CLI."""

    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override format (expected key=value): {override}")
        key, raw_value = override.split("=", 1)
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _coerce_value(raw_value)
    return result


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",")]
    return raw


__all__ = ["ConfigError", "load_config", "merge_cli_overrides"]
