"""Typed Buck2 invocation context for environment introspection.

Only configuration inputs that Buck2 understands directly are represented:
ordered mode/flag files, ``-c`` config overrides, and ``-m`` modifiers.  The
context deliberately has no shell-command or passthrough-string escape hatch;
callers turn it into an argv list and invoke Buck2 without a shell.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BuckInvocationOption:
    """One ordered Buck configuration input."""

    kind: Literal["mode", "config", "modifier"]
    value: str

    @classmethod
    def parse(cls, raw: str) -> BuckInvocationOption:
        """Parse ``TYPE=VALUE`` from the repeatable CLI option."""
        kind, separator, value = raw.partition("=")
        if not separator or kind not in {"mode", "config", "modifier"}:
            raise ValueError(
                "Buck options must use mode=VALUE, config=KEY=VALUE, " "or modifier=VALUE"
            )
        if not value or "\x00" in value:
            raise ValueError("Buck option values must be non-empty and contain no NUL")
        if kind == "mode":
            value = value[1:] if value.startswith("@") else value
            if not value:
                raise ValueError("Buck mode option must name a file after '@'")
        if kind == "config":
            key, config_separator, _config_value = value.partition("=")
            if not config_separator or not key.strip():
                raise ValueError("Buck config options must use config=KEY=VALUE")
        return cls(kind=kind, value=value)

    def to_buck_args(self) -> list[str]:
        if self.kind == "mode":
            return [f"@{self.value}"]
        if self.kind == "config":
            return ["-c", self.value]
        return ["-m", self.value]


@dataclass(frozen=True)
class BuckInvocationContext:
    """Configuration inputs that must accompany a Buck2 introspection query.

    ``mode_files``, ``config_overrides``, and ``modifiers`` retain caller
    order because Buck2 configuration precedence is order-sensitive.  Config
    override values are needed to reproduce the cquery and to fingerprint the
    context, but must never be copied into env.json; use :attr:`config_keys`
    for metadata.

    ``default_context_confirmed`` is an explicit operator assertion that the
    target should be queried with no additional context arguments.  It is
    mutually exclusive with all explicit context inputs.
    """

    mode_files: tuple[str, ...] = ()
    config_overrides: tuple[str, ...] = ()
    modifiers: tuple[str, ...] = ()
    default_context_confirmed: bool = False
    ordered_options: tuple[BuckInvocationOption, ...] = ()

    def __post_init__(self) -> None:
        # Coerce list-like callers to tuples so frozen=True is meaningful for
        # the ordered collections as well as for attribute rebinding.
        mode_files = tuple(self.mode_files)
        config_overrides = tuple(self.config_overrides)
        modifiers = tuple(self.modifiers)
        ordered_options = tuple(self.ordered_options)

        for label, values in (
            ("mode file", mode_files),
            ("config override", config_overrides),
            ("modifier", modifiers),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"Buck {label} values must be non-empty strings")
            if any("\x00" in value for value in values):
                raise ValueError(f"Buck {label} values must not contain NUL bytes")

        # Accept either the human-facing ``root//mode/debug`` form or Buck's
        # argv spelling ``@root//mode/debug`` and retain one canonical form.
        mode_files = tuple(value[1:] if value.startswith("@") else value for value in mode_files)
        if any(not value for value in mode_files):
            raise ValueError("Buck mode file values must name a file after '@'")

        for override in config_overrides:
            key, separator, _value = override.partition("=")
            if not separator or not key.strip():
                raise ValueError("Buck config overrides must use KEY=VALUE with a non-empty key")

        has_explicit = bool(mode_files or config_overrides or modifiers)
        if ordered_options and has_explicit:
            raise ValueError(
                "ordered Buck options cannot be combined with grouped mode, "
                "config, or modifier inputs"
            )
        if any(not isinstance(option, BuckInvocationOption) for option in ordered_options):
            raise ValueError("ordered Buck options must be BuckInvocationOption values")
        has_explicit = has_explicit or bool(ordered_options)
        if self.default_context_confirmed and has_explicit:
            raise ValueError(
                "default Buck context confirmation is mutually exclusive "
                "with mode files, config overrides, and modifiers"
            )

        object.__setattr__(self, "mode_files", mode_files)
        object.__setattr__(self, "config_overrides", config_overrides)
        object.__setattr__(self, "modifiers", modifiers)
        object.__setattr__(self, "ordered_options", ordered_options)

    @property
    def effective_mode_files(self) -> tuple[str, ...]:
        if self.ordered_options:
            return tuple(option.value for option in self.ordered_options if option.kind == "mode")
        return self.mode_files

    @property
    def effective_config_overrides(self) -> tuple[str, ...]:
        if self.ordered_options:
            return tuple(option.value for option in self.ordered_options if option.kind == "config")
        return self.config_overrides

    @property
    def effective_modifiers(self) -> tuple[str, ...]:
        if self.ordered_options:
            return tuple(
                option.value for option in self.ordered_options if option.kind == "modifier"
            )
        return self.modifiers

    @property
    def has_explicit_inputs(self) -> bool:
        """Whether at least one mode, config override, or modifier was supplied."""

        return bool(
            self.ordered_options or self.mode_files or self.config_overrides or self.modifiers
        )

    @property
    def source(self) -> str:
        """Schema context-source label for a requested Buck target."""

        if self.default_context_confirmed:
            return "default_confirmed"
        if self.has_explicit_inputs:
            return "explicit"
        return "unspecified"

    @property
    def config_keys(self) -> tuple[str, ...]:
        """Ordered config keys, excluding values that may contain secrets."""

        return tuple(override.partition("=")[0] for override in self.effective_config_overrides)

    @property
    def fingerprint(self) -> str:
        """SHA-256 over every full, ordered context value.

        The serialized fingerprint payload includes raw config values so a
        value-only change is detectable, while only the digest and
        :attr:`config_keys` are persisted.  Category names and the default
        confirmation bit make the encoding unambiguous.
        """

        payload: dict[str, object]
        if self.ordered_options:
            payload = {
                "ordered_options": [
                    {"kind": option.kind, "value": option.value} for option in self.ordered_options
                ],
                "default_context_confirmed": self.default_context_confirmed,
            }
        else:
            payload = {
                "mode_files": list(self.mode_files),
                "config_overrides": list(self.config_overrides),
                "modifiers": list(self.modifiers),
                "default_context_confirmed": self.default_context_confirmed,
            }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_buck_args(self) -> list[str]:
        """Build atomic Buck2 argv entries in precedence-preserving order."""

        if self.ordered_options:
            return [
                argument for option in self.ordered_options for argument in option.to_buck_args()
            ]
        args = [f"@{mode_file}" for mode_file in self.mode_files]
        for override in self.config_overrides:
            args.extend(("-c", override))
        for modifier in self.modifiers:
            args.extend(("-m", modifier))
        return args

    def redact_config_overrides(self, text: str) -> str:
        """Redact complete ``KEY=VALUE`` tokens from Buck diagnostics."""

        redacted = text
        # Longest first avoids a shorter override partially masking a longer
        # one that shares the same prefix.
        for override in sorted(
            self.effective_config_overrides,
            key=len,
            reverse=True,
        ):
            key = override.partition("=")[0]
            redacted = redacted.replace(override, f"{key}=<redacted>")
        return redacted


__all__ = ["BuckInvocationContext", "BuckInvocationOption"]
