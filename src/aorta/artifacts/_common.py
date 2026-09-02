"""Shared plumbing for the :mod:`aorta.artifacts` readers.

Private to the package. Two things live here: the exception hierarchy, and
:class:`FieldReader`, which is where the "absence is never a zero" rule is
actually enforced -- every typed accessor either returns a well-formed value
or returns ``None`` *and* records the field name, with no defaulting in
between.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar

#: How an artifact's ``schema_version`` compares to the version this build
#: was written against. Only ``"supported"`` means "we have seen this exact
#: contract"; the readers parse the other three anyway (defensively, field by
#: field) rather than refusing the file, because a reader that hard-fails on
#: an unrecognised version is useless against the older artifacts that are
#: already sitting in ticket attachments and bundle tarballs.
SchemaStatus = Literal["supported", "newer", "older", "unknown"]

T = TypeVar("T")


class ArtifactError(Exception):
    """Base for every error raised by :mod:`aorta.artifacts`."""


class ArtifactReadError(ArtifactError):
    """The file could not be read, is not JSON, or is not a JSON object.

    Distinct from a missing field: this means there is no artifact to
    interpret at all, so there is nothing to be tolerant about.
    """


class MissingFieldError(ArtifactError):
    """Raised by ``require()`` when the artifact did not carry a field.

    ``fields`` is the subset of the caller's request that was absent or
    unreadable, in the order the reader recorded it.
    """

    def __init__(self, subject: str, fields: tuple[str, ...]) -> None:
        self.subject = subject
        self.fields = fields
        joined = ", ".join(fields)
        super().__init__(f"{subject}: missing or unreadable field(s): {joined}")


class HasMissingFields:
    """Mixin giving the artifact dataclasses their ``require()`` check.

    Subclasses declare ``missing_fields`` as a real dataclass field; the
    annotation here is for type checkers only (a plain base class's
    annotations are not collected by :func:`dataclasses.dataclass`).
    """

    missing_fields: tuple[str, ...]

    def require(self, *names: str) -> None:
        """Raise :class:`MissingFieldError` if any named field is unavailable.

        With no arguments, requires *every* modelled field. This is the
        opt-in strict mode: reading stays tolerant by default so a caller
        can inspect a partial artifact, but a caller whose conclusion would
        be wrong without a field can refuse to proceed instead of reasoning
        over ``None``.
        """
        if not names:
            absent = self.missing_fields
        else:
            wanted = set(names)
            absent = tuple(n for n in self.missing_fields if n in wanted)
        if absent:
            raise MissingFieldError(type(self).__name__, absent)


def load_json_object(path: Path | str) -> dict[str, Any]:
    """Read *path* and return its top-level JSON object."""
    resolved = Path(path)
    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactReadError(f"{resolved}: cannot read artifact ({exc})") from exc
    except UnicodeDecodeError as exc:
        # A ``ValueError`` subclass, so it is not covered by the ``OSError``
        # above nor by ``ArtifactError``, and a run killed mid-write is exactly
        # how a truncated multi-byte sequence gets here -- the case the
        # tolerant callers in ``aorta.chat`` are written for.
        raise ArtifactReadError(f"{resolved}: not valid UTF-8 ({exc})") from exc
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise ArtifactReadError(f"{resolved}: not valid JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise ArtifactReadError(
            f"{resolved}: expected a JSON object at the top level, got {type(doc).__name__}"
        )
    return doc


class FieldReader:
    """Pull typed values out of a raw JSON mapping, recording what it could not.

    Every accessor follows the same contract: return the value when the key is
    present and well-formed, otherwise return ``None`` and append the key to
    :attr:`missing`. Nothing is coerced across types -- a ``failure_rate`` of
    ``"0.5"`` is recorded as unreadable rather than parsed, because silent
    coercion is how a shape change stops being visible.
    """

    def __init__(self, doc: Mapping[str, Any]) -> None:
        self._doc = doc
        self._missing: list[str] = []

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(self._missing)

    def _read(self, key: str, convert: Callable[[Any], T | None]) -> T | None:
        if key not in self._doc:
            self._missing.append(key)
            return None
        value = convert(self._doc[key])
        if value is None:
            self._missing.append(key)
        return value

    def string(self, key: str) -> str | None:
        return self._read(key, lambda v: v if isinstance(v, str) else None)

    def integer(self, key: str) -> int | None:
        # ``bool`` is an ``int`` subclass; a flag is not a count.
        return self._read(
            key, lambda v: v if isinstance(v, int) and not isinstance(v, bool) else None
        )

    def number(self, key: str) -> float | None:
        def convert(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)

        return self._read(key, convert)

    def boolean(self, key: str) -> bool | None:
        return self._read(key, lambda v: v if isinstance(v, bool) else None)

    def string_tuple(self, key: str) -> tuple[str, ...] | None:
        def convert(value: Any) -> tuple[str, ...] | None:
            # A non-str element makes the whole list unreadable rather than
            # silently shorter: dropping elements would understate e.g. how
            # many mitigations a cell applied.
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                return None
            return tuple(value)

        return self._read(key, convert)

    def count_map(self, key: str) -> dict[str, int] | None:
        def convert(value: Any) -> dict[str, int] | None:
            if not isinstance(value, dict):
                return None
            out: dict[str, int] = {}
            for name, count in value.items():
                if not isinstance(name, str) or isinstance(count, bool):
                    return None
                if not isinstance(count, int):
                    return None
                out[name] = count
            return out

        return self._read(key, convert)

    def mapping(self, key: str) -> dict[str, Any] | None:
        return self._read(key, lambda v: dict(v) if isinstance(v, dict) else None)

    def nullable_string(self, key: str) -> str | None:
        """Read a field whose JSON ``null`` is itself meaningful.

        ``None`` therefore means either "explicitly null" or "absent"; the
        two are told apart by whether the key appears in :attr:`missing`.
        """
        if key not in self._doc:
            self._missing.append(key)
            return None
        value = self._doc[key]
        if value is None or isinstance(value, str):
            return value
        self._missing.append(key)
        return None

    def record_missing(self, key: str) -> None:
        """Mark a field the caller parsed itself as unavailable."""
        self._missing.append(key)


def classify_integer_schema(value: Any, known: int) -> tuple[SchemaStatus, str | None]:
    """Compare an integer ``schema_version`` against the version we know."""
    if not isinstance(value, int) or isinstance(value, bool):
        return "unknown", f"schema_version {value!r} is not an integer; expected {known}"
    if value == known:
        return "supported", None
    if value > known:
        return (
            "newer",
            f"schema_version {value} is newer than the {known} this build was written "
            "against; fields may have been added, renamed, or removed",
        )
    return (
        "older",
        f"schema_version {value} predates the {known} this build was written against; "
        "fields may be absent",
    )


def classify_dotted_schema(value: Any, known_major: int) -> tuple[SchemaStatus, str | None]:
    """Compare a ``"MAJOR.MINOR"`` ``schema_version`` against the major we know.

    Only the major component decides compatibility. The env-probe schema bumps
    its minor for additive changes -- new keys, new blocks, richer attribution
    -- and every field this reader models has been present since 1.0, so
    pinning the exact minor would report a routine additive bump as a
    compatibility problem and train callers to ignore the signal.
    """
    if not isinstance(value, str):
        return "unknown", f"schema_version {value!r} is not a string; expected 'MAJOR.MINOR'"
    head = value.split(".", 1)[0]
    try:
        major = int(head)
    except ValueError:
        return "unknown", f"schema_version {value!r} has no numeric major component"
    if major == known_major:
        return "supported", None
    if major > known_major:
        return (
            "newer",
            f"schema_version {value!r} is a newer major than the {known_major}.x this build "
            "was written against; the shape may have changed incompatibly",
        )
    return (
        "older",
        f"schema_version {value!r} predates the {known_major}.x this build was written "
        "against; fields may be absent",
    )


__all__ = [
    "ArtifactError",
    "ArtifactReadError",
    "FieldReader",
    "HasMissingFields",
    "MissingFieldError",
    "SchemaStatus",
    "classify_dotted_schema",
    "classify_integer_schema",
    "load_json_object",
]
