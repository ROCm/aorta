"""Kernel observation adapters and deterministic top-N selection."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from aorta.report.magpie_adapter import read_magpie_report

from .models import (
    KernelIdentity,
    KernelObservation,
    KernelWorklist,
    SelectionRequirement,
)


def _non_negative_float(value: object, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative, got {parsed}")
    return parsed


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    text = str(value).strip()
    base = 16 if text.lower().startswith("0x") else 10
    try:
        parsed = int(text, base)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative, got {parsed}")
    return parsed


def _optional_int(value: object, *, field: str) -> int | None:
    if value in (None, ""):
        return None
    return _non_negative_int(value, field=field)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected a string, got {type(value).__name__}")
    text = value.strip()
    return text or None


def observations_from_magpie_report(
    report: Mapping[str, object],
    *,
    target: str,
    source: str = "magpie",
) -> tuple[KernelObservation, ...]:
    """Normalize Magpie ``kernel_summary`` entries without ranking them."""

    raw_summary = report.get("kernel_summary", [])
    if not isinstance(raw_summary, list):
        raise TypeError("Magpie kernel_summary must be a list")

    observations: list[KernelObservation] = []
    for index, raw_entry in enumerate(raw_summary):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"kernel_summary[{index}] must be an object")
        entry: Mapping[str, object] = raw_entry
        name = _optional_str(entry.get("name"))
        if name is None:
            raise ValueError(f"kernel_summary[{index}].name must be non-empty")
        observations.append(
            KernelObservation(
                identity=KernelIdentity(
                    name=name,
                    target=target,
                    code_object=_optional_str(entry.get("code_object")),
                    code_object_sha256=_optional_str(entry.get("code_object_sha256")),
                    code_object_index=_optional_int(
                        entry.get("code_object_index"),
                        field=f"kernel_summary[{index}].code_object_index",
                    ),
                    entry_offset=_optional_int(
                        entry.get("entry_offset"),
                        field=f"kernel_summary[{index}].entry_offset",
                    ),
                ),
                total_time_ms=_non_negative_float(
                    entry.get("time_ms", 0.0),
                    field=f"kernel_summary[{index}].time_ms",
                ),
                dispatch_count=_non_negative_int(
                    entry.get("calls", 0),
                    field=f"kernel_summary[{index}].calls",
                ),
                sources=(source,),
            )
        )
    return tuple(observations)


def observations_from_magpie_workspace(
    workspace: Path,
    *,
    target: str,
) -> tuple[KernelObservation, ...]:
    """Read an existing Magpie workspace through AORTA's public adapter."""

    report = read_magpie_report(workspace)
    error = report.get("error")
    if error is not None:
        raise FileNotFoundError(str(error))
    return observations_from_magpie_report(report, target=target)


def observations_from_dispatch_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    target: str,
    source: str = "dispatch_csv",
) -> tuple[KernelObservation, ...]:
    """Normalize generic dispatch rows.

    Required columns are ``name`` (or ``kernel``) and ``count``. Optional
    ``time_ms``, ``code_object``, ``code_object_index``, and ``entry_offset``
    columns carry enough identity for exact Waitcheck.
    """

    observations: list[KernelObservation] = []
    for index, row in enumerate(rows):
        name = _optional_str(row.get("name")) or _optional_str(row.get("kernel"))
        if name is None:
            raise ValueError(f"dispatch row {index} needs a name or kernel column")
        if row.get("count") in (None, ""):
            raise ValueError(f"dispatch row {index} needs a count column")
        observations.append(
            KernelObservation(
                identity=KernelIdentity(
                    name=name,
                    target=target,
                    code_object=_optional_str(row.get("code_object")),
                    code_object_sha256=_optional_str(row.get("code_object_sha256")),
                    code_object_index=_optional_int(
                        row.get("code_object_index"),
                        field=f"dispatch row {index} code_object_index",
                    ),
                    entry_offset=_optional_int(
                        row.get("entry_offset"),
                        field=f"dispatch row {index} entry_offset",
                    ),
                ),
                total_time_ms=_non_negative_float(
                    row.get("time_ms", 0.0),
                    field=f"dispatch row {index} time_ms",
                ),
                dispatch_count=_non_negative_int(
                    row.get("count", 0),
                    field=f"dispatch row {index} count",
                ),
                sources=(source,),
            )
        )
    return tuple(observations)


def observations_from_dispatch_csv(
    path: Path,
    *,
    target: str,
) -> tuple[KernelObservation, ...]:
    """Read a generic dispatch CSV; input order is never treated as ranking."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(dict(row) for row in csv.DictReader(stream))
    return observations_from_dispatch_rows(rows, target=target)


def _deduplicate(
    observations: Iterable[KernelObservation],
) -> tuple[KernelObservation, ...]:
    merged: dict[str, KernelObservation] = {}
    for observation in observations:
        key = observation.identity.stable_key
        previous = merged.get(key)
        if previous is None:
            merged[key] = observation
            continue
        if (
            previous.total_time_ms != observation.total_time_ms
            or previous.dispatch_count != observation.dispatch_count
        ):
            raise ValueError(
                "duplicate kernel identity has conflicting metrics; "
                "select one authoritative source or aggregate explicitly"
            )
        merged[key] = KernelObservation(
            identity=min(
                (previous.identity, observation.identity),
                key=lambda identity: identity.name,
            ),
            total_time_ms=previous.total_time_ms,
            dispatch_count=previous.dispatch_count,
            sources=tuple(sorted(set(previous.sources) | set(observation.sources))),
        )
    return tuple(merged.values())


def select_kernels(
    observations: Iterable[KernelObservation],
    *,
    requirement: SelectionRequirement,
    top_n: int,
) -> KernelWorklist:
    """Deduplicate and rank observations with deterministic tie-breaking."""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")
    unique = _deduplicate(observations)
    if requirement is SelectionRequirement.TOP_TIME:
        ranked = sorted(
            unique,
            key=lambda item: (
                -item.total_time_ms,
                -item.dispatch_count,
                item.identity.stable_key,
            ),
        )
    elif requirement is SelectionRequirement.TOP_DISPATCH_COUNT:
        ranked = sorted(
            unique,
            key=lambda item: (
                -item.dispatch_count,
                -item.total_time_ms,
                item.identity.stable_key,
            ),
        )
    else:  # pragma: no cover - Enum construction normally prevents this.
        raise ValueError(f"unsupported selection requirement {requirement!r}")

    return KernelWorklist(
        requirement=requirement,
        top_n=top_n,
        kernels=tuple(ranked[:top_n]),
    )
