"""Kernel observation adapters and deterministic top-N selection."""

from __future__ import annotations

import csv
import hashlib
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


def sha256_code_object(path: Path) -> str | None:
    """Lowercase SHA-256 of a non-empty code object, or ``None`` if unreadable.

    Streams the file so a large code object (hundreds of MB for an unbundled
    Tensile object) is not read into memory at once. Returns ``None`` for a missing / empty /
    unreadable path so callers can degrade gracefully rather than crash: the
    identity stays digest-less and Waitcheck fails closed
    (``code_object_identity_required``) exactly as it does today for a kernel
    source with no committed digest.
    """

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as blob:
            for chunk in iter(lambda: blob.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


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


# Runtime-internal copy kernels emitted by the ROCclr runtime (e.g.
# ``__amd_rocclr_copyBuffer``). They are not user compute and would otherwise
# dominate a dispatch-count ranking, so trace adapters drop them by default.
_RUNTIME_COPY_PREFIX = "__amd_rocclr_"


def _gemm_kernel_name(row: Mapping[str, object]) -> str:
    """Synthesize a stable GEMM kernel name from hipBLASLt shape columns.

    Matches the ``gemm_<transA><transB>_M<M>_N<N>_K<K>`` convention used by
    the sanitizer verification bundle (e.g. ``gemm_NT_M128_N128_K128``).
    """

    def _field(key: str) -> str:
        value = _optional_str(row.get(key))
        if value is None:
            raise ValueError(f"gemm CSV row missing required column {key!r}")
        return value

    trans_a = _field("transA")
    trans_b = _field("transB")
    m = _field("M")
    n = _field("N")
    k = _field("K")
    return f"gemm_{trans_a}{trans_b}_M{m}_N{n}_K{k}"


def observations_from_gemm_csv(
    path: Path,
    *,
    target: str,
    isa_dir: Path | None = None,
    source: str = "gemm_csv",
) -> tuple[KernelObservation, ...]:
    """Normalize a hipBLASLt GEMM-shape CSV into kernel observations.

    Each row is one unique GEMM shape with a ``count`` (dispatch count) and a
    ``top_solution_idx``. When ``isa_dir`` is given, the selected shape's code
    object is resolved to ``isa_dir/sol_<top_solution_idx>.hsaco`` and pinned
    on the :class:`KernelIdentity` so Waitcheck can analyze a real object. The
    CSV carries no per-shape time, so ``total_time_ms`` is left at ``0.0`` and
    callers should rank these observations by ``top_dispatch_count``.

    Input order is never treated as ranking; ``select_kernels`` ranks and
    tie-breaks deterministically.
    """

    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(
            dict(row)
            for row in csv.DictReader(
                line for line in stream if not line.lstrip().startswith("#")
            )
        )

    observations: list[KernelObservation] = []
    for index, row in enumerate(rows):
        if row.get("count") in (None, ""):
            raise ValueError(f"gemm CSV row {index} needs a count column")
        name = _gemm_kernel_name(row)
        solution_idx = _optional_int(
            row.get("top_solution_idx"),
            field=f"gemm CSV row {index} top_solution_idx",
        )
        code_object: str | None = None
        code_object_sha256: str | None = None
        if isa_dir is not None and solution_idx is not None:
            artifact = Path(isa_dir) / f"sol_{solution_idx}.hsaco"
            code_object = str(artifact)
            code_object_sha256 = sha256_code_object(artifact)
        observations.append(
            KernelObservation(
                identity=KernelIdentity(
                    name=name,
                    target=target,
                    code_object=code_object,
                    code_object_sha256=code_object_sha256,
                    code_object_index=0 if code_object is not None else None,
                ),
                total_time_ms=0.0,
                dispatch_count=_non_negative_int(
                    row.get("count", 0),
                    field=f"gemm CSV row {index} count",
                ),
                sources=(source,),
            )
        )
    return tuple(observations)


def observations_from_rocprof_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    target: str,
    drop_runtime_copies: bool = True,
    source: str = "rocprof_trace",
) -> tuple[KernelObservation, ...]:
    """Aggregate rocprofiler ``KERNEL_DISPATCH`` rows by kernel name.

    ``dispatch_count`` is the number of rows for a kernel name and
    ``total_time_ms`` is the summed ``End_Timestamp - Start_Timestamp`` (in
    nanoseconds) divided by 1e6. Non-dispatch rows are ignored and, by
    default, runtime-internal ``__amd_rocclr_*`` copy kernels are dropped.
    These observations carry no code object (trend / selection views only).
    """

    aggregated: dict[str, tuple[int, float]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        kind = _optional_str(row.get("Kind"))
        if kind is not None and kind != "KERNEL_DISPATCH":
            continue
        name = _optional_str(row.get("Kernel_Name"))
        if name is None:
            raise ValueError(f"rocprof row {index} needs a Kernel_Name column")
        if drop_runtime_copies and name.startswith(_RUNTIME_COPY_PREFIX):
            continue
        start = _non_negative_float(
            row.get("Start_Timestamp", 0.0),
            field=f"rocprof row {index} Start_Timestamp",
        )
        end = _non_negative_float(
            row.get("End_Timestamp", 0.0),
            field=f"rocprof row {index} End_Timestamp",
        )
        if end < start:
            raise ValueError(
                f"rocprof row {index}: End_Timestamp {end} precedes Start_Timestamp {start}"
            )
        prev_count, prev_ns = aggregated.get(name, (0, 0.0))
        if name not in aggregated:
            order.append(name)
        aggregated[name] = (prev_count + 1, prev_ns + (end - start))

    return tuple(
        KernelObservation(
            identity=KernelIdentity(name=name, target=target),
            total_time_ms=aggregated[name][1] / 1e6,
            dispatch_count=aggregated[name][0],
            sources=(source,),
        )
        for name in order
    )


def observations_from_rocprof_trace(
    path: Path,
    *,
    target: str,
    drop_runtime_copies: bool = True,
) -> tuple[KernelObservation, ...]:
    """Read a rocprofiler trace CSV and aggregate it by kernel name."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = tuple(dict(row) for row in csv.DictReader(stream))
    return observations_from_rocprof_rows(
        rows,
        target=target,
        drop_runtime_copies=drop_runtime_copies,
    )


def observations_from_kernel_list(
    kernels: Iterable[str | Mapping[str, object]],
    *,
    target: str,
    source: str = "kernel_list",
) -> tuple[KernelObservation, ...]:
    """Build direct kernel identities from an explicit list (no ranking).

    Each item is either a bare kernel name or a mapping carrying identity
    fields (``name``, ``code_object``, ``code_object_sha256``,
    ``code_object_index``, ``entry_offset``). Every kernel is retained with a
    ``dispatch_count`` of 1 so ``select_kernels`` keeps them all when
    ``top_n`` covers the list.
    """

    observations: list[KernelObservation] = []
    for index, item in enumerate(kernels):
        if isinstance(item, str):
            entry: Mapping[str, object] = {"name": item}
        elif isinstance(item, Mapping):
            entry = item
        else:
            raise ValueError(f"kernel_list[{index}] must be a string or mapping")
        name = _optional_str(entry.get("name"))
        if name is None:
            raise ValueError(f"kernel_list[{index}] needs a non-empty name")
        observations.append(
            KernelObservation(
                identity=KernelIdentity(
                    name=name,
                    target=target,
                    code_object=_optional_str(entry.get("code_object")),
                    code_object_sha256=_optional_str(entry.get("code_object_sha256")),
                    code_object_index=_optional_int(
                        entry.get("code_object_index"),
                        field=f"kernel_list[{index}] code_object_index",
                    ),
                    entry_offset=_optional_int(
                        entry.get("entry_offset"),
                        field=f"kernel_list[{index}] entry_offset",
                    ),
                ),
                total_time_ms=0.0,
                dispatch_count=1,
                sources=(source,),
            )
        )
    return tuple(observations)


def observations_from_consan_repro(
    variant: str,
    *,
    target: str,
    label: str | None = None,
) -> tuple[KernelObservation, ...]:
    """Build the single kernel identity for a ConSan repro variant.

    The two built-in variants ("clean"/"racy") keep their historical kernel
    labels for backward compatibility. Any other non-empty variant is accepted
    so a recipe can drive ConSan over a user-supplied command/code object: its
    kernel label is either the explicit ``label`` or a stable ``consan_{variant}``
    derived from the variant. An empty/invalid variant still fails closed.
    """

    normalized = _optional_str(variant)
    if normalized is None:
        raise ValueError(f"unsupported consan repro variant {variant!r}")
    builtin = {"clean": "consan_lds_race", "racy": "consan_lds_race_2wave"}.get(normalized)
    if builtin is not None:
        resolved_label = builtin
    else:
        explicit = _optional_str(label)
        resolved_label = explicit if explicit is not None else f"consan_{normalized}"
    return (
        KernelObservation(
            identity=KernelIdentity(name=resolved_label, target=target),
            total_time_ms=0.0,
            dispatch_count=1,
            sources=(f"consan_repro:{normalized}",),
        ),
    )


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
