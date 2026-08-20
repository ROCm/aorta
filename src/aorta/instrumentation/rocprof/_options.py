"""Option schema for the ``rocprof`` collector.

Every knob a recipe may set under ``collect: {rocprof: {...}}`` is declared
here and validated at recipe-load time, so a typo fails the whole recipe up
front instead of silently producing an unprofiled run.
"""

from __future__ import annotations

from collections.abc import Mapping

#: ``trace`` tokens -> the ``rocprofv3`` flag each one turns on. ``kernel`` is
#: the default because it is the only domain the summary parser aggregates.
TRACE_FLAGS: dict[str, str] = {
    "kernel": "--kernel-trace",
    "hip": "--hip-trace",
    "hip_runtime": "--hip-runtime-trace",
    "memory_copy": "--memory-copy-trace",
    "rccl": "--rccl-trace",
    "marker": "--marker-trace",
    "scratch": "--scratch-memory-trace",
}

#: ``rocprofv3 --output-format`` values. ``csv`` is the only one the summary
#: parser reads; the others are accepted for operators who want a pftrace /
#: rocpd artifact and will analyse it out of band.
OUTPUT_FORMATS: frozenset[str] = frozenset({"csv", "json", "pftrace", "otf2", "rocpd"})

#: ``rocprofv3 -u/--summary-units`` values.
SUMMARY_UNITS: frozenset[str] = frozenset({"sec", "msec", "usec", "nsec"})

#: Recipe-visible option keys, in the order they appear in the docs table.
OPTION_KEYS: tuple[str, ...] = (
    "trace",
    "output_format",
    "stats",
    "pmc",
    "kernel_include_regex",
    "summary_units",
)

_DEFAULTS: dict[str, str] = {
    "trace": "kernel",
    "output_format": "csv",
    "stats": "true",
}

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def split_tokens(value: str) -> list[str]:
    """Split a comma- or whitespace-separated option value into tokens."""
    return [tok for tok in value.replace(",", " ").split() if tok]


def as_bool(key: str, value: str) -> bool:
    """Parse a boolean-shaped option value.

    Raises:
        ValueError: the value is not one of the accepted true/false spellings.
    """
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(
        f"rocprof option {key!r}: expected a boolean "
        f"({'/'.join(sorted(_TRUE | _FALSE))}), got {value!r}"
    )


def validate_options(options: Mapping[str, str] | None) -> dict[str, str]:
    """Validate recipe-supplied ``rocprof`` options and apply defaults.

    Args:
        options: The raw ``collect: {rocprof: {...}}`` mapping, or ``None``
            when the collector was enabled with no options. The recipe loader
            has already guaranteed ``str -> str``; a programmatic caller that
            bypasses it is rejected here.

    Returns:
        The effective options: the caller's values merged over
        :data:`_DEFAULTS`, with every value normalised to lowercase where the
        schema is an enum. Free-form values (``pmc``,
        ``kernel_include_regex``) are passed through verbatim.

    Raises:
        ValueError: an unknown key, or a value outside its declared domain.
            The message names the valid values so it is actionable in a
            recipe-load error.
    """
    raw = dict(options or {})
    unknown = sorted(set(raw) - set(OPTION_KEYS))
    if unknown:
        raise ValueError(f"rocprof: unknown option(s) {unknown}; valid: {list(OPTION_KEYS)}")
    for key, value in raw.items():
        if not isinstance(value, str):
            raise ValueError(
                f"rocprof option {key!r}: must be a string, got " f"{type(value).__name__}"
            )

    effective = dict(_DEFAULTS)
    effective.update(raw)

    tokens = split_tokens(effective["trace"].lower())
    if not tokens:
        raise ValueError(
            f"rocprof option 'trace': must name at least one domain; "
            f"valid: {sorted(TRACE_FLAGS)}"
        )
    bad_trace = sorted(set(tokens) - set(TRACE_FLAGS))
    if bad_trace:
        raise ValueError(
            f"rocprof option 'trace': unknown domain(s) {bad_trace}; "
            f"valid: {sorted(TRACE_FLAGS)}"
        )
    effective["trace"] = ",".join(dict.fromkeys(tokens))

    output_format = effective["output_format"].strip().lower()
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"rocprof option 'output_format': {output_format!r} is not one of "
            f"{sorted(OUTPUT_FORMATS)}"
        )
    effective["output_format"] = output_format

    as_bool("stats", effective["stats"])

    units = effective.get("summary_units")
    if units is not None:
        units = units.strip().lower()
        if units not in SUMMARY_UNITS:
            raise ValueError(
                f"rocprof option 'summary_units': {units!r} is not one of "
                f"{sorted(SUMMARY_UNITS)}"
            )
        effective["summary_units"] = units

    if "kernel_include_regex" in effective and not effective["kernel_include_regex"].strip():
        raise ValueError("rocprof option 'kernel_include_regex': must be non-empty")
    if "pmc" in effective and not split_tokens(effective["pmc"]):
        raise ValueError(
            "rocprof option 'pmc': must name at least one counter " "(comma- or space-separated)"
        )
    return effective


__all__ = [
    "OPTION_KEYS",
    "OUTPUT_FORMATS",
    "SUMMARY_UNITS",
    "TRACE_FLAGS",
    "as_bool",
    "split_tokens",
    "validate_options",
]
