"""``rocprof`` collector: attach ``rocprofv3`` to an arbitrary command.

The collector is an **argv prefix**, not a library hook: given the command a
trial would have run anyway, it returns
``rocprofv3 <flags> -- <command>``. That is what lets ``aorta probe -- <any
command>`` be profiled without the workload knowing anything about it, and it
mirrors the emulator seam in :mod:`aorta.emulation.mirage_launch`.

Package shape follows :mod:`aorta.instrumentation.layer_numerics`: a small
public surface (:data:`OUTPUT_SUBDIR`, :func:`resolve_binary`,
:func:`build_argv_prefix`, :func:`parse_summary`, :func:`validate_options`),
strict validation of recipe-supplied options, and fail-soft post-run parsing.

Three operational notes that are properties of ``rocprofv3`` itself, not of
this wiring:

* It logs ``W<timestamp> ... simple_timer.cpp:55] [rocprofv3] ...`` lines to
  **stderr** on every run, and the probe classifier's stderr detectors see
  them. The run summary is therefore routed to
  ``<out_dir>/`` :data:`SUMMARY_FILENAME` rather than stderr, but ``--collect
  rocprof`` still perturbs stderr: it is a measurement mode, not a byte-exact
  passthrough.
* ``--summary-output-file`` takes a filename *stem* relative to ``-d``, not a
  path. Handing it an absolute path makes rocprofv3 splice that path into the
  middle of a filename and create stray directories under ``-d``, so the stem
  is passed on its own and the resulting name reconstructed here.
* When the profiled command performs no GPU work it writes no output files at
  all, which :func:`parse_summary` treats as "nothing to report" rather than
  an error.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from ._options import OPTION_KEYS, OUTPUT_FORMATS, SUMMARY_UNITS, TRACE_FLAGS, validate_options
from ._options import as_bool as _as_bool
from ._options import split_tokens as _split_tokens
from ._parse import parse_summary

#: Subdirectory (under the trial's collector directory) rocprofv3 writes into.
OUTPUT_SUBDIR: str = "rocprof"

#: ``rocprofv3 -o`` basename. With ``-o`` set, rocprofv3 writes the artifacts
#: flat in the ``-d`` directory as ``<basename>_kernel_stats.csv`` etc.; with
#: ``-o`` omitted it nests them under ``<hostname>/<pid>_kernel_stats.csv``
#: instead. The parser globs recursively so it reads either layout.
OUTPUT_BASENAME: str = "aorta"

#: Value passed to ``rocprofv3 --summary-output-file``. It is a *stem*, not a
#: path: rocprofv3 resolves it under ``-d``, prefixes it with the ``-o``
#: basename and appends ``.txt``, so an absolute path here would be spliced
#: into the middle of a filename and create stray directories.
SUMMARY_FILE_STEM: str = "rocprof_summary"

#: The file ``rocprofv3 -S`` actually writes, so the summary does not land on
#: the trial's stderr where the probe classifier would read it as workload
#: output. Derived from :data:`SUMMARY_FILE_STEM` the same way rocprofv3
#: derives it.
SUMMARY_FILENAME: str = f"{OUTPUT_BASENAME}_{SUMMARY_FILE_STEM}.txt"

#: Env var naming the rocprofv3 binary, peer of ``$MIRAGE_BIN``.
ENV_ROCPROF_BIN: str = "ROCPROF_BIN"
_DEFAULT_BIN = "rocprofv3"


class RocprofUnavailableError(RuntimeError):
    """Raised when rocprofv3 was requested but cannot be located."""


def resolve_binary() -> str:
    """Resolve the ``rocprofv3`` binary, honouring ``$ROCPROF_BIN``.

    Returns:
        An absolute path to the profiler binary.

    Raises:
        RocprofUnavailableError: neither ``$ROCPROF_BIN`` nor a ``rocprofv3``
            on ``$PATH`` resolves. Requesting a collector that cannot run is a
            clean setup failure, not a silent unprofiled run.
    """
    override = os.environ.get(ENV_ROCPROF_BIN)
    if override:
        override = os.path.expandvars(os.path.expanduser(override))
        if os.path.dirname(override):
            candidate = os.path.abspath(override)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
            raise RocprofUnavailableError(
                f"{ENV_ROCPROF_BIN}={override!r} looks like a path but does "
                "not point to an executable file."
            )
        found = shutil.which(override)
        if found:
            return found
        raise RocprofUnavailableError(f"{ENV_ROCPROF_BIN}={override!r} was not found on $PATH.")
    found = shutil.which(_DEFAULT_BIN)
    if found:
        return found
    raise RocprofUnavailableError(
        "rocprofv3 not found: set $ROCPROF_BIN to the profiler binary, or put "
        "'rocprofv3' on $PATH (it ships with ROCm). Required by "
        "'--collect rocprof'."
    )


def build_argv_prefix(
    out_dir: Path | str,
    options: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the ``rocprofv3 ... --`` prefix for one collected trial.

    Args:
        out_dir: Directory rocprofv3 writes artifacts into (``-d``). Created by
            the caller; rocprofv3 also creates it, but pre-creating keeps the
            trial tree shaped the same whether or not any GPU work happened.
        options: Recipe-supplied options; see :func:`validate_options`.

    Returns:
        An argv prefix ending in ``"--"``, ready to be concatenated with the
        command being profiled.

    Raises:
        ValueError: an option key or value is invalid.
        RocprofUnavailableError: rocprofv3 is not installed.
    """
    effective = validate_options(options)
    out = Path(out_dir)
    argv: list[str] = [
        resolve_binary(),
        "-d",
        str(out),
        "-o",
        OUTPUT_BASENAME,
        "--output-format",
        effective["output_format"],
    ]
    for token in _split_tokens(effective["trace"]):
        argv.append(TRACE_FLAGS[token])
    if _as_bool("stats", effective["stats"]):
        argv.append("--stats")
    # ``-S`` without ``--summary-output-file`` prints to stderr, where the
    # probe classifier's stderr detectors would read it as workload output.
    argv += ["-S", "--summary-output-file", SUMMARY_FILE_STEM]
    if "summary_units" in effective:
        argv += ["-u", effective["summary_units"]]
    if "kernel_include_regex" in effective:
        argv += ["--kernel-include-regex", effective["kernel_include_regex"]]
    # ``--pmc`` is variadic, so it must be the last flag before the ``--``
    # separator or it would swallow whatever followed it.
    if "pmc" in effective:
        argv += ["--pmc", *_split_tokens(effective["pmc"])]
    argv.append("--")
    return argv


def wrap_argv(
    argv: list[str] | tuple[str, ...],
    out_dir: Path | str,
    options: Mapping[str, str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return ``argv`` run under rocprofv3, writing artifacts to ``out_dir``.

    Args:
        argv: The command the trial would otherwise have run.
        out_dir: Directory rocprofv3 writes artifacts into.
        options: Recipe-supplied options; see :func:`validate_options`.
        env: Accepted for the collector registry's uniform wrap signature and
            unused -- rocprofv3 is configured entirely through flags, unlike
            Proton, which has to translate the device-visibility variables.
    """
    del env
    return [*build_argv_prefix(out_dir, options), *argv]


__all__ = [
    "ENV_ROCPROF_BIN",
    "OPTION_KEYS",
    "OUTPUT_BASENAME",
    "OUTPUT_FORMATS",
    "OUTPUT_SUBDIR",
    "SUMMARY_FILENAME",
    "SUMMARY_FILE_STEM",
    "SUMMARY_UNITS",
    "TRACE_FLAGS",
    "RocprofUnavailableError",
    "build_argv_prefix",
    "parse_summary",
    "resolve_binary",
    "validate_options",
    "wrap_argv",
]
