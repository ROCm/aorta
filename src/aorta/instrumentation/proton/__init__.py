"""``proton`` collector: attach Triton's Proton profiler to a command.

Two attach modes, both driven from the same recipe options:

* **CLI wrap** (``mode: cli``, the default) rewrites a Python launch
  ``<python> <script.py> <args>`` into
  ``<python> -m triton.profiler.proton <opts> <script.py> <args>``. It is
  deliberately ``-m triton.profiler.proton`` and never the ``proton`` console
  script: that shim is shebanged to whichever interpreter installed Triton,
  which on a typical aorta host is *not* the interpreter the workload runs
  under. Reusing the workload's own interpreter is the only way the wrap works
  for an opaque ``aorta probe -- ...`` command.
* **Env hook** (``mode: env``) leaves the command's own argv alone and hands it
  ``AORTA_PROTON_*`` variables, for a workload that calls ``proton.start()`` /
  ``proton.finalize()`` itself to get scoped or intra-kernel measurement.

Because Proton's CLI front-end runs a *script* (it is not a generic command
runner), the CLI wrap only applies to Python launches: a script path, a bare
``pytest ...``, or ``<python> -m pytest ...``, which is normalised onto the
bare spelling since Proton runs both as ``pytest.main(args)``. Anything else
raises :class:`ProtonWrapError` with the ``mode: env`` escape hatch named,
rather than silently running unprofiled.

On AMD, Proton reads ``ROCR_VISIBLE_DEVICES`` and **rejects**
``HIP_VISIBLE_DEVICES`` outright for the queue-intercepting backends (verified:
``ValueError: Proton does not work when the environment variable
HIP_VISIBLE_DEVICES is set on AMD GPUs``). When a trial's env carries the
latter it is translated (and the original unset) in the wrap, with a warning,
so a device-pinned cell still profiles the device it asked for.

The default backend is :data:`AUTO_BACKEND`, which omits Proton's ``-b``
entirely. Proton then picks the backend matching the active runtime --
``rocprofiler`` where rocprofiler-sdk is available, ``roctracer`` otherwise.
Naming a backend explicitly is a version commitment: ``rocprofiler`` is the
preferred AMD backend upstream but was added after Triton 3.7, whose CLI
rejects the name at argparse before the payload ever runs.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from ._options import (
    AUTO_BACKEND,
    BACKENDS,
    CONTEXTS,
    DATA_FORMATS,
    GRANULARITIES,
    INSTRUMENTATION_MODES,
    MODES,
    OPTION_KEYS,
    QUEUE_INTERCEPTING_BACKENDS,
    mode_argument,
    validate_options,
)
from ._parse import parse_profile, parse_summary

log = logging.getLogger(__name__)

#: Subdirectory (under the trial's collector directory) Proton writes into.
OUTPUT_SUBDIR: str = "proton"

#: Proton session name (``-n``) stem inside the output directory; Proton
#: appends the format suffix, so ``data: tree`` yields ``proton.hatchet``.
PROFILE_BASENAME: str = "proton"

#: Module Proton's command front-end lives in.
PROTON_MODULE: str = "triton.profiler.proton"

#: Env var overriding the interpreter used for the CLI wrap. Needed when the
#: probed command is not itself a ``python ...`` invocation, or when Triton
#: lives in a different interpreter than the one launching the workload.
ENV_PROTON_PYTHON: str = "AORTA_PROTON_PYTHON"

#: Prefix of the variables ``mode: env`` exports for a workload that drives
#: Proton itself.
ENV_PREFIX: str = "AORTA_PROTON_"

_PYTHON_RE = re.compile(r"^python(\d+(\.\d+)?)?$")
# Interpreter flags that take no argument and can safely stay in front of
# ``-m triton.profiler.proton``. Anything else means we cannot confidently
# locate the script, so the wrap refuses instead of guessing.
_SAFE_PYTHON_FLAGS = frozenset({"-u", "-B", "-E", "-s", "-S", "-O", "-OO", "-I", "-b", "-q"})
# ``python -m <module>`` launches the wrap can forward. Proton's front-end
# runs its target through ``runpy.run_path``, which resolves a *path* and not
# a module name -- with exactly one exception: a target whose basename is
# ``pytest`` is run in-process as ``pytest.main(args)``, which is precisely
# what ``python -m pytest args`` does. So that spelling has an exact
# equivalent and is translated to it; other modules have none.
_WRAPPABLE_MODULES = frozenset({"pytest"})


class ProtonWrapError(RuntimeError):
    """Raised when the Proton CLI wrap cannot be built for the given argv."""


def _is_python(arg: str) -> bool:
    return bool(_PYTHON_RE.match(Path(arg).name))


def _shebang_interpreter(command: str) -> str | None:
    """Return the Python interpreter a console script's ``#!`` line names.

    A ``pytest`` on ``$PATH`` is a generated console script shebanged to the
    interpreter that installed it, which in probe mode is frequently *not*
    the one aorta is running under. Reading it back is what keeps the wrap on
    the environment the operator's command would have used.

    Returns ``None`` when the target cannot be resolved or read, is not a
    script, or names a non-Python interpreter -- all cases the caller handles
    by falling back rather than guessing.
    """
    resolved = shutil.which(command)
    if resolved is None:
        return None
    try:
        with open(resolved, "rb") as stream:
            first_line = stream.readline(256)
    except OSError:
        return None
    if not first_line.startswith(b"#!"):
        return None
    try:
        parts = first_line[2:].decode("utf-8").split()
    except UnicodeDecodeError:
        return None
    if not parts:
        return None
    # ``#!/usr/bin/env python3`` names the interpreter in the argument.
    if Path(parts[0]).name == "env" and len(parts) > 1:
        parts = parts[1:]
    return parts[0] if _is_python(parts[0]) else None


def resolve_python(argv: list[str] | tuple[str, ...]) -> str:
    """Pick the interpreter that runs Proton's command front-end.

    Precedence: ``$AORTA_PROTON_PYTHON`` (an operator override for the case
    where Triton lives elsewhere), then the workload's own ``argv[0]`` when it
    is a Python interpreter, then -- for the bare ``pytest`` spelling, whose
    ``argv[0]`` is a console script rather than an interpreter -- the
    interpreter named by that script's shebang, and finally
    :data:`sys.executable`.

    The shebang step matters because the wrap replaces the command's own
    interpreter with this one. Falling straight through to
    :data:`sys.executable` for ``/some/venv/bin/pytest`` would profile a
    different environment than the one the operator asked to run, which can
    import a different dependency set or fail outright on a command that works
    unprofiled.
    """
    override = os.environ.get(ENV_PROTON_PYTHON)
    if override:
        return os.path.expandvars(os.path.expanduser(override))
    if argv and _is_python(argv[0]):
        return argv[0]
    if argv and Path(argv[0]).name in _WRAPPABLE_MODULES:
        interpreter = _shebang_interpreter(argv[0])
        if interpreter is not None:
            return interpreter
        log.warning(
            "proton: could not read an interpreter from %r; running Proton "
            "under %s instead. Set $%s if that is the wrong environment.",
            argv[0],
            sys.executable,
            ENV_PROTON_PYTHON,
        )
    return sys.executable


def build_argv_prefix(
    out_dir: Path | str,
    options: Mapping[str, str] | None = None,
    *,
    python: str | None = None,
) -> list[str]:
    """Return the ``<python> -m triton.profiler.proton <opts>`` prefix.

    The prefix is followed directly by the script and its arguments -- Proton's
    front-end collects them with ``argparse.REMAINDER``, so there is no ``--``
    separator (unlike rocprofv3).

    Args:
        out_dir: Directory the profile is written into.
        options: Recipe-supplied options; see :func:`validate_options`.
        python: Interpreter to run Proton under. Defaults to
            :data:`sys.executable`; callers with the workload's argv in hand
            should pass :func:`resolve_python` instead.

    Raises:
        ValueError: an option key or value is invalid.
    """
    effective = validate_options(options)
    out = Path(out_dir)
    argv = [
        python or sys.executable,
        "-m",
        PROTON_MODULE,
        "-n",
        str(out / PROFILE_BASENAME),
    ]
    # ``backend: auto`` means "omit ``-b``", which is what makes the default
    # work on every Triton: Proton's own ``_select_backend()`` picks the
    # backend matching the active runtime, whereas naming one that this
    # Triton's argparse does not list kills the run before the payload starts.
    if effective["backend"] != AUTO_BACKEND:
        argv += ["-b", effective["backend"]]
    argv += [
        "--context",
        effective["context"],
        "--data",
        effective["data"],
    ]
    mode = mode_argument(effective)
    if mode is not None:
        argv += ["--mode", mode]
    return argv


def build_env(
    out_dir: Path | str,
    options: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the ``AORTA_PROTON_*`` bundle for ``mode: env``.

    The variables mirror the CLI wrap's flags one-for-one so a workload that
    drives Proton itself can forward them straight into ``proton.start()``.
    ``AORTA_PROTON_NAME`` is the session name (path stem) to pass to
    ``proton.start(name)``; ``AORTA_PROTON_DIR`` is the directory it lives in,
    for a workload that wants to write sibling artifacts.

    Args:
        out_dir: Directory the profile should be written into.
        options: Recipe-supplied options; see :func:`validate_options`.

    Returns:
        A flat ``dict[str, str]`` the caller merges into the subprocess
        environment. Never mutates :data:`os.environ`.
    """
    effective = validate_options(options)
    out = Path(out_dir)
    env = {
        f"{ENV_PREFIX}DIR": str(out),
        f"{ENV_PREFIX}NAME": str(out / PROFILE_BASENAME),
        f"{ENV_PREFIX}CONTEXT": effective["context"],
        f"{ENV_PREFIX}DATA": effective["data"],
    }
    # Omitted for ``backend: auto`` so the workload passes ``backend=None`` to
    # ``proton.start()`` and gets Proton's own runtime-matched selection, the
    # same thing dropping ``-b`` gets the CLI wrap.
    if effective["backend"] != AUTO_BACKEND:
        env[f"{ENV_PREFIX}BACKEND"] = effective["backend"]
    mode = mode_argument(effective)
    if mode is not None:
        env[f"{ENV_PREFIX}MODE"] = mode
    return env


def _module_target(module: str, module_args: list[str], argv: list[str]) -> list[str]:
    """Translate a ``python -m <module>`` launch into Proton's target argv.

    Only :data:`_WRAPPABLE_MODULES` can be forwarded, and they are forwarded
    as the bare spelling Proton special-cases -- ``-m pytest`` becomes the
    ``pytest`` target, which Proton runs as ``pytest.main(args)``, the same
    call ``python -m pytest`` makes. Refusing here rather than forwarding an
    unrunnable module name keeps the failure at setup, where it names a fix,
    instead of surfacing as a ``run_path`` error from inside Proton.

    Raises:
        ProtonWrapError: ``-m`` carried no module, or one Proton cannot run.
    """
    if not module:
        raise ProtonWrapError(f"proton mode 'cli' saw '-m' with no module name in {argv!r}.")
    if module not in _WRAPPABLE_MODULES:
        raise ProtonWrapError(
            f"proton mode 'cli' cannot wrap 'python -m {module}'. Proton's "
            "front-end runs its target through runpy.run_path, which resolves "
            "a script path and not a module name; the only module it also "
            f"accepts is {sorted(_WRAPPABLE_MODULES)}. Invoke the module's "
            "script by path, or use 'proton: {mode: env}'."
        )
    return [module, *module_args]


def _split_python_launch(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``python -u script.py a b`` into interpreter flags and the target.

    Accepts the three launch spellings Proton's front-end can take over: a
    bare ``pytest ...``, a ``<python> <script.py> ...`` script launch, and a
    ``<python> -m pytest ...`` module launch (see :func:`_module_target`,
    which normalises the last onto the first).

    Raises:
        ProtonWrapError: the command is not a launch Proton's front-end can
            take over.
    """
    if not argv:
        raise ProtonWrapError("cannot wrap an empty argv with proton")
    if Path(argv[0]).name in _WRAPPABLE_MODULES:
        return [], list(argv)
    if not _is_python(argv[0]):
        raise ProtonWrapError(
            f"proton mode 'cli' needs a Python script launch, got {argv[0]!r}. "
            "Proton's command front-end executes a script, not an arbitrary "
            "command. Either invoke the workload as '<python> <script.py> "
            "...', or switch the recipe to 'proton: {mode: env}' and have the "
            "workload call proton.start()/finalize() itself."
        )
    rest = argv[1:]
    flags: list[str] = []
    for index, arg in enumerate(rest):
        if arg in _SAFE_PYTHON_FLAGS:
            flags.append(arg)
            continue
        # ``-m`` before the generic option guard: a module launch is a target,
        # not an interpreter flag, and ``python -m pytest`` must not fail
        # where the equivalent bare ``pytest`` succeeds. Both the separate
        # (``-m pytest``) and attached (``-mpytest``) spellings are handled,
        # for the same reason.
        if arg == "-m":
            module = rest[index + 1] if index + 1 < len(rest) else ""
            return flags, _module_target(module, rest[index + 2 :], argv)
        if arg.startswith("-m"):
            return flags, _module_target(arg[2:], rest[index + 1 :], argv)
        if arg.startswith("-"):
            raise ProtonWrapError(
                f"proton mode 'cli' cannot wrap interpreter option {arg!r} "
                f"(supported: {sorted(_SAFE_PYTHON_FLAGS)}, plus '-m' with "
                f"{sorted(_WRAPPABLE_MODULES)}). Invoke the script directly, "
                "or use 'proton: {mode: env}'."
            )
        return flags, rest[index:]
    raise ProtonWrapError(
        "proton mode 'cli' needs a script path after the interpreter; got " f"{argv!r}"
    )


def _device_env_prefix(
    backend: str,
    env: Mapping[str, str],
    extra: Mapping[str, str] | None = None,
) -> list[str]:
    """Build an ``env(1)`` prefix carrying device translation + extra vars.

    Proton on AMD raises outright when ``HIP_VISIBLE_DEVICES`` is *set* for a
    queue-intercepting backend, so the variable is unset and its value moved to
    ``ROCR_VISIBLE_DEVICES`` (unless the trial already set one). Presence, not
    truthiness, drives both halves: Proton rejects the variable on presence
    alone, and an explicitly empty device list conventionally means "hide every
    device", which is a selection to preserve rather than an absent one to
    ignore. Returns ``[]`` when there is nothing to carry, keeping the common
    argv unchanged.

    Raises:
        ProtonWrapError: there is something to carry but no ``env(1)`` to carry
            it with. Argv rewriting is the only environment channel the
            collector seam has, so an empty prefix here would run the command
            with none of ``assignments`` -- silently unprofiled in ``mode:
            env``, and on the wrong device (or crashing inside Proton) when a
            device translation was requested.
    """
    assignments = dict(extra or {})
    unset: list[str] = []
    hip = env.get("HIP_VISIBLE_DEVICES")
    if hip is not None and backend in QUEUE_INTERCEPTING_BACKENDS:
        unset.append("HIP_VISIBLE_DEVICES")
        rocr = env.get("ROCR_VISIBLE_DEVICES")
        assignments.setdefault("ROCR_VISIBLE_DEVICES", hip if rocr is None else rocr)
        log.warning(
            "proton: HIP_VISIBLE_DEVICES=%r is not honoured by Proton's %s "
            "backend; running with ROCR_VISIBLE_DEVICES=%r instead.",
            hip,
            backend,
            assignments["ROCR_VISIBLE_DEVICES"],
        )
    if not assignments and not unset:
        return []
    env_bin = shutil.which("env")
    if env_bin is None:
        raise ProtonWrapError(
            "proton needs 'env' on $PATH to apply "
            f"{sorted({*assignments, *unset})}, and it was not found. Argv "
            "rewriting is the only channel the collector seam has for these "
            "variables, so continuing would run the command unprofiled or on "
            "the wrong device. Install coreutils, or drop 'proton' from the "
            "collect request."
        )
    prefix = [env_bin]
    for name in unset:
        prefix += ["-u", name]
    prefix += [f"{key}={value}" for key, value in sorted(assignments.items())]
    return prefix


def wrap_argv(
    argv: list[str] | tuple[str, ...],
    out_dir: Path | str,
    options: Mapping[str, str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return ``argv`` running under Proton, writing the profile to ``out_dir``.

    Args:
        argv: The command the trial would otherwise have run.
        out_dir: Directory the profile is written into.
        options: Recipe-supplied options; see :func:`validate_options`.
        env: The environment the command will run with, read (never mutated)
            for the ``HIP_VISIBLE_DEVICES`` translation. Defaults to
            :data:`os.environ`.

    Returns:
        In ``mode: cli``, the Proton-wrapped launch. In ``mode: env``, the
        original argv fronted by an ``env(1)`` prefix carrying the
        ``AORTA_PROTON_*`` bundle -- the generic collector seam only gets to
        rewrite argv, so that is how the variables reach a command aorta does
        not otherwise own.

    Raises:
        ValueError: an option key or value is invalid.
        ProtonWrapError: ``mode: cli`` was requested for a command Proton's
            front-end cannot execute, or the wrap needs ``env(1)`` to carry
            variables into the command and it is not on ``$PATH``.
    """
    effective = validate_options(options)
    inner = list(argv)
    environ = env if env is not None else os.environ
    if effective["mode"] == "env":
        prefix = _device_env_prefix(effective["backend"], environ, build_env(out_dir, options))
        return [*prefix, *inner]

    flags, target = _split_python_launch(inner)
    python = resolve_python(inner)
    proton_argv = build_argv_prefix(out_dir, options, python=python)
    # Interpreter flags belong to the interpreter, so they must stay in front
    # of ``-m``, not be handed to Proton's own parser.
    proton_argv[1:1] = flags
    prefix = _device_env_prefix(effective["backend"], environ)
    return [*prefix, *proton_argv, *target]


__all__ = [
    "AUTO_BACKEND",
    "BACKENDS",
    "CONTEXTS",
    "DATA_FORMATS",
    "ENV_PREFIX",
    "ENV_PROTON_PYTHON",
    "GRANULARITIES",
    "INSTRUMENTATION_MODES",
    "MODES",
    "OPTION_KEYS",
    "OUTPUT_SUBDIR",
    "PROFILE_BASENAME",
    "PROTON_MODULE",
    "QUEUE_INTERCEPTING_BACKENDS",
    "ProtonWrapError",
    "build_argv_prefix",
    "build_env",
    "mode_argument",
    "parse_profile",
    "parse_summary",
    "resolve_python",
    "validate_options",
    "wrap_argv",
]
