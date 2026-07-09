"""Launch helpers that run an AORTA workload under the mirage GPU emulator.

The dispatcher threads the resolved :class:`~aorta.registry.types.Environment`
into every trial's config as ``config["_aorta_environment"]`` (an ``asdict``
of the dataclass). When that descriptor carries ``mirage_profile`` (or
``emulator``), the workload is meant to run on an emulated GPU rather than the
host's real hardware. This module turns that descriptor into a concrete launch:

* :func:`wrap_argv_for_environment` -- the **A2 (mirage exec)** path: given the
  workload's real argv (e.g. ``["torchrun", "--nproc_per_node=2", ...]``),
  return ``["mirage", "run", "--profile", <p>, "--", *argv]`` so mirage hosts a
  session, injects the rocjitsu ``LD_PRELOAD``, and runs the command on the
  emulated GPU. When the environment is *not* emulated, the argv is returned
  unchanged so existing (real-hardware) launches are byte-for-byte identical.

This module **constructs** the launch; it does not spawn anything. That keeps
it pure/unit-testable and mirrors AORTA's policy that the platform threads
tier hints while wrappers decide how to launch. The rocjitsu raw-``LD_PRELOAD``
fast path (A1) is intentionally out of scope here until the spike confirms the
asset-resolution contract; see the design doc.

Binary resolution: the mirage CLI is taken from ``$MIRAGE_BIN`` (default
``"mirage"`` on ``$PATH``), matching mirage's own ``MIRAGE_HOST_BIN`` style.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

# Reserved config key the dispatcher writes the resolved Environment into.
# Mirror of the literal used in aorta.run.dispatcher (kept as a literal here to
# avoid importing the dispatcher and creating an import cycle).
CONFIG_KEY_ENVIRONMENT = "_aorta_environment"

# Env var naming the mirage CLI binary, peer of mirage's own MIRAGE_HOST_BIN.
ENV_MIRAGE_BIN = "MIRAGE_BIN"
_DEFAULT_MIRAGE_BIN = "mirage"


class EmulationError(RuntimeError):
    """Raised when an emulated launch is requested but cannot be constructed.

    Examples: a ``mirage_profile`` is set but the ``mirage`` binary is not on
    ``$PATH`` (and ``$MIRAGE_BIN`` is unset/invalid), or the descriptor is
    internally inconsistent.
    """


@dataclass(frozen=True)
class MirageLaunchSpec:
    """A fully-resolved emulated launch.

    Attributes:
        mirage_bin: Resolved path/name of the mirage CLI.
        profile: mirage profile name the session runs under.
        inner_argv: The workload's original argv (what runs *inside* mirage).
        argv: The full argv to actually exec
            (``[mirage_bin, "run", "--profile", profile, ...flags..., "--", *inner_argv]``).
        workdir: Optional working directory passed to ``mirage run --workdir``.
        emulator: Optional backend hint carried from the environment
            (``"rocjitsu"`` etc.); informational -- the profile is authoritative.
    """

    mirage_bin: str
    profile: str
    inner_argv: tuple[str, ...]
    argv: tuple[str, ...]
    workdir: str | None = None
    emulator: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)


def _environment_descriptor(config: dict[str, Any]) -> dict[str, Any]:
    """Pull the platform-threaded environment descriptor out of a config dict.

    Returns an empty dict when the key is absent or malformed so callers can
    treat "no environment" and "non-emulated environment" uniformly.
    """
    env = config.get(CONFIG_KEY_ENVIRONMENT)
    return env if isinstance(env, dict) else {}


def is_emulated_environment(config: dict[str, Any]) -> bool:
    """True if this trial's resolved environment targets the GPU emulator.

    Keys off ``mirage_profile`` (authoritative -- a full mirage profile) or a
    non-``noop`` ``emulator`` hint. ``noop`` is mirage's pass-through backend
    (runs the command directly with no emulation), so it is treated as
    *not* emulated for launch purposes.
    """
    env = _environment_descriptor(config)
    if env.get("mirage_profile"):
        return True
    emulator = env.get("emulator")
    return bool(emulator) and emulator != "noop"


def resolve_mirage_bin() -> str:
    """Resolve the mirage CLI binary, honouring ``$MIRAGE_BIN``.

    Raises:
        EmulationError: if neither ``$MIRAGE_BIN`` nor a ``mirage`` on ``$PATH``
            can be found.
    """
    override = os.environ.get(ENV_MIRAGE_BIN)
    if override:
        # Expand ``~`` and ``$VARS`` so copy/pasted overrides like
        # ``MIRAGE_BIN=~/bin/mirage`` resolve like typical shell paths.
        override = os.path.expandvars(os.path.expanduser(override))
        # An override with a directory component -- absolute *or* relative
        # (e.g. ``./build/manylinux/bin/mirage``, as used in this repo's docs
        # and helper scripts) -- is a filesystem path, not a $PATH lookup.
        # Normalise it to an absolute path so the result stays valid even if
        # the working directory changes before mirage is exec'd.
        if os.path.dirname(override):
            candidate = os.path.abspath(override)
            if os.access(candidate, os.X_OK):
                return candidate
            raise EmulationError(
                f"{ENV_MIRAGE_BIN}={override!r} looks like a path but does not "
                "point to an executable file."
            )
        # A bare name (no directory component): resolve it on $PATH.
        found = shutil.which(override)
        if found:
            return found
        raise EmulationError(
            f"{ENV_MIRAGE_BIN}={override!r} was not found on $PATH."
        )
    found = shutil.which(_DEFAULT_MIRAGE_BIN)
    if found:
        return found
    raise EmulationError(
        "mirage CLI not found: set $MIRAGE_BIN to the mirage binary, or put "
        "'mirage' on $PATH. Required to run a workload under the GPU emulator."
    )


def resolve_emulation(
    config: dict[str, Any],
    inner_argv: list[str] | tuple[str, ...],
    *,
    workdir: str | None = None,
    reuse_session: str | None = None,
    keep_session: bool = False,
) -> MirageLaunchSpec | None:
    """Build a :class:`MirageLaunchSpec` for an emulated environment, else ``None``.

    Returns ``None`` (not an error) when the environment is not emulated, so
    callers can do ``spec = resolve_emulation(...) ; argv = spec.argv if spec
    else inner_argv``.

    Args:
        config: the workload config dict (must contain the dispatcher-threaded
            ``_aorta_environment`` for emulation to trigger).
        inner_argv: the workload's real launch argv.
        workdir: optional cwd forwarded to ``mirage run --workdir``.
        reuse_session: optional existing mirage session id
            (``mirage run --session ID``) so multiple cells share one session.
        keep_session: keep a mirage-created session alive after the exec
            (``mirage run --keep-session``).

    Raises:
        EmulationError: emulation is requested but cannot be constructed
            (missing profile, missing mirage binary, empty argv).
    """
    if not is_emulated_environment(config):
        return None

    inner = tuple(inner_argv)
    if not inner:
        raise EmulationError("cannot build an emulated launch from an empty argv")

    env = _environment_descriptor(config)
    profile = env.get("mirage_profile")
    emulator = env.get("emulator")
    if not profile:
        # An emulator hint without a profile is not launchable via `mirage run
        # --profile`. Surface a clear, actionable error rather than silently
        # running unemulated.
        raise EmulationError(
            "emulated environment requires 'mirage_profile' to launch via "
            f"`mirage run --profile`; got emulator={emulator!r} with no "
            "profile. Register a named environment that sets mirage_profile, "
            "or add the profile to the environment."
        )

    mirage_bin = resolve_mirage_bin()

    argv: list[str] = [mirage_bin, "run", "--profile", str(profile)]
    if reuse_session:
        argv += ["--session", str(reuse_session)]
    if keep_session:
        argv += ["--keep-session"]
    if workdir:
        argv += ["--workdir", str(workdir)]
    argv += ["--", *inner]

    return MirageLaunchSpec(
        mirage_bin=mirage_bin,
        profile=str(profile),
        inner_argv=inner,
        argv=tuple(argv),
        workdir=workdir,
        emulator=emulator,
    )


def wrap_argv_for_environment(
    config: dict[str, Any],
    inner_argv: list[str] | tuple[str, ...],
    *,
    workdir: str | None = None,
    reuse_session: str | None = None,
    keep_session: bool = False,
) -> list[str]:
    """Return the argv to exec: mirage-wrapped when emulated, unchanged otherwise.

    This is the one-call convenience for subprocess-shaped wrappers (e.g.
    :class:`aorta.workloads._subprocess.SubprocessWorkload`): pass the user's
    opaque argv and the trial config; get back either the same argv (real
    hardware) or the ``mirage run --profile … -- <argv>`` form (emulated).

    Raises:
        EmulationError: emulation is requested but cannot be constructed.
    """
    spec = resolve_emulation(
        config,
        inner_argv,
        workdir=workdir,
        reuse_session=reuse_session,
        keep_session=keep_session,
    )
    if spec is None:
        return list(inner_argv)
    return list(spec.argv)


__all__ = [
    "CONFIG_KEY_ENVIRONMENT",
    "ENV_MIRAGE_BIN",
    "EmulationError",
    "MirageLaunchSpec",
    "is_emulated_environment",
    "resolve_emulation",
    "resolve_mirage_bin",
    "wrap_argv_for_environment",
]
