"""Shared Docker env-forwarding helper for workload wrappers.

The platform does NOT launch Docker -- workload wrappers own their ``docker
run`` invocation (mounts, devices, entrypoints, shared memory, inner command).
This module provides the one piece those wrappers should share: turning a
controlled env-var mapping into ``-e KEY=VALUE`` flags, so every wrapper
forwards the platform's ``config['_aorta_trial_env']`` overlay the same way.

Deliberately narrow: it accepts an explicit mapping, never reads ``os.environ``,
and returns a flat ``list[str]`` the wrapper splices into its argv. Keeping the
launch logic (image selection, mounts, `--device`, `--ipc`, `--shm-size`, the
inner command) in the wrapper preserves the platform's no-Docker-launching
policy -- this helper is env-forwarding only.
"""

from __future__ import annotations

import re

# Same POSIX env-var name shape the dispatcher enforces on ``extra_env`` /
# ``Environment.env`` (``aorta.run.dispatcher._ENV_KEY_RE``). Duplicated here
# (rather than imported) so this leaf helper has no dependency on the
# dispatcher module -- a wrapper can import ``docker_env_flags`` without pulling
# in the whole run-orchestration graph.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def docker_env_flags(env: dict[str, str]) -> list[str]:
    """Turn a controlled env mapping into ``docker run`` ``-e KEY=VALUE`` flags.

    Args:
        env: An explicit ``dict[str, str]`` of environment variables to forward
            -- typically the platform-supplied ``config['_aorta_trial_env']``
            overlay. Only these vars are forwarded; the helper NEVER reads
            ``os.environ``, so a wrapper cannot accidentally leak the ambient
            host environment into the container.

    Returns:
        A flat ``list[str]`` of alternating ``"-e"`` / ``"KEY=VALUE"`` tokens,
        ready to splice into a ``docker run`` argv. Deterministic: keys are
        emitted in sorted order so the same mapping always yields byte-identical
        output (stable for diffing and test assertions).

        SECURITY: the ``KEY=VALUE`` tokens contain the RAW environment values,
        which may be secrets (tokens, endpoints). Do NOT log the returned list
        or any completed ``docker run`` argv built from it, and do not persist
        it to run artifacts unless those artifacts are already treated as
        sensitive.

    Raises:
        TypeError: ``env`` is not a mapping, or any key/value is not a ``str``.
        ValueError: a key is not a valid POSIX env-var name
            (``[A-Za-z_][A-Za-z0-9_]*``), or a value contains a NUL byte. The
            offending key is named; VALUES are never echoed in the error (they
            may be secrets).
    """
    if not isinstance(env, dict):
        raise TypeError(
            f"docker_env_flags() expects a dict[str, str], got {type(env).__name__}"
        )
    # Validate before sorting. A mixed-type key set such as
    # ``{"GOOD": "1", 2: "bad"}`` otherwise fails inside ``sorted`` with an
    # implementation-level comparison error instead of this API's clear,
    # value-redacting validation error.
    for key, value in env.items():
        if not isinstance(key, str):
            raise TypeError(
                f"docker_env_flags() env keys must be str, got {type(key).__name__}"
            )
        if not isinstance(value, str):
            # Do NOT include the value in the message -- only its type. Values
            # may carry secrets (tokens, endpoints).
            raise TypeError(
                f"docker_env_flags() env value for key {key!r} must be str, "
                f"got {type(value).__name__}"
            )
        if "\x00" in value:
            # A NUL byte cannot round-trip through a ``docker run -e`` argument
            # (execve rejects it), so reject it here for parity with the
            # dispatcher's ``os.environ`` validation. Value is NOT echoed.
            raise ValueError(
                f"docker_env_flags() env value for key {key!r} contains a NUL "
                "byte and cannot be passed as an environment variable."
            )
        if not _ENV_KEY_RE.fullmatch(key):
            raise ValueError(
                f"docker_env_flags() invalid env-var name {key!r}: must match "
                "[A-Za-z_][A-Za-z0-9_]* (POSIX env-var name shape)."
            )

    flags: list[str] = []
    for key in sorted(env):
        value = env[key]
        flags.append("-e")
        flags.append(f"{key}={value}")
    return flags


__all__ = ["docker_env_flags"]
