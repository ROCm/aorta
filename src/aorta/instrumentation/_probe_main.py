"""Dependency-free ``collect_env`` entry point for in-container probing.

The regular CLI (``aorta env probe``) imports Click and the entry-point
machinery, which need not exist inside a workload's docker image. This
module is the minimal alternative: it imports only stdlib plus
:mod:`aorta.instrumentation.environment` (itself stdlib-only), so a
container can produce a byte-identical ``env.json`` by bind-mounting the
aorta ``src`` tree and running::

    PYTHONPATH=/opt/aorta_src python -m aorta.instrumentation._probe_main /out/env.json

The output format is exactly ``EnvSnapshot.to_dict()`` serialised with
``json.dumps(..., indent=2)`` -- identical to what ``aorta env probe -o``
writes -- so the isolated-env snapshot the triage runner promotes is
indistinguishable from a local-env one on disk.

With no argument (or ``-``) the JSON goes to stdout, so the module also
works as ``docker exec ... > env.json`` without a bind mount.

Pass ``--extended`` to retain the full per-file kernel-catalog lists
(matching ``aorta env probe --extended``); the default is the compact
snapshot that drops those lists to keep the artifact small.

Pass ``--execution-context <direct|buck2_run|buck2_action>`` to stamp
``execution_context.probe_invocation`` (matching the CLI flag). Since this
entry point is the in-container / in-action probe, ``buck2_action`` is the
common value here.
"""

from __future__ import annotations

import json
import sys

from aorta.instrumentation.environment import (
    EXECUTION_CONTEXT_INVOCATIONS,
    collect_env,
    execution_context_warning,
)


def main(argv: list[str]) -> int:
    """Capture the snapshot and write it to the output path (or stdout).

    Recognizes ``--extended`` (full per-file catalog detail) and
    ``--execution-context <label>`` anywhere in argv; the first remaining
    non-flag argument is the output path (``-`` or absent -> stdout).
    """
    args = argv[1:]
    extended = "--extended" in args
    probe_invocation = "direct"
    positional: list[str] = []
    i = 0
    rest = [a for a in args if a != "--extended"]
    allowed = ", ".join(EXECUTION_CONTEXT_INVOCATIONS)
    while i < len(rest):
        a = rest[i]
        if a == "--execution-context":
            # Strict, matching Click's Choice: a missing or unknown value
            # must be a hard error, NOT a silent fall-through. Otherwise
            # `--execution-context /out/env.json` (forgotten value) would
            # eat the output path as the label, leave positional empty, and
            # dump JSON to stdout while exiting 0 -- silently failing to
            # write the very artifact this entry point exists to produce.
            if i + 1 >= len(rest) or rest[i + 1] not in EXECUTION_CONTEXT_INVOCATIONS:
                got = rest[i + 1] if i + 1 < len(rest) else "(missing)"
                sys.stderr.write(
                    f"aorta env probe: --execution-context requires one of: "
                    f"{allowed} (got {got!r})\n"
                )
                return 2
            probe_invocation = rest[i + 1]
            i += 2
            continue
        if a.startswith("--execution-context="):
            probe_invocation = a.split("=", 1)[1]
            if probe_invocation not in EXECUTION_CONTEXT_INVOCATIONS:
                sys.stderr.write(
                    f"aorta env probe: --execution-context requires one of: "
                    f"{allowed} (got {probe_invocation!r})\n"
                )
                return 2
            i += 1
            continue
        positional.append(a)
        i += 1

    snapshot = collect_env(
        detail="full" if extended else "compact",
        probe_invocation=probe_invocation,
    )

    # Same claim-vs-reality guardrail as the Click CLI (aorta env probe),
    # via the SHARED predicate so the two entry points never drift. This
    # dependency-free entry point is the one most likely used inside a Buck2
    # action / container, so it needs the warning even more. Fail-soft:
    # stderr only, never a non-zero exit.
    _ec_warning = execution_context_warning(
        probe_invocation, snapshot.container_detected
    )
    if _ec_warning is not None:
        sys.stderr.write(_ec_warning + "\n")

    snapshot_dict = snapshot.to_dict()
    text = json.dumps(snapshot_dict, indent=2)

    out = positional[0] if positional and positional[0] != "-" else None
    if out is None:
        sys.stdout.write(text + "\n")
        return 0

    # collect_env() never raises, but the write can (unwritable mount,
    # full disk). Surface it as a non-zero exit + stderr line rather than
    # a traceback so the wrapper's ``&& <workload>`` chain stops cleanly.
    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        sys.stderr.write(f"aorta env probe: failed to write {out}: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
