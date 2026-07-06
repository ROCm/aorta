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
"""

from __future__ import annotations

import json
import sys

from aorta.instrumentation.environment import collect_env


def main(argv: list[str]) -> int:
    """Capture the snapshot and write it to ``argv[1]`` (or stdout)."""
    snapshot_dict = collect_env().to_dict()
    text = json.dumps(snapshot_dict, indent=2)

    out = argv[1] if len(argv) > 1 and argv[1] != "-" else None
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
        sys.stderr.write(f"aorta probe: failed to write {out}: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
