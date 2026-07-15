#!/usr/bin/env python3
"""Invoke the aorta CLI with a JSON argv list (used inside mirage containers).

Reads ``AORTA_CLI_JSON`` from the environment, or ``sys.argv[1]`` if set.
Example::

    AORTA_CLI_JSON='["triage","run","--recipe","recipes/emulated/gpu-smoke-emulated.yaml"]' \\
        python3 scripts/emulation/aorta_cli_runner.py
"""
from __future__ import annotations

import json
import os
import sys

from aorta.cli import main


def parse_aorta_cli_json(raw: str) -> list[str]:
    """Decode ``AORTA_CLI_JSON`` into a CLI argv list.

    Raises:
        SystemExit: on invalid JSON or a non-list/non-string payload.
    """
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"aorta_cli_runner: invalid AORTA_CLI_JSON: {exc}") from exc
    if not isinstance(decoded, list):
        raise SystemExit(
            "aorta_cli_runner: AORTA_CLI_JSON must be a JSON list of strings, "
            f"got {type(decoded).__name__}"
        )
    if not all(isinstance(arg, str) for arg in decoded):
        raise SystemExit(
            "aorta_cli_runner: AORTA_CLI_JSON must be a JSON list of strings"
        )
    return decoded


if __name__ == "__main__":
    raw = os.environ.get("AORTA_CLI_JSON") or (sys.argv[1] if len(sys.argv) > 1 else "[]")
    args = parse_aorta_cli_json(raw)
    sys.argv = ["aorta", *args]
    raise SystemExit(main() or 0)
