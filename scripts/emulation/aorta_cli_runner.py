#!/usr/bin/env python3
"""Invoke the aorta CLI with a JSON argv list (used inside mirage containers).

Reads ``AORTA_CLI_JSON`` from the environment, or ``sys.argv[1]`` if set.
Example::

    AORTA_CLI_JSON='["triage","run","--recipe","recipes/gpu-smoke-emulated.yaml"]' \\
        python3 scripts/emulation/aorta_cli_runner.py
"""
from __future__ import annotations

import json
import os
import sys

from aorta.cli import main

if __name__ == "__main__":
    raw = os.environ.get("AORTA_CLI_JSON") or (sys.argv[1] if len(sys.argv) > 1 else "[]")
    args = json.loads(raw)
    sys.argv = ["aorta", *args]
    raise SystemExit(main() or 0)
