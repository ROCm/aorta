#!/usr/bin/env python3
"""Append sanitizer nightly results to local history (placeholder for Pages publish)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: update_history.py <incoming-dir>", file=sys.stderr)
        return 2
    incoming = Path(argv[1])
    history = Path("sanitizer-history.json")
    payload = {"date": datetime.now(tz=timezone.utc).date().isoformat(), "cases": {}}
    for case in ("waitcheck", "consan-clean", "consan-racy"):
        report = incoming / case / "sanitizer_report.json"
        if report.is_file():
            payload["cases"][case] = json.loads(report.read_text(encoding="utf-8"))
    if history.is_file():
        existing = json.loads(history.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []
    existing.append(payload)
    history.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
