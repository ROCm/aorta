#!/usr/bin/env python3
"""Write <case>/expected.json from <case>/kernel.hip and <case>/stderr.txt (C1.T4's golden files).

The goldens were reviewed by eye when first written. Rerunning this after a
change to feedback.py rewrites them; review every diff before committing it, or the test checks the
code against itself.
"""

import json
import sys
from pathlib import Path

from aorlta.rollout.feedback import build_feedback, compile_errors

HERE = Path(__file__).resolve().parent


def main() -> int:
    for case in sorted(p for p in HERE.iterdir() if p.is_dir()):
        source = (case / "kernel.hip").read_text(encoding="utf-8")
        for suffix in ("", "-rocm-7.2.4"):
            stderr = (case / f"stderr{suffix}.txt").read_text(encoding="utf-8")
            errors = compile_errors(stderr, source, "kernel.hip")
            fb = build_feedback(errors)
            out = {"errors": errors, "shown": fb.shown, "total": fb.total, "feedback": fb.text.split("\n")}
            text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
            (case / f"expected{suffix}.json").write_text(text, encoding="utf-8")
            print(f"{case.name}{suffix}: {fb.shown}/{fb.total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
