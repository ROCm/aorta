#!/usr/bin/env python3
"""``fpsan_command`` for daily-fpsan-mismatch: a dropped-accumulate candidate against ``row_dot``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpsan_pair import main  # noqa: E402

raise SystemExit(main(["mismatch"]))
