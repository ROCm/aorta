"""Measurement tests: make the shared synthetic builders (measure_helpers.py) importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
