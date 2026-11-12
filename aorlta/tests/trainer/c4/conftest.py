"""C4 tests: make the shared builders in this directory importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
