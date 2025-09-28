"""CLI entry point for the AORTA FSDP2 benchmarking workload."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from aorta.training.fsdp_trainer import main_cli


if __name__ == "__main__":
    main_cli()
