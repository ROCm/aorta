#!/usr/bin/env python3
"""Prepare gfx950 GEMM code-object fixtures for sanitizer CI."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


def _read_top_solution_indices(csv_path: Path, top_n: int) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(
            csv.DictReader(
                line for line in stream if not line.lstrip().startswith("#")
            )
        )
    ranked = sorted(rows, key=lambda row: int(row["count"]), reverse=True)
    return [row["top_solution_idx"] for row in ranked[:top_n]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bundler = shutil.which("clang-offload-bundler")
    if bundler is None:
        raise SystemExit("clang-offload-bundler not found on PATH")
    co_files = sorted(Path("/opt/rocm/lib/hipblaslt/library").glob("TensileLibrary_*SB*gfx950.co"))
    if not co_files:
        raise SystemExit("no gfx950 hipBLASLt code-object bundle found")
    base = args.out / "base_gfx950.hsaco"
    subprocess.run(
        [
            bundler,
            "--type=o",
            "--unbundle",
            f"--input={co_files[0]}",
            f"--output={base}",
            "--targets=hipv4-amdgcn-amd-amdhsa--gfx950",
        ],
        check=True,
    )
    for idx in _read_top_solution_indices(args.csv, args.top_n):
        shutil.copy2(base, args.out / f"sol_{idx}.hsaco")
    base.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
