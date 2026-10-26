#!/usr/bin/env python3
"""Mix demonstration files (rocjitsu_demos.py output) for a continual post-training cycle.

Each input is ``PATH[:N]``: take N whole episodes from PATH, sampled with a fixed
seed (all of them when N is omitted). An episode is kept or dropped with every one of
its turns, since a demonstration that starts mid-episode teaches replies to states
the policy only reaches by the turns that were dropped. Every input must have been
rendered under one system prompt; the output carries that prompt's hash in its
``.stats.json``, which ``sft_rocjitsu.py`` checks.

    mix_demos.py --out mix.jsonl new.jsonl old.jsonl:300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def episode_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    """One episode: its input file, scenario, trainer iteration (0 outside a wire log) and index."""
    return row["source"], row["scenario_id"], row.get("iteration", 0), row["episode"]


def read(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stats_path = path.with_suffix(".stats.json")
    if not stats_path.is_file():
        raise ValueError(f"{path} has no {stats_path.name}, so the system prompt it was rendered under is unknown")
    return rows, json.loads(stats_path.read_text())


def take(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    episodes: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        episodes.setdefault(episode_key(row), []).append(row)
    keys = sorted(episodes)
    if n is not None and n < len(keys):
        keys = sorted(random.Random(seed).sample(keys, n))
    return [row for k in keys for row in sorted(episodes[k], key=lambda r: r["step"])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("inputs", nargs="+", help="PATH or PATH:N (N whole episodes, sampled)")
    a = ap.parse_args(argv)
    out: list[dict[str, Any]] = []
    shas: set[str] = set()
    parts = []
    for spec in a.inputs:
        path_text, _, n_text = spec.rpartition(":") if ":" in spec else (spec, "", "")
        if not path_text:
            path_text, n_text = spec, ""
        path = Path(path_text)
        try:
            n = int(n_text) if n_text else None
            rows, stats = read(path)
        except ValueError as exc:
            print(f"[refused] {exc}", file=sys.stderr)
            return 3
        if n is not None and n < 0:
            print(f"[refused] {spec}: N must be >= 0", file=sys.stderr)
            return 3
        shas.add(stats.get("system_sha256"))
        kept = take(rows, n, a.seed)
        out.extend(kept)
        parts.append({"path": str(path), "asked": n, "episodes": len({episode_key(r) for r in kept}),
                      "samples": len(kept), "of_episodes": len({episode_key(r) for r in rows})})
    if len(shas) != 1 or None in shas:
        print(f"[refused] the inputs were rendered under {len(shas)} system prompt(s): {sorted(map(str, shas))}",
              file=sys.stderr)
        return 3
    keys = [episode_key(r) for r in out if r["step"] == 1]
    if len(keys) != len(set(keys)):
        print("[refused] an episode appears in more than one input", file=sys.stderr)
        return 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    stats = {"system_sha256": shas.pop(), "seed": a.seed, "inputs": parts, "samples": len(out),
             "episodes": len(keys), "episodes_by_source": dict(Counter(k[0] for k in keys))}
    a.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[mix] {len(keys)} episodes, {len(out)} samples -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
