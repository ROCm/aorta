#!/usr/bin/env python3
"""Merge the shards of one eval column (``eval_rocjitsu --shard K/N``) into the column an unsharded run writes.

Every reply is seeded per (scenario, episode, step) and every scenario is rolled out
on its own, so a shard's groups are the unsharded column's groups for those
scenarios. This puts them back in the split's order and recomputes the metrics. It
refuses shards that disagree on any setting but their scenario lists, that overlap,
that are unfinished, or that do not together cover the split exactly as
``eval_rocjitsu`` picks it.

    merge_columns.py --corpus-root <archive> --split val --out <dir>/after.json shard-*/after.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_rocjitsu as ev  # noqa: E402
import rocjitsu_env as rj  # noqa: E402

PER_SHARD = ("scenarios", "corpus")


def merge(shards: list[dict[str, Any]], picked: list[str]) -> dict[str, Any]:
    """The unsharded column for the scenario ids ``picked``, in that order, from its shards."""
    if not shards:
        raise ValueError("no shards to merge")
    common = {k: v for k, v in shards[0]["config"].items() if k not in PER_SHARD}
    for i, shard in enumerate(shards):
        if "metrics" not in shard:
            raise ValueError(f"shard {i} is unfinished (it has no metrics)")
        config = {k: v for k, v in shard["config"].items() if k not in PER_SHARD}
        if config != common:
            keys = sorted(k for k in set(config) | set(common) if config.get(k) != common.get(k))
            raise ValueError(f"shard {i} differs from shard 0 in {keys}")
    groups: dict[str, dict[str, Any]] = {}
    wire: dict[str, list[dict[str, Any]]] = {}
    corpus: dict[str, Any] = {}
    for i, shard in enumerate(shards):
        for group in shard["groups"]:
            sid = group["scenario_id"]
            if sid in groups:
                raise ValueError(f"{sid} is in more than one shard")
            groups[sid] = group
        for row in shard["wire"]:
            wire.setdefault(row["scenario_id"], []).append(row)
        corpus.update(shard["config"]["corpus"])
    if set(groups) != set(picked):
        missing = sorted(set(picked) - set(groups))
        extra = sorted(set(groups) - set(picked))
        raise ValueError(f"the shards hold {len(groups)} of the split's {len(picked)} scenarios "
                         f"(missing {missing[:3]}{'...' if len(missing) > 3 else ''}, "
                         f"not in the split {extra[:3]}{'...' if len(extra) > 3 else ''})")
    if set(wire) != set(groups):
        raise ValueError("the shards' wire rows and groups name different scenarios")
    merged_config = dict(common, scenarios=list(picked), corpus={sid: corpus[sid] for sid in picked})
    doc = {"groups": [groups[sid] for sid in picked], "wire": [row for sid in picked for row in wire[sid]],
           "config": {k: merged_config[k] for k in shards[0]["config"]}}
    doc["metrics"] = ev.summarize(ev.episode_rows(doc))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-faulty", type=int, default=0)
    ap.add_argument("--max-clean", type=int, default=-1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("shards", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"{a.out} exists; refusing to overwrite it", file=sys.stderr)
        return ev.EXIT_REFUSED
    docs = [json.loads(p.read_text()) for p in a.shards]
    split = {d["config"].get("split") for d in docs}
    if split != {a.split}:
        print(f"the shards were run on split(s) {sorted(map(str, split))}, not {a.split}", file=sys.stderr)
        return ev.EXIT_REFUSED
    chosen = ev.pick_scenarios(rj.load_corpus(a.corpus_root, split=a.split), a.max_faulty, a.max_clean)
    picked = [s.scenario_id for s in chosen]
    try:
        doc = merge(docs, picked)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return ev.EXIT_REFUSED
    stale = sorted(s.scenario_id for s in chosen if doc["config"]["corpus"][s.scenario_id] != s.digest)
    if stale:
        print(f"{len(stale)} scenario(s) were computed on another revision of the corpus, e.g. {stale[:3]}",
              file=sys.stderr)
        return ev.EXIT_REFUSED
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ev.write(a.out, doc)
    m = doc["metrics"]["all"]
    print(f"[merge] {len(docs)} shard(s) -> {a.out}: {len(doc['groups'])} scenarios, mean {m['mean']:+.4f}, "
          f"solved {m['solved_rate']:.4f}, exact {m['exact_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
