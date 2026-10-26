#!/usr/bin/env python3
"""Demonstrations for a warm start: solved episodes, built only from what the agent sees.

Two sources, both replayed through :class:`rocjitsu_env.Episode` so each step's
prompt is exactly the one the trainer would render:

* ``recipe``: the scripted evidence-window recipe, written against the agent's
  own observations. It names runtime kernels from ``list_kernels`` and reads
  offsets only from the instruction lines a ``disasm`` reply shows, so no answer
  rests on anything the agent was not shown.
* ``wire``: the policy's own episodes from a trainer ``wire.jsonl``, replayed
  reply by reply. A replay whose reward differs from the recorded one is refused,
  since it would mean the environment or the archive is not the one the run saw.

Only solved episodes are kept: the exact location on a faulty scenario, "none"
on a clean control.

    rocjitsu_demos.py --corpus-root <corpus> --split train --out demos.jsonl \\
        [--recipe] [--wire run/wire.jsonl ...] [--max-wire-per-scenario 2]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocjitsu_env as rj  # noqa: E402

SYNC = ("s_waitcnt", "s_barrier")


def runtime_kernels(listing: Sequence[dict[str, Any]]) -> set[str]:
    """Kernels ``list_kernels`` names as the HIP runtime's own (copies, fills, unnamed helpers)."""
    return {k["kernel"] for k in listing if k["name"].startswith(("__amd_rocclr", "?"))}


def visible_listing(result: Any) -> list[tuple[int, str]]:
    """The ``(offset, text)`` lines a ``disasm`` reply shows the agent, and no more."""
    out = []
    for line in str(result or "").splitlines()[: rj.DISASM_MAX_LINES]:
        off, _, text = line.partition(": ")
        try:
            out.append((int(off, 16), text.strip()))
        except ValueError:
            continue
    return out


def pick_in_window(lines: Sequence[tuple[int, str]], lo: int, hi: int) -> int:
    """``baselines2.pick_in_window``: the last sync instruction in ``[lo, hi]``, else the last line before ``hi``."""
    sync = [o for o, t in lines if lo <= o <= hi and t.startswith(SYNC)]
    if sync:
        return max(sync)
    before = [o for o, _ in lines if o < hi]
    return before[-1] if before else hi


def window(lo: int, hi: int) -> tuple[int, int]:
    """A ``disasm`` range ending at ``hi`` that fits in one reply and covers ``[lo, hi]`` when it can."""
    return max(0, lo - 0x40, hi - 0x1F0), hi


def recipe(episode: rj.Episode) -> Iterator[dict[str, Any]]:
    """The evidence-window recipe on the agent's observations; ``episode.last_result`` is each reply's object."""
    yield {"tool": "list_kernels"}
    runtime = runtime_kernels(episode.last_result or [])
    yield {"tool": "waitcheck"}
    diag = next((d for d in episode.last_result or [] if d["kernel"] not in runtime), None)
    if diag is not None:
        kernel, cls = diag["kernel"], "waitcnt"
        lo, hi = sorted((diag["producer_offset"], diag["consumer_offset"]))
    else:
        yield {"tool": "run_sim", "kernel": "all"}
        found = None
        for kr in episode.last_result or []:
            if kr["kernel"] in runtime:
                continue
            for race in kr.get("races") or []:
                offs = [m["offset"] for m in race["instructions"] if m.get("offset") is not None]
                if offs:
                    found = kr["kernel"], ("barrier" if race["type"] == "LDS" else "waitcnt"), min(offs), max(offs)
                    break
            if found is None:
                for m in kr.get("memory_wait") or []:
                    if m.get("consumer_offset") is not None:
                        prod = m["producer_offset"] if m.get("producer_offset") is not None else m["consumer_offset"]
                        found = (kr["kernel"], "waitcnt", *sorted((prod, m["consumer_offset"])))
                        break
            if found is not None:
                break
        if found is None:
            yield {"answer": {"class": "none"}}
            return
        kernel, cls, lo, hi = found
    start, end = window(lo, hi)
    yield {"tool": "disasm", "kernel": kernel, "start": rj.hexoff(start), "end": rj.hexoff(end)}
    offset = pick_in_window(visible_listing(episode.last_result), lo, hi)
    yield {"answer": {"class": cls, "kernel": kernel, "offset": rj.hexoff(offset)}}


def is_quirk(diag: dict[str, Any]) -> bool:
    """waitcheck's s_load -> s_load write-after-write diagnostic (:data:`rocjitsu_env.QUIRKS_V2`)."""
    return (re.search(r"before def of s\d+", str(diag.get("what", ""))) is not None
            and str(diag.get("producer", "")).startswith("s_load")
            and str(diag.get("consumer", "")).startswith("s_load"))


def quirk_recipe(episode: rj.Episode, inspect: bool = False) -> Iterator[dict[str, Any]]:
    """The recipe with waitcheck's known quirk ignored, as an expert would; used only where it answers none.

    ``inspect`` first reads the flagged code and simulates that kernel, so the demonstration
    shows the quirk being looked at and set aside rather than never examined.
    """
    yield {"tool": "list_kernels"}
    runtime = runtime_kernels(episode.last_result or [])
    yield {"tool": "waitcheck"}
    diags = [d for d in episode.last_result or [] if d["kernel"] not in runtime]
    real = [d for d in diags if not is_quirk(d)]
    if real or not diags:
        return
    if inspect:
        d = diags[0]
        lo, hi = sorted((d["producer_offset"], d["consumer_offset"]))
        start, end = window(lo, hi)
        yield {"tool": "disasm", "kernel": d["kernel"], "start": rj.hexoff(start), "end": rj.hexoff(end)}
        yield {"tool": "run_sim", "kernel": d["kernel"]}
    else:
        yield {"tool": "run_sim", "kernel": "all"}
    for kr in episode.last_result or []:
        if kr["kernel"] not in runtime and (kr.get("races") or kr.get("memory_wait")):
            return
    yield {"answer": {"class": "none"}}


def quirk_demos(scenarios: Sequence[rj.Scenario]) -> tuple[list[rj.Episode], Counter]:
    """Verified expert episodes on clean controls whose only waitcheck diagnostics are the quirk."""
    kept, stats = [], Counter()
    for s in scenarios:
        if not s.clean:
            continue
        for index, inspect in enumerate((False, True)):
            ep = play(s, functools.partial(quirk_recipe, inspect=inspect), index=index)
            if ep.answer is None:
                stats["not_quirk_only"] += 1
                continue
            if ep.outcome != "clean_correct":
                stats["not_clean_correct"] += 1
                continue
            stats["kept"] += 1
            kept.append(ep)
    return kept, stats


def play(scenario: rj.Scenario, policy, index: int = 0) -> rj.Episode:
    """One episode of ``policy`` (a generator over the episode) with the trainer's per-episode seeding."""
    episode = rj.Episode(scenario, index=index)
    actions = policy(episode)
    while episode.observe() is not None:
        try:
            action = next(actions)
        except StopIteration:
            break
        episode.act(json.dumps(action))
    return episode


def samples(episode: rj.Episode, source: str, iteration: int = 0) -> list[dict[str, Any]]:
    """The ``(prompt, completion)`` pairs of a solved episode, as the trainer's loss would see them.

    ``(source, scenario_id, iteration, episode)`` names the episode; ``mix_demos`` keys on it.
    """
    return [{"scenario_id": episode.scenario.scenario_id, "source": source, "iteration": iteration,
             "episode": episode.index, "step": s.n, "prompt": s.prompt, "completion": s.raw, "outcome": episode.outcome,
             "total": round(episode.score().total, 4)} for s in episode.steps]


def replay_wire(rows: Sequence[dict[str, Any]], scenarios: dict[str, rj.Scenario],
                budget: int = rj.MAX_TURNS) -> Iterator[tuple[rj.Episode, dict[str, Any]]]:
    """Every episode in a trainer wire log, rebuilt reply by reply, with its recorded row."""
    episodes: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[(row["iteration"], row["scenario_id"], row["episode"])].append(row)
    for (_it, sid, index), steps in sorted(episodes.items()):
        if sid not in scenarios:
            continue
        steps.sort(key=lambda r: r["step"])
        episode = rj.Episode(scenarios[sid], index=index, budget=budget)
        for row in steps:
            if episode.observe() is None:
                break
            episode.act(row["raw"])
        recorded = steps[-1]
        if round(episode.score().total, 4) != round(recorded["reward"], 4) or episode.outcome != recorded["terminal"]:
            raise SystemExit(f"replay of {sid} episode {index} (iteration {_it}) scored "
                             f"{episode.score().total:.4f}/{episode.outcome}, the run recorded "
                             f"{recorded['reward']:.4f}/{recorded['terminal']}: not the environment the run saw")
        yield episode, recorded


def budget_of(config: dict[str, Any], where: Path) -> int:
    """The turn budget an input was generated under; the replayed prompts state it."""
    budget = config.get("max_episode_steps")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise SystemExit(f"{where} records no usable max_episode_steps ({budget!r})")
    return budget


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recipe", action="store_true", help="add the recipe's solved episodes")
    ap.add_argument("--wire", type=Path, action="append", default=[], help="a trainer wire.jsonl to harvest")
    ap.add_argument("--max-wire-per-scenario", type=int, default=2,
                    help="at most this many distinct solved policy episodes per scenario")
    ap.add_argument("--max-wire-iteration", type=int, default=0, help="only iterations <= this (0 = all)")
    ap.add_argument("--column", type=Path, action="append", default=[],
                    help="an eval_rocjitsu column (before.json / after.json) whose episodes to harvest; it must "
                         "have been generated under the system prompt the demos are built under")
    ap.add_argument("--best-first", action="store_true",
                    help="per scenario, keep the highest-scoring solved episodes rather than the first ones")
    ap.add_argument("--quirk-demos", type=int, default=0, metavar="N",
                    help="add N copies of each verified quirk-handling episode (quirk_recipe, two variants) on the "
                         "split's clean controls whose only waitcheck diagnostics are the known quirk")
    ap.add_argument("--generated-under", default=None, metavar="PROFILE",
                    help="accept --column inputs generated under this prompt profile instead of the one the demos "
                         "are built under (context distillation: the episodes were sampled with a fuller prompt "
                         "and are trained under the deployment prompt); both hashes go into the stats")
    a = ap.parse_args(argv)
    scenarios = {s.scenario_id: s for s in rj.load_corpus(a.corpus_root, split=a.split)}
    system = rj.system_for(list(scenarios.values()))
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"split": a.split, "scenarios": len(scenarios),
                             "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                             "reward": rj.resolve_reward(), "prompt_profile": rj.resolve_prompt()}
    if a.recipe:
        played = [play(s, recipe) for s in scenarios.values()]
        kept = [e for e in played if e.solved]
        stats["recipe"] = {"played": len(played), "solved": len(kept),
                           "outcomes": dict(Counter(e.outcome for e in played))}
        for e in kept:
            out.extend(samples(e, "recipe"))
    # (identity, turn budget, rows) per input; the identity is the input's resolved path.
    sources: list[tuple[str, int, list[dict[str, Any]]]] = []
    expected = stats["system_sha256"]
    for path in a.wire:
        log_path = path.parent / "train-log.json"
        if not log_path.is_file():
            raise SystemExit(f"{path} has no {log_path.name} beside it, so its system prompt and turn budget "
                             "are unknown")
        log = json.loads(log_path.read_text())
        if log.get("system_prompt_sha256") != expected:
            raise SystemExit(f"{path} was generated under system prompt {str(log.get('system_prompt_sha256'))[:12]}; "
                             f"expected {expected[:12]} (AORTA_RJ_PROMPT / AORTA_RJ_REWARD differ)")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if a.max_wire_iteration:
            rows = [r for r in rows if r["iteration"] <= a.max_wire_iteration]
        sources.append((f"wire:{path.resolve()}", budget_of(log, log_path), rows))
    if a.generated_under is not None:
        source_system = rj.system_prompt_v2(rj.resolve_reward(), profile=a.generated_under)
        if rj.system_for(list(scenarios.values())) != rj.system_prompt_v2(rj.resolve_reward()):
            raise SystemExit("--generated-under is defined for corpus-v2 system prompts only")
        expected = hashlib.sha256(source_system.encode()).hexdigest()
        stats["generated_under"] = {"profile": a.generated_under, "system_sha256": expected}
    for path in a.column:
        doc = json.loads(path.read_text())
        recorded = doc["config"].get("system_sha256")
        if recorded != expected:
            raise SystemExit(f"{path} was generated under system prompt {str(recorded)[:12]}; expected "
                             f"{expected[:12]} (AORTA_RJ_PROMPT / AORTA_RJ_REWARD / --generated-under differ)")
        sources.append((f"column:{path.resolve()}", budget_of(doc["config"], path),
                        [dict(r, iteration=0) for r in doc["wire"]]))
    if a.quirk_demos:
        kept, qstats = quirk_demos(list(scenarios.values()))
        stats["quirk"] = dict(qstats, copies=a.quirk_demos, scenarios=len({e.scenario.scenario_id for e in kept}))
        for copy in range(a.quirk_demos):
            for e in kept:
                for row in samples(e, f"quirk:{copy}"):
                    out.append(row)
    seen: Counter[str] = Counter()
    distinct: set[tuple[str, ...]] = set()
    wire_stats = Counter()
    candidates: list[tuple[str, rj.Episode, int]] = []
    for source, budget, rows in sources:
        for episode, row in replay_wire(rows, scenarios, budget=budget):
            wire_stats["episodes"] += 1
            if episode.solved:
                wire_stats["solved"] += 1
                candidates.append((source, episode, row["iteration"]))
    if a.best_first:
        candidates.sort(key=lambda c: -c[1].score().total)
    for source, episode, iteration in candidates:
        key = (episode.scenario.scenario_id, *(s.raw for s in episode.steps))
        sid = episode.scenario.scenario_id
        if key in distinct or seen[sid] >= a.max_wire_per_scenario:
            continue
        distinct.add(key)
        seen[sid] += 1
        out.extend(samples(episode, source, iteration))
    stats["wire"] = dict(wire_stats)
    stats["wire_scenarios_kept"] = len(seen)
    stats["samples"] = len(out)
    stats["episodes_by_source"] = dict(Counter(s["source"] for s in out if s["step"] == 1))
    stats["clean_episodes"] = sum(1 for s in out if s["step"] == 1 and scenarios[s["scenario_id"]].clean)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(s) + "\n" for s in out))
    a.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
