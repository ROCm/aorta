#!/usr/bin/env python3
"""Scripted and constant policies for the rocjitsu environment, CPU only.

  rocjitsu_baselines.py --corpus-root <corpus-dir> --split test --out <json>

Every policy plays through :class:`rocjitsu_env.Episode` by emitting the same
JSON text a model would, so a baseline and a model are scored by one parser,
one renderer and one reward. The scripted policies read the structured result
of their last tool call (``Episode.last_result``), which is the information the
rendered text carries.

Policies (the prototype's ``baselines.py``, re-expressed):
  oracle             the injected truth, no tools (the ceiling, 7.0)
  constant_best      the one fixed answer scoring best on the split
  random_location    a random kernel, class and instruction
  waitcheck_only     waitcheck; its first diagnostic's slot, else "none"
  race_top_hit       run_sim(all); the slot before the first report's later PC
  evidence_window    waitcheck, then run_sim(all) if it was silent; the sync
                     instruction nearest the consumer inside the evidence window

The listing a scripted policy reads is requested as a window around the
evidence (``disasm`` with ``start``/``end``), since a whole listing can exceed
the per-call limit.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rocjitsu_env as rj  # noqa: E402

SYNC = ("s_waitcnt", "s_barrier", "s_nop")
Policy = Callable[[rj.Episode, random.Random], None]


def _act(ep: rj.Episode, action: dict[str, Any]) -> None:
    ep.act(json.dumps(action))


def answer(ep: rj.Episode, cls: str, kernel: str | None = None, offset: int | None = None) -> None:
    ans: dict[str, Any] = {"class": cls}
    if cls != "none":
        ans.update(kernel=kernel, offset=rj.hexoff(offset) if offset is not None else None)
    _act(ep, {"answer": ans})


def window(ep: rj.Episode, kernel: str, lo: int, hi: int) -> list[tuple[int, str]]:
    """The listing of ``[lo - 0x40, hi]``: enough for every choice below.

    A range longer than one call's cap comes back cut short at its end, which is
    where every choice below looks, so the rest is fetched a page at a time while
    a turn is left for the answer.
    """
    start, lines = max(0, lo - 0x40), []
    while True:
        _act(ep, {"tool": "disasm", "kernel": kernel, "start": rj.hexoff(start), "end": rj.hexoff(hi)})
        page = ep.last_result
        lines += page["lines"]
        if not page.get("omitted") or not page["lines"] or ep.budget - ep.turns_used <= 1:
            return lines
        start = page["lines"][-1][0] + 1


def pick_in_window(lst: Sequence[tuple[int, str]], lo: int, hi: int) -> int:
    cands = [(o, t) for o, t in lst if lo <= o <= hi and t.startswith(SYNC)]
    if cands:
        return max(cands)[0]
    before = [o for o, _ in lst if o < hi]
    return before[-1] if before else hi


def p_oracle(ep: rj.Episode, rng: random.Random) -> None:
    s = ep.scenario
    if s.clean:
        answer(ep, "none")
    else:
        answer(ep, s.family, s.kernel, s.offsets[0] if s.offsets else None)


def make_constant(ans: dict[str, Any]) -> Policy:
    def p(ep: rj.Episode, rng: random.Random) -> None:
        answer(ep, ans["class"], ans.get("kernel"), ans.get("offset"))
    return p


def p_random(ep: rj.Episode, rng: random.Random) -> None:
    kernel = rng.choice(ep.kernels)
    offs = ep.scenario.corpus.meta["kernel_insn_offsets"][kernel]
    answer(ep, rng.choice(["barrier", "waitcnt"]), kernel, rng.choice(offs))


def _from_waitcheck(ep: rj.Episode) -> bool:
    _act(ep, {"tool": "waitcheck"})
    diags = ep.last_result
    if not diags:
        return False
    d = diags[0]
    lo, hi = sorted((d["producer_offset"], d["consumer_offset"]))
    off = pick_in_window(window(ep, d["kernel"], lo, hi), lo, hi)
    answer(ep, "waitcnt", d["kernel"], off)
    return True


def _first_sim_evidence(ep: rj.Episode) -> tuple[str, str, int, int] | None:
    _act(ep, {"tool": "run_sim", "kernel": "all"})
    for kr in ep.last_result:
        if kr["races"]:
            r = kr["races"][0]
            offs = [m["offset"] for m in r["instructions"] if m["offset"] is not None]
            if offs:
                return kr["kernel"], ("barrier" if r["type"] == "LDS" else "waitcnt"), min(offs), max(offs)
        if kr["memory_wait"]:
            m = kr["memory_wait"][0]
            lo, hi = sorted((m["producer_offset"], m["consumer_offset"]))
            return kr["kernel"], "waitcnt", lo, hi
    return None


def p_waitcheck_only(ep: rj.Episode, rng: random.Random) -> None:
    if not _from_waitcheck(ep):
        answer(ep, "none")


def p_race_top_hit(ep: rj.Episode, rng: random.Random) -> None:
    ev = _first_sim_evidence(ep)
    if not ev:
        answer(ep, "none")
        return
    kernel, cls, lo, hi = ev
    before = [o for o, _ in window(ep, kernel, lo, hi) if o < hi]
    answer(ep, cls, kernel, before[-1] if before else hi)


def p_evidence_window(ep: rj.Episode, rng: random.Random) -> None:
    if _from_waitcheck(ep):
        return
    ev = _first_sim_evidence(ep)
    if not ev:
        answer(ep, "none")
        return
    kernel, cls, lo, hi = ev
    answer(ep, cls, kernel, pick_in_window(window(ep, kernel, lo, hi), lo, hi))


SCRIPTED: dict[str, Policy] = {
    "oracle": p_oracle,
    "random_location": p_random,
    "waitcheck_only": p_waitcheck_only,
    "race_top_hit": p_race_top_hit,
    "evidence_window": p_evidence_window,
}


def _scored(ans: dict[str, Any], s: rj.Scenario) -> float:
    ep = rj.Episode(s)
    make_constant(ans)(ep, random.Random(0))
    return ep.score().total


def _location_offsets(s: rj.Scenario) -> set[int]:
    """The offsets worth trying on ``s``: its accepted offsets, and each instruction of the kernel as built
    within :data:`rocjitsu_env.NEAR_INSTRUCTIONS` of one of them.

    An instruction only the served listing has can earn nothing more: :meth:`rocjitsu_env.Episode._near`
    places it at the nearest built instruction, which earns the same near credit on every scenario and is
    a candidate here, and it can be exact only by being an accepted offset."""
    insns = s.insn_offsets
    out = set(s.offsets)
    for t in s.offsets:
        if t in insns:
            i = insns.index(t)
            out |= set(insns[max(0, i - rj.NEAR_INSTRUCTIONS):i + rj.NEAR_INSTRUCTIONS + 1])
    return out


def best_constant(scenarios: Sequence[rj.Scenario]) -> tuple[dict[str, Any], float]:
    """The single answer with the best mean over ``scenarios``, searched exhaustively.

    Candidates: "none", and every (kernel, class, offset) for a kernel a faulty scenario uses, any answer
    class, and an offset from :func:`_location_offsets` of a scenario on that kernel (no offset if there
    is none). No other offset can do better: location credit is never negative, so any other offset, and
    no offset, score no more than an accepted offset does. Every candidate is scored by playing it through
    :class:`rocjitsu_env.Episode`. Ties keep the first: "none", then kernels, classes and offsets in order.
    """
    if not scenarios:
        raise ValueError("no scenarios")
    classes = [c for c in (rj.CLASSES_V2 if any(s.layout == "v2" for s in scenarios) else rj.CLASSES) if c != "none"]
    best: dict[str, Any] = {"class": "none", "kernel": None, "offset": None}
    best_v = sum(_scored(best, s) for s in scenarios) / len(scenarios)
    by_kernel: dict[str, list[rj.Scenario]] = {}
    for s in scenarios:
        if not s.clean:
            by_kernel.setdefault(s.kernel, []).append(s)
    for kernel in sorted(by_kernel):
        on = by_kernel[kernel]
        # Off this kernel an answer naming it scores the same whatever its class and offset
        # (kernel_wrong on a faulty scenario, false_alarm on a clean one).
        off_kernel = sum(_scored({"class": classes[0], "kernel": kernel, "offset": None}, s)
                         for s in scenarios if s.clean or s.kernel != kernel)
        offsets = sorted(set().union(*(_location_offsets(s) for s in on)))
        for cls in classes:
            for off in offsets or [None]:
                ans = {"class": cls, "kernel": kernel, "offset": off}
                mean = (off_kernel + sum(_scored(ans, s) for s in on)) / len(scenarios)
                if mean > best_v:
                    best, best_v = ans, mean
    return best, best_v


def play(policy: Policy, scenarios: Sequence[rj.Scenario], seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for i, s in enumerate(scenarios):
        ep = rj.Episode(s, index=i)
        policy(ep, rng)
        out.append(ep.summary())
    return out


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The metrics every report uses, from per-episode summaries."""
    if not rows:
        return None
    n = len(rows)
    faulty = [r for r in rows if not rj.is_clean_id(r["scenario_id"])]
    return {
        "n": n,
        "mean": round(statistics.mean(r["total"] for r in rows), 4),
        "solved_rate": round(sum(r["solved"] for r in rows) / n, 4),
        "exact_rate": round(sum(r["outcome"] == "exact" for r in faulty) / max(len(faulty), 1), 4),
        "near_rate": round(sum(r["outcome"] == "near" for r in faulty) / max(len(faulty), 1), 4),
        "kernel_rate": round(sum(r["outcome"] in ("exact", "near", "kernel_only") for r in faulty)
                             / max(len(faulty), 1), 4),
        "no_answer_rate": round(sum(r["outcome"] == "no_answer" for r in rows) / n, 4),
        "mean_tool_cost": round(statistics.mean(r["tool_cost"] for r in rows), 4),
        "mean_turns": round(statistics.mean(r["steps"] for r in rows), 3),
    }


def report(scenarios: Sequence[rj.Scenario], seed: int = 0) -> dict[str, Any]:
    const, const_v = best_constant(scenarios)
    # The scripted policies read v1-shaped tool results; corpus-v2 has its own in rjenv2/baselines2.
    scripted = {"oracle": p_oracle} if any(s.layout == "v2" for s in scenarios) else SCRIPTED
    policies = dict(scripted, constant_best=make_constant(const))
    out: dict[str, Any] = {"n_scenarios": len(scenarios), "constant_best_answer": const,
                           "constant_best_mean": round(const_v, 4), "policies": {}}
    for name, pol in policies.items():
        rows = play(pol, scenarios, seed)
        out["policies"][name] = {
            "all": aggregate(rows),
            "faulty": aggregate([r for r in rows if not rj.is_clean_id(r["scenario_id"])]),
            "clean": aggregate([r for r in rows if rj.is_clean_id(r["scenario_id"])]),
            "per_scenario": [{"sid": r["scenario_id"], "total": r["total"], "outcome": r["outcome"]}
                             for r in rows],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    scenarios = rj.load_corpus(args.corpus_root, split=None if args.split == "all" else args.split)
    res = report(scenarios, args.seed)
    for name, r in res["policies"].items():
        print(f"{name:16s} {r['all']}")
    print("constant_best answer:", res["constant_best_answer"], res["constant_best_mean"])
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
