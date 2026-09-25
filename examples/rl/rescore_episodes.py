#!/usr/bin/env python3
"""What an episode run produced, and what a constant scores against the corpus.

Three jobs, none of which needs a GPU.

**Constants.** A constant policy emits the same reply whatever the prompt, so
its score is fully determined by the archived answer key -- no model, no
rollouts. That makes "does a constant do worse inside an episode than as a
single reply" decidable without spending anything, and it is the one place the
multi-step claim can be checked for free.

The mechanism is specific. As a single reply a constant is a complete policy:
one reply, scored. Inside an episode the same constant is handed a prompt whose
candidate list has **shrunk by the names it just spent**, so its second reply
proposes names that are no longer on the menu, ``llm._step_from_content`` drops
them all, and the loop stops on ``proposal_unresolved``. A constant cannot walk
a decision tree, and the reward is supposed to notice. The exception is a
constant whose first reply already covers the answer key -- which is exactly
the memorisation the corpus cannot rule out, and why ``always_the_cover`` is in
the list.

**A training run's wire.** ``wire.jsonl`` carries one row per *step*, with the
episode's reward and advantage and the terminal it ended on; this reduces it
per scenario and iteration to reward, depth, and the two resolver rates.

**A terminal replay.** The environment is offline and deterministic, so feeding
an episode's recorded replies back through :class:`episode_env.Episode`
reproduces it exactly; the only thing that can differ is a classification or
scoring rule that has since changed. That makes ``--replay`` an A/B on the rule
with everything else held fixed -- and, with no rule change, a check that a
recorded run is reproducible from its own wire.

Usage
-----

    python examples/rl/rescore_episodes.py --corpus-root <dir> --constants
    python examples/rl/rescore_episodes.py --corpus-root <dir> --wire <run>/wire.jsonl
    python examples/rl/rescore_episodes.py --corpus-root <dir> --wire <eval>/after.json --replay
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import episode_env as env  # noqa: E402

from aorta.agent.policy import AgentPolicy  # noqa: E402

#: ``--replay`` refused: the record was scored against different archives, so
#: a replay would report an archive change as a rule change. Distinct from 1,
#: which means the rule did move something.
EXIT_CORPUS_CHANGED = 3

#: Step events that a single reply cannot fire, or that a log-derived episode
#: cannot show. Reported as a block so a zero is visible next to a non-zero
#: rather than absent from a table.
MULTI_STEP_EVENTS = ("name_not_offered", "name_already_tried", "malformed_reply")

#: A set cover of the corpus's resolvers: three names that, proposed together
#: on step 1, name a resolver on every resolvable scenario. Hard-coded rather
#: than recomputed so what is scored is a specific fixed answer and not
#: whatever a greedy cover returns after a corpus edit; a test asserts it still
#: covers the corpus.
COVER = ["gpu_max_hw_queues_2", "pytorch_no_cuda_memory_caching", "xnack"]

#: Three names an earlier single-step policy converged to on several different
#: scenarios at once, two of them resolvers somewhere and one of them a
#: registered mitigation whose variable nothing on the measured stack reads
#: (aorta#500). Carried beside ``COVER`` because they answer different
#: questions: what memorising the answer key achieves, and what the reflex a
#: single-step run learned is worth.
LEARNED_REFLEX = ["hip_launch_blocking", "hsa_disable_cache", "tf32_off"]


def advantages(values: list[float]) -> tuple[list[float], float, float]:
    """The GRPO normaliser the trainer uses: mean-centred, over ``sd + 1e-4``.

    Duplicated rather than imported from the trainer, which imports ``torch``
    at module scope; a test asserts the two agree.
    """
    mean = sum(values) / len(values)
    sd = (
        (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        if len(values) > 1
        else 0.0
    )
    return [(v - mean) / (sd + 1e-4) for v in values], mean, sd


def _body(**kwargs: Any) -> str:
    obj: dict[str, Any] = {
        "verdict": "fail",
        "detectors": [],
        "category": "unknown",
        "hypothesis": "",
        "next_mitigations": [],
        "confidence": 0.5,
        "stop": False,
    }
    obj.update(kwargs)
    return json.dumps(obj)


def constant_policies(scenario: env.Scenario) -> dict[str, str]:
    """One reply per constant, built from what the scenario offers."""
    offered = sorted(scenario.offered)
    counts: Counter[str] = Counter()
    for sig in scenario.grid.signatures.values():
        counts.update(sig.detectors)
    commonest = [counts.most_common(1)[0][0]] if counts else []
    return {
        "say_nothing": _body(),
        "always_claim_unresolvable": _body(stop=True),
        "always_the_cover": _body(next_mitigations=COVER),
        "always_the_learned_reflex": _body(next_mitigations=LEARNED_REFLEX),
        "cite_the_commonest_detector": _body(detectors=commonest),
        "cite_every_detector": _body(detectors=sorted(counts)),
        "shotgun_everything": _body(next_mitigations=offered),
        "always_exhaust_and_stop": _body(next_mitigations=offered, stop=True),
        "always_prose": "I would check the memory allocator first.",
        "pick_the_first_name": _body(next_mitigations=offered[:1]),
    }


def run_constants(scenarios: list[env.Scenario], policy: AgentPolicy) -> dict[str, Any]:
    """Every constant through every scenario, as an episode and as one reply.

    One episode per scenario because a constant is deterministic: a group of
    identical episodes has an advantage of exactly zero and says nothing a
    single one does not. What is measured is the *return*.

    The single-reply column uses the episode's cell accounting (one baseline
    cell plus one per proposed name), so the two columns share a scale and
    their difference is the regime.
    """
    rows: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        for name, reply in constant_policies(scenario).items():
            group, samples, _wire = env.rollout_scenario(
                scenario,
                1,
                policy,
                lambda users, reply=reply: [reply] * len(users),
                advantage_fn=advantages,
            )
            single = env.score_completion_compat(
                reply,
                scenario,
                cells_added=len(set(env.runnable_names(reply, list(scenario.offered)))),
            )
            row = rows.setdefault(
                name,
                {
                    "policy": name,
                    "per_scenario": {},
                    "per_scenario_single": {},
                    "events": Counter(),
                    "terminals": Counter(),
                    "steps": 0,
                },
            )
            row["per_scenario"][scenario.scenario_id] = round(group["rewards"][0], 4)
            row["per_scenario_single"][scenario.scenario_id] = round(single.total, 4)
            row["events"].update(group["events"])
            row["terminals"].update(group["terminals"])
            row["steps"] += len(samples)
    for row in rows.values():
        values = list(row["per_scenario"].values())
        singles = list(row["per_scenario_single"].values())
        row["mean"] = round(sum(values) / len(values), 4)
        row["mean_single_step"] = round(sum(singles) / len(singles), 4)
        row["delta"] = round(row["mean"] - row["mean_single_step"], 4)
        row["events"] = dict(row["events"])
        row["terminals"] = dict(row["terminals"])
        row["mean_steps"] = round(row["steps"] / max(len(row["per_scenario"]), 1), 3)
    return rows


# ---------------------------------------------------------------------------
# a run's wire
# ---------------------------------------------------------------------------


def read_wire(path: Path) -> list[dict[str, Any]]:
    """Rows from a trainer ``wire.jsonl`` or an ``eval_episodes`` column JSON.

    An eval column carries its rows under ``wire`` and no ``iteration`` key;
    those rows are given iteration 0 so one reader serves both.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        doc = json.loads(text)
        if not isinstance(doc, dict) or not isinstance(doc.get("wire"), list):
            raise ValueError(f"{path}: a .json input must be an eval column with a 'wire' list")
        return [{"iteration": 0, **row} for row in doc["wire"]]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def recorded_corpus(path: Path) -> dict[str, str] | None:
    """The per-scenario archive digests a record was scored against, if it says.

    An eval column carries them as ``config.corpus``; a trainer run as
    ``corpus`` in the ``train-log.json`` beside its wire. ``None`` means the
    record predates the field, which is not the same as agreeing.
    """
    if path.suffix == ".json":
        corpus = json.loads(path.read_text(encoding="utf-8")).get("config", {}).get("corpus")
    else:
        log = path.with_name("train-log.json")
        corpus = json.loads(log.read_text(encoding="utf-8")).get("corpus") if log.is_file() else None
    return corpus if isinstance(corpus, dict) else None


def corpus_mismatch(
    recorded: dict[str, str], rows: list[dict[str, Any]], scenarios: list[env.Scenario]
) -> list[str]:
    """Scenarios the record names whose archive is not the one it was scored against.

    Same scenario ID, different contents: a replay would then report what the
    archive changed as if the rule had, so this is checked before replaying.
    A scenario the record names but did not digest counts as a mismatch too.
    """
    current = {s.scenario_id: s.digest for s in scenarios}
    named = {r["scenario_id"] for r in rows}
    return sorted(sid for sid in named if recorded.get(sid) is None
                  or recorded.get(sid) != current.get(sid))


def _episodes(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("iteration", 0), row["scenario_id"], row["episode"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: r["step"])
    return grouped


def summarise(rows: list[dict[str, Any]], scenarios: list[env.Scenario]) -> dict[str, Any]:
    """Per scenario and iteration: reward, depth, step-1 and episode resolver rate."""
    by_id = {s.scenario_id: s for s in scenarios}
    per: dict[tuple[str, int], list[list[dict[str, Any]]]] = defaultdict(list)
    for (iteration, scenario_id, _), steps in sorted(_episodes(rows).items()):
        per[(scenario_id, iteration)].append(steps)

    iterations = sorted({i for _, i in per})
    out: dict[str, Any] = {"iterations": iterations, "scenarios": {}}
    for scenario_id in sorted({s for s, _ in per}):
        scenario = by_id.get(scenario_id)
        want = set(scenario.resolvers) if scenario else set()
        menu = list(scenario.offered) if scenario else []

        def hits(raw: str, menu: list[str] = menu, want: set[str] = want) -> bool:
            return bool(set(env.runnable_names(raw, menu)) & want)

        series = []
        for iteration in iterations:
            episodes = per.get((scenario_id, iteration), [])
            if not episodes:
                continue
            rewards = [steps[0]["reward"] for steps in episodes]
            series.append(
                {
                    "iteration": iteration,
                    "episodes": len(episodes),
                    "mean_reward": round(sum(rewards) / len(rewards), 4),
                    "mean_steps": round(sum(len(s) for s in episodes) / len(episodes), 3),
                    # None, not 0.0, where there is no resolver to name.
                    "step1_resolver_rate": (
                        round(sum(hits(s[0]["raw"]) for s in episodes) / len(episodes), 4)
                        if want
                        else None
                    ),
                    "episode_resolver_rate": (
                        round(
                            sum(any(hits(r["raw"]) for r in s) for s in episodes)
                            / len(episodes),
                            4,
                        )
                        if want
                        else None
                    ),
                    "terminals": dict(Counter(s[0].get("terminal") for s in episodes)),
                }
            )
        out["scenarios"][scenario_id] = series
    return out


def event_table(train_log: dict[str, Any]) -> dict[str, Any]:
    """Every event fired, per iteration, from the trainer's own record.

    Counted from ``train-log.json`` rather than re-scored: re-scoring answers
    "what would this scorer say now", and the question is "what did the run
    score".
    """
    per_iteration: list[dict[str, Any]] = []
    for row in train_log.get("iterations", []):
        events: Counter[str] = Counter()
        terminals: Counter[str] = Counter()
        for group in row.get("groups", []):
            events.update(group.get("events") or {})
            terminals.update(group.get("terminals") or {})
        per_iteration.append(
            {
                "iteration": row.get("iteration"),
                "events": dict(events),
                "terminals": dict(terminals),
                "steps_total": row.get("steps_total"),
            }
        )
    totals: Counter[str] = Counter()
    for row in per_iteration:
        totals.update(row["events"])
    return {"per_iteration": per_iteration, "totals": dict(totals)}


def replay(
    rows: list[dict[str, Any]], scenarios: list[env.Scenario], policy: AgentPolicy
) -> dict[str, Any]:
    """Re-drive every recorded episode and compare it with the record.

    Refuses rather than skips on a scenario it cannot find: a replay that
    silently dropped the scenarios it did not know would report agreement on a
    subset as agreement. And refuses an empty wire, for the same reason: zero
    episodes replayed is zero mismatches, and "nothing disagreed" is not
    "reproducible".
    """
    if not rows:
        raise ValueError("the wire holds no rows: there is nothing to replay")
    by_id = {s.scenario_id: s for s in scenarios}
    missing = sorted({r["scenario_id"] for r in rows} - set(by_id))
    if missing:
        raise ValueError(f"the wire names scenario(s) {missing} that are not in the corpus")

    moves: Counter[tuple[str, str]] = Counter()
    per_scenario: dict[str, dict[str, Any]] = {}
    mismatched_steps = 0
    inconsistent = 0
    max_reward_diff = 0.0
    for (_it, scenario_id, _ep), group in sorted(_episodes(rows).items()):
        scenario = by_id[scenario_id]
        episode = env.Episode(scenario=scenario, index=0, policy=policy)
        for row in group:
            prompt = episode.observe()
            if prompt is None:
                break
            episode.pending_prompt = prompt
            episode.act(row["raw"])
        episode.observe()
        # The replay must consume exactly the recorded replies. A mismatch
        # means the driver no longer walks the same path, which makes every
        # number below a comparison of two different runs.
        mismatched_steps += int(len(episode.steps) != len(group))

        total = env.score(episode).total
        recorded = float(group[0]["reward"])
        # Every row carries a copy of the episode's reward and terminal, and
        # each copy is compared: checking only row 0 would read a record
        # whose later rows were edited or truncated as reproducing exactly.
        # A non-finite reward on either side is a mismatch, not a match:
        # `abs(nan - x)` is NaN, and `max(0.0, nan)` keeps 0.0, so a damaged
        # record would otherwise read as reproducing exactly.
        for row in group:
            difference = abs(total - float(row["reward"]))
            max_reward_diff = max(
                max_reward_diff, difference if math.isfinite(difference) else math.inf
            )
        copies = {(str(row.get("reward")), row.get("terminal"), row.get("episode_steps"))
                  for row in group}
        inconsistent += int(len(copies) > 1)
        was, now = group[0].get("terminal", "?"), episode.terminal
        moves[(was, now)] += 1
        bucket = per_scenario.setdefault(
            scenario_id, {"episodes": 0, "recorded": 0.0, "replay": 0.0, "moved": 0}
        )
        bucket["episodes"] += 1
        bucket["recorded"] += recorded
        bucket["replay"] += total
        bucket["moved"] += int(was != now)

    for bucket in per_scenario.values():
        n = bucket["episodes"] or 1
        bucket["mean_recorded"] = bucket.pop("recorded") / n
        bucket["mean_replay"] = bucket.pop("replay") / n
        bucket["delta"] = bucket["mean_replay"] - bucket["mean_recorded"]
    return {
        "episodes": sum(b["episodes"] for b in per_scenario.values()),
        "mismatched_steps": mismatched_steps,
        "inconsistent_records": inconsistent,
        "terminals_moved": sum(n for (a, b), n in moves.items() if a != b),
        "max_reward_diff": max_reward_diff,
        "moves": {f"{a} -> {b}": n for (a, b), n in sorted(moves.items())},
        "per_scenario": per_scenario,
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------


def print_constants(rows: dict[str, Any]) -> None:
    print("=" * 100)
    print("constant policies: the same reply, scored as one reply and as an episode")
    print("=" * 100)
    print(f"  {'policy':<30} {'one reply':>10} {'episode':>10} {'delta':>9} "
          f"{'steps/ep':>9}  terminals")
    for name, row in sorted(rows.items(), key=lambda kv: -kv[1]["mean"]):
        terminals = ", ".join(f"{k}:{v}" for k, v in sorted(row["terminals"].items()))
        print(f"  {name:<30} {row['mean_single_step']:>+10.4f} {row['mean']:>+10.4f} "
              f"{row['delta']:>+9.4f} {row['mean_steps']:>9.2f}  {terminals}")


def print_wire(summary: dict[str, Any]) -> None:
    print("=" * 100)
    print("per scenario, by iteration")
    print("=" * 100)
    for scenario_id, series in summary["scenarios"].items():
        print(f"\n  {scenario_id}")
        print(f"    {'it':>3} {'reward':>9} {'steps/ep':>9} {'step1 hit':>10} "
              f"{'episode hit':>12}  terminals")
        for row in series:
            s1 = "-" if row["step1_resolver_rate"] is None else f"{row['step1_resolver_rate']:.3f}"
            ep = "-" if row["episode_resolver_rate"] is None else f"{row['episode_resolver_rate']:.3f}"
            terminals = ", ".join(f"{k}:{v}" for k, v in sorted(row["terminals"].items()))
            print(f"    {row['iteration']:>3} {row['mean_reward']:>+9.4f} "
                  f"{row['mean_steps']:>9.2f} {s1:>10} {ep:>12}  {terminals}")
    print("\n  ⚠ 'episode hit' is not comparable with 'step1 hit': an episode gets up to")
    print("    eight draws from a shrinking menu, so it is high by arithmetic.")


def print_events(table: dict[str, Any]) -> None:
    print()
    print("events fired (the multi-step-only ones are marked)")
    totals = table["totals"]
    for name in sorted(totals, key=lambda n: -totals[n]):
        mark = "  <-- multi-step only" if name in MULTI_STEP_EVENTS else ""
        print(f"  {name:<34} {totals[name]:>8}{mark}")
    for name in MULTI_STEP_EVENTS:
        if name not in totals:
            print(f"  {name:<34} {0:>8}  <-- never fired")


def print_replay(result: dict[str, Any]) -> None:
    print(f"\n== replay: {result['episodes']} episodes, "
          f"max |replayed - recorded reward| = {result['max_reward_diff']:.3g}")
    if result["mismatched_steps"]:
        print(f"  ⚠ {result['mismatched_steps']} episode(s) replayed a different number "
              "of steps -- the comparison is NOT like for like")
    if result.get("inconsistent_records"):
        print(f"  ⚠ {result['inconsistent_records']} episode(s) whose rows disagree with "
              "each other on the reward, terminal or step count: the record is damaged")
    for move, n in result["moves"].items():
        was, now = move.split(" -> ")
        print(f"  {'   ' if was == now else ' * '}{move:<50} {n:>4}")
    print(f"\n  {'scenario':<24}{'eps':>5}{'moved':>7}{'recorded':>11}{'replay':>10}{'delta':>9}")
    for name, b in sorted(result["per_scenario"].items()):
        print(f"  {name:<24}{b['episodes']:>5}{b['moved']:>7}"
              f"{b['mean_recorded']:>11.4f}{b['mean_replay']:>10.4f}{b['delta']:>+9.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", default=None,
                        help=f"directory holding the archived matrices "
                             f"(default: ${env.CORPUS_ROOT_ENV})")
    parser.add_argument("--constants", action="store_true")
    parser.add_argument("--wire", type=Path, default=None,
                        help="a trainer wire.jsonl or an eval_episodes column JSON")
    parser.add_argument("--replay", action="store_true",
                        help="re-drive every recorded episode through the current "
                             "environment and scorer and report what moved")
    parser.add_argument("--max-episode-steps", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    if not (args.constants or args.wire):
        parser.print_help()
        return 2
    # The candidate filter logs a warning per dropped name, which is right for
    # an operator and thousands of lines here; every drop is already in the
    # episode's own record.
    logging.getLogger("aorta.agent.llm").setLevel(logging.ERROR)

    scenarios = env.load_corpus(args.corpus_root)
    policy = AgentPolicy(max_iterations=args.max_episode_steps)
    result: dict[str, Any] = {}
    status = 0

    if args.constants:
        result["constants"] = run_constants(scenarios, policy)
        print_constants(result["constants"])

    if args.wire:
        rows = read_wire(args.wire)
        if args.replay:
            recorded = recorded_corpus(args.wire)
            if recorded is None:
                print("  ⚠ the record carries no corpus digests, so the archives it was "
                      "scored against cannot be checked: a difference below may be the "
                      "archive rather than the rule", file=sys.stderr)
            else:
                stale = corpus_mismatch(recorded, rows, scenarios)
                if stale:
                    print(f"[refused] the corpus differs from the one the record was scored "
                          f"against for {stale}: a replay would report the archive change "
                          "as a rule change", file=sys.stderr)
                    return EXIT_CORPUS_CHANGED
            result["replay"] = replay(rows, scenarios, policy)
            result["replay"]["corpus_verified"] = recorded is not None
            print_replay(result["replay"])
            # Non-zero when the record does not reproduce, so a script can
            # tell "the rule moved something" from "nothing moved".
            # Any difference counts: a reclassified terminal with the same
            # step count and the same total is still a record that did not
            # reproduce, and the terminal is what the next reader acts on.
            replayed = result["replay"]
            if (replayed["mismatched_steps"] or replayed["terminals_moved"]
                    or replayed["inconsistent_records"] or replayed["max_reward_diff"] > 1e-9):
                status = 1
        else:
            result["wire"] = summarise(rows, scenarios)
            print_wire(result["wire"])
            train_log = args.wire.with_name("train-log.json")
            if train_log.is_file():
                result["events"] = event_table(json.loads(train_log.read_text()))
                print_events(result["events"])

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return status


if __name__ == "__main__":
    sys.exit(main())
