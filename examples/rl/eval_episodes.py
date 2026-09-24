#!/usr/bin/env python3
"""Run a *fixed* checkpoint through the episode corpus and score it, paired.

Why this exists
===============
A training run's own reward is measured on the scenarios the policy is being
optimised on, at the training temperature, interleaved with the optimiser steps
that change the policy underneath the measurement. That is the right instrument
for "is the objective moving" and the wrong one for "is the model better". This
answers the second question and nothing else: load weights, roll out episodes,
do not backpropagate, print the table, write the JSON.

⚠ **This is not a held-out evaluation, and calling it one would be the single
most misleading thing available here.** The corpus has seven scenarios and a
policy trained on it trained on all seven; there is no unseen scenario to hold
out without retraining. What is controlled is narrower, and every output says
so:

  - the **weights are fixed** for the whole run, so no measurement is taken
    against a policy that moved during it;
  - the two checkpoints are compared **paired** -- each scenario's sampling
    stream is derived from its own name, so both columns see the same episode
    count and sampling parameters regardless of the order they ran in or how
    many processes it took, and the difference between them is the weights.

That removes the two confounds that make a training curve uninterpretable. It
does not turn the training corpus into a test set.

**Greedy decoding was the discarded alternative.** It looks like the obvious
way to remove sampling noise from a comparison, but the policy is trained and
sampled at ``temperature 0.7 / top_p 0.95``, and a greedy readout measures the
argmax of a distribution nobody uses. The noise is real and the answer to it is
more episodes, which are cheap here: there is no backward pass.

Usage
-----

    python examples/rl/eval_episodes.py --corpus-root <corpus-dir> \\
        --after <trained-checkpoint> --out <eval-dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import zlib
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Refused to run: the comparison asked for could not mean what it says.
EXIT_REFUSED = 3

# The trainer's sampling defaults, held fixed so a comparison is not measuring
# two things at once. Defaults rather than required flags so an explicit
# experiment can override them and say so; a test asserts they match the
# trainer's parser.
TRAIN_TEMPERATURE = 0.7
TRAIN_TOP_P = 0.95
TRAIN_MAX_NEW_TOKENS = 320
TRAIN_GEN_BATCH = 4
TRAIN_MAX_EPISODE_STEPS = 8


class _GenArgs:
    """The fields ``train_grpo_step.generate`` / ``episode_rollouts`` read off ``args``.

    A small namespace rather than the trainer's own parser, which requires
    ``--out`` and carries optimiser flags that mean nothing here. Constructing
    it makes the coupling visible: if ``generate`` grows a field, this fails at
    the attribute rather than quietly sampling differently.
    """

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def two_proportion_z(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Pooled two-proportion z for (b - a). 0.0 when it is undefined.

    Undefined in two cases that both occur: an unresolvable scenario
    contributes no trials, and a scenario both checkpoints always solve has
    zero pooled variance. 0.0 -- "no evidence of a difference" -- is the
    fail-closed reading; ``inf`` or ``nan`` would make a scenario nobody can
    get wrong look like the strongest result in the table.
    """
    if n_a <= 0 or n_b <= 0:
        return 0.0
    pooled = (hits_a + hits_b) / (n_a + n_b)
    var = pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)
    if var <= 0.0:
        return 0.0
    return ((hits_b / n_b) - (hits_a / n_a)) / math.sqrt(var)


def step1_counts(group: dict[str, Any]) -> tuple[int, int]:
    """(episodes that named a resolver on step 1, episodes that could have).

    Read off the event counters rather than off ``step1_resolver_rate``, which
    is None for an unresolvable scenario: multiplying a None back out by ``n``
    is how an unresolvable group ends up contributing zeros to a pooled rate it
    has no business being in.
    """
    if group.get("step1_resolver_rate") is None:
        return 0, 0
    return int(group["events"].get("resolver_named_on_step_1", 0)), int(group["n"])


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Pair two eval columns scenario by scenario, refusing an unpaired one.

    ⚠ **Refuses rather than intersects.** Comparing the scenarios two columns
    happen to share would answer a different question than the one asked --
    most damagingly when one column stopped partway and its surviving
    scenarios are the cheap ones. A caller who wants a subset says so with
    ``--scenarios`` on both sides.
    """
    a = {g["scenario_id"]: g for g in before["groups"]}
    b = {g["scenario_id"]: g for g in after["groups"]}
    for label, column, got in (("before", before, a), ("after", after, b)):
        missing = sorted(set(column["config"]["scenarios"]) - set(got))
        if missing:
            raise ValueError(
                f"the {label} column is incomplete: it asked for "
                f"{sorted(column['config']['scenarios'])} and finished {sorted(got)}, "
                f"missing {missing}. Rerun with --reuse to finish it."
            )
    if set(a) != set(b):
        raise ValueError(
            f"the two columns did not see the same scenarios: only in before "
            f"{sorted(set(a) - set(b))}, only in after {sorted(set(b) - set(a))}"
        )
    mismatched = sorted(s for s in a if a[s]["n"] != b[s]["n"])
    if mismatched:
        raise ValueError(
            f"the two columns used different episode counts on {mismatched}; "
            "a paired comparison needs the same n on both sides"
        )
    for field in ("temperature", "top_p", "max_new_tokens", "max_episode_steps", "seed"):
        if before["config"][field] != after["config"][field]:
            raise ValueError(
                f"the two columns differ in {field!r}: {before['config'][field]!r} vs "
                f"{after['config'][field]!r}. A paired comparison has to hold sampling "
                "fixed, or the difference it reports is not the weights."
            )

    rows = []
    pooled = [0, 0, 0, 0]
    for scenario in sorted(a):
        ha, na = step1_counts(a[scenario])
        hb, nb = step1_counts(b[scenario])
        pooled = [pooled[0] + ha, pooled[1] + na, pooled[2] + hb, pooled[3] + nb]
        rows.append({
            "scenario_id": scenario,
            "resolvable": na > 0,
            "step1_before": (ha / na) if na else None,
            "step1_after": (hb / nb) if nb else None,
            "step1_z": two_proportion_z(ha, na, hb, nb),
            "reward_before": a[scenario]["reward_mean"],
            "reward_after": b[scenario]["reward_mean"],
            "converged_before": a[scenario]["converged_rate"],
            "converged_after": b[scenario]["converged_rate"],
        })
    return {
        "before": before["config"]["init_from"],
        "after": after["config"]["init_from"],
        "scenarios": rows,
        "pooled_step1_before": (pooled[0] / pooled[1]) if pooled[1] else None,
        "pooled_step1_after": (pooled[2] / pooled[3]) if pooled[3] else None,
        "pooled_step1_z": two_proportion_z(*pooled),
        # The resolvable-only mean is quoted apart on purpose: an unresolvable
        # scenario can only score by reaching the unresolvable terminal, so a
        # corpus mean that folds it in moves for reasons unrelated to resolving.
        "reward_before": _mean([r["reward_before"] for r in rows]),
        "reward_after": _mean([r["reward_after"] for r in rows]),
        "resolvable_reward_before": _mean([r["reward_before"] for r in rows if r["resolvable"]]),
        "resolvable_reward_after": _mean([r["reward_after"] for r in rows if r["resolvable"]]),
    }


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def print_comparison(result: dict[str, Any]) -> None:
    print("=" * 100)
    print("paired episode evaluation -- fixed weights, no optimiser step")
    print(f"  before: {result['before']}")
    print(f"  after:  {result['after']}")
    print("=" * 100)
    print(f"  {'scenario':<24}{'step1 before':>14}{'step1 after':>13}"
          f"{'z':>8}{'reward before':>15}{'reward after':>14}")
    for row in result["scenarios"]:
        print(f"  {row['scenario_id']:<24}{_fmt(row['step1_before'], '.3f'):>14}"
              f"{_fmt(row['step1_after'], '.3f'):>13}{row['step1_z']:>+8.2f}"
              f"{row['reward_before']:>+15.3f}{row['reward_after']:>+14.3f}")
    print()
    print(f"  pooled step-1 resolver rate  {_fmt(result['pooled_step1_before'], '.3f')} -> "
          f"{_fmt(result['pooled_step1_after'], '.3f')}  z = {result['pooled_step1_z']:+.2f}")
    print(f"  mean reward, all scenarios   {_fmt(result['reward_before'], '+.3f')} -> "
          f"{_fmt(result['reward_after'], '+.3f')}")
    print(f"  mean reward, resolvable only {_fmt(result['resolvable_reward_before'], '+.3f')}"
          f" -> {_fmt(result['resolvable_reward_after'], '+.3f')}")
    print()
    print("  ⚠ Not a held-out set: the policy trained on all of these scenarios. What is\n"
          "    controlled is that the weights are fixed and the two columns are paired on\n"
          "    seed, scenario, episode count and sampling parameters.")
    if abs(result["pooled_step1_z"]) < 2.0:
        print("  ⚠ |z| < 2 on the pooled rate: this run does not distinguish the two\n"
              "    checkpoints. Report it as that, not as a tie and not as a win.")


#: The config fields a column is fully determined by. ``scenarios`` is checked
#: separately: it is derived from the corpus, and a stale column carrying a
#: different list is refused by ``compare``.
_REUSE_FIELDS = (
    "init_from", "model", "param_dtype", "episodes_per_scenario", "max_episode_steps",
    "temperature", "top_p", "max_new_tokens", "seed",
)


def column_source(checkpoint: Path | None, args: argparse.Namespace) -> str:
    """What ``config.init_from`` says for a column; ``None`` is the base model.

    One function because ``evaluate`` writes this string and ``reusable_column``
    checks it, and the two drifting apart would make every reuse a refusal or,
    worse, the wrong column reusable.
    """
    return str(checkpoint) if checkpoint else args.model


def _config(args: argparse.Namespace, source: str) -> dict[str, Any]:
    return {
        "init_from": source,
        "model": args.model,
        "param_dtype": args.param_dtype,
        "episodes_per_scenario": args.episodes_per_scenario,
        "max_episode_steps": args.max_episode_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }


def reusable_column(path: Path, source: str, args: argparse.Namespace) -> dict[str, Any]:
    """Load a column already on disk, or refuse it. Never silently recomputes.

    A two-column run at n=64 is hundreds of episodes on a GPU, and a run can be
    interrupted (preemption, a node fault) with one column done and the other
    not; without this, recovering means paying for the finished column again.

    ⚠ It **refuses** a column that does not match the current settings rather
    than recomputing it. A reused column is invisible in the output table, so a
    mismatch silently repaired would print a comparison of two different
    experiments with nothing on the page saying so.

    Reuse is sound only because of :func:`scenario_seed`: every scenario's
    sampling stream is derived from its own name, so it does not matter which
    process a scenario ran in or how many ran ahead of it.
    """
    column = json.loads(path.read_text())
    want = _config(args, source)
    differing = [
        f"{field}: on disk {column['config'].get(field)!r}, asked for {want[field]!r}"
        for field in _REUSE_FIELDS
        if column["config"].get(field) != want[field]
    ]
    if differing:
        raise ValueError(
            f"{path} was produced under different settings and cannot be reused -- "
            + "; ".join(differing) + ". Point --out somewhere else."
        )
    if args.scenarios:
        wanted = sorted(s.strip() for s in args.scenarios.split(",") if s.strip())
        if sorted(column["config"].get("scenarios", [])) != wanted:
            raise ValueError(
                f"{path} covers scenarios {sorted(column['config'].get('scenarios', []))}, "
                f"but --scenarios asked for {wanted}"
            )
    return column


def scenario_seed(base: int, scenario_id: str) -> int:
    """A sampling stream per *(seed, scenario)* rather than per column.

    The obvious design -- one ``manual_seed(args.seed)`` per column -- makes the
    stream a scenario sees depend on how many scenarios ran ahead of it *in that
    process*, so a resumed column would silently pair the control's scenario 4
    against a trained scenario 4 drawn from elsewhere in the stream. Deriving
    the seed from the scenario name makes each scenario's stream independent of
    order and of where the run was interrupted.

    ``zlib.crc32`` rather than ``hash()``, because ``hash()`` of a ``str`` is
    salted per process by ``PYTHONHASHSEED`` and the two columns routinely run
    in different processes.
    """
    return (base + zlib.crc32(scenario_id.encode())) % (2**31 - 1)


def _column_payload(
    args: argparse.Namespace,
    scenarios: list[Any],
    groups: list[dict[str, Any]],
    wire: list[dict[str, Any]],
    checkpoint: Path | None,
) -> dict[str, Any]:
    """The on-disk shape of one column.

    ``config.scenarios`` is the set the run was *asked* for and ``groups`` what
    it got through, so a partial column is self-describing and ``compare``
    refuses it rather than comparing a survivor set.
    """
    config = _config(args, column_source(checkpoint, args))
    config["scenarios"] = [s.scenario_id for s in scenarios]
    return {"config": config, "groups": groups, "wire": wire}


def evaluate(
    checkpoint: Path | None,
    args: argparse.Namespace,
    done: dict[str, Any] | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """One checkpoint, N episodes per scenario, no gradient anywhere.

    ``checkpoint=None`` is the base model, the control column. ``done`` is a
    partial column to carry forward, whose scenarios are skipped;
    ``on_progress`` is called with the column after every scenario so the unit
    of lost work on an interruption is one scenario.
    """
    # Deferred so the pure logic above imports and tests without torch.
    import episode_env
    import torch
    import train_grpo_step
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from aorta.agent.policy import AgentPolicy

    only = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenarios = episode_env.load_corpus(args.corpus_root, only=only)
    carried = list((done or {}).get("groups", []))
    carried_wire = list((done or {}).get("wire", []))
    finished = {g["scenario_id"] for g in carried}
    todo = [s for s in scenarios if s.scenario_id not in finished]
    if finished:
        print(f"[resume] {len(finished)} scenario(s) already done: {sorted(finished)}",
              flush=True)
    if not todo:
        return _column_payload(args, scenarios, carried, carried_wire, checkpoint)

    source = column_source(checkpoint, args)
    print(f"[load] {source} in {args.param_dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=getattr(torch, args.param_dtype),
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(args.device)
    model.config.use_cache = True
    model.eval()
    # Nothing here builds a graph, and a parameter that cannot require grad
    # cannot be updated by a stray optimiser someone adds later.
    for p in model.parameters():
        p.requires_grad_(False)

    policy = AgentPolicy(max_iterations=args.max_episode_steps)
    gen = _GenArgs(
        group=args.episodes_per_scenario,
        gen_batch=args.gen_batch,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        log_episodes=0,
    )
    # One scenario per `episode_rollouts` call, so the seed is set per
    # scenario and the column is persisted after each one. The batching that
    # makes episodes affordable is within a scenario's group and is untouched.
    for scenario in todo:
        torch.manual_seed(scenario_seed(args.seed, scenario.scenario_id))
        groups, _, wire = train_grpo_step.episode_rollouts(model, tok, [scenario], gen, policy)
        carried.extend(groups)
        carried_wire.extend(wire)
        if on_progress is not None:
            on_progress(_column_payload(args, scenarios, carried, carried_wire, checkpoint))
    return _column_payload(args, scenarios, carried, carried_wire, checkpoint)


def _column(path: Path, checkpoint: Path | None, label: str, args: argparse.Namespace) -> dict[str, Any]:
    """Reuse the column at ``path`` if it is there and matches, else compute it.

    Says which it did, every time: a reused column is otherwise
    indistinguishable in the output from one that cost a GPU-hour.
    """
    source = column_source(checkpoint, args)
    done = None
    if path.exists():
        done = reusable_column(path, source, args)
        covered = {g["scenario_id"] for g in done["groups"]}
        if not set(done["config"]["scenarios"]) - covered:
            print(f"[reuse] {label} column from {path} ({source})", flush=True)
            return done
        print(f"[resume] {label} column from {path}: {len(covered)} of "
              f"{len(done['config']['scenarios'])} scenarios already done", flush=True)
    else:
        print(f"[eval] {label} column", flush=True)
    column = evaluate(
        checkpoint, args, done,
        on_progress=lambda c: path.write_text(json.dumps(c, indent=2)),
    )
    path.write_text(json.dumps(column, indent=2))
    return column


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--corpus-root", default=None,
                        help="directory holding the archived matrices "
                             "(default: $AORTA_RL_CORPUS_ROOT)")
    parser.add_argument("--before", type=Path, default=None,
                        help="the control checkpoint; omit for the base model, which "
                             "is what the trainer starts from")
    parser.add_argument("--after", type=Path, required=True,
                        help="the trained checkpoint under test")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for before.json, after.json, comparison.json")
    parser.add_argument("--episodes-per-scenario", type=int, default=64,
                        help="the precision dial: the standard error on a paired "
                             "difference in a step-1 rate near 0.5 is about 0.088 at "
                             "n=64 and 0.063 at n=128")
    parser.add_argument("--max-episode-steps", type=int, default=TRAIN_MAX_EPISODE_STEPS)
    parser.add_argument("--gen-batch", type=int, default=TRAIN_GEN_BATCH)
    parser.add_argument("--temperature", type=float, default=TRAIN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TRAIN_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=TRAIN_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--param-dtype", choices=("bfloat16", "float32"), default="float32",
                        help="float32, what the trainer's checkpoints hold; bfloat16 "
                             "evaluates a cast copy, which is a different measurement")
    parser.add_argument("--seed", type=int, default=20260923,
                        help="deliberately not the trainer's default seed, so the "
                             "policy is not scored on a replay of its own rollouts' "
                             "sampling stream")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--reuse", action="store_true",
                        help="load a column already in --out instead of recomputing "
                             "it, provided its settings match; without this flag an "
                             "existing column is a refusal, not an overwrite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.before is not None and args.before.resolve() == args.after.resolve():
        # Would print a table of zeros with z = 0.00 and look like a carefully
        # measured null result. `is not None` matters: an unset --before is the
        # base-model control, which is the run this script exists for.
        print(f"--before and --after are both {args.after}; comparing a checkpoint with "
              "itself measures sampling noise, not training.", file=sys.stderr)
        return EXIT_REFUSED
    if args.episodes_per_scenario < 1:
        print("--episodes-per-scenario must be >= 1", file=sys.stderr)
        return EXIT_REFUSED

    existing = [p for p in (args.out / "before.json", args.out / "after.json") if p.exists()]
    if existing and not args.reuse:
        # Each column is GPU time, and a half-overwritten --out is worse than
        # either: the surviving column is from the old settings and nothing in
        # the printed table says so.
        print(f"{[str(p) for p in existing]} already exist. Pass --reuse to compare against "
              "them (their settings are checked), or point --out somewhere else.",
              file=sys.stderr)
        return EXIT_REFUSED

    logging.getLogger("aorta.agent.llm").setLevel(logging.ERROR)
    args.out.mkdir(parents=True, exist_ok=True)
    before = _column(args.out / "before.json", args.before, "control", args)
    after = _column(args.out / "after.json", args.after, "trained", args)

    result = compare(before, after)
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2))
    print_comparison(result)
    print(f"\nwrote {args.out}/before.json, after.json, comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
