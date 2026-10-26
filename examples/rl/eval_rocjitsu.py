#!/usr/bin/env python3
"""Held-out evaluation for the rocjitsu environment: fixed weights, no gradient.

  eval_rocjitsu.py --corpus-root <corpus> --split test --out <dir> \\
      [--before <ckpt>] [--after <ckpt>] [--column before|after|both] [--compare]

One *column* is one checkpoint rolled out over the split's scenarios
(``--before`` defaults to the base model, ``--model``). Each reply is seeded
from ``(--seed, scenario, episode, step)`` and generated on its own, as
``eval_episodes`` does, so two columns are paired episode for episode and their
difference is the weights. ``--column`` computes one column (two can run at
once on two GPUs); ``--compare`` reads both and writes ``comparison.json``.

Every report carries, on the same scenarios, the CPU baselines -- the scripted
policies and the best constant answer -- and the entrenchment numbers: how
concentrated the action sequences and the answers are.

Metrics, per column, over all / faulty / clean scenarios: mean score, solved
rate (kernel + class + exact offset, or "none" on a clean control), location
exactness (exact offset with the right kernel), near and kernel rates, tool
cost, turns, no-answer and malformed rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rocjitsu_baselines as rb  # noqa: E402
import rocjitsu_env as rj  # noqa: E402

EXIT_REFUSED = 3
COLUMNS = ("before", "after")


class _GenArgs:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def pick_scenarios(scenarios: list[rj.Scenario], max_faulty: int, max_clean: int) -> list[rj.Scenario]:
    """A deterministic spread: every k-th faulty scenario in (kernel, op, id) order, then clean ones."""
    faulty = sorted((s for s in scenarios if not s.clean), key=lambda s: (s.kernel, s.op, s.scenario_id))
    clean = [s for s in scenarios if s.clean]
    if max_faulty and max_faulty < len(faulty):
        step = len(faulty) / max_faulty
        faulty = [faulty[int(i * step)] for i in range(max_faulty)]
    if max_clean >= 0:
        clean = clean[:max_clean]
    return faulty + clean


def episode_rows(column: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for g in column["groups"] for row in g["episodes"]]


def entrenchment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    faulty = [r for r in rows if not rj.is_clean_id(r["scenario_id"])]
    answers = Counter(str((r["answer"] or {}).get("kernel") or "?") + "@" + str((r["answer"] or {}).get("offset"))
                      for r in faulty if r["answer"] and r["answer"].get("class") != "none")
    seq = rj.sequence_stats(Counter(r["sequence"] for r in rows))
    top_answer = answers.most_common(1)[0] if answers else (None, 0)
    return {
        "sequences": seq,
        "first_tool": dict(Counter(r["sequence"].split(">")[0] for r in rows).most_common(6)),
        "answers_distinct": len(answers),
        "answer_top": top_answer[0],
        "answer_top_share": round(top_answer[1] / max(sum(answers.values()), 1), 4),
        "answered_none_on_faulty": sum(1 for r in faulty if (r["answer"] or {}).get("class") == "none"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    faulty = [r for r in rows if not rj.is_clean_id(r["scenario_id"])]
    clean = [r for r in rows if rj.is_clean_id(r["scenario_id"])]
    return {"all": rb.aggregate(rows), "faulty": rb.aggregate(faulty), "clean": rb.aggregate(clean),
            "entrenchment": entrenchment(rows)}


def gen_args(args: argparse.Namespace) -> _GenArgs:
    """The sampler's settings; each one is also in :func:`column_config`, so the column records what was used."""
    return _GenArgs(group=args.episodes_per_scenario, gen_batch=args.gen_batch, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, device=args.device,
                    log_episodes=0)


#: The settings ``train_grpo_step.generate`` passes to ``model.generate``; the rest are the checkpoint's own.
EXPLICIT_GENERATION = ("do_sample", "temperature", "top_p", "top_k", "use_cache", "max_new_tokens", "pad_token_id",
                       "output_logits", "return_dict_in_generate")


def inherited_generation(settings: dict[str, Any]) -> dict[str, Any]:
    """A generation config's settings that ``generate`` does not override, without its bookkeeping fields."""
    return {k: v for k, v in sorted(settings.items())
            if k not in EXPLICIT_GENERATION and k != "transformers_version" and not k.startswith("_")}


def generation_settings(source: str) -> dict[str, Any] | None:
    """:func:`inherited_generation` of ``source``'s generation config; ``None`` if it ships none."""
    from transformers import GenerationConfig

    try:
        config = GenerationConfig.from_pretrained(source, local_files_only=True)
    except OSError:
        return None
    return inherited_generation(config.to_dict())


def column_config(args: argparse.Namespace, source: str, scenarios: list[rj.Scenario], system_sha256: str,
                  backend: Any, runtime: dict[str, Any], checkpoint: dict[str, Any] | None = None,
                  generation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything a column was sampled under, so two columns can be checked before they are paired."""
    return {"source": source, "checkpoint": checkpoint, "generation": generation,
            "seed": args.seed, "split": args.split, "episodes": args.episodes_per_scenario,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_episode_steps": args.max_episode_steps, "param_dtype": args.param_dtype,
            "seeding": "per-episode-step", "scenarios": [s.scenario_id for s in scenarios],
            "corpus": {s.scenario_id: s.digest for s in scenarios}, "blas_backend": backend,
            "system_sha256": system_sha256, "prompt_profile": rj.resolve_prompt(),
            "runtime": runtime}


#: Settings a resumed column must share with the run resuming it, besides source, seed, prompt and top_k.
#: ``checkpoint`` is by content: the trainer rewrites ``checkpoint-last`` in place.
RESUME_KEYS = ("split", "episodes", "temperature", "top_p", "max_new_tokens", "max_episode_steps", "param_dtype",
               "seeding", "prompt_profile", "scenarios", "corpus", "blas_backend", "runtime", "checkpoint",
               "generation")


def check_resumable(path: Path, recorded: dict[str, Any], source: str, seed: int, system_sha256: str,
                    top_k: int, proposed: dict[str, Any] | None = None) -> None:
    """Refuse to append to a column computed under other settings: the result would be two columns in one."""
    sha = recorded.get("system_sha256")
    if sha is not None and sha != system_sha256:
        raise SystemExit(f"{path} was computed under another system prompt ({sha[:12]}, now "
                         f"{system_sha256[:12]}); refusing to resume it")
    if recorded["source"] != source or recorded["seed"] != seed:
        raise SystemExit(f"{path} was computed from {recorded['source']} seed {recorded['seed']}; "
                         f"refusing to resume it as {source} seed {seed}")
    if recorded.get("top_k") != top_k:
        raise SystemExit(f"{path} was sampled with top_k {recorded.get('top_k')}; refusing to resume it "
                         f"with top_k {top_k}")
    differ = [k for k in RESUME_KEYS if proposed is not None and recorded.get(k) != proposed.get(k)]
    if differ:
        raise SystemExit(f"{path} was computed with other {', '.join(differ)}; refusing to resume it")


def remaining(done: dict[str, Any], scenarios: list[rj.Scenario]) -> list[rj.Scenario]:
    """The scenarios a resumed column still lacks; with any left, its old metrics are dropped."""
    finished = {g["scenario_id"] for g in done["groups"]}
    todo = [s for s in scenarios if s.scenario_id not in finished]
    if todo:
        # Until the new scenarios are in, the old metrics describe a different column.
        done.pop("metrics", None)
    return todo


def execution_setup() -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply ``$AORTA_PREFERRED_BLAS`` and return the BLAS setup and package versions to record.

    ``apply()`` returns None when the variable is unset, so the record is
    ``eval_episodes.blas_backend()`` taken afterwards: torch's version, the library
    torch reports, and every ``*BLAS*`` environment variable.
    """
    import blas_backend
    import eval_episodes

    blas_backend.apply()
    return eval_episodes.blas_backend(), eval_episodes.runtime_versions()


def run_column(checkpoint: Path | None, args: argparse.Namespace, scenarios: list[rj.Scenario],
               path: Path, system: str | None = None) -> dict[str, Any]:
    backend, runtime = execution_setup()
    import torch
    import train_grpo_step
    from eval_episodes import checkpoint_identity, episode_seed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = str(checkpoint) if checkpoint else args.model
    system = system if system is not None else rj.system_for(scenarios)
    system_sha256 = hashlib.sha256(system.encode()).hexdigest()
    config = column_config(args, source, scenarios, system_sha256, backend, runtime,
                           checkpoint=checkpoint_identity(checkpoint, args),
                           generation=generation_settings(source))
    done: dict[str, Any] = {"groups": [], "wire": []}
    if path.exists():
        if not args.reuse:
            raise SystemExit(f"{path} exists; pass --reuse to resume it or choose another --out")
        done = json.loads(path.read_text())
        check_resumable(path, done["config"], source, args.seed, system_sha256, args.top_k, proposed=config)
    todo = remaining(done, scenarios)
    print(f"[eval] {source}: {len(todo)} of {len(scenarios)} scenario(s) to run", flush=True)
    if todo:
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=getattr(torch, args.param_dtype),
                                                     local_files_only=True, attn_implementation="sdpa")
        model = model.to(args.device)
        model.config.use_cache = True
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[eval] loaded in {time.time() - t0:.0f}s", flush=True)
        gen = gen_args(args)
        policy = rj.make_policy(args.max_episode_steps)
        for scenario in todo:
            groups, _, wire = train_grpo_step.episode_rollouts(
                model, tok, [scenario], gen, policy, env=rj, system=system,
                episode_seed=lambda sid, index, step: episode_seed(args.seed, sid, index, step),
            )
            done["groups"].extend(groups)
            done["wire"].extend(wire)
            done["config"] = config
            write(path, done)
    done["config"] = config
    done["metrics"] = summarize(episode_rows(done))
    write(path, done)
    return done


def write(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)


def mcnemar(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    gained = sum(1 for a, b in pairs if b and not a)
    lost = sum(1 for a, b in pairs if a and not b)
    z = (gained - lost) / math.sqrt(gained + lost) if gained + lost else 0.0
    return {"pairs": len(pairs), "gained": gained, "lost": lost, "z": round(z, 3)}


def bootstrap_mean_diff(diffs: list[float], draws: int = 4000, seed: int = 0) -> dict[str, Any]:
    if not diffs:
        return {"n": 0}
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return {"n": len(diffs), "mean": round(statistics.mean(diffs), 4),
            "ci95": [round(means[int(0.025 * draws)], 4), round(means[int(0.975 * draws) - 1], 4)]}


def check_complete(column: dict[str, Any], name: str) -> None:
    """Refuse a column that does not hold exactly one row per configured (scenario, episode)."""
    config = column["config"]
    if column.get("incomplete") or "metrics" not in column:
        raise SystemExit(f"the {name} column is incomplete; it does not pair")
    want = {(sid, i) for sid in config["scenarios"] for i in range(config["episodes"])}
    rows = Counter((r["scenario_id"], r["index"]) for r in episode_rows(column))
    missing, extra = want - set(rows), set(rows) - want
    repeated = sorted(k for k, n in rows.items() if n > 1)
    if missing or extra or repeated:
        raise SystemExit(f"the {name} column does not hold one row per configured episode "
                         f"({len(missing)} missing, {len(extra)} not configured, {len(repeated)} repeated); "
                         "it does not pair")


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if before["config"]["scenarios"] != after["config"]["scenarios"]:
        raise SystemExit("the two columns ran different scenarios; they do not pair")
    for key in ("seed", "episodes", "temperature", "top_p", "max_new_tokens", "max_episode_steps", "corpus"):
        if before["config"][key] != after["config"][key]:
            raise SystemExit(f"the two columns differ in {key}; they do not pair")
    # Read with .get so a column without a field pairs only with another without it.
    for key in ("seeding", "param_dtype", "blas_backend", "runtime", "generation"):
        if before["config"].get(key) != after["config"].get(key):
            raise SystemExit(f"the two columns differ in {key}; they do not pair")
    tokenizers = [(c["config"].get("checkpoint") or {}).get("tokenizer") for c in (before, after)]
    if tokenizers[0] != tokenizers[1]:
        raise SystemExit("the two columns were tokenized from different trees; they do not pair")
    check_complete(before, "before")
    check_complete(after, "after")
    top_k = (before["config"].get("top_k"), after["config"].get("top_k"))
    if None not in top_k and top_k[0] != top_k[1]:
        raise SystemExit(f"the two columns were sampled with top_k {top_k[0]} and {top_k[1]}; they do not pair")
    prompts = (before["config"].get("system_sha256"), after["config"].get("system_sha256"))
    if None in prompts or prompts[0] != prompts[1]:
        raise SystemExit(f"the two columns ran under different or unrecorded system prompts {prompts}; "
                         "they do not pair")
    b_rows = {(r["scenario_id"], r["index"]): r for r in episode_rows(before)}
    a_rows = {(r["scenario_id"], r["index"]): r for r in episode_rows(after)}
    keys = sorted(set(b_rows) & set(a_rows))
    per_scenario: dict[str, list[float]] = {}
    for k in keys:
        per_scenario.setdefault(k[0], []).append(a_rows[k]["total"] - b_rows[k]["total"])
    faulty_keys = [k for k in keys if not rj.is_clean_id(k[0])]
    return {
        "episodes_paired": len(keys),
        "top_k": {"before": top_k[0], "after": top_k[1],
                  "checked": None not in top_k},
        "score_diff_by_scenario": bootstrap_mean_diff([statistics.mean(v) for v in per_scenario.values()]),
        "solved_mcnemar": mcnemar([(int(b_rows[k]["solved"]), int(a_rows[k]["solved"])) for k in keys]),
        "exact_mcnemar_faulty": mcnemar([(int(b_rows[k]["outcome"] == "exact"), int(a_rows[k]["outcome"] == "exact"))
                                         for k in faulty_keys]),
        "before": before["metrics"],
        "after": after["metrics"],
    }


def baselines_for(scenarios: list[rj.Scenario]) -> dict[str, Any]:
    res = rb.report(scenarios)
    keep = ("oracle", "evidence_window", "race_top_hit", "waitcheck_only", "random_location", "constant_best")
    return {"constant_best_answer": res["constant_best_answer"],
            "policies": {k: {p: res["policies"][k][p] for p in ("all", "faulty", "clean")}
                         for k in keep if k in res["policies"]}}


ROWS = (("mean", "score"), ("solved_rate", "solved"), ("exact_rate", "exact loc"), ("near_rate", "near loc"),
        ("kernel_rate", "kernel"), ("mean_tool_cost", "tool cost"), ("mean_turns", "turns"),
        ("no_answer_rate", "no answer"))


def print_table(title: str, columns: dict[str, dict[str, Any] | None]) -> None:
    names = list(columns)
    print(f"\n{title}")
    print(f"  {'metric':<11}" + "".join(f"{n:>17}" for n in names))
    for key, label in ROWS:
        cells = []
        for n in names:
            v = (columns[n] or {}).get(key)
            cells.append(f"{v:>17.3f}" if isinstance(v, (int, float)) else f"{'-':>17}")
        print(f"  {label:<11}" + "".join(cells))
    print(f"  {'n':<11}" + "".join(f"{(columns[n] or {}).get('n', '-'):>17}" for n in names))


def check_selection(cols: dict[str, dict[str, Any]], scenarios: list[rj.Scenario], system: str) -> None:
    """Refuse columns computed on another selection than the one the baselines are about to be scored on."""
    ids = [s.scenario_id for s in scenarios]
    corpus = {s.scenario_id: s.digest for s in scenarios}
    system_sha256 = hashlib.sha256(system.encode()).hexdigest()
    for name, col in cols.items():
        config = col["config"]
        for key, want in (("scenarios", ids), ("corpus", corpus), ("system_sha256", system_sha256)):
            if config.get(key) != want:
                raise SystemExit(f"{name}.json was computed with other {key} than this selection; "
                                 "pass the --split, --max-faulty, --max-clean and --shard it was run with")


def report(out: Path, scenarios: list[rj.Scenario], system: str | None = None) -> dict[str, Any]:
    cols = {c: json.loads((out / f"{c}.json").read_text()) for c in COLUMNS if (out / f"{c}.json").exists()}
    for name, col in cols.items():
        check_complete(col, name)
    check_selection(cols, scenarios, rj.system_for(scenarios) if system is None else system)
    base = baselines_for(scenarios) if scenarios else {"constant_best_answer": None, "policies": {}}
    result: dict[str, Any] = {"baselines": base, "columns": {c: v["metrics"] for c, v in cols.items()},
                              "sources": {c: v["config"]["source"] for c, v in cols.items()}}
    if len(cols) == 2:
        result["paired"] = compare(cols["before"], cols["after"])
    for part in ("all", "faulty", "clean"):
        table = {f"{c} model": cols[c]["metrics"][part] for c in cols}
        for name in ("evidence_window", "constant_best", "oracle"):
            if name in base["policies"]:
                table[name] = base["policies"][name][part]
        print_table(f"[{part}]", table)
    for c in cols:
        print(f"\n[entrenchment] {c}: {json.dumps(cols[c]['metrics']['entrenchment'])}")
    if "paired" in result:
        p = result["paired"]
        print(f"\n[paired] {p['episodes_paired']} episode pairs; score diff by scenario "
              f"{p['score_diff_by_scenario']}; solved {p['solved_mcnemar']}; exact {p['exact_mcnemar_faulty']}")
        tk = p["top_k"]
        print(f"[paired] top_k before {tk['before']} after {tk['after']}"
              f"{'' if tk['checked'] else ' (not recorded in both columns, so not checked here)'}")
    (out / "comparison.json").write_text(json.dumps(result, indent=1))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--split", default="test",
                        help="any split the corpus defines (v1: train, test; v2: val, test, test_reencode, ...); "
                             "the corpus loader refuses one it does not have")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--before", type=Path, default=None, help="control checkpoint; default the base model")
    parser.add_argument("--after", type=Path, default=None, help="trained checkpoint")
    parser.add_argument("--column", choices=("before", "after", "both", "none"), default="both")
    parser.add_argument("--compare", action="store_true", help="report from the columns in --out")
    parser.add_argument("--episodes-per-scenario", type=int, default=2)
    parser.add_argument("--max-faulty", type=int, default=0, help="0 = every faulty scenario of the split")
    parser.add_argument("--max-clean", type=int, default=-1, help="-1 = every clean control of the split")
    parser.add_argument("--max-episode-steps", type=int, default=rj.MAX_TURNS)
    parser.add_argument("--gen-batch", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20,
                        help="passed to generate and recorded in the column; 20 is the base model's own value")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--param-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="run only scenarios K, K+N, K+2N, ... of the picked list (0 <= K < N), so one "
                             "column can be split across devices; the N shards cover the list exactly once, "
                             "and every reply is seeded per (scenario, episode, step), so no seed moves")
    return parser


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard wants K/N, got {text!r}") from None
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"--shard {text}: need N >= 1 and 0 <= K < N")
    return k, n


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes_per_scenario < 1 or args.max_episode_steps < 1 or args.max_new_tokens < 1:
        print("--episodes-per-scenario, --max-episode-steps and --max-new-tokens must be >= 1",
              file=sys.stderr)
        return EXIT_REFUSED
    if args.max_episode_steps != rj.MAX_TURNS:
        # The system prompt states MAX_TURNS whatever the budget is.
        print(f"--max-episode-steps must be {rj.MAX_TURNS}: the system prompt states that budget", file=sys.stderr)
        return EXIT_REFUSED
    if args.column in ("after", "both") and args.after is None and not args.compare:
        if args.column == "after":
            print("--column after needs --after", file=sys.stderr)
            return EXIT_REFUSED
    if args.before is not None and args.after is not None and args.before.resolve() == args.after.resolve():
        print("--before and --after are the same checkpoint", file=sys.stderr)
        return EXIT_REFUSED
    try:
        shard = parse_shard(args.shard)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_REFUSED
    scenarios = pick_scenarios(rj.load_corpus(args.corpus_root, split=args.split), args.max_faulty,
                               args.max_clean)
    if not scenarios:
        print(f"split {args.split} selects no scenario", file=sys.stderr)
        return EXIT_REFUSED
    system = rj.system_for(scenarios)
    if shard is not None:
        scenarios = scenarios[shard[0]::shard[1]]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[eval] split {args.split}: {len(scenarios)} scenario(s), {args.episodes_per_scenario} episode(s) each",
          flush=True)
    if args.column in ("before", "both"):
        run_column(args.before, args, scenarios, args.out / "before.json", system=system)
    if args.column in ("after", "both") and args.after is not None:
        run_column(args.after, args, scenarios, args.out / "after.json", system=system)
    if args.compare or args.column == "both":
        report(args.out, scenarios, system=system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
