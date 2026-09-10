#!/usr/bin/env python3
"""Measure how the served engine treats temperature and seed, before trusting it.

Three things the rollout depends on, none of which should be assumed:

**Does the engine honour a seed, and by what name?** The OpenAI-standard
`seed=` is one candidate; TokenSpeed's SGLang-compat layer shims slime's
`sampling_seed` onto its own `seed`, which is reachable through `extra_body`.
An engine that silently *ignores* an unknown key is the dangerous case: the run
looks seeded, is not, and is unreproducible in a way nothing reports. So the
test is behavioural -- same seed twice must match, different seeds must diverge
-- rather than a check that the request was accepted.

**Does raising the temperature break the format gate?** §2 of the run report
concluded the gate is enforced by grammar-constrained decoding rather than by
the model, which predicts that JSON validity survives any temperature: token
choice varies *within* the grammar. If tier 0 or tier 1 failures appear as
temperature rises, that conclusion is wrong and the gate is weaker than stated.

**Does temperature actually produce diversity here?** A group of identical
completions is the defect being fixed, so the fix has to be observed working
rather than inferred from the flag being passed.

Usage
-----

    python examples/rl/probe_seed.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model openai/Qwen/Qwen3-8B \
        --temperatures 0 0.6 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proposal_reward import Proposal, score_proposal  # noqa: E402

# A prompt shaped like the real one: the proposer's contract, so the grammar
# path exercised here is the grammar path the rollout uses.
SYSTEM = (
    "You are an AORTA probe agent. Propose ONLY registered mitigation "
    "names from the candidate list. Never propose shell commands or argv. "
    "Return strict JSON with keys: category, hypothesis, next_mitigations "
    "(list of strings), confidence (0-1), stop (bool)."
)
USER = json.dumps(
    {
        "symptom": "Sanitizer run 'consan-racy' on gfx950 returned verdict 'fail'. "
                   "Per-sanitizer: consan=fail (64 findings).",
        "candidates": ["hip_launch_blocking", "amd_log_level_4"],
        "tried": ["hsa_no_sdma"],
    },
    indent=2,
)
CANDIDATES = ["hsa_no_sdma", "hip_launch_blocking", "amd_log_level_4", "none"]
TRIED = ["hsa_no_sdma"]


def call(
    model: str,
    *,
    temperature: float | None,
    seed: int | None,
    seed_mode: str,
) -> dict[str, Any]:
    import litellm

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        if seed_mode == "top_level":
            kwargs["seed"] = seed
        else:
            kwargs["extra_body"] = {seed_mode: seed}
    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result
        return {"error": f"{type(exc).__name__}: {exc}", "content": ""}
    return {"error": "", "content": response.choices[0].message.content or ""}


def graded(content: str) -> dict[str, Any]:
    score = score_proposal(Proposal("probe", content, CANDIDATES, TRIED))
    return {"tier": score.tier, "reward": round(score.reward, 4)}


def probe_seed_modes(model: str, temperature: float) -> dict[str, Any]:
    """Find a seed key the engine actually honours, behaviourally.

    Accepted-but-ignored is the failure this is built to catch, so a mode only
    counts as working when the same seed reproduces *and* a different seed
    diverges. Reproducing alone is what an ignored seed looks like at
    temperature 0, and diverging alone is what it looks like at temperature > 0.
    """
    out: dict[str, Any] = {}
    for mode in ("top_level", "sampling_seed", "seed"):
        same_a = call(model, temperature=temperature, seed=1234, seed_mode=mode)
        same_b = call(model, temperature=temperature, seed=1234, seed_mode=mode)
        diff = call(model, temperature=temperature, seed=9999, seed_mode=mode)
        errors = [r["error"] for r in (same_a, same_b, diff) if r["error"]]
        reproducible = (
            not errors and same_a["content"] == same_b["content"]
        )
        diverges = not errors and same_a["content"] != diff["content"]
        out[mode] = {
            "accepted": not errors,
            "error": errors[0] if errors else "",
            "same_seed_reproduces": reproducible,
            "different_seed_diverges": diverges,
            "honoured": bool(reproducible and diverges),
        }
    return out


def probe_temperature(model: str, temperature: float, draws: int) -> dict[str, Any]:
    """Unseeded draws at one temperature: diversity, and format survival."""
    results = [
        call(model, temperature=temperature, seed=None, seed_mode="top_level")
        for _ in range(draws)
    ]
    contents = [r["content"] for r in results]
    scores = [graded(c) for c in contents]
    tiers = [s["tier"] for s in scores]
    return {
        "temperature": temperature,
        "draws": draws,
        "errors": [r["error"] for r in results if r["error"]],
        "distinct": len(set(contents)),
        "tiers": tiers,
        "min_tier": min(tiers) if tiers else None,
        "format_gate_pass": sum(1 for t in tiers if t >= 3),
        "parsed": sum(1 for t in tiers if t >= 1),
        "rewards": [s["reward"] for s in scores],
        "spread": round(max(s["reward"] for s in scores)
                        - min(s["reward"] for s in scores), 4) if scores else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.0, 0.6, 1.0])
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed-probe-temperature", type=float, default=1.0,
                        help="seed behaviour is only observable where sampling "
                             "is non-trivial, so probe it away from 0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key

    report: dict[str, Any] = {"model": args.model, "base_url": args.base_url}

    print("=" * 72)
    print("Temperature: does diversity appear, and does the format gate hold?")
    print("=" * 72)
    report["temperatures"] = []
    for temperature in args.temperatures:
        row = probe_temperature(args.model, temperature, args.draws)
        report["temperatures"].append(row)
        print(f"  t={temperature:<4g} distinct {row['distinct']}/{row['draws']}  "
              f"tiers {row['tiers']}  reward spread {row['spread']:.4f}  "
              f"parsed {row['parsed']}/{row['draws']}  "
              f"gate {row['format_gate_pass']}/{row['draws']}")
        for error in row["errors"][:2]:
            print(f"         error: {error[:140]}")

    print()
    print("=" * 72)
    print(f"Seed: which key does the engine honour? (t={args.seed_probe_temperature:g})")
    print("=" * 72)
    report["seed_modes"] = probe_seed_modes(
        args.model, args.seed_probe_temperature
    )
    for mode, row in report["seed_modes"].items():
        verdict = "HONOURED" if row["honoured"] else "no"
        print(f"  {mode:<16} {verdict:<9} accepted={row['accepted']} "
              f"same-seed-reproduces={row['same_seed_reproduces']} "
              f"different-seed-diverges={row['different_seed_diverges']}")
        if row["error"]:
            print(f"         error: {row['error'][:140]}")

    honoured = [m for m, r in report["seed_modes"].items() if r["honoured"]]
    print()
    if honoured:
        print(f"  use --seed-mode {honoured[0]}")
    else:
        print("  no seed key is honoured; rollouts will be diverse but not "
              "reproducible. Say so in the write-up.")
    report["honoured_modes"] = honoured

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
