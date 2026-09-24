#!/usr/bin/env python3
"""Measure how the served engine treats temperature and seed, before trusting it.

Three things the rollout depends on, none of which should be assumed:

**Does the engine honour a seed, and by what name?** The OpenAI-standard
`seed=` is one candidate; TokenSpeed's SGLang-compat layer shims slime's
`sampling_seed` onto its own `seed`, which is reachable through `extra_body`.
An engine that silently *ignores* an unknown key is the dangerous case: the run
looks seeded, is not, and is unreproducible in a way nothing reports. So the
test is behavioural -- a seed must replay across repeated draws and two seeds
must diverge -- rather than a check that the request was accepted.

Behavioural is not the same as conclusive, and this probe reports a third
outcome rather than pretending otherwise. An engine that ignores the seed and
happens to repeat itself replays perfectly; an engine that honours the seed can
still hand two seeds the same completion when the grammar leaves little to
choose. Both look like the answer they are not. What separates them is whether
the engine samples *at all* at this temperature, which is measurable, so the
seed probe draws its own unseeded control and refuses to conclude where the
control says the question is unanswerable. See :func:`seed_verdict`.

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


#: The three things this probe can conclude about a seed key. A bool could
#: only carry two of them, and the missing one is the honest answer in two of
#: the four observable combinations -- see :func:`seed_verdict`.
HONOURED = "honoured"
IGNORED = "ignored"
INCONCLUSIVE = "inconclusive"

#: Two arbitrary, different seeds. Their values do not matter; that they differ
#: is the whole experiment.
SEED_A = 1234
SEED_B = 9999

#: Smallest number of draws per seed at which "reproduces" is a claim. One draw
#: is trivially identical to itself, so a single-draw run would report every
#: accepted key as replaying and reach HONOURED having observed nothing.
MIN_SEED_REPEATS = 2


def probe_unseeded_control(
    model: str, temperature: float, repeats: int
) -> dict[str, Any]:
    """Unseeded draws at the seed probe's own temperature: the null hypothesis.

    Drawn here rather than read off the temperature table, because the table is
    swept over ``--temperatures`` and the seed probe runs at
    ``--seed-probe-temperature``; those coincide by default and a control that
    is only valid by default is not a control. It is drawn once and shared by
    all three modes, since "does this engine sample" is a property of the
    engine and not of which key the seed was sent under.

    ``samples`` is computed over the completions that came back, the same rule
    :func:`probe_temperature` follows and for the same reason: a failed call
    records ``content: ""``, which counted would make an outage read as either
    diversity or collapse. It says *sampling was observed*, never *sampling was
    ruled out* -- a control that lost two of its three draws reports ``False``,
    which is why the verdict that depends on it is INCONCLUSIVE rather than a
    finding.
    """
    results = [
        call(model, temperature=temperature, seed=None, seed_mode="top_level")
        for _ in range(repeats)
    ]
    errors = [r["error"] for r in results if r["error"]]
    contents = [r["content"] for r in results if not r["error"]]
    return {
        "temperature": temperature,
        "draws": repeats,
        "delivered": len(contents),
        "distinct": len(set(contents)),
        "errors": errors,
        "samples": len(set(contents)) > 1,
    }


def seed_verdict(
    *, replays: bool, diverges: bool, control_samples: bool
) -> tuple[str, str]:
    """``(verdict, why)`` for one seed key, from the three observations.

    The rule, and why each branch is where it is:

    * **Not replaying is conclusive on its own.** An honoured seed reproduces
      by definition, so two draws that differ under one seed rule honouring
      out, and they do it without the control -- which is what makes this the
      direction the probe can still report on an engine it could not
      characterise. (It assumes the engine has no nondeterminism *outside*
      sampling; batching effects would show here as a false IGNORED, and that
      is the residual this probe cannot see.)
    * **Replaying proves nothing while the engine does not sample.** An engine
      returning one completion whatever you ask replays under every key,
      including keys it discards. This is the false HONOURED, and the repeat
      count does not fix it: more identical draws from a collapsed engine are
      more of the same non-evidence. Only the control separates them.
    * **Replaying without divergence is genuinely ambiguous.** Either the seed
      is ignored by an engine with very little entropy left under the grammar,
      or it is honoured and two seeds landed on the same completion -- the
      false negative. Nothing in this experiment distinguishes those, so it
      says so instead of picking one.

    Note what is *not* here: a threshold on how many repeats make a replay
    convincing. There is no such constant to invent, because the repeat count
    is not what carries the conclusion -- the control is. Repeats only have to
    reach :data:`MIN_SEED_REPEATS`, below which "reproduces" is not a claim.
    """
    if not replays:
        return IGNORED, "the same seed gave different completions"
    if not control_samples:
        return INCONCLUSIVE, (
            "the seed replayed, but unseeded draws at this temperature did not "
            "differ either, so an ignored seed looks the same"
        )
    if not diverges:
        return INCONCLUSIVE, (
            "the seed replayed and two seeds gave the same completion, which "
            "is both an ignored seed and an honoured one with nothing left to "
            "choose"
        )
    return HONOURED, "the seed replayed, two seeds diverged, and the engine samples"


def seed_draw_order(repeats: int) -> list[int]:
    """The order the two seeds are drawn in: ABBA, repeated.

    All of `SEED_A`'s draws used to precede all of `SEED_B`'s, so the seed was
    confounded with request order. An engine that ignores the key but whose
    output moves with server state -- a cache warming, a batch changing shape
    -- replays within each contiguous block and differs between them, which is
    exactly `replays and diverges`: a false HONOURED, measured at 3/3 keys
    against a fake engine whose output changed once every three requests.

    Plain alternation (ABAB) closes that and opens its mirror image: an engine
    whose output flips on every request hands each seed one parity, so each
    replays and the two differ. ABBA closes both. Each seed lands on both
    parities, and each seed's first draw precedes the other's last, so no single
    change point can put one seed entirely before it and the other after.

    What it cannot close is a drift with a longer period that happens to match
    the pattern. The unseeded control is the guard there: an engine that moves
    with server state shows it as diversity where no seed is set.
    """
    order: list[int] = []
    for i in range(repeats):
        order += [SEED_A, SEED_B] if i % 2 == 0 else [SEED_B, SEED_A]
    return order


def probe_seed_modes(
    model: str,
    temperature: float,
    repeats: int = 3,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find a seed key the engine actually honours, behaviourally.

    Accepted-but-ignored is the failure this is built to catch, so a mode never
    counts as working on acceptance alone. Each seed is drawn ``repeats`` times
    rather than twice: two draws that agree are one coincidence away from a
    false HONOURED, and the cost of another draw is a request.

    ``control`` is the unseeded measurement from
    :func:`probe_unseeded_control`, passed in rather than taken because it is
    one measurement shared by three modes. Absent, every replay is treated as
    uncharacterised and the strongest available verdict is INCONCLUSIVE --
    fail-closed, since the alternative is to assume the engine samples, which
    is the assumption that manufactures the false HONOURED.
    """
    control_samples = bool(control and control.get("samples"))
    out: dict[str, Any] = {}
    for mode in ("top_level", "sampling_seed", "seed"):
        draws: dict[int, list[dict[str, Any]]] = {SEED_A: [], SEED_B: []}
        for seed in seed_draw_order(repeats):
            draws[seed].append(
                call(model, temperature=temperature, seed=seed, seed_mode=mode)
            )
        errors = [r["error"] for rows in draws.values() for r in rows if r["error"]]
        by_seed = {
            seed: {r["content"] for r in rows} for seed, rows in draws.items()
        }
        replays = not errors and all(len(seen) == 1 for seen in by_seed.values())
        diverges = not errors and by_seed[SEED_A] != by_seed[SEED_B]
        if errors:
            verdict, why = INCONCLUSIVE, "the engine refused at least one draw"
        else:
            verdict, why = seed_verdict(
                replays=replays,
                diverges=diverges,
                control_samples=control_samples,
            )
        out[mode] = {
            "accepted": not errors,
            "error": errors[0] if errors else "",
            "repeats": repeats,
            "same_seed_reproduces": replays,
            "different_seed_diverges": diverges,
            "verdict": verdict,
            "why": why,
            "honoured": verdict == HONOURED,
        }
    return out


def probe_temperature(model: str, temperature: float, draws: int) -> dict[str, Any]:
    """Unseeded draws at one temperature: diversity, and format survival.

    Only the draws that came back are measured. A failed call records
    ``content: ""``, which is both a distinct string and a tier-0 score -- so
    counting it makes an outage read as either sampling diversity or a format
    regression, on the one statistic this probe exists to produce. Same defect
    as the transport-error rows in ``run_e2e.py`` and ``rescore_e2e.py``, and
    the same remedy: keep the failures, report them separately, and leave them
    out of the arithmetic about the model.

    ``delivered`` is reported next to ``draws`` so a thin sample reads as thin.
    A ``distinct`` of 1 over 8 draws is the finding this probe was written for;
    a ``distinct`` of 1 over 1 delivered draw is not a finding at all.
    """
    results = [
        call(model, temperature=temperature, seed=None, seed_mode="top_level")
        for _ in range(draws)
    ]
    errors = [r["error"] for r in results if r["error"]]
    contents = [r["content"] for r in results if not r["error"]]
    scores = [graded(c) for c in contents]
    tiers = [s["tier"] for s in scores]
    return {
        "temperature": temperature,
        "draws": draws,
        "delivered": len(contents),
        "errors": errors,
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
    parser.add_argument("--draws", type=int, default=5,
                        help="completions per temperature; must be at least 1")
    parser.add_argument("--seed-probe-temperature", type=float, default=1.0,
                        help="seed behaviour is only observable where sampling "
                             "is non-trivial, so probe it away from 0")
    parser.add_argument("--seed-repeats", type=int, default=3,
                        help="draws per seed, and the same count for the "
                             f"unseeded control; must be at least "
                             f"{MIN_SEED_REPEATS}")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    # `--draws 0` made every counter in the temperature table honest and
    # meaningless at once: no request is sent, `delivered 0/0` and `distinct
    # 0/0` print, and the probe exits 0 as though it had measured something. A
    # recorded diversity measurement over zero completions is the shape this
    # tool exists to refuse -- it is what "distinct 1/5 out of four transport
    # failures" was, with the sample size taken to its limit. Rejected at the
    # boundary rather than defended at every counter, the same way `run_e2e`
    # bounds `--replicates`.
    if args.draws < 1:
        parser.error("--draws must be at least 1")
    # Same boundary, for the same reason, one experiment over. A single draw
    # per seed is identical to itself whatever the engine did with the key, so
    # `--seed-repeats 1` would report every accepted mode as reproducing and
    # let a false HONOURED out of a probe whose entire purpose is to catch
    # accepted-but-ignored. Refused here rather than defended in the verdict,
    # where it would read as a rule about engines instead of about arithmetic.
    if args.seed_repeats < MIN_SEED_REPEATS:
        parser.error(f"--seed-repeats must be at least {MIN_SEED_REPEATS}")

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
        # Denominated in `delivered`, not `draws`, and the difference is the
        # one reading this probe exists to make. All three counters are
        # computed over the completions that came back, so quoting them over
        # what was *asked for* prints observations nobody made: four transport
        # failures out of five draws showed as `distinct 1/5`, which is
        # byte-for-byte the greedy-decoding collapse signature this tool was
        # written to detect. An outage and a non-sampling engine read the same,
        # and the one thing separating them -- the errors -- is on the lines
        # below rather than in the number.
        #
        # `delivered` is printed beside it rather than only used as the
        # denominator, because a probe that quietly asks for five and reports
        # over two is still hiding the shortfall, just with a truthful ratio.
        delivered = row["delivered"]
        print(f"  t={temperature:<4g} delivered {delivered}/{row['draws']}  "
              f"distinct {row['distinct']}/{delivered}  "
              f"tiers {row['tiers']}  reward spread {row['spread']:.4f}  "
              f"parsed {row['parsed']}/{delivered}  "
              f"gate {row['format_gate_pass']}/{delivered}")
        for error in row["errors"][:2]:
            print(f"         error: {error[:140]}")

    print()
    print("=" * 72)
    print(f"Seed: which key does the engine honour? (t={args.seed_probe_temperature:g})")
    print("=" * 72)
    control = probe_unseeded_control(
        args.model, args.seed_probe_temperature, args.seed_repeats
    )
    report["seed_control"] = control
    # Printed before the table rather than only stored, because it is what
    # every INCONCLUSIVE below means: a reader who cannot see that the engine
    # returned one completion to unseeded draws has no way to tell a refusal
    # to conclude from a measurement that failed.
    print(f"  control (unseeded)  distinct {control['distinct']}/"
          f"{control['delivered']} of {control['draws']}  "
          f"engine-samples={control['samples']}")
    for error in control["errors"][:2]:
        print(f"         error: {error[:140]}")
    report["seed_modes"] = probe_seed_modes(
        args.model, args.seed_probe_temperature, args.seed_repeats, control
    )
    for mode, row in report["seed_modes"].items():
        print(f"  {mode:<16} {row['verdict'].upper():<13} "
              f"accepted={row['accepted']} "
              f"same-seed-reproduces={row['same_seed_reproduces']} "
              f"different-seed-diverges={row['different_seed_diverges']}")
        print(f"         {row['why']}")
        if row["error"]:
            print(f"         error: {row['error'][:140]}")

    honoured = [m for m, r in report["seed_modes"].items() if r["honoured"]]
    ignored = [m for m, r in report["seed_modes"].items() if r["verdict"] == IGNORED]
    unclear = [
        m for m, r in report["seed_modes"].items() if r["verdict"] == INCONCLUSIVE
    ]
    print()
    if honoured:
        print(f"  use --seed-mode {honoured[0]}")
    elif unclear:
        # Deliberately not folded into the sentence below. "No seed key is
        # honoured" is a finding; "this run could not tell" is the absence of
        # one, and printing the first where the second is true is how a probe
        # launders its own inconclusive result into evidence.
        print(f"  no key was shown to be honoured, and {len(unclear)} of "
              f"{len(report['seed_modes'])} could not be decided here "
              f"({', '.join(unclear)}). Re-run at a temperature where the "
              "control samples before writing anything down.")
    else:
        print("  no seed key is honoured; rollouts will be diverse but not "
              "reproducible. Say so in the write-up.")
    report["honoured_modes"] = honoured
    report["ignored_modes"] = ignored
    report["inconclusive_modes"] = unclear

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
