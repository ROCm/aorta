#!/usr/bin/env python3
"""Seam demonstration: a graded reward for recipe synthesis, importing aorta.

This is not a workload, not a training script, and not wired into CI. It exists
to demonstrate one seam: **a reward function that lives in a trainer repo but
imports aorta, so the thing grading the model is the same code the CPU gate
runs.** Everything below calls public aorta APIs -- `load_recipe`, the registry,
the workload's own config validation. Nothing here reimplements a check.

Why that seam matters. The alternative is a reward function that reimplements
"is this recipe valid" against a copy of the schema. That copy drifts, and when
it drifts the model is optimised against a validator no longer describing the
product -- a reward that silently stops measuring the thing it names. Importing
the real validator makes drift impossible by construction: if aorta's rules
change, the reward changes with them in the same commit.

Grading
-------

Candidate recipe text in, tier out. The tiers are cumulative -- tier N means
tiers 1..N all passed -- and each is a real call:

    1  parses as YAML
    2  `aorta.triage.recipe.load_recipe` accepts the schema and cells
    3  every mitigation and environment resolves in the registry
    4  the named workload resolves and every cell's merged config validates
    5  no cell produced an "ignoring unknown key" warning

Tier 5 is the one worth explaining. `tokenspeed_serve` *warns* on an unknown
`workload_config` key and carries on with the default, so a recipe that says
`concurrency:` where the schema says `max_concurrency:` is entirely valid and
benchmarks something other than what it asked for. That is the failure mode the
docs spend the most words on, it is invisible to tiers 1-4, and it is free to
detect by capturing the warning. A reward that stopped at tier 4 would pay full
marks for a recipe that measures the wrong thing.

Tier 6 in the plan -- "does the recipe express the *asked-for* shape" -- is
deliberately not implemented. It needs a rubric or a judge model, and inventing
an automatic proxy for it is how reward hacking starts.

The novelty gate
----------------

Every tier above is a property of the artifact, not of the model's work, so a
policy that emits `recipes/tokenspeed/tokenspeed-serve-load.yaml`
character-for-character earns tier 5 for retrieval. That is not a caveat to
document, it is a reward the policy will find, so it is scored:

    reward = (tier / MAX_TIER) * novelty_multiplier

The multiplier is 1.0 below `MEMORISATION_SOFT` similarity to the nearest
committed recipe, tapers linearly to 0.0 at `MEMORISATION_HARD`, and is 0.0 at
or above it. Two thresholds rather than one because each alone is gameable: a
pure cliff lets a policy park just underneath it, and a pure taper never
actually refuses to pay for a verbatim copy.

Similarity is measured on a *canonical* form -- YAML re-parsed and re-emitted
with sorted keys, comments gone, and the arbitrary `ticket` label dropped -- so
the obvious evasions do not work. Renaming the ticket, reordering keys,
reindenting, or stripping comments leaves a copied recipe at ~1.0 similarity and
a reward of zero. Only changing what the recipe *does* moves it.

What the gate still cannot do: it measures distance from the committed corpus,
which is a proxy for the training corpus and not the same set. It also cannot
tell a novel recipe from a novel *useless* one -- tier 6, "does this express the
asked-for shape", needs a rubric or a judge model and is deliberately still
unimplemented, because inventing an automatic proxy for it is how reward hacking
starts. A real run still needs held-out prompts and tier-6 grading; the gate
removes the single largest way to score well without working, not all of them.

Usage
-----

    python examples/rl/recipe_reward.py                     # built-in demo
    python examples/rl/recipe_reward.py path/to/recipe.yaml  # grade files
    python examples/rl/recipe_reward.py --no-novelty-gate path/to/recipe.yaml
    python examples/rl/recipe_reward.py --json recipes/tokenspeed/*.yaml
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aorta.registry import (
    UnknownEnvironmentError,
    UnknownMitigationError,
    get_environment,
    get_mitigation,
)
from aorta.run.discovery import get_workload_class
from aorta.triage.recipe import load_recipe

MAX_TIER = 5

# Below SOFT a candidate is treated as its own work; at or above HARD it is
# treated as retrieval and paid nothing. See "The novelty gate" above for why
# there are two thresholds instead of one.
MEMORISATION_SOFT = 0.80
MEMORISATION_HARD = 0.95

# Fields that identify a recipe without changing what it measures. Two
# candidates differing only here are the same recipe for novelty purposes, so
# renaming the ticket cannot buy a policy out of the gate.
_IDENTITY_KEYS = ("ticket",)

TIER_NAMES = {
    0: "does not parse",
    1: "parses as YAML",
    2: "load_recipe accepts the schema",
    3: "mitigations and environments resolve",
    4: "every cell's workload config validates",
    5: "no unknown-key warnings",
}


@dataclass
class Grade:
    """The graded result for one candidate recipe."""

    tier: int = 0
    reward: float = 0.0
    tier_reward: float = 0.0
    failed_at: str | None = None
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    nearest_committed: tuple[str, float] | None = None
    novelty_multiplier: float = 1.0
    memorised: bool = False

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tier": self.tier,
            "reward": self.reward,
            "tier_reward": self.tier_reward,
            "tier_name": TIER_NAMES[self.tier],
            "failed_at": self.failed_at,
            "reason": self.reason,
            "warnings": self.warnings,
            "novelty_multiplier": self.novelty_multiplier,
            "memorised": self.memorised,
        }
        if self.nearest_committed is not None:
            out["nearest_committed_recipe"] = self.nearest_committed[0]
            out["nearest_committed_similarity"] = round(self.nearest_committed[1], 4)
        return out


class _WarningTrap(logging.Handler):
    """Collect WARNING+ records emitted while a workload validates its config.

    Tier 5 exists because this is the only place the "unknown key" signal
    surfaces: the workload logs it and proceeds with the default.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def _cell_configs(recipe: Any) -> list[tuple[str, dict[str, Any]]]:
    """(cell name, merged workload_config) for every cell, base config underneath.

    Mirrors how the runner composes a cell's config: the recipe-level
    `workload_config` is the base and each cell's block overrides it.
    """
    base = dict(getattr(recipe, "workload_config", None) or {})
    cells = getattr(recipe, "cells", None) or []
    out: list[tuple[str, dict[str, Any]]] = []
    for index, cell in enumerate(cells):
        merged = dict(base)
        merged.update(dict(getattr(cell, "workload_config", None) or {}))
        out.append((getattr(cell, "name", None) or f"cell[{index}]", merged))
    if not out:
        out.append(("(no cells)", base))
    return out


class NoConfigOnlySeam(Exception):
    """The workload offers no way to validate a config without a machine."""


def _validate_config(workload_cls: type, config: dict[str, Any]) -> None:
    """Run the workload's own config validation, without touching hardware.

    Uses `_validated_config`, the pure-config half of `setup()` -- `setup()`
    itself then goes on to require docker and a readable /dev/kfd, so calling it
    here would make the reward depend on the grader having a GPU.

    There used to be an `else: instance.setup()` fallback for workloads without
    that half, and it was worse than a missing feature. Only `tokenspeed_serve`
    and `hrx_perf` define `_validated_config`, so the fallback was the *common*
    path, and what it ran was not validation: `GpuSmokeWorkload.setup()` imports
    torch, requires `torch.cuda.is_available()` and selects GPU 0, and
    `LlmDeterminismWorkload.setup()` calls `dist.init_process_group`. A reward
    function initialising a process group is not a grading strategy. On a
    CPU-only grader the same fallback marked correct recipes down at tier 4 with
    a torch import error as the reason.

    So this raises instead, and the caller reports the gap by name. That does
    mean eight of the ten workload families cannot currently reach tier 5, which
    is a real limitation and the reason it is stated here rather than absorbed:
    the fix is upstream, a public `Workload.validate_config()`, which would let a
    reward function commit to a supported surface instead of to an underscore.
    """
    instance = workload_cls(config)
    validator = getattr(instance, "_validated_config", None)
    if not callable(validator):
        raise NoConfigOnlySeam(
            f"{workload_cls.__name__} exposes no config-only validation seam "
            "(`_validated_config`), and its `setup()` acquires hardware -- torch, "
            "a visible GPU, or a process group -- so this grader cannot check "
            "the cell without one. Not a judgement on the recipe."
        )
    validator()


def grade_recipe_text(
    text: str,
    *,
    corpus: dict[str, str] | None = None,
) -> Grade:
    """Grade one candidate recipe. Never raises -- a bad candidate is a low tier."""
    grade = Grade()

    # Tier 1 -- YAML.
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        grade.failed_at, grade.reason = "tier1_yaml", f"{type(exc).__name__}: {exc}"
        return _finish(grade, text, corpus)
    if not isinstance(data, dict):
        grade.failed_at = "tier1_yaml"
        grade.reason = f"recipe root must be a mapping, got {type(data).__name__}"
        return _finish(grade, text, corpus)
    grade.tier = 1

    # `load_recipe` takes a path, so the candidate has to reach the disk. The
    # suffix matters: the loader dispatches YAML vs JSON on it.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.yaml"
        path.write_text(text, encoding="utf-8")

        # Tiers 2 and 3 -- `load_recipe` checks the schema *and* resolves
        # registry names, so the two are separated by which error it raises.
        try:
            recipe = load_recipe(path)
        except (UnknownMitigationError, UnknownEnvironmentError) as exc:
            grade.tier = 2
            grade.failed_at = "tier3_registry"
            grade.reason = f"{type(exc).__name__}: {exc}"
            return _finish(grade, text, corpus)
        except Exception as exc:  # noqa: BLE001 - any schema failure is tier 2
            grade.failed_at = "tier2_load_recipe"
            grade.reason = f"{type(exc).__name__}: {exc}"
            return _finish(grade, text, corpus)
        grade.tier = 2

        # Tier 3 -- re-resolve explicitly. `load_recipe` has already done this,
        # so this cannot fail here; it is spelled out because the tier is a
        # named contract and a future loader that stopped resolving eagerly
        # should fail this tier rather than silently pass it.
        try:
            for cell in getattr(recipe, "cells", None) or []:
                for name in getattr(cell, "mitigations", None) or []:
                    if str(name) != "none":
                        get_mitigation(str(name))
                environment = getattr(cell, "environment", None)
                if environment:
                    get_environment(str(environment))
        except (UnknownMitigationError, UnknownEnvironmentError) as exc:
            grade.failed_at = "tier3_registry"
            grade.reason = f"{type(exc).__name__}: {exc}"
            return _finish(grade, text, corpus)
        grade.tier = 3

        # Tier 4 -- the workload's own validation, per cell.
        workload_name = str(getattr(recipe, "workload", "") or "")
        try:
            workload_cls = get_workload_class(workload_name)
        except Exception as exc:  # noqa: BLE001
            grade.failed_at = "tier4_workload"
            grade.reason = f"unknown workload {workload_name!r}: {exc}"
            return _finish(grade, text, corpus)

        trap = _WarningTrap()
        root = logging.getLogger()
        root.addHandler(trap)
        previous_level = root.level
        root.setLevel(logging.WARNING)
        try:
            for cell_name, config in _cell_configs(recipe):
                # A work_dir the validator can stat, so a recipe is not marked
                # down for the grader's filesystem.
                config = dict(config)
                config.setdefault("work_dir", str(Path(tmp) / "work"))
                try:
                    _validate_config(workload_cls, config)
                except NoConfigOnlySeam as exc:
                    # Distinguished from a bad config on purpose. The recipe may
                    # be perfect; we simply cannot say. Reported under its own
                    # name so a run can be read as "ungraded here" rather than
                    # counted as evidence that the model wrote something wrong.
                    grade.failed_at = "tier4_ungradeable"
                    grade.reason = f"cell {cell_name!r}: {exc}"
                    return _finish(grade, text, corpus)
                except Exception as exc:  # noqa: BLE001
                    grade.failed_at = "tier4_workload"
                    grade.reason = f"cell {cell_name!r}: {type(exc).__name__}: {exc}"
                    return _finish(grade, text, corpus)
        finally:
            root.removeHandler(trap)
            root.setLevel(previous_level)
        grade.tier = 4

        # Tier 5 -- the silent-misconfiguration guard.
        unknown = [m for m in trap.records if "unknown" in m.lower()]
        grade.warnings = unknown
        if unknown:
            grade.failed_at = "tier5_unknown_keys"
            grade.reason = "; ".join(unknown[:4])
            return _finish(grade, text, corpus)
        grade.tier = 5

    return _finish(grade, text, corpus)


def canonicalise(text: str) -> str:
    """Reduce a recipe to what it *does*, for similarity comparison.

    Re-emitting the parsed YAML with sorted keys collapses the whole class of
    cosmetic edits -- comments, indentation, key order, quoting style -- and
    dropping the identity fields collapses renaming. What survives is the
    recipe's actual content, so similarity measures copying rather than
    formatting. Unparseable text is compared raw; it cannot score above tier 0
    anyway, so its novelty is moot.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if k not in _IDENTITY_KEYS}
    return yaml.safe_dump(data, sort_keys=True, default_flow_style=False)


def novelty_multiplier(similarity: float) -> float:
    """Reward scale for a candidate this close to the nearest committed recipe."""
    if similarity < MEMORISATION_SOFT:
        return 1.0
    if similarity >= MEMORISATION_HARD:
        return 0.0
    span = MEMORISATION_HARD - MEMORISATION_SOFT
    return round((MEMORISATION_HARD - similarity) / span, 4)


def _finish(grade: Grade, text: str, corpus: dict[str, str] | None) -> Grade:
    grade.tier_reward = grade.tier / MAX_TIER
    grade.reward = grade.tier_reward

    if not corpus:
        return grade

    candidate = canonicalise(text)
    best_name, best_ratio = None, 0.0
    for name, committed in corpus.items():
        ratio = difflib.SequenceMatcher(None, candidate, canonicalise(committed)).ratio()
        if ratio > best_ratio:
            best_name, best_ratio = name, ratio
    if best_name is None:
        return grade

    grade.nearest_committed = (best_name, best_ratio)
    grade.novelty_multiplier = novelty_multiplier(best_ratio)
    grade.memorised = best_ratio >= MEMORISATION_HARD
    grade.reward = round(grade.tier_reward * grade.novelty_multiplier, 4)
    return grade


def load_corpus(root: Path) -> dict[str, str]:
    """Every committed recipe, for the memorisation check."""
    corpus: dict[str, str] = {}
    for path in sorted(root.rglob("*.yaml")):
        try:
            corpus[str(path.relative_to(root.parent))] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return corpus


# --------------------------------------------------------------------------- #
# Built-in demonstration
# --------------------------------------------------------------------------- #

_GOOD = """\
schema_version: 1
ticket: DEMO-REWARD-GOOD
workload: tokenspeed_serve
trials: 1
steps: 1
workload_config:
  image: lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78
  model: Qwen/Qwen3-0.6B
  input_len: 128
  output_len: 128
  num_prompts: 32
cells:
  - name: concurrency-8
    mitigations: [none]
    environment: local
    workload_config:
      max_concurrency: 8
  - name: concurrency-16
    mitigations: [none]
    environment: local
    workload_config:
      max_concurrency: 16
"""

# Tier 4, not tier 5. `concurrency` is not a key this workload has; it is
# warned about and dropped, and both cells then run the default concurrency --
# a two-cell scaling study whose axis does not vary. Valid, and measuring
# nothing.
_SILENTLY_WRONG = _GOOD.replace("DEMO-REWARD-GOOD", "DEMO-REWARD-TYPO").replace(
    "      max_concurrency: 8", "      concurrency: 8"
).replace("      max_concurrency: 16", "      concurrency: 16")

# Tier 3: schema is fine, the mitigation name is invented.
_BAD_MITIGATION = _GOOD.replace("DEMO-REWARD-GOOD", "DEMO-REWARD-MITIGATION").replace(
    "mitigations: [none]", "mitigations: [turbo_mode_max]"
)

# Tier 2: valid YAML, not a valid recipe.
_BAD_SCHEMA = """\
schema_version: 1
ticket: DEMO-REWARD-SCHEMA
workload: tokenspeed_serve
cells: "should be a list"
"""

# Tier 4: a real key, a rejected value -- rollout with ignore_eos: true is a
# contradiction the workload refuses.
_BAD_CONFIG = _GOOD.replace("DEMO-REWARD-GOOD", "DEMO-REWARD-CONFIG").replace(
    "  num_prompts: 32", "  num_prompts: 32\n  rollout: true\n  ignore_eos: true"
)

# Tier 1.
_BAD_YAML = "schema_version: 1\nticket: [unclosed\n"

DEMO_CASES = (
    ("well-formed two-cell scaling study", _GOOD, 5),
    ("typo'd key: `concurrency` for `max_concurrency`", _SILENTLY_WRONG, 4),
    ("invented mitigation name", _BAD_MITIGATION, 2),
    ("rollout: true with ignore_eos: true", _BAD_CONFIG, 3),
    ("cells is a string", _BAD_SCHEMA, 1),
    ("unterminated YAML flow sequence", _BAD_YAML, 0),
)


def _cosmetic_mutation(text: str) -> str:
    """A copy edited only in ways that do not change what the recipe measures.

    This is the evasion the gate has to survive: reword the ticket, drop the
    comments, reflow the indentation. The canonical form is identical, so the
    similarity is unchanged and the reward stays zero.
    """
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    body = "\n".join(lines)
    data = yaml.safe_load(body)
    if isinstance(data, dict):
        data["ticket"] = "DEMO-RENAMED-TO-EVADE"
        data = dict(reversed(list(data.items())))
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, indent=4)


def _novelty_demo(corpus: dict[str, str]) -> int:
    """Score a verbatim copy, a cosmetic copy, and a genuinely novel recipe."""
    print("=" * 72)
    print("The novelty gate: does copying still pay?")
    print("=" * 72)
    print(
        f"reward = (tier/{MAX_TIER}) * novelty, novelty tapering 1.0 -> 0.0 between "
        f"{MEMORISATION_SOFT:.2f} and {MEMORISATION_HARD:.2f} similarity\n"
    )

    if not corpus:
        print("!! no committed recipes found; cannot demonstrate the gate")
        return 1

    # Copy a committed tokenspeed_serve recipe: same workload as the novel
    # candidate below, so the two differ in novelty and nothing else. Grading
    # the corpus to hunt for a target would cost one full validation per recipe.
    target_name, target_text = None, None
    for name, text in sorted(corpus.items()):
        if "tokenspeed-serve" in name and "rollout" not in name:
            target_name, target_text = name, text
            break
    if target_text is None:
        target_name, target_text = sorted(corpus.items())[0]
    print(f"copy target: {target_name}\n")

    cases = (
        ("verbatim copy of a committed recipe", target_text, True),
        ("same recipe, ticket renamed and keys reordered", _cosmetic_mutation(target_text), True),
        ("a genuinely novel valid recipe", _GOOD, False),
    )

    failures = 0
    for label, text, expect_memorised in cases:
        grade = grade_recipe_text(text, corpus=corpus)
        ok = grade.memorised == expect_memorised and grade.tier == MAX_TIER
        if not ok:
            failures += 1
        nearest, ratio = grade.nearest_committed or ("(none)", 0.0)
        print(f"{'ok ' if ok else '!! '}{label}")
        print(
            f"       tier {grade.tier}/{MAX_TIER} (would pay {grade.tier_reward:.2f}) "
            f"* novelty {grade.novelty_multiplier:.2f} = reward {grade.reward:.2f}"
        )
        print(f"       nearest: {nearest} at {ratio:.3f} canonical similarity")
        print(f"       memorised: {grade.memorised} (expected {expect_memorised})")
        print()

    print("The first two cases are the point: both are tier 5, both would have")
    print("earned full marks from the tiers alone, and both now pay nothing. The")
    print("third is equally well-formed and keeps its full reward, which is what")
    print("makes the gate a novelty term and not just a difficulty penalty.\n")
    return failures


def run_demo(corpus: dict[str, str] | None) -> int:
    print("Seam demonstration: recipe-synthesis reward, graded by aorta itself.")
    print(f"Tiers are cumulative; tier reward = tier / {MAX_TIER}.\n")
    failures = 0
    for label, text, expected in DEMO_CASES:
        # Graded without the corpus so the tier ladder reads as a tier ladder;
        # the gate gets its own section below.
        grade = grade_recipe_text(text, corpus=None)
        ok = "ok " if grade.tier == expected else "!! "
        if grade.tier != expected:
            failures += 1
        print(f"{ok} tier {grade.tier}/{MAX_TIER}  reward {grade.reward:.2f}  {label}")
        print(f"       {TIER_NAMES[grade.tier]}")
        if grade.reason:
            print(f"       stopped at {grade.failed_at}: {grade.reason[:160]}")
        print()
    print("The tier-4 case is the point of tier 5: it is a valid recipe whose")
    print("concurrency axis does not vary, because the key naming that axis was")
    print("dropped with a warning. Tiers 1-4 cannot see it.\n")

    failures += _novelty_demo(corpus or {})
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade candidate aorta recipes with aorta's own validators.",
    )
    parser.add_argument("recipes", nargs="*", type=Path,
                        help="recipe files to grade; omit for the built-in demo")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--no-novelty-gate", action="store_true",
                        help="score tiers only, without penalising corpus copies")
    parser.add_argument("--recipes-root", type=Path, default=None,
                        help="corpus root for the novelty gate "
                             "(default: <repo>/recipes)")
    args = parser.parse_args(argv)

    # The gate is on by default: it is part of the reward, not a diagnostic, and
    # a scorer that silently omits it pays full marks for retrieval.
    corpus = None
    if not args.no_novelty_gate:
        root = args.recipes_root or (Path(__file__).resolve().parents[2] / "recipes")
        corpus = load_corpus(root) if root.is_dir() else {}

    if not args.recipes:
        return run_demo(corpus)

    results = {}
    worst = 0
    for path in args.recipes:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            results[str(path)] = {"tier": 0, "reason": str(exc)}
            continue
        grade = grade_recipe_text(text, corpus=corpus)
        results[str(path)] = grade.as_dict()
        worst = max(worst, MAX_TIER - grade.tier)
        if not args.json:
            print(f"tier {grade.tier}/{MAX_TIER}  reward {grade.reward:.2f}  {path}")
            print(f"     {TIER_NAMES[grade.tier]}")
            if grade.reason:
                print(f"     stopped at {grade.failed_at}: {grade.reason[:200]}")
            if grade.nearest_committed:
                name, ratio = grade.nearest_committed
                print(f"     nearest committed recipe: {name} ({ratio:.3f} similarity)")
                if grade.memorised:
                    print(f"     MEMORISED: novelty gate zeroed a tier-{grade.tier} "
                          f"reward of {grade.tier_reward:.2f}")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
