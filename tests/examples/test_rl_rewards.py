# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the RL reward seams under ``examples/rl``.

These are examples rather than package modules, so they are loaded by path.
They are still worth testing: both are scorers, and a scorer that silently stops
detecting what it claims to detect is the failure mode the whole design is
guarding against. The novelty gate and the degenerate-policy floor are the two
things most likely to rot unnoticed, so they are what these pin down.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rl"


def _load(name: str):
    """Import an example module by path, registered so dataclasses resolve.

    ``@dataclass`` looks the owning module up in ``sys.modules`` to resolve
    string annotations, so a module loaded without registering it there raises
    on the first dataclass. Hence the assignment before ``exec_module``.
    """
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipe_reward():
    return _load("recipe_reward")


@pytest.fixture(scope="module")
def triage_reward():
    return _load("triage_reward")


@pytest.fixture(scope="module")
def proposal_reward():
    return _load("proposal_reward")


def _real_detector_ids() -> set[str]:
    """Every detector ID the classifier tiers can actually emit.

    Read from the tier modules' own ``DETECTOR_*`` constants and ID frozensets
    so a rename shows up as a test failure rather than as a fixture quietly
    citing a detector that no longer exists.
    """
    ids: set[str] = set()
    for name in (
        "tier1_process",
        "tier2_hang",
        "tier3_kernel",
        "tier4_patterns",
        "tier5_custom",
        "verdict",
    ):
        module = importlib.import_module(f"aorta.probe.classifier.{name}")
        for key, value in vars(module).items():
            if key.startswith("DETECTOR") and isinstance(value, str) and ":" in value:
                ids.add(value)
            elif key.endswith("DETECTOR_IDS") and isinstance(value, frozenset):
                ids |= {v for v in value if isinstance(v, str) and ":" in v}
    return ids


# --------------------------------------------------------------------------- #
# recipe_reward: the novelty gate
# --------------------------------------------------------------------------- #


def test_the_taper_runs_from_full_reward_to_none(recipe_reward):
    m = recipe_reward
    assert m.novelty_multiplier(0.0) == 1.0
    assert m.novelty_multiplier(m.MEMORISATION_SOFT - 0.01) == 1.0
    assert m.novelty_multiplier(m.MEMORISATION_HARD) == 0.0
    assert m.novelty_multiplier(1.0) == 0.0
    midpoint = (m.MEMORISATION_SOFT + m.MEMORISATION_HARD) / 2
    assert 0.0 < m.novelty_multiplier(midpoint) < 1.0


def test_the_taper_is_monotonic_so_copying_more_never_pays_more(recipe_reward):
    m = recipe_reward
    steps = [i / 100 for i in range(0, 101)]
    values = [m.novelty_multiplier(s) for s in steps]
    assert values == sorted(values, reverse=True)


def test_canonicalising_ignores_edits_that_change_nothing(recipe_reward):
    """The evasions the gate has to survive: rename, reorder, reindent, uncomment."""
    original = (
        "# a leading comment\n"
        "schema_version: 1\n"
        "ticket: TICKET-ONE\n"
        "workload: tokenspeed_serve\n"
        "trials: 1\n"
        "steps: 1\n"
        "cells:\n"
        "  - name: only\n"
        "    mitigations: [none]\n"
        "    environment: local\n"
    )
    evaded = (
        "steps: 1\n"
        "trials: 1\n"
        "workload: tokenspeed_serve\n"
        "ticket: TICKET-RENAMED-TO-EVADE\n"
        "schema_version: 1\n"
        "cells:\n"
        "    -   name: only\n"
        "        environment: local\n"
        "        mitigations:\n"
        "            - none\n"
    )
    assert recipe_reward.canonicalise(original) == recipe_reward.canonicalise(evaded)


def test_canonicalising_still_separates_recipes_that_differ(recipe_reward):
    base = "schema_version: 1\nticket: T\nworkload: tokenspeed_serve\ntrials: 1\n"
    changed = base.replace("trials: 1", "trials: 8")
    assert recipe_reward.canonicalise(base) != recipe_reward.canonicalise(changed)


_REPO = Path(__file__).resolve().parents[2]

_TWO_CELLS = (
    "schema_version: 1\n"
    "ticket: T\n"
    "workload: tokenspeed_serve\n"
    "trials: 1\n"
    "{confound}"
    "cells:\n"
    "  - name: {first}\n"
    "    mitigations: [none]\n"
    "    environment: local\n"
    "    workload_config:\n"
    "      max_concurrency: {conc}\n"
    "  - name: {second}\n"
    "    mitigations: [none]\n"
    "    environment: local\n"
    "    workload_config:\n"
    "      max_concurrency: 16\n"
)


def _two_cells(first="conc-8", second="conc-16", conc=8, baseline=None):
    confound = f"confound:\n  baseline_cell: {baseline}\n" if baseline else ""
    return _TWO_CELLS.format(
        confound=confound, first=first, second=second, conc=conc
    )


def _pad_cell_names(text: str, pad: int) -> str:
    data = yaml.safe_load(text)
    for cell in data["cells"]:
        cell["name"] = f"{cell['name']}-{'x' * pad}"
    return yaml.safe_dump(data, sort_keys=False)


def test_padding_cell_names_cannot_buy_a_committed_recipe_out_of_the_gate(
    recipe_reward,
):
    """The review's reproduction, on the committed recipe it was measured on.

    Before the cells were relabelled, forty characters on each of this recipe's
    six cell names took a verbatim copy from similarity 1.000 to 0.691 and its
    novelty multiplier from 0.0 to 1.0: full reward for retrieval. Names have
    no length limit, so any gate threshold was a pad length away.
    """
    rel = "recipes/tokenspeed/tokenspeed-serve-load.yaml"
    committed = (_REPO / rel).read_text(encoding="utf-8")
    padded = _pad_cell_names(committed, 40)
    assert padded != committed

    assert recipe_reward.canonicalise(padded) == recipe_reward.canonicalise(committed)
    grade = recipe_reward.grade_recipe_text(padded, corpus={rel: committed})
    assert grade.nearest_committed == (rel, pytest.approx(1.0))
    assert grade.novelty_multiplier == 0.0
    assert grade.memorised is True


def test_renaming_the_baseline_cell_moves_its_reference_with_it(recipe_reward):
    """`confound.baseline_cell` is a second copy of a cell name.

    Relabelling the cells alone would leave the padded name in the reference,
    which is one field of unlimited length -- the same hole, one field wide.
    """
    original = _two_cells(baseline="conc-16")
    long_name = "renamed-" + "y" * 200
    renamed = _two_cells(first="also-renamed", second=long_name, baseline=long_name)
    assert recipe_reward.canonicalise(renamed) == recipe_reward.canonicalise(original)


def test_which_cell_is_the_baseline_still_separates_recipes(recipe_reward):
    """Narrowness: relabelling must not erase what the runner reads from a name.

    The confound classifier measures every row against the baseline, so two
    recipes that differ only in which cell that is measure different things.
    Dropping `cells[*].name` outright, as suggested in review, made the
    by-prefix pair below identical.
    """
    canon = recipe_reward.canonicalise
    prefix = recipe_reward._BASELINE_CELL_PREFIX

    # Chosen by an explicit reference.
    assert canon(_two_cells(baseline="conc-8")) != canon(_two_cells(baseline="conc-16"))
    # Chosen by the prefix, with no reference.
    assert canon(_two_cells(first=f"{prefix}a", second="b")) != canon(
        _two_cells(first="a", second=f"{prefix}b")
    )
    # A reference naming no cell is a tier 2 rejection, not a cell to invent a
    # label for; it is kept as written rather than guessed at.
    dangling = yaml.safe_load(canon(_two_cells(baseline="no-such-cell")))
    assert dangling["confound"]["baseline_cell"] == "no-such-cell"


def test_the_kept_prefix_is_the_one_the_runner_picks_a_baseline_by(recipe_reward):
    """Tripwire against the runner: `_BASELINE_CELL_PREFIX` is a copied literal.

    If `resolve_baseline` stopped reading it, or read a different one, the
    canonical form would be keeping the wrong part of a name. Both cells
    carry `[none]`, so without the prefix rule the first would win.
    """
    from aorta.triage.confound import resolve_baseline
    from aorta.triage.recipe import Cell

    prefix = recipe_reward._BASELINE_CELL_PREFIX
    by_prefix = [
        Cell(name="plain", mitigations=("none",), environment="local"),
        Cell(name=f"{prefix}x", mitigations=("none",), environment="local"),
    ]
    assert resolve_baseline(by_prefix, None).name == f"{prefix}x"

    # `baseline` alone -- the first cell of tokenspeed-serve-gptoss.yaml -- is
    # not a baseline by name, so the canonical form must not mark it as one.
    bare = prefix.rstrip("-")
    not_by_prefix = [
        Cell(name="plain", mitigations=("none",), environment="local"),
        Cell(name=bare, mitigations=("none",), environment="local"),
    ]
    assert resolve_baseline(not_by_prefix, None).name == "plain"
    labels = [
        c["name"]
        for c in yaml.safe_load(
            recipe_reward.canonicalise(_two_cells(first=bare, second=f"{prefix}x"))
        )["cells"]
    ]
    assert labels == ["cell-0", f"{prefix}cell-1"]


def test_relabelling_keeps_every_cell_field_but_the_name(recipe_reward):
    """Narrowness: cells that differ in configuration must still differ.

    Checked on every cell of a committed recipe, field for field, so a
    relabelling that dropped or rewrote anything but the name would show here
    rather than as two different recipes quietly scoring as one.
    """
    committed = (_REPO / "recipes/tokenspeed/tokenspeed-serve-load.yaml").read_text(
        encoding="utf-8"
    )
    original = yaml.safe_load(committed)["cells"]
    canonical = yaml.safe_load(recipe_reward.canonicalise(committed))["cells"]
    assert len(canonical) == len(original)
    for before, after in zip(original, canonical, strict=True):
        assert {k: v for k, v in after.items() if k != "name"} == {
            k: v for k, v in before.items() if k != "name"
        }

    canon = recipe_reward.canonicalise
    assert canon(_two_cells(conc=8)) != canon(_two_cells(conc=32))
    assert canon(_two_cells(conc=8)) != canon(_two_cells(first="a", second="b", conc=32))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cells: not-a-list\n", {"cells": "not-a-list"}),
        (
            "cells: [1, [2], {name: 7}, {mitigations: [none]}]\n",
            {"cells": [1, [2], {"name": 7}, {"mitigations": ["none"]}]},
        ),
        (
            "confound: {baseline_cell: [a]}\ncells: [{name: a}]\n",
            {"confound": {"baseline_cell": ["a"]}, "cells": [{"name": "cell-0"}]},
        ),
        (
            "confound: not-a-mapping\ncells: [{name: a}]\n",
            {"confound": "not-a-mapping", "cells": [{"name": "cell-0"}]},
        ),
    ],
)
def test_relabelling_passes_malformed_shapes_through(recipe_reward, text, expected):
    """It runs on model output, including text that fails tier 2."""
    assert yaml.safe_load(recipe_reward.canonicalise(text)) == expected


def test_unparseable_text_canonicalises_without_raising(recipe_reward):
    broken = "ticket: [unclosed\n"
    assert recipe_reward.canonicalise(broken) == broken


def test_a_verbatim_corpus_copy_earns_nothing(recipe_reward):
    """The whole point: full tier marks, zero reward."""
    corpus = {"recipes/committed.yaml": recipe_reward._GOOD}
    grade = recipe_reward.grade_recipe_text(recipe_reward._GOOD, corpus=corpus)

    assert grade.tier == recipe_reward.MAX_TIER
    assert grade.tier_reward == 1.0
    assert grade.memorised is True
    assert grade.novelty_multiplier == 0.0
    assert grade.reward == 0.0
    assert grade.nearest_committed is not None
    assert grade.nearest_committed[1] == pytest.approx(1.0)


def test_a_cosmetically_edited_copy_earns_nothing_either(recipe_reward):
    corpus = {"recipes/committed.yaml": recipe_reward._GOOD}
    evaded = recipe_reward._cosmetic_mutation(recipe_reward._GOOD)

    # Not the same bytes -- so a raw-text comparison would have paid out.
    assert evaded != recipe_reward._GOOD
    grade = recipe_reward.grade_recipe_text(evaded, corpus=corpus)
    assert grade.tier == recipe_reward.MAX_TIER
    assert grade.memorised is True
    assert grade.reward == 0.0


def test_none_does_not_cost_a_proposal_its_availability_tier(proposal_reward):
    """`none` is registered, deliberately not offered, and dropped by the loop.

    `AgentPolicy.validate_step` removes it before `run_agent_loop` iterates, so
    `["none", "tf32_off"]` is a proposal the real consumer accepts. Checking
    availability against the raw list stopped it at tier 5 for naming a
    mitigation the consumer never looks up -- the same list/cells confusion the
    precision term was already fixed for.
    """
    raw = json.dumps(
        {
            # From this branch's category set; #484 widens it and the coupling
            # is by import, so naming one of the new labels here would fail
            # until that lands.
            "category": "checkpoint_race",
            "hypothesis": "lds race",
            "next_mitigations": ["none", "tf32_off"],
            "confidence": 0.5,
            "stop": False,
        }
    )
    score = proposal_reward.score_proposal(
        proposal_reward.Proposal("p", raw, ["tf32_off"], [])
    )
    assert score.stopped_at != "tier5_available", score.detail
    assert score.tier == proposal_reward.MAX_TIER, (score.tier, score.detail)


def test_registered_is_not_the_same_as_acceptable(proposal_reward, monkeypatch):
    """Tier 4 re-derived half of `validate_step`, and kept the wrong half.

    The policy refuses a proposed name for three reasons. This tier checked
    one of them -- registry membership -- and not the other two: a name that
    looks like shell/argv, and a name that does not round-trip through a probe
    cell directory name, which is how the loop recovers tried and winning
    mitigations afterwards. A registered-but-unsafe name (the policy's own
    comment names a sidecar mitigation containing `/`) therefore reached tier
    5 with a full reward while `consumer_outcome` on the very same reply said
    `policy_stop`.

    That is the invariant the `stop: true` branch was added to restore,
    failing from the other side: MAX_TIER has to mean the real consumer would
    accept this proposal and run cells for it. The registry is stubbed because
    no *builtin* name is unsafe today -- the seam is reachable through a
    sidecar registry, and the point of asking the policy instead of copying its
    rules is that the tier does not depend on which of them happens to be
    reachable this week.
    """
    unsafe = "sidecar/thing"
    for module in (proposal_reward, importlib.import_module("aorta.agent.policy")):
        monkeypatch.setattr(module, "get_mitigation", lambda name, **kw: object())

    raw = json.dumps({
        "category": "checkpoint_race",
        "hypothesis": "lds race",
        "next_mitigations": [unsafe],
        "confidence": 0.5,
        "stop": False,
    })
    score = proposal_reward.score_proposal(
        proposal_reward.Proposal("p", raw, [unsafe], [])
    )

    assert score.tier == 3, (score.tier, score.detail)
    assert score.stopped_at == "tier4_registry", score.detail
    assert "cell name" in score.detail, score.detail
    # The two channels now agree, which is the whole point.
    assert score.consumer_outcome == "policy_stop", score.consumer_outcome


def test_a_fenced_reply_is_not_reported_as_a_silent_stop(proposal_reward):
    """The tier and the consumer outcome answer different questions.

    `LiteLLMProposer` strips a ```json fence before parsing, so a fenced reply
    is one the production path accepts. The format gate can stay strict about
    it; recording `silent_stop` cannot, because that is a claim about the loop
    stalling on a reply the loop consumes.
    """
    body = {
        "category": "checkpoint_race",
        "hypothesis": "lds race",
        "next_mitigations": ["tf32_off"],
        "confidence": 0.5,
        "stop": False,
    }
    fenced = f"```json\n{json.dumps(body)}\n```"
    score = proposal_reward.score_proposal(
        proposal_reward.Proposal("p", fenced, ["tf32_off"], [])
    )

    # Strict on the format gate: the raw text is not a bare JSON object.
    assert score.stopped_at == "tier1_json"
    # Honest about the consumer: it strips the fence and gets a usable step.
    assert score.consumer_outcome != "silent_stop", score.consumer_outcome


def test_unparseable_prose_is_still_a_silent_stop(proposal_reward):
    """Establishes the fence fix did not make every reply look consumable."""
    score = proposal_reward.score_proposal(
        proposal_reward.Proposal("p", "The collective timed out on rank 3.", ["tf32_off"], [])
    )
    assert score.stopped_at == "tier1_json"
    assert score.consumer_outcome == "silent_stop"


def test_hrx_perf_cannot_reach_tier_5_on_a_partial_seam(recipe_reward):
    """Its `_validated_config` is not the whole pure-config check.

    The seam covers bench/size/iters/warmup; `setup()` also validates
    `gpu_arch`, `timeout_sec > 0` and `keep_build` being a bool, none of which
    touch hipcc or a GPU. Treating the seam as the whole validator let a recipe
    with a nonsense `gpu_arch` read as fully valid, so it is reported
    ungradeable instead -- distinct from `tier4_workload`, which would blame the
    recipe.
    """
    from aorta.workloads.hrx_perf import HrxPerfWorkload

    assert "HrxPerfWorkload" in recipe_reward._PARTIAL_CONFIG_SEAMS
    with pytest.raises(recipe_reward.NoConfigOnlySeam):
        recipe_reward._validate_config(HrxPerfWorkload, {"bench": "gemm"})


def test_an_inline_docker_environment_reaches_tier_3(recipe_reward):
    """`{docker: <ref>}` is a valid environment, and tier 3 said otherwise.

    `load_recipe` rewrites it to an auto-name `_inline_<hash>` and records it
    on `recipe.inline_environments`; the name is deliberately absent from the
    global registry, because the definition is the recipe. Re-resolving every
    cell through plain `get_environment` therefore raised
    `UnknownEnvironmentError` on a recipe the loader had just accepted, and the
    grade came back at tier 2 -- the tier that means the schema is wrong.
    """
    text = yaml.safe_dump(
        {
            "schema_version": 1,
            "ticket": "INLINE-1",
            "workload": "gpu_smoke",
            "trials": 1,
            "steps": 1,
            "cells": [
                {
                    "name": "inline-env",
                    "mitigations": ["none"],
                    "environment": {"docker": "ubuntu:24.04"},
                }
            ],
        }
    )
    grade = recipe_reward.grade_recipe_text(text)

    assert grade.tier >= 3, (grade.tier, grade.failed_at, grade.reason)
    assert grade.failed_at != "tier3_registry", grade.reason


def test_an_unknown_environment_still_fails_tier_3(recipe_reward):
    """The skip above is scoped to names the loader itself minted.

    A genuinely unregistered environment must still be caught, or the fix has
    turned tier 3 off rather than corrected it.
    """
    text = yaml.safe_dump(
        {
            "schema_version": 1,
            "ticket": "INLINE-2",
            "workload": "gpu_smoke",
            "trials": 1,
            "steps": 1,
            "cells": [
                {
                    "name": "bad-env",
                    "mitigations": ["none"],
                    "environment": "no_such_environment",
                }
            ],
        }
    )
    grade = recipe_reward.grade_recipe_text(text)

    assert grade.tier == 2
    assert grade.failed_at == "tier3_registry", grade.reason


def _mitigations_file(tmp_path, mitigations):
    """A `--mitigations-file` sidecar, in the shape the registry reads."""
    side = tmp_path / "extra_mitigations.json"
    side.write_text(
        json.dumps({"version": 1, "mitigations": mitigations}), encoding="utf-8"
    )
    return side


def _sidecar_recipe(ticket, mitigation):
    return yaml.safe_dump(
        {
            "schema_version": 1,
            "ticket": ticket,
            "workload": "gpu_smoke",
            "trials": 1,
            "steps": 1,
            "cells": [
                {"name": "side", "mitigations": [mitigation], "environment": "local"}
            ],
        }
    )


def test_a_sidecar_supplied_mitigation_survives_tier_3(recipe_reward, tmp_path):
    """The grader was standalone-recipes-only and nothing said so.

    `load_recipe` was called with no `sidecar_files`, so a recipe naming a
    mitigation the operator supplied through `--mitigations-file` -- valid,
    runnable, and accepted by `aorta triage` and `aorta probe` -- failed tier 3
    here as an unregistered name. On a training run that is a correct candidate
    taught to be wrong, and `tier3_registry` reads as the model having invented
    a mitigation.

    `grade_recipe_text` now takes the same argument the CLIs take and forwards
    it to the same place, so the candidate is graded against the registry view
    it would actually run under. That also makes the tier-3 block's own sidecar
    threading reachable: `recipe.sidecar_files` is populated, so `get_mitigation`
    and `get_environment` re-resolve against what the loader used.
    """
    side = _mitigations_file(tmp_path, {"sidecar_only_knob": {"SIDECAR_ONLY": "1"}})
    text = _sidecar_recipe("SIDECAR-1", "sidecar_only_knob")

    without = recipe_reward.grade_recipe_text(text)
    assert without.failed_at == "tier3_registry", without.reason
    assert "sidecar_only_knob" in without.reason

    grade = recipe_reward.grade_recipe_text(text, sidecar_files=(side,))

    assert grade.tier >= 3, (grade.tier, grade.failed_at, grade.reason)
    assert grade.failed_at != "tier3_registry", grade.reason


def test_an_unknown_mitigation_still_fails_tier_3_with_sidecars_threaded(
    recipe_reward, tmp_path
):
    """Narrowness: threading the sidecars must not switch the mitigation axis off.

    The same shape as the inline-environment pair above. A name in neither the
    builtins nor the sidecar has to stay a tier-3 failure, or the fix has stopped
    checking rather than started checking correctly.
    """
    side = _mitigations_file(tmp_path, {"sidecar_only_knob": {"SIDECAR_ONLY": "1"}})
    text = _sidecar_recipe("SIDECAR-2", "no_such_mitigation")

    grade = recipe_reward.grade_recipe_text(text, sidecar_files=(side,))

    assert grade.tier == 2
    assert grade.failed_at == "tier3_registry", grade.reason


def test_the_recipe_cli_forwards_its_mitigations_file(recipe_reward, tmp_path, capsys):
    """The flag is the whole point: a grader nobody can pass sidecars to has none.

    Spelled as `aorta triage --mitigations-file` spells it, repeatable for the
    same reason, and forwarded to `load_recipe`'s own argument -- so scoring a
    candidate from a sidecar run is the same flag rather than a second concept.
    """
    side = _mitigations_file(tmp_path, {"sidecar_only_knob": {"SIDECAR_ONLY": "1"}})
    recipe = tmp_path / "candidate.yaml"
    recipe.write_text(_sidecar_recipe("SIDECAR-CLI", "sidecar_only_knob"), "utf-8")

    recipe_reward.main([str(recipe), "--json", "--no-novelty-gate"])
    bare = json.loads(capsys.readouterr().out)[str(recipe)]

    recipe_reward.main(
        [str(recipe), "--json", "--no-novelty-gate", "--mitigations-file", str(side)]
    )
    supplied = json.loads(capsys.readouterr().out)[str(recipe)]

    assert bare["failed_at"] == "tier3_registry", bare["reason"]
    assert supplied["failed_at"] != "tier3_registry", supplied["reason"]
    assert supplied["tier"] >= 3, supplied


def test_an_unreadable_recipe_fails_the_novelty_gate_closed(recipe_reward, tmp_path):
    """A skipped recipe is one a verbatim copy of it scores full marks against.

    The same defect as an empty corpus root, reached one file at a time -- and
    worse for being partial: the other files stay readable, so nothing in the
    output looks short while the CLI still reports the gate as enabled.
    """
    root = tmp_path / "recipes"
    root.mkdir()
    (root / "readable.yaml").write_text("schema_version: 1\n")
    unreadable = root / "locked.yaml"
    unreadable.write_text("schema_version: 1\n")
    unreadable.chmod(0o000)
    try:
        with pytest.raises(recipe_reward.UnreadableCorpus, match="locked.yaml"):
            recipe_reward.load_corpus(root)
    finally:
        unreadable.chmod(0o644)

    # And readable again once the cause is gone, so the guard is the file and
    # not the directory.
    assert set(recipe_reward.load_corpus(root)) == {
        "recipes/readable.yaml",
        "recipes/locked.yaml",
    }


def test_the_grader_only_injects_a_scratch_key_the_workload_takes(recipe_reward):
    """The grader's own convenience must not become the model's error.

    `work_dir` was added to every cell so the validator had somewhere to stat.
    `HrxPerfWorkload` does not accept it, so its validator logged "ignoring
    unknown workload_config key 'work_dir'" -- and tier 5 is precisely the tier
    that fails on an unknown-key warning. Every otherwise-valid `hrx_perf`
    recipe therefore lost the top tier over a key the model never wrote.
    """
    from aorta.workloads.hrx_perf import HrxPerfWorkload
    from aorta.workloads.tokenspeed_serve import TokenSpeedServeWorkload

    assert recipe_reward._scratch_keys_accepted_by(HrxPerfWorkload) == ()
    assert recipe_reward._scratch_keys_accepted_by(TokenSpeedServeWorkload) == ("work_dir",)


def test_the_injected_scratch_key_does_not_trip_the_unknown_key_guard(recipe_reward):
    """The above, demonstrated through the warning tier 5 actually reads."""
    import logging

    from aorta.workloads.hrx_perf import HrxPerfWorkload

    class _Trap(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[str] = []

        def emit(self, record):
            self.records.append(record.getMessage())

    trap = _Trap()
    root = logging.getLogger()
    root.addHandler(trap)
    previous = root.level
    root.setLevel(logging.WARNING)
    try:
        config = {"bench": "gemm"}
        for key in recipe_reward._scratch_keys_accepted_by(HrxPerfWorkload):
            config[key] = "/tmp/scratch"
        try:
            recipe_reward._validate_config(HrxPerfWorkload, config)
        except Exception:  # noqa: BLE001 - the warning is what is under test
            pass
    finally:
        root.removeHandler(trap)
        root.setLevel(previous)

    assert [m for m in trap.records if "unknown" in m.lower()] == []


def test_a_genuinely_novel_valid_recipe_keeps_its_full_reward(recipe_reward):
    """Otherwise the gate is a difficulty penalty, not a novelty term."""
    novel = recipe_reward._GOOD
    # A committed corpus of something structurally unrelated.
    corpus = {
        "recipes/other.yaml": yaml.safe_dump(
            {
                "schema_version": 1,
                "ticket": "OTHER-1",
                "workload": "some_other_workload",
                "trials": 9,
                "steps": 4,
                "cells": [{"name": f"c{i}", "mitigations": ["none"]} for i in range(6)],
            }
        )
    }
    grade = recipe_reward.grade_recipe_text(novel, corpus=corpus)

    assert grade.tier == recipe_reward.MAX_TIER
    assert grade.memorised is False
    assert grade.novelty_multiplier == 1.0
    assert grade.reward == 1.0


def test_without_a_corpus_the_gate_cannot_fire(recipe_reward):
    """No corpus means no novelty claim, so the tier reward stands unmodified."""
    grade = recipe_reward.grade_recipe_text(recipe_reward._GOOD, corpus=None)
    assert grade.reward == grade.tier_reward
    assert grade.memorised is False
    assert grade.nearest_committed is None


def test_a_malformed_copy_is_not_rescued_by_being_a_copy(recipe_reward):
    """The gate scales the tier reward; it never invents one."""
    corpus = {"recipes/committed.yaml": recipe_reward._BAD_YAML}
    grade = recipe_reward.grade_recipe_text(recipe_reward._BAD_YAML, corpus=corpus)
    assert grade.tier == 0
    assert grade.reward == 0.0


def test_grading_never_calls_a_workloads_setup(recipe_reward, monkeypatch):
    """A reward function must not acquire hardware to compute a number.

    Only `tokenspeed_serve` and `hrx_perf` expose `_validated_config`, so the old
    `else: instance.setup()` fallback was the common path -- and `setup()` on the
    other eight imports torch, selects GPU 0, or calls `dist.init_process_group`.
    Grading is a CPU activity; a grader that initialises a process group is not
    grading.
    """
    calls = []

    class Hardware:
        def __init__(self, config):
            self.config = config

        def setup(self):  # pragma: no cover -- the assertion is that this never runs
            calls.append("setup")
            raise AssertionError("grading called setup()")

    with pytest.raises(recipe_reward.NoConfigOnlySeam) as excinfo:
        recipe_reward._validate_config(Hardware, {})

    assert not calls
    # The message has to say it is a gap in the grader, not a fault in the
    # recipe, or the reason string reads as the model's mistake.
    assert "_validated_config" in str(excinfo.value)
    assert "Not a judgement on the recipe" in str(excinfo.value)


def test_an_ungradeable_workload_is_named_apart_from_an_invalid_config(
    recipe_reward, monkeypatch
):
    """`tier4_ungradeable` and `tier4_workload` are different findings.

    Folding the first into the second would let "we could not check this" be
    counted as "the model wrote something wrong", which is the same conflation
    the tier ladder exists to avoid elsewhere.
    """
    class Hardware:
        def __init__(self, config):
            self.config = config

        def setup(self):  # pragma: no cover
            raise AssertionError("grading called setup()")

    monkeypatch.setattr(recipe_reward, "get_workload_class", lambda _name: Hardware)
    grade = recipe_reward.grade_recipe_text(recipe_reward._GOOD, corpus=None)

    assert grade.tier == 3
    assert grade.failed_at == "tier4_ungradeable"


# --------------------------------------------------------------------------- #
# triage_reward: labelling and the degenerate floor
# --------------------------------------------------------------------------- #


def test_labels_come_from_the_resolver_not_the_stored_field(triage_reward):
    """A stored verdict that disagrees with the rules is corrected and flagged."""
    doc = triage_reward._run("mislabelled", "pass", ["tier1:exit_nonzero"], [])
    label = triage_reward.label_run(doc)

    assert label.verdict == "fail"
    assert label.stored_verdict == "pass"
    assert label.stale is True


def test_an_agreeing_run_is_not_flagged_stale(triage_reward):
    doc = triage_reward._run("clean", "pass", [], [])
    label = triage_reward.label_run(doc)
    assert label.verdict == "pass"
    assert label.stale is False


def test_a_corrupt_stored_verdict_disagrees_rather_than_going_quiet(triage_reward):
    """`verdict: "fal"` is a rotted archive, and rot is what `stale` reports.

    The value is explicitly present and is not the recomputed verdict, so it is
    a disagreement. Normalising it away made `stale` false, which is the reading
    reserved for an archive that recorded nothing -- so the one file that most
    needs flagging was the one that looked cleanest.
    """
    doc = triage_reward._run("rotted", "pass", [], [])
    doc["verdict"] = "fal"
    label = triage_reward.label_run(doc)

    assert label.verdict == "pass"
    assert label.stale is True
    # Carried through verbatim: an operator reading the STALE line has to be
    # able to see that the stored value was not a verdict at all.
    assert label.stored_verdict == "fal"


def test_a_stored_verdict_of_the_wrong_type_is_also_a_disagreement(triage_reward):
    doc = triage_reward._run("rotted-type", "pass", [], [])
    doc["verdict"] = 5
    label = triage_reward.label_run(doc)
    assert label.stale is True
    assert label.stored_verdict == "5"


def test_an_archive_that_recorded_no_verdict_is_not_called_stale(triage_reward):
    """Absent stays absent -- this is the case the invalid one was stealing."""
    for missing in ({}, {"verdict": None}):
        doc = triage_reward._run("silent", "pass", [], [])
        doc.pop("verdict")
        doc.update(missing)
        label = triage_reward.label_run(doc)
        assert label.stored_verdict is None
        assert label.stale is False


def test_one_non_object_result_does_not_abort_the_whole_sweep(triage_reward, tmp_path):
    """A JSON list parses fine and then kills `label_run` from outside its try.

    `AttributeError` is not in the loader's except clause, so a single rotted
    file scored none of the directory rather than none of itself.
    """
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "result.json").write_text(
        json.dumps(triage_reward._run("ok", "fail", ["tier1:exit_nonzero"], []))
    )
    (tmp_path / "listy").mkdir()
    (tmp_path / "listy" / "result.json").write_text("[]")
    (tmp_path / "scalar").mkdir()
    (tmp_path / "scalar" / "result.json").write_text('"fail"')

    runs = triage_reward.load_runs(tmp_path)

    assert [src for src, _ in runs] == [str(tmp_path / "good" / "result.json")]
    # and the survivor is still labellable, which is the point of not aborting
    assert triage_reward.label_run(runs[0][1]).verdict == "fail"


def test_one_non_object_sanitizer_report_does_not_abort_the_sweep(
    triage_reward, tmp_path
):
    (tmp_path / "listy").mkdir()
    (tmp_path / "listy" / "sanitizer_report.json").write_text("[]")
    assert triage_reward.load_sanitizer_reports(tmp_path) == []


def test_an_infra_only_run_is_an_error_not_a_failure(triage_reward):
    doc = triage_reward._run("launch", "error", [], ["tier1:exec_failed"])
    assert triage_reward.label_run(doc).verdict == "error"


def test_a_failure_alongside_infra_noise_still_reproduced(triage_reward):
    """fail > error: the bug reproducing outranks the trial also being flaky."""
    doc = triage_reward._run("both", "fail", ["tier1:sigabrt"], ["tier1:timeout"])
    label = triage_reward.label_run(doc)
    assert label.verdict == "fail"
    assert "tier1:sigabrt" in label.failure_detectors
    assert "tier1:timeout" in label.error_detectors


def test_a_detector_recorded_on_the_wrong_side_is_re_partitioned(triage_reward):
    """The recorded split is recombined and re-split through aorta's own rule."""
    doc = triage_reward._run("misfiled", "error", ["tier1:timeout"], [])
    label = triage_reward.label_run(doc)
    assert label.error_detectors == ["tier1:timeout"]
    assert label.failure_detectors == []
    assert label.verdict == "error"


@pytest.mark.parametrize(
    "key", ["failure_detectors_fired", "error_detectors_fired"]
)
def test_a_detector_field_written_as_a_bare_string_is_refused(triage_reward, key):
    """One character per detector ID is corpus rot nothing downstream can see.

    `list(doc.get(key) or [])` accepted any iterable, and a writer that emitted
    `"failure_detectors_fired": "tier1:sigsegv"` instead of a one-element list
    got `['t', 'i', 'e', 'r', ...]`. Every character went through
    `partition_detectors` as an ID; none are known, so all fourteen sorted to
    the failure side, the trial was labelled a reproduction, and those
    fabricated IDs became the *ground truth* the attribution F1 is scored
    against -- a model naming the real detector scores zero against them.

    The label that comes out is well-formed and the verdict is a legal verdict,
    so the refusal has to happen here or not at all.
    """
    doc = triage_reward._run("stringy", "fail", [], [])
    doc[key] = "tier1:sigsegv"

    with pytest.raises(TypeError) as excinfo:
        triage_reward.label_run(doc)

    # The message has to name the field and the shape: the operator's next move
    # is to find the writer that emitted it.
    assert key in str(excinfo.value)
    assert "not a list" in str(excinfo.value)


def test_a_detector_list_holding_a_non_string_is_refused_too(triage_reward):
    """A list is the right container and still the wrong contents.

    `partition_detectors` matches IDs by prefix, so a `None` or an int in the
    list is an unmatchable value that lands on the failure side exactly as a
    stray character does -- same fabrication, one shape further in.
    """
    doc = triage_reward._run("listy", "fail", [], [])
    doc["error_detectors_fired"] = ["tier1:timeout", None, 7]

    with pytest.raises(TypeError) as excinfo:
        triage_reward.label_run(doc)

    assert "non-string detector ID(s)" in str(excinfo.value)
    # Both offenders named, not just the first: a writer emitting one bad type
    # usually emits the rest of them too.
    assert "None" in str(excinfo.value) and "7" in str(excinfo.value)


@pytest.mark.parametrize("absent", [{}, {"failure_detectors_fired": None}])
def test_a_detector_field_the_archive_never_recorded_is_still_empty(
    triage_reward, absent
):
    """Narrowness: absent is a real state, not a shape the writer got wrong.

    A clean run fires no failure detectors, and a missing key and a JSON null
    are both how an archive says so. Refusing either would reject the most
    common document in the corpus.
    """
    doc = triage_reward._run("clean", "pass", [], [])
    doc.pop("failure_detectors_fired")
    doc.update(absent)

    label = triage_reward.label_run(doc)

    assert label.verdict == "pass"
    assert label.failure_detectors == []


def test_a_well_formed_detector_list_is_untouched(triage_reward):
    """Narrowness: the shape the archive actually writes must still pass."""
    doc = triage_reward._run("good", "fail", ["tier1:sigsegv"], ["tier1:timeout"])
    label = triage_reward.label_run(doc)
    assert label.failure_detectors == ["tier1:sigsegv"]
    assert label.error_detectors == ["tier1:timeout"]


def test_a_run_with_a_rotted_detector_field_is_skipped_not_fatal(
    triage_reward, tmp_path
):
    """The sweep loses the rotted run, not the directory.

    `TypeError` is raised from `label_run`, which the sweep calls from outside
    the loader's `try` -- the same shape as the `AttributeError` that made one
    non-object `result.json` score none of the directory. So the loader checks
    the field itself and turns it into a named skip beside its other ones.
    """
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "result.json").write_text(
        json.dumps(triage_reward._run("ok", "fail", ["tier1:exit_nonzero"], []))
    )
    rotted = triage_reward._run("rotted", "fail", [], [])
    rotted["failure_detectors_fired"] = "tier1:sigsegv"
    (tmp_path / "rotted").mkdir()
    (tmp_path / "rotted" / "result.json").write_text(json.dumps(rotted))

    runs = triage_reward.load_runs(tmp_path)

    assert [src for src, _ in runs] == [str(tmp_path / "good" / "result.json")]
    assert triage_reward.label_run(runs[0][1]).verdict == "fail"


def test_a_correct_answer_earns_the_full_reward(triage_reward):
    doc = triage_reward._run("x", "fail", ["tier1:sigsegv", "tier3:amdgpu_reset"], [])
    label = triage_reward.label_run(doc)
    answer = triage_reward.Answer("fail", ["tier3:amdgpu_reset", "tier1:sigsegv"])
    score = triage_reward.score_answer(answer, label)

    assert score.verdict_correct is True
    assert score.attribution_f1 == pytest.approx(1.0)
    assert score.reward == pytest.approx(1.0)


def test_a_clean_run_needs_no_citation_to_score_perfectly(triage_reward):
    label = triage_reward.label_run(triage_reward._run("clean", "pass", [], []))
    score = triage_reward.score_answer(triage_reward.Answer("pass", []), label)
    assert score.reward == pytest.approx(1.0)


def test_the_right_verdict_with_an_invented_reason_is_docked(triage_reward):
    """The 'right answer, wrong reason' case the attribution term exists for."""
    label = triage_reward.label_run(
        triage_reward._run("x", "fail", ["tier1:exit_nonzero"], [])
    )
    score = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier9:invented"]), label
    )

    assert score.verdict_correct is True
    assert score.attribution_f1 == 0.0
    assert score.reward == pytest.approx(triage_reward.VERDICT_WEIGHT)
    assert score.reward < 1.0


def test_a_partly_right_citation_earns_partial_credit(triage_reward):
    label = triage_reward.label_run(
        triage_reward._run("x", "fail", ["tier1:sigsegv", "tier3:amdgpu_reset"], [])
    )
    half = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier1:sigsegv"]), label
    )
    none = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier9:invented"]), label
    )
    full = triage_reward.score_answer(
        triage_reward.Answer("fail", ["tier1:sigsegv", "tier3:amdgpu_reset"]), label
    )
    assert none.reward < half.reward < full.reward


def test_confusing_a_failure_for_an_infra_error_loses_the_verdict_term(triage_reward):
    label = triage_reward.label_run(
        triage_reward._run("x", "fail", ["tier1:exit_nonzero"], [])
    )
    score = triage_reward.score_answer(
        triage_reward.Answer("error", ["tier1:exit_nonzero"]), label
    )
    assert score.verdict_correct is False
    assert score.reward == pytest.approx(triage_reward.ATTRIBUTION_WEIGHT)


def test_the_always_pass_policy_scores_well_enough_to_need_reporting(triage_reward):
    """The floor a real policy has to clear, on a corpus skewed towards pass."""
    labelled = [
        (f"synthetic:{d['cell_name']}", triage_reward.label_run(d))
        for d in triage_reward.FIXTURES
    ]
    always_pass = triage_reward.score_policy(
        "always pass", lambda _: triage_reward.Answer("pass", []), labelled
    )
    oracle = triage_reward.score_policy(
        "oracle",
        lambda lb: triage_reward.Answer(lb.verdict, sorted(lb.cited_detectors)),
        labelled,
    )

    assert oracle["mean_reward"] == pytest.approx(1.0)
    # Reading nothing is worth real reward, which is exactly why the demo prints
    # this number next to the policy's.
    assert always_pass["mean_reward"] > 0.2
    assert always_pass["mean_reward"] < oracle["mean_reward"]


def test_the_fixtures_cover_all_three_verdicts(triage_reward):
    """A fixture set missing a class would hide the term that detects it."""
    verdicts = {triage_reward.label_run(d).verdict for d in triage_reward.FIXTURES}
    assert verdicts == {"pass", "fail", "error"}


def test_no_fixture_is_stale_against_the_current_rules(triage_reward):
    """The fixtures encode today's precedence; if this fails, the rules moved."""
    stale = [d["cell_name"] for d in triage_reward.FIXTURES
             if triage_reward.label_run(d).stale]
    assert stale == []


def test_every_fixture_cites_only_detectors_a_tier_can_emit(triage_reward):
    """A fixture citing an invented ID trains attribution on a fake vocabulary.

    The attribution term is scored against exactly these IDs, so a fixture that
    names something no classifier tier produces rewards the model for citing a
    detector it will never see in a real run. Collected from the tier modules'
    own constants rather than hard-coded, so a renamed detector fails here
    instead of rotting silently.
    """
    real = _real_detector_ids()
    assert real, "no detector constants found; the classifier layout moved"
    for doc in triage_reward.FIXTURES:
        label = triage_reward.label_run(doc)
        invented = sorted(label.cited_detectors - real)
        assert not invented, f"{doc['cell_name']} cites unknown {invented}"


def test_the_debugging_fixtures_cover_the_failure_shapes_the_agent_sees(
    triage_reward,
):
    """The vertical is debugging, so the corpus has to contain its shapes.

    One representative per autopsy category the proposal contract enumerates
    and the classifier can actually evidence. Without these the fixture set
    only exercises signal/exit-code failures, which is not what `aorta agent`
    is pointed at.
    """
    cited = set()
    for doc in triage_reward.FIXTURES:
        cited |= triage_reward.label_run(doc).cited_detectors
    for required in (
        "tier4:collective_timeout",  # rccl_hang
        "tier4:hip_error",  # illegal_mem
        "tier3:vm_l2_fault",  # illegal_mem, the underlying fault
        "tier3:thermal_throttle",  # thermal_throttle
        "tier3:xgmi_link_error",  # fabric
        "tier4:nan_signature",  # numerics, with a zero exit code
    ):
        assert required in cited, f"no fixture evidences {required}"


def test_an_advisory_warn_is_never_part_of_the_justification(triage_reward):
    """`tier3:vram_growth` is a warn, so citing it must not earn attribution.

    The reset-with-warn fixture carries it precisely as a red herring: a policy
    that lists every detector it can see, warns included, should be docked.
    """
    doc = next(
        d for d in triage_reward.FIXTURES if d["cell_name"] == "reset-with-warn"
    )
    label = triage_reward.label_run(doc)
    assert "tier3:vram_growth" not in label.cited_detectors
    assert label.verdict == "fail"

    everything = triage_reward.Answer(
        verdict="fail",
        detectors=sorted(label.cited_detectors | {"tier3:vram_growth"}),
    )
    exact = triage_reward.Answer(
        verdict="fail", detectors=sorted(label.cited_detectors)
    )
    assert (
        triage_reward.score_answer(everything, label).reward
        < triage_reward.score_answer(exact, label).reward
    )


# --------------------------------------------------------------------------- #
# proposal_reward: the contract `aorta agent` actually enforces
# --------------------------------------------------------------------------- #


def test_an_on_contract_proposal_reaches_the_top_tier(proposal_reward):
    proposal = proposal_reward.Proposal(
        "valid",
        json.dumps(
            {
                "category": "rccl_hang",
                "hypothesis": "collective timed out on every rank",
                "next_mitigations": ["nccl_launch_order_implicit"],
                "confidence": 0.6,
                "stop": False,
            }
        ),
        ["nccl_launch_order_implicit", "tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.tier == proposal_reward.MAX_TIER
    assert score.reward == 1.0
    assert score.consumer_outcome == "accepted"


def test_precision_prices_the_cells_the_loop_runs_not_the_names_written(
    proposal_reward,
):
    """`AgentPolicy.validate_step` collapses repeats before `run_agent_loop`
    iterates, so a repeated name is one cell, not two.

    The precision term's whole justification is the GPU cost a wide proposal
    incurs. Charging for a cell that will never be created is therefore not a
    harsher version of the same rule -- it prices work that does not happen, and
    the term stops meaning what its docstring says.
    """
    def score(names):
        return proposal_reward.score_proposal(
            proposal_reward.Proposal(
                "dupes",
                json.dumps(
                    {
                        "category": "rccl_hang",
                        "hypothesis": "collective timed out on every rank",
                        "next_mitigations": names,
                        "confidence": 0.6,
                        "stop": False,
                    }
                ),
                ["nccl_launch_order_implicit", "tf32_off"],
            )
        )

    once = score(["nccl_launch_order_implicit"])
    twice = score(["nccl_launch_order_implicit", "nccl_launch_order_implicit"])

    assert twice.n_mitigations == 2, "the written length is still reported"
    assert twice.n_cells == 1
    assert twice.precision == once.precision
    assert twice.reward == pytest.approx(once.reward)


def test_a_proposal_of_only_none_is_the_stop_the_consumer_reads(proposal_reward):
    """`validate_step` drops `none` -- it is the no-op baseline and already a
    cell -- so a proposal naming nothing else normalises to an empty list, which
    `run_agent_loop` reads as a stop. Scoring it as a one-cell proposal credited
    a step the loop will not take."""
    score = proposal_reward.score_proposal(
        proposal_reward.Proposal(
            "only-none",
            json.dumps(
                {
                    "category": "rccl_hang",
                    "hypothesis": "nothing to try",
                    "next_mitigations": ["none"],
                    "confidence": 0.6,
                    "stop": False,
                }
            ),
            ["nccl_launch_order_implicit"],
        )
    )
    assert score.n_mitigations == 1
    assert score.n_cells == 0
    assert score.stopped_at == "tier4_registry"


def test_a_hallucinated_mitigation_is_docked_and_would_stop_the_search(
    proposal_reward,
):
    """The defect this reward exists for.

    `LiteLLMProposer` filters unrecognised names out before the policy sees
    them, so a proposal naming only invented mitigations reaches the loop as a
    well-formed step with nothing to try, and the search ends. Nothing raises.
    The reward has to notice what the consumer does not.
    """
    proposal = proposal_reward.Proposal(
        "hallucinated",
        json.dumps(
            {
                "category": "rccl_hang",
                "hypothesis": "disable peer-to-peer",
                "next_mitigations": ["rccl_p2p_disable"],
                "confidence": 0.9,
                "stop": False,
            }
        ),
        ["nccl_launch_order_implicit", "tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.stopped_at == "tier4_registry"
    assert score.reward < 1.0
    assert score.consumer_outcome == "silent_stop"


def test_an_invented_category_is_rejected_loudly_by_the_consumer(proposal_reward):
    """A bad category is the one thing `AgentPolicy` raises on."""
    proposal = proposal_reward.Proposal(
        "bad category",
        json.dumps(
            {
                "category": "rccl_timeout",
                "hypothesis": "h",
                "next_mitigations": ["tf32_off"],
                "confidence": 0.5,
                "stop": False,
            }
        ),
        ["tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.stopped_at == "tier3_category"
    assert score.consumer_outcome == "policy_stop"


def test_coerced_fields_still_lose_reward_even_though_the_consumer_accepts(
    proposal_reward,
):
    """`AgentStep.from_dict` repairs a bad type; the contract still says no.

    This is why the consumer cannot be used as the oracle: it accepts a
    proposal whose confidence it silently zeroed.
    """
    proposal = proposal_reward.Proposal(
        "string confidence",
        json.dumps(
            {
                "category": "rccl_hang",
                "hypothesis": "h",
                "next_mitigations": ["tf32_off"],
                "confidence": "high",
                "stop": False,
            }
        ),
        ["tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.stopped_at == "tier2_schema"
    assert score.consumer_outcome == "accepted"


@pytest.mark.parametrize(
    ("why", "mitigations", "detail"),
    [
        ("ints", [1, 2], "next_mitigations[0]=int"),
        ("objects", [{"name": "tf32_off"}], "next_mitigations[0]=dict"),
        # The mixed case is the one a coercion hides best: the first name is
        # real, so the proposal looks right in every summary that prints one.
        ("a good name and a null", ["tf32_off", None], "next_mitigations[1]=NoneType"),
        ("a nested list", [["tf32_off"]], "next_mitigations[0]=list"),
    ],
)
def test_mitigation_names_that_are_not_strings_stop_at_the_schema_tier(
    proposal_reward, why, mitigations, detail
):
    """`REQUIRED_KEYS` can say `list`; it cannot say "a list of what".

    So a reply naming `[1, 2]` cleared tier 2, and tier 4 read it through
    `[str(m) for m in ...]` -- `1` became `"1"`, an object became its repr, and
    `get_mitigation` was asked about names the reply never wrote. The reply was
    then charged for hallucinating them, which is wrong in the *forgiving*
    direction: it earns two tiers of credit for a schema it did not meet, and
    the `unknown` list reports names with no source in the output, so a reader
    comparing the two cannot tell what the model actually said.

    Refused at tier 2, which is the tier that reads the raw object rather than
    the coerced `AgentStep` precisely so a repaired type is caught rather than
    scored. A hand-written `str()` three tiers down is that same repair.
    """
    proposal = proposal_reward.Proposal(
        why,
        json.dumps(
            {
                "category": "rccl_hang",
                "hypothesis": "h",
                "next_mitigations": mitigations,
                "confidence": 0.5,
                "stop": False,
            }
        ),
        ["tf32_off"],
    )
    score = proposal_reward.score_proposal(proposal)

    assert score.stopped_at == "tier2_schema", score.detail
    assert score.tier == 1
    assert detail in score.detail


def test_a_list_of_strings_is_still_a_schema_the_tier_accepts(proposal_reward):
    """Narrowness. The refusal is about the element type, not about the list.

    Including the empty list, which is a *legal* reply -- the implicit spelling
    of a decision to stop -- and has to keep reaching tier 4, where the ladder
    already has a verdict for it. A guard that refused it here would move that
    verdict and change what an empty proposal scores.
    """
    for mitigations, expected in ((["tf32_off"], "tier5"), ([], "tier4_registry")):
        proposal = proposal_reward.Proposal(
            f"{len(mitigations)} name(s)",
            json.dumps(
                {
                    "category": "rccl_hang",
                    "hypothesis": "h",
                    "next_mitigations": mitigations,
                    "confidence": 0.5,
                    "stop": False,
                }
            ),
            ["tf32_off"],
        )
        score = proposal_reward.score_proposal(proposal)
        assert score.tier >= 2, (mitigations, score.stopped_at, score.detail)
        assert score.stopped_at != "tier2_schema", (mitigations, score.detail)
        assert expected in (score.stopped_at or "tier5"), (mitigations, score.stopped_at)


def _sidecar(tmp_path, name="rl_sidecar_flag"):
    """A `--mitigations-file` sidecar defining one ad-hoc mitigation."""
    path = tmp_path / "mitigations.json"
    path.write_text(
        json.dumps({"version": 1, "mitigations": {name: {"RL_SIDECAR": "1"}}}),
        encoding="utf-8",
    )
    return path


_SIDECAR_PROPOSAL = json.dumps(
    {
        "category": "rccl_hang",
        "hypothesis": "h",
        "next_mitigations": ["rl_sidecar_flag"],
        "confidence": 0.5,
        "stop": False,
    }
)


def test_a_sidecar_mitigation_is_not_a_hallucination(proposal_reward, tmp_path):
    """Scored against a bare registry, `--mitigations-file` names read as invented.

    The loop the model answered ran with some `AgentPolicy`, and a sidecar puts
    names in that policy's registry view that are in no other. This module
    imported `get_mitigation` and `AgentPolicy` precisely so it could not drift
    from the consumer -- and then called both without the sidecars, which is
    the drift in a different place: the same names, resolved against a
    different registry.

    Both halves point the same way, which is what makes it costly. Tier 4 calls
    a runnable mitigation "unregistered ... silently dropped by the proposer",
    and the replay records `policy_stop` for a step the real policy returned
    normalised. So a run using ad-hoc mitigations trains the model away from
    the very names the operator supplied them to make available.
    """
    sidecar = _sidecar(tmp_path)
    candidates = ["rl_sidecar_flag", "none"]

    bare = proposal_reward.score_proposal(
        proposal_reward.Proposal("no sidecar", _SIDECAR_PROPOSAL, candidates)
    )
    assert bare.stopped_at == "tier4_registry", bare.detail
    assert "rl_sidecar_flag" in bare.detail
    assert bare.consumer_outcome == "policy_stop"

    scored = proposal_reward.score_proposal(
        proposal_reward.Proposal(
            "with sidecar",
            _SIDECAR_PROPOSAL,
            candidates,
            sidecar_files=(sidecar,),
        )
    )
    assert scored.tier == proposal_reward.MAX_TIER, (scored.stopped_at, scored.detail)
    assert scored.reward == 1.0
    assert scored.consumer_outcome == "accepted"


def test_a_corpus_row_carries_the_sidecars_its_loop_ran_with(
    proposal_reward, tmp_path
):
    """Two sources, and the row wins.

    The row records what that loop actually ran with; the CLI argument is the
    caller's guess for rows that recorded nothing. A corpus that mixes runs --
    which is the ordinary shape, since `--out` is one directory per sweep and
    not per policy -- needs both, and needs the per-row fact to be the one that
    decides.
    """
    sidecar = _sidecar(tmp_path)
    corpus = tmp_path / "proposal.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "proposal",
                    "workload_family": "synthetic_hip_lds",
                    "proposal": {
                        "name": name,
                        "raw": _SIDECAR_PROPOSAL,
                        "candidates": ["rl_sidecar_flag", "none"],
                        "tried": [],
                        **({"sidecar_files": [str(sidecar)]} if on_row else {}),
                    },
                }
            )
            for name, on_row in (("row-carries-it", True), ("row-is-silent", False))
        )
        + "\n",
        encoding="utf-8",
    )

    on_row, silent = (p for p, _ in proposal_reward.load_corpus(corpus))
    assert on_row.sidecar_files == (sidecar,)
    assert silent.sidecar_files == (), "a row that recorded none has none"

    # The argument fills the silent row and does not override the other.
    on_row, silent = (
        p for p, _ in proposal_reward.load_corpus(corpus, (tmp_path / "other.json",))
    )
    assert on_row.sidecar_files == (sidecar,)
    assert silent.sidecar_files == (tmp_path / "other.json",)


def test_the_scorer_cli_takes_the_sidecars_the_run_was_given(
    proposal_reward, tmp_path, capsys
):
    """End to end, because the threading is only worth anything at the entry point.

    `--mitigations-file` is spelled as `aorta agent` spells it and forwarded to
    the same place, so scoring a corpus from a sidecar run is the same command
    with the same flag rather than a second concept to learn.
    """
    sidecar = _sidecar(tmp_path)
    corpus = tmp_path / "proposal.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "kind": "proposal",
                "workload_family": "synthetic_hip_lds",
                "proposal": {
                    "name": "sidecar-name",
                    "raw": _SIDECAR_PROPOSAL,
                    "candidates": ["rl_sidecar_flag", "none"],
                    "tried": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        proposal_reward.main(["--corpus", str(corpus), "--json"]) == 0
    )
    bare = json.loads(capsys.readouterr().out)

    assert (
        proposal_reward.main(
            ["--corpus", str(corpus), "--json", "--mitigations-file", str(sidecar)]
        )
        == 0
    )
    supplied = json.loads(capsys.readouterr().out)

    def _tier(payload):
        rows = payload["proposals"] if "proposals" in payload else payload["scores"]
        return rows[0]["tier"]

    assert _tier(bare) < proposal_reward.MAX_TIER
    assert _tier(supplied) == proposal_reward.MAX_TIER


_LOOP_STATE = {
    "candidates": ["hsa_no_sdma", "hip_launch_blocking", "amd_log_level_4", "none"],
    "tried": ["hsa_no_sdma"],
}


def _proposal_corpus(tmp_path, **spec_overrides):
    """A one-row proposal corpus; an override of `KeyError` deletes the key."""
    spec = {
        "name": "p",
        "raw": _completion("illegal_mem", mitigations=("hip_launch_blocking",)),
        **_LOOP_STATE,
    }
    for key, value in spec_overrides.items():
        if value is KeyError:
            spec.pop(key, None)
        else:
            spec[key] = value
    corpus = tmp_path / "proposal.jsonl"
    corpus.write_text(json.dumps({
        "kind": "proposal", "workload_family": "f", "proposal": spec,
    }) + "\n", encoding="utf-8")
    return corpus


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidates", "hip_launch_blocking"),
        ("candidates", {"hip_launch_blocking": True}),
        ("candidates", ["hip_launch_blocking", 7]),
        ("candidates", None),
        ("candidates", KeyError),
        ("tried", "hsa_no_sdma"),
        ("tried", ["hsa_no_sdma", 7]),
        ("tried", None),
        ("tried", KeyError),
        ("sidecar_files", "extra.json"),
        ("sidecar_files", ["extra.json", 7]),
    ],
)
def test_a_proposal_row_whose_loop_state_is_not_a_list_of_names_is_refused(
    proposal_reward, tmp_path, field, value
):
    """`list()` read a string as one name per character.

    `"candidates": "hip_launch_blocking"` offered nineteen one-character
    names, so a valid proposal scored `tier5_available` rather than the row
    being refused. `tried` fails the other way, and so does leaving it out: an
    absent or null `tried` read as `[]` says nothing was tried, re-offers the
    mitigation the row says already ran, and a proposal naming it scored 1.0
    where the row as written scores 0.8. That is why absence is refused for
    these two, unlike the detector lists in `triage_reward` -- `build_corpus.py`
    has written both on every row since its first one.

    Refused rather than skipped, as `triage_reward.load_corpus` refuses: the
    `.jsonl` is generated, and a dropped row shrinks the scored set silently.
    """
    corpus = _proposal_corpus(tmp_path, **{field: value})

    with pytest.raises(ValueError) as excinfo:
        proposal_reward.load_corpus(corpus)

    message = str(excinfo.value)
    assert "proposal.jsonl:1" in message, message
    assert f"proposal.{field}" in message, message
    assert "build_corpus.py" in message, message


def test_a_well_formed_proposal_row_still_loads(proposal_reward, tmp_path):
    """Narrowness: the refusal is about shape, not about being empty or absent.

    An empty `tried` is the ordinary first step, and `sidecar_files` is
    optional -- absent, null or empty all mean the row recorded none, so the
    caller's argument fills it, exactly as before.
    """
    fallback = _sidecar(tmp_path)
    for sidecar in (KeyError, None, []):
        corpus = _proposal_corpus(tmp_path, tried=[], sidecar_files=sidecar)
        (proposal, _), = proposal_reward.load_corpus(corpus, (fallback,))
        assert proposal.candidates == _LOOP_STATE["candidates"]
        assert proposal.tried == []
        assert proposal.sidecar_files == (fallback,)
        assert proposal_reward.score_proposal(proposal).tier == (
            proposal_reward.MAX_TIER
        )

    # And the row as written scores what its loop state says: the tried
    # mitigation is not offered, which is the fact an absent `tried` erased.
    corpus = _proposal_corpus(
        tmp_path, raw=_completion("illegal_mem", mitigations=("hsa_no_sdma",))
    )
    (proposal, _), = proposal_reward.load_corpus(corpus)
    assert proposal_reward.score_proposal(proposal).stopped_at == "tier5_available"


def test_prose_around_the_object_earns_nothing(proposal_reward):
    proposal = proposal_reward.Proposal(
        "prose", 'Here you go: {"category": "rccl_hang"}', ["tf32_off"]
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.tier == 0
    assert score.reward == 0.0


def test_a_registered_but_unavailable_mitigation_is_docked_one_tier(
    proposal_reward,
):
    """Registered is not the same as offered.

    Re-proposing something already tried is silently filtered too, so it costs
    an iteration rather than raising.
    """
    body = {
        "category": "rccl_hang",
        "hypothesis": "h",
        "next_mitigations": ["hsa_no_sdma"],
        "confidence": 0.5,
        "stop": False,
    }
    proposal = proposal_reward.Proposal(
        "already tried",
        json.dumps(body),
        ["hsa_no_sdma", "tf32_off"],
        tried=["hsa_no_sdma"],
    )
    score = proposal_reward.score_proposal(proposal)
    assert score.tier == 4
    assert score.stopped_at == "tier5_available"
    assert score.consumer_outcome == "silent_stop"


def test_the_ladder_is_monotonic_so_being_more_wrong_never_pays_more(
    proposal_reward,
):
    """The tier ladder still spans 0 to MAX_TIER, and the ungraded part is flat.

    ``reward == tier / MAX_TIER`` used to hold for every fixture. It no longer
    does, deliberately: that identity is exactly what made the reward saturate,
    since it left the reward blind to *how* a tier was reached. It still holds
    wherever neither graded term applies -- a committed category and a
    mitigation list inside the free budget -- which covers every fixture that
    predates fixes 1 and 2, so a regression in the ungraded path is still
    caught here.
    """
    m = proposal_reward
    tiers = [m.score_proposal(f).tier for f in m.FIXTURES]
    assert min(tiers) == 0
    assert max(tiers) == m.MAX_TIER
    for f in m.FIXTURES:
        score = m.score_proposal(f)
        ungraded = (
            score.category_credit == 1.0
            and score.n_mitigations <= m.FREE_MITIGATIONS
        )
        if ungraded:
            assert score.reward == pytest.approx(score.tier / m.MAX_TIER)
        else:
            assert score.reward < score.tier / m.MAX_TIER


def test_the_reward_is_the_documented_sum_of_graded_tier_steps(proposal_reward):
    """Pins the formula, so the two graded terms cannot drift from the docstring."""
    m = proposal_reward
    for f in m.FIXTURES:
        score = m.score_proposal(f)
        expected = m.TIER_STEP * min(score.tier, 2)
        if score.tier >= 3:
            expected += m.TIER_STEP * score.category_credit
        if score.tier >= 4:
            expected += m.TIER_STEP * (score.tier - 3) * score.precision
        assert score.reward == pytest.approx(expected)


def test_every_fixture_earns_the_reward_it_is_meant_to(proposal_reward):
    """The companion to the tier map: four fixtures now share tier 5 and differ."""
    expected = proposal_reward._reward_expectations()
    actual = {
        f.name: proposal_reward.score_proposal(f).reward
        for f in proposal_reward.FIXTURES
    }
    assert set(actual) == set(expected)
    for name, want in expected.items():
        assert actual[name] == pytest.approx(want), name
    # The point of the whole exercise: tier 5 is no longer a single value.
    top = {
        f.name: proposal_reward.score_proposal(f).reward
        for f in proposal_reward.FIXTURES
        if proposal_reward.score_proposal(f).tier == proposal_reward.MAX_TIER
    }
    assert len(set(top.values())) > 1, top


# --------------------------------------------------------------------------- #
# Fix 1: declining to classify must not earn full marks
#
# `unknown` is a member of the closed autopsy set, so a proposal that refuses
# the classification task used to clear the tier that exists to test it and
# reach 1.0. The trap in fixing this is that on the committed corpus `unknown`
# is the *honest* answer a probe step can give on every scenario -- four are
# labelled `unknown` and the other five carry evidence-only labels a probe step
# may not assert -- so a penalty that pushes the policy towards a confident
# wrong label is worse than the saturation it removes. These tests pin the
# ordering that keeps that from happening.
# --------------------------------------------------------------------------- #


def _proposal(proposal_reward, category="rccl_hang", mitigations=None, **over):
    body = {
        "category": category,
        "hypothesis": "h",
        "next_mitigations": (
            ["nccl_launch_order_implicit"] if mitigations is None else mitigations
        ),
        "confidence": 0.5,
        "stop": False,
    }
    body.update(over)
    return proposal_reward.Proposal(
        "under test",
        json.dumps(body),
        ["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off", "xnack"],
    )


def test_declining_to_classify_no_longer_earns_full_marks(proposal_reward):
    """The headline of fix 1."""
    m = proposal_reward
    score = m.score_proposal(_proposal(m, category="unknown"))
    assert score.tier == m.MAX_TIER
    assert score.reward < 1.0
    assert score.category_credit == m.ABSTENTION_CREDIT
    assert score.abstained is True


def test_an_abstention_still_clears_the_tier_the_consumer_accepts(proposal_reward):
    """Docked, not rejected -- the reward must not disagree with `AgentPolicy`.

    `unknown` is in `AUTOPSY_CATEGORIES` and `validate_step` accepts it, so a
    grader that failed the tier would be scoring a contract the consumer does
    not enforce. That drift is the failure mode this module is built to avoid,
    which is why fix 1 grades the step instead of closing the set.
    """
    m = proposal_reward
    score = m.score_proposal(_proposal(m, category="unknown"))
    assert score.tier >= 3
    assert score.stopped_at == ""
    assert score.consumer_outcome == "accepted"


def test_declining_beats_leaving_the_closed_set(proposal_reward):
    """The ordering that stops fix 1 rewarding dishonesty.

    committed > declined > outside the set. The middle term is the load-bearing
    one: an honest `unknown` has to stay worth more than an invented category,
    or the reward pays a policy to guess its way out of the abstention penalty.
    """
    m = proposal_reward
    committed = m.score_proposal(_proposal(m, category="rccl_hang")).reward
    declined = m.score_proposal(_proposal(m, category="unknown")).reward
    outside = m.score_proposal(_proposal(m, category="rccl_timeout")).reward
    assert committed > declined > outside


def test_the_abstention_dock_is_one_partial_step_not_a_whole_tier(proposal_reward):
    """Pins how *small* the dock is, which is the actual design decision.

    Excluding `unknown` from tier 3's accepted set -- the first option the
    report offered -- would have dropped an abstention to 0.4 while any in-set
    category, right or wrong, still earned 1.0: a 0.6 gradient pointing at
    "invent a confident label". Grading the step instead costs one partial
    step. If this test starts failing upwards, that gradient is being rebuilt.
    """
    m = proposal_reward
    committed = m.score_proposal(_proposal(m, category="rccl_hang")).reward
    declined = m.score_proposal(_proposal(m, category="unknown")).reward
    dock = committed - declined
    assert dock == pytest.approx(m.TIER_STEP * (1.0 - m.ABSTENTION_CREDIT))
    # Strictly smaller than dropping the abstention a whole tier would be.
    assert dock < m.TIER_STEP
    # And far smaller than the exclusion alternative it was chosen over.
    assert dock < (committed - 2 * m.TIER_STEP)


# --------------------------------------------------------------------------- #
# Fix 2: hedging across the candidate set has to cost something
#
# `run_agent_loop` appends every proposed name to the mitigation axis and runs a
# probe cell for each, while charging the whole proposal one unit of the
# iteration budget -- so shotgunning is free against the limit the policy
# enforces and expensive in the resource the operator pays for.
# --------------------------------------------------------------------------- #


def test_a_shotgun_proposal_no_longer_ties_a_targeted_one(proposal_reward):
    """The headline of fix 2, at the size the real run produced."""
    m = proposal_reward
    offered = [f"m{i}" for i in range(17)]
    wide = m.precision_credit(len(offered))
    narrow = m.precision_credit(1)
    assert narrow == 1.0
    assert wide < narrow
    # Same tier, same category, different reward.
    one = m.score_proposal(_proposal(m, mitigations=["nccl_launch_order_implicit"]))
    many = m.score_proposal(
        _proposal(
            m,
            mitigations=["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off"],
        )
    )
    assert one.tier == many.tier == m.MAX_TIER
    assert many.reward < one.reward


def test_naming_a_primary_and_one_fallback_costs_nothing(proposal_reward):
    """The perversity a pure brevity term creates, neutralised at its own margin.

    With no correctness signal the reward cannot tell a right name from a wrong
    one, so any brevity term makes a one-name proposal beat a two-name proposal
    that *contains* the right name. `1/len` puts the largest step the term can
    produce exactly there. A free pair makes the two tie instead, so the reward
    never pays a policy to drop a correct name in order to look decisive.

    It only relocates the problem: at three names and up a confident wrong
    single name still wins. That needs a correctness signal, not a better shape.
    """
    m = proposal_reward
    assert m.FREE_MITIGATIONS >= 2
    one = m.score_proposal(_proposal(m, mitigations=["nccl_launch_order_implicit"]))
    pair = m.score_proposal(
        _proposal(m, mitigations=["nccl_launch_order_implicit", "hsa_no_sdma"])
    )
    assert pair.reward == pytest.approx(one.reward)
    # And the relocation is real, so it is pinned rather than left implicit.
    three = m.score_proposal(
        _proposal(
            m,
            mitigations=["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off"],
        )
    )
    assert three.reward < one.reward


def test_hedging_never_pays_more_than_being_precise(proposal_reward):
    m = proposal_reward
    credits = [m.precision_credit(n) for n in range(1, 41)]
    assert credits == sorted(credits, reverse=True)
    assert credits[0] == 1.0
    assert credits[-1] < 0.1


def test_precision_is_the_reciprocal_of_the_cell_count_past_the_free_pair(
    proposal_reward,
):
    """The cost model stated as arithmetic: k names is k probe cells."""
    m = proposal_reward
    for n in range(1, m.FREE_MITIGATIONS + 1):
        assert m.precision_credit(n) == 1.0
    for n in range(m.FREE_MITIGATIONS + 1, 25):
        assert m.precision_credit(n) == pytest.approx(m.FREE_MITIGATIONS / n)
    # An empty list never reaches the block, but the function is still total.
    assert m.precision_credit(0) == 0.0


def test_the_two_saturation_routes_are_now_separately_visible(proposal_reward):
    """Abstaining and shotgunning were both routes to 1.0; now they compound.

    The recorded model did both at once on most scenarios, so the reward has to
    dock both independently rather than collapsing them into one penalty.
    """
    m = proposal_reward
    wide = ["nccl_launch_order_implicit", "hsa_no_sdma", "tf32_off", "xnack"]
    clean = m.score_proposal(_proposal(m)).reward
    abstains = m.score_proposal(_proposal(m, category="unknown")).reward
    shotguns = m.score_proposal(_proposal(m, mitigations=wide)).reward
    both = m.score_proposal(
        _proposal(m, category="unknown", mitigations=wide)
    ).reward
    assert both < abstains < clean
    assert both < shotguns < clean


def test_a_constant_that_reads_nothing_is_no_longer_worth_a_diagnosis(
    proposal_reward,
):
    """The saturation, restated as the comparison that failed before.

    Both abstaining constants scored 1.0 on the shipped fixtures, tying the
    on-contract baseline. They no longer do.
    """
    rows = {row["policy"]: row for row in proposal_reward.baselines()}
    diagnosis = rows["always the same valid proposal"]["mean_reward"]
    assert diagnosis == 1.0
    for constant in ("always abstain, one mitigation", "always abstain, shotgun everything"):
        assert rows[constant]["mean_reward"] < diagnosis
    # The one that stays uncomfortably high, and is worth keeping in view: a
    # single-name abstention reads no input and still clears 0.85, because a
    # cheap answer is most of what a form reward can see.
    assert rows["always abstain, one mitigation"]["mean_reward"] > 0.85


def test_every_fixture_stops_where_it_is_meant_to(proposal_reward):
    """Pins each failure mode to its tier, so a loosened check is visible."""
    expected = proposal_reward._fixture_expectations()
    actual = {
        f.name: proposal_reward.score_proposal(f).tier
        for f in proposal_reward.FIXTURES
    }
    assert actual == expected


def test_the_categories_come_from_the_agent_not_a_copy(proposal_reward):
    """The closed set is imported, so adding a category updates the reward."""
    from aorta.agent.llm import AUTOPSY_CATEGORIES

    assert proposal_reward.AUTOPSY_CATEGORIES is AUTOPSY_CATEGORIES


def test_a_fixed_valid_proposal_scores_full_marks_without_diagnosing_anything(
    proposal_reward,
):
    """The ceiling of a format reward, stated as a test.

    If this ever fails, the reward has started measuring substance and the
    docstring's claim that it is only a gate is wrong.
    """
    rows = {row["policy"]: row for row in proposal_reward.baselines()}
    fixed = rows["always the same valid proposal"]
    assert fixed["mean_reward"] == 1.0
    assert fixed["accepted_rate"] == 1.0


# --------------------------------------------------------------------------- #
# triage_reward: the sanitizer label source
#
# Waitcheck and ConSan are the two tools the CIA architecture highlights, so
# their reports are on-domain evidence for the root-cause half rather than a
# separate concern. These run against the reports committed under
# recipes/sanitizers/survey, which are the only real labelled failure evidence
# in the tree.
# --------------------------------------------------------------------------- #

_SURVEY = Path(__file__).resolve().parents[2] / "recipes" / "sanitizers" / "survey"


def test_the_committed_sanitizer_reports_all_label(triage_reward):
    """Every committed report loads and yields a verdict from the tools' own set."""
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    assert len(labelled) == 6, "the committed survey reports moved or changed"
    verdicts = {label.verdict for _, label in labelled}
    assert verdicts <= {"pass", "warn", "fail", "not_checked", "error"}
    assert verdicts == {"pass", "warn", "error"}


def test_a_wait_hazard_is_cited_by_its_namespaced_code(triage_reward):
    """Attribution reads like a detector ID and cannot be confused with one."""
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    hazard = next(
        label for src, label in labelled if "gemm_f32_waitcheck" in src
    )
    assert hazard.verdict == "warn"
    assert hazard.cited_detectors == {"waitcheck:wait_hazard"}


def test_a_sanitizer_that_did_not_run_attributes_nothing(triage_reward):
    """An `error` verdict means no observation, so there is nothing to cite.

    Mirrors the probe side, where a trial that never validly ran is `error` and
    carries no failure detectors.
    """
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    errored = [label for src, label in labelled if "consan" in src]
    assert errored, "expected the ConSan reports to be present"
    for label in errored:
        assert label.verdict == "error"
        assert label.failure_detectors == []


def _mixed_verdict_report() -> dict:
    """A real `error` report with a real `warn` check spliced into it.

    Both halves are committed survey reports, so the finding shape, the
    sanitizer names and the verdicts are the ones aorta's own model produces --
    only the pairing is synthetic. It has to be: every committed `error` report
    is a ConSan load rejection whose checks found nothing, which is precisely
    why this defect survived six reports and a full test sweep.
    """
    import copy

    reports = _SURVEY / "reports"
    doc = json.loads(
        (reports / "gemm_f32_consan" / "sanitizer_report.json").read_text()
    )
    warned = json.loads(
        (reports / "gemm_f32_waitcheck" / "sanitizer_report.json").read_text()
    )
    check = copy.deepcopy(warned["checks"][0])
    # One finding rather than thirty-two, and none of them duplicated through
    # `kernel_results`: the count is not what is under test and a single code
    # makes the assertion below read as the one fact it is.
    check["findings"] = check["findings"][:1]
    check["kernel_results"] = []
    doc["checks"].append(check)
    return doc


def test_an_error_report_cites_nothing_even_when_a_lesser_check_found_something(
    triage_reward,
):
    """`error` outranks `warn`, so an `error` report can carry `warn` findings.

    `overall_verdict` is the max-ranked check verdict, so a run where ConSan
    failed to load *and* Waitcheck found hazards is an `error` report holding
    `waitcheck:wait_hazard`. Those codes are the evidence for the warning, not
    for the error -- nothing observed why the sanitizer did not run, because by
    definition it did not run to observe it.

    Filing them under `error_detectors` handed the scorer an oracle that paid
    full attribution credit for citing evidence of the wrong event. That is the
    "right answer, wrong reason" case the attribution term exists to dock,
    arriving through the ground truth rather than through the answer, where no
    amount of docking can reach it.
    """
    label = triage_reward.label_sanitizer_report(_mixed_verdict_report())

    assert label.verdict == "error"
    assert label.error_detectors == []
    assert label.failure_detectors == []
    assert label.cited_detectors == set()


def test_the_warn_half_of_that_report_still_cites_its_finding(triage_reward):
    """Narrowness: the rule is about the verdict, not about losing findings.

    The same check, with the ConSan error removed so the report ranks `warn`,
    has to cite exactly what it saw. If this fails, the fix above stopped
    citing evidence rather than stopping the miscitation, and the attribution
    half of the reward is measuring nothing on the one label source that has
    real findings in it.
    """
    doc = _mixed_verdict_report()
    doc["checks"] = [doc["checks"][-1]]
    doc["overall_verdict"] = "warn"
    doc["execution_status"] = "complete"

    label = triage_reward.label_sanitizer_report(doc)

    assert label.verdict == "warn"
    assert label.failure_detectors == ["waitcheck:wait_hazard"]
    assert label.cited_detectors == {"waitcheck:wait_hazard"}


def test_the_post_training_doc_states_the_live_category_count():
    """The reviewer entry point quotes a taxonomy size the code can contradict.

    It carried "this branch is behind main and still has eight" after the
    branch had merged the eleven-name set; the count is the one claim in that
    section a test can check, so it is checked against the set the graders
    import rather than against a literal.
    """
    from aorta.agent.llm import AUTOPSY_CATEGORIES

    text = (_REPO / "docs/tokenspeed-rl-post-training.md").read_text(encoding="utf-8")
    claimed = re.findall(r"taxonomy from 8 categories to (\d+)", text)
    assert claimed == [str(len(AUTOPSY_CATEGORIES))], claimed
    assert "still\nhas eight" not in text and "still has eight" not in text


def _fail_and_warn_report() -> dict:
    """The real Waitcheck `warn` check beside the ConSan check promoted to `fail`.

    No committed report ranks `fail`, so the failing half is synthetic: the
    real ConSan check with its verdict raised and one finding of its own,
    shaped like Waitcheck's but renamed so the two checks' evidence cannot be
    confused in the assertion. The model refuses two checks from one sanitizer,
    which is why the halves come from different ones.
    """
    import copy

    doc = _mixed_verdict_report()
    failed, warned = doc["checks"][0], doc["checks"][-1]
    finding = copy.deepcopy(warned["findings"][0])
    finding["sanitizer"] = failed["sanitizer"]
    finding["code"] = "synthetic_failure"
    failed.update(verdict="fail", state="ran", reason=None, returncode=None)
    failed["findings"] = [finding]
    doc["checks"] = [failed, warned]
    doc["overall_verdict"] = "fail"
    doc["execution_status"] = "complete"
    return doc


def test_a_fail_report_cites_only_the_checks_that_failed(triage_reward):
    """The error-case rule, one rank down.

    `codes` gathered findings from every check, so an overall `fail` that also
    carried a check which merely warned cited the warning's hazard as evidence
    for the failure -- right answer, wrong reason, in the ground truth.
    """
    label = triage_reward.label_sanitizer_report(_fail_and_warn_report())

    assert label.verdict == "fail"
    assert label.failure_detectors == ["consan:synthetic_failure"]
    assert label.cited_detectors == {"consan:synthetic_failure"}


def test_two_checks_at_the_overall_verdict_both_count(triage_reward):
    """Narrowness: the rule filters by verdict, it does not pick one check."""
    doc = _fail_and_warn_report()
    doc["checks"][1]["verdict"] = "fail"

    label = triage_reward.label_sanitizer_report(doc)

    assert label.verdict == "fail"
    assert sorted(label.failure_detectors) == [
        "consan:synthetic_failure", "waitcheck:wait_hazard",
    ]


def test_the_committed_survey_attribution_is_unchanged_by_the_verdict_filter(
    triage_reward,
):
    """No committed report mixes verdicts in a way the filter touches.

    Pinned so the claim in `label_sanitizer_report`'s comment -- no existing
    corpus row's ground truth moves -- is checked rather than remembered.
    """
    by_case = {
        Path(src).parent.name: sorted(label.failure_detectors)
        for src, label in triage_reward.load_sanitizer_reports(_SURVEY)
    }
    assert by_case["gemm_f32_waitcheck"] == ["waitcheck:wait_hazard"]
    assert {case: codes for case, codes in by_case.items() if codes} == {
        "gemm_f32_waitcheck": ["waitcheck:wait_hazard"],
    }


def test_a_rotted_report_is_rejected_rather_than_relabelled(triage_reward, tmp_path):
    """The corpus-rot signal for this source is aorta's own consistency check.

    `SanitizerReport.from_dict` recomputes the overall verdict from the checks
    and raises when the stored value contradicts it, so a report whose verdict
    has been tampered with cannot be silently trained on.
    """
    import copy

    source = next(_SURVEY.rglob("sanitizer_report.json"))
    doc = json.loads(source.read_text(encoding="utf-8"))

    tampered = copy.deepcopy(doc)
    tampered["overall_verdict"] = "pass" if doc["overall_verdict"] != "pass" else "fail"
    with pytest.raises((ValueError, KeyError, TypeError)):
        triage_reward.label_sanitizer_report(tampered)

    # And the loader skips it instead of aborting the whole corpus.
    (tmp_path / "sanitizer_report.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    assert triage_reward.load_sanitizer_reports(tmp_path) == []


def test_the_untampered_report_still_labels(triage_reward):
    """Guards the test above: rejection must be caused by the tampering."""
    source = next(_SURVEY.rglob("sanitizer_report.json"))
    doc = json.loads(source.read_text(encoding="utf-8"))
    label = triage_reward.label_sanitizer_report(doc, source=str(source))
    assert label.verdict == doc["overall_verdict"]
    assert label.stale is False


def test_scoring_works_unchanged_across_both_label_sources(triage_reward):
    """One scorer, two label spaces.

    The sanitizer vocabulary includes `warn`, which the probe split has no
    equivalent for, so the two spaces are deliberately not mapped onto each
    other. `score_answer` is string equality plus set F1, so it spans both
    without needing to know which it is looking at.
    """
    labelled = triage_reward.load_sanitizer_reports(_SURVEY)
    for _, label in labelled:
        oracle = triage_reward.Answer(label.verdict, sorted(label.cited_detectors))
        assert triage_reward.score_answer(oracle, label).reward == pytest.approx(1.0)

        wrong = triage_reward.Answer("pass" if label.verdict != "pass" else "fail", [])
        assert triage_reward.score_answer(wrong, label).reward < 1.0


# --------------------------------------------------------------------------- #
# build_corpus.py -- turning real sanitizer runs into scorable examples.
#
# The two properties worth pinning are the ones that would quietly ruin a
# corpus: counting lanes of one race as many examples, and losing the workload
# family that makes the corpus splittable.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def build_corpus():
    return _load("build_corpus")


def _build(build_corpus, tmp_path, results):
    out = tmp_path / "corpus"
    build_corpus.main([
        "--results", str(results),
        "--baselines", str(
            Path(__file__).resolve().parents[2]
            / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
        ),
        "--out", str(out),
    ])
    return out, json.loads((out / "manifest.json").read_text())


def test_the_committed_reports_build_into_a_corpus(build_corpus, tmp_path):
    _, manifest = _build(build_corpus, tmp_path, _SURVEY)
    assert manifest["scenarios"] == 6
    assert manifest["examples"]["triage"] == 6
    assert manifest["rejected_reports"] == 0


def test_a_corpus_with_nothing_in_it_is_a_failed_build(build_corpus, tmp_path, capsys):
    """Every report rejected wrote an empty corpus and exited 0.

    The `no sanitizer reports under ...` check above covers a wrong
    `--results` path. This is the other way to end up with nothing: reports
    were found and `triage_example` rejected all of them, which is a baselines
    file that does not match this results tree, a schema that moved, or a
    sweep from another workload. The build wrote two empty `.jsonl` files, a
    manifest saying `"scenarios": 0`, and told automation it had succeeded.

    Empty is worse than crashed here, because an empty corpus is a *valid*
    corpus: training on it is a no-op run that looks like a run, and the
    failure surfaces as a model that did not improve rather than as a build
    that did not build. `recipe_reward` refuses an empty corpus root one layer
    over for the same reason.
    """
    results = tmp_path / "results" / "consan-racy"
    results.mkdir(parents=True)
    # Well-formed JSON, not a sanitizer report -- the shape a schema change or
    # a results tree from another workload produces.
    (results / "sanitizer_report.json").write_text(
        json.dumps({"not": "a sanitizer report"}), encoding="utf-8"
    )
    out = tmp_path / "corpus"

    code = build_corpus.main([
        "--results", str(results.parent),
        "--baselines", str(
            Path(__file__).resolve().parents[2]
            / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
        ),
        "--out", str(out),
    ])
    err = capsys.readouterr().err

    assert code == 1, err
    assert "rejected" in err and str(out) in err, err
    # Refused before `mkdir`, so no later step finds a directory to mistake for
    # a corpus.
    assert not out.exists()


def _refuse(build_corpus, tmp_path, out):
    """Run a build that finds reports and rejects every one of them."""
    results = tmp_path / "rejected" / "consan-racy"
    results.mkdir(parents=True)
    (results / "sanitizer_report.json").write_text(
        json.dumps({"not": "a sanitizer report"}), encoding="utf-8"
    )
    return build_corpus.main([
        "--results", str(results.parent),
        "--baselines", str(
            Path(__file__).resolve().parents[2]
            / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
        ),
        "--out", str(out),
    ])


def test_a_non_object_report_costs_one_scenario_not_the_whole_build(
    build_corpus, tmp_path, capsys
):
    """One truncated artifact used to abort the corpus with a traceback.

    `json.loads` returns whatever the file holds, and `[]`, `"partial"`,
    `null` and `0` are all valid JSON. Only `OSError` and `JSONDecodeError`
    were caught, so a syntactically valid non-object reached `Scenario` and
    `doc.get("checks")` raised `AttributeError` -- uncaught, out through
    `main`, no corpus written at all.

    Costing the build every *other* scenario over one bad file is the wrong
    trade in both directions: a sweep of fifty reports with one truncated
    write produces nothing, and the traceback names `AttributeError` rather
    than the file. The bytes being JSON was never the question, so this is
    skipped on the same terms as a file that is not JSON at all.

    The surviving scenario is the point of the second half -- a rule that
    rejected the whole directory would satisfy the first assertion too.
    """
    results = tmp_path / "results"
    (results / "consan-racy").mkdir(parents=True)
    (results / "consan-racy" / "sanitizer_report.json").write_text(
        json.dumps(["not", "an", "object"]), encoding="utf-8"
    )
    for report in sorted(_SURVEY.rglob("sanitizer_report.json")):
        case = results / report.parent.name
        case.mkdir(parents=True)
        (case / "sanitizer_report.json").write_text(
            report.read_text(encoding="utf-8"), encoding="utf-8"
        )

    out = tmp_path / "corpus"
    code = build_corpus.main([
        "--results", str(results),
        "--baselines", str(
            Path(__file__).resolve().parents[2]
            / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
        ),
        "--out", str(out),
    ])
    err = capsys.readouterr().err

    assert code == 0, err
    assert "not a report" in err and "consan-racy" in err, err
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["scenarios"] == 6, manifest


def test_a_refused_rebuild_does_not_leave_the_previous_corpus_consumable(
    build_corpus, tmp_path, capsys
):
    """The fail-closed guarantee held only for a first build.

    `test_a_corpus_with_nothing_in_it_is_a_failed_build` asserts no directory
    is created, which is the whole story exactly once. On a rebuild the
    refusal came before `mkdir` and left the previous `triage.jsonl`,
    `proposal.jsonl` and manifest in place -- readable, well-formed, and now
    stale. The command exited 1 and the corpus on disk said otherwise, which
    is worse than either alone: a trainer pointed at `--out` cannot tell this
    from a build that succeeded, and trains on the run before last while the
    pipeline reports a failure nobody is looking at.
    """
    out, first = _build(build_corpus, tmp_path, _SURVEY)
    assert first["scenarios"] == 6
    capsys.readouterr()

    code = _refuse(build_corpus, tmp_path, out)
    err = capsys.readouterr().err

    assert code == 1, err
    assert "removed the previous corpus" in err, err
    for name in build_corpus.CORPUS_FILES:
        assert not (out / name).exists(), f"{name} survived a refused build"


def test_a_refused_build_leaves_a_directory_that_is_not_a_corpus_alone(
    build_corpus, tmp_path, capsys
):
    """The narrowness control: removing whatever `--out` names would pass above.

    `--out` is an operator-supplied path and a typo is the normal way it ends
    up somewhere that matters. Keyed on the manifest, so a directory this
    script never wrote a corpus into loses nothing -- not even a `*.jsonl`,
    which is the shape `discard_corpus` removes where a manifest is present.
    """
    out = tmp_path / "not-a-corpus"
    out.mkdir()
    bystander = out / "notes.txt"
    bystander.write_text("mine", encoding="utf-8")
    rows = out / "events.jsonl"
    rows.write_text("{}\n", encoding="utf-8")

    code = _refuse(build_corpus, tmp_path, out)
    err = capsys.readouterr().err

    assert code == 1, err
    assert "removed the previous corpus" not in err, err
    assert bystander.read_text(encoding="utf-8") == "mine"
    assert rows.read_text(encoding="utf-8") == "{}\n"


def _seed_previous_corpus(build_corpus, tmp_path):
    """A published corpus plus the neighbours a real `--out` accumulates."""
    out, manifest = _build(build_corpus, tmp_path, _SURVEY)
    assert manifest["scenarios"] == 6
    (out / "triage.v1.jsonl").write_text("older schema\n", encoding="utf-8")
    (out / "manifest.v0.json").write_text("{}\n", encoding="utf-8")
    (out / "README.md").write_text("provenance\n", encoding="utf-8")
    (out / "scenario_labels.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    return out


_COMMITTED_INPUTS = ["README.md", "scenario_labels.json"]


def test_a_refused_rebuild_removes_every_generated_name_not_just_three(
    build_corpus, tmp_path, capsys
):
    """`discard_corpus` removed `CORPUS_FILES`; `publish` removes `_is_generated`.

    So a refused rebuild left `triage.v1.jsonl` readable -- output from an
    older schema that a successful build would have deleted, kept by the path
    whose whole job is to leave nothing stale. The committed inputs beside it
    are the narrowness half and must survive either way.
    """
    out = _seed_previous_corpus(build_corpus, tmp_path)
    capsys.readouterr()

    code = _refuse(build_corpus, tmp_path, out)
    err = capsys.readouterr().err

    assert code == 1, err
    assert "removed the previous corpus" in err, err
    assert sorted(p.name for p in out.iterdir()) == _COMMITTED_INPUTS


def test_the_manifest_is_removed_last(build_corpus, tmp_path, monkeypatch):
    """An interrupted discard must leave the marker that says to finish it.

    The manifest is what `_holds_a_corpus` keys on. Removed first, a discard
    that died part-way would leave rows behind in a directory no later failure
    recognises as a corpus.
    """
    out = _seed_previous_corpus(build_corpus, tmp_path)
    order: list[str] = []
    real_unlink = Path.unlink

    def recording_unlink(self, *args, **kwargs):
        order.append(self.name)
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    assert build_corpus.discard_corpus(out) is True
    assert order[-1] == "manifest.json", order
    assert set(order) == {
        "triage.jsonl", "proposal.jsonl", "manifest.json",
        "triage.v1.jsonl", "manifest.v0.json",
    }


def _duplicate_results(tmp_path):
    tree = tmp_path / "dup"
    shutil.copytree(_SURVEY, tree / "a")
    shutil.copytree(_SURVEY, tree / "b")
    return tree


@pytest.mark.parametrize(
    ("failure", "raised"),
    [
        ("baselines-missing", FileNotFoundError),
        ("baselines-not-json", json.JSONDecodeError),
        ("duplicate-scenario-ids", "DuplicateScenario"),
    ],
)
def test_a_build_that_raises_still_discards_the_previous_corpus(
    build_corpus, tmp_path, capsys, failure, raised
):
    """The two refusals discarded; the failures beside them did not.

    A missing or unparseable --baselines, or two reports sharing a scenario
    id, raised straight out of `main` and left the last corpus consumable --
    the stale-corpus state the refusals prevent, reached by the commoner door.
    The exception still propagates: the discard is added, not substituted.
    """
    out = _seed_previous_corpus(build_corpus, tmp_path)
    capsys.readouterr()
    results, baselines = _SURVEY, _VERDICT_BASELINES
    if failure == "baselines-missing":
        baselines = tmp_path / "nope.json"
    elif failure == "baselines-not-json":
        baselines = tmp_path / "baselines.json"
        baselines.write_text("{not json", encoding="utf-8")
    else:
        results = _duplicate_results(tmp_path)
    if isinstance(raised, str):
        raised = getattr(build_corpus, raised)

    with pytest.raises(raised):
        build_corpus.main([
            "--results", str(results), "--baselines", str(baselines),
            "--out", str(out),
        ])

    assert "removed the previous corpus" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == _COMMITTED_INPUTS


@pytest.mark.parametrize("usage_error", ["run-meta-missing", "results-omitted"])
def test_a_usage_error_leaves_the_previous_corpus_alone(
    build_corpus, tmp_path, capsys, usage_error
):
    """Narrowness: exit 2 means the build never ran, so --out is not touched.

    A command line with one wrong argument is no evidence its --out is the
    directory the operator meant. Discarding here would turn a typo in
    --run-meta into the loss of whatever --out named.
    """
    out = _seed_previous_corpus(build_corpus, tmp_path)
    before = sorted(p.name for p in out.iterdir())
    argv = ["--baselines", str(_VERDICT_BASELINES), "--out", str(out)]
    if usage_error == "run-meta-missing":
        argv += ["--results", str(_SURVEY), "--run-meta", str(tmp_path / "meta.jsno")]

    with pytest.raises(SystemExit) as excinfo:
        build_corpus.main(argv)

    assert excinfo.value.code == 2
    assert "removed the previous corpus" not in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == before


def test_a_failed_publish_keeps_publishs_rollback(build_corpus, tmp_path, monkeypatch):
    """Narrowness: the discard covers the build, not the swap.

    `publish` rolls a failed write back to the previous directory, which is its
    own tested contract, and a failure writing to --out is the least promising
    moment to write to it again.
    """
    out = _seed_previous_corpus(build_corpus, tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def failing_publish(target, payload):
        raise OSError("disk full")

    monkeypatch.setattr(build_corpus, "publish", failing_publish)
    with pytest.raises(OSError, match="disk full"):
        build_corpus.main([
            "--results", str(_SURVEY), "--baselines", str(_VERDICT_BASELINES),
            "--out", str(out),
        ])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_a_build_that_dies_mid_write_does_not_publish_half_of_it(
    build_corpus, tmp_path
):
    """Writing in place published one build's triage beside another's proposals.

    Every failure after the first `open` left `--out` holding a mix of two
    builds, with the exit code saying the build failed and the directory
    saying it had not. The scenario ids line up well enough for `run_e2e` to
    key its label map on them, so nothing downstream detects the mismatch.

    Staged beside `--out` and swapped in, so the previous corpus is what
    survives a failed build rather than a splice of both. The payload here
    fails on the second file, which is the case in-place writing got wrong.
    """
    out = tmp_path / "corpus"
    out.mkdir()
    for name in build_corpus.CORPUS_FILES:
        (out / name).write_text(f"previous {name}\n", encoding="utf-8")

    with pytest.raises(TypeError):
        build_corpus.publish(
            out,
            {"triage.jsonl": "fresh\n", "proposal.jsonl": None},  # None: dies here
        )

    for name in build_corpus.CORPUS_FILES:
        assert (out / name).read_text(encoding="utf-8") == f"previous {name}\n"
    assert not list(tmp_path.glob(".corpus.*")), "a scratch directory was left behind"


def test_a_successful_rebuild_replaces_the_corpus_rather_than_merging_into_it(
    build_corpus, tmp_path
):
    """A stale file from the last build is gone, not left for a reader to find.

    In-place writing only ever truncated the three names it was about to
    write, so anything else a previous build (or a previous *version* of this
    script) had put there stayed, and stayed readable.
    """
    out = tmp_path / "corpus"
    out.mkdir()
    (out / "triage.v1.jsonl").write_text("from the version before\n", encoding="utf-8")

    _, manifest = _build(build_corpus, tmp_path, _SURVEY)

    assert manifest["scenarios"] == 6
    assert not (out / "triage.v1.jsonl").exists()
    assert (out / "triage.jsonl").is_file()


def test_a_rebuild_keeps_the_files_it_did_not_write(build_corpus, tmp_path):
    """The other half of the test above, and the one the swap got wrong.

    `publish` replaces the *directory*, not the three names in it, so a
    rebuild deleted everything else in `--out`. The `--out` in every
    documented invocation is `examples/rl/corpus`, which commits two files
    this script does not produce: `README.md`, the provenance record for the
    sweep the published measurements were taken against, and
    `scenario_labels.json`, hand-written ground truth that `.gitignore`
    re-admits by name. The labels are the worse loss -- they are input, and
    nothing regenerates them -- so a rebuild silently destroyed the one file
    in that directory a rebuild cannot reproduce.

    `_is_generated` is the line, not an allowlist of those two names: a
    committed input beside a future corpus survives without anyone
    remembering to extend a list, and stale `*.jsonl` output is still forfeit.
    """
    out = tmp_path / "corpus"
    out.mkdir()
    (out / "README.md").write_text("provenance\n", encoding="utf-8")
    (out / "scenario_labels.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    (out / "notes").mkdir()
    (out / "notes" / "sweep.txt").write_text("kept too\n", encoding="utf-8")
    # Generated, and from a naming scheme `CORPUS_FILES` no longer lists: still
    # destroyed, or this fix would have traded one silent staleness for another.
    (out / "triage.v1.jsonl").write_text("from the version before\n", encoding="utf-8")

    _, manifest = _build(build_corpus, tmp_path, _SURVEY)

    assert manifest["scenarios"] == 6
    assert (out / "README.md").read_text(encoding="utf-8") == "provenance\n"
    assert (
        out / "scenario_labels.json"
    ).read_text(encoding="utf-8") == '{"schema_version": 1}\n'
    assert (out / "notes" / "sweep.txt").read_text(encoding="utf-8") == "kept too\n"
    assert not (out / "triage.v1.jsonl").exists()
    assert (out / "triage.jsonl").is_file()


def test_a_failed_publish_leaves_the_files_it_did_not_write_alone(
    build_corpus, tmp_path
):
    """Carrying entries across must not put them at risk to do it.

    Moving them into staging would have been cheaper and would have emptied
    `--out` before the swap, so a failure between the two left the provenance
    record in a scratch directory the `finally` then deletes -- destroying it
    on exactly the path that was supposed to change nothing.
    """
    out = tmp_path / "corpus"
    out.mkdir()
    (out / "README.md").write_text("provenance\n", encoding="utf-8")
    for name in build_corpus.CORPUS_FILES:
        (out / name).write_text(f"previous {name}\n", encoding="utf-8")

    with pytest.raises(TypeError):
        build_corpus.publish(
            out,
            {"triage.jsonl": "fresh\n", "proposal.jsonl": None},  # None: dies here
        )

    assert (out / "README.md").read_text(encoding="utf-8") == "provenance\n"
    for name in build_corpus.CORPUS_FILES:
        assert (out / name).read_text(encoding="utf-8") == f"previous {name}\n"
    assert not list(tmp_path.glob(".corpus.*")), "a scratch directory was left behind"


def _not_a_directory(tmp_path, shape):
    """An `--out` that exists and is not a directory, and how to read it back."""
    target = tmp_path / "notes.txt"
    target.write_text("mine\n", encoding="utf-8")
    if shape == "file":
        return target
    link = tmp_path / "corpus"
    link.symlink_to(target.name if shape == "link-to-file" else "nowhere")
    return link


def _entries(directory):
    return sorted(
        (p.name, os.readlink(p) if p.is_symlink() else p.is_dir())
        for p in directory.iterdir()
    )


@pytest.mark.parametrize("shape", ["file", "link-to-file", "dangling-link"])
def test_an_out_that_is_not_a_directory_is_refused_not_moved_aside(
    build_corpus, tmp_path, capsys, shape
):
    """`--out notes.txt` hid the file rather than rejecting it.

    The swap renames whatever `--out` names, so the file became
    `.notes.txt.previous-<pid>` with a corpus directory in its place, and the
    cleanup's `rmtree(..., ignore_errors=True)` cannot remove a file -- so it
    stayed there, hidden. A link to a file went the same way. Refused at the
    command line, before the build, and again at the swap for any other caller.
    """
    out = _not_a_directory(tmp_path, shape)
    before = _entries(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        build_corpus.main([
            "--results", str(_SURVEY),
            "--baselines", str(
                Path(__file__).resolve().parents[2]
                / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
            ),
            "--out", str(out),
        ])
    assert excinfo.value.code == 2
    assert "is not a directory" in capsys.readouterr().err

    with pytest.raises(NotADirectoryError):
        build_corpus.publish(out, {"triage.jsonl": "fresh\n"})

    assert _entries(tmp_path) == before, "--out was moved, replaced or hidden"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "mine\n"


def test_a_link_to_a_directory_is_followed_rather_than_replaced(
    build_corpus, tmp_path
):
    """The other thing `ignore_errors` swallowed, and the worse one.

    The carry-across reads *through* a link while the swap renamed the link
    itself, and `rmtree` refuses a link, so the link was hidden like the file
    above -- and the directory it pointed at kept the previous corpus, readable
    by anything following the link, beside a build that reported success.

    Followed rather than refused, which is the narrowness half: a link is how
    a corpus directory gets put on a bigger disk, and it is where the reads
    already went. So the build lands in the target and the link survives.
    """
    real = tmp_path / "real"
    real.mkdir()
    for name in build_corpus.CORPUS_FILES:
        (real / name).write_text(f"previous {name}\n", encoding="utf-8")
    (real / "README.md").write_text("provenance\n", encoding="utf-8")
    (tmp_path / "corpus").symlink_to("real")

    _, manifest = _build(build_corpus, tmp_path, _SURVEY)

    assert manifest["scenarios"] == 6
    assert os.readlink(tmp_path / "corpus") == "real"
    assert json.loads((real / "manifest.json").read_text())["scenarios"] == 6
    assert (real / "triage.jsonl").read_text(encoding="utf-8").startswith("{")
    assert (real / "README.md").read_text(encoding="utf-8") == "provenance\n"
    assert _entries(tmp_path) == [("corpus", "real"), ("real", True)], (
        "a scratch directory or a hidden copy was left behind"
    )


_VERDICT_BASELINES = (
    Path(__file__).resolve().parents[2]
    / "recipes/sanitizers/fixtures/expected/verdict_baselines.json"
)


def _build_with_run_meta(build_corpus, tmp_path, run_meta):
    out = tmp_path / "corpus"
    argv = [
        "--results", str(_SURVEY), "--baselines", str(_VERDICT_BASELINES),
        "--out", str(out),
    ]
    if run_meta is not None:
        argv += ["--run-meta", str(run_meta)]
    return out, build_corpus.main(argv)


def _run_meta_file(tmp_path, body):
    path = tmp_path / "run_meta.json"
    path.write_text(body, encoding="utf-8")
    return path


def _not_a_run_meta_file(tmp_path, shape):
    if shape == "misspelled":
        _run_meta_file(tmp_path, '{"image": "img@sha256:abc"}')
        return tmp_path / "run_meta.jsno"
    if shape == "directory":
        (tmp_path / "run_meta.json").mkdir()
        return tmp_path / "run_meta.json"
    (tmp_path / "run_meta.json").symlink_to("nowhere.json")
    return tmp_path / "run_meta.json"


@pytest.mark.parametrize("shape", ["misspelled", "directory", "dangling-link"])
def test_a_run_meta_that_is_not_a_file_is_a_usage_error(
    build_corpus, tmp_path, capsys, shape
):
    """A misspelled `--run-meta` built a corpus without the provenance asked for.

    `args.run_meta.exists()` sent it down the branch an omitted flag takes, so
    the build exited 0 and every row's provenance was silently short the keys
    the caller had supplied -- no provenance, recorded as though none had been
    given. Refused before the build, like a wrong `--out`, so nothing is
    published.
    """
    with pytest.raises(SystemExit) as excinfo:
        _build_with_run_meta(
            build_corpus, tmp_path, _not_a_run_meta_file(tmp_path, shape)
        )
    assert excinfo.value.code == 2
    assert "--run-meta" in capsys.readouterr().err
    assert not (tmp_path / "corpus").exists(), "a corpus was published anyway"


@pytest.mark.parametrize(
    ("body", "said"),
    [
        ('["image"]', "JSON array"),
        ("null", "JSON null"),
        ('"img@sha256:abc"', "JSON string"),
        ("7", "JSON number"),
        ("image=img@sha256:abc", "could not be read as JSON"),
        ('{"report": "nightly-0910", "image": "x"}', "sets 'report'"),
    ],
)
def test_run_meta_that_cannot_be_recorded_is_refused_before_the_build(
    build_corpus, tmp_path, capsys, body, said
):
    """The object is splatted into every row's provenance, after `report`.

    A list or a `null` raised `TypeError` from `triage_example` once the whole
    results tree had been read. An object setting `report` was worse: it exited
    0 having replaced every example's pointer back to its own source report
    with one string.
    """
    with pytest.raises(SystemExit) as excinfo:
        _build_with_run_meta(build_corpus, tmp_path, _run_meta_file(tmp_path, body))
    assert excinfo.value.code == 2
    assert said in capsys.readouterr().err
    assert not (tmp_path / "corpus").exists(), "a corpus was published anyway"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (None, {}),
        ("{}", {}),
        ('{"image": "img@sha256:abc", "commit": "7b2d7a8c"}',
         {"image": "img@sha256:abc", "commit": "7b2d7a8c"}),
    ],
)
def test_usable_run_meta_reaches_every_example_and_the_manifest(
    build_corpus, tmp_path, body, expected
):
    """Narrowness: omitting the flag is still `{}`, and a real object lands.

    Every row keeps its own `report`, so the shared keys are added beside the
    per-example pointer rather than in place of it.
    """
    run_meta = None if body is None else _run_meta_file(tmp_path, body)
    out, code = _build_with_run_meta(build_corpus, tmp_path, run_meta)
    assert code == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run_meta"] == expected
    for name in ("triage.jsonl", "proposal.jsonl"):
        rows = [json.loads(x) for x in (out / name).read_text().splitlines() if x]
        assert rows
        for row in rows:
            provenance = dict(row["provenance"])
            report = provenance.pop("report")
            assert report.endswith("sanitizer_report.json"), report
            assert provenance == expected


def test_every_example_carries_a_workload_family(build_corpus, tmp_path):
    """Stratification is free now and expensive to retrofit, so it is enforced."""
    out, manifest = _build(build_corpus, tmp_path, _SURVEY)
    assert "unknown" not in manifest["workload_families"]
    for name in ("triage.jsonl", "proposal.jsonl"):
        rows = [json.loads(x) for x in (out / name).read_text().splitlines() if x]
        assert rows
        for row in rows:
            assert row["workload_family"] != "unknown"


def test_lanes_of_one_race_collapse_to_one_site(build_corpus, tmp_path):
    """64 findings from one race are one piece of evidence, not 64.

    Built by replaying a real finding across lane masks, which is exactly the
    shape ConSan emits for a two-wave LDS race: same instruction pair, one
    record per lane.
    """
    source = json.loads(
        (_SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json")
        .read_text()
    )
    # Reduced to the one check carrying the lanes, so the counts below are the
    # dedup rule and nothing else.
    check = source["checks"][0]
    source["checks"] = [check]
    template = (check.get("findings") or [])[0]
    check["kernel_results"] = []
    lanes = []
    for index in range(64):
        lane = json.loads(json.dumps(template))
        lane["metadata"] = dict(lane.get("metadata") or {})
        lane["metadata"].update({
            "first_inst": "0x8", "second_inst": "0x28", "kind": "1",
            "first_lane_mask": hex(1 << (index % 32)),
            "first_lds": f"[{index * 4},{index * 4 + 4})",
        })
        lanes.append(lane)
    check["findings"] = lanes

    results = tmp_path / "results" / "lds_reduce_consan"
    results.mkdir(parents=True)
    (results / "sanitizer_report.json").write_text(json.dumps(source))

    _, manifest = _build(build_corpus, tmp_path, tmp_path / "results")
    assert manifest["findings"]["raw"] == 64
    assert manifest["findings"]["distinct_sites"] == 1


def test_two_runs_sharing_a_case_name_are_refused(build_corpus, tmp_path):
    """The worst failure this corpus can have, because nothing detects it.

    `collect()` accepts an arbitrary tree and derives the id from the leaf
    directory, so an archived sweep holding `<run>/<case>/sanitizer_report.json`
    produced two scenarios with one id. `run_e2e` keys its label map and its
    GRPO groups on that id, so one report was scored against the other's label
    -- and both halves are individually well-formed, so nothing downstream
    could see it.

    Refused rather than disambiguated, because a path-derived id would vary
    with where `--results` points and these ids key recorded measurements.
    """
    results = tmp_path / "results"
    source = (
        _SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json"
    ).read_text()
    for run in ("runA", "runB"):
        case = results / run / "gemm_f32_waitcheck"
        case.mkdir(parents=True)
        (case / "sanitizer_report.json").write_text(source)

    with pytest.raises(build_corpus.DuplicateScenario, match="gemm_f32_waitcheck"):
        build_corpus.collect(results)


def test_ids_do_not_depend_on_which_root_was_given(build_corpus):
    """Stability is why the collision is refused instead of disambiguated.

    The same report has to carry the same id whether a rebuild points at
    `survey/` or at `survey/reports/`, because every recorded number is keyed
    on it. A relative-path id would have changed under the shallower root.
    """
    deep = {s.scenario_id for s in build_corpus.collect(_SURVEY / "reports")}
    shallow = {s.scenario_id for s in build_corpus.collect(_SURVEY)}

    assert deep == shallow
    assert "gemm_f32_waitcheck" in deep
    # And the id is still the leaf name, which is what the committed corpus and
    # the recorded rollouts use.
    assert all("/" not in i for i in deep), deep


def test_distinct_waitcheck_hazards_do_not_collapse(build_corpus, tmp_path):
    """The mirror of the test above, and the case it did not cover.

    Collapsing lanes of one race is right; collapsing distinct hazards is not,
    and the site key could not tell the difference because it read only the
    ConSan metadata keys. A Waitcheck finding has none of them -- `entry_offset`
    is null and the producer/consumer offsets are in `metadata.context_1` and
    `context_2` -- so every finding in a check hashed identically. The committed
    `gemm_f32_waitcheck` report carries 32 hazards at 32 distinct offset pairs
    and reported one site, discarding 31 evidence locations before anything
    downstream could see them.
    """
    out, manifest = _build(build_corpus, tmp_path, _SURVEY)
    assert manifest["findings"]["raw"] == 32
    assert manifest["findings"]["distinct_sites"] == 32

    row = next(
        json.loads(line)
        for line in (out / "triage.jsonl").read_text().splitlines()
        if json.loads(line).get("scenario_id") == "gemm_f32_waitcheck"
    )
    assert row["finding_counts"]["distinct_sites"] == 32
    # The evidence is the point of keeping them apart, so the row has to carry
    # 32 different places rather than 32 copies of one.
    contexts = {
        (e["metadata"].get("context_1"), e["metadata"].get("context_2"))
        for e in row["distinct_evidence"]
    }
    assert len(contexts) == 32


def test_the_corpus_scores_through_the_triage_scorer(
    build_corpus, triage_reward, tmp_path
):
    """The corpus is consumable with no conversion pass, and the oracle is perfect."""
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    rows = triage_reward.load_corpus(out / "triage.jsonl")
    assert len(rows) == 6
    for _, label, family in rows:
        assert family != "unknown"
        oracle = triage_reward.Answer(label.verdict, sorted(label.cited_detectors))
        assert triage_reward.score_answer(oracle, label).reward == pytest.approx(1.0)


def test_a_baseline_disagreement_is_emitted_but_not_scored_by_default(
    build_corpus, triage_reward, tmp_path
):
    """Two different jobs, and conflating them trains the defect.

    `build_corpus.py` keeps a row whose observed verdict contradicts the
    committed baseline, because that row is the evidence a tool defect happened.
    But the label on it is the *observed* verdict, so scoring it rewards a policy
    for reproducing the defect -- on exactly the scenarios where we already know
    the right answer and know the tool got it wrong. Emit, flag, and let the
    consumer skip.
    """
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    corpus = out / "triage.jsonl"
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]

    # No committed survey scenario is baseline-gated, so manufacture the
    # disagreement rather than waiting for a sweep that has one.
    gated = rows[0]
    gated["ground_truth"]["agrees"] = False
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    kept = triage_reward.load_corpus(corpus)
    everything = triage_reward.load_corpus(corpus, include_disagreements=True)

    assert len(everything) == len(rows)
    assert len(kept) == len(rows) - 1
    assert gated["example_id"] not in {example_id for example_id, _, _ in kept}
    assert gated["example_id"] in {example_id for example_id, _, _ in everything}


def test_an_ungated_scenario_is_not_treated_as_a_disagreement(
    build_corpus, triage_reward, tmp_path
):
    """`agrees` is `None` for a scenario no baseline covers, and most of the
    corpus is uncovered. Skipping those too would silently discard the majority
    of the data on a filter that reads as narrow."""
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    rows = [
        json.loads(line)
        for line in (out / "triage.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ungated = [r for r in rows if (r.get("ground_truth") or {}).get("agrees") is None]
    assert ungated, "precondition: the survey corpus has ungated scenarios"

    kept = {example_id for example_id, _, _ in triage_reward.load_corpus(out / "triage.jsonl")}
    assert {r["example_id"] for r in ungated} <= kept


def test_the_corpus_scores_through_the_proposal_scorer(
    build_corpus, proposal_reward, tmp_path
):
    """Every proposal variant lands on the tier it was synthesised to land on."""
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    rows = proposal_reward.load_corpus(out / "proposal.jsonl")
    assert rows
    by_variant = {}
    for proposal, _ in rows:
        by_variant.setdefault(proposal.name.rsplit(":", 1)[1], set()).add(
            proposal_reward.score_proposal(proposal).tier
        )
    assert by_variant["valid"] == {5}
    assert by_variant["hallucinated_name"] == {3}
    assert by_variant["invalid_category"] == {2}
    assert by_variant["already_tried"] == {4}


# --------------------------------------------------------------------------- #
# rescore_e2e: re-reading a recorded run under a changed grader
#
# The offline half of `run_e2e.py`. The property worth pinning hardest is the
# one that decides whether the acceptance test can be satisfied at all: a group
# whose completions are byte-identical cannot have within-group spread under
# *any* reward, so a zero there is a statement about the rollout and not about
# the grader. Conflating the two would have the reward blamed for a sampling
# defect, or a sampling fix credited to the reward.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rescore_e2e():
    return _load("rescore_e2e")


def _recorded(completions, candidates=None, tried=None):
    """A minimal `run_e2e.py` results document, one scenario per completion list."""
    candidates = candidates or ["hip_launch_blocking", "amd_log_level_4", "none"]
    tried = tried or []
    proposals = []
    for scenario, raws in completions.items():
        for index, raw in enumerate(raws):
            proposals.append({
                "scenario_id": scenario,
                "sample": index,
                "raw": raw,
                # The "before" reward, as the original run recorded it.
                "reward": 1.0,
                "tier": 5,
            })
    return {
        "meta": {
            "condition": "test",
            "model": "test-model",
            "candidates": candidates,
            "tried": tried,
        },
        "proposals": proposals,
    }


def _completion(category="unknown", mitigations=("amd_log_level_4",), hypothesis="h"):
    return json.dumps({
        "category": category,
        "hypothesis": hypothesis,
        "next_mitigations": list(mitigations),
        "confidence": 0.7,
        "stop": False,
    })


def test_rescoring_reads_the_raw_completion_not_the_stored_reward(rescore_e2e):
    """Otherwise the re-score would just echo the number it was meant to replace."""
    doc = _recorded({"s": [_completion()] * 3})
    rows = rescore_e2e.rescore_recorded(doc)
    assert len(rows) == 3
    for row in rows:
        assert row["reward_before"] == 1.0
        assert row["reward_after"] < 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("candidates", "hip_launch_blocking"), ("tried", "hsa_no_sdma"),
     ("candidates", ["hip_launch_blocking", 7]), ("tried", None)],
)
def test_a_results_file_whose_loop_state_is_not_a_list_is_refused(
    rescore_e2e, tmp_path, capsys, field, value
):
    """Every completion is re-scored against `meta`, so a string there re-grades the file.

    `list()` read `"hip_launch_blocking"` as nineteen one-character names, so
    every recorded completion naming a real mitigation fell to
    `tier5_available`, and a `tried` of characters re-offered the mitigation
    the run had already tried. Nothing in the output said so.

    Refused for the whole invocation rather than skipped, and the second half
    is why: a well-formed file that passes `--check-determinism` beside a
    malformed one would otherwise report success over a file it never read.
    """
    doc = _recorded({"s": [_completion()] * 2})
    doc["meta"][field] = value
    for measure in (rescore_e2e.rescore_recorded, rescore_e2e.analyse):
        with pytest.raises(ValueError, match=rf"meta\.{field}"):
            measure(doc)

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_recorded(
        {"s": [_completion(hypothesis=f"h{i}") for i in range(3)]}
    )), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")

    # The control: on its own the good file passes the determinism check.
    assert rescore_e2e.main([str(good), "--json", "--check-determinism"]) == 0
    capsys.readouterr()

    assert rescore_e2e.main(
        [str(good), str(bad), "--json", "--check-determinism"]
    ) == 2
    captured = capsys.readouterr()
    assert f"refusing {bad}" in captured.err, captured.err
    assert f"meta.{field}" in captured.err, captured.err
    assert captured.out == "", "a refused invocation printed results anyway"


def test_identical_completions_cannot_produce_within_group_spread(rescore_e2e):
    """The reason acceptance criterion 3 cannot be met on the recorded rollouts.

    A reward is a function of the completion and the loop state, and the loop
    state is constant inside a group. So five copies of one completion earn five
    copies of one reward, and the within-group spread GRPO needs is exactly zero
    no matter what the grader does. This is the rollout's defect, not the
    reward's: `LiteLLMProposer.propose` sends no temperature.
    """
    doc = _recorded({"s": [_completion()] * 5})
    result = rescore_e2e.analyse(doc)
    group = result["per_scenario"]["s"]

    assert group["distinct_completions"] == 1
    assert group["spread_within_group"] == 0.0
    assert result["criteria"]["3_within_group_spread_nonzero"]["holds"] is False


def test_differing_completions_do_produce_within_group_spread(rescore_e2e):
    """The guard on the test above: the zero must come from the data, not the grader.

    Same grader, same scenario, same loop state -- only the completions differ,
    and the spread appears. So the reward is not what blocks criterion 3, and a
    rollout sampled at a non-zero temperature would produce a usable advantage.
    """
    doc = _recorded({
        "s": [
            _completion(category="rccl_hang", mitigations=("amd_log_level_4",)),
            _completion(category="unknown", mitigations=("amd_log_level_4",)),
            _completion(
                category="unknown",
                mitigations=("amd_log_level_4", "hip_launch_blocking"),
            ),
            "not JSON at all",
        ]
    })
    result = rescore_e2e.analyse(doc)
    group = result["per_scenario"]["s"]

    assert group["distinct_completions"] == 4
    assert group["spread_within_group"] > 0.0
    assert result["criteria"]["3_within_group_spread_nonzero"]["holds"] is True


def test_the_constant_templates_are_scored_on_the_models_own_loop_state(
    rescore_e2e, proposal_reward
):
    """A constant measured against a different candidate set proves nothing."""
    doc = _recorded({"s": [_completion()]})
    result = rescore_e2e.analyse(doc)
    references = result["references"]

    # Two offered names, so the shotgun template names both.
    assert references["abstain_and_shotgun"]["n_mitigations"] == 2
    assert references["abstain_and_pick_first"]["n_mitigations"] == 1
    # Every constant is on contract except the prose one, which is the point:
    # they fail on substance, not on format.
    assert references["always_prose"]["tier"] == 0
    for name in ("oracle_contract_perfect", "abstain_and_shotgun", "honest_abstainer"):
        assert references[name]["tier"] == proposal_reward.MAX_TIER


def test_the_contract_perfect_reference_still_tops_the_ladder(rescore_e2e):
    """Criterion 2. It commits to a category and spends one cell, so it is 1.0.

    Worth reading with `REFERENCE_CATEGORY`'s comment: this reference is perfect
    against *this reward*, and on the committed corpus its category is knowingly
    wrong -- as is every category a probe step may commit to, because the race
    label the corpus uses is one a probe step may not assert.
    """
    doc = _recorded({"s": [_completion()]})
    result = rescore_e2e.analyse(doc)
    assert result["references"]["oracle_contract_perfect"]["reward"] == 1.0
    assert result["criteria"]["2_reference_at_top"]["holds"] is True


def _probe_step_refuses(category: str) -> bool:
    """Put the category to the real consumer, not to a copy of its set."""
    from aorta.agent.llm import AgentStep
    from aorta.agent.policy import AgentPolicy, PolicyViolation

    try:
        AgentPolicy().validate_step(
            AgentStep(
                category=category, hypothesis="h", next_mitigations=[],
                confidence=0.5, stop=False,
            )
        )
    except PolicyViolation:
        return True
    return False


def test_the_reference_category_is_one_a_probe_step_may_commit_to(
    rescore_e2e, proposal_reward
):
    """The reference is the contract's ceiling, so the consumer must accept it.

    Review suggested moving it to `gpu_race`, the label #484 gives the corpus's
    three race scenarios. That label is evidence-only, and on the tree with #484
    the reference then failed tier 4, reaching tier 3 for 0.6, so criterion 2 would
    have compared the model against something that is not the ceiling. Asked of
    `AgentPolicy` rather than of a copied category set, so a taxonomy change
    that withdraws the name from probe steps fails here -- and nothing pins the
    literal, since any committable category is an equally arbitrary ceiling.
    """
    from aorta.agent.llm import AUTOPSY_CATEGORIES

    category = rescore_e2e.REFERENCE_CATEGORY
    assert category in AUTOPSY_CATEGORIES
    assert category != proposal_reward.ABSTENTION_CATEGORY, "the reference commits"
    assert not _probe_step_refuses(category)


def test_no_category_a_probe_may_commit_to_is_right_on_any_labelled_scenario(
    rescore_e2e, proposal_reward
):
    """Holds `REFERENCE_CATEGORY`'s comment to the labels file it describes.

    The comment said "wrong on 8 of 9" and "the closed set has no category for
    that failure" after #484 had added `gpu_race` and labelled the corpus with
    it; the recount is 9 of 9. What makes the reference unfixable rather than
    merely stale is that the race label is refused from a probe step. If a label
    ever becomes one a probe step may commit to, this fails, and the reference
    should move onto it -- which is the change review asked for, once it is one
    the consumer would accept.

    The labels arrive with #484 through this PR's base, so this skips on a tree
    without them and runs on the PR's merge ref, which is what CI checks out.
    """
    path = _REPO / "examples/rl/corpus/scenario_labels.json"
    if not path.is_file():
        pytest.skip("scenario_labels.json is not on this tree; it lands with #484")
    labels = {
        name: row["category"]
        for name, row in json.loads(path.read_text(encoding="utf-8"))["scenarios"].items()
    }
    assert len(labels) == 9, sorted(labels)

    reference = rescore_e2e.REFERENCE_CATEGORY
    assert [name for name, label in labels.items() if label == reference] == []

    committed = set(labels.values()) - {proposal_reward.ABSTENTION_CATEGORY}
    assert committed, "every row is `unknown`; the comment's argument no longer applies"
    assert {c for c in committed if not _probe_step_refuses(c)} == set(), (
        "a scenario now carries a label a probe step may commit to; "
        "REFERENCE_CATEGORY should move onto it"
    )
    assert labels["consan-racy"] in committed


def test_an_abstaining_one_name_constant_is_the_hard_case_for_criterion_one(
    rescore_e2e,
):
    """The honest failure the re-score found, pinned so it cannot be lost.

    `abstain_and_pick_first` and `honest_abstainer` are the same policy up to
    the hypothesis text, and this reward cannot tell them apart -- it does not
    read the hypothesis. So the cheapest legal abstention scores what the best
    honest answer scores, and on a wide candidate set it beats a model that
    hedges. Fixing that needs a term the constant cannot satisfy.
    """
    doc = _recorded({"s": [_completion()]})
    result = rescore_e2e.analyse(doc)
    references = result["references"]
    assert (
        references["abstain_and_pick_first"]["reward"]
        == references["honest_abstainer"]["reward"]
    )


# --------------------------------------------------------------------------- #
# run_e2e: the rollout's sampling parameters
#
# The temperature and the seed live in the rollout driver rather than in
# `LiteLLMProposer`, and that placement is the point: the proposer is the
# production path, and a diagnostic tool that returns a different answer each
# time it is asked is worse, not better. These pin the placement and the
# per-sample derivation, plus the two aggregate fields that tell a sampling
# defect apart from a saturated reward.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def run_e2e():
    return _load("run_e2e")


def test_the_proposer_itself_still_sends_no_temperature(run_e2e):
    """The load-bearing constraint: `aorta agent` must stay reproducible.

    Sampling diversity is a property of how training data is collected. If it
    leaks into the shipped proposer, every real triage becomes stochastic --
    the same evidence stops yielding the same recommendation, and the nightly
    `llm_determinism` entry starts measuring noise. Asserted against the
    proposer's source rather than its behaviour, so it holds without a server.
    """
    import inspect

    from aorta.agent.llm import LiteLLMProposer

    source = inspect.getsource(LiteLLMProposer)
    assert "temperature" not in source
    assert "seed" not in source


def test_each_sample_in_a_group_gets_its_own_seed(run_e2e):
    """A single seed per run would leave the group identical, which is the bug."""
    seeds = [run_e2e.sample_seed(7, "consan-racy", i) for i in range(5)]
    assert len(set(seeds)) == 5


def test_the_seed_is_reproducible_from_one_integer(run_e2e):
    """Diversity that nobody can re-derive is not evidence."""
    first = [run_e2e.sample_seed(7, "consan-racy", i) for i in range(5)]
    again = [run_e2e.sample_seed(7, "consan-racy", i) for i in range(5)]
    assert first == again
    # And a different base gives a different rollout.
    assert first != [run_e2e.sample_seed(8, "consan-racy", i) for i in range(5)]


def test_seeds_do_not_collide_across_scenarios(run_e2e):
    """Otherwise two groups would share draws and the corpus would be smaller."""
    seeds = [
        run_e2e.sample_seed(7, scenario, i)
        for scenario in ("consan-racy", "consan-clean", "waitcheck", "waitcheck-tiny")
        for i in range(5)
    ]
    assert len(set(seeds)) == len(seeds)
    # Engines reject out-of-range seeds, so stay inside int32.
    assert all(0 <= s < 2**31 for s in seeds)


def test_the_seed_is_not_derived_from_the_salted_builtin_hash(run_e2e):
    """`hash()` is salted per process, so a run would not replay tomorrow."""
    import subprocess
    import sys

    code = (
        f"import sys; sys.path.insert(0, {str(_EXAMPLES)!r});"
        "from run_e2e import sample_seed;"
        "print(sample_seed(7, 'consan-racy', 0))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, "seed changed across processes"


def _proposal_rows(raws):
    return [
        {
            "scenario_id": scenario,
            "sample": index,
            "raw": raw,
            "reward": 1.0,
            "tier": 5,
            "failure_kind": "on_contract",
            "consumer_outcome": "accepted",
            "category_claimed": "unknown",
            "mitigations_claimed": ["amd_log_level_4"],
            "offered": ["hip_launch_blocking", "amd_log_level_4"],
            "transport_error": "",
        }
        for scenario, group in raws.items()
        for index, raw in enumerate(group)
    ]


@pytest.mark.parametrize(
    ("case", "kind"),
    [
        ("explicit-stop", "explicit_stop"),
        ("empty-list", "empty_mitigations"),
        ("normalises-to-no-cells", "no_cells_after_normalising"),
        ("unregistered-name", "hallucinated_mitigation"),
        ("policy-rejected-name", "policy_rejected"),
    ],
)
def test_each_tier4_refusal_has_its_own_failure_kind(
    run_e2e, proposal_reward, monkeypatch, case, kind
):
    """Four of tier 4's five refusals were counted as registry hallucinations.

    `failure_kind` matched `detail` for "no mitigation proposed" and called
    everything else in the tier `hallucinated_mitigation`, so a model that
    stopped, one that proposed only `none`, and one that named a registered
    but unsafe mitigation were all reported as inventing names. Scored end to
    end, so a stop site that forgets to set `tier4_reason` fails here.
    """
    m = proposal_reward
    if case == "explicit-stop":
        proposal = _proposal(m, stop=True)
    elif case == "empty-list":
        proposal = _proposal(m, mitigations=[])
    elif case == "normalises-to-no-cells":
        proposal = _proposal(m, mitigations=["none", "none"])
    elif case == "unregistered-name":
        proposal = _proposal(m, mitigations=["not_a_registered_mitigation"])
    else:
        unsafe = "sidecar/thing"
        for module in (m, importlib.import_module("aorta.agent.policy")):
            monkeypatch.setattr(module, "get_mitigation", lambda name, **kw: object())
        proposal = _proposal(m, mitigations=[unsafe])

    score = m.score_proposal(proposal)

    assert score.stopped_at == "tier4_registry", (score.stopped_at, score.detail)
    assert score.tier4_reason in m.TIER4_REASONS
    assert run_e2e.failure_kind(score) == kind


def test_an_unrecognised_tier4_reason_is_unknown_not_a_hallucination(
    run_e2e, proposal_reward
):
    """The fallback was the defect: an unexpected refusal must show as new."""
    score = proposal_reward.Score(tier=3, stopped_at="tier4_registry", detail="?")
    assert run_e2e.failure_kind(score) == "unknown"
    kinds = [run_e2e.failure_kind(proposal_reward.Score(
        tier=3, stopped_at="tier4_registry", tier4_reason=reason,
    )) for reason in proposal_reward.TIER4_REASONS]
    assert len(set(kinds)) == len(kinds), kinds


def test_the_other_tiers_keep_their_failure_kinds(run_e2e, proposal_reward):
    """Narrowness: only tier 4 moved off the prose."""
    m = proposal_reward
    assert run_e2e.failure_kind(m.score_proposal(_proposal(m))) == "on_contract"
    not_offered = m.score_proposal(_proposal(m, mitigations=["hip_launch_blocking"]))
    assert run_e2e.failure_kind(not_offered) == "registered_but_not_offered"
    assert run_e2e.failure_kind(m.score_proposal(
        m.Proposal("p", "not json", ["tf32_off"])
    )) == "malformed_json"
    assert run_e2e.failure_kind(
        m.score_proposal(_proposal(m, category="not_a_category"))
    ) == "category_outside_set"


def test_a_collapsed_group_is_reported_apart_from_a_degenerate_one(run_e2e):
    """The distinction that decides whether the reward or the rollout is at fault.

    Identical completions give zero spread under any reward, so a group that
    collapsed to one completion is a sampling defect. Zero spread across
    several distinct completions is the reward saturating. The first end-to-end
    run could not tell these apart and attributed all nine groups to the
    reward; both were true, and only one was fixable by grading.
    """
    collapsed = run_e2e.aggregate(_proposal_rows({"s": ["same"] * 5}), [])["proposal"]
    assert collapsed["per_scenario"]["s"]["distinct_completions"] == 1
    assert collapsed["collapsed_groups"] == 1
    assert collapsed["degenerate_groups"] == 1

    # Distinct completions that happen to score the same: degenerate, but the
    # rollout did its job, so the reward is what needs work.
    rows = _proposal_rows({"s": [f"different-{i}" for i in range(5)]})
    saturated = run_e2e.aggregate(rows, [])["proposal"]
    assert saturated["per_scenario"]["s"]["distinct_completions"] == 5
    assert saturated["collapsed_groups"] == 0
    assert saturated["degenerate_groups"] == 1


def test_the_effective_n_does_not_count_a_completion_twice(run_e2e):
    """45 draws that collapse to 9 completions are 9 observations, not 45."""
    collapsed = run_e2e.aggregate(
        _proposal_rows({f"s{g}": ["same"] * 5 for g in range(9)}), []
    )["proposal"]
    assert collapsed["n"] == 45
    assert collapsed["effective_n"] == 9

    sampled = run_e2e.aggregate(
        _proposal_rows({f"s{g}": [f"r{g}-{i}" for i in range(5)] for g in range(9)}), []
    )["proposal"]
    assert sampled["n"] == 45
    assert sampled["effective_n"] == 45


def test_a_verdict_outside_the_vocabulary_is_rejected(triage_reward, tmp_path):
    """A corpus written by a newer builder fails loudly, not as a silent mismatch."""
    corpus = tmp_path / "triage.jsonl"
    corpus.write_text(json.dumps({
        "kind": "triage", "example_id": "triage:x", "workload_family": "f",
        "label": {"verdict": "catastrophe", "failure_detectors": [],
                  "error_detectors": []},
    }) + "\n")
    with pytest.raises(ValueError, match="outside this scorer's vocabulary"):
        triage_reward.load_corpus(corpus)


@pytest.mark.parametrize("verdict", ["error", "not_checked"])
@pytest.mark.parametrize("field", ["error_detectors", "failure_detectors"])
def test_a_corpus_row_that_pins_findings_on_an_error_is_refused(
    triage_reward, tmp_path, verdict, field
):
    """The same rule, enforced on the way back in.

    `label_sanitizer_report` stopped citing finding codes for a report that
    says no sanitizer ran, but `load_corpus` reads a committed `.jsonl` and
    trains on the label stored in it. A corpus built before that fix carries
    the miscitation, and reading it back unexamined lets the defect outlive the
    commit that removed it -- in the one artifact nothing else inspects,
    because the report it was derived from need not travel with it.

    Refused rather than emptied, matching the vocabulary check above: a
    generated artifact one `build_corpus.py` run replaces is a corpus to
    rebuild, not a row to repair, and silently dropping the list would score
    the run against a ground truth the file does not contain.

    Both fields, though `error_detectors` is the one the old code populated.
    `cited_detectors` is their union, so a rule enforced on one half is a rule
    a row can walk past through the other.
    """
    corpus = tmp_path / "triage.jsonl"
    label = {"verdict": verdict, "failure_detectors": [], "error_detectors": []}
    label[field] = ["waitcheck:wait_hazard"]
    corpus.write_text(json.dumps({
        "kind": "triage", "example_id": "triage:x", "workload_family": "f",
        "label": label,
    }) + "\n")

    with pytest.raises(ValueError) as excinfo:
        triage_reward.load_corpus(corpus)

    message = str(excinfo.value)
    assert "triage.jsonl:1" in message, message
    assert "waitcheck:wait_hazard" in message, message
    # The fix is a rebuild, and nothing else in this function can tell the
    # reader that, so the message has to.
    assert "build_corpus.py" in message, message


def test_the_rows_that_corpus_check_is_not_about_still_load(triage_reward, tmp_path):
    """Narrowness, on both axes the refusal could over-reach along.

    An `error` row that cites nothing is what the builder now writes, and a
    `fail` or `warn` row citing its findings is the evidence the attribution
    half of the reward exists to measure. Refusing either turns a rule about
    two verdicts into a corpus that will not load at all.
    """
    corpus = tmp_path / "triage.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in [
        {"kind": "triage", "example_id": "triage:a", "workload_family": "f",
         "label": {"verdict": "error", "failure_detectors": [],
                   "error_detectors": []}},
        {"kind": "triage", "example_id": "triage:b", "workload_family": "f",
         "label": {"verdict": "fail", "failure_detectors": ["consan:race"],
                   "error_detectors": []}},
        {"kind": "triage", "example_id": "triage:c", "workload_family": "f",
         "label": {"verdict": "warn", "failure_detectors": ["waitcheck:wait_hazard"],
                   "error_detectors": []}},
    ]) + "\n", encoding="utf-8")

    labels = {example_id: label for example_id, label, _ in
              triage_reward.load_corpus(corpus)}

    assert set(labels) == {"triage:a", "triage:b", "triage:c"}
    assert labels["triage:a"].cited_detectors == set()
    assert labels["triage:b"].cited_detectors == {"consan:race"}
    assert labels["triage:c"].cited_detectors == {"waitcheck:wait_hazard"}


def _triage_corpus(tmp_path, **label):
    corpus = tmp_path / "triage.jsonl"
    corpus.write_text(json.dumps({
        "kind": "triage", "example_id": "triage:x", "workload_family": "f",
        "label": {"verdict": "fail", "failure_detectors": [],
                  "error_detectors": [], **label},
    }) + "\n", encoding="utf-8")
    return corpus


@pytest.mark.parametrize("verdict", ["fail", "error"])
@pytest.mark.parametrize("field", ["failure_detectors", "error_detectors"])
@pytest.mark.parametrize(
    "value", ["consan:1", {"consan:1": True}, ["consan:1", 7]],
    ids=["bare-string", "object", "non-string-entry"],
)
def test_a_corpus_row_whose_detector_list_is_not_a_list_of_ids_is_refused(
    triage_reward, tmp_path, verdict, field, value
):
    """`list()` made `"consan:1"` eight one-character detector IDs.

    On a `fail` row those became the ground truth attribution F1 is scored
    against, so a model citing the real detector scored against characters.
    `_detector_list` already refuses this shape on the `result.json` path; the
    corpus loader coerced it on the way back in.

    The `error` rows pin the ordering: the "error cites nothing" check reads
    the same two fields, and run ahead of validation it refused with the
    characters quoted back as the row's citation. The refusal has to name
    the shape, not a miscitation that is not in the file.
    """
    corpus = _triage_corpus(tmp_path, verdict=verdict, **{field: value})

    with pytest.raises(ValueError) as excinfo:
        triage_reward.load_corpus(corpus)

    message = str(excinfo.value)
    assert "triage.jsonl:1" in message, message
    assert field in message, message
    assert "build_corpus.py" in message, message
    assert "cites" not in message, message


def test_absent_or_null_detector_lists_still_load_as_empty(triage_reward, tmp_path):
    """Narrowness: absent is a real state for a detector list, unlike a string.

    A clean run fired no failure detectors, and `_detector_list` reads a
    missing key and a JSON null as that. Refusing them would reject rows that
    say exactly what the archive meant.
    """
    corpus = tmp_path / "triage.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in [
        {"kind": "triage", "example_id": "triage:a", "workload_family": "f",
         "label": {"verdict": "pass"}},
        {"kind": "triage", "example_id": "triage:b", "workload_family": "f",
         "label": {"verdict": "error", "failure_detectors": None,
                   "error_detectors": None}},
    ]) + "\n", encoding="utf-8")

    labels = {eid: label for eid, label, _ in triage_reward.load_corpus(corpus)}

    assert {eid: lb.cited_detectors for eid, lb in labels.items()} == {
        "triage:a": set(), "triage:b": set(),
    }


def test_a_corpus_row_whose_stale_flag_is_not_a_boolean_is_refused(
    triage_reward, tmp_path
):
    """`bool("false")` is true, so the row printed as STALE when it said it was not."""
    with pytest.raises(ValueError, match=r"triage\.jsonl:1: stale is a JSON str"):
        triage_reward.load_corpus(_triage_corpus(tmp_path, stale="false"))

    # Narrowness: both booleans, and absence, which is what `Label` defaults.
    for stale, expected in ((True, True), (False, False), (None, False)):
        (_, label, _), = triage_reward.load_corpus(
            _triage_corpus(tmp_path, stale=stale)
        )
        assert label.stale is expected
    (_, label, _), = triage_reward.load_corpus(_triage_corpus(tmp_path))
    assert label.stale is False


def test_shotgun_counts_the_names_offered_not_the_length_written(run_e2e):
    """A list can be as long as the offered set without covering it.

    `shotgun_all_offered` exists to make one of the two routes to a saturated
    reward visible -- naming everything, so no choice was made. Comparing list
    lengths counts a proposal that repeats one name as having named them all,
    which reports the opposite of what happened and puts the wrong number in a
    measurement record.
    """
    rows = _proposal_rows({"s0": ["a"], "s1": ["b"], "s2": ["c"]})
    rows[0]["mitigations_claimed"] = ["amd_log_level_4", "amd_log_level_4"]
    rows[1]["mitigations_claimed"] = ["hip_launch_blocking", "amd_log_level_4"]
    rows[2]["mitigations_claimed"] = ["amd_log_level_4"]

    summary = run_e2e.aggregate(rows, [])["proposal"]

    assert summary["shotgun_all_offered"] == 1, (
        "only the second proposal names both offered mitigations"
    )


def test_a_disagreement_row_is_dropped_before_it_costs_a_rollout(
    run_e2e, build_corpus, triage_reward, tmp_path, capsys
):
    """The driver has to agree with the scorer about which rows exist.

    `load_corpus` skips baseline disagreements, so driving the rollout from the
    raw JSONL sampled prompts that cannot be graded and then died indexing the
    label map -- after the GPU time had been spent. The filter is applied to
    both drives, not just triage: a report the baseline contradicts is the same
    report the proposal prompt is built from.
    """
    out, _ = _build(build_corpus, tmp_path, _SURVEY)
    corpus = out / "triage.jsonl"
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    rows[0]["ground_truth"]["agrees"] = False
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    labels = {
        example_id: label for example_id, label, _ in triage_reward.load_corpus(corpus)
    }
    assert f"triage:{rows[0]['scenario_id']}" not in labels

    kept = [r for r in rows if f"triage:{r['scenario_id']}" in labels]
    assert len(kept) == len(rows) - 1

    # The rows the driver keeps are exactly the rows it can grade, so the
    # indexing that used to raise cannot.
    for row in kept:
        assert labels[f"triage:{row['scenario_id']}"] is not None


def test_agreement_means_the_whole_baseline_contract_not_just_the_verdict(
    build_corpus, tmp_path
):
    """A report can keep its verdict and lose the evidence it is meant to cite.

    The committed baseline checks four things -- `overall_verdict`,
    `execution_status`, the per-sanitizer verdicts and the `finding_shape`
    substrings. Deciding agreement from the first alone marks a report that
    regressed on any of the other three as agreeing, and then hands its
    regressed evidence to a reward as ground truth.
    """
    source = _SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json"
    doc = json.loads(source.read_text(encoding="utf-8"))

    # `consan-racy` is one of the three case names build_corpus gates, so
    # placing the report there is what puts it under a baseline at all.
    results = tmp_path / "results" / "consan-racy"
    results.mkdir(parents=True)
    (results / "sanitizer_report.json").write_text(json.dumps(doc), encoding="utf-8")

    def build(baseline: dict) -> dict:
        baselines = tmp_path / f"baselines-{len(list(tmp_path.iterdir()))}.json"
        baselines.write_text(json.dumps({"consan_racy": baseline}), encoding="utf-8")
        out = tmp_path / f"corpus-{baselines.stem}"
        build_corpus.main([
            "--results", str(results.parent),
            "--baselines", str(baselines),
            "--out", str(out),
        ])
        line = (out / "triage.jsonl").read_text().splitlines()[0]
        return json.loads(line)["ground_truth"]

    matching = build(
        {
            "overall_verdict": doc["overall_verdict"],
            "execution_status": doc["execution_status"],
        }
    )
    assert matching["agrees"] is True
    assert matching["disagreements"] == []

    # Same top-level verdict, different execution status: invisible to a
    # verdict-only comparison, and a real regression.
    status_regressed = build(
        {"overall_verdict": doc["overall_verdict"], "execution_status": "timeout"}
    )
    assert status_regressed["agrees"] is False
    assert any("execution_status" in p for p in status_regressed["disagreements"])

    # Same verdict and status, but the finding the baseline names is gone.
    evidence_lost = build(
        {
            "overall_verdict": doc["overall_verdict"],
            "execution_status": doc["execution_status"],
            "finding_shape": {"waitcheck": "a finding this report does not carry"},
        }
    )
    assert evidence_lost["agrees"] is False
    assert any("finding_shape" in p or "shape" in p for p in evidence_lost["disagreements"])


@pytest.fixture(scope="module")
def nccl_roundtrip_check():
    return _load("nccl_roundtrip_check")


def _gen(text):
    return {"status": 200 if text is not None else 500, "text": text}


def _lifecycle(*, start=200, update=200, finish=200):
    """One start -> update -> finish step as `lifecycle_update` records it.

    All three legs, because all three now decide the verdict: a rejected start
    means the engine never entered the update state and a rejected finish means
    it may still be in it, so neither is a step whose completions say anything
    about weights.
    """
    return {
        "start": {"status": start},
        "update": {"status": update},
        "finish": {"status": finish},
    }


def test_the_round_trip_is_only_proven_when_both_completions_exist(
    nccl_roundtrip_check,
):
    """The failure this driver exists to prevent, reachable inside the driver.

    A failed generation carries `text=None`, which compares unequal to the
    baseline -- indistinguishable, to a bare comparison, from the weights
    having moved. Ordered naively, a 500 after the perturb followed by a
    healthy restore reports PROVEN: the strongest verdict this tool can give,
    from a completion B that never existed.
    """
    decide = nccl_roundtrip_check.decide_verdict
    baseline = _gen("A")

    proven, changed, recovered = decide(
        baseline=baseline,
        perturbed=_gen("B"),
        restored=_gen("A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )
    assert (proven, changed, recovered) == ("PROVEN", True, True)

    verdict, changed, recovered = decide(
        baseline=baseline,
        perturbed=_gen(None),
        restored=_gen("A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )
    assert verdict == "POST_UPDATE_GENERATION_FAILED"
    # Not `False` either: there was nothing to compare, and a boolean here
    # would read as an observation that was never made.
    assert changed is None and recovered is None


def test_a_rejected_restore_update_is_named_rather_than_scored(
    nccl_roundtrip_check,
):
    """`C == A` proves faithfulness only if the restore was actually applied.

    A restore whose `/update_weights` was rejected leaves the perturbed weights
    in place, so C equalling A would mean the perturb never landed -- the
    opposite of what PROVEN claims. Only the perturb status was checked before.
    """
    decide = nccl_roundtrip_check.decide_verdict
    verdict, _, _ = decide(
        baseline=_gen("A"),
        perturbed=_gen("B"),
        restored=_gen("A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(update=500),
    )
    assert verdict == "RESTORE_UPDATE_REJECTED"


def _peer_argv(tmp_path, rounds, rank="0", world_size="2", src=None):
    return [
        sys.executable,
        str(_EXAMPLES / "nccl_weight_peer.py"),
        "--run-id",
        "R1",
        "--rank",
        rank,
        "--world-size",
        world_size,
        *(() if src is None else ("--src", src)),
        "--master-address",
        "127.0.0.1",
        "--master-port",
        "29500",
        # Deliberately absent: it is what makes this test an ordering test. A
        # path that cannot be loaded fails *before* the old check ran and
        # *after* the new one does, so the message below appears only when the
        # validation precedes the load.
        "--model-path",
        str(tmp_path / "no-such-model"),
        "--tensors",
        "a,b",
        "--rounds",
        rounds,
        "--plan-out",
        str(tmp_path / "plan.json"),
    ]


@pytest.mark.parametrize(
    ("rounds", "expected"),
    [
        ("perturb,typo", "unknown round kind"),
        ("", "--rounds is empty"),
    ],
)
def test_a_bad_rounds_argument_is_rejected_before_the_group_is_joined(
    tmp_path, rounds, expected
):
    """A typo used to be caught only after this peer had joined the group.

    `build_round` raises on an unknown kind, and it is called from the round
    loop, which runs after `join_group` has returned. So a misspelled
    `--rounds` rendezvoused first and then exited without posting the broadcast
    the driver's `/update_weights` call is blocking on: the driver hung for its
    full timeout, with the engine left mid-update, on what is really a
    command-line error.

    The assertion is an ordering one rather than a message one. `--model-path`
    points at nothing, so loading the checkpoint fails before the old check was
    reached and after the new one is, which means this message can only appear
    if the validation now comes first. It also precedes the torch import, so a
    misspelled argument costs no import, no store and no group.
    """
    proc = subprocess.run(
        _peer_argv(tmp_path, rounds), capture_output=True, text=True, timeout=180
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, output
    assert expected in output, output
    assert not (tmp_path / "plan.json").exists(), "the plan must not be published"


def test_a_valid_rounds_argument_passes_the_new_check(tmp_path):
    """Narrowness: the check must reject typos, not the documented kinds.

    All three of `perturb`, `restore` and `recv` are kinds `build_round`
    materialises, so none of them may be refused. This gets past the validation
    and fails later on the unloadable model path, which is the point -- the
    rejection message must be absent.
    """
    proc = subprocess.run(
        _peer_argv(tmp_path, "perturb,restore,recv"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = proc.stdout + proc.stderr

    assert "unknown round kind" not in output, output
    assert "--rounds is empty" not in output, output


@pytest.mark.parametrize(
    ("rounds", "kwargs", "expected"),
    [
        # A group this peer is alone in: the rendezvous completes immediately
        # and there is no engine rank to broadcast to.
        ("perturb,restore", {"world_size": "1"}, "--world-size is 1"),
        ("perturb,restore", {"world_size": "0"}, "--world-size is 0"),
        # A rank outside the group. The store never sees the peer this one
        # thinks it is, so both halves wait out their own timeouts.
        (
            "perturb,restore",
            {"rank": "2", "world_size": "2"},
            "--rank is 2, which is not a rank in a --world-size 2 group",
        ),
        (
            "perturb,restore",
            {"rank": "-1"},
            "--rank is -1, which is not a rank in a --world-size 2 group",
        ),
        # A broadcast root that is in no rank's group is a root no rank can
        # match, so the collective cannot complete even once both sides join.
        # `recv` rather than a sending round on purpose: a sending round with
        # `--src != --rank` is already a role error, and its message names
        # `--src` too, so it would pass this assertion with no bounds check at
        # all. `recv` *wants* a `--src` that is not this peer, which leaves the
        # range as the only thing that can reject 5.
        (
            "recv",
            {"rank": "0", "world_size": "2", "src": "5"},
            "--src is 5, which is not a rank in a --world-size 2 group",
        ),
    ],
)
def test_a_rank_outside_the_group_is_rejected_before_the_rendezvous(
    tmp_path, rounds, kwargs, expected
):
    """`--world-size`, `--rank` and `--src` were taken entirely on trust.

    `rendezvous` reads `world_size` as how many peers to wait for and `rank` as
    which one this is, and validates neither against the other. So a typo in
    either published a plan that looked complete, the driver POSTed
    `/update_weights` against it, and both halves then sat in their own
    30-minute timeouts -- the same cost a misspelled `--rounds` used to have,
    for the same reason, one argument over.

    Ordering is the assertion again: `--model-path` points at nothing, so the
    load fails before any later check is reached. This message can only appear
    if the bounds are checked up here with the other argument errors, ahead of
    the torch import, the store and the group.
    """
    proc = subprocess.run(
        _peer_argv(tmp_path, rounds, **kwargs),
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, output
    assert expected in output, output
    assert not (tmp_path / "plan.json").exists(), "the plan must not be published"


def test_a_rank_inside_the_group_passes_the_bounds_check(tmp_path):
    """Narrowness: a peer at a non-zero rank in a wider group is legitimate.

    Rank 0 in a world of 2 is the smallest valid layout, not the only one --
    the module documents a TP=N engine occupying ranks `1..N` with
    `world_size` `N + 1`, so refusing anything but the default would reject
    every real topology. This gets past the bounds and fails later on the
    unloadable model path, which is the point.
    """
    proc = subprocess.run(
        _peer_argv(tmp_path, "perturb,restore", rank="3", world_size="8", src="3"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = proc.stdout + proc.stderr

    assert "--world-size is" not in output, output
    assert "is not a rank in a" not in output, output


def _serve_stubs(
    tmp_path,
    sampling="triton",
    grammar="xgrammar",
    models_code="200",
    models_body='{"data":[{"id":"Qwen/Qwen3-8B"}]}',
):
    """A PATH on which the real `up()` runs with no docker and no engine.

    The existing tests in this file all extract `backends` and drive it
    directly, which is why the defect below survived: the bug was not in
    `backends` but in how `up` called it, and no test had ever executed `up`.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "docker-calls.log"
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "docker $*" >> "{calls}"\n'
        'case "$1" in\n'
        # `-aq` is the name-collision probe and must answer empty; `-q` is the
        # liveness poll and must answer running.
        '  ps) for a in "$@"; do [ "$a" = "-aq" ] && exit 0; done; echo deadbeefcafe ;;\n'
        # `--cidfile` is modelled rather than ignored, because ownership is now
        # keyed on it: docker writes the id there when it *creates* the
        # container, which is what lets the guard be armed before this call.
        # Measured against docker 29.1.3 -- see `_CIDFILE` in the script.
        "  run)\n"
        '    prev=""; for a in "$@"; do\n'
        '      if [ "$prev" = "--cidfile" ]; then\n'
        '        if [ -e "$a" ]; then\n'
        "          echo 'docker: container ID file found, make sure the other"
        " container isn'\"'\"'t running or delete '\"$a\" >&2\n"
        "          exit 125\n"
        "        fi\n"
        "        printf '%s' 0123456789abcdef > \"$a\"\n"
        "      fi\n"
        '      prev="$a"\n'
        "    done\n"
        "    echo 0123456789abcdef ;;\n"
        '  logs) echo "stub container log line" ;;\n'
        "esac\nexit 0\n"
    )
    # `-w '\\n%{http_code}'` is appended by `models`, so the stub has to answer
    # in curl's shape: body, newline, status. Both halves are overridable: the
    # status so the HTTP-error path is reachable without a server, and the body
    # because a 200 is no longer the whole answer -- `models` reads the list
    # and requires `${MODEL}` to be on it.
    models_code = models_code or "200"
    assert "'" not in models_body, "the stub embeds the body in single quotes"
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        'url="${@: -1}"\n'
        'wants_code=0\n'
        'for a in "$@"; do case "$a" in *http_code*) wants_code=1 ;; esac; done\n'
        'case "$url" in\n'
        "  *get_server_info*) printf '%s' "
        f"'{{\"sampling_backend\":\"{sampling}\",\"grammar_backend\":\"{grammar}\"}}'"
        "; exit 0 ;;\n"
        "  *health*) echo 200; exit 0 ;;\n"
        "  *v1/models*)\n"
        f"    printf '%s' '{models_body}'\n"
        f"    [ \"$wants_code\" = 1 ] && printf '\\n%s' '{models_code}'\n"
        "    exit 0 ;;\n"
        "esac\nexit 0\n"
    )
    for name in ("docker", "curl"):
        (bin_dir / name).chmod(0o755)
    return bin_dir


def _serve_env(tmp_path, bin_dir):
    return {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "TS_OUT_DIR": str(tmp_path / "out"),
        "TS_LOG_DIR": str(tmp_path / "logs"),
        "TS_HF_HOME": str(tmp_path / "hf"),
        "TS_READY_SEC": "10",
    }


def test_a_failed_backends_check_still_writes_the_failure_log(tmp_path):
    """`set -e` made the `teardown_failed` branch after `backends` unreachable.

    `backends; rc=$?` reads as a status check and is not one: `set -euo
    pipefail` is in force, so a non-zero return from a bare function call exits
    the shell at that line. `rc=$?` never ran, the `teardown_failed` call under
    it never ran, and the EXIT trap's `_release_owned` handled the exit instead.

    The container still got removed, so the leak this script has been fixed for
    four times did not come back -- what was lost was the diagnosis.
    `_release_owned` does not capture `docker logs`, so a bring-up that failed
    on an unusable engine left no `server-failure.log` and no FAIL line naming
    which of the two backends was wrong, on the one path where the reason is
    the whole point.

    The exit code survived, because bash propagates a failing command's status
    through an EXIT trap. That is what made this look correct: the observable
    everyone checks was right and the observable that matters was gone.
    """
    bin_dir = _serve_stubs(tmp_path, sampling="greedy")
    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "up"],
        capture_output=True,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
        timeout=120,
    )
    output = proc.stdout + proc.stderr

    # `backends`'s own code, not a constant: 57 is the sampling verdict.
    assert proc.returncode == 57, output
    # `teardown_failed`'s message, which is what proves the branch was reached.
    # `_release_owned` prints "bring-up did not complete" instead.
    assert "engine is not usable for rollouts" in output, output
    # And the log it exists to capture. `docker rm -f` takes the container's
    # logs with it, so this file is the only record of why the engine was
    # rejected.
    assert (tmp_path / "logs" / "server-failure.log").is_file(), sorted(
        p.name for p in (tmp_path / "logs").iterdir()
    )


def test_a_usable_engine_still_brings_up_clean(tmp_path):
    """The narrowness check: the fix must not make bring-up fail generally.

    Same harness, same image, only the reported backends differ -- so a failure
    here would mean the new `|| rc=$?` broke the success path rather than that
    the engine was rejected.
    """
    bin_dir = _serve_stubs(tmp_path, sampling="triton")
    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "up"],
        capture_output=True,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_models_subcommand_does_not_pass_on_a_silent_gateway(tmp_path):
    """A failed fetch printed "(no response)" and returned 0.

    Found by auditing the class rather than reported: `curl ... || echo "(no
    response)"` makes the fallback the *last* command, so its zero status is
    the function's, and `serve_for_rollouts.sh models` answered "fine" for a
    gateway that said nothing. Same shape as `backends`'s old `|| echo '{}'`,
    which is already fixed and tested two functions further down.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text("#!/usr/bin/env bash\nexit 7\n")
    (bin_dir / "docker").write_text("#!/usr/bin/env bash\nexit 0\n")
    for name in ("docker", "curl"):
        (bin_dir / name).chmod(0o755)

    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "models"],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
        timeout=60,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "unverified" in proc.stdout + proc.stderr


def test_down_does_not_report_a_failed_removal_as_nothing_to_remove(tmp_path):
    """`down` said "no ts-rollout-serve" for two different things.

    `docker rm -f ... && echo removed || echo "no ${NAME}"` exits 0 whatever
    happens, and "no ${NAME}" means "there was nothing to remove" -- so a
    daemon this command could not reach reported the container as already gone
    while it was still running and still holding the GPU. Releasing the device
    is the entire job of this subcommand.

    The stub fails every `docker` call, which covers the probe as well as the
    removal: reading an unreachable `docker ps` as "nothing there" is the same
    defect one level down, and is how a first attempt at this would go wrong.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n"
        'echo "Cannot connect to the Docker daemon" >&2\nexit 1\n'
    )
    (bin_dir / "docker").chmod(0o755)

    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "down"],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
        timeout=60,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "no ts-rollout-serve" not in proc.stdout, output


def test_a_greedy_engine_is_fatal_to_the_rollout_server(tmp_path):
    """Warning here while the aorta side fails with exit 57 was the asymmetry.

    Two halves of one defect behaving differently is invisible unless someone
    reads both, which is how it survived. A greedy engine answers every sampled
    request 200 with the argmax, so a rollout driven against it produces a group
    of identical completions and an identically zero advantage -- the whole run
    wasted with nothing in it looking wrong.

    Drives the extracted `backends` function against a stub that reports greedy,
    rather than bringing up a container.
    """
    script = _EXAMPLES / "serve_for_rollouts.sh"
    body = subprocess.run(
        ["awk", "/^(_backend_field|backends)\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "backends" in body, "failed to extract the function"

    # A stub `curl` that answers /get_server_info with a greedy engine.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(
        '#!/usr/bin/env bash\necho \'{"sampling_backend":"greedy","grammar_backend":"xgrammar"}\'\n'
    )
    (bin_dir / "curl").chmod(0o755)

    harness = tmp_path / "drive.sh"
    # Both variables, because both are compared against what the engine
    # reports and both have a default in the script -- an unset `GRAMMAR` here
    # is a harness the real script never produces.
    harness.write_text(
        f'CONTROL=1\nSAMPLING=triton\nGRAMMAR=xgrammar\n{body}\nbackends\n'
    )
    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )
    output = proc.stdout + proc.stderr

    # 57, the same code and the same reading as `tokenspeed_serve`'s
    # `rollout_sampling_ignored`.
    assert proc.returncode == 57, output
    assert "FAIL" in output and "greedy" in output, output

    # And silent when the engine agrees, so the check is the mismatch and not
    # a function that always fails.
    # Both fields, because `backends` now validates both. The old stub named
    # only the sampler, so the grammar check -- which is the larger failure,
    # every `response_format` request returning 500 -- had nothing to read.
    (bin_dir / "curl").write_text(
        '#!/usr/bin/env bash\n'
        'echo \'{"sampling_backend":"triton","grammar_backend":"xgrammar"}\'\n'
    )
    (bin_dir / "curl").chmod(0o755)
    ok = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr


@pytest.mark.parametrize(
    ("info", "why"),
    [
        ('{"sampling_backend":"triton","grammar_backend":"none"}', "grammar none"),
        # Spaced, so the compact-JSON stub is not the reason this passes -- the
        # same trap that hid the sampling check's narrow match.
        (
            '{ "sampling_backend" : "triton" , "grammar_backend" : "none" }',
            "grammar none, spaced",
        ),
        ('{"sampling_backend":"triton"}', "grammar field absent"),
    ],
)
def test_a_none_grammar_backend_is_fatal(tmp_path, info, why):
    """The larger blast radius of the two backend checks.

    `LiteLLMProposer.propose` sends `response_format={"type": "json_object"}`
    on *every* call, and an engine on `--grammar-backend none` answers that
    with a 500 -- so the failure is total rather than degraded, and `up` was
    reporting such a server as ready. A greedy sampler at least returns text.

    Exit 59, not 57: "the engine ignored the sampling parameters" is a
    different statement, and this script has been corrected once already for
    giving two defects one verdict name.
    """
    script = _EXAMPLES / "serve_for_rollouts.sh"
    body = subprocess.run(
        ["awk", "/^(_backend_field|backends)\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(f"#!/usr/bin/env bash\nprintf '%s' '{info}'\n")
    (bin_dir / "curl").chmod(0o755)

    harness = tmp_path / "drive.sh"
    harness.write_text(
        f"CONTROL=1\nSAMPLING=triton\nGRAMMAR=xgrammar\n{body}\nbackends\n"
    )
    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )
    assert proc.returncode == 59, f"{why}: {proc.stdout + proc.stderr}"
    assert "grammar" in (proc.stdout + proc.stderr).lower(), why


@pytest.mark.parametrize(
    ("sampling", "grammar", "info", "code", "needle"),
    [
        # The case the greedy-only test let through: `greedy` asked for, and
        # an engine that did not apply it. Both conjuncts of the old check are
        # satisfied -- reported is not greedy -- so it exited 0.
        (
            "greedy",
            "xgrammar",
            '{"sampling_backend":"triton","grammar_backend":"xgrammar"}',
            60,
            "sampling-backend greedy",
        ),
        # Neither value is the degraded one, so no blocklist can catch this.
        (
            "triton",
            "xgrammar",
            '{"sampling_backend":"flashinfer","grammar_backend":"xgrammar"}',
            60,
            "sampling-backend triton",
        ),
        # The grammar side of the same hole.
        (
            "triton",
            "none",
            '{"sampling_backend":"triton","grammar_backend":"xgrammar"}',
            61,
            "grammar-backend none",
        ),
        # Spaced, so the compact-JSON stub is not what makes this pass.
        (
            "greedy",
            "xgrammar",
            '{ "sampling_backend" : "triton" , "grammar_backend" : "xgrammar" }',
            60,
            "sampling-backend greedy",
        ),
    ],
)
def test_a_backend_the_engine_did_not_apply_is_fatal_either_way(
    tmp_path, sampling, grammar, info, code, needle
):
    """The check was a blocklist and the question is an equality.

    `[ reported = greedy ] && [ SAMPLING != greedy ]` asks "did the engine
    fall back to the bad value", which coincides with "did the override apply"
    only when the requested value is the good one. So `TS_SAMPLING=greedy` --
    a deterministic control run, the case where pinning the backend matters
    most -- served by an engine reporting `triton` passed, and the run was
    recorded as deterministic while being sampled.

    Declining an override is not hypothetical: an unrecognised backend name or
    a build without that kernel is exactly when the flag was worth passing,
    and it is the case the engine answers by ignoring it.

    Distinct codes per backend, continuing what 57 and 59 started: `up`
    propagates `backends`'s own status, so the number alone tells the two
    halves of a bring-up failure apart.
    """
    script = _EXAMPLES / "serve_for_rollouts.sh"
    body = subprocess.run(
        ["awk", "/^(_backend_field|backends)\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(f"#!/usr/bin/env bash\nprintf '%s' '{info}'\n")
    (bin_dir / "curl").chmod(0o755)

    harness = tmp_path / "drive.sh"
    harness.write_text(
        f"CONTROL=1\nSAMPLING={sampling}\nGRAMMAR={grammar}\n{body}\nbackends\n"
    )
    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode == code, output
    assert needle in output, output


def test_an_engine_that_applied_both_overrides_is_silent(tmp_path):
    """Narrowness for the equality above: agreement is still exit 0.

    Including `greedy` and `none` on both sides, which the widened check has
    to keep allowing -- they are legitimate requests, and a check that refused
    the values rather than the disagreement would ban a deterministic control
    run outright.
    """
    script = _EXAMPLES / "serve_for_rollouts.sh"
    body = subprocess.run(
        ["awk", "/^(_backend_field|backends)\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(
        '#!/usr/bin/env bash\n'
        'printf \'%s\' \'{"sampling_backend":"greedy","grammar_backend":"none"}\'\n'
    )
    (bin_dir / "curl").chmod(0o755)

    harness = tmp_path / "drive.sh"
    harness.write_text(
        f"CONTROL=1\nSAMPLING=greedy\nGRAMMAR=none\n{body}\nbackends\n"
    )
    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_an_explicitly_requested_none_grammar_is_allowed(tmp_path):
    """`TS_GRAMMAR=none` is a caller saying they do not need JSON responses.

    The check is a mismatch between what was asked for and what was applied,
    not a ban on the value -- the same shape as the sampling check, where
    `greedy` is legitimate if greedy is what was requested.
    """
    script = _EXAMPLES / "serve_for_rollouts.sh"
    body = subprocess.run(
        ["awk", "/^(_backend_field|backends)\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(
        '#!/usr/bin/env bash\n'
        'echo \'{"sampling_backend":"triton","grammar_backend":"none"}\'\n'
    )
    (bin_dir / "curl").chmod(0o755)

    harness = tmp_path / "drive.sh"
    harness.write_text(f"CONTROL=1\nSAMPLING=triton\nGRAMMAR=none\n{body}\nbackends\n")
    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    ("stub", "why"),
    [
        ("exit 7", "curl failed"),
        ("echo ''", "empty body"),
        ("echo '{}'", "no sampling_backend field"),
        ("echo '{\"grammar_backend\": \"xgrammar\"}'", "other fields only"),
        # Pretty-printed and spaced: a genuinely greedy engine in a shape the
        # old compact-only grep read as non-greedy.
        ('echo \'{"sampling_backend": "greedy"}\'', "spaced greedy"),
    ],
)
def test_backends_does_not_fail_open(tmp_path, stub, why):
    """Every way of not knowing must fail, not pass.

    `|| echo '{}'` made a curl failure, a timeout or an empty body count zero
    greedy matches -- so "could not check" read exactly like "checked and it is
    fine", and `up` left the GPU server running. The compact-only match was the
    same hole in a different place: `json.dumps` writes `": "` by default, so a
    greedy engine answering in pretty JSON slipped through, and the test stub
    happened to emit the compact form which is why it looked fine.
    """
    script = _EXAMPLES / "serve_for_rollouts.sh"
    body = subprocess.run(
        ["awk", "/^(_backend_field|backends)\\(\\) \\{/,/^\\}$/", str(script)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "curl").write_text(f"#!/usr/bin/env bash\n{stub}\n")
    (bin_dir / "curl").chmod(0o755)

    harness = tmp_path / "drive.sh"
    harness.write_text(
        f"CONTROL=1\nSAMPLING=triton\nGRAMMAR=xgrammar\n{body}\nbackends\n"
    )
    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )
    assert proc.returncode == 57, f"{why}: {proc.stdout + proc.stderr}"


def test_the_coordination_key_never_reaches_the_engine(nccl_roundtrip_check, monkeypatch):
    """The plan's `run_id` is for the driver, not for `update_info`.

    Stamping it on the plan to defeat a stale file -- which is what the
    previous commit did -- put it into the POST body, and the engine rejects an
    unknown `update_info` key with a 500. So every round failed before a single
    broadcast: the fix for one silent failure created a loud one, and only on
    hardware, where no test here would have seen it.
    """
    sent: list[dict] = []

    def fake_call(base, method, path, body=None, timeout=None):
        sent.append({"path": path, "body": body})
        return 200, {"ok": True}, 0.01

    monkeypatch.setattr(nccl_roundtrip_check, "call", fake_call)
    plan = {
        "run_id": "this-run",
        "names": ["w"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
    }
    nccl_roundtrip_check.lifecycle_update("http://c", plan, "perturb")

    update = next(s for s in sent if s["path"] == "/update_weights")
    info = update["body"]["update_info"]
    assert "run_id" not in info, info
    # And nothing else was dropped with it.
    assert set(info) == {"names", "dtype_names", "shapes"}
    # The plan itself is untouched, since the driver still needs the id.
    assert plan["run_id"] == "this-run"


def test_a_stale_plan_is_not_this_runs_plan(nccl_roundtrip_check, tmp_path, capsys):
    """`--plan` is a fixed shared path, so a leftover plan is the normal state.

    Accepting it meant reading a previous run's tensor names and shapes while
    this run's peer wrote its own, then mismatching the HTTP update against the
    collective actually broadcast -- which hangs rather than failing, so it
    would not even have reported.
    """
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"run_id": "an-older-run", "names": ["w"]}))

    found, stale = nccl_roundtrip_check.wait_for_plan(plan, "this-run", 0)

    assert found is None
    # Reported apart from "nothing appeared", because the two send an operator
    # to different places: clear the path, versus the peer never got that far.
    assert stale == "an-older-run"

    # The same path, once this run's peer publishes, is accepted immediately.
    plan.write_text(json.dumps({"run_id": "this-run", "names": ["w"]}))
    found, stale = nccl_roundtrip_check.wait_for_plan(plan, "this-run", 0)
    assert found is not None and found["names"] == ["w"]
    assert stale is None


def test_no_plan_at_all_is_a_different_verdict(nccl_roundtrip_check, tmp_path):
    """`PEER_PLAN_NEVER_APPEARED` has to stay distinguishable from stale."""
    found, stale = nccl_roundtrip_check.wait_for_plan(
        tmp_path / "absent.json", "this-run", 0
    )
    assert found is None and stale is None


def test_a_rejected_lifecycle_publishes_no_weight_observations(nccl_roundtrip_check):
    """The verdict and the JSON must not disagree.

    `changed` and `recovered` were computed from the completions alone, so a
    rejected start or finish gave a verdict naming the rejection while the
    report still published `weights_changed_under_perturb: true` -- the JSON
    asserting an observation the verdict had just disowned. A rejected start
    means the engine never entered the update state and a rejected finish means
    it may still be in it, so in neither case are those completions evidence
    about weights.
    """
    decide = nccl_roundtrip_check.decide_verdict
    verdict, changed, recovered = decide(
        baseline=_gen("A"),
        perturbed=_gen("B"),
        restored=_gen("A"),
        perturb_lifecycle=_lifecycle(finish=500),
        restore_lifecycle=_lifecycle(),
    )
    assert verdict == "LIFECYCLE_REJECTED_FINISH"
    assert changed is None and recovered is None


@pytest.mark.parametrize("bad", [10**400, "high", None])
def test_an_unusable_confidence_scores_low_rather_than_raising(proposal_reward, bad):
    """A scorer has to survive the bad output it exists to score.

    `AgentStep.from_dict` coerces `confidence` with `float()`, and a JSON
    integer has no width limit -- so `10**400` raised `OverflowError` out of the
    scorer and aborted the whole batch instead of scoring one proposal low.
    """
    raw = json.dumps(
        {
            "category": "checkpoint_race",
            "hypothesis": "h",
            "next_mitigations": ["tf32_off"],
            "confidence": bad,
            "stop": False,
        }
    )
    score = proposal_reward.score_proposal(
        proposal_reward.Proposal("p", raw, ["tf32_off"], [])
    )
    assert score.tier < proposal_reward.MAX_TIER, (score.tier, score.detail)


@pytest.mark.parametrize(
    ("leg", "expected"),
    [
        ("start", "LIFECYCLE_REJECTED_START"),
        ("finish", "LIFECYCLE_REJECTED_FINISH"),
    ],
)
def test_a_rejected_start_or_finish_is_not_proven(nccl_roundtrip_check, leg, expected):
    """PROVEN required the whole update step, not just `/update_weights`.

    Only the two update statuses used to reach the verdict, so a rejected
    `/start_weight_update` -- the engine never entered the update state -- or a
    rejected `/finish_weight_update` -- it may still be in it -- still came back
    PROVEN whenever the two completions happened to round-trip. The completions
    are the same in both halves of this test; only the leg differs.
    """
    decide = nccl_roundtrip_check.decide_verdict
    verdict, _, _ = decide(
        baseline=_gen("A"),
        perturbed=_gen("B"),
        restored=_gen("A"),
        perturb_lifecycle=_lifecycle(**{leg: 500}),
        restore_lifecycle=_lifecycle(),
    )
    assert verdict == expected

    # And on the restore side, named separately so a reader knows which step.
    verdict, _, _ = decide(
        baseline=_gen("A"),
        perturbed=_gen("B"),
        restored=_gen("A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(**{leg: 500}),
    )
    assert verdict == f"RESTORE_{expected}"


def test_the_two_real_negative_results_still_come_back(nccl_roundtrip_check):
    """Guarding the failures must not swallow the findings.

    `HTTP_OK_BUT_WEIGHTS_UNCHANGED` is the defect this branch measured, and
    `CHANGED_BUT_NOT_FAITHFUL` is the one a perturbation-only test would miss.
    Both are reachable with everything healthy.
    """
    decide = nccl_roundtrip_check.decide_verdict
    unchanged, _, _ = decide(
        baseline=_gen("A"),
        perturbed=_gen("A"),
        restored=_gen("A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )
    assert unchanged == "HTTP_OK_BUT_WEIGHTS_UNCHANGED"

    unfaithful, _, _ = decide(
        baseline=_gen("A"),
        perturbed=_gen("B"),
        restored=_gen("C"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )
    assert unfaithful == "CHANGED_BUT_NOT_FAITHFUL"


# --------------------------------------------------------------------------- #
# The weight-transfer verdict: what the evidence can actually carry
# --------------------------------------------------------------------------- #
#
# Reviewer finding, and it is the one that reaches outside this repository:
# one greedy completion per phase cannot prove that weights changed. These
# drive the real `decide_verdict`, because the ordering and the strength of
# each branch *is* the check.


def _phase(*texts):
    """A phase as `generate_phase` records it: one entry per replicate."""
    generated = [t for t in texts if t is not None]
    return {
        "status": 200 if generated else 500,
        "text": texts[0],
        "texts": list(texts),
        "replicates": len(texts),
        "distinct": len(set(generated)),
        "ok": len(generated) == len(texts) and bool(texts),
        "stable": len(set(generated)) == 1 and len(generated) == len(texts),
    }


# The two completions a temperature-0 engine on this stack actually alternates
# between. `docs/tokenspeed-rl-e2e-sanitizer-routing.md` §5.4 measures 2 of 8
# distinct completions across separate temperature-0 requests on the greedy
# backend: the argmax moves with batch composition, so it is not the sampler
# and turning the temperature down cannot remove it.
_JITTER_A = " Paris. The capital of France is also the capital of the French Republic."
_JITTER_B = " Paris. The capital of France is also the capital of the Republic of France."


def test_argmax_jitter_alone_cannot_produce_proven(nccl_roundtrip_check):
    """A no-op transfer plus decoder noise must not read as a weight change.

    This is the false positive the driver exists to prevent, arriving by a
    route the driver used to be blind to. Every HTTP leg is 200 -- which is
    exactly what the recorded `HTTP_OK_BUT_PEER_NEVER_SENT` measurement saw --
    and no weight moves. The engine simply returns its minority completion once
    and its majority completion twice, which this repository has measured it
    doing 1-2 times in 8.

    Compared as single strings that is `B != A` and `C == A`: PROVEN, from
    nothing. Simulated over the recorded jitter rate it happened 11% of the
    time. Compared as phases it cannot happen at all, because the coincidence
    has to repeat on every replicate.
    """
    decide = nccl_roundtrip_check.decide_verdict

    verdict, changed, recovered = decide(
        baseline=_phase(_JITTER_A, _JITTER_A, _JITTER_A),
        # One jittered draw, then the engine settles back. Nothing was
        # transferred; this is the same weights answering three times.
        perturbed=_phase(_JITTER_B, _JITTER_A, _JITTER_A),
        restored=_phase(_JITTER_A, _JITTER_A, _JITTER_A),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )
    assert verdict == "HTTP_OK_BUT_WEIGHTS_UNCHANGED"
    assert changed is False

    # And the single-draw spelling of the same run, which is what the driver
    # used to do: one jittered B against one baseline A reads as PROVEN.
    single = decide(
        baseline=_phase(_JITTER_A),
        perturbed=_phase(_JITTER_B),
        restored=_phase(_JITTER_A),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )[0]
    assert single == "PROVEN", (
        "kept as the control: with one draw per phase the verdict is "
        "indistinguishable from a real transfer, which is why --replicates "
        "defaults above 1"
    )


def test_a_baseline_that_disagrees_with_itself_is_its_own_verdict(
    nccl_roundtrip_check,
):
    """If the observable is not a function of the weights, nothing else is.

    The baseline replicates are the noise control: they are drawn before
    anything is touched, so a disagreement among them is the decoder and only
    the decoder. Reported as its own verdict rather than quietly compared,
    because the honest answer is that this engine cannot support this method --
    not that the weights did or did not move.
    """
    decide = nccl_roundtrip_check.decide_verdict
    verdict, changed, recovered = decide(
        baseline=_phase(_JITTER_A, _JITTER_B, _JITTER_A),
        perturbed=_phase("!!!!", "!!!!", "!!!!"),
        restored=_phase(_JITTER_A, _JITTER_A, _JITTER_A),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )
    assert verdict == "BASELINE_NONDETERMINISTIC"
    # Not `False`, and not `True`: there was no trustworthy comparison, and a
    # boolean here would read as an observation nobody made. Same rule the
    # rejected-lifecycle branch already follows.
    assert changed is None and recovered is None


def test_a_missing_peer_marker_outranks_a_perfect_round_trip(nccl_roundtrip_check):
    """Evidence of the transport beats inference from model behaviour.

    `dist.broadcast` on the sending rank returns only once the other rank posts
    its matching receive, so the peer's round marker is proof a collective was
    matched. Absent, after the engine has already answered 200, the engine
    posted nothing -- and then whatever the model emitted afterwards is not
    about weights that were pushed, however cleanly it round-trips.

    The completions here are the textbook PROVEN shape, deliberately: the point
    is that the marker overrules them.
    """
    decide = nccl_roundtrip_check.decide_verdict
    verdict, _, _ = decide(
        baseline=_phase("A", "A", "A"),
        perturbed=_phase("B", "B", "B"),
        restored=_phase("A", "A", "A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
        peer_sent_perturb=False,
    )
    assert verdict == "HTTP_OK_BUT_PEER_NEVER_SENT"

    # A restore that never went out is the same finding one leg later.
    assert decide(
        baseline=_phase("A", "A", "A"),
        perturbed=_phase("B", "B", "B"),
        restored=_phase("A", "A", "A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
        peer_sent_perturb=True,
        peer_sent_restore=False,
    )[0] == "HTTP_OK_BUT_PEER_NEVER_SENT"

    # `None` means the check was not run -- the caller passed `--peer-grace 0`
    # -- and must not read as either answer.
    assert decide(
        baseline=_phase("A", "A", "A"),
        perturbed=_phase("B", "B", "B"),
        restored=_phase("A", "A", "A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
        peer_sent_perturb=None,
        peer_sent_restore=None,
    )[0] == "PROVEN"


def test_a_missing_peer_marker_also_withdraws_the_two_booleans(nccl_roundtrip_check):
    """The verdict disowned the transfer and the JSON asserted it anyway.

    `comparable` gated on the lifecycle and the generations but not on the
    peer evidence, so the run above -- verdict `HTTP_OK_BUT_PEER_NEVER_SENT` --
    still published `weights_changed_under_perturb: true` and
    `weights_recovered_under_restore: true` beside it. A reader taking the
    booleans, which is what a machine does, gets the opposite of the finding.

    It is the same defect the lifecycle half of `comparable` was already fixed
    for, one rank further up the ladder, and here the evidence is stronger:
    a missing marker is a direct observation that no collective was posted,
    where the completions are an inference from model behaviour.
    """
    decide = nccl_roundtrip_check.decide_verdict
    textbook = dict(
        baseline=_phase("A", "A", "A"),
        perturbed=_phase("B", "B", "B"),
        restored=_phase("A", "A", "A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )

    for leg in ({"peer_sent_perturb": False},
                {"peer_sent_perturb": True, "peer_sent_restore": False}):
        verdict, changed, recovered = decide(**textbook, **leg)
        assert verdict == "HTTP_OK_BUT_PEER_NEVER_SENT", leg
        assert changed is None and recovered is None, (leg, changed, recovered)

    # Narrowness, and the reason the test is `is not False` rather than
    # truthiness: `None` is a check that was not performed -- `--peer-grace 0`
    # -- and reading it as a failure would withdraw the booleans from every run
    # without a peer, which is most of them.
    verdict, changed, recovered = decide(
        **textbook, peer_sent_perturb=None, peer_sent_restore=None
    )
    assert verdict == "PROVEN"
    assert changed is True and recovered is True


def test_the_driver_reads_the_marker_the_peer_actually_writes(
    nccl_roundtrip_check, tmp_path
):
    """The two scripts have to agree on the filename, and nothing else checks.

    The peer has written `<plan-out>.roundN.done` since it was first committed
    and this driver ignored it, so there has never been anything holding the
    two spellings together. Derived here from the peer's own source rather than
    restated, so a rename on either side fails this instead of silently turning
    the strongest evidence the driver has into a permanent `appeared: False`.
    """
    plan = tmp_path / "plan.json"
    peer_source = (_EXAMPLES / "nccl_weight_peer.py").read_text()
    assert '.round{index}.done' in peer_source, (
        "the peer's marker filename moved; update the driver's "
        "peer_round_marker to match"
    )

    marker = nccl_roundtrip_check.peer_round_marker(str(plan), 2)
    assert marker == tmp_path / "plan.json.round2.done"

    # Absent: reported as absent rather than waited on forever.
    absent = nccl_roundtrip_check.wait_for_peer_round(str(plan), 1, 0.05, "r1", "perturb")
    assert absent["appeared"] is False

    # Present: found, and without burning the grace period.
    nccl_roundtrip_check.peer_round_marker(str(plan), 1).write_text(
        json.dumps({"run_id": "r1", "kind": "perturb"})
    )
    present = nccl_roundtrip_check.wait_for_peer_round(str(plan), 1, 30.0, "r1", "perturb")
    assert present["appeared"] is True
    assert present["waited_seconds"] < 5.0


def test_a_phase_missing_a_replicate_is_not_a_comparison(nccl_roundtrip_check):
    """Two completions and a 500 is a partial observation, not a narrow one.

    Scoring the replicates that survived is the same defect as scoring a single
    `text=None` -- an absent observation standing in for evidence -- so a phase
    counts only when every draw came back.
    """
    decide = nccl_roundtrip_check.decide_verdict
    verdict, changed, recovered = decide(
        baseline=_phase("A", "A", "A"),
        perturbed=_phase("B", "B", None),
        restored=_phase("A", "A", "A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
    )
    assert verdict == "POST_UPDATE_GENERATION_FAILED"
    assert changed is None and recovered is None


def test_a_real_weight_change_is_still_proven(nccl_roundtrip_check):
    """Narrowness: the replicates must not cost the verdict it exists to give.

    A deterministic baseline, a perturb that moves every draw, and a restore
    that returns every draw. This is the shape a working transport produces and
    it must still be PROVEN -- otherwise the fix has bought its precision by
    making the tool unable to say anything.
    """
    decide = nccl_roundtrip_check.decide_verdict
    verdict, changed, recovered = decide(
        baseline=_phase("A", "A", "A"),
        perturbed=_phase("!!!!", "!!!!", "!!!!"),
        restored=_phase("A", "A", "A"),
        perturb_lifecycle=_lifecycle(),
        restore_lifecycle=_lifecycle(),
        peer_sent_perturb=True,
        peer_sent_restore=True,
    )
    assert (verdict, changed, recovered) == ("PROVEN", True, True)


# --------------------------------------------------------------------------- #
# The peer's round roles: `--src` must not contradict what the round is for
# --------------------------------------------------------------------------- #


def _peer_role_argv(tmp_path, rounds, extra=()):
    argv = _peer_argv(tmp_path, rounds)
    return argv + list(extra)


def _run_peer(tmp_path, rounds, extra=()):
    return subprocess.run(
        _peer_role_argv(tmp_path, rounds, extra),
        capture_output=True,
        text=True,
        timeout=180,
    )


# Evidence that the peer got past argument validation and into real work.
#
# *Which* of these it hits depends entirely on what the environment has
# installed -- no torch on a bare login node, torch but no safetensors on the
# CPU lane, both plus a real `--model-path` in anger -- so asserting any one of
# them by name makes the test a statement about the runner rather than about
# the script. Asserting that it reached exactly none of them is the ordering
# claim these tests are actually making: a role error has to be paid before the
# import and before the rendezvous, because a peer that joins the group and
# then exits leaves the driver blocked on a broadcast that never comes.
_PAST_VALIDATION = ("import torch", "safetensors", "no .safetensors")


def _reached_real_work(output):
    return [marker for marker in _PAST_VALIDATION if marker in output]


def test_a_recv_round_refuses_to_broadcast_from_this_peer(tmp_path):
    """`--src` defaulting to `--rank` inverts the one round that receives.

    `recv` exists to tell "the engine posts no collective" from "the engine
    posts one but as the root", and it does that by poisoning a buffer and
    seeing whether the poison survives. With the default `--src` this peer is
    the broadcast root, so it *sends* the poisoned buffer -- and the root's
    buffer is the source, so it comes back untouched and the diagnostic reports
    that nothing arrived. Driven against the real `build_round` and the real
    broadcast contract, a working collective scored as a dead one:

        --src omitted -> poison [-1,-1,-1,-1] -> [-1,-1,-1,-1], "changed: False"
        --src 1       -> poison [-1,-1,-1,-1] -> [3.14, ...],   "changed: True"

    So the failure is silent and it is inverted, which is the worst pair: the
    one configuration a reader would reach for first is the one that cannot
    work, and it reports the finding the tool was run to look for.
    """
    proc = _run_peer(tmp_path, "recv")
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, output
    assert "--src resolved to 0" in output, output
    assert "this peer's own --rank" in output, output
    # Rejected before the torch import and before the rendezvous, for the same
    # reason the `--rounds` kind check moved there: a peer that has already
    # joined the group and then exits leaves the driver blocked on an
    # `/update_weights` whose broadcast is never posted, so a command-line
    # error costs the full timeout instead of nothing.
    assert _reached_real_work(output) == [], output


def test_a_recv_round_with_an_engine_source_is_allowed(tmp_path):
    """Narrowness: the check must reject the inversion, not the round kind.

    Same invocation with an engine rank as the root. It still fails -- there is
    no checkpoint at `--model-path`, and a test runner may not have torch or
    safetensors either -- but it has to fail *past the role check*, which is
    what makes this the narrowness control rather than a second copy of the
    test above.
    """
    proc = _run_peer(tmp_path, "recv", ("--src", "1"))
    output = proc.stdout + proc.stderr

    assert "--src resolved to" not in output, output
    assert "this peer's own --rank" not in output, output
    assert _reached_real_work(output), output


def test_a_sending_round_refuses_a_source_that_is_not_this_peer(tmp_path):
    """The mirror, which nobody reported and which is just as quiet.

    `perturb` and `restore` exist to push tensors at the engine, so this peer
    has to be the root. Point `--src` elsewhere and this peer silently becomes
    a receiver on a round it logs as "sent", overwriting the payload it
    believes it is delivering. Checked because the reported defect is one half
    of a cross product and fixing only the reported half leaves the other
    reachable from the same flag.
    """
    proc = _run_peer(tmp_path, "perturb,restore", ("--src", "1"))
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, output
    assert "would receive into the payload it believes it is sending" in output


def test_mixing_sending_and_receiving_rounds_is_refused(tmp_path):
    """One `--src` cannot be this peer and not this peer at the same time.

    Reported as a contradiction in the request rather than as a wrong value,
    because there is no value that would satisfy it -- whichever way it is set,
    one of the two rounds runs inverted.
    """
    proc = _run_peer(tmp_path, "perturb,recv")
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, output
    assert "separate invocations" in output, output


# --------------------------------------------------------------------------- #
# A provider outage is not model output
# --------------------------------------------------------------------------- #


def _row(scenario, sample, raw, reward, tier, error=""):
    return {
        "scenario_id": scenario,
        "sample": sample,
        "raw": raw,
        "reward": reward,
        "tier": tier,
        "transport_error": error,
        "failure_kind": "",
        "consumer_outcome": "accepted" if raw else "silent_stop",
        "category_claimed": "checkpoint_race" if raw else None,
        "mitigations_claimed": ["tf32_off"] if raw else None,
        "offered": ["tf32_off", "xnack"],
    }


def test_a_failed_call_does_not_enter_the_reward_mean(run_e2e):
    """A provider failure records an empty completion, which scores tier 0.

    Left in the statistics it is indistinguishable from a model that emitted
    nothing usable, so an outage reads as malformed output and drags the mean
    toward zero. The same shape as the `text=None` hole in `decide_verdict`:
    an absent observation standing in for an observation.

    Both numbers are reported, so excluding the row is visible rather than
    silent -- `n` is what was scored and `requested` is what was asked for.
    """
    proposals = [
        _row("s1", 0, '{"a": 1}', 1.0, 5),
        _row("s1", 1, '{"a": 2}', 1.0, 5),
        _row("s1", 2, "", 0.0, 0, error="APIConnectionError: connection reset"),
    ]
    out = run_e2e.aggregate(proposals, [])["proposal"]

    assert out["n"] == 2
    assert out["requested"] == 3
    assert out["transport_errors"] == 1
    assert out["delivered_rate"] == round(2 / 3, 4)
    # The mean over what came back, not over what was asked for. The failed row
    # would have pulled 1.0 down to 0.6667.
    assert out["mean_reward"] == 1.0
    # And the format rates, which are the same claim about a different axis: a
    # call that never reached the provider did not fail to parse.
    assert out["parse_rate"] == 1.0
    assert out["tier_distribution"]["tier_0"] == 0


def test_an_outage_is_not_reported_as_a_collapsed_group(run_e2e):
    """The worse half: a whole group failing looks like greedy decoding.

    `distinct_completions == 1` is this file's own diagnosis for a sampler that
    is not sampling -- the defect that cost the first end-to-end run. Three
    failed calls all record `raw: ""`, so the group reports one distinct
    completion and zero spread, which is that signature exactly, produced by an
    outage instead. `requested` next to `n` is what makes the difference
    legible in the per-scenario block.
    """
    proposals = [
        _row("s1", 0, '{"a": 1}', 1.0, 5),
        _row("s1", 1, '{"a": 2}', 0.6, 3),
        _row("s2", 0, "", 0.0, 0, error="Timeout"),
        _row("s2", 1, "", 0.0, 0, error="Timeout"),
        _row("s2", 2, "", 0.0, 0, error="Timeout"),
    ]
    out = run_e2e.aggregate(proposals, [])["proposal"]

    # The wholly-failed group is not a group: it contributed no observation.
    assert "s2" not in out["per_scenario"], out["per_scenario"]
    assert out["collapsed_groups"] == 0
    assert out["degenerate_groups"] == 0
    assert out["per_scenario"]["s1"]["requested"] == 2
    assert out["transport_errors"] == 3
    # `offered` describes the request, so it survives even a total outage.
    assert out["offered_count"] == 2


def test_the_rescorer_skips_rows_that_never_reached_the_provider(rescore_e2e):
    """Second instance of the same class, on the path that outlives the GPU.

    `rescore_e2e` re-scores from the archived wire text, so every recorded run
    is replayed through it long after the hardware is gone -- and it read
    `row["raw"]` for every row, transport errors included, dropping the
    `transport_error` field entirely on the way out. Fixing only `run_e2e`
    would leave the durable half of the measurement still counting outages as
    empty completions.
    """
    doc = {
        "meta": {"candidates": ["tf32_off", "xnack", "none"], "tried": []},
        "proposals": [
            _row("s1", 0, json.dumps({
                "category": "checkpoint_race", "hypothesis": "h",
                "next_mitigations": ["tf32_off"], "confidence": 0.5,
                "stop": False,
            }), 1.0, 5),
            _row("s1", 1, "", 0.0, 0, error="APIConnectionError: reset"),
        ],
    }
    rows = rescore_e2e.rescore_recorded(doc)
    assert len(rows) == 1
    assert rows[0]["sample"] == 0

    analysis = rescore_e2e.analyse(doc)
    assert analysis["n"] == 1
    assert analysis["requested"] == 2
    assert analysis["transport_errors"] == 1
    # The empty row would have been a second, distinct "completion" here, so a
    # group of one real answer would have reported spread it does not have.
    assert analysis["per_scenario"]["s1"]["distinct_completions"] == 1
    assert analysis["per_scenario"]["s1"]["spread_within_group"] == 0.0


# --------------------------------------------------------------------------- #
# Reaching the top tier has to mean the loop would run cells
# --------------------------------------------------------------------------- #


def _proposal_body(**over):
    body = {
        "category": "checkpoint_race",
        "hypothesis": "h",
        "next_mitigations": ["tf32_off"],
        "confidence": 0.5,
        "stop": False,
        "stop_reason": "",
    }
    body.update(over)
    return json.dumps(body)


def test_a_proposal_that_stops_cannot_reach_the_top_tier(proposal_reward):
    """`stop: true` ended the search and still collected a full 1.0.

    The tier-4 comment already said an empty list is a decision to stop and
    fails there -- but that is only the implicit spelling. `run_agent_loop`
    honours the flag and breaks before building any probe cell, so a reply that
    set `stop` while naming valid, offered mitigations had those names dropped
    on the floor and was paid top marks for them. `_consumer_outcome` recorded
    `silent_stop` for the same reply at the same time, so the module was
    already contradicting itself in its own output.

    That makes a constant policy that terminates every search the cheapest way
    to the top of this ladder, which is the exploit the reward exists to price.
    """
    offered = ["tf32_off", "xnack"]
    stopping = proposal_reward.Proposal(
        "stopping", _proposal_body(stop=True), offered + ["none"], []
    )
    score = proposal_reward.score_proposal(stopping)

    assert score.tier < 4, (score.tier, score.reward)
    assert score.reward < 1.0
    assert score.stopped_at == "tier4_registry"
    assert "never tried" in score.detail

    # Narrowness: the identical reply that does not stop is still on contract.
    running = proposal_reward.Proposal(
        "running", _proposal_body(stop=False), offered + ["none"], []
    )
    assert proposal_reward.score_proposal(running).tier == proposal_reward.MAX_TIER


def test_the_top_tier_implies_the_loop_would_accept_it(proposal_reward):
    """The invariant behind the fix, enumerated rather than spot-checked.

    `stop: true` was one spelling of "this reply is really a stop"; the point
    of enumerating is that there could have been others, and finding out by
    listing them is cheaper than finding out from a training run. Every way a
    reply can end the search is walked here against both the ladder and
    `consumer_outcome`, and the two must never disagree at the top.
    """
    offered = ["tf32_off", "xnack"]
    candidates = offered + ["none"]
    cases = {
        "on contract": _proposal_body(),
        "stop true, one valid name": _proposal_body(stop=True),
        "stop true, two valid names": _proposal_body(
            stop=True, next_mitigations=["tf32_off", "xnack"]
        ),
        "stop true, with a stop_reason": _proposal_body(
            stop=True, stop_reason="found it"
        ),
        "stop true, empty list": _proposal_body(stop=True, next_mitigations=[]),
        "empty list": _proposal_body(next_mitigations=[]),
        "only `none`": _proposal_body(next_mitigations=["none"]),
        "only repeats": _proposal_body(next_mitigations=["tf32_off", "tf32_off"]),
        "stop as a string": _proposal_body(stop="true"),
    }

    violations = []
    top = 0
    for label, raw in cases.items():
        score = proposal_reward.score_proposal(
            proposal_reward.Proposal(label, raw, candidates, [])
        )
        if score.tier == proposal_reward.MAX_TIER:
            top += 1
            if score.consumer_outcome != "accepted":
                violations.append((label, score.reward, score.consumer_outcome))

    assert violations == [], (
        "a proposal reached the top of the ladder that the real loop would not "
        f"act on: {violations}"
    )
    # And the enumeration has to be able to fail: at least one case must
    # actually reach the top, or this passes by testing nothing.
    assert top >= 1


def test_the_stopping_constant_is_priced_in_the_baseline_table(proposal_reward):
    """It used to sit at the top of this table, so it stays in it.

    The degenerate-policy table is where a constant that beats a real model
    becomes visible, and this one scored a clean 1.0 on every fixture. Keeping
    the row rather than deleting it with the defect means the table goes on
    showing that the cheapest possible policy is priced.
    """
    rows = {r["policy"]: r for r in proposal_reward.baselines()}
    stopping = rows["always stop, naming valid mitigations"]

    assert stopping["mean_reward"] < 1.0, stopping
    assert stopping["accepted_rate"] == 0.0, stopping
    # Strictly worse than the same proposal that lets the search continue,
    # which is the ordering the reward has to hold.
    assert stopping["mean_reward"] < rows["always the same valid proposal"]["mean_reward"]


# --------------------------------------------------------------------------- #
# `hold`: ownership is transferred, never cleared and re-installed
# --------------------------------------------------------------------------- #


def _serve_trace(tmp_path, bin_dir, subcommand, timeout=8):
    """Run the real script under xtrace and return (trace, docker calls).

    xtrace records the commands the shell actually executed, which is the only
    way to observe a trap being cleared: `trap -p` cannot be read from outside
    the process, and the handover window the reviewer found is one statement
    wide, so no signal can be timed into it from a test. The invariant behind
    the window is observable, though -- whether the guard is ever disarmed --
    and that is what this reads.

    `hold` does not terminate: holding is its whole job, and the stub keeps the
    container running forever. So it is signalled once the trace has been
    collected, rather than waited on.
    """
    proc = subprocess.Popen(
        ["bash", "-x", str(_EXAMPLES / "serve_for_rollouts.sh"), subcommand],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
    )
    try:
        output = proc.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        proc.terminate()
        output = proc.communicate(timeout=30)[0]
    calls_file = tmp_path / "docker-calls.log"
    calls = calls_file.read_text() if calls_file.exists() else ""
    return output, calls


def test_hold_never_disarms_the_ownership_guard(tmp_path):
    """The handover window, observed as the thing that opens it.

    `up` used to clear `_OWNED` and all three traps before returning, and
    `hold` installed replacements on the following command. A signal in that
    interval exited with the container running and nothing registered to remove
    it -- the leak the guard exists to prevent, reintroduced by the handover.

    Both orderings were wrong for opposite reasons: arming the traps *before*
    `up` meant a name collision removed another invocation's live server, which
    is why they were armed late in the first place. So ownership is transferred
    instead -- the traps armed just before `docker run -d` are never removed --
    and the observable is that `trap -` does not appear on this path at all.
    """
    bin_dir = _serve_stubs(tmp_path)
    trace, _ = _serve_trace(tmp_path, bin_dir, "hold")

    assert "trap - INT TERM EXIT" not in trace, trace[-3000:]
    # And the guard really was armed: ownership taken, phase advanced.
    assert "_OWNED=1" in trace
    assert "_PHASE=hold" in trace


def test_plain_up_still_releases_the_ownership_guard(tmp_path):
    """Narrowness, and it is the half that would leak if the flag inverted.

    `up` on its own hands the container to a *later* invocation, so it must
    still release: keeping the guard armed here would make the EXIT trap remove
    the container `up` just brought up successfully, turning the fix for a leak
    into a fix that deletes the server.
    """
    bin_dir = _serve_stubs(tmp_path)
    trace, calls = _serve_trace(tmp_path, bin_dir, "up")

    assert "trap - INT TERM EXIT" in trace
    assert "_PHASE=hold" not in trace
    # The observable that matters: a successful `up` removes nothing.
    assert "rm -f" not in calls, calls


def test_a_signalled_hold_still_removes_the_container(tmp_path):
    """`hold` installs no traps of its own now, so the cleanup must survive.

    Deleting `hold`'s own traps is only safe if the ones it inherits do the
    same job. The stub keeps the container "running" forever, so the wait loop
    never ends and the only way out is the signal.
    """
    bin_dir = _serve_stubs(tmp_path)
    proc = subprocess.Popen(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "hold"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
    )
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        proc.terminate()
    output = proc.communicate(timeout=30)[0]

    calls = (tmp_path / "docker-calls.log").read_text()
    assert "rm -f" in calls, calls
    assert "signalled; tearing down" in output, output


def test_hold_on_a_name_collision_removes_nothing(tmp_path):
    """The catastrophic direction, and the reason the traps were armed late.

    `hold` now declares that it wants ownership retained *before* calling `up`,
    which is the same shape as the arrangement that once destroyed another
    invocation's server. It is safe because the declaration only ever retains
    ownership and never asserts it: `_OWNED` is still set at one place, on the
    line before `docker run -d`, and the collision path exits before reaching
    it. Armed before rather than after so that a container created by a `run`
    that never returns a status -- killed mid-flight, or interrupted -- is
    still covered; what keeps that safe is the cidfile, which names only the
    id this invocation itself created.

    This is the test that would catch getting that wrong, and getting it wrong
    is worse than the leak being fixed.
    """
    bin_dir = _serve_stubs(tmp_path)
    # `-aq` is the collision probe: answering non-empty means the name belongs
    # to somebody else and this invocation must create and remove nothing.
    (bin_dir / "docker").write_text(
        "#!/usr/bin/env bash\n"
        f'echo "docker $*" >> "{tmp_path / "docker-calls.log"}"\n'
        'case "$1" in\n'
        '  ps) echo someone-elses-container ;;\n'
        "  run) echo 0123456789abcdef ;;\n"
        '  logs) echo "stub container log line" ;;\n'
        "esac\nexit 0\n"
    )
    (bin_dir / "docker").chmod(0o755)

    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "hold"],
        capture_output=True,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
        timeout=60,
    )
    output = proc.stdout + proc.stderr
    calls = (tmp_path / "docker-calls.log").read_text()

    assert "already exists" in output, output
    assert "rm -f" not in calls, calls
    assert "docker run" not in calls, calls


def _serve_run(tmp_path, bin_dir, subcommand="up", timeout=60):
    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), subcommand],
        capture_output=True,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
        timeout=timeout,
    )
    calls_file = tmp_path / "docker-calls.log"
    return (
        proc.returncode,
        proc.stdout + proc.stderr,
        calls_file.read_text() if calls_file.exists() else "",
    )


def test_an_engine_serving_a_different_model_does_not_get_handed_over(tmp_path):
    """A 200 from /v1/models was read as the answer to a question about names.

    `models` fetched the list and checked only the status line, so an engine
    advertising some *other* model passed bring-up. It is the expensive way to
    get this wrong: the prefix is stripped on the wire, so every rollout
    request 404s against a container that answered /health_generate, reported
    the right backends, and looks healthy at every level the driver can see.
    What the driver reports is a wall of failed requests with the model name
    nowhere in it.

    So the container is torn down here, where the cause still has a name.
    """
    bin_dir = _serve_stubs(
        tmp_path, models_body='{"data":[{"id":"meta-llama/Llama-3.1-8B"}]}'
    )
    code, output, calls = _serve_run(tmp_path, bin_dir)

    assert code == 58, output
    assert "Qwen/Qwen3-8B" in output and "Llama-3.1-8B" in output, output
    # Torn down, not left holding a GPU -- and through `teardown_failed`, so
    # the logs of the engine that was wrong survive the failure.
    assert "rm -f" in calls, calls
    assert (tmp_path / "logs" / "server-failure.log").exists()


def test_a_model_id_that_merely_contains_ours_is_not_ours(tmp_path):
    """Why the ids are extracted instead of the body grepped for `${MODEL}`.

    `grep -q "${MODEL}"` against the raw body is the one-line version of this
    check and it is satisfied by the wrong things: a quantised or instruct
    variant whose id contains ours as a prefix, and our own name quoted back
    inside an error message. Both are engines that 404 every request.
    """
    bin_dir = _serve_stubs(
        tmp_path, models_body='{"data":[{"id":"Qwen/Qwen3-8B-Instruct-AWQ"}]}'
    )
    code, output, _ = _serve_run(tmp_path, bin_dir)

    assert code == 58, output


def test_an_unreadable_model_list_is_unverified_rather_than_fatal(tmp_path):
    """The narrowness control, and the reason `models` has two codes.

    Failing bring-up on anything non-zero from `models` would be the opposite
    error: /health_generate has already answered and `backends` is what decides
    usability, so a list that could not be read is a question left unanswered,
    not a wrong answer. Only 58 -- the list was read and we are not on it --
    tears the container down.

    The subcommand still reports the unreadable list as a failure, because a
    caller running `serve_for_rollouts.sh models` is asking exactly that.
    """
    bin_dir = _serve_stubs(tmp_path, models_body='{"data":[]}')

    code, output, calls = _serve_run(tmp_path, bin_dir)
    assert code == 0, output
    assert "rm -f" not in calls, calls

    code, output, _ = _serve_run(tmp_path, bin_dir, subcommand="models")
    assert code == 56, output


def test_one_invocation_does_not_clear_another_invocations_cidfile(tmp_path):
    """The cidfile was a fixed path, so concurrent servers shared one file.

    `LOG_DIR` is keyed on `TS_LOG_DIR` and not on `TS_NAME`, so two invocations
    serving different models into the same log directory both used
    `${LOG_DIR}/container.cid`. The clear-before-run then deleted the peer's
    file, docker wrote our id over it, and the loser's teardown removed
    whichever id it found -- a live server torn down by a process that never
    started it. (Without the clear it is the other failure: docker exits 125
    with "container ID file found" and refuses to start at all.)

    The path now carries this process's pid, which no peer can write, so the
    clear is safe by construction rather than by being the only invocation.
    """
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    peer = logs / "container.cid"
    peer.write_text("peerscontainerid")

    bin_dir = _serve_stubs(tmp_path)
    code, output, calls = _serve_run(tmp_path, bin_dir)

    assert code == 0, output
    assert peer.read_text() == "peerscontainerid", calls
    assert "peerscontainerid" not in calls, calls
    cidfile = re.search(r"--cidfile (\S+)", calls)
    assert cidfile, calls
    assert re.fullmatch(r".*/ts-rollout-serve\.\d+\.cid", cidfile.group(1)), calls


# --------------------------------------------------------------------------- #
# Follow-ups from the review pass on the fixes above
# --------------------------------------------------------------------------- #


def test_a_marker_from_a_previous_run_is_not_this_runs_evidence(
    nccl_roundtrip_check, tmp_path
):
    """A stale round marker would undo the reason the marker check exists.

    `--plan` is a fixed shared path, so a marker left behind by a previous run
    is the ordinary state of that directory -- the same reason the plan itself
    is matched on `run_id` rather than on the file existing. It matters more
    here: the marker is the one observation standing between a silently dead
    transport and `PROVEN`, so accepting a stale one suppresses
    `HTTP_OK_BUT_PEER_NEVER_SENT` on a run where nothing was ever broadcast.

    Reported with the id it saw, because "the peer never got here" and "this
    directory needs clearing" send an operator to different places.
    """
    plan = tmp_path / "plan.json"
    marker = nccl_roundtrip_check.peer_round_marker(str(plan), 1)
    marker.write_text(json.dumps({"run_id": "a-previous-run", "kind": "perturb"}))

    stale = nccl_roundtrip_check.wait_for_peer_round(str(plan), 1, 0.05, "this-run", "perturb")
    assert stale["appeared"] is False
    assert stale["run_id_seen"] == "a-previous-run"

    marker.write_text(json.dumps({"run_id": "this-run", "kind": "perturb"}))
    fresh = nccl_roundtrip_check.wait_for_peer_round(str(plan), 1, 30.0, "this-run", "perturb")
    assert fresh["appeared"] is True
    assert fresh["waited_seconds"] < 5.0


def test_the_peer_stamps_its_markers_with_the_run_id(tmp_path):
    """The writer half, read off the peer's own source.

    The driver validating an id the peer never writes would reject every
    marker, which reads exactly like a transport that never sends -- a
    permanent false negative in place of the false positive just closed. So
    the two halves are pinned together rather than separately.
    """
    source = (_EXAMPLES / "nccl_weight_peer.py").read_text()
    assert '"run_id": args.run_id' in source, source[-2000:]
    # Published by rename for the same reason the plan is: a marker that exists
    # has to be a marker that is complete, or the driver reads a partial write
    # as a malformed id and waits out the whole grace period.
    assert "os.replace(marker_tmp, marker)" in source


def test_criterion_three_cannot_pass_over_scenarios_that_vanished(rescore_e2e):
    """Dropping failed rows must not turn an outage into perfect diversity.

    Excluding transport-error rows is right, and it has a consequence: a
    wholly failed scenario leaves `per_scenario` entirely, so `all(...)` runs
    over the survivors. In the limit every call fails, `per_scenario` is empty,
    and `all([])` is True -- a total outage reported as criterion 3 holding.

    The criterion has to see the scenarios that were asked about, not the ones
    that answered.
    """
    def row(scenario, sample, raw, reward, tier, error=""):
        return _row(scenario, sample, raw, reward, tier, error)

    good = json.dumps({
        "category": "checkpoint_race", "hypothesis": "h",
        "next_mitigations": ["tf32_off"], "confidence": 0.5, "stop": False,
    })
    # Scores differently from `good` rather than merely differing as text: the
    # criterion is about reward spread, so a second on-contract reply would
    # give a group with two distinct completions and no spread, which is a
    # different finding entirely.
    other = "not JSON, just a sentence."
    meta = {"candidates": ["tf32_off", "xnack", "none"], "tried": []}

    # s1 answered and has spread; s2 failed entirely and is gone.
    partial = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [
            row("s1", 0, good, 1.0, 5),
            row("s1", 1, other, 1.0, 5),
            row("s2", 0, "", 0.0, 0, "Timeout"),
            row("s2", 1, "", 0.0, 0, "Timeout"),
        ],
    })
    c3 = partial["criteria"]["3_within_group_spread_nonzero"]
    assert c3["holds"] is False, c3
    assert c3["detail"]["groups_missing"] == ["s2"], c3
    assert c3["detail"]["groups_requested"] == 2

    # The limit case: nothing came back at all.
    total = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [row("s1", 0, "", 0.0, 0, "Timeout")],
    })
    assert total["criteria"]["3_within_group_spread_nonzero"]["holds"] is False

    # Narrowness: a clean run still holds, so this is not just always False.
    clean = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [row("s1", 0, good, 1.0, 5), row("s1", 1, other, 1.0, 5)],
    })
    c3_clean = clean["criteria"]["3_within_group_spread_nonzero"]
    assert c3_clean["detail"]["groups_missing"] == []
    assert c3_clean["holds"] is True, c3_clean


def test_the_seed_probe_does_not_count_failed_draws_as_diversity(monkeypatch):
    """Third instance of the class, on the probe whose whole output is `distinct`.

    A failed call records `content: ""`, which is a distinct string and a
    tier-0 score. Counted, an outage reads as either sampling diversity or a
    format regression -- on the one statistic this probe exists to produce, and
    the statistic that decided the `--sampling-backend` finding.
    """
    probe_seed = _load("probe_seed")

    answers = [
        {"error": "", "content": '{"a": 1}'},
        {"error": "", "content": '{"a": 1}'},
        {"error": "APIConnectionError: reset", "content": ""},
        {"error": "APIConnectionError: reset", "content": ""},
    ]
    calls = iter(answers)
    monkeypatch.setattr(probe_seed, "call", lambda *a, **k: next(calls))

    out = probe_seed.probe_temperature("m", 0.7, 4)

    # Two identical completions and two failures: one distinct completion, not
    # two, and the collapse is real rather than manufactured by the outage.
    assert out["distinct"] == 1
    assert out["draws"] == 4
    assert out["delivered"] == 2
    assert len(out["errors"]) == 2
    # And the format statistics describe the two replies that arrived.
    assert out["tiers"] == out["tiers"][:2]
    assert 0 not in out["tiers"]


def test_the_console_line_denominates_in_what_arrived(monkeypatch, capsys):
    """The JSON was fixed and the line a human reads still said `1/5`.

    `distinct`, `parsed` and `format_gate_pass` are all computed over the
    completions that came back, and all three were printed over `draws` -- so
    four transport failures out of five drew `distinct 1/5`, which is exactly
    the greedy-decoding collapse signature this probe was written to detect.
    The errors were on the following lines, but the number is what gets pasted
    into a write-up, and `1/5` is a claim about sampling.

    `delivered` is printed too, not just used as the denominator: a truthful
    ratio over a shortfall nobody mentioned is still a hidden shortfall.
    """
    probe_seed = _load("probe_seed")

    answers = [{"error": "", "content": '{"category": "race", "confidence": 0.5}'}]
    answers += [{"error": "APIConnectionError: reset", "content": ""}] * 4
    calls = iter(answers)
    monkeypatch.setattr(probe_seed, "call", lambda *a, **k: next(calls))
    monkeypatch.setattr(probe_seed, "probe_seed_modes", lambda *a, **k: {})
    monkeypatch.setattr(
        probe_seed,
        "probe_unseeded_control",
        lambda *a, **k: {"temperature": 1.0, "draws": 0, "delivered": 0,
                         "distinct": 0, "errors": [], "samples": False},
    )

    probe_seed.main([
        "--base-url", "http://127.0.0.1:1/v1",
        "--model", "m",
        "--temperatures", "1.0",
        "--draws", "5",
    ])
    line = next(
        l for l in capsys.readouterr().out.splitlines() if "distinct" in l
    )

    assert "delivered 1/5" in line, line
    assert "distinct 1/1" in line, line
    # The collapse signature must not appear on a run that never collapsed.
    assert "/5" not in line.split("delivered 1/5", 1)[1], line


def test_a_marker_for_the_wrong_round_is_not_this_phases_evidence(
    nccl_roundtrip_check, tmp_path
):
    """The driver numbers rounds by position; the peer records what it sent.

    `perturb` is round 1 and `restore` is round 2 only because that is what
    the documented `--rounds perturb,restore` produces. The peer takes them in
    either order, so under `--rounds restore,perturb` round 1 is the restore --
    the marker appears on time, with this run's id, and the proof that a
    collective was matched gets attached to the wrong phase.

    Position is this driver's assumption. The kind is the peer's own record,
    so comparing them turns the assumption into a check.
    """
    plan = tmp_path / "plan.json"
    nccl_roundtrip_check.peer_round_marker(str(plan), 1).write_text(
        json.dumps({"run_id": "this-run", "kind": "restore"})
    )

    wrong = nccl_roundtrip_check.wait_for_peer_round(
        str(plan), 1, 0.05, "this-run", "perturb"
    )
    assert wrong["appeared"] is False
    assert wrong["kind_expected"] == "perturb"
    assert wrong["kind_seen"] == "restore"
    # Not reported as stale: the id matched, so "clear the directory" would be
    # the wrong advice. It is a `--rounds` ordering problem.
    assert wrong["run_id_seen"] is None

    # Narrowness: the same marker read as the round it actually is.
    right = nccl_roundtrip_check.wait_for_peer_round(
        str(plan), 1, 30.0, "this-run", "restore"
    )
    assert right["appeared"] is True


def test_an_empty_tensor_list_cannot_manufacture_round_markers(tmp_path):
    """A no-op plan would forge the one piece of evidence that cannot be faked.

    With an empty `--tensors` the peer joins the group, broadcasts nothing,
    writes its round markers and exits 0. Those markers are exactly what the
    driver now reads as proof a collective was matched, so an empty plan
    produces the evidence rather than the absence of it -- and it is a typo
    away, since `--tensors ""` and a trailing comma both land here.

    Harmless while the driver ignored the markers. Not harmless now, which is
    why it is rejected by the same commit that started reading them.
    """
    argv = _peer_argv(tmp_path, "perturb,restore")
    argv[argv.index("--tensors") + 1] = ""
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    output = proc.stdout + proc.stderr

    assert proc.returncode != 0, output
    assert "--tensors is empty" in output, output
    # Before the checkpoint load and before the plan is published, so nothing
    # downstream sees a partial run.
    assert _reached_real_work(output) == [], output
    assert not (tmp_path / "plan.json").exists()


# --------------------------------------------------------------------------- #
# Review pass 2026-09-21: the blocking failure path, and the fail-open family
# --------------------------------------------------------------------------- #


def test_a_rejected_start_does_not_post_the_blocking_update(
    nccl_roundtrip_check, monkeypatch
):
    """The verdict was already decided and the driver spent 30 minutes on it.

    `/update_weights` blocks while the workers receive every broadcast in the
    plan, and `call` gives it `TIMEOUT_S` -- 1800 seconds. When
    `/start_weight_update` is rejected the engine never entered the update
    state, so there is no receive to match and nothing to wait for; the step
    posted it anyway, and `_lifecycle_failure` only got to say "start" half an
    hour later. On the one failure path this driver exists to detect, it was
    unusable in practice.
    """
    posted = []

    def fake_call(base, method, path, body=None, timeout=nccl_roundtrip_check.TIMEOUT_S):
        posted.append((path, timeout))
        if path == "/start_weight_update":
            return 500, {"error": "engine busy"}, 0.003
        return 200, {"ok": True}, 0.01

    monkeypatch.setattr(nccl_roundtrip_check, "call", fake_call)
    step = nccl_roundtrip_check.lifecycle_update("http://c", {"names": ["w"]}, "perturb")

    assert [p for p, _ in posted] == ["/start_weight_update"]
    assert "update" not in step and "finish" not in step
    # Named rather than merely absent, so a truncated record does not read as a
    # driver that crashed here.
    assert step["skipped"] == ["update", "finish"]
    # And the verdict is unchanged: `_lifecycle_failure` walks the legs in
    # order and reports `start` whether or not the later keys exist.
    assert nccl_roundtrip_check._lifecycle_failure(step) == "start"


def test_a_rejected_update_still_posts_finish(nccl_roundtrip_check, monkeypatch):
    """Narrowness, and the reason the early return is on `start` only.

    A rejected `/update_weights` is a different situation: start succeeded, so
    the engine *is* mid-update, and `/finish_weight_update` is what takes it
    back out. Skipping it on the same reasoning would leave the engine in the
    update state for whatever runs next.
    """
    posted = []

    def fake_call(base, method, path, body=None, timeout=nccl_roundtrip_check.TIMEOUT_S):
        posted.append(path)
        if path == "/update_weights":
            return 500, {"error": "nope"}, 0.01
        return 200, {"ok": True}, 0.01

    monkeypatch.setattr(nccl_roundtrip_check, "call", fake_call)
    step = nccl_roundtrip_check.lifecycle_update("http://c", {"names": ["w"]}, "perturb")

    assert posted == [
        "/start_weight_update",
        "/update_weights",
        "/finish_weight_update",
    ]
    assert nccl_roundtrip_check._lifecycle_failure(step) == "update"


def test_the_report_survives_a_step_that_never_posted_an_update(
    nccl_roundtrip_check,
):
    """The half the suggested patch left out, and it is not cosmetic.

    `main` reads `report[...]['update']['status']` in four places -- two log
    lines and both `update_seconds` entries -- so returning early without
    touching them raises `KeyError: 'update'` on the line *after* the early
    return, before the next `flush()`. That replaces a thirty-minute hang with
    a traceback and no verdict file at all, which is worse: today's behaviour
    at least writes the report eventually.
    """
    stopped = {
        "label": "perturb",
        "start": {"status": 500, "seconds": 0.003, "body": {}},
        "skipped": ["update", "finish"],
        "skipped_reason": "start was not accepted",
        "total_seconds": 0.003,
    }
    # Both readers tolerate it, and the timing reader says `None` rather than
    # `0.0`: a leg that never ran has no duration, and zero would read as an
    # update that returned instantly -- which this driver has really recorded.
    assert nccl_roundtrip_check._log_step(stopped) is False
    assert nccl_roundtrip_check._update_seconds(stopped) is None

    ran = {"label": "perturb", "update": {"status": 200, "seconds": 1.25}}
    assert nccl_roundtrip_check._log_step(ran) is True
    assert nccl_roundtrip_check._update_seconds(ran) == 1.25


def test_check_determinism_fails_on_an_outage_rather_than_passing(rescore_e2e, capsys):
    """A flag that answers "is the rollout sampling?" from zero completions.

    It scanned only the scenarios that survived, so a scenario whose every
    request failed was simply absent, `collapsed` stayed empty and the command
    returned 0. Worse than silent: this is the flag someone runs *because* they
    suspect the rollout is degenerate.
    """
    def row(scenario, sample, raw, reward, tier, error=""):
        return _row(scenario, sample, raw, reward, tier, error)

    meta = {"candidates": ["tf32_off", "xnack", "none"], "tried": []}

    total_outage = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [
            row("s1", 0, "", 0.0, 0, "Timeout"),
            row("s1", 1, "", 0.0, 0, "Timeout"),
        ],
    })
    assert total_outage["per_scenario"] == {}

    rc = rescore_e2e.check_determinism([dict(total_outage, source="run.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "delivered no completions" in err, err
    # The advice has to name the cause. "Set a temperature" is what the
    # collapsed branch says and it is the wrong fix for an outage.
    assert "temperature" not in err, err


def test_a_thin_group_is_not_reported_as_a_collapsed_one(rescore_e2e, capsys):
    """The sibling: one delivered completion out of eight is not collapse.

    `distinct_completions == 1` is true either way, so a group that mostly
    failed to arrive was reported as degenerate sampling and the operator was
    told to set a temperature -- on a rollout that may already have been
    sampling correctly. Delivery is checked before diversity so the advice
    names the cause rather than whichever check ran first.
    """
    def row(scenario, sample, raw, reward, tier, error=""):
        return _row(scenario, sample, raw, reward, tier, error)

    good = json.dumps({
        "category": "checkpoint_race", "hypothesis": "h",
        "next_mitigations": ["tf32_off"], "confidence": 0.5, "stop": False,
    })
    meta = {"candidates": ["tf32_off", "xnack", "none"], "tried": []}

    thin = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [
            row("s1", 0, good, 1.0, 5),
            row("s1", 1, "", 0.0, 0, "Timeout"),
            row("s1", 2, "", 0.0, 0, "Timeout"),
        ],
    })
    assert thin["per_scenario"]["s1"]["n"] == 1
    assert thin["per_scenario"]["s1"]["requested"] == 3

    rc = rescore_e2e.check_determinism([dict(thin, source="run.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "delivered 1/3" in err, err
    assert "temperature" not in err, err


def test_a_genuinely_collapsed_group_still_reports_collapse(rescore_e2e, capsys):
    """Narrowness: full delivery plus one distinct completion is the real thing.

    This is the finding the flag exists for -- the greedy-decoding defect that
    cost the first end-to-end run -- so the delivery checks must not swallow it.
    """
    def row(scenario, sample, raw, reward, tier, error=""):
        return _row(scenario, sample, raw, reward, tier, error)

    same = json.dumps({
        "category": "checkpoint_race", "hypothesis": "h",
        "next_mitigations": ["tf32_off"], "confidence": 0.5, "stop": False,
    })
    meta = {"candidates": ["tf32_off", "xnack", "none"], "tried": []}

    collapsed = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [row("s1", 0, same, 1.0, 5), row("s1", 1, same, 1.0, 5)],
    })
    rc = rescore_e2e.check_determinism([dict(collapsed, source="run.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "single distinct completion" in err, err
    assert "Set a temperature" in err, err


def test_a_group_of_one_is_not_reported_as_a_collapsed_one(rescore_e2e, capsys):
    """The third way to satisfy `distinct_completions == 1` without collapsing.

    A group that requested one completion and got it is fully delivered, so it
    clears both delivery checks, and one completion is one distinct completion,
    so it met the collapsed test on both counts. The operator was told to set a
    temperature -- and there is no temperature at which a single draw is
    diverse. Within-group spread is undefined for a group of one, so this run
    cannot answer the question the flag asks, at any sampling setting.

    Not a corrupt results file either: `run_e2e` refuses `--samples 0` and
    allows `--samples 1`, so this is a file the driver is willing to produce.
    The advice has to be "rerun with more samples", which is the one thing that
    can change the answer.
    """
    def row(scenario, sample, raw, reward, tier, error=""):
        return _row(scenario, sample, raw, reward, tier, error)

    one = json.dumps({
        "category": "checkpoint_race", "hypothesis": "h",
        "next_mitigations": ["tf32_off"], "confidence": 0.5, "stop": False,
    })
    meta = {"candidates": ["tf32_off", "xnack", "none"], "tried": []}

    singleton = rescore_e2e.analyse({
        "meta": meta, "proposals": [row("s1", 0, one, 1.0, 5)],
    })
    # The premise: it looks exactly like collapse to the old test.
    assert singleton["per_scenario"]["s1"]["requested"] == 1
    assert singleton["per_scenario"]["s1"]["n"] == 1
    assert singleton["per_scenario"]["s1"]["distinct_completions"] == 1

    rc = rescore_e2e.check_determinism([dict(singleton, source="run.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "single completion" in err, err
    assert "--samples 2" in err, err
    # The wrong advice, and the reason this is a bug rather than a wording
    # nit: a knob that cannot move the answer sends the operator to tune the
    # rollout over a result the rollout did not produce. Matched on the
    # imperative rather than on the bare word, unlike the two delivery tests
    # above: this branch names the temperature on purpose, to say that no
    # setting of it would help.
    assert "Set a temperature" not in err, err


def test_a_two_sample_group_that_really_collapsed_is_still_caught(
    rescore_e2e, capsys
):
    """Narrowness: two is where collapse becomes observable, and it is checked.

    The singleton branch is a `requested < 2` test placed ahead of `collapsed`,
    so an off-by-one there -- `<= 2`, or dropping the matching `>= 2` guard on
    `collapsed` -- silently excuses the smallest group that can actually
    demonstrate the greedy-decoding defect the flag was written for.
    """
    def row(scenario, sample, raw, reward, tier, error=""):
        return _row(scenario, sample, raw, reward, tier, error)

    same = json.dumps({
        "category": "checkpoint_race", "hypothesis": "h",
        "next_mitigations": ["tf32_off"], "confidence": 0.5, "stop": False,
    })
    meta = {"candidates": ["tf32_off", "xnack", "none"], "tried": []}

    pair = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [row("s1", 0, same, 1.0, 5), row("s1", 1, same, 1.0, 5)],
    })
    assert pair["per_scenario"]["s1"]["requested"] == 2

    rc = rescore_e2e.check_determinism([dict(pair, source="run.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "single distinct completion" in err, err
    assert "Set a temperature" in err, err
    assert "--samples 2" not in err, err


def test_a_singleton_group_is_named_before_the_collapse_advice(
    rescore_e2e, capsys
):
    """Order, not just presence: one unanswerable group silences the advice.

    `check_determinism` returns on the first non-empty outcome, so a run
    holding both a group of one and a genuinely collapsed group has to report
    the group of one. The results file cannot answer the question for every
    group, and "set a temperature" on the strength of the groups it *can*
    answer for is advice built on a partial read of the run.
    """
    def row(scenario, sample, raw, reward, tier, error=""):
        return _row(scenario, sample, raw, reward, tier, error)

    same = json.dumps({
        "category": "checkpoint_race", "hypothesis": "h",
        "next_mitigations": ["tf32_off"], "confidence": 0.5, "stop": False,
    })
    meta = {"candidates": ["tf32_off", "xnack", "none"], "tried": []}

    mixed = rescore_e2e.analyse({
        "meta": meta,
        "proposals": [
            row("alone", 0, same, 1.0, 5),
            row("collapsed", 0, same, 1.0, 5),
            row("collapsed", 1, same, 1.0, 5),
        ],
    })

    rc = rescore_e2e.check_determinism([dict(mixed, source="run.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "single completion" in err, err
    assert "run.json: alone" in err, err
    assert "Set a temperature" not in err, err
    # And the collapsed group is not named either: the run has one report to
    # give, and it is the one about what cannot be answered.
    assert "single distinct completion" not in err, err


def test_the_determinism_flag_describes_what_it_now_does(rescore_e2e, capsys):
    """The help text still described only the check the flag started with.

    Two failure modes were added to `check_determinism` -- a scenario that
    delivered nothing, and a group short of its requested completions -- and
    `--help` went on saying it fails "if any group's completions are
    byte-identical". An operator reading that sees a non-zero exit for a run
    whose groups are all distinct and has no reason to look for the outage,
    which is the advice-reversing case the delivery checks exist to catch.

    A fourth followed -- a group that requested a single completion -- and it
    is the one an operator is likeliest to hit deliberately, because
    `--samples 1` is a supported way to drive `run_e2e`. Pinned against the
    branches rather than spot-checked, so adding a fifth reason to fail
    without saying so fails here.
    """
    with pytest.raises(SystemExit):
        rescore_e2e.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    flag = help_text.split("--check-determinism", 1)[1]

    assert "byte-identical" in flag, flag
    assert "delivered nothing" in flag, flag
    assert "fewer completions than were requested" in flag, flag
    assert "requested a single completion" in flag, flag


def test_an_infra_error_the_producer_recorded_stays_an_error(triage_reward):
    """`meta:` IDs bypass the resolver, and upstream says so in its own words.

    `partition_detectors` recognises exactly `tier1:timeout` and
    `tier1:exec_failed`. `verdict.py`'s module docstring states that
    `error_detectors_fired` "can still carry additional `meta:` infra-error
    IDs, but those are written **directly** by a workload that bypasses
    `resolve` entirely ... they are not produced by, and do not flow through,
    this resolver."

    So recombining the two stored lists and re-splitting them asked the wrong
    authority. `SubprocessWorkload` writes `meta:env_file_validation_failed`
    for a rejected `probe.env`, where the subprocess never launched and the
    trial made no observation at all -- and re-partitioning moved it onto
    `failure_detectors`, scoring a trial that never ran as a reproduction,
    marking the archive stale, and handing the attribution term a ground truth
    that cites an infrastructure error as the cause.
    """
    doc = {
        "verdict": "error",
        "exit_code": 2,
        "failure_detectors_fired": [],
        "error_detectors_fired": ["meta:env_file_validation_failed"],
    }
    label = triage_reward.label_run(doc)

    assert label.verdict == "error"
    assert label.error_detectors == ["meta:env_file_validation_failed"]
    assert label.failure_detectors == []
    assert label.stale is False


def test_the_meta_carve_out_is_positional_not_by_prefix(triage_reward):
    """`meta:` is not a synonym for "infra error", and keying on it would break.

    `resolve` synthesises `meta:missing_pass_signal` as a genuine *failure*
    signal and appends it to the failure list. A rule that read the prefix
    would move that onto the error side -- the same defect in the opposite
    direction, turning a real reproduction into an infra flake. What is
    preserved is where the producer put it.
    """
    real_failure = triage_reward.label_run({
        "verdict": "fail",
        "failure_detectors_fired": ["meta:missing_pass_signal"],
        "error_detectors_fired": [],
    })
    assert real_failure.verdict == "fail"
    assert real_failure.failure_detectors == ["meta:missing_pass_signal"]
    assert real_failure.error_detectors == []
    assert real_failure.stale is False


def test_re_partitioning_still_corrects_a_genuinely_misfiled_detector(triage_reward):
    """Narrowness: the carve-out must not disable the thing this function is for.

    A `tier1:timeout` recorded on the failure side is exactly what
    `partition_detectors` owns, and it still gets moved and still marks the
    archive stale.
    """
    label = triage_reward.label_run({
        "verdict": "fail",
        "failure_detectors_fired": ["tier1:timeout"],
        "error_detectors_fired": [],
    })
    assert label.verdict == "error"
    assert label.error_detectors == ["tier1:timeout"]
    assert label.stale is True

    # And fail still beats error in precedence when both are real.
    mixed = triage_reward.label_run({
        "verdict": "fail",
        "failure_detectors_fired": ["tier4:nan_signature"],
        "error_detectors_fired": ["meta:env_file_validation_failed"],
    })
    assert mixed.verdict == "fail"
    assert mixed.error_detectors == ["meta:env_file_validation_failed"]


def test_the_recipe_cli_fails_when_a_recipe_falls_short(recipe_reward, tmp_path, capsys):
    """It computed `worst` and returned 0, so it could not gate anything.

    The grade went to stdout and the thing automation reads said "fine" either
    way. `run_demo`, one function up in the same file, has returned
    `1 if failures else 0` all along -- so the correct shape was already here.
    """
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a recipe\n", encoding="utf-8")
    assert recipe_reward.main([str(bad), "--no-novelty-gate"]) == 1
    capsys.readouterr()


def test_an_unreadable_recipe_is_a_max_deficit_not_a_skip(
    recipe_reward, tmp_path, capsys
):
    """The half that makes the exit code honest.

    The `except OSError` branch `continue`d without touching `worst`, so an
    unreadable path was indistinguishable from one that scored top marks.
    Wiring the exit code without this would have shipped a gate that passes the
    inputs it never looked at -- the same defect one branch earlier.
    """
    missing = tmp_path / "nope.yaml"
    assert not missing.exists()
    assert recipe_reward.main([str(missing), "--no-novelty-gate"]) == 1
    capsys.readouterr()


def _graded_as(recipe_reward, monkeypatch, tmp_path, grade):
    """Run the CLI over one recipe whose grade is fixed by the caller.

    The tier ladder is stubbed rather than driven, because the defect is in
    how `main` turns a grade into an exit code and the input that exposes it
    -- a copy that reaches the *top* tier -- needs tier 4 to pass, which needs
    the aorta workload registry, which needs `torch`. Making the tripwire
    depend on that would mean it stops guarding anything on a host without a
    GPU stack, which is most of them. The real gate is exercised by
    `test_a_verbatim_corpus_copy_earns_nothing` above; this is about the wire.
    """
    root = tmp_path / "recipes"
    root.mkdir()
    (root / "committed.yaml").write_text(recipe_reward._GOOD, encoding="utf-8")
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(recipe_reward._GOOD, encoding="utf-8")
    # `**_` rather than a named `sidecar_files`: the stub stands in for the
    # grader's whole signature, and pinning it here means every keyword `main`
    # learns to forward breaks seven tests that are about the exit code.
    monkeypatch.setattr(
        recipe_reward, "grade_recipe_text", lambda text, corpus=None, **_: grade
    )
    return recipe_reward.main([str(candidate), "--recipes-root", str(root)])


def test_a_memorised_copy_does_not_pass_the_gate_on_its_tier(
    recipe_reward, monkeypatch, tmp_path, capsys
):
    """The exit code read the tier deficit, and a copy has none.

    A verbatim copy of a committed recipe passes every tier -- it is a valid
    recipe, that is the whole point of copying it -- so it reaches tier 5, the
    deficit is 0, and `main` returned 0 for an input the novelty gate had just
    scored 0.00. The printed line said MEMORISED and the JSON said
    `"memorised": true`; the only channel that disagreed was the one
    automation reads, which is the channel the exit code exists to be.

    Retrieval passing a gate that exists to price retrieval is the worst case
    this scorer has.
    """
    memorised = recipe_reward.Grade(
        tier=recipe_reward.MAX_TIER,
        tier_reward=1.0,
        reward=0.0,
        nearest_committed=("recipes/committed.yaml", 1.0),
        novelty_multiplier=0.0,
        memorised=True,
    )
    assert _graded_as(recipe_reward, monkeypatch, tmp_path, memorised) == 1
    out = capsys.readouterr().out
    assert "MEMORISED" in out, out


def test_a_top_tier_original_still_passes(recipe_reward, monkeypatch, tmp_path, capsys):
    """Narrowness. Failing every top-tier recipe would be the easy over-fix."""
    original = recipe_reward.Grade(
        tier=recipe_reward.MAX_TIER,
        tier_reward=1.0,
        reward=1.0,
        nearest_committed=("recipes/committed.yaml", 0.11),
        novelty_multiplier=1.0,
        memorised=False,
    )
    assert _graded_as(recipe_reward, monkeypatch, tmp_path, original) == 0
    capsys.readouterr()


def test_the_soft_zone_stays_a_gradient_rather_than_a_second_cliff(
    recipe_reward, monkeypatch, tmp_path, capsys
):
    """Only the hard zone fails, and the distinction is the design.

    Between `MEMORISATION_SOFT` and `MEMORISATION_HARD` the gate *scales* the
    reward: a recipe that resembles a committed one is worth less, not worth
    nothing. Failing on that would collapse the taper this scorer is built
    around into a threshold, and the taper is what stops a near-copy being
    indistinguishable from an original.
    """
    resembling = recipe_reward.Grade(
        tier=recipe_reward.MAX_TIER,
        tier_reward=1.0,
        reward=0.5,
        nearest_committed=("recipes/committed.yaml", 0.85),
        novelty_multiplier=0.5,
        memorised=False,
    )
    assert _graded_as(recipe_reward, monkeypatch, tmp_path, resembling) == 0
    capsys.readouterr()


def test_a_run_that_delivered_nothing_is_not_a_measurement(run_e2e):
    """Third member of the transport-error family, on the driver's own exit.

    A reporter's exit code answers "did I produce a measurement", not "was the
    measurement good" -- a poor reward is a finding and stays 0. But a run
    where every request failed wrote a file with no model output in it, and
    returning 0 says the measurement happened.
    """
    outage = run_e2e.aggregate(
        [_row("s1", 0, "", 0.0, 0, error="Timeout")],
        [],
    )
    assert outage["proposal"]["delivered_rate"] == 0.0
    assert run_e2e.is_empty_measurement(outage) is True

    # A partial run is deliberately *not* caught: one delivered completion is a
    # thin measurement, which `delivered_rate` already reports and which is a
    # finding rather than a failure to measure.
    partial = run_e2e.aggregate(
        [
            _row("s1", 0, '{"a": 1}', 1.0, 5),
            _row("s1", 1, "", 0.0, 0, error="Timeout"),
        ],
        [],
    )
    assert partial["proposal"]["delivered_rate"] == 0.5
    assert run_e2e.is_empty_measurement(partial) is False

    # And a run that asked for nothing is not an outage either -- there is no
    # measurement to be missing, so this must not turn an empty corpus into a
    # transport failure.
    assert run_e2e.is_empty_measurement(run_e2e.aggregate([], [])) is False


def test_a_group_size_of_zero_is_refused_at_the_argument(
    run_e2e, build_corpus, tmp_path, capsys
):
    """`--samples 0` wrote a clean-looking results file over nothing.

    The driver looped `range(0)` per scenario, aggregated means of empty
    lists, and exited 0 -- the same "nothing was measured and the exit code
    said fine" that `is_empty_measurement` above refuses, reached from the
    argument rather than from the transport. It is worse here because the file
    is well-formed: `n: 0`, a `condition` string, and a name that makes it look
    like a comparable row in a sweep directory.

    Rejected rather than clamped: `samples_per_scenario` in the results file
    quotes the command line, so a clamp would record a group size the run did
    not use.
    """
    # A real corpus, so that without the check the run would get all the way
    # to writing the file -- `range(0)` makes no calls, so nothing else stops
    # it and there is no network to need.
    corpus, _ = _build(build_corpus, tmp_path, _SURVEY)
    out = tmp_path / "results.json"

    with pytest.raises(SystemExit) as excinfo:
        run_e2e.main([
            "--corpus", str(corpus / "triage.jsonl"),
            "--base-url", "http://127.0.0.1:1/v1",
            "--model", "openai/whatever",
            "--samples", "0",
            "--skip-triage",
            "--out", str(out),
        ])
    assert excinfo.value.code == 2
    assert "--samples 0" in capsys.readouterr().err
    # Refused before anything ran, so nothing was written.
    assert not out.exists()


def test_a_group_of_one_is_allowed_and_says_why_it_looks_collapsed(
    run_e2e, build_corpus, monkeypatch, tmp_path, capsys
):
    """Narrowness, against the easy over-fix of demanding two.

    One completion per scenario is a real measurement -- the format gate, the
    tier ladder and the triage half all score it -- and it is the cheapest way
    to smoke-test a driver or a serving change. Refusing it would remove a
    legitimate mode to fix an argument that draws nothing at all.

    What a group of one cannot do is measure within-group spread, and this
    report leads with that: every group comes back `degenerate: true` with
    `distinct_completions: 1`, which is byte-for-byte the signature of the
    greedy-decoding collapse the first end-to-end run hit. Same numbers,
    entirely different cause, so the run says which one it is.
    """
    single = run_e2e.aggregate([_row("s1", 0, '{"a": 1}', 1.0, 5)], [])
    group = single["proposal"]["per_scenario"]["s1"]
    assert group["degenerate"] is True and group["distinct_completions"] == 1
    # ...and that shape is not an outage, so nothing else refuses it either.
    assert run_e2e.is_empty_measurement(single) is False

    # The CLI accepts it, and says which of the two causes this is. The drives
    # are stubbed because the question here is the argument, not the rollout.
    #
    # The recorder is stubbed for the same reason and one more: it imports
    # `litellm` in its constructor, and `litellm` is not on the CPU lane --
    # `aorta[chat]` carries it. Leaving it real made this test pass here and
    # fail every CPU job with a `ModuleNotFoundError` that has nothing to do
    # with `--samples 1`. It is reached *after* the warning is printed, so the
    # claim was already true at the point the import ended the run; a red lane
    # whose failure is a dependency is still a red lane nobody can read.
    corpus, _ = _build(build_corpus, tmp_path, _SURVEY)

    class _NoTransport:
        """The recorder's surface that `main` touches, and nothing else."""

        def __init__(self, **_kwargs):
            self.calls = []
            self.seed_mode = None

        def install(self):
            pass

        def restore(self):
            pass

    monkeypatch.setattr(run_e2e, "drive_proposals", lambda *a, **k: [])
    monkeypatch.setattr(run_e2e, "RecordingLiteLLM", _NoTransport)
    out = tmp_path / "one.json"
    assert run_e2e.main([
        "--corpus", str(corpus / "triage.jsonl"),
        "--base-url", "http://127.0.0.1:1/v1",
        "--model", "openai/whatever",
        "--samples", "1",
        "--skip-triage",
        "--out", str(out),
    ]) == 0
    assert out.exists()
    assert "--samples 1" in capsys.readouterr().err


def test_the_rollout_driver_refuses_a_corrupt_label_before_sampling(
    run_e2e, monkeypatch, tmp_path
):
    """`loop_state` still copies the label's detector lists with `list()`.

    That is safe only because every row it sees has been through
    `triage_reward.load_corpus` first -- `main` drives only the rows the
    scorer's label map contains -- so a string list there is refused before
    it can become one-character detector IDs in the prompt the model is
    shown. Pinned here because that is a property of `main`'s ordering, and a
    reorder would break it with nothing else going red.
    """
    corpus = tmp_path / "triage.jsonl"
    corpus.write_text(json.dumps({
        "kind": "triage", "example_id": "triage:s", "scenario_id": "s",
        "workload_family": "f",
        "label": {"verdict": "fail", "failure_detectors": "consan:1",
                  "error_detectors": []},
    }) + "\n", encoding="utf-8")

    def _no_rollout(*_args, **_kwargs):
        raise AssertionError("sampled rollouts for a corpus that should be refused")

    monkeypatch.setattr(run_e2e, "drive_proposals", _no_rollout)
    monkeypatch.setattr(run_e2e, "drive_triage", _no_rollout)
    monkeypatch.setattr(run_e2e, "RecordingLiteLLM", _no_rollout)
    monkeypatch.setenv("OPENAI_API_BASE", "unset")
    monkeypatch.setenv("OPENAI_API_KEY", "unset")

    with pytest.raises(ValueError, match=r"triage\.jsonl:1: failure_detectors"):
        run_e2e.main([
            "--corpus", str(corpus),
            "--base-url", "http://127.0.0.1:1/v1",
            "--model", "openai/whatever",
            "--out", str(tmp_path / "out.json"),
        ])
    assert not (tmp_path / "out.json").exists()


def test_the_models_subcommand_fails_on_an_http_error(tmp_path):
    """`curl` exits 0 for any response it manages to read, including a 500.

    So the transport check this function already had caught a dead socket and
    let an error *page* through as the answer -- printed under "advertised
    models", subcommand exit 0. One layer up from the `(no response)` bug this
    same function was corrected for: first a failed fetch read as fine, then a
    successful fetch of a failure did.
    """
    bin_dir = _serve_stubs(tmp_path, models_code="500")
    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "models"],
        capture_output=True,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
        timeout=120,
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode == 56, output
    assert "answered HTTP 500" in output, output


def test_the_models_subcommand_still_passes_on_a_healthy_gateway(tmp_path):
    """Narrowness: a 200 must still print the body and exit 0.

    The status is parsed off the end of the response, so getting the split
    wrong would either eat the body or report every healthy gateway as broken.
    """
    bin_dir = _serve_stubs(tmp_path, models_code="200")
    proc = subprocess.run(
        ["bash", str(_EXAMPLES / "serve_for_rollouts.sh"), "models"],
        capture_output=True,
        text=True,
        env=_serve_env(tmp_path, bin_dir),
        timeout=120,
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode == 0, output
    assert "Qwen/Qwen3-8B" in output, output
    # The status must not be printed as though it were part of the body.
    assert "200" not in proc.stdout.replace("Qwen/Qwen3-8B", ""), proc.stdout


# --------------------------------------------------------------------------- #
# Review pass 2026-09-23: the second collective into a group still on round 1
# --------------------------------------------------------------------------- #


def _drive_roundtrip(
    mod, tmp_path, monkeypatch, *, perturb_update=200, marker=True, peer_grace="0.05"
):
    """Run ``main`` end to end against a stub control plane.

    Every HTTP call in this driver goes through ``call``, so replacing that one
    function is enough to run the whole sequence in-process and record exactly
    which requests were posted -- which is the thing under test here. Asserting
    on the verdict alone would not catch it: the verdict was already right, and
    the defect was the update posted on the way to it.
    """
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "names": ["w"],
                "dtype_names": ["float32"],
                "shapes": [[1]],
            }
        )
    )
    if marker:
        mod.peer_round_marker(str(plan), 1).write_text(
            json.dumps({"run_id": "r1", "kind": "perturb"})
        )
    # Round 2's marker is always in place, so a restore that does get posted
    # comes back healthy: nothing about the *second* round is what stops it.
    mod.peer_round_marker(str(plan), 2).write_text(
        json.dumps({"run_id": "r1", "kind": "restore"})
    )

    posted: list[str] = []
    # Baseline A, perturb B, restore A -- a clean round trip, three draws each.
    texts = ["A"] * 3 + ["B"] * 3 + ["A"] * 3

    def fake_call(base, method, path, body=None, timeout=mod.TIMEOUT_S):
        posted.append(path)
        if path == "/v1/completions":
            drawn = sum(1 for p in posted if p == "/v1/completions") - 1
            return 200, {"choices": [{"text": texts[min(drawn, len(texts) - 1)]}]}, 0.01
        if path == "/update_weights":
            # The first lifecycle is the perturb; a second one is the restore
            # this test is asking about.
            if sum(1 for p in posted if p == "/start_weight_update") == 1:
                return perturb_update, {"detail": "stub"}, 0.01
        return 200, {"ok": True}, 0.01

    monkeypatch.setattr(mod, "call", fake_call)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nccl_roundtrip_check.py",
            "--model", "m",
            "--plan", str(plan),
            "--plan-run-id", "r1",
            "--master-port", "29500",
            "--world-size", "2",
            "--peer-grace", peer_grace,
            "--out", str(out),
        ],
    )
    code = mod.main()
    return code, json.loads(out.read_text()), posted


def test_a_rejected_perturb_update_does_not_post_a_second_collective(
    nccl_roundtrip_check, tmp_path, monkeypatch
):
    """The restore would have consumed the perturb payload, not replaced it.

    Broadcasts are ordered. A rejected perturb update leaves the peer possibly
    still blocked in its round-1 send, and both rounds carry identical shapes,
    so the restore update matches *that* pending payload: the engine receives
    the zeroed weights while the driver labels the step "restore", and round 2
    is the one left blocked. `decide_verdict` already named this run correctly
    at the end -- the damage was done by the update posted on the way there,
    which is why this asserts on the requests and not only on the verdict.
    """
    code, report, posted = _drive_roundtrip(
        nccl_roundtrip_check, tmp_path, monkeypatch, perturb_update=500
    )

    assert posted.count("/start_weight_update") == 1
    assert posted.count("/update_weights") == 1
    assert report["verdict"] == "UPDATE_REJECTED"
    assert report["stopped_after_round"] == 1
    # Recorded as a declined step, not left absent: "the driver crashed here"
    # and "the driver refused to post this" are different things to conclude.
    assert report["restore"]["skipped"] == ["start", "update", "finish"]
    assert "pending" in report["restore"]["skipped_reason"]
    assert "after_restore" not in report
    # The same withdrawal `decide_verdict` makes for this verdict.
    assert report["weights_changed_under_perturb"] is None
    assert report["weights_recovered_under_restore"] is None
    # The exit code follows the verdict: this is the code the full path
    # returned for `UPDATE_REJECTED` before the run learned to stop early.
    assert code == 1


def test_a_missing_round_one_marker_also_stops_before_the_restore(
    nccl_roundtrip_check, tmp_path, monkeypatch
):
    """The other half, and the one a status-code check cannot see.

    Here every leg came back 200 and the engine is satisfied -- but the peer's
    round-1 marker never appeared, which says no collective was matched. That
    is the same blocked peer as above reached by a different route, so posting
    the restore into it has the same consequence.
    """
    code, report, posted = _drive_roundtrip(
        nccl_roundtrip_check, tmp_path, monkeypatch, marker=False
    )

    assert posted.count("/start_weight_update") == 1
    assert report["verdict"] == "HTTP_OK_BUT_PEER_NEVER_SENT"
    assert report["perturb"]["peer"]["appeared"] is False
    assert report["restore"]["skipped"] == ["start", "update", "finish"]
    assert code == 1


def test_a_completed_perturb_round_still_gets_its_restore(
    nccl_roundtrip_check, tmp_path, monkeypatch
):
    """Narrowness. A guard that stops every run proves nothing at all.

    Accepted lifecycle plus a round-1 marker is the case the whole driver
    exists to measure, and it must still post both rounds and reach a real
    verdict.
    """
    code, report, posted = _drive_roundtrip(nccl_roundtrip_check, tmp_path, monkeypatch)

    assert posted.count("/start_weight_update") == 2
    assert posted.count("/finish_weight_update") == 2
    assert report["verdict"] == "PROVEN"
    assert "stopped_after_round" not in report
    assert report["weights_changed_under_perturb"] is True
    assert report["weights_recovered_under_restore"] is True
    assert code == 0


@pytest.mark.parametrize(
    "lifecycle,peer,expected",
    [
        (_lifecycle(start=500), None, "LIFECYCLE_REJECTED_START"),
        (_lifecycle(update=500), None, "UPDATE_REJECTED"),
        (_lifecycle(finish=500), None, "LIFECYCLE_REJECTED_FINISH"),
        (_lifecycle(), False, "HTTP_OK_BUT_PEER_NEVER_SENT"),
        (_lifecycle(), True, None),
        # `None` is "the check was not performed", which the ladder is explicit
        # is evidence either way -- so it must not stop the run.
        (_lifecycle(), None, None),
    ],
)
def test_the_early_stop_names_the_verdict_the_full_run_would_have(
    nccl_roundtrip_check, lifecycle, peer, expected
):
    """The drift guard: one ladder, read from two places.

    `main` now decides whether to continue from the same two rungs
    `decide_verdict` uses to name the run, so a copy of that naming in `main`
    would be free to drift from the verdict the report carries. This pins the
    helper to both its answers -- the verdict strings, and the `None` that
    means the perturb round completed and the restore may be posted.
    """
    assert nccl_roundtrip_check.round_one_verdict(lifecycle, peer) == expected

    # And when it does name a verdict, it is the one the full ladder reaches
    # from the same evidence -- with an otherwise perfect round trip below it,
    # so nothing except these two rungs can be supplying the answer.
    if expected is not None:
        assert nccl_roundtrip_check.decide_verdict(
            baseline=_gen("A"),
            perturbed=_gen("B"),
            restored=_gen("A"),
            perturb_lifecycle=lifecycle,
            restore_lifecycle=_lifecycle(),
            peer_sent_perturb=peer,
            peer_sent_restore=True,
        )[0] == expected


def test_the_harness_builds_the_proposer_the_agent_would_build(run_e2e, monkeypatch):
    """`LiteLLMProposer` and "what `aorta agent` uses" stopped being one thing.

    Phase 5b put `litellm` onto the shared chat provider layer, so
    `make_proposer("litellm")` returns `ChatProviderProposer` when the chat
    extra is installed and the direct `LiteLLMProposer` only as the agent-only
    fallback. Naming the class here measured whichever one this file picked,
    under a docstring calling it the real path.
    """
    from aorta.agent.llm import LiteLLMProposer

    asked = []

    def fake_make_proposer(backend, *, model=None):
        asked.append((backend, model))
        return LiteLLMProposer(model=model)

    monkeypatch.setattr(run_e2e, "make_proposer", fake_make_proposer)
    proposer = run_e2e.agent_proposer("openai/Qwen/Qwen3-8B")

    assert asked == [("litellm", "openai/Qwen/Qwen3-8B")]
    assert isinstance(proposer, LiteLLMProposer)


def test_a_shared_provider_install_is_refused_not_mismeasured(run_e2e, monkeypatch):
    """The fail-open version of this is a results file of zeros.

    The recorder wraps `litellm.completion`, the boundary the direct proposer
    crosses; the chat layer need not cross it. Recording nothing means every
    sample carries an empty `Recorded()` and scores as a tier-1 failure -- a
    whole run of zero reward that reads like a finding about the model rather
    than a harness pointed at the wrong seam. And the two paths differ in the
    thing tier 1 grades: one sends `response_format` and the other does not.
    """
    from aorta.agent.llm import ChatProviderProposer

    monkeypatch.setattr(
        run_e2e,
        "make_proposer",
        lambda backend, *, model=None: ChatProviderProposer(backend, model=model),
    )

    with pytest.raises(SystemExit) as refusal:
        run_e2e.agent_proposer("openai/Qwen/Qwen3-8B")

    message = str(refusal.value)
    # Names what was built, and both reasons the measurement would not transfer.
    assert "ChatProviderProposer" in message
    assert "litellm.completion" in message
    assert "response_format" in message


# --------------------------------------------------------------------------- #
# Review pass 2026-09-23, second round
# --------------------------------------------------------------------------- #


def test_a_corrupt_twin_does_not_make_the_good_report_a_collision(
    build_corpus, tmp_path, capsys
):
    """The loader's two rules contradicted each other on one archive.

    The id was claimed when the path was first seen, before the file had been
    shown to be a report. So a corrupt `runA/<case>/sanitizer_report.json` --
    skipped on its own line, one scenario lost, exactly as intended -- then
    made the valid `runB/<case>/sanitizer_report.json` a duplicate, and this
    refusal aborts the whole build. A file that contributes no scenario cannot
    make anything ambiguous, which is the only thing the collision guard is
    about.
    """
    results = tmp_path / "results"
    source = (
        _SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json"
    ).read_text()
    (results / "runA" / "gemm_f32_waitcheck").mkdir(parents=True)
    (results / "runA" / "gemm_f32_waitcheck" / "sanitizer_report.json").write_text(
        "{not json"
    )
    (results / "runB" / "gemm_f32_waitcheck").mkdir(parents=True)
    (results / "runB" / "gemm_f32_waitcheck" / "sanitizer_report.json").write_text(
        source
    )

    scenarios = build_corpus.collect(results)

    assert [s.scenario_id for s in scenarios] == ["gemm_f32_waitcheck"]
    assert scenarios[0].report_path.parent.parent.name == "runB"
    assert "skipped" in capsys.readouterr().err


def test_an_object_that_is_not_a_report_does_not_make_the_good_one_a_collision(
    build_corpus, tmp_path, capsys
):
    """The same fault as above, one layer in, and the layer the fix stopped at.

    Claiming the id after `isinstance(doc, dict)` fixed unparseable twins and
    left `{}` -- which is an object, and is not a report. `SanitizerReport`
    rejects it, `triage_example` prints `rejected` and drops the scenario, so
    it contributes nothing; but the id was already reserved, so the valid
    `runB/<case>/sanitizer_report.json` was refused as a duplicate and the
    whole build aborted over a scenario that was going to be discarded.

    A truncated writer is the ordinary way to produce this: the bytes are
    valid JSON at every prefix that closes a brace, so `json.JSONDecodeError`
    is exactly the wrong thing to wait for.

    The id is withheld rather than the scenario dropped. A document that is
    not a report is still a *rejected* report, and
    `test_a_corpus_with_nothing_in_it_is_a_failed_build` depends on that
    count: rejecting every report found has to exit 1 rather than write an
    empty corpus. Both paths are collected, and only the valid one keys
    `seen`.
    """
    results = tmp_path / "results"
    source = (
        _SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json"
    ).read_text()
    (results / "runA" / "gemm_f32_waitcheck").mkdir(parents=True)
    (results / "runA" / "gemm_f32_waitcheck" / "sanitizer_report.json").write_text("{}")
    (results / "runB" / "gemm_f32_waitcheck").mkdir(parents=True)
    (results / "runB" / "gemm_f32_waitcheck" / "sanitizer_report.json").write_text(
        source
    )

    # The assertion is that this returns at all: it used to raise
    # `DuplicateScenario` and take the build with it.
    scenarios = build_corpus.collect(results)

    assert [s.scenario_id for s in scenarios] == ["gemm_f32_waitcheck"] * 2
    assert {s.report_path.parent.parent.name for s in scenarios} == {"runA", "runB"}
    assert "claims no scenario id" in capsys.readouterr().err
    # And the valid report is the one that survives to a row, which is the
    # scenario the abort was costing.
    assert build_corpus.triage_example(scenarios[0], {}, {}) is None
    assert build_corpus.triage_example(scenarios[1], {}, {}) is not None


def test_two_good_reports_sharing_a_case_name_are_still_refused(
    build_corpus, tmp_path
):
    """Narrowness. Moving the claim must not retire the guard.

    Two well-formed reports under one case name is the failure nothing
    downstream can see -- `run_e2e` keys its label map and its GRPO groups on
    the id, so one report is scored against the other's label -- and it is
    still a refusal.
    """
    results = tmp_path / "results"
    source = (
        _SURVEY / "reports" / "gemm_f32_waitcheck" / "sanitizer_report.json"
    ).read_text()
    for run in ("runA", "runB"):
        case = results / run / "gemm_f32_waitcheck"
        case.mkdir(parents=True)
        (case / "sanitizer_report.json").write_text(source)

    with pytest.raises(build_corpus.DuplicateScenario, match="gemm_f32_waitcheck"):
        build_corpus.collect(results)


def test_a_probe_that_draws_nothing_is_not_a_successful_probe(monkeypatch):
    """`--draws 0` printed `delivered 0/0  distinct 0/0` and exited 0.

    Every counter is computed over the completions that came back, so with
    none asked for they are all honest and all meaningless -- and a recorded
    diversity measurement over zero completions is the exact shape this tool
    was written to refuse, with the sample size taken to its limit. Rejected at
    the boundary, so no counter downstream has to defend against it.
    """
    probe_seed = _load("probe_seed")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no request should be sent for an empty probe")

    monkeypatch.setattr(probe_seed, "call", fail_if_called)

    for draws in ("0", "-1"):
        with pytest.raises(SystemExit) as refusal:
            probe_seed.main([
                "--base-url", "http://127.0.0.1:1/v1",
                "--model", "m",
                "--temperatures", "1.0",
                "--draws", draws,
            ])
        # argparse's own exit code for a usage error, not a verdict of 1.
        assert refusal.value.code == 2, draws


# --------------------------------------------------------------------------- #
# Review pass 2026-09-23, third round: what three draws can and cannot prove
# --------------------------------------------------------------------------- #


def _seeded_engine(behaviour: dict | None = None):
    """A fake ``call`` whose completion is a function of the seed it was given.

    ``behaviour`` maps a seed (or ``None`` for the unseeded control) to the
    content returned. A seed absent from the map gets a fresh unique string on
    every draw, which is what an engine that ignores the key looks like.
    """
    fixed = behaviour or {}
    counter = itertools.count()

    def call(model, *, temperature, seed, seed_mode):
        if seed in fixed:
            return {"error": "", "content": fixed[seed]}
        return {"error": "", "content": f'{{"draw": {next(counter)}}}'}

    return call


def test_a_seed_that_does_not_replay_is_ignored_without_needing_the_control():
    """The one direction this probe can still conclude on an engine it cannot
    characterise.

    An honoured seed reproduces by definition, so draws that differ under a
    fixed seed rule honouring out on their own. That matters beyond tidiness:
    it is the finding this probe actually produced -- seeds accepted and
    ignored -- and it is the finding that does *not* rest on the control or on
    the repeat count. The residual it cannot see is nondeterminism outside
    sampling, which would show here as a false IGNORED.
    """
    probe_seed = _load("probe_seed")
    verdict, why = probe_seed.seed_verdict(
        replays=False, diverges=True, control_samples=False
    )
    assert verdict == probe_seed.IGNORED
    assert "different completions" in why


def test_a_replay_on_an_engine_that_does_not_sample_decides_nothing():
    """The false HONOURED, in its systematic form.

    An engine returning the same completion whatever you ask replays under
    every key, including one it throws away. More draws do not help -- they are
    more of the same non-evidence -- so the control is what separates them, and
    with the control silent the verdict has to be silent too.
    """
    probe_seed = _load("probe_seed")
    verdict, why = probe_seed.seed_verdict(
        replays=True, diverges=True, control_samples=False
    )
    assert verdict == probe_seed.INCONCLUSIVE
    assert "unseeded" in why


def test_two_seeds_landing_on_one_completion_is_not_a_verdict_of_ignored():
    """The false negative, which the old bool reported as `no`.

    A grammar-constrained decode can leave so little to choose that two seeds
    agree while both are honoured. That is indistinguishable here from an
    ignored seed on a low-entropy engine, and picking either is a claim the
    experiment did not earn.
    """
    probe_seed = _load("probe_seed")
    verdict, why = probe_seed.seed_verdict(
        replays=True, diverges=False, control_samples=True
    )
    assert verdict == probe_seed.INCONCLUSIVE
    assert "nothing left to choose" in why


def test_a_replayed_diverging_seed_on_a_sampling_engine_is_still_honoured():
    """The narrowness control for the two refusals above.

    A third outcome is only worth having if it does not swallow the second.
    Without this, `seed_verdict` returning INCONCLUSIVE unconditionally passes
    every other test in this group, and the probe would have stopped being able
    to report a working seed key at all.
    """
    probe_seed = _load("probe_seed")
    verdict, _ = probe_seed.seed_verdict(
        replays=True, diverges=True, control_samples=True
    )
    assert verdict == probe_seed.HONOURED


def test_an_engine_that_ignores_seeds_is_reported_ignored(monkeypatch):
    """End to end on the engine this probe was pointed at.

    Recorded in the routing write-up: with the triton backend the same seed
    twice gave different completions on all three keys, while unseeded draws
    at the same temperature were fully diverse. That is the IGNORED branch, and
    it survives the move away from a bool.
    """
    probe_seed = _load("probe_seed")
    monkeypatch.setattr(probe_seed, "call", _seeded_engine())

    control = probe_seed.probe_unseeded_control("m", 1.0, 3)
    assert control["samples"] is True
    modes = probe_seed.probe_seed_modes("m", 1.0, 3, control)

    assert {row["verdict"] for row in modes.values()} == {probe_seed.IGNORED}
    assert not [m for m, row in modes.items() if row["honoured"]]


def test_a_collapsed_engine_cannot_be_used_to_rule_a_seed_out(monkeypatch):
    """The reading the old bool got wrong, and the reason for the third value.

    Every completion identical: the seed replays, two seeds agree, and unseeded
    draws agree as well. The old code called that `no` -- a finding about the
    seed -- when every byte of it is a finding about the temperature. Being
    unable to tell is the correct answer and is now the reported one.
    """
    probe_seed = _load("probe_seed")
    monkeypatch.setattr(
        probe_seed, "call", lambda *a, **k: {"error": "", "content": "{}"}
    )

    control = probe_seed.probe_unseeded_control("m", 0.0, 3)
    assert control["samples"] is False
    modes = probe_seed.probe_seed_modes("m", 0.0, 3, control)

    assert {row["verdict"] for row in modes.values()} == {probe_seed.INCONCLUSIVE}
    assert all(row["same_seed_reproduces"] for row in modes.values())


def test_a_working_seed_key_is_found_end_to_end(monkeypatch):
    """The whole point of the probe, still reachable through the new verdict.

    Content is a function of the seed and the unseeded control is diverse,
    which is what an engine honouring the key looks like from outside.
    """
    probe_seed = _load("probe_seed")
    monkeypatch.setattr(
        probe_seed,
        "call",
        _seeded_engine({probe_seed.SEED_A: '{"a": 1}',
                        probe_seed.SEED_B: '{"a": 2}'}),
    )

    control = probe_seed.probe_unseeded_control("m", 1.0, 3)
    modes = probe_seed.probe_seed_modes("m", 1.0, 3, control)

    assert {row["verdict"] for row in modes.values()} == {probe_seed.HONOURED}


def test_a_missing_control_cannot_be_read_as_a_sampling_engine(monkeypatch):
    """Fail closed where the control is absent rather than assume it passed.

    `probe_seed_modes` takes the control as an argument, so a caller can omit
    it. Defaulting that to "the engine samples" would reinstate the false
    HONOURED through the one path that never measured anything.
    """
    probe_seed = _load("probe_seed")
    monkeypatch.setattr(
        probe_seed,
        "call",
        _seeded_engine({probe_seed.SEED_A: '{"a": 1}',
                        probe_seed.SEED_B: '{"a": 2}'}),
    )

    modes = probe_seed.probe_seed_modes("m", 1.0, 3)

    assert {row["verdict"] for row in modes.values()} == {probe_seed.INCONCLUSIVE}


def test_one_draw_per_seed_is_refused_rather_than_reported():
    """`--seed-repeats 1` is a replay claim about a single completion.

    One draw is identical to itself whatever the engine did with the key, so
    every accepted mode would report `same_seed_reproduces` and a diverse
    engine would hand back HONOURED on no evidence at all -- the exact failure
    this probe exists to catch, manufactured by its own flag.
    """
    probe_seed = _load("probe_seed")
    for repeats in ("1", "0", "-1"):
        with pytest.raises(SystemExit) as refusal:
            probe_seed.main([
                "--base-url", "http://127.0.0.1:1/v1",
                "--model", "m",
                "--temperatures", "1.0",
                "--draws", "1",
                "--seed-repeats", repeats,
            ])
        assert refusal.value.code == 2, repeats


def test_an_undecided_run_does_not_print_a_finding(monkeypatch, capsys):
    """The sentence a human pastes into the write-up, on a run that decided
    nothing.

    "No seed key is honoured" is a finding; "this run could not tell" is the
    absence of one. Printing the first where the second is true is how a probe
    launders its own inconclusive result into evidence -- and the write-up is
    where that evidence goes.
    """
    probe_seed = _load("probe_seed")
    monkeypatch.setattr(
        probe_seed, "call", lambda *a, **k: {"error": "", "content": "{}"}
    )

    probe_seed.main([
        "--base-url", "http://127.0.0.1:1/v1",
        "--model", "m",
        "--temperatures", "0.0",
        "--draws", "2",
        "--seed-repeats", "2",
    ])
    out = capsys.readouterr().out

    assert "could not be decided" in out
    assert "no seed key is honoured" not in out
    assert "engine-samples=False" in out


def test_one_seed_replaying_is_not_the_key_replaying(monkeypatch):
    """Both seeds have to replay, not just the one that happened to.

    Checking only the first seed's draws makes "reproduces" a statement about
    one value of the parameter. An engine that replays under 1234 and samples
    freely under 9999 is not honouring the key -- it is doing something else
    that this probe would have reported as a working seed mode, which is the
    accepted-but-ignored shape wearing a different hat.
    """
    probe_seed = _load("probe_seed")
    monkeypatch.setattr(
        probe_seed, "call", _seeded_engine({probe_seed.SEED_A: '{"a": 1}'})
    )

    control = probe_seed.probe_unseeded_control("m", 1.0, 3)
    modes = probe_seed.probe_seed_modes("m", 1.0, 3, control)

    assert {row["verdict"] for row in modes.values()} == {probe_seed.IGNORED}
    assert not any(row["same_seed_reproduces"] for row in modes.values())


def _order_following_engine(state):
    """A fake ``call`` that ignores the seed; its output is a function of the
    request's position alone -- server state, batch shape, a warming cache."""
    counter = itertools.count()

    def call(model, *, temperature, seed, seed_mode):
        return {"error": "", "content": f'{{"state": {state(next(counter))}}}'}

    return call


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(lambda n: n // 3, id="drifts-every-3-requests"),
        pytest.param(lambda n: n % 2, id="flips-every-request"),
        *[pytest.param(lambda n, k=k: int(n >= k), id=f"changes-once-after-{k}")
          for k in range(1, 6)],
    ],
)
def test_an_engine_whose_output_follows_request_order_is_never_honoured(
    monkeypatch, state
):
    """The seed was confounded with request order.

    Every `SEED_A` draw went out before every `SEED_B` draw, so an engine that
    ignores the key but drifts every three requests replayed within each block
    and differed between them: HONOURED on all three keys, measured. The
    every-request case is what plain alternation would have got wrong, and the
    single change points cover every position in the first mode's six draws.
    The control is given as sampling, which is what such an engine shows.
    """
    probe_seed = _load("probe_seed")
    monkeypatch.setattr(probe_seed, "call", _order_following_engine(state))

    modes = probe_seed.probe_seed_modes("m", 1.0, 3, {"samples": True})

    assert not [m for m, row in modes.items() if row["honoured"]], modes


@pytest.mark.parametrize("repeats", [2, 3, 4, 5, 6])
def test_the_draw_order_gives_no_request_position_to_one_seed_alone(repeats):
    """The property the behavioural test above samples, stated for every size.

    Each seed is drawn `repeats` times, lands on both parities, and starts
    before the other finishes -- so neither a flip-every-request engine nor a
    single change point can hand the two seeds disjoint outputs.
    """
    probe_seed = _load("probe_seed")
    order = probe_seed.seed_draw_order(repeats)
    positions = {
        seed: [i for i, drawn in enumerate(order) if drawn == seed]
        for seed in (probe_seed.SEED_A, probe_seed.SEED_B)
    }
    a, b = positions[probe_seed.SEED_A], positions[probe_seed.SEED_B]
    assert len(order) == 2 * repeats
    assert len(a) == len(b) == repeats
    assert {i % 2 for i in a} == {i % 2 for i in b} == {0, 1}
    assert min(a) < max(b) and min(b) < max(a)


def test_an_outage_in_the_control_does_not_certify_sampling(monkeypatch):
    """A control that lost draws reports "not observed", never "ruled out".

    Failed calls record `content: ""`, so counting them would make an outage
    read as diversity -- the same defect the temperature table was fixed for,
    arriving in the measurement every INCONCLUSIVE below now rests on. One
    delivered completion cannot show sampling, and the verdict that consumes
    this is a refusal to conclude rather than a finding, which is the direction
    a half-delivered control should push.
    """
    probe_seed = _load("probe_seed")
    answers = iter([
        {"error": "", "content": '{"a": 1}'},
        {"error": "APIConnectionError: reset", "content": ""},
        {"error": "APIConnectionError: reset", "content": ""},
    ])
    monkeypatch.setattr(probe_seed, "call", lambda *a, **k: next(answers))

    control = probe_seed.probe_unseeded_control("m", 1.0, 3)

    assert control["delivered"] == 1
    assert control["distinct"] == 1
    assert control["samples"] is False
    assert len(control["errors"]) == 2


# --------------------------------------------------------------------------- #
# Review pass 2026-09-23: the opt-out a typo produces
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["-1", "-0.05", "-120"])
def test_a_negative_peer_grace_is_refused_rather_than_disabling_the_check(
    nccl_roundtrip_check, tmp_path, monkeypatch, capsys, value
):
    """`> 0` gates the marker check, so every negative value is a silent opt-out.

    The peer's `roundN.done` markers are the only direct evidence that a
    collective happened at all -- `decide_verdict` ranks a missing marker above
    a perfect-looking round trip precisely because the HTTP legs can all return
    200 on a group that never sent. Dropping them turns the strongest check in
    this script off, and `--peer-grace -1` does exactly that while reading like
    a request to wait.

    Refused rather than clamped: clamping to 0 honours the reading nobody
    meant, and clamping to the default waits two minutes on an argument that
    asked for less than none. Checked next to `--replicates`, before any of the
    run happens, because the cost of the old behaviour was a weaker verdict on
    a run that had already been paid for.
    """
    # Every request this driver makes goes through `call`, so a `call` that
    # refuses to be reached is how "nothing ran" is asserted rather than
    # inferred. Without it the run reaches a health check against an engine
    # that is not there and this test fails by timing out on real sockets --
    # which is also what the argument itself used to cost.
    def no_requests(*a, **k):
        raise AssertionError("the run started despite an invalid --peer-grace")

    monkeypatch.setattr(nccl_roundtrip_check, "call", no_requests)
    monkeypatch.setattr(sys, "argv", [
        "nccl_roundtrip_check.py",
        "--model", "m",
        "--plan", str(tmp_path / "plan.json"),
        "--plan-run-id", "r1",
        "--master-port", "29500",
        "--world-size", "2",
        "--peer-grace", value,
        "--out", str(tmp_path / "report.json"),
    ])

    with pytest.raises(SystemExit) as excinfo:
        nccl_roundtrip_check.main()

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--peer-grace" in err, err
    # The value is echoed, because a negative one usually arrives from a
    # generated command line where the caller cannot see what was substituted.
    assert value in err, err
    # And the message has to say what 0 does, or the reader's next move is to
    # try 0 expecting it to mean "no limit".
    assert "0 is how the check is disabled" in err, err
    # Nothing ran: the refusal is an argument error, not a verdict.
    assert not (tmp_path / "report.json").exists()


def test_the_documented_zero_opt_out_still_works(
    nccl_roundtrip_check, tmp_path, monkeypatch
):
    """Narrowness: `0` is the opt-out `--help` names, and it has to stay one.

    A `>= 0` bound would be the easy over-reach, and it would break the only
    supported way to run this script against an engine with no peer attached.
    Driven all the way through `main` rather than stopping at the parse, so
    this also pins that 0 still reaches the `> 0` gate as a skipped check
    rather than as a zero-second wait that fails every marker.
    """
    code, report, posted = _drive_roundtrip(
        nccl_roundtrip_check, tmp_path, monkeypatch, marker=False, peer_grace="0"
    )

    # The marker for round 1 is deliberately absent, and with the check off
    # that is not allowed to cost the run its verdict.
    assert code == 0
    assert report["verdict"] == "PROVEN"
    assert report["weights_changed_under_perturb"] is True
    # No `peer` block at all, rather than one recording `appeared: false`:
    # nobody asked, which is the distinction this module keeps everywhere the
    # peer evidence is missing. A zero-second wait would have written one.
    assert "peer" not in report["perturb"]
    assert "peer" not in report["restore"]
    assert posted.count("/update_weights") == 2


def test_the_help_names_zero_as_the_way_to_disable_it(
    nccl_roundtrip_check, monkeypatch, capsys
):
    """The refusal above points the reader at `0`, so `--help` must too.

    Pinned because the two texts are the whole contract: an operator who hits
    the refusal is told 0 disables the check, and the only place that claim is
    documented for someone who has not hit it is this help string.
    """
    monkeypatch.setattr(sys, "argv", ["nccl_roundtrip_check.py", "--help"])
    with pytest.raises(SystemExit):
        nccl_roundtrip_check.main()

    flag = " ".join(capsys.readouterr().out.split()).split("--peer-grace", 1)[1]
    assert "0 disables the check" in flag, flag
